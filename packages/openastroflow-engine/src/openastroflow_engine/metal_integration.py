"""Safe ctypes bridge for the optional Apple Metal ordinary integrator.

The native library owns the Metal device, command queue, and compiled shader.
Python owns every input/output array and keeps it alive for the complete ABI
call.  Metal is an optional accelerator: capability, ABI, frame-count, or
numerical-gate failure falls back to the full portable CPU integration without
truncating the stack.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from contextlib import nullcontext
import ctypes
import ctypes.util
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from lightframeqc.content_hash import file_sha256
from .calibration import (
    CalibrationError,
    DEFAULT_MEMORY_BUDGET,
    FitsFloatWriter,
    FrameExpression,
    IntegrationParameters,
    IntegrationMapPaths,
    IntegrationResult,
    _StatsAccumulator,
    _atomic_publish_file,
    _canonical_expression,
    _expression_rows,
    _combined_integration_weights,
    _estimate_rejection_sigma_floor,
    _open_expression_sources,
    _prepare_transient_rejection,
    _ordinary_integration_tile,
    _temporary_output,
    _validate_expression_shapes,
    integrate_expressions,
)
from .hardware import HardwareProfile, detect_hardware
from .native_kernels import (
    _candidate_library_paths,
    load_native_kernels,
)
from .performance_profile import ExecutionTuning, select_execution_tuning


NATIVE_ABI_VERSION = 1
MAXIMUM_METAL_LINEAR_FIT_FRAMES = 64
MAXIMUM_METAL_REJECTION_FRAMES = 512
MAXIMUM_NATIVE_REJECTION_FRAMES = 512
# Conservative live-buffer model for one full-stack ordinary-integration row.
# Preparation includes the Float32 values, median/MAD advanced-index scratch,
# finite/accepted masks, and the returned UInt8 rejection mask.  An in-flight
# Metal call retains values/mask, a normalized Float32 copy, shared MTL input
# copies, and output/count buffers.  Fixed bytes cover image-sized statistics,
# FITS-map conversion, and output arrays that do not scale with frame count.
PREPARATION_BYTES_PER_SAMPLE = 24
METAL_INFLIGHT_BYTES_PER_SAMPLE = 14
METAL_FIXED_BYTES_PER_PIXEL = 96
SUPPORTED_BACKENDS = frozenset(
    {"auto", "portable-cpu", "generic-apple-metal", "m3-pro-tuned"}
)


class MetalIntegrationError(RuntimeError):
    """An optional native accelerator could not produce trusted output."""


class _RequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("image_height", ctypes.c_uint32),
        ("first_row", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("frame_count", ctypes.c_uint32),
        ("grid_width", ctypes.c_uint32),
        ("grid_height", ctypes.c_uint32),
        ("frame_major_samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_size_t),
        ("frame_major_scale_grid", ctypes.POINTER(ctypes.c_float)),
        ("scale_grid_count", ctypes.c_size_t),
        ("frame_major_zero_offset_grid", ctypes.POINTER(ctypes.c_float)),
        ("zero_offset_grid_count", ctypes.c_size_t),
        ("frame_weights", ctypes.POINTER(ctypes.c_float)),
        ("weight_count", ctypes.c_size_t),
        ("range_low", ctypes.c_float),
        ("low_tolerance", ctypes.c_float),
        ("high_tolerance", ctypes.c_float),
        ("fit_bisection_iterations", ctypes.c_uint32),
        ("rejection_iterations", ctypes.c_uint32),
        ("output_scale", ctypes.c_float),
        ("output_offset", ctypes.c_float),
    ]


class _OutputV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("integrated", ctypes.POINTER(ctypes.c_float)),
        ("accepted_samples", ctypes.POINTER(ctypes.c_uint16)),
        ("rejected_samples", ctypes.POINTER(ctypes.c_uint16)),
        ("pixel_capacity", ctypes.c_size_t),
    ]


class _MaskedRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("image_height", ctypes.c_uint32),
        ("first_row", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("frame_count", ctypes.c_uint32),
        ("grid_width", ctypes.c_uint32),
        ("grid_height", ctypes.c_uint32),
        ("frame_major_samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_size_t),
        ("frame_major_rejection_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("rejection_mask_count", ctypes.c_size_t),
        ("frame_major_scale_grid", ctypes.POINTER(ctypes.c_float)),
        ("scale_grid_count", ctypes.c_size_t),
        ("frame_major_zero_offset_grid", ctypes.POINTER(ctypes.c_float)),
        ("zero_offset_grid_count", ctypes.c_size_t),
        ("frame_weights", ctypes.POINTER(ctypes.c_float)),
        ("weight_count", ctypes.c_size_t),
        ("rejection_bits", ctypes.c_uint32),
        ("output_scale", ctypes.c_float),
        ("output_offset", ctypes.c_float),
    ]


class _StatsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("executed_on_gpu", ctypes.c_uint32),
        ("wall_seconds", ctypes.c_double),
        ("gpu_seconds", ctypes.c_double),
        ("submitted_buffer_bytes", ctypes.c_uint64),
        ("recommended_working_set_bytes", ctypes.c_uint64),
        ("maximum_buffer_bytes", ctypes.c_uint64),
        ("device_name", ctypes.c_char * 128),
    ]


@dataclass(frozen=True, slots=True)
class NativeTileResult:
    integrated: NDArray[np.float32]
    accepted: NDArray[np.uint16]
    rejected: NDArray[np.uint16]
    stats: Mapping[str, Any]


def _masked_input_summary(
    samples: NDArray[np.float32], rejection_mask: NDArray[np.uint8]
) -> tuple[
    float,
    NDArray[np.uint16],
    NDArray[np.uint16],
    NDArray[np.uint16],
]:
    """Compute normalization range and portable counts in one bounded pass."""

    source_values = np.asarray(samples, dtype=np.float32)
    mask_values = np.asarray(rejection_mask, dtype=np.uint8)
    if source_values.ndim != 3 or mask_values.shape != source_values.shape:
        raise ValueError("masked integration counts require equal frame-major arrays")
    accepted = np.zeros(source_values.shape[1:], dtype=np.uint16)
    rejected = np.zeros_like(accepted)
    unavailable = np.zeros_like(accepted)
    input_normalization = 1.0
    # Accumulate one frame at a time instead of retaining three frame-stack
    # sized boolean temporaries.  The public policy caps frame_count at 512, so
    # UInt16 accumulation cannot overflow.
    for frame_values, frame_mask in zip(source_values, mask_values, strict=True):
        finite = np.isfinite(frame_values)
        masked = frame_mask != 0
        np.add(accepted, finite & ~masked, out=accepted, casting="unsafe")
        np.add(rejected, finite & masked, out=rejected, casting="unsafe")
        np.add(unavailable, ~finite, out=unavailable, casting="unsafe")
        active = finite & ~masked
        if np.any(active):
            frame_maximum = float(
                np.max(frame_values, where=active, initial=np.float32(0.0))
            )
            frame_minimum = float(
                np.min(frame_values, where=active, initial=np.float32(0.0))
            )
            input_normalization = max(
                input_normalization, abs(frame_minimum), abs(frame_maximum)
            )
    return input_normalization, accepted, rejected, unavailable


def _portable_mask_counts(
    samples: NDArray[np.float32], rejection_mask: NDArray[np.uint8]
) -> tuple[NDArray[np.uint16], NDArray[np.uint16], NDArray[np.uint16]]:
    """Separate accepted samples, finite rejections, and unavailable coverage."""

    _, accepted, rejected, unavailable = _masked_input_summary(
        samples, rejection_mask
    )
    return accepted, rejected, unavailable


@dataclass(frozen=True, slots=True)
class _PendingTile:
    first_row: int
    future: Future[NativeTileResult]


def _metal_parallel_bytes_per_row(
    *,
    width: int,
    frame_count: int,
    preparation_workers: int,
    inflight_buffers: int,
) -> int:
    """Conservatively estimate peak live ordinary-Metal bytes for one row."""

    if min(width, frame_count, preparation_workers, inflight_buffers) < 1:
        raise ValueError("Metal memory-plan dimensions must be positive")
    return width * (
        frame_count
        * (
            PREPARATION_BYTES_PER_SAMPLE * preparation_workers
            + METAL_INFLIGHT_BYTES_PER_SAMPLE * inflight_buffers
        )
        + METAL_FIXED_BYTES_PER_PIXEL
    )


def _metal_memory_plan(
    *,
    width: int,
    height: int,
    frame_count: int,
    configured_workers: int,
    inflight_buffers: int,
    target_tile_rows: int,
    memory_budget_bytes: int,
) -> tuple[int, int, int]:
    """Choose workers/tile rows while keeping the modeled peak under budget."""

    if min(height, target_tile_rows, memory_budget_bytes) < 1:
        raise ValueError("Metal memory-plan limits must be positive")
    workers = max(1, configured_workers)
    target_rows = min(height, target_tile_rows)
    # Retain the tuned tile height when possible: reducing independent mask
    # preparation workers avoids doubling Metal dispatch count on deep stacks.
    while workers > 1:
        candidate = _metal_parallel_bytes_per_row(
            width=width,
            frame_count=frame_count,
            preparation_workers=workers,
            inflight_buffers=inflight_buffers,
        )
        if candidate * target_rows <= memory_budget_bytes:
            break
        workers -= 1
    bytes_per_row = _metal_parallel_bytes_per_row(
        width=width,
        frame_count=frame_count,
        preparation_workers=workers,
        inflight_buffers=inflight_buffers,
    )
    if bytes_per_row > memory_budget_bytes:
        raise MetalIntegrationError(
            "one full-stack Metal preparation row exceeds the memory budget"
        )
    tile_rows = max(
        1,
        min(height, target_rows, memory_budget_bytes // bytes_per_row),
    )
    return workers, tile_rows, bytes_per_row


def _sha256(path: Path) -> str:
    return "sha256:" + file_sha256(path)


class NativeMetalExecutor:
    """Validated owner of one opaque native Metal executor."""

    def __init__(
        self,
        *,
        library_path: str | os.PathLike[str] | None = None,
        metal_source_path: str | os.PathLike[str] | None = None,
    ) -> None:
        candidates = _candidate_library_paths(library_path)
        if not candidates:
            raise MetalIntegrationError("native library was not found")
        self.library_path = candidates[0]
        try:
            self._library = ctypes.CDLL(str(self.library_path), use_errno=True)
        except OSError as error:
            raise MetalIntegrationError(f"cannot load native library: {error}") from error
        self._bind()
        if int(self._library.oaf_native_abi_version()) != NATIVE_ABI_VERSION:
            raise MetalIntegrationError("native ABI version differs from Python adapter")
        error = ctypes.create_string_buffer(1024)
        available = ctypes.c_uint32(0)
        status = int(
            self._library.oaf_native_metal_available_v1(
                ctypes.byref(available), error, ctypes.sizeof(error)
            )
        )
        if status != 0:
            raise MetalIntegrationError(self._error_text(error, status))
        if available.value != 1:
            raise MetalIntegrationError("Metal device is unavailable")
        source: bytes | None = None
        if metal_source_path:
            source_path = Path(metal_source_path).expanduser().resolve(strict=True)
            if not source_path.is_file():
                raise MetalIntegrationError("Metal source path is not a file")
            source = os.fsencode(source_path)
        handle = ctypes.c_void_p()
        status = int(
            self._library.oaf_native_metal_executor_create_v1(
                source, ctypes.byref(handle), error, ctypes.sizeof(error)
            )
        )
        if status != 0 or not handle.value:
            raise MetalIntegrationError(self._error_text(error, status))
        self._handle = handle

    def _bind(self) -> None:
        library = self._library
        required = (
            "oaf_native_abi_version",
            "oaf_native_cpu_linear_fit_v1",
            "oaf_native_cpu_masked_weighted_v1",
            "oaf_native_metal_available_v1",
            "oaf_native_metal_executor_create_v1",
            "oaf_native_metal_executor_destroy_v1",
            "oaf_native_metal_linear_fit_v1",
            "oaf_native_metal_masked_weighted_v1",
        )
        missing = [name for name in required if not hasattr(library, name)]
        if missing:
            raise MetalIntegrationError(
                "native library misses required symbols: " + ", ".join(missing)
            )
        library.oaf_native_abi_version.argtypes = []
        library.oaf_native_abi_version.restype = ctypes.c_uint32
        common = [
            ctypes.POINTER(_RequestV1),
            ctypes.POINTER(_OutputV1),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.oaf_native_cpu_linear_fit_v1.argtypes = common
        library.oaf_native_cpu_linear_fit_v1.restype = ctypes.c_int
        masked_common = [
            ctypes.POINTER(_MaskedRequestV1),
            ctypes.POINTER(_OutputV1),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.oaf_native_cpu_masked_weighted_v1.argtypes = masked_common
        library.oaf_native_cpu_masked_weighted_v1.restype = ctypes.c_int
        library.oaf_native_metal_available_v1.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.oaf_native_metal_available_v1.restype = ctypes.c_int
        library.oaf_native_metal_executor_create_v1.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.oaf_native_metal_executor_create_v1.restype = ctypes.c_int
        library.oaf_native_metal_executor_destroy_v1.argtypes = [ctypes.c_void_p]
        library.oaf_native_metal_executor_destroy_v1.restype = None
        library.oaf_native_metal_linear_fit_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_RequestV1),
            ctypes.POINTER(_OutputV1),
            ctypes.POINTER(_StatsV1),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.oaf_native_metal_linear_fit_v1.restype = ctypes.c_int
        library.oaf_native_metal_masked_weighted_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MaskedRequestV1),
            ctypes.POINTER(_OutputV1),
            ctypes.POINTER(_StatsV1),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.oaf_native_metal_masked_weighted_v1.restype = ctypes.c_int

    @staticmethod
    def _error_text(buffer: ctypes.Array[Any], status: int) -> str:
        detail = bytes(buffer.value).decode("utf-8", errors="replace").strip()
        return detail or f"native call failed with status {status}"

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self._library.oaf_native_metal_executor_destroy_v1(handle)
            handle.value = None

    def __enter__(self) -> NativeMetalExecutor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _request_and_output(
        samples: NDArray[np.float32],
        *,
        image_height: int,
        first_row: int,
        scales: NDArray[np.float32],
        offsets: NDArray[np.float32],
        weights: NDArray[np.float32],
        sigma_clip: float,
    ) -> tuple[_RequestV1, _OutputV1, NDArray[np.float32], NDArray[np.uint16], NDArray[np.uint16]]:
        values = np.ascontiguousarray(samples, dtype=np.float32)
        if values.ndim != 3:
            raise MetalIntegrationError("native samples must be frame-major 3-D")
        frames, rows, width = values.shape
        if frames < 5 or frames > MAXIMUM_METAL_LINEAR_FIT_FRAMES:
            raise MetalIntegrationError("Metal exact-request frame count is outside [5, 64]")
        if first_row < 0 or first_row + rows > image_height:
            raise MetalIntegrationError("native tile rows are outside the image")
        scale_values = np.ascontiguousarray(scales, dtype=np.float32)
        offset_values = np.ascontiguousarray(offsets, dtype=np.float32)
        weight_values = np.ascontiguousarray(weights, dtype=np.float32)
        expected_grid = frames * 4
        if scale_values.size != expected_grid or offset_values.size != expected_grid:
            raise MetalIntegrationError("native 2x2 normalization grid cardinality differs")
        if weight_values.shape != (frames,):
            raise MetalIntegrationError("native weight cardinality differs")
        integrated = np.empty((rows, width), dtype=np.float32)
        accepted = np.empty((rows, width), dtype=np.uint16)
        rejected = np.empty((rows, width), dtype=np.uint16)
        request = _RequestV1(
            ctypes.sizeof(_RequestV1),
            width,
            image_height,
            first_row,
            rows,
            frames,
            2,
            2,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.size,
            scale_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scale_values.size,
            offset_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            offset_values.size,
            weight_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weight_values.size,
            -np.finfo(np.float32).max,
            sigma_clip,
            sigma_clip,
            10,
            8,
            1.0,
            0.0,
        )
        output = _OutputV1(
            ctypes.sizeof(_OutputV1),
            integrated.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            accepted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            rejected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            integrated.size,
        )
        # Return every owner: ctypes structures only hold raw pointers.
        request._array_owners = (values, scale_values, offset_values, weight_values)  # type: ignore[attr-defined]
        return request, output, integrated, accepted, rejected

    def run_tile(
        self,
        samples: NDArray[np.float32],
        *,
        image_height: int,
        first_row: int,
        scales: NDArray[np.float32],
        offsets: NDArray[np.float32],
        weights: NDArray[np.float32],
        sigma_clip: float,
        compare_cpu: bool,
    ) -> NativeTileResult:
        request, output, integrated, accepted, rejected = self._request_and_output(
            samples,
            image_height=image_height,
            first_row=first_row,
            scales=scales,
            offsets=offsets,
            weights=weights,
            sigma_clip=sigma_clip,
        )
        error = ctypes.create_string_buffer(1024)
        stats = _StatsV1()
        stats.struct_size = ctypes.sizeof(_StatsV1)
        status = int(
            self._library.oaf_native_metal_linear_fit_v1(
                self._handle,
                ctypes.byref(request),
                ctypes.byref(output),
                ctypes.byref(stats),
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            raise MetalIntegrationError(self._error_text(error, status))
        integrated[accepted == 0] = np.nan
        parity: dict[str, Any] = {"performed": False}
        if compare_cpu:
            cpu_integrated = np.empty_like(integrated)
            cpu_accepted = np.empty_like(accepted)
            cpu_rejected = np.empty_like(rejected)
            cpu_output = _OutputV1(
                ctypes.sizeof(_OutputV1),
                cpu_integrated.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                cpu_accepted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                cpu_rejected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                cpu_integrated.size,
            )
            error.value = b""
            status = int(
                self._library.oaf_native_cpu_linear_fit_v1(
                    ctypes.byref(request),
                    ctypes.byref(cpu_output),
                    error,
                    ctypes.sizeof(error),
                )
            )
            if status != 0:
                raise MetalIntegrationError(self._error_text(error, status))
            cpu_integrated[cpu_accepted == 0] = np.nan
            finite_equal = bool(
                np.array_equal(np.isfinite(integrated), np.isfinite(cpu_integrated))
            )
            counts_equal = bool(
                np.array_equal(accepted, cpu_accepted)
                and np.array_equal(rejected, cpu_rejected)
            )
            finite = np.isfinite(cpu_integrated) & np.isfinite(integrated)
            difference = np.abs(integrated[finite] - cpu_integrated[finite]).astype(
                np.float64, copy=False
            )
            max_abs = float(np.max(difference)) if difference.size else 0.0
            rmse = (
                float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
                if difference.size
                else 0.0
            )
            max_abs_limit = 2.0e-6
            rmse_limit = 2.0e-7
            parity = {
                "performed": True,
                "domain": "native-input-float32",
                "pixelsCompared": int(np.count_nonzero(finite)),
                "finiteMaskEqual": finite_equal,
                "rejectionCountsEqual": counts_equal,
                "maxAbs": max_abs,
                "rmse": rmse,
                "maxAbsLimit": max_abs_limit,
                "rmseLimit": rmse_limit,
                "passed": finite_equal
                and counts_equal
                and max_abs <= max_abs_limit
                and rmse <= rmse_limit,
            }
            if not parity["passed"]:
                raise MetalIntegrationError(
                    "CPU/Metal numerical gate failed: "
                    f"finite={finite_equal}, counts={counts_equal}, "
                    f"maxAbs={max_abs:.9g}, rmse={rmse:.9g}"
                )
        device_name = bytes(stats.device_name).split(b"\0", 1)[0].decode(
            "utf-8", errors="replace"
        )
        return NativeTileResult(
            integrated=integrated,
            accepted=accepted,
            rejected=rejected,
            stats={
                "deviceName": device_name,
                "executedOnGpu": stats.executed_on_gpu == 1,
                "wallSeconds": stats.wall_seconds,
                "gpuSeconds": stats.gpu_seconds,
                "submittedBufferBytes": stats.submitted_buffer_bytes,
                "recommendedWorkingSetBytes": stats.recommended_working_set_bytes,
                "maximumBufferBytes": stats.maximum_buffer_bytes,
                "parityGate": parity,
            },
        )

    @staticmethod
    def _masked_request_and_output(
        samples: NDArray[np.float32],
        rejection_mask: NDArray[np.uint8],
        *,
        image_height: int,
        first_row: int,
        scales: NDArray[np.float32],
        offsets: NDArray[np.float32],
        weights: NDArray[np.float32],
    ) -> tuple[
        _MaskedRequestV1,
        _OutputV1,
        NDArray[np.float32],
        NDArray[np.uint16],
        NDArray[np.uint16],
    ]:
        values = np.ascontiguousarray(samples, dtype=np.float32)
        mask = np.ascontiguousarray(rejection_mask, dtype=np.uint8)
        if values.ndim != 3 or mask.shape != values.shape:
            raise MetalIntegrationError(
                "masked native samples and mask must be same-shaped frame-major 3-D"
            )
        frames, rows, width = values.shape
        if frames < 1 or frames > MAXIMUM_METAL_REJECTION_FRAMES:
            raise MetalIntegrationError(
                f"masked Metal frame count is outside [1, {MAXIMUM_METAL_REJECTION_FRAMES}]"
            )
        if first_row < 0 or first_row + rows > image_height:
            raise MetalIntegrationError("masked native tile rows are outside the image")
        scale_values = np.ascontiguousarray(scales, dtype=np.float32)
        offset_values = np.ascontiguousarray(offsets, dtype=np.float32)
        weight_values = np.ascontiguousarray(weights, dtype=np.float32)
        expected_grid = frames * 4
        if scale_values.size != expected_grid or offset_values.size != expected_grid:
            raise MetalIntegrationError("masked native 2x2 grid cardinality differs")
        if weight_values.shape != (frames,):
            raise MetalIntegrationError("masked native weight cardinality differs")
        integrated = np.empty((rows, width), dtype=np.float32)
        accepted = np.empty((rows, width), dtype=np.uint16)
        rejected = np.empty((rows, width), dtype=np.uint16)
        request = _MaskedRequestV1(
            ctypes.sizeof(_MaskedRequestV1),
            width,
            image_height,
            first_row,
            rows,
            frames,
            2,
            2,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.size,
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            mask.size,
            scale_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scale_values.size,
            offset_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            offset_values.size,
            weight_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weight_values.size,
            1,
            1.0,
            0.0,
        )
        request._array_owners = (  # type: ignore[attr-defined]
            values,
            mask,
            scale_values,
            offset_values,
            weight_values,
        )
        output = _OutputV1(
            ctypes.sizeof(_OutputV1),
            integrated.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            accepted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            rejected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            integrated.size,
        )
        return request, output, integrated, accepted, rejected

    def run_masked_tile(
        self,
        samples: NDArray[np.float32],
        rejection_mask: NDArray[np.uint8],
        *,
        image_height: int,
        first_row: int,
        scales: NDArray[np.float32],
        offsets: NDArray[np.float32],
        weights: NDArray[np.float32],
        compare_cpu: bool,
    ) -> NativeTileResult:
        source_values = np.asarray(samples, dtype=np.float32)
        mask_values = np.asarray(rejection_mask, dtype=np.uint8)
        try:
            (
                input_normalization,
                portable_accepted,
                portable_rejected,
                unavailable,
            ) = _masked_input_summary(source_values, mask_values)
        except ValueError as error:
            raise MetalIntegrationError(str(error)) from error
        native_values = np.ascontiguousarray(
            source_values / np.float32(input_normalization), dtype=np.float32
        )
        request, output, integrated, accepted, rejected = (
            self._masked_request_and_output(
                native_values,
                rejection_mask,
                image_height=image_height,
                first_row=first_row,
                scales=scales,
                offsets=offsets,
                weights=weights,
            )
        )
        error = ctypes.create_string_buffer(1024)
        stats = _StatsV1()
        stats.struct_size = ctypes.sizeof(_StatsV1)
        status = int(
            self._library.oaf_native_metal_masked_weighted_v1(
                self._handle,
                ctypes.byref(request),
                ctypes.byref(output),
                ctypes.byref(stats),
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            raise MetalIntegrationError(self._error_text(error, status))
        # The native ABI deliberately accounts for every unavailable sample in
        # its rejected counter.  Product receipts/maps use the portable
        # semantics instead: finite sigma rejection is distinct from missing
        # registration coverage.  Verify the native accounting before
        # translating it, so a kernel/count drift still fails closed.
        native_accounted_rejected = np.asarray(
            portable_rejected + unavailable, dtype=np.uint16
        )
        if not np.array_equal(accepted, portable_accepted) or not np.array_equal(
            rejected, native_accounted_rejected
        ):
            raise MetalIntegrationError(
                "native masked integration counts disagree with portable semantics"
            )
        integrated[accepted == 0] = np.nan
        parity: dict[str, Any] = {"performed": False}
        if compare_cpu:
            cpu_integrated = np.empty_like(integrated)
            cpu_accepted = np.empty_like(accepted)
            cpu_rejected = np.empty_like(rejected)
            cpu_output = _OutputV1(
                ctypes.sizeof(_OutputV1),
                cpu_integrated.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                cpu_accepted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                cpu_rejected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                cpu_integrated.size,
            )
            error.value = b""
            status = int(
                self._library.oaf_native_cpu_masked_weighted_v1(
                    ctypes.byref(request),
                    ctypes.byref(cpu_output),
                    error,
                    ctypes.sizeof(error),
                )
            )
            if status != 0:
                raise MetalIntegrationError(self._error_text(error, status))
            cpu_integrated[cpu_accepted == 0] = np.nan
            finite_equal = bool(
                np.array_equal(np.isfinite(integrated), np.isfinite(cpu_integrated))
            )
            counts_equal = bool(
                np.array_equal(accepted, cpu_accepted)
                and np.array_equal(rejected, cpu_rejected)
            )
            finite = np.isfinite(cpu_integrated) & np.isfinite(integrated)
            difference = np.abs(integrated[finite] - cpu_integrated[finite]).astype(
                np.float64, copy=False
            )
            max_abs = float(np.max(difference)) if difference.size else 0.0
            rmse = (
                float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
                if difference.size
                else 0.0
            )
            max_abs_limit = 2.0e-6
            rmse_limit = 2.0e-7
            parity = {
                "performed": True,
                "domain": "normalized-float32",
                "pixelsCompared": int(np.count_nonzero(finite)),
                "finiteMaskEqual": finite_equal,
                "rejectionCountsEqual": counts_equal,
                "maxAbs": max_abs,
                "rmse": rmse,
                "maxAbsLimit": max_abs_limit,
                "rmseLimit": rmse_limit,
                "passed": finite_equal
                and counts_equal
                and max_abs <= max_abs_limit
                and rmse <= rmse_limit,
            }
            if not parity["passed"]:
                raise MetalIntegrationError(
                    "masked CPU/Metal numerical gate failed: "
                    f"finite={finite_equal}, counts={counts_equal}, "
                    f"maxAbs={max_abs:.9g}, rmse={rmse:.9g}"
                )
        device_name = bytes(stats.device_name).split(b"\0", 1)[0].decode(
            "utf-8", errors="replace"
        )
        integrated *= np.float32(input_normalization)
        return NativeTileResult(
            integrated=integrated,
            accepted=accepted,
            rejected=portable_rejected,
            stats={
                "deviceName": device_name,
                "executedOnGpu": stats.executed_on_gpu == 1,
                "wallSeconds": stats.wall_seconds,
                "gpuSeconds": stats.gpu_seconds,
                "submittedBufferBytes": stats.submitted_buffer_bytes,
                "recommendedWorkingSetBytes": stats.recommended_working_set_bytes,
                "maximumBufferBytes": stats.maximum_buffer_bytes,
                "inputNormalizationScale": 1.0 / input_normalization,
                "outputRescale": input_normalization,
                "parityGate": parity,
            },
        )


def _requested_selection(
    requested: str, hardware: HardwareProfile, tuning: ExecutionTuning
) -> tuple[str, str | None]:
    if requested not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported ordinary integration backend: {requested}")
    if requested == "portable-cpu":
        return "portable-cpu", None
    if not hardware.apple_silicon:
        return "portable-cpu", "Metal was requested or preferred on a non-Apple-silicon host"
    m3_tuning_ready = (
        hardware.m3_pro_tuned and tuning.profile_id == "apple-m3-pro-tuned-v1"
    )
    if requested == "m3-pro-tuned" and not m3_tuning_ready:
        return (
            "generic-apple-metal",
            "M3 Pro tuning was not applied because the chip or memory gate did not match",
        )
    if requested == "generic-apple-metal":
        return requested, None
    return (
        "m3-pro-tuned" if m3_tuning_ready else "generic-apple-metal",
        None,
    )


def _cpu_with_receipt(
    expressions: tuple[FrameExpression, ...],
    output_path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None,
    parameters: IntegrationParameters,
    requested_backend: str,
    hardware: HardwareProfile,
    tuning: ExecutionTuning,
    fallback_reason: str | None,
    quality_weights: Sequence[float] | None,
    map_paths: IntegrationMapPaths | None,
    durable: bool = True,
    selection_policy: str = "fallback",
) -> IntegrationResult:
    result = integrate_expressions(
        expressions,
        output_path,
        metadata=metadata,
        parameters=parameters,
        quality_weights=quality_weights,
        map_paths=map_paths,
        native_threads=max(1, tuning.cpu_workers),
        durable=durable,
    )
    return replace(
        result,
        execution={
            **dict(result.execution),
            "requestedBackend": requested_backend,
            "selectedBackend": "portable-cpu",
            "selectionPolicy": selection_policy,
            "acceleratorUsed": False,
            "fallbackReason": fallback_reason,
            "hardware": hardware.serializable(),
            "performanceProfile": tuning.serializable(),
            "inputFrameCount": len(expressions),
            "inputFramesTruncated": False,
            "cpuWorkersUsed": 1,
            "tileRows": result.tile_rows,
            "fastMath": False,
        },
    )


NATIVE_CPU_AUTO_REASON = (
    "auto selected the native CPU kernels: per-pixel rejection statistics "
    "dominate ordinary integration and the multithreaded native reduction "
    "avoids the Metal path's host-side normalization, buffer copies and "
    "parity reruns; request generic-apple-metal or m3-pro-tuned explicitly to "
    "use the Metal reduction"
)


def integrate_registered_group(
    expressions: Iterable[FrameExpression],
    output_path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    parameters: IntegrationParameters | None = None,
    requested_backend: str = "auto",
    native_library_path: str | os.PathLike[str] | None = None,
    metal_source_path: str | os.PathLike[str] | None = None,
    hardware: HardwareProfile | None = None,
    tuning: ExecutionTuning | None = None,
    metal_executor: NativeMetalExecutor | None = None,
    metal_unavailable_reason: str | None = None,
    quality_weights: Sequence[float] | None = None,
    map_paths: IntegrationMapPaths | None = None,
    durable: bool = True,
) -> IntegrationResult:
    """Integrate one registered Light group with explicit, audited fallback.

    ``auto`` prefers the multithreaded native CPU kernels whenever they are
    loaded: the measured M-series cost of ordinary integration is the
    per-pixel rejection statistics, which both paths compute on the CPU, and
    the Metal weighted mean adds host-side normalization, copies and parity
    reruns that cost more than its GPU time.  Explicit Metal requests still
    run the audited Metal path.
    """

    integration = parameters or IntegrationParameters()
    integration.validate()
    canonical = tuple(_canonical_expression(item) for item in expressions)
    if not canonical:
        raise CalibrationError("NO_INPUTS", "at least one integration input is required")
    profile = hardware or detect_hardware()
    selected_tuning = tuning or select_execution_tuning(profile)
    selected, selection_note = _requested_selection(
        requested_backend, profile, selected_tuning
    )
    if requested_backend == "auto" and load_native_kernels() is not None:
        selected, selection_note = "portable-cpu", NATIVE_CPU_AUTO_REASON
    if selected == "portable-cpu":
        return _cpu_with_receipt(
            canonical,
            output_path,
            metadata=metadata,
            parameters=integration,
            requested_backend=requested_backend,
            hardware=profile,
            tuning=selected_tuning,
            fallback_reason=selection_note,
            quality_weights=quality_weights,
            map_paths=map_paths,
            durable=durable,
            selection_policy=(
                "native-cpu-kernels-preferred"
                if selection_note == NATIVE_CPU_AUTO_REASON
                else "requested" if requested_backend == "portable-cpu" else "fallback"
            ),
        )
    if len(canonical) > MAXIMUM_METAL_REJECTION_FRAMES:
        return _cpu_with_receipt(
            canonical,
            output_path,
            metadata=metadata,
            parameters=integration,
            requested_backend=requested_backend,
            hardware=profile,
            tuning=selected_tuning,
            fallback_reason=(
                f"{len(canonical)} frames exceed the full-stack Metal policy limit of "
                f"{MAXIMUM_METAL_REJECTION_FRAMES}; all frames used by portable CPU"
            ),
            quality_weights=quality_weights,
            map_paths=map_paths,
            durable=durable,
        )

    destination = Path(output_path)
    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output", path=str(destination)
        )
    temporary = _temporary_output(destination)
    map_destinations = map_paths.resolved(destination) if map_paths is not None else {}
    map_temporaries = {
        name: _temporary_output(path) for name, path in map_destinations.items()
    }
    try:
        owns_executor = False
        executor = metal_executor
        if executor is None and metal_unavailable_reason is not None:
            return _cpu_with_receipt(
                canonical,
                output_path,
                metadata=metadata,
                parameters=integration,
                requested_backend=requested_backend,
                hardware=profile,
                tuning=selected_tuning,
                fallback_reason=metal_unavailable_reason,
                quality_weights=quality_weights,
                map_paths=map_paths,
            )
        if executor is None:
            try:
                executor = NativeMetalExecutor(
                    library_path=native_library_path,
                    metal_source_path=metal_source_path,
                )
                owns_executor = True
            except MetalIntegrationError as error:
                return _cpu_with_receipt(
                    canonical,
                    output_path,
                    metadata=metadata,
                    parameters=integration,
                    requested_backend=requested_backend,
                    hardware=profile,
                    tuning=selected_tuning,
                    fallback_reason=str(error),
                    quality_weights=quality_weights,
                    map_paths=map_paths,
                )

        try:
            executor_context = executor if owns_executor else nullcontext(executor)
            with executor_context as active_executor, ExitStack() as stack:
                sources = _open_expression_sources(stack, canonical)
                shape = _validate_expression_shapes(canonical, sources)
                height, width = shape
                (
                    weights64,
                    serialized_noise_weights,
                    serialized_quality_weights,
                    serialized_weights,
                ) = _combined_integration_weights(
                    canonical, sources, shape, integration, quality_weights
                )
                rejection_sigma_floor = _estimate_rejection_sigma_floor(
                    canonical, sources, shape, integration
                )
                transient_model = _prepare_transient_rejection(
                    canonical, sources, shape, integration, weights64,
                    workers=max(1, selected_tuning.cpu_workers),
                )
                weights = np.ascontiguousarray(weights64, dtype=np.float32)
                scales = np.ones((len(canonical), 2, 2), dtype=np.float32)
                offsets = np.zeros((len(canonical), 2, 2), dtype=np.float32)
                inflight = max(1, selected_tuning.gpu_inflight_buffers)
                configured_cpu_workers = max(1, selected_tuning.cpu_workers)
                if integration.max_memory_bytes == DEFAULT_MEMORY_BUDGET:
                    integration_memory_budget = (
                        selected_tuning.integration_memory_bytes
                    )
                    memory_budget_source = "hardware-profile"
                else:
                    integration_memory_budget = min(
                        selected_tuning.integration_memory_bytes,
                        integration.max_memory_bytes,
                    )
                    memory_budget_source = "explicit-integration-cap"
                (
                    preparation_worker_limit,
                    tile_rows,
                    parallel_bytes_per_row,
                ) = _metal_memory_plan(
                    width=width,
                    height=height,
                    frame_count=len(canonical),
                    configured_workers=configured_cpu_workers,
                    inflight_buffers=inflight,
                    target_tile_rows=selected_tuning.integration_tile_rows,
                    memory_budget_bytes=integration_memory_budget,
                )
                output_metadata = dict(metadata or {})
                output_metadata.setdefault("OAFSTATE", "UNSOLVED_WORKING")
                output_metadata.setdefault("OAFNFRM", len(canonical))
                output_metadata.setdefault("OAFREJ", integration.sigma_clip)
                output_metadata.setdefault("OAFACCEL", "METAL")
                statistics = _StatsAccumulator()
                accepted_total = 0
                rejected_total = 0
                gpu_wall = 0.0
                gpu_seconds = 0.0
                submitted = 0
                device_name = ""
                input_scale_min = math.inf
                input_scale_max = 0.0
                output_rescale_min = math.inf
                output_rescale_max = 0.0
                parity_samples: list[dict[str, Any]] = []
                pending: deque[_PendingTile] = deque()
                row_ranges = [
                    (first_row, min(height, first_row + tile_rows))
                    for first_row in range(0, height, tile_rows)
                ]
                parity_first_rows = {
                    row_ranges[0][0],
                    row_ranges[len(row_ranges) // 2][0],
                    row_ranges[-1][0],
                }
                cpu_workers_used = min(preparation_worker_limit, len(row_ranges))
                kernel_threads_per_worker = max(
                    1, configured_cpu_workers // max(1, cpu_workers_used)
                )

                map_writers: dict[str, FitsFloatWriter] = {}

                def consume(item: _PendingTile, writer: FitsFloatWriter) -> None:
                    nonlocal accepted_total, rejected_total, gpu_wall, gpu_seconds
                    nonlocal submitted, device_name
                    nonlocal input_scale_min, input_scale_max
                    nonlocal output_rescale_min, output_rescale_max
                    tile = item.future.result()
                    writer.write_rows(item.first_row, tile.integrated)
                    if map_writers:
                        accepted = tile.accepted.astype(np.float32)
                        map_writers["acceptedSampleCount"].write_rows(
                            item.first_row, accepted
                        )
                        map_writers["coverageFraction"].write_rows(
                            item.first_row,
                            accepted / np.float32(len(canonical)),
                        )
                        map_writers["rejectionCount"].write_rows(
                            item.first_row, tile.rejected.astype(np.float32)
                        )
                    statistics.update(tile.integrated)
                    accepted_total += int(np.sum(tile.accepted, dtype=np.uint64))
                    rejected_total += int(np.sum(tile.rejected, dtype=np.uint64))
                    gpu_wall += float(tile.stats["wallSeconds"])
                    gpu_seconds += float(tile.stats["gpuSeconds"])
                    submitted += int(tile.stats["submittedBufferBytes"])
                    device_name = str(tile.stats["deviceName"])
                    input_scale = float(tile.stats["inputNormalizationScale"])
                    output_rescale = float(tile.stats["outputRescale"])
                    input_scale_min = min(input_scale_min, input_scale)
                    input_scale_max = max(input_scale_max, input_scale)
                    output_rescale_min = min(output_rescale_min, output_rescale)
                    output_rescale_max = max(output_rescale_max, output_rescale)
                    tile_parity = tile.stats["parityGate"]
                    if bool(tile_parity.get("performed")):
                        parity_samples.append(
                            {
                                "firstRow": item.first_row,
                                "rowCount": int(tile.integrated.shape[0]),
                                **dict(tile_parity),
                            }
                        )

                def prepare(
                    first_row: int, stop: int
                ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
                    values = np.empty(
                        (len(canonical), stop - first_row, width),
                        dtype=np.float32,
                    )
                    for index, expression in enumerate(canonical):
                        values[index] = _expression_rows(
                            expression,
                            sources,
                            first_row,
                            stop,
                            division_floor=integration.division_floor,
                        )
                    finite, _, accepted = _ordinary_integration_tile(
                        values, integration, rejection_sigma_floor, transient_model,
                        first_row, kernel_threads_per_worker,
                    )
                    return values, np.asarray(finite & ~accepted, dtype=np.uint8)

                with ExitStack() as output_stack:
                    writer = output_stack.enter_context(
                        FitsFloatWriter(
                            temporary, shape, output_metadata, durable=durable
                        )
                    )
                    map_metadata = {
                        "acceptedSampleCount": {
                            "IMAGETYP": "Integration accepted-sample count",
                            "OAFMAP": "ACCEPTED_COUNT",
                            "OAFNFRM": len(canonical),
                        },
                        "coverageFraction": {
                            "IMAGETYP": "Integration coverage fraction",
                            "OAFMAP": "COVERAGE",
                            "OAFNFRM": len(canonical),
                        },
                        "rejectionCount": {
                            "IMAGETYP": "Integration rejection count",
                            "OAFMAP": "REJECTION_COUNT",
                            "OAFNFRM": len(canonical),
                            "OAFREJ": integration.sigma_clip,
                        },
                    }
                    for name, temporary_map in map_temporaries.items():
                        map_writers[name] = output_stack.enter_context(
                            FitsFloatWriter(
                                temporary_map, shape, map_metadata[name],
                                durable=durable,
                            )
                        )
                    with ThreadPoolExecutor(
                        max_workers=inflight,
                        thread_name_prefix="oaf-metal",
                    ) as gpu_pool, ThreadPoolExecutor(
                        max_workers=cpu_workers_used,
                        thread_name_prefix="oaf-mask",
                    ) as cpu_pool:
                        ranges = iter(row_ranges)
                        prepared: deque[
                            tuple[
                                int,
                                Future[
                                    tuple[NDArray[np.float32], NDArray[np.uint8]]
                                ],
                            ]
                        ] = deque()

                        def submit_preparation() -> bool:
                            try:
                                first_row, stop = next(ranges)
                            except StopIteration:
                                return False
                            prepared.append(
                                (first_row, cpu_pool.submit(prepare, first_row, stop))
                            )
                            return True

                        for _ in range(cpu_workers_used):
                            submit_preparation()
                        while prepared:
                            first_row, preparation = prepared.popleft()
                            values, rejection_mask = preparation.result()
                            submit_preparation()
                            future = gpu_pool.submit(
                                active_executor.run_masked_tile,
                                values,
                                rejection_mask,
                                image_height=height,
                                first_row=first_row,
                                scales=scales,
                                offsets=offsets,
                                weights=weights,
                                compare_cpu=first_row in parity_first_rows,
                            )
                            pending.append(_PendingTile(first_row, future))
                            if len(pending) >= inflight:
                                consume(pending.popleft(), writer)
                        while pending:
                            consume(pending.popleft(), writer)
                final_statistics = statistics.result()
                if final_statistics.finite_pixels == 0:
                    raise MetalIntegrationError("Metal integration produced no finite pixels")
                if len(parity_samples) != len(parity_first_rows):
                    raise MetalIntegrationError(
                        "CPU/Metal parity sampling did not cover first/middle/last tiles"
                    )
                parity_pixels = sum(
                    int(sample["pixelsCompared"]) for sample in parity_samples
                )
                parity_max_abs = max(
                    float(sample["maxAbs"]) for sample in parity_samples
                )
                parity_rmse = (
                    math.sqrt(
                        sum(
                            float(sample["rmse"]) ** 2
                            * int(sample["pixelsCompared"])
                            for sample in parity_samples
                        )
                        / parity_pixels
                    )
                    if parity_pixels
                    else 0.0
                )
                parity_max_abs_limit = min(
                    float(sample["maxAbsLimit"]) for sample in parity_samples
                )
                parity_rmse_limit = min(
                    float(sample["rmseLimit"]) for sample in parity_samples
                )
                parity_finite_equal = all(
                    bool(sample["finiteMaskEqual"]) for sample in parity_samples
                )
                parity_counts_equal = all(
                    bool(sample["rejectionCountsEqual"])
                    for sample in parity_samples
                )
                parity_gate: Mapping[str, Any] = {
                    "scope": "first-middle-last-tiles",
                    "performed": True,
                    "domain": "normalized-float32",
                    "sampledTileCount": len(parity_samples),
                    "sampledFirstRows": sorted(parity_first_rows),
                    "pixelsCompared": parity_pixels,
                    "finiteMaskEqual": parity_finite_equal,
                    "rejectionCountsEqual": parity_counts_equal,
                    "maxAbs": parity_max_abs,
                    "rmse": parity_rmse,
                    "maxAbsLimit": parity_max_abs_limit,
                    "rmseLimit": parity_rmse_limit,
                    "passed": parity_finite_equal
                    and parity_counts_equal
                    and parity_max_abs <= parity_max_abs_limit
                    and parity_rmse <= parity_rmse_limit,
                    "samples": sorted(
                        parity_samples, key=lambda sample: int(sample["firstRow"])
                    ),
                }
                if not parity_gate["passed"]:
                    raise MetalIntegrationError(
                        "aggregate CPU/Metal first/middle/last parity gate failed"
                    )
            _atomic_publish_file(temporary, destination)
            for name, map_destination in map_destinations.items():
                _atomic_publish_file(map_temporaries[name], map_destination)
            return IntegrationResult(
                output_path=str(destination),
                shape=shape,
                frame_count=len(canonical),
                tile_rows=tile_rows,
                weights=serialized_weights,
                rejected_samples=rejected_total,
                accepted_samples=accepted_total,
                statistics=final_statistics,
                noise_weights=serialized_noise_weights,
                quality_weights=serialized_quality_weights,
                map_paths={
                    name: str(path) for name, path in map_destinations.items()
                },
                output_sha256=writer.sha256,
                map_sha256={
                    name: map_writer.sha256
                    for name, map_writer in map_writers.items()
                    if map_writer.sha256 is not None
                },
                execution={
                    "requestedBackend": requested_backend,
                    "selectedBackend": selected,
                    "acceleratorUsed": True,
                    "fallbackReason": selection_note,
                    "hardware": profile.serializable(),
                    "performanceProfile": selected_tuning.serializable(),
                    "nativeAbiVersion": NATIVE_ABI_VERSION,
                    "executorScope": "integration-group" if owns_executor else "pipeline",
                    "nativeLibrary": {
                        "path": str(executor.library_path),
                        "sha256": _sha256(executor.library_path),
                    },
                    "deviceName": device_name,
                    "inputFrameCount": len(canonical),
                    "inputFramesTruncated": False,
                    "maximumNativeFrames": MAXIMUM_NATIVE_REJECTION_FRAMES,
                    "maximumExactMetalFrames": MAXIMUM_METAL_REJECTION_FRAMES,
                    "algorithm": "cpu-mad-mask-plus-metal-full-stack-weighted-v1",
                    "rejectionMask": {
                        "producer": "portable-cpu",
                        "method": "median-mad-sigma",
                        "sigma": integration.sigma_clip,
                        "scope": "all-frames-per-pixel",
                        "partialMeanBatching": False,
                        "sigmaFloor": rejection_sigma_floor.serializable(),
                        "spatialTransients": transient_model.serializable(),
                    },
                    "tileRows": tile_rows,
                    "integrationMemoryBudgetBytes": integration_memory_budget,
                    "memoryBudgetSource": memory_budget_source,
                    "memoryModel": {
                        "kind": "conservative-live-buffer-v2",
                        "preparationBytesPerSample": PREPARATION_BYTES_PER_SAMPLE,
                        "metalInflightBytesPerSample": METAL_INFLIGHT_BYTES_PER_SAMPLE,
                        "fixedBytesPerPixel": METAL_FIXED_BYTES_PER_PIXEL,
                        "parallelBytesPerRow": parallel_bytes_per_row,
                        "estimatedPeakBytes": parallel_bytes_per_row * tile_rows,
                    },
                    "inflightBuffers": inflight,
                    "peakInflightBuffers": min(inflight, len(row_ranges)),
                    "configuredCpuWorkers": configured_cpu_workers,
                    "cpuWorkersUsed": cpu_workers_used,
                    "kernelThreadsPerWorker": kernel_threads_per_worker,
                    "tilesSubmitted": math.ceil(height / tile_rows),
                    "gpuWallSecondsSum": gpu_wall,
                    "gpuSecondsSum": gpu_seconds,
                    "submittedBufferBytes": submitted,
                    "fastMath": False,
                    "numericDomain": {
                        "parity": "normalized-float32",
                        "inputNormalizationScaleMinimum": input_scale_min,
                        "inputNormalizationScaleMaximum": input_scale_max,
                        "outputRescaleMinimum": output_rescale_min,
                        "outputRescaleMaximum": output_rescale_max,
                    },
                    "parityGate": dict(parity_gate),
                },
            )
        except (MetalIntegrationError, OSError, ValueError) as error:
            if temporary.exists():
                temporary.unlink()
            for map_temporary in map_temporaries.values():
                if map_temporary.exists():
                    map_temporary.unlink()
            return _cpu_with_receipt(
                canonical,
                output_path,
                metadata=metadata,
                parameters=integration,
                requested_backend=requested_backend,
                hardware=profile,
                tuning=selected_tuning,
                fallback_reason=f"Metal execution rejected: {error}",
                quality_weights=quality_weights,
                map_paths=map_paths,
            )
    finally:
        if temporary.exists():
            temporary.unlink()
        for map_temporary in map_temporaries.values():
            if map_temporary.exists():
                map_temporary.unlink()


__all__ = [
    "NATIVE_CPU_AUTO_REASON",
    "MAXIMUM_METAL_REJECTION_FRAMES",
    "MAXIMUM_NATIVE_REJECTION_FRAMES",
    "MetalIntegrationError",
    "NATIVE_ABI_VERSION",
    "NativeMetalExecutor",
    "SUPPORTED_BACKENDS",
    "integrate_registered_group",
]
