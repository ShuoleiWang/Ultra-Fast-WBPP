"""ctypes bridge for the portable multithreaded CPU kernels.

The native library (``engine/native``) exports three kernels that reproduce the
NumPy reference arithmetic of the ordinary mono pipeline value for value:

* ``warp_lanczos3``: normalized, domain-bounded 6x6 Lanczos-3 affine warp,
* ``mad_rejection``: full-stack per-pixel median/MAD sigma clipping,
* ``masked_weighted_mean``: exact Float64 frame-order weighted mean.

Every caller keeps its NumPy implementation as the portable fallback.  The
kernels are optional: a missing library, a library without the symbols, an ABI
mismatch, or ``OPENASTROFLOW_DISABLE_NATIVE_KERNELS=1`` leaves the Python path
in charge.  Because ctypes releases the GIL for the duration of a call, several
Python worker threads can run kernels concurrently, and each kernel additionally
splits its own work across ``threads`` native threads.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
import os
from pathlib import Path
import stat
import sys
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray


NATIVE_ABI_VERSION = 1
DISABLE_ENVIRONMENT_VARIABLE = "OPENASTROFLOW_DISABLE_NATIVE_KERNELS"
WARP_KERNEL_ID = "native-cpu-lanczos3-warp-v2"
MAD_KERNEL_ID = "native-cpu-mad-rejection-v1"
MEAN_KERNEL_ID = "native-cpu-masked-mean-v1"
TILE_OFFSET_KERNEL_ID = "native-cpu-tile-offsets-v1"
_MAXIMUM_KERNEL_THREADS = 64


class NativeKernelError(RuntimeError):
    """A native kernel could not be loaded or refused a request."""


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libopenastroflow_native.dylib"
    if os.name == "nt":
        return "openastroflow_native.dll"
    return "libopenastroflow_native.so"


def _candidate_library_paths(explicit: str | os.PathLike[str] | None) -> tuple[Path, ...]:
    """Return trusted, existing native-library candidates in priority order."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    environment = os.environ.get("OPENASTROFLOW_NATIVE_LIBRARY", "").strip()
    if environment:
        candidates.append(Path(environment).expanduser())
    module = Path(__file__).resolve()
    name = _library_filename()
    candidates.append(module.parent / "native" / name)
    try:
        repository = module.parents[4]
    except IndexError:
        repository = module.parent
    candidates.extend(
        (
            repository / "build" / "native-metal" / name,
            repository / "build" / "native" / name,
            repository / "engine" / "native" / "build" / name,
        )
    )
    found = ctypes.util.find_library("openastroflow_native")
    if found and os.path.isabs(found):
        candidates.append(Path(found))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if not resolved.is_file():
            continue
        if os.name != "nt":
            metadata = resolved.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o022:
                continue
            if metadata.st_uid not in {0, os.getuid()}:
                continue
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return tuple(unique)


class _WarpRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("source_width", ctypes.c_uint32),
        ("source_height", ctypes.c_uint32),
        ("output_width", ctypes.c_uint32),
        ("first_row", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("source_samples", ctypes.POINTER(ctypes.c_float)),
        ("source_sample_count", ctypes.c_size_t),
        ("inverse", ctypes.c_double * 6),
        ("domain_scale", ctypes.c_float),
        ("reserved_scale", ctypes.c_float),
    ]


class _MadRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("frame_count", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("minimum_rejection_frames", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("frame_major_samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_size_t),
        ("sigma_clip", ctypes.c_float),
        ("group_sigma_floor", ctypes.c_float),
        ("absolute_floor", ctypes.c_float),
        ("epsilon_floor", ctypes.c_float),
    ]


class _MeanRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("frame_count", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("frame_major_samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_size_t),
        ("frame_major_accepted", ctypes.POINTER(ctypes.c_uint8)),
        ("accepted_count", ctypes.c_size_t),
        ("frame_weights", ctypes.POINTER(ctypes.c_double)),
        ("weight_count", ctypes.c_size_t),
    ]


class _MeanOutputV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("integrated", ctypes.POINTER(ctypes.c_float)),
        ("accepted_samples", ctypes.POINTER(ctypes.c_uint16)),
        ("rejected_samples", ctypes.POINTER(ctypes.c_uint16)),
        ("pixel_capacity", ctypes.c_size_t),
    ]


class _TileOffsetRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("tile_count", ctypes.c_uint32),
        ("minimum_samples", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("target", ctypes.POINTER(ctypes.c_double)),
        ("reference", ctypes.POINTER(ctypes.c_double)),
        ("sample_count", ctypes.c_size_t),
        ("boundaries", ctypes.POINTER(ctypes.c_uint64)),
        ("boundary_count", ctypes.c_size_t),
        ("scale", ctypes.c_double),
        ("lower_quantile", ctypes.c_double),
        ("upper_quantile", ctypes.c_double),
        ("residual_clip_sigma", ctypes.c_double),
    ]


class _TileOffsetOutputV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset", ctypes.POINTER(ctypes.c_double)),
        ("count", ctypes.POINTER(ctypes.c_uint32)),
        ("residual_mad", ctypes.POINTER(ctypes.c_double)),
        ("valid", ctypes.POINTER(ctypes.c_uint8)),
        ("tile_capacity", ctypes.c_size_t),
    ]


_ERROR_BYTES = 1024
_REQUIRED_SYMBOLS = (
    "oaf_native_abi_version",
    "oaf_native_cpu_warp_lanczos3_v1",
    "oaf_native_cpu_mad_rejection_v1",
    "oaf_native_cpu_masked_mean_v1",
    "oaf_native_cpu_tile_offsets_v1",
    "oaf_native_default_kernel_threads_v1",
)


def _thread_count(value: int | None) -> int:
    if value is None:
        return default_kernel_threads()
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("native kernel thread count must be a positive integer")
    return min(int(value), _MAXIMUM_KERNEL_THREADS)


class NativeKernels:
    """Validated owner of one loaded native kernel library."""

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        self.library_path = Path(library_path).expanduser().resolve(strict=True)
        try:
            self._library = ctypes.CDLL(str(self.library_path))
        except OSError as error:
            raise NativeKernelError(
                f"unable to load native kernels from {self.library_path}: {error}"
            ) from error
        missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(self._library, name)]
        if missing:
            raise NativeKernelError(
                "native library misses portable kernel symbols: " + ", ".join(missing)
            )
        library = self._library
        library.oaf_native_abi_version.argtypes = []
        library.oaf_native_abi_version.restype = ctypes.c_uint32
        version = int(library.oaf_native_abi_version())
        if version != NATIVE_ABI_VERSION:
            raise NativeKernelError(
                f"native ABI version {version} differs from {NATIVE_ABI_VERSION}"
            )
        error_arguments = [ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
        library.oaf_native_cpu_warp_lanczos3_v1.argtypes = [
            ctypes.POINTER(_WarpRequestV1),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            *error_arguments,
        ]
        library.oaf_native_cpu_warp_lanczos3_v1.restype = ctypes.c_int
        library.oaf_native_cpu_mad_rejection_v1.argtypes = [
            ctypes.POINTER(_MadRequestV1),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            *error_arguments,
        ]
        library.oaf_native_cpu_mad_rejection_v1.restype = ctypes.c_int
        library.oaf_native_cpu_masked_mean_v1.argtypes = [
            ctypes.POINTER(_MeanRequestV1),
            ctypes.POINTER(_MeanOutputV1),
            *error_arguments,
        ]
        library.oaf_native_cpu_masked_mean_v1.restype = ctypes.c_int
        library.oaf_native_cpu_tile_offsets_v1.argtypes = [
            ctypes.POINTER(_TileOffsetRequestV1),
            ctypes.POINTER(_TileOffsetOutputV1),
            *error_arguments,
        ]
        library.oaf_native_cpu_tile_offsets_v1.restype = ctypes.c_int
        library.oaf_native_default_kernel_threads_v1.argtypes = []
        library.oaf_native_default_kernel_threads_v1.restype = ctypes.c_uint32
        self.hardware_threads = max(1, int(library.oaf_native_default_kernel_threads_v1()))

    @staticmethod
    def _raise(buffer: ctypes.Array[Any], status: int, kernel: str) -> None:
        detail = bytes(buffer.value).decode("utf-8", errors="replace").strip()
        raise NativeKernelError(
            f"{kernel} failed with status {status}: {detail or 'no detail'}"
        )

    def warp_lanczos3(
        self,
        source: NDArray[np.float32],
        inverse: NDArray[np.float64],
        *,
        first_row: int,
        row_count: int,
        output_width: int,
        domain_scale: float,
        threads: int | None = None,
    ) -> NDArray[np.float32]:
        """Warp one band of output rows from a native Float32 source image."""

        source_values = np.ascontiguousarray(source, dtype=np.float32)
        if source_values.ndim != 2:
            raise ValueError("warp source must be a two-dimensional image")
        matrix = np.ascontiguousarray(inverse, dtype=np.float64)
        if matrix.shape == (3, 3):
            if not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=0.0):
                raise ValueError("warp inverse must be affine")
            matrix = matrix[:2]
        if matrix.shape != (2, 3):
            raise ValueError("warp inverse must be a 2x3 or affine 3x3 matrix")
        if row_count < 1 or output_width < 1 or first_row < 0:
            raise ValueError("warp band geometry must be positive")
        height, width = source_values.shape
        request = _WarpRequestV1()
        request.struct_size = ctypes.sizeof(_WarpRequestV1)
        request.source_width = int(width)
        request.source_height = int(height)
        request.output_width = int(output_width)
        request.first_row = int(first_row)
        request.row_count = int(row_count)
        request.threads = _thread_count(threads)
        request.source_samples = source_values.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        request.source_sample_count = source_values.size
        request.inverse = (ctypes.c_double * 6)(*(float(value) for value in matrix.ravel()))
        request.domain_scale = float(domain_scale)
        destination = np.empty((int(row_count), int(output_width)), dtype=np.float32)
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.oaf_native_cpu_warp_lanczos3_v1(
                ctypes.byref(request),
                destination.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                destination.size,
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            self._raise(error, status, "native Lanczos-3 warp")
        return destination

    def mad_rejection(
        self,
        samples: NDArray[np.float32],
        *,
        sigma_clip: float,
        minimum_rejection_frames: int,
        group_sigma_floor: float,
        absolute_floor: float,
        epsilon_floor: float,
        threads: int | None = None,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
        """Return (accepted, center) for a frame-major (F, R, W) Float32 stack."""

        values = np.ascontiguousarray(samples, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("MAD rejection samples must be frame-major 3-D")
        frames, rows, width = values.shape
        request = _MadRequestV1()
        request.struct_size = ctypes.sizeof(_MadRequestV1)
        request.frame_count = int(frames)
        request.row_count = int(rows)
        request.width = int(width)
        request.minimum_rejection_frames = int(minimum_rejection_frames)
        request.threads = _thread_count(threads)
        request.frame_major_samples = values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.sample_count = values.size
        request.sigma_clip = float(sigma_clip)
        request.group_sigma_floor = float(group_sigma_floor)
        request.absolute_floor = float(absolute_floor)
        request.epsilon_floor = float(epsilon_floor)
        accepted = np.empty(values.shape, dtype=np.uint8)
        center = np.empty((rows, width), dtype=np.float32)
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.oaf_native_cpu_mad_rejection_v1(
                ctypes.byref(request),
                accepted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                accepted.size,
                center.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                center.size,
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            self._raise(error, status, "native MAD rejection")
        return accepted.view(np.bool_), center

    def masked_weighted_mean(
        self,
        samples: NDArray[np.float32],
        accepted: NDArray[Any],
        weights: NDArray[np.float64],
        *,
        threads: int | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint16], NDArray[np.uint16]]:
        """Return (integrated, accepted_count, rejected_count) for one tile."""

        values = np.ascontiguousarray(samples, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("masked mean samples must be frame-major 3-D")
        mask = np.ascontiguousarray(accepted)
        if mask.dtype == np.bool_:
            mask = mask.view(np.uint8)
        elif mask.dtype != np.uint8:
            mask = np.ascontiguousarray(mask != 0).view(np.uint8)
        if mask.shape != values.shape:
            raise ValueError("masked mean acceptance mask must match the samples")
        frame_weights = np.ascontiguousarray(weights, dtype=np.float64).ravel()
        frames, rows, width = values.shape
        if frame_weights.size != frames:
            raise ValueError("masked mean weight count differs from the frame count")
        request = _MeanRequestV1()
        request.struct_size = ctypes.sizeof(_MeanRequestV1)
        request.frame_count = int(frames)
        request.row_count = int(rows)
        request.width = int(width)
        request.threads = _thread_count(threads)
        request.frame_major_samples = values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.sample_count = values.size
        request.frame_major_accepted = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        request.accepted_count = mask.size
        request.frame_weights = frame_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.weight_count = frame_weights.size
        integrated = np.empty((rows, width), dtype=np.float32)
        accepted_count = np.empty((rows, width), dtype=np.uint16)
        rejected_count = np.empty((rows, width), dtype=np.uint16)
        output = _MeanOutputV1()
        output.struct_size = ctypes.sizeof(_MeanOutputV1)
        output.integrated = integrated.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output.accepted_samples = accepted_count.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16)
        )
        output.rejected_samples = rejected_count.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16)
        )
        output.pixel_capacity = integrated.size
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.oaf_native_cpu_masked_mean_v1(
                ctypes.byref(request),
                ctypes.byref(output),
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            self._raise(error, status, "native masked weighted mean")
        return integrated, accepted_count, rejected_count

    def tile_offsets(
        self,
        target: NDArray[np.float64],
        reference: NDArray[np.float64],
        boundaries: NDArray[np.uint64],
        *,
        scale: float,
        lower_quantile: float,
        upper_quantile: float,
        minimum_samples: int,
        residual_clip_sigma: float,
        threads: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uint32], NDArray[np.float64], NDArray[np.bool_]]:
        """Return (offset, count, residual_mad, valid) per concatenated tile."""

        target_values = np.ascontiguousarray(target, dtype=np.float64).ravel()
        reference_values = np.ascontiguousarray(reference, dtype=np.float64).ravel()
        bounds = np.ascontiguousarray(boundaries, dtype=np.uint64).ravel()
        if target_values.size != reference_values.size:
            raise ValueError("tile offset target and reference sample counts differ")
        if bounds.size < 2:
            raise ValueError("tile offsets require at least one tile")
        tiles = int(bounds.size - 1)
        request = _TileOffsetRequestV1()
        request.struct_size = ctypes.sizeof(_TileOffsetRequestV1)
        request.tile_count = tiles
        request.minimum_samples = int(minimum_samples)
        request.threads = _thread_count(threads)
        request.target = target_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.reference = reference_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.sample_count = target_values.size
        request.boundaries = bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        request.boundary_count = bounds.size
        request.scale = float(scale)
        request.lower_quantile = float(lower_quantile)
        request.upper_quantile = float(upper_quantile)
        request.residual_clip_sigma = float(residual_clip_sigma)
        offset = np.empty(tiles, dtype=np.float64)
        count = np.empty(tiles, dtype=np.uint32)
        residual_mad = np.empty(tiles, dtype=np.float64)
        valid = np.empty(tiles, dtype=np.uint8)
        output = _TileOffsetOutputV1()
        output.struct_size = ctypes.sizeof(_TileOffsetOutputV1)
        output.offset = offset.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output.count = count.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        output.residual_mad = residual_mad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output.valid = valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        output.tile_capacity = tiles
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.oaf_native_cpu_tile_offsets_v1(
                ctypes.byref(request), ctypes.byref(output), error, ctypes.sizeof(error)
            )
        )
        if status != 0:
            self._raise(error, status, "native tile offsets")
        return offset, count, residual_mad, valid.view(np.bool_)


_LOCK = threading.Lock()
_CACHE: dict[str, NativeKernels | None] = {}


def native_kernels_disabled() -> bool:
    return os.environ.get(DISABLE_ENVIRONMENT_VARIABLE, "").strip() not in {"", "0", "false", "no"}


def load_native_kernels(
    library_path: str | os.PathLike[str] | None = None,
) -> NativeKernels | None:
    """Return the cached native kernels, or ``None`` when the NumPy path must run.

    Candidates are tried in priority order and the first library exporting the
    complete kernel symbol set wins, so an older library without the kernels
    never blocks a newer sibling.
    """

    if native_kernels_disabled():
        return None
    key = os.path.normcase(str(Path(library_path).expanduser())) if library_path else ""
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        loaded: NativeKernels | None = None
        for candidate in _candidate_library_paths(library_path):
            try:
                loaded = NativeKernels(candidate)
                break
            except NativeKernelError:
                continue
        _CACHE[key] = loaded
        return loaded


def reset_native_kernel_cache() -> None:
    """Forget loaded libraries; used by tests that switch libraries or the env."""

    with _LOCK:
        _CACHE.clear()


@lru_cache(maxsize=1)
def default_kernel_threads() -> int:
    """Threads per kernel call: the hardware profile's CPU worker count."""

    try:
        from .hardware import detect_hardware
        from .performance_profile import select_execution_tuning

        workers = int(select_execution_tuning(detect_hardware()).cpu_workers)
    except Exception:  # pragma: no cover - defensive: tuning never blocks pixels
        workers = int(os.cpu_count() or 1)
    return max(1, min(workers, _MAXIMUM_KERNEL_THREADS))


__all__ = [
    "DISABLE_ENVIRONMENT_VARIABLE",
    "MAD_KERNEL_ID",
    "MEAN_KERNEL_ID",
    "NATIVE_ABI_VERSION",
    "NativeKernelError",
    "NativeKernels",
    "TILE_OFFSET_KERNEL_ID",
    "WARP_KERNEL_ID",
    "default_kernel_threads",
    "load_native_kernels",
    "native_kernels_disabled",
    "reset_native_kernel_cache",
]
