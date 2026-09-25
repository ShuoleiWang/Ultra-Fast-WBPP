"""Resampling one calibrated Light onto the reference grid (Lanczos-3, or an exact half turn), and the provenance it records."""

from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..calibration.inputs import numeric_domain_metadata
from ..native_kernels import WARP_KERNEL_ID, load_native_kernels
from ..platform import remove_file
from .integration import (
    CalibrationError,
    FitsFloatWriter,
    FitsFrame,
    FrameInfo,
    PixelStatistics,
    MemoryFrame,
)
from .parameters import OUTPUT_STATE, PixelTransform


# v3: tap weights come from the deterministic 2048-interval table
# (`lanczos_table.py`, cubic Lagrange interpolation of nodes computed by the
# module's own series) instead of the host's libm; the interpolation contract
# is unchanged, the weights differ from the exact ones by less than 1e-12
# before Float32 rounding, and the result no longer depends on the C library.
LANCZOS3_REGISTRATION_ALGORITHM = (
    "normalized-lanczos-3-domain-union-support-clamp-v3-table2048"
)


NUMPY_WARP_KERNEL_ID = "numpy-lanczos3-warp-v3-table2048"


# Native warp per output row: Float32 band, finite mask, statistics selection
# and the big-endian conversion inside the FITS writer.
NATIVE_WARP_BYTES_PER_PIXEL = 16


def _inverse_coordinates(
    inverse: NDArray[np.float64],
    output_x: NDArray[np.float64],
    output_y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map output pixel coordinates to input coordinates.

    The evaluation order ``((m00*x) + (m01*y)) + m02`` and, for a projective
    map, the division by ``((m20*x) + (m21*y)) + m22`` are the arithmetic
    contract the native warp kernel reproduces value for value.
    """

    input_x = inverse[0, 0] * output_x + inverse[0, 1] * output_y + inverse[0, 2]
    input_y = inverse[1, 0] * output_x + inverse[1, 1] * output_y + inverse[1, 2]
    if not (inverse[2, 0] == 0.0 and inverse[2, 1] == 0.0 and inverse[2, 2] == 1.0):
        denominator = inverse[2, 0] * output_x + inverse[2, 1] * output_y + inverse[2, 2]
        input_x = input_x / denominator
        input_y = input_y / denominator
    return input_x, input_y


def _exact_half_turn_translation(
    transform: PixelTransform, shape: tuple[int, int]
) -> tuple[int, int] | None:
    """Recognize only an integer half-turn up to float64 arithmetic roundoff.

    Coefficient checks alone can hide a displacement on a long image axis.
    Bound the residual at every corner, separately for x and y; the affine
    residual between corners cannot exceed those bounds. Eight float64 epsilons
    allow numerical noise (including sin(pi)), not a fitted angular tolerance.
    """
    matrix = transform.validated_matrix()
    if not np.array_equal(matrix[2], (0.0, 0.0, 1.0)):
        return None
    linear_error = matrix[:2, :2] + np.eye(2)
    roundoff = 8.0 * np.finfo(np.float64).eps
    if np.any(np.abs(linear_error) > roundoff):
        return None
    translation = np.rint(matrix[:2, 2])
    translation_error = matrix[:2, 2] - translation
    if np.any(
        np.abs(translation_error) > roundoff * np.maximum(1.0, np.abs(translation))
    ):
        return None
    height, width = shape
    corners = np.asarray(
        ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)),
        dtype=np.float64,
    )
    residual = corners @ linear_error.T + translation_error
    coordinate_scale = np.maximum(1.0, np.maximum(corners, np.abs(translation)))
    if np.any(np.abs(residual) > roundoff * coordinate_scale):
        return None
    return int(translation[0]), int(translation[1])


def _registration_provenance(
    transform: PixelTransform, shape: tuple[int, int], resampler: str
) -> dict[str, str]:
    if transform.is_identity:
        actual, algorithm = "identity-exact", "identity-exact-copy-v1"
    elif _exact_half_turn_translation(transform, shape) is not None:
        actual, algorithm = "half-turn-exact", "half-turn-integer-copy-v1"
    else:
        actual = resampler
        algorithm = (
            LANCZOS3_REGISTRATION_ALGORITHM
            if resampler == "lanczos-3-clamped"
            else "bilinear-2x2-v1"
        )
    return {"resampler": actual, "resamplerAlgorithm": algorithm}


def _registration_metadata(
    info: FrameInfo,
    transform: PixelTransform,
    *,
    resampler: str,
    source_exposure_seconds: float | None = None,
) -> dict[str, Any]:
    actual_resampler = _registration_provenance(transform, info.shape, resampler)[
        "resampler"
    ]
    uses_lanczos = actual_resampler == "lanczos-3-clamped"
    return {
        "IMAGETYP": "Registered Light",
        "FILTER": info.filter_name,
        "OBJECT": info.target,
        "EXPTIME": info.exposure_seconds,
        "OAFSTATE": OUTPUT_STATE,
        "OAFREG": (
            "IDENTITY"
            if transform.is_identity
            else "PROJECTIVE" if transform.is_projective else "AFFINE"
        ),
        "OAFRSAMP": actual_resampler.upper(),
        "OAFRCLMP": "DOMAIN_UNION_SUPPORT" if uses_lanczos else None,
        "OAFRMARG": 2 if uses_lanczos else 0,
        "OAFSRCEX": source_exposure_seconds,
        **numeric_domain_metadata(info),
    }


def _registration_bytes_per_pixel(
    transform: PixelTransform, resampler: str, shape: tuple[int, int]
) -> int:
    if transform.is_identity:
        return 24
    if _exact_half_turn_translation(transform, shape) is not None:
        # Includes the destination tile, scaled FITS conversion and previous
        # iteration's values/statistics buffers while the next tile is read.
        return 32
    return 192 if resampler == "lanczos-3-clamped" else 112


def _read_half_turn_rows(
    source: Any, y0: int, y1: int, translation: tuple[int, int]
) -> NDArray[np.float32]:
    height, width = source.shape
    tx, ty = translation
    values = np.full((y1 - y0, width), np.nan, dtype=np.float32)
    left, right = max(0, tx - width + 1), min(width, tx + 1)
    top, bottom = max(y0, ty - height + 1), min(y1, ty + 1)
    if left < right and top < bottom:
        # Read at most this output tile's row count. Reversal is a view of the
        # physical Float32 values, so it neither interpolates nor clamps them.
        rows = source.read_rows(ty - bottom + 1, ty - top + 1)
        values[top - y0 : bottom - y0, left:right] = rows[
            ::-1, tx - right + 1 : tx - left + 1
        ][:, ::-1]
    return values


def _open_registration_source(source: Path | MemoryFrame) -> Any:
    if isinstance(source, MemoryFrame):
        return nullcontext(source)
    return FitsFrame(source)


def _register_frame(
    source: Path | MemoryFrame,
    destination: Path,
    transform: PixelTransform,
    info: FrameInfo,
    *,
    max_memory_bytes: int,
    resampler: str,
    source_exposure_seconds: float | None = None,
    native_threads: int | None = None,
    execution: dict[str, Any] | None = None,
    durable: bool = True,
) -> PixelStatistics:
    """Resample one calibrated Light (file or in-memory) into a new FITS.

    Exact identity and integer half-turn transforms copy pixels.  General
    transforms use the native multithreaded Lanczos-3 kernel when it is
    available and the budget holds the decoded source; otherwise the NumPy
    reference resampler runs on bounded coordinate tiles.  Both produce
    value-identical output.  ``execution`` receives the backend actually used.
    """

    matrix = transform.validated_matrix()
    inverse = np.linalg.inv(matrix)
    with _open_registration_source(source) as frame:
        height, width = frame.shape
        is_identity = transform.is_identity
        half_turn = _exact_half_turn_translation(transform, frame.shape)
        kernels = None
        source_values: NDArray[np.float32] | None = None
        domain_scale = frame.info.normalized_unit_scale
        if is_identity:
            warp_backend = "identity-copy"
        elif half_turn is not None:
            warp_backend = "half-turn-copy"
        else:
            warp_backend = "numpy"
        if (
            warp_backend == "numpy"
            and resampler == "lanczos-3-clamped"
            and domain_scale is not None
            and math.isfinite(domain_scale)
            and domain_scale > 0.0
        ):
            kernels = load_native_kernels()
        if kernels is not None:
            decoded_bytes = 0 if isinstance(frame, MemoryFrame) else height * width * 4
            native_row_bytes = width * NATIVE_WARP_BYTES_PER_PIXEL
            if decoded_bytes + native_row_bytes <= max_memory_bytes:
                warp_backend = "native-cpu"
                tile_rows = max(
                    1,
                    min(height, (max_memory_bytes - decoded_bytes) // native_row_bytes),
                )
            else:
                kernels = None
        if warp_backend != "native-cpu":
            bytes_per_pixel = _registration_bytes_per_pixel(
                transform, resampler, frame.shape
            )
            bytes_per_row = width * bytes_per_pixel
            if bytes_per_row > max_memory_bytes:
                raise CalibrationError(
                    "MEMORY_BUDGET_TOO_SMALL", "one registration row exceeds memory budget"
                )
            tile_rows = max(1, min(height, max_memory_bytes // bytes_per_row))
        temporary = destination.with_name(f".{destination.name}.partial")
        if temporary.exists() or os.path.lexists(temporary):
            raise CalibrationError(
                "OUTPUT_EXISTS", "registration temporary already exists", path=str(temporary)
            )
        stats_min = math.inf
        stats_max = -math.inf
        stats_sum = 0.0
        finite_total = 0
        invalid_total = 0
        digest: str | None = None
        try:
            with FitsFloatWriter(
                temporary,
                frame.shape,
                _registration_metadata(
                    info,
                    transform,
                    resampler=resampler,
                    source_exposure_seconds=source_exposure_seconds,
                ),
                durable=durable,
            ) as writer:
                if warp_backend == "native-cpu":
                    source_values = frame.full_values()
                for y0 in range(0, height, tile_rows):
                    y1 = min(height, y0 + tile_rows)
                    if is_identity:
                        values = frame.read_rows(y0, y1)
                    elif half_turn is not None:
                        values = _read_half_turn_rows(frame, y0, y1, half_turn)
                    elif warp_backend == "native-cpu":
                        assert kernels is not None and source_values is not None
                        values = kernels.warp_lanczos3(
                            source_values,
                            inverse,
                            first_row=y0,
                            row_count=y1 - y0,
                            output_width=width,
                            domain_scale=float(domain_scale),
                            threads=native_threads,
                        )
                    else:
                        output_y = np.arange(y0, y1, dtype=np.float64)[:, None]
                        output_x = np.arange(width, dtype=np.float64)[None, :]
                        input_x, input_y = _inverse_coordinates(inverse, output_x, output_y)
                        input_x = np.broadcast_to(input_x, (y1 - y0, width))
                        input_y = np.broadcast_to(input_y, (y1 - y0, width))
                        if resampler == "lanczos-3-clamped":
                            values = frame.sample_lanczos3_clamped(input_x, input_y)
                        else:
                            values = frame.sample_bilinear(input_x, input_y)
                    finite = np.isfinite(values)
                    count = int(np.count_nonzero(finite))
                    finite_total += count
                    invalid_total += int(values.size - count)
                    if count:
                        selected = values[finite]
                        stats_min = min(stats_min, float(np.min(selected)))
                        stats_max = max(stats_max, float(np.max(selected)))
                        stats_sum += float(np.sum(selected, dtype=np.float64))
                    writer.write_rows(y0, values)
            digest = writer.sha256
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise CalibrationError(
                    "OUTPUT_EXISTS",
                    "refusing to overwrite registered frame",
                    path=str(destination),
                ) from error
            remove_file(temporary, missing_ok=False)
        finally:
            remove_file(temporary)
    if execution is not None:
        execution.update(
            {
                "warpBackend": warp_backend,
                "warpKernel": (
                    WARP_KERNEL_ID
                    if warp_backend == "native-cpu"
                    else NUMPY_WARP_KERNEL_ID
                    if warp_backend == "numpy" and resampler == "lanczos-3-clamped"
                    else warp_backend
                ),
                "tileRows": tile_rows,
                "nativeThreads": native_threads if warp_backend == "native-cpu" else None,
                "sha256": digest,
            }
        )
    return PixelStatistics(
        finite_pixels=finite_total,
        invalid_pixels=invalid_total,
        minimum=stats_min if finite_total else None,
        maximum=stats_max if finite_total else None,
        mean=stats_sum / finite_total if finite_total else None,
    )
