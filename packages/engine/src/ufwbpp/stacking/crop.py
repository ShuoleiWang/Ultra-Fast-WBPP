"""The rectangle every registered frame covers, and cropping masters to it."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..platform import remove_file
from .integration import CalibrationError, FitsFloatWriter, FitsFrame, FrameInfo, PixelStatistics
from .parameters import PixelTransform
from .warp import _exact_half_turn_translation, _inverse_coordinates


def histogram_rectangle(
    heights: NDArray[np.int64], row_index: int
) -> tuple[int, int, int, int, int] | None:
    best: tuple[int, int, int, int, int] | None = None
    stack: list[tuple[int, int]] = []
    width = int(heights.size)
    if not width:
        return None
    # Repeating a height cannot push or pop the monotone stack. Registered
    # footprints typically leave thousands of adjacent columns at the same
    # height, so locate run boundaries in NumPy and visit only those in Python.
    # Keep the original coordinates and sentinel: candidates and their exact
    # lexicographic tie-break remain identical, including masks with holes.
    boundaries = np.flatnonzero(heights[1:] != heights[:-1]) + 1
    positions = [0, *boundaries.tolist(), width]
    levels = [int(heights[0]), *heights[boundaries].tolist(), 0]
    for x, height in zip(positions, levels, strict=True):
        start = x
        while stack and stack[-1][1] > height:
            left, previous_height = stack.pop()
            area = previous_height * (x - left)
            candidate = (
                area,
                row_index - previous_height + 1,
                left,
                row_index + 1,
                x,
            )
            if best is None or candidate > best:
                best = candidate
            start = left
        if height and (not stack or stack[-1][1] < height):
            stack.append((start, height))
    return best


def _shared_auto_crop(
    light_groups: Mapping[str, Sequence[Path]],
    light_info: Mapping[Path, FrameInfo],
    transforms: Mapping[Path, PixelTransform],
    *,
    enabled: bool,
    max_memory_bytes: int,
    resampler: str,
) -> tuple[tuple[int, int, int, int] | None, dict[str, tuple[int, int, int, int]]]:
    """Intersect the per-filter valid crops of one registration run.

    Returns ``(shared_crop, crop_by_filter)``; ``shared_crop`` is ``None`` when
    auto-crop is disabled.  All groups must share the registered geometry,
    which is the source frame shape of every Light of the run.
    """

    if not enabled or not light_groups:
        return None, {}
    shapes = {
        filter_name: light_info[paths[0]].shape for filter_name, paths in light_groups.items()
    }
    if len(set(shapes.values())) != 1:
        raise CalibrationError(
            "REGISTRATION_GEOMETRY_MISMATCH",
            "filter groups of one run must share the registered frame geometry: "
            + ", ".join(f"{name}={shape}" for name, shape in sorted(shapes.items())),
        )
    crops: dict[str, tuple[int, int, int, int]] = {}
    for filter_name, paths in sorted(light_groups.items()):
        crops[filter_name] = _common_valid_crop(
            shapes[filter_name],
            [transforms[path] for path in paths],
            max_memory_bytes=max_memory_bytes,
            resampler=resampler,
        )
    shared = (
        max(crop[0] for crop in crops.values()),
        max(crop[1] for crop in crops.values()),
        min(crop[2] for crop in crops.values()),
        min(crop[3] for crop in crops.values()),
    )
    if shared[2] <= shared[0] or shared[3] <= shared[1]:
        raise CalibrationError(
            "AUTOCROP_TOO_SMALL",
            "the filter groups of this run have no common fully covered rectangle",
        )
    return shared, crops


def _common_valid_crop(
    shape: tuple[int, int],
    transforms: Sequence[PixelTransform],
    *,
    max_memory_bytes: int,
    resampler: str,
) -> tuple[int, int, int, int]:
    if not transforms:
        raise CalibrationError("NO_REGISTERED_INPUTS", "no registration transforms")
    height, width = shape
    bytes_per_row = width * 56
    if bytes_per_row > max_memory_bytes:
        raise CalibrationError(
            "MEMORY_BUDGET_TOO_SMALL", "one crop-mask row exceeds memory budget"
        )
    tile_rows = max(1, min(height, max_memory_bytes // bytes_per_row))
    inverses = []
    margins = []
    for transform in transforms:
        half_turn = _exact_half_turn_translation(transform, shape)
        if half_turn is not None:
            tx, ty = half_turn
            # Match the exact integer map actually used by the row copier.
            inverses.append(
                np.asarray(
                    ((-1, 0, tx), (0, -1, ty), (0, 0, 1)), dtype=np.float64
                )
            )
        else:
            inverses.append(np.linalg.inv(transform.validated_matrix()))
        margins.append(
            2
            if not transform.is_identity
            and half_turn is None
            and resampler == "lanczos-3-clamped"
            else 0
        )
    # Every frame's valid pixels of a row form one run (the frame's footprint
    # is convex), so each frame contributes a per-row interval, found with
    # the exact per-pixel test by bisection; the common mask row is the
    # intersection of the intervals.  Rows the bisection cannot bracket are
    # evaluated pixel by pixel, so every mask row equals the dense one.
    heights = np.zeros(width, dtype=np.int64)
    best: tuple[int, int, int, int, int] | None = None
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        first = np.zeros(y1 - y0, dtype=np.int64)
        last = np.full(y1 - y0, width - 1, dtype=np.int64)
        for inverse, interpolation_margin in zip(inverses, margins, strict=True):
            frame_first, frame_last = _valid_row_runs(
                inverse, interpolation_margin, y0, y1, width, height
            )
            first = np.maximum(first, frame_first)
            last = np.minimum(last, frame_last)
        for local_y in range(y1 - y0):
            run_first = int(first[local_y])
            run_last = int(last[local_y])
            if run_last < run_first:
                heights.fill(0)
            else:
                heights[:run_first] = 0
                heights[run_first : run_last + 1] += 1
                heights[run_last + 1 :] = 0
            candidate = histogram_rectangle(heights, y0 + local_y)
            if candidate is not None and (best is None or candidate > best):
                best = candidate
    if best is None or best[0] == 0:
        raise CalibrationError(
            "AUTOCROP_EMPTY", "registered inputs have no common finite rectangle"
        )
    _, top, left, bottom, right = best
    return top, left, bottom, right


def _valid_row_runs(
    inverse: NDArray[np.float64],
    interpolation_margin: int,
    y0: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Per-row ``(first, last)`` valid output columns of one frame, rows y0..y1.

    A column is valid when its inverse-mapped coordinates lie inside the
    source frame less the interpolation margin, the dense mask test.  Rows
    without a valid column return ``last < first``.
    """

    rows = y1 - y0
    output_y = np.arange(y0, y1, dtype=np.float64)
    x_low = float(interpolation_margin)
    x_high = float(width - 1 - interpolation_margin)
    y_low = float(interpolation_margin)
    y_high = float(height - 1 - interpolation_margin)

    def valid_at(columns: NDArray[np.int64]) -> NDArray[np.bool_]:
        input_x, input_y = _inverse_coordinates(
            inverse, columns.astype(np.float64), output_y
        )
        return (
            (input_x >= x_low) & (input_x <= x_high)
            & (input_y >= y_low) & (input_y <= y_high)
        )

    # Analytic run centre of each row (the geometry is affine or a
    # near-identity projective map, so the source-centre column bounds the
    # run's interior); the exact test decides whether it is inside.
    centre_x = 0.5 * (x_low + x_high)
    centre_y = 0.5 * (y_low + y_high)
    a, b, c = inverse[0]
    d, e, f = inverse[1]
    g, h, i = inverse[2]
    # Solve for x with y fixed: the column mapping to input (centre_x, *) or,
    # when the x row is degenerate, to input (*, centre_y).
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator_x = centre_x * (h * output_y + i) - (b * output_y + c)
        denominator_x = a - centre_x * g
        numerator_y = centre_y * (h * output_y + i) - (e * output_y + f)
        denominator_y = d - centre_y * g
        guess = np.where(
            np.abs(denominator_x) >= np.abs(denominator_y),
            numerator_x / denominator_x,
            numerator_y / denominator_y,
        )
    guess = np.where(np.isfinite(guess), guess, 0.5 * (width - 1))
    inside = np.clip(np.rint(guess), 0, width - 1).astype(np.int64)
    bracketed = valid_at(inside)
    first = np.zeros(rows, dtype=np.int64)
    last = np.full(rows, -1, dtype=np.int64)
    if np.any(bracketed):
        # Bisection on the single run: ``low`` is outside (or virtual -1 /
        # width), ``high`` is inside.
        low = np.full(rows, -1, dtype=np.int64)
        high = inside.copy()
        while True:
            active = high - low > 1
            if not np.any(active):
                break
            middle = (low + high) // 2
            probe = valid_at(np.clip(middle, 0, width - 1))
            high = np.where(active & probe, middle, high)
            low = np.where(active & ~probe, middle, low)
        first_bisect = high
        low = inside.copy()
        high = np.full(rows, width, dtype=np.int64)
        while True:
            active = high - low > 1
            if not np.any(active):
                break
            middle = (low + high) // 2
            probe = valid_at(np.clip(middle, 0, width - 1))
            low = np.where(active & probe, middle, low)
            high = np.where(active & ~probe, middle, high)
        last_bisect = low
        first = np.where(bracketed, first_bisect, first)
        last = np.where(bracketed, last_bisect, last)
    unbracketed = np.flatnonzero(~bracketed)
    if unbracketed.size:
        # Rows whose centre guess is outside (edge rows of a tilted frame,
        # or an unexpected geometry): the dense row test decides, a bounded
        # number of rows at a time.
        output_x = np.arange(width, dtype=np.float64)[None, :]
        chunk = max(1, (8 * 1024 * 1024) // (width * 40))
        for start in range(0, unbracketed.size, chunk):
            selected = unbracketed[start : start + chunk]
            input_x, input_y = _inverse_coordinates(
                inverse, output_x, output_y[selected][:, None]
            )
            dense = (
                (input_x >= x_low) & (input_x <= x_high)
                & (input_y >= y_low) & (input_y <= y_high)
            )
            any_valid = np.any(dense, axis=1)
            dense_first = np.argmax(dense, axis=1)
            dense_last = width - 1 - np.argmax(dense[:, ::-1], axis=1)
            first[selected] = np.where(any_valid, dense_first, 0)
            last[selected] = np.where(any_valid, dense_last, -1)
    return first, last


def _crop_fits(
    source_path: Path,
    destination: Path,
    crop: tuple[int, int, int, int],
    metadata: Mapping[str, Any],
    *,
    max_memory_bytes: int,
    durable: bool = True,
) -> tuple[PixelStatistics, str | None]:
    """Crop one FITS into a new file; returns statistics and the writer digest."""
    top, left, bottom, right = crop
    with FitsFrame(source_path) as source:
        height, width = source.shape
        if not (0 <= top < bottom <= height and 0 <= left < right <= width):
            raise CalibrationError("AUTOCROP_INVALID", "crop is outside image bounds")
        output_shape = (bottom - top, right - left)
        bytes_per_row = output_shape[1] * 12
        if bytes_per_row > max_memory_bytes:
            raise CalibrationError(
                "MEMORY_BUDGET_TOO_SMALL", "one crop row exceeds memory budget"
            )
        tile_rows = max(1, min(output_shape[0], max_memory_bytes // bytes_per_row))
        temporary = destination.with_name(f".{destination.name}.partial")
        if temporary.exists() or os.path.lexists(temporary):
            raise CalibrationError("OUTPUT_EXISTS", "crop temporary already exists")
        finite_total = 0
        invalid_total = 0
        total = 0.0
        minimum = math.inf
        maximum = -math.inf
        try:
            with FitsFloatWriter(
                temporary, output_shape, metadata, durable=durable
            ) as writer:
                output_y = 0
                for source_y in range(top, bottom, tile_rows):
                    source_y1 = min(bottom, source_y + tile_rows)
                    values = source.read_rows(source_y, source_y1)[:, left:right]
                    finite = np.isfinite(values)
                    count = int(np.count_nonzero(finite))
                    finite_total += count
                    invalid_total += int(values.size - count)
                    if count:
                        selected = values[finite]
                        minimum = min(minimum, float(np.min(selected)))
                        maximum = max(maximum, float(np.max(selected)))
                        total += float(np.sum(selected, dtype=np.float64))
                    writer.write_rows(output_y, values)
                    output_y += values.shape[0]
            digest = writer.sha256
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise CalibrationError(
                    "OUTPUT_EXISTS", "refusing to overwrite master", path=str(destination)
                ) from error
            remove_file(temporary, missing_ok=False)
        finally:
            remove_file(temporary)
    return PixelStatistics(
        finite_pixels=finite_total,
        invalid_pixels=invalid_total,
        minimum=minimum if finite_total else None,
        maximum=maximum if finite_total else None,
        mean=total / finite_total if finite_total else None,
    ), digest
