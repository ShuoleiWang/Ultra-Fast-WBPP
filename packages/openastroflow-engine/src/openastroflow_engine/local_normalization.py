"""Conservative, bounded-memory local normalization for registered sky frames.

The model is fitted in a common registered coordinate system as

    reference(x, y) ~= scale(x, y) * target(x, y) + offset(x, y)

on a coarse grid.  Bright stars and high-surface-brightness structure are
excluded from the fit with paired quantile and residual clipping.  Scale and
offset grids are bilinearly interpolated, so the correction contains no spatial
frequencies smaller than the configured tile size.  Every non-reference model
must pass sample, correlation, coefficient and residual-improvement gates.

This is intentionally an opt-in, conservative implementation.  It does not
claim algorithmic equivalence to PixInsight LocalNormalization; ambiguous data
fails closed instead of silently applying a global median normalization.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from lightframeqc.content_hash import file_sha256
from .calibration import CalibrationError, FitsFloatWriter, FitsFrame, PixelStatistics
from .platform import remove_tree


LOCAL_NORMALIZATION_VERSION = "paired-background-grid-v1"


@dataclass(frozen=True, slots=True)
class LocalNormalizationParameters:
    enabled: bool = False
    tile_size_pixels: int = 256
    minimum_frames: int = 3
    minimum_samples_per_tile: int = 512
    minimum_valid_tiles: int = 12
    minimum_valid_tile_fraction: float = 0.70
    background_upper_quantile: float = 0.70
    background_lower_quantile: float = 0.05
    residual_clip_sigma: float = 3.5
    minimum_tile_correlation: float = 0.15
    minimum_median_correlation: float = 0.40
    minimum_residual_improvement: float = 0.10
    minimum_scale: float = 0.75
    maximum_scale: float = 1.35
    maximum_scale_grid_span: float = 0.20
    max_memory_bytes: int = 256 * 1024**2

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("local normalization enabled must be boolean")
        for name, floor in (
            ("tile_size_pixels", 64),
            ("minimum_frames", 2),
            ("minimum_samples_per_tile", 32),
            ("minimum_valid_tiles", 4),
            ("max_memory_bytes", 1024),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"{name} must be an integer >= {floor}")
        for name in (
            "minimum_valid_tile_fraction",
            "background_upper_quantile",
            "background_lower_quantile",
            "minimum_tile_correlation",
            "minimum_median_correlation",
            "minimum_residual_improvement",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.background_lower_quantile >= self.background_upper_quantile:
            raise ValueError("background quantile bounds are reversed")
        if not math.isfinite(self.residual_clip_sigma) or self.residual_clip_sigma < 2.0:
            raise ValueError("residual_clip_sigma must be finite and >= 2")
        if not (
            math.isfinite(self.minimum_scale)
            and math.isfinite(self.maximum_scale)
            and 0 < self.minimum_scale < 1 < self.maximum_scale
        ):
            raise ValueError("scale bounds must straddle one")
        if (
            not math.isfinite(self.maximum_scale_grid_span)
            or not 0 < self.maximum_scale_grid_span <= 0.5
        ):
            raise ValueError("maximum_scale_grid_span must be in (0, 0.5]")

    def serializable(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "algorithm": LOCAL_NORMALIZATION_VERSION,
            "tileSizePixels": self.tile_size_pixels,
            "minimumFrames": self.minimum_frames,
            "minimumSamplesPerTile": self.minimum_samples_per_tile,
            "minimumValidTiles": self.minimum_valid_tiles,
            "minimumValidTileFraction": self.minimum_valid_tile_fraction,
            "backgroundQuantiles": [
                self.background_lower_quantile,
                self.background_upper_quantile,
            ],
            "residualClipSigma": self.residual_clip_sigma,
            "minimumTileCorrelation": self.minimum_tile_correlation,
            "minimumMedianCorrelation": self.minimum_median_correlation,
            "minimumResidualImprovement": self.minimum_residual_improvement,
            "scaleBounds": [self.minimum_scale, self.maximum_scale],
            "maximumScaleGridSpan": self.maximum_scale_grid_span,
            "maxMemoryBytes": self.max_memory_bytes,
            "equivalentToPixInsight": False,
        }


@dataclass(frozen=True, slots=True)
class LocalNormalizationResult:
    normalized_paths: tuple[str, ...]
    receipt_path: str
    receipt: dict[str, Any]


def _sha256(path: Path) -> str:
    return "sha256:" + file_sha256(path)


def _canonical_matrix(value: Sequence[Sequence[float]] | None) -> NDArray[np.float64]:
    matrix = np.eye(3, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise CalibrationError("LOCAL_NORMALIZATION_TRANSFORM_INVALID", "transform must be finite 3x3")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise CalibrationError("LOCAL_NORMALIZATION_TRANSFORM_INVALID", "transform is singular")
    return matrix


def _project(matrix: NDArray[np.float64], x: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    output_x = np.full(np.shape(x), np.nan, dtype=np.float64)
    output_y = np.full(np.shape(y), np.nan, dtype=np.float64)
    numerator_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    numerator_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    np.divide(numerator_x, denominator, out=output_x, where=valid)
    np.divide(numerator_y, denominator, out=output_y, where=valid)
    valid &= np.isfinite(output_x) & np.isfinite(output_y)
    return output_x, output_y, valid


def _robust_fit(
    target: NDArray[np.float32],
    reference: NDArray[np.float32],
    parameters: LocalNormalizationParameters,
) -> tuple[float, float, float, float, float, int, float] | None:
    x = np.asarray(target, dtype=np.float64).ravel()
    y = np.asarray(reference, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < parameters.minimum_samples_per_tile:
        return None
    x_bounds = np.quantile(
        x, [parameters.background_lower_quantile, parameters.background_upper_quantile]
    )
    y_bounds = np.quantile(
        y, [parameters.background_lower_quantile, parameters.background_upper_quantile]
    )
    background = (
        (x >= x_bounds[0]) & (x <= x_bounds[1])
        & (y >= y_bounds[0]) & (y <= y_bounds[1])
    )
    x = x[background]
    y = y[background]
    if x.size < parameters.minimum_samples_per_tile:
        return None

    def fit(x_value: NDArray[np.float64], y_value: NDArray[np.float64]) -> tuple[float, float, float]:
        x_center = float(np.median(x_value))
        y_center = float(np.median(y_value))
        dx = x_value - x_center
        dy = y_value - y_center
        variance_x = float(np.dot(dx, dx))
        variance_y = float(np.dot(dy, dy))
        if variance_x <= np.finfo(np.float64).eps or variance_y <= np.finfo(np.float64).eps:
            return math.nan, math.nan, math.nan
        covariance = float(np.dot(dx, dy))
        scale = covariance / variance_x
        offset = y_center - scale * x_center
        correlation = covariance / math.sqrt(variance_x * variance_y)
        return scale, offset, correlation

    scale, offset, correlation = fit(x, y)
    if not all(math.isfinite(value) for value in (scale, offset, correlation)):
        return None
    residual = y - (scale * x + offset)
    center = float(np.median(residual))
    sigma = 1.4826 * float(np.median(np.abs(residual - center)))
    if math.isfinite(sigma) and sigma > np.finfo(np.float64).eps:
        keep = np.abs(residual - center) <= parameters.residual_clip_sigma * sigma
        if int(np.count_nonzero(keep)) >= parameters.minimum_samples_per_tile:
            x = x[keep]
            y = y[keep]
            scale, offset, correlation = fit(x, y)
            if not all(math.isfinite(value) for value in (scale, offset, correlation)):
                return None
            residual = y - (scale * x + offset)
    before = y - x
    before_mad = 1.4826 * float(np.median(np.abs(before - np.median(before))))
    after_mad = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    if before_mad <= np.finfo(np.float64).eps:
        improvement = 1.0 if after_mad <= np.finfo(np.float64).eps else 0.0
    else:
        improvement = 1.0 - after_mad / before_mad
    return scale, offset, correlation, before_mad, after_mad, int(x.size), float(np.median(x))


def _fill_grid(grid: NDArray[np.float64], valid: NDArray[np.bool_]) -> NDArray[np.float64]:
    result = grid.copy()
    if not np.any(valid):
        raise CalibrationError("LOCAL_NORMALIZATION_MODEL_INSUFFICIENT", "model grid has no valid nodes")
    fallback = float(np.median(result[valid]))
    result[~valid] = fallback
    # Iterative four-neighbour fill preserves local structure without adding a
    # heavyweight interpolation dependency.  Remaining edge holes use median.
    known = valid.copy()
    original_missing = ~valid
    for _ in range(result.shape[0] + result.shape[1]):
        changed = False
        for y, x in zip(*np.where(~known), strict=True):
            neighbours: list[float] = []
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < result.shape[0] and 0 <= xx < result.shape[1] and known[yy, xx]:
                    neighbours.append(float(result[yy, xx]))
            if neighbours:
                result[y, x] = float(np.median(neighbours))
                known[y, x] = True
                changed = True
        if not changed or np.all(known):
            break
    result[original_missing & ~known] = fallback
    return result


def _grid_coordinates(length: int, tile_size: int) -> NDArray[np.float64]:
    starts = np.arange(0, length, tile_size, dtype=np.int64)
    stops = np.minimum(starts + tile_size, length)
    return (starts + stops - 1).astype(np.float64) / 2.0


def _sample_grid(
    grid: NDArray[np.float64],
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    x_clipped = np.clip(x, x_nodes[0], x_nodes[-1])
    y_clipped = np.clip(y, y_nodes[0], y_nodes[-1])
    x_hi = np.searchsorted(x_nodes, x_clipped, side="right")
    y_hi = np.searchsorted(y_nodes, y_clipped, side="right")
    x_hi = np.clip(x_hi, 1, len(x_nodes) - 1) if len(x_nodes) > 1 else np.zeros_like(x_hi)
    y_hi = np.clip(y_hi, 1, len(y_nodes) - 1) if len(y_nodes) > 1 else np.zeros_like(y_hi)
    x_lo = x_hi - 1 if len(x_nodes) > 1 else x_hi
    y_lo = y_hi - 1 if len(y_nodes) > 1 else y_hi
    x_denominator = x_nodes[x_hi] - x_nodes[x_lo]
    y_denominator = y_nodes[y_hi] - y_nodes[y_lo]
    wx = np.zeros_like(x_clipped)
    wy = np.zeros_like(y_clipped)
    np.divide(x_clipped - x_nodes[x_lo], x_denominator, out=wx, where=x_denominator != 0)
    np.divide(y_clipped - y_nodes[y_lo], y_denominator, out=wy, where=y_denominator != 0)
    top = grid[y_lo, x_lo] * (1.0 - wx) + grid[y_lo, x_hi] * wx
    bottom = grid[y_hi, x_lo] * (1.0 - wx) + grid[y_hi, x_hi] * wx
    return top * (1.0 - wy) + bottom * wy


def _fit_model(
    target: FitsFrame,
    reference: FitsFrame,
    input_to_reference: NDArray[np.float64],
    reference_input_to_common: NDArray[np.float64],
    parameters: LocalNormalizationParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any], NDArray[np.float64], NDArray[np.float64]]:
    reference_height, reference_width = reference.shape
    x_nodes = _grid_coordinates(reference_width, parameters.tile_size_pixels)
    y_nodes = _grid_coordinates(reference_height, parameters.tile_size_pixels)
    if len(x_nodes) < 2 or len(y_nodes) < 2:
        raise CalibrationError(
            "LOCAL_NORMALIZATION_MODEL_INSUFFICIENT",
            "frame is too small for a two-dimensional local model",
        )
    bytes_per_tile = parameters.tile_size_pixels**2 * 80
    if bytes_per_tile > parameters.max_memory_bytes:
        raise CalibrationError(
            "LOCAL_NORMALIZATION_MEMORY_BUDGET",
            "one model tile exceeds max_memory_bytes",
        )
    inverse = np.linalg.inv(input_to_reference)
    reference_inverse = np.linalg.inv(reference_input_to_common)
    shape = (len(y_nodes), len(x_nodes))
    scale_grid = np.full(shape, np.nan, dtype=np.float64)
    offset_grid = np.full(shape, np.nan, dtype=np.float64)
    correlations = np.full(shape, np.nan, dtype=np.float64)
    before_values = np.full(shape, np.nan, dtype=np.float64)
    after_values = np.full(shape, np.nan, dtype=np.float64)
    sample_counts = np.zeros(shape, dtype=np.int64)
    target_centers = np.full(shape, np.nan, dtype=np.float64)
    for grid_y, center_y in enumerate(y_nodes):
        y0 = max(0, int(round(center_y - parameters.tile_size_pixels / 2)))
        y1 = min(reference_height, y0 + parameters.tile_size_pixels)
        y0 = max(0, y1 - parameters.tile_size_pixels)
        for grid_x, center_x in enumerate(x_nodes):
            x0 = max(0, int(round(center_x - parameters.tile_size_pixels / 2)))
            x1 = min(reference_width, x0 + parameters.tile_size_pixels)
            x0 = max(0, x1 - parameters.tile_size_pixels)
            tile_shape = (y1 - y0, x1 - x0)
            yy, xx = np.indices(tile_shape, dtype=np.float64)
            xx += x0
            yy += y0
            input_x, input_y, valid = _project(inverse, xx, yy)
            reference_x, reference_y, reference_valid = _project(
                reference_inverse, xx, yy
            )
            reference_values = reference.sample_bilinear(reference_x, reference_y)
            reference_values[~reference_valid] = np.nan
            target_values = target.sample_bilinear(input_x, input_y)
            target_values[~valid] = np.nan
            fit = _robust_fit(target_values, reference_values, parameters)
            if fit is None:
                continue
            scale, offset, correlation, before, after, count, target_center = fit
            if (
                correlation < parameters.minimum_tile_correlation
                or not parameters.minimum_scale <= scale <= parameters.maximum_scale
            ):
                continue
            scale_grid[grid_y, grid_x] = scale
            offset_grid[grid_y, grid_x] = offset
            correlations[grid_y, grid_x] = correlation
            before_values[grid_y, grid_x] = before
            after_values[grid_y, grid_x] = after
            sample_counts[grid_y, grid_x] = count
            target_centers[grid_y, grid_x] = target_center
    valid = np.isfinite(scale_grid) & np.isfinite(offset_grid)
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = valid_count / valid.size
    if valid_count < parameters.minimum_valid_tiles or valid_fraction < parameters.minimum_valid_tile_fraction:
        raise CalibrationError(
            "LOCAL_NORMALIZATION_MODEL_INSUFFICIENT",
            f"only {valid_count}/{valid.size} model tiles passed",
        )
    median_correlation = float(np.median(correlations[valid]))
    before_mad = float(np.median(before_values[valid]))
    after_mad = float(np.median(after_values[valid]))
    improvement = 1.0 if before_mad <= np.finfo(np.float64).eps and after_mad <= np.finfo(np.float64).eps else (
        1.0 - after_mad / before_mad if before_mad > np.finfo(np.float64).eps else 0.0
    )
    if median_correlation < parameters.minimum_median_correlation:
        raise CalibrationError(
            "LOCAL_NORMALIZATION_CORRELATION_GATE",
            f"median tile correlation {median_correlation:.4f} is below the gate",
        )
    if improvement < parameters.minimum_residual_improvement:
        raise CalibrationError(
            "LOCAL_NORMALIZATION_RESIDUAL_GATE",
            f"residual improvement {improvement:.4f} is below the gate",
        )
    raw_scale_median = float(np.median(scale_grid[valid]))
    raw_scale_mad = 1.4826 * float(
        np.median(np.abs(scale_grid[valid] - raw_scale_median))
    )
    if raw_scale_mad > parameters.maximum_scale_grid_span / 2.0:
        raise CalibrationError(
            "LOCAL_NORMALIZATION_SCALE_UNSTABLE",
            f"tile-scale MAD {raw_scale_mad:.4f} is too large to identify a safe model",
        )
    # Tile regressions can be weakly degenerate with an additive sky gradient.
    # A shrinkage estimator retains genuine low-frequency throughput changes
    # while preventing noisy local slopes from modulating extended targets.
    winsor_half_span = max(0.02, min(3.0 * raw_scale_mad, 0.10))
    regularized_scale = raw_scale_median + 0.25 * (
        np.clip(
            scale_grid,
            raw_scale_median - winsor_half_span,
            raw_scale_median + winsor_half_span,
        )
        - raw_scale_median
    )
    offset_grid[valid] += (
        scale_grid[valid] - regularized_scale[valid]
    ) * target_centers[valid]
    scale_grid[valid] = regularized_scale[valid]
    scale_span = float(np.max(scale_grid[valid]) - np.min(scale_grid[valid]))
    if scale_span > parameters.maximum_scale_grid_span:
        raise CalibrationError(
            "LOCAL_NORMALIZATION_SCALE_SPAN_GATE",
            f"local scale span {scale_span:.4f} exceeds the gate",
        )
    fitted_scale = _fill_grid(scale_grid, valid)
    fitted_offset = _fill_grid(offset_grid, valid)
    evidence = {
        "validTiles": valid_count,
        "totalTiles": int(valid.size),
        "validTileFraction": valid_fraction,
        "medianTileCorrelation": median_correlation,
        "beforeResidualMad": before_mad,
        "afterResidualMad": after_mad,
        "residualImprovement": improvement,
        "scaleMinimum": float(np.min(fitted_scale)),
        "scaleMedian": float(np.median(fitted_scale)),
        "scaleMaximum": float(np.max(fitted_scale)),
        "scaleGridSpan": scale_span,
        "rawScaleMedian": raw_scale_median,
        "rawScaleMad": raw_scale_mad,
        "scaleRegularization": "winsorized-25-percent-local-deviation",
        "sampleCount": int(np.sum(sample_counts[valid])),
    }
    return fitted_scale, fitted_offset, evidence, x_nodes, y_nodes


def _write_model(path: Path, grid: NDArray[np.float64], kind: str) -> None:
    with FitsFloatWriter(
        path,
        (grid.shape[0], grid.shape[1]),
        {"IMAGETYP": f"Local Normalization {kind} Model", "OAFLN": LOCAL_NORMALIZATION_VERSION},
    ) as writer:
        writer.write_rows(0, grid.astype(np.float32))


def _apply_model(
    source: FitsFrame,
    destination: Path,
    input_to_reference: NDArray[np.float64],
    scale_grid: NDArray[np.float64],
    offset_grid: NDArray[np.float64],
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
    parameters: LocalNormalizationParameters,
) -> PixelStatistics:
    height, width = source.shape
    bytes_per_row = width * 72
    if bytes_per_row > parameters.max_memory_bytes:
        raise CalibrationError("LOCAL_NORMALIZATION_MEMORY_BUDGET", "one output row exceeds max_memory_bytes")
    tile_rows = max(1, min(height, parameters.max_memory_bytes // bytes_per_row))
    finite_total = 0
    invalid_total = 0
    minimum = math.inf
    maximum = -math.inf
    total = 0.0
    with FitsFloatWriter(
        destination,
        source.shape,
        {"IMAGETYP": "Locally Normalized Light", "OAFLN": LOCAL_NORMALIZATION_VERSION},
    ) as writer:
        for y0 in range(0, height, tile_rows):
            y1 = min(height, y0 + tile_rows)
            values = source.read_rows(y0, y1)
            yy, xx = np.indices(values.shape, dtype=np.float64)
            yy += y0
            reference_x, reference_y, valid_transform = _project(input_to_reference, xx, yy)
            scale = _sample_grid(scale_grid, x_nodes, y_nodes, reference_x, reference_y)
            offset = _sample_grid(offset_grid, x_nodes, y_nodes, reference_x, reference_y)
            valid = valid_transform & np.isfinite(values) & np.isfinite(scale) & np.isfinite(offset)
            normalized = np.full(values.shape, np.nan, dtype=np.float32)
            normalized[valid] = (
                values[valid].astype(np.float64) * scale[valid] + offset[valid]
            ).astype(np.float32)
            count = int(np.count_nonzero(np.isfinite(normalized)))
            finite_total += count
            invalid_total += int(normalized.size - count)
            if count:
                selected = normalized[np.isfinite(normalized)]
                minimum = min(minimum, float(np.min(selected)))
                maximum = max(maximum, float(np.max(selected)))
                total += float(np.sum(selected, dtype=np.float64))
            writer.write_rows(y0, normalized)
    return PixelStatistics(
        finite_pixels=finite_total,
        invalid_pixels=invalid_total,
        minimum=minimum if finite_total else None,
        maximum=maximum if finite_total else None,
        mean=total / finite_total if finite_total else None,
    )


def normalize_registered_group(
    paths: Sequence[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    *,
    input_to_reference: Sequence[Sequence[Sequence[float]]] | None = None,
    reference_index: int = 0,
    parameters: LocalNormalizationParameters | None = None,
) -> LocalNormalizationResult:
    parameters = parameters or LocalNormalizationParameters(enabled=True)
    parameters.validate()
    if not parameters.enabled:
        raise CalibrationError("LOCAL_NORMALIZATION_DISABLED", "local normalization is disabled")
    canonical = tuple(Path(path).expanduser().resolve(strict=True) for path in paths)
    if len(canonical) < parameters.minimum_frames:
        raise CalibrationError(
            "LOCAL_NORMALIZATION_FRAME_COUNT",
            f"requires at least {parameters.minimum_frames} frames",
        )
    if not 0 <= reference_index < len(canonical):
        raise CalibrationError("LOCAL_NORMALIZATION_REFERENCE_INVALID", "reference index is out of range")
    if len({os.path.normcase(str(path)) for path in canonical}) != len(canonical):
        raise CalibrationError("LOCAL_NORMALIZATION_DUPLICATE_INPUT", "input frames are duplicated")
    matrices = tuple(
        _canonical_matrix(value)
        for value in (
            input_to_reference
            if input_to_reference is not None
            else [None] * len(canonical)
        )
    )
    if len(matrices) != len(canonical):
        raise CalibrationError("LOCAL_NORMALIZATION_TRANSFORM_COUNT", "transform count differs from inputs")
    destination = Path(output_directory).expanduser().resolve(strict=False)
    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError("OUTPUT_EXISTS", "local normalization directory must be new", path=str(destination))
    destination.mkdir(parents=True, exist_ok=False)
    completed = False
    try:
        models_dir = destination / "models"
        frames_dir = destination / "frames"
        models_dir.mkdir()
        frames_dir.mkdir()
        records: list[dict[str, Any]] = []
        outputs: list[str] = []
        with ExitStack() as stack:
            frames = [stack.enter_context(FitsFrame(path)) for path in canonical]
            reference = frames[reference_index]
            reference_shape = reference.shape
            for frame in frames:
                if frame.shape != reference_shape:
                    raise CalibrationError("LOCAL_NORMALIZATION_GEOMETRY_MISMATCH", "frame geometries differ")
            for index, (path, frame, matrix) in enumerate(zip(canonical, frames, matrices, strict=True)):
                if index == reference_index:
                    x_nodes = _grid_coordinates(reference_shape[1], parameters.tile_size_pixels)
                    y_nodes = _grid_coordinates(reference_shape[0], parameters.tile_size_pixels)
                    scale_grid = np.ones((len(y_nodes), len(x_nodes)), dtype=np.float64)
                    offset_grid = np.zeros_like(scale_grid)
                    evidence = {
                        "reference": True,
                        "validTiles": int(scale_grid.size),
                        "totalTiles": int(scale_grid.size),
                        "validTileFraction": 1.0,
                        "medianTileCorrelation": 1.0,
                        "beforeResidualMad": 0.0,
                        "afterResidualMad": 0.0,
                        "residualImprovement": 1.0,
                        "scaleMinimum": 1.0,
                        "scaleMedian": 1.0,
                        "scaleMaximum": 1.0,
                        "scaleGridSpan": 0.0,
                        "sampleCount": 0,
                    }
                else:
                    scale_grid, offset_grid, evidence, x_nodes, y_nodes = _fit_model(
                        frame,
                        reference,
                        matrix,
                        matrices[reference_index],
                        parameters,
                    )
                scale_path = models_dir / f"{index + 1:05d}_scale.fits"
                offset_path = models_dir / f"{index + 1:05d}_offset.fits"
                output_path = frames_dir / f"{index + 1:05d}_normalized.fits"
                _write_model(scale_path, scale_grid, "Scale")
                _write_model(offset_path, offset_grid, "Offset")
                statistics = _apply_model(
                    frame,
                    output_path,
                    matrix,
                    scale_grid,
                    offset_grid,
                    x_nodes,
                    y_nodes,
                    parameters,
                )
                outputs.append(str(output_path))
                records.append(
                    {
                        "index": index,
                        "reference": index == reference_index,
                        "sourceSha256": _sha256(path),
                        "normalized": {
                            "path": str(output_path.relative_to(destination)),
                            "sha256": _sha256(output_path),
                            "sizeBytes": output_path.stat().st_size,
                            "statistics": statistics.serializable(),
                        },
                        "model": {
                            "scalePath": str(scale_path.relative_to(destination)),
                            "scaleSha256": _sha256(scale_path),
                            "offsetPath": str(offset_path.relative_to(destination)),
                            "offsetSha256": _sha256(offset_path),
                            "gridShape": [int(scale_grid.shape[0]), int(scale_grid.shape[1])],
                            "inputToReference": matrix.tolist(),
                        },
                        "evidence": evidence,
                    }
                )
        receipt = {
            "schemaVersion": 1,
            "algorithm": LOCAL_NORMALIZATION_VERSION,
            "status": "APPLIED",
            "pixInsightEquivalent": False,
            "parameters": parameters.serializable(),
            "referenceIndex": reference_index,
            "frames": records,
            "scienceGuard": {
                "brightStructureExcludedFromFit": True,
                "modelMinimumSpatialScalePixels": parameters.tile_size_pixels,
                "ambiguousModelsFailClosed": True,
            },
        }
        receipt_path = destination / "receipt.json"
        with receipt_path.open("xb") as stream:
            stream.write((json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        completed = True
        return LocalNormalizationResult(tuple(outputs), str(receipt_path), receipt)
    finally:
        if not completed:
            remove_tree(destination)


__all__ = [
    "LOCAL_NORMALIZATION_VERSION",
    "LocalNormalizationParameters",
    "LocalNormalizationResult",
    "normalize_registered_group",
]
