"""Stellar-scale plus low-frequency additive normalization for registered mono frames.

Multiplicative throughput is intentionally one scalar per frame. Additive sky
structure may use a 128-pixel node grid smoothed at seven nodes (about 896
pixels sigma). Coefficients are fitted only from paired low/mid-intensity pixels
after registration. Bright stars and bright target cores are excluded by joint
quantiles, then a residual MAD clip removes remaining compact outliers.

Multiplicative scale is never inferred from background covariance.  A caller
may provide a content-bound scale measured from same-filter matched stellar
apertures during registration.  If that evidence is unavailable, the
implementation falls back to an audited offset-only correction.  It does not
claim PixInsight ImageIntegration equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .native_kernels import TILE_OFFSET_KERNEL_ID, load_native_kernels
from .calibration import CalibrationError, FitsFrame


GLOBAL_NORMALIZATION_VERSION = "matched-stellar-scale-plus-smoothed-additive-grid-v4"


class _OffsetGridSpanUnsafe(CalibrationError):
    """Reject an optional grid while retaining the measurements for its receipt."""

    def __init__(self, path: str, evidence: dict[str, Any]) -> None:
        self.evidence = evidence
        limits = evidence["limits"]
        super().__init__(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_SPAN_UNSAFE",
            "additive correction exceeds its sky-relative gate: "
            f"p05-p95={evidence['offsetP05P95Fraction']:.6g} "
            f"(maximum {limits['maximumOffsetP05P95Fraction']:.6g}), "
            f"span={evidence['offsetSpanFraction']:.6g} "
            f"(maximum {limits['maximumOffsetSpanFraction']:.6g}), "
            f"neighbor-sigma={evidence['neighborDifferenceSigmaFraction']:.6g} "
            f"(maximum {limits['maximumOffsetNeighborSigmaFraction']:.6g}); "
            f"reference-sky={evidence['referenceSky']:.6g}",
            path=path,
        )


@dataclass(frozen=True, slots=True)
class GlobalNormalizationParameters:
    enabled: bool = True
    maximum_samples: int = 262_144
    minimum_samples: int = 4_096
    lower_quantile: float = 0.05
    upper_quantile: float = 0.70
    residual_clip_sigma: float = 3.5
    minimum_scale: float = 0.50
    maximum_scale: float = 2.00
    allow_offset_only_fallback: bool = True
    offset_tile_size_pixels: int = 128
    offset_smoothing_sigma_nodes: float = 7.0
    minimum_samples_per_offset_tile: int = 512
    minimum_valid_offset_tiles: int = 12
    minimum_valid_offset_tile_fraction: float = 0.70
    minimum_offset_residual_improvement: float = 0.10
    offset_grid_not_needed_sigma_fraction: float = 0.03
    maximum_offset_p05_p95_fraction: float = 0.25
    maximum_offset_span_fraction: float = 0.35
    maximum_offset_neighbor_sigma_fraction: float = 0.002

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("global normalization enabled must be boolean")
        if (
            isinstance(self.maximum_samples, bool)
            or not isinstance(self.maximum_samples, int)
            or self.maximum_samples < 1_024
        ):
            raise ValueError("maximum_samples must be an integer >= 1024")
        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(self.minimum_samples, int)
            or self.minimum_samples < 128
            or self.minimum_samples > self.maximum_samples
        ):
            raise ValueError(
                "minimum_samples must be an integer in [128, maximum_samples]"
            )
        if not (
            math.isfinite(self.lower_quantile)
            and math.isfinite(self.upper_quantile)
            and 0.0 <= self.lower_quantile < self.upper_quantile <= 0.90
        ):
            raise ValueError(
                "global normalization quantiles must satisfy 0 <= lower < upper <= 0.90"
            )
        if (
            not math.isfinite(self.residual_clip_sigma)
            or self.residual_clip_sigma < 2.0
        ):
            raise ValueError("residual_clip_sigma must be finite and >= 2")
        if not (
            math.isfinite(self.minimum_scale)
            and math.isfinite(self.maximum_scale)
            and 0.0 < self.minimum_scale < 1.0 < self.maximum_scale
        ):
            raise ValueError("global normalization scale bounds must straddle one")
        if not isinstance(self.allow_offset_only_fallback, bool):
            raise ValueError("allow_offset_only_fallback must be boolean")
        for name, floor in (
            ("offset_tile_size_pixels", 64),
            ("minimum_samples_per_offset_tile", 32),
            ("minimum_valid_offset_tiles", 4),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"{name} must be an integer >= {floor}")
        for name in (
            "minimum_valid_offset_tile_fraction",
            "minimum_offset_residual_improvement",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "offset_smoothing_sigma_nodes",
            "offset_grid_not_needed_sigma_fraction",
            "maximum_offset_p05_p95_fraction",
            "maximum_offset_span_fraction",
            "maximum_offset_neighbor_sigma_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def serializable(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "algorithm": GLOBAL_NORMALIZATION_VERSION,
            "maximumSamples": self.maximum_samples,
            "minimumSamples": self.minimum_samples,
            "backgroundQuantiles": [self.lower_quantile, self.upper_quantile],
            "residualClipSigma": self.residual_clip_sigma,
            "scaleBounds": [self.minimum_scale, self.maximum_scale],
            "allowOffsetOnlyFallback": self.allow_offset_only_fallback,
            "offsetTileSizePixels": self.offset_tile_size_pixels,
            "offsetSmoothingSigmaNodes": self.offset_smoothing_sigma_nodes,
            "minimumSamplesPerOffsetTile": self.minimum_samples_per_offset_tile,
            "minimumValidOffsetTiles": self.minimum_valid_offset_tiles,
            "minimumValidOffsetTileFraction": self.minimum_valid_offset_tile_fraction,
            "minimumOffsetResidualImprovement": self.minimum_offset_residual_improvement,
            "offsetGridNotNeededSigmaFraction": self.offset_grid_not_needed_sigma_fraction,
            "maximumOffsetP05P95Fraction": self.maximum_offset_p05_p95_fraction,
            "maximumOffsetSpanFraction": self.maximum_offset_span_fraction,
            "maximumOffsetNeighborSigmaFraction": self.maximum_offset_neighbor_sigma_fraction,
            "equivalentToPixInsight": False,
        }


@dataclass(frozen=True, slots=True)
class GlobalNormalizationCoefficient:
    source_path: str
    scale: float
    offset: float
    mode: str
    evidence: dict[str, Any]
    offset_grid: tuple[tuple[float, ...], ...] = ()
    offset_grid_x: tuple[float, ...] = ()
    offset_grid_y: tuple[float, ...] = ()

    def serializable(self, *, reference: bool = False) -> dict[str, Any]:
        return {
            "source": self.source_path,
            "reference": reference,
            "mode": self.mode,
            "scale": self.scale,
            "offset": self.offset,
            "evidence": dict(self.evidence),
            "offsetGrid": (
                {
                    "values": [list(row) for row in self.offset_grid],
                    "xNodes": list(self.offset_grid_x),
                    "yNodes": list(self.offset_grid_y),
                    "sha256": _offset_grid_sha256(
                        self.offset_grid, self.offset_grid_x, self.offset_grid_y
                    ),
                }
                if self.offset_grid
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class GlobalNormalizationResult:
    coefficients: tuple[GlobalNormalizationCoefficient, ...]
    reference_index: int
    receipt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StellarScaleHint:
    source_path: str
    reference_path: str
    filter_name: str
    source_sha256: str
    reference_sha256: str
    scale: float | None
    status: str
    evidence: dict[str, Any]

    def serializable(self) -> dict[str, Any]:
        return {
            "source": self.source_path,
            "reference": self.reference_path,
            "filter": self.filter_name,
            "sourceSha256": self.source_sha256,
            "referenceSha256": self.reference_sha256,
            "scale": self.scale,
            "status": self.status,
            "evidence": dict(self.evidence),
        }


def _uniform_integer_indices(length: int, count: int) -> NDArray[np.int64]:
    if count <= 1:
        return np.zeros(1, dtype=np.int64)
    values = np.linspace(0, length - 1, count, dtype=np.float64)
    return np.rint(values).astype(np.int64)


def _sample_coordinates(
    shape: tuple[int, int], maximum_samples: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    height, width = shape
    target_rows = min(
        height,
        max(1, math.isqrt(maximum_samples * height // max(1, width))),
    )
    target_columns = min(width, max(1, maximum_samples // target_rows))
    return (
        _uniform_integer_indices(height, target_rows),
        _uniform_integer_indices(width, target_columns),
    )


def _offset_grid_sha256(
    grid_value: Sequence[Sequence[float]],
    x_nodes_value: Sequence[float],
    y_nodes_value: Sequence[float],
) -> str:
    digest = hashlib.sha256()
    digest.update(GLOBAL_NORMALIZATION_VERSION.encode("ascii") + b"\0")
    for value in (grid_value, x_nodes_value, y_nodes_value):
        array = np.asarray(value, dtype="<f8")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _grid_coordinates(length: int, tile_size: int) -> NDArray[np.float64]:
    starts = np.arange(0, length, tile_size, dtype=np.int64)
    stops = np.minimum(starts + tile_size, length)
    return (starts + stops - 1).astype(np.float64) / 2.0


def _fill_grid(
    grid: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    if not np.any(valid):
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_UNDERCONSTRAINED",
            "additive offset grid has no valid tiles",
        )
    result = grid.copy()
    known = valid.copy()
    fallback = float(np.median(result[valid]))
    result[~known] = fallback
    for _ in range(result.shape[0] + result.shape[1]):
        pending = np.argwhere(~known)
        if not pending.size:
            break
        updates: list[tuple[int, int, float]] = []
        for y, x in pending:
            neighbours = [
                result[yy, xx]
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                if 0 <= yy < result.shape[0]
                and 0 <= xx < result.shape[1]
                and known[yy, xx]
            ]
            if neighbours:
                updates.append((int(y), int(x), float(np.median(neighbours))))
        if not updates:
            break
        for y, x, value in updates:
            result[y, x] = value
            known[y, x] = True
    result[~known] = fallback
    return result


def _sample_grid_points(
    grid: NDArray[np.float64],
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    x_clipped = np.clip(x, x_nodes[0], x_nodes[-1])
    y_clipped = np.clip(y, y_nodes[0], y_nodes[-1])
    x_hi = np.clip(np.searchsorted(x_nodes, x_clipped, side="right"), 1, len(x_nodes) - 1)
    y_hi = np.clip(np.searchsorted(y_nodes, y_clipped, side="right"), 1, len(y_nodes) - 1)
    x_lo = x_hi - 1
    y_lo = y_hi - 1
    wx = (x_clipped - x_nodes[x_lo]) / (x_nodes[x_hi] - x_nodes[x_lo])
    wy = (y_clipped - y_nodes[y_lo]) / (y_nodes[y_hi] - y_nodes[y_lo])
    top = grid[y_lo, x_lo] * (1.0 - wx) + grid[y_lo, x_hi] * wx
    bottom = grid[y_hi, x_lo] * (1.0 - wx) + grid[y_hi, x_hi] * wx
    return top * (1.0 - wy) + bottom * wy


class _ReferenceSampleCache:
    """Run-local sampled reference pixels shared by every target frame."""

    def __init__(self, reference: FitsFrame) -> None:
        self.reference = reference
        self._indices: dict[tuple[str, int], NDArray[np.int64]] = {}
        self._last_index_objects: dict[
            tuple[str, int], NDArray[np.int64]
        ] = {}
        self._values: dict[
            tuple[tuple[str, int], int], NDArray[np.float32]
        ] = {}
        self._rows: dict[int, NDArray[np.float32]] = {}
        self._lock = threading.Lock()
        self.requests = 0
        self.hits = 0
        self.cached_samples = 0

    def sample(
        self,
        role: tuple[str, int],
        y: int,
        x_indices: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        # Target frames are fitted concurrently; the lock keeps the cache and
        # its evidence counters deterministic.
        with self._lock:
            self.requests += 1
            expected = self._indices.get(role)
            if expected is None:
                expected = np.asarray(x_indices, dtype=np.int64).copy()
                expected.setflags(write=False)
                self._indices[role] = expected
                self._last_index_objects[role] = x_indices
            elif self._last_index_objects.get(role) is not x_indices:
                if not np.array_equal(expected, x_indices):
                    raise RuntimeError("reference sample-cache coordinate role changed")
                self._last_index_objects[role] = x_indices
            key = (role, int(y))
            cached = self._values.get(key)
            if cached is not None:
                self.hits += 1
                return cached
            values = np.asarray(
                self.reference.read_rows(int(y), int(y) + 1)[0, x_indices],
                dtype=np.float32,
            )
            values.setflags(write=False)
            self._values[key] = values
            self.cached_samples += int(values.size)
            return values

    def rows(self, y_indices: NDArray[np.int64]) -> NDArray[np.float32]:
        """Return the requested reference rows, decoding each row once."""

        with self._lock:
            width = self.reference.shape[1]
            result = np.empty((len(y_indices), width), dtype=np.float32)
            for position, y in enumerate(y_indices):
                self.requests += 1
                row = self._rows.get(int(y))
                if row is None:
                    row = self.reference.read_rows(int(y), int(y) + 1)[0]
                    row.setflags(write=False)
                    self._rows[int(y)] = row
                    self.cached_samples += int(row.size)
                else:
                    self.hits += 1
                result[position] = row
            return result

    def serializable(self) -> dict[str, Any]:
        return {
            "strategy": "deterministic-coordinate-reference-sample-cache-v1",
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.requests - self.hits,
            "coordinateRoles": len(self._indices),
            "cachedRows": len(self._rows),
            "cachedSamples": self.cached_samples,
            "cachedBytes": self.cached_samples * np.dtype(np.float32).itemsize,
            "fullReferenceMaterialized": False,
        }


def _paired_samples(
    target: FitsFrame,
    reference: FitsFrame,
    parameters: GlobalNormalizationParameters,
    reference_cache: _ReferenceSampleCache,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
]:
    y_indices, x_indices = _sample_coordinates(
        reference.shape, parameters.maximum_samples
    )
    target_chunks: list[NDArray[np.float64]] = []
    reference_chunks: list[NDArray[np.float64]] = []
    x_chunks: list[NDArray[np.float64]] = []
    y_chunks: list[NDArray[np.float64]] = []
    # One gather per frame replaces one read per sampled row; the per-row
    # pairing and selection below are unchanged.
    target_sampled = target.read_sampled_rows(y_indices)[:, x_indices]
    reference_sampled = reference_cache.rows(y_indices)[:, x_indices]
    for position, y in enumerate(y_indices):
        target_row = target_sampled[position]
        reference_row = reference_sampled[position]
        finite = np.isfinite(target_row) & np.isfinite(reference_row)
        if np.any(finite):
            target_chunks.append(
                np.asarray(target_row[finite], dtype=np.float64)
            )
            reference_chunks.append(
                np.asarray(reference_row[finite], dtype=np.float64)
            )
            x_chunks.append(np.asarray(x_indices[finite], dtype=np.float64))
            y_chunks.append(np.full(int(np.count_nonzero(finite)), float(y)))
    if not target_chunks:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_SAMPLES_INSUFFICIENT",
            "registered frame has no paired finite normalization samples",
            path=str(target.path),
        )
    x = np.concatenate(target_chunks)
    y = np.concatenate(reference_chunks)
    sample_x = np.concatenate(x_chunks)
    sample_y = np.concatenate(y_chunks)
    paired_before_selection = int(x.size)
    x_bounds = np.quantile(x, (parameters.lower_quantile, parameters.upper_quantile))
    y_bounds = np.quantile(y, (parameters.lower_quantile, parameters.upper_quantile))
    background = (
        (x >= x_bounds[0])
        & (x <= x_bounds[1])
        & (y >= y_bounds[0])
        & (y <= y_bounds[1])
    )
    return (
        x[background],
        y[background],
        sample_x[background],
        sample_y[background],
        paired_before_selection,
    )


def _location_and_mad(values: NDArray[np.float64]) -> tuple[float, float]:
    location = float(np.median(values))
    dispersion = float(1.4826 * np.median(np.abs(values - location)))
    return location, dispersion


def _tile_offset(
    target: NDArray[np.float64],
    reference: NDArray[np.float64],
    scale: float,
    parameters: GlobalNormalizationParameters,
) -> tuple[float, int, float] | None:
    finite = np.isfinite(target) & np.isfinite(reference)
    x = target[finite]
    y = reference[finite]
    if x.size < parameters.minimum_samples_per_offset_tile:
        return None
    x_bounds = np.quantile(x, (parameters.lower_quantile, parameters.upper_quantile))
    y_bounds = np.quantile(y, (parameters.lower_quantile, parameters.upper_quantile))
    selected = (
        (x >= x_bounds[0])
        & (x <= x_bounds[1])
        & (y >= y_bounds[0])
        & (y <= y_bounds[1])
    )
    residual = y[selected] - scale * x[selected]
    if residual.size < parameters.minimum_samples_per_offset_tile:
        return None
    center, sigma = _location_and_mad(residual)
    if sigma > np.finfo(np.float64).eps:
        keep = np.abs(residual - center) <= parameters.residual_clip_sigma * sigma
        if int(np.count_nonzero(keep)) >= parameters.minimum_samples_per_offset_tile:
            residual = residual[keep]
    offset = float(np.median(residual))
    _, residual_mad = _location_and_mad(residual - offset)
    if not math.isfinite(offset) or not math.isfinite(residual_mad):
        return None
    return offset, int(residual.size), residual_mad


def _smooth_offset_grid(
    offsets: NDArray[np.float64],
    valid: NDArray[np.bool_],
    global_offset: float,
    sigma_nodes: float,
) -> NDArray[np.float64]:
    weights = valid.astype(np.float64)
    numerator = gaussian_filter(
        np.where(valid, offsets, 0.0), sigma=sigma_nodes, mode="nearest"
    )
    denominator = gaussian_filter(weights, sigma=sigma_nodes, mode="nearest")
    result = np.full(offsets.shape, global_offset, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 1e-6,
    )
    return result


def _fit_additive_offset_grid(
    target: FitsFrame,
    reference: FitsFrame,
    scale: float,
    sample_target: NDArray[np.float64],
    sample_reference: NDArray[np.float64],
    sample_x: NDArray[np.float64],
    sample_y: NDArray[np.float64],
    parameters: GlobalNormalizationParameters,
    reference_cache: _ReferenceSampleCache,
    native_threads: int | None = None,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[float, ...],
    dict[str, Any],
]:
    height, width = reference.shape
    x_nodes = _grid_coordinates(width, parameters.offset_tile_size_pixels)
    y_nodes = _grid_coordinates(height, parameters.offset_tile_size_pixels)
    if len(x_nodes) < 2 or len(y_nodes) < 2:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_UNDERCONSTRAINED",
            "image is too small for a two-dimensional low-frequency offset grid",
            path=str(target.path),
        )
    offsets = np.full((len(y_nodes), len(x_nodes)), np.nan, dtype=np.float64)
    sample_counts = np.zeros(offsets.shape, dtype=np.int64)
    residual_mads = np.full(offsets.shape, np.nan, dtype=np.float64)
    maximum_tile_samples = 2_048
    kernels = load_native_kernels()
    tile_targets: list[NDArray[np.float64]] = []
    tile_references: list[NDArray[np.float64]] = []
    tile_positions: list[tuple[int, int]] = []
    for grid_y in range(len(y_nodes)):
        y0 = grid_y * parameters.offset_tile_size_pixels
        y1 = min(height, y0 + parameters.offset_tile_size_pixels)
        tile_height = y1 - y0
        target_rows = min(
            tile_height,
            max(
                1,
                math.isqrt(
                    maximum_tile_samples
                    * tile_height
                    // max(1, parameters.offset_tile_size_pixels)
                ),
            ),
        )
        y_indices = _uniform_integer_indices(tile_height, target_rows) + y0
        x_indices_by_tile: list[NDArray[np.int64]] = []
        for grid_x in range(len(x_nodes)):
            x0 = grid_x * parameters.offset_tile_size_pixels
            x1 = min(width, x0 + parameters.offset_tile_size_pixels)
            count = min(x1 - x0, max(1, maximum_tile_samples // target_rows))
            x_indices_by_tile.append(
                _uniform_integer_indices(x1 - x0, count) + x0
        )
        # One band read per tile row replaces one read per sampled row; the
        # sampled rows and columns, and hence every tile's sample sequence
        # (row-major), are exactly those of the per-row formulation.
        target_band = target.read_rows(int(y0), int(y1))
        sampled_target = target_band[y_indices - y0]
        sampled_reference = reference_cache.rows(y_indices)
        for grid_x, x_indices in enumerate(x_indices_by_tile):
            tile_target = np.asarray(
                sampled_target[:, x_indices], dtype=np.float64
            ).ravel()
            tile_reference = np.asarray(
                sampled_reference[:, x_indices], dtype=np.float64
            ).ravel()
            if kernels is None:
                fit = _tile_offset(tile_target, tile_reference, scale, parameters)
                if fit is None:
                    continue
                offsets[grid_y, grid_x], sample_counts[grid_y, grid_x], residual_mads[
                    grid_y, grid_x
                ] = fit
            else:
                tile_targets.append(tile_target)
                tile_references.append(tile_reference)
                tile_positions.append((grid_y, grid_x))
    if kernels is not None and tile_targets:
        boundaries = np.concatenate(
            ([0], np.cumsum([tile.size for tile in tile_targets]))
        ).astype(np.uint64)
        tile_offsets, tile_counts, tile_mads, tile_valid = kernels.tile_offsets(
            np.concatenate(tile_targets),
            np.concatenate(tile_references),
            boundaries,
            scale=scale,
            lower_quantile=parameters.lower_quantile,
            upper_quantile=parameters.upper_quantile,
            minimum_samples=parameters.minimum_samples_per_offset_tile,
            residual_clip_sigma=parameters.residual_clip_sigma,
            threads=native_threads,
        )
        for index, (grid_y, grid_x) in enumerate(tile_positions):
            if not tile_valid[index]:
                continue
            offsets[grid_y, grid_x] = float(tile_offsets[index])
            sample_counts[grid_y, grid_x] = int(tile_counts[index])
            residual_mads[grid_y, grid_x] = float(tile_mads[index])

    valid = np.isfinite(offsets)
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = valid_count / offsets.size
    if (
        valid_count < parameters.minimum_valid_offset_tiles
        or valid_fraction < parameters.minimum_valid_offset_tile_fraction
    ):
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_UNDERCONSTRAINED",
            f"only {valid_count}/{offsets.size} additive offset tiles passed",
            path=str(target.path),
        )
    residual = sample_reference - scale * sample_target
    scalar_offset = float(np.median(residual))
    reference_sky = max(abs(float(np.median(sample_reference))), 1e-9)
    raw_offset_sigma = float(
        1.4826 * np.median(np.abs(offsets[valid] - scalar_offset))
    )
    if raw_offset_sigma <= parameters.offset_grid_not_needed_sigma_fraction * reference_sky:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_NOT_BENEFICIAL",
            f"low-frequency offset sigma {raw_offset_sigma:.6g} is already below "
            f"{parameters.offset_grid_not_needed_sigma_fraction:.6g} of reference sky",
            path=str(target.path),
        )

    row_index, column_index = np.indices(offsets.shape)
    training = valid & ((row_index + column_index) % 2 == 0)
    holdout = valid & ~training
    if np.count_nonzero(training) < 4 or np.count_nonzero(holdout) < 4:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_UNDERCONSTRAINED",
            "checkerboard offset-grid validation needs at least four training and holdout tiles",
            path=str(target.path),
        )
    training_model = _smooth_offset_grid(
        offsets,
        training,
        scalar_offset,
        parameters.offset_smoothing_sigma_nodes,
    )
    before_holdout = offsets[holdout] - scalar_offset
    after_holdout = offsets[holdout] - training_model[holdout]
    _, before_holdout_mad = _location_and_mad(before_holdout)
    _, after_holdout_mad = _location_and_mad(after_holdout)
    before_holdout_span = float(
        np.percentile(before_holdout, 95) - np.percentile(before_holdout, 5)
    )
    after_holdout_span = float(
        np.percentile(after_holdout, 95) - np.percentile(after_holdout, 5)
    )
    mad_improvement = (
        1.0 - after_holdout_mad / before_holdout_mad
        if before_holdout_mad > np.finfo(np.float64).eps
        else 0.0
    )
    span_improvement = (
        1.0 - after_holdout_span / before_holdout_span
        if before_holdout_span > np.finfo(np.float64).eps
        else 0.0
    )
    tolerance = 1.02
    if (
        after_holdout_mad > before_holdout_mad * tolerance
        or after_holdout_span > before_holdout_span * tolerance
        or max(mad_improvement, span_improvement)
        < parameters.minimum_offset_residual_improvement
    ):
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_NOT_BENEFICIAL",
            "checkerboard holdout did not validate the additive offset grid",
            path=str(target.path),
        )

    fitted = _smooth_offset_grid(
        offsets,
        valid,
        scalar_offset,
        parameters.offset_smoothing_sigma_nodes,
    )
    offset_span = float(np.max(fitted) - np.min(fitted))
    offset_p05, offset_p95 = (
        float(value) for value in np.percentile(fitted, (5, 95))
    )
    offset_p05_p95_span = offset_p95 - offset_p05
    vertical_center, vertical_sigma = _location_and_mad(
        np.diff(fitted, axis=0).ravel()
    )
    horizontal_center, horizontal_sigma = _location_and_mad(
        np.diff(fitted, axis=1).ravel()
    )
    neighbor_sigma = max(vertical_sigma, horizontal_sigma)
    offset_span_fraction = offset_span / reference_sky
    offset_p05_p95_fraction = offset_p05_p95_span / reference_sky
    neighbor_sigma_fraction = neighbor_sigma / reference_sky
    if (
        offset_p05_p95_fraction > parameters.maximum_offset_p05_p95_fraction
        or offset_span_fraction > parameters.maximum_offset_span_fraction
        or neighbor_sigma_fraction
        > parameters.maximum_offset_neighbor_sigma_fraction
    ):
        raise _OffsetGridSpanUnsafe(
            str(target.path),
            {
                "model": "BILINEAR_ADDITIVE_GRID",
                "status": "REJECTED",
                "applied": False,
                "referenceSky": reference_sky,
                "offsetP05P95Span": offset_p05_p95_span,
                "offsetP05P95Fraction": offset_p05_p95_fraction,
                "offsetSpan": offset_span,
                "offsetSpanFraction": offset_span_fraction,
                "neighborDifferenceRobustSigma": neighbor_sigma,
                "neighborDifferenceSigmaFraction": neighbor_sigma_fraction,
                "limits": {
                    "maximumOffsetP05P95Fraction": parameters.maximum_offset_p05_p95_fraction,
                    "maximumOffsetSpanFraction": parameters.maximum_offset_span_fraction,
                    "maximumOffsetNeighborSigmaFraction": parameters.maximum_offset_neighbor_sigma_fraction,
                },
            },
        )

    before = residual - scalar_offset
    sampled_grid = _sample_grid_points(fitted, x_nodes, y_nodes, sample_x, sample_y)
    after = residual - sampled_grid
    _, before_mad = _location_and_mad(before)
    after_center, after_mad = _location_and_mad(after)
    improvement = (
        1.0
        if before_mad <= np.finfo(np.float64).eps
        and after_mad <= np.finfo(np.float64).eps
        else 1.0 - after_mad / before_mad
        if before_mad > np.finfo(np.float64).eps
        else 0.0
    )
    grid_tuple = tuple(tuple(float(value) for value in row) for row in fitted)
    x_tuple = tuple(float(value) for value in x_nodes)
    y_tuple = tuple(float(value) for value in y_nodes)
    evidence = {
        "model": "BILINEAR_ADDITIVE_GRID",
        "coordinateConvention": "absolute-zero-based-pixel-centers",
        "boundaryConvention": "clamp-to-nearest-grid-node",
        "tileSizePixels": parameters.offset_tile_size_pixels,
        "tileStatisticsKernel": (
            TILE_OFFSET_KERNEL_ID if kernels is not None else "numpy-tile-offset-v1"
        ),
        "smoothingSigmaNodes": parameters.offset_smoothing_sigma_nodes,
        "effectiveSmoothingSigmaPixels": (
            parameters.offset_tile_size_pixels
            * parameters.offset_smoothing_sigma_nodes
        ),
        "gridShape": [len(y_nodes), len(x_nodes)],
        "validTiles": valid_count,
        "totalTiles": int(offsets.size),
        "validTileFraction": valid_fraction,
        "sampleCount": int(np.sum(sample_counts[valid])),
        "tileResidualMadMedian": float(np.median(residual_mads[valid])),
        "offsetMinimum": float(np.min(fitted)),
        "offsetMedian": float(np.median(fitted)),
        "offsetMaximum": float(np.max(fitted)),
        "offsetSpan": offset_span,
        "offsetSpanFraction": offset_span_fraction,
        "offsetP05P95Span": offset_p05_p95_span,
        "offsetP05P95Fraction": offset_p05_p95_fraction,
        "neighborDifferenceMedian": {
            "x": horizontal_center,
            "y": vertical_center,
        },
        "neighborDifferenceRobustSigmaByAxis": {
            "x": horizontal_sigma,
            "y": vertical_sigma,
        },
        "neighborDifferenceRobustSigma": neighbor_sigma,
        "neighborDifferenceSigmaFraction": neighbor_sigma_fraction,
        "rawOffsetRobustSigma": raw_offset_sigma,
        "rawOffsetSigmaFraction": raw_offset_sigma / reference_sky,
        "referenceSky": reference_sky,
        "checkerboardHoldout": {
            "trainingTiles": int(np.count_nonzero(training)),
            "holdoutTiles": int(np.count_nonzero(holdout)),
            "beforeMad": before_holdout_mad,
            "afterMad": after_holdout_mad,
            "madImprovement": mad_improvement,
            "beforeP05P95Span": before_holdout_span,
            "afterP05P95Span": after_holdout_span,
            "spanImprovement": span_improvement,
        },
        "scalarResidualMad": before_mad,
        "gridResidualMedian": after_center,
        "gridResidualMad": after_mad,
        "residualImprovement": improvement,
        "sha256": _offset_grid_sha256(grid_tuple, x_tuple, y_tuple),
    }
    return grid_tuple, x_tuple, y_tuple, evidence


def _fit_coefficient(
    target: FitsFrame,
    reference: FitsFrame,
    parameters: GlobalNormalizationParameters,
    scale_hint: StellarScaleHint | None,
    reference_cache: _ReferenceSampleCache,
    native_threads: int | None = None,
) -> GlobalNormalizationCoefficient:
    x, y, sample_x, sample_y, paired_before_selection = _paired_samples(
        target, reference, parameters, reference_cache
    )
    if x.size < parameters.minimum_samples:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_SAMPLES_INSUFFICIENT",
            f"only {x.size} low/mid-intensity paired samples remain; "
            f"require {parameters.minimum_samples}",
            path=str(target.path),
        )

    hint_scale = scale_hint.scale if scale_hint is not None else None
    stellar_scale_accepted = bool(
        scale_hint is not None
        and scale_hint.status in {"REFERENCE_IDENTITY", "STELLAR_SCALE_ACCEPTED"}
        and hint_scale is not None
        and math.isfinite(hint_scale)
        and parameters.minimum_scale <= hint_scale <= parameters.maximum_scale
    )
    if not stellar_scale_accepted and not parameters.allow_offset_only_fallback:
        reason = (
            "missing stellar scale hint"
            if scale_hint is None
            else f"stellar scale hint is {scale_hint.status}"
        )
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_STELLAR_SCALE_UNAVAILABLE",
            reason,
            path=str(target.path),
        )
    scale = float(hint_scale) if stellar_scale_accepted else 1.0
    residual = y - scale * x
    residual_center, residual_sigma = _location_and_mad(residual)
    if residual_sigma > np.finfo(np.float64).eps:
        keep = (
            np.abs(residual - residual_center)
            <= parameters.residual_clip_sigma * residual_sigma
        )
        if int(np.count_nonzero(keep)) >= parameters.minimum_samples:
            x = x[keep]
            y = y[keep]
            sample_x = sample_x[keep]
            sample_y = sample_y[keep]
            residual = y - scale * x
    offset = float(np.median(residual))
    if not math.isfinite(offset):
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_INVALID",
            "background offset estimate is non-finite",
            path=str(target.path),
        )
    fixed_scale_before = y - scale * x
    before_center, before_mad = _location_and_mad(fixed_scale_before)
    scalar_after = fixed_scale_before - offset
    after_center, after_mad = _location_and_mad(scalar_after)
    common_evidence: dict[str, Any] = {
        "pairedSamplesBeforeSelection": paired_before_selection,
        "selectedBackgroundSamples": int(x.size),
        "brightStructureExcludedByJointQuantiles": True,
        "multiplicativeScaleSpatialOrder": 0,
        "beforeResidualMedian": before_center,
        "beforeResidualMad": before_mad,
        "afterResidualMedian": after_center,
        "afterResidualMad": after_mad,
        "stellarScale": (
            scale_hint.serializable() if scale_hint is not None else None
        ),
        "backgroundCovarianceUsedForScale": False,
        "fallbackReason": (
            None
            if stellar_scale_accepted
            else "STELLAR_SCALE_UNAVAILABLE_OR_UNSAFE"
        ),
    }
    offset_grid: tuple[tuple[float, ...], ...] = ()
    offset_grid_x: tuple[float, ...] = ()
    offset_grid_y: tuple[float, ...] = ()
    try:
        (
            offset_grid,
            offset_grid_x,
            offset_grid_y,
            grid_evidence,
        ) = _fit_additive_offset_grid(
            target,
            reference,
            scale,
            x,
            y,
            sample_x,
            sample_y,
            parameters,
            reference_cache,
            native_threads,
        )
        offset = 0.0
        mode = (
            "STELLAR_SCALE_ADDITIVE_GRID"
            if stellar_scale_accepted
            else "UNIT_SCALE_ADDITIVE_GRID_STELLAR_UNAVAILABLE"
        )
        common_evidence["additiveModel"] = grid_evidence
        common_evidence["scalarOffsetFallback"] = False
    except CalibrationError as error:
        if error.code not in {
            "GLOBAL_NORMALIZATION_OFFSET_GRID_UNDERCONSTRAINED",
            "GLOBAL_NORMALIZATION_OFFSET_GRID_NOT_BENEFICIAL",
            "GLOBAL_NORMALIZATION_OFFSET_GRID_SPAN_UNSAFE",
        }:
            raise
        # Rejecting spatial correction does not invalidate the already fitted
        # finite scalar offset or the independently checked stellar scale.
        # Keep both scalar coefficients; never apply or clip the rejected grid.
        mode = (
            "STELLAR_SCALE_SCALAR_OFFSET"
            if stellar_scale_accepted
            else "UNIT_SCALE_SCALAR_OFFSET_STELLAR_UNAVAILABLE"
        )
        common_evidence["additiveModel"] = {
            "model": "SCALAR_OFFSET_FALLBACK",
            "status": "FALLBACK",
            "reasonCode": error.code,
            "reason": str(error),
            "coordinateConvention": "not-applicable",
        }
        common_evidence["scalarOffsetFallback"] = True
        if isinstance(error, _OffsetGridSpanUnsafe):
            common_evidence["additiveModel"]["rejectedGrid"] = error.evidence
    return GlobalNormalizationCoefficient(
        source_path=str(target.path),
        scale=scale,
        offset=offset,
        mode=mode,
        evidence=common_evidence,
        offset_grid=offset_grid,
        offset_grid_x=offset_grid_x,
        offset_grid_y=offset_grid_y,
    )


def fit_registered_group_global_normalization(
    paths: Sequence[str | os.PathLike[str]],
    *,
    reference_index: int,
    parameters: GlobalNormalizationParameters | None = None,
    stellar_scale_hints: Sequence[StellarScaleHint | None] | None = None,
    workers: int = 1,
) -> GlobalNormalizationResult:
    """Fit every target against the reference; ``workers`` fits targets
    concurrently and cannot change any coefficient."""

    parameters = parameters or GlobalNormalizationParameters()
    parameters.validate()
    if workers < 1:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_WORKERS_INVALID", "workers must be positive"
        )
    if not parameters.enabled:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_DISABLED", "global normalization is disabled"
        )
    canonical = tuple(Path(path).expanduser().resolve(strict=True) for path in paths)
    if not canonical:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_INPUT_EMPTY", "at least one frame is required"
        )
    if not 0 <= reference_index < len(canonical):
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_REFERENCE_INVALID",
            "reference index is out of range",
        )
    if len({os.path.normcase(str(path)) for path in canonical}) != len(canonical):
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_DUPLICATE_INPUT",
            "registered inputs are duplicated",
        )
    hints = tuple(stellar_scale_hints or (None,) * len(canonical))
    if len(hints) != len(canonical):
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_HINT_COUNT_MISMATCH",
            "stellar scale hint count differs from registered inputs",
        )
    reference_path = str(canonical[reference_index])
    for index, (path, hint) in enumerate(zip(canonical, hints, strict=True)):
        if hint is None:
            continue
        if not isinstance(hint, StellarScaleHint):
            raise CalibrationError(
                "GLOBAL_NORMALIZATION_HINT_INVALID",
                "stellar scale hints must use the typed identity-bound contract",
                path=str(path),
            )
        if hint.source_path != str(path) or hint.reference_path != reference_path:
            raise CalibrationError(
                "GLOBAL_NORMALIZATION_HINT_IDENTITY_MISMATCH",
                "stellar scale hint source/reference does not bind this registered group",
                path=str(path),
            )
        if index == reference_index and (
            hint.status != "REFERENCE_IDENTITY" or hint.scale != 1.0
        ):
            raise CalibrationError(
                "GLOBAL_NORMALIZATION_HINT_REFERENCE_INVALID",
                "reference stellar scale hint must be identity",
                path=str(path),
            )
        if index != reference_index and hint.status == "REFERENCE_IDENTITY":
            raise CalibrationError(
                "GLOBAL_NORMALIZATION_HINT_REFERENCE_INVALID",
                "only the selected reference may carry an identity scale hint",
                path=str(path),
            )

    frames: list[FitsFrame] = []
    reference_cache: _ReferenceSampleCache | None = None
    try:
        for path in canonical:
            frame = FitsFrame(path)
            frames.append(frame.__enter__())
        reference = frames[reference_index]
        reference_cache = _ReferenceSampleCache(reference)
        for frame in frames:
            if frame.shape != reference.shape:
                raise CalibrationError(
                    "GLOBAL_NORMALIZATION_GEOMETRY_MISMATCH",
                    "registered frame geometries differ",
                    path=str(frame.path),
                )
        def fit(index: int) -> GlobalNormalizationCoefficient:
            frame = frames[index]
            if index == reference_index:
                return GlobalNormalizationCoefficient(
                    str(frame.path),
                    1.0,
                    0.0,
                    "REFERENCE_IDENTITY",
                    {
                        "selectedBackgroundSamples": 0,
                        "brightStructureExcludedByJointQuantiles": True,
                        "multiplicativeScaleSpatialOrder": 0,
                        "additiveModel": {"model": "REFERENCE_IDENTITY"},
                        "stellarScale": (
                            hints[index].serializable()
                            if hints[index] is not None
                            else None
                        ),
                        "backgroundCovarianceUsedForScale": False,
                        "fallbackReason": None,
                    },
                )
            assert reference_cache is not None
            return _fit_coefficient(
                frame,
                reference,
                parameters,
                hints[index],
                reference_cache,
                native_threads,
            )

        fit_workers = max(1, min(workers, len(frames)))
        native_threads = max(1, workers // fit_workers)
        if fit_workers == 1:
            coefficients = [fit(index) for index in range(len(frames))]
        else:
            with ThreadPoolExecutor(
                max_workers=fit_workers, thread_name_prefix="oaf-normalize"
            ) as executor:
                coefficients = list(executor.map(fit, range(len(frames))))
    finally:
        while frames:
            frames.pop().close()

    receipt = {
        "schemaVersion": 1,
        "algorithm": GLOBAL_NORMALIZATION_VERSION,
        "status": "APPLIED",
        "pixInsightEquivalent": False,
        "parameters": parameters.serializable(),
        "referenceIndex": reference_index,
        "referenceSampleCache": (
            reference_cache.serializable()
            if reference_cache is not None
            else {"strategy": "not-initialized"}
        ),
        "frames": [
            item.serializable(reference=index == reference_index)
            for index, item in enumerate(coefficients)
        ],
        "scienceGuard": {
            "brightStructureExcludedFromFit": True,
            "multiplicativeScaleFromBackgroundCovariance": False,
            "multiplicativeScaleSpatiallyConstant": True,
            "additiveModelMayChangeOnlyGuardedLowFrequencies": True,
            "additiveEffectiveSmoothingSigmaPixels": (
                parameters.offset_tile_size_pixels
                * parameters.offset_smoothing_sigma_nodes
            ),
            "checkerboardHoldoutRequiredForGrid": True,
            "skyRelativeAmplitudeAndFirstDifferenceGates": True,
            "unsafeAdditiveGridFallsBackToScalarOffset": True,
            "missingStellarScaleFallsBackToOffsetOnly": (
                parameters.allow_offset_only_fallback
            ),
            "noNormalizationFallback": False,
        },
    }
    return GlobalNormalizationResult(
        tuple(coefficients), reference_index, receipt
    )


__all__ = [
    "GLOBAL_NORMALIZATION_VERSION",
    "GlobalNormalizationCoefficient",
    "GlobalNormalizationParameters",
    "GlobalNormalizationResult",
    "StellarScaleHint",
    "fit_registered_group_global_normalization",
]
