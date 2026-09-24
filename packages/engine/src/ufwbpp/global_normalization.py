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

Sky-proportional response.  Residual flat-field error multiplies the sky, so
its imprint on a frame (edge roll-off, a sky-dependent gradient) is
proportional to that frame's sky level while the object and any additive
instrumental signature are not.  When the sky level varies enough across a
group, a per-tile robust regression of tile background against frame sky
separates the two; the sky-proportional part is removed from every frame in
proportion to its own sky before the frames are matched to the reference, so
the master keeps the object and loses the sky-proportional structure that
matching alone cannot remove from the reference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .native_kernels import TILE_OFFSET_KERNEL_ID, load_native_kernels
from .calibration import CalibrationError, FitsFrame


GLOBAL_NORMALIZATION_VERSION = "matched-stellar-scale-plus-smoothed-additive-grid-v5"
SKY_RESPONSE_VERSION = "sky-proportional-edge-response-theil-sen-sensor-frame-v2"


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
    # The additive grid is skipped only when the spread of the tile offsets is
    # explained by their own measurement noise (or is a negligible fraction of
    # the sky); every applied grid still has to pass the checkerboard holdout.
    offset_grid_not_needed_sigma_fraction: float = 0.002
    offset_grid_not_needed_noise_multiple: float = 1.0
    maximum_offset_p05_p95_fraction: float = 0.25
    maximum_offset_span_fraction: float = 0.35
    maximum_offset_neighbor_sigma_fraction: float = 0.002
    sky_response_correction: bool = True
    sky_response_minimum_frames: int = 6
    sky_response_minimum_sky_ratio: float = 1.25
    sky_response_minimum_amplitude_fraction: float = 0.001
    # A residual flat-field error beyond 5% of the sky is not credible; a
    # larger "response" is a night-to-night gradient that happens to
    # correlate with the sky and must not be treated as instrumental.
    sky_response_maximum_fraction: float = 0.05
    # Gates on the tilt-free response.  The applied model is the set of
    # edge profiles (the response of each node row/column band along the
    # four sensor edges, the interior being zero): an edge is kept when its
    # profile is significant against the interior row/column noise and the
    # two interleaved halves of the group (ordered by sky) agree on it; the
    # first and second half in time only have to agree in sign overall,
    # and only when both halves have enough sky variation of their own;
    # the model must not add between-frame structure.
    sky_response_minimum_halves_correlation: float = 0.60
    sky_response_minimum_time_halves_correlation: float = 0.0
    sky_response_maximum_residual_ratio: float = 1.0
    sky_response_smoothing_sigma_nodes: float = 1.0
    sky_response_edge_significance_sigma: float = 3.0
    sky_response_minimum_halves_agreement: float = 0.4
    sky_response_object_fraction: float = 0.02
    sky_response_rows_per_tile: int = 32
    sky_response_column_stride: int = 2

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
            "offset_grid_not_needed_noise_multiple",
            "maximum_offset_p05_p95_fraction",
            "maximum_offset_span_fraction",
            "maximum_offset_neighbor_sigma_fraction",
            "sky_response_minimum_sky_ratio",
            "sky_response_minimum_amplitude_fraction",
            "sky_response_maximum_fraction",
            "sky_response_smoothing_sigma_nodes",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.sky_response_correction, bool):
            raise ValueError("sky_response_correction must be boolean")
        if self.sky_response_minimum_sky_ratio <= 1.0:
            raise ValueError("sky_response_minimum_sky_ratio must exceed 1")
        for name in (
            "sky_response_minimum_halves_correlation",
            "sky_response_minimum_time_halves_correlation",
            "sky_response_maximum_residual_ratio",
            "sky_response_minimum_halves_agreement",
            "sky_response_object_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            not math.isfinite(self.sky_response_edge_significance_sigma)
            or self.sky_response_edge_significance_sigma < 0.0
        ):
            raise ValueError("sky_response_edge_significance_sigma must be finite and >= 0")
        for name, floor in (
            ("sky_response_minimum_frames", 3),
            ("sky_response_rows_per_tile", 2),
            ("sky_response_column_stride", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"{name} must be an integer >= {floor}")

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
            "offsetGridNotNeededNoiseMultiple": self.offset_grid_not_needed_noise_multiple,
            "maximumOffsetP05P95Fraction": self.maximum_offset_p05_p95_fraction,
            "maximumOffsetSpanFraction": self.maximum_offset_span_fraction,
            "maximumOffsetNeighborSigmaFraction": self.maximum_offset_neighbor_sigma_fraction,
            "skyResponse": {
                "algorithm": SKY_RESPONSE_VERSION,
                "enabled": self.sky_response_correction,
                "minimumFrames": self.sky_response_minimum_frames,
                "minimumSkyRatio": self.sky_response_minimum_sky_ratio,
                "minimumAmplitudeFraction": self.sky_response_minimum_amplitude_fraction,
                "maximumFraction": self.sky_response_maximum_fraction,
                "minimumHalvesCorrelation": self.sky_response_minimum_halves_correlation,
                "minimumTimeHalvesCorrelation": self.sky_response_minimum_time_halves_correlation,
                "maximumResidualRatio": self.sky_response_maximum_residual_ratio,
                "smoothingSigmaNodes": self.sky_response_smoothing_sigma_nodes,
                "edgeSignificanceSigma": self.sky_response_edge_significance_sigma,
                "minimumHalvesAgreement": self.sky_response_minimum_halves_agreement,
                "objectFraction": self.sky_response_object_fraction,
                "rowsPerTile": self.sky_response_rows_per_tile,
                "columnStride": self.sky_response_column_stride,
            },
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


@lru_cache(maxsize=4096)
def _uniform_integer_indices(length: int, count: int) -> NDArray[np.int64]:
    """``count`` rounded uniform positions in ``[0, length)``, shared read-only.

    Every tile of a frame asks for the same few (length, count) pairs, so the
    array is built once; callers never write to it.
    """

    if count <= 1:
        values = np.zeros(1, dtype=np.int64)
    else:
        values = np.rint(np.linspace(0, length - 1, count, dtype=np.float64)).astype(np.int64)
    values.setflags(write=False)
    return values


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


def _sample_grid_lattice(
    grid: NDArray[np.float64],
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``_sample_grid_points`` on the product lattice ``y`` x ``x``.

    Returns ``(len(y), len(x))`` values equal element for element to sampling
    the meshgrid points: the node search and the weights are per axis, and
    each point's bilinear expression is evaluated in the same order.
    """

    x_clipped = np.clip(x, x_nodes[0], x_nodes[-1])
    y_clipped = np.clip(y, y_nodes[0], y_nodes[-1])
    x_hi = np.clip(np.searchsorted(x_nodes, x_clipped, side="right"), 1, len(x_nodes) - 1)
    y_hi = np.clip(np.searchsorted(y_nodes, y_clipped, side="right"), 1, len(y_nodes) - 1)
    x_lo = x_hi - 1
    y_lo = y_hi - 1
    wx = ((x_clipped - x_nodes[x_lo]) / (x_nodes[x_hi] - x_nodes[x_lo]))[None, :]
    wy = ((y_clipped - y_nodes[y_lo]) / (y_nodes[y_hi] - y_nodes[y_lo]))[:, None]
    rows_lo = grid[y_lo]
    rows_hi = grid[y_hi]
    top = rows_lo[:, x_lo] * (1.0 - wx) + rows_lo[:, x_hi] * wx
    bottom = rows_hi[:, x_lo] * (1.0 - wx) + rows_hi[:, x_hi] * wx
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
        self._lattices: dict[tuple[int, int], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
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

    def sky_lattice(
        self,
        grid: NDArray[np.float64],
        x_nodes: NDArray[np.float64],
        y_nodes: NDArray[np.float64],
        columns: NDArray[np.float64],
        rows: NDArray[np.float64],
        band: int,
    ) -> NDArray[np.float64]:
        """``_sample_grid_lattice`` of one sky-response grid on one tile row's
        sampled lattice, evaluated once per group: the lattice of a tile row
        is the same for every frame, and so is the shared response grid."""

        key = (id(grid), band)
        with self._lock:
            cached = self._lattices.get(key)
        if cached is not None and cached[0] is grid:
            return cached[1]
        values = _sample_grid_lattice(grid, x_nodes, y_nodes, columns, rows)
        values.setflags(write=False)
        with self._lock:
            self._lattices.setdefault(key, (grid, values))
        return values

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
    sky_response: _SkyResponse | None = None,
    target_sky: float = 0.0,
    reference_sky: float = 0.0,
    reference_response: _SkyResponse | None = None,
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
    if sky_response is not None and sky_response.applied:
        # Match the frames with their sky-proportional structure removed.
        x = x - sky_response.at(sample_x, sample_y, target_sky)
        y = y - (reference_response or sky_response).at(sample_x, sample_y, reference_sky)
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


@dataclass(frozen=True, slots=True)
class _SkyResponse:
    """Sky-proportional response of one registered group, on the offset grid nodes.

    ``grid`` is the response in the sensor frame (the geometry the residual
    flat-field structure lives in); ``frame_grids`` holds the same response
    carried onto the registered nodes of every frame through its registration
    transform, and is ``None`` when no frame was rotated or shifted by more
    than a node spacing, in which case ``grid`` serves every frame.
    """

    status: str
    grid: NDArray[np.float64] | None
    x_nodes: NDArray[np.float64]
    y_nodes: NDArray[np.float64]
    skies: tuple[float, ...]
    evidence: dict[str, Any]
    frame_grids: tuple[NDArray[np.float64], ...] | None = None

    @property
    def applied(self) -> bool:
        return self.status == "APPLIED" and self.grid is not None

    def for_frame(self, index: int) -> _SkyResponse:
        """The response on the registered nodes of frame ``index``."""

        if self.frame_grids is None:
            return self
        return replace(self, grid=self.frame_grids[index], frame_grids=None)

    def at(
        self, x: NDArray[np.float64], y: NDArray[np.float64], sky: float
    ) -> NDArray[np.float64]:
        """Sky-proportional term (frame units) of a frame with the given sky."""

        assert self.grid is not None
        return sky * _sample_grid_points(self.grid, self.x_nodes, self.y_nodes, x, y)


def _map_points(
    matrix: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply a 3x3 (affine or projective) pixel transform to points."""

    u = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    v = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return u / w, v / w


def _sample_node_map(
    grid: NDArray[np.float64],
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bilinear sample of a node map; NaN beyond the outer tiles (half a node
    spacing past the outer nodes, where the outer tile's level still holds)
    or next to a NaN node."""

    half_x = 0.5 * float(x_nodes[1] - x_nodes[0]) if len(x_nodes) > 1 else 0.0
    half_y = 0.5 * float(y_nodes[1] - y_nodes[0]) if len(y_nodes) > 1 else 0.0
    inside = (
        (x >= x_nodes[0] - half_x)
        & (x <= x_nodes[-1] + half_x)
        & (y >= y_nodes[0] - half_y)
        & (y <= y_nodes[-1] + half_y)
    )
    values = _sample_grid_points(grid, x_nodes, y_nodes, x, y)
    return np.where(inside & np.isfinite(values), values, np.nan)


def _sensor_frame_transforms(
    transforms: Sequence[Any] | None,
    count: int,
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
) -> list[NDArray[np.float64]] | None:
    """Registration matrices (sensor -> registered) when any of them moves the
    node grid by more than a tenth of a node spacing; ``None`` otherwise."""

    if transforms is None:
        return None
    matrices: list[NDArray[np.float64]] = []
    for transform in transforms:
        matrix = np.eye(3) if transform is None else np.asarray(transform, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or matrix[2, 2] == 0.0:
            raise CalibrationError(
                "GLOBAL_NORMALIZATION_TRANSFORM_INVALID",
                "registration transforms must be finite 3x3 matrices",
            )
        matrices.append(matrix / matrix[2, 2])
    if len(matrices) != count:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_TRANSFORM_COUNT_MISMATCH",
            "registration transform count differs from registered inputs",
        )
    grid_y, grid_x = np.meshgrid(y_nodes, x_nodes, indexing="ij")
    spacing = min(
        float(np.min(np.diff(x_nodes))) if len(x_nodes) > 1 else np.inf,
        float(np.min(np.diff(y_nodes))) if len(y_nodes) > 1 else np.inf,
    )
    for matrix in matrices:
        mapped_x, mapped_y = _map_points(matrix, grid_x, grid_y)
        if np.max(np.hypot(mapped_x - grid_x, mapped_y - grid_y)) > 0.1 * spacing:
            return matrices
    return None


def _orientation_classes(
    matrices: Sequence[NDArray[np.float64]],
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
    tolerance_fraction: float = 1.5,
) -> NDArray[np.int64]:
    """Group frames whose registration moves the node grid alike (within one
    and a half node spacings, so dithers and pointing offsets of a night stay
    together while a rotation or a half-turn separates): the sky-fixed
    structure then lies on the same sensor tiles for every frame of a class,
    to within the tile size."""

    grid_y, grid_x = np.meshgrid(y_nodes, x_nodes, indexing="ij")
    mapped = [np.stack(_map_points(matrix, grid_x, grid_y)) for matrix in matrices]
    spacing = min(
        float(np.min(np.diff(x_nodes))) if len(x_nodes) > 1 else np.inf,
        float(np.min(np.diff(y_nodes))) if len(y_nodes) > 1 else np.inf,
    )
    tolerance = tolerance_fraction * spacing
    classes = np.full(len(matrices), -1, dtype=np.int64)
    next_class = 0
    for index in range(len(matrices)):
        if classes[index] >= 0:
            continue
        classes[index] = next_class
        for other in range(index + 1, len(matrices)):
            if classes[other] < 0 and float(
                np.max(np.hypot(*(mapped[other] - mapped[index])))
            ) <= tolerance:
                classes[other] = next_class
        next_class += 1
    return classes


def _sensor_frame_slopes(
    registered_levels: NDArray[np.float64],
    skies: NDArray[np.float64],
    matrices: Sequence[NDArray[np.float64]],
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
    classes: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Theil-Sen slope of tile level against sky on the sensor nodes.

    Every frame's registered levels are read at the registered position of
    each sensor node.  Slopes come from pairs of frames of one orientation
    class only (see ``_orientation_classes``); the classes share the
    response.  Returns slopes, intercepts, the sensor-frame levels and the
    classes.
    """

    grid_y, grid_x = np.meshgrid(y_nodes, x_nodes, indexing="ij")
    sensor = np.stack(
        [
            _sample_node_map(registered_levels[index], x_nodes, y_nodes, *_map_points(matrix, grid_x, grid_y))
            for index, matrix in enumerate(matrices)
        ]
    )
    if classes is None:
        classes = _orientation_classes(matrices, x_nodes, y_nodes)
    beta, alpha = _theil_sen_slopes(skies, sensor, classes=classes)
    return beta, alpha, sensor, classes


def _response_on_registered_nodes(
    response: NDArray[np.float64],
    matrices: Sequence[NDArray[np.float64]],
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """The sensor-frame response carried onto every frame's registered nodes."""

    grid_y, grid_x = np.meshgrid(y_nodes, x_nodes, indexing="ij")
    grids = []
    for matrix in matrices:
        sensor_x, sensor_y = _map_points(np.linalg.inv(matrix), grid_x, grid_y)
        grids.append(
            np.ascontiguousarray(
                _sample_grid_points(response, x_nodes, y_nodes, sensor_x, sensor_y), dtype=np.float64
            )
        )
    return tuple(grids)


def _tile_backgrounds(
    frame: FitsFrame,
    parameters: GlobalNormalizationParameters,
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Robust background level of every offset tile of one frame.

    Only ``rows_per_tile`` rows per tile row and every ``column_stride``-th
    column are decoded, so a frame costs a few megabytes of reads.  Within a
    tile the samples between the lower and upper background quantiles are
    kept and their median is the tile level, mirroring the paired selection.

    Tiles whose samples are all finite are evaluated together per tile row:
    each tile's samples are sorted once, the quantiles come from the same
    NumPy linear interpolation on the sorted rows, the kept samples are the
    sorted run inside the bounds, and the median is its middle order
    statistic (the mean of the two middle values for an even count), exactly
    the values the per-tile evaluation yields.  Tiles touching NaN samples
    take the per-tile path.
    """

    height, width = frame.shape
    tile = parameters.offset_tile_size_pixels
    stride = parameters.sky_response_column_stride
    rows: list[NDArray[np.int64]] = []
    for grid_y in range(len(y_nodes)):
        y0 = grid_y * tile
        y1 = min(height, y0 + tile)
        rows.append(_uniform_integer_indices(y1 - y0, min(y1 - y0, parameters.sky_response_rows_per_tile)) + y0)
    y_indices = np.concatenate(rows)
    sampled = frame.read_sampled_rows(y_indices)[:, ::stride]
    levels = np.full((len(y_nodes), len(x_nodes)), np.nan, dtype=np.float64)
    minimum = max(16, parameters.minimum_samples_per_offset_tile // 8)
    minimum_kept = max(8, minimum // 2)
    quantiles = (parameters.lower_quantile, parameters.upper_quantile)
    # Column slice of every tile among the sampled (strided) columns.
    column_starts = [-(-grid_x * tile // stride) for grid_x in range(len(x_nodes))]
    column_stops = [-(-min(width, grid_x * tile + tile) // stride) for grid_x in range(len(x_nodes))]

    def per_tile(selected: NDArray[np.float64]) -> float:
        selected = selected[np.isfinite(selected)]
        if selected.size < minimum:
            return math.nan
        bounds = np.quantile(selected, quantiles)
        background = selected[(selected >= bounds[0]) & (selected <= bounds[1])]
        if background.size >= minimum_kept:
            return float(np.median(background))
        return math.nan

    row_start = 0
    for grid_y, band_rows in enumerate(rows):
        band = sampled[row_start : row_start + len(band_rows)].astype(np.float64)
        row_start += len(band_rows)
        widths = [column_stops[grid_x] - column_starts[grid_x] for grid_x in range(len(x_nodes))]
        finite_columns = np.all(np.isfinite(band), axis=0)
        for tile_width in sorted(set(widths)):
            members = [grid_x for grid_x in range(len(x_nodes)) if widths[grid_x] == tile_width]
            samples_per_tile = band.shape[0] * tile_width
            vectorised = [
                grid_x for grid_x in members
                if samples_per_tile >= minimum
                and bool(np.all(finite_columns[column_starts[grid_x]:column_stops[grid_x]]))
            ]
            for grid_x in members:
                if grid_x not in vectorised:
                    levels[grid_y, grid_x] = per_tile(
                        band[:, column_starts[grid_x]:column_stops[grid_x]].ravel()
                    )
            if not vectorised:
                continue
            block = np.stack(
                [band[:, column_starts[grid_x]:column_stops[grid_x]].ravel() for grid_x in vectorised]
            )
            ordered = np.sort(block, axis=1)
            bounds = np.quantile(ordered, quantiles, axis=1)
            first = np.count_nonzero(ordered < bounds[0][:, None], axis=1)
            last = np.count_nonzero(ordered <= bounds[1][:, None], axis=1)
            count = last - first
            middle = first + count // 2
            upper = np.take_along_axis(ordered, middle[:, None], axis=1)[:, 0]
            lower = np.take_along_axis(ordered, np.maximum(middle - 1, 0)[:, None], axis=1)[:, 0]
            median = np.where(count % 2 == 1, upper, (lower + upper) / 2)
            usable = count >= minimum_kept
            for position, grid_x in enumerate(vectorised):
                if usable[position]:
                    levels[grid_y, grid_x] = float(median[position])
    return levels


SKY_RESPONSE_INTERIOR_MARGIN = 0.15


def _interior_plane_fit(
    response: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64] | None:
    """Robust plane of a node map fitted on the interior of the frame.

    The outer 15% on every side is excluded, so an edge roll-off cannot leak
    into the plane.  Returns the plane evaluated on every node (planes are
    defined everywhere), or ``None`` when too few interior nodes are usable.
    """

    rows, columns = np.indices(response.shape)
    height, width = response.shape
    interior = (
        (rows >= SKY_RESPONSE_INTERIOR_MARGIN * height)
        & (rows < (1.0 - SKY_RESPONSE_INTERIOR_MARGIN) * height)
        & (columns >= SKY_RESPONSE_INTERIOR_MARGIN * width)
        & (columns < (1.0 - SKY_RESPONSE_INTERIOR_MARGIN) * width)
    )
    design = np.column_stack(
        (np.ones(response.size), columns.ravel() / max(width - 1, 1), rows.ravel() / max(height - 1, 1))
    )
    values = response.ravel()
    usable = (valid & interior & np.isfinite(response)).ravel()
    keep = usable
    if np.count_nonzero(keep) < 6:
        return None
    coefficients = np.zeros(3)
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = values - design @ coefficients
        scale = 1.4826 * float(np.nanmedian(np.abs(residual[keep])))
        keep = usable & (np.abs(np.nan_to_num(residual, nan=np.inf)) <= 3.0 * scale + 1e-12)
        if np.count_nonzero(keep) < 6:
            break
    return (design @ coefficients).reshape(response.shape)


def _high_pass_response(
    response: NDArray[np.float64],
    valid: NDArray[np.bool_],
    sigma_nodes: float,
) -> NDArray[np.float64]:
    """Remove the global tilt of a response map.

    A robust plane fitted on the interior of the frame (see
    ``_interior_plane_fit``) captures the light-pollution or moonlit gradient
    that happens to scale with the sky; the per-frame additive fit against
    the reference already models such tilts.  Everything else, notably the
    edge roll-off of a flat-field residual, stays in the response.
    ``sigma_nodes`` is unused (kept for the call signature).
    """

    plane = _interior_plane_fit(response, valid)
    if plane is None:
        return np.where(valid, response, np.nan)
    return np.where(valid, response - plane, np.nan)


def _theil_sen_slopes(
    skies: NDArray[np.float64],
    levels: NDArray[np.float64],
    maximum_pairs: int = 2016,
    classes: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-tile Theil-Sen slope and intercept of tile level against frame sky.

    With ``classes``, only pairs of frames of the same class form slopes:
    frames of different registration orientations see the object on
    different sensor tiles, so their intercepts differ and a pair across
    classes would read that difference as a slope.
    """

    count = len(skies)
    first, second = np.triu_indices(count, 1)
    if classes is not None:
        same = classes[first] == classes[second]
        first, second = first[same], second[same]
    if first.size > maximum_pairs:
        step = int(math.ceil(first.size / maximum_pairs))
        first, second = first[::step], second[::step]
    delta = skies[second] - skies[first]
    usable = np.abs(delta) > 1e-9
    first, second, delta = first[usable], second[usable], delta[usable]
    slopes = (levels[second] - levels[first]) / delta[:, None, None]
    with _quiet_nan():
        beta = np.nanmedian(slopes, axis=0)
        # A node needs three pairs; fewer give a slope no better than noise.
        beta = np.where(np.count_nonzero(np.isfinite(slopes), axis=0) >= 3, beta, np.nan)
        alpha = np.nanmedian(levels - beta[None, :, :] * skies[:, None, None], axis=0)
    return beta, alpha


def _group_tile_levels(
    frames: Sequence[FitsFrame],
    parameters: GlobalNormalizationParameters,
    workers: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Tile background levels of every frame, and their node coordinates."""

    reference_shape = frames[0].shape
    x_nodes = _grid_coordinates(reference_shape[1], parameters.offset_tile_size_pixels)
    y_nodes = _grid_coordinates(reference_shape[0], parameters.offset_tile_size_pixels)

    def levels_of(index: int) -> NDArray[np.float64]:
        return _tile_backgrounds(frames[index], parameters, x_nodes, y_nodes)

    fit_workers = max(1, min(workers, len(frames)))
    if fit_workers == 1:
        levels = np.stack([levels_of(index) for index in range(len(frames))])
    else:
        with ThreadPoolExecutor(max_workers=fit_workers, thread_name_prefix="ufwbpp-sky") as executor:
            levels = np.stack(list(executor.map(levels_of, range(len(frames)))))
    skies = np.nanmedian(levels.reshape(len(frames), -1), axis=1)
    return levels, skies, x_nodes, y_nodes


def _interior_plane(
    values: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Robust interior plane of a node map, evaluated on every node."""

    plane = _interior_plane_fit(values, valid)
    if plane is None:
        return np.full(values.shape, np.nan)
    return plane


@dataclass(frozen=True, slots=True)
class _LowOrderTarget:
    """Common low-order correction that moves the master's background tilt.

    The master inherits the tilt of the additive reference.  The target tilt
    is the flattest convex combination of the frames' own tilts (in the
    registered geometry, on the reference scale): a master whose background
    is a weighted mix of the observed backgrounds, never flatter than the
    observations allow, and no astrophysical plane is invented or removed
    beyond what the frames disagree on.  When the nights' gradients point in
    different directions (meridian flips, moon on the other side) they
    cancel; a single night keeps its common tilt.  The difference between
    the reference's tilt and the target is added to every frame's grid, so
    frame-to-frame matching is unchanged.
    """

    correction: NDArray[np.float64] | None
    evidence: dict[str, Any]


def _flattest_convex_combination(planes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weights >= 0 summing to one that minimise the RMS of the mixed plane.

    A non-negative least-squares fit of ``sum_i w_i plane_i`` to zero, with
    the sum-to-one constraint as a heavily weighted extra equation
    (Lawson-Hanson, deterministic).
    """

    from scipy.optimize import nnls

    count, samples = planes.shape
    scale = float(np.max(np.abs(planes))) if planes.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.full(count, 1.0 / count)
    constraint = 1e3 * scale
    design = np.vstack((planes.T, np.full((1, count), constraint)))
    target = np.concatenate((np.zeros(samples), [constraint]))
    weights, _ = nnls(design, target)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(count, 1.0 / count)
    return weights / total


def _fit_low_order_target(
    levels: NDArray[np.float64],
    skies: NDArray[np.float64],
    scales: Sequence[float],
    reference_index: int,
    minimum_frames: int,
) -> _LowOrderTarget:
    count = levels.shape[0]
    if count < minimum_frames:
        return _LowOrderTarget(
            None,
            {"status": "NOT_APPLICABLE", "reason": f"{count} frames are fewer than the {minimum_frames} required"},
        )
    valid = np.all(np.isfinite(levels), axis=0)
    if np.count_nonzero(valid) < 12 or not np.all(np.isfinite(skies)):
        return _LowOrderTarget(None, {"status": "NOT_APPLICABLE", "reason": "too few tiles"})
    planes = np.full(levels.shape, np.nan)
    for index in range(count):
        centred = levels[index] - skies[index]
        planes[index] = float(scales[index]) * _interior_plane(np.where(valid, centred, np.nan), valid)
    if not np.all(np.isfinite(planes[:, valid])):
        return _LowOrderTarget(None, {"status": "NOT_APPLICABLE", "reason": "frame tilts are undefined"})
    planes -= np.nanmedian(planes[:, valid], axis=1)[:, None, None]
    samples = planes[:, valid]

    def amplitude(plane: NDArray[np.float64]) -> float:
        return float(np.percentile(plane, 99) - np.percentile(plane, 1))

    # The weights are fitted where every frame has a level; the planes are
    # defined on every node, so the correction has no seam at the border
    # tiles that some frames miss.
    weights = _flattest_convex_combination(samples)
    target = np.tensordot(weights, planes, axes=1)
    amplitudes = [amplitude(samples[index]) for index in range(count)]
    flattest_frame = int(np.argmin(amplitudes))
    correction = np.nan_to_num(target - planes[reference_index], nan=0.0)
    support = [int(index) for index in np.flatnonzero(weights > 1e-6)]
    return _LowOrderTarget(
        correction,
        {
            "status": "APPLIED",
            "rule": "flattest-convex-combination-of-frame-tilts-v1",
            "chosen": "convex-combination",
            "supportFrames": support,
            "weights": [round(float(weights[index]), 4) for index in support],
            "referencePlaneAmplitude": amplitudes[reference_index],
            "targetPlaneAmplitude": amplitude(weights @ samples),
            "flattestFramePlaneAmplitude": amplitudes[flattest_frame],
            "flattestFrame": flattest_frame,
            "groupMeanPlaneAmplitude": amplitude(samples.mean(axis=0)),
            "correctionAmplitude": float(np.max(correction) - np.min(correction)),
        },
    )


SKY_RESPONSE_EDGES = ("top", "bottom", "left", "right")


class _quiet_nan(warnings.catch_warnings):
    """Silence numpy's all-NaN reductions: an empty node is a NaN by design."""

    def __enter__(self) -> None:  # type: ignore[override]
        super().__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)


def _row_medians(values: NDArray[np.float64]) -> NDArray[np.float64]:
    with _quiet_nan():
        return np.nanmedian(values, axis=1)


def _json_finite(value: Any) -> Any:
    """Receipt-safe copy: non-finite floats become ``None``."""

    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _edge_depths(shape: tuple[int, int]) -> tuple[int, int]:
    """Node rows and columns from each edge that the edge model covers: the
    outer 15% of the grid, the part the interior plane never sees."""

    return (
        max(1, int(math.floor(SKY_RESPONSE_INTERIOR_MARGIN * shape[0]))),
        max(1, int(math.floor(SKY_RESPONSE_INTERIOR_MARGIN * shape[1]))),
    )


def _edge_profiles(
    response: NDArray[np.float64], valid: NDArray[np.bool_]
) -> tuple[dict[str, NDArray[np.float64]], dict[str, float]]:
    """Median response of every node band along each edge, from the edge
    inwards, relative to the interior (whose median is zero), with the
    noise of an interior row/column median."""

    masked = np.where(valid, response, np.nan)
    rows, columns = response.shape
    depth_y, depth_x = _edge_depths(response.shape)
    inner_rows = slice(depth_y, rows - depth_y)
    inner_columns = slice(depth_x, columns - depth_x)
    with _quiet_nan():
        row_medians = np.nanmedian(masked[:, inner_columns], axis=1)
        column_medians = np.nanmedian(masked[inner_rows, :], axis=0)
    profiles = {
        "top": row_medians[:depth_y],
        "bottom": row_medians[::-1][:depth_y],
        "left": column_medians[:depth_x],
        "right": column_medians[::-1][:depth_x],
    }
    interior_rows = row_medians[inner_rows]
    interior_columns = column_medians[inner_columns]
    with _quiet_nan():
        interior = float(np.nanmedian(masked[inner_rows, inner_columns]))
    profiles = {
        name: np.nan_to_num(values - interior, nan=0.0) for name, values in profiles.items()
    }
    noise = {
        "row": _location_and_mad(interior_rows[np.isfinite(interior_rows)])[1]
        if np.count_nonzero(np.isfinite(interior_rows)) >= 3 else float("nan"),
        "column": _location_and_mad(interior_columns[np.isfinite(interior_columns)])[1]
        if np.count_nonzero(np.isfinite(interior_columns)) >= 3 else float("nan"),
    }
    return profiles, noise


def _tapered(profile: NDArray[np.float64]) -> NDArray[np.float64]:
    """The profile continued linearly from its innermost band to zero over as
    many further bands, so the model joins the interior without a step."""

    depth = profile.size
    if depth == 0:
        return profile
    ramp = profile[-1] * (np.arange(depth, 0, -1, dtype=np.float64) / (depth + 1.0))
    return np.concatenate((profile, ramp))


def _edge_model(profiles: dict[str, NDArray[np.float64]], shape: tuple[int, int]) -> NDArray[np.float64]:
    """The response grid built from the kept edge profiles, each tapered into
    the interior (zero elsewhere)."""

    model = np.zeros(shape, dtype=np.float64)
    rows, columns = shape
    for depth, value in enumerate(_tapered(profiles.get("top", np.zeros(0)))):
        if depth < rows:
            model[depth, :] += value
    for depth, value in enumerate(_tapered(profiles.get("bottom", np.zeros(0)))):
        if depth < rows:
            model[rows - 1 - depth, :] += value
    for depth, value in enumerate(_tapered(profiles.get("left", np.zeros(0)))):
        if depth < columns:
            model[:, depth] += value
    for depth, value in enumerate(_tapered(profiles.get("right", np.zeros(0)))):
        if depth < columns:
            model[:, columns - 1 - depth] += value
    return model


def _peak(profile: NDArray[np.float64]) -> float:
    """Signed value of largest magnitude of a profile."""

    if profile.size == 0:
        return 0.0
    return float(profile[int(np.argmax(np.abs(profile)))])


def _fit_sky_response(
    frames: Sequence[FitsFrame],
    parameters: GlobalNormalizationParameters,
    workers: int,
    precomputed: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None = None,
    transforms: Sequence[Any] | None = None,
) -> _SkyResponse:
    """Sky-proportional response of the group, regressed in the sensor frame.

    Residual flat-field structure is fixed on the sensor, so frames taken
    after a meridian flip (or with any rotation) carry it on the opposite
    side of the registered geometry.  Regressing in registered coordinates
    would then attribute the night-to-night difference to the sky; the
    levels are therefore read in the sensor frame through each frame's
    registration transform, the response is fitted there, and carried back
    onto every frame's registered nodes.
    """

    reference_shape = frames[0].shape
    x_nodes = _grid_coordinates(reference_shape[1], parameters.offset_tile_size_pixels)
    y_nodes = _grid_coordinates(reference_shape[0], parameters.offset_tile_size_pixels)

    def skipped(status: str, reason: str, **extra: Any) -> _SkyResponse:
        return _SkyResponse(
            status,
            None,
            x_nodes,
            y_nodes,
            tuple(extra.pop("skies", ())),
            _json_finite({"algorithm": SKY_RESPONSE_VERSION, "status": status, "reason": reason, **extra}),
        )

    if not parameters.sky_response_correction:
        return skipped("DISABLED", "sky-proportional response correction is disabled")
    if len(frames) < parameters.sky_response_minimum_frames:
        return skipped(
            "NOT_APPLICABLE",
            f"{len(frames)} frames are fewer than the {parameters.sky_response_minimum_frames} required",
        )
    if len(x_nodes) < 3 or len(y_nodes) < 3:
        return skipped("NOT_APPLICABLE", "image is too small for a sky-response grid")

    if precomputed is None:
        levels, skies, _, _ = _group_tile_levels(frames, parameters, workers)
    else:
        levels, skies, _, _ = precomputed
    if not np.all(np.isfinite(skies)) or np.any(skies <= 0):
        return skipped("NOT_APPLICABLE", "a frame has no measurable sky level", skies=[float(v) for v in skies])
    matrices = _sensor_frame_transforms(transforms, len(frames), x_nodes, y_nodes)
    registered_levels = levels
    classes: NDArray[np.int64] | None = (
        _orientation_classes(matrices, x_nodes, y_nodes) if matrices is not None else None
    )

    def usable_sky_ratio(members: NDArray[np.int64]) -> float:
        """Largest sky range among the orientation classes of ``members``:
        slopes only come from frames of one class."""

        labels = classes[members] if classes is not None else np.zeros(len(members), dtype=np.int64)
        ratios = [
            float(np.max(skies[members][labels == label]) / np.min(skies[members][labels == label]))
            for label in np.unique(labels)
            if np.count_nonzero(labels == label) >= 2
        ]
        return max(ratios) if ratios else 1.0

    sky_ratio = usable_sky_ratio(np.arange(len(frames)))
    common: dict[str, Any] = {
        "skyLevels": [float(value) for value in skies],
        "skyRatio": sky_ratio,
        "skyRatioMeasure": "largest-within-orientation-class",
        "tileGrid": [int(len(y_nodes)), int(len(x_nodes))],
        "regressionFrame": "sensor" if matrices is not None else "registered",
    }
    if classes is not None:
        common["orientationClasses"] = [int(value) for value in classes]
    if sky_ratio < parameters.sky_response_minimum_sky_ratio:
        return skipped(
            "NOT_APPLICABLE",
            f"sky ratio {sky_ratio:.4g} is below {parameters.sky_response_minimum_sky_ratio:.4g}; "
            "the sky-proportional and additive structure cannot be separated",
            skies=[float(v) for v in skies],
            **common,
        )
    if matrices is None:
        valid = np.all(np.isfinite(levels), axis=0)
        if np.count_nonzero(valid) < max(parameters.minimum_valid_offset_tiles, 3 * 3):
            return skipped("NOT_APPLICABLE", "too few tiles have a level in every frame", skies=[float(v) for v in skies], **common)
        beta, alpha = _theil_sen_slopes(skies, levels)
    else:
        # ``levels`` become the levels in the sensor frame; the halves below
        # regress the same quantity within the same orientation classes.  A
        # sensor node that some frames do not cover (the outermost band of a
        # rotated frame) keeps its slope from the frames that do.
        assert classes is not None
        beta, alpha, levels, _ = _sensor_frame_slopes(levels, skies, matrices, x_nodes, y_nodes, classes)
        valid = np.count_nonzero(np.isfinite(levels), axis=0) >= 3
        if np.count_nonzero(valid) < max(parameters.minimum_valid_offset_tiles, 3 * 3):
            return skipped("NOT_APPLICABLE", "too few sensor tiles have a level in three frames", skies=[float(v) for v in skies], **common)
    beta_scale = float(np.nanmedian(beta[valid]))
    if not math.isfinite(beta_scale) or beta_scale <= 0:
        return skipped("NOT_APPLICABLE", "the sky response has no positive median slope", skies=[float(v) for v in skies], **common)
    # Tiles the object dominates (more than a few per cent of the sky above
    # it) cannot measure a per-cent response; they are left out of the fit.
    sky_median = float(np.median(skies))
    object_tiles = np.zeros(valid.shape, dtype=bool)
    for label in np.unique(classes) if classes is not None else (0,):
        members = classes == label if classes is not None else np.ones(len(frames), dtype=bool)
        with _quiet_nan():
            structure = np.nanmedian(levels[members] - skies[members][:, None, None], axis=0)
        object_tiles |= np.nan_to_num(structure, nan=0.0) > parameters.sky_response_object_fraction * sky_median
    valid = valid & ~object_tiles
    common["objectTilesExcluded"] = int(np.count_nonzero(object_tiles))
    if np.count_nonzero(valid) < max(parameters.minimum_valid_offset_tiles, 3 * 3):
        return skipped("NOT_APPLICABLE", "too few object-free tiles", skies=[float(v) for v in skies], **common)
    full_response = beta / beta_scale - 1.0
    response = _high_pass_response(full_response, valid, parameters.offset_smoothing_sigma_nodes)
    _, low_order_amplitude = _location_and_mad((full_response - np.nan_to_num(response))[valid])
    common["lowOrderResponseMadFraction"] = low_order_amplitude
    common["lowOrderResponseLeftToAdditiveGrid"] = True
    common["lowOrderModel"] = "robust-interior-plane"
    def half_response(half: NDArray[np.int64]) -> NDArray[np.float64] | None:
        """Tilt-free response fitted on a subset of the frames."""

        if len(half) < 3:
            return None
        half_beta, _ = _theil_sen_slopes(
            skies[half], levels[half], classes=None if classes is None else classes[half]
        )
        half_scale = float(np.nanmedian(half_beta[valid]))
        if not math.isfinite(half_scale) or half_scale <= 0:
            return None
        return _high_pass_response(half_beta / half_scale - 1.0, valid, parameters.offset_smoothing_sigma_nodes)

    def node_correlation(first: NDArray[np.float64] | None, second: NDArray[np.float64] | None) -> float:
        if first is None or second is None:
            return float("nan")
        pair = np.isfinite(first) & np.isfinite(second) & valid
        if np.count_nonzero(pair) < 8 or np.std(first[pair]) == 0 or np.std(second[pair]) == 0:
            return float("nan")
        return float(np.corrcoef(first[pair], second[pair])[0, 1])

    order = np.argsort(skies, kind="stable")
    halves = (half_response(order[::2]), half_response(order[1::2]))
    sky_correlation = node_correlation(halves[0], halves[1])
    common["halvesCorrelation"] = sky_correlation
    profiles, noise = _edge_profiles(response, valid)
    amplitude = max(abs(_peak(profile)) for profile in profiles.values())
    if math.isfinite(sky_correlation) and sky_correlation >= parameters.sky_response_minimum_halves_correlation:
        # The two halves reproduce the whole map: apply it as it is, lightly
        # smoothed, with the object tiles filled from their neighbours.
        common["responseModel"] = "node-map"
        _, node_amplitude = _location_and_mad(response[valid])
        amplitude = max(amplitude, node_amplitude)
        common["responseAmplitudeFraction"] = amplitude
        common["responseAmplitudeMeasure"] = "max(largest-edge-profile-value, node-mad)"
        if amplitude < parameters.sky_response_minimum_amplitude_fraction:
            return skipped(
                "NOT_NEEDED",
                f"sky-proportional structure {amplitude:.3g} is below {parameters.sky_response_minimum_amplitude_fraction:.3g} of the sky",
                skies=[float(v) for v in skies],
                **common,
            )
        filled = _fill_grid(np.where(valid, response, np.nan), valid)
        grid = np.ascontiguousarray(
            gaussian_filter(filled, sigma=parameters.sky_response_smoothing_sigma_nodes, mode="nearest"),
            dtype=np.float64,
        )
        kept_edges: list[str] = []

        def half_summary(half: NDArray[np.int64]) -> NDArray[np.float64] | None:
            estimate = half_response(half)
            return None if estimate is None else np.where(valid, estimate, np.nan)

    else:
        # Only the edge bands are reproducible: apply their profiles.  An
        # edge is kept when its profile is significant against the interior
        # row/column noise, is shaped like a roll-off (largest at or next to
        # the edge, decaying inwards) and the two halves agree on it.
        common["responseModel"] = "edge-band-profiles"
        common["responseAmplitudeFraction"] = amplitude
        common["responseAmplitudeMeasure"] = "largest-edge-profile-value"
        if amplitude < parameters.sky_response_minimum_amplitude_fraction:
            return skipped(
                "NOT_NEEDED",
                f"sky-proportional edge structure {amplitude:.3g} is below {parameters.sky_response_minimum_amplitude_fraction:.3g} of the sky",
                skies=[float(v) for v in skies],
                **common,
            )
        half_edges = tuple(
            None if estimate is None else _edge_profiles(estimate, valid)[0] for estimate in halves
        )
        edges: dict[str, dict[str, Any]] = {}
        kept: dict[str, NDArray[np.float64]] = {}
        for name in SKY_RESPONSE_EDGES:
            profile = profiles[name]
            sigma = noise["row"] if name in ("top", "bottom") else noise["column"]
            peak = _peak(profile)
            significant = (
                math.isfinite(sigma)
                and abs(peak) >= parameters.sky_response_edge_significance_sigma * sigma
                and abs(peak) >= parameters.sky_response_minimum_amplitude_fraction
            )
            edge_shaped = bool(
                profile.size > 0
                and int(np.argmax(np.abs(profile))) <= 1
                and abs(profile[-1]) <= 0.5 * abs(profile[0])
            )
            agreement = float("nan")
            if half_edges[0] is not None and half_edges[1] is not None:
                first, second = _peak(half_edges[0][name]), _peak(half_edges[1][name])
                if first * second > 0 and first * peak > 0:
                    agreement = min(abs(first), abs(second)) / max(abs(first), abs(second))
                else:
                    agreement = 0.0
            reproducible = math.isfinite(agreement) and agreement >= parameters.sky_response_minimum_halves_agreement
            edges[name] = {
                "profile": [float(value) for value in profile],
                "peak": peak,
                "noiseSigma": sigma,
                "significant": bool(significant),
                "edgeShaped": edge_shaped,
                "halvesAgreement": agreement,
                "kept": bool(significant and edge_shaped and reproducible),
            }
            if significant and edge_shaped and reproducible:
                kept[name] = profile
        common["edges"] = edges
        kept_edges = sorted(kept)
        if not kept:
            return skipped(
                "REJECTED",
                "no edge of the sky response is significant, roll-off shaped and reproduced by the two frame halves",
                skies=[float(v) for v in skies],
                **common,
            )
        grid = np.ascontiguousarray(_edge_model(kept, response.shape), dtype=np.float64)

        def half_summary(half: NDArray[np.int64]) -> NDArray[np.float64] | None:
            estimate = half_response(half)
            if estimate is None:
                return None
            half_profiles = _edge_profiles(estimate, valid)[0]
            return np.concatenate([half_profiles[name] for name in kept_edges])

    common["keptEdges"] = kept_edges
    # The first and second half in time must not disagree in sign, when
    # each half has enough sky variation to measure a response at all.
    sequence = np.arange(len(frames))
    time_halves = (sequence[: len(frames) // 2], sequence[len(frames) // 2 :])
    time_ratios = [usable_sky_ratio(half) if len(half) else 1.0 for half in time_halves]
    common["timeHalvesSkyRatios"] = time_ratios
    if min(time_ratios) >= parameters.sky_response_minimum_sky_ratio:
        first, second = half_summary(time_halves[0]), half_summary(time_halves[1])
        if first is not None and second is not None:
            pair = np.isfinite(first) & np.isfinite(second)
            if np.count_nonzero(pair) >= 2 and np.std(first[pair]) > 0 and np.std(second[pair]) > 0:
                time_correlation = float(np.corrcoef(first[pair], second[pair])[0, 1])
                common["timeHalvesCorrelation"] = time_correlation
                if time_correlation < parameters.sky_response_minimum_time_halves_correlation:
                    return skipped(
                        "REJECTED",
                        f"the first and second half of the sequence disagree on the sky response (correlation {time_correlation:.3g})",
                        skies=[float(v) for v in skies],
                        **common,
                    )
    else:
        common["timeHalvesCorrelation"] = None
        common["timeHalvesNotTestable"] = "a time half has too little sky variation"

    extreme = float(np.max(np.abs(grid)))
    common.update(
        {
            "responseMinimumFraction": float(np.min(grid)),
            "responseMaximumFraction": float(np.max(grid)),
            "rowMedianFractions": [float(value) for value in np.median(grid, axis=1)],
            "columnMedianFractions": [float(value) for value in np.median(grid, axis=0)],
            "additiveRowMedians": [float(value) for value in _row_medians(np.where(valid, alpha, np.nan))],
        }
    )
    if extreme > parameters.sky_response_maximum_fraction:
        return skipped(
            "REJECTED",
            f"sky response reaches {extreme:.3g} of the sky, above {parameters.sky_response_maximum_fraction:.3g}",
            skies=[float(v) for v in skies],
            **common,
        )
    frame_grids = (
        _response_on_registered_nodes(grid, matrices, x_nodes, y_nodes)
        if matrices is not None
        else None
    )
    # The model must reduce the between-frame differences on the tiles it
    # touches and never add to them: before = each frame's tile levels less
    # its own scalar and the group's common structure; after = the same
    # with the applied response removed in proportion to each frame's sky.
    # Judged in the registered frame, where the object is common to all
    # frames and only the sensor-fixed response moves from frame to frame.
    gate_valid = np.all(np.isfinite(registered_levels), axis=0)
    applied = frame_grids if frame_grids is not None else (grid,) * len(frames)
    model = np.stack([beta_scale * skies[index] * applied[index] for index in range(len(frames))])
    touched = gate_valid & np.any(model != 0.0, axis=0)

    def between_frame_structure(values: NDArray[np.float64]) -> NDArray[np.float64]:
        own = values - np.nanmedian(values.reshape(len(frames), -1), axis=1)[:, None, None]
        own = own - np.nanmedian(own, axis=0)[None, :, :]
        # Compare at the scales the response corrects: high-pass.
        return np.stack([
            _high_pass_response(frame, gate_valid, parameters.offset_smoothing_sigma_nodes) for frame in own
        ])

    centred = between_frame_structure(registered_levels)
    residual = between_frame_structure(registered_levels - model)
    before_rms = float(np.nanmedian([np.nanstd(frame[touched]) for frame in centred]))
    after_rms = float(np.nanmedian([np.nanstd(frame[touched]) for frame in residual]))
    common["gateTiles"] = int(np.count_nonzero(touched))
    residual_ratio = after_rms / before_rms if before_rms > 0 else float("inf")
    common["residualRmsBefore"] = before_rms
    common["residualRmsAfter"] = after_rms
    common["residualRatio"] = residual_ratio
    if not math.isfinite(residual_ratio) or residual_ratio > parameters.sky_response_maximum_residual_ratio:
        return skipped(
            "REJECTED",
            f"the sky-proportional model leaves {residual_ratio:.3g} of the between-frame background structure",
            skies=[float(v) for v in skies],
            **common,
        )
    evidence = {
        "algorithm": SKY_RESPONSE_VERSION,
        "status": "APPLIED",
        "coordinateConvention": "zero-based-pixel-centres-bilinear-node-grid",
        "xNodes": [float(value) for value in x_nodes],
        "yNodes": [float(value) for value in y_nodes],
        "sha256": _offset_grid_sha256(
            tuple(tuple(float(v) for v in row) for row in grid),
            tuple(float(v) for v in x_nodes),
            tuple(float(v) for v in y_nodes),
        ),
        **common,
    }
    return _SkyResponse(
        "APPLIED", grid, x_nodes, y_nodes, tuple(float(v) for v in skies), _json_finite(evidence), frame_grids
    )


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
    sky_response: _SkyResponse | None = None,
    target_sky: float = 0.0,
    reference_sky_level: float = 0.0,
    reference_response: _SkyResponse | None = None,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[float, ...],
    dict[str, Any],
]:
    height, width = reference.shape
    x_nodes = _grid_coordinates(width, parameters.offset_tile_size_pixels)
    y_nodes = _grid_coordinates(height, parameters.offset_tile_size_pixels)
    correct_sky = sky_response is not None and sky_response.applied
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
        band_columns = np.concatenate(x_indices_by_tile)
        column_starts = np.concatenate(
            ([0], np.cumsum([indices.size for indices in x_indices_by_tile]))
        )
        if correct_sky:
            # The sky terms of the whole band's sampled lattice at once; each
            # tile then takes its own columns, so the per-tile samples are
            # exactly those of the per-tile evaluation.
            assert sky_response is not None and sky_response.grid is not None
            reference_term = reference_response or sky_response
            assert reference_term.grid is not None
            band_rows = y_indices.astype(np.float64)
            target_term = target_sky * reference_cache.sky_lattice(
                sky_response.grid, sky_response.x_nodes, sky_response.y_nodes,
                band_columns.astype(np.float64), band_rows, grid_y,
            )
            reference_sky_term = reference_sky_level * reference_cache.sky_lattice(
                reference_term.grid, reference_term.x_nodes, reference_term.y_nodes,
                band_columns.astype(np.float64), band_rows, grid_y,
            )
        # The band's sampled columns are gathered once; each tile's samples
        # are its contiguous run, in the same row-major order.
        band_target = np.asarray(sampled_target[:, band_columns], dtype=np.float64)
        band_reference = np.asarray(sampled_reference[:, band_columns], dtype=np.float64)
        if correct_sky:
            band_target = band_target - target_term
            band_reference = band_reference - reference_sky_term
        for grid_x in range(len(x_indices_by_tile)):
            columns = slice(int(column_starts[grid_x]), int(column_starts[grid_x + 1]))
            tile_target = band_target[:, columns].ravel()
            tile_reference = band_reference[:, columns].ravel()
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
    # Standard error of one tile's median offset from its own residual
    # dispersion: the grid is pointless when the tile offsets scatter no more
    # than their measurement noise.
    tile_offset_noise = float(
        np.median(
            1.2533 * residual_mads[valid] / np.sqrt(np.maximum(sample_counts[valid], 1))
        )
    )
    not_needed_floor = max(
        parameters.offset_grid_not_needed_sigma_fraction * reference_sky,
        parameters.offset_grid_not_needed_noise_multiple * tile_offset_noise,
    )
    if raw_offset_sigma <= not_needed_floor:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_NOT_BENEFICIAL",
            f"low-frequency offset sigma {raw_offset_sigma:.6g} is within "
            f"{parameters.offset_grid_not_needed_noise_multiple:.3g}x its tile noise "
            f"{tile_offset_noise:.6g} or below "
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
        "tileOffsetNoise": tile_offset_noise,
        "referenceSky": reference_sky,
        "skyResponseRemovedBeforeFit": bool(correct_sky),
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


def _with_sky_response(
    grid: tuple[tuple[float, ...], ...],
    scalar_offset: float,
    scale: float,
    sky: float,
    sky_response: _SkyResponse,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    """Fold a frame's sky-proportional term into its additive grid.

    Integration evaluates ``scale * pixel + grid``; removing ``sky * response``
    from the frame before scaling therefore subtracts ``scale * sky * response``
    from the grid.  A scalar offset becomes a grid of that constant.
    """

    assert sky_response.grid is not None
    base = (
        np.asarray(grid, dtype=np.float64)
        if grid
        else np.full(sky_response.grid.shape, scalar_offset, dtype=np.float64)
    )
    if base.shape != sky_response.grid.shape:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_UNDERCONSTRAINED",
            "additive grid and sky-response grid geometries differ",
        )
    combined = base - scale * sky * sky_response.grid
    return (
        tuple(tuple(float(value) for value in row) for row in combined),
        tuple(float(value) for value in sky_response.x_nodes),
        tuple(float(value) for value in sky_response.y_nodes),
    )


def _with_low_order_target(
    coefficient: GlobalNormalizationCoefficient,
    low_order: _LowOrderTarget,
    x_nodes: NDArray[np.float64],
    y_nodes: NDArray[np.float64],
) -> GlobalNormalizationCoefficient:
    """Add the group's common low-order correction to one frame's grid."""

    if low_order.correction is None or not np.any(low_order.correction):
        return coefficient
    base = (
        np.asarray(coefficient.offset_grid, dtype=np.float64)
        if coefficient.offset_grid
        else np.full(low_order.correction.shape, coefficient.offset, dtype=np.float64)
    )
    if base.shape != low_order.correction.shape:
        raise CalibrationError(
            "GLOBAL_NORMALIZATION_OFFSET_GRID_UNDERCONSTRAINED",
            "additive grid and low-order target geometries differ",
        )
    combined = base + low_order.correction
    evidence = dict(coefficient.evidence)
    evidence["lowOrderTarget"] = {"applied": True, "chosen": low_order.evidence.get("chosen")}
    return GlobalNormalizationCoefficient(
        source_path=coefficient.source_path,
        scale=coefficient.scale,
        offset=0.0,
        mode=coefficient.mode + "+LOW_ORDER_TARGET",
        evidence=evidence,
        offset_grid=tuple(tuple(float(value) for value in row) for row in combined),
        offset_grid_x=tuple(float(value) for value in x_nodes),
        offset_grid_y=tuple(float(value) for value in y_nodes),
    )


def _fit_coefficient(
    target: FitsFrame,
    reference: FitsFrame,
    parameters: GlobalNormalizationParameters,
    scale_hint: StellarScaleHint | None,
    reference_cache: _ReferenceSampleCache,
    native_threads: int | None = None,
    sky_response: _SkyResponse | None = None,
    target_sky: float = 0.0,
    reference_sky: float = 0.0,
    reference_response: _SkyResponse | None = None,
) -> GlobalNormalizationCoefficient:
    x, y, sample_x, sample_y, paired_before_selection = _paired_samples(
        target,
        reference,
        parameters,
        reference_cache,
        sky_response,
        target_sky,
        reference_sky,
        reference_response,
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
            sky_response,
            target_sky,
            reference_sky,
            reference_response,
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
    if sky_response is not None and sky_response.applied:
        offset_grid, offset_grid_x, offset_grid_y = _with_sky_response(
            offset_grid, offset, scale, target_sky, sky_response
        )
        offset = 0.0
        mode = f"{mode}+SKY_RESPONSE"
        common_evidence["skyResponse"] = {
            "applied": True,
            "targetSky": target_sky,
            "referenceSky": reference_sky,
            "subtractedFromFrame": "targetSky * response(x, y) before the stellar scale",
        }
    else:
        common_evidence["skyResponse"] = {"applied": False}
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
    transforms: Sequence[Any] | None = None,
) -> GlobalNormalizationResult:
    """Fit every target against the reference; ``workers`` fits targets
    concurrently and cannot change any coefficient.  ``transforms`` are the
    frames' registration matrices (sensor -> registered pixels); they let
    the sky-proportional response be regressed in the sensor frame."""

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
        fit_workers = max(1, min(workers, len(frames)))
        native_threads = max(1, workers // fit_workers)
        group_levels = _group_tile_levels(frames, parameters, workers)
        sky_response = _fit_sky_response(
            frames, parameters, workers, precomputed=group_levels, transforms=transforms
        )
        skies = sky_response.skies if sky_response.applied else (0.0,) * len(frames)
        reference_response = sky_response.for_frame(reference_index)
        low_order = _fit_low_order_target(
            group_levels[0],
            group_levels[1],
            [
                float(hint.scale)
                if hint is not None and hint.scale is not None and math.isfinite(hint.scale)
                else 1.0
                for hint in hints
            ],
            reference_index,
            parameters.sky_response_minimum_frames,
        )

        def fit(index: int) -> GlobalNormalizationCoefficient:
            frame = frames[index]
            if index == reference_index:
                evidence: dict[str, Any] = {
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
                    "skyResponse": {"applied": False},
                }
                grid: tuple[tuple[float, ...], ...] = ()
                grid_x: tuple[float, ...] = ()
                grid_y: tuple[float, ...] = ()
                mode = "REFERENCE_IDENTITY"
                if sky_response.applied:
                    grid, grid_x, grid_y = _with_sky_response(
                        (), 0.0, 1.0, skies[index], reference_response
                    )
                    mode = "REFERENCE_IDENTITY+SKY_RESPONSE"
                    evidence["skyResponse"] = {
                        "applied": True,
                        "targetSky": skies[index],
                        "referenceSky": skies[index],
                        "subtractedFromFrame": "referenceSky * response(x, y)",
                    }
                coefficient = GlobalNormalizationCoefficient(
                    str(frame.path), 1.0, 0.0, mode, evidence, grid, grid_x, grid_y
                )
            else:
                assert reference_cache is not None
                coefficient = _fit_coefficient(
                    frame,
                    reference,
                    parameters,
                    hints[index],
                    reference_cache,
                    native_threads,
                    sky_response.for_frame(index),
                    skies[index],
                    skies[reference_index],
                    reference_response,
                )
            return _with_low_order_target(coefficient, low_order, group_levels[2], group_levels[3])

        if fit_workers == 1:
            coefficients = [fit(index) for index in range(len(frames))]
        else:
            with ThreadPoolExecutor(
                max_workers=fit_workers, thread_name_prefix="ufwbpp-normalize"
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
        "skyResponse": sky_response.evidence,
        "lowOrderTarget": low_order.evidence,
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
            "skyProportionalResponseRemovedPerFrame": sky_response.applied,
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
