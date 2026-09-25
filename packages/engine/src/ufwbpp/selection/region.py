"""Per-frame region weight maps from the Light Frame QC spatial grid.

Light Frame QC scores every frame on a grid (16x16 by default) laid over the
group reference's preview (``FrameResult.grid``): per-cell star completeness
against the reference, the median transparency residual of the matched stars
(positive = dimmer than the frame's global transparency), the connected
missing-star component the occlusion gate judged, and the background and
texture differences against the reference.  A region weight map turns that
evidence into a multiplicative weight per cell::

    m = clip(1 - max(0, residual - dead band) / full loss, 0, 1)
        * [cell is not missing its stars]
        * [background delta within 3 sigma, or texture ratio above 0.5]

where the residual is the consensus dimming residual (this frame against the
cohort's clear envelope) and only patches of at least three dimmed cells
count, so faint-star outliers and shared flat-field structure never dim a
clean frame.

Missing-star components of at least two cells are dilated by one cell in
every direction (the margin for the occluder's soft edge); the zero regions
then fade in over about one cell through a Gaussian ramp while the dimming
term keeps its cell values, and the zeros stay at exactly zero.  Cells without evidence keep weight 1.  Registered frames share
the reference's pixel grid, so the nodes are the cell centres in normalized
coordinates; ``pixel_nodes`` turns them into pixel coordinates once the frame
shape is known.  Integration multiplies the frame weight by the bilinear
interpolation of the map at every sample; the rejection statistics are
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..stacking.integration import evaluate_weight_grid_rows

REGION_WEIGHT_ALGORITHM = "region-weights-grid-v1"

# Cell-median transparency residuals of clean frames scatter by about 0.035
# mag (MADN) with single-cell excursions to 0.3 mag from faint stars; the
# dead band and the coherence requirement below keep clean maps at exactly 1.
DIMMING_DEAD_BAND_MAG = 0.08
# The weight reaches zero this many magnitudes beyond the dead band.
DIMMING_FULL_LOSS_MAG = 0.45
# A dimmed area must span this many 8-connected cells whose median residual
# clears the dead band by ``DIMMING_PATCH_MARGIN_MAG``; a cloud patch is
# contiguous and deep, a cluster of faint-star outliers is neither.
DIMMING_MINIMUM_CELLS = 4
DIMMING_PATCH_MARGIN_MAG = 0.04
# A map without zero cells whose mean is above this is cosmetic: it cannot
# change the master measurably and only costs the per-sample reduction.
COSMETIC_MEAN_WEIGHT = 0.99
BACKGROUND_DELTA_SIGMA = 3.0
TEXTURE_RATIO_FLOOR = 0.50
SMOOTHING_SIGMA_CELLS = 1.0
# Maps whose smallest weight is above this are not worth a per-sample pass.
NEGLIGIBLE_FLOOR = 0.98


@dataclass(frozen=True, slots=True)
class RegionWeightMap:
    """A per-frame multiplicative weight grid in normalized reference coordinates."""

    path: str
    rows: int
    columns: int
    nodes: tuple[tuple[float, ...], ...]
    zero_cells: int
    minimum_weight: float
    mean_weight: float
    evidence: Mapping[str, Any]

    @property
    def zero_fraction(self) -> float:
        return self.zero_cells / float(self.rows * self.columns)

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(self.nodes, dtype=np.float64)

    def pixel_nodes(self, height: int, width: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Node coordinates (x, y) in pixels of a registered frame of this shape."""

        x = tuple((column + 0.5) / self.columns * float(width) for column in range(self.columns))
        y = tuple((row + 0.5) / self.rows * float(height) for row in range(self.rows))
        return x, y

    def transformed(
        self, matrix: Sequence[Sequence[float]], height: int, width: int
    ) -> RegionWeightMap:
        """Resample the map into another pixel frame of the same dimensions.

        ``matrix`` (3x3, affine or projective) maps a pixel of the target
        frame to the pixel of this map's frame.  Registered Lights live in the
        pixel pipeline's reference frame while the QC grid lives in the QC
        reference's frame; the two can differ by dithers or a meridian flip.
        Each target cell centre is mapped and the map is read there
        bilinearly; positions outside this map's frame carry no evidence and
        get weight 1.
        """

        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
            raise ValueError("region map transform must be a finite 3x3 matrix")
        source = self.as_array()
        rows, columns = source.shape
        centre_x = (np.arange(columns) + 0.5) / columns * float(width)
        centre_y = (np.arange(rows) + 0.5) / rows * float(height)
        grid_x, grid_y = np.meshgrid(centre_x, centre_y)
        homogeneous = np.stack((grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)))
        mapped = transform @ homogeneous
        with np.errstate(divide="ignore", invalid="ignore"):
            mapped_x = mapped[0] / mapped[2]
            mapped_y = mapped[1] / mapped[2]
        inside = (
            np.isfinite(mapped_x)
            & np.isfinite(mapped_y)
            & (mapped_x >= 0.0)
            & (mapped_x < float(width))
            & (mapped_y >= 0.0)
            & (mapped_y < float(height))
        )
        # Fractional cell coordinates of the mapped centres in this map's grid.
        fx = np.clip(mapped_x / float(width) * columns - 0.5, 0.0, columns - 1.0)
        fy = np.clip(mapped_y / float(height) * rows - 0.5, 0.0, rows - 1.0)
        x0 = np.clip(np.floor(fx).astype(int), 0, columns - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, rows - 2)
        wx = fx - x0
        wy = fy - y0
        values = (
            source[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + source[y0, x0 + 1] * wx * (1.0 - wy)
            + source[y0 + 1, x0] * (1.0 - wx) * wy
            + source[y0 + 1, x0 + 1] * wx * wy
        )
        # A blanked cell stays blanked: the nearest source cell decides, so
        # the margin around a blocked area keeps its width after a sub-cell
        # registration shift instead of fading through interpolation.
        nearest = source[
            np.clip(np.rint(fy).astype(int), 0, rows - 1),
            np.clip(np.rint(fx).astype(int), 0, columns - 1),
        ]
        values = np.where(nearest <= 0.0, 0.0, values)
        values = np.where(inside, values, 1.0).reshape(rows, columns)
        values = np.clip(values, 0.0, 1.0)
        return RegionWeightMap(
            path=self.path,
            rows=rows,
            columns=columns,
            nodes=tuple(tuple(float(value) for value in row) for row in values),
            zero_cells=int(np.count_nonzero(values <= 0.0)),
            minimum_weight=float(values.min()),
            mean_weight=float(values.mean()),
            evidence={**dict(self.evidence), "frame": "registered"},
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "algorithm": REGION_WEIGHT_ALGORITHM,
            "frame": str(self.evidence.get("frame", "qc-reference")),
            "rows": self.rows,
            "columns": self.columns,
            "zeroCells": self.zero_cells,
            "zeroFraction": self.zero_fraction,
            "minimumWeight": self.minimum_weight,
            "meanWeight": self.mean_weight,
            "evidence": dict(self.evidence),
            "nodes": [[round(float(value), 4) for value in row] for row in self.nodes],
        }


def _float_grid(values: Any, rows: int, columns: int) -> NDArray[np.float64]:
    grid = np.full((rows, columns), np.nan, dtype=np.float64)
    if not isinstance(values, Sequence) or len(values) != rows:
        return grid
    for row_index, row in enumerate(values):
        if not isinstance(row, Sequence) or len(row) != columns:
            return np.full((rows, columns), np.nan, dtype=np.float64)
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if np.isfinite(number):
                grid[row_index, column_index] = number
    return grid


_FOUR_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT_NEIGHBOURS = _FOUR_NEIGHBOURS + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(
    mask: NDArray[np.bool_], *, diagonal: bool = False
) -> list[list[tuple[int, int]]]:
    rows, columns = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    neighbours = _EIGHT_NEIGHBOURS if diagonal else _FOUR_NEIGHBOURS
    for start_row in range(rows):
        for start_column in range(columns):
            if not mask[start_row, start_column] or seen[start_row, start_column]:
                continue
            seen[start_row, start_column] = True
            queue = [(start_row, start_column)]
            component: list[tuple[int, int]] = []
            while queue:
                row, column = queue.pop()
                component.append((row, column))
                for row_step, column_step in neighbours:
                    next_row, next_column = row + row_step, column + column_step
                    if (
                        0 <= next_row < rows
                        and 0 <= next_column < columns
                        and mask[next_row, next_column]
                        and not seen[next_row, next_column]
                    ):
                        seen[next_row, next_column] = True
                        queue.append((next_row, next_column))
            components.append(component)
    return components


def _dilate(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """One-cell dilation over the eight neighbours (a full margin, corners included)."""

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = mask.copy()
    rows, columns = mask.shape
    for row_shift in (0, 1, 2):
        for column_shift in (0, 1, 2):
            result |= padded[row_shift : row_shift + rows, column_shift : column_shift + columns]
    return result


def _smooth(values: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Gaussian smoothing with edge extension; the kernel is renormalized."""

    radius = max(1, int(np.ceil(2.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    rows, columns = values.shape
    horizontal = np.zeros((rows + 2 * radius, columns), dtype=np.float64)
    for index, weight in enumerate(kernel):
        horizontal += weight * padded[:, index : index + columns]
    result = np.zeros((rows, columns), dtype=np.float64)
    for index, weight in enumerate(kernel):
        result += weight * horizontal[index : index + rows, :]
    return result


def region_weight_map(grid: Mapping[str, Any] | None, path: str) -> RegionWeightMap | None:
    """Build the region weight map of one frame from its QC grid, or None.

    None means the frame needs no map: the grid is absent or too small, or
    every cell's weight is above ``NEGLIGIBLE_FLOOR``.
    """

    if not isinstance(grid, Mapping):
        return None
    try:
        rows = int(grid.get("rows", 0))
        columns = int(grid.get("columns", 0))
    except (TypeError, ValueError):
        return None
    if rows < 4 or columns < 4:
        return None
    expected = _float_grid(grid.get("expectedStars"), rows, columns)
    matched = _float_grid(grid.get("matchedStars"), rows, columns)
    background = _float_grid(grid.get("backgroundDeltaRobustSigma"), rows, columns)
    texture = _float_grid(grid.get("textureRatio"), rows, columns)
    # The consensus residual (frame minus the cohort's clear envelope, median
    # centred) separates this frame's cloud from patterns every frame shares
    # (flat-field and reference-frame structure); the raw residual against
    # the reference is the fallback for cohorts too small for a consensus.
    residual = _float_grid(grid.get("consensusDimmingResidualMag"), rows, columns)
    residual_source = "consensus"
    if not np.any(np.isfinite(residual)):
        residual = _float_grid(grid.get("transparencyResidualMag"), rows, columns)
        residual_source = "reference"

    weights = np.ones((rows, columns), dtype=np.float64)

    # Spatial dimming: cells beyond the dead band that form a deep patch.
    candidate = np.isfinite(residual) & (residual > DIMMING_DEAD_BAND_MAG)
    dimmed = np.zeros((rows, columns), dtype=bool)
    for component in _components(candidate, diagonal=True):
        if len(component) < DIMMING_MINIMUM_CELLS:
            continue
        values = np.asarray([residual[row, column] for row, column in component])
        if float(np.median(values)) < DIMMING_DEAD_BAND_MAG + DIMMING_PATCH_MARGIN_MAG:
            continue
        for row, column in component:
            dimmed[row, column] = True
    # The ramp removes cloud structure (zero beyond about half a magnitude);
    # the transparency factor 10^(-0.4 r) tracks the lower signal-to-noise
    # of an attenuated, unrescaled sample.
    ramp = np.clip(
        1.0 - (residual - DIMMING_DEAD_BAND_MAG) / DIMMING_FULL_LOSS_MAG, 0.0, 1.0
    )
    with np.errstate(invalid="ignore"):
        transmission = np.power(10.0, -0.4 * np.where(np.isfinite(residual), residual, 0.0))
    loss = ramp * transmission
    weights[dimmed] = loss[dimmed]

    # Missing stars, judged as the occlusion gate judges them: cells with at
    # least three expected stars whose completeness falls below the larger
    # of 0.20 and 35% of the frame's completeness.
    supported = np.isfinite(expected) & (expected >= 3) & np.isfinite(matched)
    expected_total = float(np.nansum(np.where(supported, expected, 0.0)))
    matched_total = float(np.nansum(np.where(supported, matched, 0.0)))
    global_completeness = matched_total / expected_total if expected_total > 0 else 0.0
    completeness = np.divide(
        matched, expected, out=np.full((rows, columns), np.nan), where=supported & (expected > 0)
    )
    missing_candidate = supported & np.isfinite(completeness) & (
        completeness <= max(0.20, global_completeness * 0.35)
    )
    # A lone missing cell is a sparse or faint-star cell, not an occluder;
    # blocked areas span cells.  Components count, and each gets its margin.
    missing = np.zeros((rows, columns), dtype=bool)
    zeros = np.zeros((rows, columns), dtype=bool)
    for component in _components(missing_candidate, diagonal=True):
        if len(component) < 2:
            continue
        component_mask = np.zeros((rows, columns), dtype=bool)
        for row, column in component:
            component_mask[row, column] = True
        missing |= component_mask
        zeros |= _dilate(component_mask)

    # A background anomaly whose texture is also gone (an opaque blocker or a
    # bright cloud bank) rather than a smooth gradient.
    anomalous = (
        np.isfinite(background)
        & (np.abs(background) >= BACKGROUND_DELTA_SIGMA)
        & np.isfinite(texture)
        & (texture <= TEXTURE_RATIO_FLOOR)
    )
    zeros |= anomalous

    # The dimming term is kept cell by cell (its noise is handled by the dead
    # band); only the zero regions get a soft edge, so a blocked area fades
    # in over about one cell beyond its margin.
    ramp = 1.0 - _smooth(zeros.astype(np.float64), SMOOTHING_SIGMA_CELLS)
    smoothed = np.clip(weights * ramp, 0.0, 1.0)
    smoothed[zeros] = 0.0
    # Cells the evidence never touched stay at exactly 1 unless a zero
    # region's ramp reaches them.
    untouched = ~(dimmed | zeros)
    smoothed[untouched & (smoothed > NEGLIGIBLE_FLOOR)] = 1.0
    if float(smoothed.min()) > NEGLIGIBLE_FLOOR:
        return None
    if not np.any(smoothed <= 0.0) and float(smoothed.mean()) > COSMETIC_MEAN_WEIGHT:
        return None
    return RegionWeightMap(
        path=str(path),
        rows=rows,
        columns=columns,
        nodes=tuple(tuple(float(value) for value in row) for row in smoothed),
        zero_cells=int(np.count_nonzero(smoothed <= 0.0)),
        minimum_weight=float(smoothed.min()),
        mean_weight=float(smoothed.mean()),
        evidence={
            "dimmedCells": int(np.count_nonzero(dimmed)),
            "dimmedCandidateCells": int(np.count_nonzero(candidate)),
            "missingCells": int(np.count_nonzero(missing)),
            "backgroundCells": int(np.count_nonzero(anomalous)),
            "zeroCellsWithMargin": int(np.count_nonzero(zeros)),
            "residualSource": residual_source,
            "frame": "qc-reference",
        },
    )


def _resolved(path: str) -> str:
    try:
        return str(Path(path).resolve(strict=True))
    except OSError:
        return str(Path(path))


def region_weight_maps(
    results: Iterable[Any], paths: Iterable[str] | None = None
) -> dict[str, RegionWeightMap]:
    """Maps for the QC results whose (resolved) path is in ``paths`` (all when None)."""

    wanted = None if paths is None else {_resolved(str(path)) for path in paths}
    maps: dict[str, RegionWeightMap] = {}
    for result in results:
        resolved = _resolved(str(result.path))
        if wanted is not None and resolved not in wanted:
            continue
        built = region_weight_map(getattr(result, "grid", None), resolved)
        if built is not None:
            maps[resolved] = built
    return maps


__all__ = [
    "REGION_WEIGHT_ALGORITHM",
    "RegionWeightMap",
    "evaluate_weight_grid_rows",
    "region_weight_map",
    "region_weight_maps",
]
