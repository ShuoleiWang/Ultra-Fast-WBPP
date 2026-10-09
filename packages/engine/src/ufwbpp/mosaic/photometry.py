"""Photometric and background matching of mosaic panels on one canvas.

Every panel master already lies on the canvas lattice, so an overlap is the
same pixels seen by two panels.  Two quantities tie the panels together:

* the **scale** of each panel, from aperture photometry of the stars both
  panels measured in their overlap (an aperture of three times the larger
  FWHM makes the ratio insensitive to the panels' different seeing), solved
  for all overlaps at once by robust weighted least squares instead of being
  propagated panel by panel, so the error of one overlap never accumulates
  along a chain and closing loops are a check;
* an additive **plane** per panel, from the binned differences of the
  scaled panels in the overlaps.  Only differences are observable: a common
  plane added to every panel changes no overlap, so the solution keeps the
  area-weighted consensus sky (the minimum-norm solution) and removes only
  what the panels disagree on.  Real large-scale sky, a galaxy halo or IFN,
  is identical in the panels and therefore never removed.  Bins where the
  uncertainty of the scale times the local brightness could reach a tenth
  of the noise are left out, so a bright galaxy's residual can never be
  absorbed into a panel's plane; those bins then give an independent
  "extended-structure" scale that is compared with the stars'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray


PHOTOMETRY_ALGORITHM = "mosaic-star-photometry-network-v1"
BACKGROUND_ALGORITHM = "mosaic-plane-network-min-norm-v1"
BIN_PIXELS = 32
MINIMUM_OVERLAP_STARS = 8
MINIMUM_STAR_SNR = 30.0
MINIMUM_OVERLAP_PIXELS = 64 * 64
MINIMUM_COVERAGE_FRACTION = 0.5
BOOTSTRAP_SAMPLES = 200
HUBER_THRESHOLD = 1.5
# A bin is used for the background only where the scale's uncertainty times
# the bin's structure stays below this fraction of the bin noise.
SCALE_LEAK_FRACTION = 0.1
# Aperture radius in units of each panel's own FWHM.
APERTURE_FWHM = 2.5


@dataclass(frozen=True)
class PanelImage:
    """One panel master of one filter on its canvas window."""

    key: str
    path: Path
    origin: tuple[int, int]
    shape: tuple[int, int]
    exposure_seconds: float | None = None
    coverage_path: Path | None = None
    count_path: Path | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) on the canvas, exclusive ends."""

        return self.origin[0], self.origin[1], self.origin[0] + self.shape[1], self.origin[1] + self.shape[0]

    def _read(
        self, path: Path, x0: int, y0: int, x1: int, y1: int, opened: Mapping[Path, Any] | None = None
    ) -> NDArray[np.float32]:
        bx0, by0, _, _ = self.box
        if opened is not None and path in opened:
            return np.array(opened[path][0].data[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0], dtype=np.float32)
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            return np.array(hdul[0].data[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0], dtype=np.float32)

    def read(
        self, x0: int, y0: int, x1: int, y1: int, *, opened: Mapping[Path, Any] | None = None
    ) -> NDArray[np.float32]:
        """Canvas box of the master; NaN where the panel has no valid sample.
        ``opened`` maps a plane's path to its open HDU list, for many small
        reads."""

        values = self._read(self.path, x0, y0, x1, y1, opened)
        if self.coverage_path is not None:
            coverage = self._read(self.coverage_path, x0, y0, x1, y1, opened)
            values[~(np.isfinite(coverage) & (coverage >= MINIMUM_COVERAGE_FRACTION))] = np.nan
        return values

    def counts(
        self, x0: int, y0: int, x1: int, y1: int, *, opened: Mapping[Path, Any] | None = None
    ) -> NDArray[np.float32] | None:
        if self.count_path is None:
            return None
        return self._read(self.count_path, x0, y0, x1, y1, opened)

    @property
    def planes(self) -> tuple[Path, ...]:
        """The files this panel reads."""

        return tuple(dict.fromkeys(path for path in (self.path, self.coverage_path, self.count_path) if path is not None))


def _intersection(left: PanelImage, right: PanelImage) -> tuple[int, int, int, int] | None:
    a, b = left.box, right.box
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[2] - box[0] >= BIN_PIXELS and box[3] - box[1] >= BIN_PIXELS else None


@dataclass
class EdgeMeasurement:
    left: str
    right: str
    box: tuple[int, int, int, int]
    valid_pixels: int
    # Per matched star: canvas position, flux in each panel, centroid offset.
    star_x: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    star_y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    flux_left: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    flux_right: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    offset_x: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    offset_y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    aperture_radius: float = 0.0
    # Each panel's seeing (pixels) from its stars' half-light radii; 0 when
    # no star was measured.
    fwhm_left: float = 0.0
    fwhm_right: float = 0.0
    # Per background bin: canvas centre, the two medians and the bin noise.
    bin_x: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    bin_y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    bin_left: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    bin_right: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    bin_noise: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    @property
    def width(self) -> int:
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])


def _flux_slope(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Slope of ``left = slope*right + offset`` by Huber-weighted least
    squares.  The offset absorbs an additive error common to the apertures
    (a background map raised by crowded star light differs between two
    panels of different seeing), which biases faint stars' ratios; the
    slope, carried by the bright stars, is the scale ratio."""

    design = np.column_stack((right, np.ones_like(right)))
    weights = np.ones_like(right)
    slope = float(np.median(left / right))
    for _ in range(6):
        solution, *_ = np.linalg.lstsq(design * weights[:, None], left * weights, rcond=None)
        slope = float(solution[0])
        residual = left - design @ solution
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) or 1.0
        normalized = np.abs(residual) / scale
        weights = np.where(normalized <= HUBER_THRESHOLD, 1.0, np.sqrt(HUBER_THRESHOLD / np.maximum(normalized, 1e-12)))
    return slope


def _log_flux_ratio(
    left: NDArray[np.float64], right: NDArray[np.float64], rng: np.random.Generator
) -> tuple[float, float]:
    """ln(scale ratio) of two panels' fluxes of the same stars and its
    bootstrap standard error."""

    slope = _flux_slope(left, right)
    if left.size < 3 or slope <= 0.0:
        return math.log(slope) if slope > 0.0 else math.nan, math.inf
    draws = rng.integers(0, left.size, size=(BOOTSTRAP_SAMPLES, left.size))
    slopes = np.asarray([_flux_slope(left[draw], right[draw]) for draw in draws])
    slopes = slopes[slopes > 0.0]
    if slopes.size < BOOTSTRAP_SAMPLES // 2:
        return math.log(slope), math.inf
    return math.log(slope), float(np.std(np.log(slopes), ddof=1))


def _aperture_photometry(
    work: NDArray[np.float32],
    residual: NDArray[np.float64],
    invalid: NDArray[np.bool_],
    star_mask: NDArray[np.bool_],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fwhm: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.bool_]]:
    """Fluxes of stars in one panel.

    The aperture is ``APERTURE_FWHM`` times this panel's own FWHM, so two
    panels of different seeing enclose the same fraction of a star of the
    same profile.  Each star's background is the median of its own annulus
    with every detection masked: on the steep flank of a bright galaxy a
    smooth background map leaves a residual that the aperture area turns
    into a percent-level flux error, an annulus median follows the local
    level (a linear gradient cancels).
    """

    import sep

    height, width = work.shape
    # Each star's local level (median of its annulus, other stars masked)
    # and, from it, its half-light radius on a growth curve.
    outer_probe = 4.0 * fwhm
    probe_inner, probe_outer = outer_probe + 2.0, outer_probe + 2.0 + max(6.0, outer_probe)
    reach = int(math.ceil(max(probe_outer, 3.0 * APERTURE_FWHM * fwhm)))
    levels = np.full(x.shape, np.nan, dtype=np.float64)
    half_light = np.full(x.shape, np.nan, dtype=np.float64)
    for index, (sx, sy) in enumerate(zip(x, y)):
        ix, iy = int(round(sx)), int(round(sy))
        sl_y = slice(max(0, iy - reach), min(height, iy + reach + 1))
        sl_x = slice(max(0, ix - reach), min(width, ix + reach + 1))
        dy = np.arange(sl_y.start, sl_y.stop)[:, None] - sy
        dx = np.arange(sl_x.start, sl_x.stop)[None, :] - sx
        distance = np.sqrt(dx * dx + dy * dy)
        cut_invalid = invalid[sl_y, sl_x]
        own = distance <= outer_probe
        others = star_mask[sl_y, sl_x] & ~own
        ring = (distance >= probe_inner) & (distance <= probe_outer) & ~others & ~cut_invalid
        if ring.sum() < 12 or cut_invalid[own].any():
            continue
        level = float(np.median(work[sl_y, sl_x][ring]))
        levels[index] = level
        inside = own & ~others
        order = np.argsort(distance[inside])
        cumulative = np.cumsum((work[sl_y, sl_x][inside] - level)[order])
        if cumulative.size == 0 or cumulative[-1] <= 0:
            continue
        half_light[index] = float(distance[inside][order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])
    finite = np.isfinite(half_light)
    median_half_light = float(np.median(half_light[finite])) if finite.sum() >= 3 else fwhm / 2.0
    median_half_light = float(np.clip(median_half_light, 0.3 * fwhm, 1.5 * fwhm))
    # A Gaussian's half-light radius is half its FWHM.
    radius = float(np.clip(APERTURE_FWHM * 2.0 * median_half_light, 3.0, 40.0))
    sums, _, flags = sep.sum_circle(np.ascontiguousarray(work, dtype=np.float64), x, y, radius, mask=invalid, subpix=5)
    flux = np.asarray(sums, dtype=np.float64) - levels * (math.pi * radius * radius)
    good = (np.asarray(flags) == 0) & np.isfinite(flux) & (flux > 0)
    return flux, radius, good


def measure_edge(left: PanelImage, right: PanelImage) -> EdgeMeasurement | None:
    """Stars and background bins of one overlap."""

    import sep
    from ufwbpp_registration import DetectionConfig, detect_stars

    box = _intersection(left, right)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    a = left.read(x0, y0, x1, y1)
    b = right.read(x0, y0, x1, y1)
    valid = np.isfinite(a) & np.isfinite(b)
    edge = EdgeMeasurement(left.key, right.key, box, int(valid.sum()))
    if edge.valid_pixels < MINIMUM_OVERLAP_PIXELS:
        return edge
    mean = np.where(valid, 0.5 * (a + b), np.nan).astype(np.float32)
    try:
        catalog = detect_stars(
            mean,
            DetectionConfig(detection_sigma=5.0, maximum_stars=1500, background_box=64, preview_long_edge=256),
        )
    except ValueError:
        catalog = None
    invalid = np.ascontiguousarray(~valid)
    work_a = np.ascontiguousarray(np.where(valid, a, np.nanmedian(a[valid])), dtype=np.float32)
    work_b = np.ascontiguousarray(np.where(valid, b, np.nanmedian(b[valid])), dtype=np.float32)
    # A local background map (SEP's clipped mesh) instead of an annulus: an
    # annulus in a crowded field or on a galaxy holds other stars and biases
    # every flux low.
    background_a = sep.Background(work_a, mask=invalid, bw=64, bh=64, fw=3, fh=3)
    background_b = sep.Background(work_b, mask=invalid, bw=64, bh=64, fw=3, fh=3)
    residual_a = np.ascontiguousarray(work_a - background_a.back(), dtype=np.float64)
    residual_b = np.ascontiguousarray(work_b - background_b.back(), dtype=np.float64)
    star_mask = np.zeros(valid.shape, dtype=bool)
    if catalog is not None and catalog.count:
        fwhm = float(np.median(catalog.fwhm)) if catalog.fwhm.size else 3.0
        px, py = catalog.points[:, 0], catalog.points[:, 1]
        # Every detection is masked out of the local backgrounds (and of the
        # background bins below), at the size of the largest aperture.
        mask_radius = float(np.clip(APERTURE_FWHM * 1.5 * fwhm, 4.0, 60.0))
        yy, xx = np.ogrid[: valid.shape[0], : valid.shape[1]]
        for sx, sy in zip(px, py):
            ix, iy = int(round(sx)), int(round(sy))
            r = int(math.ceil(mask_radius))
            sl_y = slice(max(0, iy - r), min(valid.shape[0], iy + r + 1))
            sl_x = slice(max(0, ix - r), min(valid.shape[1], ix + r + 1))
            star_mask[sl_y, sl_x] |= (xx[:, sl_x] - sx) ** 2 + (yy[sl_y, :] - sy) ** 2 <= mask_radius**2
        flux_a, radius_a, good_a = _aperture_photometry(work_a, residual_a, invalid, star_mask, px, py, fwhm)
        flux_b, radius_b, good_b = _aperture_photometry(work_b, residual_b, invalid, star_mask, px, py, fwhm)
        sigma = max(fwhm / 2.3548, 0.8)
        cx_a, cy_a, wflag_a = sep.winpos(residual_a, px, py, sigma, mask=invalid)
        cx_b, cy_b, wflag_b = sep.winpos(residual_b, px, py, sigma, mask=invalid)
        keep = (
            good_a
            & good_b
            & (np.asarray(wflag_a) == 0)
            & (np.asarray(wflag_b) == 0)
            & np.isfinite(cx_a)
            & np.isfinite(cx_b)
        )
        keep &= flux_a > MINIMUM_STAR_SNR * float(background_a.globalrms) * math.sqrt(math.pi) * radius_a
        keep &= flux_b > MINIMUM_STAR_SNR * float(background_b.globalrms) * math.sqrt(math.pi) * radius_b
        # Saturated or nonlinear cores: the brightest peaks of the overlap.
        if catalog.peak.size >= 50:
            keep &= catalog.peak < float(np.percentile(catalog.peak, 98.0))
        edge.star_x = px[keep] + x0
        edge.star_y = py[keep] + y0
        edge.flux_left = np.asarray(flux_a[keep], dtype=np.float64)
        edge.flux_right = np.asarray(flux_b[keep], dtype=np.float64)
        edge.offset_x = np.asarray(cx_a[keep] - cx_b[keep], dtype=np.float64)
        edge.offset_y = np.asarray(cy_a[keep] - cy_b[keep], dtype=np.float64)
        edge.aperture_radius = max(radius_a, radius_b)
        edge.fwhm_left = radius_a / APERTURE_FWHM
        edge.fwhm_right = radius_b / APERTURE_FWHM
    usable = valid & ~star_mask
    rows, columns = valid.shape[0] // BIN_PIXELS, valid.shape[1] // BIN_PIXELS
    if rows and columns:
        cut = (slice(0, rows * BIN_PIXELS), slice(0, columns * BIN_PIXELS))
        shape = (rows, BIN_PIXELS, columns, BIN_PIXELS)
        blocks_a = np.where(usable, a, np.nan)[cut].reshape(shape).transpose(0, 2, 1, 3).reshape(rows, columns, -1)
        blocks_b = np.where(usable, b, np.nan)[cut].reshape(shape).transpose(0, 2, 1, 3).reshape(rows, columns, -1)
        filled = np.isfinite(blocks_a).sum(axis=2)
        enough = filled >= 0.5 * BIN_PIXELS * BIN_PIXELS
        with np.errstate(all="ignore"):
            median_a = np.nanmedian(blocks_a, axis=2)
            median_b = np.nanmedian(blocks_b, axis=2)
            difference = blocks_a - blocks_b
            spread = 1.4826 * np.nanmedian(np.abs(difference - np.nanmedian(difference, axis=2)[..., None]), axis=2)
        noise = 1.2533 * spread / np.sqrt(np.maximum(filled, 1))
        good = enough & np.isfinite(median_a) & np.isfinite(median_b) & np.isfinite(noise) & (noise > 0)
        by, bx = np.nonzero(good)
        edge.bin_x = x0 + (bx + 0.5) * BIN_PIXELS
        edge.bin_y = y0 + (by + 0.5) * BIN_PIXELS
        edge.bin_left = median_a[good].astype(np.float64)
        edge.bin_right = median_b[good].astype(np.float64)
        edge.bin_noise = noise[good].astype(np.float64)
    return edge


@dataclass
class ScaleSolution:
    scales: dict[str, float]
    log_scale_errors: dict[str, float]
    reference: str
    edges: list[dict[str, Any]]
    chi2_per_dof: float | None
    covariance: NDArray[np.float64]
    order: list[str]

    def ratio_error(self, left: str, right: str) -> float:
        i, j = self.order.index(left), self.order.index(right)
        variance = self.covariance[i, i] + self.covariance[j, j] - 2.0 * self.covariance[i, j]
        return math.sqrt(max(variance, 0.0))


def _connected(keys: Sequence[str], pairs: Sequence[tuple[str, str]]) -> bool:
    if len(keys) <= 1:
        return True
    adjacency: dict[str, set[str]] = {key: set() for key in keys}
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {keys[0]}
    frontier = [keys[0]]
    while frontier:
        for neighbour in adjacency[frontier.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
    return len(seen) == len(keys)


def solve_scales(
    panels: Sequence[PanelImage],
    edges: Sequence[EdgeMeasurement],
    *,
    seed: int = 20261007,
) -> ScaleSolution:
    """One scale per panel from every overlap's star ratios (robust network)."""

    order = [panel.key for panel in panels]
    index = {key: position for position, key in enumerate(order)}
    rng = np.random.default_rng(seed)
    observations: list[tuple[int, int, float, float, EdgeMeasurement]] = []
    for edge in edges:
        if edge.flux_left.size < MINIMUM_OVERLAP_STARS:
            continue
        location, error = _log_flux_ratio(edge.flux_left, edge.flux_right, rng)
        if not (math.isfinite(location) and math.isfinite(error)):
            continue
        observations.append((index[edge.left], index[edge.right], location, max(error, 1e-5), edge))
    if not _connected(order, [(order[i], order[j]) for i, j, _, _, _ in observations]):
        raise MosaicPhotometryError(
            "MOSAIC_PHOTOMETRIC_GRAPH_DISCONNECTED",
            "the overlaps with enough common stars do not connect every panel",
        )
    count = len(order)
    if count == 1:
        return ScaleSolution({order[0]: 1.0}, {order[0]: 0.0}, order[0], [], None, np.zeros((1, 1)), order)
    # Unknowns: L = ln(scale).  Scaled panels agree: L_i - L_j = -ln(F_i/F_j).
    design = np.zeros((len(observations) + 1, count))
    target = np.zeros(len(observations) + 1)
    sigma = np.ones(len(observations) + 1)
    for row, (i, j, location, error, _) in enumerate(observations):
        design[row, i], design[row, j], target[row], sigma[row] = 1.0, -1.0, -location, error
    design[-1, :] = 1.0  # gauge: the mean of L is zero (shifted below)
    sigma[-1] = 1e-9
    weights = np.ones(len(observations) + 1)
    for _ in range(4):
        scaled = design * (weights / sigma)[:, None]
        solution, *_ = np.linalg.lstsq(scaled, target * weights / sigma, rcond=None)
        normalized = (target - design @ solution) / sigma
        normalized[-1] = 0.0
        weights = np.where(np.abs(normalized) <= HUBER_THRESHOLD, 1.0, np.sqrt(HUBER_THRESHOLD / np.maximum(np.abs(normalized), 1e-12)))
        weights[-1] = 1.0
    scaled = design * (weights / sigma)[:, None]
    covariance = np.linalg.pinv(scaled.T @ scaled)
    residual = (target - design @ solution) / sigma
    dof = len(observations) - (count - 1)
    chi2 = float(np.sum(np.square(residual[:-1]))) / dof if dof > 0 else None
    # Gauge: the panel with the most flux per second (the most transparent
    # sky) keeps its scale; the others are brought up to it.
    exposure = np.asarray([panel.exposure_seconds or 1.0 for panel in panels], dtype=np.float64)
    reference_index = int(np.argmin(solution + np.log(exposure)))
    solution = solution - solution[reference_index]
    scales = {key: float(math.exp(solution[position])) for key, position in index.items()}
    errors = {key: float(math.sqrt(max(covariance[position, position], 0.0))) for key, position in index.items()}
    edge_records = []
    for row, (i, j, location, error, edge) in enumerate(observations):
        post, post_error = _log_flux_ratio(
            edge.flux_left * math.exp(solution[i]), edge.flux_right * math.exp(solution[j]), rng
        )
        offsets = np.hypot(edge.offset_x, edge.offset_y)
        edge_records.append(
            {
                "left": order[i],
                "right": order[j],
                "stars": int(edge.flux_left.size),
                "apertureRadiusPixels": edge.aperture_radius,
                "logRatio": location,
                "logRatioError": error,
                "normalizedResidual": float(residual[row]),
                "weight": float(weights[row]),
                "correctedRatioMinusOne": math.expm1(post),
                "correctedRatioError": post_error,
                "astrometricOffsetRmsPixels": float(np.sqrt(np.mean(np.square(offsets)))),
                "astrometricOffsetMedian": [float(np.median(edge.offset_x)), float(np.median(edge.offset_y))],
            }
        )
    return ScaleSolution(scales, errors, order[reference_index], edge_records, chi2, covariance, order)


class MosaicPhotometryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass
class PlaneSolution:
    """b_k(x, y) = c0 + c1*u + c2*v, u and v the canvas offsets from its
    centre in units of ``length``: subtracted from each scaled panel."""

    coefficients: dict[str, tuple[float, float, float]]
    centre: tuple[float, float]
    length: float
    diagnostics: dict[str, Any]

    def evaluate(self, key: str, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        c0, c1, c2 = self.coefficients[key]
        return c0 + c1 * (x - self.centre[0]) / self.length + c2 * (y - self.centre[1]) / self.length


def solve_planes(
    panels: Sequence[PanelImage],
    edges: Sequence[EdgeMeasurement],
    scales: ScaleSolution,
    canvas_shape: tuple[int, int],
) -> PlaneSolution:
    """The additive plane of every panel from the binned overlap differences
    of the scaled panels, keeping the area-weighted consensus sky."""

    order = [panel.key for panel in panels]
    index = {key: position for position, key in enumerate(order)}
    height, width = canvas_shape
    centre = ((width - 1) / 2.0, (height - 1) / 2.0)
    length = max(width, height) / 2.0
    rows: list[NDArray[np.float64]] = []
    targets: list[float] = []
    sigmas: list[float] = []
    edge_records = []
    galaxy_checks = []
    for edge in edges:
        if edge.bin_x.size == 0:
            continue
        i, j = index[edge.left], index[edge.right]
        ci, cj = scales.scales[edge.left], scales.scales[edge.right]
        left = ci * edge.bin_left
        right = cj * edge.bin_right
        difference = left - right
        noise = np.hypot(ci * edge.bin_noise, cj * edge.bin_noise)
        structure = 0.5 * (left + right)
        structure = structure - float(np.median(structure))
        relative_error = scales.ratio_error(edge.left, edge.right)
        background = np.abs(structure) * relative_error <= SCALE_LEAK_FRACTION * noise
        u = (edge.bin_x - centre[0]) / length
        v = (edge.bin_y - centre[1]) / length
        for position in np.flatnonzero(background):
            row = np.zeros(3 * len(order))
            row[3 * i : 3 * i + 3] = (1.0, u[position], v[position])
            row[3 * j : 3 * j + 3] = (-1.0, -u[position], -v[position])
            rows.append(row)
            targets.append(float(difference[position]))
            sigmas.append(float(noise[position]))
        edge_records.append(
            {
                "left": edge.left,
                "right": edge.right,
                "bins": int(edge.bin_x.size),
                "backgroundBins": int(background.sum()),
                "scaleLeakMaskedBins": int((~background).sum()),
            }
        )
        galaxy_checks.append((edge, i, j, difference, noise, structure, background))
    coefficients = {key: (0.0, 0.0, 0.0) for key in order}
    diagnostics: dict[str, Any] = {"algorithm": BACKGROUND_ALGORITHM, "edges": edge_records}
    if rows:
        design = np.vstack(rows)
        target = np.asarray(targets)
        sigma = np.asarray(sigmas)
        weights = np.ones(len(target))
        for _ in range(4):
            scaled = design * (weights / sigma)[:, None]
            solution, *_ = np.linalg.lstsq(scaled, target * weights / sigma, rcond=None)
            normalized = (target - design @ solution) / sigma
            weights = np.where(np.abs(normalized) <= HUBER_THRESHOLD, 1.0, np.sqrt(HUBER_THRESHOLD / np.maximum(np.abs(normalized), 1e-12)))
        planes = solution.reshape(len(order), 3)
        # Minimum norm: the common plane every panel could carry is the
        # area-weighted consensus, which stays in the data.
        areas = np.asarray([panel.shape[0] * panel.shape[1] for panel in panels], dtype=np.float64)
        planes = planes - (areas[:, None] * planes).sum(axis=0) / areas.sum()
        coefficients = {key: tuple(float(value) for value in planes[index[key]]) for key in order}
        residual = (target - design @ solution) / sigma
        diagnostics["chi2PerBin"] = float(np.mean(np.square(residual)))
        diagnostics["robustWeightsBelowOne"] = int(np.count_nonzero(weights < 1.0))
    solution = PlaneSolution(coefficients, centre, length, diagnostics)
    # The extended-structure scale check: with the planes fixed, the bright
    # bins' remaining difference regressed on their structure.
    checks = []
    for edge, i, j, difference, noise, structure, background in galaxy_checks:
        bright = ~background
        if bright.sum() < 8:
            continue
        model = solution.evaluate(edge.left, edge.bin_x[bright], edge.bin_y[bright]) - solution.evaluate(
            edge.right, edge.bin_x[bright], edge.bin_y[bright]
        )
        remaining = difference[bright] - model
        level = 0.5 * (scales.scales[edge.left] * edge.bin_left[bright] + scales.scales[edge.right] * edge.bin_right[bright])
        weight = 1.0 / np.square(noise[bright])
        delta = float(np.sum(weight * remaining * level) / np.sum(weight * level * level))
        delta_error = float(1.0 / math.sqrt(float(np.sum(weight * level * level))))
        star_error = scales.ratio_error(edge.left, edge.right)
        combined = math.hypot(delta_error, star_error)
        checks.append(
            {
                "left": edge.left,
                "right": edge.right,
                "brightBins": int(bright.sum()),
                "extendedScaleMinusOne": delta,
                "extendedScaleError": delta_error,
                "agreesWithStars": bool(abs(delta) <= 3.0 * combined or abs(delta) <= 0.003),
            }
        )
    diagnostics["extendedStructureScaleChecks"] = checks
    return solution


__all__ = [
    "BACKGROUND_ALGORITHM",
    "PHOTOMETRY_ALGORITHM",
    "EdgeMeasurement",
    "MosaicPhotometryError",
    "PanelImage",
    "PlaneSolution",
    "ScaleSolution",
    "measure_edge",
    "solve_planes",
    "solve_scales",
]
