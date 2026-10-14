"""Photometric and background matching of mosaic panels on one canvas.

Every panel master already lies on the canvas lattice, so an overlap is the
same pixels seen by two panels.  Two quantities tie the panels together:

* the **gain** of each panel, a scale times a linear gradient, from aperture
  photometry of the stars both panels measured in their overlap (each
  panel's aperture is 2.5 times its own FWHM, so the ratio is insensitive to
  the panels' different seeing), in bins along each overlap, solved for all
  overlaps at once by robust weighted least squares instead of being
  propagated panel by panel, so the error of one overlap never accumulates
  along a chain and closing loops are a check.  The gradient carries what a
  master inherits from frames whose transparency varied across the field
  (thin cloud, differential extinction at high airmass); the BIC decides
  for a whole component whether gradients are fitted at all.  The overlaps
  with enough common stars are the edges of the panels' photometric graph;
  each connected component is solved on its own, with its own gauge
  (:func:`photometric_components`);
* an additive **quadratic** per panel (six terms), from the binned
  differences of the scaled panels in the overlaps.  Only differences are
  observable: a common polynomial added to every panel changes no overlap,
  so the solution keeps the area-weighted consensus sky (the minimum-norm
  solution) and removes only what the panels disagree on.  Real large-scale
  sky, a galaxy halo or IFN, is identical in the panels and therefore never
  removed.  Bins where the uncertainty of the scale times the local
  brightness could reach a tenth of the noise are left out, so a bright
  galaxy's residual can never be absorbed into a panel's polynomial; those
  bins then give an independent "extended-structure" scale that is compared
  with the stars'.

Both networks are solved through their normal equations from sparse designs
(a bin touches two panels), so a 6x6 mosaic costs the memory of its bins,
not of a dense design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zlib

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags


PHOTOMETRY_ALGORITHM = "mosaic-star-photometry-gain-plane-network-v1"
BACKGROUND_ALGORITHM = "mosaic-quadratic-network-min-norm-v1"
BIN_PIXELS = 32
MINIMUM_OVERLAP_STARS = 8
MINIMUM_STAR_SNR = 30.0
# Overlap misregistration: median star offsets in at most OFFSET_BINS
# equal-count bins of at least OFFSET_BIN_STARS stars along the overlap.
OFFSET_BINS = 8
OFFSET_BIN_STARS = 8
# The gain network: flux-flux slopes in at most GAIN_BINS equal-count bins
# of at least GAIN_BIN_STARS stars along each overlap, and a prior on each
# panel's gain gradient (per half of its component's box) of
# GAIN_GRADIENT_PRIOR.
GAIN_BINS = 6
GAIN_BIN_STARS = 24
GAIN_GRADIENT_PRIOR = 0.05
MINIMUM_OVERLAP_PIXELS = 64 * 64
MINIMUM_COVERAGE_FRACTION = 0.5
BOOTSTRAP_SAMPLES = 200
HUBER_THRESHOLD = 1.5
# A bin is used for the background only where the scale's uncertainty times
# the bin's structure stays below this fraction of the bin noise.
SCALE_LEAK_FRACTION = 0.1
# Aperture radius in units of each panel's own FWHM.
APERTURE_FWHM = 2.5
# The additive model per panel: 1, u, v, u², uv, v².  The prior on a
# panel's three curvature terms: its bending across itself is within
# QUADRATIC_PRIOR_NOISE times the per-pixel noise of a panel difference
# (0.14 of one panel's) unless the overlaps show more (the strips along a
# panel's edges hardly measure its curvature across them; without the prior
# the minimum-norm solution carries noise into the panel).
PLANE_TERMS = 6
QUADRATIC_PRIOR_NOISE = 0.1
# Unknowns per panel of the gain network: ln scale and the two gradients.
_GAIN_TERMS = 3
# Singular values of a normal matrix below this fraction of its largest are
# the unobservable common terms (the gauge) and stay at zero.
_NORMAL_RCOND = 1e-10


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
    # Per background bin: canvas centre, the two medians and the bin noise
    # (that of the median of the pixel differences in the bin).
    bin_x: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    bin_y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    bin_left: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    bin_right: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    bin_noise: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    @property
    def width(self) -> int:
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])


def _huber_weights(normalized: NDArray[np.float64]) -> NDArray[np.float64]:
    size = np.abs(normalized)
    return np.where(size <= HUBER_THRESHOLD, 1.0, np.sqrt(HUBER_THRESHOLD / np.maximum(size, 1e-12)))


def _flux_slopes(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    """Slope of ``left = slope*right + offset`` by Huber-weighted least
    squares, for every row of two (samples, stars) arrays at once (one
    row is one bootstrap draw).  The offset absorbs an additive error
    common to the apertures (a background map raised by crowded star light
    differs between two panels of different seeing), which biases faint
    stars' ratios; the slope, carried by the bright stars, is the scale
    ratio."""

    left = np.atleast_2d(np.asarray(left, dtype=np.float64))
    right = np.atleast_2d(np.asarray(right, dtype=np.float64))
    weights = np.ones_like(right)
    slope = np.full(left.shape[0], np.nan)
    for _ in range(6):
        squared = weights * weights
        total = squared.sum(axis=1)
        mean_x = (squared * right).sum(axis=1) / total
        mean_y = (squared * left).sum(axis=1) / total
        dx = right - mean_x[:, None]
        sxx = (squared * dx * dx).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = np.where(sxx > 0.0, (squared * dx * (left - mean_y[:, None])).sum(axis=1) / sxx, np.nan)
        offset = mean_y - slope * mean_x
        residual = left - (slope[:, None] * right + offset[:, None])
        spread = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=1)[:, None]), axis=1)
        spread = np.where(spread > 0.0, spread, 1.0)
        weights = _huber_weights(residual / spread[:, None])
    return slope


def _log_flux_ratio(
    left: NDArray[np.float64], right: NDArray[np.float64], rng: np.random.Generator
) -> tuple[float, float]:
    """ln(scale ratio) of two panels' fluxes of the same stars and its
    bootstrap standard error."""

    slope = float(_flux_slopes(left, right)[0])
    if left.size < 3 or not slope > 0.0:
        return math.log(slope) if slope > 0.0 else math.nan, math.inf
    draws = rng.integers(0, left.size, size=(BOOTSTRAP_SAMPLES, left.size))
    slopes = _flux_slopes(left[draws], right[draws])
    slopes = slopes[np.isfinite(slopes) & (slopes > 0.0)]
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
class _Design:
    """A least-squares design assembled block by block as sparse triples:
    a row touches the unknowns of two panels, a dozen entries at most."""

    unknowns: int
    rows: list[NDArray[np.int64]] = field(default_factory=list)
    columns: list[NDArray[np.int64]] = field(default_factory=list)
    values: list[NDArray[np.float64]] = field(default_factory=list)
    targets: list[NDArray[np.float64]] = field(default_factory=list)
    sigmas: list[NDArray[np.float64]] = field(default_factory=list)
    count: int = 0

    def add(
        self,
        columns: NDArray[np.int64],
        values: NDArray[np.float64],
        targets: NDArray[np.float64],
        sigmas: NDArray[np.float64],
    ) -> None:
        """Rows with the given (rows, entries) columns and values."""

        rows = len(targets)
        if rows == 0:
            return
        self.rows.append(np.repeat(np.arange(self.count, self.count + rows), columns.shape[1]))
        self.columns.append(np.asarray(columns, dtype=np.int64).ravel())
        self.values.append(np.asarray(values, dtype=np.float64).ravel())
        self.targets.append(np.asarray(targets, dtype=np.float64))
        self.sigmas.append(np.asarray(sigmas, dtype=np.float64))
        self.count += rows

    def matrix(self) -> csr_matrix:
        return csr_matrix(
            (np.concatenate(self.values), (np.concatenate(self.rows), np.concatenate(self.columns))),
            shape=(self.count, self.unknowns),
        )


def _robust_solve(
    design: _Design, *, robust_rows: int, covariance: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Huber-weighted least squares through the normal equations of a
    sparse design; the rows from ``robust_rows`` on (priors) keep unit
    weight.  The minimum-norm solution leaves the unobservable common terms
    at zero.  Returns the solution, the row weights, the normalized
    residuals and, when asked, the covariance of the solution."""

    matrix = design.matrix()
    target = np.concatenate(design.targets)
    sigma = np.concatenate(design.sigmas)
    robust = np.arange(design.count) < robust_rows
    weights = np.ones(design.count)
    solution = np.zeros(design.unknowns)
    for _ in range(4):
        scaled = diags(weights / sigma) @ matrix
        normal = (scaled.T @ scaled).toarray()
        solution = np.linalg.lstsq(normal, scaled.T @ (target * weights / sigma), rcond=_NORMAL_RCOND)[0]
        normalized = (target - matrix @ solution) / sigma
        normalized[~robust] = 0.0
        weights = _huber_weights(normalized)
    residual = (target - matrix @ solution) / sigma
    if not covariance:
        return solution, weights, residual, None
    scaled = diags(weights / sigma) @ matrix
    return solution, weights, residual, np.linalg.pinv((scaled.T @ scaled).toarray(), rcond=_NORMAL_RCOND, hermitian=True)


@dataclass
class ScaleSolution:
    """Each panel's gain ``scales[key] * exp(a*u + b*v)``, with u and v the
    canvas offsets from ``centre`` (the centre of the panels' box) in units
    of ``length``: the reference panel's gain is 1 at ``centre``.
    ``log_scale_errors`` are the errors of each ln scale relative to the
    reference; ``covariance`` is over (ln scale, a, b) of every panel in
    ``order``."""

    scales: dict[str, float]
    log_scale_errors: dict[str, float]
    reference: str
    edges: list[dict[str, Any]]
    chi2_per_dof: float | None
    covariance: NDArray[np.float64]
    order: list[str]
    gradients: dict[str, tuple[float, float]]
    centre: tuple[float, float]
    length: float
    # "gainPlane" or "scale" (gradients held at zero), chosen by the BIC.
    model: str = "scale"
    criterion: dict[str, float] = field(default_factory=dict)

    def gain(self, key: str, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        a, b = self.gradients[key]
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if a == 0.0 and b == 0.0:
            return np.full(np.broadcast_shapes(x.shape, y.shape), self.scales[key])
        u = (x - self.centre[0]) / self.length
        v = (y - self.centre[1]) / self.length
        return self.scales[key] * np.exp(a * u + b * v)

    def ratio_error(self, left: str, right: str, x: float | None = None, y: float | None = None) -> float:
        """Standard error of ln(gain_left / gain_right) at canvas (x, y), the
        canvas centre by default."""

        u = 0.0 if x is None else (x - self.centre[0]) / self.length
        v = 0.0 if y is None else (y - self.centre[1]) / self.length
        i, j = self.order.index(left), self.order.index(right)
        direction = np.zeros(self.covariance.shape[0])
        direction[3 * i : 3 * i + 3] += (1.0, u, v)
        direction[3 * j : 3 * j + 3] -= (1.0, u, v)
        return math.sqrt(max(float(direction @ self.covariance @ direction), 0.0))


def _systematic_offset(edge: EdgeMeasurement) -> tuple[float, float, int]:
    """Misregistration of one overlap: the RMS of the median star offset in
    equal-count bins along the overlap, less the noise of those medians, and
    the robust per-star scatter within the bins.  Centroid noise and a few
    blended stars set the plain RMS of per-star offsets, not the panels'
    registration; a scatter taken over the whole overlap would also hold
    the variation between bins that is being measured."""

    dx, dy = edge.offset_x, edge.offset_y
    bins = int(np.clip(dx.size // OFFSET_BIN_STARS, 1, OFFSET_BINS))
    x0, y0, x1, y1 = edge.box
    along = edge.star_x if x1 - x0 >= y1 - y0 else edge.star_y
    groups = np.array_split(np.argsort(along, kind="stable"), bins)
    medians = np.asarray([(np.median(dx[group]), np.median(dy[group])) for group in groups], dtype=np.float64)
    within_x = np.concatenate([dx[group] - median[0] for group, median in zip(groups, medians)])
    within_y = np.concatenate([dy[group] - median[1] for group, median in zip(groups, medians)])
    scatter_x = 1.4826 * float(np.median(np.abs(within_x)))
    scatter_y = 1.4826 * float(np.median(np.abs(within_y)))
    # The standard error of a median is sqrt(pi/2) times that of a mean.
    noise = [(math.pi / 2.0) * (scatter_x**2 + scatter_y**2) / group.size for group in groups]
    systematic = math.sqrt(max(0.0, float(np.mean(np.sum(np.square(medians), axis=1))) - float(np.mean(noise))))
    return systematic, math.hypot(scatter_x, scatter_y), bins


def _ratio_bins(edge: EdgeMeasurement, rng: np.random.Generator) -> list[tuple[float, float, float, float]]:
    """(x, y, ln ratio, error) of the flux-flux slope in up to GAIN_BINS
    equal-count bins of at least GAIN_BIN_STARS stars along the overlap."""

    bins = int(np.clip(edge.flux_left.size // GAIN_BIN_STARS, 1, GAIN_BINS))
    x0, y0, x1, y1 = edge.box
    along = edge.star_x if x1 - x0 >= y1 - y0 else edge.star_y
    observations = []
    for group in np.array_split(np.argsort(along, kind="stable"), bins):
        location, error = _log_flux_ratio(edge.flux_left[group], edge.flux_right[group], rng)
        if math.isfinite(location) and math.isfinite(error):
            observations.append(
                (float(np.mean(edge.star_x[group])), float(np.mean(edge.star_y[group])), location, max(error, 1e-5))
            )
    return observations


@dataclass(frozen=True)
class ScaleObservation:
    """One overlap's flux ratio for the gain network: ln ratio and its
    bootstrap error over all its common stars, the same in bins along it
    (x, y, ln ratio, error), and the seed of its bootstraps."""

    edge: EdgeMeasurement
    location: float
    error: float
    bins: tuple[tuple[float, float, float, float], ...]
    seed: tuple[int, int]


def scale_observations(
    edges: Sequence[EdgeMeasurement], *, seed: int = 20261007
) -> dict[tuple[str, str], ScaleObservation]:
    """The flux ratio of every overlap that gives one: at least
    MINIMUM_OVERLAP_STARS common stars with a positive flux-flux slope and
    a finite bootstrap error.  Each overlap's bootstraps are seeded by its
    two panels, so its numbers do not depend on the other overlaps."""

    observations: dict[tuple[str, str], ScaleObservation] = {}
    for edge in edges:
        if edge.flux_left.size < MINIMUM_OVERLAP_STARS:
            continue
        edge_seed = (seed, zlib.crc32(f"{edge.left}\n{edge.right}".encode()))
        rng = np.random.default_rng(edge_seed)
        location, error = _log_flux_ratio(edge.flux_left, edge.flux_right, rng)
        if not (math.isfinite(location) and math.isfinite(error)):
            continue
        binned = _ratio_bins(edge, rng)
        if binned:
            observations[(edge.left, edge.right)] = ScaleObservation(
                edge, location, max(error, 1e-5), tuple(binned), edge_seed
            )
    return observations


def _components(keys: Sequence[str], pairs: Sequence[tuple[str, str]]) -> list[set[str]]:
    """The connected components of the graph of ``keys`` and ``pairs``, in
    the order of their first key."""

    adjacency: dict[str, set[str]] = {key: set() for key in keys}
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[str] = set()
    components: list[set[str]] = []
    for key in keys:
        if key in seen:
            continue
        members = {key}
        frontier = [key]
        while frontier:
            for neighbour in adjacency[frontier.pop()]:
                if neighbour not in members:
                    members.add(neighbour)
                    frontier.append(neighbour)
        seen |= members
        components.append(members)
    return components


def photometric_components(
    panels: Sequence[PanelImage], observations: Mapping[tuple[str, str], ScaleObservation]
) -> list[list[PanelImage]]:
    """The connected components of the panels' photometric graph, whose
    edges are the overlaps with a scale observation; largest first (panels,
    then area, then the first panel's position), each in the panels' order."""

    position = {panel.key: number for number, panel in enumerate(panels)}
    pairs = [pair for pair in observations if pair[0] in position and pair[1] in position]
    components = [
        sorted((panels[position[key]] for key in members), key=lambda item: position[item.key])
        for members in _components([panel.key for panel in panels], pairs)
    ]
    components.sort(
        key=lambda members: (
            -len(members),
            -sum(item.shape[0] * item.shape[1] for item in members),
            position[members[0].key],
        )
    )
    return components


def solve_scales(
    panels: Sequence[PanelImage], observations: Mapping[tuple[str, str], ScaleObservation]
) -> ScaleSolution:
    """One gain plane per panel, g = s * exp(a*u + b*v), from the binned
    star ratios of the overlaps between ``panels`` (robust network).  A gain
    gradient common to all panels is unobservable; a weak prior on each
    gradient keeps the minimum-norm solution, the panels' consensus."""

    order = [panel.key for panel in panels]
    index = {key: position for position, key in enumerate(order)}
    boxes = [panel.box for panel in panels]
    bx0, by0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    bx1, by1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    centre = ((bx0 + bx1 - 1) / 2.0, (by0 + by1 - 1) / 2.0)
    length = max(bx1 - bx0, by1 - by0) / 2.0
    used = [item for (left_key, right_key), item in observations.items() if left_key in index and right_key in index]
    if len(_components(order, [(item.edge.left, item.edge.right) for item in used])) > 1:
        raise MosaicPhotometryError(
            "MOSAIC_PHOTOMETRIC_GRAPH_DISCONNECTED", "the overlaps with enough common stars do not connect every panel"
        )
    count = len(order)
    if count == 1:
        return ScaleSolution(
            {order[0]: 1.0}, {order[0]: 0.0}, order[0], [], None, np.zeros((3, 3)), order,
            {order[0]: (0.0, 0.0)}, centre, length,
        )
    # Unknowns per panel: L = ln(scale), a, b.  Scaled panels agree at every
    # bin: (L_i + a_i u + b_i v) - (L_j + a_j u + b_j v) = -ln(F_i/F_j).
    left = np.asarray([index[item.edge.left] for item in used for _ in item.bins], dtype=np.int64)
    right = np.asarray([index[item.edge.right] for item in used for _ in item.bins], dtype=np.int64)
    binned = np.asarray([entry for item in used for entry in item.bins], dtype=np.float64).reshape(-1, 4)
    u = (binned[:, 0] - centre[0]) / length
    v = (binned[:, 1] - centre[1]) / length
    owner = np.asarray([number for number, item in enumerate(used) for _ in item.bins], dtype=np.int64)
    measured = int(binned.shape[0])

    def fit(
        gradients: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        terms = _GAIN_TERMS if gradients else 1
        design = _Design(terms * count)
        if gradients:
            columns = np.column_stack((3 * left, 3 * left + 1, 3 * left + 2, 3 * right, 3 * right + 1, 3 * right + 2))
            values = np.column_stack((np.ones(measured), u, v, -np.ones(measured), -u, -v))
        else:
            columns = np.column_stack((left, right))
            values = np.column_stack((np.ones(measured), -np.ones(measured)))
        design.add(columns, values, -binned[:, 2], binned[:, 3])
        if gradients:
            # The prior: every gradient is zero, to GAIN_GRADIENT_PRIOR.
            prior_columns = (3 * np.arange(count)[:, None] + np.asarray([1, 2])[None, :]).reshape(-1, 1)
            design.add(
                prior_columns, np.ones((2 * count, 1)), np.zeros(2 * count), np.full(2 * count, GAIN_GRADIENT_PRIOR)
            )
        solution, weights, residual, covariance = _robust_solve(design, robust_rows=measured, covariance=True)
        assert covariance is not None
        residual, weights = residual[:measured], weights[:measured]
        size = np.abs(residual)
        loss = float(np.sum(np.where(size <= HUBER_THRESHOLD, size**2, 2.0 * HUBER_THRESHOLD * size - HUBER_THRESHOLD**2)))
        if not gradients:
            full = np.zeros(_GAIN_TERMS * count)
            full[0::_GAIN_TERMS] = solution
            full_covariance = np.zeros((_GAIN_TERMS * count, _GAIN_TERMS * count))
            full_covariance[0::_GAIN_TERMS, 0::_GAIN_TERMS] = covariance
            solution, covariance = full, full_covariance
        return solution, weights, covariance, residual, loss

    # The gradients are fitted only where the overlaps carry them: the gain
    # plane must lower the BIC of the robust loss against one scale per panel.
    scale_only, plane = fit(False), fit(True)
    parameters = {"scale": count - 1, "gainPlane": 3 * (count - 1)}
    criterion = {
        name: result[4] + parameters[name] * math.log(max(measured, 1))
        for name, result in (("scale", scale_only), ("gainPlane", plane))
    }
    model = "gainPlane" if criterion["gainPlane"] < criterion["scale"] else "scale"
    solution, edge_weights, covariance, residual, _ = plane if model == "gainPlane" else scale_only
    dof = measured - parameters[model]
    chi2 = float(np.sum(np.square(residual))) / dof if dof > 0 else None
    # Gauge: the panel with the most flux per second (the most transparent
    # sky) keeps its scale; the others are brought up to it.
    levels = solution[0::3].copy()
    exposure = np.asarray([panel.exposure_seconds or 1.0 for panel in panels], dtype=np.float64)
    reference_index = int(np.argmin(levels + np.log(exposure)))
    levels -= levels[reference_index]
    scales = {key: float(math.exp(levels[position])) for key, position in index.items()}
    gradients = {key: (float(solution[3 * position + 1]), float(solution[3 * position + 2])) for key, position in index.items()}
    # Each ln scale's error relative to the reference, which has none.
    reference_term = 3 * reference_index
    errors = {
        key: float(
            math.sqrt(
                max(
                    covariance[3 * position, 3 * position]
                    + covariance[reference_term, reference_term]
                    - 2.0 * covariance[3 * position, reference_term],
                    0.0,
                )
            )
        )
        for key, position in index.items()
    }
    solved = ScaleSolution(
        scales, errors, order[reference_index], [], chi2, covariance, order, gradients, centre, length, model, criterion
    )
    for number, item in enumerate(used):
        edge = item.edge
        gain_left = solved.gain(edge.left, edge.star_x, edge.star_y)
        gain_right = solved.gain(edge.right, edge.star_x, edge.star_y)
        post, post_error = _log_flux_ratio(
            edge.flux_left * gain_left, edge.flux_right * gain_right, np.random.default_rng((*item.seed, 1))
        )
        offsets = np.hypot(edge.offset_x, edge.offset_y)
        systematic, scatter, bins = _systematic_offset(edge)
        measured_post = math.isfinite(post) and math.isfinite(post_error)
        record: dict[str, Any] = {
            "left": edge.left,
            "right": edge.right,
            "stars": int(edge.flux_left.size),
            "apertureRadiusPixels": edge.aperture_radius,
            "logRatio": item.location,
            "logRatioError": item.error,
            "ratioBins": len(item.bins),
            "weight": float(np.mean(edge_weights[owner == number])),
            "fwhmPixels": max(edge.fwhm_left, edge.fwhm_right),
            "astrometricOffsetRmsPixels": float(np.sqrt(np.mean(np.square(offsets)))),
            "astrometricOffsetMedian": [float(np.median(edge.offset_x)), float(np.median(edge.offset_y))],
            "astrometricSystematicPixels": systematic,
            "astrometricScatterPixels": scatter,
            "astrometricBins": bins,
            # None when the corrected ratio has no finite bootstrap error.
            "correctedRatioMinusOne": math.expm1(post) if measured_post else None,
            "correctedRatioError": max(post_error, 1e-5) if measured_post else None,
            "normalizedResidual": post / max(post_error, 1e-5) if measured_post else None,
        }
        solved.edges.append(record)
    return solved


class MosaicPhotometryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _polynomial_terms(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """The six terms of the additive model at (u, v): 1, u, v, u², uv, v²."""

    return np.column_stack((np.ones_like(u), u, v, u * u, u * v, v * v))


@dataclass
class PlaneSolution:
    """b_k(x, y) = c0 + c1*u + c2*v + c3*u² + c4*u*v + c5*v², u and v the
    canvas offsets from its centre in units of ``length``: subtracted from
    each scaled panel."""

    coefficients: dict[str, tuple[float, ...]]
    centre: tuple[float, float]
    length: float
    diagnostics: dict[str, Any]

    def evaluate(self, key: str, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        c0, c1, c2, c3, c4, c5 = self.coefficients[key]
        u = (np.asarray(x, dtype=np.float64) - self.centre[0]) / self.length
        v = (np.asarray(y, dtype=np.float64) - self.centre[1]) / self.length
        return c0 + c1 * u + c2 * v + c3 * u * u + c4 * u * v + c5 * v * v


def solve_planes(
    panels: Sequence[PanelImage],
    edges: Sequence[EdgeMeasurement],
    scales: ScaleSolution,
    canvas_shape: tuple[int, int],
) -> PlaneSolution:
    """The additive quadratic of every panel from the binned overlap
    differences of the scaled panels, keeping the area-weighted consensus
    sky."""

    order = [panel.key for panel in panels]
    index = {key: position for position, key in enumerate(order)}
    height, width = canvas_shape
    centre = ((width - 1) / 2.0, (height - 1) / 2.0)
    length = max(width, height) / 2.0
    design = _Design(PLANE_TERMS * len(order))
    edge_records = []
    galaxy_checks = []
    bin_noises: dict[str, list[NDArray[np.float64]]] = {key: [] for key in order}
    offsets = np.arange(PLANE_TERMS)
    for edge in edges:
        if edge.bin_x.size == 0:
            continue
        i, j = index[edge.left], index[edge.right]
        ci = scales.gain(edge.left, edge.bin_x, edge.bin_y)
        cj = scales.gain(edge.right, edge.bin_x, edge.bin_y)
        left = ci * edge.bin_left
        right = cj * edge.bin_right
        difference = left - right
        # sqrt(2) times the noise of the difference (bin_noise is already
        # that of the difference): the unit of the robust weights, the leak
        # mask and the curvature prior below.
        noise = np.hypot(ci * edge.bin_noise, cj * edge.bin_noise)
        bin_noises[edge.left].append(noise)
        bin_noises[edge.right].append(noise)
        structure = 0.5 * (left + right)
        structure = structure - float(np.median(structure))
        relative_error = scales.ratio_error(edge.left, edge.right, float(np.mean(edge.bin_x)), float(np.mean(edge.bin_y)))
        background = np.abs(structure) * relative_error <= SCALE_LEAK_FRACTION * noise
        u = (edge.bin_x[background] - centre[0]) / length
        v = (edge.bin_y[background] - centre[1]) / length
        terms = _polynomial_terms(u, v)
        columns = np.broadcast_to(
            np.concatenate((PLANE_TERMS * i + offsets, PLANE_TERMS * j + offsets)), (u.size, 2 * PLANE_TERMS)
        )
        design.add(columns, np.hstack((terms, -terms)), difference[background], noise[background])
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
    coefficients = {key: (0.0,) * PLANE_TERMS for key in order}
    diagnostics: dict[str, Any] = {"algorithm": BACKGROUND_ALGORITHM, "terms": PLANE_TERMS, "edges": edge_records}
    measured = design.count
    if measured:
        # The curvature prior of every panel, in the units of its terms: a
        # bin's difference noise is that of a median of BIN_PIXELS² pixel
        # differences, so a pixel's is BIN_PIXELS / 1.2533 of it.
        for position, panel in enumerate(panels):
            stacked = np.concatenate(bin_noises[panel.key]) if bin_noises[panel.key] else np.empty(0)
            pixel_noise = float(np.median(stacked)) * BIN_PIXELS / 1.2533 / math.sqrt(2.0) if stacked.size else 0.0
            if not pixel_noise > 0.0:
                continue
            half_u, half_v = panel.shape[1] / 2.0 / length, panel.shape[0] / 2.0 / length
            sigmas = QUADRATIC_PRIOR_NOISE * pixel_noise / np.asarray([half_u * half_u, half_u * half_v, half_v * half_v])
            columns = (PLANE_TERMS * position + np.asarray([3, 4, 5])).reshape(-1, 1)
            design.add(columns, np.ones((3, 1)), np.zeros(3), sigmas)
        solution, weights, residual, _ = _robust_solve(design, robust_rows=measured)
        residual, weights = residual[:measured], weights[:measured]
        planes = solution.reshape(len(order), PLANE_TERMS)
        # Minimum norm: the common polynomial every panel could carry is the
        # area-weighted consensus, which stays in the data.
        areas = np.asarray([panel.shape[0] * panel.shape[1] for panel in panels], dtype=np.float64)
        planes = planes - (areas[:, None] * planes).sum(axis=0) / areas.sum()
        coefficients = {key: tuple(float(value) for value in planes[index[key]]) for key in order}
        diagnostics["chi2PerBin"] = float(np.mean(np.square(residual)))
        diagnostics["robustWeightsBelowOne"] = int(np.count_nonzero(weights < 1.0))
    solution = PlaneSolution(coefficients, centre, length, diagnostics)
    # The extended-structure scale check: with the polynomials fixed, the
    # bright bins' remaining difference regressed on their structure.
    checks = []
    for edge, i, j, difference, noise, structure, background in galaxy_checks:
        bright = ~background
        if bright.sum() < 8:
            continue
        model = solution.evaluate(edge.left, edge.bin_x[bright], edge.bin_y[bright]) - solution.evaluate(
            edge.right, edge.bin_x[bright], edge.bin_y[bright]
        )
        remaining = difference[bright] - model
        bx, by = edge.bin_x[bright], edge.bin_y[bright]
        level = 0.5 * (scales.gain(edge.left, bx, by) * edge.bin_left[bright] + scales.gain(edge.right, bx, by) * edge.bin_right[bright])
        weight = 1.0 / np.square(noise[bright])
        delta = float(np.sum(weight * remaining * level) / np.sum(weight * level * level))
        delta_error = float(1.0 / math.sqrt(float(np.sum(weight * level * level))))
        star_error = scales.ratio_error(edge.left, edge.right, float(np.mean(bx)), float(np.mean(by)))
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
    "PLANE_TERMS",
    "EdgeMeasurement",
    "MosaicPhotometryError",
    "PanelImage",
    "PlaneSolution",
    "ScaleObservation",
    "ScaleSolution",
    "measure_edge",
    "photometric_components",
    "scale_observations",
    "solve_planes",
    "solve_scales",
]
