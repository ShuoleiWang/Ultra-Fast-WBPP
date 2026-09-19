"""Temporal, spatially coherent trail masks for ordinary integration.

A trail must be absent from the robust registered reference and have distributed
positive evidence along a long narrow corridor. Ordinary MAD rejection decisions
are preserved; the corridor mask can reject additional samples.

Detection (v2) integrates each frame's residual against the temporal median
along every line of every dyadic length with a fast Radon transform (the
multi-scale streak detection of Nir, Zackay & Ofek 2018, AJ 156, 229), so a
trail is found from the sum of its whole length rather than from individual
bright samples: a 0.3 sigma-per-pixel satellite across a frame integrates to a
30-40 sigma line, and a trail that fades or blinks along its length keeps the
extent over which its running mean stays positive.

At pixels affected by that additional rejection, an optional group-relative sky
model adjusts the temporary integration values to preserve the original accepted
background. This avoids stripes when the removed frame has a different sky level.
Stored inputs and pixels outside those corridors remain unchanged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from .native_kernels import RADON_KERNEL_ID, load_native_kernels
from .residual_background import ResidualBackgroundAlignment

ALGORITHM = "temporal-residual-radon-line-corridor-v2"
# (n, blocks, shift_index, columns, z) of one dyadic level's line peaks.
LevelPeaks = tuple[int, NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], NDArray[np.float32]]
MINIMUM_LENGTH_PIXELS = 256
# Dyadic line lengths tested, in preview bins: from FRT_MINIMUM_LEVEL_BINS up
# to the padded frame height.  A detection needs a line sum of
# FRT_DETECTION_Z (after per-level standardisation) and, on the preview, a
# core mean of at least CORE_MINIMUM_SIGMA, a peaked cross-track profile and
# a significance of at least SIGNIFICANCE_MINIMUM.
FRT_MINIMUM_LEVEL_BINS = 16
FRT_DETECTION_Z = 6.5
FRT_MINIMUM_COVERAGE = 0.6
FRT_MAXIMUM_CANDIDATES_PER_LEVEL = 24
# A line is valid at block length n when its weight count reaches
# max(FRT_MINIMUM_COUNT, FRT_MINIMUM_COVERAGE*n); a level standardises its z
# only when at least FRT_MINIMUM_SCALE_SAMPLES lines are valid.
FRT_MINIMUM_COUNT = 8.0
FRT_MINIMUM_SCALE_SAMPLES = 64
NUMPY_RADON_KERNEL_ID = "numpy-radon-peaks-v1"
CORE_MINIMUM_SIGMA = 0.2
SIGNIFICANCE_MINIMUM = 6.0
EXTENT_WINDOW_BINS = 32
EXTENT_THRESHOLD_SIGMA = 0.12
MINIMUM_SUPPORTED_SEGMENTS = 4
CORE_CONCENTRATION_MINIMUM = 0.6
OBJECT_MASK_SIGMA = 30.0
ANCHOR_STAR_SIGMA = 300.0
ANCHOR_CENTRE_FRACTION = 0.2
ANCHOR_MAXIMUM_LENGTH_PIXELS = 1600


@dataclass(frozen=True, slots=True)
class TransientTrail:
    frame: int
    normal_x: float
    normal_y: float
    distance: float
    start: float
    stop: float
    half_width: float
    seed_count: int
    supported_segments: int
    profile_sigma: float

    def serializable(self) -> dict[str, Any]:
        return {
            "frameIndex": self.frame,
            "normal": [self.normal_x, self.normal_y],
            "distancePixels": self.distance,
            "alongStartPixels": self.start,
            "alongStopPixels": self.stop,
            "halfWidthPixels": self.half_width,
            "seedCount": self.seed_count,
            "supportedSegments": self.supported_segments,
            "profileSigma": self.profile_sigma,
        }


@dataclass(frozen=True, slots=True)
class TransientRejectionModel:
    bin_factor: int
    trails: tuple[TransientTrail, ...] = ()
    status: str = "APPLIED"
    background_alignment: ResidualBackgroundAlignment | None = None
    line_kernel: str = NUMPY_RADON_KERNEL_ID

    def normalize_rejected_rows(self, values: NDArray[np.float32], first_row: int,
                                original: NDArray[np.bool_], accepted: NDArray[np.bool_]) -> None:
        """Adjust a temporary tile only where the spatial mask removed samples.

        ``original`` is the ordinary MAD acceptance mask; ``accepted`` includes
        the additional trail rejection. The background correction is anchored to
        the original weighted acceptance set, never to a single reference frame.
        """
        if self.background_alignment is None:
            return
        changed = np.any(original & ~accepted, axis=0)
        if not np.any(changed):
            return
        rows, cols = np.nonzero(changed)
        self._normalize_pixels(values, first_row, rows, cols,
                               [original[index, rows, cols] for index in range(values.shape[0])])

    def apply_corridors(self, values: NDArray[np.float32], accepted: NDArray[np.bool_],
                        first_row: int, enough_samples: NDArray[np.bool_]) -> int:
        """Reject every trail corridor of this row band and normalise the changed pixels.

        The same decisions and values as ``reject_rows`` followed by
        ``normalize_rejected_rows`` on a copy of the original mask, without the
        whole-tile passes: the corridors are evaluated on their candidate
        pixels, the pixels whose acceptance they change are collected before
        any flag is cleared, and the background model is evaluated there only.
        Returns the number of changed samples.
        """
        height, width = accepted.shape[1:]
        corridors: list[tuple[int, NDArray[np.intp], NDArray[np.intp]]] = []
        changed: list[NDArray[np.intp]] = []
        for trail in self.trails:
            rows, cols = _corridor_candidates(trail, first_row, height, width)
            if rows.size == 0:
                continue
            x = cols.astype(np.float64)
            y = (rows + first_row).astype(np.float64)
            distance = x * trail.normal_x + y * trail.normal_y - trail.distance
            along = -x * trail.normal_y + y * trail.normal_x
            corridor = ((np.abs(distance) <= trail.half_width)
                        & (along >= trail.start) & (along <= trail.stop)
                        & enough_samples[rows, cols])
            rows, cols = rows[corridor], cols[corridor]
            corridors.append((trail.frame, rows, cols))
            newly = accepted[trail.frame, rows, cols]
            changed.append(rows[newly] * width + cols[newly])
        if not corridors:
            return 0
        linear = np.unique(np.concatenate(changed))
        if linear.size == 0:
            for frame, rows, cols in corridors:
                accepted[frame, rows, cols] = False
            return 0
        rows, cols = np.divmod(linear, width)
        original = [accepted[index, rows, cols] for index in range(values.shape[0])]
        for frame, corridor_rows, corridor_cols in corridors:
            accepted[frame, corridor_rows, corridor_cols] = False
        if self.background_alignment is not None:
            self._normalize_pixels(values, first_row, rows, cols, original)
        return int(linear.size)

    def _normalize_pixels(self, values: NDArray[np.float32], first_row: int,
                          rows: NDArray[np.intp], cols: NDArray[np.intp],
                          original: list[NDArray[np.bool_]]) -> None:
        """Group-relative sky alignment at the pixels ``(rows, cols)`` of the band.

        ``original[f]`` is the ordinary acceptance of frame ``f`` at those
        pixels.  The model is evaluated at the pixels alone with the per-pixel
        expressions of a whole-tile evaluation, accumulated over frames in the
        same order, so the values are identical.
        """
        alignment = self.background_alignment
        assert alignment is not None
        weights = np.asarray(alignment.weights, dtype=np.float64)
        correction = alignment.corrections_at(
            cols.astype(np.float64), (rows + first_row).astype(np.float64)
        )
        # Frame-sequential Float64 sums, the order np.sum(axis=0) uses on a
        # frame-major tile.
        denominator = np.zeros(rows.size, dtype=np.float64)
        numerator = np.zeros(rows.size, dtype=np.float64)
        for index in range(values.shape[0]):
            denominator += original[index] * weights[index]
            numerator += np.where(original[index], correction[index], 0) * weights[index]
        anchor = np.divide(
            numerator, denominator, out=np.zeros_like(denominator), where=denominator > 0,
        )
        # Equivalent to mean_new(X-C) + mean_original(C). No correction is
        # applied elsewhere, so ordinary integration controls stay bitwise equal.
        for index in range(values.shape[0]):
            values[index, rows, cols] += (correction[index] - anchor).astype(np.float32)

    def serializable(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM,
            "status": self.status,
            "binFactor": self.bin_factor,
            "reference": "registered-group-temporal-median",
            "backgroundRemoval": "detection-only-clipped-smooth-residual",
            "minimumLengthPixels": MINIMUM_LENGTH_PIXELS,
            "detection": "dyadic-fast-radon-multiscale",
            "detectionZ": FRT_DETECTION_Z,
            "minimumLevelBins": FRT_MINIMUM_LEVEL_BINS,
            "coreMinimumSigma": CORE_MINIMUM_SIGMA,
            "extentThresholdSigma": EXTENT_THRESHOLD_SIGMA,
            "minimumSupportedSegments": MINIMUM_SUPPORTED_SEGMENTS,
            "coreConcentrationMinimum": CORE_CONCENTRATION_MINIMUM,
            "objectMaskSigma": OBJECT_MASK_SIGMA,
            "anchorStarSigma": ANCHOR_STAR_SIGMA,
            "minimumProfileSigma": SIGNIFICANCE_MINIMUM,
            "tileInvariant": True,
            "intensitiesModified": self.background_alignment is not None,
            "storedInputsModified": False,
            "skyAlignmentApplication": "additional-spatial-rejection-pixels-only",
            "skyReference": "original-mad-accepted-weighted-background",
            "ordinaryMadDecisions": "unchanged-original-input-values",
            "backgroundAlignment": (self.background_alignment.serializable()
                                    if self.background_alignment is not None else None),
            "trails": [trail.serializable() for trail in self.trails],
            "lineKernel": self.line_kernel,
        }

    def reject_rows(self, accepted: NDArray[np.bool_], first_row: int,
                    enough_samples: NDArray[np.bool_]) -> None:
        """Clear the accepted flags inside every trail corridor of this row band.

        The corridor test is evaluated only on the pixels of each row that can
        satisfy it (the analytic cross-track and along-track intervals with a
        two-pixel margin), with the same Float64 expressions the full-tile
        evaluation used, so the decisions are identical at a fraction of the
        work.
        """
        height, width = accepted.shape[1:]
        for trail in self.trails:
            rows, cols = _corridor_candidates(trail, first_row, height, width)
            if rows.size == 0:
                continue
            x = cols.astype(np.float64)
            y = (rows + first_row).astype(np.float64)
            distance = x * trail.normal_x + y * trail.normal_y - trail.distance
            along = -x * trail.normal_y + y * trail.normal_x
            corridor = ((np.abs(distance) <= trail.half_width)
                        & (along >= trail.start) & (along <= trail.stop)
                        & enough_samples[rows, cols])
            accepted[trail.frame, rows[corridor], cols[corridor]] = False


def _corridor_candidates(trail: TransientTrail, first_row: int, height: int,
                         width: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Pixels of a row band that can lie in a trail corridor: (rows, columns).

    For every band row the cross-track condition ``|x nx + y ny - d| <= h``
    and the along-track condition ``start <= -x ny + y nx <= stop`` are each
    an interval of x (or the whole row, or nothing, when the normal component
    vanishes); their intersection, widened by two pixels against rounding, is
    the candidate run of the row.  Rows are band-relative.
    """

    y = np.arange(first_row, first_row + height, dtype=np.float64)
    low = np.full(height, -np.inf)
    high = np.full(height, np.inf)
    keep = np.ones(height, dtype=bool)
    nx, ny = trail.normal_x, trail.normal_y
    if abs(nx) > 1e-9:
        a = (trail.distance - trail.half_width - y * ny) / nx
        b = (trail.distance + trail.half_width - y * ny) / nx
        low = np.maximum(low, np.minimum(a, b))
        high = np.minimum(high, np.maximum(a, b))
    else:
        keep &= np.abs(y * ny - trail.distance) <= trail.half_width + 1.0
    if abs(ny) > 1e-9:
        a = (y * nx - trail.stop) / ny
        b = (y * nx - trail.start) / ny
        low = np.maximum(low, np.minimum(a, b))
        high = np.minimum(high, np.maximum(a, b))
    else:
        keep &= (y * nx >= trail.start - 1.0) & (y * nx <= trail.stop + 1.0)
    low = np.maximum(np.floor(low) - 2.0, 0.0)
    high = np.minimum(np.ceil(high) + 2.0, float(width - 1))
    keep &= high >= low
    band_rows = np.flatnonzero(keep)
    if band_rows.size == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    starts = low[band_rows].astype(np.intp)
    lengths = (high[band_rows] - low[band_rows]).astype(np.intp) + 1
    rows = np.repeat(band_rows, lengths)
    offsets = np.arange(rows.size, dtype=np.intp) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    cols = np.repeat(starts, lengths) + offsets
    return rows, cols


def _next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


def fast_radon_levels(image: NDArray[np.float32], minimum_rows: int) -> list[tuple[int, NDArray[np.float32]]]:
    """Dyadic fast Radon transform for lines within 45 degrees of the columns.

    ``image`` is ``(H, W)`` with ``H`` a power of two.  Returns, for every
    block length ``n = 2**k >= minimum_rows``, an array ``F`` of shape
    ``(H // n, 2n - 1, W + 2H)`` where ``F[b, s + n - 1, x]`` is the sum along
    the dyadic line from ``(x - H, b n)`` to ``(x - H + s, b n + n - 1)``
    (columns are padded by ``H`` zeros on both sides so every shift stays
    inside).  Level ``k`` is built from level ``k - 1`` as
    ``F_k[b, s, x] = F_{k-1}[2b, h, x] + F_{k-1}[2b+1, h, x + (s - h)]`` with
    ``h = trunc(s / 2)``, the standard O(H W log H) recursion.
    """

    height, width = image.shape
    pad = height
    padded = np.zeros((height, width + 2 * pad), dtype=np.float32)
    padded[:, pad : pad + width] = image
    current = padded[:, None, :]  # (blocks=H, shifts=1, Wp): n = 1, s = 0
    levels: list[tuple[int, NDArray[np.float32]]] = []
    n = 1
    columns = np.arange(padded.shape[1])
    while n < height:
        half = n
        n *= 2
        shifts = np.arange(-(n - 1), n)
        halves = np.trunc(shifts / 2.0).astype(np.int64)
        deltas = shifts - halves
        half_index = halves + (half - 1)
        top = current[0::2][:, half_index, :]
        bottom = current[1::2][:, half_index, :]
        take = np.clip(columns[None, :] + deltas[:, None], 0, padded.shape[1] - 1)
        shifted = np.take_along_axis(bottom, np.broadcast_to(take[None], bottom.shape), axis=2)
        current = top + shifted
        if n >= minimum_rows:
            levels.append((n, current))
    return levels


def _line_normal_form(p0: NDArray[np.float64], p1: NDArray[np.float64]) -> tuple[float, float, float, float, float]:
    """Normal (nx, ny), distance and along-track coordinates of a segment."""

    direction = p1 - p0
    length = float(np.hypot(direction[0], direction[1]))
    if length <= 0:
        raise ValueError("degenerate segment")
    tx, ty = direction / length
    nx, ny = -ty, tx
    distance = float(p0[0] * nx + p0[1] * ny)
    a0 = float(-p0[0] * ny + p0[1] * nx)
    a1 = float(-p1[0] * ny + p1[1] * nx)
    return nx, ny, distance, min(a0, a1), max(a0, a1)


def _standardised_z(sums: NDArray[np.float32], counts: NDArray[np.float32], minimum_count: float) -> NDArray[np.float32]:
    valid = counts >= minimum_count
    z = np.zeros(sums.shape, dtype=np.float32)
    np.divide(sums, np.sqrt(np.maximum(counts, 1.0)), out=z, where=valid)
    sample = z[valid]
    if sample.size >= FRT_MINIMUM_SCALE_SAMPLES:
        scale = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
        if np.isfinite(scale) and scale > 0:
            z /= np.float32(scale)
    z[~valid] = 0.0
    return z


def _numpy_level_peaks(image: NDArray[np.float32], weights: NDArray[np.float32],
                       minimum_rows: int) -> list[LevelPeaks]:
    """NumPy reference of the native ``radon_line_peaks`` kernel.

    Per dyadic level ``n >= minimum_rows``: the peaks of the standardised line
    z (``z >= FRT_DETECTION_Z`` and the maximum of the (1, 5, 7) window) as
    ``(n, blocks, shift_index, columns, z)`` in ``np.nonzero`` order.
    """

    from scipy.ndimage import maximum_filter

    sums = fast_radon_levels(image, minimum_rows)
    counts = fast_radon_levels(weights, minimum_rows)
    levels: list[LevelPeaks] = []
    for (n, level_sums), (_, level_counts) in zip(sums, counts, strict=True):
        z = _standardised_z(level_sums, level_counts, max(8.0, FRT_MINIMUM_COVERAGE * n))
        peaks = (z >= FRT_DETECTION_Z) & (z == maximum_filter(z, size=(1, 5, 7)))
        blocks, shift_index, xs = np.nonzero(peaks)
        levels.append((n, blocks, shift_index, xs, z[blocks, shift_index, xs]))
    return levels


def _level_peaks(residual_z: NDArray[np.float32], detect: NDArray[np.bool_], size: int,
                 minimum_rows: int, kernels: Any, threads: int | None) -> list[LevelPeaks]:
    """Line peaks of one orientation from the native kernel or its NumPy reference."""

    if kernels is not None:
        return kernels.radon_line_peaks(
            residual_z, detect.astype(np.uint8), size=size, minimum_rows=minimum_rows,
            detection_z=FRT_DETECTION_Z, minimum_coverage=FRT_MINIMUM_COVERAGE,
            minimum_count=FRT_MINIMUM_COUNT, minimum_scale_samples=FRT_MINIMUM_SCALE_SAMPLES,
            threads=threads,
        )
    h, w = residual_z.shape
    image = np.zeros((size, w), dtype=np.float32)
    image[:h] = residual_z
    weights = np.zeros((size, w), dtype=np.float32)
    weights[:h] = detect
    return _numpy_level_peaks(image, weights, minimum_rows)


def _candidate_lines(residual_z: NDArray[np.float32], detect: NDArray[np.bool_], minimum_rows: int,
                     *, kernels: Any = None, threads: int | None = None):
    """Multi-scale line candidates of one frame: (z, p0, p1) in preview coords."""

    height, width = residual_z.shape
    size = _next_power_of_two(max(height, width))
    candidates: list[tuple[float, NDArray[np.float64], NDArray[np.float64]]] = []
    for transpose in (False, True):
        img = np.ascontiguousarray(residual_z.T) if transpose else residual_z
        wgt = np.ascontiguousarray(detect.T) if transpose else detect
        for n, blocks, shift_index, xs, z_values in _level_peaks(
                img, wgt, size, minimum_rows, kernels, threads):
            if blocks.size > FRT_MAXIMUM_CANDIDATES_PER_LEVEL:
                order = np.argsort(-z_values)[:FRT_MAXIMUM_CANDIDATES_PER_LEVEL]
                blocks, shift_index, xs = blocks[order], shift_index[order], xs[order]
                z_values = z_values[order]
            for b, si, x, z_line in zip(blocks, shift_index, xs, z_values):
                shift = int(si) - (n - 1)
                x0 = float(int(x) - size)
                y0 = float(int(b) * n)
                p0 = np.array([x0, y0])
                p1 = np.array([x0 + shift, y0 + n - 1])
                if transpose:
                    p0, p1 = p0[::-1], p1[::-1]
                candidates.append((float(z_line), p0, p1))
    candidates.sort(key=lambda item: -item[0])
    return candidates


def _robust_sky(reference: NDArray[np.float32], common: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Large-scale sky of the reference that ignores objects (clipped smoothing)."""

    from scipy.ndimage import gaussian_filter

    level = float(np.median(reference[common])) if np.any(common) else 0.0
    filled = np.where(common, reference, level).astype(np.float64)
    weight = common.astype(np.float64)
    sky = np.full(reference.shape, level)
    for _ in range(3):
        residual = filled - sky
        scale = 1.4826 * float(np.median(np.abs(residual[common] - np.median(residual[common])))) if np.any(common) else 1.0
        clipped = np.clip(residual, -3.0 * scale, 3.0 * scale)
        sky = sky + gaussian_filter(clipped * weight, 24) / np.maximum(gaussian_filter(weight, 24), 0.05)
    return sky


def _anchored_on_star(result: dict, anchors: NDArray[np.bool_]) -> bool:
    """True when the line's extent is centred on a bright star it passes through.

    Diffraction spikes and halo asymmetries of a bright star under a seeing
    or transparency difference are line-like residuals of one frame, but they
    are symmetric about the star; a satellite that happens to cross a bright
    star has its extent centred elsewhere.
    """

    ys, xs = np.nonzero(anchors)
    if ys.size == 0:
        return False
    nx, ny = result["nx"], result["ny"]
    cross = xs * nx + ys * ny - result["distance"]
    along = -xs * ny + ys * nx
    near = np.abs(cross) <= 2.5
    if not np.any(near):
        return False
    start, stop = result["start"], result["stop"]
    length = max(stop - start, 1.0)
    middle = 0.5 * (start + stop)
    return bool(np.any(np.abs(along[near] - middle) <= ANCHOR_CENTRE_FRACTION * length))


def _merge_into(trail: dict, result: dict, half_width: float) -> bool:
    """Union ``result`` into ``trail`` when both describe the same line.

    Same line: normals within 2 degrees and the new segment's end points
    within 3 bins of the accepted line.  The new extent is projected onto the
    accepted line's along-track axis.
    """

    nx, ny = trail["nx"], trail["ny"]
    dot = nx * result["nx"] + ny * result["ny"]
    if abs(dot) < 0.99939:
        return False
    rx, ry, rd = result["nx"], result["ny"], result["distance"]
    ends = []
    for a in (result["start"], result["stop"]):
        point = np.array([rd * rx - a * ry, rd * ry + a * rx])
        cross = point[0] * nx + point[1] * ny - trail["distance"]
        if abs(cross) > 3.0:
            return False
        ends.append(float(-point[0] * ny + point[1] * nx))
    trail["start"] = min(trail["start"], min(ends))
    trail["stop"] = max(trail["stop"], max(ends))
    trail["half_width"] = max(trail["half_width"], half_width)
    return True


def _segment_inside(trail: dict, p0: NDArray[np.float64], p1: NDArray[np.float64]) -> bool:
    """True when a candidate segment is already covered by an accepted trail.

    Either both end points lie inside the corridor, or the segment is nearly
    parallel (within 10 degrees) and its midpoint lies inside the corridor:
    an oblique dyadic line that only crosses a bright trail.
    """

    nx, ny = trail["nx"], trail["ny"]
    reach = trail["half_width"] + 2.0
    inside = []
    for point in (p0, p1, 0.5 * (p0 + p1)):
        cross = point[0] * nx + point[1] * ny - trail["distance"]
        along = -point[0] * ny + point[1] * nx
        inside.append(abs(cross) <= reach and trail["start"] - 8 <= along <= trail["stop"] + 8)
    if inside[0] and inside[1]:
        return True
    direction = p1 - p0
    length = float(np.hypot(direction[0], direction[1]))
    if length <= 0:
        return False
    cos_angle = abs((-direction[1] * nx + direction[0] * ny) / length)
    return bool(inside[2] and cos_angle > 0.985)


def _refine_line(residual, valid, xx, yy, nx, ny, distance, start, stop):
    """Best (nx, ny, distance) around a candidate inside its along-track range."""

    margin = 12.0
    cross0 = xx * nx + yy * ny - distance
    along0 = -xx * ny + yy * nx
    roi = valid & (np.abs(cross0) <= margin) & (along0 >= start - 64) & (along0 <= stop + 64)
    ry, rx = np.nonzero(roi)
    if ry.size < 32:
        return None
    rv = residual[ry, rx]
    fx, fy = xx[ry, rx], yy[ry, rx]
    base_angle = np.arctan2(ny, nx)
    best = None
    for dtheta in np.deg2rad(np.arange(-2.0, 2.001, 0.1)):
        cnx, cny = float(np.cos(base_angle + dtheta)), float(np.sin(base_angle + dtheta))
        cross = fx * cnx + fy * cny
        for drho in np.arange(-1.5, 1.51, 0.5):
            core = np.abs(cross - (distance + drho)) <= 0.5
            count = np.count_nonzero(core)
            if count < 16:
                continue
            score = float(np.sum(rv[core])) / np.sqrt(count)
            if best is None or score > best[0]:
                best = (score, cnx, cny, distance + drho)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _tracked_extent(core_along, core_values, sigma, seed_along):
    """Grow the extent outward from ``seed_along`` while the running mean of
    the core samples stays positive; gaps shorter than two windows are bridged
    (a blinking aircraft, a star mask on the line)."""

    window = min(EXTENT_WINDOW_BINS, core_values.size)
    running = np.convolve(core_values, np.ones(window) / window, mode="same")
    on = running > EXTENT_THRESHOLD_SIGMA * sigma
    centre = int(np.argmin(np.abs(core_along - seed_along)))
    if not on[centre]:
        near = np.flatnonzero(on)
        if near.size == 0:
            return None
        centre = int(near[np.argmin(np.abs(near - centre))])
        if abs(core_along[centre] - seed_along) > 2 * window:
            return None
    gap_limit = 2 * window
    lo = hi = centre
    gap = 0
    for index in range(centre - 1, -1, -1):
        if on[index]:
            lo, gap = index, 0
        else:
            gap += 1
            if gap > gap_limit:
                break
    gap = 0
    for index in range(centre + 1, core_values.size):
        if on[index]:
            hi, gap = index, 0
        else:
            gap += 1
            if gap > gap_limit:
                break
    return float(core_along[lo]), float(core_along[hi])


def _measure_line(residual: NDArray[np.float64], valid: NDArray[np.bool_], sigma: float,
                  nx: float, ny: float, distance: float, start: float, stop: float):
    """Refine a candidate line on the preview and measure its extent/profile.

    Returns None when the line is not a peaked, significant, distributed
    positive residual.  Otherwise a dict with the refined normal form, the
    along-track extent, the core statistics and the cross-track profile.
    The extent is tracked outward along the refined line beyond the
    candidate block, and the line is re-refined on the grown extent, so one
    dyadic block of a long or fading trail yields the whole trail.
    """

    height, width = residual.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    seed_mid = 0.5 * (start + stop)
    ext_start, ext_stop = start, stop
    cnx = cny = cdist = None
    for _ in range(3):
        refined = _refine_line(residual, valid, xx, yy, nx, ny, distance, ext_start, ext_stop)
        if refined is None:
            return None
        cnx, cny, cdist = refined
        band = valid & (np.abs(xx * cnx + yy * cny - cdist) <= 1.0)
        by, bx = np.nonzero(band)
        if by.size < 32:
            return None
        along = -bx * cny + by * cnx
        order = np.argsort(along)
        core_along = along[order]
        core_values = residual[by, bx][order]
        extent = _tracked_extent(core_along, core_values, sigma, seed_mid)
        if extent is None:
            return None
        new_start, new_stop = extent
        seed_mid = 0.5 * (new_start + new_stop)
        nx, ny, distance = cnx, cny, cdist
        grown = (new_stop - new_start) > 1.1 * (ext_stop - ext_start) + 8
        ext_start, ext_stop = new_start, new_stop
        if not grown:
            break
    inside = (core_along >= ext_start) & (core_along <= ext_stop)
    core_inside = core_values[inside]
    if core_inside.size < 16:
        return None
    clipped = np.clip(core_inside, -4 * sigma, 4 * sigma)
    mean_core = float(np.mean(clipped))
    significance = mean_core / sigma * np.sqrt(core_inside.size)
    if mean_core < CORE_MINIMUM_SIGMA * sigma or significance < SIGNIFICANCE_MINIMUM:
        return None
    edges = np.linspace(ext_start, ext_stop, 9)
    along_inside = core_along[inside]
    supported = 0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        segment = (along_inside >= lo) & (along_inside < hi)
        if np.count_nonzero(segment) >= 4 and np.mean(clipped[segment]) > EXTENT_THRESHOLD_SIGMA * sigma:
            supported += 1
    if supported < MINIMUM_SUPPORTED_SEGMENTS:
        return None
    # Cross-track profile over the extent (median per one-bin band).
    wide = valid & (np.abs(xx * cnx + yy * cny - cdist) <= 8.5)
    wy, wx = np.nonzero(wide)
    rel = wx * cnx + wy * cny - cdist
    walong = -wx * cny + wy * cnx
    wv = residual[wy, wx]
    in_extent = (walong >= ext_start) & (walong <= ext_stop)
    profile = np.zeros(17)
    counts = np.zeros(17, dtype=np.int64)
    for offset in range(-8, 9):
        band = in_extent & (np.abs(rel - offset) < 0.5)
        counts[offset + 8] = np.count_nonzero(band)
        profile[offset + 8] = float(np.median(wv[band])) if counts[offset + 8] >= 8 else 0.0
    wings = [profile[8 + k] for k in (-3, -2, 2, 3) if counts[8 + k] >= 8]
    if len(wings) < 3:
        return None
    wing = float(np.mean(np.abs(wings)))
    if profile[8] < 2.0 * wing + 0.1 * sigma:
        return None
    # A trail is narrow: the three core bands must hold most of the positive
    # cross-track flux within +-6 bins.  A bright star's asymmetric halo or a
    # reflection ghost in one frame is a broad blob that fails this.
    positive = np.clip(profile[2:15], 0.0, None)
    if positive.sum() <= 0 or positive[5:8].sum() < CORE_CONCENTRATION_MINIMUM * positive.sum():
        return None
    return {
        "nx": cnx, "ny": cny, "distance": cdist, "start": ext_start, "stop": ext_stop,
        "core_count": int(core_inside.size), "mean_core": mean_core,
        "significance": float(significance), "supported": supported, "profile": profile,
    }


def detect_transient_trails(values: NDArray[np.float32], bin_factor: int,
                            *, workers: int = 1) -> TransientRejectionModel:
    """Fit deterministic corridors from block-mean registered frame samples.

    Frames are analysed independently against the shared temporal median, so
    ``workers`` frames run concurrently; the fitted corridors are identical
    for any worker count and keep frame order.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    frame_count, height, width = values.shape
    if frame_count < 5 or min(height, width) * bin_factor < MINIMUM_LENGTH_PIXELS:
        return TransientRejectionModel(bin_factor, status="NOT_APPLICABLE")
    finite = np.isfinite(values)
    common = np.sum(finite, axis=0) >= max(5, (frame_count + 1) // 2)
    reference = np.zeros((height, width), dtype=np.float32)
    if np.any(common):
        reference[common] = np.nanmedian(values[:, common], axis=0)
    # Shared stars, diffraction spikes, and resolved structure cannot seed
    # a trail; the structure map depends only on the reference.  Bright
    # extended objects and star halos are masked from detection as well:
    # under a small scale or seeing difference their residual scales with
    # their own brightness, and a galaxy's major axis integrates like a line.
    structure = reference - gaussian_filter(reference, 3)
    sky = _robust_sky(reference, common)
    elevation = np.where(common, reference - sky, 0.0)
    minimum_rows = max(2, min(_next_power_of_two(FRT_MINIMUM_LEVEL_BINS),
                              _next_power_of_two(max(1, MINIMUM_LENGTH_PIXELS // (2 * bin_factor)))))
    frame_workers = max(1, min(workers, frame_count))
    # Frames are the unit of concurrency; a lone frame lets the kernel split
    # its own levels across the tuning row's thread budget instead.
    kernels = load_native_kernels()
    kernel_threads = 1 if frame_workers > 1 else None

    def frame_trails(frame_index: int) -> list[TransientTrail]:
        trails: list[TransientTrail] = []
        valid = common & finite[frame_index]
        if np.count_nonzero(valid) < 256:
            return trails
        residual = np.where(valid, values[frame_index] - reference, 0.0)
        location = float(np.median(residual[valid]))
        sigma = float(1.4826 * np.median(np.abs(residual[valid] - location)))
        if not np.isfinite(sigma) or sigma <= 0:
            return trails
        # Clipping keeps a satellite itself from raising the background model.
        clipped = np.clip(residual - location, -3 * sigma, 3 * sigma)
        support = gaussian_filter(valid.astype(np.float32), 8)
        background = gaussian_filter(np.where(valid, clipped, 0), 8)
        background /= np.maximum(support, 0.01)
        residual = residual - location - background
        sigma = float(1.4826 * np.median(np.abs(
            residual[valid] - np.median(residual[valid]))))
        if not np.isfinite(sigma) or sigma <= 0:
            return trails
        # The final corridor can cross protected structure, using other
        # frames there.
        protected = binary_dilation(structure > 4 * sigma, iterations=2)
        protected |= binary_dilation(elevation > OBJECT_MASK_SIGMA * sigma, iterations=2)
        anchors = structure > ANCHOR_STAR_SIGMA * sigma
        # Normalized smoothing remains valid at a footprint boundary: requiring
        # almost full support would truncate every fitted trail by several
        # smoothing radii before the image edge.
        detect = valid & ~protected & (support > 0.25)
        residual_z = np.where(detect, np.clip(residual / sigma, -5.0, 5.0), 0.0).astype(np.float32)
        candidates = _candidate_lines(residual_z, detect, minimum_rows,
                                      kernels=kernels, threads=kernel_threads)
        measured = []
        # Accepted corridors are blanked before the next candidate is measured,
        # so oblique dyadic lines that only borrow a bright trail's samples
        # lose their support ("clean" order: strongest line first).
        work_residual = residual.copy()
        work_detect = detect.copy()
        yy, xx = np.mgrid[:height, :width].astype(np.float64)
        for z_line, p0, p1 in candidates:
            try:
                nx, ny, distance, start, stop = _line_normal_form(p0, p1)
            except ValueError:
                continue
            # Skip candidates whose segment already lies inside an accepted
            # corridor (shorter or oblique dyadic lines through the same trail).
            if any(_segment_inside(t, p0, p1) for t in measured):
                continue
            result = _measure_line(work_residual, work_detect, sigma, nx, ny, distance, start, stop)
            if result is None:
                continue
            if ((result["stop"] - result["start"]) * bin_factor <= ANCHOR_MAXIMUM_LENGTH_PIXELS
                    and _anchored_on_star(result, anchors)):
                continue
            length = result["stop"] - result["start"]
            if length * bin_factor < MINIMUM_LENGTH_PIXELS:
                continue
            profile = result["profile"]
            threshold = max(0.15, 3 / np.sqrt(max(result["core_count"], 1) / 3)) * sigma
            left = right = 8
            while left > 0 and profile[left - 1] > threshold:
                left -= 1
            while right < 16 and profile[right + 1] > threshold:
                right += 1
            half_width = float(max(8 - left, right - 8) + 1.5)
            # The profile is measured over +-8 bins; a brighter trail's wings
            # are simply capped there.  The corridor never covers more than
            # 5% of the frame: the concentration test above already refused
            # blobs, so a wider profile only narrows the corridor.
            half_width = min(half_width, 8.5, height * width * 0.05 / (2.0 * max(length, 1.0)))
            # Merge with an accepted line of the same geometry (other levels
            # or blocks of the same trail): keep the union of the extents,
            # projected onto the accepted line.
            merged = False
            for t in measured:
                if _merge_into(t, result, half_width):
                    merged = True
                    break
            if merged:
                continue
            result["half_width"] = half_width
            result["z_line"] = z_line
            measured.append(result)
            cross = xx * result["nx"] + yy * result["ny"] - result["distance"]
            along = -xx * result["ny"] + yy * result["nx"]
            blank = ((np.abs(cross) <= half_width + 1.0)
                     & (along >= result["start"] - 4) & (along <= result["stop"] + 4))
            work_residual[blank] = 0.0
            work_detect[blank] = False
        shift = (bin_factor - 1) / 2
        for t in measured:
            nx, ny = t["nx"], t["ny"]
            trails.append(TransientTrail(
                frame_index, nx, ny,
                float(t["distance"] * bin_factor + shift * (nx + ny)),
                float(t["start"] * bin_factor + shift * (nx - ny) - 2 * bin_factor),
                float(t["stop"] * bin_factor + shift * (nx - ny) + 2 * bin_factor),
                t["half_width"] * bin_factor, int(t["core_count"]), int(t["supported"]),
                float(t["significance"]),
            ))
        trails.sort(key=lambda trail: (-trail.profile_sigma, trail.distance))
        return trails

    if frame_workers == 1:
        per_frame = [frame_trails(index) for index in range(frame_count)]
    else:
        with ThreadPoolExecutor(
            max_workers=frame_workers, thread_name_prefix="oaf-trails"
        ) as executor:
            per_frame = list(executor.map(frame_trails, range(frame_count)))
    trails = [trail for frame in per_frame for trail in frame]
    return TransientRejectionModel(
        bin_factor, tuple(trails),
        line_kernel=RADON_KERNEL_ID if kernels is not None else NUMPY_RADON_KERNEL_ID,
    )
