"""The protected set of the two-scale blend (docs/mosaic-plan.md §6.2).

Outside it the mosaic is the inverse-variance mean of the panels.  A bright
star or a galaxy core in an overlap is different: panels of different
seeing hold different profiles of it, and their mean is neither.  Such a
blob takes its small scales from one panel ``a``, the one with the highest
blend weight at its peak, while its large scales stay averaged:

    M = Σ W_k J_k + ρ Σ W_k HP(J_a − J_k)

``J_k`` are the matched panels, ``W_k`` the normalized blend weights, ``HP``
the high band (the image minus its σ = 16 px normalized Gaussian low band)
and ``ρ`` is 1 on the blob and falls to 0 over 8 px.  Where ``ρ`` is 0 the
blend is untouched, and where one panel alone covers a pixel the correction
vanishes.

A blob grows from every local maximum above 50 σ of a panel's high band in
an overlap.  Its footprint is a core disk of two FWHM plus the connected
region around it where two panels' high bands differ significantly (their
different seeing or a residual misalignment), within twice the radius at
which a Moffat (β = 2.5) star of that peak in the wider seeing fades to
1 σ.  Footprints that touch are one blob.  A blob that no panel covers
whole, or that only one panel weighs in on, is left to the mean and
counted; where the ramps of two blobs meet, a pixel follows the blob with
the larger ``ρ``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


PROTECT_ALGORITHM = "mosaic-protected-high-band-v1"
LOW_BAND_SIGMA = 16.0
PROTECT_PEAK_SIGMA = 50.0
PROTECT_RAMP_PIXELS = 8.0
MAXIMUM_PROTECT_RADIUS = 128.0
# The low band is a normalized Gaussian on 4x4 bins, interpolated back.
_BINNING = 4
_HALO = int(4 * LOW_BAND_SIGMA)
_WING_BETA = 2.5
# A blob extends over the connected region where some panel's high band
# differs from the chosen panel's by this many σ after a 1.5 px smoothing.
_MISMATCH_SIGMA = 2.0
_MISMATCH_SMOOTHING = 1.5
_STRIP_ROWS = 512
_PEAK_WINDOW = 5

Box = tuple[int, int, int, int]
# Panel key and canvas box -> the panel's matched values (NaN where it has
# none) and its blend weight (0 there).
Sampler = Callable[[str, Box], tuple[NDArray[np.float64], NDArray[np.float64]]]


def low_band(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """The normalized Gaussian low band (σ = ``LOW_BAND_SIGMA``) of the
    finite values, computed on 4x4 bins and interpolated back bilinearly."""

    height, width = values.shape
    rows, columns = -(-height // _BINNING), -(-width // _BINNING)
    valid = np.isfinite(values)
    sums = np.zeros((rows * _BINNING, columns * _BINNING))
    counts = np.zeros_like(sums)
    sums[:height, :width] = np.where(valid, values, 0.0)
    counts[:height, :width] = valid
    shape = (rows, _BINNING, columns, _BINNING)
    sigma = LOW_BAND_SIGMA / _BINNING
    numerator = ndimage.gaussian_filter(sums.reshape(shape).sum(axis=(1, 3)), sigma, mode="constant", truncate=4.0)
    denominator = ndimage.gaussian_filter(counts.reshape(shape).sum(axis=(1, 3)), sigma, mode="constant", truncate=4.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        binned = np.where(denominator > 1e-6, numerator / denominator, np.nan)

    def axis(size: int, bins: int) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        position = np.clip((np.arange(size) + 0.5) / _BINNING - 0.5, 0.0, bins - 1.0)
        lower = np.minimum(np.floor(position).astype(np.int64), max(bins - 2, 0))
        upper = np.minimum(lower + 1, bins - 1)
        return lower, upper, position - lower

    top, bottom, fy = axis(height, rows)
    left, right, fx = axis(width, columns)
    by_rows = binned[top] * (1.0 - fy)[:, None] + binned[bottom] * fy[:, None]
    return by_rows[:, left] * (1.0 - fx)[None, :] + by_rows[:, right] * fx[None, :]


def high_band(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """The values minus their low band (NaN where the values are)."""

    return values - low_band(values)


@dataclass(frozen=True)
class ProtectedBlob:
    """One blob of the protected set, on its canvas box."""

    box: Box
    panel: str
    peak_sigma: float
    # The radius of a disk of the footprint's area.
    radius: float
    # ρ over the box, and per other panel HP(J_panel − J_other) (0 where
    # undefined).
    ramp: NDArray[np.float32]
    high_bands: Mapping[str, NDArray[np.float32]]

    def serializable(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "panel": self.panel,
            "peakSigma": self.peak_sigma,
            "footprintRadiusPixels": self.radius,
            "protectedPixels": int(np.count_nonzero(self.ramp >= 1.0)),
        }


@dataclass(frozen=True)
class _Seed:
    x: int
    y: int
    peak: float
    fwhm: float
    # The footprint: a mask over its canvas box.
    box: Box
    footprint: NDArray[np.bool_]


def _wing_radius(peak: float, fwhm: float) -> float:
    """Where a Moffat (β = 2.5) star of this peak (σ) fades to 1 σ."""

    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / _WING_BETA) - 1.0))
    wing = alpha * max(peak, 1.0) ** (1.0 / (2.0 * _WING_BETA))
    return float(np.clip(wing, 2.0 * fwhm, MAXIMUM_PROTECT_RADIUS / 2.0))


def _intersection(a: Box, b: Box) -> Box | None:
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[0] < box[2] and box[1] < box[3] else None


def _disk(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> NDArray[np.bool_]:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius * radius


def _seeds(
    keys: Sequence[str],
    boxes: Mapping[str, Box],
    sigma: Mapping[str, float],
    fwhm: Mapping[str, float],
    sample: Sampler,
) -> list[_Seed]:
    """Local maxima above ``PROTECT_PEAK_SIGMA`` of the panels' high bands
    where two panels overlap, each with its footprint."""

    halo = _HALO + int(MAXIMUM_PROTECT_RADIUS)
    seeds: list[_Seed] = []
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            overlap = _intersection(boxes[left], boxes[right])
            if overlap is None:
                continue
            x0, y0, x1, y1 = overlap
            width = max(fwhm[left], fwhm[right])
            noise = math.hypot(sigma[left], sigma[right]) / (2.0 * math.sqrt(math.pi) * _MISMATCH_SMOOTHING)
            for top in range(y0, y1, _STRIP_ROWS):
                bottom = min(y1, top + _STRIP_ROWS)
                read = (x0 - halo, top - halo, x1 + halo, bottom + halo)
                values = [sample(key, read)[0] for key in (left, right)]
                significance = [high_band(value) / sigma[key] for value, key in zip(values, (left, right))]
                held = np.isfinite(significance[0]) & np.isfinite(significance[1])
                peak = np.where(held, np.fmax(significance[0], significance[1]), -np.inf)
                maxima = (peak >= PROTECT_PEAK_SIGMA) & (
                    peak == ndimage.maximum_filter(peak, size=_PEAK_WINDOW, mode="constant", cval=-np.inf)
                )
                maxima[: halo] = maxima[halo + bottom - top :] = False
                maxima[:, : halo] = maxima[:, halo + x1 - x0 :] = False
                if not maxima.any():
                    continue
                difference = high_band(values[0] - values[1])
                smoothed = ndimage.gaussian_filter(np.where(held, difference, 0.0), _MISMATCH_SMOOTHING)
                labels, _ = ndimage.label(held & (np.abs(smoothed) > _MISMATCH_SIGMA * noise))
                for row, column in zip(*np.nonzero(maxima)):
                    seed_peak = float(peak[row, column])
                    reach = int(math.ceil(2.0 * _wing_radius(seed_peak, width)))
                    window = (
                        slice(max(0, row - reach), min(peak.shape[0], row + reach + 1)),
                        slice(max(0, column - reach), min(peak.shape[1], column + reach + 1)),
                    )
                    centre = (column - window[1].start, row - window[0].start)
                    shape = (window[0].stop - window[0].start, window[1].stop - window[1].start)
                    core = _disk(shape, centre, 2.0 * width)
                    local = labels[window]
                    touching = np.unique(local[core & (local > 0)])
                    footprint = (core | np.isin(local, touching[touching > 0])) & _disk(shape, centre, reach)
                    rows, columns = np.nonzero(footprint)
                    r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(columns.min()), int(columns.max()) + 1
                    origin = (read[0] + int(window[1].start), read[1] + int(window[0].start))
                    seeds.append(
                        _Seed(
                            int(column) + read[0],
                            int(row) + read[1],
                            seed_peak,
                            width,
                            (origin[0] + c0, origin[1] + r0, origin[0] + c1, origin[1] + r1),
                            footprint[r0:r1, c0:c1],
                        )
                    )
    return seeds


def _touching(a: _Seed, b: _Seed) -> bool:
    """Whether two footprints overlap or are adjacent."""

    grown = (a.box[0] - 1, a.box[1] - 1, a.box[2] + 1, a.box[3] + 1)
    shared = _intersection(grown, b.box)
    if shared is None:
        return False
    mask_a = np.pad(a.footprint, 1)
    mask_a = ndimage.binary_dilation(mask_a, structure=np.ones((3, 3), dtype=bool))
    region_a = mask_a[shared[1] - grown[1] : shared[3] - grown[1], shared[0] - grown[0] : shared[2] - grown[0]]
    region_b = b.footprint[shared[1] - b.box[1] : shared[3] - b.box[1], shared[0] - b.box[0] : shared[2] - b.box[0]]
    return bool((region_a & region_b).any())


def _clusters(seeds: Sequence[_Seed]) -> list[list[_Seed]]:
    """Seeds whose footprints touch."""

    if not seeds:
        return []
    centres = np.asarray([((s.box[0] + s.box[2]) / 2.0, (s.box[1] + s.box[3]) / 2.0) for s in seeds])
    extents = np.asarray([math.hypot(s.box[2] - s.box[0], s.box[3] - s.box[1]) / 2.0 + 1.0 for s in seeds])
    candidates = cKDTree(centres).query_pairs(r=2.0 * float(extents.max()), output_type="ndarray")
    pairs = [(i, j) for i, j in candidates if _touching(seeds[i], seeds[j])]
    rows = np.asarray([i for i, _ in pairs], dtype=np.int64)
    columns = np.asarray([j for _, j in pairs], dtype=np.int64)
    graph = coo_matrix((np.ones(len(pairs)), (rows, columns)), shape=(len(seeds), len(seeds)))
    count, labels = connected_components(graph, directed=False)
    clusters: list[list[_Seed]] = [[] for _ in range(count)]
    for seed, label in zip(seeds, labels):
        clusters[label].append(seed)
    return clusters


def _blob(
    cluster: Sequence[_Seed],
    keys: Sequence[str],
    boxes: Mapping[str, Box],
    sample: Sampler,
    canvas_box: Box,
) -> ProtectedBlob | None:
    ramp_pixels = int(math.ceil(PROTECT_RAMP_PIXELS))
    box = (
        max(canvas_box[0], min(seed.box[0] for seed in cluster) - ramp_pixels),
        max(canvas_box[1], min(seed.box[1] for seed in cluster) - ramp_pixels),
        min(canvas_box[2], max(seed.box[2] for seed in cluster) + ramp_pixels),
        min(canvas_box[3], max(seed.box[3] for seed in cluster) + ramp_pixels),
    )
    footprint = np.zeros((box[3] - box[1], box[2] - box[0]), dtype=bool)
    for seed in cluster:
        shared = _intersection(seed.box, box)
        if shared is not None:
            footprint[shared[1] - box[1] : shared[3] - box[1], shared[0] - box[0] : shared[2] - box[0]] |= seed.footprint[
                shared[1] - seed.box[1] : shared[3] - seed.box[1], shared[0] - seed.box[0] : shared[2] - seed.box[0]
            ]
    distance = ndimage.distance_transform_edt(~footprint)
    read = (box[0] - _HALO, box[1] - _HALO, box[2] + _HALO, box[3] + _HALO)
    inner = (slice(_HALO, _HALO + box[3] - box[1]), slice(_HALO, _HALO + box[2] - box[0]))
    samples = {key: sample(key, read) for key in keys if _intersection(boxes[key], box) is not None}
    # A panel that weighs in on the whole footprint and its ramp, else on
    # the footprint (its ramp then ends where its weight does).
    covering = [
        key for key, (_, weight) in samples.items() if (weight[inner][distance <= PROTECT_RAMP_PIXELS] > 0).all()
    ] or [key for key, (_, weight) in samples.items() if (weight[inner][footprint] > 0).all()]
    if not covering:
        return None
    brightest = max(cluster, key=lambda seed: seed.peak)
    at_peak = (brightest.y - read[1], brightest.x - read[0])
    chosen = max(covering, key=lambda key: (float(samples[key][1][at_peak]), key))
    values_chosen = samples[chosen][0]
    high_bands: dict[str, NDArray[np.float32]] = {}
    for key, (values, weight) in samples.items():
        if key == chosen or not (weight[inner] > 0).any():
            continue
        band = high_band(values_chosen - values)[inner]
        high_bands[key] = np.where(np.isfinite(band), band, 0.0).astype(np.float32)
    if not high_bands:
        return None
    ramp = np.clip(1.0 - distance / PROTECT_RAMP_PIXELS, 0.0, 1.0)
    # Where the chosen panel has no weight, the blend stays as it is: a
    # pixel that one panel alone covers is that panel's.
    ramp[~(samples[chosen][1][inner] > 0)] = 0.0
    return ProtectedBlob(
        box=box,
        panel=chosen,
        peak_sigma=brightest.peak,
        radius=math.sqrt(float(footprint.sum()) / math.pi),
        ramp=ramp.astype(np.float32),
        high_bands=high_bands,
    )


def protected_set(
    keys: Sequence[str],
    boxes: Mapping[str, Box],
    sigma: Mapping[str, float],
    fwhm: Mapping[str, float],
    sample: Sampler,
    canvas_box: Box,
) -> tuple[list[ProtectedBlob], dict[str, Any]]:
    """The blobs of the protected set and its summary for the receipt.

    ``boxes`` are the panels' canvas boxes, ``sigma`` their pixel noise in
    matched units and ``fwhm`` their seeing (pixels)."""

    seeds = _seeds(keys, boxes, sigma, fwhm, sample)
    clusters = _clusters(seeds)
    blobs: list[ProtectedBlob] = []
    left_to_the_mean = 0
    for cluster in clusters:
        blob = _blob(cluster, keys, boxes, sample, canvas_box)
        if blob is None:
            left_to_the_mean += 1
        else:
            blobs.append(blob)
    by_panel: dict[str, int] = {}
    for blob in blobs:
        by_panel[blob.panel] = by_panel.get(blob.panel, 0) + 1
    brightest = sorted(blobs, key=lambda blob: -blob.peak_sigma)[:20]
    return blobs, {
        "algorithm": PROTECT_ALGORITHM,
        "lowBandSigmaPixels": LOW_BAND_SIGMA,
        "peakThresholdSigma": PROTECT_PEAK_SIGMA,
        "rampPixels": PROTECT_RAMP_PIXELS,
        "seeds": len(seeds),
        "blobs": len(blobs),
        "blobsLeftToTheMean": left_to_the_mean,
        "blobsByPanel": dict(sorted(by_panel.items())),
        "brightest": [blob.serializable() for blob in brightest],
    }


__all__ = ["PROTECT_ALGORITHM", "ProtectedBlob", "high_band", "low_band", "protected_set"]
