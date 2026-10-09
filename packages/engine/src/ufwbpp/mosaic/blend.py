"""Blending the matched panels of one filter into the mosaic.

Every output pixel is the weighted mean of the scaled, plane-corrected
panels that cover it, except on the protected set (``protect.py``): there a
bright star or galaxy core takes its small scales from one panel.  A
panel's weight is the product of

* its inverse variance (the panel's noise, scaled with its photometric
  factor, and the number of frames that reached the pixel), so overlaps get
  the signal-to-noise gain of both panels;
* an edge taper ``min(1, d/D)`` of the distance ``d`` to the end of the
  panel's valid area, so a panel fades in over ``D`` pixels instead of
  starting with a step (``D`` is at most 256 px and at most 0.4 of the
  narrowest overlap of the panel);
* its validity (finite and covered by at least half of its frames).

Off the protected set a weighted mean of the panels never leaves their
range.  The output stays linear and the noise of every pixel is propagated
into the NOISE plane.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..image_io.fits import FitsFloatWriter
from .photometry import PanelImage, PlaneSolution, ScaleSolution
from .protect import Box, ProtectedBlob, protected_set


BLEND_ALGORITHM = "mosaic-inverse-variance-taper-blend-v2-protected-high-band"
MAXIMUM_TAPER_PIXELS = 256
TAPER_OVERLAP_FRACTION = 0.4
_TAPER_BINNING = 4
BAND_ROWS = 512
MASK_NO_COVERAGE = 1
MASK_SINGLE_PANEL = 2
MASK_PROTECTED = 4


@dataclass(frozen=True)
class _PanelWeights:
    panel: PanelImage
    scale: float
    noise: float
    median_count: float
    taper: NDArray[np.float32]
    taper_length: float


def panel_noise(panel: PanelImage) -> float:
    """Background noise of a master: SEP's global background RMS."""

    import sep

    values = panel.read(*panel.box)
    finite = np.isfinite(values)
    work = np.ascontiguousarray(np.where(finite, values, np.nanmedian(values)), dtype=np.float32)
    background = sep.Background(work, mask=~finite, bw=64, bh=64)
    return float(background.globalrms)


def _taper_map(panel: PanelImage) -> NDArray[np.float32]:
    """Distance (pixels) to the end of the valid area, on a 4x binned grid."""

    values = panel.read(*panel.box)
    valid = np.isfinite(values)
    height, width = valid.shape
    rows, columns = math.ceil(height / _TAPER_BINNING), math.ceil(width / _TAPER_BINNING)
    padded = np.zeros((rows * _TAPER_BINNING, columns * _TAPER_BINNING), dtype=bool)
    padded[:height, :width] = valid
    binned = padded.reshape(rows, _TAPER_BINNING, columns, _TAPER_BINNING).all(axis=(1, 3))
    distance = ndimage.distance_transform_edt(np.pad(binned, 1, constant_values=False))[1:-1, 1:-1]
    return (distance * _TAPER_BINNING).astype(np.float32)


def _taper_rows(weights: _PanelWeights, y0: int, y1: int, x0: int, x1: int) -> NDArray[np.float32]:
    """The taper of window rows [y0, y1), columns [x0, x1), at full
    resolution."""

    row_index = np.minimum(np.arange(y0, y1) // _TAPER_BINNING, weights.taper.shape[0] - 1)
    column_index = np.minimum(np.arange(x0, x1) // _TAPER_BINNING, weights.taper.shape[1] - 1)
    distance = weights.taper[row_index[:, None], column_index[None, :]]
    return np.minimum(1.0, distance / np.float32(weights.taper_length)).astype(np.float32)


def _panel_values(
    weights: _PanelWeights, planes: PlaneSolution, box: Box, opened: Mapping[Path, Any] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Matched values, blend weight and pixel variance of one panel over a
    canvas box inside it; the weight is 0 where the panel has no valid
    sample."""

    left, top, right, bottom = box
    px0, py0, _, _ = weights.panel.box
    values = weights.panel.read(left, top, right, bottom, opened=opened).astype(np.float64)
    valid = np.isfinite(values)
    corrected = weights.scale * values - planes.evaluate(
        weights.panel.key,
        np.arange(left, right, dtype=np.float64)[None, :],
        np.arange(top, bottom, dtype=np.float64)[:, None],
    )
    taper = _taper_rows(weights, top - py0, bottom - py0, left - px0, right - px0)
    counts = weights.panel.counts(left, top, right, bottom, opened=opened)
    depth = (
        np.where(np.isfinite(counts) & (counts > 0), counts, 0.0) / weights.median_count
        if counts is not None
        else np.ones_like(values)
    )
    pixel_variance = np.square(weights.scale * weights.noise) / np.maximum(depth, 1e-6)
    weight = np.where(valid & (depth > 0), taper.astype(np.float64) / pixel_variance, 0.0)
    return corrected, weight, pixel_variance


def _sampler(prepared: Sequence[_PanelWeights], planes: PlaneSolution, opened: Mapping[Path, Any]) -> Any:
    """A panel's matched values and blend weight over any canvas box (NaN
    and 0 outside the panel), for the protected set; ``opened`` holds the
    panels' open planes."""

    by_key = {weights.panel.key: weights for weights in prepared}

    def sample(key: str, box: Box) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        weights = by_key[key]
        values = np.full((box[3] - box[1], box[2] - box[0]), np.nan)
        weight = np.zeros_like(values)
        px0, py0, px1, py1 = weights.panel.box
        inner = (max(box[0], px0), max(box[1], py0), min(box[2], px1), min(box[3], py1))
        if inner[0] < inner[2] and inner[1] < inner[3]:
            corrected, panel_weight, _ = _panel_values(weights, planes, inner, opened)
            region = (slice(inner[1] - box[1], inner[3] - box[1]), slice(inner[0] - box[0], inner[2] - box[0]))
            values[region] = np.where(panel_weight > 0, corrected, np.nan)
            weight[region] = panel_weight
        return values, weight

    return sample


def _owners(blobs: Sequence[ProtectedBlob], band_start: int, band_end: int, x0: int, width: int) -> NDArray[np.int64]:
    """Per band pixel, the blob with the largest ρ (-1 for none): where two
    blobs' ramps meet, a pixel follows one of them."""

    best = np.zeros((band_end - band_start, width))
    owner = np.full(best.shape, -1, dtype=np.int64)
    for index, blob in enumerate(blobs):
        shared = _overlap(blob.box, (x0, band_start, x0 + width, band_end))
        if shared is None:
            continue
        sx0, sy0, sx1, sy1 = shared
        band = (slice(sy0 - band_start, sy1 - band_start), slice(sx0 - x0, sx1 - x0))
        ramp = blob.ramp[sy0 - blob.box[1] : sy1 - blob.box[1], sx0 - blob.box[0] : sx1 - blob.box[0]]
        better = ramp > best[band]
        best[band] = np.where(better, ramp, best[band])
        owner[band] = np.where(better, index, owner[band])
    return owner


def _accumulate_protection(
    blob: ProtectedBlob,
    index: int,
    key: str,
    region: Box,
    weight: NDArray[np.float64],
    pixel_variance: NDArray[np.float64],
    band_start: int,
    x0: int,
    owner: NDArray[np.int64],
    correction: NDArray[np.float64],
    rho: NDArray[np.float64],
    chosen_weight: NDArray[np.float64],
    chosen_variance: NDArray[np.float64],
) -> None:
    """Add one panel's share of a blob's correction to the band."""

    shared = _overlap(blob.box, region)
    if shared is None or (key != blob.panel and key not in blob.high_bands):
        return
    sx0, sy0, sx1, sy1 = shared
    band = (slice(sy0 - band_start, sy1 - band_start), slice(sx0 - x0, sx1 - x0))
    panel = (slice(sy0 - region[1], sy1 - region[1]), slice(sx0 - region[0], sx1 - region[0]))
    own = (slice(sy0 - blob.box[1], sy1 - blob.box[1]), slice(sx0 - blob.box[0], sx1 - blob.box[0]))
    ramp = np.where(owner[band] == index, blob.ramp[own], 0.0)
    if key == blob.panel:
        inside = ramp > 0
        rho[band] = np.where(inside, ramp, rho[band])
        chosen_weight[band] = np.where(inside, weight[panel], chosen_weight[band])
        chosen_variance[band] = np.where(inside, pixel_variance[panel], chosen_variance[band])
    else:
        correction[band] += ramp * weight[panel] * blob.high_bands[key][own]


def _overlap(a: Box, b: Box) -> Box | None:
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[0] < box[2] and box[1] < box[3] else None


def blend_filter(
    panels: Sequence[PanelImage],
    scales: ScaleSolution,
    planes: PlaneSolution,
    *,
    noises: Mapping[str, float],
    fwhm: Mapping[str, float],
    overlap_widths: Mapping[str, float],
    canvas_box: tuple[int, int, int, int],
    header: Mapping[str, Any],
    science_path: Path,
    noise_path: Path,
    coverage_path: Path,
    mask_path: Path,
    durable: bool = True,
) -> dict[str, Any]:
    """Blend one filter's panels onto the canvas box (x0, y0, x1, y1);
    ``fwhm`` is every panel's seeing in pixels."""

    x0, y0, x1, y1 = canvas_box
    width, height = x1 - x0, y1 - y0
    prepared = []
    for panel in panels:
        counts = panel.counts(*panel.box)
        median_count = float(np.nanmedian(counts)) if counts is not None and np.isfinite(counts).any() else 1.0
        length = min(float(MAXIMUM_TAPER_PIXELS), TAPER_OVERLAP_FRACTION * float(overlap_widths.get(panel.key, math.inf)))
        prepared.append(
            _PanelWeights(
                panel=panel,
                scale=scales.scales[panel.key],
                noise=float(noises[panel.key]),
                median_count=max(median_count, 1.0),
                taper=_taper_map(panel),
                taper_length=max(length, 1.0),
            )
        )
    # The protected set reads many small boxes: every plane is opened once
    # and closed before anything else touches the files.
    opened = {path: fits.open(path, mode="readonly", memmap=True) for panel in panels for path in panel.planes}
    try:
        blobs, protection = protected_set(
            [weights.panel.key for weights in prepared],
            {weights.panel.key: weights.panel.box for weights in prepared},
            {weights.panel.key: weights.scale * weights.noise for weights in prepared},
            fwhm,
            _sampler(prepared, planes, opened),
            canvas_box,
        )
    finally:
        for hdul in opened.values():
            hdul.close()
        opened.clear()
    statistics = {
        "finitePixels": 0,
        "singlePanelPixels": 0,
        "overlapPixels": 0,
        "uncoveredPixels": 0,
        "protectedPixels": 0,
    }
    science_header = dict(header)
    with (
        FitsFloatWriter(science_path, (height, width), science_header, durable=durable) as science,
        FitsFloatWriter(noise_path, (height, width), {**header, "OAFPLANE": "NOISE"}, durable=durable) as noise_writer,
        FitsFloatWriter(coverage_path, (height, width), {**header, "OAFPLANE": "COVERAGE"}, durable=durable) as coverage_writer,
        FitsFloatWriter(mask_path, (height, width), {**header, "OAFPLANE": "MASK"}, durable=durable) as mask_writer,
    ):
        for band_start in range(y0, y1, BAND_ROWS):
            band_end = min(y1, band_start + BAND_ROWS)
            rows = band_end - band_start
            numerator = np.zeros((rows, width), dtype=np.float64)
            denominator = np.zeros((rows, width), dtype=np.float64)
            variance = np.zeros((rows, width), dtype=np.float64)
            count = np.zeros((rows, width), dtype=np.float32)
            active = [blob for blob in blobs if blob.box[1] < band_end and blob.box[3] > band_start]
            # The protected set: ρ Σ w_k HP(J_a − J_k), and the chosen
            # panel's weight and variance for the noise there.
            correction = np.zeros((rows, width), dtype=np.float64) if active else None
            rho = np.zeros((rows, width), dtype=np.float64) if active else None
            chosen_weight = np.zeros((rows, width), dtype=np.float64) if active else None
            chosen_variance = np.zeros((rows, width), dtype=np.float64) if active else None
            owner = _owners(active, band_start, band_end, x0, width) if active else None
            for weights in prepared:
                px0, py0, px1, py1 = weights.panel.box
                top, bottom = max(band_start, py0), min(band_end, py1)
                left, right = max(x0, px0), min(x1, px1)
                if top >= bottom or left >= right:
                    continue
                corrected, weight, pixel_variance = _panel_values(weights, planes, (left, top, right, bottom))
                if not (weight > 0).any():
                    continue
                sl = (slice(top - band_start, bottom - band_start), slice(left - x0, right - x0))
                numerator[sl] += np.where(weight > 0, weight * corrected, 0.0)
                denominator[sl] += weight
                variance[sl] += np.square(weight) * pixel_variance
                count[sl] += (weight > 0).astype(np.float32)
                for index, blob in enumerate(active):
                    _accumulate_protection(
                        blob, index, weights.panel.key, (left, top, right, bottom), weight, pixel_variance,
                        band_start, x0, owner, correction, rho, chosen_weight, chosen_variance,
                    )
            covered = denominator > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                if correction is None:
                    band_science = np.where(covered, numerator / denominator, np.nan).astype(np.float32)
                    band_noise = np.where(covered, np.sqrt(variance) / denominator, np.nan).astype(np.float32)
                else:
                    band_science = np.where(covered, (numerator + correction) / denominator, np.nan).astype(np.float32)
                    # Small scales from the chosen panel alone where ρ = 1.
                    noise_squared = (
                        np.square(1.0 - rho) * variance / np.square(denominator)
                        + 2.0 * rho * (1.0 - rho) * chosen_weight * chosen_variance / denominator
                        + np.square(rho) * chosen_variance
                    )
                    band_noise = np.where(
                        covered,
                        np.where(rho > 0, np.sqrt(noise_squared), np.sqrt(variance) / denominator),
                        np.nan,
                    ).astype(np.float32)
            protected = covered & (rho > 0) if rho is not None else np.zeros_like(covered)
            mask = (
                np.where(covered, 0, MASK_NO_COVERAGE)
                + np.where(count == 1, MASK_SINGLE_PANEL, 0)
                + np.where(protected, MASK_PROTECTED, 0)
            )
            science.write_rows(band_start - y0, band_science)
            noise_writer.write_rows(band_start - y0, band_noise)
            coverage_writer.write_rows(band_start - y0, count)
            mask_writer.write_rows(band_start - y0, mask.astype(np.float32))
            statistics["finitePixels"] += int(covered.sum())
            statistics["singlePanelPixels"] += int((count == 1).sum())
            statistics["overlapPixels"] += int((count >= 2).sum())
            statistics["uncoveredPixels"] += int((~covered).sum())
            statistics["protectedPixels"] += int(protected.sum())
    return {
        "algorithm": BLEND_ALGORITHM,
        "taperPixels": {weights.panel.key: weights.taper_length for weights in prepared},
        "noise": {weights.panel.key: weights.noise for weights in prepared},
        "protectedSet": protection,
        "sha256": {
            "science": science.sha256,
            "noise": noise_writer.sha256,
            "coverage": coverage_writer.sha256,
            "mask": mask_writer.sha256,
        },
        **statistics,
    }


__all__ = ["BLEND_ALGORITHM", "blend_filter", "panel_noise"]
