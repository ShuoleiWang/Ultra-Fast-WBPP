"""Blending the matched panels of one filter into the mosaic.

Every output pixel is the weighted mean of the scaled, plane-corrected
panels that cover it.  A panel's weight is the product of

* its inverse variance (the panel's noise, scaled with its photometric
  factor, and the number of frames that reached the pixel), so overlaps get
  the signal-to-noise gain of both panels;
* an edge taper ``min(1, d/D)`` of the distance ``d`` to the end of the
  panel's valid area, so a panel fades in over ``D`` pixels instead of
  starting with a step (``D`` is at most 256 px and at most 0.4 of the
  narrowest overlap of the panel);
* its validity (finite and covered by at least half of its frames).

A weighted mean of the panels never leaves their range, the output stays
linear and the noise of every pixel is propagated into the NOISE plane.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..image_io.fits import FitsFloatWriter
from .photometry import PanelImage, PlaneSolution, ScaleSolution


BLEND_ALGORITHM = "mosaic-inverse-variance-taper-blend-v1"
MAXIMUM_TAPER_PIXELS = 256
TAPER_OVERLAP_FRACTION = 0.4
_TAPER_BINNING = 4
BAND_ROWS = 512
MASK_NO_COVERAGE = 1
MASK_SINGLE_PANEL = 2


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


def _taper_rows(weights: _PanelWeights, y0: int, y1: int) -> NDArray[np.float32]:
    """The taper of window rows [y0, y1) at full resolution."""

    height, width = weights.panel.shape
    row_index = np.minimum(np.arange(y0, y1) // _TAPER_BINNING, weights.taper.shape[0] - 1)
    column_index = np.minimum(np.arange(width) // _TAPER_BINNING, weights.taper.shape[1] - 1)
    distance = weights.taper[row_index[:, None], column_index[None, :]]
    return np.minimum(1.0, distance / np.float32(weights.taper_length)).astype(np.float32)


def blend_filter(
    panels: Sequence[PanelImage],
    scales: ScaleSolution,
    planes: PlaneSolution,
    *,
    noises: Mapping[str, float],
    overlap_widths: Mapping[str, float],
    canvas_box: tuple[int, int, int, int],
    header: Mapping[str, Any],
    science_path: Path,
    noise_path: Path,
    coverage_path: Path,
    mask_path: Path,
    durable: bool = True,
) -> dict[str, Any]:
    """Blend one filter's panels onto the canvas box (x0, y0, x1, y1)."""

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
    statistics = {"finitePixels": 0, "singlePanelPixels": 0, "overlapPixels": 0, "uncoveredPixels": 0}
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
            for weights in prepared:
                px0, py0, px1, py1 = weights.panel.box
                top, bottom = max(band_start, py0), min(band_end, py1)
                left, right = max(x0, px0), min(x1, px1)
                if top >= bottom or left >= right:
                    continue
                values = weights.panel.read(left, top, right, bottom).astype(np.float64)
                valid = np.isfinite(values)
                if not valid.any():
                    continue
                corrected = weights.scale * values - planes.evaluate(
                    weights.panel.key,
                    np.arange(left, right, dtype=np.float64)[None, :],
                    np.arange(top, bottom, dtype=np.float64)[:, None],
                )
                taper = _taper_rows(weights, top - py0, bottom - py0)[:, left - px0 : right - px0]
                counts = weights.panel.counts(left, top, right, bottom)
                depth = (
                    np.where(np.isfinite(counts) & (counts > 0), counts, 0.0) / weights.median_count
                    if counts is not None
                    else np.ones_like(values)
                )
                pixel_variance = np.square(weights.scale * weights.noise) / np.maximum(depth, 1e-6)
                weight = np.where(valid & (depth > 0), taper.astype(np.float64) / pixel_variance, 0.0)
                sl = (slice(top - band_start, bottom - band_start), slice(left - x0, right - x0))
                numerator[sl] += np.where(weight > 0, weight * corrected, 0.0)
                denominator[sl] += weight
                variance[sl] += np.square(weight) * pixel_variance
                count[sl] += (weight > 0).astype(np.float32)
            covered = denominator > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                band_science = np.where(covered, numerator / denominator, np.nan).astype(np.float32)
                band_noise = np.where(covered, np.sqrt(variance) / denominator, np.nan).astype(np.float32)
            mask = np.where(covered, 0, MASK_NO_COVERAGE) + np.where(count == 1, MASK_SINGLE_PANEL, 0)
            science.write_rows(band_start - y0, band_science)
            noise_writer.write_rows(band_start - y0, band_noise)
            coverage_writer.write_rows(band_start - y0, count)
            mask_writer.write_rows(band_start - y0, mask.astype(np.float32))
            statistics["finitePixels"] += int(covered.sum())
            statistics["singlePanelPixels"] += int((count == 1).sum())
            statistics["overlapPixels"] += int((count >= 2).sum())
            statistics["uncoveredPixels"] += int((~covered).sum())
    return {
        "algorithm": BLEND_ALGORITHM,
        "taperPixels": {weights.panel.key: weights.taper_length for weights in prepared},
        "noise": {weights.panel.key: weights.noise for weights in prepared},
        "sha256": {
            "science": science.sha256,
            "noise": noise_writer.sha256,
            "coverage": coverage_writer.sha256,
            "mask": mask_writer.sha256,
        },
        **statistics,
    }


__all__ = ["BLEND_ALGORITHM", "blend_filter", "panel_noise"]
