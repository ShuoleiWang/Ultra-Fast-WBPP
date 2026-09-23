"""Display-only Blink diagnostics; never used for science pixels.

The detail panel and background panel are complementary, not interchangeable:
removing the smooth background in the detail panel is safe for inspection only
when the background difference is also shown. All display scales belong to a
channel reference, not to individual frames. No frame is automatically dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy import ndimage

DISPLAY_ALGORITHM = "blink-complementary-display-v2"
BACKGROUND_CELL = 48
BACKGROUND_LEVEL = 0.22


def local_noise(image: np.ndarray) -> float:
    """MAD of disjoint diagonal Haar differences, insensitive to a sky plane.

    (a-b-c+d)/2 has the same variance as independent white input noise. This
    is a display scale estimate, not a calibrated uncertainty after resampling.
    Measure it on the unwarped preview, before interpolation changes covariance.
    """
    image = np.asarray(image, dtype=np.float32)
    rows, columns = image.shape
    data = image[: rows // 2 * 2, : columns // 2 * 2]
    difference = (data[::2, ::2] - data[::2, 1::2] - data[1::2, ::2] + data[1::2, 1::2]) / 2
    finite = difference[np.isfinite(difference)]
    if finite.size < 32:
        raise ValueError("insufficient finite pixels for the display noise scale")
    sigma = float(1.4826 * np.median(np.abs(finite - np.median(finite))))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("constant or quantized preview has no measurable display noise")
    return sigma


def background_grid(image: np.ndarray, cell: int = BACKGROUND_CELL) -> tuple[np.ndarray, np.ndarray]:
    """Robust cell medians and measured coverage; stars do not set the scale.

    Cells with less than half finite coverage remain unknown. Interpolation is
    for the displayed background/detail split only, never missing-data repair.
    """
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    rows, columns = math.ceil(h / cell), math.ceil(w / cell)
    padded = np.full((rows * cell, columns * cell), np.nan, np.float32)
    padded[:h, :w] = image
    blocks = padded.reshape(rows, cell, columns, cell).transpose(0, 2, 1, 3).reshape(rows, columns, -1)
    valid = np.sum(np.isfinite(blocks), axis=-1) >= cell * cell / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grid = np.nanmedian(blocks, axis=-1)
    grid[~valid] = np.nan
    return grid, valid


def background_surface(image: np.ndarray, cell: int = BACKGROUND_CELL) -> np.ndarray:
    grid, valid = background_grid(image, cell)
    if not np.any(valid):
        raise ValueError("insufficient sky coverage for background inspection")
    # Nearest supported cells supply interpolation at frame borders. The source
    # mask is restored below, so the filled grid cannot fabricate sky coverage.
    if not np.all(valid):
        nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        grid = grid[tuple(nearest)]
    y = (np.arange(image.shape[0], dtype=np.float32) + 0.5) / cell - 0.5
    x = (np.arange(image.shape[1], dtype=np.float32) + 0.5) / cell - 0.5
    surface = ndimage.map_coordinates(grid, np.meshgrid(y, x, indexing="ij"), order=1, mode="nearest")
    surface[~np.isfinite(image)] = np.nan
    return np.asarray(surface, dtype=np.float32)


def detail_transfer(signal: np.ndarray, reference_noise: float) -> np.ndarray:
    """Shared monotone soft-shoulder curve, without a finite white clip point.

    Zero signal maps to 22% display level. A frame's transparency and noise are
    not normalized away. No denoising, sharpening or histogram equalization.
    """
    if not math.isfinite(reference_noise) or reference_noise <= 0:
        raise ValueError("reference noise must be finite and positive")
    value = np.asarray(signal, dtype=np.float32) / np.float32(reference_noise)
    positive = np.arcsinh(np.maximum(value, 0) / 2)
    rendered = BACKGROUND_LEVEL + (1 - BACKGROUND_LEVEL) * positive / (positive + 1.5)
    negative = BACKGROUND_LEVEL * np.exp(np.minimum(value, 0) / 3)
    rendered = np.where(value < 0, negative, rendered)
    return np.asarray(np.nan_to_num(rendered, nan=0), dtype=np.float32)


def background_transfer(residual: np.ndarray, reference_noise: float) -> np.ndarray:
    """Signed background differences: orange positive, blue negative, grey zero.

    The fixed scale is in reference preview noise units, NOT cell sigma or a
    statistical significance. Unmeasured cells are dark, never neutral grey.
    """
    if not math.isfinite(reference_noise) or reference_noise <= 0:
        raise ValueError("reference noise must be finite and positive")
    value = np.asarray(residual, dtype=np.float32) / np.float32(reference_noise)
    magnitude = 2 / np.pi * np.arctan(np.abs(value) / 2)
    neutral = np.array([0.20, 0.22, 0.25], np.float32)
    positive = np.array([1.0, 0.62, 0.20], np.float32)
    negative = np.array([0.18, 0.65, 1.0], np.float32)
    color = neutral + magnitude[..., None] * (np.where((value >= 0)[..., None], positive, negative) - neutral)
    return np.where(np.isfinite(value)[..., None], color, 0).astype(np.float32)


@dataclass(frozen=True)
class DisplayReference:
    image: np.ndarray
    background: np.ndarray
    noise: float

    @classmethod
    def from_image(cls, image: np.ndarray) -> "DisplayReference":
        return cls(np.asarray(image, np.float32), background_surface(image), local_noise(image))


@dataclass(frozen=True)
class DiagnosticPreview:
    detail: np.ndarray
    field: np.ndarray
    background_difference: np.ndarray | None
    background_rgb: np.ndarray | None
    relative_noise: float
    matched_signal_noise: float | None
    finite_fraction: float


def diagnostic_preview(
    image: np.ndarray,
    reference: DisplayReference,
    *,
    source_noise: float,
    flux_scale: float | None,
    registered: bool,
) -> DiagnosticPreview:
    """Inspect one already-calibrated image on the reference grid.

    Detail: I_i - B_i, same reference curve, NO multiplicative gain.
    Field: I_i - median(I_i), same curve, preserves the full background shape.
    Background: robust cell medians of g_i I_i - I_ref, removing ONLY its
    constant offset. This cancels the fixed astronomical scene without fitting
    away the differential gradient/cloud. It requires verified registration and
    measured positive photometry; an unavailable map is explicit.
    """
    data = np.asarray(image, np.float32)
    if data.shape != reference.image.shape:
        raise ValueError("frame and reference must share the same grid")
    if not math.isfinite(source_noise) or source_noise <= 0:
        raise ValueError("source noise must be finite and positive")
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("frame has no finite pixels")
    detail = detail_transfer(data - background_surface(data), reference.noise)
    field = detail_transfer(data - np.median(data[finite]), reference.noise)
    residual = color = None
    gain_valid = flux_scale is not None and math.isfinite(flux_scale) and flux_scale > 0
    if registered and gain_valid:
        difference = float(flux_scale) * data - reference.image
        residual, covered = background_grid(difference)
        if np.any(covered):
            residual -= np.median(residual[covered])
            color = background_transfer(residual, reference.noise)
        else:
            residual = None
    return DiagnosticPreview(
        detail=detail,
        field=field,
        background_difference=residual,
        background_rgb=color,
        relative_noise=source_noise / reference.noise,
        matched_signal_noise=float(flux_scale) * source_noise / reference.noise if gain_valid else None,
        finite_fraction=float(np.mean(finite)),
    )


def choose_display_reference(frames: list[dict], noise_by_index: dict[int, float]) -> int:
    """Rank the existing measured candidate set with calibrated local noise.

    This is a display-reference experiment, independent of the gate and the
    integration reference. Input flags and human decisions are never changed.
    """
    ranked = []
    for frame in frames:
        metrics = frame["metrics"]
        noise = noise_by_index.get(frame["index"])
        transparency = metrics.get("transparency")
        fwhm = metrics.get("fwhmNative")
        if not frame["score"]["candidate"] or not frame["normalization"]["registered"]:
            continue
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (noise, transparency, fwhm)):
            continue
        roundness = 1 - min(1, max(0, metrics.get("ellipticity") or 0))
        completeness = min(1, max(0, metrics.get("sourceRatio") or 0))
        score = (transparency / noise / fwhm) ** 2 * roundness * completeness
        if score > 0:
            ranked.append((-score, frame["index"]))
    if not ranked:
        raise ValueError("no measured display-reference candidate")
    return min(ranked)[1]
