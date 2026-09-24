"""Registered, photometrically normalized, shared-stretch previews for blinking.

Every frame of a channel is warped onto its blink reference with the
similarity transform the quality analysis already estimated (no new star
extraction), brought to the reference's sky and flux with a linear model
from the measurements, and rendered with one stretch per channel derived
from the reference.  What the user then sees while blinking is exactly the
difference the global normalization will have to absorb: a 30 ADU gradient
in a clean frame is about one sigma at filmstrip scale, a normalized
moonlit gradient several.

Inputs are the float16 linear previews kept by ``measure_paths`` (one read
per frame); outputs are a small grayscale JPEG (or PNG) for the filmstrip
and a PNG at the preview's own scale for zooming.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import math
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .blink_diagnostic_render import DiagnosticFrameSpec
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage

from lightframeqc.parallel import FrameRunner
from lightframeqc.readers import read_frame_preview

FILMSTRIP_FORMATS = ("jpeg", "png")
# The shared screen transfer function.  The shadows are clipped this many
# reference sigmas below the sky and the midtone is solved so the sky lands
# on the target, exactly as a PixInsight-style auto-stretch does; the
# highlight clip only decides where the far highlights saturate, because the
# midtone already fixes the curve around the background.
STRETCH_TARGET = 0.25
STRETCH_HARD_TARGET = 0.45
STRETCH_SHADOWS_SIGMA = -2.8
STRETCH_WHITE_SIGMA = 1000.0


@dataclass(frozen=True, slots=True)
class ChannelStretch:
    """One channel's screen transfer, derived from its blink reference.

    ``black``/``white`` are the clip points in the frames' own units and
    ``shadows``/``target`` the coefficients they came from, so a receipt
    states both what was applied and how it was chosen.
    """

    black: float
    white: float
    midtone: float
    target: float
    shadows: float
    sky_reference: float
    sigma_reference: float
    mode: str = "stf"

    def serializable(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "black": round(self.black, 3),
            "white": round(self.white, 3),
            "shadowsClip": round(self.shadows, 4),
            "midtone": round(self.midtone, 8),
            "target": round(self.target, 4),
            "skyReference": round(self.sky_reference, 3),
            "sigmaReference": round(self.sigma_reference, 3),
        }


def screen_transfer(
    sky: float,
    sigma: float,
    target: float = STRETCH_TARGET,
    shadows: float = STRETCH_SHADOWS_SIGMA,
) -> ChannelStretch:
    """Solve the screen transfer that puts ``sky`` on ``target``.

    The shadows are clipped at ``c0 = sky + shadows * sigma`` and the
    midtone ``m`` is the exact solution of ``MTF(m, x0) = target`` for the
    normalized sky ``x0 = (sky - c0) / (white - c0)``:

        ``m = x0 (target - 1) / (2 target x0 - target - x0)``

    which has no zero denominator for ``0 < target < 1`` and ``0 < x0 < 1``.
    """

    if not math.isfinite(sky):
        sky = 0.0
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = max(abs(sky) * 1e-3, 1e-3)
    if not 0.0 < target < 1.0 or not math.isfinite(shadows) or shadows >= 0.0:
        raise ValueError("screen transfer needs 0 < target < 1 and a negative shadows clip")
    black = sky + shadows * sigma
    white = sky + STRETCH_WHITE_SIGMA * sigma
    x0 = min(max((sky - black) / (white - black), 1e-9), 1.0 - 1e-9)
    midtone = x0 * (target - 1.0) / (2.0 * target * x0 - target - x0)
    return ChannelStretch(
        black=black,
        white=white,
        midtone=min(max(midtone, 1e-6), 1.0 - 1e-6),
        target=target,
        shadows=shadows,
        sky_reference=sky,
        sigma_reference=sigma,
    )


def apply_stf(image: np.ndarray, stretch: ChannelStretch) -> np.ndarray:
    """``MTF(m, x)`` on ``x = clip((v - black) / (white - black), 0, 1)``.

    ``(2m - 1) x - m`` is negative for every ``m`` in ``(0, 1)`` and every
    ``x`` in ``[0, 1]`` (it is ``-m`` at 0 and ``m - 1`` at 1), so the
    transfer has no pole inside the displayed range.
    """

    span = max(stretch.white - stretch.black, 1e-6)
    x = (np.asarray(image, dtype=np.float32) - np.float32(stretch.black)) / np.float32(span)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(x, 0.0, 1.0, out=x)
    midtone = np.float32(stretch.midtone)
    return np.asarray(
        ((midtone - np.float32(1.0)) * x) / ((np.float32(2.0) * midtone - np.float32(1.0)) * x - midtone),
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class PreviewCalibration:
    """Master previews applied before rendering: flat (required) and pedestal.

    A flat alone removes the vignetting that scales with the sky level and
    would otherwise dominate a normalized moonlit frame; the constant
    pedestal it leaves is the same in every frame and does not disturb a
    blink.  A dark or bias, when supplied, removes it too.
    """

    flat_path: str
    pedestal_path: str | None
    long_edge: int

    def serializable(self) -> dict[str, Any]:
        return {
            "flat": Path(self.flat_path).name,
            "pedestal": Path(self.pedestal_path).name if self.pedestal_path else None,
        }


@dataclass(frozen=True, slots=True)
class FramePreviewSpec:
    """Everything one worker needs to render one frame (picklable)."""

    index: int
    linear_path: str
    filmstrip_path: str
    zoom_path: str
    # A second filmstrip image of the same geometry under a harder stretch,
    # so the interface can offer a contrast toggle without re-rendering.
    filmstrip_hard_path: str | None
    # Frame -> blink reference, preview pixels, rows [[a, b, tx], [c, d, ty]];
    # None renders the frame unregistered.
    transform: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    output_shape: tuple[int, int]
    sky: float
    flux_scale: float
    reference_sky: float
    stretch: ChannelStretch
    stretch_hard: ChannelStretch | None = None
    filmstrip_format: str = "jpeg"
    jpeg_quality: int = 85
    filmstrip_divisor: int = 2
    calibration: PreviewCalibration | None = None
    diagnostic: DiagnosticFrameSpec | None = None


@dataclass(frozen=True, slots=True)
class FramePreviewResult:
    index: int
    filmstrip_path: str | None
    zoom_path: str | None
    filmstrip_bytes: int
    zoom_bytes: int
    filmstrip_shape: tuple[int, int]
    zoom_shape: tuple[int, int]
    coverage: float | None
    registered: bool
    filmstrip_hard_path: str | None = None
    filmstrip_hard_bytes: int = 0
    # The sky level the normalization subtracted (calibrated when masters
    # were applied) and whether they were.
    sky: float | None = None
    calibrated: bool = False
    error: str | None = None
    diagnostic_previews: dict[str, str | None] | None = None
    diagnostics: dict[str, Any] | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "filmstripBytes": self.filmstrip_bytes,
            "zoomBytes": self.zoom_bytes,
            "coverage": None if self.coverage is None else round(self.coverage, 4),
            "registered": self.registered,
            "error": self.error,
        }


@lru_cache(maxsize=16)
def _master_preview(path: str, long_edge: int) -> np.ndarray:
    """A master's block-mean preview, loaded once per worker process."""

    preview = read_frame_preview(Path(path), max_long_edge=long_edge)
    return np.asarray(preview.data, dtype=np.float32)


def calibrate_linear(image: np.ndarray, calibration: PreviewCalibration | None) -> tuple[np.ndarray, bool]:
    """``(image - pedestal) / flat`` at preview scale; unchanged without masters.

    Masters in the normalized [0, 1] domain are scaled to the Lights' 16-bit
    range; the flat is normalized to its median so the sky level survives.
    Geometry mismatches leave the frame uncalibrated rather than failing.
    """

    if calibration is None:
        return image, False
    try:
        flat = _master_preview(calibration.flat_path, calibration.long_edge)
    except Exception:
        return image, False
    if flat.shape != image.shape:
        return image, False
    result = np.asarray(image, dtype=np.float32)
    if calibration.pedestal_path is not None:
        try:
            pedestal = _master_preview(calibration.pedestal_path, calibration.long_edge)
        except Exception:
            pedestal = None
        if pedestal is not None and pedestal.shape == image.shape:
            scale = 65535.0 if float(np.nanmax(pedestal)) <= 1.001 and float(np.nanmedian(result)) > 2.0 else 1.0
            result = result - pedestal * np.float32(scale)
    flat_norm = flat / max(float(np.nanmedian(flat)), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = result / np.where(flat_norm > 0.05, flat_norm, np.nan)
    return np.asarray(result, dtype=np.float32), True


def safe_stem(path: str | os.PathLike[str], limit: int = 64) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(path).stem).strip("-._") or "frame"
    return stem[:limit]


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """NaN-aware ``factor``x``factor`` block mean (partial edge blocks kept)."""

    if factor <= 1:
        return image
    rows, columns = image.shape
    out_rows = math.ceil(rows / factor)
    out_columns = math.ceil(columns / factor)
    padded = np.full((out_rows * factor, out_columns * factor), np.nan, dtype=np.float32)
    padded[:rows, :columns] = image
    blocks = padded.reshape(out_rows, factor, out_columns, factor)
    with warnings.catch_warnings():
        # An all-NaN block (outside the warped frame) is NaN, not a warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3)).astype(np.float32)


def robust_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    center = float(np.median(finite))
    return 1.4826 * float(np.median(np.abs(finite - center)))


def channel_statistics(reference_linear: np.ndarray, filmstrip_divisor: int = 2) -> tuple[float, float]:
    """The reference's sky and MADN at the scale the filmstrip is judged at."""

    small = block_mean(np.asarray(reference_linear, dtype=np.float32), filmstrip_divisor)
    finite = small[np.isfinite(small)]
    sky = float(np.median(finite)) if finite.size else 0.0
    sigma = robust_sigma(small)
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = max(abs(sky) * 1e-3, 1e-3)
    return sky, sigma


def channel_stretch(
    reference_linear: np.ndarray,
    filmstrip_divisor: int = 2,
    target: float = STRETCH_TARGET,
) -> ChannelStretch:
    """The channel's screen transfer from the reference's own statistics."""

    sky, sigma = channel_statistics(reference_linear, filmstrip_divisor)
    return screen_transfer(sky, sigma, target=target)


def compose_to_reference(
    frame_matrix: Sequence[Sequence[float]] | None,
    reference_matrix: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """``B = M_r^-1 . M_i``: frame -> blink reference from the QC transforms
    (both frame -> QC reference, 3x3 in x/y order)."""

    if frame_matrix is None or reference_matrix is None:
        return None
    try:
        frame = np.asarray(frame_matrix, dtype=np.float64).reshape(3, 3)
        reference = np.asarray(reference_matrix, dtype=np.float64).reshape(3, 3)
        composed = np.linalg.inv(reference) @ frame
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(composed)):
        return None
    return (
        (float(composed[0, 0]), float(composed[0, 1]), float(composed[0, 2])),
        (float(composed[1, 0]), float(composed[1, 1]), float(composed[1, 2])),
    )


def warp_to_reference(
    image: np.ndarray,
    transform: tuple[tuple[float, float, float], tuple[float, float, float]],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Bilinear warp of ``image`` (frame pixels) onto the reference grid.

    ``scipy.ndimage.affine_transform`` wants the output -> input mapping in
    (row, column) order, so the frame -> reference matrix is inverted and its
    axes swapped; pixels outside the frame become NaN.
    """

    forward = np.array(
        [[*transform[0]], [*transform[1]], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    inverse = np.linalg.inv(forward)
    matrix = np.array(
        [[inverse[1, 1], inverse[1, 0]], [inverse[0, 1], inverse[0, 0]]], dtype=np.float64
    )
    offset = np.array([inverse[1, 2], inverse[0, 2]], dtype=np.float64)
    source = np.asarray(image, dtype=np.float32)
    # NaN inside the frame (masked pixels) must not spread through the
    # interpolation; they are filled with the sky and restored afterwards.
    invalid = ~np.isfinite(source)
    if np.any(invalid):
        fill = float(np.nanmedian(source)) if not np.all(invalid) else 0.0
        source = np.where(invalid, np.float32(fill), source)
    warped = ndimage.affine_transform(
        source, matrix, offset=offset, output_shape=output_shape, order=1, mode="constant", cval=np.nan
    )
    if np.any(invalid):
        mask = ndimage.affine_transform(
            invalid.astype(np.float32), matrix, offset=offset, output_shape=output_shape, order=0, mode="constant", cval=1.0
        )
        warped = np.where(mask > 0.5, np.nan, warped)
    return np.asarray(warped, dtype=np.float32)


def stretch_to_8bit(image: np.ndarray, stretch: ChannelStretch) -> np.ndarray:
    return np.asarray(np.rint(apply_stf(image, stretch) * 255.0), dtype=np.uint8)


def _encode(pixels: np.ndarray, *, format_name: str, quality: int) -> bytes:
    encoded = BytesIO()
    image = Image.fromarray(pixels, mode="L")
    if format_name == "jpeg":
        image.save(encoded, format="JPEG", quality=int(quality), optimize=False)
    else:
        image.save(encoded, format="PNG", optimize=False, compress_level=1)
    return encoded.getvalue()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()


def render_frame(spec: FramePreviewSpec) -> FramePreviewResult:
    """Render one frame's filmstrip and zoom previews; importable for a pool."""

    try:
        if spec.diagnostic is not None:
            from .blink_diagnostic_render import render_diagnostic_frame
            return render_diagnostic_frame(spec)
        linear = np.load(spec.linear_path, allow_pickle=False).astype(np.float32)
        if linear.ndim != 2:
            raise ValueError("linear preview is not two-dimensional")
        linear, calibrated = calibrate_linear(linear, spec.calibration)
        # The measured sky is the raw preview's; a calibrated frame gets its
        # own median (the stars barely move it).
        sky = float(np.nanmedian(linear)) if calibrated else spec.sky
        registered = spec.transform is not None
        if registered:
            assert spec.transform is not None
            image = warp_to_reference(linear, spec.transform, spec.output_shape)
        else:
            image = linear
        finite = np.isfinite(image)
        coverage = float(np.count_nonzero(finite)) / image.size if image.size else 0.0
        # Linear photometric normalization to the reference: same sky, same
        # flux scale, so the shared stretch means the same on every frame.
        normalized = (image - np.float32(sky)) * np.float32(spec.flux_scale) + np.float32(
            spec.reference_sky
        )
        zoom_pixels = stretch_to_8bit(normalized, spec.stretch)
        small = block_mean(normalized, spec.filmstrip_divisor)
        filmstrip_pixels = stretch_to_8bit(small, spec.stretch)
        zoom_bytes = _encode(zoom_pixels, format_name="png", quality=spec.jpeg_quality)
        filmstrip_bytes = _encode(
            filmstrip_pixels, format_name=spec.filmstrip_format, quality=spec.jpeg_quality
        )
        _write_new(Path(spec.zoom_path), zoom_bytes)
        _write_new(Path(spec.filmstrip_path), filmstrip_bytes)
        # The harder variant is one more transfer and encode of an array that
        # is already in memory; it never decides whether the frame rendered.
        hard_path, hard_bytes = None, b""
        if spec.filmstrip_hard_path is not None and spec.stretch_hard is not None:
            hard_bytes = _encode(
                stretch_to_8bit(small, spec.stretch_hard),
                format_name=spec.filmstrip_format,
                quality=spec.jpeg_quality,
            )
            _write_new(Path(spec.filmstrip_hard_path), hard_bytes)
            hard_path = spec.filmstrip_hard_path
        return FramePreviewResult(
            index=spec.index,
            filmstrip_path=spec.filmstrip_path,
            zoom_path=spec.zoom_path,
            filmstrip_bytes=len(filmstrip_bytes),
            zoom_bytes=len(zoom_bytes),
            filmstrip_hard_path=hard_path,
            filmstrip_hard_bytes=len(hard_bytes),
            filmstrip_shape=(int(filmstrip_pixels.shape[1]), int(filmstrip_pixels.shape[0])),
            zoom_shape=(int(zoom_pixels.shape[1]), int(zoom_pixels.shape[0])),
            coverage=coverage,
            registered=registered,
            sky=sky,
            calibrated=calibrated,
        )
    except Exception as error:  # one broken preview never stops the session
        return FramePreviewResult(
            index=spec.index,
            filmstrip_path=None,
            zoom_path=None,
            filmstrip_bytes=0,
            zoom_bytes=0,
            filmstrip_shape=(0, 0),
            zoom_shape=(0, 0),
            coverage=None,
            registered=False,
            error=f"{type(error).__name__}: {error}"[:300],
        )


def render_previews(
    specs: Sequence[FramePreviewSpec],
    *,
    workers: int = 1,
    runner: FrameRunner | None = None,
) -> list[FramePreviewResult]:
    """Render every spec, in order, with the shared frame worker pool."""

    if not specs:
        return []
    owned = FrameRunner(workers, len(specs)) if runner is None else None
    active = owned if owned is not None else runner
    assert active is not None
    try:
        return active.map(render_frame, list(specs))
    finally:
        if owned is not None:
            owned.close()


__all__ = [
    "ChannelStretch",
    "FILMSTRIP_FORMATS",
    "FramePreviewResult",
    "FramePreviewSpec",
    "PreviewCalibration",
    "STRETCH_HARD_TARGET",
    "STRETCH_TARGET",
    "apply_stf",
    "block_mean",
    "calibrate_linear",
    "channel_statistics",
    "channel_stretch",
    "screen_transfer",
    "compose_to_reference",
    "render_frame",
    "render_previews",
    "safe_stem",
    "stretch_to_8bit",
    "warp_to_reference",
]
