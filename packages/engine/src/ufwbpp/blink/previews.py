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

from pathlib import Path
from typing import Sequence

import numpy as np

from lightframeqc.parallel import FrameRunner

from .diagnostic_render import render_diagnostic_frame
from .imaging import (
    FramePreviewResult,
    FramePreviewSpec,
    block_mean,
    calibrate_linear,
    encode_image,
    stretch_to_8bit,
    warp_to_reference,
    write_new,
)


def render_frame(spec: FramePreviewSpec) -> FramePreviewResult:
    """Render one frame's filmstrip and zoom previews; importable for a pool."""

    try:
        if spec.diagnostic is not None:
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
        zoom_bytes = encode_image(zoom_pixels, format_name="png", quality=spec.jpeg_quality)
        filmstrip_bytes = encode_image(
            filmstrip_pixels, format_name=spec.filmstrip_format, quality=spec.jpeg_quality
        )
        write_new(Path(spec.zoom_path), zoom_bytes)
        write_new(Path(spec.filmstrip_path), filmstrip_bytes)
        # The harder variant is one more transfer and encode of an array that
        # is already in memory; it never decides whether the frame rendered.
        hard_path, hard_bytes = None, b""
        if spec.filmstrip_hard_path is not None and spec.stretch_hard is not None:
            hard_bytes = encode_image(
                stretch_to_8bit(small, spec.stretch_hard),
                format_name=spec.filmstrip_format,
                quality=spec.jpeg_quality,
            )
            write_new(Path(spec.filmstrip_hard_path), hard_bytes)
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


__all__ = ["render_frame", "render_previews"]
