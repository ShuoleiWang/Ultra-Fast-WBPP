"""Production transport/rendering of complementary Blink display diagnostics.

This module is called only by blink-measure when the desktop explicitly selects
v2. It never changes quality-gate decisions or science-product pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from io import BytesIO
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .blink_diagnostics import DISPLAY_ALGORITHM, DisplayReference, choose_display_reference, diagnostic_preview, local_noise
from .blink_native_crops import native_atlas, star_positions
from .blink_previews import FramePreviewResult, PreviewCalibration, _master_preview, _write_new, calibrate_linear, warp_to_reference

if TYPE_CHECKING:
    from .blink_previews import FramePreviewSpec


@dataclass(frozen=True, slots=True)
class DiagnosticFrameSpec:
    source_path: str
    reference_path: str
    output_directory: str
    source_noise: float | None
    flux_scale: float | None
    calibration: PreviewCalibration | None
    calibration_status: str
    calibration_note: str | None
    native_scale: int
    native_supported: bool


def prepare_diagnostic_specs(evidence, results, measurements, linear_paths, session, request, long_edge):
    # Import here to keep the legacy renderer independent of this extension.
    from .blink_session import _preview_calibration, _preview_specs

    prepared = {}
    noise = {}
    calibration_by_path = {}
    notes = {}
    root = session / "linear" / "diagnostic"
    root.mkdir(parents=True, exist_ok=False)
    for index, frame in enumerate(evidence.inputs):
        measurement = measurements.get(frame.path)
        path = linear_paths.get(frame.path)
        if measurement is None or path is None or not path.is_file():
            continue
        data = np.load(path, allow_pickle=False).astype(np.float32)
        calibration = _preview_calibration(request, frame.filter_name, measurement.metadata.exposure_seconds, long_edge)
        complete = False
        note = "missing-flat-or-pedestal"
        if calibration is not None and calibration.pedestal_path is not None:
            try:
                complete = all(_master_preview(p, long_edge).shape == data.shape for p in (calibration.flat_path, calibration.pedestal_path))
                note = "master-geometry-mismatch"
            except Exception:
                note = "master-unreadable"
        if complete:
            data, complete = calibrate_linear(data, calibration)
        if not complete:
            calibration = None
        calibration_by_path[frame.path] = calibration
        notes[frame.path] = None if complete else note
        target = root / f"{index:04d}.npy"
        with target.open("xb") as stream:
            np.save(stream, data)
        prepared[frame.path] = target
        try:
            noise[index] = local_noise(data)
        except ValueError:
            noise[index] = None

    references = dict(evidence.references)
    records = evidence.frame_records()
    for record, frame in zip(records, evidence.inputs, strict=True):
        record["normalization"] = {"registered": frame.registration_ok}
    for channel_id in evidence.channel_ids():
        members = [r for r in records if r["channelId"] == channel_id]
        try:
            index = choose_display_reference(members, noise)
        except ValueError:
            # An unscored but measurable channel still needs a labelled visual
            # reference. A channel with no noise estimate will fail rendering.
            index = next((r["index"] for r in members if r["reference"]), members[0]["index"])
        frame = evidence.inputs[index]
        references[channel_id] = replace(references[channel_id], path=frame.path, source_sha256=frame.source_sha256, rule="calibrated-local-noise-psf-v2")
    evidence = replace(evidence, references=references)
    # These files are already calibrated. Do not calibrate them a second time.
    raw_request = replace(request, master_flats=(), master_darks=(), master_bias=None)
    specs, stretches, geometry = _preview_specs(evidence, results, measurements, prepared, session, request.previews, raw_request, long_edge)
    frames = {frame.path: frame for frame in evidence.inputs}
    output = []
    for spec in specs:
        frame = evidence.inputs[spec.index]
        reference = evidence.references[frame.channel_id]
        ref_frame = frames[reference.path]
        metadata = measurements[frame.path].metadata
        calibration = calibration_by_path[frame.path]
        status = "calibrated" if calibration else "uncalibrated"
        ref_calibration = calibration_by_path.get(reference.path)
        # Mixing calibrated and uncalibrated frames makes the difference map
        # uninterpretable. Mark it unavailable instead of disguising it.
        compatible = bool(calibration) == bool(ref_calibration)
        gain = ref_frame.transparency / frame.transparency if ref_frame.transparency and frame.transparency and frame.transparency > 0 and compatible else None
        diagnostic = DiagnosticFrameSpec(
            source_path=frame.path,
            reference_path=str(prepared[reference.path]),
            output_directory=str(session / "diagnostic"),
            source_noise=noise.get(spec.index),
            flux_scale=gain,
            calibration=calibration,
            calibration_status=status,
            calibration_note=notes[frame.path],
            native_scale=max(1, math.ceil(max(metadata.width, metadata.height) / long_edge)),
            native_supported=metadata.channels == 1 and metadata.cfa_pattern.upper() in {"NONE", "UNKNOWN", "MONO", "MONOCHROME", ""},
        )
        output.append(replace(spec, diagnostic=diagnostic))
        reference_index = next(r["index"] for r in records if r["path"] == reference.path)
        geometry[frame.channel_id]["display"] = {
            "algorithm": DISPLAY_ALGORITHM,
            "noiseReference": noise.get(reference_index),
            "referenceCalibration": "calibrated" if ref_calibration else "uncalibrated",
        }
        geometry[frame.channel_id]["calibration"] = ref_calibration.serializable() if ref_calibration else None
    return evidence, output, stretches, geometry


@lru_cache(maxsize=2)
def _reference(path: str) -> DisplayReference:
    return DisplayReference.from_image(np.load(path, allow_pickle=False))


@lru_cache(maxsize=2)
def _star_positions(path: str):
    return star_positions(_reference(path))


def _encode(values: np.ndarray, *, jpeg: bool = False) -> bytes:
    image = Image.fromarray(np.rint(np.clip(np.nan_to_num(values), 0, 1) * 255).astype(np.uint8))
    stream = BytesIO()
    if jpeg:
        image.save(stream, format="JPEG", quality=85)
    else:
        image.save(stream, format="PNG", compress_level=3)
    return stream.getvalue()


def render_diagnostic_frame(spec: FramePreviewSpec) -> FramePreviewResult:
    diagnostic = spec.diagnostic
    assert diagnostic is not None
    if diagnostic.source_noise is None:
        raise ValueError("frame noise unavailable; inspect source data")
    reference = _reference(diagnostic.reference_path)
    data = np.load(spec.linear_path, allow_pickle=False).astype(np.float32)
    aligned = warp_to_reference(data, spec.transform, spec.output_shape) if spec.transform is not None else data
    result = diagnostic_preview(aligned, reference, source_noise=diagnostic.source_noise, flux_scale=diagnostic.flux_scale, registered=spec.transform is not None)
    zoom = _encode(result.detail)
    film_image = Image.fromarray(np.rint(result.detail * 255).astype(np.uint8))
    film_shape = (math.ceil(result.detail.shape[1] / spec.filmstrip_divisor), math.ceil(result.detail.shape[0] / spec.filmstrip_divisor))
    film_image = film_image.resize(film_shape, Image.Resampling.BOX)
    buffer = BytesIO()
    film_image.save(buffer, format="JPEG" if spec.filmstrip_format == "jpeg" else "PNG", quality=spec.jpeg_quality)
    film = buffer.getvalue()
    _write_new(Path(spec.zoom_path), zoom)
    _write_new(Path(spec.filmstrip_path), film)
    root = Path(diagnostic.output_directory)
    paths: dict[str, str | None] = {"field": None, "background": None, "nativeSignal": None, "nativeShape": None}
    def write(kind, values):
        path = root / f"{spec.index:04d}-{kind}.png"
        _write_new(path, _encode(values))
        paths[kind] = path.relative_to(root.parent).as_posix()
    write("field", result.field)
    if result.background_rgb is not None:
        write("background", result.background_rgb)
    native_status = "unsupported"
    shape_regions = 0
    if spec.transform is None:
        native_status = "unregistered"
    elif diagnostic.native_supported:
        try:
            transform = np.vstack([spec.transform, [0, 0, 1]])
            calibration = diagnostic.calibration
            atlas = native_atlas({"path": diagnostic.source_path}, transform, _star_positions(diagnostic.reference_path), reference,
                calibration.flat_path if calibration else None, calibration.pedestal_path if calibration else None, diagnostic.native_scale)
            if atlas is not None:
                write("nativeSignal", atlas.signal)
                write("nativeShape", atlas.shape)
                native_status = "ready"
                shape_regions = atlas.shape_regions
        except Exception:
            native_status = "unavailable"
    return FramePreviewResult(
        index=spec.index, filmstrip_path=spec.filmstrip_path, zoom_path=spec.zoom_path,
        filmstrip_bytes=len(film), zoom_bytes=len(zoom), filmstrip_shape=film_shape,
        zoom_shape=(result.detail.shape[1], result.detail.shape[0]), coverage=result.finite_fraction,
        registered=spec.transform is not None, sky=float(np.nanmedian(data)), calibrated=diagnostic.calibration is not None,
        diagnostic_previews=paths,
        diagnostics={
            "algorithm": DISPLAY_ALGORITHM, "calibration": diagnostic.calibration_status,
            "calibrationNote": diagnostic.calibration_note,
            "relativeSignal": 1 / diagnostic.flux_scale if diagnostic.flux_scale else None,
            "relativeNoise": result.relative_noise, "matchedSignalNoise": result.matched_signal_noise,
            "backgroundStatus": "ready" if result.background_rgb is not None else "unavailable",
            "backgroundSpan": float(np.nanpercentile(result.background_difference, 95) - np.nanpercentile(result.background_difference, 5)) / reference.noise if result.background_difference is not None else None,
            "nativeStatus": native_status, "shapeRegions": shape_regions,
        },
    )
