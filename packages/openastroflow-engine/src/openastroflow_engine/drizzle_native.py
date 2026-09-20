"""Drizzle integration on the native multithreaded kernel.

Each calibrated (unregistered) Light is dropped onto the ``scale`` times finer
output grid through its registration matrix, with the very same per-frame
normalization (multiplicative scale, additive offset and offset grid), the
same integration weights and the same per-sample rejection decisions the
ordinary integration of the group produced: the rejection mask lives on the
reference grid and a pixel is dropped only when the mask accepts it at the
pixel's registered position, exactly as PixInsight's DrizzleIntegration reads
the ImageIntegration rejection maps.  Kernels: square (exact quadrilateral
overlap), circular (exact disc overlap), gaussian and point.  A CFA pattern
turns the run into a Bayer drizzle whose three colour planes are dropped from
the mosaic's own pixels.

The output is one multi-extension FITS (``SCI`` primary, ``WHT`` weights,
``COVERAGE`` contributing frame counts) followed by a JSON receipt; both are
create-only and the receipt is the commit marker.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray

from .calibration import CalibrationError, FitsFrame, _plain_header_value
from .native_kernels import (
    DRIZZLE_KERNEL_ID,
    DRIZZLE_KERNELS,
    default_kernel_threads,
    load_native_kernels,
)

DRIZZLE_ALGORITHM = "native-drizzle-weighted-mean-v1"
SUPPORTED_SCALES = (1, 2, 3, 4)
SUPPORTED_KERNELS = tuple(DRIZZLE_KERNELS)
MINIMUM_PIXFRAC = 0.1
DEFAULT_ACCUMULATOR_BYTES = 2 * 1024**3
CFA_PATTERNS: dict[str, tuple[int, int, int, int]] = {
    # (y & 1, x & 1) -> channel 0 R, 1 G, 2 B for the 2x2 Bayer tile.
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}


class DrizzleError(CalibrationError):
    """A drizzle request or execution failed."""


@dataclass(frozen=True, slots=True)
class DrizzleFrame:
    """One calibrated Light and the products of its ordinary integration."""

    calibrated_path: str
    source_path: str
    input_to_reference: tuple[tuple[float, float, float], ...]
    weight: float
    exposure_seconds: float
    normalization_scale: float = 1.0
    normalization_offset: float = 0.0
    offset_grid: tuple[tuple[float, ...], ...] = ()
    offset_grid_x: tuple[float, ...] = ()
    offset_grid_y: tuple[float, ...] = ()
    # Optional region weight map of the integration (node values in [0, 1]).
    weight_grid: tuple[tuple[float, ...], ...] = ()
    weight_grid_x: tuple[float, ...] = ()
    weight_grid_y: tuple[float, ...] = ()
    # Packed row bits of the accepted-sample mask on the reference grid
    # (``np.packbits`` along the rows); None accepts every sample.
    accepted_mask_bits: NDArray[np.uint8] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DrizzleGroupInputs:
    """Everything the drizzle of one filter group needs from the ordinary
    integration: calibrated frames, transforms, normalization, weights and
    rejection masks, plus the master's header metadata."""

    filter_name: str
    frames: tuple[DrizzleFrame, ...]
    reference_shape: tuple[int, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DrizzleGroupRequest:
    frames: tuple[DrizzleFrame, ...]
    reference_shape: tuple[int, int]
    output_path: str
    receipt_path: str
    scale: int = 2
    pixfrac: float = 0.9
    kernel: str = "square"
    cfa_pattern: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    max_accumulator_bytes: int = DEFAULT_ACCUMULATOR_BYTES
    threads: int | None = None
    durable: bool = True

    def validate(self) -> None:
        if not self.frames:
            raise DrizzleError("DRIZZLE_NO_INPUTS", "drizzle requires at least one frame")
        if self.scale not in SUPPORTED_SCALES:
            raise DrizzleError("DRIZZLE_SCALE_INVALID", f"drizzle scale must be one of {SUPPORTED_SCALES}")
        if not (math.isfinite(self.pixfrac) and MINIMUM_PIXFRAC <= self.pixfrac <= 1.0):
            raise DrizzleError("DRIZZLE_PIXFRAC_INVALID", f"drop shrink must be in [{MINIMUM_PIXFRAC}, 1]")
        if self.kernel not in SUPPORTED_KERNELS:
            raise DrizzleError("DRIZZLE_KERNEL_INVALID", f"drizzle kernel must be one of {SUPPORTED_KERNELS}")
        if self.cfa_pattern is not None and self.cfa_pattern.upper() not in CFA_PATTERNS:
            raise DrizzleError("DRIZZLE_CFA_PATTERN_INVALID", f"CFA pattern must be one of {tuple(CFA_PATTERNS)}")
        height, width = self.reference_shape
        if height < 1 or width < 1:
            raise DrizzleError("DRIZZLE_GEOMETRY_INVALID", "reference shape must be positive")
        if self.max_accumulator_bytes < 16 * width * self.scale:
            raise DrizzleError("DRIZZLE_MEMORY_BUDGET_TOO_SMALL", "one output row exceeds the accumulator budget")
        for index, frame in enumerate(self.frames):
            matrix = np.asarray(frame.input_to_reference, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise DrizzleError("DRIZZLE_TRANSFORM_INVALID", f"frames[{index}] transform must be a finite 3x3 matrix")
            if abs(float(np.linalg.det(matrix))) <= 1e-12:
                raise DrizzleError("DRIZZLE_TRANSFORM_INVALID", f"frames[{index}] transform is singular")
            if not (math.isfinite(frame.weight) and frame.weight >= 0.0):
                raise DrizzleError("DRIZZLE_WEIGHT_INVALID", f"frames[{index}] weight must be finite and non-negative")
            if not (math.isfinite(frame.exposure_seconds) and frame.exposure_seconds > 0.0):
                raise DrizzleError("DRIZZLE_EXPOSURE_INVALID", f"frames[{index}] exposure must be positive")
            for label, grid, x_nodes, y_nodes in (
                ("offset", frame.offset_grid, frame.offset_grid_x, frame.offset_grid_y),
                ("weight", frame.weight_grid, frame.weight_grid_x, frame.weight_grid_y),
            ):
                if grid and (
                    len(grid) != len(y_nodes) or any(len(row) != len(x_nodes) for row in grid)
                ):
                    raise DrizzleError("DRIZZLE_GRID_INVALID", f"frames[{index}] {label} grid does not match its nodes")
            bits = frame.accepted_mask_bits
            if bits is not None and bits.shape != (height, (width + 7) // 8):
                raise DrizzleError("DRIZZLE_MASK_INVALID", f"frames[{index}] rejection mask does not match the reference grid")


@dataclass(frozen=True, slots=True)
class DrizzleGroupResult:
    output_path: str
    receipt_path: str
    output_sha256: str
    receipt: Mapping[str, Any]


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


PERCENTILE_SAMPLE_LIMIT = 4_000_000


def _percentiles(values: NDArray[np.floating[Any]]) -> dict[str, float]:
    """Advisory percentiles; large arrays are strided down to a fixed sample
    so the summary costs milliseconds instead of a partial sort of the image."""

    if values.size == 0:
        return {}
    flat = values.reshape(-1)
    if flat.size > PERCENTILE_SAMPLE_LIMIT:
        flat = flat[:: -(-flat.size // PERCENTILE_SAMPLE_LIMIT)]
    quantiles = np.percentile(flat, (5, 25, 50, 75, 95))
    return {f"p{level}": float(value) for level, value in zip((5, 25, 50, 75, 95), quantiles, strict=True)}


def _dither_evidence(frames: Sequence[DrizzleFrame]) -> dict[str, Any]:
    """Sub-pixel translation phases of the registration matrices (advisory)."""

    phases = []
    for frame in frames:
        matrix = np.asarray(frame.input_to_reference, dtype=np.float64)
        centre = matrix @ np.array([0.0, 0.0, 1.0])
        centre = centre[:2] / centre[2]
        phases.append((float(centre[0] % 1.0), float(centre[1] % 1.0)))
    array = np.asarray(phases, dtype=np.float64)
    distinct = 0
    if array.size:
        rounded = np.round(array / 0.15).astype(np.int64)
        distinct = int(len({tuple(item) for item in rounded.tolist()}))
    return {
        "framePhases": [[round(x, 4), round(y, 4)] for x, y in phases],
        "distinctPhaseBins": distinct,
        "phaseBinPixels": 0.15,
        "status": "SAMPLED" if distinct >= 3 or len(frames) < 3 else "FEW_DISTINCT_PHASES",
    }


def drizzle_group(request: DrizzleGroupRequest) -> DrizzleGroupResult:
    """Drizzle one filter group and publish the FITS artifact plus its receipt."""

    request.validate()
    kernels = load_native_kernels()
    if kernels is None or not hasattr(kernels, "drizzle_band"):
        raise DrizzleError("DRIZZLE_KERNEL_UNAVAILABLE", "the native drizzle kernel library is not loaded")
    output_path = Path(request.output_path).expanduser().resolve(strict=False)
    receipt_path = Path(request.receipt_path).expanduser().resolve(strict=False)
    for path in (output_path, receipt_path):
        if path.exists() or os.path.lexists(path):
            raise DrizzleError("OUTPUT_EXISTS", "refusing to overwrite drizzle output", path=str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    scale = int(request.scale)
    height, width = request.reference_shape
    output_height, output_width = height * scale, width * scale
    channels = 3 if request.cfa_pattern else 1
    pattern = CFA_PATTERNS[request.cfa_pattern.upper()] if request.cfa_pattern else (0, 1, 1, 2)
    threads = default_kernel_threads() if request.threads is None else max(1, int(request.threads))
    bytes_per_row = output_width * 16 * channels
    band_rows = max(1, min(output_height, int(request.max_accumulator_bytes // bytes_per_row)))
    scale_matrix = np.diag([float(scale), float(scale), 1.0])

    science = np.full((channels, output_height, output_width), np.nan, dtype=np.float32)
    weight_map = np.zeros((channels, output_height, output_width), dtype=np.float32)
    coverage = np.zeros((output_height, output_width), dtype=np.int16)
    accepted_input_pixels = 0
    total_input_pixels = 0
    frame_timing: list[float] = []
    frames_masks: list[NDArray[np.uint8] | None] = []
    for frame in request.frames:
        if frame.accepted_mask_bits is None:
            frames_masks.append(None)
            total_input_pixels += height * width
            accepted_input_pixels += height * width
        else:
            unpacked = np.unpackbits(frame.accepted_mask_bits, axis=1, count=width).astype(np.uint8)
            frames_masks.append(unpacked)
            total_input_pixels += height * width
            accepted_input_pixels += int(np.count_nonzero(unpacked))

    sources: list[FitsFrame] = []
    reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oaf-drizzle-read")
    try:
        for frame in request.frames:
            opened = FitsFrame(frame.calibrated_path).__enter__()
            sources.append(opened)
            if opened.shape != (height, width):
                raise DrizzleError(
                    "DRIZZLE_GEOMETRY_INVALID",
                    f"calibrated frame shape {opened.shape} differs from the reference {request.reference_shape}",
                    path=frame.calibrated_path,
                )
        active = [
            (frame, source, mask)
            for frame, source, mask in zip(request.frames, sources, frames_masks, strict=True)
            if frame.weight > 0.0
        ]
        for band_start in range(0, output_height, band_rows):
            band_stop = min(output_height, band_start + band_rows)
            rows = band_stop - band_start
            band_sum = np.zeros((channels, rows, output_width), dtype=np.float64)
            band_weight = np.zeros((channels, rows, output_width), dtype=np.float64)
            band_coverage = np.zeros((rows, output_width), dtype=np.int16)
            touched = np.zeros((rows, output_width), dtype=np.uint8)
            # The kernel runs without the GIL, so the next frame is decoded
            # while the current one is dropped.
            pending = reader.submit(active[0][1].full_values) if active else None
            for index, (frame, source, mask) in enumerate(active):
                frame_started = time.perf_counter()
                values = pending.result() if pending is not None else source.full_values()
                pending = (
                    reader.submit(active[index + 1][1].full_values)
                    if index + 1 < len(active)
                    else None
                )
                forward = scale_matrix @ np.asarray(frame.input_to_reference, dtype=np.float64)
                touched.fill(0)
                for channel in range(channels):
                    kernels.drizzle_band(
                        values,
                        source_row0=0,
                        forward=forward,
                        scale=scale,
                        pixfrac=request.pixfrac,
                        kernel=request.kernel,
                        output_sum=band_sum[channel],
                        output_weight=band_weight[channel],
                        output_row0=band_start,
                        normalization_scale=frame.normalization_scale,
                        normalization_offset=frame.normalization_offset,
                        grid=np.asarray(frame.offset_grid, dtype=np.float64) if frame.offset_grid else None,
                        grid_x_nodes=np.asarray(frame.offset_grid_x, dtype=np.float64) if frame.offset_grid else None,
                        grid_y_nodes=np.asarray(frame.offset_grid_y, dtype=np.float64) if frame.offset_grid else None,
                        weight_grid=np.asarray(frame.weight_grid, dtype=np.float64) if frame.weight_grid else None,
                        weight_grid_x_nodes=np.asarray(frame.weight_grid_x, dtype=np.float64) if frame.weight_grid else None,
                        weight_grid_y_nodes=np.asarray(frame.weight_grid_y, dtype=np.float64) if frame.weight_grid else None,
                        mask=mask,
                        cfa_pattern=pattern,
                        channel=channel if channels == 3 else 255,
                        frame_weight=frame.weight,
                        threads=threads,
                        output_touched=touched,
                    )
                band_coverage += touched
                frame_timing.append(time.perf_counter() - frame_started)
            covered = band_weight > 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                band_image = np.where(covered, band_sum / band_weight, np.nan)
            science[:, band_start:band_stop] = band_image.astype(np.float32)
            weight_map[:, band_start:band_stop] = band_weight.astype(np.float32)
            coverage[band_start:band_stop] = band_coverage
    finally:
        reader.shutdown(wait=True)
        while sources:
            sources.pop().close()

    output_pixels = output_height * output_width
    covered_pixels = int(np.count_nonzero(coverage > 0))
    null_pixels = output_pixels - covered_pixels
    coverage_fraction = covered_pixels / output_pixels if output_pixels else 0.0
    covered_weights = weight_map[0][coverage > 0] if channels == 1 else weight_map.sum(axis=0)[coverage > 0]
    dither = _dither_evidence(request.frames)
    elapsed = time.perf_counter() - started

    header_metadata = {
        **dict(request.metadata),
        "OAFDRZ": "NATIVE",
        "OAFDRZSC": scale,
        "OAFDRZPF": float(request.pixfrac),
        "OAFDRZKN": request.kernel,
        "OAFNFRM": len(request.frames),
    }
    if request.cfa_pattern:
        header_metadata["OAFDRZCF"] = request.cfa_pattern.upper()
    primary = fits.PrimaryHDU(science[0] if channels == 1 else science)
    primary.name = "SCI"
    for key, value in header_metadata.items():
        if value is not None:
            primary.header[str(key).strip().upper()] = _plain_header_value(value)
    weight_hdu = fits.ImageHDU(weight_map[0] if channels == 1 else weight_map, name="WHT")
    coverage_hdu = fits.ImageHDU(coverage, name="COVERAGE")
    staged_output = output_path.with_name(f".{output_path.name}.partial")
    staged_receipt = receipt_path.with_name(f".{receipt_path.name}.partial")
    for path in (staged_output, staged_receipt):
        if path.exists() or os.path.lexists(path):
            raise DrizzleError("OUTPUT_EXISTS", "drizzle staging file already exists", path=str(path))
    try:
        # No FITS checksums: the receipt's SHA-256 binds the artifact, and the
        # ordinary masters carry none either, so the later header rewrites
        # (solved-state promotion, same-grid solve) do not re-sum the image.
        fits.HDUList([primary, weight_hdu, coverage_hdu]).writeto(staged_output, overwrite=False, checksum=False)
        if request.durable:
            with staged_output.open("r+b") as stream:
                os.fsync(stream.fileno())
        output_digest = _sha256_of(staged_output)
        receipt_core: dict[str, Any] = {
            "schemaVersion": 3,
            "stage": "drizzle",
            "status": "succeeded",
            "algorithm": DRIZZLE_ALGORITHM,
            "backend": {"id": DRIZZLE_KERNEL_ID, "device": "CPU", "threads": threads},
            "recipe": {
                "scale": scale,
                "pixfrac": float(request.pixfrac),
                "kernel": request.kernel,
                "cfaPattern": request.cfa_pattern.upper() if request.cfa_pattern else None,
                "inputUnits": "normalized-integration-frame",
            },
            "geometry": {
                "referenceHeight": height,
                "referenceWidth": width,
                "outputHeight": output_height,
                "outputWidth": output_width,
                "channels": channels,
                "outputPixelCentre": "output index = scale * reference index",
            },
            "inputs": [
                {
                    "index": index,
                    "source": frame.source_path,
                    "calibrated": frame.calibrated_path,
                    "weight": float(frame.weight),
                    "exposureSeconds": float(frame.exposure_seconds),
                    "normalization": {
                        "scale": float(frame.normalization_scale),
                        "offset": float(frame.normalization_offset),
                        "offsetGrid": bool(frame.offset_grid),
                    },
                    "regionWeightGrid": bool(frame.weight_grid),
                    "rejectionMask": frame.accepted_mask_bits is not None,
                    "inputToReference": [list(map(float, row)) for row in frame.input_to_reference],
                }
                for index, frame in enumerate(request.frames)
            ],
            "science": {
                "normalization": "per-frame integration scale, offset and offset grid at the registered position",
                "rejection": "ordinary integration acceptance mask sampled at the registered position",
                "weights": "ordinary integration frame weights",
                "dither": dither,
            },
            "statistics": {
                "inputFrames": len(request.frames),
                "totalInputPixels": total_input_pixels,
                "acceptedInputPixels": accepted_input_pixels,
                "outputPixels": output_pixels,
                "coveredPixels": covered_pixels,
                "coverageFraction": coverage_fraction,
                "nullPixels": null_pixels,
                "nullPixelFraction": null_pixels / output_pixels if output_pixels else 0.0,
                "coveragePercentiles": _percentiles(coverage[coverage > 0].astype(np.float64)),
                "coveredWeightPercentiles": _percentiles(np.asarray(covered_weights, dtype=np.float64)),
                "wallSeconds": round(elapsed, 3),
                "frameSeconds": [round(value, 3) for value in frame_timing],
            },
            "artifact": {
                "path": str(output_path),
                "mediaType": "image/fits",
                "sha256": output_digest,
                "sizeBytes": staged_output.stat().st_size,
                "extensions": ["SCI", "WHT", "COVERAGE"],
            },
        }
        receipt: dict[str, Any] = {
            "receiptId": "sha256:" + hashlib.sha256(_canonical_json(receipt_core)).hexdigest(),
            **receipt_core,
        }
        with staged_receipt.open("xb") as stream:
            stream.write(_canonical_json(receipt))
            stream.flush()
            if request.durable:
                os.fsync(stream.fileno())
        os.link(staged_output, output_path)
        try:
            os.link(staged_receipt, receipt_path)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
    finally:
        staged_output.unlink(missing_ok=True)
        staged_receipt.unlink(missing_ok=True)
    return DrizzleGroupResult(
        output_path=str(output_path),
        receipt_path=str(receipt_path),
        output_sha256=output_digest,
        receipt=receipt,
    )


def verify_drizzle_receipt(receipt_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Re-read a drizzle receipt and check its artifact still hashes the same."""

    path = Path(receipt_path)
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict) or payload.get("stage") != "drizzle" or payload.get("status") != "succeeded":
        raise DrizzleError("DRIZZLE_RECEIPT_INVALID", "drizzle receipt is not a succeeded drizzle stage", path=str(path))
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise DrizzleError("DRIZZLE_RECEIPT_INVALID", "drizzle receipt lacks its artifact", path=str(path))
    artifact_path = Path(str(artifact.get("path")))
    if _sha256_of(artifact_path) != artifact.get("sha256"):
        raise DrizzleError("DRIZZLE_ARTIFACT_CHANGED", "drizzle artifact differs from its receipt", path=str(artifact_path))
    core = {key: value for key, value in payload.items() if key != "receiptId"}
    if payload.get("receiptId") != "sha256:" + hashlib.sha256(_canonical_json(core)).hexdigest():
        raise DrizzleError("DRIZZLE_RECEIPT_INVALID", "drizzle receipt identifier does not match its content", path=str(path))
    return payload


__all__ = [
    "CFA_PATTERNS",
    "DRIZZLE_ALGORITHM",
    "DrizzleError",
    "DrizzleFrame",
    "DrizzleGroupInputs",
    "DrizzleGroupRequest",
    "DrizzleGroupResult",
    "SUPPORTED_KERNELS",
    "SUPPORTED_SCALES",
    "drizzle_group",
    "verify_drizzle_receipt",
]
