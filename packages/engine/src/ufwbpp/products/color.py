"""Atomic, WCS-safe construction of linear RGB and display products.

The color product boundary accepts only independently solved, co-registered
monochrome masters.  It validates the celestial WCS at the four corners and
the image centre before copying the reference WCS into a three-plane linear
FITS product.  Preview rendering applies one luminance-derived multiplier to
all three channels at each pixel, so the asinh stretch does not independently
rebalance channel colours.

All artifacts are first written into a sibling staging directory.  The output
directory is then published with a platform no-replace primitive; caller-owned
inputs are never opened for update and an existing destination is never
overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Any, Callable, Mapping
import zlib

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray

from ..integrity import canonical_json_document, sha256_digest
from ..platform import remove_tree
from ..solvers.base import validate_wcs_header
from .publication import (
    ColorProductError,
    _fsync_directory,
    _best_effort_fsync_directory,
    _publish_directory_no_replace,
)


COLOR_PRODUCT_VERSION = "ultra-fast-wbpp-color-product-v1"
_CHANNELS = ("R", "G", "B")


@dataclass(frozen=True, slots=True)
class ColorProductRequest:
    red_path: str
    green_path: str
    blue_path: str
    output_directory: str
    luminance_path: str | None = None
    wcs_tolerance_pixels: float = 0.05
    preview_max_long_edge: int = 2048
    asinh_softness: float = 12.0
    # File stem of the FITS cube and both previews; ``None`` picks ``RGB`` or
    # ``LRGB``.  PixInsight labels an opened image with its file stem, so the
    # default names the view after the channels it holds.
    product_name: str | None = None


@dataclass(frozen=True, slots=True)
class ColorProductResult:
    output_directory: str
    linear_rgb_path: str
    preview_tiff_path: str
    preview_png_path: str
    receipt_path: str
    receipt: Mapping[str, Any]

    def serializable(self) -> dict[str, Any]:
        return {
            "outputDirectory": self.output_directory,
            "linearRgbPath": self.linear_rgb_path,
            "previewTiffPath": self.preview_tiff_path,
            "previewPngPath": self.preview_png_path,
            "receiptPath": self.receipt_path,
            "receipt": dict(self.receipt),
        }


@dataclass(frozen=True, slots=True)
class _Channel:
    name: str
    path: Path
    data: NDArray[np.float32]
    header: fits.Header
    wcs: WCS
    shape: tuple[int, int]
    identity: Mapping[str, Any]
    wcs_validation: Mapping[str, Any]


DirectoryPublisher = Callable[[Path, Path], None]


def _regular_identity(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as error:
        raise ColorProductError("INPUT_UNREADABLE", str(error), path=str(path)) from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ColorProductError(
            "INPUT_NOT_REGULAR_FILE", "input must be a non-symlink regular file", path=str(path)
        )
    return {
        "path": str(path),
        "sha256": sha256_digest(path),
        "sizeBytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _combined_image_header(hdul: fits.HDUList, image_hdu: fits.hdu.base.ExtensionHDU) -> fits.Header:
    header = hdul[0].header.copy()
    if image_hdu is not hdul[0]:
        header.extend(image_hdu.header, update=True, strip=True)
    return header


def _normalized_color_filter(value: Any) -> str | None:
    token = "".join(character for character in str(value).upper() if character.isalnum())
    aliases = {
        "R": "R",
        "RED": "R",
        "G": "G",
        "GREEN": "G",
        "B": "B",
        "BLUE": "B",
        "L": "L",
        "LUM": "L",
        "LUMINANCE": "L",
    }
    return aliases.get(token)


def _read_channel(name: str, value: str) -> _Channel:
    if not isinstance(value, str) or not value.strip():
        raise ColorProductError("CHANNEL_MISSING", f"{name} channel path is required")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise ColorProductError("CHANNEL_MISSING", str(error), path=value) from error
    identity = _regular_identity(path)
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=False,
            do_not_scale_image_data=False,
            uint=True,
            checksum=False,
        ) as hdul:
            image_hdu = next(
                (hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2),
                None,
            )
            if image_hdu is None:
                raise ColorProductError(
                    "CHANNEL_IMAGE_INVALID", "a two-dimensional image HDU is required", path=str(path)
                )
            data = np.asarray(image_hdu.data, dtype=np.float32).copy()
            header = _combined_image_header(hdul, image_hdu)
    except ColorProductError:
        raise
    except Exception as error:
        raise ColorProductError("CHANNEL_READ_FAILED", str(error), path=str(path)) from error

    shape = tuple(int(item) for item in data.shape)
    observed_filter = header.get("FILTER")
    normalized_filter = _normalized_color_filter(observed_filter)
    if normalized_filter is None:
        raise ColorProductError(
            "CHANNEL_FILTER_UNSUPPORTED",
            f"{name} input requires an explicit R/G/B/L FILTER value; found {observed_filter!r}",
            path=str(path),
        )
    if normalized_filter != name:
        raise ColorProductError(
            "CHANNEL_FILTER_MISMATCH",
            f"request assigns {name}, but FITS FILTER={observed_filter!r} identifies {normalized_filter}",
            path=str(path),
        )
    if header.get("OAFSTATE") != "SOLVED" or header.get("OAFWCS") != "SOLVED":
        raise ColorProductError(
            "CHANNEL_NOT_SOLVED",
            "OAFSTATE=SOLVED and OAFWCS=SOLVED are required",
            path=str(path),
        )
    validation = validate_wcs_header(header, image_shape=shape)
    if not validation.valid:
        raise ColorProductError(
            "CHANNEL_WCS_INVALID", f"{validation.code}: {validation.message}", path=str(path)
        )
    try:
        wcs = WCS(header, relax=False).celestial
    except Exception as error:
        raise ColorProductError("CHANNEL_WCS_INVALID", str(error), path=str(path)) from error
    channel = _Channel(
        name=name,
        path=path,
        data=data,
        header=header,
        wcs=wcs,
        shape=shape,
        identity={
            "channel": name,
            "observedFilter": str(observed_filter),
            "normalizedFilter": normalized_filter,
            **identity,
        },
        wcs_validation=validation.serializable(),
    )
    _verify_channel_stat(channel)
    return channel


def _verify_channel_stat(channel: _Channel) -> None:
    try:
        info = channel.path.lstat()
    except OSError as error:
        raise ColorProductError(
            "SOURCE_CHANGED", "channel disappeared during execution", path=str(channel.path)
        ) from error
    current = (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)
    expected = (
        channel.identity["sizeBytes"],
        channel.identity["mtimeNs"],
        channel.identity["device"],
        channel.identity["inode"],
    )
    if channel.path.is_symlink() or current != expected:
        raise ColorProductError(
            "SOURCE_CHANGED", "channel identity changed during execution", path=str(channel.path)
        )


def _five_points(shape: tuple[int, int]) -> NDArray[np.float64]:
    height, width = shape
    return np.asarray(
        [
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (0.0, float(height - 1)),
            (float(width - 1), float(height - 1)),
            ((width - 1) / 2.0, (height - 1) / 2.0),
        ],
        dtype=np.float64,
    )


def _wcs_consistency(
    reference: _Channel,
    candidate: _Channel,
    *,
    tolerance_pixels: float,
) -> dict[str, Any]:
    points = _five_points(reference.shape)
    try:
        reference_world = np.asarray(reference.wcs.all_pix2world(points, 0), dtype=np.float64)
        candidate_world = np.asarray(candidate.wcs.all_pix2world(points, 0), dtype=np.float64)
        candidate_on_reference = np.asarray(
            reference.wcs.all_world2pix(candidate_world, 0), dtype=np.float64
        )
        if not all(
            np.all(np.isfinite(value))
            for value in (reference_world, candidate_world, candidate_on_reference)
        ):
            raise ValueError("a five-point WCS transform produced non-finite coordinates")
        residuals = np.hypot(
            candidate_on_reference[:, 0] - points[:, 0],
            candidate_on_reference[:, 1] - points[:, 1],
        )
        reference_sky = SkyCoord(
            reference_world[:, 0] * u.deg, reference_world[:, 1] * u.deg, frame="icrs"
        )
        candidate_sky = SkyCoord(
            candidate_world[:, 0] * u.deg, candidate_world[:, 1] * u.deg, frame="icrs"
        )
        separations = reference_sky.separation(candidate_sky).arcsec
    except Exception as error:
        raise ColorProductError(
            "CHANNEL_WCS_COMPARISON_FAILED", str(error), path=str(candidate.path)
        ) from error
    maximum_pixels = float(np.max(residuals))
    if maximum_pixels > tolerance_pixels:
        raise ColorProductError(
            "CHANNEL_WCS_MISMATCH",
            (
                f"{candidate.name} differs from {reference.name} by "
                f"{maximum_pixels:.6g} px at the five validation points "
                f"(limit {tolerance_pixels:.6g} px)"
            ),
            path=str(candidate.path),
        )
    return {
        "referenceChannel": reference.name,
        "candidateChannel": candidate.name,
        "samplePixels": points.tolist(),
        "residualPixels": [float(item) for item in residuals],
        "separationArcseconds": [float(item) for item in separations],
        "maxResidualPixels": maximum_pixels,
        "maxSeparationArcseconds": float(np.max(separations)),
        "tolerancePixels": tolerance_pixels,
    }


def _validate_request(request: ColorProductRequest) -> tuple[Path, float, int, float]:
    try:
        output = Path(request.output_directory).expanduser().resolve(strict=False)
    except (OSError, TypeError) as error:
        raise ColorProductError("OUTPUT_PATH_INVALID", str(error)) from error
    if not str(request.output_directory).strip():
        raise ColorProductError("OUTPUT_PATH_INVALID", "output_directory is required")
    if output.exists() or output.is_symlink():
        raise ColorProductError("OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(output))
    tolerance = request.wcs_tolerance_pixels
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ColorProductError("WCS_TOLERANCE_INVALID", "wcs_tolerance_pixels must be numeric")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 1.0:
        raise ColorProductError("WCS_TOLERANCE_INVALID", "WCS tolerance must be in (0, 1] pixels")
    edge = request.preview_max_long_edge
    if isinstance(edge, bool) or not isinstance(edge, int) or edge < 16:
        raise ColorProductError("PREVIEW_SIZE_INVALID", "preview_max_long_edge must be at least 16")
    softness = request.asinh_softness
    if isinstance(softness, bool) or not isinstance(softness, (int, float)):
        raise ColorProductError("ASINH_SOFTNESS_INVALID", "asinh_softness must be numeric")
    softness = float(softness)
    if not math.isfinite(softness) or softness <= 0.0:
        raise ColorProductError("ASINH_SOFTNESS_INVALID", "asinh_softness must be finite and positive")
    return output, tolerance, edge, softness


def _product_name(request: ColorProductRequest) -> str:
    name = request.product_name
    if name is None:
        return "LRGB" if request.luminance_path is not None else "RGB"
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,63}", name):
        raise ColorProductError(
            "PRODUCT_NAME_INVALID", "product_name must be 1-64 letters, digits, '_' or '-'"
        )
    return name


def _matched_luminance_rgb(
    rgb: NDArray[np.float32], luminance: NDArray[np.float32]
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    rgb_luminance = (
        0.2126 * rgb[0].astype(np.float64)
        + 0.7152 * rgb[1].astype(np.float64)
        + 0.0722 * rgb[2].astype(np.float64)
    )
    selected = np.isfinite(rgb_luminance) & np.isfinite(luminance)
    if int(np.count_nonzero(selected)) < 64:
        raise ColorProductError(
            "LUMINANCE_FINITE_PIXELS_INSUFFICIENT",
            "at least 64 finite L/RGB samples are required",
        )
    rgb_low, rgb_high = (float(item) for item in np.percentile(rgb_luminance[selected], (1.0, 99.0)))
    l_low, l_high = (float(item) for item in np.percentile(luminance[selected], (1.0, 99.0)))
    if rgb_high <= rgb_low or l_high <= l_low:
        raise ColorProductError("LUMINANCE_DYNAMIC_RANGE_EMPTY", "L or RGB luminance has no usable range")
    gain = (rgb_high - rgb_low) / (l_high - l_low)
    offset = rgb_low - gain * l_low
    matched = luminance.astype(np.float64) * gain + offset
    epsilon = max((rgb_high - rgb_low) * 1e-8, np.finfo(np.float32).tiny)
    factor = np.ones(rgb_luminance.shape, dtype=np.float64)
    valid = selected & (np.abs(rgb_luminance) > epsilon)
    factor[valid] = matched[valid] / rgb_luminance[valid]
    factor = np.clip(factor, 0.0, 100.0)
    result = (rgb.astype(np.float64) * factor[np.newaxis, :, :]).astype(np.float32)
    return result, {
        "method": "affine-percentile-luminance-replacement-color-ratio-preserving",
        "percentiles": [1.0, 99.0],
        "gain": gain,
        "offset": offset,
        "factorClippedRange": [0.0, 100.0],
    }


def _block_mean_rgb(rgb: NDArray[np.float32], max_long_edge: int) -> tuple[NDArray[np.float32], int]:
    _, height, width = rgb.shape
    factor = max(1, math.ceil(max(height, width) / max_long_edge))
    if factor == 1:
        return rgb.copy(), factor
    out_height = math.ceil(height / factor)
    out_width = math.ceil(width / factor)
    output = np.full((3, out_height, out_width), np.nan, dtype=np.float32)
    x_starts = np.arange(0, width, factor, dtype=np.int64)
    for channel in range(3):
        for out_y in range(out_height):
            y0 = out_y * factor
            y1 = min(height, y0 + factor)
            block = rgb[channel, y0:y1].astype(np.float64, copy=False)
            finite = np.isfinite(block)
            sums = np.sum(np.where(finite, block, 0.0), axis=0)
            counts = np.sum(finite, axis=0)
            reduced_sum = np.add.reduceat(sums, x_starts)
            reduced_count = np.add.reduceat(counts, x_starts)
            row = np.full(out_width, np.nan, dtype=np.float64)
            np.divide(reduced_sum, reduced_count, out=row, where=reduced_count > 0)
            output[channel, out_y] = row.astype(np.float32)
    return output, factor


def _color_preserving_asinh(
    rgb: NDArray[np.float32], softness: float
) -> tuple[NDArray[np.uint16], dict[str, Any]]:
    working = rgb.astype(np.float64, copy=False)
    luminance = 0.2126 * working[0] + 0.7152 * working[1] + 0.0722 * working[2]
    finite = np.isfinite(luminance)
    if int(np.count_nonzero(finite)) < 64:
        raise ColorProductError("PREVIEW_FINITE_PIXELS_INSUFFICIENT", "at least 64 finite pixels are required")
    black, white = (float(item) for item in np.percentile(luminance[finite], (0.1, 99.8)))
    if not math.isfinite(black) or not math.isfinite(white) or white <= black:
        raise ColorProductError("PREVIEW_DYNAMIC_RANGE_EMPTY", "RGB luminance has no usable range")
    target = np.zeros(luminance.shape, dtype=np.float64)
    normalized = np.clip((luminance[finite] - black) / (white - black), 0.0, None)
    target[finite] = np.arcsinh(normalized * softness) / math.asinh(softness)
    scale = np.zeros(luminance.shape, dtype=np.float64)
    usable = finite & (luminance > max(abs(black) * 1e-12, np.finfo(np.float32).tiny))
    scale[usable] = target[usable] / luminance[usable]
    stretched = np.moveaxis(working * scale[np.newaxis, :, :], 0, -1)
    stretched[~np.isfinite(stretched)] = 0.0
    clipped = np.clip(stretched, 0.0, 1.0)
    saturated = float(np.count_nonzero(stretched > 1.0) / stretched.size)
    pixels = np.rint(clipped * 65535.0).astype(np.uint16)
    return pixels, {
        "method": "shared-luminance-multiplier-asinh",
        "blackPoint": black,
        "whitePoint": white,
        "softness": softness,
        "saturatedSampleFraction": saturated,
        "bitDepth": 16,
        "channelOrder": "RGB",
    }


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_png16(path: Path, pixels: NDArray[np.uint16]) -> None:
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("PNG writer requires RGB pixels")
    big_endian = np.asarray(pixels, dtype=">u2")
    raw = b"".join(b"\x00" + big_endian[row].tobytes(order="C") for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_tiff16(path: Path, pixels: NDArray[np.uint16]) -> None:
    """Write a baseline, uncompressed, chunky little-endian 16-bit RGB TIFF."""

    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("TIFF writer requires RGB pixels")
    tags: list[tuple[int, int, int, bytes | int]] = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 3, struct.pack("<3H", 16, 16, 16)),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (273, 4, 1, 0),  # strip offset is filled after the extra-value area is known
        (274, 3, 1, 1),
        (277, 3, 1, 3),
        (278, 4, 1, height),
        (279, 4, 1, int(width * height * 3 * 2)),
        (282, 5, 1, struct.pack("<2I", 72, 1)),
        (283, 5, 1, struct.pack("<2I", 72, 1)),
        (284, 3, 1, 1),
        (296, 3, 1, 2),
        (339, 3, 3, struct.pack("<3H", 1, 1, 1)),
    ]
    tags.sort(key=lambda item: item[0])
    ifd_size = 2 + len(tags) * 12 + 4
    extra_start = 8 + ifd_size
    extra = bytearray()
    encoded_entries: list[tuple[int, int, int, bytes]] = []
    for tag, kind, count, value in tags:
        if isinstance(value, int):
            if kind == 3:
                inline = struct.pack("<H", value) + b"\x00\x00"
            else:
                inline = struct.pack("<I", value)
            encoded_entries.append((tag, kind, count, inline))
            continue
        if len(extra) % 2:
            extra.append(0)
        offset = extra_start + len(extra)
        extra.extend(value)
        encoded_entries.append((tag, kind, count, struct.pack("<I", offset)))
    if len(extra) % 2:
        extra.append(0)
    pixel_offset = extra_start + len(extra)
    encoded_entries = [
        (tag, kind, count, struct.pack("<I", pixel_offset) if tag == 273 else value)
        for tag, kind, count, value in encoded_entries
    ]
    with path.open("xb") as stream:
        stream.write(b"II")
        stream.write(struct.pack("<HI", 42, 8))
        stream.write(struct.pack("<H", len(encoded_entries)))
        for tag, kind, count, value in encoded_entries:
            stream.write(struct.pack("<HHI", tag, kind, count))
            stream.write(value)
        stream.write(struct.pack("<I", 0))
        stream.write(extra)
        stream.write(np.asarray(pixels, dtype="<u2").tobytes(order="C"))
        stream.flush()
        os.fsync(stream.fileno())


def _write_linear_fits(path: Path, rgb: NDArray[np.float32], reference: _Channel, has_luminance: bool) -> None:
    header = reference.wcs.to_header(relax=True)
    for key in ("OBJECT", "INSTRUME", "DATE-OBS"):
        if key in reference.header:
            header[key] = reference.header[key]
    header["OAFSTATE"] = ("SOLVED", "Color product retains verified common WCS")
    header["OAFWCS"] = ("SOLVED", "Five-point channel WCS agreement verified")
    header["OAFPROD"] = ("LINEAR_RGB", "Ultra-Fast WBPP product kind")
    header["OAFVERS"] = (COLOR_PRODUCT_VERSION, "Ultra-Fast WBPP color product version")
    header["OAFRGB"] = ("RGB", "Plane order")
    header["OAFLUM"] = (bool(has_luminance), "Luminance replacement applied")
    header.add_history("Ultra-Fast WBPP: R/G/B celestial WCS agreed at corners and centre")
    fits.PrimaryHDU(data=np.asarray(rgb, dtype=np.float32), header=header).writeto(
        path, overwrite=False, checksum=True, output_verify="exception"
    )
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _artifact(path: Path, root: Path, kind: str) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "kind": kind,
        "sha256": sha256_digest(path),
        "sizeBytes": info.st_size,
    }


def build_color_product(
    request: ColorProductRequest,
    *,
    publisher: DirectoryPublisher = _publish_directory_no_replace,
) -> ColorProductResult:
    """Build and atomically publish a solved linear RGB product and previews."""

    output, tolerance, preview_edge, softness = _validate_request(request)
    product_name = _product_name(request)
    values = {"R": request.red_path, "G": request.green_path, "B": request.blue_path}
    if request.luminance_path is not None:
        values["L"] = request.luminance_path
    channels = {name: _read_channel(name, value) for name, value in values.items()}
    if len({str(channel.path) for channel in channels.values()}) != len(channels):
        raise ColorProductError("DUPLICATE_CHANNEL", "each channel must name a distinct input file")
    reference = channels["R"]
    for channel in channels.values():
        if channel.shape != reference.shape:
            raise ColorProductError(
                "CHANNEL_SHAPE_MISMATCH",
                f"{channel.name} shape {channel.shape} differs from R shape {reference.shape}",
                path=str(channel.path),
            )
    wcs_evidence = [
        _wcs_consistency(reference, channels[name], tolerance_pixels=tolerance)
        for name in (*_CHANNELS[1:], *(("L",) if "L" in channels else ()))
    ]
    rgb = np.stack([channels[name].data for name in _CHANNELS]).astype(np.float32, copy=False)
    luminance_evidence: dict[str, Any] | None = None
    if "L" in channels:
        rgb, luminance_evidence = _matched_luminance_rgb(rgb, channels["L"].data)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ColorProductError("OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(output))
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        linear = staging / f"{product_name}.fits"
        tiff = staging / f"{product_name}.tiff"
        png = staging / f"{product_name}.png"
        receipt_path = staging / "receipt.json"
        _write_linear_fits(linear, rgb, reference, "L" in channels)
        preview_rgb, block_size = _block_mean_rgb(rgb, preview_edge)
        preview_pixels, stretch = _color_preserving_asinh(preview_rgb, softness)
        _write_tiff16(tiff, preview_pixels)
        _write_png16(png, preview_pixels)
        artifacts = [
            _artifact(linear, staging, "LINEAR_RGB_FITS"),
            _artifact(tiff, staging, "ASINH_RGB_TIFF_16"),
            _artifact(png, staging, "ASINH_RGB_PNG_16"),
        ]
        receipt_core: dict[str, Any] = {
            "schemaVersion": 1,
            "productVersion": COLOR_PRODUCT_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "state": "SOLVED",
            "inputs": [channels[name].identity for name in (*_CHANNELS, *(("L",) if "L" in channels else ()))],
            "geometry": {"shape": list(reference.shape), "channelOrder": list(_CHANNELS)},
            "wcs": {
                "state": "SOLVED",
                "referenceChannel": "R",
                "referenceValidation": reference.wcs_validation,
                "channelValidations": {
                    name: channels[name].wcs_validation for name in channels
                },
                "fivePointConsistency": wcs_evidence,
            },
            "luminance": luminance_evidence,
            "preview": {
                **stretch,
                "blockSize": block_size,
                "shape": list(preview_pixels.shape[:2]),
                "colorPreserving": True,
            },
            "artifacts": artifacts,
            "publication": {
                "atomic": True,
                "noReplace": True,
                "receiptIsCommitMarker": True,
                "parentDirectoryFsync": "BEST_EFFORT_AFTER_COMMIT",
            },
        }
        receipt_id = "sha256:" + hashlib.sha256(canonical_json_document(receipt_core)).hexdigest()
        receipt = {"receiptId": receipt_id, **receipt_core}
        with receipt_path.open("xb") as stream:
            stream.write(canonical_json_document(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        for channel in channels.values():
            _verify_channel_stat(channel)
        _fsync_directory(staging)
        try:
            publisher(staging, output)
        except ColorProductError:
            raise
        except Exception as error:
            raise ColorProductError("ATOMIC_PUBLICATION_FAILED", str(error), path=str(output)) from error
        _best_effort_fsync_directory(output.parent)
    finally:
        remove_tree(staging)

    return ColorProductResult(
        output_directory=str(output),
        linear_rgb_path=str(output / linear.name),
        preview_tiff_path=str(output / tiff.name),
        preview_png_path=str(output / png.name),
        receipt_path=str(output / "receipt.json"),
        receipt=receipt,
    )


# Descriptive alias for GUI/recipe callers.
create_color_product = build_color_product


__all__ = [
    "COLOR_PRODUCT_VERSION",
    "ColorProductError",
    "ColorProductRequest",
    "ColorProductResult",
    "build_color_product",
    "create_color_product",
]
