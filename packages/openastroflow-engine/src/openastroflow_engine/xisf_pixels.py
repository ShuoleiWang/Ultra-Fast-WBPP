"""Fail-closed XISF-to-FITS pixel staging for the portable engine.

XISF is a container format, while the portable calibration kernels consume
read-only, row-addressable FITS images.  This module is the deliberately narrow
bridge between those two boundaries.  It accepts exactly one two-dimensional
mono/CFA image, validates the declared decoded size before allocation, and
creates a private Float32 FITS staging copy on the same filesystem as the run.

Uncompressed attachments are streamed by rows.  Compressed attachments use
bounded decoders for the codecs supported by ``xisf`` (zlib, lz4 and zstd).
Inline/embedded blocks, multiple images and complex/multichannel samples remain
fail-closed; header-only inventory support must never imply pixel support.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping
import zlib

import lz4.block
import numpy as np
import zstandard
from xisf import XISF

from .calibration import CalibrationError, FitsFloatWriter


XISF_PIXEL_BRIDGE_VERSION = "xisf-private-fits-v2"


@dataclass(frozen=True, slots=True)
class XisfDecodePolicy:
    """Explicit image-size and working-set ceilings for XISF pixel execution."""

    max_decoded_image_bytes: int = 2 * 1024**3
    max_compressed_working_set_bytes: int = 768 * 1024**2
    max_compression_ratio: float = 512.0
    tile_rows: int = 256
    max_xml_header_bytes: int = 16 * 1024**2

    def validate(self) -> None:
        for name in (
            "max_decoded_image_bytes",
            "max_compressed_working_set_bytes",
            "tile_rows",
            "max_xml_header_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not math.isfinite(self.max_compression_ratio)
            or self.max_compression_ratio < 1.0
        ):
            raise ValueError("max_compression_ratio must be finite and at least 1")

    def serializable(self) -> dict[str, Any]:
        return {
            "bridgeVersion": XISF_PIXEL_BRIDGE_VERSION,
            "maxDecodedImageBytes": self.max_decoded_image_bytes,
            "maxCompressedWorkingSetBytes": self.max_compressed_working_set_bytes,
            "maxCompressionRatio": self.max_compression_ratio,
            "tileRows": self.tile_rows,
            "maxXmlHeaderBytes": self.max_xml_header_bytes,
            "acceptedGeometry": "single-image-2d-mono-or-cfa",
            "acceptedCompression": ["none", "zlib", "lz4", "zstd"],
        }


@dataclass(frozen=True, slots=True)
class XisfConversionReceipt:
    source_sha256: str
    converted_sha256: str
    source_size_bytes: int
    converted_size_bytes: int
    width: int
    height: int
    source_dtype: str
    source_sample_format: str
    source_bounds: str | None
    source_bounds_parsed: tuple[float, float] | None
    numeric_domain: str
    normalized_unit_scale: float | None
    compression: str
    decoder: str
    decoded_image_bytes: int
    peak_working_set_bound_bytes: int
    image_id: str
    image_type: str
    container_image_count: int
    ignored_auxiliary_images: tuple[dict[str, Any], ...]

    def serializable(self) -> dict[str, Any]:
        return {
            "bridgeVersion": XISF_PIXEL_BRIDGE_VERSION,
            "sourceSha256": self.source_sha256,
            "convertedSha256": self.converted_sha256,
            "sourceSizeBytes": self.source_size_bytes,
            "convertedSizeBytes": self.converted_size_bytes,
            "geometry": [self.height, self.width],
            "sourceDtype": self.source_dtype,
            "sourceSampleFormat": self.source_sample_format,
            "sourceBounds": self.source_bounds,
            "sourceBoundsParsed": (
                list(self.source_bounds_parsed)
                if self.source_bounds_parsed is not None
                else None
            ),
            "numericDomain": self.numeric_domain,
            "normalizedUnitScale": self.normalized_unit_scale,
            "compression": self.compression,
            "decoder": self.decoder,
            "decodedImageBytes": self.decoded_image_bytes,
            "peakWorkingSetBoundBytes": self.peak_working_set_bound_bytes,
            "selectedImage": {"id": self.image_id, "type": self.image_type},
            "containerImageCount": self.container_image_count,
            "ignoredAuxiliaryImages": [dict(item) for item in self.ignored_auxiliary_images],
            "sourceReadOnly": True,
            "stagingFormat": "FITS_FLOAT32",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat(follow_symlinks=False)
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _preflight_xml_header(source: Path, policy: XisfDecodePolicy) -> None:
    """Bound XML parsing and reject DTD/entity expansion before ``xisf`` sees it."""

    try:
        with source.open("rb") as stream:
            prefix = stream.read(16)
            if len(prefix) != 16 or prefix[:8] != b"XISF0100":
                raise CalibrationError(
                    "XISF_SIGNATURE_INVALID", "not an XISF 1.0 file", path=str(source)
                )
            header_size = struct.unpack("<I", prefix[8:12])[0]
            if header_size < 16 or header_size > policy.max_xml_header_bytes:
                raise CalibrationError(
                    "XISF_XML_HEADER_BUDGET",
                    f"XML header size {header_size} exceeds the accepted bound",
                    path=str(source),
                )
            if 16 + header_size > source.stat().st_size:
                raise CalibrationError(
                    "XISF_XML_HEADER_TRUNCATED", "XML header escapes the source file", path=str(source)
                )
            header = stream.read(header_size)
    except CalibrationError:
        raise
    except OSError as error:
        raise CalibrationError("XISF_HEADER_ERROR", str(error), path=str(source)) from error
    if len(header) != header_size:
        raise CalibrationError(
            "XISF_XML_HEADER_TRUNCATED", "XML header is truncated", path=str(source)
        )
    folded = header.upper()
    if b"<!DOCTYPE" in folded or b"<!ENTITY" in folded:
        raise CalibrationError(
            "XISF_XML_DTD_FORBIDDEN",
            "DOCTYPE and ENTITY declarations are forbidden",
            path=str(source),
        )
    if b"\x00" in header:
        raise CalibrationError(
            "XISF_XML_HEADER_INVALID", "XML header contains NUL bytes", path=str(source)
        )


def preflight_xisf_header(
    source_path: str | os.PathLike[str], policy: XisfDecodePolicy | None = None
) -> None:
    """Apply the bounded XML/DTD gate without decoding an image block."""

    selected = policy or XisfDecodePolicy()
    selected.validate()
    source = Path(source_path).expanduser().resolve(strict=True)
    _preflight_xml_header(source, selected)


def _checked_product(*values: int) -> int:
    result = 1
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CalibrationError("XISF_GEOMETRY_INVALID", "invalid XISF geometry")
        result *= value
        if result > 2**63 - 1:
            raise CalibrationError("XISF_GEOMETRY_OVERFLOW", "XISF geometry overflows")
    return result


def _file_dtype(metadata: Mapping[str, Any]) -> np.dtype[Any]:
    try:
        dtype = np.dtype(metadata["dtype"])
    except Exception as error:
        raise CalibrationError("XISF_SAMPLE_FORMAT_INVALID", str(error)) from error
    if dtype.kind not in {"u", "i", "f"} or dtype.itemsize not in {1, 2, 4, 8}:
        raise CalibrationError(
            "XISF_SAMPLE_FORMAT_UNSUPPORTED",
            f"real integer/float samples are required, found {dtype}",
        )
    if dtype.itemsize == 1:
        return dtype
    byte_order = str(metadata.get("byteOrder") or "little").casefold()
    if byte_order not in {"little", "big"}:
        raise CalibrationError(
            "XISF_BYTE_ORDER_UNSUPPORTED", f"unsupported byte order {byte_order!r}"
        )
    return dtype.newbyteorder("<" if byte_order == "little" else ">")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _xisf_numeric_domain(
    image: Mapping[str, Any], dtype: np.dtype[Any]
) -> tuple[str, float | None, str | None, tuple[float, float] | None]:
    raw_bounds = image.get("bounds")
    bounds = str(raw_bounds).strip() if raw_bounds is not None else None
    parsed: tuple[float, float] | None = None
    if bounds is not None:
        pieces = bounds.split(":")
        if len(pieces) != 2 or any(not piece.strip() for piece in pieces):
            raise CalibrationError(
                "XISF_BOUNDS_INVALID",
                f"bounds must contain exactly two finite endpoints, found {bounds!r}",
            )
        try:
            low, high = (float(piece.strip()) for piece in pieces)
        except ValueError as error:
            raise CalibrationError(
                "XISF_BOUNDS_INVALID",
                f"bounds must contain exactly two finite endpoints, found {bounds!r}",
            ) from error
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise CalibrationError(
                "XISF_BOUNDS_INVALID",
                f"bounds must be finite and strictly increasing, found {bounds!r}",
            )
        parsed = (low, high)
    if dtype.kind == "f":
        if parsed is None or not (
            math.isclose(parsed[0], 0.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(parsed[1], 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise CalibrationError(
                "XISF_FLOAT_BOUNDS_UNSUPPORTED",
                "Float XISF calibration inputs must explicitly declare bounds 0:1",
            )
        return "NORMALIZED_UNIT", 1.0, bounds, parsed
    if dtype.kind in {"u", "i"}:
        bits = dtype.itemsize * 8
        expected = float((1 << bits) - 1)
        if dtype.kind == "u" and (
            parsed is None
            or (
                math.isclose(parsed[0], 0.0, rel_tol=0.0, abs_tol=1e-9)
                and math.isclose(parsed[1], expected, rel_tol=0.0, abs_tol=0.5)
            )
        ):
            return "SENSOR_CODE", expected, bounds, parsed
    return "UNDECLARED", None, bounds, parsed


def _fits_metadata(
    image: Mapping[str, Any],
    source_sha256: str,
    dtype: np.dtype[Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keywords = image.get("FITSKeywords", {})
    # Only acquisition/geometry metadata needed by the engine crosses the
    # staging boundary.  In particular COMMENT, HISTORY and PixInsight's
    # ProcessingHistory are not copied because they can contain private paths.
    safe_keywords = {
        "IMAGETYP", "FILTER", "OBJECT", "INSTRUME", "EXPTIME", "EXPOSURE",
        "CCD-TEMP", "CCD_TEMP", "SENSORT", "SENSOR-T", "CAMTEMP", "GAIN",
        "EGAIN", "CAMGAIN", "OFFSET", "CAMOFFSET", "XBINNING", "YBINNING",
        "CCDBINX", "CCDBINY", "BINNING", "BAYERPAT", "BAYERPATN", "CFAPAT",
        "CFAPATTERN", "READOUTM", "READOUT", "READMODE", "READOUTMODE",
        "DATE-OBS", "DATE-END", "RA", "DEC", "OBJCTRA", "OBJCTDEC", "EQUINOX",
        "FOCALLEN", "XPIXSZ", "YPIXSZ", "TELESCOP",
    }
    if isinstance(keywords, Mapping):
        for raw_key, entries in keywords.items():
            key = str(raw_key).strip().upper()
            if key not in safe_keywords:
                continue
            if isinstance(entries, list) and entries and isinstance(entries[0], Mapping):
                value = entries[0].get("value")
                if value is not None and isinstance(
                    value, (str, bool, int, float, np.generic)
                ):
                    result[key] = _plain(value)

    properties = image.get("XISFProperties", {})
    scalar_properties: dict[str, Any] = {}
    if isinstance(properties, Mapping):
        for identifier, description in properties.items():
            if not isinstance(description, Mapping) or "value" not in description:
                continue
            value = description["value"]
            if isinstance(value, np.ndarray) and value.ndim != 0:
                continue
            if isinstance(value, (str, bool, int, float, np.generic)):
                scalar_properties[str(identifier)] = _plain(value)
    aliases = {
        "Instrument:ExposureTime": "EXPTIME",
        "Instrument:FrameExposureTime": "EXPTIME",
        "Instrument:Camera:XBinning": "XBINNING",
        "Instrument:Camera:YBinning": "YBINNING",
        "Instrument:Camera:Name": "INSTRUME",
        "Instrument:Camera:Gain": "GAIN",
        "Instrument:Camera:Offset": "OFFSET",
        "Instrument:Camera:ReadoutMode": "READOUTM",
        "Instrument:Filter:Name": "FILTER",
        "Observation:Object:Name": "OBJECT",
        "Observation:Time:Start": "DATE-OBS",
    }
    for identifier, keyword in aliases.items():
        if keyword not in result and identifier in scalar_properties:
            result[keyword] = scalar_properties[identifier]
    image_type = image.get("imageType")
    if "IMAGETYP" not in result and image_type is not None:
        result["IMAGETYP"] = str(image_type)
    result["OAFXISF"] = True
    result["OAFXSHA"] = source_sha256.removeprefix("sha256:")[:16]
    result["OAFXSFMT"] = str(image.get("sampleFormat") or "")
    numeric_domain, normalized_unit_scale, bounds, _parsed_bounds = _xisf_numeric_domain(
        image, dtype
    )
    result["OAFNDOM"] = numeric_domain
    if normalized_unit_scale is not None:
        result["OAFNSCL"] = normalized_unit_scale
    if bounds is not None:
        result["OAFXBD"] = bounds
    return result


def _compact_image_type(value: Any) -> str:
    return "".join(character for character in str(value or "").upper() if character.isalnum())


def _declared_image_type(image: Mapping[str, Any]) -> str:
    value = image.get("imageType")
    if value is not None and str(value).strip():
        return str(value)
    keywords = image.get("FITSKeywords", {})
    if isinstance(keywords, Mapping):
        entries = keywords.get("IMAGETYP", keywords.get("IMAGETYPE"))
        if isinstance(entries, list) and entries and isinstance(entries[0], Mapping):
            candidate = entries[0].get("value")
            if candidate is not None:
                return str(candidate).strip().strip("'\"")
    return ""


def _select_science_image(
    images: list[Mapping[str, Any]], source: Path
) -> tuple[Mapping[str, Any], tuple[dict[str, Any], ...]]:
    if not images:
        raise CalibrationError("XISF_IMAGE_MISSING", "XISF contains no image", path=str(source))
    science_types = {
        "LIGHT", "LIGHTFRAME", "FLAT", "FLATFRAME", "DARK", "DARKFRAME",
        "BIAS", "BIASFRAME", "MASTERLIGHT", "MASTERFLAT", "MASTERDARK",
        "MASTERBIAS",
    }
    auxiliary_types = {
        "REJECTIONMAPLOW", "REJECTIONMAPHIGH", "REJECTIONMAP", "WEIGHTMAP",
        "COVERAGEMAP",
    }
    candidates = [
        image for image in images
        if _compact_image_type(_declared_image_type(image)) in science_types
    ]
    if len(images) == 1 and not candidates:
        candidates = [images[0]]
    if len(candidates) != 1:
        raise CalibrationError(
            "XISF_SCIENCE_IMAGE_AMBIGUOUS",
            f"expected one science image, found {len(candidates)} among {len(images)} images",
            path=str(source),
        )
    selected = candidates[0]
    auxiliaries: list[dict[str, Any]] = []
    for image in images:
        if image is selected:
            continue
        image_type = _compact_image_type(_declared_image_type(image))
        if image_type not in auxiliary_types:
            raise CalibrationError(
                "XISF_MULTI_IMAGE_UNSUPPORTED",
                "non-science XISF images must be recognized auxiliary maps",
                path=str(source),
            )
        geometry = image.get("geometry")
        auxiliaries.append(
            {
                "id": str(image.get("id") or ""),
                "type": _declared_image_type(image),
                "geometry": [int(value) for value in geometry]
                if isinstance(geometry, (tuple, list))
                else [],
            }
        )
    return selected, tuple(auxiliaries)


def _attachment(metadata: Mapping[str, Any], source: Path) -> tuple[int, int]:
    location = metadata.get("location")
    if not isinstance(location, (tuple, list)) or len(location) != 3:
        raise CalibrationError(
            "XISF_STORAGE_UNSUPPORTED", "XISF image location is malformed", path=str(source)
        )
    if str(location[0]).casefold() != "attachment":
        raise CalibrationError(
            "XISF_STORAGE_UNSUPPORTED",
            "only attachment-backed XISF pixels are supported",
            path=str(source),
        )
    try:
        offset, size = int(location[1]), int(location[2])
    except (TypeError, ValueError) as error:
        raise CalibrationError(
            "XISF_STORAGE_INVALID", "attachment offset/size is invalid", path=str(source)
        ) from error
    if offset < 0 or size < 1 or offset + size > source.stat().st_size:
        raise CalibrationError(
            "XISF_STORAGE_INVALID", "attachment escapes the source file", path=str(source)
        )
    return offset, size


def _bounded_zlib(data: bytes, expected_size: int) -> bytes:
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(data, expected_size + 1)
    if len(decoded) > expected_size or decoder.unconsumed_tail:
        raise CalibrationError(
            "XISF_COMPRESSION_BOMB", "zlib output exceeds the declared decoded size"
        )
    decoded += decoder.flush(expected_size + 1 - len(decoded))
    if len(decoded) != expected_size or not decoder.eof:
        raise CalibrationError(
            "XISF_DECODE_SIZE_MISMATCH", "zlib output differs from the declared size"
        )
    return decoded


def _decode_compressed(
    data: bytes, metadata: Mapping[str, Any], expected_size: int
) -> tuple[bytes, str]:
    compression = metadata.get("compression")
    if not isinstance(compression, (tuple, list)) or len(compression) != 3:
        raise CalibrationError(
            "XISF_COMPRESSION_INVALID", "compression metadata is malformed"
        )
    codec = str(compression[0]).casefold()
    try:
        declared_size = int(compression[1])
    except (TypeError, ValueError) as error:
        raise CalibrationError(
            "XISF_COMPRESSION_INVALID", "decoded size is invalid"
        ) from error
    if declared_size != expected_size:
        raise CalibrationError(
            "XISF_DECODE_SIZE_MISMATCH",
            "compression decoded size disagrees with image geometry",
        )
    try:
        if codec.startswith("zlib"):
            decoded = _bounded_zlib(data, expected_size)
            decoder = "python-zlib-bounded"
        elif codec.startswith("lz4"):
            decoded = lz4.block.decompress(data, uncompressed_size=expected_size)
            decoder = "python-lz4-block-bounded"
        elif codec.startswith("zstd"):
            decoded = zstandard.ZstdDecompressor().decompress(
                data, max_output_size=expected_size
            )
            decoder = "python-zstandard-bounded"
        else:
            raise CalibrationError(
                "XISF_COMPRESSION_UNSUPPORTED", f"unsupported XISF codec {codec!r}"
            )
    except CalibrationError:
        raise
    except Exception as error:
        raise CalibrationError("XISF_DECODE_FAILED", str(error)) from error
    if len(decoded) != expected_size:
        raise CalibrationError(
            "XISF_DECODE_SIZE_MISMATCH",
            "decoder output differs from the declared size",
        )
    item_size = compression[2]
    if item_size is not None:
        try:
            item_size = int(item_size)
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                "XISF_SHUFFLE_INVALID", "shuffle item size is invalid"
            ) from error
        if item_size not in {1, 2, 4, 8} or len(decoded) % item_size:
            raise CalibrationError(
                "XISF_SHUFFLE_INVALID", "shuffle item size is incompatible"
            )
        shuffled = np.frombuffer(decoded, dtype=np.uint8).reshape((item_size, -1))
        decoded = shuffled.T.tobytes()
    return decoded, decoder


def _write_rows(
    writer: FitsFloatWriter,
    raw: bytes | memoryview,
    dtype: np.dtype[Any],
    *,
    width: int,
    height: int,
    tile_rows: int,
) -> None:
    samples = np.frombuffer(raw, dtype=dtype, count=width * height)
    if samples.size != width * height:
        raise CalibrationError("XISF_DECODE_SIZE_MISMATCH", "pixel block is truncated")
    image = samples.reshape((height, width))
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        writer.write_rows(y0, image[y0:y1])


def convert_xisf_to_fits(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
    *,
    policy: XisfDecodePolicy | None = None,
) -> XisfConversionReceipt:
    """Create a private FITS staging image without modifying the XISF source."""

    policy = policy or XisfDecodePolicy()
    policy.validate()
    source = Path(source_path).expanduser().resolve(strict=True)
    destination = Path(destination_path).expanduser().resolve(strict=False)
    if source.suffix.casefold() != ".xisf":
        raise CalibrationError("PIXEL_FORMAT_UNSUPPORTED", "source is not XISF", path=str(source))
    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError("OUTPUT_EXISTS", "refusing to overwrite staging FITS", path=str(destination))
    before = _stat_identity(source)
    source_sha256 = _sha256(source)
    _preflight_xml_header(source, policy)
    try:
        document = XISF(str(source))
        images = document.get_images_metadata()
    except Exception as error:
        raise CalibrationError("XISF_HEADER_ERROR", str(error), path=str(source)) from error
    metadata, auxiliary_images = _select_science_image(images, source)
    try:
        geometry = tuple(int(value) for value in metadata["geometry"])
    except Exception as error:
        raise CalibrationError("XISF_GEOMETRY_INVALID", str(error), path=str(source)) from error
    if len(geometry) != 3:
        raise CalibrationError(
            "XISF_GEOMETRY_UNSUPPORTED", "expected W:H:C geometry", path=str(source)
        )
    width, height, channels = geometry
    if channels != 1:
        raise CalibrationError(
            "XISF_CHANNELS_UNSUPPORTED",
            "pixel execution accepts mono/CFA images with exactly one stored channel",
            path=str(source),
        )
    dtype = _file_dtype(metadata)
    decoded_bytes = _checked_product(width, height, channels, dtype.itemsize)
    if decoded_bytes > policy.max_decoded_image_bytes:
        raise CalibrationError(
            "XISF_DECODE_BUDGET_EXCEEDED",
            f"declared image needs {decoded_bytes} bytes; limit is {policy.max_decoded_image_bytes}",
            path=str(source),
        )
    offset, stored_size = _attachment(metadata, source)
    compression = metadata.get("compression")
    compression_name = "none"
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists() or os.path.lexists(temporary):
        raise CalibrationError("OUTPUT_EXISTS", "staging temporary already exists", path=str(temporary))
    destination.parent.mkdir(parents=True, exist_ok=True)
    decoder_name = "attachment-row-stream"
    peak_bound = min(decoded_bytes, policy.tile_rows * width * dtype.itemsize) + (
        policy.tile_rows * width * np.dtype(np.float32).itemsize
    )
    try:
        with FitsFloatWriter(
            temporary,
            (height, width),
            _fits_metadata(metadata, source_sha256, dtype),
        ) as writer:
            if compression is None:
                if stored_size < decoded_bytes:
                    raise CalibrationError(
                        "XISF_DECODE_SIZE_MISMATCH",
                        "uncompressed attachment is smaller than declared pixels",
                        path=str(source),
                    )
                row_bytes = width * dtype.itemsize
                rows = max(1, min(policy.tile_rows, height))
                with source.open("rb") as stream:
                    for y0 in range(0, height, rows):
                        y1 = min(height, y0 + rows)
                        stream.seek(offset + y0 * row_bytes)
                        raw = stream.read((y1 - y0) * row_bytes)
                        if len(raw) != (y1 - y0) * row_bytes:
                            raise CalibrationError(
                                "XISF_DECODE_SIZE_MISMATCH",
                                "uncompressed attachment is truncated",
                                path=str(source),
                            )
                        values = np.frombuffer(raw, dtype=dtype).reshape((y1 - y0, width))
                        writer.write_rows(y0, values)
            else:
                codec = str(compression[0]).casefold() if isinstance(compression, (tuple, list)) and compression else "unknown"
                compression_name = codec
                ratio = decoded_bytes / stored_size
                if ratio > policy.max_compression_ratio:
                    raise CalibrationError(
                        "XISF_COMPRESSION_BOMB",
                        f"declared compression ratio {ratio:.1f} exceeds {policy.max_compression_ratio:.1f}",
                        path=str(source),
                    )
                # Input bytes + decoded bytes + an unshuffle/decode copy.  The
                # Float32 FITS destination is memory-mapped and written in tiles.
                peak_bound = stored_size + 2 * decoded_bytes + min(
                    decoded_bytes, policy.tile_rows * width * np.dtype(np.float32).itemsize
                )
                if peak_bound > policy.max_compressed_working_set_bytes:
                    raise CalibrationError(
                        "XISF_DECODE_BUDGET_EXCEEDED",
                        f"compressed decode bound is {peak_bound} bytes; limit is "
                        f"{policy.max_compressed_working_set_bytes}",
                        path=str(source),
                    )
                with source.open("rb") as stream:
                    stream.seek(offset)
                    raw = stream.read(stored_size)
                if len(raw) != stored_size:
                    raise CalibrationError(
                        "XISF_DECODE_SIZE_MISMATCH", "compressed attachment is truncated", path=str(source)
                    )
                decoded, decoder_name = _decode_compressed(raw, metadata, decoded_bytes)
                _write_rows(
                    writer,
                    decoded,
                    dtype,
                    width=width,
                    height=height,
                    tile_rows=policy.tile_rows,
                )
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise CalibrationError("OUTPUT_EXISTS", "refusing to overwrite staging FITS", path=str(destination)) from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    after = _stat_identity(source)
    if before != after or _sha256(source) != source_sha256:
        if destination.exists():
            destination.unlink()
        raise CalibrationError("SOURCE_CHANGED", "XISF source changed during decode", path=str(source))
    (
        numeric_domain,
        normalized_unit_scale,
        source_bounds,
        source_bounds_parsed,
    ) = _xisf_numeric_domain(
        metadata, dtype
    )
    return XisfConversionReceipt(
        source_sha256=source_sha256,
        converted_sha256=_sha256(destination),
        source_size_bytes=before[0],
        converted_size_bytes=destination.stat().st_size,
        width=width,
        height=height,
        source_dtype=str(dtype),
        source_sample_format=str(metadata.get("sampleFormat") or ""),
        source_bounds=source_bounds,
        source_bounds_parsed=source_bounds_parsed,
        numeric_domain=numeric_domain,
        normalized_unit_scale=normalized_unit_scale,
        compression=compression_name,
        decoder=decoder_name,
        decoded_image_bytes=decoded_bytes,
        peak_working_set_bound_bytes=peak_bound,
        image_id=str(metadata.get("id") or ""),
        image_type=_declared_image_type(metadata),
        container_image_count=len(images),
        ignored_auxiliary_images=auxiliary_images,
    )


__all__ = [
    "XISF_PIXEL_BRIDGE_VERSION",
    "XisfConversionReceipt",
    "XisfDecodePolicy",
    "convert_xisf_to_fits",
    "preflight_xisf_header",
]
