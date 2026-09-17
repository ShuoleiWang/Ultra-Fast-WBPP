"""Minimal reader/writer for monolithic XISF 1.0 files (MIT, no external codec).

Only what the pipeline needs from the container format is implemented: the
XML header of a monolithic ``.xisf`` file, the image metadata of every
``<Image>`` element, attachment-backed pixel blocks with the standard codecs
(zlib, lz4, lz4hc, zstd, each optionally byte-shuffled), and a writer for
planar single-block images. The public surface mirrors the subset of the
``xisf`` package that the code base used, so callers and fixtures are
unchanged:

* ``XISF(path).get_images_metadata()`` returns one ``dict`` per image with
  ``geometry`` (``(width, height, channels)``), ``location``
  (``("attachment", position, size)``), ``compression``
  (``(codec, uncompressed_size, shuffle_item_size)``) when present, ``dtype``
  (NumPy dtype of one sample), the plain string attributes of the element,
  ``FITSKeywords`` (``{name: [{"value": ..., "comment": ...}, ...]}``) and
  ``XISFProperties`` (``{id: {"id": ..., "type": ..., "value": ...}}``).
* ``XISF(path).get_file_metadata()`` returns the ``<Metadata>`` properties.
* ``XISF(path).read_image(index, data_format="channels_last")`` decodes one
  image to an ``(height, width, channels)`` array of the file's sample type.
* ``XISF.write(path, array, ...)`` writes ``(height, width, channels)``
  arrays as a monolithic file and returns ``(bytes_written, codec)``.

The header is parsed with the standard library after a fail-closed check
that it declares no DTD or entities; decoders are bounded by the declared
uncompressed size. See https://pixinsight.com/doc/docs/XISF-1.0-spec/ for
the format.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import struct
from typing import Any, Mapping
import xml.etree.ElementTree as ElementTree
import zlib

import numpy as np
from numpy.typing import NDArray

__all__ = ["XISF", "XisfError"]

SIGNATURE = b"XISF0100"
NAMESPACE = "http://www.pixinsight.com/xisf"
_NS = f"{{{NAMESPACE}}}"
DEFAULT_MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_PROPERTY_BLOCK_BYTES = 64 * 1024 * 1024
BLOCK_ALIGNMENT = 4096

_SAMPLE_FORMATS: dict[str, str] = {
    "UInt8": "u1",
    "UInt16": "u2",
    "UInt32": "u4",
    "UInt64": "u8",
    "Float32": "f4",
    "Float64": "f8",
    "Complex32": "c8",
    "Complex64": "c16",
}
_DTYPE_TO_SAMPLE_FORMAT: dict[str, str] = {
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
    "complex64": "Complex32",
    "complex128": "Complex64",
}
_CODECS = ("zlib", "lz4", "lz4hc", "zstd")


class XisfError(ValueError):
    """The file is not a monolithic XISF 1.0 file this reader accepts."""


def _sample_dtype(sample_format: str, byte_order: str | None) -> np.dtype[Any]:
    try:
        code = _SAMPLE_FORMATS[sample_format]
    except KeyError as error:
        raise XisfError(f"unsupported sampleFormat {sample_format!r}") from error
    order = "<" if byte_order in (None, "little") else ">" if byte_order == "big" else None
    if order is None:
        raise XisfError(f"unsupported byteOrder {byte_order!r}")
    if code == "u1":
        return np.dtype(code)
    return np.dtype(order + code)


def _parse_location(value: str) -> tuple[Any, ...]:
    parts = value.split(":")
    kind = parts[0]
    if kind == "attachment":
        if len(parts) != 3:
            raise XisfError(f"malformed attachment location {value!r}")
        try:
            return ("attachment", int(parts[1]), int(parts[2]))
        except ValueError as error:
            raise XisfError(f"malformed attachment location {value!r}") from error
    if kind in ("inline", "embedded"):
        return tuple(parts)
    if kind in ("url", "path"):
        return tuple(parts)
    raise XisfError(f"unsupported location {value!r}")


def _parse_compression(value: str) -> tuple[str, int, int | None]:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise XisfError(f"malformed compression {value!r}")
    codec = parts[0]
    base = codec[:-3] if codec.endswith("+sh") else codec
    if base not in _CODECS:
        raise XisfError(f"unsupported compression codec {codec!r}")
    try:
        size = int(parts[1])
        item = int(parts[2]) if len(parts) == 3 else None
    except ValueError as error:
        raise XisfError(f"malformed compression {value!r}") from error
    if codec.endswith("+sh") and item is None:
        raise XisfError(f"shuffled compression without item size {value!r}")
    return codec, size, item


_SCALAR_PROPERTY_TYPES: dict[str, str] = {
    "Int8": "i1", "Int16": "i2", "Int32": "i4", "Int64": "i8",
    "UInt8": "u1", "UInt16": "u2", "UInt32": "u4", "UInt64": "u8",
    "Float32": "f4", "Float64": "f8", "Complex32": "c8", "Complex64": "c16",
}
_ARRAY_PROPERTY_ITEMS: dict[str, str] = {
    "I8": "i1", "I16": "i2", "I32": "i4", "I64": "i8",
    "UI8": "u1", "UI16": "u2", "UI32": "u4", "UI64": "u8",
    "F32": "f4", "F64": "f8", "C32": "c8", "C64": "c16",
}


def _keyword_value(value: str) -> str:
    """A FITS keyword value with one pair of surrounding quotes removed."""

    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def _decode_inline(element: ElementTree.Element, encoding: str) -> bytes:
    text = "".join((element.text or "").split())
    try:
        if encoding == "base64":
            return base64.b64decode(text, validate=True)
        if encoding == "hex":
            return binascii.unhexlify(text)
    except (binascii.Error, ValueError) as error:
        raise XisfError(f"{encoding} data block is malformed") from error
    raise XisfError(f"unsupported inline encoding {encoding!r}")


def _property(element: ElementTree.Element, reader: XISF | None = None) -> dict[str, Any]:
    """A ``<Property>`` as ``{"id", "type", "value", ...}``; scalar values are
    typed (``int``/``float``/``bool``/``complex``), strings and time points
    stay strings, vectors and matrices become NumPy arrays."""

    entry: dict[str, Any] = {key: value for key, value in element.attrib.items()}
    kind = str(entry.get("type", "String"))
    raw = entry.get("value")
    if kind in _SCALAR_PROPERTY_TYPES and raw is not None:
        try:
            if kind.startswith("Float"):
                entry["value"] = float(raw)
            elif kind.startswith("Complex"):
                entry["value"] = complex(raw)
            else:
                entry["value"] = int(raw)
        except ValueError:
            entry["value"] = raw
    elif kind == "Boolean" and raw is not None:
        entry["value"] = str(raw).strip().lower() in ("1", "true")
    elif kind.endswith(("Vector", "Matrix")) and kind[:-6] in _ARRAY_PROPERTY_ITEMS:
        item = _ARRAY_PROPERTY_ITEMS[kind[:-6]]
        parts = str(entry.get("location", "")).split(":")
        for key in ("length", "rows", "columns"):
            if key in entry:
                try:
                    entry[key] = int(entry[key])
                except ValueError:
                    pass
        entry["dtype"] = np.dtype(item)
        data: bytes | None = None
        if parts[0] == "attachment":
            entry["location"] = _parse_location(str(entry.get("location", "")))
            if reader is not None and int(entry["location"][2]) <= MAX_PROPERTY_BLOCK_BYTES:
                data = reader._attachment(int(entry["location"][1]), int(entry["location"][2]))
        else:
            if "location" in entry:
                entry["location"] = parts
            if parts[0] == "inline" and len(parts) == 2:
                data = _decode_inline(element, parts[1])
            elif parts[0] == "embedded":
                child = element.find(f"{_NS}Data")
                if child is None:
                    child = element.find("Data")
                if child is not None:
                    data = _decode_inline(child, child.attrib.get("encoding", "base64"))
        if data is None:
            entry["value"] = None
        else:
            values = np.frombuffer(data, dtype=np.dtype("<" + item))
            if kind.endswith("Matrix"):
                try:
                    values = values.reshape(int(entry.get("rows", 0)), int(entry.get("columns", 0)))
                except ValueError:
                    pass
            entry["value"] = values
    elif raw is None:
        # A string property carries its value as text; an empty element reads
        # as ``None``, as the previous reader reported it.
        entry["value"] = None if "location" in entry else element.text
    return entry


def _unshuffle(data: bytes, item_size: int) -> bytes:
    if item_size <= 0 or len(data) % item_size:
        raise XisfError("shuffle item size does not divide the block")
    return np.frombuffer(data, dtype=np.uint8).reshape(item_size, -1).T.tobytes()


def _shuffle(data: bytes, item_size: int) -> bytes:
    if item_size <= 0 or len(data) % item_size:
        raise XisfError("shuffle item size does not divide the block")
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, item_size).T.tobytes()


def _decompress(data: bytes, codec: str, size: int) -> bytes:
    base = codec[:-3] if codec.endswith("+sh") else codec
    if base == "zlib":
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(data, size + 1)
        if len(decoded) > size or decoder.unconsumed_tail:
            raise XisfError("zlib block exceeds its declared size")
        decoded += decoder.flush(size + 1 - len(decoded))
        if len(decoded) != size or not decoder.eof:
            raise XisfError("zlib block differs from its declared size")
        return decoded
    if base in ("lz4", "lz4hc"):
        import lz4.block

        decoded = lz4.block.decompress(data, uncompressed_size=size)
    elif base == "zstd":
        import zstandard

        decoded = zstandard.ZstdDecompressor().decompress(data, max_output_size=size)
    else:  # pragma: no cover - guarded by _parse_compression
        raise XisfError(f"unsupported compression codec {codec!r}")
    if len(decoded) != size:
        raise XisfError(f"{base} block differs from its declared size")
    return decoded


def _compress(data: bytes, codec: str, level: int | None) -> bytes:
    if codec == "zlib":
        return zlib.compress(data, 6 if level is None else level)
    if codec in ("lz4", "lz4hc"):
        import lz4.block

        if codec == "lz4hc":
            return lz4.block.compress(
                data, mode="high_compression", compression=9 if level is None else level,
                store_size=False,
            )
        return lz4.block.compress(data, store_size=False)
    if codec == "zstd":
        import zstandard

        return zstandard.ZstdCompressor(level=3 if level is None else level).compress(data)
    raise XisfError(f"unsupported compression codec {codec!r}")


class XISF:
    """A monolithic XISF file: parsed header plus on-demand pixel decoding."""

    def __init__(self, path: str | os.PathLike[str], *, max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES) -> None:
        self._path = Path(path)
        self._images: list[dict[str, Any]] = []
        self._file_metadata: dict[str, dict[str, Any]] = {}
        self._elements: list[ElementTree.Element] = []
        self._read_header(max_header_bytes)

    # ------------------------------------------------------------------ header
    def _read_header(self, max_header_bytes: int) -> None:
        with self._path.open("rb") as stream:
            prefix = stream.read(16)
            if len(prefix) != 16 or prefix[:8] != SIGNATURE:
                raise XisfError("not a monolithic XISF 1.0 file")
            (header_length,) = struct.unpack("<I", prefix[8:12])
            if header_length == 0 or header_length > max_header_bytes:
                raise XisfError(f"XISF header length {header_length} is outside the accepted range")
            header = stream.read(header_length)
        if len(header) != header_length:
            raise XisfError("XISF header is truncated")
        lowered = header.lower()
        if b"<!doctype" in lowered or b"<!entity" in lowered:
            raise XisfError("XISF header declares a DTD or entities")
        try:
            root = ElementTree.fromstring(header.decode("utf-8").lstrip("﻿"))
        except (ElementTree.ParseError, UnicodeDecodeError) as error:
            raise XisfError(f"XISF header is not well-formed XML: {error}") from error
        if root.tag not in (f"{_NS}xisf", "xisf"):
            raise XisfError("XISF header root element is not <xisf>")
        namespace = _NS if root.tag.startswith(_NS) else ""
        metadata = root.find(f"{namespace}Metadata")
        if metadata is not None:
            for child in metadata.findall(f"{namespace}Property"):
                entry = _property(child, self)
                self._file_metadata[str(entry.get("id"))] = entry
        for element in root.findall(f"{namespace}Image"):
            self._elements.append(element)
            self._images.append(self._image_metadata(element, namespace))

    def _image_metadata(self, element: ElementTree.Element, namespace: str) -> dict[str, Any]:
        metadata: dict[str, Any] = dict(element.attrib)
        try:
            geometry = tuple(int(value) for value in str(metadata["geometry"]).split(":"))
        except (KeyError, ValueError) as error:
            raise XisfError("XISF image geometry is missing or malformed") from error
        if len(geometry) < 2 or any(value <= 0 for value in geometry):
            raise XisfError(f"unsupported XISF geometry {metadata.get('geometry')!r}")
        metadata["geometry"] = geometry
        if "sampleFormat" not in metadata:
            raise XisfError("XISF image declares no sampleFormat")
        metadata["dtype"] = _sample_dtype(str(metadata["sampleFormat"]), metadata.get("byteOrder"))
        if "location" in metadata:
            metadata["location"] = _parse_location(str(metadata["location"]))
        if "compression" in metadata:
            metadata["compression"] = _parse_compression(str(metadata["compression"]))
        keywords: dict[str, list[dict[str, str]]] = {}
        for keyword in element.findall(f"{namespace}FITSKeyword"):
            name = keyword.attrib.get("name")
            if name is None:
                continue
            keywords.setdefault(name, []).append(
                {
                    "value": _keyword_value(keyword.attrib.get("value", "")),
                    "comment": keyword.attrib.get("comment", ""),
                }
            )
        properties: dict[str, dict[str, Any]] = {}
        for child in element.findall(f"{namespace}Property"):
            entry = _property(child, self)
            properties[str(entry.get("id"))] = entry
        metadata["FITSKeywords"] = keywords
        metadata["XISFProperties"] = properties
        return metadata

    def _attachment(self, position: int, size: int) -> bytes:
        with self._path.open("rb") as stream:
            stream.seek(position)
            data = stream.read(size)
        if len(data) != size:
            raise XisfError("XISF attachment is truncated")
        return data

    # ------------------------------------------------------------------ public
    def get_images_metadata(self) -> list[dict[str, Any]]:
        return [dict(image) for image in self._images]

    def get_file_metadata(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._file_metadata.items()}

    def read_image(self, n: int = 0, data_format: str = "channels_last") -> NDArray[Any]:
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        if n < 0 or n >= len(self._images):
            raise IndexError(f"image index {n} is outside [0, {len(self._images) - 1}]")
        metadata = self._images[n]
        geometry = metadata["geometry"]
        width, height = int(geometry[0]), int(geometry[1])
        channels = int(geometry[2]) if len(geometry) > 2 else 1
        dtype: np.dtype[Any] = metadata["dtype"]
        expected = width * height * channels * dtype.itemsize
        raw = self._block(self._elements[n], metadata, expected)
        samples = np.frombuffer(raw, dtype=dtype)
        if metadata.get("pixelStorage", "planar") == "normal":
            array = samples.reshape(height, width, channels)
        else:
            array = samples.reshape(channels, height, width).transpose(1, 2, 0)
        if data_format == "channels_first":
            array = np.moveaxis(array, -1, 0)
        return np.ascontiguousarray(array)

    def _block(self, element: ElementTree.Element, metadata: Mapping[str, Any], expected: int) -> bytes:
        location = metadata.get("location")
        if not location:
            raise XisfError("XISF image has no data location")
        compression = metadata.get("compression")
        stored_size = int(compression[1]) if compression else expected
        if location[0] == "attachment":
            data = self._attachment(int(location[1]), int(location[2]))
        elif location[0] == "inline":
            data = _decode_inline(element, str(location[1]) if len(location) > 1 else "base64")
        elif location[0] == "embedded":
            data_element = element.find(f"{_NS}Data")
            if data_element is None:
                data_element = element.find("Data")
            if data_element is None:
                raise XisfError("embedded XISF image has no <Data> element")
            data = _decode_inline(data_element, data_element.attrib.get("encoding", "base64"))
        else:
            raise XisfError(f"unsupported XISF data location {location[0]!r}")
        if compression:
            codec, uncompressed_size, item_size = compression
            if uncompressed_size != expected:
                raise XisfError("declared uncompressed size disagrees with the image geometry")
            data = _decompress(data, str(codec), int(uncompressed_size))
            if item_size is not None:
                data = _unshuffle(data, int(item_size))
        if len(data) != expected or stored_size != expected:
            raise XisfError("XISF data block size disagrees with the image geometry")
        return data

    # ------------------------------------------------------------------ writer
    @staticmethod
    def write(
        fname: str | os.PathLike[str],
        im_data: NDArray[Any],
        creator_app: str | None = None,
        image_metadata: Mapping[str, Any] | None = None,
        xisf_metadata: Mapping[str, Any] | None = None,
        codec: str | None = None,
        shuffle: bool = False,
        level: int | None = None,
    ) -> tuple[int, str | None]:
        """Write ``im_data`` (``height, width, channels``) as a planar image.

        ``image_metadata`` may carry plain string attributes of the
        ``<Image>`` element (``id``, ``imageType``, ...), ``FITSKeywords``
        (``{name: [{"value": ..., "comment": ...}]}``) and ``XISFProperties``
        (``{id: {"type": ..., "value": ...}}``); ``xisf_metadata`` carries
        file-level properties in the same layout. Returns the number of bytes
        written and the codec recorded in the header (``None`` when the block
        is stored as is).
        """

        array = np.asarray(im_data)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3:
            raise ValueError("im_data must have shape (height, width, channels)")
        sample_format = _DTYPE_TO_SAMPLE_FORMAT.get(array.dtype.name)
        if sample_format is None:
            raise ValueError(f"unsupported dtype {array.dtype} for XISF")
        height, width, channels = array.shape
        planar = np.ascontiguousarray(array.transpose(2, 0, 1)).astype(array.dtype.newbyteorder("<"), copy=False)
        raw = planar.tobytes()
        used_codec: str | None = None
        block = raw
        if codec is not None:
            if codec not in _CODECS:
                raise ValueError(f"unsupported codec {codec!r}")
            used_codec = codec
            if shuffle and array.dtype.itemsize > 1:
                used_codec = f"{codec}+sh"
                block = _shuffle(raw, array.dtype.itemsize)
            block = _compress(block, codec, level)

        image = ElementTree.Element("Image")
        attributes: dict[str, str] = {}
        keywords: Mapping[str, Any] = {}
        properties: Mapping[str, Any] = {}
        for key, value in (image_metadata or {}).items():
            if key == "FITSKeywords":
                keywords = value
            elif key == "XISFProperties":
                properties = value
            elif key in ("geometry", "sampleFormat", "location", "compression", "dtype", "pixelStorage", "byteOrder"):
                continue
            else:
                attributes[str(key)] = str(value)
        image.set("geometry", f"{width}:{height}:{channels}")
        image.set("sampleFormat", sample_format)
        if sample_format.startswith("Float") and "bounds" not in attributes:
            image.set("bounds", "0:1")
        if "colorSpace" not in attributes:
            image.set("colorSpace", "Gray" if channels == 1 else "RGB")
        for key, value in attributes.items():
            image.set(key, value)
        if used_codec is not None:
            suffix = f":{array.dtype.itemsize}" if used_codec.endswith("+sh") else ""
            image.set("compression", f"{used_codec}:{len(raw)}{suffix}")
        for name, entries in keywords.items():
            for entry in entries:
                keyword = ElementTree.SubElement(image, "FITSKeyword")
                keyword.set("name", str(name))
                keyword.set("value", str(entry.get("value", "")))
                keyword.set("comment", str(entry.get("comment", "")))
        for identifier, entry in properties.items():
            _append_property(image, str(identifier), entry)

        root = ElementTree.Element("xisf")
        root.set("version", "1.0")
        root.set("xmlns", NAMESPACE)
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:schemaLocation", f"{NAMESPACE} http://pixinsight.com/xisf/xisf-1.0.xsd")
        root.append(image)
        metadata = ElementTree.SubElement(root, "Metadata")
        file_properties: dict[str, dict[str, Any]] = {
            "XISF:CreationTime": {"type": "String", "value": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
            "XISF:CreatorApplication": {"type": "String", "value": creator_app or "lightframeqc.xisf"},
            "XISF:CreatorOS": {"type": "String", "value": platform.system() or "unknown"},
            "XISF:BlockAlignmentSize": {"type": "UInt16", "value": str(BLOCK_ALIGNMENT)},
        }
        for identifier, entry in (xisf_metadata or {}).items():
            file_properties[str(identifier)] = dict(entry)
        for identifier, entry in file_properties.items():
            _append_property(metadata, identifier, entry)

        # The attachment position depends on the header length, which
        # depends on the position; a fixed-width placeholder resolves it.
        placeholder = "attachment:{position:020d}:{size}"
        image.set("location", placeholder.format(position=0, size=len(block)))
        header = _serialize(root)
        header_length = len(header)
        position = -(-(16 + header_length) // BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT
        image.set("location", placeholder.format(position=position, size=len(block)))
        header = _serialize(root)
        if len(header) != header_length:  # pragma: no cover - fixed-width placeholder
            raise RuntimeError("XISF header length changed while resolving the attachment position")
        padding = position - 16 - header_length
        with Path(fname).open("wb") as stream:
            stream.write(SIGNATURE)
            stream.write(struct.pack("<I", header_length))
            stream.write(b"\0\0\0\0")
            stream.write(header)
            stream.write(b"\0" * padding)
            stream.write(block)
        return position + len(block), used_codec


def _append_property(parent: ElementTree.Element, identifier: str, entry: Mapping[str, Any]) -> None:
    element = ElementTree.SubElement(parent, "Property")
    element.set("id", identifier)
    element.set("type", str(entry.get("type", "String")))
    value = entry.get("value")
    if str(entry.get("type", "String")) == "String":
        element.text = "" if value is None else str(value)
    elif value is not None:
        element.set("value", str(value))
    for key in ("comment", "format"):
        if key in entry and entry[key] is not None:
            element.set(key, str(entry[key]))


def _serialize(root: ElementTree.Element) -> bytes:
    body = ElementTree.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + body).encode("utf-8")
