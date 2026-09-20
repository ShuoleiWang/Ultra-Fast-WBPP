"""Read-only FITS/XISF preview readers.

The quality-control pipeline never needs a full-resolution working copy of an
uncompressed frame.  This module therefore reduces images to a block-mean
preview while reading a bounded number of source rows at a time.  The same
integer block size is used on both axes, so preview coordinates retain the
source aspect ratio and have a simple, explicit scale.

Compressed FITS HDUs and compressed/inline XISF blocks cannot always be
streamed by their Python decoders.  Those paths are guarded by
``max_full_decode_bytes`` and fail with a specific error instead of silently
allocating an unbounded array.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
from .xisf import XISF

from .cfa import even_block_size, is_cfa_pattern
from .metadata import _header_lookup, normalize_metadata
from .models import FrameMetadata


DEFAULT_PREVIEW_LONG_EDGE = 2048
DEFAULT_MAX_FULL_DECODE_BYTES = 512 * 1024 * 1024

_FITS_ENDINGS = (".fit", ".fits", ".fts", ".fit.fz", ".fits.fz", ".fts.fz")
_XISF_ENDINGS = (".xisf",)


class FrameReadError(RuntimeError):
    """Base class for a frame that cannot be decoded safely.

    ``code`` is stable and suitable for a report or CLI exit diagnostic.  The
    exception message deliberately includes the affected path.
    """

    def __init__(self, code: str, path: str | os.PathLike[str], detail: str) -> None:
        self.code = code
        self.path = str(path)
        self.detail = detail
        super().__init__(f"{code}: {self.path}: {detail}")


class UnsupportedFrameFormatError(FrameReadError):
    """The input has no supported FITS/XISF filename ending."""


class FrameMemoryLimitError(FrameReadError):
    """A decoder would have to exceed the configured full-decode limit."""


class FrameDiscoveryError(FrameReadError):
    """An explicit discovery input is missing or otherwise invalid."""


@dataclass(frozen=True, slots=True)
class ImagePreview:
    """A read-only, native-endian, two-dimensional Float32 preview.

    ``data`` is always luminance in preview-pixel coordinates.  Mono inputs are
    unchanged apart from block averaging; RGB/multi-channel inputs use the
    finite mean of their channels.  ``block_size`` is the source-pixel width
    and height represented by a normal preview pixel.  Edge pixels may cover a
    smaller partial block.
    """

    path: str
    data: NDArray[np.float32]
    metadata: FrameMetadata
    source_width: int
    source_height: int
    source_channels: int
    block_size: int
    reader_backend: str
    image_index: int = 0

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("preview data must be two-dimensional")
        if self.data.dtype != np.dtype(np.float32):
            raise ValueError("preview data must have dtype float32")
        if not self.data.flags.c_contiguous:
            raise ValueError("preview data must be C-contiguous")
        if self.source_width < 1 or self.source_height < 1 or self.source_channels < 1:
            raise ValueError("source geometry must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")

    @property
    def pixels(self) -> NDArray[np.float32]:
        """Alias used by callers that prefer an image-oriented name."""

        return self.data

    @property
    def preview_width(self) -> int:
        return int(self.data.shape[1])

    @property
    def preview_height(self) -> int:
        return int(self.data.shape[0])

    @property
    def scale_x(self) -> float:
        return self.source_width / self.preview_width

    @property
    def scale_y(self) -> float:
        return self.source_height / self.preview_height

    def as_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        """Return report-friendly metadata, optionally including the ndarray."""

        value: dict[str, Any] = {
            "path": self.path,
            "sourceWidth": self.source_width,
            "sourceHeight": self.source_height,
            "sourceChannels": self.source_channels,
            "previewWidth": self.preview_width,
            "previewHeight": self.preview_height,
            "blockSize": self.block_size,
            "scaleX": self.scale_x,
            "scaleY": self.scale_y,
            "readerBackend": self.reader_backend,
            "imageIndex": self.image_index,
            "metadata": self.metadata.serializable(),
        }
        if include_data:
            value["data"] = self.data
        return value


def _format_name(path: Path) -> str | None:
    name = path.name.casefold()
    if name.endswith(_XISF_ENDINGS):
        return "xisf"
    if name.endswith(_FITS_ENDINGS):
        return "fits"
    return None


def is_supported_frame_path(path: str | os.PathLike[str]) -> bool:
    """Return whether ``path`` has a supported, case-insensitive ending."""

    return _format_name(Path(path)) is not None


def _input_sequence(
    inputs: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
) -> Sequence[str | os.PathLike[str]]:
    if isinstance(inputs, (str, os.PathLike)):
        return (inputs,)
    return tuple(inputs)


def discover_paths(
    inputs: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
) -> list[Path]:
    """Discover FITS/XISF files recursively with deterministic ordering.

    Unsupported files encountered *inside* a directory are ignored.  An
    explicitly supplied unsupported file is an error, since silently omitting
    it would make a CLI request look successful while losing user intent.
    Directory symlinks are not traversed.  Canonical file paths are deduplicated.
    """

    requested = _input_sequence(inputs)
    if not requested:
        raise FrameDiscoveryError("NO_INPUTS", "<inputs>", "no input paths were supplied")

    discovered: dict[str, Path] = {}
    for raw_path in requested:
        expanded = Path(raw_path).expanduser()
        if not expanded.exists():
            raise FrameDiscoveryError("INPUT_NOT_FOUND", expanded, "path does not exist")
        if expanded.is_file():
            if not is_supported_frame_path(expanded):
                raise UnsupportedFrameFormatError(
                    "UNSUPPORTED_FORMAT",
                    expanded,
                    "expected .fit/.fits/.fts[.fz] or .xisf",
                )
            canonical = expanded.resolve(strict=True)
            discovered[os.path.normcase(str(canonical))] = canonical
            continue
        if not expanded.is_dir():
            raise FrameDiscoveryError(
                "INPUT_NOT_REGULAR", expanded, "path is neither a regular file nor directory"
            )

        for root, directory_names, file_names in os.walk(expanded, followlinks=False):
            root_path = Path(root)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if not (root_path / name).is_symlink()
            )
            for name in sorted(file_names):
                candidate = root_path / name
                if not is_supported_frame_path(candidate) or not candidate.is_file():
                    continue
                canonical = candidate.resolve(strict=True)
                discovered[os.path.normcase(str(canonical))] = canonical

    if not discovered:
        raise FrameDiscoveryError(
            "NO_SUPPORTED_FRAMES", "<inputs>", "no FITS or XISF images were found"
        )
    return sorted(discovered.values(), key=lambda path: os.path.normcase(str(path)))


def _validated_path(path: str | os.PathLike[str]) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.exists():
        raise FrameReadError("INPUT_NOT_FOUND", expanded, "path does not exist")
    if not expanded.is_file():
        raise FrameReadError("INPUT_NOT_FILE", expanded, "path is not a regular file")
    return expanded.resolve(strict=True)


def _validated_limits(max_long_edge: int, max_full_decode_bytes: int | None) -> None:
    if isinstance(max_long_edge, bool) or max_long_edge < 1:
        raise ValueError("max_long_edge must be a positive integer")
    if max_full_decode_bytes is not None:
        if isinstance(max_full_decode_bytes, bool) or max_full_decode_bytes < 1:
            raise ValueError("max_full_decode_bytes must be positive or None")


def _block_size(width: int, height: int, max_long_edge: int) -> int:
    return max(1, math.ceil(max(width, height) / max_long_edge))


_CFA_HEADER_KEYS = ("BAYERPAT", "BAYERPATN", "CFAPAT", "CFAPATTERN", "PCL:CFASourcePattern")


def _preview_block_size(width: int, height: int, max_long_edge: int, header: Mapping[str, Any]) -> int:
    """Block size of a preview; even for a Bayer mosaic so every preview
    pixel averages the same number of red, green and blue samples and the
    preview is a luminance image rather than a colour-aliased one."""

    factor = _block_size(width, height, max_long_edge)
    pattern = _header_lookup({str(k): v for k, v in header.items()}, _CFA_HEADER_KEYS)
    if is_cfa_pattern(pattern):
        return even_block_size(factor)
    return factor


RowsReader = Callable[[int, int], NDArray[np.float64]]


def _block_mean_preview(
    read_rows: RowsReader,
    *,
    width: int,
    height: int,
    block_size: int,
    max_work_bytes: int = 32 * 1024 * 1024,
) -> NDArray[np.float32]:
    """Reduce a source through a bounded-row callback, ignoring non-finite pixels."""

    output_height = math.ceil(height / block_size)
    output_width = math.ceil(width / block_size)
    output = np.full((output_height, output_width), np.nan, dtype=np.float32)
    x_starts = np.arange(0, width, block_size, dtype=np.int64)

    bytes_per_output_row = max(1, width * block_size * np.dtype(np.float64).itemsize)
    rows_per_chunk = max(1, min(64, max_work_bytes // bytes_per_output_row))
    for output_y0 in range(0, output_height, rows_per_chunk):
        output_y1 = min(output_height, output_y0 + rows_per_chunk)
        source_y0 = output_y0 * block_size
        source_y1 = min(height, output_y1 * block_size)
        band = np.asarray(read_rows(source_y0, source_y1), dtype=np.float64)
        expected_shape = (source_y1 - source_y0, width)
        if band.shape != expected_shape:
            raise ValueError(
                f"row reader returned {band.shape}, expected {expected_shape}"
            )

        # Every complete block row of the band reduces at once; the vertical
        # sums accumulate the block's rows in order and the column segments
        # reduce in order, exactly as the one-row-at-a-time loop did, so the
        # means are value-identical.  A trailing partial block row (image
        # height not a multiple of the block size) takes the same path alone.
        rows_in_band = band.shape[0]
        complete_rows = rows_in_band // block_size
        pieces = []
        if complete_rows:
            pieces.append(band[: complete_rows * block_size].reshape(complete_rows, block_size, width))
        if complete_rows * block_size < rows_in_band:
            pieces.append(band[complete_rows * block_size :][None, :, :])
        next_output_y = output_y0
        for source_blocks in pieces:
            finite = np.isfinite(source_blocks)
            vertical_sum = np.sum(
                np.where(finite, source_blocks, 0.0), axis=1, dtype=np.float64
            )
            vertical_count = np.sum(finite, axis=1, dtype=np.int64)
            block_sum = np.add.reduceat(vertical_sum, x_starts, axis=1)
            block_count = np.add.reduceat(vertical_count, x_starts, axis=1)
            reduced = np.full(block_sum.shape, np.nan, dtype=np.float64)
            np.divide(block_sum, block_count, out=reduced, where=block_count > 0)
            count = reduced.shape[0]
            output[next_output_y : next_output_y + count] = reduced.astype(np.float32, copy=False)
            next_output_y += count

    return np.ascontiguousarray(output)


def _finite_channel_mean(array: NDArray[Any], axis: int) -> NDArray[np.float64]:
    values = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(values)
    total = np.sum(np.where(finite, values, 0.0), axis=axis, dtype=np.float64)
    count = np.sum(finite, axis=axis, dtype=np.int64)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _fits_header_dict(header: fits.Header) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for card in header.cards:
        key = str(card.keyword).strip()
        if not key or key in {"COMMENT", "HISTORY"}:
            continue
        result[key] = _plain_value(card.value)
    return result


def _fits_layout(shape: tuple[int, ...], path: Path) -> tuple[int, int, int, str]:
    if len(shape) == 2:
        height, width = shape
        return int(width), int(height), 1, "mono"
    if len(shape) != 3:
        raise FrameReadError(
            "UNSUPPORTED_GEOMETRY", path, f"FITS image has unsupported shape {shape}"
        )
    if 1 <= shape[0] <= 4:
        channels, height, width = shape
        return int(width), int(height), int(channels), "channels_first"
    if 1 <= shape[-1] <= 4:
        height, width, channels = shape
        return int(width), int(height), int(channels), "channels_last"
    raise FrameReadError(
        "UNSUPPORTED_GEOMETRY",
        path,
        f"cannot identify the channel axis in FITS shape {shape}",
    )


def _fits_image_candidates(
    hdul: fits.HDUList,
) -> list[tuple[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU, tuple[int, ...]]]:
    candidates: list[tuple[Any, tuple[int, ...]]] = []
    for hdu in hdul:
        naxis = int(hdu.header.get("NAXIS", 0) or 0)
        if naxis not in {2, 3}:
            continue
        # FITS stores NAXIS1 first; NumPy exposes the reversed shape.
        shape = tuple(
            int(hdu.header.get(f"NAXIS{axis}", 0) or 0)
            for axis in range(naxis, 0, -1)
        )
        if all(dimension > 0 for dimension in shape):
            candidates.append((hdu, shape))
    return candidates


def _select_fits_hdu(
    hdul: fits.HDUList, image_index: int, path: Path
) -> tuple[
    fits.hdu.base.ExtensionHDU | fits.PrimaryHDU,
    tuple[int, ...],
    int,
]:
    candidates = _fits_image_candidates(hdul)
    if not candidates:
        raise FrameReadError("NO_IMAGE", path, "FITS file contains no 2-D image HDU")
    if image_index < 0 or image_index >= len(candidates):
        raise FrameReadError(
            "IMAGE_INDEX_RANGE",
            path,
            f"image_index {image_index} is outside [0, {len(candidates) - 1}]",
        )
    hdu, shape = candidates[image_index]
    return hdu, shape, len(candidates)


def _read_fits_preview(
    path: Path,
    *,
    max_long_edge: int,
    image_index: int,
    max_full_decode_bytes: int | None,
) -> ImagePreview:
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            lazy_load_hdus=True,
            do_not_scale_image_data=True,
            uint=False,
            checksum=False,
        ) as hdul:
            hdu, declared_shape, image_count = _select_fits_hdu(
                hdul, image_index, path
            )
            bitpix = abs(int(hdu.header.get("BITPIX", 0) or 0))
            estimated_bytes = math.prod(declared_shape) * max(1, bitpix // 8)
            if isinstance(hdu, fits.CompImageHDU):
                if (
                    max_full_decode_bytes is not None
                    and estimated_bytes > max_full_decode_bytes
                ):
                    raise FrameMemoryLimitError(
                        "FULL_DECODE_LIMIT",
                        path,
                        f"compressed FITS image needs about {estimated_bytes} decoded bytes; "
                        f"limit is {max_full_decode_bytes}",
                    )

            data = hdu.data
            if data is None:
                raise FrameReadError("NO_IMAGE", path, "selected FITS HDU has no data")
            width, height, channels, layout = _fits_layout(tuple(data.shape), path)
            header = _fits_header_dict(hdu.header)
            factor = _preview_block_size(width, height, max_long_edge, header)
            bscale = float(hdu.header.get("BSCALE", 1.0) or 1.0)
            bzero = float(hdu.header.get("BZERO", 0.0) or 0.0)
            blank = hdu.header.get("BLANK")

            def read_rows(y0: int, y1: int) -> NDArray[np.float64]:
                if layout == "mono":
                    raw = np.asanyarray(data[y0:y1, :])
                    values = np.asarray(raw, dtype=np.float64)
                    if blank is not None and np.issubdtype(raw.dtype, np.integer):
                        values[np.asarray(raw) == blank] = np.nan
                elif layout == "channels_first":
                    raw = np.asanyarray(data[:, y0:y1, :])
                    values = np.asarray(raw, dtype=np.float64)
                    if blank is not None and np.issubdtype(raw.dtype, np.integer):
                        values[np.asarray(raw) == blank] = np.nan
                    values = _finite_channel_mean(values, axis=0)
                else:
                    raw = np.asanyarray(data[y0:y1, :, :])
                    values = np.asarray(raw, dtype=np.float64)
                    if blank is not None and np.issubdtype(raw.dtype, np.integer):
                        values[np.asarray(raw) == blank] = np.nan
                    values = _finite_channel_mean(values, axis=2)
                if bscale != 1.0 or bzero != 0.0:
                    values = values * bscale + bzero
                return values

            preview = _block_mean_preview(
                read_rows,
                width=width,
                height=height,
                block_size=factor,
            )
            metadata = normalize_metadata(
                FrameMetadata(
                    path=str(path),
                    width=width,
                    height=height,
                    channels=channels,
                    image_count=image_count,
                    header=header,
                )
            )
            backend = "astropy-fits-compressed" if isinstance(hdu, fits.CompImageHDU) else "astropy-fits-memmap"
    except FrameReadError:
        raise
    except Exception as error:
        raise FrameReadError("FITS_DECODE_ERROR", path, str(error)) from error

    preview.setflags(write=False)
    return ImagePreview(
        path=str(path),
        data=preview,
        metadata=metadata,
        source_width=width,
        source_height=height,
        source_channels=channels,
        block_size=factor,
        reader_backend=backend,
        image_index=image_index,
    )


def _xisf_header_dict(image_metadata: Mapping[str, Any]) -> dict[str, Any]:
    header: dict[str, Any] = {}
    fits_keywords = image_metadata.get("FITSKeywords", {})
    if isinstance(fits_keywords, Mapping):
        for key, entries in fits_keywords.items():
            if isinstance(entries, list) and entries:
                entry = entries[0]
                if isinstance(entry, Mapping) and "value" in entry:
                    header[str(key)] = _plain_value(entry["value"])

    properties = image_metadata.get("XISFProperties", {})
    property_values: dict[str, Any] = {}
    if isinstance(properties, Mapping):
        for identifier, description in properties.items():
            if not isinstance(description, Mapping) or "value" not in description:
                continue
            value = description["value"]
            if isinstance(value, np.ndarray) and value.ndim != 0:
                continue
            property_values[str(identifier)] = _plain_value(value)

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
        if keyword not in header and identifier in property_values:
            header[keyword] = property_values[identifier]
    for identifier, value in property_values.items():
        header.setdefault(identifier, value)

    header["XISF:SAMPLEFORMAT"] = _plain_value(image_metadata.get("sampleFormat"))
    header["XISF:COLORSPACE"] = _plain_value(image_metadata.get("colorSpace"))
    if image_metadata.get("imageType") is not None:
        header["XISF:IMAGETYPE"] = _plain_value(image_metadata.get("imageType"))
    return header


def _xisf_file_dtype(metadata: Mapping[str, Any]) -> np.dtype[Any]:
    dtype = np.dtype(metadata["dtype"])
    if dtype.itemsize == 1:
        return dtype
    byte_order = str(metadata.get("byteOrder", "little")).casefold()
    if byte_order not in {"little", "big"}:
        raise ValueError(f"unsupported XISF byteOrder {byte_order!r}")
    return dtype.newbyteorder("<" if byte_order == "little" else ">")


def _read_xisf_preview(
    path: Path,
    *,
    max_long_edge: int,
    image_index: int,
    max_full_decode_bytes: int | None,
) -> ImagePreview:
    try:
        document = XISF(str(path))
        images = document.get_images_metadata()
        if not images:
            raise FrameReadError("NO_IMAGE", path, "XISF file contains no image")
        if image_index < 0 or image_index >= len(images):
            raise FrameReadError(
                "IMAGE_INDEX_RANGE",
                path,
                f"image_index {image_index} is outside [0, {len(images) - 1}]",
            )
        image_metadata = images[image_index]
        geometry = tuple(int(value) for value in image_metadata["geometry"])
        if len(geometry) != 3:
            raise FrameReadError(
                "UNSUPPORTED_GEOMETRY", path, f"XISF geometry is {geometry}, expected W:H:C"
            )
        width, height, channels = geometry
        if width < 1 or height < 1 or channels < 1:
            raise FrameReadError(
                "UNSUPPORTED_GEOMETRY", path, f"invalid XISF geometry {geometry}"
            )
        dtype = _xisf_file_dtype(image_metadata)
        decoded_bytes = width * height * channels * dtype.itemsize
        header = _xisf_header_dict(image_metadata)
        factor = _preview_block_size(width, height, max_long_edge, header)
        location = tuple(image_metadata["location"])
        direct_attachment = location and location[0] == "attachment" and "compression" not in image_metadata

        if direct_attachment:
            _, offset, stored_size = location
            if int(stored_size) < decoded_bytes:
                raise FrameReadError(
                    "TRUNCATED_IMAGE",
                    path,
                    f"XISF attachment has {stored_size} bytes, expected at least {decoded_bytes}",
                )
            with path.open("rb") as stream:

                def read_rows(y0: int, y1: int) -> NDArray[np.float64]:
                    row_count = y1 - y0
                    channel_bytes = row_count * width * dtype.itemsize
                    total = np.zeros((row_count, width), dtype=np.float64)
                    count = np.zeros((row_count, width), dtype=np.int16)
                    for channel in range(channels):
                        position = (
                            int(offset)
                            + channel * height * width * dtype.itemsize
                            + y0 * width * dtype.itemsize
                        )
                        stream.seek(position)
                        raw = stream.read(channel_bytes)
                        if len(raw) != channel_bytes:
                            raise FrameReadError(
                                "TRUNCATED_IMAGE",
                                path,
                                f"short XISF attachment read at byte {position}",
                            )
                        values = np.frombuffer(raw, dtype=dtype).reshape(row_count, width)
                        numeric = np.asarray(values, dtype=np.float64)
                        finite = np.isfinite(numeric)
                        total += np.where(finite, numeric, 0.0)
                        count += finite
                    result = np.full((row_count, width), np.nan, dtype=np.float64)
                    np.divide(total, count, out=result, where=count > 0)
                    return result

                preview = _block_mean_preview(
                    read_rows,
                    width=width,
                    height=height,
                    block_size=factor,
                )
            backend = "xisf-python-attachment-stream"
        else:
            if max_full_decode_bytes is not None and decoded_bytes > max_full_decode_bytes:
                storage = "compressed" if "compression" in image_metadata else str(location[0])
                raise FrameMemoryLimitError(
                    "FULL_DECODE_LIMIT",
                    path,
                    f"{storage} XISF image needs about {decoded_bytes} decoded bytes; "
                    f"limit is {max_full_decode_bytes}",
                )
            decoded = np.asarray(document.read_image(image_index, data_format="channels_last"))
            if decoded.shape != (height, width, channels):
                raise FrameReadError(
                    "DECODED_GEOMETRY_MISMATCH",
                    path,
                    f"decoder returned {decoded.shape}, expected {(height, width, channels)}",
                )

            def read_decoded_rows(y0: int, y1: int) -> NDArray[np.float64]:
                return _finite_channel_mean(decoded[y0:y1, :, :], axis=2)

            preview = _block_mean_preview(
                read_decoded_rows,
                width=width,
                height=height,
                block_size=factor,
            )
            backend = "xisf-python-full-decode"

        metadata = normalize_metadata(
            FrameMetadata(
                path=str(path),
                width=width,
                height=height,
                channels=channels,
                image_count=len(images),
                header=header,
            )
        )
    except FrameReadError:
        raise
    except Exception as error:
        raise FrameReadError("XISF_DECODE_ERROR", path, str(error)) from error

    preview.setflags(write=False)
    return ImagePreview(
        path=str(path),
        data=preview,
        metadata=metadata,
        source_width=width,
        source_height=height,
        source_channels=channels,
        block_size=factor,
        reader_backend=backend,
        image_index=image_index,
    )


def _probe_fits_metadata(path: Path, *, image_index: int) -> FrameMetadata:
    """Read only FITS headers; never request an HDU's pixel array."""

    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            lazy_load_hdus=True,
            do_not_scale_image_data=True,
            uint=False,
            checksum=False,
        ) as hdul:
            hdu, declared_shape, image_count = _select_fits_hdu(
                hdul, image_index, path
            )
            width, height, channels, _ = _fits_layout(declared_shape, path)
            return normalize_metadata(
                FrameMetadata(
                    path=str(path),
                    width=width,
                    height=height,
                    channels=channels,
                    image_count=image_count,
                    header=_fits_header_dict(hdu.header),
                )
            )
    except FrameReadError:
        raise
    except Exception as error:
        raise FrameReadError("FITS_HEADER_ERROR", path, str(error)) from error


def _probe_xisf_metadata(path: Path, *, image_index: int) -> FrameMetadata:
    """Parse the XISF XML header without reading an image data block."""

    try:
        document = XISF(str(path))
        images = document.get_images_metadata()
        if not images:
            raise FrameReadError("NO_IMAGE", path, "XISF file contains no image")
        if image_index < 0 or image_index >= len(images):
            raise FrameReadError(
                "IMAGE_INDEX_RANGE",
                path,
                f"image_index {image_index} is outside [0, {len(images) - 1}]",
            )
        image_metadata = images[image_index]
        geometry = tuple(int(value) for value in image_metadata["geometry"])
        if len(geometry) != 3:
            raise FrameReadError(
                "UNSUPPORTED_GEOMETRY",
                path,
                f"XISF geometry is {geometry}, expected W:H:C",
            )
        width, height, channels = geometry
        if width < 1 or height < 1 or channels < 1:
            raise FrameReadError(
                "UNSUPPORTED_GEOMETRY", path, f"invalid XISF geometry {geometry}"
            )
        return normalize_metadata(
            FrameMetadata(
                path=str(path),
                width=width,
                height=height,
                channels=channels,
                image_count=len(images),
                header=_xisf_header_dict(image_metadata),
            )
        )
    except FrameReadError:
        raise
    except Exception as error:
        raise FrameReadError("XISF_HEADER_ERROR", path, str(error)) from error


def probe_frame_metadata(
    path: str | os.PathLike[str], *, image_index: int = 0
) -> FrameMetadata:
    """Probe FITS/XISF role and acquisition metadata without decoding pixels."""

    if isinstance(image_index, bool) or image_index < 0:
        raise ValueError("image_index must be a non-negative integer")
    source = _validated_path(path)
    format_name = _format_name(source)
    if format_name == "fits":
        return _probe_fits_metadata(source, image_index=image_index)
    if format_name == "xisf":
        return _probe_xisf_metadata(source, image_index=image_index)
    raise UnsupportedFrameFormatError(
        "UNSUPPORTED_FORMAT",
        source,
        "expected .fit/.fits/.fts[.fz] or .xisf",
    )


def read_frame_preview(
    path: str | os.PathLike[str],
    *,
    max_long_edge: int = DEFAULT_PREVIEW_LONG_EDGE,
    image_index: int = 0,
    max_full_decode_bytes: int | None = DEFAULT_MAX_FULL_DECODE_BYTES,
) -> ImagePreview:
    """Read one FITS/XISF image into a bounded block-mean preview.

    Input files are opened exclusively in read-only/binary-read mode.  The
    returned NumPy array has its write flag disabled to make accidental
    in-memory mutation visible to callers as well.
    """

    _validated_limits(max_long_edge, max_full_decode_bytes)
    if isinstance(image_index, bool) or image_index < 0:
        raise ValueError("image_index must be a non-negative integer")
    source = _validated_path(path)
    format_name = _format_name(source)
    if format_name == "fits":
        return _read_fits_preview(
            source,
            max_long_edge=max_long_edge,
            image_index=image_index,
            max_full_decode_bytes=max_full_decode_bytes,
        )
    if format_name == "xisf":
        return _read_xisf_preview(
            source,
            max_long_edge=max_long_edge,
            image_index=image_index,
            max_full_decode_bytes=max_full_decode_bytes,
        )
    raise UnsupportedFrameFormatError(
        "UNSUPPORTED_FORMAT",
        source,
        "expected .fit/.fits/.fts[.fz] or .xisf",
    )


# Short, discoverable alias for interactive callers.
read_preview = read_frame_preview


__all__ = [
    "DEFAULT_MAX_FULL_DECODE_BYTES",
    "DEFAULT_PREVIEW_LONG_EDGE",
    "FrameDiscoveryError",
    "FrameMemoryLimitError",
    "FrameReadError",
    "ImagePreview",
    "UnsupportedFrameFormatError",
    "discover_paths",
    "is_supported_frame_path",
    "probe_frame_metadata",
    "read_frame_preview",
    "read_preview",
]
