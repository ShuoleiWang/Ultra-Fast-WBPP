"""Row-band access to uncompressed FITS image data without a memory map.

A memory map is the natural way to stream a FITS image band by band and it
stays the default on POSIX.  On Windows every page a mapping touches is a
soft fault that the kernel serves one at a time: reading 32-row bands of a
26 MP frame through a map measured 2.7x slower than ``seek`` + ``readinto``
on the same laptop, 4.6x for the 64-row uint16 bands of the QC preview.
:class:`FitsBandReader` reads the same bytes at the same offsets as the map
would and views them with the same big-endian dtype, so every decoding path
above it (BZERO/BSCALE, BLANK, the Float32 conversions) sees identical values
whichever transport ran; the differential tests hold that invariant.

The reader offers the slice of the ndarray interface the readers use:
``reader[y0:y1]``, ``reader[y0:y1, x0:x1]``, a row gather ``reader[rows]``
and an element gather ``reader[y, x]``.  Band reads return read-only views
of a buffer that belongs to the calling thread and is reused by that
thread's next read; callers convert or copy before reading again, exactly
as they already did with the map (the conversions copy).  Gathers return
fresh arrays.  Several threads may share one reader: each has its own
buffer and the file position is guarded by a lock.
"""

from __future__ import annotations

import operator
import os
from pathlib import Path
import threading
from typing import Any, Literal

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray


ReaderMode = Literal["memmap", "buffered"]

# ``LIGHTFRAMEQC_FITS_READER=memmap|buffered`` overrides the platform default
# (an A/B switch for measurements and the differential tests).
FITS_READER_ENVIRONMENT = "LIGHTFRAMEQC_FITS_READER"
BUFFERED_BACKEND = "astropy-fits-buffered"
MEMMAP_BACKEND = "astropy-fits-memmap"

_BITPIX_DTYPES = {
    8: np.dtype(">u1"),
    16: np.dtype(">i2"),
    32: np.dtype(">i4"),
    64: np.dtype(">i8"),
    -32: np.dtype(">f4"),
    -64: np.dtype(">f8"),
}


def fits_reader_mode() -> ReaderMode:
    """``buffered`` on Windows, ``memmap`` elsewhere, unless overridden."""

    requested = os.environ.get(FITS_READER_ENVIRONMENT, "").strip().casefold()
    if requested in {"memmap", "buffered"}:
        return requested  # type: ignore[return-value]
    return "buffered" if os.name == "nt" else "memmap"


def fits_bitpix_dtype(bitpix: int) -> np.dtype[Any]:
    """The stored (big-endian) dtype of an image HDU with this ``BITPIX``."""

    try:
        return _BITPIX_DTYPES[int(bitpix)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unsupported BITPIX {bitpix!r}") from error


class FitsBandReader:
    """Rows of one uncompressed 2-D image HDU read with ``seek`` + ``readinto``.

    ``data_offset`` is the byte offset of the HDU's data segment (astropy's
    ``hdu.fileinfo()["datLoc"]``), ``dtype`` the stored big-endian dtype and
    ``shape`` ``(height, width)``.  The file is opened unbuffered: every read
    is one system call for exactly the band's bytes.
    """

    ndim = 2

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        data_offset: int,
        dtype: np.dtype[Any],
        shape: tuple[int, int],
    ) -> None:
        height, width = (int(shape[0]), int(shape[1]))
        if len(shape) != 2 or height < 1 or width < 1:
            raise ValueError("a band reader needs a positive two-dimensional shape")
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.shape = (height, width)
        self.size = height * width
        self._offset = int(data_offset)
        self._row_bytes = width * self.dtype.itemsize
        self._file: Any = open(self.path, "rb", buffering=0)
        self._lock = threading.Lock()
        self._local = threading.local()
        end = self._offset + height * self._row_bytes
        if end > os.fstat(self._file.fileno()).st_size:
            self._file.close()
            raise ValueError(
                f"{self.path}: the file is shorter than its declared image ({end} bytes)"
            )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        stream = self._file
        if stream is not None:
            self._file = None
            stream.close()
        # Drop this thread's cache; other threads' buffers go with the object.
        self._local.__dict__.clear()

    @property
    def closed(self) -> bool:
        return self._file is None

    def __enter__(self) -> "FitsBandReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.shape[0]

    # -- reads -------------------------------------------------------------

    def _read_into(self, buffer: memoryview, offset: int) -> None:
        stream = self._file
        if stream is None:
            raise ValueError("I/O operation on closed band reader")
        with self._lock:
            stream.seek(offset)
            filled = 0
            while filled < len(buffer):
                count = stream.readinto(buffer[filled:])
                if not count:
                    raise OSError(f"{self.path}: unexpected end of file while reading image rows")
                filled += count

    def _buffer(self, nbytes: int) -> bytearray:
        """This thread's reusable buffer, replaced (never resized) when it
        grows so an outstanding view keeps the bytes it was given."""

        buffer = getattr(self._local, "buffer", None)
        if buffer is None or len(buffer) < nbytes:
            buffer = bytearray(max(nbytes, 2 * self._row_bytes))
            self._local.buffer = buffer
            self._local.band = None
        return buffer

    def _band(self, y0: int, y1: int) -> NDArray[Any]:
        """Rows ``[y0, y1)`` as a read-only ``(y1 - y0, width)`` view."""

        height, width = self.shape
        if y0 < 0 or y1 > height or y0 > y1:
            raise IndexError(f"rows [{y0}, {y1}) are outside the image height {height}")
        if y1 == y0:
            return np.empty((0, width), dtype=self.dtype)
        cached = getattr(self._local, "band", None)
        if cached is not None and cached[0] <= y0 and y1 <= cached[1]:
            # Resampling gathers the same source rows tap after tap; the
            # cached band serves them without another read.
            return cached[2][y0 - cached[0] : y1 - cached[0]]
        rows = y1 - y0
        nbytes = rows * self._row_bytes
        buffer = self._buffer(nbytes)
        self._read_into(memoryview(buffer)[:nbytes], self._offset + y0 * self._row_bytes)
        view = np.frombuffer(buffer, dtype=self.dtype, count=rows * width).reshape(rows, width)
        view.setflags(write=False)
        self._local.band = (y0, y1, view)
        return view

    def rows(self, y0: int, y1: int) -> NDArray[Any]:
        """Rows ``[y0, y1)``; a view valid until this thread's next read."""

        return self._band(int(y0), int(y1))

    def _gather_rows(self, indices: NDArray[np.int64]) -> NDArray[Any]:
        """The listed rows, each read on its own, as a fresh array."""

        height, width = self.shape
        flat = np.asarray(indices, dtype=np.int64).ravel()
        flat = np.where(flat < 0, flat + height, flat)
        if flat.size and (flat.min() < 0 or flat.max() >= height):
            raise IndexError("row index out of range")
        result = np.empty((flat.size, width), dtype=self.dtype)
        if flat.size == 0:
            return result.reshape(indices.shape + (width,))
        span = int(flat.max()) - int(flat.min()) + 1
        if span <= 2 * flat.size:
            # Dense enough that one band read beats one read per row.
            band = self._band(int(flat.min()), int(flat.max()) + 1)
            np.take(band, flat - int(flat.min()), axis=0, out=result)
            return result.reshape(indices.shape + (width,))
        target = memoryview(result).cast("B")
        row_bytes = self._row_bytes
        for slot, row in enumerate(flat.tolist()):
            self._read_into(
                target[slot * row_bytes : (slot + 1) * row_bytes], self._offset + row * row_bytes
            )
        return result.reshape(indices.shape + (width,))

    def _gather_elements(self, y: NDArray[Any], x: NDArray[Any]) -> NDArray[Any]:
        """``data[y, x]`` for same-shaped index arrays, as a fresh array."""

        height, width = self.shape
        y = np.asarray(y, dtype=np.int64)
        x = np.asarray(x, dtype=np.int64)
        y, x = np.broadcast_arrays(y, x)
        y = np.where(y < 0, y + height, y)
        x = np.where(x < 0, x + width, x)
        if y.size == 0:
            return np.empty(y.shape, dtype=self.dtype)
        if y.min() < 0 or y.max() >= height or x.min() < 0 or x.max() >= width:
            raise IndexError("index out of range")
        y0 = int(y.min())
        band = self._band(y0, int(y.max()) + 1)
        return band[y - y0, x]

    def _column_view(self, band: NDArray[Any], columns: Any) -> NDArray[Any]:
        if isinstance(columns, slice):
            return band[:, columns]
        return band[:, np.asarray(columns)]

    def __getitem__(self, key: Any) -> NDArray[Any]:
        if isinstance(key, tuple):
            if len(key) > 2:
                raise IndexError("too many indices for a two-dimensional image")
            rows = key[0]
            columns = key[1] if len(key) == 2 else slice(None)
        else:
            rows, columns = key, slice(None)
        height, _width = self.shape
        if isinstance(rows, slice):
            y0, y1, step = rows.indices(height)
            if step != 1:
                raise IndexError("row slices with a step are not supported by the band reader")
            return self._column_view(self._band(y0, max(y0, y1)), columns)
        if isinstance(rows, (int, np.integer)):
            y = operator.index(rows)
            if y < 0:
                y += height
            if y < 0 or y >= height:
                raise IndexError("row index out of range")
            band = self._band(y, y + 1)
            return band[0][columns] if isinstance(columns, slice) else band[0][np.asarray(columns)]
        indices = np.asarray(rows)
        if indices.dtype == np.bool_:
            raise IndexError("boolean row masks are not supported by the band reader")
        if isinstance(columns, slice):
            gathered = self._gather_rows(indices.astype(np.int64))
            return gathered[..., columns]
        return self._gather_elements(indices, np.asarray(columns))


def image_hdu_shape(header: fits.Header) -> tuple[int, ...]:
    """``(NAXISn ... NAXIS1)`` of an image header, NumPy order."""

    naxis = int(header.get("NAXIS", 0) or 0)
    return tuple(int(header.get(f"NAXIS{axis}", 0) or 0) for axis in range(naxis, 0, -1))


def open_fits_image_data(
    hdu: Any,
    path: str | os.PathLike[str],
    *,
    mode: ReaderMode | None = None,
) -> Any:
    """``hdu.data`` (the memory map) or a :class:`FitsBandReader` over it.

    The reader serves uncompressed two-dimensional images only; cubes,
    compressed HDUs and the memmap mode return astropy's array so the
    decoding code above is the same either way.  ``None`` when the HDU has
    no data.
    """

    selected = mode or fits_reader_mode()
    if selected == "memmap" or isinstance(hdu, fits.CompImageHDU):
        return hdu.data
    shape = image_hdu_shape(hdu.header)
    if len(shape) != 2 or any(dimension < 1 for dimension in shape):
        return hdu.data
    info = hdu.fileinfo()
    if not info or info.get("datLoc") is None:
        return hdu.data
    return FitsBandReader(
        path,
        data_offset=int(info["datLoc"]),
        dtype=fits_bitpix_dtype(int(hdu.header["BITPIX"])),
        shape=(shape[0], shape[1]),
    )


def close_image_data(data: Any) -> None:
    """Release a reader returned by :func:`open_fits_image_data` (no-op for arrays)."""

    if isinstance(data, FitsBandReader):
        data.close()


def reader_backend(data: Any) -> str:
    return BUFFERED_BACKEND if isinstance(data, FitsBandReader) else MEMMAP_BACKEND


__all__ = [
    "BUFFERED_BACKEND",
    "FITS_READER_ENVIRONMENT",
    "FitsBandReader",
    "MEMMAP_BACKEND",
    "close_image_data",
    "fits_bitpix_dtype",
    "fits_reader_mode",
    "image_hdu_shape",
    "open_fits_image_data",
    "reader_backend",
]
