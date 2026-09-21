"""The buffered FITS band reader returns exactly the memory map's bytes.

The reader is the Windows transport; these tests force it on every host and
compare it against astropy's memory map over synthetic files covering the
storage forms the pipeline meets: unsigned 16-bit (``BZERO`` 32768), scaled
integers (``BSCALE``/``BZERO``), ``BLANK``, 32-bit integers and IEEE floats
with NaN payloads.
"""

from __future__ import annotations

from pathlib import Path
import threading

from astropy.io import fits
import numpy as np
import pytest

from lightframeqc.fits_bands import (
    FITS_READER_ENVIRONMENT,
    FitsBandReader,
    fits_bitpix_dtype,
    fits_reader_mode,
    open_fits_image_data,
)
from lightframeqc.native_psf import open_native_image
from lightframeqc.readers import read_frame_preview


HEIGHT, WIDTH = 97, 131


def _write_variants(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(20260921)
    files: dict[str, Path] = {}

    unsigned = rng.integers(0, 65536, size=(HEIGHT, WIDTH), dtype=np.uint16)
    unsigned[3, 7] = 0
    unsigned[50, 60] = 65535
    files["uint16"] = root / "uint16.fits"
    fits.writeto(files["uint16"], unsigned, fits.Header([("IMAGETYP", "Light"), ("FILTER", "R")]))

    scaled = fits.PrimaryHDU(rng.integers(-2000, 2000, size=(HEIGHT, WIDTH), dtype=np.int16))
    scaled.header["BSCALE"] = 0.125
    scaled.header["BZERO"] = 1000.5
    scaled.header["BLANK"] = -32768
    scaled.header["IMAGETYP"] = "Light"
    data = scaled.data
    data[10:12, 20:30] = -32768
    files["scaled-blank"] = root / "scaled.fits"
    # ``do_not_scale_image_data`` keeps the stored values exactly as given.
    fits.HDUList([scaled]).writeto(files["scaled-blank"], output_verify="exception")

    signed32 = rng.integers(-(2**31), 2**31 - 1, size=(HEIGHT, WIDTH), dtype=np.int32)
    files["int32"] = root / "int32.fits"
    fits.writeto(files["int32"], signed32, fits.Header([("IMAGETYP", "Light")]))

    floats = rng.normal(1000.0, 30.0, size=(HEIGHT, WIDTH)).astype(np.float32)
    floats[5, 5] = np.nan
    floats[6, 6] = np.inf
    floats[7, 7] = -np.inf
    # A NaN with a payload: byte identity means the payload survives too.
    floats.view(np.uint32)[8, 8] = 0x7FC00123
    files["float32"] = root / "float32.fits"
    fits.writeto(files["float32"], floats, fits.Header([("IMAGETYP", "Light")]))

    doubles = rng.normal(0.5, 0.01, size=(HEIGHT, WIDTH))
    files["float64"] = root / "float64.fits"
    fits.writeto(files["float64"], doubles, fits.Header([("IMAGETYP", "Light")]))
    return files


def _open_both(path: Path):
    hdul = fits.open(
        path, mode="readonly", memmap=True, lazy_load_hdus=True,
        do_not_scale_image_data=True, uint=False, checksum=False,
    )
    hdu = hdul[0]
    mapped = open_fits_image_data(hdu, path, mode="memmap")
    reader = open_fits_image_data(hdu, path, mode="buffered")
    assert isinstance(reader, FitsBandReader)
    return hdul, mapped, reader


def _same(left: np.ndarray, right: np.ndarray) -> None:
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert np.asarray(left).tobytes() == np.asarray(right).tobytes()


def test_reader_matches_memmap_bytes_for_every_storage_form(tmp_path: Path) -> None:
    for label, path in _write_variants(tmp_path).items():
        hdul, mapped, reader = _open_both(path)
        try:
            assert reader.shape == mapped.shape == (HEIGHT, WIDTH), label
            assert reader.dtype == mapped.dtype, label
            assert reader.dtype.byteorder in (">", "|"), label
            _same(reader[:], mapped[:])
            _same(reader[0:1], mapped[0:1])
            _same(reader[13:45], mapped[13:45])
            _same(reader[HEIGHT - 1 : HEIGHT], mapped[HEIGHT - 1 : HEIGHT])
            _same(reader[-3:], mapped[-3:])
            _same(reader[20:50, 7:100], mapped[20:50, 7:100])
            _same(reader[20:50, :], mapped[20:50, :])
            _same(reader[4], mapped[4])
            _same(reader[-1, 5:9], mapped[-1, 5:9])
            rows = np.array([0, 5, 5, 96, 33, 2])
            _same(reader[rows], mapped[rows])
            _same(reader[np.arange(0, HEIGHT, 3)], mapped[np.arange(0, HEIGHT, 3)])
            _same(reader[np.arange(0, HEIGHT, 2)], mapped[np.arange(0, HEIGHT, 2)])
            _same(reader[rows, 10:20], mapped[rows, 10:20])
            y = np.array([[0, 1, 96], [50, 50, 12]])
            x = np.array([[0, 130, 65], [1, 2, -1]])
            _same(reader[y, x], mapped[y, x])
            _same(reader[np.array([], dtype=np.int64)], mapped[np.array([], dtype=np.int64)])
            assert reader[10:10].shape == (0, WIDTH)
            assert not reader[13:45].flags.writeable
        finally:
            reader.close()
            hdul.close()
        assert reader.closed


def test_reader_rejects_what_it_does_not_implement_and_truncated_files(tmp_path: Path) -> None:
    files = _write_variants(tmp_path)
    hdul, _mapped, reader = _open_both(files["uint16"])
    try:
        with pytest.raises(IndexError):
            reader[::2]
        with pytest.raises(IndexError):
            reader[np.array([True, False])]
        with pytest.raises(IndexError):
            reader[HEIGHT]
        with pytest.raises(IndexError):
            reader[np.array([HEIGHT])]
        with pytest.raises(IndexError):
            reader[np.array([0]), np.array([WIDTH])]
        with pytest.raises(IndexError):
            reader[0:1, 0:1, 0:1]
    finally:
        reader.close()
        hdul.close()
    with pytest.raises(ValueError):
        reader[0:1]
    truncated = tmp_path / "truncated.fits"
    payload = files["uint16"].read_bytes()
    truncated.write_bytes(payload[: len(payload) - 3000])
    with fits.open(truncated, memmap=True, lazy_load_hdus=True, do_not_scale_image_data=True) as hdul:
        with pytest.raises(ValueError, match="shorter"):
            open_fits_image_data(hdul[0], truncated, mode="buffered")
    with pytest.raises(ValueError):
        fits_bitpix_dtype(24)
    assert fits_bitpix_dtype(-32) == np.dtype(">f4") and fits_bitpix_dtype(16) == np.dtype(">i2")


def test_reader_mode_follows_the_platform_unless_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.delenv(FITS_READER_ENVIRONMENT, raising=False)
    assert fits_reader_mode() == ("buffered" if os.name == "nt" else "memmap")
    monkeypatch.setenv(FITS_READER_ENVIRONMENT, "buffered")
    assert fits_reader_mode() == "buffered"
    monkeypatch.setenv(FITS_READER_ENVIRONMENT, " MEMMAP ")
    assert fits_reader_mode() == "memmap"
    monkeypatch.setenv(FITS_READER_ENVIRONMENT, "nonsense")
    assert fits_reader_mode() == ("buffered" if os.name == "nt" else "memmap")


def test_concurrent_band_reads_of_one_reader_are_isolated_per_thread(tmp_path: Path) -> None:
    files = _write_variants(tmp_path)
    hdul, mapped, reader = _open_both(files["float32"])
    expected = {y0: np.asarray(mapped[y0 : y0 + 8]).tobytes() for y0 in range(0, HEIGHT - 8)}
    failures: list[str] = []

    def worker(seed: int) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(300):
            y0 = int(rng.integers(0, HEIGHT - 8))
            band = reader[y0 : y0 + 8]
            # Hold the view across another read on this thread's neighbours'
            # buffers: only this thread's next read may replace it.
            digest = band.tobytes()
            if digest != expected[y0]:
                failures.append(f"thread {seed} row {y0}")
                return

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(6)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30.0)
    finally:
        reader.close()
        hdul.close()
    assert failures == []


def test_preview_and_native_stamps_are_identical_in_both_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = _write_variants(tmp_path)
    for label, path in files.items():
        previews: dict[str, object] = {}
        stamps: dict[str, bytes] = {}
        for mode in ("memmap", "buffered"):
            monkeypatch.setenv(FITS_READER_ENVIRONMENT, mode)
            preview = read_frame_preview(path, max_long_edge=40)
            previews[mode] = preview
            with open_native_image(str(path)) as image:
                assert image is not None
                stamp = image[10:31, 20:41]
                edge = image[HEIGHT - 5 : HEIGHT, WIDTH - 5 : WIDTH]
                stamps[mode] = stamp.tobytes() + edge.tobytes()
        left, right = previews["memmap"], previews["buffered"]
        assert left.data.tobytes() == right.data.tobytes(), label
        assert left.data.dtype == right.data.dtype and left.block_size == right.block_size
        assert left.metadata == right.metadata, label
        assert left.reader_backend == "astropy-fits-memmap"
        assert right.reader_backend == "astropy-fits-buffered"
        assert stamps["memmap"] == stamps["buffered"], label
