"""``FitsFrame`` decodes identically through the memory map and the band reader.

The band reader is the Windows transport for FITS rows; these tests force it
on every host and require byte-identical Float32 values, samples and
integrated masters against the memory-map path, including a threaded
integration that reads shared calibration masters from several threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from lightframeqc.fits_bands import FITS_READER_ENVIRONMENT, FitsBandReader
from ufwbpp.calibration import (
    CalibrationError,
    FitsFrame,
    FrameExpression,
    IntegrationMapPaths,
    IntegrationParameters,
    integrate_expressions,
)


HEIGHT, WIDTH = 61, 83


def _write_variants(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(1234)
    files: dict[str, Path] = {}
    # The declared numeric domain lets the Lanczos-3 sampler run on every
    # storage form (unsigned integers declare theirs through BITPIX/BZERO).
    header = fits.Header(
        [("IMAGETYP", "Light"), ("FILTER", "R"), ("EXPTIME", 120.0), ("OAFNDOM", "TEST"), ("OAFNSCL", 65535.0)]
    )

    unsigned = rng.integers(0, 65536, size=(HEIGHT, WIDTH), dtype=np.uint16)
    unsigned[0, 0] = 0
    unsigned[1, 1] = 65535
    files["uint16"] = root / "uint16.fits"
    fits.writeto(files["uint16"], unsigned, header)

    hdu = fits.PrimaryHDU(rng.integers(-3000, 3000, size=(HEIGHT, WIDTH), dtype=np.int16), header=header)
    hdu.header["BSCALE"] = 0.25
    hdu.header["BZERO"] = 512.75
    hdu.header["BLANK"] = -32768
    hdu.data[4:6, 8:12] = -32768
    files["scaled-blank"] = root / "scaled.fits"
    fits.HDUList([hdu]).writeto(files["scaled-blank"])

    files["uint32"] = root / "uint32.fits"
    fits.writeto(files["uint32"], rng.integers(0, 2**32, size=(HEIGHT, WIDTH), dtype=np.uint32), header)

    floats = rng.normal(1000.0, 25.0, size=(HEIGHT, WIDTH)).astype(np.float32)
    floats[2, 3] = np.nan
    floats[3, 4] = np.inf
    floats.view(np.uint32)[5, 6] = 0x7FC00321
    files["float32"] = root / "float32.fits"
    fits.writeto(files["float32"], floats, header)

    files["float64"] = root / "float64.fits"
    fits.writeto(files["float64"], rng.normal(0.4, 0.05, size=(HEIGHT, WIDTH)), header)
    return files


def _same(left: np.ndarray, right: np.ndarray) -> None:
    assert left.dtype == right.dtype and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def test_fits_frame_values_are_byte_identical_in_both_transports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = _write_variants(tmp_path)
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.5, WIDTH - 0.5, 37), np.linspace(-0.5, HEIGHT - 0.5, 29)
    )
    grid_x = grid_x + 0.37
    grid_y = grid_y - 0.21
    grid_x[0, 0] = np.nan
    rows = np.array([0, 7, 7, HEIGHT - 1, 30])
    y_index = np.array([[0, 1], [HEIGHT - 1, 30]])
    x_index = np.array([[0, WIDTH - 1], [5, 6]])
    for label, path in files.items():
        results: dict[str, tuple[np.ndarray, ...]] = {}
        for mode in ("memmap", "buffered"):
            monkeypatch.setenv(FITS_READER_ENVIRONMENT, mode)
            with FitsFrame(path) as frame:
                if mode == "buffered":
                    assert isinstance(frame._data, FitsBandReader), label
                    assert frame._hdul is None, "the reader owns the only handle"
                else:
                    assert not isinstance(frame._data, FitsBandReader)
                results[mode] = (
                    frame.full_values(),
                    frame.read_rows(0, 1),
                    frame.read_rows(11, 43),
                    frame.read_rows(HEIGHT - 2, HEIGHT),
                    frame.read_sampled_rows(rows),
                    frame.read_sampled_rows(np.arange(0, HEIGHT, 3)),
                    frame._physical_index(y_index, x_index),
                    frame.sample_bilinear(grid_x, grid_y),
                    frame.sample_lanczos3_clamped(grid_x, grid_y),
                )
                info = frame.info
            assert info.shape == (HEIGHT, WIDTH)
        for left, right in zip(results["memmap"], results["buffered"], strict=True):
            _same(left, right)
        # The frame is closed either way: the reader releases its handle.
        with FitsFrame(path) as frame:
            data = frame._data
        if isinstance(data, FitsBandReader):
            assert data.closed


def test_fits_frame_reports_a_truncated_file_as_invalid_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = _write_variants(tmp_path)
    payload = files["uint16"].read_bytes()
    truncated = tmp_path / "short.fits"
    truncated.write_bytes(payload[: len(payload) - 2880 - 100])
    monkeypatch.setenv(FITS_READER_ENVIRONMENT, "buffered")
    with pytest.raises(CalibrationError) as error:
        with FitsFrame(truncated):
            pass
    assert error.value.code == "FITS_DATA_INVALID"


def _integrate(root: Path, lights: list[Path], dark: Path, flat: Path, stem: str) -> tuple[str, ...]:
    output = root / f"{stem}.fits"
    maps = IntegrationMapPaths(root / f"{stem}-a.fits", root / f"{stem}-c.fits", root / f"{stem}-r.fits")
    integrate_expressions(
        [FrameExpression(str(path), subtract_path=str(dark), divide_path=str(flat)) for path in lights],
        output,
        parameters=IntegrationParameters(sigma_clip=3.0, minimum_rejection_frames=3, max_memory_bytes=64 * 1024, max_statistics_samples=200),
        map_paths=maps,
        # Several band-reader threads read the shared dark and flat at once.
        native_threads=4,
    )
    return tuple(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (output, maps.accepted_count, maps.coverage, maps.rejection_count)
    )


def test_threaded_integration_over_shared_masters_is_identical_in_both_transports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(99)
    header = fits.Header([("IMAGETYP", "Light"), ("FILTER", "R")])
    lights = []
    for index in range(7):
        values = rng.integers(2000, 4000, size=(HEIGHT, WIDTH), dtype=np.uint16)
        values[index, index] = 60000
        path = tmp_path / f"light-{index}.fits"
        fits.writeto(path, values, header)
        lights.append(path)
    dark = tmp_path / "dark.fits"
    fits.writeto(dark, rng.normal(500.0, 3.0, size=(HEIGHT, WIDTH)).astype(np.float32), fits.Header([("IMAGETYP", "Master Dark")]))
    flat = tmp_path / "flat.fits"
    fits.writeto(flat, rng.uniform(0.8, 1.2, size=(HEIGHT, WIDTH)).astype(np.float32), fits.Header([("IMAGETYP", "Master Flat")]))
    monkeypatch.setenv(FITS_READER_ENVIRONMENT, "memmap")
    mapped = _integrate(tmp_path / "m", lights, dark, flat, "master")
    monkeypatch.setenv(FITS_READER_ENVIRONMENT, "buffered")
    buffered = _integrate(tmp_path / "b", lights, dark, flat, "master")
    assert mapped == buffered
    # The same FitsFrame read from many threads at once stays exact.
    with FitsFrame(dark) as frame:
        assert isinstance(frame._data, FitsBandReader)
        expected = frame.full_values()

        def band(y0: int) -> bytes:
            return frame.read_rows(y0, min(HEIGHT, y0 + 5)).tobytes()

        with ThreadPoolExecutor(max_workers=8) as pool:
            digests = list(pool.map(band, [y0 for _ in range(20) for y0 in range(0, HEIGHT, 5)]))
    reference = [expected[y0 : min(HEIGHT, y0 + 5)].tobytes() for _ in range(20) for y0 in range(0, HEIGHT, 5)]
    assert digests == reference
