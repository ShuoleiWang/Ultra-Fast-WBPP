from __future__ import annotations

import hashlib
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest
from lightframeqc.xisf import XISF

from lightframeqc.readers import (
    FrameDiscoveryError,
    FrameMemoryLimitError,
    UnsupportedFrameFormatError,
    discover_paths,
    read_frame_preview,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _block_mean_2x2(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    assert height % 2 == 0 and width % 2 == 0
    return image.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))


def test_reads_unsigned_fits_by_bounded_block_mean_without_mutating_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "M31_R_300s.FITS"
    pixels = (np.arange(48, dtype=np.uint16).reshape(6, 8) + 40_000)
    hdu = fits.PrimaryHDU(pixels)
    hdu.header["FILTER"] = "R"
    hdu.header["EXPTIME"] = 300.0
    hdu.writeto(source)
    before_hash = _sha256(source)
    before_stat = source.stat()

    # An uncompressed memmap is streamed, so the full-decode ceiling is not
    # relevant even when it is deliberately smaller than one source row.
    preview = read_frame_preview(
        source,
        max_long_edge=4,
        max_full_decode_bytes=1,
    )

    np.testing.assert_allclose(preview.data, _block_mean_2x2(pixels), rtol=0, atol=0)
    assert preview.data.shape == (3, 4)
    assert preview.data.dtype == np.float32
    assert preview.data.flags.c_contiguous
    assert not preview.data.flags.writeable
    assert preview.block_size == 2
    assert preview.source_width == 8
    assert preview.source_height == 6
    assert preview.source_channels == 1
    assert preview.reader_backend == "astropy-fits-memmap"
    assert preview.metadata.filter_name == "R"
    assert preview.metadata.exposure_seconds == pytest.approx(300.0)
    assert _sha256(source) == before_hash
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_fits_partial_edge_blocks_ignore_nonfinite_samples(tmp_path: Path) -> None:
    source = tmp_path / "partial.fit"
    pixels = np.arange(35, dtype=np.float32).reshape(5, 7)
    pixels[0, 0] = np.nan
    pixels[4, 6] = np.nan
    fits.writeto(source, pixels)

    preview = read_frame_preview(source, max_long_edge=4)

    assert preview.block_size == 2
    assert preview.data.shape == (3, 4)
    assert preview.data[0, 0] == pytest.approx(np.mean([1.0, 7.0, 8.0]))
    assert np.isnan(preview.data[2, 3])


def test_streams_uncompressed_multichannel_xisf_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "light.xisf"
    base = np.arange(48, dtype=np.uint16).reshape(6, 8)
    pixels = np.stack((base, base + 100, base + 200), axis=2)
    image_metadata = {
        "FITSKeywords": {
            "FILTER": [{"value": "'Ha'", "comment": ""}],
            "EXPTIME": [{"value": "120.0", "comment": "seconds"}],
        }
    }
    _, actual_codec = XISF.write(
        str(source),
        pixels,
        image_metadata=image_metadata,
        codec=None,
    )
    assert actual_codec is None
    before_hash = _sha256(source)
    before_stat = source.stat()

    preview = read_frame_preview(
        source,
        max_long_edge=4,
        max_full_decode_bytes=1,
    )

    expected_luminance = pixels.astype(np.float64).mean(axis=2)
    np.testing.assert_allclose(
        preview.data,
        _block_mean_2x2(expected_luminance),
        rtol=0,
        atol=0,
    )
    assert preview.reader_backend == "xisf-python-attachment-stream"
    assert preview.source_channels == 3
    assert preview.metadata.filter_name == "HA"
    assert preview.metadata.exposure_seconds == pytest.approx(120.0)
    assert not preview.data.flags.writeable
    assert _sha256(source) == before_hash
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_compressed_xisf_fails_closed_before_exceeding_decode_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "compressed.xisf"
    pixels = np.zeros((64, 96, 1), dtype=np.float32)
    pixels[10:20, 30:50, 0] = 1.0
    _, actual_codec = XISF.write(str(source), pixels, codec="zlib")
    assert actual_codec == "zlib"
    before_hash = _sha256(source)

    with pytest.raises(FrameMemoryLimitError) as raised:
        read_frame_preview(source, max_long_edge=64, max_full_decode_bytes=1_024)

    assert raised.value.code == "FULL_DECODE_LIMIT"
    assert _sha256(source) == before_hash

    preview = read_frame_preview(source, max_long_edge=64)
    assert preview.reader_backend == "xisf-python-full-decode"
    assert preview.data.shape == (32, 48)
    np.testing.assert_allclose(
        preview.data,
        _block_mean_2x2(pixels[:, :, 0]),
        rtol=0,
        atol=0,
    )


def test_discovers_supported_files_recursively_deterministically_and_once(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "z-subdir"
    nested.mkdir()
    first = tmp_path / "B.FITS"
    second = nested / "a.xisf"
    ignored = nested / "notes.txt"
    first.write_bytes(b"fits placeholder")
    second.write_bytes(b"xisf placeholder")
    ignored.write_text("not a frame", encoding="utf-8")

    discovered = discover_paths([tmp_path, first])

    expected = sorted((first.resolve(), second.resolve()), key=lambda path: str(path))
    assert discovered == expected


def test_discovery_rejects_explicit_bad_paths_but_ignores_directory_noise(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "frame.png"
    unsupported.write_bytes(b"png")

    with pytest.raises(UnsupportedFrameFormatError) as raised:
        discover_paths(unsupported)
    assert raised.value.code == "UNSUPPORTED_FORMAT"

    with pytest.raises(FrameDiscoveryError) as raised:
        discover_paths(tmp_path)
    assert raised.value.code == "NO_SUPPORTED_FRAMES"

    with pytest.raises(FrameDiscoveryError) as raised:
        discover_paths(tmp_path / "missing")
    assert raised.value.code == "INPUT_NOT_FOUND"
