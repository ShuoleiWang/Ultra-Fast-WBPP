from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

from astropy.io import fits
import numpy as np
from PIL import Image
import pytest

import ufwbpp.color_product as color_product_module

from ufwbpp.color_product import (
    ColorProductError,
    ColorProductRequest,
    build_color_product,
)


def _solved_header(*, crval1: float = 150.0, filter_name: str = "R") -> fits.Header:
    header = fits.Header()
    header["OBJECT"] = "SYNTHETIC-RGB"
    header["FILTER"] = filter_name
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.5
    header["CRPIX2"] = 12.5
    header["CRVAL1"] = crval1
    header["CRVAL2"] = 20.0
    header["CD1_1"] = -0.0004
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.0004
    header["OAFSTATE"] = "SOLVED"
    header["OAFWCS"] = "SOLVED"
    return header


def _write_channel(
    path: Path,
    *,
    channel: str,
    multiplier: float,
    crval1: float = 150.0,
) -> Path:
    y, x = np.indices((24, 32), dtype=np.float32)
    data = multiplier * (100.0 + x + 2.0 * y)
    fits.writeto(
        path,
        data.astype(np.float32),
        _solved_header(crval1=crval1, filter_name=channel),
        overwrite=False,
        checksum=True,
    )
    return path


def _channels(tmp_path: Path) -> dict[str, Path]:
    return {
        "R": _write_channel(tmp_path / "R.fits", channel="R", multiplier=1.0),
        "G": _write_channel(tmp_path / "G.fits", channel="G", multiplier=0.8),
        "B": _write_channel(tmp_path / "B.fits", channel="B", multiplier=0.6),
        "L": _write_channel(tmp_path / "L.fits", channel="L", multiplier=1.3),
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_color_product_writes_linear_fits_and_16_bit_previews(tmp_path: Path) -> None:
    channels = _channels(tmp_path)
    before = {name: _digest(path) for name, path in channels.items()}
    output = tmp_path / "color-product"
    result = build_color_product(
        ColorProductRequest(
            red_path=str(channels["R"]),
            green_path=str(channels["G"]),
            blue_path=str(channels["B"]),
            luminance_path=str(channels["L"]),
            output_directory=str(output),
        )
    )

    assert result.output_directory == str(output)
    with fits.open(result.linear_rgb_path, checksum=True) as hdul:
        assert hdul[0].data.shape == (3, 24, 32)
        assert hdul[0].data.dtype.kind == "f"
        assert hdul[0].header["OAFSTATE"] == "SOLVED"
        assert hdul[0].header["OAFWCS"] == "SOLVED"
        assert hdul[0].header["OAFPROD"] == "LINEAR_RGB"
        assert hdul[0].header["OAFLUM"]

    png = Path(result.preview_png_path).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, bit_depth, color_type = struct.unpack(">IIBB", png[16:26])
    assert (width, height, bit_depth, color_type) == (32, 24, 16, 2)
    with Image.open(result.preview_tiff_path) as image:
        assert image.size == (32, 24)
        assert tuple(image.tag_v2[258]) == (16, 16, 16)

    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "SOLVED"
    assert receipt["preview"]["colorPreserving"] is True
    assert receipt["preview"]["bitDepth"] == 16
    assert len(receipt["wcs"]["fivePointConsistency"]) == 3
    assert {item["channel"] for item in receipt["inputs"]} == {"R", "G", "B", "L"}
    assert {name: _digest(path) for name, path in channels.items()} == before
    # PixInsight names an opened image after its file stem, so the cube and
    # its previews are called after the channels they hold.
    assert sorted(path.name for path in output.iterdir()) == ["LRGB.fits", "LRGB.png", "LRGB.tiff", "receipt.json"]
    assert [item["path"] for item in receipt["artifacts"]] == ["LRGB.fits", "LRGB.tiff", "LRGB.png"]

    with pytest.raises(ColorProductError) as captured:
        build_color_product(
            ColorProductRequest(
                red_path=str(channels["R"]),
                green_path=str(channels["G"]),
                blue_path=str(channels["B"]),
                output_directory=str(output),
            )
        )
    assert captured.value.code == "OUTPUT_EXISTS"


def test_color_product_name_defaults_to_channels_and_rejects_unsafe_names(tmp_path: Path) -> None:
    channels = _channels(tmp_path)
    request = ColorProductRequest(
        red_path=str(channels["R"]), green_path=str(channels["G"]), blue_path=str(channels["B"]),
        output_directory=str(tmp_path / "rgb"),
    )
    result = build_color_product(request)
    assert Path(result.linear_rgb_path).name == "RGB.fits"
    assert Path(result.preview_png_path).name == "RGB.png"
    named = build_color_product(replace(request, output_directory=str(tmp_path / "named"), product_name="NGC7331_RGB"))
    assert Path(named.linear_rgb_path).name == "NGC7331_RGB.fits"
    for bad in ("", "../x", "a b", "-lead", "x" * 65):
        with pytest.raises(ColorProductError) as info:
            build_color_product(replace(request, output_directory=str(tmp_path / "bad"), product_name=bad))
        assert info.value.code == "PRODUCT_NAME_INVALID"
        assert not (tmp_path / "bad").exists()


def test_color_product_rejects_missing_channel_without_publication(tmp_path: Path) -> None:
    channels = _channels(tmp_path)
    output = tmp_path / "missing"
    with pytest.raises(ColorProductError) as captured:
        build_color_product(
            ColorProductRequest(
                red_path=str(channels["R"]),
                green_path="",
                blue_path=str(channels["B"]),
                output_directory=str(output),
            )
        )
    assert captured.value.code == "CHANNEL_MISSING"
    assert not output.exists()


def test_color_product_rejects_five_point_wcs_mismatch(tmp_path: Path) -> None:
    channels = _channels(tmp_path)
    _write_channel(
        tmp_path / "bad-G.fits",
        channel="G",
        multiplier=0.8,
        crval1=150.01,
    )
    output = tmp_path / "mismatch"
    with pytest.raises(ColorProductError) as captured:
        build_color_product(
            ColorProductRequest(
                red_path=str(channels["R"]),
                green_path=str(tmp_path / "bad-G.fits"),
                blue_path=str(channels["B"]),
                output_directory=str(output),
            )
        )
    assert captured.value.code == "CHANNEL_WCS_MISMATCH"
    assert not output.exists()


def test_color_product_rejects_swapped_filter_roles(tmp_path: Path) -> None:
    channels = _channels(tmp_path)
    output = tmp_path / "swapped"
    with pytest.raises(ColorProductError) as captured:
        build_color_product(
            ColorProductRequest(
                red_path=str(channels["B"]),
                green_path=str(channels["G"]),
                blue_path=str(channels["R"]),
                output_directory=str(output),
            )
        )
    assert captured.value.code == "CHANNEL_FILTER_MISMATCH"
    assert not output.exists()


def test_color_product_atomic_publish_failure_leaves_no_product(tmp_path: Path) -> None:
    channels = _channels(tmp_path)
    output = tmp_path / "atomic-failure"

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("injected publication failure")

    with pytest.raises(ColorProductError) as captured:
        build_color_product(
            ColorProductRequest(
                red_path=str(channels["R"]),
                green_path=str(channels["G"]),
                blue_path=str(channels["B"]),
                output_directory=str(output),
            ),
            publisher=fail_publish,
        )
    assert captured.value.code == "ATOMIC_PUBLICATION_FAILED"
    assert not output.exists()
    assert not list(tmp_path.glob(".atomic-failure.staging-*"))


def test_post_commit_parent_fsync_failure_is_not_reported_as_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels = _channels(tmp_path)
    output = tmp_path / "committed"
    original = color_product_module._fsync_directory
    calls = 0

    def fail_second_call(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        original(path)

    from ufwbpp import publication
    monkeypatch.setattr(color_product_module, "_fsync_directory", fail_second_call)
    monkeypatch.setattr(publication, "_fsync_directory", fail_second_call)
    result = build_color_product(
        ColorProductRequest(
            red_path=str(channels["R"]),
            green_path=str(channels["G"]),
            blue_path=str(channels["B"]),
            output_directory=str(output),
        )
    )
    assert calls == 2
    assert Path(result.receipt_path).is_file()
    assert result.receipt["publication"]["parentDirectoryFsync"] == "BEST_EFFORT_AFTER_COMMIT"
