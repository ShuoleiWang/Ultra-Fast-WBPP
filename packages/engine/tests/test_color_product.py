from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct

from astropy.io import fits
import numpy as np
from PIL import Image
import pytest

import ufwbpp.products.color as color_product_module

from ufwbpp.products.color import (
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

    from ufwbpp.products import publication
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


def test_super_luminance_weights_channels_by_their_pixel_noise_and_keeps_the_reference(tmp_path: Path) -> None:
    """One field in L, R, G, B and Ha: the super luminance takes the
    broadband channels at weights scale^2 / noise^2, leaves out G (stars 30 %
    wider) and Ha, keeps L's flux scale, sky level and footprint, and lowers
    L's noise by the predicted gain.  Faint structure that every channel
    shares does not move the weights, and B, whose stars are faintest, still
    measures as sharp as L."""

    from scipy import ndimage

    from ufwbpp.products.super_luminance import build_super_luminance

    shape = (720, 720)
    sky = np.random.default_rng(3)
    count = 260
    rows, columns = sky.uniform(12, shape[0] - 12, count), sky.uniform(12, shape[1] - 12, count)
    fluxes = 10 ** sky.uniform(3.0, 4.2, count)
    common = ndimage.gaussian_filter(sky.normal(0.0, 1.0, shape), 6.0)
    common *= 6.0 / common.std()
    # filter: (star scale, sky noise, FWHM, sky level)
    channels = {
        "l": (1.0, 2.0, 3.0, 300.0), "r": (0.5, 1.5, 3.0, 90.0), "g": (0.6, 1.5, 3.9, 140.0),
        "b": (0.4, 3.0, 3.0, 70.0), "ha": (0.1, 1.0, 3.0, 20.0),
    }
    truth = {}
    paths = {}
    for index, (key, (scale, noise, fwhm, level)) in enumerate(channels.items()):
        sigma = fwhm / 2.3548
        image = level + scale * common
        for row, column, flux in zip(rows, columns, fluxes):
            y0, x0 = int(row) - 10, int(column) - 10
            yy, xx = np.mgrid[y0 : y0 + 21, x0 : x0 + 21]
            image[y0 : y0 + 21, x0 : x0 + 21] += scale * flux / (2 * math.pi * sigma**2) * np.exp(
                -((yy - row) ** 2 + (xx - column) ** 2) / (2 * sigma**2)
            )
        truth[key] = image.copy()
        image += np.random.default_rng(10 + index).normal(0.0, noise, shape)
        if key == "l":
            image[:40, :40] = np.nan
        if key == "r":
            image[300:310] = np.nan
        paths[key] = tmp_path / f"{key.upper()}.fits"
        fits.writeto(paths[key], image.astype(np.float32), fits.Header({"FILTER": key.upper(), "CRVAL1": 150.0}))

    record = build_super_luminance(paths, "l", tmp_path / "SuperL.fits", filter_names={key: key.upper() for key in paths})

    assert record["status"] == "CREATED", record
    assert {key: entry["included"] for key, entry in record["channels"].items()} == {
        "l": True, "r": True, "g": False, "b": True,
    }
    assert "wider" in record["channels"]["g"]["reason"]
    assert record["channels"]["b"]["fwhmRatio"] == pytest.approx(1.0, abs=0.05)
    for key in ("r", "b"):
        assert record["channels"][key]["scale"] == pytest.approx(channels[key][0], rel=0.03)
    expected = {key: channels[key][0] ** 2 / channels[key][1] ** 2 for key in ("l", "r", "b")}
    total = sum(expected.values())
    for key, value in expected.items():
        assert record["channels"][key]["weight"] == pytest.approx(value / total, rel=0.1)
    assert record["effectiveGain"] == pytest.approx(math.sqrt(total / expected["l"]), rel=0.05)
    assert record["stars"]["fwhmRatio"] == pytest.approx(1.0, abs=0.03)
    result = fits.getdata(tmp_path / "SuperL.fits")
    header = fits.getheader(tmp_path / "SuperL.fits")
    assert header["FILTER"] == "SuperL" and header["OAFSLCH"] == "L,R,B" and header["CRVAL1"] == 150.0
    # L's footprint and sky level, with no step where R has no data; the
    # noise falls by the predicted gain.
    assert np.isnan(result[:40, :40]).all() and np.isfinite(result[40:]).all()
    residual = result - truth["l"]
    assert float(np.nanmedian(residual)) == pytest.approx(0.0, abs=0.1)
    assert float(np.median(residual[300:310, 40:])) == pytest.approx(0.0, abs=0.3)
    noise = 1.4826 * float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
    assert noise == pytest.approx(channels["l"][1] / record["effectiveGain"], rel=0.05)
