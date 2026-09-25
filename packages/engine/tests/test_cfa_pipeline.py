"""OSC / Bayer (CFA) support: the shared helpers, the per-channel flat
scaling, the colour channel groups of the pixel pipeline, the Bayer drizzle
and the project layout, on a synthetic RGGB project."""

from __future__ import annotations

import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from lightframeqc import cfa
from lightframeqc.cfa import (
    bilinear_debayer,
    channel_mask,
    channel_medians,
    luminance,
    pattern_layout,
    shifted_pattern,
    superpixel_luminance,
)
from lightframeqc.native_psf import open_native_image
from lightframeqc.readers import read_frame_preview
from ufwbpp_registration.pipeline import read_full_image
from ufwbpp.stacking.integration import (
    FitsFrame,
    FrameExpression,
    _canonical_expression,
    _expression_rows,
    _expression_sampled_rows,
)
from ufwbpp.stacking.drizzle_native import (
    DrizzleError,
    DrizzleFrame,
    DrizzleGroupRequest,
    drizzle_group,
    verify_drizzle_receipt,
)
from ufwbpp.workflows.single_target import E2ERequest, E2EState, IntegrationMode, run_e2e
from ufwbpp.workflows.contracts import DrizzleOptions
from ufwbpp.native_kernels import load_native_kernels
from ufwbpp.workflows.project import ProjectE2EError, classify_project_layout
from ufwbpp.inventory import inventory_project

from test_e2e import FakeSolver, _header, _stars, _subpixel_shift, _write

requires_native = pytest.mark.skipif(
    load_native_kernels() is None, reason="native kernel library not available"
)


# ------------------------------------------------------------------ helpers
def test_pattern_helpers_cover_the_four_bayer_layouts() -> None:
    assert cfa.normalize_pattern(" rggb ") == "RGGB"
    assert cfa.normalize_pattern("NONE") == "NONE" and cfa.normalize_pattern("mono") == "NONE"
    assert cfa.normalize_pattern(None) == "UNKNOWN" and cfa.normalize_pattern("auto") == "UNKNOWN"
    assert cfa.is_cfa_pattern("GBRG") and not cfa.is_cfa_pattern("NONE") and not cfa.is_cfa_pattern("CYGM")
    assert pattern_layout("BGGR") == (2, 1, 1, 0)
    with pytest.raises(ValueError):
        pattern_layout("CYGM")
    # Sub-windows at odd offsets see the tile phase-shifted.
    assert shifted_pattern("RGGB", 0, 1) == "GRBG"
    assert shifted_pattern("RGGB", 1, 0) == "GBRG"
    assert shifted_pattern("RGGB", 1, 1) == "BGGR"
    assert shifted_pattern("GRBG", 1, 1) == "GBRG"
    mask = channel_mask((4, 4), "RGGB", 1)
    assert mask.sum() == 8 and mask[0, 1] and mask[1, 0] and not mask[0, 0]
    assert cfa.even_block_size(3) == 4 and cfa.even_block_size(4) == 4


def _planes_and_mosaic(pattern: str, shape: tuple[int, int] = (48, 60)) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    truth = np.stack(
        [100.0 + 0.5 * xx + 0.2 * yy, 200.0 + 20.0 * np.sin(xx / 9.0) + 0.1 * yy, 50.0 + 0.1 * xx + 0.3 * yy]
    )
    mosaic = np.zeros(shape)
    for channel in range(3):
        mask = channel_mask(shape, pattern, channel)
        mosaic[mask] = truth[channel][mask]
    return truth, mosaic


@pytest.mark.parametrize("pattern", sorted(cfa.CFA_PATTERNS))
def test_bilinear_debayer_keeps_samples_interpolates_smoothly_and_is_band_invariant(
    pattern: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth, mosaic = _planes_and_mosaic(pattern)
    planes = bilinear_debayer(mosaic, pattern)
    assert planes.shape == (3, 48, 60) and planes.dtype == np.float32
    for channel in range(3):
        mask = channel_mask(mosaic.shape, pattern, channel)
        np.testing.assert_array_equal(planes[channel][mask], truth[channel][mask].astype(np.float32))
    assert np.abs(planes - truth)[:, 2:-2, 2:-2].max() < 0.1
    assert np.abs(planes - truth).max() < 1.5  # edge replication
    monkeypatch.setattr(cfa, "DEBAYER_BAND_ROWS", 5)
    np.testing.assert_array_equal(bilinear_debayer(mosaic, pattern), planes)
    medians = channel_medians(mosaic, pattern)
    assert abs(medians[1] - float(np.median(truth[1][channel_mask(mosaic.shape, pattern, 1)]))) < 1e-9
    lum = luminance(mosaic, pattern)
    assert np.allclose(lum, planes.mean(axis=0), atol=1e-4)
    assert superpixel_luminance(mosaic, pattern).shape == (24, 30)


@requires_native
def test_native_debayer_is_value_identical_to_the_numpy_reference() -> None:
    from ufwbpp.native_kernels import load_native_kernels

    kernels = load_native_kernels()
    assert kernels is not None and cfa.debayer_backend() == "accelerated"
    rng = np.random.default_rng(3)
    try:
        for pattern in sorted(cfa.CFA_PATTERNS):
            for shape in ((7, 9), (64, 80), (65, 81), (1, 1), (2, 3), (130, 97)):
                mosaic = rng.uniform(0.0, 1000.0, shape).astype(np.float32)
                if shape[0] > 4:
                    mosaic[rng.integers(0, shape[0], 20), rng.integers(0, shape[1], 20)] = np.nan
                    mosaic[0, :] = np.nan
                native = kernels.debayer_bilinear(mosaic, pattern_layout(pattern))
                cfa.set_debayer_accelerator(None)
                reference = bilinear_debayer(mosaic, pattern)
                cfa.set_debayer_accelerator(lambda m, layout: kernels.debayer_bilinear(m, layout))
                assert np.array_equal(native, reference, equal_nan=True), (pattern, shape)
                assert np.array_equal(bilinear_debayer(mosaic, pattern), reference, equal_nan=True)
    finally:
        cfa.set_debayer_accelerator(lambda m, layout: kernels.debayer_bilinear(m, layout))
    with pytest.raises(ValueError):
        kernels.debayer_bilinear(np.zeros((4, 4), np.float32), (0, 1, 1, 3))


def test_debayer_treats_non_finite_samples_as_missing() -> None:
    _truth, mosaic = _planes_and_mosaic("RGGB")
    mosaic[10, 10] = np.nan  # a red sample
    planes = bilinear_debayer(mosaic, "RGGB")
    assert np.isnan(planes[0, 10, 10])
    assert np.isfinite(planes[0, 10, 11]) and np.isfinite(planes[1, 10, 10]) and np.isfinite(planes[2, 10, 10])
    assert int(np.isnan(planes).sum()) == 1


def test_cfa_frames_get_even_preview_blocks_and_luminance_stamps(tmp_path: Path) -> None:
    _truth, mosaic = _planes_and_mosaic("RGGB", shape=(90, 120))
    header = fits.Header()
    header["BAYERPAT"] = "RGGB"
    header["IMAGETYP"] = "LIGHT"
    path = tmp_path / "mosaic.fits"
    fits.PrimaryHDU(mosaic.astype(np.uint16), header=header).writeto(path)
    mono_path = tmp_path / "mono.fits"
    fits.PrimaryHDU(mosaic.astype(np.uint16)).writeto(mono_path)
    # 120 / 40 = 3 blocks for a mono frame; a Bayer frame rounds up to 4.
    assert read_frame_preview(mono_path, max_long_edge=40).block_size == 3
    preview = read_frame_preview(path, max_long_edge=40)
    assert preview.block_size == 4 and preview.metadata.cfa_pattern == "RGGB"
    with open_native_image(str(path)) as image:
        assert image is not None and image.cfa_pattern == "RGGB"
        stamp = image[5:21, 7:30]
    full = luminance(mosaic.astype(np.uint16), "RGGB")
    np.testing.assert_allclose(stamp, full[5:21, 7:30], atol=1e-3)
    with open_native_image(str(mono_path)) as image:
        assert image is not None and image.cfa_pattern is None
    np.testing.assert_allclose(read_full_image(str(path)), full, atol=1e-3)
    np.testing.assert_array_equal(read_full_image(str(mono_path)), mosaic.astype(np.uint16).astype(np.float32))


def test_pattern_scales_multiply_by_bayer_position(tmp_path: Path) -> None:
    source = tmp_path / "s.fits"
    flat = tmp_path / "f.fits"
    fits.PrimaryHDU(np.full((6, 8), 100.0, np.float32)).writeto(source)
    fits.PrimaryHDU(np.full((6, 8), 2.0, np.float32)).writeto(flat)
    expression = _canonical_expression(
        FrameExpression(str(source), divide_path=str(flat), scale=2.0, pattern_scales=(1.0, 2.0, 3.0, 4.0))
    )
    with FitsFrame(source) as light, FitsFrame(flat) as response:
        sources = {expression.source_path: light, expression.divide_path: response}
        rows = _expression_rows(expression, sources, 1, 4, division_floor=1e-6)
        sampled = _expression_sampled_rows(expression, sources, np.array([1, 3]), division_floor=1e-6)
    assert rows[0, :4].tolist() == [300.0, 400.0, 300.0, 400.0]  # odd row: positions (1,0),(1,1)
    assert rows[1, :4].tolist() == [100.0, 200.0, 100.0, 200.0]
    np.testing.assert_array_equal(sampled[0], rows[0])
    np.testing.assert_array_equal(sampled[1], rows[2])
    assert expression.serializable()["patternScales"] == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(Exception):
        _canonical_expression(FrameExpression(str(source), pattern_scales=(1.0, 0.0, 1.0, 1.0)))


# ---------------------------------------------------------- synthetic project
GAINS = {"R": 0.8, "G": 1.0, "B": 0.65}


def _gain_map(shape: tuple[int, int], pattern: str = "RGGB") -> np.ndarray:
    gains = np.empty(shape, dtype=np.float64)
    for index, channel in enumerate(pattern_layout(pattern)):
        gains[index >> 1 :: 2, index & 1 :: 2] = GAINS[("R", "G", "B")[channel]]
    return gains


@pytest.fixture(scope="module")
def cfa_project(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, ...]]:
    """An RGGB one-shot-colour project: a grey star field seen through
    channel gains, a colour-tinted flat, and mono-agnostic darks and biases."""

    root = tmp_path_factory.mktemp("e2e-cfa")
    shape = (128, 128)
    rng = np.random.default_rng(11)
    y, x = np.indices(shape, dtype=np.float64)
    response = np.clip(0.82 + 0.18 * (1.0 - ((x - 63.5) ** 2 + (y - 63.5) ** 2) / (2 * 92.0**2)), 0.72, 1.0)
    gains = _gain_map(shape)
    bias_level, dark_signal = 1000.0, 14.0

    def header(role: str, *, exposure: float, observed_at: str) -> fits.Header:
        value = _header(role, exposure=exposure, observed_at=observed_at, filter_name="LP")
        value["BAYERPAT"] = "RGGB"
        value["INSTRUME"] = "SYNTHETIC-OSC"
        return value

    biases = tuple(
        _write(root / "BIAS" / f"bias_{i:02d}.fits", bias_level + rng.normal(0.0, 1.0, shape), header("Bias", exposure=0.001, observed_at="2026-01-01T18:00:00Z"))
        for i in range(3)
    )
    darks = tuple(
        _write(root / "DARK" / f"dark_{i:02d}.fits", bias_level + dark_signal + rng.normal(0.0, 1.2, shape), header("Dark", exposure=60.0, observed_at="2026-01-01T18:10:00Z"))
        for i in range(3)
    )
    flats = tuple(
        _write(root / "FLAT" / f"flat_{i:02d}.fits", bias_level + 24_000.0 * response * gains + rng.normal(0.0, 3.0, shape), header("Flat", exposure=2.0, observed_at="2026-01-01T18:20:00Z"))
        for i in range(3)
    )
    base = _stars(shape)
    shifts = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
    lights = tuple(
        _write(
            root / "LIGHT" / f"light_{i:02d}.fits",
            bias_level + dark_signal + _subpixel_shift(base, dx, dy) * response * gains + rng.normal(0.0, 1.0, shape),
            header("Light", exposure=60.0, observed_at=f"2026-01-01T20:{i:02d}:00Z"),
        )
        for i, (dx, dy) in enumerate(shifts)
    )
    return {"biases": biases, "darks": darks, "flats": flats, "lights": lights}


def _request(project: dict[str, tuple[Path, ...]], output: Path, *, mode: IntegrationMode = IntegrationMode.ORDINARY) -> E2ERequest:
    return E2ERequest(
        light_files=tuple(str(path) for path in project["lights"]),
        flat_files=tuple(str(path) for path in project["flats"]),
        dark_files=tuple(str(path) for path in project["darks"]),
        bias_files=tuple(str(path) for path in project["biases"]),
        output_directory=str(output),
        integration_mode=mode,
        workers=2,
        ra_hint_degrees=150.0,
        dec_hint_degrees=20.0,
        field_of_view_degrees=3.0,
        search_radius_degrees=5.0,
        drizzle=DrizzleOptions(scale=2, pixfrac=0.9, kernel="square", tile_rows=64),
    )


def test_bayer_lights_integrate_into_three_colour_channel_masters(cfa_project, tmp_path: Path) -> None:
    output = tmp_path / "cfa-ordinary"
    result = run_e2e(_request(cfa_project, output), solver_backends=(FakeSolver(),))
    assert result.success is True and result.state is E2EState.SOLVED, result.message
    products = {fits.getheader(path)["FILTER"]: Path(path) for path in result.product_paths}
    assert set(products) == {"R", "G", "B"}
    headers = {name: fits.getheader(path) for name, path in products.items()}
    for name, header in headers.items():
        assert header["OAFCFA"] == "RGGB" and header["OAFCFACH"] == name and header["OAFCFAF"] == "LP"
        assert header["OAFSTATE"] == "SOLVED" and header["OAFNFRM"] == 8
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    pixel_receipt = json.loads((output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))
    groups = pixel_receipt["statistics"]["integrationGroups"]
    assert {name: (group["sourceFilter"], group["cfaChannel"], group["cfaPattern"]) for name, group in groups.items()} == {
        "R": ("LP", "R", "RGGB"), "G": ("LP", "G", "RGGB"), "B": ("LP", "B", "RGGB")
    }
    # One crop shared by the three channels, so they combine without resampling.
    assert len({tuple(group["crop"]) for group in groups.values()}) == 1
    flat_record = pixel_receipt["statistics"]["calibration"]["masterFlat:LP"]
    medians = flat_record["cfaChannelMedians"]
    assert medians["separateChannelScaling"] is True
    assert medians["R"] / medians["G"] == pytest.approx(0.8, rel=0.03)
    assert medians["B"] / medians["G"] == pytest.approx(0.65, rel=0.03)
    first = next(iter(pixel_receipt["registration"].values()))
    assert first["calibration"]["debayer"]["algorithm"] == "bilinear-same-colour-neighbours-v1"
    assert set(first["outputs"]) == {"R", "G", "B"}
    # Separate flat scaling keeps the sensor's channel response: the sky of
    # each channel master follows the synthetic gains.
    skies = {name: float(np.nanmedian(fits.getdata(path))) for name, path in products.items()}
    assert skies["R"] / skies["G"] == pytest.approx(0.8, rel=0.03)
    assert skies["B"] / skies["G"] == pytest.approx(0.65, rel=0.03)


@requires_native
def test_bayer_drizzle_drops_each_colour_from_the_mosaic(cfa_project, tmp_path: Path) -> None:
    output = tmp_path / "cfa-drizzle"
    result = run_e2e(_request(cfa_project, output, mode=IntegrationMode.DRIZZLE), solver_backends=(FakeSolver(),))
    assert result.success is True and result.state is E2EState.SOLVED, result.message
    products = {fits.getheader(path)["FILTER"]: Path(path) for path in result.product_paths}
    assert set(products) == {"R", "G", "B"}
    for name, path in products.items():
        header = fits.getheader(path)
        assert header["OAFDRZ"] == "NATIVE" and header["OAFDRZSC"] == 2
        assert header["OAFDRZCF"] == "RGGB" and header["OAFDRZCH"] == name
        receipt = json.loads((output / "receipts" / f"drizzle_{name}.json").read_text(encoding="ascii"))
        assert receipt["recipe"]["cfaPattern"] == "RGGB" and receipt["recipe"]["cfaChannel"] == name
        # Eight whole- and half-pixel dithers of 1/4 (R, B) or 1/2 (G) of the
        # samples cover the 2x grid.
        assert receipt["statistics"]["coverageFraction"] > 0.95
        with fits.open(path) as hdul:
            assert [hdu.name for hdu in hdul][:1] == ["SCI"]


def test_single_channel_bayer_drizzle_request(tmp_path: Path) -> None:
    yy, xx = np.indices((16, 16))
    mosaic = np.where((yy % 2 == 0) & (xx % 2 == 0), 10.0, np.where((yy % 2 == 1) & (xx % 2 == 1), 30.0, 20.0)).astype(np.float32)
    path = tmp_path / "cfa.fits"
    header = fits.Header()
    header["IMAGETYP"] = "Light"
    fits.PrimaryHDU(mosaic, header=header).writeto(path)
    frames = tuple(
        DrizzleFrame(str(path), "/raw/cfa.fits", ((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0)), 1.0, 30.0)
        for dx, dy in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    )
    with pytest.raises(DrizzleError) as error:
        DrizzleGroupRequest(frames=frames, reference_shape=(16, 16), output_path=str(tmp_path / "x.fits"), receipt_path=str(tmp_path / "x.json"), channel=2).validate()
    assert error.value.code == "DRIZZLE_CFA_CHANNEL_INVALID"
    if load_native_kernels() is None:
        pytest.skip("native kernel library not available")
    request = DrizzleGroupRequest(
        frames=frames, reference_shape=(16, 16), output_path=str(tmp_path / "blue.fits"), receipt_path=str(tmp_path / "blue.json"),
        scale=1, pixfrac=1.0, kernel="square", cfa_pattern="RGGB", channel=2, durable=False,
    )
    result = drizzle_group(request)
    receipt = verify_drizzle_receipt(result.receipt_path)
    assert receipt["recipe"]["cfaChannel"] == "B" and receipt["geometry"]["channels"] == 1
    science = np.asarray(fits.getdata(result.output_path), dtype=np.float64)
    assert science.shape == (16, 16)
    assert np.allclose(science[2:14, 2:14], 30.0)
    assert fits.getheader(result.output_path)["OAFDRZCH"] == "B"


# ------------------------------------------------------------- project layer
def test_project_layout_expands_bayer_lights_into_colour_channel_panels(cfa_project, tmp_path: Path) -> None:
    inventory = inventory_project([path.parent for path in (cfa_project["lights"][0], cfa_project["flats"][0])])
    layout = classify_project_layout(inventory)
    assert [(panel.filter_name, panel.filter_key, panel.source_filter, panel.cfa_channel, panel.cfa_pattern) for panel in layout.panels] == [
        ("B", "b", "LP", "B", "RGGB"), ("G", "g", "LP", "G", "RGGB"), ("R", "r", "LP", "R", "RGGB"),
    ]
    assert all(len(panel.light_files) == 8 for panel in layout.panels)
    descriptor, panels = layout.target_runs[0]
    assert len(panels) == 3 and len(descriptor.light_files) == 8  # each Light once per run
    assert layout.panels[0].serializable()["cfaChannel"] == "B"
    # A mono R Light next to the Bayer set would fight over the R channel panel.
    mono_dir = tmp_path / "mono"
    mono_dir.mkdir()
    header = _header("Light", exposure=60.0, observed_at="2026-01-01T21:00:00Z", filter_name="R")
    header["OBJECT"] = "SYNTHETIC-FIELD"
    _write(mono_dir / "light_R_00.fits", np.full((128, 128), 1200.0), header)
    mixed = inventory_project([cfa_project["lights"][0].parent, mono_dir])
    with pytest.raises(ProjectE2EError) as error:
        classify_project_layout(mixed)
    assert error.value.code == "CFA_CHANNEL_FILTER_COLLISION"
