"""WBPP grouping keywords pair each night's Lights with that night's Flats."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from astropy.io import fits
import numpy as np

from test_e2e import FakeSolver, _header, _stars, _subpixel_shift, _write
from test_pixel_pipeline import _parameters, _write_frame
from ufwbpp.stacking.pipeline import run_portable_pipeline
from ufwbpp.workflows.contracts import DrizzleOptions
from ufwbpp.workflows.single_target import E2ERequest, E2EState, run_e2e

HEIGHT, WIDTH = 12, 16


def _night_responses() -> dict[str, np.ndarray]:
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    gradient = np.broadcast_to(np.linspace(0.8, 1.2, WIDTH, dtype=np.float64), (HEIGHT, WIDTH))
    dust = 1.0 - 0.3 * np.exp(-((x - 10.0) ** 2 + (y - 5.0) ** 2) / (2 * 1.5**2))
    return {"1": np.array(gradient), "2": dust}


def _two_nights(root: Path, *, keyword_folders: bool) -> tuple[list[Path], list[Path], list[Path], np.ndarray]:
    """Four Lights of night 1 and two of night 2, each night with its own
    vignetting or dust and raw Flats that show it."""

    y, x = np.mgrid[:HEIGHT, :WIDTH]
    signal = (500.0 + 3.0 * x + 2.0 * y).astype(np.float64)
    biases = [
        _write_frame(root / "Bias" / f"bias_{index}.fits", "Bias", np.full((HEIGHT, WIDTH), 100.0 + index), exposure=0.001)
        for index in range(3)
    ]
    flats: list[Path] = []
    lights: list[Path] = []
    for night, response in _night_responses().items():
        folder = root / (f"NIGHT_{night}" if keyword_folders else f"night{night}")
        flats.extend(
            _write_frame(folder / "Flats" / f"flat_{index}.fits", "Flat", 100.0 + 20_000.0 * response, filter_name="L", exposure=2.0)
            for index in range(3)
        )
        count = 4 if night == "1" else 2
        lights.extend(
            _write_frame(folder / "Lights" / f"light_{index}.fits", "Light", 100.0 + signal * response, filter_name="L")
            for index in range(count)
        )
    return biases, flats, lights, signal


def _calibrated_ratios(output: Path, signal: np.ndarray) -> list[float]:
    """The relative spread of each calibrated Light against the signal: a
    Light flat-fielded by its own night's Flat is the signal up to a scale."""

    spreads = []
    for path in sorted((output / "calibrated").glob("*.fits")):
        ratio = fits.getdata(path).astype(np.float64) / signal
        spreads.append(float(np.std(ratio) / np.mean(ratio)))
    return spreads


def test_raw_flats_of_each_night_calibrate_that_nights_lights(tmp_path: Path) -> None:
    parameters = replace(
        _parameters(),
        calibration_workflow="mono-standard-v1",
        materialize_calibrated_lights=True,
        auto_crop=False,
    )
    biases, flats, lights, signal = _two_nights(tmp_path / "wbpp", keyword_folders=True)
    output = tmp_path / "per-night"
    run_portable_pipeline(
        bias_files=biases, flat_files=flats, light_files=lights, output_directory=output, parameters=parameters
    )
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    matching = receipt["calibrationMatching"]
    assert [group["key"] for group in matching["groups"]["FLAT"]] == ["L|NIGHT=1", "L|NIGHT=2"]
    assert {(item["flat"], item["lightCount"]) for item in matching["lightPairings"]} == {
        ("L|NIGHT=1", 4),
        ("L|NIGHT=2", 2),
    }
    assert matching["warnings"] == []
    assert sorted(path.name for path in (output / "masters").glob("master_flat_*.fits")) == [
        "master_flat_L_NIGHT_1.fits",
        "master_flat_L_NIGHT_2.fits",
    ]
    assert max(_calibrated_ratios(output, signal)) < 2e-3

    # The same frames without WBPP's keyword folders form one Flat group, as
    # WBPP without a keyword would: every Light keeps the other night's pattern.
    biases, flats, lights, signal = _two_nights(tmp_path / "plain", keyword_folders=False)
    merged = tmp_path / "merged"
    run_portable_pipeline(
        bias_files=biases, flat_files=flats, light_files=lights, output_directory=merged, parameters=parameters
    )
    receipt = json.loads((merged / "receipt.json").read_text(encoding="utf-8"))
    assert [group["key"] for group in receipt["calibrationMatching"]["groups"]["FLAT"]] == ["L"]
    # Night 2's dust survives at a tenth of the signal.
    assert max(_calibrated_ratios(merged, signal)) > 0.1


def test_a_night_without_flats_is_left_unflattened_with_a_warning(tmp_path: Path) -> None:
    parameters = replace(
        _parameters(),
        calibration_workflow="mono-standard-v1",
        materialize_calibrated_lights=True,
        auto_crop=False,
    )
    biases, flats, lights, signal = _two_nights(tmp_path / "wbpp", keyword_folders=True)
    night_one_flats = [path for path in flats if "NIGHT_1" in str(path)]
    output = tmp_path / "result"
    run_portable_pipeline(
        bias_files=biases, flat_files=night_one_flats, light_files=lights, output_directory=output, parameters=parameters
    )
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    (warning,) = [item for item in receipt["calibrationMatching"]["warnings"] if item["code"] == "FLAT_MISSING"]
    assert warning["details"]["frameCount"] == 2
    # One Flat group is left, so its key is the plain filter name.
    assert {(item["flat"], item["lightCount"]) for item in receipt["calibrationMatching"]["lightPairings"]} == {
        ("L", 4),
        (None, 2),
    }
    ratios = _calibrated_ratios(output, signal)
    assert len(ratios) == 6 and max(sorted(ratios)[:4]) < 2e-3 and max(ratios) > 0.02


def _two_night_star_project(root: Path) -> dict[str, tuple[Path, ...]]:
    shape = (128, 128)
    rng = np.random.default_rng(97)
    y, x = np.indices(shape, dtype=np.float64)
    vignette = np.clip(0.82 + 0.18 * (1.0 - ((x - 63.5) ** 2 + (y - 63.5) ** 2) / (2 * 92.0**2)), 0.72, 1.0)
    dust = 1.0 - 0.35 * np.exp(-((x - 88.0) ** 2 + (y - 40.0) ** 2) / (2 * 6.0**2))
    responses = {"1": vignette, "2": vignette * dust}
    bias_level = 1000.0
    biases = tuple(
        _write(
            root / "BIAS" / f"bias_{index:02d}.fits",
            bias_level + rng.normal(0.0, 1.0, shape),
            _header("Bias", exposure=0.001, observed_at="2026-01-01T18:00:00Z"),
        )
        for index in range(3)
    )
    base = _stars(shape)
    flats: list[Path] = []
    lights: list[Path] = []
    shifts = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.25))
    for night, response in responses.items():
        day = "01" if night == "1" else "02"
        flats.extend(
            _write(
                root / f"NIGHT_{night}" / "FLAT" / f"flat_R_{index:02d}.fits",
                bias_level + 24_000.0 * response + rng.normal(0.0, 3.0, shape),
                _header("Flat", exposure=2.0, observed_at=f"2026-01-{day}T18:20:00Z"),
            )
            for index in range(3)
        )
        for index, (dx, dy) in enumerate(shifts):
            raw = bias_level + _subpixel_shift(base, dx, dy) * response + rng.normal(0.0, 1.0, shape)
            lights.append(
                _write(
                    root / f"NIGHT_{night}" / "LIGHT" / f"light_R_{index:02d}.fits",
                    raw,
                    _header("Light", exposure=60.0, observed_at=f"2026-01-{day}T20:{index:02d}:00Z"),
                )
            )
    return {"biases": biases, "flats": tuple(flats), "lights": tuple(lights)}


def test_e2e_registration_and_pixels_share_the_per_night_masters(tmp_path: Path) -> None:
    project = _two_night_star_project(tmp_path / "input")
    output = tmp_path / "output"
    request = E2ERequest(
        light_files=tuple(str(path) for path in project["lights"]),
        flat_files=tuple(str(path) for path in project["flats"]),
        bias_files=tuple(str(path) for path in project["biases"]),
        output_directory=str(output),
        workers=2,
        ra_hint_degrees=150.0,
        dec_hint_degrees=20.0,
        field_of_view_degrees=3.0,
        search_radius_degrees=5.0,
        drizzle=DrizzleOptions(scale=2, pixfrac=0.9, kernel="square", tile_rows=64),
    )
    result = run_e2e(request, solver_backends=(FakeSolver(),))
    assert result.success is True and result.state is E2EState.SOLVED, result
    registration = json.loads((output / "receipts" / "registration-calibration.json").read_text(encoding="utf-8"))
    assert sorted(registration["masterFlats"]) == ["R|NIGHT=1", "R|NIGHT=2"]
    assert {record["mode"] for record in registration["masterFlats"].values()} == {"BUILT_FROM_RAW"}
    pixels = json.loads((output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))
    statistics = pixels["statistics"]["calibration"]
    assert statistics["masterFlat:R|NIGHT=1"]["mode"] == "REUSED_E2E_GENERATED_MASTER"
    assert statistics["masterFlat:R|NIGHT=2"]["mode"] == "REUSED_E2E_GENERATED_MASTER"
    assert {(item["flat"], item["lightCount"]) for item in pixels["calibrationMatching"]["lightPairings"]} == {
        ("R|NIGHT=1", 5),
        ("R|NIGHT=2", 5),
    }
    # Night 2's dust is divided out by night 2's Flat: the master shows no dip
    # where it was (a Flat merged over both nights would leave about half).
    (product,) = [Path(path) for path in result.product_paths]
    master = fits.getdata(product).astype(np.float64)
    y, x = np.indices(master.shape)
    spot = np.hypot(x - 88.0, y - 40.0) < 3.0
    ring = (np.hypot(x - 88.0, y - 40.0) > 14.0) & (np.hypot(x - 88.0, y - 40.0) < 20.0)
    background = np.nanmedian(master[ring])
    assert abs(np.nanmedian(master[spot]) / background - 1.0) < 0.02


def test_wbpp_panel_folders_become_mosaic_panels_of_one_target(tmp_path: Path) -> None:
    from conftest import write_frame

    from ufwbpp.inventory import inventory_project
    from ufwbpp.workflows.project import classify_project_layout

    for panel in ("1", "2"):
        for index in range(2):
            light = write_frame(tmp_path / f"PANEL_{panel}" / "Lights" / f"light_{index}.fits", "Light", target="M31")
            fits.setval(light, "DATE-OBS", value=f"2026-09-01T2{panel}:0{index}:00Z")
    write_frame(tmp_path / "Flats" / "flat.fits", "Flat", exposure=2)
    layout = classify_project_layout(inventory_project([tmp_path]))
    assert sorted(panel.target for panel in layout.panels) == ["M31 PANEL 1", "M31 PANEL 2"]
    assert {len(panel.light_files) for panel in layout.panels} == {2}



def test_a_frame_without_any_image_type_is_a_light_as_in_wbpp(tmp_path: Path) -> None:
    from conftest import write_frame

    from ufwbpp.inventory import inventory_project
    from ufwbpp.models import AssetRole

    unnamed = write_frame(tmp_path / "M31" / "frame_001.fits", "Light")
    fits.delval(unnamed, "IMAGETYP")
    focus = write_frame(tmp_path / "M31" / "frame_002.fits", "Light")
    fits.setval(focus, "IMAGETYP", value="FOCUS")
    inventory = inventory_project([tmp_path])
    roles = {Path(asset.path).name: asset.role for asset in inventory.assets}
    assert roles == {"frame_001.fits": AssetRole.LIGHT, "frame_002.fits": AssetRole.UNKNOWN}
    codes = {issue.code: Path(issue.path).name for issue in inventory.issues}
    assert codes["ROLE_DEFAULTED_TO_LIGHT"] == "frame_001.fits"
    assert codes["ROLE_UNKNOWN"] == "frame_002.fits"
