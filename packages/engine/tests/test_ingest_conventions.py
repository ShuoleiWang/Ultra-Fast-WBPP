"""Header spellings, folder layouts and missing keywords of other people's
data run to a calibrated, solved master, and a calibration the recipe
requires is never silently left out."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Callable

from astropy.io import fits
import numpy as np
import pytest

from test_e2e import FakeSolver, _header, _stars, _subpixel_shift, _write
from ufwbpp.inventory import inventory_project
from ufwbpp.models import AssetRole
from ufwbpp.recipe import Recipe
from ufwbpp.runtime import build_e2e_request
from ufwbpp.workflows.contracts import E2EError
from ufwbpp.workflows.project import ProjectE2ERequest, classify_project_layout, run_project_e2e
from ufwbpp.workflows.single_target import E2EState, run_e2e
from ufwbpp.workflows.sources import _validated_sources

SHAPE = (128, 128)
BIAS_LEVEL = 1000.0
SKY_LEVEL = 500.0
SHIFTS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
STANDARD = Recipe.from_dict({"calibration": {"workflow": "mono-standard-v1", "bias": "OPTIONAL"}})
Edit = Callable[[str, fits.Header], None]


def _vignetting() -> np.ndarray:
    y, x = np.indices(SHAPE, dtype=np.float64)
    return np.clip(0.82 + 0.18 * (1.0 - ((x - 63.5) ** 2 + (y - 63.5) ** 2) / (2 * 92.0**2)), 0.72, 1.0)


def _frame(path: Path, role: str, data: np.ndarray, *, exposure: float, observed_at: str, edit: Edit | None) -> Path:
    header = _header(role, exposure=exposure, observed_at=observed_at)
    if edit is not None:
        edit(role, header)
    return _write(path, data, header)


def _night(
    root: Path,
    *,
    day: str = "01",
    edit: Edit | None = None,
    light_folder: str = "LIGHT",
    shifts: tuple[tuple[float, float], ...] = SHIFTS,
    dark_signal: np.ndarray | float = 14.0,
    darks: Path | None = None,
    seed: int = 73,
) -> None:
    """One night of 60 s Lights of the star field with their Bias, 60 s Darks
    and 2 s Flats; the Lights and Flats share one vignetting."""

    rng = np.random.default_rng(seed)
    response = _vignetting()
    for index in range(3):
        _frame(
            root / "BIAS" / f"bias_{index:02d}.fits",
            "Bias",
            BIAS_LEVEL + rng.normal(0.0, 1.0, SHAPE),
            exposure=0.001,
            observed_at=f"2026-01-{day}T18:00:00Z",
            edit=edit,
        )
        _frame(
            (darks or root / "DARK") / f"dark_{index:02d}.fits",
            "Dark",
            BIAS_LEVEL + dark_signal + rng.normal(0.0, 1.2, SHAPE),
            exposure=60.0,
            observed_at=f"2026-01-{day}T18:10:00Z",
            edit=edit,
        )
    for index in range(6):
        _frame(
            root / "FLAT" / f"flat_{index:02d}.fits",
            "Flat",
            BIAS_LEVEL + 24_000.0 * response + rng.normal(0.0, 3.0, SHAPE),
            exposure=2.0,
            observed_at=f"2026-01-{day}T18:20:00Z",
            edit=edit,
        )
    base = _stars(SHAPE)
    for index, (dx, dy) in enumerate(shifts):
        raw = BIAS_LEVEL + dark_signal + _subpixel_shift(base, dx, dy) * response + rng.normal(0.0, 1.0, SHAPE)
        _frame(
            root / light_folder / f"img_{index:02d}.fits",
            "Light",
            raw,
            exposure=60.0,
            observed_at=f"2026-01-{day}T20:{index:02d}:00Z",
            edit=edit,
        )


def _request(root: Path, output: Path, recipe: Recipe = STANDARD):
    inventory = inventory_project([root])
    request = build_e2e_request(
        inventory,
        recipe,
        output,
        workers=2,
        ra_hint_degrees=150.0,
        dec_hint_degrees=20.0,
        field_of_view_degrees=3.0,
        search_radius_degrees=5.0,
    )
    return inventory, request


def _corner_to_centre(path: Path) -> float:
    """Background of a corner over the centre: 1 when the vignetting was
    divided out, 0.95 when it was not."""

    data = fits.getdata(path, memmap=False).astype(np.float64)
    height, width = data.shape
    centre = np.nanmedian(data[height // 2 - 10 : height // 2 + 10, width // 2 - 10 : width // 2 + 10])
    return float(np.nanmedian(data[4:20, 4:20]) / centre)


def _spelled(**filters: str) -> Edit:
    def edit(role: str, header: fits.Header) -> None:
        if role in filters:
            header["FILTER"] = filters[role]

    return edit


def _maxim_image_types(role: str, header: fits.Header) -> None:
    header["IMAGETYP"] = {"Light": "Light Frame", "Flat": "Flat Field", "Dark": "Dark Frame", "Bias": "Bias Frame"}[role]


def _bare_lights(role: str, header: fits.Header) -> None:
    """A camera without a filter wheel and a writer that names neither the
    frame type nor the target."""

    if role in {"Light", "Flat"}:
        del header["FILTER"]
    if role == "Light":
        del header["IMAGETYP"]
        del header["OBJECT"]


@pytest.mark.parametrize(
    ("edit", "light_folder", "channel", "target"),
    [
        pytest.param(_spelled(Light="Red", Flat="Red"), "LIGHT", "R", "SYNTHETIC-FIELD", id="lights-and-flats-red"),
        pytest.param(_spelled(Flat="Red"), "LIGHT", "R", "SYNTHETIC-FIELD", id="lights-r-flats-red"),
        pytest.param(_maxim_image_types, "LIGHT", "R", "SYNTHETIC-FIELD", id="light-frame-flat-field"),
        pytest.param(_bare_lights, "玫瑰星云/2026-01-01", "NOFILTER", "玫瑰星云", id="no-type-filter-or-object"),
    ],
)
def test_other_peoples_conventions_reach_a_flat_fielded_solved_master(
    tmp_path: Path, edit: Edit, light_folder: str, channel: str, target: str
) -> None:
    _night(tmp_path / "in", edit=edit, light_folder=light_folder)
    inventory, request = _request(tmp_path / "in", tmp_path / "out")
    lights = [asset for asset in inventory.assets if asset.role is AssetRole.LIGHT]
    assert {(asset.filter_name, asset.target) for asset in lights} == {(channel, target)} and len(lights) == 8

    result = run_project_e2e(
        ProjectE2ERequest(inventory, request, request.output_directory), solver_backends=(FakeSolver(),)
    )

    assert result.success is True, result
    (product,) = sorted(Path(result.output_directory).glob("*.fits"))
    assert product.name == f"{channel}.fits"
    assert fits.getheader(product)["FILTER"] == channel
    assert _corner_to_centre(product) == pytest.approx(1.0, abs=0.02)


def test_a_light_without_the_flat_the_recipe_requires_stops_the_run(tmp_path: Path) -> None:
    # The Flats are of another filter: no plan would start this run, so the
    # request is built without one, as a caller that skips the plan would.
    _night(tmp_path / "in", edit=_spelled(Flat="G"))
    inventory = inventory_project([tmp_path / "in"])
    optional = Recipe.from_dict({"calibration": {"workflow": "mono-standard-v1", "bias": "OPTIONAL", "flat": "OPTIONAL"}})
    request = build_e2e_request(
        inventory, optional, tmp_path / "out", workers=2, ra_hint_degrees=150.0, dec_hint_degrees=20.0,
        field_of_view_degrees=3.0, search_radius_degrees=5.0,
    )
    assert request.pipeline_parameters.required_calibration == ()
    assert _request(tmp_path / "in", tmp_path / "unused", optional)[1].pipeline_parameters.required_calibration == ()
    required = replace(request, pipeline_parameters=replace(request.pipeline_parameters, required_calibration=("FLAT",)))

    stages: list[tuple[str, str]] = []
    with pytest.raises(E2EError) as raised:
        run_e2e(required, solver_backends=(FakeSolver(),), progress=lambda event: stages.append((event.stage.value, event.status)))

    assert raised.value.code == "FLAT_MATCH_MISSING"
    assert "no flat fits 8 Light(s) of filter R and the recipe requires one" in str(raised.value)
    assert not (tmp_path / "out").exists()
    # It stops where the calibration is paired for registration: no Light is registered first.
    assert ("calibration", "started") in stages and not any(stage == "registration" for stage, _ in stages)


def test_the_recipes_requirements_reach_the_runs_pairing(tmp_path: Path) -> None:
    _night(tmp_path / "in", shifts=SHIFTS[:2])
    _, request = _request(tmp_path / "in", tmp_path / "out")
    assert request.pipeline_parameters.required_calibration == ("FLAT",)
    strict = Recipe.from_dict({"calibration": {"dark": "REQUIRED"}})
    assert _request(tmp_path / "in", tmp_path / "out", strict)[1].pipeline_parameters.required_calibration == (
        "BIAS",
        "DARK",
        "FLAT",
    )


def test_night_folders_with_flat_darks_take_the_library_dark_of_the_lights_exposure(tmp_path: Path) -> None:
    # A dark signal that matters: +200..300 ADU in 60 s, a thirtieth of it in the 2 s Flat-Darks.
    x = np.indices(SHAPE, dtype=np.float64)[1]
    dark_signal = 200.0 + 100.0 * (x / (SHAPE[1] - 1))
    night = tmp_path / "in" / "NIGHT_1"
    _night(night, dark_signal=dark_signal, darks=tmp_path / "in" / "DarkLibrary")
    rng = np.random.default_rng(5)
    for index in range(5):
        _frame(
            night / "DARKFLAT" / f"darkflat_{index:02d}.fits",
            "DARKFLAT",
            BIAS_LEVEL + dark_signal * (2.0 / 60.0) + rng.normal(0.0, 1.2, SHAPE),
            exposure=2.0,
            observed_at="2026-01-01T18:30:00Z",
            edit=None,
        )
    _, request = _request(tmp_path / "in", tmp_path / "out")

    result = run_e2e(request, solver_backends=(FakeSolver(),))

    assert result.success is True and result.state is E2EState.SOLVED, result
    matching = json.loads((tmp_path / "out" / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))[
        "calibrationMatching"
    ]
    assert matching["algorithm"] == "wbpp-compatible-calibration-matching-v2"
    assert [(item["dark"], item["lightCount"]) for item in matching["lightPairings"]] == [("60", 8)]
    assert matching["flatPairings"]["R"]["dark"] == "2"
    assert matching["warnings"] == []
    (product,) = [Path(path) for path in result.product_paths]
    # The sky of the scene times the flat's median response (about 486); with
    # the night's 2 s Flat-Darks the 60 s dark signal stayed in it (about 730).
    assert float(np.nanmedian(fits.getdata(product, memmap=False))) == pytest.approx(SKY_LEVEL, abs=30.0)


def test_a_project_with_bias_frames_of_each_night_builds_one_master_per_night(tmp_path: Path) -> None:
    for day, shifts, seed in (("01", SHIFTS[:4], 73), ("02", SHIFTS[4:], 74)):
        _night(tmp_path / "in" / f"DATE_2026-01-{day}", day=day, shifts=shifts, seed=seed)
    inventory, request = _request(tmp_path / "in", tmp_path / "out")

    result = run_project_e2e(
        ProjectE2ERequest(inventory, request, request.output_directory), solver_backends=(FakeSolver(),)
    )

    assert result.success is True, result
    output = Path(result.output_directory)
    shared = json.loads((output / "details" / "shared-calibration" / "receipt.json").read_text(encoding="utf-8"))
    assert [len(shared[key]) for key in ("masterBiasFiles", "masterDarkFiles", "masterFlatFiles")] == [2, 2, 2]
    (pixels,) = output.glob("details/runs/*/receipts/pixel-pipeline.json")
    pairings = json.loads(pixels.read_text(encoding="utf-8"))["calibrationMatching"]["lightPairings"]
    assert sorted((item["bias"], item["dark"], item["flat"], item["lightCount"]) for item in pairings) == [
        ("bias|DATE=2026-01-01", "60|DATE=2026-01-01", "R|DATE=2026-01-01", 4),
        ("bias|DATE=2026-01-02", "60|DATE=2026-01-02", "R|DATE=2026-01-02", 4),
    ]
    assert _corner_to_centre(output / "R.fits") == pytest.approx(1.0, abs=0.02)


def test_targets_named_beyond_ascii_are_separate_targets_with_safe_run_folders(tmp_path: Path) -> None:
    from conftest import write_frame

    for index, name in enumerate(("玫瑰星云", "面纱星云", "NGC 7000 北", "NGC 7000 南")):
        light = write_frame(tmp_path / name / "LIGHT" / "img_0001.fits", "Light")
        fits.delval(light, "OBJECT")
        fits.setval(light, "DATE-OBS", value=f"2026-08-31T12:0{index}:00Z")

    layout = classify_project_layout(inventory_project([tmp_path]))

    assert sorted(panel.target for panel in layout.panels) == ["NGC 7000 北", "NGC 7000 南", "玫瑰星云", "面纱星云"]
    assert len(layout.target_keys) == 4
    assert all(panel.panel_id.isascii() and panel.panel_id.replace("_", "").isalnum() for panel in layout.panels)


def test_a_light_without_a_declared_sensor_range_is_refused_before_any_measurement(tmp_path: Path) -> None:
    _night(tmp_path / "in", shifts=SHIFTS[:3])
    _, request = _request(tmp_path / "in", tmp_path / "out")
    converted = Path(request.light_files[0])
    data = fits.getdata(converted, memmap=False).astype(np.float32)
    header = fits.getheader(converted)
    converted.unlink()
    fits.writeto(converted, data, header)

    with pytest.raises(E2EError) as raised:
        _validated_sources(request, None)

    assert raised.value.code == "CALIBRATION_NUMERIC_DOMAIN_AMBIGUOUS"
    assert raised.value.path == str(converted.resolve())
