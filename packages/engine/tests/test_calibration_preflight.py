from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astropy.io import fits
import pytest

from ufwbpp.calibration_preflight import inspect_calibration
from ufwbpp.cli import main
from ufwbpp.recipe import Recipe
from conftest import write_frame


def _inputs(root: Path, *, dark_role: str = "Dark") -> dict[str, Path]:
    frames = {
        "light": write_frame(root / "night-1" / "light.fits", "Light"),
        "bias": write_frame(root / "calibration" / "bias.fits", "Bias", exposure=0.001),
        "flat": write_frame(root / "calibration" / "flat.fits", "Flat", exposure=2),
        "dark": write_frame(root / "calibration" / "dark.fits", dark_role),
    }
    for path in frames.values():
        fits.setval(path, "CCD-TEMP", value=-10.0)
    return frames


def _codes(report: dict) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


def test_multidate_lights_share_matching_calibration_with_explicit_optical_warning(tmp_path: Path) -> None:
    frames = _inputs(tmp_path)
    light2 = write_frame(tmp_path / "night-2" / "light.fits", "Light")
    fits.setval(light2, "CCD-TEMP", value=-10.0)
    fits.setval(light2, "DATE-OBS", value="2026-09-02T18:00:00Z")
    before = {path: path.read_bytes() for path in (*frames.values(), light2)}

    report = inspect_calibration([str(tmp_path)])

    assert report["calibrationReady"] is True
    assert report["status"] == "READY"
    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["lightCount"] == 2
    assert group["observedDates"] == ["2026-08-31", "2026-09-02"]
    assert group["matches"] == {name: {"rawCount": 1, "masterCount": 0} for name in ("FLAT", "DARK", "BIAS")}
    assert "CAPTURE_SESSION_NOT_VERIFIED" in _codes(report)
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_present_flat_with_wrong_filter_is_missing_for_light(tmp_path: Path) -> None:
    frames = _inputs(tmp_path)
    fits.setval(frames["flat"], "FILTER", value="G")

    report = inspect_calibration([str(tmp_path)])

    assert report["calibrationReady"] is False
    assert "FLAT_MATCH_MISSING" in _codes(report)
    assert report["groups"][0]["matches"]["FLAT"]["rawCount"] == 0


@pytest.mark.parametrize("requirement", ["REQUIRED", "OPTIONAL"])
def test_missing_bias_is_distinct_from_competing_sources(tmp_path: Path, requirement: str) -> None:
    frames = _inputs(tmp_path)
    bias_bytes = frames["bias"].read_bytes()
    frames["bias"].unlink()
    recipe = Recipe.from_dict({"calibration": {"bias": requirement}})

    missing = inspect_calibration([str(tmp_path)], recipe)

    assert "BIAS_MATCH_MISSING" in _codes(missing)
    assert "BIAS_SOURCE_AMBIGUOUS" not in _codes(missing)
    assert missing["calibrationReady"] is (requirement == "OPTIONAL")

    frames["bias"].write_bytes(bias_bytes)
    master = write_frame(tmp_path / "calibration" / "master-bias.fits", "Master Bias", exposure=0.001)
    fits.setval(master, "CCD-TEMP", value=-10.0)

    competing = inspect_calibration([str(tmp_path)], recipe)

    assert "BIAS_SOURCE_AMBIGUOUS" in _codes(competing)
    assert "BIAS_MATCH_MISSING" not in _codes(competing)
    assert competing["calibrationReady"] is False


def test_mixed_raw_camera_library_is_blocked_before_master_construction(tmp_path: Path) -> None:
    frames = _inputs(tmp_path)
    wrong_flat = write_frame(tmp_path / "different-camera" / "flat.fits", "Flat", exposure=2)
    fits.setval(wrong_flat, "INSTRUME", value="DIFFERENT-CAMERA")

    report = inspect_calibration([str(tmp_path)])

    assert report["calibrationReady"] is False
    failure = next(issue for issue in report["issues"] if issue["code"] == "CALIBRATION_PROFILE_UNSUPPORTED")
    assert str(wrong_flat) in failure["paths"]
    # A valid Flat being present must not hide the unsupported extra library.
    assert report["groups"][0]["matches"]["FLAT"]["rawCount"] == 1
    assert frames["flat"].exists()


def test_same_exposure_raw_dark_temperature_libraries_are_not_silently_combined(tmp_path: Path) -> None:
    _inputs(tmp_path)
    warm = write_frame(tmp_path / "warm-dark" / "dark.fits", "Dark")
    fits.setval(warm, "CCD-TEMP", value=5.0)

    report = inspect_calibration([str(tmp_path)])

    assert report["calibrationReady"] is False
    assert "DARK_TEMPERATURE_MISMATCH" in _codes(report)


def test_exact_flat_dark_with_wrong_temperature_is_blocked(tmp_path: Path) -> None:
    _inputs(tmp_path)
    flat_dark = write_frame(tmp_path / "flat-dark" / "dark.fits", "Dark", exposure=2)
    fits.setval(flat_dark, "CCD-TEMP", value=5.0)

    report = inspect_calibration([str(tmp_path)])

    assert report["calibrationReady"] is False
    assert "FLAT_DARK_MISMATCH" in _codes(report)


def test_missing_raw_bias_temperature_is_reported_before_shared_master_build(tmp_path: Path) -> None:
    frames = _inputs(tmp_path)
    fits.delval(frames["bias"], "CCD-TEMP")

    report = inspect_calibration([str(tmp_path)])

    assert report["calibrationReady"] is False
    assert "RAW_MASTER_METADATA_MISSING" in _codes(report)


def test_master_dark_requires_explicit_bias_semantics_even_with_matching_header(tmp_path: Path) -> None:
    frames = _inputs(tmp_path, dark_role="Master Dark")
    unconfirmed = inspect_calibration([str(tmp_path)])
    assert "MASTER_DARK_BIAS_SEMANTICS_REQUIRED" in _codes(unconfirmed)
    assert unconfirmed["calibrationReady"] is False
    override = {
        "sourceSha256": "sha256:" + hashlib.sha256(frames["dark"].read_bytes()).hexdigest(),
        "camera": "ASI2600MM PRO", "gain": 100, "offset": 50,
        "binning": [1, 1], "filter": "R", "cfaPattern": "NONE",
        "readoutMode": "MODE 1", "temperatureCelsius": -10,
        "exposureSeconds": 120, "biasIncluded": True,
    }

    confirmed = inspect_calibration([str(tmp_path)], Recipe.from_dict({"calibration": {"masterMetadataOverrides": [override]}}))

    assert confirmed["calibrationReady"] is True
    assert "MASTER_DARK_BIAS_SEMANTICS_REQUIRED" not in _codes(confirmed)


def test_calibration_cli_returns_blocked_report_without_solver_or_pixel_execution(tmp_path: Path, capsys) -> None:
    frames = _inputs(tmp_path / "input")
    frames["flat"].unlink()
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schemaVersion": 1, "paths": [str(tmp_path / "input")], "recipe": {}}))

    code = main(["calibration-check", "--request-json", str(request), "--compact"])

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schemaVersion"] == 1
    assert report["status"] == "BLOCKED"
    assert "FLAT_MATCH_MISSING" in _codes(report)
