from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
from astropy.io import fits

from lightframeqc import cli


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_light(path: Path) -> None:
    """Write a small but measurable FITS light frame for the CLI smoke test."""
    rng = np.random.default_rng(20260809)
    height = width = 256
    yy, xx = np.mgrid[:height, :width]
    image = rng.normal(1000.0, 3.0, size=(height, width))
    stars = (
        (28.0, 31.0, 900.0),
        (61.0, 45.0, 1200.0),
        (103.0, 26.0, 850.0),
        (151.0, 54.0, 1100.0),
        (211.0, 34.0, 950.0),
        (39.0, 104.0, 1000.0),
        (89.0, 91.0, 1300.0),
        (139.0, 119.0, 1050.0),
        (201.0, 102.0, 1150.0),
        (54.0, 180.0, 800.0),
        (119.0, 205.0, 1250.0),
        (193.0, 188.0, 900.0),
    )
    for x0, y0, amplitude in stars:
        image += amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * 1.7**2))

    hdu = fits.PrimaryHDU(image.astype(np.float32))
    hdu.header["IMAGETYP"] = "Light Frame"
    hdu.header["FILTER"] = "=1+1<script>alert(1)</script>"
    hdu.header["EXPTIME"] = 60.0
    hdu.header["INSTRUME"] = "Synthetic Camera"
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    hdu.writeto(path)


def _reject_json_constant(value: str) -> None:
    raise AssertionError(f"non-standard JSON constant emitted: {value}")


def test_distribution_metadata_exposes_console_script_and_readme_contract() -> None:
    pyproject = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert pyproject["project"]["scripts"] == {"light-frame-qc": "lightframeqc.cli:main"}
    assert pyproject["tool"]["setuptools"]["package-dir"] == {"": "src"}

    readme = (PACKAGE_ROOT / pyproject["project"]["readme"]).read_text(encoding="utf-8")
    assert "pip install -e ." in readme
    assert "light-frame-qc doctor" in readme
    assert "不会修改 FITS/XISF header" in readme


def test_doctor_emits_machine_readable_dependency_inventory(capsys) -> None:
    return_code = cli.main(["doctor"])

    captured = capsys.readouterr()
    assert return_code == 0
    assert captured.err == ""
    inventory = json.loads(captured.out, parse_constant=_reject_json_constant)
    assert inventory["python"]
    assert inventory["light-frame-qc"]
    assert inventory["numpy"]
    assert inventory["astropy"]
    assert inventory["sep"]


def test_show_config_emits_strict_validated_json(capsys) -> None:
    return_code = cli.main(["show-config"])

    captured = capsys.readouterr()
    assert return_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out, parse_constant=_reject_json_constant)
    assert payload["grid_rows"] == 16
    assert payload["grid_columns"] == 16
    assert payload["minimum_overlap_fraction"] == 0.5


def test_show_gate_policy_emits_digest_and_mandatory_boundaries(capsys) -> None:
    return_code = cli.main(["show-gate-policy"])

    captured = capsys.readouterr()
    assert return_code == 0
    payload = json.loads(captured.out, parse_constant=_reject_json_constant)
    assert payload["policyDigest"].startswith("sha256:")
    assert payload["policy"]["minimum_pass_group_frames"] == 8
    assert payload["policy"]["minimum_night_frames_for_pass"] == 3


def test_analyze_empty_directory_fails_closed_without_reports(tmp_path: Path, capsys) -> None:
    empty_input = tmp_path / "empty"
    empty_input.mkdir()
    output = tmp_path / "output"

    return_code = cli.main(
        ["analyze", str(empty_input), "--output", str(output), "--no-thumbnails"]
    )

    captured = capsys.readouterr()
    assert return_code == 2
    assert "NO_SUPPORTED_FRAMES" in captured.err
    assert not (output / "results.json").exists()
    assert not (output / "frames.csv").exists()
    assert not (output / "report.html").exists()
    assert not any(output.iterdir())


def test_analyze_synthetic_light_writes_strict_reports_without_touching_input(
    tmp_path: Path, capsys
) -> None:
    input_path = tmp_path / "synthetic-light.fits"
    output = tmp_path / "report"
    _write_synthetic_light(input_path)

    before_bytes = input_path.read_bytes()
    before_digest = hashlib.sha256(before_bytes).hexdigest()
    before_stat = input_path.stat()

    arguments = ["analyze", str(input_path), "--output", str(output)]
    first_return_code = cli.main(arguments)
    first_captured = capsys.readouterr()
    assert first_return_code == 0, first_captured.err
    assert first_captured.err == ""
    assert "No input frame was modified, moved, or deleted." in first_captured.out
    first_thumbnails = set((output / "thumbnails").glob("*.png"))
    assert len(first_thumbnails) == 1

    # Reusing an existing output directory must not overwrite the first run's
    # review asset or turn a successful analysis into an error.
    second_return_code = cli.main(arguments)
    second_captured = capsys.readouterr()
    assert second_return_code == 0, second_captured.err
    assert second_captured.err == ""
    assert "No input frame was modified, moved, or deleted." in second_captured.out
    all_thumbnails = set((output / "thumbnails").glob("*.png"))
    assert len(all_thumbnails) == 2
    assert first_thumbnails < all_thumbnails

    json_path = output / "results.json"
    csv_path = output / "frames.csv"
    html_path = output / "report.html"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert html_path.is_file()

    json_text = json_path.read_text(encoding="utf-8")
    payload = json.loads(json_text, parse_constant=_reject_json_constant)
    assert payload["schemaVersion"] == 3
    assert len(payload["frames"]) == 1
    assert payload["frames"][0]["path"] == str(input_path.resolve())
    assert set(payload["frames"][0]["sourceIdentity"]) == {
        "sha256",
        "sizeBytes",
        "mtimeNs",
        "device",
        "inode",
    }
    assert payload["frames"][0]["qualityGate"]["disposition"] == "REVIEW"
    assert payload["frames"][0]["qualityGate"]["version"].startswith(
        "quality-gate-v1@sha256:"
    )
    assert "NaN" not in json_text
    assert "Infinity" not in json_text

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["path"] == str(input_path.resolve())
    assert rows[0]["filter"].startswith("'=")
    assert rows[0]["decision"] in {"KEEP", "REJECT", "REVIEW", "UNASSESSABLE"}

    html = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower()
    assert "synthetic-light.fits" in html
    assert "Light Frame QC" in html
    assert "=1+1&lt;SCRIPT&gt;ALERT(1)&lt;/SCRIPT&gt;" in html
    assert "=1+1<script>alert(1)</script>" not in html

    after_bytes = input_path.read_bytes()
    after_stat = input_path.stat()
    assert hashlib.sha256(after_bytes).hexdigest() == before_digest
    assert after_bytes == before_bytes
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_mode == before_stat.st_mode
