"""The master tolerance gate: bulk/outlier policy on synthetic masters,
accepted-sample counts and the evaluator-report comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from astropy.io import fits
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("master_tolerance_gate", REPOSITORY / "benchmarks" / "master_tolerance_gate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32)).writeto(path, overwrite=True)
    return path


def test_compare_master_accepts_last_ulp_and_tile_shifts_and_rejects_large_changes(tmp_path: Path) -> None:
    gate = _load_module()
    rng = np.random.default_rng(7)
    baseline = rng.normal(1000.0, 4.0, size=(256, 256)).astype(np.float32)
    baseline[0, 0] = np.nan
    candidate = baseline.copy()
    candidate[10:20, :] = np.nextafter(candidate[10:20, :], np.float32(np.inf))  # one ulp
    candidate[100:110, 100:110] += np.float32(0.004)  # a tile shift of 1e-3 sigma
    candidate[200, 200] += np.float32(6.0)  # one flipped rejection decision, 1.5 sigma
    base_path = _write(tmp_path / "base" / "L.fits", baseline)
    cand_path = _write(tmp_path / "cand" / "L.fits", candidate)
    policy = dict(ulps=16.0, bulk_sigma=0.002, bulk_fraction=0.9999, outlier_fraction=1e-4, outlier_sigma=2.0)
    record = gate.compare_master(base_path, cand_path, **policy)
    assert record["status"] == "PASS", record
    assert record["outlierPixels"] == 1
    assert 1.2 < record["maxOutlierInSigma"] < 1.8
    assert not record["sha256Identical"]

    identical = gate.compare_master(base_path, base_path, **policy)
    assert identical["status"] == "PASS" and identical["sha256Identical"] and identical["outlierPixels"] == 0

    candidate[200, 200] = baseline[200, 200] + np.float32(20.0)  # a 5 sigma outlier
    cand_path = _write(tmp_path / "cand" / "L.fits", candidate)
    record = gate.compare_master(base_path, cand_path, **policy)
    assert record["status"] == "FAIL" and "exceeds" in record["reason"]

    candidate[200, 200] = baseline[200, 200]
    candidate[:64, :] += np.float32(0.5)  # a quarter of the image shifted by 0.12 sigma
    cand_path = _write(tmp_path / "cand" / "L.fits", candidate)
    record = gate.compare_master(base_path, cand_path, **policy)
    assert record["status"] == "FAIL" and "within" in record["reason"]

    candidate[:] = baseline
    candidate[5, 5] = np.nan
    cand_path = _write(tmp_path / "cand" / "L.fits", candidate)
    assert gate.compare_master(base_path, cand_path, **policy)["reason"] == "NaN masks differ"


def test_compare_counts_reads_every_integration_map(tmp_path: Path) -> None:
    gate = _load_module()
    counts = np.full((8, 8), 30.0, dtype=np.float32)
    for run, name in (("base", "L"), ("cand", "L")):
        _write(tmp_path / run / "details" / "runs" / "run-1" / "coverage" / f"{name}_acceptedSampleCount.fits", counts)
    record = gate.compare_counts(tmp_path / "base", tmp_path / "cand", count_fraction=1e-4)
    assert record["status"] == "PASS" and record["maps"]["L_acceptedSampleCount.fits"]["baseline"] == 30 * 64
    counts[0, 0] = 29.0
    _write(tmp_path / "cand" / "details" / "runs" / "run-1" / "coverage" / "L_acceptedSampleCount.fits", counts)
    record = gate.compare_counts(tmp_path / "base", tmp_path / "cand", count_fraction=1e-4)
    assert record["status"] == "FAIL"
    assert gate.compare_counts(tmp_path / "none", tmp_path / "none", count_fraction=1e-4)["status"] == "NOT_APPLICABLE"


def _evaluation(path: Path, *, g8: float, status: str, label: str) -> Path:
    report = {
        "filter": "R",
        "metrics": [
            {"family": "noise", "metric": "G_8", "status": "PASS", "verdict": True},
            {"family": "photometry", "metric": "flat-topped star count", "status": status, "verdict": True},
        ],
        "info": {"noise": {"G": {"4": [1.02, 1.01, 1.03], "8": [g8, g8 - 0.006, g8 + 0.009]}}, "depth": {"dm": 0.0067, "ci": [-0.06, 0.07]}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    (path.parent / "summary.md").write_text(f"# Summary\n\n**R verdict: {label}** — details\n", encoding="utf-8")
    return path


def test_compare_evaluations_requires_identical_statuses_and_reports_label_boundary_crossings(tmp_path: Path) -> None:
    gate = _load_module()
    base = _evaluation(tmp_path / "base" / "R.json", g8=1.019993, status="PASS", label="EQUIVALENT")
    cand = _evaluation(tmp_path / "cand" / "R.json", g8=1.020002, status="PASS", label="INCONCLUSIVE")
    record = gate.compare_evaluations([(base, cand)])
    assert record["status"] == "PASS"
    entry = record["reports"][0]
    assert entry["filter"] == "R" and entry["changedVerdicts"] == [] and entry["exceededTolerance"] == []
    assert entry["label"] == {"baseline": "EQUIVALENT", "candidate": "INCONCLUSIVE"}
    assert "threshold" in entry["note"]
    assert abs(entry["headline"]["shift"]["G_8"] - 9e-6) < 1e-9

    worse = _evaluation(tmp_path / "worse" / "R.json", g8=1.019993, status="FAIL", label="WORSE")
    record = gate.compare_evaluations([(base, worse)])
    assert record["status"] == "FAIL" and record["reports"][0]["changedVerdicts"] == ["photometry/flat-topped star count: PASS -> FAIL"]

    shifted = _evaluation(tmp_path / "shifted" / "R.json", g8=1.03, status="PASS", label="EQUIVALENT")
    record = gate.compare_evaluations([(base, shifted)])
    assert record["status"] == "FAIL" and record["reports"][0]["exceededTolerance"] == ["G_8"]
    assert gate.compare_evaluations([(base, shifted)], tolerance=0.02)["status"] == "PASS"
    assert gate.compare_evaluations([])["status"] == "NOT_APPLICABLE"


def test_main_writes_the_report_and_exit_status(tmp_path: Path) -> None:
    gate = _load_module()
    rng = np.random.default_rng(3)
    for name in gate.PRODUCT_NAMES:
        data = rng.normal(500.0, 3.0, size=(32, 32)).astype(np.float32)
        _write(tmp_path / "base" / f"{name}.fits", data)
        _write(tmp_path / "cand" / f"{name}.fits", data)
    output = tmp_path / "gate.json"
    assert gate.main(["--baseline", str(tmp_path / "base"), "--candidate", str(tmp_path / "cand"), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS" and all(record["sha256Identical"] for record in report["masters"].values())
    assert report["integrationCounts"]["status"] == "NOT_APPLICABLE" and report["evaluations"]["status"] == "NOT_APPLICABLE"
    (tmp_path / "cand" / "B.fits").unlink()
    assert gate.main(["--baseline", str(tmp_path / "base"), "--candidate", str(tmp_path / "cand")]) == 1
