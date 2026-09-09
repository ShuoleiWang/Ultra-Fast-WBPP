"""Pixel-to-admission regression cases; no catalog or quality metrics are forged.

These synthetic FITS cases establish detector/gate wiring, not a real-data
recall estimate. In particular the wrong-field case uses a different, sparse
catalog; identifying a whole consistently wrong target requires external truth.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from lightframeqc.analysis import analyze_measurements
from lightframeqc.config import DEFAULT_CONFIG
from lightframeqc.measure import measure_paths
from lightframeqc.models import FrameResult, GateDisposition
from lightframeqc.quality_gate import evaluate_quality_gate


def _scene(*, seed: int = 17, sx: float = 1.25, sy: float = 1.25, count: int = 90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((512, 512), dtype=np.float64)
    points: list[np.ndarray] = []
    while len(points) < count:
        point = rng.uniform(20, 492, 2)
        if all(np.linalg.norm(point - other) >= 18 for other in points):
            points.append(point)
    for cx, cy in points:
        amplitude = rng.lognormal(6.8, 0.6)
        xa, xb = max(0, int(cx) - 18), min(512, int(cx) + 19)
        ya, yb = max(0, int(cy) - 18), min(512, int(cy) + 19)
        y, x = np.mgrid[ya:yb, xa:xb]
        # Conserve integrated stellar signal while broadening the PSF.
        image[ya:yb, xa:xb] += amplitude * 1.25**2 / (sx * sy) * np.exp(
            -0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2)
        )
    return image


def _write(
    root: Path,
    name: str,
    signal: np.ndarray,
    sky: np.ndarray,
    *,
    night: int,
    minute: int,
    airmass: float,
    noise_seed: int,
) -> Path:
    header = fits.Header(
        {
            "IMAGETYP": "LIGHT", "FILTER": "R", "OBJECT": "SYNTHETIC-FIELD",
            "INSTRUME": "SYNTHETIC-CAMERA", "EXPTIME": 60.0, "GAIN": 100.0,
            "OFFSET": 20.0, "XBINNING": 1, "YBINNING": 1, "BAYERPAT": "NONE",
            "DATE-OBS": (datetime(2026, 8, 9 + night, 18) + timedelta(minutes=minute)).isoformat() + "Z",
            "AIRMASS": airmass,
        }
    )
    noise = np.random.default_rng(noise_seed).normal(0, 3, signal.shape)
    path = root / f"{name}.fits"
    fits.writeto(path, np.asarray(signal + sky + noise, dtype=np.float32), header)
    return path


def _assess(paths: list[Path], output: Path) -> dict[str, FrameResult]:
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    config = replace(DEFAULT_CONFIG, make_thumbnails=False)
    measurements = measure_paths(paths, output, config, workers=1)
    _, results = analyze_measurements(measurements, config)
    evaluate_quality_gate(results, measurements)
    assert all(measurement.status == "MEASURED" for measurement in measurements)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == before[path] for path in paths)
    return {Path(result.path).stem: result for result in results}


def test_real_pixels_exclude_six_severe_defects_and_keep_two_clear_nights(tmp_path: Path) -> None:
    signal = _scene()
    y, x = np.indices(signal.shape)
    sky = 1000 + 0.02 * x + 0.01 * y
    paths: list[Path] = []
    for night in (0, 1):
        for index in range(10):
            air = 1 + index * 0.1
            scale = 10 ** (-0.4 * 0.45 * (air - 1)) * (0.85 if night else 1)
            paths.append(_write(tmp_path, f"clear_n{night}_{index:02}", signal * scale, sky,
                                night=night, minute=index * 6, airmass=air, noise_seed=100 + len(paths)))
    defects = [
        ("trailing", _scene(sx=5), sky, "GATE_COHERENT_TRAILING_HARD"),
        ("defocus", _scene(sx=3.6, sy=3.6), sky, "GATE_FOCUS_SEEING_REVIEW"),
        ("wrong_field", _scene(seed=99, count=40), sky, "GATE_REGISTRATION_REVIEW"),
        ("cloud", signal * 0.08, sky + 700, "GATE_MULTI_FAMILY_CLOUD_HARD"),
        ("occlusion", signal * (x >= 180), np.where(x < 180, 110, sky), "GATE_SOURCE_RETENTION_REVIEW"),
        ("patch_cloud", signal * np.where(x < 180, 0.08, 1), sky + np.where(x < 180, 500, 0), "GATE_SPATIAL_DIMMING_STRONG"),
    ]
    for index, (name, pixels, background, _) in enumerate(defects):
        paths.append(_write(tmp_path, name, pixels, background, night=0, minute=15 + index * 2,
                            airmass=1.3, noise_seed=100 + len(paths)))

    results = _assess(paths, tmp_path / "qc")

    for name, result in results.items():
        if name.startswith("clear_"):
            assert result.quality_gate.disposition is GateDisposition.PASS, result.quality_gate
    for name, _, _, expected_code in defects:
        gate = results[name].quality_gate
        assert gate.disposition in {GateDisposition.REVIEW, GateDisposition.HARD_FAIL}, (name, gate)
        assert expected_code in {item.code for item in gate.evidence}, (name, gate)
    assert results["patch_cloud"].features.spatial_dimming_p90_mag >= 0.45
    assert min(result.features.transparency_ratio for name, result in results.items() if name.startswith("clear_")) < 0.65


def test_whole_defocused_night_without_optional_nina_hfr_is_excluded(tmp_path: Path) -> None:
    y, x = np.indices((512, 512))
    sky = 1000 + 0.02 * x + 0.01 * y
    paths: list[Path] = []
    for night in (0, 1):
        signal = _scene(sx=3.6, sy=3.6) if night else _scene()
        for index in range(10):
            air = 1 + index * 0.1
            scale = 10 ** (-0.4 * 0.45 * (air - 1))
            paths.append(_write(tmp_path, f"night{night}_{index:02}", signal * scale, sky,
                                night=night, minute=index * 6, airmass=air, noise_seed=200 + len(paths)))

    results = _assess(paths, tmp_path / "qc")

    for name, result in results.items():
        assert result.features.nina_hfr_pixels is None
        gate = result.quality_gate
        if name.startswith("night0_"):
            assert gate.disposition is GateDisposition.PASS, gate
        else:
            assert gate.disposition is GateDisposition.REVIEW, gate
            assert "GATE_NIGHT_FOCUS_SHIFT_REVIEW" in {item.code for item in gate.evidence}


def test_denser_wrong_field_cannot_displace_the_normal_majority_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lightframeqc.analysis as analysis_module

    real_register = analysis_module.register_star_catalogs
    calls: list[tuple[int, int, int]] = []

    def counted_register(source, reference, *, thresholds):
        calls.append((len(source["points"]), len(reference["points"]), thresholds.max_control_points))
        return real_register(source, reference, thresholds=thresholds)

    monkeypatch.setattr(analysis_module, "register_star_catalogs", counted_register)
    signal = _scene()
    y, x = np.indices(signal.shape)
    sky = 1000 + 0.02 * x + 0.01 * y
    paths: list[Path] = []
    for index in range(10):
        air = 1 + index * 0.1
        scale = 10 ** (-0.4 * 0.45 * (air - 1))
        paths.append(_write(tmp_path, f"clear_{index:02}", signal * scale, sky,
                            night=0, minute=index * 6, airmass=air, noise_seed=300 + index))
    paths.append(_write(tmp_path, "zz_denser_wrong_field", _scene(seed=99, count=180), sky,
                        night=0, minute=29, airmass=1.5, noise_seed=399))

    results = _assess(paths, tmp_path / "qc")

    wrong = results["zz_denser_wrong_field"]
    assert wrong.star_count > max(result.star_count for name, result in results.items() if name.startswith("clear_"))
    assert wrong.quality_gate.disposition is GateDisposition.REVIEW
    assert "REFERENCE_CONNECTIVITY_UNRESOLVED" in wrong.registration.error
    assert all(result.reference_path != wrong.path for result in results.values())
    assert sum(control_count == 32 for _, _, control_count in calls) == 3
    assert all(control_count == 32 for source_count, reference_count, control_count in calls if max(source_count, reference_count) > 100)
    for name, result in results.items():
        if name.startswith("clear_"):
            assert result.quality_gate.disposition is GateDisposition.PASS, result.quality_gate
