from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightframeqc.analysis as analysis
import lightframeqc.analysis_cache as cache_module
from lightframeqc.config import DEFAULT_CONFIG
from lightframeqc.models import (
    FileIdentity, FrameFeatures, FrameMeasurement, FrameMetadata, FrameRole,
    GateDisposition, QualityGateResult, RegistrationMetrics, Star,
)


CONFIG = replace(DEFAULT_CONFIG, grid_rows=4, grid_columns=4)


def _frame(index: int, filter_name: str = "L") -> FrameMeasurement:
    return FrameMeasurement(
        metadata=FrameMetadata(
            path=f"/synthetic/{filter_name}-{index}.fits", width=64, height=64,
            channels=1, filter_name=filter_name, camera="TEST", target="M42",
            exposure_seconds=60.0, role=FrameRole.LIGHT,
            observed_at=datetime(2026, 9, 1, 12, index, tzinfo=timezone.utc),
            header={"OBJECT": "M42"},
        ),
        stars=[Star(
            x=5.0 + column * 9, y=5.0 + row * 9,
            flux=1000.0 + row * 100 + column * 10, peak=100.0,
            a=1.2, b=1.1, theta=0.0, fwhm=2.5, ellipticity=0.08,
        ) for row in range(6) for column in range(6)],
        preview_width=64, preview_height=64, image_median=100.0, image_mad=2.0,
        background_grid=[[100.0] * 4 for _ in range(4)],
        texture_grid=[[2.0] * 4 for _ in range(4)],
        identity=FileIdentity(f"{index:064x}", 1000, 2000, 1, index),
        thumbnail_path=f"/cold/thumb-{index}.jpg",
    )


@pytest.fixture(autouse=True)
def _small_deterministic_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    # These are cache contract tests. Exact synthetic catalogs need no triangle
    # search; exercise all group features/scoring using their known transform.
    monkeypatch.setattr(analysis, "_registration", lambda frame, reference, config, **kwargs: analysis._identity_registration(frame))


def _run(frames: list[FrameMeasurement], directory: Path, config=CONFIG, stats=None):
    return analysis.analyze_measurements(frames, config, cache_directory=directory, cache_stats=stats)


def _forbid_analysis(*args, **kwargs):
    pytest.fail("Warm cohort should skip field splitting and group analysis")


def test_warm_cache_preserves_all_results_rebinds_current_objects_and_has_no_gate(tmp_path, monkeypatch):
    frames = [_frame(0), _frame(1)]
    stats: dict[str, int] = {}
    cold = _run(frames, tmp_path, stats=stats)
    fresh = [replace(frame, metadata=replace(frame.metadata), identity=replace(frame.identity), thumbnail_path=f"/warm/thumb-{index}.jpg") for index, frame in enumerate(frames)]
    # A later caller may attach a gate; it never enters persisted analysis.
    cold[1][0].quality_gate = QualityGateResult(GateDisposition.PASS)
    monkeypatch.setattr(analysis, "_split_auto_fields", _forbid_analysis)
    monkeypatch.setattr(analysis, "_analyze_group", _forbid_analysis)
    warm = _run(fresh, tmp_path, replace(CONFIG, make_thumbnails=False), stats)
    assert cold[0] == warm[0]
    assert stats == {"misses": 1, "hits": 1}
    for index, (before, after) in enumerate(zip(cold[1], warm[1], strict=True)):
        expected = asdict(before)
        expected.update(thumbnail_path=fresh[index].thumbnail_path, quality_gate=None)
        assert asdict(after) == expected
        assert after.metadata is fresh[index].metadata
        assert after.identity is fresh[index].identity
        assert isinstance(after.features, FrameFeatures)
        assert isinstance(after.registration, RegistrationMetrics)
        assert after.metadata.role is FrameRole.LIGHT
        assert after.metadata.observed_at == frames[index].metadata.observed_at
    entry = next(tmp_path.rglob("*.json"))
    text = entry.read_text()
    assert "quality_gate" not in text and "thumbnail_path" not in text and "identity" not in text
    if os.name == "posix":
        assert entry.stat().st_mode & 0o777 == 0o600
        assert entry.parent.stat().st_mode & 0o777 == 0o700


def test_all_filters_and_separate_filter_runs_share_complete_cohort_entries(tmp_path, monkeypatch):
    frames = [_frame(0, "L"), _frame(1, "L"), _frame(2, "B"), _frame(3, "B")]
    cold = _run(frames, tmp_path)
    monkeypatch.setattr(analysis, "_split_auto_fields", _forbid_analysis)
    monkeypatch.setattr(analysis, "_analyze_group", _forbid_analysis)
    groups, results = [], []
    stats: dict[str, int] = {}
    for filter_name in ("B", "L"):
        separate = _run([frame for frame in frames if frame.metadata.filter_name == filter_name], tmp_path, stats=stats)
        groups.extend(separate[0])
        results.extend(separate[1])
    assert sorted(groups, key=lambda group: group["groupId"]) == cold[0]
    assert [asdict(result) for result in sorted(results, key=lambda result: result.path)] == [asdict(result) for result in cold[1]]
    assert stats == {"hits": 2}


@pytest.mark.parametrize("change", ["config", "identity", "stars", "metadata", "pixels", "add_frame", "implementation"])
def test_scientific_input_changes_invalidate_complete_cohort(tmp_path, monkeypatch, change):
    frames = [_frame(0), _frame(1)]
    _run(frames, tmp_path)
    config = CONFIG
    if change == "config":
        config = replace(CONFIG, minimum_overlap_fraction=0.45)
    elif change == "identity":
        frames[0].identity = replace(frames[0].identity, sha256="f" * 64)
    elif change == "stars":
        frames[0].stars[0] = replace(frames[0].stars[0], flux=1200.0)
    elif change == "metadata":
        frames[0].metadata.header["NOTE"] = "new metadata"
    elif change == "pixels":
        frames[0].background_grid[0][0] = 90.0
    elif change == "add_frame":
        frames.append(_frame(2))
    elif change == "implementation":
        monkeypatch.setattr(cache_module, "_implementation_fingerprint", lambda: "new-implementation")
    stats: dict[str, int] = {}
    _run(frames, tmp_path, config, stats)
    assert stats == {"misses": 1}


def test_auto_field_split_is_part_of_cached_work(tmp_path, monkeypatch):
    frames = [_frame(0), _frame(1)]
    for frame in frames:
        frame.metadata.target = "UNKNOWN"
    cold = _run(frames, tmp_path)
    assert cold[0][0]["groupId"].endswith("-field-1")
    monkeypatch.setattr(analysis, "_split_auto_fields", _forbid_analysis)
    assert _run(frames, tmp_path)[0] == cold[0]


@pytest.mark.parametrize("damage", ["json", "checksum", "indices", "features", "boolean_feature", "grid", "nonfinite", "gate"])
def test_corrupt_entries_fall_back_to_analysis_even_with_matching_checksum(tmp_path, damage):
    frames = [_frame(0), _frame(1)]
    cold = _run(frames, tmp_path)
    path = next(tmp_path.rglob("*.json"))
    envelope = json.loads(path.read_text())
    first = envelope["payload"]["results"][0]
    if damage == "json":
        path.write_text("{")
    elif damage == "checksum":
        envelope["sha256"] = "0" * 64
        path.write_text(json.dumps(envelope))
    else:
        if damage == "indices":
            first["registration"]["source_indices"][0] = 9999
        elif damage == "features":
            first["features"]["cloud_score"] = "KEEP"
        elif damage == "boolean_feature":
            first["features"]["fragmented_trailing_detected"] = 1
        elif damage == "grid":
            first["grid"]["rows"] = 123
        elif damage == "nonfinite":
            first["features"]["image_median"] = float("nan")
        elif damage == "gate":
            first["quality_gate"] = {"disposition": "PASS"}
        encoded = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        envelope["sha256"] = hashlib.sha256(encoded).hexdigest()
        path.write_text(json.dumps(envelope))
    stats: dict[str, int] = {}
    repaired = _run(frames, tmp_path, stats=stats)
    assert stats == {"misses": 1}
    assert repaired[0] == cold[0]
    assert [asdict(result) for result in repaired[1]] == [asdict(result) for result in cold[1]]


def test_cache_is_optional_and_io_failures_are_advisory(tmp_path, monkeypatch):
    frames = [_frame(0)]
    expected = analysis.analyze_measurements(frames, CONFIG)
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    assert _run(frames, blocked) == expected
    monkeypatch.setattr(cache_module, "_implementation_fingerprint", lambda: (_ for _ in ()).throw(ValueError("unavailable loader")))
    assert _run(frames, tmp_path / "disabled") == expected
    assert not (tmp_path / "disabled").exists()


def test_cache_retention_and_entry_size_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_MAX_ENTRIES", 2)
    for index in range(5):
        _run([_frame(index)], tmp_path)
    assert len(list(tmp_path.rglob("*.json"))) == 2
    monkeypatch.setattr(cache_module, "_MAX_ENTRY_BYTES", 10)
    _run([_frame(7)], tmp_path / "oversized")
    assert not list((tmp_path / "oversized").rglob("*.json"))
    monkeypatch.setattr(cache_module, "_MAX_ENTRY_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(cache_module, "_MAX_TOTAL_BYTES", 1)
    _run([_frame(8)], tmp_path)
    assert not list(tmp_path.rglob("*.json"))


def test_frozen_loader_code_and_dependency_versions_invalidate_fingerprint(monkeypatch):
    code = compile("value = 1", "frozen_module.py", "exec")
    loader = SimpleNamespace(get_code=lambda name: code)
    frozen = SimpleNamespace(__name__="lightframeqc.frozen", __spec__=SimpleNamespace(loader=loader))
    dependency = SimpleNamespace(__version__="1.0")
    monkeypatch.setattr(cache_module.importlib, "import_module", lambda name: frozen if name.startswith("lightframeqc.") else dependency)
    cache_module.implementation_fingerprint.cache_clear()
    try:
        original = cache_module._implementation_fingerprint()
        code = compile("value = 2", "frozen_module.py", "exec")
        cache_module.implementation_fingerprint.cache_clear()
        changed_code = cache_module._implementation_fingerprint()
        assert changed_code != original
        dependency.__version__ = "2.0"
        cache_module.implementation_fingerprint.cache_clear()
        assert cache_module._implementation_fingerprint() != changed_code
    finally:
        cache_module.implementation_fingerprint.cache_clear()
