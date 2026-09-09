from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from openastroflow_engine.e2e import (
    E2EError,
    E2ERequest,
    bind_review_approval_selections,
)
from openastroflow_engine.quality_preflight import inspect_light_quality

from conftest import write_frame


@pytest.fixture(autouse=True)
def isolated_qc_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENASTROFLOW_QC_CACHE_DIR", str(tmp_path / "analysis-cache"))


def test_quality_preflight_reuses_analysis_but_remeasures_pixels_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openastroflow_engine.quality_preflight as preflight

    light = write_frame(tmp_path / "light.fit", "Light")
    calls = {"measure": 0, "gate": 0}
    measure, gate = preflight.measure_paths, preflight.evaluate_quality_gate

    def measured(*args, **kwargs):
        calls["measure"] += 1
        return measure(*args, **kwargs)

    def gated(*args, **kwargs):
        calls["gate"] += 1
        return gate(*args, **kwargs)

    monkeypatch.setattr(preflight, "measure_paths", measured)
    monkeypatch.setattr(preflight, "evaluate_quality_gate", gated)
    first = preflight.inspect_light_quality([light], workers=1)
    second = preflight.inspect_light_quality([light], workers=1)
    assert first["frames"] == second["frames"]
    assert first["counts"] == second["counts"]
    assert second["analysisCache"].get("hits") == 1
    assert calls == {"measure": 2, "gate": 2}
    assert all(value >= 0 for value in second["timings"].values())


def test_quality_cache_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from openastroflow_engine.quality_cache import quality_cache_directory

    monkeypatch.setenv("OPENASTROFLOW_QC_CACHE_DIR", "off")
    assert quality_cache_directory() is None


def test_quality_preflight_returns_real_content_bound_frame_gate(tmp_path: Path) -> None:
    light = write_frame(tmp_path / "盾牌座 Light.fit", "Light")
    before = light.read_bytes()

    result = inspect_light_quality([light], workers=1)

    assert result["schemaVersion"] == 1
    assert result["gatePolicyDigest"].startswith("sha256:")
    assert sum(result["counts"].values()) == 1
    assert result["frames"][0]["path"] == str(light.resolve())
    assert result["frames"][0]["sourceSha256"] == (
        "sha256:" + hashlib.sha256(before).hexdigest()
    )
    assert result["frames"][0]["disposition"] in {"REVIEW", "HARD_FAIL"}
    assert result["frames"][0]["previewDataUrl"].startswith("data:image/png;base64,")
    assert result["frames"][0]["previewSha256"].startswith("sha256:")
    assert light.read_bytes() == before


def test_review_selection_is_bound_to_current_full_request(tmp_path: Path) -> None:
    light = tmp_path / "light.fit"
    flat = tmp_path / "flat.fit"
    bias = tmp_path / "bias.fit"
    for path, value in ((light, b"light"), (flat, b"flat"), (bias, b"bias")):
        path.write_bytes(value)
    request = E2ERequest(
        light_files=(str(light),),
        flat_files=(str(flat),),
        bias_files=(str(bias),),
        output_directory=str(tmp_path / "output"),
    )
    source_sha256 = "sha256:" + hashlib.sha256(b"light").hexdigest()
    policy_digest = request.gate_policy.canonical_digest()

    bound = bind_review_approval_selections(
        request,
        [{"sourceSha256": source_sha256, "gatePolicyDigest": policy_digest}],
    )

    assert bound.review_approvals[0].source_sha256 == source_sha256
    assert bound.review_approvals[0].gate_policy_digest == policy_digest
    assert bound.review_approvals[0].request_digest.startswith("sha256:")
    with pytest.raises(E2EError, match="policy differs"):
        bind_review_approval_selections(
            request,
            [
                {
                    "sourceSha256": source_sha256,
                    "gatePolicyDigest": "sha256:" + "0" * 64,
                }
            ],
        )
