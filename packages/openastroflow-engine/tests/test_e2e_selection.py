"""E2E behaviour of the unattended selection policy."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from openastroflow_engine.e2e import run_e2e
from openastroflow_engine.selection import SelectionParameters
from test_e2e import FakeSolver, _request, synthetic_project  # noqa: F401  (fixture re-export)


def _master_hashes(output: Path) -> set[tuple[str, str]]:
    receipt = json.loads((output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))
    hashes = {
        (item["kind"], item.get("sha256"))
        for item in receipt.get("outputs", [])
        if "MASTER_LIGHT" in str(item.get("kind"))
    }
    assert hashes, "pixel-pipeline receipt lists no master light outputs"
    return hashes


def test_unattended_policy_keeps_thin_evidence_review_frame_with_reduced_weight(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]]
) -> None:
    output = tmp_path / "unattended"
    request = replace(
        _request(synthetic_project, output, lights=synthetic_project["lights"]),
        selection=SelectionParameters(policy="unattended-v1"),
    )
    result = run_e2e(request, solver_backends=(FakeSolver(),))
    assert result.success is True
    review_path = str(synthetic_project["lights"][8].resolve(strict=True))
    assert review_path in result.passed_light_paths
    assert result.excluded_light_paths == ()

    manifest = json.loads((output / "qc" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manualReviewApprovals"]["defaultDisposition"] == "DECIDED_BY_SELECTION_POLICY"
    # Published receipts carry share-safe source paths (source/<id>/<name>),
    # so frames are matched by their file name.
    review_name = Path(review_path).name
    decisions = {Path(item["path"]).name: item for item in manifest["selection"]["frames"]}
    kept = decisions[review_name]
    assert kept["action"] == "KEEP" and kept["confidence"] == 0.5
    # The weight multiplier is the confidence times the balanced-priority PSF factor.
    assert kept["weightMultiplier"] == pytest.approx(0.5 * kept["psfFactor"])
    assert 0.5 < kept["psfFactor"] < 2.0
    assert any("GATE_INSUFFICIENT_NIGHT_BASELINE" in reason["code"] for reason in kept["reasons"])
    assert manifest["selection"]["counts"] == {"KEEP": 8, "EXCLUDE": 0, "KEEP_REDUCED_WEIGHT": 1}

    selection = json.loads((output / "qc" / "selection.json").read_text(encoding="utf-8"))
    reports = selection["counterfactual"]
    assert reports and all(report["tilesUsed"] >= 1 for report in reports.values())
    frames = [frame for report in reports.values() for frame in report["frames"]]
    assert len(frames) == 9
    assert all(isinstance(frame["deltaDepthMag"], float) for frame in frames)
    annotated = {Path(item["path"]).name: item for item in selection["frames"]}
    assert Path(annotated[review_name]["counterfactual"]["path"]).name == review_name
    assert annotated[review_name]["confidence"] == 0.5

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["qualityControl"]["selectionPolicy"] == "unattended-v1"
    assert receipt["qualityControl"]["selection"] == "qc/selection.json"
    pixel = json.loads((output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))
    weights = {
        Path(path).name: float(record["qualityWeight"])
        for path, record in pixel["registration"].items()
        if isinstance(record, dict) and "qualityWeight" in record
    }
    review_weight = weights[Path(review_path).name]
    others = sorted(weight for name, weight in weights.items() if name != Path(review_path).name)
    # The thin-evidence frame keeps half of its registration quality weight.
    assert review_weight < 0.75 * others[len(others) // 2]


def test_unattended_policy_is_bit_identical_when_every_frame_passes(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]]
) -> None:
    """Depth priority applies no PSF factor, so an all-PASS set integrates identically."""

    legacy_output = tmp_path / "legacy"
    unattended_output = tmp_path / "unattended"
    base = _request(synthetic_project, legacy_output)
    legacy = run_e2e(base, solver_backends=(FakeSolver(),))
    unattended = run_e2e(
        replace(
            base,
            output_directory=str(unattended_output),
            selection=SelectionParameters(policy="unattended-v1", priority="depth"),
        ),
        solver_backends=(FakeSolver(),),
    )
    assert legacy.success and unattended.success
    assert legacy.passed_light_paths == unattended.passed_light_paths
    assert _master_hashes(legacy_output) == _master_hashes(unattended_output)
    selection = json.loads((unattended_output / "qc" / "selection.json").read_text(encoding="utf-8"))
    assert selection["counts"] == {"KEEP": 8, "EXCLUDE": 0, "KEEP_REDUCED_WEIGHT": 0}


def test_confirmed_harmful_frame_triggers_a_second_integration_pass(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame the counterfactual confirms harmful is removed and the group integrated again."""

    import openastroflow_engine.e2e as e2e_module
    from openastroflow_engine.selection import policy as policy_module

    # Two candidates: at most one of them can be the group's normalization
    # reference, which the second pass must keep.
    victims = {str(synthetic_project["lights"][index].resolve(strict=True)) for index in (3, 4)}
    original_annotate = policy_module.annotate_with_counterfactual

    def rigged_annotate(decisions, report, parameters):
        annotated = original_annotate(decisions, report, parameters)
        return [
            replace(item, suggestion="EXCLUDE_CONFIRMED_HARMFUL")
            if item.path in victims and item.admitted and item.suggestion != "EXCLUDE_CONFIRMED_HARMFUL"
            else item
            for item in annotated
        ]

    monkeypatch.setattr(e2e_module, "annotate_with_counterfactual", rigged_annotate)
    output = tmp_path / "second-pass"
    request = replace(
        _request(synthetic_project, output),
        selection=SelectionParameters(policy="unattended-v1", priority="depth"),
    )
    result = run_e2e(request, solver_backends=(FakeSolver(),))
    assert result.success is True
    selection = json.loads((output / "qc" / "selection.json").read_text(encoding="utf-8"))
    excluded = set(result.excluded_light_paths)
    assert excluded and excluded <= victims, selection.get("reintegration")
    assert len(result.passed_light_paths) == 8 - len(excluded)
    assert selection["integrationPasses"] == 2
    assert selection["reintegration"]["status"] == "APPLIED"
    assert {Path(item).name for item in selection["reintegration"]["excluded"]} == {Path(item).name for item in excluded}
    kept = victims - excluded
    assert {Path(item).name for item in selection["reintegration"]["keptBecause"]} == {Path(item).name for item in kept}
    # The confirming counterfactual numbers of the first pass survive in the record.
    evidence = selection["reintegration"]["passes"][0]["evidence"]
    assert {Path(item).name for item in evidence} == {Path(item).name for item in victims}
    assert all("deltaDepthMag" in item for item in evidence.values())
    decisions = {Path(item["path"]).name: item for item in selection["frames"]}
    for path in excluded:
        assert decisions[Path(path).name]["action"] == "EXCLUDE"
        assert decisions[Path(path).name]["reasons"][-1]["code"] == "SEL_COUNTERFACTUAL_CONFIRMED_HARMFUL"
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["qualityControl"]["passedLights"] == 8 - len(excluded)
    assert receipt["qualityControl"]["excludedLights"] == len(excluded)
    # The second pass measured only the remaining frames.
    frames = [frame for report in selection["counterfactual"].values() for frame in report["frames"]]
    assert len(frames) == 8 - len(excluded)
    assert not (output / "work").exists()


def test_region_weight_maps_weight_samples_and_are_recorded(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame with a region weight map keeps its clean area at full weight and
    its blanked area contributes nothing; receipts record the map."""

    import numpy as np

    import openastroflow_engine.e2e as e2e_module
    from openastroflow_engine.selection.region import RegionWeightMap

    target = str(synthetic_project["lights"][3].resolve(strict=True))
    nodes = np.ones((4, 4))
    nodes[:, 0] = 0.0  # the left quarter of the field is blanked

    def synthetic_maps(results, paths=None):
        return {
            target: RegionWeightMap(
                target, 4, 4, tuple(tuple(map(float, row)) for row in nodes), 4, 0.0, 0.75, {"synthetic": 4}
            )
        }

    monkeypatch.setattr(e2e_module, "region_weight_maps", synthetic_maps)
    plain_output = tmp_path / "plain"
    mapped_output = tmp_path / "mapped"
    base = _request(synthetic_project, plain_output)
    plain = run_e2e(
        replace(
            base,
            selection=SelectionParameters(policy="unattended-v1", priority="depth", region_weights=False),
        ),
        solver_backends=(FakeSolver(),),
    )
    mapped = run_e2e(
        replace(
            base,
            output_directory=str(mapped_output),
            selection=SelectionParameters(policy="unattended-v1", priority="depth", region_weights=True),
        ),
        solver_backends=(FakeSolver(),),
    )
    assert plain.success and mapped.success
    assert plain.passed_light_paths == mapped.passed_light_paths
    # The map changes the pixels of the target's group only.
    assert _master_hashes(plain_output) != _master_hashes(mapped_output)

    selection = json.loads((mapped_output / "qc" / "selection.json").read_text(encoding="utf-8"))
    assert selection["parameters"]["regionWeights"] is True
    region = selection["regionWeights"]
    assert region["algorithm"] == "region-weights-grid-v1"
    assert [Path(item["path"]).name for item in region["frames"]] == [Path(target).name]
    assert region["frames"][0]["zeroFraction"] == 0.25
    assert region["frames"][0]["nodes"][0][0] == 0.0 and region["frames"][0]["nodes"][0][3] == 1.0
    decisions = {Path(item["path"]).name: item for item in selection["frames"]}
    assert decisions[Path(target).name]["action"] == "KEEP"
    manifest = json.loads((mapped_output / "qc" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["regionWeights"]["frames"][0]["zeroCells"] == 4

    pixel = json.loads((mapped_output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))
    groups = pixel["statistics"]["integrationGroups"]
    mapped_groups = {name: record for name, record in groups.items() if record["regionWeightMaps"]}
    assert len(mapped_groups) == 1
    (record,) = mapped_groups.values()
    assert [Path(item["path"]).name for item in record["regionWeightMaps"]] == [Path(target).name]
    assert record["regionWeightMaps"][0]["frame"] == "registered"
    assert 0.0 < record["regionWeightMaps"][0]["meanWeight"] < 1.0
    assert '"regionWeights"' in json.dumps(record["integration"])
    for name, other in groups.items():
        if name not in mapped_groups:
            assert '"regionWeights"' not in json.dumps(other["integration"])
    receipt = json.loads((mapped_output / "receipt.json").read_text(encoding="utf-8"))
    executions = receipt["integration"]["ordinaryExecutions"]
    (mapped_name,) = mapped_groups
    assert executions[mapped_name]["regionWeights"]["frames"] == 1
    assert executions[mapped_name]["regionWeights"]["reducer"] in {
        "native-cpu-masked-mean-v2",
        "numpy-reference",
    }
    # The counterfactual still reports every admitted frame.
    frames = [frame for report in selection["counterfactual"].values() for frame in report["frames"]]
    assert len(frames) == 8
