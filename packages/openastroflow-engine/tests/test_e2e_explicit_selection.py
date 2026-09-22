"""E2E behaviour of the explicit (blink review) selection policy.

The bit-identity assertion is the guarantee of the selection path: an
explicit KEEP of exactly the frames the legacy gate admits integrates to
byte-identical masters, because registration, normalization and integration
never see the selection.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine.e2e import (
    E2EError,
    E2ERequest,
    E2EResult,
    ExplicitDecision,
    ExplicitSelection,
    parse_explicit_selection,
    run_e2e,
)
from openastroflow_engine.inventory import inventory_project
from openastroflow_engine.project_e2e import ProjectE2EError, ProjectE2ERequest, run_project_e2e
from openastroflow_engine.recipe import Recipe
from openastroflow_engine.runtime import RuntimeConfigurationError, build_e2e_request
from openastroflow_engine.selection import SelectionParameters
from test_e2e import FakeSolver, _header, _request, _write, synthetic_project  # noqa: F401  (fixture re-export)
from test_project_e2e import FakePanelRunner, FakeReproject, ManagedCopySolver, _project


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_file(
    decisions: dict[Path, str], *, undecided: str = "ERROR", origin: dict[str, str] | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "ultra-fast-wbpp-selection",
        "policy": "explicit-v1",
        "undecided": undecided,
        "decisions": [
            {"sourceSha256": _digest(path), "decision": decision}
            for path, decision in decisions.items()
        ],
    }
    if origin is not None:
        value["origin"] = origin
    return value


def _explicit_request(base: E2ERequest, selection: ExplicitSelection) -> E2ERequest:
    return replace(
        base,
        selection=SelectionParameters(policy="explicit-v1"),
        explicit_selection=selection,
    )


def _master_hashes(output: Path) -> set[tuple[str, str]]:
    receipt = json.loads((output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))
    hashes = {
        (item["kind"], item.get("sha256"))
        for item in receipt.get("outputs", [])
        if "MASTER_LIGHT" in str(item.get("kind"))
    }
    assert hashes, "pixel-pipeline receipt lists no master light outputs"
    return hashes


def _moonlit_light(directory: Path) -> Path:
    """A registrable Light with twice the sky and half the stars of the field.

    The blink flags pre-drop it (bright sky and a low source ratio against the
    channel's clean set) while the registration still finds its stars.
    """

    shape = (128, 128)
    rng = np.random.default_rng(20260901)
    y, x = np.indices(shape, dtype=np.float64)
    positions: list[tuple[float, float]] = []
    while len(positions) < 58:
        candidate = (float(rng.uniform(8, 120)), float(rng.uniform(8, 120)))
        if all(np.hypot(candidate[0] - px, candidate[1] - py) >= 8.0 for px, py in positions):
            positions.append(candidate)
    image = np.full(shape, 2500.0, dtype=np.float64)
    for index, (cx, cy) in enumerate(positions):
        if index % 2:
            continue
        amplitude = 1800.0 + 80.0 * (index % 9)
        image += amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.8**2))
    image += rng.normal(0.0, 2.0, shape)
    response = 0.82 + 0.18 * (1.0 - ((x - 63.5) ** 2 + (y - 63.5) ** 2) / (2 * 92.0**2))
    raw = 1000.0 + 14.0 + image * np.clip(response, 0.72, 1.0)
    return _write(
        directory / "light_R_moonlit.fits",
        raw,
        _header("Light", exposure=60.0, observed_at="2026-01-01T20:30:00Z"),
    )


def _sparse_light(directory: Path) -> Path:
    """Five stars where the field has none: far below the 12 matches a
    registration needs, so the quality pass finds no transform."""

    shape = (128, 128)
    rng = np.random.default_rng(5)
    y, x = np.indices(shape, dtype=np.float64)
    image = np.full(shape, 500.0, dtype=np.float64)
    for cx, cy in ((12.0, 12.0), (100.0, 20.0), (60.0, 70.0), (20.0, 110.0), (110.0, 100.0)):
        image += 2000.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.8**2))
    image += rng.normal(0.0, 2.0, shape)
    return _write(
        directory / "light_R_sparse.fits",
        1000.0 + 14.0 + image,
        _header("Light", exposure=60.0, observed_at="2026-01-01T20:31:00Z"),
    )


def test_selection_file_parsing_is_strict_and_digested() -> None:
    lights = {Path(f"/x/{index}.fits"): "KEEP" for index in range(2)}
    raw = {
        "schemaVersion": 1,
        "kind": "ultra-fast-wbpp-selection",
        "policy": "explicit-v1",
        "undecided": "DROP",
        "origin": {"sessionId": "s1", "blinkManifestSha256": "sha256:" + "a" * 64},
        "decisions": [
            {"sourceSha256": "sha256:" + "1" * 64, "decision": "KEEP", "defaultDecision": "DROP", "flags": ["BLINK_SKY_BRIGHT"], "note": "faint gradient acceptable"},
            {"sourceSha256": "sha256:" + "2" * 64, "decision": "DROP"},
        ],
    }
    del lights
    selection = parse_explicit_selection(raw)
    assert selection.undecided == "DROP"
    assert selection.origin == {"sessionId": "s1", "blinkManifestSha256": "sha256:" + "a" * 64}
    assert selection.decisions[0].flags == ("BLINK_SKY_BRIGHT",)
    assert selection.digest.startswith("sha256:")
    assert parse_explicit_selection(raw).digest == selection.digest
    assert parse_explicit_selection({**raw, "undecided": "KEEP"}).digest != selection.digest
    assert selection.serializable()["decisions"][1] == {"sourceSha256": "sha256:" + "2" * 64, "decision": "DROP"}
    for broken in (
        {**raw, "extra": 1},
        {**raw, "kind": "other"},
        {**raw, "policy": "unattended-v1"},
        {**raw, "undecided": "MAYBE"},
        {**raw, "origin": {"host": "x"}},
        {**raw, "decisions": [{"sourceSha256": "sha256:" + "1" * 64, "decision": "MAYBE"}]},
        {**raw, "decisions": [{"sourceSha256": "SHA256:" + "1" * 64, "decision": "KEEP"}]},
        {**raw, "decisions": [raw["decisions"][0], raw["decisions"][0]]},
        {**raw, "decisions": [{"sourceSha256": "sha256:" + "1" * 64, "decision": "KEEP", "extra": True}]},
        [],
    ):
        with pytest.raises(E2EError) as excinfo:
            parse_explicit_selection(broken)
        assert excinfo.value.code == "SELECTION_INVALID"


def test_explicit_keep_of_the_gate_set_is_bit_identical_to_the_legacy_gate(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]]
) -> None:
    legacy_output = tmp_path / "legacy"
    explicit_output = tmp_path / "explicit"
    base = _request(synthetic_project, legacy_output)
    legacy = run_e2e(base, solver_backends=(FakeSolver(),))
    assert legacy.success is True
    kept = {Path(path): "KEEP" for path in legacy.passed_light_paths}
    assert len(kept) == 8
    selection = parse_explicit_selection(
        _selection_file(kept, origin={"sessionId": "blink-test", "createdAt": "2026-09-22T00:00:00Z"})
    )
    explicit = run_e2e(
        _explicit_request(replace(base, output_directory=str(explicit_output)), selection),
        solver_backends=(FakeSolver(),),
    )
    assert explicit.success is True
    assert explicit.passed_light_paths == legacy.passed_light_paths
    assert explicit.excluded_light_paths == ()
    assert _master_hashes(legacy_output) == _master_hashes(explicit_output)

    receipt = json.loads((explicit_output / "receipt.json").read_text(encoding="utf-8"))
    quality = receipt["qualityControl"]
    assert quality["selectionPolicy"] == "explicit-v1"
    block = quality["selection"]
    assert block["policy"] == "explicit-v1"
    assert block["selectionDigest"] == selection.digest
    assert block["origin"] == {"sessionId": "blink-test", "createdAt": "2026-09-22T00:00:00Z"}
    assert block["flagsPolicyDigest"].startswith("sha256:")
    assert block["counts"] == {
        "keep": 8,
        "drop": 0,
        "overriddenExcludeFlags": 0,
        "overriddenGateHardFail": 0,
        "undecided": 0,
    }
    assert len(block["frames"]) == 8
    assert {item["decision"] for item in block["frames"]} == {"KEEP"}
    assert all(item["path"].startswith("source/") for item in block["frames"])
    blink = quality["blink"]
    assert blink["manifest"] == "qc/blink.json"
    assert blink["referenceRule"] == "psf-signal-weight-proxy-v1"
    assert len(blink["referenceBySource"]) == 1
    (reference,) = blink["referenceBySource"].values()
    assert reference["sourceSha256"] in {_digest(path) for path in kept}
    assert blink["pipelineReferences"]["registration"].startswith("source/")
    assert set(blink["pipelineReferences"]["normalization"]) == {"R"}
    evidence = json.loads((explicit_output / "qc" / "blink.json").read_text(encoding="utf-8"))
    assert evidence["kind"] == "blink-evidence-v1"
    assert evidence["counts"]["frames"] == 8
    assert len(evidence["frames"]) == 8 and sum(frame["reference"] for frame in evidence["frames"]) == 1
    assert evidence["channels"][0]["reference"]["index"] in range(8)
    assert all(frame["path"].startswith("source/") for frame in evidence["frames"])
    assert receipt["qualityControl"]["screening"]["frames"] == []
    manifest = json.loads((explicit_output / "qc" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manualReviewApprovals"]["defaultDisposition"] == "DECIDED_BY_EXPLICIT_SELECTION"
    assert manifest["explicitSelection"]["selectionDigest"] == selection.digest


def test_dropping_a_frame_changes_the_admitted_set_and_records_the_reason(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]]
) -> None:
    output = tmp_path / "dropped"
    base = _request(synthetic_project, output)
    lights = [Path(path) for path in base.light_files]
    decisions = {path: "KEEP" for path in lights}
    decisions[lights[3]] = "DROP"
    selection = parse_explicit_selection(_selection_file(decisions))
    result = run_e2e(_explicit_request(base, selection), solver_backends=(FakeSolver(),))
    assert result.success is True
    assert len(result.passed_light_paths) == 7
    assert result.excluded_light_paths == (str(lights[3].resolve(strict=True)),)
    reference = tmp_path / "reference"
    baseline = run_e2e(replace(base, output_directory=str(reference)), solver_backends=(FakeSolver(),))
    assert baseline.success and _master_hashes(reference) != _master_hashes(output)

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    screening = receipt["qualityControl"]["screening"]
    assert screening["admitted"] == 7 and screening["excluded"] == 1
    dropped = [frame for frame in screening["frames"] if frame.get("reason") == "USER_DROP"]
    assert len(dropped) == 1
    assert dropped[0]["path"].endswith(lights[3].name)
    assert dropped[0]["admitted"] is False and dropped[0]["disposition"] == "PASS"
    assert dropped[0]["flags"] == []
    block = receipt["qualityControl"]["selection"]
    assert block["counts"]["keep"] == 7 and block["counts"]["drop"] == 1
    assert [item["decision"] for item in block["frames"] if item["path"].endswith(lights[3].name)] == ["DROP"]


def test_keep_against_an_exclude_flag_is_honoured_and_recorded(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]]
) -> None:
    moonlit = _moonlit_light(tmp_path / "extra")
    output = tmp_path / "override"
    base = _request(synthetic_project, output, lights=(*synthetic_project["lights"][:8], moonlit))
    lights = [Path(path) for path in base.light_files]
    selection = parse_explicit_selection(_selection_file({path: "KEEP" for path in lights}))
    result = run_e2e(_explicit_request(base, selection), solver_backends=(FakeSolver(),))
    assert result.success is True, result.message
    assert len(result.passed_light_paths) == 9
    evidence = json.loads((output / "qc" / "blink.json").read_text(encoding="utf-8"))
    moon = next(frame for frame in evidence["frames"] if frame["name"] == moonlit.name)
    assert moon["defaultDecision"] == "DROP"
    assert {flag["code"] for flag in moon["flags"]} >= {"BLINK_SKY_BRIGHT"}
    assert moon["metrics"]["skyRatio"] > 1.6
    assert moon["reference"] is False
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    block = receipt["qualityControl"]["selection"]
    assert block["counts"]["overriddenExcludeFlags"] == 1
    kept = next(item for item in block["frames"] if item["path"].endswith(moonlit.name))
    assert kept["decision"] == "KEEP" and kept["defaultDecision"] == "DROP"
    assert "EXCLUDE_FLAGS" in kept["overrode"]
    overrides = [frame for frame in receipt["qualityControl"]["screening"]["frames"] if frame.get("reason") == "USER_KEEP_OVERRIDE"]
    assert len(overrides) == 1 and overrides[0]["path"].endswith(moonlit.name)
    assert "BLINK_SKY_BRIGHT" in overrides[0]["flags"]
    assert overrides[0]["admitted"] is True
    # A kept frame the gate did not pass never anchors the registration.
    registration = json.loads((output / "receipts" / "registration.json").read_text(encoding="utf-8"))
    assert not registration["referencePath"].endswith(moonlit.name)


def test_incomplete_selection_fails_closed_unless_undecided_is_decided(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]]
) -> None:
    base = _request(synthetic_project, tmp_path / "incomplete")
    lights = [Path(path) for path in base.light_files]
    partial = {path: "KEEP" for path in lights[:7]}
    with pytest.raises(E2EError) as excinfo:
        run_e2e(_explicit_request(base, parse_explicit_selection(_selection_file(partial))), solver_backends=(FakeSolver(),))
    assert excinfo.value.code == "SELECTION_INCOMPLETE"
    assert not (tmp_path / "incomplete").exists()
    output = tmp_path / "undecided-drop"
    result = run_e2e(
        _explicit_request(
            replace(base, output_directory=str(output)),
            parse_explicit_selection(_selection_file(partial, undecided="DROP")),
        ),
        solver_backends=(FakeSolver(),),
    )
    assert result.success is True
    assert len(result.passed_light_paths) == 7
    block = json.loads((output / "receipt.json").read_text(encoding="utf-8"))["qualityControl"]["selection"]
    assert block["counts"]["undecided"] == 1 and block["undecided"] == "DROP"
    undecided = [item for item in block["frames"] if item["undecided"]]
    assert len(undecided) == 1 and undecided[0]["decision"] == "DROP"


def test_keep_of_an_unregistrable_frame_and_unknown_digests_are_refused(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]]
) -> None:
    blank = _sparse_light(tmp_path / "extra")
    base = _request(synthetic_project, tmp_path / "unregistrable", lights=(*synthetic_project["lights"][:8], blank))
    lights = [Path(path) for path in base.light_files]
    with pytest.raises(E2EError) as excinfo:
        run_e2e(
            _explicit_request(base, parse_explicit_selection(_selection_file({path: "KEEP" for path in lights}))),
            solver_backends=(FakeSolver(),),
        )
    assert excinfo.value.code == "SELECTION_UNREGISTRABLE"
    assert excinfo.value.path is not None and excinfo.value.path.endswith(blank.name)
    # Dropping it is the reviewer's normal answer and runs.
    decisions = {path: "KEEP" for path in lights}
    decisions[blank] = "DROP"
    output = tmp_path / "blank-dropped"
    result = run_e2e(
        _explicit_request(replace(base, output_directory=str(output)), parse_explicit_selection(_selection_file(decisions))),
        solver_backends=(FakeSolver(),),
    )
    assert result.success is True and len(result.passed_light_paths) == 8
    evidence = json.loads((output / "qc" / "blink.json").read_text(encoding="utf-8"))
    blank_frame = next(frame for frame in evidence["frames"] if frame["name"] == blank.name)
    assert "BLINK_UNREGISTRABLE" in {flag["code"] for flag in blank_frame["flags"]}
    # A digest that names no current Light is refused before any work.
    foreign = parse_explicit_selection(
        {**_selection_file({path: "KEEP" for path in lights[:8]}), "decisions": [
            *_selection_file({path: "KEEP" for path in lights[:8]})["decisions"],
            {"sourceSha256": "sha256:" + "f" * 64, "decision": "DROP"},
        ]}
    )
    with pytest.raises(E2EError) as excinfo:
        run_e2e(
            _explicit_request(replace(base, output_directory=str(tmp_path / "foreign"), light_files=base.light_files[:8]), foreign),
            solver_backends=(FakeSolver(),),
        )
    assert excinfo.value.code == "SELECTION_SOURCE_UNKNOWN"


def test_policy_conflicts_are_refused_before_any_work(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]]
) -> None:
    base = _request(synthetic_project, tmp_path / "conflict")
    lights = [Path(path) for path in base.light_files]
    selection = parse_explicit_selection(_selection_file({path: "KEEP" for path in lights}))
    with pytest.raises(E2EError) as excinfo:
        run_e2e(replace(base, explicit_selection=selection), solver_backends=(FakeSolver(),))
    assert excinfo.value.code == "SELECTION_POLICY_CONFLICT"
    with pytest.raises(E2EError) as excinfo:
        run_e2e(replace(base, selection=SelectionParameters(policy="explicit-v1")), solver_backends=(FakeSolver(),))
    assert excinfo.value.code == "SELECTION_POLICY_CONFLICT"
    with pytest.raises(E2EError) as excinfo:
        run_e2e(
            replace(base, selection=SelectionParameters(policy="unattended-v1"), explicit_selection=selection),
            solver_backends=(FakeSolver(),),
        )
    assert excinfo.value.code == "SELECTION_POLICY_CONFLICT"
    assert not (tmp_path / "conflict").exists()
    # The runtime refuses a recipe policy that disagrees with a supplied selection.
    inventory = inventory_project([path.parent for path in (synthetic_project["lights"][0], synthetic_project["flats"][0], synthetic_project["darks"][0], synthetic_project["biases"][0])])
    recipe = Recipe.from_dict({"selection": {"policy": "unattended-v1"}})
    with pytest.raises(RuntimeConfigurationError) as runtime_error:
        build_e2e_request(inventory, recipe, tmp_path / "runtime", workers=1, explicit_selection=selection)
    assert runtime_error.value.code == "SELECTION_POLICY_CONFLICT"
    with pytest.raises(RuntimeConfigurationError) as runtime_error:
        build_e2e_request(inventory, Recipe.from_dict({"selection": {"policy": "explicit-v1"}}), tmp_path / "runtime", workers=1)
    assert runtime_error.value.code == "SELECTION_POLICY_CONFLICT"
    request = build_e2e_request(inventory, Recipe.from_dict({}), tmp_path / "runtime", workers=1, explicit_selection=selection)
    assert request.selection.policy == "explicit-v1" and request.explicit_selection is selection
    plain = build_e2e_request(inventory, Recipe.from_dict({}), tmp_path / "runtime", workers=1)
    assert plain.selection.policy == "legacy-gate" and plain.explicit_selection is None


def test_project_layer_binds_decisions_per_target_run(tmp_path: Path) -> None:
    inventory, base, lights = _project(tmp_path, filters=("R", "G"))
    decisions = {path: ("DROP" if path.name == "light-2.fits" else "KEEP") for path in lights}
    selection = parse_explicit_selection(_selection_file(decisions))
    base = _explicit_request(base, selection)
    observed: list[E2ERequest] = []
    runner = FakePanelRunner()

    def capture(request: E2ERequest, **kwargs: Any) -> E2EResult:
        observed.append(request)
        return runner(request, **kwargs)

    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=capture,
        mosaic_provider=FakeReproject().provider(),
    )
    assert result.success is True
    assert len(observed) == 4
    for request in observed:
        assert request.selection.policy == "explicit-v1"
        assert request.explicit_selection is not None
        own = {_digest(Path(path)) for path in request.light_files}
        assert {item.source_sha256 for item in request.explicit_selection.decisions} == own
        assert request.explicit_selection.digest == selection.digest
    receipt = json.loads((Path(result.output_directory) / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["execution"]["screening"]["frames"]
    # The fake panel runner writes no selection policy; the real one does
    # (asserted on the run receipt above); the aggregate stays None here.
    assert receipt["execution"]["screening"]["selectionPolicy"] is None

    stale = replace(base, output_directory=str(tmp_path / "product-stale"))
    foreign = replace(
        selection,
        decisions=(*selection.decisions, ExplicitDecision(source_sha256="sha256:" + "f" * 64, decision="DROP")),
    )
    with pytest.raises(ProjectE2EError) as excinfo:
        run_project_e2e(
            ProjectE2ERequest(inventory, replace(stale, explicit_selection=foreign), stale.output_directory),
            solver_backends=(ManagedCopySolver(),),
            panel_runner=FakePanelRunner(),
            mosaic_provider=FakeReproject().provider(),
        )
    assert excinfo.value.code == "SELECTION_SOURCE_UNKNOWN"
    with pytest.raises(ProjectE2EError) as excinfo:
        run_project_e2e(
            ProjectE2ERequest(
                inventory, stale, stale.output_directory,
                review_selections=({"sourceSha256": _digest(lights[0]), "gatePolicyDigest": base.gate_policy.canonical_digest()},),
            ),
            solver_backends=(ManagedCopySolver(),),
            panel_runner=FakePanelRunner(),
            mosaic_provider=FakeReproject().provider(),
        )
    assert excinfo.value.code == "SELECTION_POLICY_CONFLICT"


def test_cli_selection_flag_and_request_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from openastroflow_engine import cli

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(_selection_file({}) | {"decisions": [{"sourceSha256": "sha256:" + "1" * 64, "decision": "KEEP"}]}),
        encoding="utf-8",
    )
    loaded = cli._load_selection_file(str(selection_path))
    assert loaded.decisions[0].decision == "KEEP"
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(E2EError) as excinfo:
        cli._load_selection_file(str(broken))
    assert excinfo.value.code == "SELECTION_INVALID"
    light = tmp_path / "input" / "light.fits"
    light.parent.mkdir()
    fits.writeto(light, np.zeros((4, 4), dtype=np.uint16), _header("Light", exposure=60.0, observed_at="2026-01-01T20:00:00Z"))
    request_path = tmp_path / "request.json"
    body = {
        "schemaVersion": 1,
        "sources": [{"hostPath": str(light.parent), "expectedRole": "LIGHT"}],
        "outputDirectory": str(tmp_path / "out"),
        "selection": {"schemaVersion": 1, "kind": "ultra-fast-wbpp-selection", "policy": "explicit-v1", "decisions": [{"sourceSha256": "sha256:" + "1" * 64, "decision": "KEEP"}]},
    }
    request_path.write_text(json.dumps(body), encoding="utf-8")
    _paths, _name, _output, _recipe, options = cli._load_project_request(str(request_path))
    assert options["selection"].decisions[0].source_sha256 == "sha256:" + "1" * 64
    request_path.write_text(
        json.dumps({**body, "reviewSelections": [{"sourceSha256": "sha256:" + "2" * 64, "gatePolicyDigest": "sha256:" + "3" * 64}]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigurationError) as runtime_error:
        cli._load_project_request(str(request_path))
    assert runtime_error.value.code == "SELECTION_POLICY_CONFLICT"
    request_path.write_text(json.dumps({**body, "selection": {"kind": "x"}}), encoding="utf-8")
    with pytest.raises(E2EError) as excinfo:
        cli._load_project_request(str(request_path))
    assert excinfo.value.code == "SELECTION_INVALID"


def test_cli_run_with_selection_file_on_a_synthetic_project(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ultra-fast-wbpp run … --selection`` reaches the pipeline: the fake
    solver stands in for the installed one, everything else is real."""

    import openastroflow_engine.runtime as runtime_module
    from openastroflow_engine import cli
    from test_cli_worker import _registry as ready_registry

    # The planner probes the solver before the run starts; a host without
    # solve-field (CI) must see a ready descriptor, and the run itself gets
    # the fake solver.
    monkeypatch.setattr(runtime_module, "default_registry", ready_registry)
    monkeypatch.setattr(runtime_module, "select_solver_chain", lambda registry, requested: (FakeSolver(),))
    lights = list(synthetic_project["lights"][:8])
    decisions = {path: "KEEP" for path in lights}
    decisions[lights[5]] = "DROP"
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(_selection_file(decisions)), encoding="utf-8")
    output = tmp_path / "cli-run"
    inputs = [str(path) for path in (*lights, *synthetic_project["flats"], *synthetic_project["darks"], *synthetic_project["biases"])]
    code = cli.main(["run", *inputs, "--output", str(output), "--selection", str(selection_path), "--workers", "2", "--fov", "3.0", "--ra", "150", "--dec", "20"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    result = json.loads(captured.out)
    assert result["success"] is True
    assert len(result["passedLightPaths"]) == 7 and len(result["excludedLightPaths"]) == 1
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["qualityControl"]["selectionPolicy"] == "explicit-v1"
    assert receipt["qualityControl"]["selection"]["counts"]["drop"] == 1
    assert (output / "qc" / "blink.json").is_file()
    # A selection with a recipe that asks for the unattended policy is refused.
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps({"selection": {"policy": "unattended-v1"}}), encoding="utf-8")
    code = cli.main(["run", *inputs, "--output", str(tmp_path / "cli-conflict"), "--selection", str(selection_path), "--recipe", str(recipe_path), "--workers", "2"])
    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.err)["error"]["code"] == "SELECTION_POLICY_CONFLICT"
