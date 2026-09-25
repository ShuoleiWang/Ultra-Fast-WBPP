"""Contracts for the GitHub workflows: the CI gate, path-based job selection,
timeouts, action pins and the release workflow's early checks.

``main`` requires only the ``CI gate`` status, so a job that the gate does not
wait for, or a skip it accepts without reason, would merge untested changes.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
WORKFLOWS = REPOSITORY / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
SELECTED_JOBS = {"python", "rust", "frontend"}


def _load(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(document: dict) -> dict:
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1).
    return document.get("on", document.get(True))


@pytest.mark.parametrize("name", sorted(path.name for path in WORKFLOWS.glob("*.yml")))
def test_every_job_has_a_timeout_and_every_action_is_pinned_by_commit(name: str) -> None:
    for job_id, job in _load(name)["jobs"].items():
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, int) and 0 < timeout <= 120, (name, job_id)
        for step in job.get("steps", []):
            if "uses" in step:
                assert PINNED_ACTION.match(step["uses"]), (name, job_id, step["uses"])


def test_ci_gate_needs_every_job_and_accepts_a_skip_only_for_an_unselected_area() -> None:
    jobs = _load("ci.yml")["jobs"]
    gate = jobs["ci-gate"]
    assert gate["name"] == "CI gate"  # the status check main requires
    assert gate["if"] == "always()"
    assert set(gate["needs"]) == set(jobs) - {"ci-gate"}
    assert set(jobs) - {"changes", "source-check", "ci-gate"} == SELECTED_JOBS

    outputs = jobs["changes"]["outputs"]
    step = gate["steps"][0]
    for job_id in SELECTED_JOBS:
        assert jobs[job_id]["needs"] == "changes"
        assert jobs[job_id]["if"] == f"needs.changes.outputs.{job_id} == 'true'"
        assert job_id in outputs
        assert step["env"][f"{job_id.upper()}_RESULT"] == f"${{{{ needs.{job_id}.result }}}}"
        assert step["env"][f"{job_id.upper()}_SELECTED"] == f"${{{{ needs.changes.outputs.{job_id} }}}}"
    # The classifier and the source checks always run: a skip of either fails.
    assert "if" not in jobs["source-check"] and "needs" not in jobs["source-check"]
    assert '"${result}" == skipped && "${selected}" == false' in step["run"]
    assert 'require "Changed areas" "${CHANGES_RESULT}" true' in step["run"]
    assert 'require "Public tree and links" "${SOURCE_CHECK_RESULT}" true' in step["run"]


def test_ci_selects_jobs_in_a_job_not_with_workflow_path_filters() -> None:
    document = _load("ci.yml")
    triggers = _triggers(document)
    for event in ("push", "pull_request"):
        assert not {"paths", "paths-ignore"} & set(triggers.get(event) or {})
    changes = document["jobs"]["changes"]["steps"]
    assert changes[0]["with"]["fetch-depth"] == 0
    assert "scripts/ci_changed_areas.py" in changes[1]["run"]
    source = document["jobs"]["source-check"]["steps"][1]["run"]
    assert "scripts/check_public_tree.py" in source and "scripts/check_local_links.py" in source


def test_ci_cancels_superseded_runs_only_for_pull_requests() -> None:
    concurrency = _load("ci.yml")["concurrency"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
    # Pushes to main are grouped by commit, so none waits for or replaces another.
    assert "github.sha" in concurrency["group"]


def test_ci_matrices() -> None:
    jobs = _load("ci.yml")["jobs"]
    # The interpreter the bundles freeze on the shipped platforms, the oldest
    # supported one on Linux.
    assert jobs["python"]["strategy"]["matrix"] == {
        "include": [
            {"os": "macos-14", "python": "3.12"},
            {"os": "windows-latest", "python": "3.12"},
            {"os": "ubuntu-latest", "python": "3.11"},
        ]
    }
    rust = jobs["rust"]
    assert rust["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-14", "windows-latest"]
    fmt = next(step for step in rust["steps"] if "cargo fmt" in str(step.get("run", "")))
    assert fmt["if"] == "runner.os == 'Linux'"


def test_release_checks_versions_first_and_installs_node_modules_before_the_tests() -> None:
    jobs = _load("release.yml")["jobs"]
    assert "scripts/check_release_versions.py" in jobs["versions"]["steps"][1]["run"]
    assert jobs["bundle"]["needs"] == "versions"

    steps = jobs["bundle"]["steps"]
    npm_ci = next(i for i, step in enumerate(steps) if str(step.get("run", "")).strip() == "npm ci")
    tests = next(i for i, step in enumerate(steps) if "python -m pytest" in str(step.get("run", "")))
    assert npm_ci < tests  # the Tauri config schema test needs node_modules
    assert steps[npm_ci]["working-directory"] == "apps/desktop"

    upload = next(step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@"))
    assert upload["with"]["name"] == "${{ steps.artifact.outputs.name }}"
    naming = next(step for step in steps if step.get("id") == "artifact")
    assert "${REF_NAME//[^A-Za-z0-9._-]/-}" in naming["run"]
    assert "github.ref_name" not in naming["run"]  # passed through env, never interpolated
