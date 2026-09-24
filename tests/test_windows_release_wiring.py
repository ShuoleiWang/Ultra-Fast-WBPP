"""Contracts for the Windows CI/release wiring.

These tests read the workflow files and the installer gate script so that the
Windows-specific steps (patched SEP wheel, static-CRT native build, installed
MSI attestation, prerelease notes) cannot silently disappear from the
pipelines that produce end-user artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPOSITORY / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = REPOSITORY / ".github" / "workflows" / "release.yml"
MSI_GATE = REPOSITORY / "scripts" / "windows" / "attest-installed-msi.ps1"


def _steps(workflow: Path, job: str) -> list[dict]:
    yaml = pytest.importorskip("yaml")
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    return document["jobs"][job]["steps"]


def _index(steps: list[dict], needle: str) -> int:
    for index, step in enumerate(steps):
        if needle in str(step.get("run", "")):
            return index
    raise AssertionError(f"no step runs {needle!r}")


def test_ci_windows_python_job_tests_against_the_patched_sep_and_static_crt_kernels() -> None:
    steps = _steps(CI_WORKFLOW, "python")

    sep_step = steps[_index(steps, "scripts/build_sep_wheel.py --install")]
    assert sep_step["if"] == "runner.os == 'Windows'"
    native = _index(steps, "scripts/build_native_runtime.py")
    assert "runner.os == 'Windows' && '--require-static-crt'" in steps[native]["run"]
    tests = _index(steps, "python -m pytest")
    # The patched wheel and the static-CRT gate both run before the suite so
    # the Windows tests exercise the bytes the sidecar ships.
    assert _index(steps, "scripts/build_sep_wheel.py --install") < native < tests
    assert _index(steps, "pip check") > _index(steps, "scripts/build_sep_wheel.py --install")

    # CI may reuse a cached wheel (the script re-runs every gate and skips only
    # the compile); the cache is keyed on the script, the patch and the
    # interpreter, restored before the build and written from main only.
    assert "--reuse-wheel" in sep_step["run"]
    restore = next(i for i, step in enumerate(steps) if str(step.get("uses", "")).startswith("actions/cache/restore@"))
    save = next(i for i, step in enumerate(steps) if str(step.get("uses", "")).startswith("actions/cache/save@"))
    assert restore < _index(steps, "scripts/build_sep_wheel.py --install") < save < tests
    key = steps[restore]["with"]["key"]
    assert "hashFiles('scripts/build_sep_wheel.py', 'packaging/patches/**')" in key
    assert "steps.setup-python.outputs.python-version" in key
    assert "github.ref == 'refs/heads/main'" in steps[save]["if"]

    rust = _steps(CI_WORKFLOW, "rust")
    assert "cargo test --workspace --locked" in rust[_index(rust, "cargo test --workspace")]["run"]


def test_release_windows_job_attests_the_installed_msi_before_collecting_installers() -> None:
    steps = _steps(RELEASE_WORKFLOW, "bundle")

    sep = _index(steps, "scripts/build_sep_wheel.py --install")
    assert steps[sep]["if"] == "runner.os == 'Windows'"
    # The shipped wheel is always compiled from the verified source.
    assert "--reuse-wheel" not in steps[sep]["run"]
    native = _index(steps, "scripts/build_native_runtime.py")
    assert "runner.os == 'Windows' && '--require-static-crt'" in steps[native]["run"]
    freeze = _index(steps, "scripts/build_engine_sidecar.py")
    build = _index(steps, "npm run tauri:build")
    gate = _index(steps, "scripts/windows/attest-installed-msi.ps1")
    collect = _index(steps, "scripts/collect_release_artifacts.py")
    # sep must be installed before the sidecar is frozen; the MSI gate runs on
    # the built bundle and before the installers are collected and attested.
    assert sep < freeze < build < gate < collect
    assert steps[gate]["if"] == "runner.os == 'Windows'"
    assert steps[gate]["shell"] == "pwsh"
    assert "target/release/bundle/msi" in steps[gate]["run"]
    assert "-MaxStartSeconds 20" in steps[gate]["run"]
    assert ".bundled.manifest.json" in steps[gate]["run"]


def test_release_prerelease_publishes_both_platforms_with_smartscreen_wording() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    steps = _steps(RELEASE_WORKFLOW, "publish-prerelease")

    download = next(step for step in steps if str(step.get("uses", "")).startswith("actions/download-artifact"))
    assert download["with"]["pattern"] == "ultra-fast-wbpp-*"
    notes = steps[_index(steps, "release-notes.md")]["run"]
    for phrase in (
        "SmartScreen",
        '"More info", then "Run anyway"',
        "User Account Control",
        "unknown publisher",
        "Authenticode signing is a release gate",
        "%LOCALAPPDATA%",
        "SHA256SUMS",
    ):
        assert phrase in notes
    release_command = text.split('gh release create "${GITHUB_REF_NAME}"', 1)[1]
    assert "--notes-file release-notes.md" in release_command
    assert "Windows x64" in release_command
    assert "--draft" in release_command and "--prerelease" in release_command


def test_msi_gate_script_installs_silently_attests_and_uninstalls() -> None:
    script = MSI_GATE.read_text(encoding="utf-8")

    assert '"/i", "`"$msiPath`"", "/qn", "/norestart"' in script
    assert '"/x", "`"$msiPath`"", "/qn", "/norestart"' in script
    assert "CurrentVersion\\Uninstall" in script
    assert "InstallLocation" in script
    assert "resources\\ufwbpp-engine" in script
    assert "scripts\\attest_bundled_runtime.py" in script
    assert "--resource-root" in script and "--target" in script and "--output" in script
    assert '"--main-executable", $mainExe' in script
    assert "uninstall left files behind" in script
    # Failures are collected so the uninstall always runs and is never masked.
    assert "$failures" in script
    # "$matches" is PowerShell's automatic variable; only the comment names it.
    assert "$matches = " not in script and "$matches +=" not in script
