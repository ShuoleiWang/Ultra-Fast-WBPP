from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import ci_changed_areas as areas


EVERYTHING = {"python", "rust", "frontend"}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Documentation selects nothing; the source checks still run.
        ("README.md", set()),
        ("docs/recipes/mono-standard.json", set()),
        ("apps/desktop/README.md", set()),
        ("engine/native/README.md", set()),
        ("validation/standard-mono-masters-20260906.json", set()),
        ("assets/branding/ultra-fast-wbpp-mark.svg", set()),
        ("CITATION.cff", set()),
        # Package READMEs are package metadata (pyproject readme, package data).
        ("packages/engine/README.md", {"python"}),
        # Legal texts the desktop bundle ships.
        ("THIRD_PARTY_NOTICES.md", {"python", "rust"}),
        ("docs/licensing.md", {"python", "rust"}),
        ("LICENSE", {"python", "rust"}),
        ("NOTICE", {"python", "rust"}),
        ("LICENSES/GPL-3.0.txt", {"python", "rust"}),
        # Workflows, this script and Git metadata select everything.
        (".github/workflows/ci.yml", EVERYTHING),
        (".github/pull_request_template.md", EVERYTHING),
        ("scripts/ci_changed_areas.py", EVERYTHING),
        (".gitattributes", EVERYTHING),
        # Code areas.
        ("packages/engine/src/ufwbpp/cli.py", {"python"}),
        ("engine/native/src/FusedIntegration.cpp", {"python"}),
        ("engine/native/metal/FusedIntegration.metal", {"python"}),
        ("scripts/build_sep_wheel.py", {"python"}),
        ("scripts/windows/attest-installed-msi.ps1", {"python"}),
        ("packaging/patches/sep-1.4.1-zero-initialised-buffers.patch", {"python"}),
        ("resources/catalogs/astap-external-v1.json", {"python"}),
        ("Makefile", {"python"}),
        ("Cargo.lock", {"rust"}),
        (".cargo/config.toml", {"rust"}),
        ("apps/desktop/src-tauri/src/lib.rs", {"rust"}),
        # Also read by the Python contract tests.
        ("Cargo.toml", {"rust", "python"}),
        ("apps/desktop/src-tauri/Cargo.toml", {"rust", "python"}),
        ("apps/desktop/src-tauri/tauri.conf.json", {"rust", "python"}),
        ("apps/desktop/src/App.tsx", {"frontend"}),
        ("apps/desktop/package-lock.json", {"frontend"}),
        ("apps/desktop/package.json", {"frontend", "python"}),
        # Anything the table does not know selects everything.
        ("crates/new-crate/src/lib.rs", EVERYTHING),
        ("pyproject.toml", EVERYTHING),
    ],
)
def test_areas_for_each_kind_of_path(path: str, expected: set[str]) -> None:
    assert areas.areas_for(path) == frozenset(expected)


def test_non_pull_request_events_and_unknown_or_empty_diffs_select_everything() -> None:
    assert areas.select_areas("push", ["README.md"]) == areas.EVERYTHING
    assert areas.select_areas("workflow_dispatch", ["README.md"]) == areas.EVERYTHING
    assert areas.select_areas("pull_request", None) == areas.EVERYTHING
    assert areas.select_areas("pull_request", []) == areas.EVERYTHING
    assert areas.select_areas("pull_request", ["docs/windows.md", "README.md"]) == areas.NOTHING
    assert areas.select_areas("pull_request", ["docs/x.md", "apps/desktop/src/App.tsx", "Cargo.lock"]) == {
        "frontend",
        "rust",
    }


def test_diff_lists_renames_as_deletion_and_addition() -> None:
    command = areas.diff_command("base", "head")
    assert command[:3] == ["git", "diff", "--name-only"]
    assert "--no-renames" in command and "-z" in command
    assert command[-1] == "base...head"


def _git(repository: Path, *arguments: str) -> str:
    # No user or system configuration (commit signing, hooks, line endings).
    empty_config = repository.parent / "empty-gitconfig"
    empty_config.touch()
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(empty_config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_main_moves_a_python_file_into_docs_and_still_selects_python(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "packages" / "engine").mkdir(parents=True)
    (repository / "packages" / "engine" / "notes.py").write_text("# notes\n", encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")
    (repository / "docs").mkdir()
    _git(repository, "mv", "packages/engine/notes.py", "docs/notes.md")
    _git(repository, "commit", "-q", "-m", "move")
    head = _git(repository, "rev-parse", "HEAD")
    output = tmp_path / "github-output"

    code = areas.main(
        [
            "--event",
            "pull_request",
            "--base",
            base,
            "--head",
            head,
            "--repository",
            str(repository),
            "--github-output",
            str(output),
        ]
    )

    assert code == 0
    assert output.read_text(encoding="utf-8").splitlines() == ["python=true", "rust=false", "frontend=false"]


def test_main_selects_everything_when_the_diff_cannot_be_computed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "github-output"

    code = areas.main(["--event", "pull_request", "--base", "", "--head", "", "--github-output", str(output)])

    assert code == 0
    assert output.read_text(encoding="utf-8").splitlines() == ["python=true", "rust=true", "frontend=true"]
    assert "::warning::" in capsys.readouterr().out
