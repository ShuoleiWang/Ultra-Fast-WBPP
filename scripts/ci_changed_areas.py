#!/usr/bin/env python3
"""Select the CI jobs a pull request needs from the paths it changes.

The first job of ``.github/workflows/ci.yml`` runs

    python3 scripts/ci_changed_areas.py --event pull_request \\
        --base <base sha> --head <head sha> --github-output "$GITHUB_OUTPUT"

and the Python, Rust and frontend jobs run only when their area is selected.
The public-tree and local-link checks do not depend on this: they run on every
pull request.  The table is conservative:

* anything but a pull request (a push to ``main``, a manual run) selects every
  area, and so does a pull request whose diff cannot be computed or is empty;
* the CI workflow, this script, the Git metadata that decides how files are
  checked out or found (``.gitattributes``, ``.gitignore``) and every path the
  table does not recognise select every area;
* the other workflows select the Python area, whose contract tests read them
  (no CI job runs them: a release change is proved by a manual run of the
  release workflow);
* documentation selects nothing: ``*.md`` outside ``packages/`` (whose
  READMEs are package metadata), ``docs/`` (with the dated evidence
  receipts in ``docs/evidence/``), ``assets/`` artwork, ``CITATION.cff`` and
  the issue and pull request templates under ``.github/``;
* the legal texts the desktop bundle ships (``LICENSE``, ``NOTICE``,
  ``LICENSES/``, ``THIRD_PARTY_NOTICES.md``, ``docs/licensing.md``) select the
  Python and Rust areas, whatever their suffix.

Renames are listed as a deletion plus an addition (``--no-renames``), so a file
moved out of a code area still selects that area.

Before a push, ``make pr-check`` asks the same question of the local checkout

    python3 scripts/ci_changed_areas.py --worktree --base origin/main

(its commits, uncommitted edits and new files against the merge base) and runs
the suites of the selected areas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence


AREAS = ("python", "rust", "frontend")
EVERYTHING = frozenset(AREAS)
NOTHING: frozenset[str] = frozenset()
PYTHON = frozenset({"python"})
RUST = frozenset({"rust"})
FRONTEND = frozenset({"frontend"})

SELF = "scripts/ci_changed_areas.py"
CI_WORKFLOW = ".github/workflows/ci.yml"
SELECT_EVERYTHING_FILES = frozenset({".gitattributes", ".gitignore"})
WORKFLOW_PREFIX = ".github/workflows/"

# Copied into the bundle's legal/ folder by tauri.conf.json and checked by the
# bundle attestation tests.
BUNDLED_LEGAL_FILES = frozenset({"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "docs/licensing.md"})
BUNDLED_LEGAL_PREFIXES = ("LICENSES/",)

DOCUMENTATION_FILES = frozenset({"CITATION.cff", ".github/CODEOWNERS"})
DOCUMENTATION_PREFIXES = ("docs/", "assets/", ".github/ISSUE_TEMPLATE/")

# Read by the Python contract tests (bundle contract, release versions) as
# well as by their own toolchain.
RUST_AND_PYTHON_FILES = frozenset(
    {
        "Cargo.toml",
        "apps/desktop/src-tauri/Cargo.toml",
        "apps/desktop/src-tauri/tauri.conf.json",
        "apps/desktop/src-tauri/tauri.prerelease.conf.json",
        "apps/desktop/src-tauri/tauri.demo.conf.json",
    }
)
DESKTOP_PACKAGE = "apps/desktop/package.json"

RUST_FILES = frozenset({"Cargo.lock"})
RUST_PREFIXES = ("apps/desktop/src-tauri/", ".cargo/")
FRONTEND_PREFIXES = ("apps/desktop/",)
# native/ is built and ctest-checked by the Python jobs; the root test
# configuration and the lint configuration belong to them too.
PYTHON_FILES = frozenset({"Makefile", "pytest.ini", "conftest.py", "ruff.toml", "requirements-dev.txt"})
PYTHON_PREFIXES = (
    "packages/",
    "native/",
    "tests/",
    "scripts/",
    "packaging/",
    "tools/",
    "benchmarks/",
)


def areas_for(path: str) -> frozenset[str]:
    """The CI areas one repository-relative POSIX path selects."""

    if path in {SELF, CI_WORKFLOW} or path in SELECT_EVERYTHING_FILES:
        return EVERYTHING
    if path.startswith(WORKFLOW_PREFIX):
        return PYTHON
    if path in BUNDLED_LEGAL_FILES or path.startswith(BUNDLED_LEGAL_PREFIXES):
        return PYTHON | RUST
    if (
        (path.endswith(".md") and not path.startswith("packages/"))
        or path in DOCUMENTATION_FILES
        or path.startswith(DOCUMENTATION_PREFIXES)
    ):
        return NOTHING
    if path in RUST_AND_PYTHON_FILES:
        return RUST | PYTHON
    if path in RUST_FILES or path.startswith(RUST_PREFIXES):
        return RUST
    if path == DESKTOP_PACKAGE:
        return FRONTEND | PYTHON
    if path.startswith(FRONTEND_PREFIXES):
        return FRONTEND
    if path in PYTHON_FILES or path.startswith(PYTHON_PREFIXES):
        return PYTHON
    return EVERYTHING


def select_areas(event: str, paths: Iterable[str] | None) -> frozenset[str]:
    """Areas to run for an event; ``paths`` is ``None`` when the diff is unknown."""

    if event != "pull_request" or paths is None:
        return EVERYTHING
    selected: set[str] = set()
    seen = False
    for path in paths:
        seen = True
        selected |= areas_for(path)
        if selected == EVERYTHING:
            break
    return EVERYTHING if not seen else frozenset(selected)


def diff_command(base: str, head: str) -> list[str]:
    return ["git", "diff", "--name-only", "--no-renames", "-z", f"{base}...{head}"]


def _git_paths(command: list[str], *, repository: Path) -> list[str] | None:
    """The NUL- or newline-separated output of a Git command, or ``None`` when it fails."""

    try:
        completed = subprocess.run(command, cwd=str(repository), capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    separator = b"\0" if "-z" in command else b"\n"
    return [raw.decode("utf-8", errors="replace") for raw in completed.stdout.split(separator) if raw]


def changed_paths(base: str, head: str, *, repository: Path) -> list[str] | None:
    """Paths the pull request changes, or ``None`` when Git cannot tell."""

    if not base or not head:
        return None
    return _git_paths(diff_command(base, head), repository=repository)


def worktree_paths(base: str, *, repository: Path) -> list[str] | None:
    """Paths the checkout changes against its merge base with ``base``.

    Commits, uncommitted edits and untracked files that are not ignored: what
    a pull request from this checkout would change once everything is committed.
    """

    merge_base = _git_paths(["git", "merge-base", base, "HEAD"], repository=repository)
    if not merge_base:
        return None
    tracked = _git_paths(["git", "diff", "--name-only", "--no-renames", "-z", merge_base[0]], repository=repository)
    untracked = _git_paths(["git", "ls-files", "--others", "--exclude-standard", "-z"], repository=repository)
    if tracked is None or untracked is None:
        return None
    return sorted({*tracked, *untracked})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event", default="pull_request", help="github.event_name")
    parser.add_argument("--worktree", action="store_true", help="classify the local checkout against --base")
    parser.add_argument("--base", default="", help="pull request base commit")
    parser.add_argument("--head", default="", help="pull request head commit")
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--github-output", type=Path, default=None, help="append name=value lines here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths: list[str] | None = None
    if arguments.worktree:
        arguments.event = "pull_request"
        paths = worktree_paths(arguments.base or "origin/main", repository=arguments.repository)
        if paths is None:
            print("cannot list the checkout's changes; selecting every CI job")
    elif arguments.event == "pull_request":
        paths = changed_paths(arguments.base, arguments.head, repository=arguments.repository)
        if paths is None:
            print("::warning::cannot list the pull request's changes; selecting every CI job")
    # A pull request with an empty diff runs everything; a checkout without changes has nothing to test.
    selected = NOTHING if arguments.worktree and paths == [] else select_areas(arguments.event, paths)
    if paths is None:
        print(f"event {arguments.event}: every CI job runs")
    else:
        print(f"event {arguments.event}: {len(paths)} changed path(s)")
        for path in paths[:200]:
            areas = ", ".join(sorted(areas_for(path))) or "documentation only"
            print(f"  {path}: {areas}")
        if len(paths) > 200:
            print(f"  ... {len(paths) - 200} more")
    lines = [f"{area}={'true' if area in selected else 'false'}" for area in AREAS]
    print("\n".join(lines))
    if arguments.github_output is not None:
        with arguments.github_output.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
