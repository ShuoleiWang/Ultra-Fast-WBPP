#!/usr/bin/env python3
"""Fail when the desktop's version fields disagree, or disagree with a release tag.

The release tooling reads the desktop version from three files:
``apps/desktop/package.json`` (installer and asset names),
``apps/desktop/src-tauri/tauri.conf.json`` (the application bundle) and the
desktop crate's ``Cargo.toml`` (resolving ``version.workspace = true``).  They
must be identical.  A release tag must be ``v<version>`` or
``v<version>-<label>``: the label lets a pre-release tag such as
``v0.1.0-alpha.2`` ship ``0.1.0`` fields, because the Windows MSI accepts only
numeric version components.

    python3 scripts/check_release_versions.py [--tag v0.1.0-alpha.2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_JSON = Path("apps/desktop/package.json")
TAURI_CONFIG = Path("apps/desktop/src-tauri/tauri.conf.json")
DESKTOP_CARGO = Path("apps/desktop/src-tauri/Cargo.toml")
WORKSPACE_CARGO = Path("Cargo.toml")
TAG_LABEL = re.compile(r"[0-9A-Za-z][0-9A-Za-z.-]*")


class VersionFieldError(RuntimeError):
    """A version field is missing or unreadable."""


def _json_version(root: Path, relative: Path) -> str:
    try:
        version = json.loads((root / relative).read_text(encoding="utf-8"))["version"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise VersionFieldError(f"{relative.as_posix()}: no readable version ({error})") from error
    if not isinstance(version, str) or not version:
        raise VersionFieldError(f"{relative.as_posix()}: version is not a non-empty string")
    return version


def _cargo_version(root: Path) -> str:
    try:
        package: dict[str, Any] = tomllib.loads((root / DESKTOP_CARGO).read_text(encoding="utf-8"))["package"]
        version = package["version"]
        if isinstance(version, dict) and version.get("workspace") is True:
            workspace = tomllib.loads((root / WORKSPACE_CARGO).read_text(encoding="utf-8"))
            version = workspace["workspace"]["package"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise VersionFieldError(f"{DESKTOP_CARGO.as_posix()}: no readable version ({error})") from error
    if not isinstance(version, str) or not version:
        raise VersionFieldError(f"{DESKTOP_CARGO.as_posix()}: version is not a non-empty string")
    return version


def desktop_versions(root: Path = REPO_ROOT) -> dict[str, str]:
    """The version each of the three files declares, keyed by its path."""

    return {
        PACKAGE_JSON.as_posix(): _json_version(root, PACKAGE_JSON),
        TAURI_CONFIG.as_posix(): _json_version(root, TAURI_CONFIG),
        DESKTOP_CARGO.as_posix(): _cargo_version(root),
    }


def tag_problem(tag: str, version: str) -> str | None:
    """Why ``tag`` cannot release ``version``, or ``None`` when it can."""

    if not tag.startswith("v"):
        return f"tag {tag!r} does not start with 'v'"
    rest = tag[1:]
    if rest == version:
        return None
    if rest.startswith(version + "-") and TAG_LABEL.fullmatch(rest[len(version) + 1 :]):
        return None
    return f"tag {tag!r} does not release version {version!r} (expected v{version} or v{version}-<label>)"


def check(root: Path = REPO_ROOT, *, tag: str | None = None) -> list[str]:
    """Every disagreement; an empty list means the fields and the tag agree."""

    try:
        versions = desktop_versions(root)
    except VersionFieldError as error:
        return [str(error)]
    problems: list[str] = []
    distinct = sorted(set(versions.values()))
    if len(distinct) != 1:
        problems.append(
            "desktop version fields disagree: "
            + ", ".join(f"{path}={value}" for path, value in versions.items())
        )
    elif tag is not None:
        problem = tag_problem(tag, distinct[0])
        if problem is not None:
            problems.append(problem)
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--tag", default=None, help="release tag to check against, e.g. v0.1.0-alpha.2")
    arguments = parser.parse_args(argv)
    problems = check(arguments.root, tag=arguments.tag)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    versions = desktop_versions(arguments.root)
    summary = {"versions": versions, "tag": arguments.tag}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
