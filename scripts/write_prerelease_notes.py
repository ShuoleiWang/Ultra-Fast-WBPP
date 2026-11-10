#!/usr/bin/env python3
"""Write the draft prerelease's notes and title for the platforms that passed.

The release workflow's ``bundle`` matrix uploads a platform's artifacts only
after its bundle was built and attested (the upload is the job's last step), so
the downloaded ``SHA256SUMS-<target>`` files say which platforms passed. Each
one is checked against the downloaded files; a file that no checksum list
covers, or a checksum that does not match, fails the step. A platform without
artifacts is named in the notes as missing, and at least one must be present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Sequence


class PrereleaseNotesError(RuntimeError):
    """The downloaded assets cannot back a draft prerelease."""


PLATFORMS = (
    {
        "target": "aarch64-apple-darwin",
        "title": "macOS arm64",
        "notes": (
            "**macOS 15 or later (Apple Silicon, DMG):** ad-hoc signed, not notarized. Gatekeeper will\n"
            "refuse the first launch; open System Settings, Privacy & Security and press\n"
            '"Open Anyway" (the right-click, "Open" bypass no longer works on macOS 15 and later).\n'
            "The bundle passed its deployment audit: every Mach-O slice targets macOS 15.0 or\n"
            "older and loads only system libraries or files inside the app."
        ),
    },
    {
        "target": "x86_64-pc-windows-msvc",
        "title": "Windows x64",
        "notes": (
            "**Windows 10/11 x64 (MSI and NSIS installers):** not Authenticode-signed.\n"
            "- The NSIS installer (`*-setup.exe`, per-user, installs under `%LOCALAPPDATA%`)\n"
            '  triggers Microsoft Defender SmartScreen: choose "More info", then "Run anyway".\n'
            "- The MSI installs per machine (Program Files) and shows a User Account Control\n"
            "  prompt from an unknown publisher; accept it only for a build whose SHA-256 you\n"
            "  have verified. The MSI embeds the WebView2 bootstrapper and installs the\n"
            "  runtime silently when it is missing.\n"
            "- Authenticode signing is a release gate: stable releases will carry a signature\n"
            "  and these warnings will not appear.\n"
            "\n"
            "The installers passed the installed-runtime attestation in this workflow run: the\n"
            "frozen engine's DLL/PYD import closure resolves against Windows system DLLs or\n"
            "shipped files, the native kernel DLL links the C runtime statically, and the\n"
            "installed runtime passed the launch budget before the artifacts were collected."
        ),
    },
)
HEADER = (
    "Unsigned validation build for maintainer review; not a stable release. Verify the\n"
    "included `SHA256SUMS-<target>` files before installing anything."
)
_SUMS_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9._+-]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_files(assets: Path, target: str) -> set[str]:
    sums = assets / f"SHA256SUMS-{target}"
    listed: set[str] = set()
    for number, line in enumerate(sums.read_text(encoding="ascii").splitlines(), 1):
        match = _SUMS_LINE.fullmatch(line)
        if match is None:
            raise PrereleaseNotesError(f"{sums.name}:{number} is not a checksum line")
        digest, name = match.groups()
        path = assets / name
        if path.is_symlink() or not path.is_file():
            raise PrereleaseNotesError(f"{sums.name} lists {name}, which was not downloaded")
        if _sha256(path) != digest:
            raise PrereleaseNotesError(f"{name} does not match {sums.name}")
        listed.add(name)
    if f"release-metadata-{target}.json" not in listed:
        raise PrereleaseNotesError(f"{sums.name} does not cover the release metadata")
    return listed


def present_platforms(assets: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The platforms whose checksummed artifacts were downloaded, and the others."""

    files = (
        {path.name for path in assets.iterdir()} if assets.is_dir() and not assets.is_symlink() else set()
    )
    present: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    covered: set[str] = set()
    for platform in PLATFORMS:
        sums = f"SHA256SUMS-{platform['target']}"
        if sums in files:
            covered |= _verified_files(assets, platform["target"]) | {sums}
            present.append(platform)
        else:
            missing.append(platform)
    stray = sorted(files - covered)
    if stray:
        raise PrereleaseNotesError(
            "downloaded files that no platform's SHA256SUMS covers: " + ", ".join(stray)
        )
    if not present:
        raise PrereleaseNotesError(
            "no platform built and attested its bundle: nothing to put in a draft release"
        )
    return present, missing


def render(present: Sequence[dict[str, str]], missing: Sequence[dict[str, str]], run_url: str) -> tuple[str, str]:
    """The release notes (Markdown) and the parenthesised title text."""

    sections = [HEADER, *(platform["notes"] for platform in present)]
    if missing:
        names = " and ".join(platform["title"] for platform in missing)
        sections.append(
            f"**Missing: {names}.** This draft has no {names} build: its bundle job in\n"
            f"[the workflow run]({run_url}) did not upload attested artifacts (it failed,\n"
            "timed out or was cancelled before its attestation passed; the job log says\n"
            "which step). Rerun the failed job before publishing a release that needs it."
        )
    title = " + ".join(platform["title"] for platform in present)
    if missing:
        title += " only"
    return "\n\n".join(sections) + "\n", f"{title} unsigned prerelease"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--title", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    arguments = parser.parse_args(argv)
    try:
        present, missing = present_platforms(arguments.assets)
        notes, title = render(present, missing, arguments.run_url)
        arguments.notes.write_text(notes, encoding="utf-8", newline="\n")
        arguments.title.write_text(title + "\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError, PrereleaseNotesError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "platforms": [platform["target"] for platform in present],
                "missing": [platform["target"] for platform in missing],
                "title": title,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
