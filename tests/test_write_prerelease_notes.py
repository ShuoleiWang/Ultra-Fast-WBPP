from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.write_prerelease_notes import PrereleaseNotesError, main, present_platforms, render


MACOS = "aarch64-apple-darwin"
WINDOWS = "x86_64-pc-windows-msvc"
RUN_URL = "https://github.com/owner/repo/actions/runs/1"


def _platform(assets: Path, target: str, bundle: str) -> None:
    assets.mkdir(exist_ok=True)
    names = [bundle, f"ufwbpp-engine-{target}.manifest.json", f"release-metadata-{target}.json"]
    for name in names:
        (assets / name).write_bytes(f"{target}:{name}".encode())
    (assets / f"SHA256SUMS-{target}").write_text(
        "".join(
            f"{hashlib.sha256((assets / name).read_bytes()).hexdigest()}  {name}\n"
            for name in sorted(names)
        ),
        encoding="ascii",
    )


def test_both_platforms_give_the_full_draft(tmp_path: Path) -> None:
    assets = tmp_path / "release-assets"
    _platform(assets, MACOS, "Ultra-Fast-WBPP_0.1.0_aarch64.dmg")
    _platform(assets, WINDOWS, "Ultra-Fast-WBPP_0.1.0_x64-setup.exe")

    notes, title = render(*present_platforms(assets), RUN_URL)
    assert title == "macOS arm64 + Windows x64 unsigned prerelease"
    assert "macOS 15 or later" in notes and "SmartScreen" in notes
    assert "Missing" not in notes


def test_a_failed_platform_is_named_and_the_other_is_still_drafted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "release-assets"
    _platform(assets, WINDOWS, "Ultra-Fast-WBPP_0.1.0_x64_en-US.msi")
    notes_path, title_path = tmp_path / "release-notes.md", tmp_path / "release-title.txt"

    assert main(
        ["--assets", str(assets), "--notes", str(notes_path), "--title", str(title_path), "--run-url", RUN_URL]
    ) == 0
    notes = notes_path.read_text(encoding="utf-8")
    assert title_path.read_text(encoding="utf-8") == "Windows x64 only unsigned prerelease\n"
    assert "**Missing: macOS arm64.**" in notes and RUN_URL in notes
    assert "Gatekeeper" not in notes and "SmartScreen" in notes
    assert json.loads(capsys.readouterr().out)["missing"] == [MACOS]


def test_no_platform_or_unverifiable_assets_draft_nothing(tmp_path: Path) -> None:
    assets = tmp_path / "release-assets"
    with pytest.raises(PrereleaseNotesError, match="no platform built and attested"):
        present_platforms(assets)  # download-artifact found nothing
    assets.mkdir()
    with pytest.raises(PrereleaseNotesError, match="no platform built and attested"):
        present_platforms(assets)

    _platform(assets, MACOS, "Ultra-Fast-WBPP_0.1.0_aarch64.dmg")
    (assets / "stray.dmg").write_bytes(b"x")
    with pytest.raises(PrereleaseNotesError, match="no platform's SHA256SUMS covers: stray.dmg"):
        present_platforms(assets)
    (assets / "stray.dmg").unlink()
    (assets / "Ultra-Fast-WBPP_0.1.0_aarch64.dmg").write_bytes(b"tampered")
    with pytest.raises(PrereleaseNotesError, match="does not match SHA256SUMS-aarch64-apple-darwin"):
        present_platforms(assets)
    (assets / "Ultra-Fast-WBPP_0.1.0_aarch64.dmg").unlink()
    with pytest.raises(PrereleaseNotesError, match="which was not downloaded"):
        present_platforms(assets)
