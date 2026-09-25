from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_release_versions as versions


def _tree(root: Path, *, package: str, tauri: str, cargo: str, workspace: str | None = None) -> Path:
    desktop = root / "apps" / "desktop"
    (desktop / "src-tauri").mkdir(parents=True)
    (desktop / "package.json").write_text(json.dumps({"name": "desktop", "version": package}), encoding="utf-8")
    (desktop / "src-tauri" / "tauri.conf.json").write_text(json.dumps({"version": tauri}), encoding="utf-8")
    (desktop / "src-tauri" / "Cargo.toml").write_text(f'[package]\nname = "desktop"\n{cargo}\n', encoding="utf-8")
    if workspace is not None:
        (root / "Cargo.toml").write_text(
            f'[workspace]\nmembers = []\n\n[workspace.package]\nversion = "{workspace}"\n', encoding="utf-8"
        )
    return root


def test_the_repository_fields_agree() -> None:
    # Every pull request runs this: the three fields are bumped together.
    assert versions.check() == []
    assert len(set(versions.desktop_versions().values())) == 1


def test_disagreeing_fields_are_named(tmp_path: Path) -> None:
    root = _tree(tmp_path, package="0.2.0", tauri="0.1.0", cargo='version = "0.1.0"')

    problems = versions.check(root)

    assert len(problems) == 1
    assert "apps/desktop/package.json=0.2.0" in problems[0]
    assert "apps/desktop/src-tauri/tauri.conf.json=0.1.0" in problems[0]


def test_workspace_inherited_cargo_version_is_resolved(tmp_path: Path) -> None:
    root = _tree(tmp_path, package="0.3.0", tauri="0.3.0", cargo="version.workspace = true", workspace="0.3.0")

    assert versions.check(root, tag="v0.3.0") == []
    assert versions.desktop_versions(root)["apps/desktop/src-tauri/Cargo.toml"] == "0.3.0"


@pytest.mark.parametrize(
    ("tag", "accepted"),
    [
        ("v0.1.0", True),
        ("v0.1.0-alpha.1", True),
        ("v0.1.0-rc1", True),
        ("0.1.0", False),
        ("v0.1.1", False),
        ("v0.1.0.1", False),
        ("v0.1.0-", False),
        ("v0.1.00", False),
    ],
)
def test_tag_must_release_the_field_version(tmp_path: Path, tag: str, accepted: bool) -> None:
    root = _tree(tmp_path, package="0.1.0", tauri="0.1.0", cargo='version = "0.1.0"')

    assert (versions.check(root, tag=tag) == []) is accepted


def test_main_fails_closed_on_a_missing_field(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _tree(tmp_path, package="0.1.0", tauri="0.1.0", cargo='description = "no version"')

    assert versions.main(["--root", str(root)]) == 1
    assert "Cargo.toml" in capsys.readouterr().err
