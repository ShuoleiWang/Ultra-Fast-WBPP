from __future__ import annotations

from pathlib import Path

from scripts.check_local_links import check


def test_accepts_existing_anchor_and_web_links(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "target.md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[local](docs/target.md#target) [anchor](#here) [web](https://example.com)\n",
        encoding="utf-8",
    )
    assert check(tmp_path) == []


def test_reports_missing_and_repository_escape(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "[missing](none.md)\n[escape](../../outside.md)\n", encoding="utf-8"
    )
    failures = check(tmp_path)
    assert {item["code"] for item in failures} == {
        "LINK_TARGET_MISSING",
        "LINK_ESCAPES_REPOSITORY",
    }


def test_readme_images_are_checked_including_spaces(tmp_path: Path) -> None:
    (tmp_path / "GUI screenshot.png").write_bytes(b"fixture")
    (tmp_path / "README.md").write_text(
        "![GUI](<GUI screenshot.png>)\n![missing](assets/missing.gif)\n",
        encoding="utf-8",
    )
    failures = check(tmp_path)
    assert len(failures) == 1
    assert failures[0]["target"] == "assets/missing.gif"
    assert failures[0]["code"] == "LINK_TARGET_MISSING"
