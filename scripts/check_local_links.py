#!/usr/bin/env python3
"""Fail when a checked-in Markdown link points to a missing local file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Iterable
from urllib.parse import unquote


# Image targets use the same local-path contract as ordinary links. Keeping
# them in this check catches missing README screenshots before publication.
LINK = re.compile(r"\[[^\]]*\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
IGNORED_PARTS = {".git", ".venv", "build", "dist", "node_modules", "target"}


def _markdown_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.md")):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not any(part in IGNORED_PARTS for part in relative.parts):
            yield path


def check(root: Path) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for document in _markdown_files(root):
        text = document.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in LINK.finditer(line):
                raw = match.group("target").strip("<>")
                if raw.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_text = unquote(raw.split("#", 1)[0])
                if not path_text:
                    continue
                portable = PurePosixPath(path_text.replace("\\", "/"))
                if portable.is_absolute() or ".." in portable.parts:
                    # Parent links are valid only when they still resolve below
                    # the repository root; resolve and check that boundary.
                    candidate = (document.parent / path_text).resolve(strict=False)
                else:
                    candidate = (document.parent / portable).resolve(strict=False)
                try:
                    candidate.relative_to(root)
                except ValueError:
                    failures.append(
                        {
                            "document": document.relative_to(root).as_posix(),
                            "line": line_number,
                            "target": raw,
                            "code": "LINK_ESCAPES_REPOSITORY",
                        }
                    )
                    continue
                if not candidate.exists():
                    failures.append(
                        {
                            "document": document.relative_to(root).as_posix(),
                            "line": line_number,
                            "target": raw,
                            "code": "LINK_TARGET_MISSING",
                        }
                    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    failures = check(root)
    print(
        json.dumps(
            {"ok": not failures, "failureCount": len(failures), "failures": failures},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["check", "main"]
