#!/usr/bin/env python3
"""Fail closed when the public source tree contains local/private artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys


IGNORED_PARTS = {
    ".git",
    ".ultra-fast-wbpp",
    ".openastroflow",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
}
LOCAL_IGNORED_PREFIXES = (
    # Frozen worker trees are generated here immediately before a Tauri bundle.
    # Git-indexed files below these paths are still audited because
    # tracked-file inspection bypasses local exclusions.
    "apps/desktop/src-tauri/binaries/",
    "apps/desktop/src-tauri/resources/ufwbpp-engine/",
    "packages/engine/src/ufwbpp/native/",
)
RAW_SUFFIXES = {".fit", ".fits", ".fts", ".fz", ".xisf", ".xdrz", ".xnml"}
BINARY_RUNTIME_SUFFIXES = {".dll", ".dylib", ".exe", ".so"}
TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cff",
    ".cmake",
    ".cpp",
    ".css",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".metal",
    ".mjs",
    ".mm",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yml",
    ".yaml",
}
FORBIDDEN_PATTERNS = (
    (
        "MAC_HOME_PATH",
        re.compile(r"/" + r"Users/(?!example(?:/|\b))[^/\s\"']+"),
    ),
    ("WINDOWS_HOME_PATH", re.compile(r"[A-Za-z]:\\Users\\(?!example(?:\\|\b))")),
    ("LINUX_HOME_PATH", re.compile(r"/" + r"home/(?!example(?:/|\b))[^/\s\"']+")),
    (
        "PIXINSIGHT_INSTALL_PATH",
        re.compile(r"/Applications/" + r"PixInsight(?:/|\b)"),
    ),
    ("PCL_SOURCE_INCLUDE", re.compile(r"#\s*include\s*[<\"]pcl/")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    line: int | None
    message: str

    def serializable(self) -> dict[str, object]:
        return {
            "code": self.code,
            "path": self.path,
            "line": self.line,
            "message": self.message,
        }


def _ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    portable = relative.as_posix()
    return portable.startswith(LOCAL_IGNORED_PREFIXES) or any(
        part in IGNORED_PARTS or part.endswith(".egg-info")
        for part in relative.parts
    )


def _tracked_paths(root: Path) -> tuple[Path, ...] | None:
    """Return publishable working-tree files, or ``None`` outside Git.

    Release audits inspect what Git will publish, including a file force-added
    below a normally ignored local-state directory. Include new, nonignored
    files too: a pre-commit audit must not silently omit newly written source.
    Ignored, untracked build outputs remain outside this source audit.
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    paths: list[Path] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            relative = Path(raw.decode("utf-8"))
        except UnicodeDecodeError:
            continue
        candidate = root / relative
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        paths.append(candidate)
    return tuple(sorted(paths))


def inspect(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    tracked = _tracked_paths(root)
    candidates = tracked if tracked is not None else tuple(sorted(root.rglob("*")))
    for path in candidates:
        if tracked is None and _ignored(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(Finding("SYMLINK", relative, None, "symlinks are not published"))
            continue
        if not path.is_file():
            continue
        if path.suffix.casefold() in RAW_SUFFIXES and "tests/fixtures/" not in relative:
            findings.append(
                Finding("RAW_DATA", relative, None, "astronomical data must not be published")
            )
        if path.suffix.casefold() in BINARY_RUNTIME_SUFFIXES:
            findings.append(
                Finding(
                    "BINARY_RUNTIME",
                    relative,
                    None,
                    "runtime binaries must be built from checked-in source",
                )
            )
        if path.stat().st_size > 10 * 1024 * 1024:
            findings.append(
                Finding("LARGE_FILE", relative, None, "file exceeds the 10 MiB source limit")
            )
        if path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(Finding("TEXT_ENCODING", relative, None, "text is not UTF-8"))
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for code, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        Finding(code, relative, number, "forbidden public-tree content")
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    arguments = parser.parse_args()
    root = Path(arguments.root).resolve(strict=True)
    findings = inspect(root)
    payload = {
        "schemaVersion": 1,
        "root": str(root),
        "ok": not findings,
        "findingCount": len(findings),
        "findings": [finding.serializable() for finding in findings],
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
