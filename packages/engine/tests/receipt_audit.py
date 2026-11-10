"""Audit of a published result: every digest a JSON document cites for a file
of the result is the SHA-256 of that file as published.

A receipt names a file and its digest in one JSON object: ``path`` (or
``relativePath``, ``output``) beside ``sha256``, or ``<name>`` / ``<name>Path``
beside ``<name>Sha256`` (``receipt`` and ``receiptSha256``, ``outputPath`` and
``outputSha256``).  Paths are resolved the way the engine writes them:
``artifact/<relative>`` below a published tree that encloses the document
(the result, or the target run the document belongs to), anything else below
the document's own directory.  A path that names no file of the result (a
source, a deleted work file) is not a citation of the result.

A ``receipt.json`` that carries a ``receiptId`` is identified by the SHA-256
of the canonical JSON of the rest of the document.

One citation is a history, not a statement about the published file: a solver
attempt whose artifact a later attempt of the same filter rewrote in place
(the catalog refit, the adoption of a same-grid solve).  The last attempt
that names the file states its published bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ufwbpp.integrity import canonical_json_document


def _sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def _resolve(value: str, document: Path, root: Path) -> Path | None:
    """The file of the result a string of ``document`` names, if any."""

    if not value or len(value) > 1024 or "\n" in value or "\\" in value:
        return None
    if value.startswith("artifact/"):
        relative = value[len("artifact/") :]
        bases = [document.parent, *document.parent.parents]
        bases = bases[: bases.index(root) + 1] if root in bases else [root]
    else:
        relative = value
        bases = [document.parent]
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        return None
    for base in bases:
        candidate = base / relative
        try:
            if candidate.is_file():
                return candidate
        except (OSError, ValueError):  # a sentence, not a path
            return None
    return None


def _digest_keys(path_key: str, item: dict[str, Any]) -> list[str]:
    names = [f"{path_key}Sha256", f"{path_key.removesuffix('Path')}Sha256"]
    if path_key in ("path", "relativePath", "output"):
        names.append("sha256")
    return [name for name in dict.fromkeys(names) if isinstance(item.get(name), str)]


def _superseded_attempt_artifacts(attempts: list[Any]) -> set[int]:
    """Artifacts of attempts whose file a later attempt of the list names again."""

    superseded: set[int] = set()
    later: set[str] = set()
    for attempt in reversed(attempts):
        artifact = attempt.get("artifact") if isinstance(attempt, dict) else None
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
            if artifact["path"] in later:
                superseded.add(id(artifact))
            later.add(artifact["path"])
    return superseded


def audit_published_digests(root: Path) -> tuple[list[str], list[str]]:
    """``(citations checked, problems)`` of the published tree ``root``.

    Each checked citation reads ``<document>: <json path> -> <file>``.
    """

    root = Path(root)
    cache: dict[Path, str] = {}
    checked: list[str] = []
    problems: list[str] = []

    def walk(value: Any, trail: str, document: Path, skipped: set[int]) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{trail}[{index}]", document, skipped)
            return
        if not isinstance(value, dict):
            return
        if isinstance(value.get("attempts"), list):
            skipped = skipped | _superseded_attempt_artifacts(value["attempts"])
        if id(value) not in skipped:
            for key, item in value.items():
                target = _resolve(item, document, root) if isinstance(item, str) else None
                if target is None:
                    continue
                where = f"{document.relative_to(root).as_posix()}: {trail}.{key} -> {target.relative_to(root).as_posix()}"
                for name in _digest_keys(key, value):
                    checked.append(f"{where} ({name})")
                    if value[name].removeprefix("sha256:") != _sha256(target, cache):
                        problems.append(f"{where}: {name} is not the SHA-256 of the published file")
                if key in ("path", "relativePath") and isinstance(value.get("sizeBytes"), int):
                    if value["sizeBytes"] != target.stat().st_size:
                        problems.append(f"{where}: sizeBytes is not the size of the published file")
        for key, item in value.items():
            walk(item, f"{trail}.{key}", document, skipped)

    for document in sorted(root.rglob("*.json")):
        if document.name.startswith("._"):
            continue
        value = json.loads(document.read_text(encoding="utf-8"))
        if document.name == "receipt.json" and isinstance(value, dict) and "receiptId" in value:
            core = {key: item for key, item in value.items() if key != "receiptId"}
            name = document.relative_to(root).as_posix()
            checked.append(f"{name}: receiptId")
            if value["receiptId"] != "sha256:" + hashlib.sha256(canonical_json_document(core)).hexdigest():
                problems.append(f"{name}: receiptId is not the SHA-256 of the document's canonical JSON")
        walk(value, "$", document, set())
    return checked, problems
