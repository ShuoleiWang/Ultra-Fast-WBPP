"""Making receipts shareable: local paths become source references or basenames and stat identities are dropped."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

from ..platform import rename_with_retry
from .contracts import E2EError
from .sources import _SourceIdentity


_LOCAL_STAT_KEYS = {
    "device",
    "inode",
    "mtimens",
    "mtime_ns",
    "sourcedevice",
    "sourceinode",
}


def _source_redactor(source_tokens: Mapping[str, str]) -> Callable[[str], str]:
    """The replacement of every source path by its public token, longest path first.

    A receipt tree holds tens of thousands of strings and a project hundreds
    of sources, so the paths are ordered once, and a string that names none of
    the sources' folders (it then names no source) is returned untouched.
    """

    ordered = sorted(source_tokens.items(), key=lambda item: -len(item[0]))
    folders = tuple(dict.fromkeys(os.path.dirname(private) for private, _ in ordered))

    def redact(value: str) -> str:
        if not any(folder in value for folder in folders):
            return value
        for private, public in ordered:
            value = value.replace(private, public)
        return value

    return redact


def _share_safe_string(
    value: str,
    *,
    staging: PurePath,
    source_tokens: Mapping[str, str],
) -> str:
    """Redact host paths while retaining an auditable opaque source token."""

    return _redacted_string(value, staging, _source_redactor(source_tokens))


def _redacted_string(value: str, staging: PurePath, redact: Callable[[str], str]) -> str:
    result = redact(value)
    staging_text = str(staging)
    # Artifact identities use forward slashes on every host. Replacing only
    # the Windows staging prefix leaves backslashes in the suffix and breaks
    # the exact path/hash/size binding when generated masters are handed off.
    for prefix in dict.fromkeys((staging_text, staging.as_posix())):
        if result == prefix:
            return "artifact/."
        separator = "\\" if isinstance(staging, PureWindowsPath) and "\\" in prefix else "/"
        if result.startswith(prefix + separator):
            relative = result[len(prefix) + 1 :]
            if isinstance(staging, PureWindowsPath):
                relative = PureWindowsPath(relative).as_posix()
            return "artifact/" + relative
    result = result.replace(staging_text + os.sep, "artifact/")
    # Any remaining absolute path is an execution-environment detail (solver
    # executable, temporary catalog path, etc.).  Preserve only the basename;
    # the backend/version/catalog identities remain elsewhere in the receipt.
    if os.path.isabs(result):
        return "local-redacted/" + Path(result).name
    return result


def _share_safe_value(
    value: Any,
    *,
    staging: Path,
    source_tokens: Mapping[str, str],
) -> Any:
    return _redacted_value(value, staging, _source_redactor(source_tokens))


def _redacted_value(value: Any, staging: Path, redact: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return _redacted_string(value, staging, redact)
    if isinstance(value, list):
        return [_redacted_value(item, staging, redact) for item in value]
    if isinstance(value, dict):
        return {
            _redacted_string(key, staging, redact): _redacted_value(item, staging, redact)
            for key, item in value.items()
            if key.casefold() not in _LOCAL_STAT_KEYS
        }
    return value


def _sanitize_shareable_tree(
    staging: Path, identities: Sequence[_SourceIdentity]
) -> None:
    """Rewrite every staged JSON document to a share-safe public form."""

    redact = _source_redactor(
        {item.path: f"source/{item.source_id}/{Path(item.path).name}" for item in identities}
    )
    for path in sorted(staging.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise E2EError(
                "PUBLIC_RECEIPT_SANITIZE_FAILED", str(error), path=str(path)
            ) from error
        sanitized = _redacted_value(value, staging, redact)
        encoded = (
            json.dumps(
                sanitized,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        temporary = path.with_name(path.name + ".privacy.tmp")
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        rename_with_retry(temporary, path, replace=True)


def _share_safe_receipt_core(
    value: Mapping[str, Any],
    *,
    staging: Path,
    identities: Sequence[_SourceIdentity],
) -> dict[str, Any]:
    """Sanitize a receipt assembled after the staged JSON rewrite.

    The top-level receipt is created after ``_sanitize_shareable_tree``. Late
    execution evidence therefore needs the same boundary explicitly; without
    it, a native-library or diagnostic path can be reintroduced after every
    existing JSON document was already made share-safe.
    """

    source_tokens = {
        item.path: f"source/{item.source_id}/{Path(item.path).name}"
        for item in identities
    }
    sanitized = _share_safe_value(
        dict(value), staging=staging, source_tokens=source_tokens
    )
    if not isinstance(sanitized, dict):  # pragma: no cover - defensive boundary
        raise E2EError(
            "PUBLIC_RECEIPT_SANITIZE_FAILED",
            "receipt sanitizer did not return an object",
        )
    return sanitized
