"""Hash-bound human adjudications for WBPP preparation.

An adjudication never identifies a frame by path alone.  The content digest is
part of the key so that a decision cannot silently follow a replaced file.
This module intentionally contains no filesystem mutation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class AdjudicationError(ValueError):
    """An adjudication document is ambiguous, stale, or malformed."""


class AdjudicationAction(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


def normalize_sha256(value: str) -> str:
    """Return a lowercase, untagged SHA-256 digest or raise."""

    text = str(value).strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AdjudicationError("sha256 must contain exactly 64 hexadecimal digits")
    return text


def canonical_path(value: str | Path) -> str:
    text = str(value)
    if not text or "\x00" in text:
        raise AdjudicationError("adjudication path is empty or contains NUL")
    return str(Path(text).expanduser().resolve(strict=False))


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise AdjudicationError(f"duplicate adjudication JSON key: {key}")
        result[key] = item
    return result


def _reject_nonfinite(value: str) -> None:
    raise AdjudicationError(f"non-finite adjudication value: {value}")


@dataclass(frozen=True, slots=True)
class AdjudicationRecord:
    path: str
    sha256: str
    action: AdjudicationAction
    reason: str = ""
    reviewer: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", canonical_path(self.path))
        object.__setattr__(self, "sha256", normalize_sha256(self.sha256))
        if not isinstance(self.action, AdjudicationAction):
            try:
                object.__setattr__(
                    self, "action", AdjudicationAction(str(self.action).strip().upper())
                )
            except ValueError as error:
                raise AdjudicationError(
                    f"unsupported adjudication action: {self.action!r}"
                ) from error
        if "\x00" in self.reason or "\x00" in self.reviewer:
            raise AdjudicationError("adjudication text contains NUL")

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.sha256

    def serializable(self) -> dict[str, str]:
        value = asdict(self)
        value["action"] = self.action.value
        return value


@dataclass(frozen=True, slots=True)
class AdjudicationSet:
    records: tuple[AdjudicationRecord, ...] = ()

    def by_key(self) -> dict[tuple[str, str], AdjudicationRecord]:
        result: dict[tuple[str, str], AdjudicationRecord] = {}
        path_to_sha: dict[str, str] = {}
        for record in self.records:
            previous_sha = path_to_sha.get(record.path)
            if previous_sha is not None and previous_sha != record.sha256:
                raise AdjudicationError(
                    f"multiple content identities are adjudicated for {record.path}"
                )
            path_to_sha[record.path] = record.sha256
            previous = result.get(record.key)
            if previous is not None and previous != record:
                raise AdjudicationError(
                    f"conflicting adjudications for {record.path} ({record.sha256})"
                )
            result[record.key] = record
        return result

    def serializable(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "records": [record.serializable() for record in self.records],
        }


def _record_from_mapping(value: Mapping[str, Any]) -> AdjudicationRecord:
    allowed = {"path", "sha256", "action", "reason", "reviewer"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AdjudicationError(
            "unknown adjudication field(s): " + ", ".join(str(item) for item in unknown)
        )
    missing = sorted({"path", "sha256", "action"} - set(value))
    if missing:
        raise AdjudicationError(
            "missing adjudication field(s): " + ", ".join(missing)
        )
    return AdjudicationRecord(
        path=str(value["path"]),
        sha256=str(value["sha256"]),
        action=AdjudicationAction(str(value["action"]).strip().upper()),
        reason=str(value.get("reason", "")),
        reviewer=str(value.get("reviewer", "")),
    )


def parse_adjudication(value: Any | None) -> AdjudicationSet:
    """Normalize a typed set, document mapping, or iterable of records.

    The accepted document form is deliberately strict::

        {"schemaVersion": 1, "records": [{...}]}
    """

    if value is None:
        return AdjudicationSet()
    if isinstance(value, AdjudicationSet):
        # Force duplicate/conflict validation even for an already typed value.
        value.by_key()
        return value
    if isinstance(value, (str, Path)):
        document_path = Path(value).expanduser().resolve(strict=True)
        try:
            decoded = json.loads(
                document_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_object,
                parse_constant=_reject_nonfinite,
            )
        except (OSError, json.JSONDecodeError) as error:
            raise AdjudicationError(f"cannot read adjudication document: {error}") from error
        return parse_adjudication(decoded)
    if isinstance(value, Mapping):
        allowed = {"schemaVersion", "records"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise AdjudicationError(
                "unknown adjudication document field(s): "
                + ", ".join(str(item) for item in unknown)
            )
        if value.get("schemaVersion") != 1:
            raise AdjudicationError("unsupported adjudication schemaVersion")
        records_value = value.get("records")
        if not isinstance(records_value, list):
            raise AdjudicationError("adjudication records must be an array")
        records = tuple(
            item if isinstance(item, AdjudicationRecord) else _record_from_mapping(item)
            for item in records_value
            if isinstance(item, (AdjudicationRecord, Mapping))
        )
        if len(records) != len(records_value):
            raise AdjudicationError("every adjudication record must be an object")
        result = AdjudicationSet(records)
        result.by_key()
        return result
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        records: list[AdjudicationRecord] = []
        for item in value:
            if isinstance(item, AdjudicationRecord):
                records.append(item)
            elif isinstance(item, Mapping):
                records.append(_record_from_mapping(item))
            else:
                raise AdjudicationError("every adjudication record must be an object")
        result = AdjudicationSet(tuple(records))
        result.by_key()
        return result
    raise AdjudicationError("unsupported adjudication value")


__all__ = [
    "AdjudicationAction",
    "AdjudicationError",
    "AdjudicationRecord",
    "AdjudicationSet",
    "canonical_path",
    "normalize_sha256",
    "parse_adjudication",
]
