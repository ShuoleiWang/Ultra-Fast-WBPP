"""Conservative, content-addressed preparation of WBPP input trees.

Planning is side-effect free.  Applying a plan copies into a private sibling
staging directory and publishes the complete tree with one directory rename.
No input file is ever modified, moved, linked, or opened for writing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping
import unicodedata

from .adjudication import (
    AdjudicationAction,
    AdjudicationError,
    canonical_path,
    normalize_sha256,
    parse_adjudication,
)
from .models import (
    Decision,
    FileIdentity,
    FrameMetadata,
    FrameResult,
    FrameRole,
    GateDisposition,
)


PLAN_SCHEMA_VERSION = 1
PLAN_KIND = "light-frame-qc.prepare-wbpp"
LAYOUT_ID = "per-target-light-flat-v1"
MANIFEST_NAME = "prepare-manifest.json"
COPY_BUFFER_BYTES = 4 * 1024 * 1024
_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul", "clock$"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)

# Schema-1 plans predate an explicit description of apply-time source checks.
# They retain their original, stronger (and slower) all-source SHA-256 preflight.
# Newly generated plans bind this policy into planId and can safely avoid reading
# MANIFEST_ONLY/DUPLICATE_LIGHT content that will never enter the output tree.
APPLY_VERIFICATION_POLICY = {
    "copySource": "stream-sha256-with-stat-guard",
    "copyDestination": "sha256-before-publish-stat-bound-after-rename",
    "manifestOnlySource": "stat-only-no-current-content-hash",
    "duplicateSource": "stat-only-no-current-content-hash",
}


@dataclass(frozen=True)
class _PublishedFileSnapshot:
    """Hash-bound identity of a file inside the private staging tree."""

    identity: FileIdentity
    ctime_ns: int


class PrepareError(ValueError):
    """Base class for deterministic plan/application failures."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class PreparePlanError(PrepareError):
    """Input metadata or policy cannot produce an unambiguous plan."""


class PrepareApplyError(PrepareError):
    """A validated plan cannot be safely or atomically published."""


def _compute_file_identity(path: str | os.PathLike[str]) -> FileIdentity:
    # Kept lazy so this module remains importable while optional CLI pieces are
    # assembled, and so tests can replace the single authoritative function.
    from .identity import compute_file_identity

    return compute_file_identity(path)


def _verify_file_identity_stat(
    path: str | os.PathLike[str], identity: FileIdentity
) -> None:
    from .identity import verify_file_identity_stat

    verify_file_identity_stat(path, identity)


def _probe_frame_metadata(path: str | os.PathLike[str]) -> FrameMetadata:
    from .readers import probe_frame_metadata

    return probe_frame_metadata(path)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise PreparePlanError("NON_CANONICAL_PLAN_VALUE", str(error)) from error


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def safe_slug(value: str, *, maximum_stem: int = 48) -> str:
    """Create a stable ASCII path component with a collision-resistant suffix."""

    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PreparePlanError("UNKNOWN_PATH_COMPONENT", "target/filter value is empty")
    normalized = unicodedata.normalize("NFKC", value).strip()
    folded = normalized.casefold()
    if folded in {"unknown", "?", "none", "null"}:
        raise PreparePlanError("UNKNOWN_PATH_COMPONENT", f"unresolved value {value!r}")
    ascii_text = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-.")
    stem = (stem or "value")[:maximum_stem].rstrip("-.") or "value"
    suffix = hashlib.sha256(folded.encode("utf-8")).hexdigest()[:10]
    component = f"{stem}--{suffix}"
    if component in {".", ".."} or "/" in component or "\\" in component:
        raise PreparePlanError("UNSAFE_PATH_COMPONENT", value)
    return component


def _windows_safe_component(value: str) -> bool:
    """Return whether one component is portable to supported Windows filesystems."""

    if not value or value in {".", ".."} or value.endswith((" ", ".")):
        return False
    if any(ord(character) < 32 or character in _WINDOWS_FORBIDDEN_CHARS for character in value):
        return False
    reserved_stem = value.split(".", 1)[0].rstrip(" .").casefold()
    return reserved_stem not in _WINDOWS_RESERVED_STEMS


def _safe_filter_slug(value: str) -> str:
    """Keep conventional filter identifiers readable, hash unsafe names."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or normalized.casefold() in {"unknown", "?", "none", "null"}:
        raise PreparePlanError("UNKNOWN_PATH_COMPONENT", f"unresolved filter {value!r}")
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,31}", normalized)
        and _windows_safe_component(normalized)
    ):
        return normalized
    return safe_slug(normalized, maximum_stem=24)


def _portable_source_filename(value: str) -> str:
    """Preserve normal capture names and deterministically encode unsafe ones."""

    if _windows_safe_component(value):
        return value
    source = Path(value)
    suffix = source.suffix
    stem = value[: -len(suffix)] if suffix else value
    portable_suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,15}", suffix) else ""
    candidate = safe_slug(stem or value, maximum_stem=40) + portable_suffix
    if not _windows_safe_component(candidate):
        raise PreparePlanError("UNSAFE_SOURCE_FILENAME", value)
    return candidate


def _text(value: Any) -> str:
    return str(value).strip()


def _known_text(value: Any) -> str | None:
    if value is None:
        return None
    text = _text(value)
    if not text or text.casefold() in {"unknown", "?", "null"}:
        return None
    return unicodedata.normalize("NFKC", text).casefold()


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _role(metadata: FrameMetadata) -> FrameRole:
    value = getattr(metadata, "role", FrameRole.UNKNOWN)
    if isinstance(value, FrameRole):
        return value
    try:
        return FrameRole(str(value))
    except ValueError:
        return FrameRole.UNKNOWN


def _has_authoritative_role_evidence(metadata: FrameMetadata) -> bool:
    return any(
        str(value).startswith(("FITS:", "XISF:"))
        for value in getattr(metadata, "role_evidence", [])
    )


def _decision(result: FrameResult) -> Decision:
    value = result.decision
    if isinstance(value, Decision):
        return value
    try:
        return Decision(str(value))
    except ValueError as error:
        raise PreparePlanError(
            "UNKNOWN_DECISION", f"{result.path}: {value!r}"
        ) from error


def _validate_serialized_quality_gate(
    serialized: Mapping[str, Any],
    label: str,
    error_type: type[PrepareError],
) -> GateDisposition:
    try:
        disposition = GateDisposition(str(serialized.get("disposition", "")))
    except ValueError as error:
        raise error_type("QUALITY_GATE_INVALID", f"{label}: disposition") from error
    version = str(serialized.get("version", ""))
    policy = serialized.get("policy")
    policy_digest = str(serialized.get("policyDigest", ""))
    if not isinstance(policy, Mapping):
        raise error_type("QUALITY_GATE_POLICY_MISSING", label)
    try:
        from .quality_gate import GatePolicy

        policy_values = dict(policy)
        policy_values.pop("evidence_revision", None)
        typed_policy = GatePolicy(**policy_values)
        typed_policy.validate()
    except (TypeError, ValueError) as error:
        raise error_type(
            "QUALITY_GATE_POLICY_UNSUPPORTED", f"{label}: {error}"
        ) from error
    if typed_policy.serializable() != dict(policy):
        raise error_type("QUALITY_GATE_POLICY_UNSUPPORTED", label)
    actual_digest = "sha256:" + hashlib.sha256(
        _canonical_json(dict(policy)).encode("utf-8")
    ).hexdigest()
    if policy_digest != actual_digest:
        raise error_type(
            "QUALITY_GATE_POLICY_DIGEST_MISMATCH", f"{label}: {policy_digest}"
        )
    expected_version = str(policy.get("version", "")) + "@" + actual_digest
    if version != expected_version or not re.fullmatch(
        r"quality-gate-v1@sha256:[0-9a-f]{64}", version
    ):
        raise error_type(
            "QUALITY_GATE_POLICY_UNSUPPORTED", f"{label}: {version!r}"
        )
    ranks = {"INFO": 0, "WARNING": 1, "REVIEW": 2, "ERROR": 3, "HARD_FAIL": 4}
    severities = []
    for item in serialized.get("evidence", []):
        if not isinstance(item, Mapping) or item.get("severity") not in ranks:
            raise error_type("QUALITY_GATE_INVALID", f"{label}: evidence")
        severities.append(ranks[str(item["severity"])])
    maximum = max(severities, default=0)
    if (
        disposition is GateDisposition.PASS
        and maximum >= ranks["REVIEW"]
    ) or (
        disposition is GateDisposition.REVIEW
        and (maximum < ranks["REVIEW"] or maximum >= ranks["HARD_FAIL"])
    ) or (
        disposition is GateDisposition.HARD_FAIL
        and maximum < ranks["HARD_FAIL"]
    ):
        raise error_type(
            "QUALITY_GATE_DISPOSITION_MISMATCH", label
        )
    return disposition


def _quality_gate(result: FrameResult) -> tuple[GateDisposition, dict[str, Any]]:
    gate = getattr(result, "quality_gate", None)
    if gate is None:
        raise PreparePlanError("QUALITY_GATE_MISSING", result.path)
    try:
        serialized = gate.serializable()
    except (AttributeError, TypeError, ValueError) as error:
        raise PreparePlanError("QUALITY_GATE_INVALID", f"{result.path}: {error}") from error
    disposition = _validate_serialized_quality_gate(
        serialized, result.path, PreparePlanError
    )
    return disposition, serialized


def _identity_dict(identity: FileIdentity) -> dict[str, Any]:
    digest = normalize_sha256(identity.sha256)
    fields = {
        "sha256": digest,
        "sizeBytes": int(identity.size_bytes),
        "mtimeNs": int(identity.mtime_ns),
        "device": int(identity.device),
        "inode": int(identity.inode),
    }
    if fields["sizeBytes"] < 0 or fields["mtimeNs"] < 0:
        raise PreparePlanError("INVALID_SOURCE_IDENTITY", str(fields))
    return fields


def _identity_from_dict(value: Mapping[str, Any]) -> FileIdentity:
    required = {"sha256", "sizeBytes", "mtimeNs", "device", "inode"}
    if set(value) != required:
        raise PrepareApplyError(
            "INVALID_PLAN_IDENTITY", f"identity fields are {sorted(value)}"
        )
    try:
        return FileIdentity(
            sha256=normalize_sha256(str(value["sha256"])),
            size_bytes=int(value["sizeBytes"]),
            mtime_ns=int(value["mtimeNs"]),
            device=int(value["device"]),
            inode=int(value["inode"]),
        )
    except (TypeError, ValueError, AdjudicationError) as error:
        raise PrepareApplyError("INVALID_PLAN_IDENTITY", str(error)) from error


def _same_source_identity(left: FileIdentity, right: FileIdentity) -> bool:
    return (
        normalize_sha256(left.sha256) == normalize_sha256(right.sha256)
        and left.size_bytes == right.size_bytes
        and left.mtime_ns == right.mtime_ns
        and left.device == right.device
        and left.inode == right.inode
    )


def _source_path(value: str | os.PathLike[str], *, code: str) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise PreparePlanError(code, f"symbolic-link inputs are not accepted: {requested}")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PreparePlanError(code, str(error)) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise PreparePlanError(code, f"not a regular file: {resolved}")
    return resolved


def _assert_metadata_consistent(result: FrameResult, live: FrameMetadata) -> None:
    recorded = result.metadata
    exact_fields = (
        "width",
        "height",
        "channels",
        "filter_name",
        "binning_x",
        "binning_y",
        "cfa_pattern",
    )
    mismatches = [
        name
        for name in exact_fields
        if getattr(recorded, name, None) != getattr(live, name, None)
    ]
    optional_fields = ("camera", "gain", "offset", "readout_mode", "target")
    for name in optional_fields:
        old = getattr(recorded, name, None)
        new = getattr(live, name, None)
        if name in {"gain", "offset"}:
            old_number, new_number = _number(old), _number(new)
            if old_number is not None and new_number is not None and old_number != new_number:
                mismatches.append(name)
        else:
            old_text, new_text = _known_text(old), _known_text(new)
            if old_text is not None and new_text is not None and old_text != new_text:
                mismatches.append(name)
    if mismatches:
        raise PreparePlanError(
            "RESULT_METADATA_MISMATCH",
            f"{result.path}: live metadata differs in {', '.join(sorted(set(mismatches)))}",
        )


def _profile(metadata: FrameMetadata) -> dict[str, Any]:
    if metadata.width <= 0 or metadata.height <= 0 or metadata.channels <= 0:
        raise PreparePlanError(
            "UNKNOWN_LIGHT_PROFILE", f"invalid geometry for {metadata.path}"
        )
    if metadata.binning_x <= 0 or metadata.binning_y <= 0:
        raise PreparePlanError(
            "UNKNOWN_LIGHT_PROFILE", f"invalid binning for {metadata.path}"
        )
    if not bool(getattr(metadata, "binning_known", False)):
        raise PreparePlanError(
            "UNKNOWN_BINNING_PROFILE", f"binning metadata is absent for {metadata.path}"
        )
    filter_name = _text(metadata.filter_name)
    cfa = _text(getattr(metadata, "cfa_pattern", "UNKNOWN")).upper()
    # CFA/mono state must be explicit or inferred from an unambiguous
    # monochrome camera model.  UNKNOWN == UNKNOWN is not scientific evidence.
    if not filter_name or filter_name.upper() == "UNKNOWN":
        raise PreparePlanError(
            "UNKNOWN_LIGHT_PROFILE",
            f"filter metadata is unresolved for {metadata.path}",
        )
    if cfa == "UNKNOWN":
        raise PreparePlanError(
            "UNKNOWN_CFA_PROFILE",
            f"CFA/mono state is unresolved for {metadata.path}",
        )
    return {
        "width": int(metadata.width),
        "height": int(metadata.height),
        "channels": int(metadata.channels),
        "binningX": int(metadata.binning_x),
        "binningY": int(metadata.binning_y),
        "filter": filter_name,
        "cfaPattern": cfa,
        "camera": None if _known_text(metadata.camera) is None else _text(metadata.camera),
        "gain": _number(metadata.gain),
        "offset": _number(metadata.offset),
        "readoutMode": (
            None
            if _known_text(getattr(metadata, "readout_mode", None)) is None
            else _text(metadata.readout_mode)
        ),
    }


def _profile_id(profile: Mapping[str, Any]) -> str:
    return "profile-" + _digest(profile)[:16]


def _profiles_compatible(light: Mapping[str, Any], flat: Mapping[str, Any]) -> bool:
    for name in (
        "width",
        "height",
        "channels",
        "binningX",
        "binningY",
        "filter",
        "cfaPattern",
    ):
        left, right = light[name], flat[name]
        if isinstance(left, str):
            if unicodedata.normalize("NFKC", left).casefold() != unicodedata.normalize(
                "NFKC", str(right)
            ).casefold():
                return False
        elif left != right:
            return False
    for name in ("camera", "readoutMode"):
        left, right = _known_text(light.get(name)), _known_text(flat.get(name))
        if left is not None and right is not None and left != right:
            return False
    for name in ("gain", "offset"):
        left, right = _number(light.get(name)), _number(flat.get(name))
        if left is not None and right is not None and left != right:
            return False
    return True


def _relative_path(*parts: str) -> str:
    path = PurePosixPath(*parts)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PreparePlanError("UNSAFE_DESTINATION_PATH", str(path))
    if any(not _windows_safe_component(part) for part in path.parts):
        raise PreparePlanError("UNSAFE_DESTINATION_PATH", str(path))
    return path.as_posix()


def _destination_key(relative: str) -> str:
    return "/".join(
        unicodedata.normalize("NFC", part).rstrip(" .").casefold()
        for part in PurePosixPath(relative).parts
    )


def _entry_id(prefix: str, source: str, sha256: str, destination: str | None) -> str:
    return f"{prefix}-" + _digest(
        {"source": source, "sha256": sha256, "destination": destination}
    )[:20]


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(entry.get("destinationRelativePath") or "~"),
        str(entry.get("action", "")),
        str(entry.get("sourcePath", "")),
        str(entry.get("entryId", "")),
    )


def build_prepare_plan(
    frame_results: Iterable[FrameResult],
    master_flat_paths: Iterable[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    adjudication: Any | None = None,
    qc_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, non-mutating WBPP preparation plan.

    Only LIGHT frames are accepted in ``frame_results``.  Without an explicit
    adjudication, only automatic KEEP decisions are copy-eligible.  Master
    flats are supplied separately and raw flats are deliberately unsupported.
    """

    final_destination = str(Path(destination).expanduser().resolve(strict=False))
    if not Path(final_destination).name:
        raise PreparePlanError("UNSAFE_DESTINATION", final_destination)

    adjudications = parse_adjudication(adjudication)
    adjudication_by_key = adjudications.by_key()
    adjudication_paths: dict[str, str] = {
        path: sha for path, sha in adjudication_by_key
    }
    used_adjudications: set[tuple[str, str]] = set()

    prepared_lights: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for result in sorted(frame_results, key=lambda item: canonical_path(item.path)):
        source = _source_path(result.path, code="LIGHT_SOURCE_INVALID")
        source_text = str(source)
        if source_text in seen_paths:
            raise PreparePlanError("DUPLICATE_FRAME_RESULT", source_text)
        seen_paths.add(source_text)
        if canonical_path(result.metadata.path) != source_text:
            raise PreparePlanError(
                "RESULT_PATH_MISMATCH", f"{result.path} != {result.metadata.path}"
            )
        identity = getattr(result, "identity", None)
        if identity is None:
            raise PreparePlanError("QC_SOURCE_IDENTITY_MISSING", source_text)
        try:
            _verify_file_identity_stat(source, identity)
        except (OSError, RuntimeError, ValueError) as error:
            raise PreparePlanError(
                "QC_SOURCE_MISMATCH", f"{source}: {error}"
            ) from error
        live_metadata = _probe_frame_metadata(source)
        try:
            _verify_file_identity_stat(source, identity)
        except (OSError, RuntimeError, ValueError) as error:
            raise PreparePlanError(
                "SOURCE_CHANGED_DURING_PLAN", f"{source}: {error}"
            ) from error
        _assert_metadata_consistent(result, live_metadata)
        if getattr(live_metadata, "role_conflicts", []):
            raise PreparePlanError(
                "METADATA_ROLE_CONFLICT",
                f"{source}: {', '.join(live_metadata.role_conflicts)}",
            )
        if _role(live_metadata) is FrameRole.RAW_FLAT:
            raise PreparePlanError("RAW_FLAT_UNSUPPORTED", source_text)
        if _role(live_metadata) is not FrameRole.LIGHT:
            code = (
                "UNKNOWN_FRAME_ROLE"
                if _role(live_metadata) is FrameRole.UNKNOWN
                else "NON_LIGHT_FRAME_RESULT"
            )
            raise PreparePlanError(code, f"{source}: {_role(live_metadata).value}")
        if not _has_authoritative_role_evidence(live_metadata):
            raise PreparePlanError("NON_AUTHORITATIVE_FRAME_ROLE", source_text)
        if int(getattr(live_metadata, "image_count", 1)) != 1:
            raise PreparePlanError("MULTI_IMAGE_CONTAINER", source_text)

        target = _text(live_metadata.target)
        filter_name = _text(live_metadata.filter_name)
        target_slug = safe_slug(target)
        filter_slug = _safe_filter_slug(filter_name)
        profile = _profile(live_metadata)
        profile_id = _profile_id(profile)
        identity_value = _identity_dict(identity)

        digest = identity_value["sha256"]
        key = (source_text, digest)
        record = adjudication_by_key.get(key)
        automatic = _decision(result)
        gate_disposition, gate_value = _quality_gate(result)
        stale_sha = adjudication_paths.get(source_text)
        if record is None and stale_sha is not None:
            raise PreparePlanError(
                "STALE_ADJUDICATION",
                f"{source_text}: adjudicated {stale_sha}, live {digest}",
            )
        if record is not None:
            if (
                record.action is AdjudicationAction.APPROVE
                and gate_disposition is not GateDisposition.REVIEW
            ):
                raise PreparePlanError(
                    "UNSAFE_ADJUDICATION_APPROVAL",
                    f"{source_text}: cannot approve gate {gate_disposition.value}",
                )
            used_adjudications.add(key)

        approved = gate_disposition is GateDisposition.PASS
        adjudication_value: dict[str, str] | None = None
        if record is not None:
            approved = record.action is AdjudicationAction.APPROVE
            adjudication_value = record.serializable()

        prepared_lights.append(
            {
                "result": result,
                "source": source_text,
                "identity": identity_value,
                "metadata": live_metadata,
                "target": target,
                "targetSlug": target_slug,
                "filter": filter_name,
                "filterSlug": filter_slug,
                "profile": profile,
                "profileId": profile_id,
                "automaticDecision": automatic.value,
                "qualityGate": gate_value,
                "gateDisposition": gate_disposition.value,
                "confidence": (
                    result.confidence.value
                    if hasattr(result.confidence, "value")
                    else str(result.confidence)
                ),
                "groupId": str(result.group_id),
                "reasons": sorted(str(value) for value in result.reasons),
                "warnings": sorted(str(value) for value in result.warnings),
                "adjudication": adjudication_value,
                "approved": approved,
            }
        )

    unused = sorted(set(adjudication_by_key) - used_adjudications)
    if unused:
        raise PreparePlanError(
            "UNKNOWN_ADJUDICATION_TARGET",
            ", ".join(f"{path} ({digest})" for path, digest in unused),
        )

    flat_candidates: list[dict[str, Any]] = []
    seen_flat_paths: set[str] = set()
    for raw_path in sorted(master_flat_paths, key=lambda item: canonical_path(item)):
        source = _source_path(raw_path, code="MASTER_FLAT_SOURCE_INVALID")
        source_text = str(source)
        if source_text in seen_flat_paths:
            continue
        seen_flat_paths.add(source_text)
        identity = _compute_file_identity(source)
        metadata = _probe_frame_metadata(source)
        try:
            _verify_file_identity_stat(source, identity)
        except (OSError, RuntimeError, ValueError) as error:
            raise PreparePlanError(
                "SOURCE_CHANGED_DURING_PLAN", f"{source}: {error}"
            ) from error
        if getattr(metadata, "role_conflicts", []):
            raise PreparePlanError(
                "METADATA_ROLE_CONFLICT",
                f"{source}: {', '.join(metadata.role_conflicts)}",
            )
        role = _role(metadata)
        if role is FrameRole.RAW_FLAT:
            raise PreparePlanError("RAW_FLAT_UNSUPPORTED", source_text)
        if role is not FrameRole.MASTER_FLAT:
            code = "UNKNOWN_FRAME_ROLE" if role is FrameRole.UNKNOWN else "NOT_MASTER_FLAT"
            raise PreparePlanError(code, f"{source}: {role.value}")
        if not _has_authoritative_role_evidence(metadata):
            raise PreparePlanError("NON_AUTHORITATIVE_FRAME_ROLE", source_text)
        # WBPP master-flat XISF files commonly contain the integrated master as
        # image 0 plus rejection-map images.  The container must be copied as
        # a whole, and WBPP correctly consumes its primary MasterFlat image.
        if int(getattr(metadata, "image_count", 1)) < 1:
            raise PreparePlanError("EMPTY_IMAGE_CONTAINER", source_text)
        flat_candidates.append(
            {
                "source": source_text,
                "identity": _identity_dict(identity),
                "metadata": metadata,
                "profile": _profile(metadata),
            }
        )

    # Byte-identical flat aliases cannot constitute an ambiguity.  Their
    # canonical source is the lexical first path, independent of discovery order.
    flats_by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in flat_candidates:
        flats_by_digest[candidate["identity"]["sha256"]].append(candidate)
    unique_flats: list[dict[str, Any]] = []
    for digest in sorted(flats_by_digest):
        aliases = sorted(flats_by_digest[digest], key=lambda item: item["source"])
        canonical = aliases[0]
        if any(item["profile"] != canonical["profile"] for item in aliases[1:]):
            raise PreparePlanError(
                "IDENTICAL_FLAT_METADATA_CONFLICT",
                ", ".join(item["source"] for item in aliases),
            )
        canonical = dict(canonical)
        canonical["aliases"] = [item["source"] for item in aliases[1:]]
        unique_flats.append(canonical)

    eligible = [item for item in prepared_lights if item["approved"]]
    profile_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        profile_groups[(item["targetSlug"], item["profileId"])].append(item)

    selected_flat_for_group: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(profile_groups):
        light_profile = profile_groups[key][0]["profile"]
        candidates = [
            candidate
            for candidate in unique_flats
            if _profiles_compatible(light_profile, candidate["profile"])
        ]
        if not candidates:
            sample = profile_groups[key][0]
            raise PreparePlanError(
                "MISSING_MASTER_FLAT",
                f"target={sample['target']!r}, filter={sample['filter']!r}, "
                f"profile={sample['profileId']}",
            )
        if len(candidates) > 1:
            raise PreparePlanError(
                "AMBIGUOUS_MASTER_FLAT",
                ", ".join(candidate["source"] for candidate in candidates),
            )
        selected_flat_for_group[key] = candidates[0]

    entries: list[dict[str, Any]] = []
    # Same bytes may only collapse when their scientific placement agrees.
    eligible_by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        eligible_by_digest[item["identity"]["sha256"]].append(item)
    for digest in sorted(eligible_by_digest):
        aliases = sorted(eligible_by_digest[digest], key=lambda item: item["source"])
        placement = {
            (item["targetSlug"], item["filterSlug"], item["profileId"])
            for item in aliases
        }
        if len(placement) != 1:
            raise PreparePlanError(
                "DUPLICATE_LIGHT_METADATA_CONFLICT",
                ", ".join(item["source"] for item in aliases),
            )
        canonical = aliases[0]
        filename = _portable_source_filename(Path(canonical["source"]).name)
        destination_relative = _relative_path(
            canonical["targetSlug"], "LIGHT", canonical["filterSlug"], filename
        )
        flat = selected_flat_for_group[(canonical["targetSlug"], canonical["profileId"])]
        entry = {
            "entryId": _entry_id("light", canonical["source"], digest, destination_relative),
            "role": FrameRole.LIGHT.value,
            "action": "COPY_LIGHT",
            "sourcePath": canonical["source"],
            "sourceIdentity": canonical["identity"],
            "destinationRelativePath": destination_relative,
            "destinationSha256": digest,
            "automaticDecision": canonical["automaticDecision"],
            "qualityGate": canonical["qualityGate"],
            "gateDisposition": canonical["gateDisposition"],
            "confidence": canonical["confidence"],
            "groupId": canonical["groupId"],
            "reasons": canonical["reasons"],
            "warnings": canonical["warnings"],
            "adjudication": canonical["adjudication"],
            "effectiveDisposition": "KEEP",
            "target": canonical["target"],
            "targetSlug": canonical["targetSlug"],
            "filter": canonical["filter"],
            "filterSlug": canonical["filterSlug"],
            "profileId": canonical["profileId"],
            "profile": canonical["profile"],
            "matchedFlatSha256": flat["identity"]["sha256"],
        }
        entries.append(entry)
        for duplicate in aliases[1:]:
            entries.append(
                {
                    "entryId": _entry_id("duplicate", duplicate["source"], digest, None),
                    "role": FrameRole.LIGHT.value,
                    "action": "DUPLICATE_LIGHT",
                    "sourcePath": duplicate["source"],
                    "sourceIdentity": duplicate["identity"],
                    "destinationRelativePath": None,
                    "destinationSha256": None,
                    "automaticDecision": duplicate["automaticDecision"],
                    "qualityGate": duplicate["qualityGate"],
                    "gateDisposition": duplicate["gateDisposition"],
                    "confidence": duplicate["confidence"],
                    "groupId": duplicate["groupId"],
                    "reasons": duplicate["reasons"],
                    "warnings": duplicate["warnings"],
                    "adjudication": duplicate["adjudication"],
                    "effectiveDisposition": "KEEP",
                    "target": duplicate["target"],
                    "targetSlug": duplicate["targetSlug"],
                    "filter": duplicate["filter"],
                    "filterSlug": duplicate["filterSlug"],
                    "profileId": duplicate["profileId"],
                    "profile": duplicate["profile"],
                    "matchedFlatSha256": flat["identity"]["sha256"],
                    "duplicateOf": entry["entryId"],
                }
            )

    eligible_sources = {item["source"] for item in eligible}
    for item in prepared_lights:
        if item["source"] in eligible_sources:
            continue
        entries.append(
            {
                "entryId": _entry_id(
                    "excluded", item["source"], item["identity"]["sha256"], None
                ),
                "role": FrameRole.LIGHT.value,
                "action": "MANIFEST_ONLY",
                "sourcePath": item["source"],
                "sourceIdentity": item["identity"],
                "destinationRelativePath": None,
                "destinationSha256": None,
                "automaticDecision": item["automaticDecision"],
                "qualityGate": item["qualityGate"],
                "gateDisposition": item["gateDisposition"],
                "confidence": item["confidence"],
                "groupId": item["groupId"],
                "reasons": item["reasons"],
                "warnings": item["warnings"],
                "adjudication": item["adjudication"],
                "effectiveDisposition": "EXCLUDE",
                "target": item["target"],
                "targetSlug": item["targetSlug"],
                "filter": item["filter"],
                "filterSlug": item["filterSlug"],
                "profileId": item["profileId"],
                "profile": item["profile"],
                "matchedFlatSha256": None,
            }
        )

    # Each target receives its own flat copy, even when several targets share
    # one source master.  Within a target, a reused flat is copied only once.
    flat_copy_keys: set[tuple[str, str]] = set()
    for group_key in sorted(selected_flat_for_group):
        target_slug, profile_id = group_key
        flat = selected_flat_for_group[group_key]
        digest = flat["identity"]["sha256"]
        copy_key = (target_slug, digest)
        if copy_key in flat_copy_keys:
            continue
        flat_copy_keys.add(copy_key)
        group_sample = profile_groups[group_key][0]
        filename = _portable_source_filename(Path(flat["source"]).name)
        destination_relative = _relative_path(
            target_slug, "FLAT", group_sample["filterSlug"], filename
        )
        matched_profiles = sorted(
            key[1]
            for key, selected in selected_flat_for_group.items()
            if key[0] == target_slug and selected["identity"]["sha256"] == digest
        )
        entries.append(
            {
                "entryId": _entry_id("flat", flat["source"], digest, destination_relative),
                "role": FrameRole.MASTER_FLAT.value,
                "action": "COPY_MASTER_FLAT",
                "sourcePath": flat["source"],
                "sourceIdentity": flat["identity"],
                "sourceAliases": flat["aliases"],
                "destinationRelativePath": destination_relative,
                "destinationSha256": digest,
                "automaticDecision": None,
                "qualityGate": None,
                "gateDisposition": None,
                "adjudication": None,
                "effectiveDisposition": "CALIBRATION",
                "target": group_sample["target"],
                "targetSlug": target_slug,
                "filter": group_sample["filter"],
                "filterSlug": group_sample["filterSlug"],
                "profileId": _profile_id(flat["profile"]),
                "profile": flat["profile"],
                "matchedLightProfileIds": matched_profiles,
            }
        )

    # Refuse collisions using macOS/Windows-like Unicode and case folding.
    destinations: dict[str, dict[str, Any]] = {}
    for entry in sorted(entries, key=_entry_sort_key):
        relative = entry.get("destinationRelativePath")
        if relative is None:
            continue
        key = _destination_key(relative)
        previous = destinations.get(key)
        if previous is not None:
            raise PreparePlanError(
                "DESTINATION_COLLISION",
                f"{previous['sourcePath']} and {entry['sourcePath']} -> {relative}",
            )
        destinations[key] = entry

    entries.sort(key=_entry_sort_key)
    action_counts = Counter(entry["action"] for entry in entries)
    decision_counts = Counter(
        entry["automaticDecision"]
        for entry in entries
        if entry["automaticDecision"] is not None
    )
    gate_counts = Counter(
        entry["gateDisposition"]
        for entry in entries
        if entry.get("gateDisposition") is not None
    )
    gate_versions = {
        item["qualityGate"]["version"] for item in prepared_lights
    }
    if len(gate_versions) != 1:
        raise PreparePlanError(
            "MIXED_QUALITY_GATE_POLICIES", ", ".join(sorted(gate_versions))
        )
    payload: dict[str, Any] = {
        "schemaVersion": PLAN_SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "layoutId": LAYOUT_ID,
        "destination": final_destination,
        "policy": {
            "automaticCopyGate": GateDisposition.PASS.value,
            "qualityGateVersion": next(iter(gate_versions)),
            "qualityGatePolicyDigest": prepared_lights[0]["qualityGate"]["policyDigest"],
            "qualityGatePolicy": prepared_lights[0]["qualityGate"]["policy"],
            "adjudicationActions": [
                AdjudicationAction.APPROVE.value,
                AdjudicationAction.REJECT.value,
            ],
            "rawFlatSupport": False,
            "missingMasterFlat": "FAIL",
            "ambiguousMasterFlat": "FAIL",
            "duplicateLightContent": "DEDUPLICATE",
            "destinationCollision": "FAIL",
            "applyVerification": dict(APPLY_VERIFICATION_POLICY),
        },
        "summary": {
            "frameResultCount": len(prepared_lights),
            "masterFlatCandidateCount": len(flat_candidates),
            "uniqueMasterFlatContentCount": len(
                {
                    entry["sourceIdentity"]["sha256"]
                    for entry in entries
                    if entry["action"] == "COPY_MASTER_FLAT"
                }
            ),
            "targetCount": len({item["targetSlug"] for item in prepared_lights}),
            "actionCounts": dict(sorted(action_counts.items())),
            "automaticDecisionCounts": dict(sorted(decision_counts.items())),
            "qualityGateCounts": dict(sorted(gate_counts.items())),
        },
        "entries": entries,
    }
    if qc_report is not None:
        payload["qcReport"] = dict(qc_report)
    payload["planId"] = "sha256:" + _digest(payload)
    return payload


def _validate_plan(
    plan: Mapping[str, Any], *, verify_qc_report: bool = False
) -> None:
    if plan.get("schemaVersion") != PLAN_SCHEMA_VERSION:
        raise PrepareApplyError("UNSUPPORTED_PLAN_SCHEMA", str(plan.get("schemaVersion")))
    if plan.get("kind") != PLAN_KIND or plan.get("layoutId") != LAYOUT_ID:
        raise PrepareApplyError("UNSUPPORTED_PLAN_KIND", str(plan.get("kind")))
    claimed = str(plan.get("planId", ""))
    if not claimed.startswith("sha256:"):
        raise PrepareApplyError("INVALID_PLAN_ID", claimed)
    unsigned = deepcopy(dict(plan))
    unsigned.pop("planId", None)
    actual = "sha256:" + _digest(unsigned)
    if claimed != actual:
        raise PrepareApplyError("PLAN_DIGEST_MISMATCH", f"claimed {claimed}, got {actual}")
    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise PrepareApplyError("INVALID_PLAN", "entries must be an array")
    seen_ids: set[str] = set()
    seen_destinations: set[str] = set()
    source_identities: dict[str, dict[str, Any]] = {}
    gate_versions: set[str] = set()
    light_entries: list[Mapping[str, Any]] = []
    flat_entries: list[Mapping[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PrepareApplyError("INVALID_PLAN", "entry must be an object")
        entry_id = str(entry.get("entryId", ""))
        if not entry_id or entry_id in seen_ids:
            raise PrepareApplyError("INVALID_PLAN", f"duplicate/empty entryId {entry_id!r}")
        seen_ids.add(entry_id)
        raw_identity = dict(entry.get("sourceIdentity", {}))
        source_identity = _identity_from_dict(raw_identity)
        source_path = str(entry.get("sourcePath", ""))
        previous_identity = source_identities.get(source_path)
        if not source_path or (
            previous_identity is not None and previous_identity != raw_identity
        ):
            raise PrepareApplyError(
                "INVALID_PLAN", f"inconsistent source identity for {source_path!r}"
            )
        source_identities[source_path] = raw_identity
        if entry.get("role") == FrameRole.LIGHT.value:
            light_entries.append(entry)
            gate = entry.get("qualityGate")
            if not isinstance(gate, Mapping):
                raise PrepareApplyError("INVALID_PLAN", f"missing quality gate for {entry_id}")
            gate_disposition = _validate_serialized_quality_gate(
                gate, entry_id, PrepareApplyError
            )
            disposition = gate_disposition.value
            if entry.get("gateDisposition") != disposition:
                raise PrepareApplyError("INVALID_PLAN", f"gate mismatch for {entry_id}")
            version = str(gate.get("version", ""))
            gate_versions.add(version)
            adjudication = entry.get("adjudication")
            record = None
            if adjudication is not None:
                if not isinstance(adjudication, Mapping):
                    raise PrepareApplyError("INVALID_PLAN", f"invalid adjudication for {entry_id}")
                try:
                    parsed = parse_adjudication([adjudication])
                except AdjudicationError as error:
                    raise PrepareApplyError(
                        "INVALID_PLAN", f"invalid adjudication for {entry_id}: {error}"
                    ) from error
                record = parsed.records[0]
                if record.path != canonical_path(source_path) or record.sha256 != normalize_sha256(
                    source_identity.sha256
                ):
                    raise PrepareApplyError(
                        "INVALID_PLAN", f"adjudication identity mismatch for {entry_id}"
                    )
                if (
                    record.action is AdjudicationAction.APPROVE
                    and gate_disposition is not GateDisposition.REVIEW
                ):
                    raise PrepareApplyError(
                        "INVALID_PLAN", f"unsafe adjudication approval for {entry_id}"
                    )
            copy_eligible = (
                gate_disposition is GateDisposition.PASS
                and (record is None or record.action is not AdjudicationAction.REJECT)
            ) or (
                gate_disposition is GateDisposition.REVIEW
                and record is not None
                and record.action is AdjudicationAction.APPROVE
            )
            if entry.get("action") in {"COPY_LIGHT", "DUPLICATE_LIGHT"} and not copy_eligible:
                raise PrepareApplyError("INVALID_PLAN", f"gate cannot copy {entry_id}")
            if entry.get("action") == "MANIFEST_ONLY" and copy_eligible:
                raise PrepareApplyError("INVALID_PLAN", f"eligible light omitted for {entry_id}")
        elif entry.get("role") == FrameRole.MASTER_FLAT.value:
            flat_entries.append(entry)
        relative = entry.get("destinationRelativePath")
        copy_action = entry.get("action") in {"COPY_LIGHT", "COPY_MASTER_FLAT"}
        if copy_action != (isinstance(relative, str) and bool(relative)):
            raise PrepareApplyError(
                "INVALID_PLAN", f"copy/destination mismatch for {entry_id}"
            )
        if copy_action:
            if entry.get("destinationSha256") != normalize_sha256(
                source_identity.sha256
            ):
                raise PrepareApplyError(
                    "INVALID_PLAN", f"destination digest mismatch for {entry_id}"
                )
            normalized = _relative_path(*PurePosixPath(relative).parts)
            if normalized != relative:
                raise PrepareApplyError("INVALID_PLAN", f"noncanonical path {relative}")
            key = _destination_key(relative)
            if key in seen_destinations:
                raise PrepareApplyError("DESTINATION_COLLISION", relative)
            seen_destinations.add(key)
    if len(gate_versions) > 1:
        raise PrepareApplyError(
            "INVALID_PLAN", "multiple quality-gate policies are mixed in one plan"
        )
    policy = plan.get("policy")
    if not isinstance(policy, Mapping) or (
        gate_versions and policy.get("qualityGateVersion") != next(iter(gate_versions))
    ):
        raise PrepareApplyError("INVALID_PLAN", "plan quality-gate policy mismatch")
    apply_verification = policy.get("applyVerification")
    if apply_verification is not None and apply_verification != APPLY_VERIFICATION_POLICY:
        raise PrepareApplyError(
            "INVALID_PLAN", "unsupported apply-verification policy"
        )
    if light_entries:
        first_gate = light_entries[0]["qualityGate"]
        if (
            policy.get("qualityGatePolicyDigest") != first_gate.get("policyDigest")
            or policy.get("qualityGatePolicy") != first_gate.get("policy")
        ):
            raise PrepareApplyError(
                "INVALID_PLAN", "embedded quality-gate policy mismatch"
            )

    flat_index = {
        (
            str(entry.get("targetSlug")),
            str(entry.get("filterSlug")),
            normalize_sha256(str(entry.get("sourceIdentity", {}).get("sha256", ""))),
        ): entry
        for entry in flat_entries
        if entry.get("action") == "COPY_MASTER_FLAT"
    }
    for entry in light_entries:
        if entry.get("action") != "COPY_LIGHT":
            continue
        matched_digest = normalize_sha256(str(entry.get("matchedFlatSha256", "")))
        key = (
            str(entry.get("targetSlug")),
            str(entry.get("filterSlug")),
            matched_digest,
        )
        flat = flat_index.get(key)
        if flat is None:
            raise PrepareApplyError(
                "INVALID_PLAN", f"missing matched master flat for {entry.get('entryId')}"
            )
        try:
            compatible = _profiles_compatible(
                dict(entry.get("profile", {})), dict(flat.get("profile", {}))
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PrepareApplyError(
                "INVALID_PLAN", f"invalid matched flat profile: {error}"
            ) from error
        if not compatible:
            raise PrepareApplyError(
                "INVALID_PLAN", f"incompatible matched flat for {entry.get('entryId')}"
            )

    summary = plan.get("summary")
    if not isinstance(summary, Mapping):
        raise PrepareApplyError("INVALID_PLAN", "summary must be an object")
    expected_summary = {
        "frameResultCount": len(light_entries),
        "targetCount": len({str(entry.get("targetSlug")) for entry in light_entries}),
        "actionCounts": dict(sorted(Counter(str(entry.get("action")) for entry in entries).items())),
        "automaticDecisionCounts": dict(
            sorted(
                Counter(
                    str(entry.get("automaticDecision"))
                    for entry in light_entries
                    if entry.get("automaticDecision") is not None
                ).items()
            )
        ),
        "qualityGateCounts": dict(
            sorted(Counter(str(entry.get("gateDisposition")) for entry in light_entries).items())
        ),
        "uniqueMasterFlatContentCount": len(
            {
                normalize_sha256(str(entry.get("sourceIdentity", {}).get("sha256", "")))
                for entry in flat_entries
            }
        ),
    }
    for name, expected in expected_summary.items():
        if summary.get(name) != expected:
            raise PrepareApplyError(
                "INVALID_PLAN", f"summary mismatch for {name}"
            )
    if verify_qc_report:
        _validate_qc_report(plan, light_entries)


def _read_manifest(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate manifest JSON key {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_object,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise PrepareApplyError("INVALID_EXISTING_MANIFEST", str(error)) from error
    if not isinstance(value, dict):
        raise PrepareApplyError("INVALID_EXISTING_MANIFEST", "root is not an object")
    return value


def _validate_qc_report(
    plan: Mapping[str, Any], light_entries: list[Mapping[str, Any]]
) -> None:
    raw = plan.get("qcReport")
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "sizeBytes"}:
        raise PrepareApplyError("QC_REPORT_MISSING", "plan is not bound to results.json")
    path = Path(str(raw["path"])).expanduser().resolve(strict=True)
    identity = _compute_file_identity(path)
    if (
        normalize_sha256(str(raw["sha256"])) != normalize_sha256(identity.sha256)
        or int(raw["sizeBytes"]) != identity.size_bytes
    ):
        raise PrepareApplyError("QC_REPORT_MISMATCH", str(path))
    payload = _read_manifest(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise PrepareApplyError("QC_REPORT_INVALID", "frames must be an array")
    by_path: dict[str, Mapping[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise PrepareApplyError("QC_REPORT_INVALID", "frame must be an object")
        frame_path = canonical_path(str(frame.get("path", "")))
        if frame_path in by_path:
            raise PrepareApplyError("QC_REPORT_INVALID", f"duplicate {frame_path}")
        by_path[frame_path] = frame
    for entry in light_entries:
        source_path = canonical_path(str(entry["sourcePath"]))
        frame = by_path.get(source_path)
        if frame is None:
            raise PrepareApplyError("QC_REPORT_MISMATCH", f"missing {source_path}")
        if (
            frame.get("sourceIdentity") != entry.get("sourceIdentity")
            or frame.get("qualityGate") != entry.get("qualityGate")
            or frame.get("decision") != entry.get("automaticDecision")
            or frame.get("groupId") != entry.get("groupId")
        ):
            raise PrepareApplyError("QC_REPORT_MISMATCH", source_path)


def load_prepare_plan(
    path: str | os.PathLike[str], *, require_qc_report: bool = False
) -> dict[str, Any]:
    """Load and validate a strict, unsigned-on-disk preparation plan."""

    source = Path(path).expanduser().resolve(strict=True)
    value = _read_manifest(source)
    if "status" in value:
        raise PrepareApplyError(
            "INVALID_PLAN", "a completed destination manifest is not an apply plan"
        )
    _validate_plan(value, verify_qc_report=require_qc_report)
    return value


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _hash_bound_snapshot(path: Path) -> _PublishedFileSnapshot:
    """Hash a file once and bind the digest to its full local stat identity."""

    before = path.stat()
    identity = _compute_file_identity(path)
    after = path.stat()
    if _stat_signature(before) != _stat_signature(after) or (
        identity.device,
        identity.inode,
        identity.size_bytes,
        identity.mtime_ns,
    ) != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    ):
        raise PrepareApplyError(
            "DESTINATION_DRIFT", f"file changed while snapshotting {path}"
        )
    return _PublishedFileSnapshot(identity=identity, ctime_ns=int(after.st_ctime_ns))


def _verify_published_tree(
    destination: Path,
    plan: Mapping[str, Any],
    *,
    trusted_snapshot: Mapping[str, _PublishedFileSnapshot] | None = None,
) -> dict[str, _PublishedFileSnapshot]:
    """Verify an exact tree and return hash-bound evidence for its copied files.

    The private staging tree is hashed exactly once.  After its atomic directory
    rename, ``trusted_snapshot`` lets the commit check prove that every pathname
    still resolves to the same inode/stat coordinates, without reading all image
    bytes a second time.  Existing destinations never have such trusted evidence
    and are therefore fully rehashed before returning ``ALREADY_COMPLETE``.
    """

    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PrepareApplyError("DESTINATION_DRIFT", "missing regular manifest")
    manifest = _read_manifest(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise PrepareApplyError("DESTINATION_DRIFT", "manifest is not COMPLETE")
    embedded_plan = dict(manifest)
    embedded_plan.pop("status", None)
    if embedded_plan != dict(plan):
        existing_id = str(embedded_plan.get("planId", ""))
        if existing_id != plan.get("planId"):
            raise PrepareApplyError(
                "DESTINATION_PLAN_CONFLICT",
                f"existing {existing_id}, requested {plan.get('planId')}",
            )
        raise PrepareApplyError("DESTINATION_DRIFT", "manifest differs from requested plan")
    _validate_plan(embedded_plan)

    expected_files = {MANIFEST_NAME}
    expected_directories: set[str] = set()
    source_inodes = {
        (
            _identity_from_dict(entry["sourceIdentity"]).device,
            _identity_from_dict(entry["sourceIdentity"]).inode,
        )
        for entry in plan["entries"]
    }
    snapshots: dict[str, _PublishedFileSnapshot] = {}
    for entry in plan["entries"]:
        relative = entry.get("destinationRelativePath")
        if relative is None:
            continue
        expected_files.add(relative)
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        target = destination.joinpath(*PurePosixPath(relative).parts)
        if target.is_symlink() or not target.is_file():
            raise PrepareApplyError("DESTINATION_DRIFT", f"missing regular file {relative}")
        current_stat = target.stat()
        if current_stat.st_nlink != 1:
            raise PrepareApplyError(
                "DESTINATION_DRIFT", f"hardlink count is not one for {relative}"
            )
        expected = _identity_from_dict(entry["sourceIdentity"])
        if trusted_snapshot is None:
            snapshot = _hash_bound_snapshot(target)
        else:
            snapshot = trusted_snapshot.get(relative)
            if snapshot is None:
                raise PrepareApplyError(
                    "DESTINATION_DRIFT", f"missing trusted snapshot for {relative}"
                )
            if _stat_signature(current_stat) != (
                snapshot.identity.device,
                snapshot.identity.inode,
                snapshot.identity.size_bytes,
                snapshot.identity.mtime_ns,
                snapshot.ctime_ns,
            ):
                raise PrepareApplyError(
                    "DESTINATION_DRIFT", f"post-publish identity changed for {relative}"
                )
        actual = snapshot.identity
        if (actual.device, actual.inode) in source_inodes:
            raise PrepareApplyError(
                "DESTINATION_DRIFT", f"hardlink alias detected for {relative}"
            )
        if (
            normalize_sha256(actual.sha256) != normalize_sha256(expected.sha256)
            or actual.size_bytes != expected.size_bytes
            or actual.mtime_ns != expected.mtime_ns
        ):
            raise PrepareApplyError("DESTINATION_DRIFT", f"identity mismatch for {relative}")
        snapshots[relative] = snapshot

    if trusted_snapshot is not None and set(trusted_snapshot) != set(snapshots):
        raise PrepareApplyError(
            "DESTINATION_DRIFT", "trusted snapshot contains unexpected paths"
        )

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in destination.rglob("*"):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            raise PrepareApplyError("DESTINATION_DRIFT", f"unexpected symlink {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise PrepareApplyError("DESTINATION_DRIFT", f"irregular object {relative}")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise PrepareApplyError(
            "DESTINATION_DRIFT",
            f"tree differs: files={sorted(actual_files ^ expected_files)}, "
            f"directories={sorted(actual_directories ^ expected_directories)}",
        )
    return snapshots


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Python's CRT file descriptors cannot represent a directory handle
        # suitable for FlushFileBuffers. The Rust Windows publisher owns the
        # release-grade handle contract; this portable helper is best-effort.
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory only if destination does not exist."""

    if sys.platform == "darwin":
        renamex_np = ctypes.CDLL(None, use_errno=True).renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        rename_excl = 0x00000004
        if renamex_np(
            os.fsencode(source), os.fsencode(destination), rename_excl
        ) != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                str(destination),
            )
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise PrepareApplyError(
                "ATOMIC_NOREPLACE_UNAVAILABLE", "libc.renameat2 is unavailable"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        ) != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                str(destination),
            )
        return
    if os.name == "nt":
        # Windows rename is already no-replace for an existing directory.
        os.rename(source, destination)
        return
    raise PrepareApplyError(
        "ATOMIC_NOREPLACE_UNAVAILABLE", f"unsupported platform {sys.platform}"
    )


def _write_manifest(path: Path, plan: Mapping[str, Any]) -> None:
    manifest = deepcopy(dict(plan))
    manifest["status"] = "COMPLETE"
    text = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _stream_copy(
    source_stream: BinaryIO,
    destination_stream: BinaryIO,
    digest: Any | None,
) -> int:
    """Copy a stream while optionally hashing the bytes already being read."""

    copied_bytes = 0
    while True:
        block = source_stream.read(COPY_BUFFER_BYTES)
        if not block:
            break
        destination_stream.write(block)
        copied_bytes += len(block)
        if digest is not None:
            digest.update(block)
    return copied_bytes


def _matches_expected_stat(value: os.stat_result, expected: FileIdentity) -> bool:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    ) == (
        expected.device,
        expected.inode,
        expected.size_bytes,
        expected.mtime_ns,
    )


def _path_descriptor_matches(
    path_stat: os.stat_result, descriptor_stat: os.stat_result
) -> bool:
    """Check stable fields shared by pathname and descriptor stat APIs."""

    if os.name == "nt":
        # CPython 3.12 may report different synthetic st_dev/st_ino values for
        # a pathname and its CRT descriptor. Their identities are still held
        # stable independently around the copy; only size/mtime cross the API
        # boundary here.
        return (
            int(path_stat.st_size),
            int(path_stat.st_mtime_ns),
        ) == (
            int(descriptor_stat.st_size),
            int(descriptor_stat.st_mtime_ns),
        )
    return _stat_signature(path_stat) == _stat_signature(descriptor_stat)


def _copy_entry(
    entry: Mapping[str, Any],
    staging: Path,
    verified_source_content: set[tuple[str, int, int, int, int]],
) -> None:
    """Copy one entry with one source pass and no target content read.

    The first copy of a bound source identity computes SHA-256 in the same loop
    that feeds the destination.  Reused master flats skip that duplicate digest
    computation; every resulting target is independently hashed by the private
    staging-tree verification before publication.
    """

    source = Path(str(entry["sourcePath"]))
    if source.is_symlink():
        raise PrepareApplyError("SOURCE_CHANGED", f"source became a symlink: {source}")
    expected = _identity_from_dict(entry["sourceIdentity"])
    source_content_key = (
        normalize_sha256(expected.sha256),
        expected.size_bytes,
        expected.mtime_ns,
        expected.device,
        expected.inode,
    )
    verify_source_content = source_content_key not in verified_source_content

    relative = str(entry["destinationRelativePath"])
    destination = staging.joinpath(*PurePosixPath(relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists() or destination.exists():
        raise PrepareApplyError("STAGING_COLLISION", relative)
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            source_descriptor = os.open(source, flags)
        except OSError as error:
            raise PrepareApplyError(
                "SOURCE_CHANGED", f"cannot securely open {source}: {error}"
            ) from error
        with os.fdopen(source_descriptor, "rb", closefd=True) as source_stream:
            descriptor_before = os.fstat(source_stream.fileno())
            path_before = os.lstat(source)
            if (
                stat.S_ISLNK(path_before.st_mode)
                or not stat.S_ISREG(descriptor_before.st_mode)
                or not _path_descriptor_matches(path_before, descriptor_before)
                or not _matches_expected_stat(path_before, expected)
            ):
                raise PrepareApplyError("SOURCE_CHANGED", str(source))

            digest = hashlib.sha256() if verify_source_content else None
            with temporary.open("xb") as destination_stream:
                copied_bytes = _stream_copy(
                    source_stream, destination_stream, digest
                )
                destination_stream.flush()

            descriptor_after_copy = os.fstat(source_stream.fileno())
            path_after_copy = os.lstat(source)
            if (
                _stat_signature(descriptor_before)
                != _stat_signature(descriptor_after_copy)
                or _stat_signature(path_before) != _stat_signature(path_after_copy)
                or not _path_descriptor_matches(path_after_copy, descriptor_after_copy)
            ):
                raise PrepareApplyError("SOURCE_CHANGED", str(source))
            if copied_bytes != expected.size_bytes:
                raise PrepareApplyError("SOURCE_CHANGED", str(source))
            if digest is not None and normalize_sha256(digest.hexdigest()) != normalize_sha256(
                expected.sha256
            ):
                raise PrepareApplyError("SOURCE_CHANGED", str(source))

            # Preserve copy2 metadata semantics without making a second source
            # content pass.  The source descriptor/path are checked again so a
            # concurrent mutation around copystat cannot be silently accepted.
            shutil.copystat(source, temporary, follow_symlinks=False)
            descriptor_after_metadata = os.fstat(source_stream.fileno())
            path_after_metadata = os.lstat(source)
            if (
                _stat_signature(descriptor_before)
                != _stat_signature(descriptor_after_metadata)
                or _stat_signature(path_before) != _stat_signature(path_after_metadata)
                or not _path_descriptor_matches(
                    path_after_metadata, descriptor_after_metadata
                )
            ):
                raise PrepareApplyError("SOURCE_CHANGED", str(source))

        _fsync_file(temporary)
        copied_stat = temporary.stat()
        if (int(copied_stat.st_dev), int(copied_stat.st_ino)) == (
            expected.device,
            expected.inode,
        ):
            raise PrepareApplyError("COPY_VERIFY_FAILED", f"hardlink alias: {relative}")
        if (
            int(copied_stat.st_size) != expected.size_bytes
            or int(copied_stat.st_mtime_ns) != expected.mtime_ns
            or copied_stat.st_nlink != 1
        ):
            raise PrepareApplyError("COPY_VERIFY_FAILED", relative)
        if verify_source_content:
            verified_source_content.add(source_content_key)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def apply_prepare_plan(
    plan: Mapping[str, Any], destination: str | os.PathLike[str]
) -> dict[str, Any]:
    """Apply a validated plan with all-or-nothing directory publication.

    Reapplying the same plan to a complete, byte-for-byte valid tree is a
    verified no-op.  An existing tree from another plan is never overwritten.
    """

    _validate_plan(plan)
    requested = Path(destination).expanduser().resolve(strict=False)
    planned = Path(str(plan["destination"])).expanduser().resolve(strict=False)
    if requested != planned:
        raise PrepareApplyError(
            "DESTINATION_MISMATCH", f"plan={planned}, requested={requested}"
        )

    if requested.exists() or requested.is_symlink():
        if requested.is_symlink() or not requested.is_dir():
            raise PrepareApplyError("DESTINATION_PLAN_CONFLICT", str(requested))
        _verify_published_tree(requested, plan)
        return {
            "status": "ALREADY_COMPLETE",
            "planId": plan["planId"],
            "destination": str(requested),
        }

    parent = requested.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / ("." + requested.name + ".prepare.lock")
    lock_descriptor: int | None = None
    lock_owned = False
    staging: Path | None = None
    try:
        try:
            lock_descriptor = os.open(
                lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            lock_owned = True
            os.write(lock_descriptor, (str(plan["planId"]) + "\n").encode("utf-8"))
            os.fsync(lock_descriptor)
        except FileExistsError as error:
            raise PrepareApplyError("DESTINATION_LOCKED", str(lock_path)) from error

        # Recheck after acquiring the lock; another writer may have completed
        # between the initial lookup and lock acquisition.
        if requested.exists() or requested.is_symlink():
            if requested.is_symlink() or not requested.is_dir():
                raise PrepareApplyError("DESTINATION_PLAN_CONFLICT", str(requested))
            _verify_published_tree(requested, plan)
            return {
                "status": "ALREADY_COMPLETE",
                "planId": plan["planId"],
                "destination": str(requested),
            }

        # Every generated plan explicitly says that non-copied audit entries get
        # a stat-coordinate check, not a fresh content claim.  Legacy schema-1
        # plans lack that declaration, so preserve their original all-source
        # SHA-256 preflight.  Copied sources are content-verified in their only
        # required read pass below.
        policy = plan.get("policy", {})
        streaming_policy = (
            isinstance(policy, Mapping)
            and policy.get("applyVerification") == APPLY_VERIFICATION_POLICY
        )
        checked_source_paths: set[str] = set()
        verified_source_content: set[tuple[str, int, int, int, int]] = set()
        for entry in plan["entries"]:
            source = Path(str(entry["sourcePath"]))
            source_key = str(source)
            if source_key in checked_source_paths:
                continue
            if source.is_symlink():
                raise PrepareApplyError("SOURCE_CHANGED", str(source))
            expected = _identity_from_dict(entry["sourceIdentity"])
            try:
                if streaming_policy:
                    _verify_file_identity_stat(source, expected)
                else:
                    actual = _compute_file_identity(source)
                    if not _same_source_identity(actual, expected):
                        raise PrepareApplyError("SOURCE_CHANGED", str(source))
                    verified_source_content.add(
                        (
                            normalize_sha256(expected.sha256),
                            expected.size_bytes,
                            expected.mtime_ns,
                            expected.device,
                            expected.inode,
                        )
                    )
            except (OSError, RuntimeError, ValueError) as error:
                raise PrepareApplyError("SOURCE_CHANGED", f"{source}: {error}") from error
            checked_source_paths.add(source_key)

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{requested.name}.staging-{str(plan['planId'])[7:19]}-",
                dir=parent,
            )
        )
        for entry in plan["entries"]:
            if entry["action"] in {"COPY_LIGHT", "COPY_MASTER_FLAT"}:
                _copy_entry(entry, staging, verified_source_content)
        _write_manifest(staging / MANIFEST_NAME, plan)
        staging_snapshot = _verify_published_tree(staging, plan)
        _fsync_directory(staging)

        if requested.exists() or requested.is_symlink():
            raise PrepareApplyError("DESTINATION_PLAN_CONFLICT", str(requested))
        try:
            _rename_no_replace(staging, requested)
        except FileExistsError as error:
            raise PrepareApplyError(
                "DESTINATION_PLAN_CONFLICT", str(requested)
            ) from error
        staging = None
        try:
            _fsync_directory(parent)
            _verify_published_tree(
                requested, plan, trusted_snapshot=staging_snapshot
            )
        except Exception as error:
            raise PrepareApplyError(
                "COMMIT_UNCERTAIN",
                f"published {requested}, but post-commit verification failed: {error}",
            ) from error
        return {
            "status": "APPLIED",
            "planId": plan["planId"],
            "destination": str(requested),
        }
    except PrepareError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise PrepareApplyError("PREPARE_TRANSACTION_FAILED", str(error)) from error
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if lock_owned:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "LAYOUT_ID",
    "MANIFEST_NAME",
    "PLAN_KIND",
    "PLAN_SCHEMA_VERSION",
    "PrepareApplyError",
    "PrepareError",
    "PreparePlanError",
    "apply_prepare_plan",
    "build_prepare_plan",
    "load_prepare_plan",
    "safe_slug",
]
