"""Source identities, artifact records and receipt references of one pixel run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..calibration.inputs import (
    TrustedGeneratedMaster,
    hash_calibration_file,
    file_stat_identity,
    source_identity,
    SourceIdentityCache,
)
from ..calibration.policy import acquisition_receipt
from ..path_budget import name_token
from .integration import CalibrationError, FrameInfo, PixelStatistics, read_frame_info


def _canonical_inputs(
    values: Iterable[str | os.PathLike[str]], label: str, *, required: bool
) -> tuple[Path, ...]:
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise CalibrationError(
                "INPUT_NOT_FILE", f"{label} input is not a file", path=str(path)
            )
        key = os.path.normcase(str(path))
        if key in seen:
            raise CalibrationError("DUPLICATE_INPUT", f"duplicate {label} frame", path=str(path))
        seen.add(key)
        paths.append(path)
    if required and not paths:
        raise CalibrationError("INPUT_GROUP_EMPTY", f"at least one {label} frame is required")
    return tuple(sorted(paths, key=lambda path: os.path.normcase(str(path))))


def _read_infos(paths: tuple[Path, ...], expected_role: str) -> dict[Path, FrameInfo]:
    result: dict[Path, FrameInfo] = {}
    for path in paths:
        info = read_frame_info(path)
        if info.role != expected_role:
            raise CalibrationError(
                "FRAME_ROLE_MISMATCH",
                f"expected {expected_role}, found {info.role}",
                path=str(path),
            )
        result[path] = info
    return result


def require_filter(info: FrameInfo) -> str:
    if info.filter_name == "UNKNOWN":
        raise CalibrationError(
            "FILTER_UNKNOWN", "Flat and Light frames require FILTER metadata", path=info.path
        )
    return info.filter_name


def _safe_token(value: str) -> str:
    token = name_token(value)
    if not token:
        raise CalibrationError("FILENAME_TOKEN_EMPTY", f"cannot encode group name {value!r}")
    return token


def _exposure_token(value: float) -> str:
    return format(value, ".9g").replace("-", "m").replace(".", "p")


def _content_lineage_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    payload = sorted(
        (
            {
                "role": str(item["role"]),
                "sha256": str(item["sha256"]),
                "sizeBytes": int(item["sizeBytes"]),
            }
            for item in records
        ),
        key=lambda item: (item["role"], item["sha256"], item["sizeBytes"]),
    )
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_records(
    grouped: Sequence[tuple[str, tuple[Path, ...]]],
    source_aliases: Mapping[str, Path] | None = None,
    identity_cache: SourceIdentityCache | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    records: list[dict[str, Any]] = []
    identities: dict[str, dict[str, int]] = {}
    for role, paths in grouped:
        for path in paths:
            original, digest, identity = source_identity(
                path, source_aliases, identity_cache
            )
            identities[str(original)] = identity
            records.append(
                {
                    "path": str(original),
                    "role": role,
                    "sha256": digest,
                    **identity,
                }
            )
    return records, identities


def _pixel_numeric_domain_records(
    grouped: Sequence[tuple[str, Mapping[Path, FrameInfo]]],
    source_aliases: Mapping[str, Path],
    identity_cache: SourceIdentityCache,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, infos in grouped:
        for path, info in infos.items():
            original, digest, _ = source_identity(
                path, source_aliases, identity_cache
            )
            scale = info.normalized_unit_scale
            records.append(
                {
                    "role": role,
                    "path": str(original),
                    "sourceToken": digest,
                    "storageEvidence": dict(info.numeric_domain_evidence),
                    "acquisitionMetadata": acquisition_receipt(info),
                    "numericDomain": info.numeric_domain,
                    "normalizedUnitScale": scale,
                    "authority": info.numeric_domain_authority,
                    "canonicalOperation": (
                        "DIVISIVE_RESPONSE_MEDIAN_NORMALIZED"
                        if role in {"FLAT", "MASTER_FLAT"}
                        else "VALUE_TO_NORMALIZED_UNIT"
                    ),
                    "canonicalApplicationScale": (
                        None
                        if role in {"FLAT", "MASTER_FLAT"} or scale is None
                        else 1.0 / float(scale)
                    ),
                    "consumerApplicationScales": (
                        "RECORDED_PER_CALIBRATED_OUTPUT"
                        if role
                        in {"BIAS", "DARK", "MASTER_BIAS", "MASTER_DARK"}
                        else None
                    ),
                }
            )
    return records


def _verify_source_identities(identities: Mapping[str, Mapping[str, int]]) -> None:
    for value, expected in identities.items():
        path = Path(value)
        try:
            actual = file_stat_identity(path)
        except OSError as error:
            raise CalibrationError(
                "SOURCE_CHANGED", "source disappeared during processing", path=value
            ) from error
        if actual != dict(expected):
            raise CalibrationError(
                "SOURCE_CHANGED", "source identity changed during processing", path=value
            )


def _artifact_record(
    staging: Path,
    path: Path,
    kind: str,
    *,
    statistics: PixelStatistics | None = None,
    details: Mapping[str, Any] | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Describe one staged artifact; ``sha256`` may come from a streaming writer.

    A writer-side digest covers exactly the bytes it wrote, so it equals the
    file hash without rereading large intermediates.
    """

    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path.relative_to(staging)),
        "kind": kind,
        "sha256": sha256 if sha256 is not None else hash_calibration_file(path),
        "sizeBytes": stat.st_size,
    }
    if statistics is not None:
        record["statistics"] = statistics.serializable()
    if details:
        record["details"] = dict(details)
    return record


def _path_receipt_reference(
    staging: Path,
    path: Path,
    source_aliases: Mapping[str, Path] | None = None,
    trusted_generated: Mapping[str, TrustedGeneratedMaster] | None = None,
    identity_cache: SourceIdentityCache | None = None,
) -> dict[str, Any]:
    try:
        relative = str(path.relative_to(staging))
        return {"kind": "generated", "path": relative}
    except ValueError:
        trusted = (trusted_generated or {}).get(os.path.normcase(str(path)))
        if trusted is not None:
            return {
                "kind": "e2e-generated-master",
                "role": trusted.role,
                "sha256": trusted.sha256,
                "sizeBytes": trusted.size_bytes,
            }
        original, digest, identity = source_identity(
            path, source_aliases, identity_cache
        )
        return {
            "kind": "supplied-master",
            "path": str(original),
            "sha256": digest,
            "sizeBytes": identity["sizeBytes"],
        }


def _integration_record(result: Any, staging: Path) -> dict[str, Any]:
    value = result.serializable()
    output_path = Path(value["outputPath"])
    try:
        value["outputPath"] = str(output_path.relative_to(staging))
    except ValueError:
        value["outputPath"] = str(output_path)
    return value
