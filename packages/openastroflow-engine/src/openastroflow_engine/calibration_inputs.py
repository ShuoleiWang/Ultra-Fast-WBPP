"""Content-bound calibration metadata, source identity, and generated-master reuse."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from lightframeqc.cfa import CFA_PATTERNS, normalize_pattern as normalize_cfa_pattern
from lightframeqc.content_hash import file_sha256
from .calibration_policy import STRICT, MONO_STANDARD, metadata_changes, same_metadata, cfa_for_workflow, resolve_dark_bias
from .calibration import CalibrationError, FrameInfo, read_frame_info


PIPELINE_VERSION = "portable-pixel-pipeline-v1"
OUTPUT_STATE = "UNSOLVED_WORKING"
REGISTRATION_RESAMPLERS = frozenset({"bilinear", "lanczos-3-clamped"})
# v3: tap weights come from the deterministic 2048-interval table
# (`lanczos_table.py`, cubic Lagrange interpolation of nodes computed by the
# module's own series) instead of the host's libm; the interpolation contract
# is unchanged, the weights differ from the exact ones by less than 1e-12
# before Float32 rounding, and the result no longer depends on the C library.
LANCZOS3_REGISTRATION_ALGORITHM = (
    "normalized-lanczos-3-domain-union-support-clamp-v3-table2048"
)
NUMPY_WARP_KERNEL_ID = "numpy-lanczos3-warp-v3-table2048"
# Fused calibrate+register working set per Light: the Float32 result, one
# master temporary during subtraction/division, and masks/temporaries.
FUSED_LIGHT_BYTES_PER_PIXEL = 12
# Native warp per output row: Float32 band, finite mask, statistics selection
# and the big-endian conversion inside the FITS writer.
NATIVE_WARP_BYTES_PER_PIXEL = 16


@dataclass(frozen=True, slots=True)
class MasterMetadataOverride:
    """Explicit content-bound acquisition metadata for one supplied master."""

    source_sha256: str
    camera: str | None = None
    gain: float | None = None
    offset: float | None = None
    binning_x: int | None = None
    binning_y: int | None = None
    filter_name: str | None = None
    cfa_pattern: str | None = None
    readout_mode: str | None = None
    temperature_celsius: float | None = None
    exposure_seconds: float | None = None
    bias_included: bool | None = None
    numeric_domain: str | None = None
    normalized_unit_scale: float | None = None

    def validate(self) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("master override source_sha256 must be a lowercase sha256: digest")
        for name in ("camera", "filter_name", "cfa_pattern", "readout_mode"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"master override {name} must be non-empty")
        for name in ("gain", "offset", "temperature_celsius", "exposure_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                raise ValueError(f"master override {name} must be finite")
        if self.exposure_seconds is not None and self.exposure_seconds < 0:
            raise ValueError("master override exposure_seconds must be nonnegative")
        for name in ("binning_x", "binning_y"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"master override {name} must be a positive integer")
        if self.bias_included is not None and not isinstance(self.bias_included, bool):
            raise ValueError("master override bias_included must be boolean or None")
        if (self.numeric_domain is None) != (self.normalized_unit_scale is None):
            raise ValueError(
                "master override numeric_domain and normalized_unit_scale must be declared together"
            )
        if self.numeric_domain is not None:
            if not isinstance(self.numeric_domain, str) or not self.numeric_domain.strip():
                raise ValueError("master override numeric_domain must be non-empty")
            if (
                isinstance(self.normalized_unit_scale, bool)
                or not isinstance(self.normalized_unit_scale, (int, float))
                or not math.isfinite(float(self.normalized_unit_scale))
                or float(self.normalized_unit_scale) <= 0
            ):
                raise ValueError(
                    "master override normalized_unit_scale must be finite and positive"
                )
            if self.numeric_domain.strip().upper() not in {
                "NORMALIZED_UNIT",
                "SENSOR_CODE",
            }:
                raise ValueError(
                    "master override numeric_domain must be NORMALIZED_UNIT or SENSOR_CODE"
                )

    def serializable(self) -> dict[str, Any]:
        return {
            "sourceSha256": self.source_sha256,
            "camera": self.camera,
            "gain": self.gain,
            "offset": self.offset,
            "binning": [self.binning_x, self.binning_y] if self.binning_x is not None else None,
            "filter": self.filter_name,
            "cfaPattern": self.cfa_pattern,
            "readoutMode": self.readout_mode,
            "temperatureCelsius": self.temperature_celsius,
            "exposureSeconds": self.exposure_seconds,
            "biasIncluded": self.bias_included,
            "numericDomain": self.numeric_domain,
            "normalizedUnitScale": self.normalized_unit_scale,
        }


@dataclass(frozen=True, slots=True)
class RawFrameMetadataOverride:
    source_sha256: str
    cfa_pattern: str

    def validate(self) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("raw-frame override source_sha256 must be a lowercase sha256: digest")
        if not isinstance(self.cfa_pattern, str) or not self.cfa_pattern.strip():
            raise ValueError("raw-frame override cfa_pattern must be non-empty")
        if self.cfa_pattern.strip().upper() in {"UNKNOWN", "UNSPECIFIED"}:
            raise ValueError("raw-frame override must explicitly confirm NONE or a CFA pattern")

    def serializable(self) -> dict[str, str]:
        return {
            "sourceSha256": self.source_sha256,
            "cfaPattern": self.cfa_pattern.strip().upper(),
        }


@dataclass(frozen=True, slots=True)
class _InternalSourceIdentity:
    """E2E-owned source binding for one private generated-master handoff.

    This is deliberately absent from the public recipe and worker protocol.
    It carries upstream expectations only; the consumer freshly hashes every
    bound source rather than using this object as a digest cache.
    """

    path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int

    def validate(self) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.sha256) is None:
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_SOURCE_INVALID",
                "trusted source SHA-256 must be a lowercase sha256: digest",
                path=self.path,
            )
        for name in ("size_bytes", "mtime_ns", "device", "inode"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CalibrationError(
                    "TRUSTED_GENERATED_CALIBRATION_SOURCE_INVALID",
                    f"trusted source {name} must be a nonnegative integer",
                    path=self.path,
                )

    def stat_identity(self) -> dict[str, int]:
        return {
            "sizeBytes": self.size_bytes,
            "mtimeNs": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True, slots=True)
class _TrustedCalibrationSource:
    """One exact original input bound to an in-process generated-master set."""

    role: str
    identity: _InternalSourceIdentity


@dataclass(frozen=True, slots=True)
class _TrustedGeneratedMaster:
    """Private, content-bound identity and semantics for one E2E-built master.

    This type is intentionally absent from the public recipe and worker
    protocol.  It prevents an E2E-owned intermediate from being smuggled
    through the weaker user-supplied-master interface.
    """

    role: str
    path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int
    frame_info: FrameInfo
    bias_included: bool | None = None
    application_scale: float | None = None

    def stat_identity(self) -> dict[str, int]:
        return {
            "sizeBytes": self.size_bytes,
            "mtimeNs": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True, slots=True)
class _TrustedGeneratedCalibrationSet:
    """In-process trust handoff from E2E registration calibration to pixels."""

    master_bias: _TrustedGeneratedMaster | None
    master_darks: tuple[_TrustedGeneratedMaster, ...]
    master_flats: tuple[_TrustedGeneratedMaster, ...]
    source_bindings: tuple[_TrustedCalibrationSource, ...]
    source_manifest_sha256: str
    upstream_receipt_path: str
    upstream_receipt_sha256: str
    upstream_receipt_size_bytes: int
    upstream_receipt_mtime_ns: int
    upstream_receipt_device: int
    upstream_receipt_inode: int

    def upstream_receipt_stat_identity(self) -> dict[str, int]:
        return {
            "sizeBytes": self.upstream_receipt_size_bytes,
            "mtimeNs": self.upstream_receipt_mtime_ns,
            "device": self.upstream_receipt_device,
            "inode": self.upstream_receipt_inode,
        }


_SourceIdentityCache = dict[str, tuple[str, dict[str, int]]]


def _apply_master_metadata_overrides(
    info_groups: Sequence[dict[Path, FrameInfo]],
    overrides: Sequence[MasterMetadataOverride],
    source_aliases: Mapping[str, Path],
    identity_cache: _SourceIdentityCache | None = None,
) -> tuple[dict[Path, FrameInfo], ...]:
    all_items = [(path, info) for group in info_groups for path, info in group.items()]
    digests: dict[str, list[Path]] = {}
    for path, _ in all_items:
        _, digest, _ = _source_identity(path, source_aliases, identity_cache)
        digests.setdefault(digest, []).append(path)
    override_by_path: dict[Path, MasterMetadataOverride] = {}
    for override in overrides:
        matches = digests.get(override.source_sha256, [])
        if len(matches) != 1:
            raise CalibrationError(
                "MASTER_METADATA_OVERRIDE_SOURCE_AMBIGUOUS",
                "override digest must bind exactly one supplied master",
            )
        override_by_path[matches[0]] = override
    result: list[dict[Path, FrameInfo]] = []
    for group in info_groups:
        updated: dict[Path, FrameInfo] = {}
        for path, info in group.items():
            override = override_by_path.get(path)
            if override is None:
                updated[path] = info
            else:
                changes = metadata_changes(override)
                if "cfa_pattern" in changes and info.cfa_pattern not in {"UNKNOWN", "UNSPECIFIED", ""} and changes["cfa_pattern"] != info.cfa_pattern.strip().upper():
                    raise CalibrationError("MASTER_CFA_OVERRIDE_CONFLICT", "An override cannot replace explicit source CFA metadata", path=info.path)
                if (
                    override.numeric_domain is not None
                    and info.numeric_domain_authority
                    in {"FITS_STORAGE_ENDPOINTS", "TRUSTED_XISF_CONVERSION"}
                ):
                    declared_scale = float(override.normalized_unit_scale)
                    if (
                        info.normalized_unit_scale is None
                        or not math.isclose(
                            declared_scale,
                            float(info.normalized_unit_scale),
                            rel_tol=0.0,
                            abs_tol=max(
                                1e-12,
                                8.0
                                * np.finfo(np.float64).eps
                                * max(abs(declared_scale), 1.0),
                            ),
                        )
                    ):
                        raise CalibrationError(
                            "MASTER_NUMERIC_DOMAIN_OVERRIDE_CONFLICT",
                            "hash-bound numeric-domain override conflicts with trusted storage metadata",
                            path=str(source_aliases.get(str(path), path)),
                        )
                if (
                    override.numeric_domain is not None
                    and info.numeric_domain_authority == "FITS_BUNIT_UNSUPPORTED"
                ):
                    raise CalibrationError(
                        "MASTER_NUMERIC_DOMAIN_OVERRIDE_CONFLICT",
                        "hash-bound sensor-code override conflicts with explicit FITS BUNIT",
                        path=str(source_aliases.get(str(path), path)),
                    )
                preserve_storage_authority = (
                    override.numeric_domain is not None
                    and info.numeric_domain_authority
                    in {"FITS_STORAGE_ENDPOINTS", "TRUSTED_XISF_CONVERSION"}
                )
                updated[path] = replace(
                    info,
                    **metadata_changes(override),
                    numeric_domain=(
                        info.numeric_domain
                        if preserve_storage_authority
                        else override.numeric_domain.strip().upper()
                        if override.numeric_domain is not None
                        else info.numeric_domain
                    ),
                    normalized_unit_scale=(
                        info.normalized_unit_scale
                        if preserve_storage_authority
                        else float(override.normalized_unit_scale)
                        if override.normalized_unit_scale is not None
                        else info.normalized_unit_scale
                    ),
                    numeric_domain_authority=(
                        info.numeric_domain_authority
                        if preserve_storage_authority
                        else "CONTENT_BOUND_OVERRIDE"
                        if override.numeric_domain is not None
                        else info.numeric_domain_authority
                    ),
                )
        result.append(updated)
    return tuple(result)


def _apply_raw_frame_metadata_overrides(
    info_groups: Sequence[dict[Path, FrameInfo]],
    overrides: Sequence[RawFrameMetadataOverride],
    source_aliases: Mapping[str, Path],
    identity_cache: _SourceIdentityCache | None = None,
) -> tuple[dict[Path, FrameInfo], ...]:
    all_items = [(path, info) for group in info_groups for path, info in group.items()]
    digests: dict[str, list[Path]] = {}
    for path, _ in all_items:
        _, digest, _ = _source_identity(path, source_aliases, identity_cache)
        digests.setdefault(digest, []).append(path)
    override_by_path: dict[Path, RawFrameMetadataOverride] = {}
    for override in overrides:
        matches = digests.get(override.source_sha256, [])
        if len(matches) != 1:
            raise CalibrationError(
                "RAW_FRAME_METADATA_OVERRIDE_SOURCE_AMBIGUOUS",
                "raw-frame override digest must bind exactly one current raw source",
            )
        override_by_path[matches[0]] = override
    updated_groups: list[dict[Path, FrameInfo]] = []
    for group in info_groups:
        updated: dict[Path, FrameInfo] = {}
        for path, info in group.items():
            override = override_by_path.get(path)
            if override is None:
                updated[path] = info
                continue
            current = info.cfa_pattern.strip().upper()
            confirmed = override.cfa_pattern.strip().upper()
            if current not in {"", "UNKNOWN", "UNSPECIFIED"} and current != confirmed:
                raise CalibrationError(
                    "RAW_CFA_OVERRIDE_CONFLICT",
                    f"explicit source CFA {current} cannot be replaced by {confirmed}",
                    path=str(source_aliases.get(str(path), path)),
                )
            updated[path] = replace(info, cfa_pattern=confirmed)
        updated_groups.append(updated)
    return tuple(updated_groups)


def _trust_private_xisf_numeric_domains(
    info_groups: Sequence[dict[Path, FrameInfo]],
    source_aliases: Mapping[str, Path],
) -> tuple[dict[Path, FrameInfo], ...]:
    updated_groups: list[dict[Path, FrameInfo]] = []
    for group in info_groups:
        updated: dict[Path, FrameInfo] = {}
        for path, info in group.items():
            original = source_aliases.get(str(path))
            if original is None or original.suffix.casefold() != ".xisf":
                updated[path] = info
                continue
            if (
                info.numeric_domain_authority != "SELF_DECLARED_HEADER"
                or info.numeric_domain == "UNDECLARED"
                or info.normalized_unit_scale is None
            ):
                raise CalibrationError(
                    "XISF_NUMERIC_DOMAIN_UNDECLARED",
                    "private XISF conversion lacks explicit finite bounds for its numeric domain",
                    path=str(original),
                )
            updated[path] = replace(
                info, numeric_domain_authority="TRUSTED_XISF_CONVERSION"
            )
        updated_groups.append(updated)
    return tuple(updated_groups)


def _master_dark_bias_semantics(
    paths: Sequence[Path],
    overrides: Sequence[MasterMetadataOverride],
    source_aliases: Mapping[str, Path],
    identity_cache: _SourceIdentityCache | None = None,
    workflow: str = STRICT,
) -> dict[Path, bool]:
    """Resolve content-bound choices, explicit headers and workflow defaults."""

    override_by_digest = {item.source_sha256: item for item in overrides}
    result: dict[Path, bool] = {}
    for path in paths:
        original, digest, _ = _source_identity(
            path, source_aliases, identity_cache
        )
        override = override_by_digest.get(digest)
        explicit = override.bias_included if override is not None else None
        header = read_frame_info(path).bias_included if workflow == MONO_STANDARD else None
        resolved = resolve_dark_bias(explicit, header, workflow)
        if resolved is None:
            raise CalibrationError(
                "MASTER_DARK_BIAS_SEMANTICS_REQUIRED",
                "every supplied MasterDark requires a hash-bound biasIncluded metadata override",
                path=str(original),
            )
        result[path] = resolved
    return result


def _required_mismatch(left: Any, right: Any, unknown: Any = "UNKNOWN") -> bool:
    """Unknown metadata never proves calibration compatibility."""

    if left is None or right is None or left == unknown or right == unknown:
        return True
    return left != right


def _assert_compatible(
    reference: FrameInfo,
    candidate: FrameInfo,
    *,
    compare_filter: bool = False,
    compare_exposure: bool = False,
    compare_target: bool = False,
    compare_temperature: bool = False,
    temperature_tolerance_celsius: float = 3.0,
    workflow: str = STRICT,
) -> None:
    mismatches: dict[str, Any] = {}
    if reference.shape != candidate.shape:
        mismatches["shape"] = [list(reference.shape), list(candidate.shape)]
    for name in (
        "camera",
        "gain",
        "offset",
        "binning_x",
        "binning_y",
        "cfa_pattern",
        "readout_mode",
    ):
        left = getattr(reference, name)
        right = getattr(candidate, name)
        if name == "cfa_pattern":
            # Mono frames and Bayer frames of one pattern each match among
            # themselves; a mosaic can only be calibrated by masters of the
            # same pattern (the dark/bias are pixel-wise, the flat per colour).
            left, right = cfa_for_workflow(left, workflow), cfa_for_workflow(right, workflow)
            for value in (left, right):
                if value not in {"NONE", "UNKNOWN", "UNSPECIFIED", ""} and normalize_cfa_pattern(value) not in CFA_PATTERNS:
                    mismatches[name] = [left, right]
        if not same_metadata(left, right, workflow):
            mismatches[name] = [left, right]
    if compare_filter and _required_mismatch(
        reference.filter_name, candidate.filter_name
    ):
        mismatches["filter"] = [reference.filter_name, candidate.filter_name]
    if compare_exposure:
        left = reference.exposure_seconds
        right = candidate.exposure_seconds
        if left is None or right is None or not math.isclose(
            left, right, rel_tol=0.0, abs_tol=1e-6
        ):
            mismatches["exposureSeconds"] = [left, right]
    if compare_target and _required_mismatch(reference.target, candidate.target):
        mismatches["target"] = [reference.target, candidate.target]
    if compare_temperature:
        left = reference.temperature_celsius
        right = candidate.temperature_celsius
        if (
            (left is None or right is None) and workflow != MONO_STANDARD
            or (left is not None and right is not None and (not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > temperature_tolerance_celsius))
        ):
            mismatches["temperatureCelsius"] = [left, right]
    if mismatches:
        raise CalibrationError(
            "CALIBRATION_PROFILE_MISMATCH",
            json.dumps(mismatches, sort_keys=True, separators=(",", ":")),
            path=candidate.path,
        )


def _numeric_application_scale(
    target: FrameInfo,
    additive: FrameInfo,
    *,
    target_label: str,
    additive_label: str,
) -> float:
    allowed_authorities = {
        "FITS_STORAGE_ENDPOINTS",
        "TRUSTED_XISF_CONVERSION",
        "CONTENT_BOUND_OVERRIDE",
    }
    allowed_domain = lambda value: (  # noqa: E731
        value in {"NORMALIZED_UNIT", "SENSOR_CODE"}
        or re.fullmatch(r"INTEGER_(?:8|16|32|64)_PHYSICAL_0_BASED", value)
        is not None
    )
    target_scale = target.normalized_unit_scale
    additive_scale = additive.normalized_unit_scale
    if (
        target.numeric_domain_authority not in allowed_authorities
        or additive.numeric_domain_authority not in allowed_authorities
        or not allowed_domain(target.numeric_domain)
        or not allowed_domain(additive.numeric_domain)
        or target_scale is None
        or additive_scale is None
        or not math.isfinite(target_scale)
        or not math.isfinite(additive_scale)
        or target_scale <= 0
        or additive_scale <= 0
    ):
        raise CalibrationError(
            "CALIBRATION_NUMERIC_DOMAIN_AMBIGUOUS",
            f"{target_label} domain {target.numeric_domain}/{target_scale} and "
            f"{additive_label} domain {additive.numeric_domain}/{additive_scale} "
            "do not define an additive application scale",
            path=additive.path,
        )
    scale = float(target_scale / additive_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise CalibrationError(
            "CALIBRATION_APPLICATION_SCALE_INVALID",
            "additive calibration application scale is not finite and positive",
            path=additive.path,
        )
    return scale


def _numeric_domain_metadata(info: FrameInfo) -> dict[str, Any]:
    if (
        info.normalized_unit_scale is None
        or info.numeric_domain_authority
        not in {
            "FITS_STORAGE_ENDPOINTS",
            "TRUSTED_XISF_CONVERSION",
            "CONTENT_BOUND_OVERRIDE",
        }
    ):
        raise CalibrationError(
            "CALIBRATION_NUMERIC_DOMAIN_AMBIGUOUS",
            f"numeric domain {info.numeric_domain!r} has no normalized-unit scale",
            path=info.path,
        )
    return {
        "OAFNDOM": info.numeric_domain,
        "OAFNSCL": info.normalized_unit_scale,
    }


def _hash_file(path: Path) -> str:
    return "sha256:" + file_sha256(path)


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat(follow_symlinks=False)
    return {
        "sizeBytes": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _trusted_source_manifest_sha256(
    bindings: Sequence[_TrustedCalibrationSource],
) -> str:
    payload = [
        {
            "role": binding.role,
            "path": binding.identity.path,
            "sha256": binding.identity.sha256,
            **binding.identity.stat_identity(),
        }
        for binding in sorted(
            bindings,
            key=lambda item: (item.role, os.path.normcase(item.identity.path)),
        )
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _capture_trusted_generated_master(
    path: Path,
    *,
    role: str,
    frame_info: FrameInfo,
    bias_included: bool | None = None,
    application_scale: float | None = None,
) -> _TrustedGeneratedMaster:
    canonical = path.expanduser().resolve(strict=True)
    stat_identity = _stat_identity(canonical)
    return _TrustedGeneratedMaster(
        role=role,
        path=str(canonical),
        sha256=_hash_file(canonical),
        size_bytes=stat_identity["sizeBytes"],
        mtime_ns=stat_identity["mtimeNs"],
        device=stat_identity["device"],
        inode=stat_identity["inode"],
        frame_info=replace(frame_info, path=str(canonical), role=role),
        bias_included=bias_included,
        application_scale=application_scale,
    )


def _capture_trusted_generated_calibration_set(
    *,
    master_bias: tuple[Path, FrameInfo] | None,
    master_darks: Sequence[tuple[Path, FrameInfo, bool]],
    master_flats: Sequence[tuple[Path, FrameInfo, float]],
    source_groups: Sequence[tuple[str, Sequence[Path]]],
    source_identities: Mapping[str, _InternalSourceIdentity],
    upstream_receipt_path: Path,
) -> _TrustedGeneratedCalibrationSet:
    bindings: list[_TrustedCalibrationSource] = []
    for role, paths in source_groups:
        for path in paths:
            canonical = path.expanduser().resolve(strict=True)
            identity = source_identities.get(str(canonical))
            if identity is None:
                raise CalibrationError(
                    "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                    "trusted calibration source manifest is missing an exact source identity",
                    path=str(canonical),
                )
            bindings.append(_TrustedCalibrationSource(role=role, identity=identity))
    binding_tuple = tuple(bindings)
    canonical_receipt = upstream_receipt_path.expanduser().resolve(strict=True)
    receipt_stat = _stat_identity(canonical_receipt)
    captured_bias = (
        _capture_trusted_generated_master(
            master_bias[0], role="MASTER_BIAS", frame_info=master_bias[1]
        )
        if master_bias is not None
        else None
    )
    captured_darks = tuple(
        _capture_trusted_generated_master(
            path,
            role="MASTER_DARK",
            frame_info=info,
            bias_included=bias_included,
        )
        for path, info, bias_included in master_darks
    )
    captured_flats = tuple(
        _capture_trusted_generated_master(
            path,
            role="MASTER_FLAT",
            frame_info=info,
            application_scale=application_scale,
        )
        for path, info, application_scale in master_flats
    )
    try:
        upstream_payload = json.loads(canonical_receipt.read_text(encoding="utf-8"))
        upstream_artifacts = upstream_payload["artifacts"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_RECEIPT_INVALID",
            "upstream registration-calibration receipt is not a valid artifact manifest",
            path=str(canonical_receipt),
        ) from error
    if (
        upstream_payload.get("schemaVersion") != 1
        or upstream_payload.get("stage") != "registration-calibration-masters"
        or not isinstance(upstream_artifacts, list)
    ):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_RECEIPT_INVALID",
            "upstream receipt has the wrong schema or stage",
            path=str(canonical_receipt),
        )
    for master in (
        captured_bias,
        *captured_darks,
        *captured_flats,
    ):
        if master is None:
            continue
        try:
            master_relative = Path(master.path).relative_to(
                canonical_receipt.parent.parent
            )
        except ValueError as error:
            raise CalibrationError(
                "TRUSTED_GENERATED_MASTER_PATH_MISMATCH",
                "generated master is outside the E2E-owned staging directory",
                path=master.path,
            ) from error
        expected_artifact_path = "artifact/" + master_relative.as_posix()
        matches = [
            item
            for item in upstream_artifacts
            if isinstance(item, dict)
            and item.get("path") == expected_artifact_path
            and item.get("sha256") == master.sha256
            and item.get("sizeBytes") == master.size_bytes
        ]
        if len(matches) != 1:
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_RECEIPT_MISMATCH",
                "generated master is not uniquely bound by the upstream receipt",
                path=master.path,
            )
    if captured_bias is not None and upstream_payload.get("masterBias", {}).get(
        "mode"
    ) != "BUILT_FROM_RAW":
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_RECEIPT_MISMATCH",
            "upstream receipt does not identify the generated MasterBias",
            path=captured_bias.path,
        )
    upstream_darks = upstream_payload.get("masterDarks")
    upstream_flats = upstream_payload.get("masterFlats")
    if not isinstance(upstream_darks, dict) or not isinstance(upstream_flats, dict):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_RECEIPT_INVALID",
            "upstream receipt lacks Dark or Flat semantic records",
            path=str(canonical_receipt),
        )
    for master in captured_darks:
        exposure = master.frame_info.exposure_seconds
        record = upstream_darks.get(format(float(exposure), ".9g")) if exposure else None
        if (
            not isinstance(record, dict)
            or record.get("mode") != "BUILT_FROM_RAW"
            or record.get("biasIncluded") is not master.bias_included
        ):
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_RECEIPT_MISMATCH",
                "upstream MasterDark semantics disagree with the trust handoff",
                path=master.path,
            )
    for master in captured_flats:
        record = upstream_flats.get(master.frame_info.filter_name)
        if (
            not isinstance(record, dict)
            or record.get("mode") != "BUILT_FROM_RAW"
            or record.get("applicationScale") != master.application_scale
        ):
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_RECEIPT_MISMATCH",
                "upstream MasterFlat semantics disagree with the trust handoff",
                path=master.path,
            )
    return _TrustedGeneratedCalibrationSet(
        master_bias=captured_bias,
        master_darks=captured_darks,
        master_flats=captured_flats,
        source_bindings=binding_tuple,
        source_manifest_sha256=_trusted_source_manifest_sha256(binding_tuple),
        upstream_receipt_path=str(canonical_receipt),
        upstream_receipt_sha256=_hash_file(canonical_receipt),
        upstream_receipt_size_bytes=receipt_stat["sizeBytes"],
        upstream_receipt_mtime_ns=receipt_stat["mtimeNs"],
        upstream_receipt_device=receipt_stat["device"],
        upstream_receipt_inode=receipt_stat["inode"],
    )


def _validate_trusted_generated_master(
    master: _TrustedGeneratedMaster,
) -> Path:
    if not isinstance(master, _TrustedGeneratedMaster):
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_INVALID",
            "generated calibration entries must use the private typed identity",
        )
    if master.role not in {"MASTER_BIAS", "MASTER_DARK", "MASTER_FLAT"}:
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_INVALID",
            "generated calibration master has an invalid role",
            path=master.path,
        )
    try:
        canonical = Path(master.path).expanduser().resolve(strict=True)
    except OSError as error:
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_CHANGED",
            "generated calibration master disappeared",
            path=master.path,
        ) from error
    if master.path != str(canonical):
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_PATH_MISMATCH",
            "generated calibration master path is not its exact canonical path",
            path=master.path,
        )
    if re.fullmatch(r"sha256:[0-9a-f]{64}", master.sha256) is None:
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_INVALID",
            "generated calibration master SHA-256 is invalid",
            path=master.path,
        )
    if _stat_identity(canonical) != master.stat_identity():
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_CHANGED",
            "generated calibration master stat identity changed after production",
            path=master.path,
        )
    if _hash_file(canonical) != master.sha256:
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_CHANGED",
            "generated calibration master content changed after production",
            path=master.path,
        )
    if not isinstance(master.frame_info, FrameInfo):
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_INVALID",
            "generated calibration master metadata is not typed FrameInfo",
            path=master.path,
        )
    actual = read_frame_info(canonical)
    if actual.role != master.role or actual.shape != master.frame_info.shape:
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_METADATA_MISMATCH",
            "generated calibration master role or shape disagrees with its bound metadata",
            path=master.path,
        )
    if master.role == "MASTER_DARK":
        if not isinstance(master.bias_included, bool) or master.application_scale is not None:
            raise CalibrationError(
                "TRUSTED_GENERATED_MASTER_SEMANTICS_INVALID",
                "generated MasterDark requires explicit biasIncluded semantics only",
                path=master.path,
            )
    elif master.role == "MASTER_FLAT":
        if (
            master.bias_included is not None
            or master.application_scale != 1.0
        ):
            raise CalibrationError(
                "TRUSTED_GENERATED_MASTER_SEMANTICS_INVALID",
                "generated MasterFlat must bind applicationScale=1.0",
                path=master.path,
            )
    elif master.bias_included is not None or master.application_scale is not None:
        raise CalibrationError(
            "TRUSTED_GENERATED_MASTER_SEMANTICS_INVALID",
            "generated MasterBias cannot carry Dark or Flat semantics",
            path=master.path,
        )
    return canonical


def _validate_trusted_generated_calibration_set(
    trusted: _TrustedGeneratedCalibrationSet,
    *,
    source_groups: Sequence[tuple[str, Sequence[Path]]],
    source_aliases: Mapping[str, Path],
    identity_cache: _SourceIdentityCache | None = None,
) -> dict[str, Any]:
    if not isinstance(trusted, _TrustedGeneratedCalibrationSet):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_INVALID",
            "generated-master reuse requires the private typed calibration set",
        )
    try:
        upstream_receipt = Path(trusted.upstream_receipt_path).expanduser().resolve(
            strict=True
        )
    except OSError as error:
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_RECEIPT_CHANGED",
            "upstream registration-calibration receipt disappeared",
            path=trusted.upstream_receipt_path,
        ) from error
    if trusted.upstream_receipt_path != str(upstream_receipt):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_RECEIPT_PATH_MISMATCH",
            "upstream registration-calibration receipt path is not canonical",
            path=trusted.upstream_receipt_path,
        )
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", trusted.upstream_receipt_sha256)
        is None
        or _stat_identity(upstream_receipt)
        != trusted.upstream_receipt_stat_identity()
        or _hash_file(upstream_receipt) != trusted.upstream_receipt_sha256
    ):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_RECEIPT_CHANGED",
            "upstream registration-calibration receipt changed after the trust handoff",
            path=trusted.upstream_receipt_path,
        )
    actual_bindings: list[_TrustedCalibrationSource] = []
    expected_by_role_path: dict[tuple[str, str], _InternalSourceIdentity] = {}
    for binding in trusted.source_bindings:
        if (
            not isinstance(binding, _TrustedCalibrationSource)
            or binding.role
            not in {
                "BIAS",
                "DARK",
                "FLAT",
                "MASTER_BIAS",
                "MASTER_DARK",
                "MASTER_FLAT",
                "LIGHT",
            }
            or not isinstance(binding.identity, _InternalSourceIdentity)
        ):
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                "generated calibration source manifest contains an invalid typed binding",
            )
        binding.identity.validate()
        canonical_binding = Path(binding.identity.path).expanduser().resolve(strict=True)
        if binding.identity.path != str(canonical_binding):
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                "generated calibration source binding path is not canonical",
                path=binding.identity.path,
            )
        expected_by_role_path[(binding.role, binding.identity.path)] = binding.identity
    if len(expected_by_role_path) != len(trusted.source_bindings):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
            "generated calibration source manifest contains duplicate role/path bindings",
        )
    for role, paths in source_groups:
        for path in paths:
            original = _canonical_original_path(path, source_aliases)
            expected_identity = expected_by_role_path.get((role, str(original)))
            if expected_identity is None:
                raise CalibrationError(
                    "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                    "trusted calibration consumer input is absent from the upstream source manifest",
                    path=str(original),
                )
            _, actual_sha256, actual_stat = _source_identity(
                original, None, identity_cache
            )
            if (
                actual_stat != expected_identity.stat_identity()
                or actual_sha256 != expected_identity.sha256
            ):
                raise CalibrationError(
                    "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                    "source content or stat identity changed after upstream calibration",
                    path=str(original),
                )
            actual_bindings.append(
                _TrustedCalibrationSource(
                    role=role,
                    identity=_InternalSourceIdentity(
                        path=str(original),
                        sha256=actual_sha256,
                        size_bytes=actual_stat["sizeBytes"],
                        mtime_ns=actual_stat["mtimeNs"],
                        device=actual_stat["device"],
                        inode=actual_stat["inode"],
                    ),
                )
            )
    expected_manifest = _trusted_source_manifest_sha256(trusted.source_bindings)
    actual_manifest = _trusted_source_manifest_sha256(actual_bindings)
    if (
        trusted.source_manifest_sha256 != expected_manifest
        or actual_manifest != expected_manifest
        or tuple(
            sorted(
                (
                    binding.role,
                    binding.identity.path,
                    binding.identity.sha256,
                    binding.identity.size_bytes,
                    binding.identity.mtime_ns,
                    binding.identity.device,
                    binding.identity.inode,
                )
                for binding in trusted.source_bindings
            )
        )
        != tuple(
            sorted(
                (
                    binding.role,
                    binding.identity.path,
                    binding.identity.sha256,
                    binding.identity.size_bytes,
                    binding.identity.mtime_ns,
                    binding.identity.device,
                    binding.identity.inode,
                )
                for binding in actual_bindings
            )
        )
    ):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
            "generated calibration set is not bound to this exact input manifest",
        )
    masters = tuple(
        item
        for item in (
            trusted.master_bias,
            *trusted.master_darks,
            *trusted.master_flats,
        )
        if item is not None
    )
    paths: dict[str, _TrustedGeneratedMaster] = {}
    for master in masters:
        canonical = _validate_trusted_generated_master(master)
        key = os.path.normcase(str(canonical))
        if key in paths:
            raise CalibrationError(
                "TRUSTED_GENERATED_MASTER_DUPLICATE",
                "one generated master appears more than once",
                path=str(canonical),
            )
        paths[key] = master
    return {
        "byPath": paths,
        "bias": trusted.master_bias,
        "darks": tuple(trusted.master_darks),
        "flats": tuple(trusted.master_flats),
    }


def _canonical_original_path(
    path: Path, source_aliases: Mapping[str, Path] | None
) -> Path:
    original = (source_aliases or {}).get(str(path), path)
    return Path(original).expanduser().resolve(strict=True)


def _source_identity(
    path: Path,
    source_aliases: Mapping[str, Path] | None,
    identity_cache: _SourceIdentityCache | None = None,
) -> tuple[Path, str, dict[str, int]]:
    original = _canonical_original_path(path, source_aliases)
    cache_key = os.path.normcase(str(original))
    identity = _stat_identity(original)
    if identity_cache is not None and cache_key in identity_cache:
        digest, expected_identity = identity_cache[cache_key]
        if identity != expected_identity:
            raise CalibrationError(
                "SOURCE_CHANGED",
                "source stat identity changed after its cached content digest",
                path=str(original),
            )
        return original, digest, dict(expected_identity)
    digest = _hash_file(original)
    after = _stat_identity(original)
    if identity != after:
        raise CalibrationError(
            "SOURCE_CHANGED",
            "source changed while computing its content digest",
            path=str(original),
        )
    if identity_cache is not None:
        identity_cache[cache_key] = (digest, dict(after))
    return original, digest, after


def _find_dark(
    exposure: float | None,
    masters: Mapping[float, Path],
) -> tuple[float, Path] | None:
    if exposure is None:
        return None
    for dark_exposure, path in masters.items():
        if math.isclose(exposure, dark_exposure, rel_tol=0.0, abs_tol=1e-6):
            return dark_exposure, path
    return None

