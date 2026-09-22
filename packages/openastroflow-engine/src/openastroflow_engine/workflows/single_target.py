"""Fail-closed end-to-end astrophotography orchestration.

The public pixel pipeline deliberately publishes unsolved working masters.  This
module composes the existing QC, calibration, registration, integration,
optional drizzle, and solver boundaries into one transaction.  A requested
output directory is published only after every filter has a newly solved and
independently verified celestial WCS.  Solver failure publishes a separate
``.unsolved`` evidence directory and returns an unsuccessful result.

All caller-owned files are opened read-only.  Outputs are built on the same
filesystem as their destination and committed with a no-replace directory
rename, so a successful receipt can never describe a partial output tree.
"""

from __future__ import annotations
from .contracts import (
    IntegrationMode,
    E2EState,
    ProgressStage,
    ProgressEvent,
    E2EError,
    ReviewApproval,
    ExplicitDecision,
    ExplicitSelection,
    parse_explicit_selection,
    DrizzleOptions,
    E2ERequest,
    E2EResult,
    _canonical_json,
    SELECTION_KIND,
    SELECTION_POLICY,
    MAX_SELECTION_DECISIONS,
    _SELECTION_DIGEST,
    _SELECTION_ORIGIN_KEYS,
    ProgressCallback,
)
from .solve import (
    _read_image_header,
    _wcs_headers_agree,
    _unit_vectors,
    _sky_separation_degrees,
    _solution_geometry,
    _validate_solution_against_hints,
    _grid_sample_points,
    _wcs_grid_disagreement,
    _accepted_solve_rms_pixels,
    _same_grid_signatures,
    _unify_same_grid_solutions,
    _validate_cross_filter_wcs,
    _verify_backend_result,
    _discard_owned_solver_output,
    _promote_solved_state,
    _solve_one,
    _SolverHints,
    _WCS_CARD_PATTERN,
)


from lightframeqc.content_hash import file_sha256
from lightframeqc.cfa import is_cfa_pattern
from .. import platform as platform_services
from ..platform import NoReplaceError, remove_file, remove_tree, rename_with_retry
from ..calibration_policy import apply_mono_workflow, bias_from_header, MONO_STANDARD, can_omit_bias, conflicting_profile_fields, workflow_receipt

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path, PurePath, PureWindowsPath
import re
import shutil
import stat
import tempfile
import threading
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence
import warnings

from ..quality_cache import quality_cache_directory
from ..review_preview import (
    MAX_REVIEW_PREVIEWS,
    MAX_TOTAL_PREVIEW_BYTES,
    REVIEW_DIRECTORY,
    bounded_review_preview,
    review_preview_name,
)

from astropy.io import fits
import numpy as np

from lightframeqc.analysis import analyze_measurements
from lightframeqc.config import QcConfig
from lightframeqc.measure import measure_paths
from lightframeqc.parallel import FrameRunner
from lightframeqc.source_extraction import NONDETERMINISTIC_WARNING, cached_extraction_self_test
from lightframeqc.models import FrameResult, GateDisposition
from lightframeqc.quality_gate import GatePolicy, evaluate_quality_gate
from lightframeqc.readers import probe_frame_metadata

from ..solvers.process import (
    verify_solver_execution_result,
)
from ..blink_session import BlinkEvidence, blink_evidence
from ..calibration import (
    cfa_metadata,
    CalibrationError,
    FrameExpression,
    FrameInfo,
    integrate_expressions,
    normalize_role,
    read_frame_info,
    robust_location,
)
from ..drizzle_native import (
    SUPPORTED_KERNELS as DRIZZLE_KERNELS_SUPPORTED,
    SUPPORTED_SCALES as DRIZZLE_SCALES_SUPPORTED,
    DrizzleGroupRequest,
    drizzle_group,
    verify_drizzle_receipt,
)
from ..calibration_inputs import (
    _InternalSourceIdentity,
    _apply_master_metadata_overrides,
    _assert_compatible,
    _find_dark,
    _master_dark_bias_semantics,
    _numeric_application_scale,
    _numeric_domain_metadata,
    _capture_trusted_generated_calibration_set,
)
from ..pixel_pipeline import (
    PipelineParameters,
    _require_filter,
    _run_portable_pipeline_fits,
)
from ..global_normalization import StellarScaleHint
from ..selection import (
    CounterfactualReport,
    FrameSelectionFeatures,
    LeaveOneOutAccumulator,
    SelectionDecision,
    SelectionParameters,
    annotate_with_counterfactual,
    confirmed_harmful,
    decide,
    exclude_confirmed,
    extract_features,
)
from ..selection.policy import selection_receipt
from ..selection.region import RegionWeightMap, region_weight_maps
from ..xisf_pixels import convert_xisf_to_fits, preflight_xisf_header
from ..path_budget import (
    PIXEL_PIPELINE_DIRECTORY,
    PIXEL_PIPELINE_STAGING_STEM,
    STAGING_SUFFIX,
    WORK_DIRECTORY,
    name_token,
)
from ..preview import render_auto_stretch_preview
from ..solver import (
    SolveRequest,
    SolverBackend,
    SolverResult,
    WcsValidation,
    validate_solver_result,
    validate_wcs_header,
    wcs_parity,
)


E2E_VERSION = "openastroflow-e2e-v1"
_DRIZZLE_REJECTION_RELATIVE_SIGNAL_FLOOR = 0.20
_DRIZZLE_REJECTION_GRADIENT_FLOOR = 0.75


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    path: str
    role: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int

    @property
    def source_id(self) -> str:
        payload = f"{self.role}\0{self.sha256}\0{Path(self.path).name}".encode("utf-8")
        return "src-" + hashlib.sha256(payload).hexdigest()[:20]

    def serializable(self) -> dict[str, Any]:
        """Return the share-safe identity written to published receipts."""

        return {
            "sourceId": self.source_id,
            "role": self.role,
            "displayName": Path(self.path).name,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
        }

    def local_serializable(self) -> dict[str, Any]:
        """Return execution-local binding data; never write it to a product."""

        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "mtimeNs": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True, slots=True)
class _RegistrationProducts:
    transforms: dict[str, tuple[tuple[float, float, float], ...]]
    quality_weights: dict[str, float]
    stellar_scale_hints: dict[str, StellarScaleHint]
    run: Any
    receipt: dict[str, Any]
    source_aliases: dict[str, str] = field(default_factory=dict)


def _emit(
    callback: ProgressCallback | None,
    stage: ProgressStage,
    status: str,
    message: str,
    *,
    current: int = 0,
    total: int = 0,
) -> None:
    if callback is not None:
        callback(ProgressEvent(stage, status, current, total, message))


def _sha256(path: Path) -> str:
    return "sha256:" + file_sha256(path)


def _input_frame_info(
    path: Path,
    parameters: PipelineParameters,
    *,
    override_identity_path: Path | None = None,
    override_source_identity: _SourceIdentity | _InternalSourceIdentity | None = None,
) -> FrameInfo:
    if path.suffix.casefold() != ".xisf":
        info = read_frame_info(path)
    else:
        preflight_xisf_header(path, parameters.xisf_decode)
        try:
            metadata = probe_frame_metadata(path)
        except Exception as error:
            code = getattr(error, "code", "XISF_HEADER_ERROR")
            raise CalibrationError(code, str(error), path=str(path)) from error
        header = metadata.header

        def number(*keys: str) -> float | None:
            for key in keys:
                value = header.get(key)
                try:
                    result = float(value) if value is not None else math.nan
                except (TypeError, ValueError):
                    continue
                if math.isfinite(result):
                    return result
            return None

        info = FrameInfo(
            path=str(path),
            role=normalize_role(metadata.role.value),
            shape=(metadata.height, metadata.width),
            filter_name=metadata.filter_name,
            exposure_seconds=metadata.exposure_seconds,
            temperature_celsius=number(
                "CCD-TEMP", "CCD_TEMP", "SENSORT", "SENSOR-T", "CAMTEMP"
            ),
            camera=metadata.camera,
            gain=metadata.gain,
            offset=metadata.offset,
            binning_x=metadata.binning_x,
            binning_y=metadata.binning_y,
            cfa_pattern=metadata.cfa_pattern,
            readout_mode=metadata.readout_mode,
            target=metadata.target,
            bias_included=bias_from_header(header),
        )
    identity_path = override_identity_path or path
    if (
        identity_path.suffix.casefold() == ".xisf"
        and path.suffix.casefold() != ".xisf"
    ):
        if (
            info.numeric_domain_authority != "SELF_DECLARED_HEADER"
            or info.numeric_domain == "UNDECLARED"
            or info.normalized_unit_scale is None
        ):
            raise CalibrationError(
                "XISF_NUMERIC_DOMAIN_UNDECLARED",
                "private XISF conversion lacks explicit finite bounds for its numeric domain",
                path=str(identity_path),
            )
        info = replace(
            info, numeric_domain_authority="TRUSTED_XISF_CONVERSION"
        )
    matches = []
    if parameters.raw_frame_metadata_overrides:
        if override_source_identity is None:
            digest = _sha256(identity_path)
        else:
            canonical_identity_path = identity_path.expanduser().resolve(strict=True)
            if override_source_identity.path != str(canonical_identity_path):
                raise CalibrationError(
                    "SOURCE_IDENTITY_CACHE_SET_MISMATCH",
                    "metadata override digest is not bound to this exact source path",
                    path=str(canonical_identity_path),
                )
            digest = override_source_identity.sha256
        matches = [
            item
            for item in parameters.raw_frame_metadata_overrides
            if item.source_sha256 == digest
        ]
    if len(matches) > 1:
        raise CalibrationError(
            "RAW_FRAME_METADATA_OVERRIDE_SOURCE_AMBIGUOUS",
            "multiple raw-frame overrides bind the same source",
            path=str(identity_path),
        )
    if matches:
        current = info.cfa_pattern.strip().upper()
        confirmed = matches[0].cfa_pattern.strip().upper()
        if current not in {"", "UNKNOWN", "UNSPECIFIED"} and current != confirmed:
            raise CalibrationError(
                "RAW_CFA_OVERRIDE_CONFLICT",
                f"explicit source CFA {current} cannot be replaced by {confirmed}",
                path=str(identity_path),
            )
        info = replace(info, cfa_pattern=confirmed)
    return apply_mono_workflow(info, parameters.calibration_workflow)


def _stage_e2e_xisf_inputs(
    grouped: Sequence[tuple[str, tuple[Path, ...]]],
    directory: Path,
    parameters: PipelineParameters,
    source_identities: Mapping[str, _SourceIdentity] | None = None,
) -> tuple[
    dict[str, tuple[Path, ...]],
    dict[str, Path],
    list[dict[str, Any]],
    dict[str, str],
]:
    directory.mkdir(parents=True, exist_ok=False)
    aliases: dict[str, Path] = {}
    conversions: list[dict[str, Any]] = []
    staged_digests: dict[str, str] = {}
    staged_groups: dict[str, tuple[Path, ...]] = {}
    sequence = 0
    for role, paths in grouped:
        staged_paths: list[Path] = []
        for path in paths:
            sequence += 1
            expected = (
                source_identities.get(str(path.resolve(strict=True)))
                if source_identities is not None
                else None
            )
            if source_identities is not None and expected is None:
                raise E2EError(
                    "SOURCE_IDENTITY_CACHE_SET_MISMATCH",
                    "pixel staging is missing the captured source identity",
                    path=str(path),
                )
            if path.suffix.casefold() == ".xisf":
                staged = directory / f"{sequence:06d}_{role.casefold()}.fits"
                receipt = convert_xisf_to_fits(path, staged, policy=parameters.xisf_decode)
                if expected is not None and (
                    receipt.source_sha256 != expected.sha256
                    or receipt.source_size_bytes != expected.size_bytes
                ):
                    remove_file(staged)
                    raise E2EError(
                        "SOURCE_CHANGED",
                        "XISF conversion does not match the captured source identity",
                        path=str(path),
                    )
                conversions.append({"role": role, **receipt.serializable()})
                staged_digests[str(staged)] = receipt.converted_sha256
            else:
                # FITS sources are consumed in place through read-only handles.
                # Their captured stat identity is rechecked at every trust
                # boundary and their content is rehashed once before
                # publication, so a private byte copy adds no detection that
                # the final gate does not already provide.
                staged = path
            staged_paths.append(staged)
            aliases[str(staged)] = path
        staged_groups[role] = tuple(staged_paths)
    return staged_groups, aliases, conversions, staged_digests


def _verify_staged_pixel_inputs(
    staged_digests: Mapping[str, str],
    identities: Sequence[_SourceIdentity] = (),
) -> None:
    """Recheck private conversions by content and original sources by identity."""

    for value, expected in staged_digests.items():
        path = Path(value)
        try:
            actual = _sha256(path)
        except OSError as error:
            raise E2EError(
                "PRIVATE_PIXEL_STAGING_CHANGED",
                "private pixel snapshot disappeared",
                path=value,
            ) from error
        if actual != expected:
            raise E2EError(
                "PRIVATE_PIXEL_STAGING_CHANGED",
                "private pixel snapshot changed during execution",
                path=value,
            )
    _verify_source_stat_identities(identities)


def _verify_source_stat_identities(identities: Sequence[_SourceIdentity]) -> None:
    for identity in identities:
        path = Path(identity.path)
        try:
            current = path.stat(follow_symlinks=False)
        except OSError as error:
            raise E2EError("SOURCE_CHANGED", "source disappeared", path=identity.path) from error
        actual = (current.st_size, current.st_mtime_ns, current.st_dev, current.st_ino)
        expected = (
            identity.size_bytes,
            identity.mtime_ns,
            identity.device,
            identity.inode,
        )
        if actual != expected:
            raise E2EError(
                "SOURCE_CHANGED",
                "source identity changed during execution",
                path=identity.path,
            )


_LOCAL_STAT_KEYS = {
    "device",
    "inode",
    "mtimens",
    "mtime_ns",
    "sourcedevice",
    "sourceinode",
}


def _share_safe_string(
    value: str,
    *,
    staging: PurePath,
    source_tokens: Mapping[str, str],
) -> str:
    """Redact host paths while retaining an auditable opaque source token."""

    result = value
    for private, public in sorted(source_tokens.items(), key=lambda item: -len(item[0])):
        result = result.replace(private, public)
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
    if isinstance(value, str):
        return _share_safe_string(value, staging=staging, source_tokens=source_tokens)
    if isinstance(value, list):
        return [
            _share_safe_value(item, staging=staging, source_tokens=source_tokens)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            _share_safe_string(key, staging=staging, source_tokens=source_tokens): _share_safe_value(
                item, staging=staging, source_tokens=source_tokens
            )
            for key, item in value.items()
            if key.casefold() not in _LOCAL_STAT_KEYS
        }
    return value


def _sanitize_shareable_tree(
    staging: Path, identities: Sequence[_SourceIdentity]
) -> None:
    """Rewrite every staged JSON document to a share-safe public form."""

    source_tokens = {
        item.path: f"source/{item.source_id}/{Path(item.path).name}"
        for item in identities
    }
    for path in sorted(staging.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise E2EError(
                "PUBLIC_RECEIPT_SANITIZE_FAILED", str(error), path=str(path)
            ) from error
        sanitized = _share_safe_value(
            value, staging=staging, source_tokens=source_tokens
        )
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


def _approval_config_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise E2EError("APPROVAL_CONTEXT_INVALID", f"{name} is non-finite")
        return value
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return _approval_config_value(asdict(value), name)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise E2EError(
                "APPROVAL_CONTEXT_INVALID", f"{name} contains a non-string key"
            )
        return {
            key: _approval_config_value(item, f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _approval_config_value(item, f"{name}[]") for item in value
        ]
    raise E2EError(
        "APPROVAL_CONTEXT_INVALID",
        f"{name} has unsupported type {type(value).__name__}",
    )


def _review_approval_request_digest(
    request: E2ERequest, identities: Sequence[_SourceIdentity]
) -> str:
    """Bind manual admission to every science-affecting request/input field."""

    payload = {
        "schemaVersion": 1,
        "sources": [
            identity.local_serializable()
            for identity in sorted(identities, key=lambda item: (item.role, item.path))
        ],
        "qualityControl": {
            "config": request.qc_config.serializable(),
            "gatePolicy": request.gate_policy.serializable(),
            "gatePolicyDigest": request.gate_policy.canonical_digest(),
            # The selection policy changes admission, so an unattended policy
            # is bound into the approval digest; the legacy digest is unchanged.
            **(
                {"selection": request.selection.serializable()}
                if request.selection.unattended
                else {}
            ),
        },
        "pipeline": request.pipeline_parameters.serializable(),
        "recipeDigest": request.recipe_digest,
        "workers": request.workers,
        "integrationMode": request.integration_mode.value,
        "drizzle": request.drizzle.serializable(),
        "solverQuality": {
            "raHintDegrees": request.ra_hint_degrees,
            "decHintDegrees": request.dec_hint_degrees,
            "fieldOfViewDegrees": request.field_of_view_degrees,
            "searchRadiusDegrees": request.search_radius_degrees,
            "minimumMatches": request.min_matches,
            "maximumRmsArcsec": request.max_rms_arcsec,
        },
        "registration": {
            "detection": _approval_config_value(
                request.registration_detection, "registrationDetection"
            ),
            "config": _approval_config_value(
                request.registration_config, "registrationConfig"
            ),
        },
    }
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def bind_review_approval_selections(
    request: E2ERequest, selections: Sequence[Mapping[str, str]]
) -> E2ERequest:
    """Turn GUI review choices into full approvals bound to the current request.

    The GUI supplies only the exact preflight Light digest and gate-policy
    digest.  This trusted runtime recomputes every source identity and the full
    science-request digest immediately before execution.  The normal E2E gate
    still requires that each selected source is uniquely present and remains
    REVIEW; PASS and HARD_FAIL can never be promoted.
    """

    if not selections:
        return request
    pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    policy_digest = request.gate_policy.canonical_digest()
    sources = (
        ("LIGHT", request.light_files),
        ("FLAT", request.flat_files),
        ("DARK", request.dark_files),
        ("BIAS", request.bias_files),
        ("MASTER_BIAS", request.master_bias_files),
        ("MASTER_DARK", request.master_dark_files),
        ("MASTER_FLAT", request.master_flat_files),
    )
    identities = tuple(
        _capture_source(Path(value).expanduser().resolve(strict=True), role)
        for role, values in sources
        for value in values
    )
    request_digest = _review_approval_request_digest(request, identities)
    approvals: list[ReviewApproval] = []
    seen: set[str] = set()
    for index, selection in enumerate(selections):
        if set(selection) != {"sourceSha256", "gatePolicyDigest"}:
            raise E2EError(
                "REVIEW_SELECTION_INVALID",
                f"review selection {index} has unsupported fields",
            )
        source_sha256 = selection.get("sourceSha256")
        selected_policy = selection.get("gatePolicyDigest")
        if (
            not isinstance(source_sha256, str)
            or pattern.fullmatch(source_sha256) is None
            or source_sha256 in seen
            or not isinstance(selected_policy, str)
            or pattern.fullmatch(selected_policy) is None
        ):
            raise E2EError(
                "REVIEW_SELECTION_INVALID",
                "review selections require unique lowercase source and policy SHA-256 values",
            )
        if selected_policy != policy_digest:
            raise E2EError(
                "REVIEW_APPROVAL_POLICY_DRIFT",
                "preflight approval policy differs from the current execution policy",
            )
        seen.add(source_sha256)
        approvals.append(
            ReviewApproval(
                source_sha256=source_sha256,
                gate_policy_digest=selected_policy,
                request_digest=request_digest,
            )
        )
    return replace(request, review_approvals=tuple(approvals))


def _apply_review_approvals(
    *,
    request: E2ERequest,
    identities: Sequence[_SourceIdentity],
    results: Sequence[FrameResult],
) -> tuple[set[str], str, list[dict[str, Any]]]:
    request_digest = _review_approval_request_digest(request, identities)
    policy_digest = request.gate_policy.canonical_digest()
    approvals: dict[str, ReviewApproval] = {}
    for approval in request.review_approvals:
        if approval.source_sha256 in approvals:
            raise E2EError(
                "REVIEW_APPROVAL_DUPLICATE",
                "a source SHA-256 appears in more than one REVIEW approval",
            )
        if approval.gate_policy_digest != policy_digest:
            raise E2EError(
                "REVIEW_APPROVAL_POLICY_DRIFT",
                "approval gate-policy digest differs from the executed policy",
            )
        if approval.request_digest != request_digest:
            raise E2EError(
                "REVIEW_APPROVAL_REQUEST_DRIFT",
                "approval request digest differs from the executed inputs or recipe",
            )
        approvals[approval.source_sha256] = approval

    light_identities = {
        identity.path: identity
        for identity in identities
        if identity.role == "LIGHT"
    }
    light_paths_by_sha: dict[str, list[str]] = {}
    for identity in light_identities.values():
        light_paths_by_sha.setdefault(identity.sha256, []).append(identity.path)
    ambiguous = sorted(
        digest
        for digest in approvals
        if len(light_paths_by_sha.get(digest, ())) != 1
    )
    if ambiguous:
        raise E2EError(
            "REVIEW_APPROVAL_SOURCE_AMBIGUOUS",
            "approval SHA-256 must identify exactly one current Light: "
            + ", ".join(ambiguous),
        )
    admitted: set[str] = set()
    evidence: list[dict[str, Any]] = []
    matched_approvals: set[str] = set()
    for result in results:
        canonical = str(Path(result.path).resolve(strict=True))
        identity = light_identities.get(canonical)
        if identity is None or result.quality_gate is None:
            continue
        approval = approvals.get(identity.sha256)
        if approval is None:
            continue
        if result.quality_gate.disposition is not GateDisposition.REVIEW:
            raise E2EError(
                "REVIEW_APPROVAL_SOURCE_NOT_REVIEW",
                "manual approval may admit REVIEW only, never PASS or HARD_FAIL",
                path=canonical,
            )
        if not result.registration.ok:
            # No transform in the quality pass: the registration would fail on
            # this frame and take the whole run with it.  Refuse now instead.
            raise E2EError(
                "REVIEW_APPROVAL_UNREGISTRABLE",
                "an approved REVIEW frame could not be registered in the quality pass",
                path=canonical,
            )
        matched_approvals.add(identity.sha256)
        admitted.add(canonical)
        evidence.append(
            {
                **approval.serializable(),
                "path": canonical,
                "gateDisposition": GateDisposition.REVIEW.value,
                "admitted": True,
            }
        )
    unmatched = sorted(set(approvals) - matched_approvals)
    if unmatched:
        raise E2EError(
            "REVIEW_APPROVAL_SOURCE_UNKNOWN",
            "approval SHA-256 does not uniquely identify a current REVIEW Light: "
            + ", ".join(unmatched),
        )
    return admitted, request_digest, evidence


def _apply_explicit_selection(
    *,
    request: E2ERequest,
    identities: Sequence[_SourceIdentity],
    results: Sequence[FrameResult],
    evidence: BlinkEvidence,
) -> tuple[set[str], dict[str, Any]]:
    """Admit exactly the Lights the selection keeps; record every override.

    The gate and the blink flags have run; they inform the record but never
    decide.  A KEEP on a frame without a registration transform fails closed
    (the run's registration would fail on it); a KEEP on a HARD_FAIL or on
    an EXCLUDE-flagged frame is honoured and counted as an override.
    """

    selection = request.explicit_selection
    assert selection is not None
    light_identities = {
        identity.path: identity for identity in identities if identity.role == "LIGHT"
    }
    paths_by_sha: dict[str, list[str]] = {}
    for identity in light_identities.values():
        paths_by_sha.setdefault(identity.sha256, []).append(identity.path)
    # Source identities carry the same ``sha256:`` form as the selection.
    decisions = dict(selection.by_source)
    ambiguous = sorted(digest for digest in decisions if len(paths_by_sha.get(digest, ())) > 1)
    if ambiguous:
        raise E2EError(
            "SELECTION_SOURCE_AMBIGUOUS",
            "a selection digest names more than one current Light: " + ", ".join(ambiguous),
        )
    unknown = sorted(digest for digest in decisions if digest not in paths_by_sha)
    if unknown:
        raise E2EError(
            "SELECTION_SOURCE_UNKNOWN",
            "a selection digest names no current Light: " + ", ".join(unknown),
        )
    flags_by_path = evidence.flags_by_path
    admitted: set[str] = set()
    frames: list[dict[str, Any]] = []
    counts = {
        "keep": 0,
        "drop": 0,
        "overriddenExcludeFlags": 0,
        "overriddenGateHardFail": 0,
        "undecided": 0,
    }
    undecided_paths: list[str] = []
    for result in sorted(results, key=lambda item: item.path):
        canonical = str(Path(result.path).resolve(strict=True))
        identity = light_identities.get(canonical)
        if identity is None:
            continue
        decision = decisions.get(identity.sha256)
        undecided = decision is None
        verdict = selection.undecided if undecided else decision.decision
        if undecided:
            if verdict == "ERROR":
                undecided_paths.append(canonical)
                continue
            counts["undecided"] += 1
        record = flags_by_path.get(result.path)
        flag_codes = list(record.codes) if record is not None else []
        default_decision = record.default_decision if record is not None else None
        gate = result.quality_gate
        gate_disposition = gate.disposition.value if gate is not None else None
        overrode: list[str] = []
        if verdict == "KEEP":
            if not result.registration.ok:
                raise E2EError(
                    "SELECTION_UNREGISTRABLE",
                    "a kept Light could not be registered in the quality pass",
                    path=canonical,
                )
            if record is not None and record.exclude:
                counts["overriddenExcludeFlags"] += 1
                overrode.append("EXCLUDE_FLAGS")
            if gate is not None and gate.disposition is GateDisposition.HARD_FAIL:
                counts["overriddenGateHardFail"] += 1
                overrode.append("GATE_HARD_FAIL")
            counts["keep"] += 1
            admitted.add(canonical)
        else:
            counts["drop"] += 1
        frames.append(
            {
                "sourceSha256": identity.sha256,
                "path": canonical,
                "decision": verdict,
                "undecided": undecided,
                "defaultDecision": default_decision,
                "flags": flag_codes,
                "gateDisposition": gate_disposition,
                "overrode": overrode,
                **({"note": decision.note} if decision is not None and decision.note else {}),
            }
        )
    if undecided_paths:
        raise E2EError(
            "SELECTION_INCOMPLETE",
            f"{len(undecided_paths)} Light(s) have no decision and the selection says undecided=ERROR",
            path=undecided_paths[0],
        )
    block = {
        "policy": SELECTION_POLICY,
        "selectionDigest": selection.digest,
        "origin": dict(selection.origin) if selection.origin is not None else None,
        "undecided": selection.undecided,
        "flagsPolicyDigest": evidence.flags_policy.canonical_digest(),
        "counts": counts,
        "frames": frames,
    }
    return admitted, block


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise E2EError("OUTPUT_EXISTS", "refusing to replace JSON artifact", path=str(path)) from error


def _canonical_inputs(values: Iterable[str], role: str, *, required: bool) -> tuple[Path, ...]:
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise E2EError("INPUT_PATH_INVALID", f"{role} paths must be non-empty strings")
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except OSError as error:
            raise E2EError("INPUT_MISSING", f"{role} input cannot be resolved", path=value) from error
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except OSError as error:
            raise E2EError("INPUT_STAT_FAILED", str(error), path=str(path)) from error
        if not stat.S_ISREG(mode):
            raise E2EError("INPUT_NOT_REGULAR", f"{role} input is not a regular file", path=str(path))
        key = os.path.normcase(str(path))
        if key in seen:
            raise E2EError("DUPLICATE_INPUT", f"duplicate {role} input", path=str(path))
        seen.add(key)
        paths.append(path)
    if required and not paths:
        raise E2EError("INPUT_GROUP_EMPTY", f"at least one {role} frame is required")
    return tuple(sorted(paths, key=lambda item: os.path.normcase(str(item))))


def _capture_sources(
    flattened: Sequence[tuple[str, Path]], *, workers: int = 1
) -> tuple[_SourceIdentity, ...]:
    """Hash every source once, several files at a time; order is preserved."""

    if not flattened:
        return ()
    count = max(1, min(int(workers), len(flattened), 8))
    if count == 1:
        return tuple(_capture_source(path, role) for role, path in flattened)
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="oaf-inventory") as pool:
        return tuple(pool.map(lambda item: _capture_source(item[1], item[0]), flattened))


def _capture_source(path: Path, role: str) -> _SourceIdentity:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise E2EError("INPUT_NOT_REGULAR", "input is not a regular file", path=str(path))
    digest = _sha256(path)
    after = path.stat(follow_symlinks=False)
    before_tuple = (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
    after_tuple = (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino)
    if before_tuple != after_tuple:
        raise E2EError("SOURCE_CHANGED", "source changed while inventorying", path=str(path))
    return _SourceIdentity(
        str(path),
        role,
        digest,
        after.st_size,
        after.st_mtime_ns,
        after.st_dev,
        after.st_ino,
    )


def _trusted_source_identity_bindings(
    identities: Mapping[str, _SourceIdentity],
    paths: Sequence[Path],
) -> dict[str, _InternalSourceIdentity]:
    result: dict[str, _InternalSourceIdentity] = {}
    for path in paths:
        canonical = path.expanduser().resolve(strict=True)
        identity = identities.get(str(canonical))
        if identity is None:
            raise E2EError(
                "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                "upstream source identity is missing from the generated-master handoff",
                path=str(canonical),
            )
        current = canonical.stat(follow_symlinks=False)
        actual_stat = (
            current.st_size,
            current.st_mtime_ns,
            current.st_dev,
            current.st_ino,
        )
        expected_stat = (
            identity.size_bytes,
            identity.mtime_ns,
            identity.device,
            identity.inode,
        )
        # The private pixel snapshot was copied through one open handle and its
        # digest was checked against this captured identity before this handoff.
        # Rechecking stat here detects path replacement; the final publication
        # gate performs the deliberate second full hash of every original.
        if actual_stat != expected_stat:
            raise E2EError(
                "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                "source stat identity changed before the generated-master trust handoff",
                path=str(canonical),
            )
        result[identity.path] = _InternalSourceIdentity(
            path=identity.path,
            sha256=identity.sha256,
            size_bytes=identity.size_bytes,
            mtime_ns=identity.mtime_ns,
            device=identity.device,
            inode=identity.inode,
        )
    if len(result) != len(paths):
        raise E2EError(
            "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
            "generated-master source bindings do not match the selected input set",
        )
    return result


def _verify_source(identity: _SourceIdentity) -> None:
    path = Path(identity.path)
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as error:
        raise E2EError("SOURCE_CHANGED", "source disappeared", path=identity.path) from error
    actual = (current.st_size, current.st_mtime_ns, current.st_dev, current.st_ino)
    expected = (
        identity.size_bytes,
        identity.mtime_ns,
        identity.device,
        identity.inode,
    )
    if actual != expected or _sha256(path) != identity.sha256:
        raise E2EError("SOURCE_CHANGED", "source identity changed during execution", path=identity.path)


def _verify_sources(identities: Sequence[_SourceIdentity], *, workers: int = 4) -> None:
    """Rehash every original; files are checked concurrently, failures surface as one."""

    count = max(1, min(int(workers), len(identities), 8))
    if count <= 1:
        for identity in identities:
            _verify_source(identity)
        return
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="oaf-verify") as pool:
        list(pool.map(_verify_source, identities))


def _safe_token(value: str) -> str:
    token = name_token(value)
    if not token:
        raise E2EError("FILTER_INVALID", "filter name cannot be encoded safely")
    return token


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Create-only directory publication through the platform service layer."""

    try:
        platform_services.current().rename_directory_no_replace(source, destination)
    except NoReplaceError as error:
        if error.code == "OUTPUT_EXISTS":
            message = (
                "refusing to replace output directory"
                if error.precheck
                else "output appeared during publication"
            )
            raise E2EError("OUTPUT_EXISTS", message, path=str(destination)) from error
        raise E2EError(error.code, error.message) from error


def _fsync_directory(path: Path) -> None:
    platform_services.current().fsync_directory(path)


def _validate_request(request: E2ERequest) -> None:
    if not isinstance(request, E2ERequest):
        raise E2EError("REQUEST_INVALID", "request has the wrong type")
    if isinstance(request.workers, bool) or not isinstance(request.workers, int) or request.workers < 1:
        raise E2EError("WORKER_COUNT_INVALID", "workers must be a positive integer")
    request.qc_config.validate()
    request.gate_policy.validate()
    request.pipeline_parameters.validate()
    request.drizzle.validate()
    digest_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    if request.recipe_digest is not None and (
        not isinstance(request.recipe_digest, str)
        or digest_pattern.fullmatch(request.recipe_digest) is None
    ):
        raise E2EError(
            "RECIPE_DIGEST_INVALID", "recipe_digest must be a lowercase sha256: digest"
        )
    for approval in request.review_approvals:
        if not isinstance(approval, ReviewApproval):
            raise E2EError(
                "REVIEW_APPROVAL_INVALID", "review approvals have the wrong type"
            )
        for name, value in approval.serializable().items():
            if not isinstance(value, str) or digest_pattern.fullmatch(value) is None:
                raise E2EError(
                    "REVIEW_APPROVAL_INVALID",
                    f"{name} must be a lowercase sha256: digest",
                )
    if not isinstance(request.integration_mode, IntegrationMode):
        raise E2EError("INTEGRATION_MODE_INVALID", "unknown integration mode")
    request.selection.validate()
    request.blink_flags_policy.validate()
    if request.explicit_selection is not None and not isinstance(
        request.explicit_selection, ExplicitSelection
    ):
        raise E2EError("SELECTION_INVALID", "explicit selection has the wrong type")
    if request.selection.explicit != (request.explicit_selection is not None):
        raise E2EError(
            "SELECTION_POLICY_CONFLICT",
            "selection policy explicit-v1 and a supplied selection go together",
        )
    if request.explicit_selection is not None and request.review_approvals:
        raise E2EError(
            "SELECTION_POLICY_CONFLICT",
            "an explicit selection cannot be combined with REVIEW approvals",
        )
    numeric_hints = {
        "ra_hint_degrees": request.ra_hint_degrees,
        "dec_hint_degrees": request.dec_hint_degrees,
        "field_of_view_degrees": request.field_of_view_degrees,
        "search_radius_degrees": request.search_radius_degrees,
    }
    for name, value in numeric_hints.items():
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise E2EError("SOLVER_HINT_INVALID", f"{name} must be finite")
    if (request.ra_hint_degrees is None) != (request.dec_hint_degrees is None):
        raise E2EError("SOLVER_HINT_INCOMPLETE", "RA and Dec hints must be supplied together")
    if request.ra_hint_degrees is not None and not 0 <= request.ra_hint_degrees < 360:
        raise E2EError("SOLVER_HINT_INVALID", "RA hint must be in [0, 360)")
    if request.dec_hint_degrees is not None and not -90 <= request.dec_hint_degrees <= 90:
        raise E2EError("SOLVER_HINT_INVALID", "Dec hint must be in [-90, 90]")
    for name in ("field_of_view_degrees", "search_radius_degrees"):
        value = getattr(request, name)
        if value is not None and value <= 0:
            raise E2EError("SOLVER_HINT_INVALID", f"{name} must be positive")
    if (
        isinstance(request.min_matches, bool)
        or not isinstance(request.min_matches, int)
        or request.min_matches < 12
    ):
        raise E2EError(
            "SOLVER_QUALITY_POLICY_INVALID",
            "min_matches must be an integer of at least 12",
        )
    if (
        isinstance(request.max_rms_arcsec, bool)
        or not isinstance(request.max_rms_arcsec, (int, float))
        or not math.isfinite(float(request.max_rms_arcsec))
        or float(request.max_rms_arcsec) <= 0
        or float(request.max_rms_arcsec) > 2.0
    ):
        raise E2EError(
            "SOLVER_QUALITY_POLICY_INVALID",
            "max_rms_arcsec must be finite, positive, and no greater than 2.0",
        )
    tolerance = request.same_grid_wcs_tolerance_pixels
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0
        or float(tolerance) > 5.0
    ):
        raise E2EError(
            "SOLVER_QUALITY_POLICY_INVALID",
            "same_grid_wcs_tolerance_pixels must be finite, positive, and no greater than 5.0",
        )


def _write_review_previews(
    staging: Path, qc_dir: Path, results: Sequence[FrameResult]
) -> dict[str, str]:
    """Bounded previews of every frame the gate did not pass, keyed by path.

    The desktop shows them with the run's result so an excluded frame can be
    judged without opening the raw file; values are paths relative to the run.
    """

    previews: dict[str, str] = {}
    total = 0
    for index, result in enumerate(results):
        gate = result.quality_gate
        if gate is None or gate.disposition is GateDisposition.PASS or result.thumbnail_path is None:
            continue
        if len(previews) >= MAX_REVIEW_PREVIEWS:
            break
        data = bounded_review_preview(result.thumbnail_path)
        if data is None or total + len(data) > MAX_TOTAL_PREVIEW_BYTES:
            continue
        destination = qc_dir / REVIEW_DIRECTORY / review_preview_name(index, result.path)
        destination.parent.mkdir(exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        total += len(data)
        previews[result.path] = destination.relative_to(staging).as_posix()
    return previews


def _screening_summary(
    results: Sequence[FrameResult],
    passed: Sequence[Path],
    approved_review_paths: Sequence[str],
    review_previews: Mapping[str, str],
    explicit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Counts plus the frames that needed a decision, for receipts and the desktop.

    With an explicit selection every user drop is listed (``reason``
    ``USER_DROP`` plus its blink flag codes) and so is every keep that
    overrode an EXCLUDE flag (``USER_KEEP_OVERRIDE``), next to the gate's own
    non-PASS frames, so the result page shows both kinds of decision.
    """

    counts = {disposition.value: 0 for disposition in GateDisposition}
    frames: list[dict[str, Any]] = []
    passed_set = {str(path) for path in passed}
    explicit_by_path = {
        item["path"]: item for item in (explicit or {}).get("frames", []) if isinstance(item, Mapping)
    }
    for result in sorted(results, key=lambda item: item.path):
        gate = result.quality_gate
        disposition = gate.disposition.value if gate is not None else "HARD_FAIL"
        counts[disposition] += 1
        resolved = str(Path(result.path).resolve(strict=True))
        decision = explicit_by_path.get(resolved)
        reason: str | None = None
        if decision is not None:
            if decision["decision"] == "DROP":
                reason = "USER_DROP"
            elif "EXCLUDE_FLAGS" in decision.get("overrode", ()):
                reason = "USER_KEEP_OVERRIDE"
        if gate is not None and gate.disposition is GateDisposition.PASS and reason is None:
            continue
        frames.append(
            {
                "path": resolved,
                "disposition": disposition,
                "admitted": resolved in passed_set or resolved in approved_review_paths,
                "summary": gate.summary if gate is not None else "; ".join(result.reasons) or "not measured",
                "evidence": [item.message for item in gate.evidence][:8] if gate is not None else list(result.reasons)[:8],
                "starCount": result.star_count,
                "reviewPreview": review_previews.get(result.path),
                **({"reason": reason, "flags": list(decision["flags"])} if reason is not None and decision is not None else {}),
            }
        )
    return {
        "counts": counts,
        "admitted": len(passed_set),
        "excluded": len(results) - len(passed_set),
        "frames": frames,
    }


def _qc_manifest(
    staging: Path,
    groups: Sequence[Mapping[str, Any]],
    results: Sequence[FrameResult],
    config: QcConfig,
    policy: GatePolicy,
    review_previews: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    frame_payloads: list[dict[str, Any]] = []
    for result in results:
        payload = result.serializable()
        thumbnail = payload.get("thumbnailPath")
        if isinstance(thumbnail, str):
            try:
                payload["thumbnailPath"] = Path(thumbnail).relative_to(staging).as_posix()
            except ValueError:
                payload["thumbnailPath"] = None
        payload["reviewPreviewPath"] = (review_previews or {}).get(result.path)
        frame_payloads.append(payload)
    counts = {
        disposition.value: sum(
            result.quality_gate is not None
            and result.quality_gate.disposition is disposition
            for result in results
        )
        for disposition in GateDisposition
    }
    return {
        "schemaVersion": 1,
        "stage": "quality-control",
        "gatePolicy": policy.serializable(),
        "gatePolicyDigest": policy.canonical_digest(),
        "qcConfig": config.serializable(),
        "counts": counts,
        "groups": list(groups),
        "frames": frame_payloads,
    }


def _inferred_solver_hints(
    request: E2ERequest,
    passed_results: Sequence[FrameResult],
) -> _SolverHints:
    explicit_coordinates = request.ra_hint_degrees is not None
    coordinate_pairs = [
        (result.metadata.ra_degrees, result.metadata.dec_degrees)
        for result in passed_results
        if result.metadata.ra_degrees is not None
        and result.metadata.dec_degrees is not None
        and math.isfinite(result.metadata.ra_degrees)
        and math.isfinite(result.metadata.dec_degrees)
    ]
    metadata_center: tuple[float, float] | None = None
    coordinate_spread_degrees: float | None = None
    if coordinate_pairs:
        ra_radians = np.radians([item[0] for item in coordinate_pairs])
        metadata_ra = float(
            np.degrees(
                math.atan2(float(np.mean(np.sin(ra_radians))), float(np.mean(np.cos(ra_radians))))
            )
            % 360.0
        )
        metadata_dec = float(np.median([item[1] for item in coordinate_pairs]))
        metadata_center = (metadata_ra, metadata_dec)
        coordinate_spread_degrees = max(
            _sky_separation_degrees(metadata_center, item) for item in coordinate_pairs
        )

    coordinate_evidence: dict[str, Any] = {
        "explicit": (
            {
                "raDegrees": request.ra_hint_degrees,
                "decDegrees": request.dec_hint_degrees,
            }
            if explicit_coordinates
            else None
        ),
        "metadata": (
            {
                "raDegrees": metadata_center[0],
                "decDegrees": metadata_center[1],
                "sampleCount": len(coordinate_pairs),
                "maximumSeparationFromConsensusDegrees": coordinate_spread_degrees,
            }
            if metadata_center is not None
            else None
        ),
        "consistency": "NOT_COMPARABLE",
    }
    if explicit_coordinates and metadata_center is not None:
        assert request.ra_hint_degrees is not None and request.dec_hint_degrees is not None
        explicit_to_metadata = _sky_separation_degrees(
            (request.ra_hint_degrees, request.dec_hint_degrees), metadata_center
        )
        consistency_radius = request.search_radius_degrees or 15.0
        coordinate_evidence["explicitToMetadataDegrees"] = explicit_to_metadata
        coordinate_evidence["consistencyRadiusDegrees"] = consistency_radius
        if explicit_to_metadata <= consistency_radius:
            ra = request.ra_hint_degrees
            dec = request.dec_hint_degrees
            provenance = "explicit+NINA/FITS-consistent"
            coordinate_evidence["consistency"] = "CONSISTENT"
            coordinate_evidence["selection"] = "explicit"
        else:
            # Conflicting pointing hints are never allowed to lock the solver.
            # A consensus from the actual acquired frames is a better seed; the
            # astrometry.net adapter still has an unconstrained fallback.
            ra, dec = metadata_center
            provenance = "NINA/FITS-header;conflicting-explicit-coordinate-ignored"
            coordinate_evidence["consistency"] = "CONFLICT"
            coordinate_evidence["selection"] = "metadata-consensus"
    elif explicit_coordinates:
        ra = request.ra_hint_degrees
        dec = request.dec_hint_degrees
        provenance = "explicit"
        coordinate_evidence["selection"] = "explicit"
    elif metadata_center is not None:
        ra, dec = metadata_center
        provenance = "NINA/FITS-header"
        coordinate_evidence["selection"] = "metadata-consensus"
    else:
        ra = None
        dec = None
        provenance = "blind"
        coordinate_evidence["selection"] = "blind"

    fov_samples: list[dict[str, Any]] = []
    for result in passed_results:
        header = result.metadata.header
        try:
            focal_length = float(
                header.get(
                    "FOCALLEN",
                    header.get("FOCAL", header.get("FOCALLENGTH", 0.0)),
                )
            )
            pixel_size_x_um = float(
                header.get(
                    "XPIXSZ",
                    header.get("PIXSIZE1", header.get("PIXSIZE", 0.0)),
                )
            )
            pixel_size_y_um = float(
                header.get(
                    "YPIXSZ",
                    header.get("PIXSIZE2", header.get("PIXSIZE", pixel_size_x_um)),
                )
            )
        except (TypeError, ValueError):
            continue
        width = result.metadata.width
        height = result.metadata.height
        values = (focal_length, pixel_size_x_um, pixel_size_y_um)
        if (
            focal_length <= 0
            or pixel_size_x_um <= 0
            or pixel_size_y_um <= 0
            or width <= 0
            or height <= 0
            or not all(math.isfinite(value) for value in values)
        ):
            continue
        width_degrees = math.degrees(
            2.0 * math.atan((width * pixel_size_x_um / 1000.0) / (2.0 * focal_length))
        )
        height_degrees = math.degrees(
            2.0 * math.atan((height * pixel_size_y_um / 1000.0) / (2.0 * focal_length))
        )
        if not (
            math.isfinite(width_degrees)
            and math.isfinite(height_degrees)
            and 0 < width_degrees <= 180
            and 0 < height_degrees <= 180
        ):
            continue
        fov_samples.append(
            {
                "widthDegrees": width_degrees,
                "heightDegrees": height_degrees,
                "widthPixels": width,
                "heightPixels": height,
                "focalLengthMm": focal_length,
                "pixelSizeXMicrons": pixel_size_x_um,
                "pixelSizeYMicrons": pixel_size_y_um,
            }
        )

    derived_fov: float | None = None
    fov_spread_ratio: float | None = None
    if fov_samples:
        width_estimates = np.asarray(
            [sample["widthDegrees"] for sample in fov_samples], dtype=np.float64
        )
        median_width = float(np.median(width_estimates))
        consistent = width_estimates[
            (width_estimates >= median_width * 0.8)
            & (width_estimates <= median_width * 1.2)
        ]
        if consistent.size:
            derived_fov = float(np.median(consistent))
            fov_spread_ratio = float(np.max(consistent) / np.min(consistent))

    explicit_fov = request.field_of_view_degrees
    fov_evidence: dict[str, Any] = {
        "explicitWidthDegrees": explicit_fov,
        "derivedWidthDegrees": derived_fov,
        "derivedSampleCount": len(fov_samples),
        "derivedSpreadRatio": fov_spread_ratio,
        "derivation": "2*atan((imagePixels*effectivePixelMicrons/1000)/(2*focalLengthMm))",
        "binningAssumption": "FITS XPIXSZ/YPIXSZ describe effective image pixels",
        "consistency": "NOT_COMPARABLE",
    }
    if explicit_fov is not None and derived_fov is not None:
        explicit_to_derived_ratio = explicit_fov / derived_fov
        fov_evidence["explicitToDerivedRatio"] = explicit_to_derived_ratio
        if 0.7 <= explicit_to_derived_ratio <= 1.3:
            field_of_view = explicit_fov
            fov_evidence["consistency"] = "CONSISTENT"
            fov_evidence["selection"] = "explicit"
        else:
            # A wrong scale hint is much more damaging than a missing one.  Use
            # image geometry and acquisition optics, and retain the conflict as
            # auditable evidence rather than constraining solve-field to it.
            field_of_view = derived_fov
            provenance += "+derived-FOV;conflicting-explicit-FOV-ignored"
            fov_evidence["consistency"] = "CONFLICT"
            fov_evidence["selection"] = "derived"
    elif explicit_fov is not None:
        field_of_view = explicit_fov
        fov_evidence["selection"] = "explicit-unverified"
    elif derived_fov is not None:
        field_of_view = derived_fov
        provenance += "+derived-FOV"
        fov_evidence["selection"] = "derived"
    else:
        field_of_view = None
        fov_evidence["selection"] = "unconstrained"
    return _SolverHints(
        ra,
        dec,
        field_of_view,
        request.search_radius_degrees,
        provenance,
        {
            "coordinates": coordinate_evidence,
            "fieldOfView": fov_evidence,
            "fallback": "astrometry.net removes all hints on its bounded fallback attempt",
        },
    )


def _drizzle_sampling_evidence(
    results: Sequence[FrameResult], options: DrizzleOptions
) -> dict[str, Any]:
    """Turn QC morphology into a fail-closed drizzle sampling decision."""

    direct_fwhm: list[float] = []
    hfr_fwhm: list[float] = []
    pixel_scales: list[float] = []
    for result in results:
        fwhm = result.features.median_fwhm_native_pixels
        if fwhm is not None and math.isfinite(fwhm) and fwhm > 0:
            direct_fwhm.append(float(fwhm))
        hfr = result.features.nina_hfr_pixels
        if hfr is not None and math.isfinite(hfr) and hfr > 0:
            # For a circular Gaussian, FWHM is exactly twice the half-flux
            # radius.  This is fallback evidence only; measured PSF FWHM wins.
            hfr_fwhm.append(float(2.0 * hfr))
        header = result.metadata.header
        try:
            focal_length_mm = float(header.get("FOCALLEN", header.get("FOCAL")))
            pixel_size_um = float(
                header.get("XPIXSZ", header.get("PIXSIZE1", header.get("PIXSIZE")))
            )
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(focal_length_mm)
            and focal_length_mm > 0
            and math.isfinite(pixel_size_um)
            and pixel_size_um > 0
        ):
            pixel_scales.append(206.265 * pixel_size_um / focal_length_mm)

    direct_median = float(np.median(direct_fwhm)) if direct_fwhm else None
    hfr_median = float(np.median(hfr_fwhm)) if hfr_fwhm else None
    if direct_median is not None and hfr_median is not None:
        fwhm_values = [direct_median, hfr_median]
        provenance = "QC_NATIVE_PSF_FWHM+NINA_HFR_GAUSSIAN_EQUIVALENT"
    elif direct_median is not None:
        fwhm_values = [direct_median]
        provenance = "QC_NATIVE_PSF_FWHM"
    elif hfr_median is not None:
        fwhm_values = [hfr_median]
        provenance = "NINA_HFR_GAUSSIAN_EQUIVALENT"
    else:
        fwhm_values = []
        provenance = "UNKNOWN"
    median_fwhm = float(np.median(fwhm_values)) if fwhm_values else None
    pixel_scale = float(np.median(pixel_scales)) if pixel_scales else None
    evidence: dict[str, Any] = {
        "medianNativeFwhmPixels": median_fwhm,
        "qcMedianNativeFwhmPixels": direct_median,
        "ninaHfrEquivalentFwhmPixels": hfr_median,
        "pixelScaleArcsec": pixel_scale,
        "seeingFwhmArcsec": (
            median_fwhm * pixel_scale
            if median_fwhm is not None and pixel_scale is not None
            else None
        ),
        "sampleCount": max(len(direct_fwhm), len(hfr_fwhm)),
        "qcFwhmSampleCount": len(direct_fwhm),
        "ninaHfrSampleCount": len(hfr_fwhm),
        "provenance": provenance,
        "maximumFwhmForUpsamplingPixels": options.maximum_fwhm_for_upsampling_pixels,
    }
    # The sampling evidence is advisory, as in WBPP: the user chose the scale;
    # the receipt records whether upsampling is expected to gain resolution.
    evidence["advisory"] = True
    if options.scale == 1:
        evidence["status"] = "NOT_APPLICABLE"
        return evidence
    if median_fwhm is None:
        evidence["status"] = "UNKNOWN_SAMPLING"
        evidence["recommendation"] = (
            "QC could not establish the native PSF sampling; the benefit of "
            f"{options.scale}x drizzle is unknown"
        )
        return evidence
    if (
        direct_median is not None
        and hfr_median is not None
        and (direct_median >= options.maximum_fwhm_for_upsampling_pixels)
        != (hfr_median >= options.maximum_fwhm_for_upsampling_pixels)
    ):
        evidence["status"] = "CONFLICTING_SAMPLING"
        evidence["recommendation"] = (
            "QC PSF FWHM and NINA HFR disagree across the adequately-sampled boundary"
        )
        return evidence
    if median_fwhm >= options.maximum_fwhm_for_upsampling_pixels:
        evidence["status"] = "WELL_SAMPLED"
        evidence["recommendation"] = (
            f"QC median native FWHM is {median_fwhm:.3f} px, at or above the "
            f"{options.maximum_fwhm_for_upsampling_pixels:.3f} px threshold: "
            f"{options.scale}x drizzle mainly gains sub-pixel sampling, not resolution"
        )
        return evidence
    evidence["status"] = "PASS_UNDERSAMPLED"
    return evidence


def _build_registration_masters(
    *,
    biases: tuple[Path, ...],
    darks: tuple[Path, ...],
    flats: tuple[Path, ...],
    supplied_biases: tuple[Path, ...],
    supplied_darks: tuple[Path, ...],
    supplied_flats: tuple[Path, ...],
    lights: tuple[Path, ...],
    directory: Path,
    pipeline_parameters: PipelineParameters,
    source_aliases: Mapping[str, Path] | None = None,
    source_identities: Mapping[str, _SourceIdentity] | None = None,
    xisf_conversions: Sequence[Mapping[str, Any]] = (),
) -> tuple[Any, dict[str, Any]]:
    try:
        from openastroflow_registration import CalibrationPlan
    except ImportError as error:
        raise E2EError(
            "REGISTRATION_BACKEND_UNAVAILABLE",
            "openastroflow-registration must be installed for E2E execution",
        ) from error

    parameters = pipeline_parameters.integration
    directory.mkdir(parents=True, exist_ok=False)
    def frame_info(path: Path) -> FrameInfo:
        identity_path = (source_aliases or {}).get(str(path), path)
        source_identity = (source_identities or {}).get(
            str(identity_path.expanduser().resolve(strict=True))
        )
        return _input_frame_info(
            path,
            pipeline_parameters,
            override_identity_path=identity_path,
            override_source_identity=source_identity,
        )

    grouped_roles = (
        ("BIAS", biases),
        ("DARK", darks),
        ("FLAT", flats),
        ("MASTER_BIAS", supplied_biases),
        ("MASTER_DARK", supplied_darks),
        ("MASTER_FLAT", supplied_flats),
        ("LIGHT", lights),
    )
    for role, paths in grouped_roles:
        for path in paths:
            actual = normalize_role(frame_info(path).role)
            if actual != role:
                raise E2EError(
                    "FRAME_ROLE_MISMATCH",
                    f"expected {role}, found {actual}",
                    path=str(path),
                )

    if len(supplied_biases) > 1 or (biases and supplied_biases) or (not biases and not supplied_biases and pipeline_parameters.calibration_workflow != MONO_STANDARD):
        raise E2EError(
            "BIAS_SOURCE_AMBIGUOUS",
            "registration calibration requires raw Biases or one MasterBias",
        )
    bias_infos = {path: frame_info(path) for path in biases}
    supplied_bias_infos = {path: frame_info(path) for path in supplied_biases}
    light_infos = {path: frame_info(path) for path in lights}
    dark_infos = {path: frame_info(path) for path in darks}
    supplied_dark_infos = {path: frame_info(path) for path in supplied_darks}
    flat_infos = {path: frame_info(path) for path in flats}
    supplied_flat_infos = {path: frame_info(path) for path in supplied_flats}
    (
        supplied_bias_infos,
        supplied_dark_infos,
        supplied_flat_infos,
    ) = _apply_master_metadata_overrides(
        (supplied_bias_infos, supplied_dark_infos, supplied_flat_infos),
        pipeline_parameters.master_metadata_overrides,
        dict(source_aliases or {}),
    )
    supplied_dark_bias_included = _master_dark_bias_semantics(
        supplied_darks,
        pipeline_parameters.master_metadata_overrides,
        dict(source_aliases or {}),
        workflow=pipeline_parameters.calibration_workflow,
    )
    all_infos = [*bias_infos.values(), *supplied_bias_infos.values(), *dark_infos.values(), *supplied_dark_infos.values(), *flat_infos.values(), *supplied_flat_infos.values(), *light_infos.values()]
    conflicts = conflicting_profile_fields(all_infos, pipeline_parameters.calibration_workflow)
    if conflicts:
        raise E2EError("CALIBRATION_PROFILE_MISMATCH", "Conflicting known acquisition metadata: " + ", ".join(conflicts))
    if not biases and not supplied_biases and not can_omit_bias(
        (*light_infos.values(), *flat_infos.values()),
        [*((info, True) for info in dark_infos.values()), *((info, supplied_dark_bias_included[path]) for path, info in supplied_dark_infos.items())],
        pipeline_parameters.calibration_workflow,
    ):
        raise E2EError("BIAS_REQUIRED_FOR_CALIBRATION", "Bias is required unless every Light and raw Flat has a matching Dark that includes Bias.")
    display = lambda path: (source_aliases or {}).get(str(path), path)  # noqa: E731
    reference_bias = (
        bias_infos[biases[0]] if biases else supplied_bias_infos[supplied_biases[0]] if supplied_biases else light_infos[lights[0]]
    )
    light_numeric_reference = next(iter(light_infos.values()))
    for info in light_infos.values():
        light_domain_scale = _numeric_application_scale(
            light_numeric_reference,
            info,
            target_label="registration Light domain",
            additive_label="raw Light",
        )
        if not math.isclose(light_domain_scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise E2EError(
                "REGISTRATION_LIGHT_NUMERIC_DOMAIN_MIXED",
                "registration requires one common Light numeric domain",
                path=info.path,
            )
    for info in (*bias_infos.values(), *light_infos.values()):
        _assert_compatible(reference_bias, info, workflow=pipeline_parameters.calibration_workflow)
    if biases:
        master_bias = directory / "master_bias.fits"
        bias_result = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=_numeric_application_scale(
                        reference_bias,
                        bias_infos[path],
                        target_label="MasterBias reference",
                        additive_label="raw Bias",
                    ),
                )
                for path in biases
            ),
            master_bias,
            metadata={
                "IMAGETYP": "Master Bias",
                "OAFSTATE": "UNSOLVED_WORKING",
                **cfa_metadata(reference_bias),
                **_numeric_domain_metadata(reference_bias),
            },
            parameters=parameters,
        )
        bias_record: dict[str, Any] = {
            "mode": "BUILT_FROM_RAW",
            "numericDomain": reference_bias.numeric_domain,
            "normalizedUnitScale": reference_bias.normalized_unit_scale,
            **bias_result.serializable(),
        }
    elif supplied_biases:
        master_bias = supplied_biases[0]
        bias_record = {
            "mode": "REUSED_SUPPLIED_MASTER",
            "path": str(display(master_bias)),
            "sha256": _sha256(display(master_bias)),
            "calibrationApplied": False,
            "numericDomain": reference_bias.numeric_domain,
            "normalizedUnitScale": reference_bias.normalized_unit_scale,
        }

    else:
        master_bias = None
        bias_record = {"mode": "NOT_REQUIRED_DARK_INCLUDES_BIAS"}

    dark_groups: dict[float, list[Path]] = {}
    for path in darks:
        exposure = dark_infos[path].exposure_seconds
        if exposure is None or exposure <= 0:
            raise E2EError("DARK_EXPOSURE_UNKNOWN", "Dark requires positive EXPTIME", path=str(path))
        dark_groups.setdefault(exposure, []).append(path)
    master_darks: dict[float, Path] = {}
    master_dark_domain_info: dict[float, FrameInfo] = {}
    dark_records: dict[str, Any] = {}
    for exposure, paths in sorted(dark_groups.items()):
        reference_dark = dark_infos[paths[0]]
        for path in paths:
            _assert_compatible(reference_bias, dark_infos[path], workflow=pipeline_parameters.calibration_workflow)
            _assert_compatible(
                reference_dark,
                dark_infos[path],
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=pipeline_parameters.dark_temperature_tolerance_celsius,
                workflow=pipeline_parameters.calibration_workflow,
            )
        destination = directory / f"master_dark_{format(exposure, '.9g').replace('.', 'p')}s.fits"
        result = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=_numeric_application_scale(
                        reference_dark,
                        dark_infos[path],
                        target_label="MasterDark reference",
                        additive_label="raw Dark",
                    ),
                )
                for path in paths
            ),
            destination,
            metadata={
                "IMAGETYP": "Master Dark",
                "EXPTIME": exposure,
                "OAFSTATE": "UNSOLVED_WORKING",
                "OAFBIAS": "INCLUDED",
                **cfa_metadata(reference_dark),
                **_numeric_domain_metadata(reference_dark),
            },
            parameters=parameters,
        )
        master_darks[exposure] = destination
        master_dark_domain_info[exposure] = reference_dark
        dark_records[format(exposure, ".9g")] = {
            "mode": "BUILT_FROM_RAW",
            "biasIncluded": True,
            "numericDomain": reference_dark.numeric_domain,
            "normalizedUnitScale": reference_dark.normalized_unit_scale,
            "applicationScaleToRawDarkReference": 1.0,
            **result.serializable(),
        }
    for path, info in supplied_dark_infos.items():
        exposure = info.exposure_seconds
        if exposure is None or exposure <= 0:
            raise E2EError(
                "DARK_EXPOSURE_UNKNOWN",
                "MasterDark requires positive EXPTIME",
                path=str(path),
            )
        if _find_dark(exposure, master_darks) is not None:
            raise E2EError(
                "DARK_SOURCE_AMBIGUOUS",
                "an exposure has both raw Darks and a supplied MasterDark",
                path=str(path),
            )
        master_darks[exposure] = path
        master_dark_domain_info[exposure] = info
        dark_records[format(exposure, ".9g")] = {
            "mode": "REUSED_SUPPLIED_MASTER",
            "path": str(display(path)),
            "sha256": _sha256(display(path)),
            "calibrationApplied": False,
            "biasIncluded": supplied_dark_bias_included[path],
            "numericDomain": info.numeric_domain,
            "normalizedUnitScale": info.normalized_unit_scale,
        }

    flat_groups: dict[str, list[Path]] = {}
    for path in flats:
        filter_name = _require_filter(flat_infos[path])
        flat_groups.setdefault(filter_name, []).append(path)
    supplied_flats_by_filter: dict[str, Path] = {}
    for path, info in supplied_flat_infos.items():
        filter_name = _require_filter(info)
        if filter_name in supplied_flats_by_filter:
            raise E2EError(
                "MASTER_FLAT_AMBIGUOUS",
                f"multiple supplied MasterFlats match filter {filter_name}",
            )
        if filter_name in flat_groups:
            raise E2EError(
                "FLAT_SOURCE_AMBIGUOUS",
                f"filter {filter_name} has raw Flats and a supplied MasterFlat",
            )
        supplied_flats_by_filter[filter_name] = path
    light_filters = {_require_filter(info) for info in light_infos.values()}
    missing = sorted(light_filters - set(flat_groups) - set(supplied_flats_by_filter))
    if missing:
        raise E2EError("MASTER_FLAT_MISSING", "no Flat group for: " + ", ".join(missing))

    master_flats: dict[str, Path] = {}
    flat_records: dict[str, Any] = {}
    for filter_name, paths in sorted(flat_groups.items()):
        expressions: list[FrameExpression] = []
        normalizations: list[float] = []
        calibration_sources: list[dict[str, Any]] = []
        for path in paths:
            _assert_compatible(reference_bias, flat_infos[path], workflow=pipeline_parameters.calibration_workflow)
            flat_info = flat_infos[path]
            flat_dark_match = _find_dark(flat_info.exposure_seconds, master_darks)
            if flat_dark_match is not None:
                dark_exposure, subtract_path = flat_dark_match
                dark_info = (
                    dark_infos[dark_groups[dark_exposure][0]]
                    if dark_exposure in dark_groups
                    else supplied_dark_infos[subtract_path]
                )
                _assert_compatible(
                    flat_info,
                    dark_info,
                    compare_exposure=True,
                    compare_temperature=True,
                    temperature_tolerance_celsius=pipeline_parameters.dark_temperature_tolerance_celsius,
                    workflow=pipeline_parameters.calibration_workflow,
                )
                dark_bias_included = (
                    True
                    if dark_exposure in dark_groups
                    else supplied_dark_bias_included[subtract_path]
                )
                calibration_mode = (
                    "MATCHED_BIAS_INCLUDED_DARK"
                    if dark_bias_included
                    else "MATCHED_BIAS_SUBTRACTED_DARK_PLUS_MASTER_BIAS"
                )
                flat_subtract_info = master_dark_domain_info[dark_exposure]
            else:
                subtract_path = master_bias
                dark_bias_included = True
                calibration_mode = "BIAS"
                flat_subtract_info = reference_bias
            flat_subtract_scale = _numeric_application_scale(
                flat_info,
                flat_subtract_info,
                target_label="raw Flat",
                additive_label=(
                    "MasterDark" if flat_dark_match is not None else "MasterBias"
                ),
            )
            flat_bias_scale = _numeric_application_scale(
                flat_info,
                reference_bias,
                target_label="raw Flat",
                additive_label="MasterBias",
            )
            expression = FrameExpression(
                str(path),
                subtract_path=str(subtract_path),
                subtract_scale=flat_subtract_scale,
                subtract_paths=(str(master_bias),) if not dark_bias_included else (),
                subtract_scales=(flat_bias_scale,) if not dark_bias_included else (),
            )
            location = robust_location(
                expression,
                max_samples=parameters.max_statistics_samples,
                division_floor=parameters.division_floor,
                max_memory_bytes=parameters.max_memory_bytes,
            )
            if not math.isfinite(location) or location <= parameters.division_floor:
                raise E2EError("FLAT_SIGNAL_INVALID", "Flat has no positive calibrated signal", path=str(path))
            normalizations.append(location)
            expressions.append(
                FrameExpression(
                    str(path),
                    subtract_path=str(subtract_path),
                    subtract_scale=flat_subtract_scale,
                    subtract_paths=(str(master_bias),) if not dark_bias_included else (),
                    subtract_scales=(flat_bias_scale,) if not dark_bias_included else (),
                    scale=1.0 / location,
                )
            )
            calibration_sources.append(
                {
                    "source": str(display(path)),
                    "mode": calibration_mode,
                    "subtracted": str(subtract_path),
                    "subtractedSha256": _sha256(subtract_path),
                    "targetNumericDomain": flat_info.numeric_domain,
                    "additiveNumericDomain": flat_subtract_info.numeric_domain,
                    "applicationScale": flat_subtract_scale,
                    "applicationScaleSource": "normalized-unit-domain-ratio",
                    "biasApplicationScale": (
                        flat_bias_scale if not dark_bias_included else None
                    ),
                }
            )
        destination = directory / f"master_flat_{_safe_token(filter_name)}.fits"
        result = integrate_expressions(
            expressions,
            destination,
            metadata={
                "IMAGETYP": "Master Flat",
                "FILTER": filter_name,
                "OAFSTATE": "UNSOLVED_WORKING",
                "OAFBIAS": "SUBTRACTED",
                "OAFNDOM": "DIMENSIONLESS_RESPONSE",
                "OAFNSCL": 1.0,
                **cfa_metadata(flat_infos[flat_groups[filter_name][0]]),
            },
            parameters=parameters,
        )
        master_flats[filter_name] = destination
        flat_records[filter_name] = {
            "mode": "BUILT_FROM_RAW",
            "applicationScale": 1.0,
            **result.serializable(),
            "normalizations": normalizations,
            "calibrationSources": calibration_sources,
        }
    for filter_name, path in sorted(supplied_flats_by_filter.items()):
        master_flats[filter_name] = path
        flat_records[filter_name] = {
            "mode": "REUSED_SUPPLIED_MASTER",
            "path": str(display(path)),
            "sha256": _sha256(display(path)),
            "calibrationApplied": False,
        }

    for path, light_info in light_infos.items():
        filter_name = _require_filter(light_info)
        flat_path = master_flats[filter_name]
        flat_info = (
            flat_infos[flat_groups[filter_name][0]]
            if filter_name in flat_groups
            else supplied_flat_infos[flat_path]
        )
        _assert_compatible(light_info, flat_info, compare_filter=True, workflow=pipeline_parameters.calibration_workflow)
        dark_match = _find_dark(light_info.exposure_seconds, master_darks)
        if master_darks and dark_match is None:
            raise E2EError(
                "DARK_EXPOSURE_MISMATCH",
                "no exact MasterDark matches this Light for registration calibration",
                path=str(path),
            )
        if dark_match is not None:
            exposure, dark_path = dark_match
            dark_info = (
                dark_infos[dark_groups[exposure][0]]
                if exposure in dark_groups
                else supplied_dark_infos[dark_path]
            )
            _assert_compatible(
                light_info,
                dark_info,
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=pipeline_parameters.dark_temperature_tolerance_celsius,
                workflow=pipeline_parameters.calibration_workflow,
            )
            _numeric_application_scale(
                light_info,
                master_dark_domain_info[exposure],
                target_label="registration Light",
                additive_label="MasterDark",
            )
        _numeric_application_scale(
            light_info,
            reference_bias,
            target_label="registration Light",
            additive_label="MasterBias",
        )
    bias_application_scale = _numeric_application_scale(
        light_numeric_reference,
        reference_bias,
        target_label="registration Light domain",
        additive_label="MasterBias",
    )
    dark_application_scales = {
        exposure: _numeric_application_scale(
            light_numeric_reference,
            master_dark_domain_info[exposure],
            target_label="registration Light domain",
            additive_label="MasterDark",
        )
        for exposure in master_darks
    }
    plan = CalibrationPlan(
        bias_path=str(master_bias) if master_bias is not None else None,
        dark_paths={exposure: str(path) for exposure, path in master_darks.items()},
        dark_bias_included_by_exposure={
            exposure: (
                True
                if exposure in dark_groups
                else supplied_dark_bias_included[path]
            )
            for exposure, path in master_darks.items()
        },
        dark_application_scale_by_exposure=dark_application_scales,
        flat_paths={key: str(value) for key, value in master_flats.items()},
        dark_scale=1.0,
        dark_includes_bias=True,
        bias_application_scale=bias_application_scale,
        flat_floor_fraction=0.05,
    )
    receipt = {
        "schemaVersion": 1,
        "stage": "registration-calibration-masters",
        "calibrationPolicy": workflow_receipt(pipeline_parameters.calibration_workflow),
        "xisfConversions": [dict(item) for item in xisf_conversions],
        "masterBias": bias_record,
        "masterDarks": dark_records,
        "masterFlats": flat_records,
        "registrationDarksByExposure": {
            format(exposure, ".9g"): str(path)
            for exposure, path in sorted(master_darks.items())
        },
        "registrationNumericDomain": {
            "light": light_numeric_reference.serializable(),
            "biasApplicationScale": bias_application_scale,
            "darkApplicationScaleByExposure": {
                format(exposure, ".9g"): scale
                for exposure, scale in sorted(dark_application_scales.items())
            },
            "applicationScaleSource": "normalized-unit-domain-ratio",
        },
        "artifacts": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "sizeBytes": path.stat().st_size,
            }
            for path in (master_bias, *master_darks.values(), *master_flats.values()) if path is not None
        ],
    }
    return plan, receipt


def _capture_single_field_generated_calibration(
    *,
    plan: Any,
    generated_directory: Path,
    upstream_receipt_path: Path,
    staged_inputs: Mapping[str, tuple[Path, ...]],
    source_aliases: Mapping[str, Path],
    pipeline_parameters: PipelineParameters,
    consumer_source_groups: Sequence[tuple[str, Sequence[Path]]],
    internal_source_identities: Mapping[str, _InternalSourceIdentity],
) -> Any:
    """Capture the private single-run trust handoff for generated masters."""

    generated_root = generated_directory.resolve(strict=True)

    def is_generated(path: Path) -> bool:
        return path.resolve(strict=True).is_relative_to(generated_root)

    def source_info(path: Path) -> FrameInfo:
        original = source_aliases.get(str(path), path)
        source_identity = internal_source_identities.get(
            str(original.expanduser().resolve(strict=True))
        )
        return _input_frame_info(
            path,
            pipeline_parameters,
            override_identity_path=original,
            override_source_identity=source_identity,
        )

    bias_spec: tuple[Path, FrameInfo] | None = None
    if staged_inputs["BIAS"]:
        bias_path = Path(plan.bias_path)
        if not is_generated(bias_path):
            raise E2EError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "raw Bias provenance did not produce an internal MasterBias",
                path=str(bias_path),
            )
        bias_spec = (bias_path, source_info(staged_inputs["BIAS"][0]))

    dark_specs: list[tuple[Path, FrameInfo, bool]] = []
    raw_dark_infos = {path: source_info(path) for path in staged_inputs["DARK"]}
    raw_dark_exposures = {
        float(info.exposure_seconds)
        for info in raw_dark_infos.values()
        if info.exposure_seconds is not None
    }
    for exposure in sorted(raw_dark_exposures):
        match = _find_dark(exposure, {float(key): Path(value) for key, value in plan.dark_paths.items()})
        if match is None or not is_generated(match[1]):
            raise E2EError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "raw Dark provenance did not produce an internal MasterDark",
            )
        reference = next(
            info
            for info in raw_dark_infos.values()
            if info.exposure_seconds is not None
            and math.isclose(
                float(info.exposure_seconds), exposure, rel_tol=0.0, abs_tol=1e-6
            )
        )
        dark_specs.append(
            (
                match[1],
                reference,
                bool(plan.dark_bias_included_by_exposure[match[0]]),
            )
        )

    flat_specs: list[tuple[Path, FrameInfo, float]] = []
    raw_flat_infos = {path: source_info(path) for path in staged_inputs["FLAT"]}
    for filter_name in sorted({_require_filter(info) for info in raw_flat_infos.values()}):
        flat_path = Path(plan.flat_paths[filter_name])
        if not is_generated(flat_path):
            raise E2EError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "raw Flat provenance did not produce an internal MasterFlat",
                path=str(flat_path),
            )
        reference = next(
            info for info in raw_flat_infos.values() if _require_filter(info) == filter_name
        )
        flat_specs.append((flat_path, reference, 1.0))

    return _capture_trusted_generated_calibration_set(
        master_bias=bias_spec,
        master_darks=dark_specs,
        master_flats=flat_specs,
        source_groups=consumer_source_groups,
        source_identities=internal_source_identities,
        upstream_receipt_path=upstream_receipt_path,
    )


def _register_lights(
    lights: tuple[Path, ...],
    calibration_plan: Any,
    *,
    detection: Any,
    registration: Any,
    workers: int,
    allow_projective: bool,
    source_aliases: Mapping[str, Path] | None = None,
    source_sha256_by_path: Mapping[str, str] | None = None,
    reference_candidates: Sequence[Path] | None = None,
    runner: FrameRunner | None = None,
) -> _RegistrationProducts:
    try:
        from openastroflow_registration import RegistrationConfig, run_registration
        from openastroflow_registration.quality import (
            estimate_stellar_scale_hints,
            normalize_quality_weights,
        )
    except ImportError as error:
        raise E2EError("REGISTRATION_BACKEND_UNAVAILABLE", str(error)) from error
    try:
        # Refine the preview bootstrap against full-resolution centroids with
        # the projective model: frames of another night or hour angle differ
        # from the reference by perspective terms (tilt, differential
        # refraction) that an affine fit leaves as a field-dependent
        # misregistration of several tenths of a pixel.
        selected_registration = registration or RegistrationConfig(
            refine_full_centroids=True,
            full_transform_model="projective" if allow_projective else "affine",
        )
        run = run_registration(
            [str(path) for path in lights],
            detection=detection,
            registration=selected_registration,
            calibration=calibration_plan,
            reference_candidates=(
                [str(path) for path in reference_candidates] if reference_candidates else None
            ),
            validate_warp=True,
            workers=workers,
            runner=runner,
        )
    except Exception as error:
        raise E2EError("REGISTRATION_FAILED", str(error)) from error
    transforms: dict[str, tuple[tuple[float, float, float], ...]] = {}
    transform_records: list[dict[str, Any]] = []
    reference_path = (
        str(Path(run.analyses[run.reference_index].path).resolve(strict=True))
        if hasattr(run, "analyses") and hasattr(run, "reference_index")
        else None
    )
    for item in run.transforms:
        if not item.accepted or item.full_matrix is None:
            raise E2EError("REGISTRATION_REJECTED", item.reason or "registration rejected", path=item.path)
        matrix = np.asarray(item.full_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise E2EError("REGISTRATION_MATRIX_INVALID", "full matrix is not finite 3x3", path=item.path)
        norm = float(np.linalg.norm(matrix, ord=np.inf))
        determinant = float(np.linalg.det(matrix))
        if norm == 0.0 or not math.isfinite(determinant) or abs(determinant) <= 1e-12 * norm**3:
            raise E2EError("REGISTRATION_MATRIX_INVALID", "full matrix is singular", path=item.path)
        if not allow_projective and not np.allclose(
            matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-9
        ):
            raise E2EError(
                "REGISTRATION_PROJECTIVE_UNSUPPORTED",
                "portable pixel integration currently requires affine full matrices",
                path=item.path,
            )
        expected_refined_model = (
            f"{getattr(selected_registration, 'full_transform_model', 'affine')}"
            "-full-centroid"
        )
        actual_path = str(Path(item.path).resolve(strict=True))
        is_reference = actual_path == reference_path or (
            reference_path is None and len(run.transforms) == 1
        )
        if (
            bool(getattr(selected_registration, "refine_full_centroids", False))
            and not is_reference
            and getattr(item, "transform_model", "similarity")
            != expected_refined_model
        ):
            refine_evidence = getattr(item, "full_refine_evidence", {})
            reason = refine_evidence.get(
                "reason", "full-resolution refinement was not accepted"
            )
            raise E2EError(
                "REGISTRATION_FULL_REFINEMENT_REJECTED",
                str(reason),
                path=item.path,
            )
        serialized = tuple(tuple(float(value) for value in row) for row in matrix)
        displayed_path = str((source_aliases or {}).get(actual_path, Path(actual_path)))
        transforms[displayed_path] = serialized
        transform_records.append(
            {
                "path": displayed_path,
                "filter": item.filter_name,
                "fullMatrixInputToOutput": [list(row) for row in serialized],
                "matchCount": item.match_count,
                "inlierCount": item.inlier_count,
                "inlierRatio": item.inlier_ratio,
                "rmsPreviewPixels": item.rms_preview_px,
                "rmsFullPixels": item.rms_full_px,
                "transformModel": getattr(item, "transform_model", "similarity"),
                "fullRefineInliers": getattr(item, "full_refine_inliers", 0),
                "fullRefineSeconds": getattr(item, "full_refine_seconds", 0.0),
                "fullRefineEvidence": dict(
                    getattr(item, "full_refine_evidence", {})
                ),
                "warpPearson": item.warp_pearson,
                "warpValidFraction": item.warp_valid_fraction,
            }
        )
    # Registration contributes the PSF-coherence quality weight only; the
    # integration multiplies it by its own inverse-variance noise weight
    # measured on the normalized frames, so noise must not be weighted here.
    weights = normalize_quality_weights(run.analyses)
    candidate_indices: list[int] | None = None
    if reference_candidates:
        allowed = {str(Path(path).expanduser().resolve(strict=True)) for path in reference_candidates}
        candidate_indices = [
            index
            for index, analysis in enumerate(run.analyses)
            if str(Path(analysis.path).resolve(strict=True)) in allowed
        ]
    scale_estimates = estimate_stellar_scale_hints(
        run.analyses,
        run.transforms,
        weights,
        workers=workers,
        # The normalization reference obeys the same candidate set as the
        # geometric one: the master inherits its background.
        reference_candidates=candidate_indices,
    )
    quality_weights = {
        str(
            (source_aliases or {}).get(
                str(Path(analysis.path).resolve(strict=True)),
                Path(analysis.path).resolve(strict=True),
            )
        ): float(weight)
        for analysis, weight in zip(run.analyses, weights, strict=True)
    }

    def displayed_and_digest(index: int) -> tuple[str, str]:
        actual = str(Path(run.analyses[index].path).resolve(strict=True))
        displayed = str((source_aliases or {}).get(actual, Path(actual)))
        digest = (source_sha256_by_path or {}).get(actual) or (
            source_sha256_by_path or {}
        ).get(displayed)
        if digest is None:
            digest = _sha256(Path(actual))
        return displayed, digest

    stellar_scale_hints: dict[str, StellarScaleHint] = {}
    for estimate in scale_estimates:
        source_path, source_sha256 = displayed_and_digest(estimate.source_index)
        reference_path, reference_sha256 = displayed_and_digest(
            estimate.reference_index
        )
        filter_name = str(estimate.filter_name or "UNKNOWN")
        stellar_scale_hints[source_path] = StellarScaleHint(
            source_path=source_path,
            reference_path=reference_path,
            filter_name=filter_name,
            source_sha256=source_sha256,
            reference_sha256=reference_sha256,
            scale=estimate.scale,
            status=estimate.status,
            evidence=dict(estimate.evidence),
        )
    receipt = {
        "schemaVersion": 1,
        "stage": "registration",
        "referenceIndex": run.reference_index,
        "referencePath": str(
            (source_aliases or {}).get(
                str(Path(run.analyses[run.reference_index].path).resolve(strict=True)),
                Path(run.analyses[run.reference_index].path).resolve(strict=True),
            )
        ),
        "qualityWeights": list(weights),
        "qualityWeightsBySource": quality_weights,
        "stellarScaleHints": [
            stellar_scale_hints[path].serializable()
            for path in sorted(stellar_scale_hints)
        ],
        "timingSeconds": {
            "analysisWall": run.analysis_wall_seconds,
            "registrationWall": run.registration_wall_seconds,
            "total": run.total_seconds,
        },
        "transforms": transform_records,
    }
    return _RegistrationProducts(
        transforms,
        quality_weights,
        stellar_scale_hints,
        run,
        receipt,
        {
            key: str(value)
            for key, value in (source_aliases or {}).items()
        },
    )


def _drizzle_candidates(
    *,
    staging: Path,
    work: Path,
    pipeline_result: Any,
    options: DrizzleOptions,
    sampling_evidence: Mapping[str, Any],
    threads: int | None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Drizzle every filter group from the ordinary integration's products.

    The pixel pipeline hands over, per group, the calibrated (unregistered)
    Lights, their registration matrices, the normalization coefficients, the
    integration weights and the per-sample rejection masks; the native drizzle
    reproduces the group's integration on the finer grid from exactly those.
    """

    groups = getattr(pipeline_result, "drizzle_groups", None) or {}
    if not groups:
        raise E2EError(
            "DRIZZLE_INPUTS_MISSING",
            "the pixel pipeline did not capture drizzle inputs for any filter group",
        )
    candidates: dict[str, Path] = {}
    receipts: dict[str, Any] = {}
    for filter_name, group in sorted(groups.items()):
        token = _safe_token(filter_name)
        target_dir = work / "drizzle" / token
        target_dir.mkdir(parents=True, exist_ok=False)
        # A colour channel group of a Bayer filter is a Bayer drizzle: the
        # calibrated mosaics' own samples of that colour are dropped, with the
        # channel group's normalization, weights and rejection.
        request = DrizzleGroupRequest(
            frames=group.frames,
            reference_shape=tuple(int(value) for value in group.reference_shape),
            output_path=str(target_dir / f"master_light_{token}_drizzle_unsolved.fits"),
            receipt_path=str(target_dir / "receipt.json"),
            scale=options.scale,
            pixfrac=options.pixfrac,
            kernel=options.kernel,
            cfa_pattern=group.cfa_pattern,
            channel=group.channel,
            metadata={**dict(group.metadata), "OAFCROP": "NONE"},
            max_accumulator_bytes=options.max_working_set_bytes,
            threads=threads,
            durable=False,
        )
        try:
            result = drizzle_group(request)
            verified = verify_drizzle_receipt(result.receipt_path)
        except CalibrationError as error:
            raise E2EError(error.code, str(error), path=error.path) from error
        statistics = verified.get("statistics", {})
        coverage_status = "PASS"
        if float(statistics.get("coverageFraction", 0.0)) < options.minimum_coverage_fraction:
            coverage_status = "LOW_COVERAGE"
        candidates[filter_name] = Path(result.output_path).resolve(strict=True)
        receipts[filter_name] = {
            **dict(verified),
            "coverageGate": {
                "status": coverage_status,
                "minimumCoverageFraction": float(options.minimum_coverage_fraction),
                "observedCoverageFraction": float(statistics.get("coverageFraction", 0.0)),
                "advisory": True,
            },
            "localNormalization": {"status": "NOT_APPLICABLE", "reason": "drizzle applies the global normalization coefficients"},
        }
    return candidates, {
        "mode": IntegrationMode.DRIZZLE.value,
        "options": options.serializable(),
        "sampling": dict(sampling_evidence),
        "filters": receipts,
        "localNormalizationEvidence": {"status": "NOT_APPLICABLE", "groups": {}},
    }


def _ordinary_candidates(
    pipeline_paths: Sequence[str],
    pipeline_root: Path,
    staging: Path,
    *,
    minimum_coverage_fraction: float = 0.90,
    maximum_null_fraction: float = 0.10,
) -> tuple[dict[str, Path], dict[str, Any]]:
    pipeline_receipt = json.loads((pipeline_root / "receipt.json").read_text(encoding="utf-8"))
    integration_groups = pipeline_receipt.get("statistics", {}).get("integrationGroups", {})
    candidates: dict[str, Path] = {}
    coverage: dict[str, Any] = {}
    for value in pipeline_paths:
        path = Path(value).resolve(strict=True)
        info = read_frame_info(path)
        filter_name = info.filter_name
        if filter_name in candidates:
            raise E2EError("FILTER_OUTPUT_DUPLICATE", f"multiple masters for filter {filter_name}")
        candidates[filter_name] = path
        group = integration_groups.get(filter_name, {})
        promoted_maps: dict[str, str] = {}
        token = _safe_token(filter_name)
        raw_maps = group.get("integration", {}).get("maps", {})
        required_maps = {
            "acceptedSampleCount",
            "coverageFraction",
            "rejectionCount",
        }
        if not isinstance(raw_maps, Mapping) or set(raw_maps) != required_maps:
            raise E2EError(
                "ORDINARY_INTEGRATION_MAPS_MISSING",
                f"ordinary integration lacks required maps for filter {filter_name}",
            )
        for map_name in sorted(required_maps):
            relative = raw_maps[map_name]
            if not isinstance(relative, str):
                raise E2EError(
                    "PIPELINE_RECEIPT_INVALID", "integration map path is not a string"
                )
            source = (pipeline_root / relative).resolve(strict=True)
            try:
                source.relative_to(pipeline_root.resolve(strict=True))
            except ValueError as error:
                raise E2EError(
                    "PIPELINE_RECEIPT_INVALID", "integration map escapes pipeline root"
                ) from error
            destination = staging / "coverage" / f"{token}_{map_name}.fits"
            try:
                with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=4 * 1024 * 1024)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
            except FileExistsError as error:
                raise E2EError(
                    "OUTPUT_EXISTS", "refusing to replace promoted integration map",
                    path=str(destination),
                ) from error
            if _sha256(source) != _sha256(destination):
                raise E2EError(
                    "ARTIFACT_HASH_MISMATCH",
                    "promoted integration map differs from its source",
                    path=str(destination),
                )
            promoted_maps[map_name] = str(destination.relative_to(staging))
        coverage_path = staging / promoted_maps["coverageFraction"]
        with fits.open(coverage_path, mode="readonly", memmap=True, checksum=True) as hdul:
            coverage_data = np.asarray(hdul[0].data, dtype=np.float32)
            finite_coverage = np.isfinite(coverage_data)
            supported_fraction = float(
                np.count_nonzero(finite_coverage & (coverage_data > 0))
                / coverage_data.size
            )
        with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
            # Keep only the finite mask: an array of the memory map itself
            # would outlive the ``with`` block and, on Windows, keep the
            # master locked while the run later moves it.
            finite_master = np.isfinite(hdul[0].data)
        null_fraction = float(
            1.0 - np.count_nonzero(finite_master) / finite_master.size
        )
        if (
            supported_fraction < minimum_coverage_fraction
            or null_fraction > maximum_null_fraction
        ):
            raise E2EError(
                "ORDINARY_COVERAGE_GATE_FAILED",
                f"filter {filter_name} supported fraction {supported_fraction:.6g}, "
                f"null fraction {null_fraction:.6g}; require >= {minimum_coverage_fraction:.6g} "
                f"and <= {maximum_null_fraction:.6g}",
            )
        coverage[filter_name] = {
            "mode": IntegrationMode.ORDINARY.value,
            "cropTopLeftBottomRightExclusive": group.get("crop"),
            "masterStatistics": group.get("masterStatistics"),
            "mapStatistics": group.get("mapStatistics"),
            "maps": promoted_maps,
            "weights": group.get("integration", {}).get("weightComponents"),
            "execution": group.get("integration", {}).get("execution"),
            "sourceReceipt": "receipts/pixel-pipeline.json",
            "qualityGate": {
                "status": "PASS",
                "supportedFraction": supported_fraction,
                "minimumSupportedFraction": minimum_coverage_fraction,
                "nullFraction": null_fraction,
                "maximumNullFraction": maximum_null_fraction,
                "zeroAcceptedPixelsRemainNaN": True,
            },
        }
    return candidates, {"mode": IntegrationMode.ORDINARY.value, "filters": coverage}


def _artifact_records(staging: Path, roots: Sequence[Path], *, workers: int = 4) -> list[dict[str, Any]]:
    """Digest every artifact under ``roots`` in a stable order; the files are
    independent and hashing releases the GIL, so they are digested concurrently."""

    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend([root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file()))
    count = max(1, min(int(workers), len(paths)))
    if count <= 1:
        digests = [_sha256(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="oaf-artifacts") as pool:
            digests = list(pool.map(_sha256, paths))
    return [
        {
            "path": str(path.relative_to(staging)),
            "sha256": digest,
            "sizeBytes": path.stat().st_size,
        }
        for path, digest in zip(paths, digests, strict=True)
    ]


def _relativize_solver_attempts(
    attempts: list[dict[str, Any]], staging: Path
) -> list[dict[str, Any]]:
    """Relativize outer paths while preserving signed backend evidence bytes.

    Adapter evidence contains its own receipt digest.  Recursively rewriting
    paths inside that evidence would silently invalidate the backend receipt,
    so only the E2E-owned wrapper fields are changed here.
    """

    for attempt in attempts:
        result = attempt.get("result")
        if isinstance(result, dict):
            output_path = result.get("outputPath")
            if isinstance(output_path, str):
                try:
                    result["outputPath"] = str(Path(output_path).relative_to(staging))
                except ValueError:
                    pass
        artifact = attempt.get("artifact")
        if isinstance(artifact, dict):
            artifact_path = artifact.get("path")
            if isinstance(artifact_path, str):
                try:
                    artifact["path"] = str(Path(artifact_path).relative_to(staging))
                except ValueError:
                    pass
        if attempt.get("executionVerified") is True:
            attempt["verificationBoundary"] = (
                "verified before atomic outer-directory publication; final content "
                "identity is rebound by the E2E receipt"
            )
    return attempts


def _failure_result(
    *,
    staging: Path,
    failure_directory: Path,
    code: str,
    message: str,
    passed: tuple[Path, ...],
    excluded: tuple[Path, ...],
    solver: Mapping[str, Any],
    sources: Sequence[_SourceIdentity],
    callback: ProgressCallback | None,
    screening: Mapping[str, Any] | None = None,
    selection_policy: str | None = None,
    selection: Mapping[str, Any] | None = None,
) -> E2EResult:
    private_work = staging / WORK_DIRECTORY
    remove_tree(private_work)
    _sanitize_shareable_tree(staging, sources)
    evidence_artifacts = _artifact_records(
        staging,
        tuple(
            path
            for path in (
                staging / "products",
                staging / "previews",
                staging / "qc",
                staging / "coverage",
                staging / "receipts",
            )
            if path.exists()
        ),
    )
    receipt_core = {
        "schemaVersion": 1,
        "pipelineVersion": E2E_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "state": E2EState.UNSOLVED_WORKING.value,
        "code": code,
        "message": message,
        "sources": [item.serializable() for item in sources],
        "qualityControl": {
            "manifest": "qc/manifest.json",
            "passedLights": len(passed),
            "excludedLights": len(excluded),
            **({"screening": dict(screening)} if screening is not None else {}),
            **({"selectionPolicy": selection_policy} if selection_policy is not None else {}),
            **({"selection": dict(selection)} if selection is not None else {}),
        },
        "astrometry": {"status": "UNSOLVED", **dict(solver)},
        "artifacts": evidence_artifacts,
        "publication": {
            "kind": "failure-evidence-only",
            "requestedOutputPublished": False,
            "noReplace": True,
        },
    }
    receipt_core = _share_safe_receipt_core(
        receipt_core, staging=staging, identities=sources
    )
    receipt_id = "sha256:" + hashlib.sha256(_canonical_json(receipt_core)).hexdigest()
    _write_json(staging / "receipt.json", {"receiptId": receipt_id, **receipt_core})
    _emit(callback, ProgressStage.PUBLISH, "started", "publishing UNSOLVED evidence")
    _rename_directory_no_replace(staging, failure_directory)
    _fsync_directory(failure_directory.parent)
    try:
        _emit(callback, ProgressStage.FAILED, "completed", message)
    except Exception:
        pass
    return E2EResult(
        success=False,
        code=code,
        state=E2EState.UNSOLVED_WORKING,
        output_directory=None,
        evidence_directory=str(failure_directory),
        receipt_path=str(failure_directory / "receipt.json"),
        product_paths=(),
        preview_paths=(),
        passed_light_paths=tuple(str(path) for path in passed),
        excluded_light_paths=tuple(str(path) for path in excluded),
        message=message,
    )


def run_e2e(
    request: E2ERequest,
    *,
    solver_backends: Sequence[SolverBackend],
    progress: ProgressCallback | None = None,
) -> E2EResult:
    """Run the complete workflow and atomically publish only verified WCS products.

    ``solver_backends`` is an ordered fallback chain.  Each backend result must
    pass :func:`validate_solver_result`, execution-receipt verification, and an
    independent validation of the WCS actually written to ``output_path``.
    """

    _validate_request(request)
    if not isinstance(solver_backends, Sequence) or not solver_backends:
        raise E2EError("SOLVER_CHAIN_EMPTY", "at least one solver backend is required")

    output = Path(request.output_directory).expanduser().resolve(strict=False)
    failure_directory = output.with_name(output.name + ".unsolved")
    if os.path.lexists(output):
        raise E2EError("OUTPUT_EXISTS", "output directory must be new", path=str(output))
    if os.path.lexists(failure_directory):
        raise E2EError(
            "FAILURE_OUTPUT_EXISTS",
            "move the previous UNSOLVED evidence before retrying",
            path=str(failure_directory),
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    _emit(progress, ProgressStage.INVENTORY, "started", "validating explicit source files")
    lights = _canonical_inputs(request.light_files, "LIGHT", required=True)
    flats = _canonical_inputs(request.flat_files, "FLAT", required=False)
    biases = _canonical_inputs(request.bias_files, "BIAS", required=False)
    darks = _canonical_inputs(request.dark_files, "DARK", required=False)
    master_biases = _canonical_inputs(
        request.master_bias_files, "MASTER_BIAS", required=False
    )
    master_darks = _canonical_inputs(
        request.master_dark_files, "MASTER_DARK", required=False
    )
    master_flats = _canonical_inputs(
        request.master_flat_files, "MASTER_FLAT", required=False
    )
    if len(master_biases) > 1 or (biases and master_biases) or (not biases and not master_biases and request.pipeline_parameters.calibration_workflow != MONO_STANDARD):
        raise E2EError(
            "BIAS_SOURCE_AMBIGUOUS",
            "supply exactly one Bias source mode: raw Bias frames or one MasterBias",
        )
    if not flats and not master_flats:
        raise E2EError(
            "MASTER_FLAT_MISSING", "raw Flats or supplied MasterFlats are required"
        )
    all_grouped = (
        ("LIGHT", lights),
        ("FLAT", flats),
        ("DARK", darks),
        ("BIAS", biases),
        ("MASTER_BIAS", master_biases),
        ("MASTER_DARK", master_darks),
        ("MASTER_FLAT", master_flats),
    )
    flattened = [(role, path) for role, paths in all_grouped for path in paths]
    canonical_names = [os.path.normcase(str(path)) for _, path in flattened]
    if len(set(canonical_names)) != len(canonical_names):
        raise E2EError("INPUT_ROLE_OVERLAP", "one source appears in multiple roles")
    if any(path == output or output in path.parents for _, path in flattened):
        raise E2EError("OUTPUT_ALIASES_SOURCE", "output cannot contain or alias a source")
    # Capture every source exactly once.  All preflight/override lookups below
    # reuse only this path/stat-bound identity; private snapshot creation still
    # re-reads through one open handle and final publication still rehashes the
    # originals, so eliminating duplicate inventory passes weakens no drift gate.
    identities = _capture_sources(flattened, workers=request.workers)
    identity_by_path = {identity.path: identity for identity in identities}
    raw_digests: dict[str, list[Path]] = {}
    for identity in identities:
        if identity.role in {"LIGHT", "FLAT", "DARK", "BIAS"}:
            raw_digests.setdefault(identity.sha256, []).append(Path(identity.path))
    for override in request.pipeline_parameters.raw_frame_metadata_overrides:
        matches = raw_digests.get(override.source_sha256, [])
        if len(matches) != 1:
            raise E2EError(
                "RAW_FRAME_METADATA_OVERRIDE_SOURCE_AMBIGUOUS",
                "raw-frame override digest must bind exactly one current E2E source",
            )
    for expected_role, path in flattened:
        try:
            frame_info = _input_frame_info(
                path,
                request.pipeline_parameters,
                override_source_identity=identity_by_path[str(path)],
            )
            actual_role = normalize_role(frame_info.role)
        except CalibrationError as error:
            raise E2EError(error.code, str(error), path=str(path)) from error
        if actual_role != expected_role:
            raise E2EError(
                "FRAME_ROLE_MISMATCH",
                f"expected {expected_role}, found {actual_role}",
                path=str(path),
            )
        if expected_role in {"LIGHT", "FLAT", "DARK", "BIAS"}:
            cfa = frame_info.cfa_pattern.strip().upper()
            if cfa in {"", "UNKNOWN", "UNSPECIFIED"}:
                raise E2EError(
                    "CFA_CONFIRMATION_REQUIRED",
                    f"raw {expected_role} CFA metadata is unknown; add a hash-bound rawFrameMetadataOverrides confirmation",
                    path=str(path),
                )
            if cfa != "NONE" and not is_cfa_pattern(cfa):
                raise E2EError(
                    "CFA_PATTERN_UNSUPPORTED",
                    f"CFA pattern {cfa!r} of raw {expected_role} is not supported (RGGB, BGGR, GRBG, GBRG are)",
                    path=str(path),
                )
    _emit(progress, ProgressStage.INVENTORY, "completed", f"validated {len(identities)} source files")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=STAGING_SUFFIX, dir=output.parent))
    published = False
    # One spawned worker pool serves quality-gate measurement, star-field
    # analysis and registration: each spawned worker imports NumPy, astropy,
    # SEP and the engine (about 1.7 s per worker on the Windows laptop), so
    # three pools per run cost several seconds of wall time for nothing.
    # Every frame runs the same function whichever pool executes it, so the
    # values never depend on the sharing.  Closed as soon as registration is
    # done so its processes do not sit on memory during integration.
    frame_runner = FrameRunner(request.workers, len(lights))
    try:
        work = staging / WORK_DIRECTORY
        receipts_dir = staging / "receipts"
        products_dir = staging / "products"
        previews_dir = staging / "previews"
        coverage_dir = staging / "coverage"
        for directory in (work, receipts_dir, products_dir, previews_dir, coverage_dir):
            directory.mkdir()

        _emit(progress, ProgressStage.QUALITY_CONTROL, "started", "measuring and gating Light frames")
        # Star counts, quality weights and therefore the masters depend on
        # SEP's extraction being reproducible; the verdict of its self-test
        # is part of the receipt and a failing build is called out here.
        source_extraction = cached_extraction_self_test()
        if not source_extraction["deterministic"]:
            notice = (
                f"{NONDETERMINISTIC_WARNING}: the installed sep {source_extraction['sepVersion']} "
                "returns different objects for identical input; run-to-run identical products "
                "cannot be claimed on this machine"
            )
            warnings.warn(notice, RuntimeWarning, stacklevel=2)
            _emit(progress, ProgressStage.QUALITY_CONTROL, "running", notice)
        qc_dir = staging / "qc"
        qc_dir.mkdir()
        qc_timings: dict[str, float] = {}
        qc_cache_stats: dict[str, int] = {}
        qc_measurement_stats: dict[str, Any] = {}
        qc_started = perf_counter()
        measurements = measure_paths(
            [str(path) for path in lights],
            qc_dir,
            request.qc_config,
            workers=request.workers,
            stats=qc_measurement_stats,
            runner=frame_runner,
        )
        qc_timings["measurementSeconds"] = perf_counter() - qc_started
        _emit(progress, ProgressStage.QUALITY_CONTROL, "running", f"measured {len(measurements)} Light frames; analyzing star fields")
        qc_started = perf_counter()
        qc_analysis_stats: dict[str, Any] = {}
        groups, frame_results = analyze_measurements(
            measurements, request.qc_config,
            cache_directory=quality_cache_directory(), cache_stats=qc_cache_stats,
            workers=request.workers, stats=qc_analysis_stats, runner=frame_runner,
        )
        qc_timings["analysisSeconds"] = perf_counter() - qc_started
        qc_started = perf_counter()
        evaluate_quality_gate(frame_results, measurements, request.gate_policy)
        qc_timings["gateSeconds"] = perf_counter() - qc_started
        review_previews = _write_review_previews(staging, qc_dir, frame_results)
        qc_manifest = _qc_manifest(
            staging, groups, frame_results, request.qc_config, request.gate_policy, review_previews
        )
        qc_manifest["timings"] = qc_timings
        qc_manifest["measurement"] = qc_measurement_stats
        qc_manifest["analysis"] = qc_analysis_stats
        qc_manifest["analysisCache"] = qc_cache_stats
        approved_review_paths, approval_request_digest, approval_evidence = (
            _apply_review_approvals(
                request=request,
                identities=identities,
                results=frame_results,
            )
        )
        qc_manifest["manualReviewApprovals"] = {
            "defaultDisposition": (
                "DECIDED_BY_EXPLICIT_SELECTION"
                if request.selection.explicit
                else "DECIDED_BY_SELECTION_POLICY"
                if request.selection.unattended
                else "EXCLUDED"
            ),
            "requestDigest": approval_request_digest,
            "gatePolicyDigest": request.gate_policy.canonical_digest(),
            "accepted": approval_evidence,
        }
        # The blink evidence (flags, reference, scores) is recomputed from
        # this run's own quality pass, so a run with an explicit selection
        # is self-describing without the blink session that produced it.
        blink: BlinkEvidence | None = None
        explicit_block: dict[str, Any] | None = None
        explicit_admitted: set[str] = set()
        if request.selection.explicit:
            qc_started = perf_counter()
            blink = blink_evidence(
                frame_results,
                measurements,
                flags_policy=request.blink_flags_policy,
                gate_policy=request.gate_policy,
                qc_config=request.qc_config,
            )
            explicit_admitted, explicit_block = _apply_explicit_selection(
                request=request,
                identities=identities,
                results=frame_results,
                evidence=blink,
            )
            qc_timings["blinkSeconds"] = perf_counter() - qc_started
            qc_manifest["explicitSelection"] = {
                "selectionDigest": explicit_block["selectionDigest"],
                "flagsPolicyDigest": explicit_block["flagsPolicyDigest"],
                "counts": dict(explicit_block["counts"]),
            }
        selection_features: list[FrameSelectionFeatures] = []
        selection_decisions: list[SelectionDecision] = []
        selection_confidence: dict[str, float] = {}
        selection_region_maps: dict[str, RegionWeightMap] = {}
        if request.selection.unattended:
            selection_features = extract_features(
                frame_results,
                measurements,
                night_boundary_hours=request.gate_policy.night_boundary_hours,
                observing_timezone=request.qc_config.observing_timezone,
            )
            if request.selection.region_weights:
                # Region weight maps come from the QC grid alone, so they are
                # built before the decisions that depend on them.
                selection_region_maps = region_weight_maps(frame_results)
            selection_decisions = decide(
                selection_features,
                request.selection,
                approved_paths=approved_review_paths,
                region_maps=selection_region_maps,
            )
            selection_confidence = {
                item.path: item.weight_multiplier for item in selection_decisions if item.admitted
            }
            qc_manifest["selection"] = selection_receipt(
                request.selection,
                selection_features,
                selection_decisions,
                region_maps=selection_region_maps,
            )
        _write_json(qc_dir / "manifest.json", qc_manifest)
        if blink is not None:
            _write_json(qc_dir / "blink.json", blink.serializable())
        gate_by_path = {
            str(Path(result.path).resolve(strict=True)): result.quality_gate
            for result in frame_results
        }
        if request.selection.explicit:
            passed = tuple(path for path in lights if str(path) in explicit_admitted)
        elif request.selection.unattended:
            decision_by_path = {item.path: item for item in selection_decisions}
            passed = tuple(
                path
                for path in lights
                if decision_by_path.get(str(path)) is not None
                and decision_by_path[str(path)].admitted
            )
        else:
            passed = tuple(
                path
                for path in lights
                if gate_by_path.get(str(path)) is not None
                and (
                    gate_by_path[str(path)].disposition is GateDisposition.PASS
                    or str(path) in approved_review_paths
                )
            )
        passed_set = set(passed)
        passed_set_text = {str(path) for path in passed}
        excluded = tuple(path for path in lights if path not in passed_set)
        screening = _screening_summary(
            frame_results, passed, approved_review_paths, review_previews, explicit_block
        )
        passed_results = [
            result
            for result in frame_results
            if Path(result.path).resolve(strict=True) in passed_set
        ]
        solver_hints = _inferred_solver_hints(request, passed_results)
        drizzle_sampling = (
            _drizzle_sampling_evidence(passed_results, request.drizzle)
            if request.integration_mode is IntegrationMode.DRIZZLE
            else {}
        )
        _emit(
            progress,
            ProgressStage.QUALITY_CONTROL,
            "completed",
            (
                f"{len(passed)} kept by the explicit selection "
                f"({explicit_block['counts']['overriddenExcludeFlags']} kept against an EXCLUDE flag); "
                f"{len(excluded)} dropped"
                if request.selection.explicit and explicit_block is not None
                else f"{len(passed)} admitted by selection policy {request.selection.policy} "
                f"({sum(1 for item in selection_decisions if item.admitted and item.confidence < 1.0)} "
                f"with reduced weight); {len(excluded)} excluded"
                if request.selection.unattended
                else f"{len(passed)} admitted ({len(approved_review_paths)} explicitly approved REVIEW); "
                f"{len(excluded)} REVIEW/HARD_FAIL excluded"
            ),
        )
        panel_counts: dict[tuple[str, str], list[int]] = {}
        for frame in frame_results:
            panel_key = (frame.metadata.target.strip().upper(), frame.metadata.filter_name.strip().upper())
            counts = panel_counts.setdefault(panel_key, [0, 0])
            counts[1] += 1
            counts[0] += int(Path(frame.path).resolve(strict=True) in passed_set)
        insufficient = [
            f"{target} / {filter_name}: {admitted} of {total} Light frames admitted"
            for (target, filter_name), (admitted, total) in sorted(panel_counts.items())
            if admitted < 2
        ]
        if insufficient:
            _verify_sources(identities)
            result = _failure_result(
                staging=staging,
                failure_directory=failure_directory,
                code="QC_INSUFFICIENT_LIGHTS" if passed else "NO_PASS_LIGHTS",
                message="; ".join(insufficient) + "; at least 2 admitted Light frames per target/filter are required for registration. Review the screening evidence before processing.",
                passed=passed,
                excluded=excluded,
                solver={"attempts": {}},
                sources=identities,
                callback=progress,
                screening=screening,
                selection_policy=request.selection.policy,
                selection=explicit_block,
            )
            published = True
            return result

        (
            staged_inputs,
            registration_source_aliases,
            xisf_conversions,
            staged_input_digests,
        ) = (
            _stage_e2e_xisf_inputs(
                (
                    ("LIGHT", passed),
                    ("FLAT", flats),
                    ("DARK", darks),
                    ("BIAS", biases),
                    ("MASTER_BIAS", master_biases),
                    ("MASTER_DARK", master_darks),
                    ("MASTER_FLAT", master_flats),
                ),
                work / "pixel-inputs",
                request.pipeline_parameters,
                identity_by_path,
            )
        )
        staged_light_infos = {
            str(registration_source_aliases[str(path)].resolve(strict=True)): _input_frame_info(
                path,
                request.pipeline_parameters,
                override_identity_path=registration_source_aliases[str(path)],
                override_source_identity=identity_by_path[
                    str(registration_source_aliases[str(path)].resolve(strict=True))
                ],
            )
            for path in staged_inputs["LIGHT"]
        }

        _emit(progress, ProgressStage.CALIBRATION, "started", "building registration calibration masters")
        try:
            calibration_plan, calibration_receipt = _build_registration_masters(
                biases=staged_inputs["BIAS"],
                darks=staged_inputs["DARK"],
                flats=staged_inputs["FLAT"],
                supplied_biases=staged_inputs["MASTER_BIAS"],
                supplied_darks=staged_inputs["MASTER_DARK"],
                supplied_flats=staged_inputs["MASTER_FLAT"],
                lights=staged_inputs["LIGHT"],
                directory=work / "registration-calibration",
                pipeline_parameters=request.pipeline_parameters,
                source_aliases=registration_source_aliases,
                source_identities=identity_by_path,
                xisf_conversions=xisf_conversions,
            )
        except CalibrationError as error:
            raise E2EError(error.code, str(error), path=error.path) from error
        _verify_staged_pixel_inputs(staged_input_digests, identities)
        registration_calibration_receipt_path = (
            receipts_dir / "registration-calibration.json"
        )
        # This receipt becomes the content-bound trust anchor for generated
        # masters.  Write its final share-safe representation now so the SHA
        # recorded by the pixel receipt remains verifiable after publication.
        share_safe_calibration_receipt = _share_safe_value(
            calibration_receipt,
            staging=staging,
            source_tokens={
                identity.path: f"source/{identity.source_id}/{Path(identity.path).name}"
                for identity in identities
            },
        )
        _write_json(
            registration_calibration_receipt_path,
            share_safe_calibration_receipt,
        )
        _emit(progress, ProgressStage.CALIBRATION, "completed", "calibration masters verified")

        _emit(progress, ProgressStage.REGISTRATION, "started", "measuring full-resolution transforms")
        # A manually admitted REVIEW frame may be integrated, but it never
        # anchors the registration: a cloud-covered frame scores well on
        # sharp noise blobs and would leave every real frame unregistered.
        # The reference comes from the frames that passed the gate on their
        # own; only when nothing did do the approved frames compete.
        # The same holds for a Light the blink reviewer kept against the
        # gate: it is integrated, but the reference comes from gate-PASS
        # frames while any exist.
        anchor_excluded = set(approved_review_paths)
        # A frame with a blink flag (moonlit or hazy sky, low star retention,
        # extinction, atypical background shape, ...) is integrated when the
        # reviewer keeps it, but it never anchors the group: the master
        # inherits the reference's background, and the previous rule's
        # preference for the lowest sky picked exactly such cloud-dimmed
        # frames (NGC 6822, 2026-09-22).
        if blink is None:
            blink = blink_evidence(
                frame_results,
                measurements,
                flags_policy=request.blink_flags_policy,
                gate_policy=request.gate_policy,
                qc_config=request.qc_config,
            )
        anchor_excluded.update(
            str(Path(frame.path).resolve(strict=True)) for frame in blink.flags if frame.flags
        )
        if request.selection.explicit:
            anchor_excluded.update(
                path
                for path, gate in gate_by_path.items()
                if path in passed_set_text
                and (gate is None or gate.disposition is not GateDisposition.PASS)
            )
        gate_passed_lights = [
            path for path in staged_inputs["LIGHT"]
            if str(registration_source_aliases.get(str(path), path)) not in anchor_excluded
            and str(path) not in anchor_excluded
        ]
        registration = _register_lights(
            staged_inputs["LIGHT"],
            calibration_plan,
            detection=request.registration_detection,
            registration=request.registration_config,
            workers=request.workers,
            allow_projective=True,
            reference_candidates=gate_passed_lights if gate_passed_lights and len(gate_passed_lights) < len(staged_inputs["LIGHT"]) else None,
            source_aliases=registration_source_aliases,
            source_sha256_by_path={
                str(path): (
                    staged_input_digests[str(path)]
                    if str(path) in staged_input_digests
                    else identity_by_path[
                        str(registration_source_aliases[str(path)].resolve(strict=True))
                    ].sha256
                )
                for path in staged_inputs["LIGHT"]
            },
            runner=frame_runner,
        )
        frame_runner.close()
        if selection_confidence:
            # Reduced-confidence frames keep their registration quality weight
            # scaled by the selection confidence; excluded frames never reach here.
            registration = replace(
                registration,
                quality_weights={
                    path: float(weight) * float(selection_confidence.get(path, 1.0))
                    for path, weight in registration.quality_weights.items()
                },
            )
        _write_json(receipts_dir / "registration.json", registration.receipt)
        _emit(
            progress,
            ProgressStage.REGISTRATION,
            "completed",
            f"accepted {len(registration.transforms)} full matrices",
        )

        _emit(progress, ProgressStage.INTEGRATION, "started", "calibrating, registering, and integrating PASS frames")
        pipeline_root = work / PIXEL_PIPELINE_DIRECTORY
        # Drizzle mode runs the same registered integration: its per-frame
        # normalization, weights and rejection masks are what the drizzle
        # applies to the calibrated Lights on the finer grid.
        pipeline_transforms: Mapping[str, Sequence[Sequence[float]]] = registration.transforms
        if set(pipeline_transforms) != {str(path) for path in passed}:
            raise E2EError(
                "REGISTRATION_TRANSFORM_SET_INCOMPLETE",
                "transform map must bind every and only admitted Light",
            )
        pixel_source_paths = (
            *biases,
            *darks,
            *flats,
            *master_biases,
            *master_darks,
            *master_flats,
            *passed,
        )
        selected_source_digests = {
            identity_by_path[str(path.resolve(strict=True))].sha256
            for path in pixel_source_paths
        }
        pipeline_parameters = replace(
            request.pipeline_parameters,
            raw_frame_metadata_overrides=tuple(
                override
                for override in request.pipeline_parameters.raw_frame_metadata_overrides
                if override.source_sha256 in selected_source_digests
            ),
            # Ordinary integration consumes calibrated Lights in memory; only
            # Drizzle reads them back from the pipeline directory.
            materialize_calibrated_lights=(
                request.integration_mode is IntegrationMode.DRIZZLE
            ),
            capture_drizzle_inputs=(
                request.integration_mode is IntegrationMode.DRIZZLE
            ),
            # The whole pipeline directory lives in the transient work tree;
            # the promoted products below are fsynced by this run.
            durable_intermediates=False,
        )
        # The inventory already hashed every original through one read; hand
        # those path/stat-bound digests to the pixel pipeline so it never
        # rereads a source only to recompute a digest it must then verify.
        pixel_identity_seed = {
            str(path.resolve(strict=True)): (
                identity_by_path[str(path.resolve(strict=True))].sha256,
                {
                    "sizeBytes": identity_by_path[str(path.resolve(strict=True))].size_bytes,
                    "mtimeNs": identity_by_path[str(path.resolve(strict=True))].mtime_ns,
                    "device": identity_by_path[str(path.resolve(strict=True))].device,
                    "inode": identity_by_path[str(path.resolve(strict=True))].inode,
                },
            )
            for path in pixel_source_paths
        }
        trusted_generated_calibration = None
        if (
            request.integration_mode is IntegrationMode.ORDINARY
            and bool(biases or darks or flats)
        ):
            trusted_source_identities = _trusted_source_identity_bindings(
                identity_by_path,
                pixel_source_paths,
            )
            trusted_generated_calibration = _capture_single_field_generated_calibration(
                plan=calibration_plan,
                generated_directory=work / "registration-calibration",
                upstream_receipt_path=registration_calibration_receipt_path,
                staged_inputs=staged_inputs,
                source_aliases=registration_source_aliases,
                pipeline_parameters=request.pipeline_parameters,
                consumer_source_groups=(
                    ("BIAS", biases),
                    ("DARK", darks),
                    ("FLAT", flats),
                    ("MASTER_BIAS", master_biases),
                    ("MASTER_DARK", master_darks),
                    ("MASTER_FLAT", master_flats),
                    ("LIGHT", passed),
                ),
                internal_source_identities=trusted_source_identities,
            )
        def _staged_pixel_maps(
            light_subset: Sequence[Path],
        ) -> tuple[dict[str, Any], dict[str, float], dict[str, StellarScaleHint]]:
            transforms_map = {
                str(staged): pipeline_transforms[
                    str(registration_source_aliases[str(staged)].resolve(strict=True))
                ]
                for staged in light_subset
            }
            weights_map = {
                str(staged): registration.quality_weights[
                    str(registration_source_aliases[str(staged)].resolve(strict=True))
                ]
                for staged in light_subset
            }
            light_by_original = {
                str(registration_source_aliases[str(staged)].resolve(strict=True)): staged
                for staged in light_subset
            }
            hints_map: dict[str, StellarScaleHint] = {}
            for original, staged in light_by_original.items():
                hint = registration.stellar_scale_hints[original]
                reference_staged = light_by_original.get(
                    str(Path(hint.reference_path).resolve(strict=True))
                )
                if reference_staged is None:
                    raise E2EError(
                        "STELLAR_SCALE_HINT_REFERENCE_MISMATCH",
                        "registration stellar scale reference is absent from the pixel Light set",
                        path=hint.reference_path,
                    )
                hints_map[str(staged)] = replace(
                    hint,
                    source_path=str(staged),
                    reference_path=str(reference_staged),
                )
            return transforms_map, weights_map, hints_map

        selection_observers: dict[str, LeaveOneOutAccumulator] = {}

        def _selection_observer_factory(
            filter_name: str, ordered_paths: Sequence[str]
        ) -> LeaveOneOutAccumulator | None:
            if not request.selection.unattended or request.selection.counterfactual != "analytic":
                return None
            originals = [
                str(registration_source_aliases[str(staged)].resolve(strict=True))
                for staged in ordered_paths
            ]
            accumulator = LeaveOneOutAccumulator(originals)
            selection_observers[filter_name] = accumulator
            return accumulator

        def _original_of(staged: Path) -> str:
            return str(registration_source_aliases[str(staged)].resolve(strict=True))

        result_by_original = {
            str(Path(frame.path).resolve(strict=True)): frame for frame in frame_results
        }
        measurement_by_original = {
            str(Path(item.metadata.path).resolve(strict=True)): item for item in measurements
        }

        def _registered_region_maps(light_subset: Sequence[Path]) -> dict[str, RegionWeightMap]:
            """Region maps resampled from the QC reference frame into the pipeline's.

            The QC grid lives in the QC reference's preview frame; registered
            Lights live in the pixel pipeline's reference frame.  For a Light
            ``f`` a registered pixel maps to the QC reference through
            ``S Q_f S^-1 T_f^-1`` (``T_f``: source to pipeline reference in
            native pixels, ``Q_f``: source to QC reference in preview pixels,
            ``S``: preview to native scale), so each frame's own matrices
            carry it across, including a meridian flip between nights.
            """

            registered: dict[str, RegionWeightMap] = {}
            for staged in light_subset:
                original = _original_of(staged)
                qc_map = selection_region_maps.get(original)
                if qc_map is None:
                    continue
                frame = result_by_original.get(original)
                measurement = measurement_by_original.get(original)
                transform = pipeline_transforms.get(original)
                if (
                    frame is None
                    or measurement is None
                    or transform is None
                    or frame.registration.matrix is None
                    or measurement.preview_scale_x is None
                    or measurement.preview_scale_y is None
                    or not measurement.metadata.width
                    or not measurement.metadata.height
                ):
                    continue
                try:
                    scale = np.diag(
                        [float(measurement.preview_scale_x), float(measurement.preview_scale_y), 1.0]
                    )
                    qc_matrix = np.asarray(frame.registration.matrix, dtype=np.float64)
                    pipeline_matrix = np.asarray(transform, dtype=np.float64)
                    if qc_matrix.shape != (3, 3) or pipeline_matrix.shape != (3, 3):
                        continue
                    composite = (
                        scale @ qc_matrix @ np.linalg.inv(scale) @ np.linalg.inv(pipeline_matrix)
                    )
                    registered[str(staged)] = qc_map.transformed(
                        composite,
                        int(measurement.metadata.height),
                        int(measurement.metadata.width),
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
            return registered

        light_subset: list[Path] = list(staged_inputs["LIGHT"])
        selection_reports: dict[str, CounterfactualReport] = {}
        selection_receipt_path: str | None = None
        selection_reintegration: dict[str, Any] | None = None
        reintegration_passes: list[dict[str, Any]] = []
        admitted_at_start = len(light_subset)
        pass_index = 0
        while True:
            pass_index += 1
            pass_root = pipeline_root if pass_index == 1 else work / f"{PIXEL_PIPELINE_DIRECTORY}-pass{pass_index}"
            staged_pipeline_transforms, staged_quality_weights, staged_stellar_scale_hints = (
                _staged_pixel_maps(light_subset)
            )
            selection_observers.clear()
            try:
                pipeline_result = _run_portable_pipeline_fits(
                    # The deepest level of the run's layout; see path_budget.
                    _staging_stem=PIXEL_PIPELINE_STAGING_STEM,
                    bias_files=staged_inputs["BIAS"],
                    dark_files=staged_inputs["DARK"],
                    flat_files=staged_inputs["FLAT"],
                    master_bias_file=(
                        staged_inputs["MASTER_BIAS"][0]
                        if staged_inputs["MASTER_BIAS"]
                        else None
                    ),
                    master_dark_files=staged_inputs["MASTER_DARK"],
                    master_flat_files=staged_inputs["MASTER_FLAT"],
                    light_files=light_subset,
                    output_directory=pass_root,
                    transforms=staged_pipeline_transforms,
                    quality_weights=staged_quality_weights,
                    stellar_scale_hints=staged_stellar_scale_hints,
                    parameters=pipeline_parameters,
                    _source_aliases=registration_source_aliases,
                    _xisf_conversions=xisf_conversions,
                    _trusted_generated_calibration=trusted_generated_calibration,
                    _source_identity_seed=pixel_identity_seed,
                    _integration_tile_observers=(
                        _selection_observer_factory if request.selection.unattended else None
                    ),
                    region_weight_maps=(
                        _registered_region_maps(light_subset) or None
                        if selection_region_maps
                        else None
                    ),
                )
            except CalibrationError as error:
                raise E2EError(error.code, str(error), path=error.path) from error
            if not request.selection.unattended:
                break
            fwhm_by_path = {item.path: item.fwhm_native for item in selection_features}
            selection_reports = {
                group_name: accumulator.finalize(
                    fwhm_by_frame=[fwhm_by_path.get(path) for path in accumulator.paths]
                )
                for group_name, accumulator in selection_observers.items()
            }
            annotated = list(selection_decisions)
            for report in selection_reports.values():
                annotated = annotate_with_counterfactual(annotated, report, request.selection)
            selection_decisions = annotated
            harmful = confirmed_harmful(selection_decisions)
            if (
                pass_index >= request.selection.max_integration_passes
                or request.selection.counterfactual_action != "exclude"
                or not harmful
            ):
                break
            # A confirmed-harmful frame is removed and its groups integrated once
            # more without it; frames that other frames' normalization hints
            # reference, and frames whose removal would leave a panel below the
            # registration minimum, stay (recorded as such).
            reference_originals = {
                str(Path(hint.reference_path).resolve(strict=True))
                for hint in registration.stellar_scale_hints.values()
            }
            removable: list[str] = []
            kept_reasons: dict[str, str] = {}
            for item in harmful:
                if item.path in reference_originals:
                    kept_reasons[item.path] = "normalization reference of its group"
                    continue
                removable.append(item.path)
            remaining_by_panel: dict[tuple[str, str], int] = {}
            for staged in light_subset:
                original = _original_of(staged)
                result_for = next(
                    (
                        frame
                        for frame in frame_results
                        if str(Path(frame.path).resolve(strict=True)) == original
                    ),
                    None,
                )
                if result_for is None:
                    continue
                key = (
                    result_for.metadata.target.strip().upper(),
                    result_for.metadata.filter_name.strip().upper(),
                )
                remaining_by_panel[key] = remaining_by_panel.get(key, 0) + (
                    0 if original in removable else 1
                )
            for item in harmful:
                if item.path not in removable:
                    continue
                result_for = next(
                    frame
                    for frame in frame_results
                    if str(Path(frame.path).resolve(strict=True)) == item.path
                )
                key = (
                    result_for.metadata.target.strip().upper(),
                    result_for.metadata.filter_name.strip().upper(),
                )
                if remaining_by_panel.get(key, 0) < 2:
                    removable.remove(item.path)
                    remaining_by_panel[key] = remaining_by_panel.get(key, 0) + 1
                    kept_reasons[item.path] = "fewer than 2 Lights would remain in its panel"
            # Cumulative counterfactual exclusions stay under the soft guard.
            already_removed = admitted_at_start - len(light_subset)
            allowed = int(request.selection.soft_exclusion_fraction_guard * admitted_at_start) - already_removed
            if len(removable) > max(0, allowed):
                for path in removable[max(0, allowed):]:
                    kept_reasons[path] = "cumulative exclusions would exceed the soft-exclusion guard"
                removable = removable[: max(0, allowed)]
            # The confirming counterfactual numbers travel with the record:
            # the next pass measures only the remaining frames.
            harmful_evidence = {
                item.path: item.counterfactual.serializable()
                for item in harmful
                if item.counterfactual is not None
            }
            if not removable:
                selection_reintegration = {
                    "status": "NOT_APPLIED",
                    "pass": pass_index,
                    "confirmedHarmful": [item.path for item in harmful],
                    "keptBecause": kept_reasons,
                    "evidence": harmful_evidence,
                }
                reintegration_passes.append(selection_reintegration)
                break
            selection_decisions = exclude_confirmed(selection_decisions, removable)
            removed_set = set(removable)
            light_subset = [staged for staged in light_subset if _original_of(staged) not in removed_set]
            passed = tuple(path for path in passed if str(path) not in removed_set)
            passed_set = set(passed)
            excluded = tuple(path for path in lights if path not in passed_set)
            if trusted_generated_calibration is not None:
                # The generated calibration set is bound to the exact pixel
                # input manifest, so it is captured again for the reduced set.
                trusted_generated_calibration = _capture_single_field_generated_calibration(
                    plan=calibration_plan,
                    generated_directory=work / "registration-calibration",
                    upstream_receipt_path=registration_calibration_receipt_path,
                    staged_inputs={**staged_inputs, "LIGHT": tuple(light_subset)},
                    source_aliases=registration_source_aliases,
                    pipeline_parameters=request.pipeline_parameters,
                    consumer_source_groups=(
                        ("BIAS", biases),
                        ("DARK", darks),
                        ("FLAT", flats),
                        ("MASTER_BIAS", master_biases),
                        ("MASTER_DARK", master_darks),
                        ("MASTER_FLAT", master_flats),
                        ("LIGHT", passed),
                    ),
                    internal_source_identities=trusted_source_identities,
                )
            screening = _screening_summary(
                frame_results, passed, approved_review_paths, review_previews
            )
            selection_reintegration = {
                "status": "APPLIED",
                "pass": pass_index,
                "excluded": sorted(removable),
                "keptBecause": kept_reasons,
                "evidence": harmful_evidence,
                "discardedPass": pass_root.name,
            }
            reintegration_passes.append(selection_reintegration)
            _emit(
                progress,
                ProgressStage.INTEGRATION,
                "running",
                f"{len(removable)} frame(s) measured harmful by the counterfactual; "
                "integrating again without them",
            )
            remove_tree(pass_root, ignore_errors=True)
        pipeline_root = pass_root
        if request.selection.unattended:
            selection_block = selection_receipt(
                request.selection,
                selection_features,
                selection_decisions,
                selection_reports,
                region_maps=selection_region_maps,
            )
            if selection_reintegration is not None:
                applied = [item for item in reintegration_passes if item.get("status") == "APPLIED"]
                kept: dict[str, str] = {}
                for item in reintegration_passes:
                    kept.update(item.get("keptBecause", {}))
                selection_block["reintegration"] = {
                    "status": "APPLIED" if applied else selection_reintegration["status"],
                    "excluded": sorted(path for item in applied for path in item.get("excluded", [])),
                    "keptBecause": kept,
                    "passes": reintegration_passes,
                }
            else:
                selection_block["reintegration"] = None
            selection_block["integrationPasses"] = pass_index
            _write_json(qc_dir / "selection.json", selection_block)
            selection_receipt_path = "qc/selection.json"
        shutil.copyfile(pipeline_result.receipt_path, receipts_dir / "pixel-pipeline.json")
        pixel_pipeline_receipt = json.loads(
            Path(pipeline_result.receipt_path).read_text(encoding="utf-8")
        )
        ordinary_executions = (
            {
                filter_name: group.get("integration", {}).get("execution", {})
                for filter_name, group in pixel_pipeline_receipt.get("statistics", {})
                .get("integrationGroups", {})
                .items()
            }
            if request.integration_mode is IntegrationMode.ORDINARY
            else {}
        )
        _emit(progress, ProgressStage.INTEGRATION, "completed", "linear UNSOLVED masters created")

        if request.integration_mode is IntegrationMode.DRIZZLE:
            _emit(progress, ProgressStage.DRIZZLE, "started", "executing per-filter drizzle")
            candidates, coverage = _drizzle_candidates(
                staging=staging,
                work=work,
                pipeline_result=pipeline_result,
                options=request.drizzle,
                sampling_evidence=drizzle_sampling,
                threads=None,
            )
            for filter_name in sorted(candidates):
                source_receipt = work / "drizzle" / _safe_token(filter_name) / "receipt.json"
                shutil.copyfile(source_receipt, receipts_dir / f"drizzle_{_safe_token(filter_name)}.json")
            _emit(progress, ProgressStage.DRIZZLE, "completed", f"drizzled {len(candidates)} filters")
        else:
            candidates, coverage = _ordinary_candidates(
                pipeline_result.master_light_paths,
                pipeline_root,
                staging,
                minimum_coverage_fraction=request.drizzle.minimum_coverage_fraction,
                maximum_null_fraction=request.drizzle.maximum_null_fraction,
            )
            coverage["localNormalizationEvidence"] = {"status": "DISABLED", "groups": {}}
        _write_json(coverage_dir / "coverage.json", coverage)

        solver_records: dict[str, Any] = {}
        product_paths_staged: list[Path] = []
        solved_products: dict[str, Path] = {}
        all_solved = True
        _emit(progress, ProgressStage.ASTROMETRY, "started", "solving every filter", total=len(candidates))
        ordered_candidates = sorted(candidates.items())
        solve_targets: dict[str, Path] = {}
        for filter_name, _candidate in ordered_candidates:
            token = _safe_token(filter_name)
            filter_dir = products_dir / token
            filter_dir.mkdir(parents=True, exist_ok=False)
            solve_targets[filter_name] = filter_dir / f"master_light_{token}_wcs.fits"

        solve_progress_lock = threading.Lock()
        solve_completed = 0

        def solve_filter(item: tuple[str, Path]) -> tuple[str, bool, list[dict[str, Any]]]:
            nonlocal solve_completed
            filter_name, candidate = item
            solved, attempts = _solve_one(
                input_path=candidate,
                output_path=solve_targets[filter_name],
                backends=solver_backends,
                hints=solver_hints,
                min_matches=request.min_matches,
                max_rms_arcsec=request.max_rms_arcsec,
            )
            with solve_progress_lock:
                solve_completed += 1
                _emit(
                    progress,
                    ProgressStage.ASTROMETRY,
                    "progress",
                    f"{filter_name}: {'SOLVED' if solved else 'UNSOLVED'}",
                    current=solve_completed,
                    total=len(candidates),
                )
            return filter_name, solved, attempts

        # Every filter solves in its own staging directory against read-only
        # backends, so the filters run concurrently; records keep filter order.
        solve_workers = max(1, min(request.workers, len(ordered_candidates)))
        if solve_workers == 1:
            solve_outcomes = [solve_filter(item) for item in ordered_candidates]
        else:
            with ThreadPoolExecutor(
                max_workers=solve_workers, thread_name_prefix="oaf-solve"
            ) as solve_pool:
                solve_outcomes = list(solve_pool.map(solve_filter, ordered_candidates))
        for filter_name, solved, attempts in solve_outcomes:
            candidate = candidates[filter_name]
            solved_path = solve_targets[filter_name]
            attempts = _relativize_solver_attempts(attempts, staging)
            solver_records[filter_name] = {
                "status": "SOLVED" if solved else "UNSOLVED",
                "input": str(candidate.relative_to(staging)),
                "output": str(solved_path.relative_to(staging)) if solved else None,
                "attempts": attempts,
            }
            all_solved = all_solved and solved
            if solved:
                product_paths_staged.append(solved_path)
                solved_products[filter_name] = solved_path

        # WCS tolerances are expressed in native pixels; drizzled masters have
        # ``scale`` pixels per native pixel, so the same angular agreement is
        # ``scale`` times as many of their own pixels.
        grid_pixels_per_native = (
            float(request.drizzle.scale)
            if request.integration_mode is IntegrationMode.DRIZZLE
            else 1.0
        )
        same_grid_unification: dict[str, Any] = {
            "status": "NOT_APPLICABLE",
            "reason": "single filter",
        }
        if all_solved and len(solved_products) > 1:
            same_grid_unification = _unify_same_grid_solutions(
                solved_products,
                solver_records=solver_records,
                tolerance_pixels=float(request.same_grid_wcs_tolerance_pixels)
                * grid_pixels_per_native,
                grid_signatures=_same_grid_signatures(
                    integration_mode=request.integration_mode,
                    pixel_pipeline_receipt=pixel_pipeline_receipt,
                    coverage=coverage,
                ),
            )
            if same_grid_unification["status"] == "MISMATCH":
                all_solved = False
        cross_filter_validation = (
            _validate_cross_filter_wcs(
                solved_products, tolerance_pixels=0.05 * grid_pixels_per_native
            )
            if all_solved
            else WcsValidation(
                False,
                str(same_grid_unification.get("code") or "CROSS_FILTER_WCS_SKIPPED"),
                str(
                    same_grid_unification.get("message")
                    or "one or more filter solutions failed their individual gates"
                ),
                {"sameGridUnification": same_grid_unification},
            )
        )
        all_solved = all_solved and cross_filter_validation.valid
        if not all_solved:
            _write_json(receipts_dir / "solver-attempts.json", {"filters": solver_records})
            _verify_sources(identities)
            result = _failure_result(
                staging=staging,
                failure_directory=failure_directory,
                code="ASTROMETRY_REQUIRED",
                message="one or more filters did not produce a verified new WCS solution",
                passed=passed,
                excluded=excluded,
                solver={
                    "hints": solver_hints.serializable(),
                    "qualityPolicy": {
                        "minMatches": request.min_matches,
                        "maxRmsArcsec": request.max_rms_arcsec,
                    },
                    "crossFilterValidation": cross_filter_validation.serializable(),
                    "sameGridUnification": same_grid_unification,
                    "filters": solver_records,
                },
                sources=identities,
                callback=progress,
            )
            published = True
            return result
        _emit(progress, ProgressStage.ASTROMETRY, "completed", "all filters have verified WCS")

        _emit(progress, ProgressStage.PREVIEW, "started", "rendering solved-master previews")
        preview_paths_staged: list[Path] = []
        for product in product_paths_staged:
            token = product.parent.name
            preview_path = previews_dir / f"master_light_{token}.png"
            render_auto_stretch_preview(
                product,
                preview_path,
                max_long_edge=request.pipeline_parameters.preview_max_long_edge,
                max_memory_bytes=request.pipeline_parameters.registration_memory_bytes,
            )
            preview_paths_staged.append(preview_path)
        _emit(progress, ProgressStage.PREVIEW, "completed", f"rendered {len(preview_paths_staged)} previews")

        _emit(progress, ProgressStage.VERIFY, "started", "verifying sources and final artifact identities")
        _verify_sources(identities)
        remove_tree(work)
        _sanitize_shareable_tree(staging, identities)
        artifacts = _artifact_records(
            staging,
            (products_dir, previews_dir, qc_dir, coverage_dir, receipts_dir),
        )
        receipt_core: dict[str, Any] = {
            "schemaVersion": 1,
            "pipelineVersion": E2E_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "success": True,
            "state": E2EState.SOLVED.value,
            "code": "E2E_SUCCEEDED",
            "calibrationPolicy": workflow_receipt(request.pipeline_parameters.calibration_workflow),
            "integrationMode": request.integration_mode.value,
            "sources": [item.serializable() for item in identities],
            "execution": {"sourceExtraction": dict(source_extraction)},
            "qualityControl": {
                "manifest": "qc/manifest.json",
                "passedLights": len(passed),
                "excludedLights": len(excluded),
                "screening": screening,
                "selectionPolicy": request.selection.policy,
                "selection": explicit_block if explicit_block is not None else selection_receipt_path,
                **(
                    {
                        "blink": {
                            "manifest": "qc/blink.json",
                            "flagsPolicyDigest": blink.flags_policy.canonical_digest(),
                            "referenceRule": next(iter(blink.references.values())).rule if blink.references else None,
                            "referenceBySource": {
                                channel_id: {
                                    "sourceSha256": reference.source_sha256,
                                    "path": reference.path,
                                    "candidacy": reference.candidacy,
                                }
                                for channel_id, reference in blink.references.items()
                            },
                            "pipelineReferences": {
                                "registration": registration.receipt.get("referencePath"),
                                "normalization": {
                                    filter_name: (
                                        (group.get("globalNormalization") or {}).get("referenceInput")
                                    )
                                    for filter_name, group in pixel_pipeline_receipt.get("statistics", {})
                                    .get("integrationGroups", {})
                                    .items()
                                },
                            },
                        }
                    }
                    if blink is not None
                    else {}
                ),
            },
            "registration": {
                "receipt": "receipts/registration.json",
                "fullMatrixConvention": "INPUT_TO_OUTPUT",
            },
            "integration": {
                "pixelPipelineReceipt": "receipts/pixel-pipeline.json",
                "coverage": "coverage/coverage.json",
                "ordinaryExecutions": ordinary_executions,
            },
            "astrometry": {
                "status": "SOLVED",
                "requiredForSuccess": True,
                "hints": solver_hints.serializable(),
                "qualityPolicy": {
                    "minMatches": request.min_matches,
                    "maxRmsArcsec": request.max_rms_arcsec,
                },
                "crossFilterValidation": cross_filter_validation.serializable(),
                "sameGridUnification": same_grid_unification,
                "filters": solver_records,
            },
            "artifacts": artifacts,
            "publication": {"atomic": True, "noReplace": True, "sourceMutation": False},
        }
        receipt_core = _share_safe_receipt_core(
            receipt_core, staging=staging, identities=identities
        )
        receipt_id = "sha256:" + hashlib.sha256(_canonical_json(receipt_core)).hexdigest()
        _write_json(staging / "receipt.json", {"receiptId": receipt_id, **receipt_core})
        _emit(progress, ProgressStage.VERIFY, "completed", f"verified {len(artifacts)} artifacts")
        _emit(progress, ProgressStage.PUBLISH, "started", "atomically publishing solved project")
        _rename_directory_no_replace(staging, output)
        _fsync_directory(output.parent)
        published = True
        # A UI callback must not be able to turn an already committed success
        # into an apparent processing failure.
        try:
            _emit(progress, ProgressStage.COMPLETE, "completed", "E2E project published")
        except Exception:
            pass
        return E2EResult(
            success=True,
            code="E2E_SUCCEEDED",
            state=E2EState.SOLVED,
            output_directory=str(output),
            evidence_directory=None,
            receipt_path=str(output / "receipt.json"),
            product_paths=tuple(str(output / path.relative_to(staging)) for path in product_paths_staged),
            preview_paths=tuple(str(output / path.relative_to(staging)) for path in preview_paths_staged),
            passed_light_paths=tuple(str(path) for path in passed),
            excluded_light_paths=tuple(str(path) for path in excluded),
        )
    except Exception:
        if not published:
            remove_tree(staging)
        try:
            _emit(progress, ProgressStage.FAILED, "failed", "E2E execution aborted without publishing success")
        except Exception:
            pass
        raise
    finally:
        # Idempotent: the success path closed the pool after registration;
        # a gate failure returned early with the pool still open.
        frame_runner.close()


__all__ = [
    "DrizzleOptions",
    "E2EError",
    "E2ERequest",
    "E2EResult",
    "E2EState",
    "E2E_VERSION",
    "ExplicitDecision",
    "ExplicitSelection",
    "IntegrationMode",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressStage",
    "ReviewApproval",
    "bind_review_approval_selections",
    "parse_explicit_selection",
    "run_e2e",
]
