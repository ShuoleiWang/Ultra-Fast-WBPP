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

from .calibration_policy import apply_mono_workflow, bias_from_header, MONO_STANDARD, can_omit_bias, conflicting_profile_fields, workflow_receipt

from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
import ctypes
import errno
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

from .quality_cache import quality_cache_directory

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from lightframeqc.analysis import analyze_measurements
from lightframeqc.config import DEFAULT_CONFIG, QcConfig
from lightframeqc.measure import measure_paths
from lightframeqc.models import FrameResult, GateDisposition
from lightframeqc.quality_gate import GatePolicy, evaluate_quality_gate
from lightframeqc.readers import probe_frame_metadata

from .astap_backend import verify_solver_execution_result
from .calibration import (
    CalibrationError,
    FrameExpression,
    FrameInfo,
    integrate_expressions,
    normalize_role,
    read_frame_info,
    robust_location,
)
from .drizzle_execution import (
    DrizzleExecutionRequest,
    DrizzleFrameInput,
    DrizzleProvider,
    execute_drizzle,
    verify_drizzle_result,
)
from .pixel_pipeline import (
    PipelineParameters,
    _InternalSourceIdentity,
    _apply_master_metadata_overrides,
    _assert_compatible,
    _find_dark,
    _master_dark_bias_semantics,
    _numeric_application_scale,
    _numeric_domain_metadata,
    _require_filter,
    _capture_trusted_generated_calibration_set,
    _run_portable_pipeline_fits,
)
from .local_normalization import normalize_registered_group
from .global_normalization import StellarScaleHint
from .xisf_pixels import convert_xisf_to_fits, preflight_xisf_header
from .preview import render_auto_stretch_preview
from .solver import (
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


class IntegrationMode(StrEnum):
    ORDINARY = "ordinary"
    DRIZZLE = "drizzle"


class E2EState(StrEnum):
    SOLVED = "SOLVED"
    UNSOLVED_WORKING = "UNSOLVED_WORKING"


class ProgressStage(StrEnum):
    INVENTORY = "inventory"
    QUALITY_CONTROL = "quality-control"
    CALIBRATION = "calibration"
    REGISTRATION = "registration"
    INTEGRATION = "integration"
    DRIZZLE = "drizzle"
    ASTROMETRY = "astrometry"
    PREVIEW = "preview"
    VERIFY = "verify"
    PUBLISH = "publish"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: ProgressStage
    status: str
    current: int = 0
    total: int = 0
    message: str = ""

    def serializable(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "message": self.message,
        }


ProgressCallback = Callable[[ProgressEvent], None]


class E2EError(RuntimeError):
    """Stable fail-closed error for invalid or incomplete E2E execution."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        self.code = code
        self.path = path
        detail = f"{path}: {message}" if path else message
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ReviewApproval:
    """One explicit, content- and request-bound REVIEW admission."""

    source_sha256: str
    gate_policy_digest: str
    request_digest: str

    def serializable(self) -> dict[str, str]:
        return {
            "sourceSha256": self.source_sha256,
            "gatePolicyDigest": self.gate_policy_digest,
            "requestDigest": self.request_digest,
        }


@dataclass(frozen=True, slots=True)
class DrizzleOptions:
    scale: int = 2
    pixfrac: float = 0.8
    kernel: str = "square"
    tile_rows: int = 256
    max_tile_bytes: int = 256 * 1024**2
    max_output_pixels: int = 128 * 1024**2
    max_working_set_bytes: int = 4 * 1024**3
    minimum_coverage_fraction: float = 0.90
    maximum_null_fraction: float = 0.10
    minimum_distinct_dither_phases: int = 3
    minimum_dither_phase_separation_pixels: float = 0.15
    minimum_dither_span_pixels: float = 0.35
    maximum_fwhm_for_upsampling_pixels: float = 3.0
    rejection_sigma: float = 6.0
    rejection_minimum_frames: int = 3

    def validate(self) -> None:
        if isinstance(self.scale, bool) or self.scale not in {1, 2, 3}:
            raise E2EError("DRIZZLE_SCALE_INVALID", "drizzle scale must be 1, 2, or 3")
        if not math.isfinite(self.pixfrac) or not 0.1 <= self.pixfrac <= 1.0:
            raise E2EError("DRIZZLE_PIXFRAC_INVALID", "pixfrac must be in [0.1, 1]")
        if self.kernel not in {
            "square",
            "point",
            "turbo",
            "gaussian",
            "lanczos2",
            "lanczos3",
        }:
            raise E2EError("DRIZZLE_KERNEL_INVALID", "unsupported drizzle kernel")
        if self.kernel.startswith("lanczos") and (
            self.scale != 1 or not math.isclose(self.pixfrac, 1.0)
        ):
            raise E2EError(
                "DRIZZLE_KERNEL_GEOMETRY_UNSAFE",
                "Lanczos requires scale=1 and pixfrac=1",
            )
        for name in (
            "tile_rows",
            "max_tile_bytes",
            "max_output_pixels",
            "max_working_set_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise E2EError("DRIZZLE_LIMIT_INVALID", f"{name} must be positive")
        for name in ("minimum_coverage_fraction", "maximum_null_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise E2EError(
                    "DRIZZLE_GATE_INVALID", f"{name} must be in [0, 1]"
                )
        if self.minimum_coverage_fraction < 0.90:
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "minimum_coverage_fraction cannot be below the production floor 0.90",
            )
        if self.maximum_null_fraction > 0.10:
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "maximum_null_fraction cannot exceed the production ceiling 0.10",
            )
        if (
            isinstance(self.minimum_distinct_dither_phases, bool)
            or not isinstance(self.minimum_distinct_dither_phases, int)
            or self.minimum_distinct_dither_phases < 3
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "minimum_distinct_dither_phases must be at least 3",
            )
        if not (
            math.isfinite(self.minimum_dither_phase_separation_pixels)
            and 0.0 < self.minimum_dither_phase_separation_pixels <= math.sqrt(0.5)
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "minimum dither phase separation must be in (0, sqrt(0.5)]",
            )
        if not (
            math.isfinite(self.minimum_dither_span_pixels)
            and 0.0 < self.minimum_dither_span_pixels <= 0.5
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "minimum dither span must be in (0, 0.5]",
            )
        if not (
            math.isfinite(self.maximum_fwhm_for_upsampling_pixels)
            and self.maximum_fwhm_for_upsampling_pixels > 0
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "maximum_fwhm_for_upsampling_pixels must be positive",
            )
        if self.maximum_fwhm_for_upsampling_pixels > 3.0:
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "maximum_fwhm_for_upsampling_pixels cannot exceed the production ceiling 3.0",
            )
        if not math.isfinite(self.rejection_sigma) or self.rejection_sigma < 3.0:
            raise E2EError(
                "DRIZZLE_GATE_INVALID", "rejection_sigma must be at least 3"
            )
        if (
            isinstance(self.rejection_minimum_frames, bool)
            or not isinstance(self.rejection_minimum_frames, int)
            or self.rejection_minimum_frames < 3
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "rejection_minimum_frames must be at least 3",
            )

    def serializable(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "pixfrac": self.pixfrac,
            "kernel": self.kernel,
            "tileRows": self.tile_rows,
            "maxTileBytes": self.max_tile_bytes,
            "maxOutputPixels": self.max_output_pixels,
            "maxWorkingSetBytes": self.max_working_set_bytes,
            "minimumCoverageFraction": self.minimum_coverage_fraction,
            "maximumNullFraction": self.maximum_null_fraction,
            "minimumDistinctDitherPhases": self.minimum_distinct_dither_phases,
            "minimumDitherPhaseSeparationPixels": self.minimum_dither_phase_separation_pixels,
            "minimumDitherSpanPixels": self.minimum_dither_span_pixels,
            "maximumFwhmForUpsamplingPixels": self.maximum_fwhm_for_upsampling_pixels,
            "rejectionSigma": self.rejection_sigma,
            "rejectionMinimumFrames": self.rejection_minimum_frames,
        }


@dataclass(frozen=True, slots=True)
class E2ERequest:
    light_files: tuple[str, ...]
    flat_files: tuple[str, ...]
    bias_files: tuple[str, ...]
    output_directory: str
    dark_files: tuple[str, ...] = ()
    master_bias_files: tuple[str, ...] = ()
    master_dark_files: tuple[str, ...] = ()
    master_flat_files: tuple[str, ...] = ()
    review_approvals: tuple[ReviewApproval, ...] = ()
    recipe_digest: str | None = None
    integration_mode: IntegrationMode = IntegrationMode.ORDINARY
    workers: int = 1
    qc_config: QcConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    gate_policy: GatePolicy = field(default_factory=GatePolicy)
    pipeline_parameters: PipelineParameters = field(default_factory=PipelineParameters)
    drizzle: DrizzleOptions = field(default_factory=DrizzleOptions)
    ra_hint_degrees: float | None = None
    dec_hint_degrees: float | None = None
    field_of_view_degrees: float | None = None
    search_radius_degrees: float | None = None
    min_matches: int = 12
    max_rms_arcsec: float = 2.0
    registration_detection: Any = field(default=None, repr=False, compare=False)
    registration_config: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class E2EResult:
    success: bool
    code: str
    state: E2EState
    output_directory: str | None
    evidence_directory: str | None
    receipt_path: str
    product_paths: tuple[str, ...]
    preview_paths: tuple[str, ...]
    passed_light_paths: tuple[str, ...]
    excluded_light_paths: tuple[str, ...]
    message: str | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "code": self.code,
            "state": self.state.value,
            "outputDirectory": self.output_directory,
            "evidenceDirectory": self.evidence_directory,
            "receiptPath": self.receipt_path,
            "productPaths": list(self.product_paths),
            "previewPaths": list(self.preview_paths),
            "passedLightPaths": list(self.passed_light_paths),
            "excludedLightPaths": list(self.excluded_light_paths),
            "message": self.message,
        }


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


@dataclass(frozen=True, slots=True)
class _SolverHints:
    ra_degrees: float | None
    dec_degrees: float | None
    field_of_view_degrees: float | None
    search_radius_degrees: float | None
    provenance: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "raDegrees": self.ra_degrees,
            "decDegrees": self.dec_degrees,
            "fieldOfViewDegrees": self.field_of_view_degrees,
            "searchRadiusDegrees": self.search_radius_degrees,
            "provenance": self.provenance,
            "evidence": self.evidence,
        }


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
                    staged.unlink(missing_ok=True)
                    raise E2EError(
                        "SOURCE_CHANGED",
                        "XISF conversion does not match the captured source identity",
                        path=str(path),
                    )
                conversions.append({"role": role, **receipt.serializable()})
                staged_digests[str(staged)] = receipt.converted_sha256
            elif expected is not None:
                staged = directory / f"{sequence:06d}_{role.casefold()}.fits"
                staged_digests[str(staged)] = _copy_fits_source_snapshot(
                    path, staged, expected
                )
            else:
                staged = path
            staged_paths.append(staged)
            aliases[str(staged)] = path
        staged_groups[role] = tuple(staged_paths)
    return staged_groups, aliases, conversions, staged_digests


def _copy_fits_source_snapshot(
    source: Path, destination: Path, expected: _SourceIdentity
) -> str:
    """Copy and hash one captured FITS identity through a single open handle."""

    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            before = os.fstat(input_stream.fileno())
            before_identity = (
                before.st_size,
                before.st_mtime_ns,
                before.st_dev,
                before.st_ino,
            )
            expected_identity = (
                expected.size_bytes,
                expected.mtime_ns,
                expected.device,
                expected.inode,
            )
            if before_identity != expected_identity:
                raise E2EError(
                    "SOURCE_CHANGED",
                    "FITS source changed before private snapshot",
                    path=str(source),
                )
            while block := input_stream.read(4 * 1024 * 1024):
                output_stream.write(block)
                digest.update(block)
                copied += len(block)
            output_stream.flush()
            after = os.fstat(input_stream.fileno())
            after_identity = (
                after.st_size,
                after.st_mtime_ns,
                after.st_dev,
                after.st_ino,
            )
            actual_digest = "sha256:" + digest.hexdigest()
            if (
                after_identity != before_identity
                or copied != expected.size_bytes
                or actual_digest != expected.sha256
            ):
                raise E2EError(
                    "SOURCE_CHANGED",
                    "FITS source changed while creating its private snapshot",
                    path=str(source),
                )
        return actual_digest
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _verify_staged_pixel_inputs(staged_digests: Mapping[str, str]) -> None:
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


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


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
    staging: Path,
    source_tokens: Mapping[str, str],
) -> str:
    """Redact host paths while retaining an auditable opaque source token."""

    result = value
    for private, public in sorted(source_tokens.items(), key=lambda item: -len(item[0])):
        result = result.replace(private, public)
    staging_text = str(staging)
    result = result.replace(staging_text + os.sep, "artifact/")
    if result == staging_text:
        return "artifact/."
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
        os.replace(temporary, path)


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


def _verify_sources(identities: Sequence[_SourceIdentity]) -> None:
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
        if actual != expected or _sha256(path) != identity.sha256:
            raise E2EError("SOURCE_CHANGED", "source identity changed during execution", path=identity.path)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    if not token:
        raise E2EError("FILTER_INVALID", "filter name cannot be encoded safely")
    return token


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise E2EError("OUTPUT_EXISTS", "refusing to replace output directory", path=str(destination))
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as error:
            raise E2EError("OUTPUT_EXISTS", "output appeared during publication", path=str(destination)) from error
        return
    library = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        result = library.renamex_np(encoded_source, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        result = library.renameat2(-100, encoded_source, -100, encoded_destination, 0x00000001)
    else:
        raise E2EError(
            "ATOMIC_DIRECTORY_PUBLISH_UNSUPPORTED",
            "platform has no no-replace directory rename primitive",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise E2EError("OUTPUT_EXISTS", "output appeared during publication", path=str(destination))
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise
    finally:
        os.close(descriptor)


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


def _qc_manifest(
    staging: Path,
    groups: Sequence[Mapping[str, Any]],
    results: Sequence[FrameResult],
    config: QcConfig,
    policy: GatePolicy,
) -> dict[str, Any]:
    frame_payloads: list[dict[str, Any]] = []
    for result in results:
        payload = result.serializable()
        thumbnail = payload.get("thumbnailPath")
        if isinstance(thumbnail, str):
            try:
                payload["thumbnailPath"] = str(Path(thumbnail).relative_to(staging))
            except ValueError:
                payload["thumbnailPath"] = None
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
    if options.scale == 1:
        evidence["status"] = "NOT_APPLICABLE"
        return evidence
    if median_fwhm is None:
        evidence["status"] = "REVIEW"
        raise E2EError(
            "DRIZZLE_SAMPLING_REVIEW_REQUIRED",
            "QC could not establish native PSF sampling; refusing to assume that "
            f"{options.scale}x drizzle is beneficial",
        )
    if (
        direct_median is not None
        and hfr_median is not None
        and (direct_median >= options.maximum_fwhm_for_upsampling_pixels)
        != (hfr_median >= options.maximum_fwhm_for_upsampling_pixels)
    ):
        evidence["status"] = "REVIEW_CONFLICTING_SAMPLING"
        raise E2EError(
            "DRIZZLE_SAMPLING_REVIEW_REQUIRED",
            "QC PSF FWHM and NINA HFR disagree across the adequately-sampled "
            "boundary; refusing automatic 2x/3x drizzle",
        )
    if median_fwhm >= options.maximum_fwhm_for_upsampling_pixels:
        evidence["status"] = "BLOCKED_WELL_SAMPLED"
        raise E2EError(
            "DRIZZLE_UPSCALE_NOT_RECOMMENDED",
            f"QC median native FWHM is {median_fwhm:.3f} px, at or above the "
            f"{options.maximum_fwhm_for_upsampling_pixels:.3f} px adequately-sampled threshold",
        )
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
        # Refine the preview bootstrap against full-resolution centroids.  The
        # ordinary pixel pipeline consumes affine source->reference matrices;
        # Drizzle can retain the more general projective model.
        selected_registration = registration or RegistrationConfig(
            refine_full_centroids=True,
            full_transform_model="projective" if allow_projective else "affine",
        )
        run = run_registration(
            [str(path) for path in lights],
            detection=detection,
            registration=selected_registration,
            calibration=calibration_plan,
            validate_warp=True,
            workers=workers,
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
    weights = normalize_quality_weights(run.analyses)
    quality_weights = {
        str(
            (source_aliases or {}).get(
                str(Path(analysis.path).resolve(strict=True)),
                Path(analysis.path).resolve(strict=True),
            )
        ): float(weight)
        for analysis, weight in zip(run.analyses, weights, strict=True)
    }
    scale_estimates = estimate_stellar_scale_hints(
        run.analyses,
        run.transforms,
        weights,
    )

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
        "autocrop": list(run.autocrop),
        "qualityWeights": list(weights),
        "qualityWeightsBySource": quality_weights,
        "stellarScaleHints": [
            stellar_scale_hints[path].serializable()
            for path in sorted(stellar_scale_hints)
        ],
        "timingSeconds": {
            "analysisWall": run.analysis_wall_seconds,
            "registrationWall": run.registration_wall_seconds,
            "autocropWall": run.autocrop_wall_seconds,
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


def _pipeline_calibrated_paths(pipeline_root: Path) -> dict[str, Path]:
    receipt = json.loads((pipeline_root / "receipt.json").read_text(encoding="utf-8"))
    result: dict[str, Path] = {}
    for artifact in receipt.get("outputs", []):
        if artifact.get("kind") != "CALIBRATED_LIGHT":
            continue
        details = artifact.get("details", {})
        source = details.get("source")
        relative = artifact.get("path")
        if not isinstance(source, str) or not isinstance(relative, str):
            raise E2EError("PIPELINE_RECEIPT_INVALID", "calibrated artifact lacks source binding")
        canonical_source = str(Path(source).resolve(strict=True))
        calibrated = (pipeline_root / relative).resolve(strict=True)
        try:
            calibrated.relative_to(pipeline_root.resolve(strict=True))
        except ValueError as error:
            raise E2EError("PIPELINE_RECEIPT_INVALID", "artifact escapes pipeline directory") from error
        result[canonical_source] = calibrated
    return result


def _drizzle_output_to_input_matrix(
    input_to_reference: Sequence[Sequence[float]], scale: int
) -> tuple[tuple[float, float, float], ...]:
    matrix = np.asarray(input_to_reference, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise E2EError("REGISTRATION_MATRIX_INVALID", "drizzle transform must be finite 3x3")
    norm = float(np.linalg.norm(matrix, ord=np.inf))
    determinant = float(np.linalg.det(matrix))
    if norm == 0.0 or not math.isfinite(determinant) or abs(determinant) <= 1e-12 * norm**3:
        raise E2EError("REGISTRATION_MATRIX_INVALID", "drizzle transform is singular")
    scale_to_base = np.diag((1.0 / scale, 1.0 / scale, 1.0))
    # input -> reference is M; high-resolution output -> reference is S^-1.
    # Therefore output -> input is M^-1 S^-1.  Keep the full homography.
    output_to_input = np.linalg.inv(matrix) @ scale_to_base
    return tuple(tuple(float(value) for value in row) for row in output_to_input)


def _projective_coordinates(
    matrix: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    safe = np.where(valid, denominator, 1.0)
    mapped_x = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / safe
    mapped_y = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / safe
    valid &= np.isfinite(mapped_x) & np.isfinite(mapped_y)
    return mapped_x, mapped_y, valid


def _bilinear_sample(
    image: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= width - 1.0)
        & (y >= 0.0)
        & (y <= height - 1.0)
    )
    safe_x = np.clip(np.where(inside, x, 0.0), 0.0, width - 1.0)
    safe_y = np.clip(np.where(inside, y, 0.0), 0.0, height - 1.0)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = safe_x - x0
    ty = safe_y - y0
    p00 = np.asarray(image[y0, x0], dtype=np.float32)
    p10 = np.asarray(image[y0, x1], dtype=np.float32)
    p01 = np.asarray(image[y1, x0], dtype=np.float32)
    p11 = np.asarray(image[y1, x1], dtype=np.float32)
    finite = (
        inside
        & np.isfinite(p00)
        & np.isfinite(p10)
        & np.isfinite(p01)
        & np.isfinite(p11)
    )
    sampled = (
        p00 * (1.0 - tx) * (1.0 - ty)
        + p10 * tx * (1.0 - ty)
        + p01 * (1.0 - tx) * ty
        + p11 * tx * ty
    ).astype(np.float32, copy=False)
    sampled[~finite] = np.nan
    return sampled, finite


def _mask_sample_any(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= width - 1.0)
        & (y >= 0.0)
        & (y <= height - 1.0)
    )
    safe_x = np.clip(np.where(inside, x, 0.0), 0.0, width - 1.0)
    safe_y = np.clip(np.where(inside, y, 0.0), 0.0, height - 1.0)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    selected = (
        (mask[y0, x0] != 0)
        | (mask[y0, x1] != 0)
        | (mask[y1, x0] != 0)
        | (mask[y1, x1] != 0)
    )
    return inside & selected


def _frame_normalization(
    image: np.ndarray, photometric_scale: float
) -> tuple[float, float, float]:
    maximum_samples = 262_144
    stride = max(1, int(math.ceil(math.sqrt(image.size / maximum_samples))))
    sampled = np.asarray(image[::stride, ::stride], dtype=np.float64).reshape(-1)
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size < 16:
        raise E2EError(
            "DRIZZLE_REJECTION_DATA_INVALID",
            "calibrated frame has too few finite pixels for robust rejection",
        )
    background = float(np.median(sampled))
    noise = float(1.4826 * np.median(np.abs(sampled - background)))
    if not math.isfinite(noise) or noise <= 0:
        noise = max(abs(background) * 1e-6, np.finfo(np.float32).eps)
    scale = float(photometric_scale)
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    return background, noise, scale


def _build_drizzle_rejection_masks(
    *,
    calibrated: Mapping[str, Path],
    transforms: Mapping[str, Sequence[Sequence[float]]],
    paths: Sequence[Path],
    analysis_by_path: Mapping[str, tuple[int, Any]],
    reference_shape: tuple[int, int],
    directory: Path,
    options: DrizzleOptions,
    source_exposure_seconds: Mapping[str, float],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Create independent per-frame robust rejection masks with bounded tiles.

    Pass one registers each calibrated frame into native reference tiles and
    computes a cross-frame median/MAD decision.  Pass two projects each
    frame's reference-space decisions back to its own detector pixels, which
    is the coordinate system required by ``DrizzleFrameInput``.
    """

    if len(paths) < options.rejection_minimum_frames:
        raise E2EError(
            "DRIZZLE_REJECTION_BASELINE_INSUFFICIENT",
            f"filter has {len(paths)} frames; robust rejection requires at least "
            f"{options.rejection_minimum_frames}",
        )
    directory.mkdir(parents=True, exist_ok=False)
    canonical_paths = [str(path.resolve(strict=True)) for path in paths]
    matrices: list[np.ndarray] = []
    inverse_matrices: list[np.ndarray] = []
    fluxes: list[float] = []
    exposures: list[float] = []
    for canonical in canonical_paths:
        if canonical not in transforms or canonical not in calibrated:
            raise E2EError(
                "PIPELINE_RECEIPT_INVALID",
                "rejection mask input lacks calibrated data or registration",
                path=canonical,
            )
        matrix = np.asarray(transforms[canonical], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise E2EError(
                "REGISTRATION_MATRIX_INVALID", "mask transform is not finite 3x3"
            )
        matrices.append(matrix)
        inverse_matrices.append(np.linalg.inv(matrix))
        _, analysis = analysis_by_path[canonical]
        flux = float(getattr(analysis, "total_integrated_flux", float("nan")))
        fluxes.append(flux if math.isfinite(flux) and flux > 0 else float("nan"))
        exposure = float(source_exposure_seconds.get(canonical, float("nan")))
        if not math.isfinite(exposure) or exposure <= 0:
            raise E2EError(
                "LIGHT_EXPOSURE_UNKNOWN",
                "drizzle rejection requires the original positive Light exposure",
                path=canonical,
            )
        exposures.append(exposure)
    # Calibrated pixel data has already been multiplied by
    # referenceExposure/sourceExposure.  Registration flux still describes
    # the raw exposure, so only flux/exposure estimates residual throughput.
    # Dividing by raw flux here would apply the exposure term twice and mark a
    # scientifically identical 60/300/300 second stack as an outlier.
    flux_rates = [
        flux / exposure if math.isfinite(flux) and flux > 0 else float("nan")
        for flux, exposure in zip(fluxes, exposures, strict=True)
    ]
    finite_rates = [value for value in flux_rates if math.isfinite(value) and value > 0]
    rate_center = float(np.median(finite_rates)) if finite_rates else 1.0
    photometric_scales = [
        min(max(value / rate_center, 0.25), 4.0)
        if math.isfinite(value) and value > 0
        else 1.0
        for value in flux_rates
    ]

    mask_paths: dict[str, Path] = {}
    mask_counts: dict[str, int] = {}
    normalizations: list[tuple[float, float, float]] = []
    reference_height, reference_width = reference_shape
    with ExitStack() as stack:
        images: list[np.ndarray] = []
        for index, canonical in enumerate(canonical_paths):
            try:
                hdul = stack.enter_context(
                    fits.open(
                        calibrated[canonical],
                        mode="readonly",
                        memmap=True,
                        lazy_load_hdus=False,
                    )
                )
            except (OSError, ValueError, TypeError) as error:
                raise E2EError(
                    "DRIZZLE_REJECTION_DATA_INVALID",
                    f"cannot open calibrated frame: {error}",
                    path=str(calibrated[canonical]),
                ) from error
            data = hdul[0].data
            if data is None or data.ndim != 2 or min(data.shape) < 2:
                raise E2EError(
                    "DRIZZLE_REJECTION_DATA_INVALID",
                    "calibrated frame must be a two-dimensional image",
                    path=str(calibrated[canonical]),
                )
            image = np.asanyarray(data)
            images.append(image)
            normalizations.append(
                _frame_normalization(image, photometric_scales[index])
            )

        frame_count = len(images)
        bytes_per_reference_row = max(
            reference_width * (frame_count * 12 + 64), 1
        )
        if bytes_per_reference_row > options.max_tile_bytes:
            raise E2EError(
                "DRIZZLE_REJECTION_TILE_LIMIT",
                "one robust-rejection tile row exceeds max_tile_bytes; reduce "
                "the frame group or raise the explicit memory limit",
            )
        tile_rows = max(
            1,
            min(
                options.tile_rows,
                int(options.max_tile_bytes // bytes_per_reference_row),
            ),
        )
        normalized_noises = [noise / scale for _, noise, scale in normalizations]
        noise_floor = max(
            float(np.median(normalized_noises)), np.finfo(np.float32).eps
        )

        with tempfile.TemporaryDirectory(
            prefix=".rejection-mask-work-", dir=directory
        ) as temporary_name:
            temporary = Path(temporary_name)
            reference_masks = [
                np.memmap(
                    temporary / f"reference-{index:05d}.u8",
                    dtype=np.uint8,
                    mode="w+",
                    shape=reference_shape,
                )
                for index in range(frame_count)
            ]
            for row_start in range(0, reference_height, tile_rows):
                row_stop = min(reference_height, row_start + tile_rows)
                y_reference, x_reference = np.indices(
                    (row_stop - row_start, reference_width), dtype=np.float64
                )
                y_reference += row_start
                registered_values: list[np.ndarray] = []
                registered_valid: list[np.ndarray] = []
                for image, inverse, normalization in zip(
                    images, inverse_matrices, normalizations, strict=True
                ):
                    input_x, input_y, transform_valid = _projective_coordinates(
                        inverse, x_reference, y_reference
                    )
                    sampled, sample_valid = _bilinear_sample(image, input_x, input_y)
                    background, _, scale = normalization
                    normalized = (sampled - background) / scale
                    valid = transform_valid & sample_valid & np.isfinite(normalized)
                    normalized[~valid] = np.nan
                    registered_values.append(normalized.astype(np.float32, copy=False))
                    registered_valid.append(valid)
                stack_values = np.stack(registered_values, axis=0)
                valid_stack = np.stack(registered_valid, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    center = np.nanmedian(stack_values, axis=0)
                    absolute_deviation = np.abs(stack_values - center[None, ...])
                    robust_sigma = 1.4826 * np.nanmedian(
                        absolute_deviation, axis=0
                    )
                threshold = options.rejection_sigma * np.maximum(
                    robust_sigma, noise_floor
                ) + _DRIZZLE_REJECTION_RELATIVE_SIGNAL_FLOOR * np.abs(center)
                gradient_y = (
                    np.gradient(center, axis=0)
                    if center.shape[0] > 1
                    else np.zeros_like(center)
                )
                gradient_x = np.gradient(center, axis=1)
                threshold += _DRIZZLE_REJECTION_GRADIENT_FLOOR * (
                    np.abs(gradient_x) + np.abs(gradient_y)
                )
                baseline_valid = (
                    np.count_nonzero(valid_stack, axis=0)
                    >= options.rejection_minimum_frames
                )
                outliers = (
                    valid_stack
                    & baseline_valid[None, ...]
                    & (absolute_deviation > threshold[None, ...])
                )
                for index, reference_mask in enumerate(reference_masks):
                    reference_mask[row_start:row_stop] = outliers[index].astype(
                        np.uint8, copy=False
                    )
            for reference_mask in reference_masks:
                reference_mask.flush()

            for index, (canonical, image, matrix, reference_mask) in enumerate(
                zip(
                    canonical_paths,
                    images,
                    matrices,
                    reference_masks,
                    strict=True,
                )
            ):
                mask_path = directory / f"{index + 1:05d}_rejection.fits"
                raw_input_mask = np.memmap(
                    temporary / f"input-{index:05d}.u8",
                    dtype=np.uint8,
                    mode="w+",
                    shape=image.shape,
                )
                rejected = 0
                for row_start in range(0, image.shape[0], tile_rows):
                    row_stop = min(image.shape[0], row_start + tile_rows)
                    y_input, x_input = np.indices(
                        (row_stop - row_start, image.shape[1]), dtype=np.float64
                    )
                    y_input += row_start
                    reference_x, reference_y, transform_valid = (
                        _projective_coordinates(matrix, x_input, y_input)
                    )
                    selected = transform_valid & _mask_sample_any(
                        reference_mask, reference_x, reference_y
                    )
                    raw_input_mask[row_start:row_stop] = selected.astype(
                        np.uint8, copy=False
                    )
                    rejected += int(np.count_nonzero(selected))
                raw_input_mask.flush()
                header = fits.Header()
                header["OAFPROD"] = ("REJECTION_MASK", "Ultra-Fast WBPP product role")
                header["OAFALGO"] = ("TILED_MAD_SIGMA", "Independent rejection method")
                header["OAFSIGMA"] = (options.rejection_sigma, "Sigma threshold")
                header["NREJECT"] = (
                    rejected,
                    "Pixels rejected by independent robust stack",
                )
                compressed = fits.CompImageHDU(
                    data=raw_input_mask,
                    header=header,
                    name="MASK",
                    compression_type="RICE_1",
                    tile_shape=(min(tile_rows, image.shape[0]), image.shape[1]),
                )
                mask_hdul = fits.HDUList([fits.PrimaryHDU(), compressed])
                try:
                    mask_hdul.writeto(mask_path, overwrite=False, checksum=True)
                finally:
                    mask_hdul.close()
                mapped_input = getattr(raw_input_mask, "_mmap", None)
                if mapped_input is not None:
                    mapped_input.close()
                del compressed, mask_hdul, raw_input_mask
                with mask_path.open("r+b") as stream:
                    os.fsync(stream.fileno())
                mask_paths[canonical] = mask_path
                mask_counts[canonical] = rejected
            while reference_masks:
                reference_mask = reference_masks.pop()
                mapped_file = getattr(reference_mask, "_mmap", None)
                if mapped_file is not None:
                    mapped_file.close()
                del reference_mask
            # Windows will not remove a mapped file while even a closed mmap
            # wrapper remains reachable. Drop every wrapper before the
            # TemporaryDirectory cleanup attempts to unlink its work files.
            gc.collect()

    manifest = {
        "schemaVersion": 1,
        "stage": "drizzle-rejection-masks",
        "algorithm": {
            "id": "TILED_REGISTERED_MEDIAN_MAD_SIGMA",
            "sigma": options.rejection_sigma,
            "minimumFrames": options.rejection_minimum_frames,
            "tileRows": tile_rows,
            "photometricNormalization": "registration-catalog-flux-per-source-exposure",
            "calibratedPixelsAlreadyExposureNormalized": True,
            "relativeSignalFloor": _DRIZZLE_REJECTION_RELATIVE_SIGNAL_FLOOR,
            "gradientFloor": _DRIZZLE_REJECTION_GRADIENT_FLOOR,
            "maskCompression": "FITS_RICE_1",
            "maskHdu": "MASK",
        },
        "frames": [
            {
                "source": canonical,
                "path": mask_paths[canonical].name,
                "hdu": "MASK",
                "sha256": _sha256(mask_paths[canonical]),
                "sizeBytes": mask_paths[canonical].stat().st_size,
                "rejectedPixels": mask_counts[canonical],
                "photometricScale": photometric_scales[index],
                "photometricScaleEvidence": {
                    "registrationIntegratedFlux": fluxes[index],
                    "sourceExposureSeconds": exposures[index],
                    "integratedFluxPerSecond": flux_rates[index],
                    "medianIntegratedFluxPerSecond": rate_center,
                    "calibratedExposureScaleAlreadyApplied": True,
                },
            }
            for index, canonical in enumerate(canonical_paths)
        ],
        "totalRejectedPixels": sum(mask_counts.values()),
    }
    _write_json(directory / "manifest.json", manifest)
    return mask_paths, manifest


def _drizzle_candidates(
    *,
    staging: Path,
    pipeline_root: Path,
    lights: tuple[Path, ...],
    registration: _RegistrationProducts,
    options: DrizzleOptions,
    sampling_evidence: Mapping[str, Any],
    provider: DrizzleProvider | None,
    local_normalization: Any,
    light_infos: Mapping[str, FrameInfo],
) -> tuple[dict[str, Path], dict[str, Any]]:
    calibrated = _pipeline_calibrated_paths(pipeline_root)
    analysis_by_path = {
        registration.source_aliases.get(
            str(Path(item.path).resolve(strict=True)),
            str(Path(item.path).resolve(strict=True)),
        ): (index, item)
        for index, item in enumerate(registration.run.analyses)
    }
    weights = registration.receipt["qualityWeights"]
    reference = registration.run.analyses[registration.run.reference_index]
    output_shape = (
        int(reference.source_height) * options.scale,
        int(reference.source_width) * options.scale,
    )
    groups: dict[str, list[Path]] = {}
    for path in lights:
        canonical = str(path.resolve(strict=True))
        info = light_infos.get(canonical)
        if info is None:
            raise E2EError(
                "STAGED_LIGHT_INFO_MISSING",
                "drizzle requires the preflighted staged Light metadata",
                path=canonical,
            )
        groups.setdefault(info.filter_name, []).append(path)
    candidates: dict[str, Path] = {}
    receipts: dict[str, Any] = {}
    for filter_name, paths in sorted(groups.items()):
        token = _safe_token(filter_name)
        target_dir = staging / "work" / "drizzle" / token
        target_dir.mkdir(parents=True, exist_ok=False)
        selected_calibrated = dict(calibrated)
        local_normalization_evidence: dict[str, Any] = {
            "status": "DISABLED",
            "parameters": local_normalization.serializable(),
        }
        if local_normalization.enabled:
            canonical_group = [str(path.resolve(strict=True)) for path in paths]
            reference_index = max(
                range(len(canonical_group)),
                key=lambda item: float(
                    registration.quality_weights[canonical_group[item]]
                ),
            )
            normalization_result = normalize_registered_group(
                [str(calibrated[path]) for path in canonical_group],
                staging / "work" / "drizzle-local-normalization" / token,
                input_to_reference=[
                    registration.transforms[path] for path in canonical_group
                ],
                reference_index=reference_index,
                parameters=local_normalization,
            )
            for canonical, normalized in zip(
                canonical_group,
                normalization_result.normalized_paths,
                strict=True,
            ):
                selected_calibrated[canonical] = Path(normalized)
            local_normalization_evidence = {
                "status": "APPLIED",
                "receipt": str(
                    Path(normalization_result.receipt_path).relative_to(staging)
                ),
                "receiptSha256": _sha256(Path(normalization_result.receipt_path)),
                "evidence": normalization_result.receipt,
            }
        mask_directory = staging / "coverage" / "rejection-masks" / token
        mask_paths, mask_manifest = _build_drizzle_rejection_masks(
            calibrated=selected_calibrated,
            transforms=registration.transforms,
            paths=paths,
            analysis_by_path=analysis_by_path,
            reference_shape=(
                int(reference.source_height),
                int(reference.source_width),
            ),
            directory=mask_directory,
            options=options,
            source_exposure_seconds={
                str(path.resolve(strict=True)): float(
                    light_infos[str(path.resolve(strict=True))].exposure_seconds
                )
                for path in paths
                if light_infos[str(path.resolve(strict=True))].exposure_seconds
                is not None
            },
        )
        frames: list[DrizzleFrameInput] = []
        for path in paths:
            canonical = str(path.resolve(strict=True))
            if canonical not in calibrated or canonical not in registration.transforms:
                raise E2EError("PIPELINE_RECEIPT_INVALID", "missing calibrated frame or transform", path=canonical)
            output_to_input = _drizzle_output_to_input_matrix(
                registration.transforms[canonical], options.scale
            )
            index, _ = analysis_by_path[canonical]
            info = light_infos[canonical]
            if info.exposure_seconds is None or info.exposure_seconds <= 0:
                raise E2EError("LIGHT_EXPOSURE_UNKNOWN", "Light requires positive EXPTIME", path=canonical)
            quality_weight = max(float(weights[index]), 0.05)
            frames.append(
                DrizzleFrameInput(
                    calibrated_path=str(selected_calibrated[canonical]),
                    output_to_input_projective=output_to_input,
                    rejection_mask_path=str(mask_paths[canonical]),
                    rejection_mask_hdu="MASK",
                    exposure_seconds=info.exposure_seconds,
                    weight_scale=quality_weight,
                )
            )
        output = target_dir / f"master_light_{token}_drizzle_unsolved.fits"
        receipt_path = target_dir / "receipt.json"
        drizzle_request = DrizzleExecutionRequest(
            frames=tuple(frames),
            output_path=str(output),
            receipt_path=str(receipt_path),
            output_shape=output_shape,
            scale=options.scale,
            pixfrac=options.pixfrac,
            kernel=options.kernel,
            tile_rows=options.tile_rows,
            max_tile_bytes=options.max_tile_bytes,
            max_output_pixels=options.max_output_pixels,
            max_working_set_bytes=options.max_working_set_bytes,
            minimum_coverage_fraction=options.minimum_coverage_fraction,
            maximum_null_fraction=options.maximum_null_fraction,
            minimum_distinct_dither_phases=options.minimum_distinct_dither_phases,
            minimum_dither_phase_separation_pixels=options.minimum_dither_phase_separation_pixels,
            minimum_dither_span_pixels=options.minimum_dither_span_pixels,
            median_fwhm_native_pixels=sampling_evidence.get(
                "medianNativeFwhmPixels"
            ),
            maximum_fwhm_for_upsampling_pixels=options.maximum_fwhm_for_upsampling_pixels,
            pixel_scale_arcsec=sampling_evidence.get("pixelScaleArcsec"),
        )
        result = execute_drizzle(drizzle_request, provider=provider)
        if not result.completed or result.output_path is None or result.receipt is None:
            raise E2EError(result.code, result.message)
        try:
            verified_receipt = verify_drizzle_result(drizzle_request, result)
        except Exception as error:
            code = getattr(error, "code", "DRIZZLE_VERIFICATION_FAILED")
            raise E2EError(code, str(error)) from error
        candidates[filter_name] = Path(result.output_path).resolve(strict=True)
        receipt_with_mask_evidence = dict(verified_receipt)
        receipt_with_mask_evidence["e2eRejectionMaskEvidence"] = {
            "manifest": str(
                (mask_directory / "manifest.json").relative_to(staging)
            ),
            "manifestSha256": _sha256(mask_directory / "manifest.json"),
            "totalRejectedPixels": mask_manifest["totalRejectedPixels"],
        }
        receipt_with_mask_evidence["localNormalization"] = local_normalization_evidence
        receipts[filter_name] = receipt_with_mask_evidence
    preserved_local_normalization = _preserve_local_normalization_evidence(
        staging / "work" / "drizzle-local-normalization",
        staging,
        namespace="drizzle",
    )
    for filter_name, receipt in receipts.items():
        normalization = receipt.get("localNormalization")
        if not isinstance(normalization, dict) or normalization.get("status") != "APPLIED":
            continue
        preserved = preserved_local_normalization.get("groups", {}).get(
            _safe_token(filter_name)
        )
        if not isinstance(preserved, dict):
            raise E2EError(
                "LOCAL_NORMALIZATION_EVIDENCE_MISSING",
                "Drizzle local-normalization models were not preserved",
            )
        normalization["receipt"] = preserved["manifest"]
        normalization["receiptSha256"] = preserved["manifestSha256"]
        normalization["normalizedFramesTransient"] = True
    return candidates, {
        "mode": IntegrationMode.DRIZZLE.value,
        "options": options.serializable(),
        "sampling": dict(sampling_evidence),
        "filters": receipts,
        "localNormalizationEvidence": preserved_local_normalization,
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
            master_data = np.asarray(hdul[0].data)
            null_fraction = float(
                1.0 - np.count_nonzero(np.isfinite(master_data)) / master_data.size
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


def _preserve_local_normalization_evidence(
    source_root: Path,
    staging: Path,
    *,
    namespace: str,
) -> dict[str, Any]:
    """Preserve models/metrics while large normalized execution frames stay transient."""

    if not source_root.is_dir():
        return {"status": "DISABLED", "groups": {}}
    destination_root = staging / "coverage" / "local-normalization" / namespace
    destination_root.mkdir(parents=True, exist_ok=True)
    groups: dict[str, Any] = {}
    for group in sorted(path for path in source_root.iterdir() if path.is_dir()):
        receipt_path = group / "receipt.json"
        models = group / "models"
        if not receipt_path.is_file() or not models.is_dir():
            raise E2EError(
                "LOCAL_NORMALIZATION_EVIDENCE_MISSING",
                "applied local normalization lacks receipt or model files",
                path=str(group),
            )
        destination = destination_root / group.name
        destination.mkdir(exist_ok=False)
        shutil.copytree(models, destination / "models")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for frame in receipt.get("frames", []):
            normalized = frame.get("normalized")
            if isinstance(normalized, dict):
                normalized.pop("path", None)
                normalized["transientExecutionArtifact"] = True
            model = frame.get("model")
            if isinstance(model, dict):
                for key in ("scalePath", "offsetPath"):
                    value = model.get(key)
                    if isinstance(value, str):
                        model[key] = str(Path("models") / Path(value).name)
        manifest = destination / "manifest.json"
        _write_json(manifest, receipt)
        groups[group.name] = {
            "manifest": str(manifest.relative_to(staging)),
            "manifestSha256": _sha256(manifest),
            "modelFileCount": len(tuple((destination / "models").glob("*.fits"))),
        }
    return {"status": "PRESERVED", "groups": groups}


def _read_image_header(path: Path) -> tuple[fits.Header, tuple[int, int]]:
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            lazy_load_hdus=True,
            do_not_scale_image_data=True,
            uint=False,
            checksum=False,
        ) as hdul:
            hdu = next(
                (
                    item
                    for item in hdul
                    if int(item.header.get("NAXIS", 0) or 0) == 2 and item.data is not None
                ),
                None,
            )
            if hdu is None or hdu.data is None:
                raise E2EError("SOLVED_OUTPUT_INVALID", "no two-dimensional FITS image", path=str(path))
            shape = tuple(int(value) for value in hdu.data.shape)
            if len(shape) != 2:
                raise E2EError("SOLVED_OUTPUT_INVALID", "image is not two-dimensional", path=str(path))
            return hdu.header.copy(), (shape[0], shape[1])
    except E2EError:
        raise
    except Exception as error:
        raise E2EError("SOLVED_OUTPUT_INVALID", str(error), path=str(path)) from error


def _wcs_headers_agree(
    result_header: Mapping[str, Any] | fits.Header,
    output_header: fits.Header,
    shape: tuple[int, int],
) -> bool:
    try:
        left = WCS(result_header).celestial
        right = WCS(output_header).celestial
        height, width = shape
        pixels = np.asarray(
            [[0.0, 0.0], [(width - 1) / 2.0, (height - 1) / 2.0], [width - 1.0, height - 1.0]],
            dtype=np.float64,
        )
        left_world = left.all_pix2world(pixels, 0)
        right_pixels = right.all_world2pix(left_world, 0)
        return bool(np.all(np.isfinite(right_pixels)) and np.max(np.abs(right_pixels - pixels)) <= 0.05)
    except Exception:
        return False


def _unit_vectors(world_degrees: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(world_degrees[:, 0])
    dec = np.deg2rad(world_degrees[:, 1])
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def _sky_separation_degrees(left: tuple[float, float], right: tuple[float, float]) -> float:
    vectors = _unit_vectors(np.asarray((left, right), dtype=np.float64))
    chord = float(np.linalg.norm(vectors[0] - vectors[1]))
    return float(np.rad2deg(2.0 * math.asin(min(1.0, max(0.0, chord / 2.0)))))


def _solution_geometry(
    header: Mapping[str, Any] | fits.Header,
    shape: tuple[int, int],
) -> dict[str, Any]:
    height, width = shape
    celestial = WCS(header, relax=False).celestial
    center_pixel = np.asarray([[(width - 1.0) / 2.0, (height - 1.0) / 2.0]])
    horizontal = np.asarray(
        [[0.0, (height - 1.0) / 2.0], [width - 1.0, (height - 1.0) / 2.0]],
        dtype=np.float64,
    )
    vertical = np.asarray(
        [[(width - 1.0) / 2.0, 0.0], [(width - 1.0) / 2.0, height - 1.0]],
        dtype=np.float64,
    )
    center_world = celestial.all_pix2world(center_pixel, 0)[0]
    horizontal_world = celestial.all_pix2world(horizontal, 0)
    vertical_world = celestial.all_pix2world(vertical, 0)
    matrix = np.asarray(celestial.pixel_scale_matrix, dtype=np.float64)
    column_scales = np.sqrt(np.sum(np.square(matrix), axis=0)) * 3600.0
    if (
        not np.all(np.isfinite(center_world))
        or not np.all(np.isfinite(horizontal_world))
        or not np.all(np.isfinite(vertical_world))
        or not np.all(np.isfinite(column_scales))
        or np.any(column_scales <= 0)
    ):
        raise ValueError("the WCS geometry is non-finite")
    return {
        "centerRaDegrees": float(center_world[0] % 360.0),
        "centerDecDegrees": float(center_world[1]),
        "fieldWidthDegrees": _sky_separation_degrees(
            tuple(float(value) for value in horizontal_world[0]),
            tuple(float(value) for value in horizontal_world[1]),
        ),
        "fieldHeightDegrees": _sky_separation_degrees(
            tuple(float(value) for value in vertical_world[0]),
            tuple(float(value) for value in vertical_world[1]),
        ),
        "pixelScaleArcsec": float(math.sqrt(float(column_scales[0] * column_scales[1]))),
        "axisPixelScalesArcsec": [float(column_scales[0]), float(column_scales[1])],
        "pixelAspectRatio": float(max(column_scales) / min(column_scales)),
        "rotationDegrees": float(
            math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
        ),
        "parity": wcs_parity(header).value,
        "imageShape": [height, width],
    }


def _validate_solution_against_hints(
    result: SolverResult,
    hints: _SolverHints,
) -> WcsValidation:
    try:
        if result.image_shape is None:
            raise ValueError("solver result omitted image geometry")
        geometry = _solution_geometry(result.header, result.image_shape)
    except Exception as error:
        return WcsValidation(False, "SOLVER_GEOMETRY_INVALID", str(error))
    diagnostics: dict[str, Any] = {"solution": geometry, "hints": hints.serializable()}
    if hints.ra_degrees is not None and hints.dec_degrees is not None:
        separation = _sky_separation_degrees(
            (geometry["centerRaDegrees"], geometry["centerDecDegrees"]),
            (hints.ra_degrees, hints.dec_degrees),
        )
        effective_radius = hints.search_radius_degrees if hints.search_radius_degrees is not None else 15.0
        diagnostics["centerToHintDegrees"] = separation
        diagnostics["effectiveSearchRadiusDegrees"] = effective_radius
        if separation > effective_radius:
            return WcsValidation(
                False,
                "SOLVER_CENTER_OUTSIDE_HINT_RADIUS",
                f"solved center is {separation:.6g} deg from the NINA/request hint, outside {effective_radius:.6g} deg",
                diagnostics,
            )
    if hints.field_of_view_degrees is not None:
        solved_width = float(geometry["fieldWidthDegrees"])
        height, width = (int(value) for value in geometry["imageShape"])
        ratio = solved_width / hints.field_of_view_degrees
        expected_height = hints.field_of_view_degrees * max(height - 1, 1) / max(width - 1, 1)
        height_ratio = float(geometry["fieldHeightDegrees"]) / expected_height
        hinted_scale = hints.field_of_view_degrees * 3600.0 / max(width - 1, 1)
        scale_ratio = float(geometry["pixelScaleArcsec"]) / hinted_scale
        diagnostics.update(
            {
                "fieldWidthToHintRatio": ratio,
                "fieldHeightToHintRatio": height_ratio,
                "hintDerivedPixelScaleArcsec": hinted_scale,
                "pixelScaleToHintRatio": scale_ratio,
            }
        )
        # solve-field itself is constrained to 0.8--1.2 of this hint.  The
        # slightly wider independent gate allows projection/corner-definition
        # differences without permitting a different scale solution.
        if not (
            0.7 <= ratio <= 1.3
            and 0.7 <= height_ratio <= 1.3
            and 0.7 <= scale_ratio <= 1.3
            and geometry["pixelAspectRatio"] <= 1.2
        ):
            return WcsValidation(
                False,
                "SOLVER_SCALE_OUTSIDE_HINT",
                "solved field width/pixel scale is inconsistent with the NINA or derived FOV hint",
                diagnostics,
            )
    return WcsValidation(
        True,
        "SOLVER_HINTS_VALID",
        "solved center, field of view, and pixel scale agree with acquisition hints",
        diagnostics,
    )


def _validate_cross_filter_wcs(
    products: Mapping[str, Path], *, tolerance_pixels: float = 0.05
) -> WcsValidation:
    """Require direct full-field pixel agreement between solved filter WCSes.

    Centre/scale/parity summaries cannot detect a 90-degree rotation or
    edge-only SIP drift.  Registered filter masters are required to describe
    the same pixel grid, so this gate compares centre, corners, and edge
    midpoints through each celestial transform in both directions.
    """

    if not math.isfinite(tolerance_pixels) or tolerance_pixels <= 0:
        return WcsValidation(
            False,
            "CROSS_FILTER_POLICY_INVALID",
            "cross-filter WCS tolerance must be finite and positive",
        )
    geometries: dict[str, dict[str, Any]] = {}
    headers: dict[str, fits.Header] = {}
    shapes: dict[str, tuple[int, int]] = {}
    try:
        for filter_name, path in sorted(products.items()):
            header, shape = _read_image_header(path)
            validation = validate_wcs_header(header, image_shape=shape)
            if not validation.valid:
                return validation
            geometries[filter_name] = _solution_geometry(header, shape)
            headers[filter_name] = header
            shapes[filter_name] = shape
    except Exception as error:
        return WcsValidation(False, "CROSS_FILTER_WCS_INVALID", str(error))
    if len(geometries) < 2:
        return WcsValidation(
            True,
            "CROSS_FILTER_WCS_NOT_APPLICABLE",
            "only one solved filter is present",
            {"filters": geometries},
        )
    reference_name = next(iter(geometries))
    reference = geometries[reference_name]
    reference_shape = shapes[reference_name]
    height, width = reference_shape
    samples = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
            [(width - 1.0) / 2.0, 0.0],
            [(width - 1.0) / 2.0, height - 1.0],
            [0.0, (height - 1.0) / 2.0],
            [width - 1.0, (height - 1.0) / 2.0],
            [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
        ],
        dtype=np.float64,
    )
    reference_wcs = WCS(headers[reference_name], relax=False).celestial
    comparisons: dict[str, Any] = {}
    for filter_name, geometry in geometries.items():
        if filter_name == reference_name:
            continue
        shape_agrees = shapes[filter_name] == reference_shape
        candidate_wcs = WCS(headers[filter_name], relax=False).celestial
        if shape_agrees:
            candidate_world = candidate_wcs.all_pix2world(samples, 0)
            in_reference = reference_wcs.all_world2pix(candidate_world, 0)
            reference_world = reference_wcs.all_pix2world(samples, 0)
            in_candidate = candidate_wcs.all_world2pix(reference_world, 0)
            finite = bool(
                np.all(np.isfinite(in_reference)) and np.all(np.isfinite(in_candidate))
            )
            candidate_to_reference = (
                float(np.max(np.abs(in_reference - samples))) if finite else None
            )
            reference_to_candidate = (
                float(np.max(np.abs(in_candidate - samples))) if finite else None
            )
            per_sample = (
                np.maximum(
                    np.max(np.abs(in_reference - samples), axis=1),
                    np.max(np.abs(in_candidate - samples), axis=1),
                ).tolist()
                if finite
                else [None] * len(samples)
            )
        else:
            finite = False
            candidate_to_reference = None
            reference_to_candidate = None
            per_sample = [None] * len(samples)
        rotation_delta = abs(
            ((float(geometry["rotationDegrees"]) - float(reference["rotationDegrees"]) + 180.0) % 360.0)
            - 180.0
        )
        comparisons[filter_name] = {
            "referenceFilter": reference_name,
            "shapeAgrees": shape_agrees,
            "tolerancePixels": tolerance_pixels,
            "candidateToReferenceMaximumResidualPixels": candidate_to_reference,
            "referenceToCandidateMaximumResidualPixels": reference_to_candidate,
            "sampleResidualPixels": per_sample,
            "samplePixels": samples.tolist(),
            "rotationDeltaDegrees": rotation_delta,
            "parityAgrees": reference["parity"] == geometry["parity"],
        }
        if (
            not shape_agrees
            or not finite
            or candidate_to_reference is None
            or reference_to_candidate is None
            or candidate_to_reference > tolerance_pixels
            or reference_to_candidate > tolerance_pixels
            or reference["parity"] != geometry["parity"]
        ):
            return WcsValidation(
                False,
                "CROSS_FILTER_WCS_MISMATCH",
                f"{filter_name} does not describe the same registered sky footprint as {reference_name}",
                {"filters": geometries, "comparisons": comparisons},
            )
    return WcsValidation(
        True,
        "CROSS_FILTER_WCS_VALID",
        "all filter WCSes agree directly at centre, corners, and edge midpoints",
        {
            "filters": geometries,
            "comparisons": comparisons,
            "policy": {
                "maximumResidualPixels": tolerance_pixels,
                "sampleCount": len(samples),
                "bidirectional": True,
            },
        },
    )


def _verify_backend_result(
    backend: SolverBackend,
    result: SolverResult,
) -> bool:
    backend_verifier = getattr(backend, "verify_result", None)
    if callable(backend_verifier):
        try:
            return bool(backend_verifier(result))
        except Exception:
            return False
    return bool(verify_solver_execution_result(result))


def _discard_owned_solver_output(path: Path) -> bool:
    """Remove only the exact, staging-owned output requested from a backend."""

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISDIR(mode):
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def _promote_solved_state(
    path: Path,
    *,
    backend_id: str,
    solver_verified_sha256: str,
) -> dict[str, Any]:
    """Make the E2E state cards agree with an already verified fresh WCS.

    External adapters intentionally preserve non-WCS input cards, including
    the pixel pipeline's ``UNSOLVED_WORKING`` marker.  The adapter receipt is
    verified first; this E2E-owned promotion then changes only state/history
    cards, revalidates WCS, and binds the resulting file in the outer receipt.
    """

    before_stat = path.lstat()
    if not stat.S_ISREG(before_stat.st_mode) or path.is_symlink():
        raise E2EError("SOLVED_OUTPUT_INVALID", "solver output is not a regular file", path=str(path))
    before_sha256 = _sha256(path)
    if before_sha256 != solver_verified_sha256:
        raise E2EError("SOLVED_OUTPUT_DRIFT", "solver output changed before E2E promotion", path=str(path))
    before_header, before_shape = _read_image_header(path)
    try:
        with fits.open(
            path,
            mode="update",
            memmap=True,
            do_not_scale_image_data=True,
            uint=False,
            checksum=False,
        ) as hdul:
            hdul[0].header["OAFSTATE"] = ("SOLVED", "Ultra-Fast WBPP E2E product state")
            hdul[0].header["OAFWCS"] = ("SOLVED", "Fresh WCS independently verified")
            hdul[0].header["OAFSOLVR"] = (backend_id, "Plate solver backend")
            hdul[0].header.add_history(
                "Ultra-Fast WBPP: solver evidence verified before E2E state promotion"
            )
            for hdu in hdul:
                if "CHECKSUM" in hdu.header or "DATASUM" in hdu.header:
                    hdu.add_checksum(override_datasum=True)
            hdul.flush(output_verify="exception")
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
    except Exception as error:
        raise E2EError("SOLVED_STATE_PROMOTION_FAILED", str(error), path=str(path)) from error
    after_header, after_shape = _read_image_header(path)
    validation = validate_wcs_header(after_header, image_shape=after_shape)
    if (
        not validation.valid
        or before_shape != after_shape
        or not _wcs_headers_agree(before_header, after_header, after_shape)
        or after_header.get("OAFSTATE") != "SOLVED"
        or after_header.get("OAFWCS") != "SOLVED"
    ):
        raise E2EError(
            "SOLVED_STATE_PROMOTION_INVALID",
            "state promotion changed geometry/WCS or did not persist SOLVED markers",
            path=str(path),
        )
    try:
        with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
            for hdu in hdul:
                if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
                    raise E2EError("SOLVED_OUTPUT_CHECKSUM_INVALID", "FITS checksum failed", path=str(path))
                if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
                    raise E2EError("SOLVED_OUTPUT_CHECKSUM_INVALID", "FITS datasum failed", path=str(path))
    except E2EError:
        raise
    except Exception as error:
        raise E2EError("SOLVED_OUTPUT_CHECKSUM_INVALID", str(error), path=str(path)) from error
    return {
        "kind": "E2E_SOLVED_STATE_PROMOTION",
        "solverVerifiedSha256": before_sha256,
        "finalSha256": _sha256(path),
        "sizeBytes": path.stat().st_size,
        "wcsValidation": validation.serializable(),
        "stateCards": {"OAFSTATE": "SOLVED", "OAFWCS": "SOLVED"},
    }


def _solve_one(
    *,
    input_path: Path,
    output_path: Path,
    backends: Sequence[SolverBackend],
    hints: _SolverHints,
    min_matches: int,
    max_rms_arcsec: float,
) -> tuple[bool, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for backend in backends:
        if os.path.lexists(output_path):
            attempts.append(
                {
                    "backendId": getattr(backend, "backend_id", type(backend).__name__),
                    "accepted": False,
                    "code": "UNEXPECTED_OUTPUT_EXISTS",
                }
            )
            break
        try:
            result = backend.solve(
                SolveRequest(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    ra_hint_degrees=hints.ra_degrees,
                    dec_hint_degrees=hints.dec_degrees,
                    field_of_view_degrees=hints.field_of_view_degrees,
                    search_radius_degrees=hints.search_radius_degrees,
                )
            )
        except Exception as error:
            attempts.append(
                {
                    "backendId": getattr(backend, "backend_id", type(backend).__name__),
                    "accepted": False,
                    "code": "BACKEND_EXCEPTION",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            if not _discard_owned_solver_output(output_path):
                attempts[-1]["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        if not isinstance(result, SolverResult):
            attempts.append(
                {
                    "backendId": getattr(backend, "backend_id", type(backend).__name__),
                    "accepted": False,
                    "code": "BACKEND_PROTOCOL_ERROR",
                    "message": "solver returned the wrong result type",
                }
            )
            if not _discard_owned_solver_output(output_path):
                attempts[-1]["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        try:
            validation = validate_solver_result(
                result,
                require_scientific_evidence=True,
                min_matches=min_matches,
                max_rms_arcsec=max_rms_arcsec,
            )
            hint_validation = (
                _validate_solution_against_hints(result, hints)
                if validation.valid
                else WcsValidation(
                    False,
                    "SOLVER_HINT_VALIDATION_SKIPPED",
                    "catalog quality validation failed first",
                )
            )
            verified = _verify_backend_result(backend, result)
            serialized_result = result.serializable()
        except Exception as error:
            attempts.append(
                {
                    "backendId": getattr(backend, "backend_id", type(backend).__name__),
                    "accepted": False,
                    "code": "BACKEND_PROTOCOL_ERROR",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            if not _discard_owned_solver_output(output_path):
                attempts[-1]["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        attempt = {
            "result": serialized_result,
            "validation": validation.serializable(),
            "hintValidation": hint_validation.serializable(),
            "executionVerified": verified,
            "accepted": False,
        }
        if not validation.valid or not hint_validation.valid or not verified:
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        if result.output_path is None:
            attempt["code"] = "SOLVED_OUTPUT_MISSING"
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        try:
            actual_output = Path(result.output_path).resolve(strict=True)
        except OSError:
            attempt["code"] = "SOLVED_OUTPUT_MISSING"
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        if actual_output != output_path.resolve(strict=False):
            attempt["code"] = "SOLVED_OUTPUT_PATH_MISMATCH"
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        try:
            output_header, output_shape = _read_image_header(actual_output)
        except E2EError as error:
            attempt["code"] = error.code
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        output_validation = validate_wcs_header(output_header, image_shape=output_shape)
        attempt["publishedWcsValidation"] = output_validation.serializable()
        if (
            not output_validation.valid
            or result.image_shape != output_shape
            or not _wcs_headers_agree(result.header, output_header, output_shape)
        ):
            attempt["code"] = "SOLVER_RESULT_OUTPUT_MISMATCH"
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        attempt["accepted"] = True
        attempt["artifact"] = {
            "path": str(actual_output),
            "sha256": _sha256(actual_output),
            "sizeBytes": actual_output.stat().st_size,
        }
        try:
            attempt["promotion"] = _promote_solved_state(
                actual_output,
                backend_id=result.backend_id,
                solver_verified_sha256=attempt["artifact"]["sha256"],
            )
            attempt["artifact"] = {
                "path": str(actual_output),
                "sha256": attempt["promotion"]["finalSha256"],
                "sizeBytes": attempt["promotion"]["sizeBytes"],
            }
        except E2EError as error:
            attempt["accepted"] = False
            attempt["code"] = error.code
            attempt["message"] = str(error)
            attempts.append(attempt)
            return False, attempts
        attempts.append(attempt)
        return True, attempts
    return False, attempts


def _artifact_records(staging: Path, roots: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            records.append(
                {
                    "path": str(path.relative_to(staging)),
                    "sha256": _sha256(path),
                    "sizeBytes": path.stat().st_size,
                }
            )
    return records


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
) -> E2EResult:
    private_work = staging / "work"
    if private_work.exists():
        shutil.rmtree(private_work)
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
    drizzle_provider: DrizzleProvider | None = None,
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
    identities = tuple(_capture_source(path, role) for role, path in flattened)
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
            if cfa != "NONE":
                raise E2EError(
                    "CFA_PIXEL_PIPELINE_UNSUPPORTED",
                    f"CFA/Bayer raw {expected_role} is blocked until phase-preserving calibration, registration, and color reconstruction are implemented",
                    path=str(path),
                )
    _emit(progress, ProgressStage.INVENTORY, "completed", f"validated {len(identities)} source files")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    published = False
    try:
        work = staging / "work"
        receipts_dir = staging / "receipts"
        products_dir = staging / "products"
        previews_dir = staging / "previews"
        coverage_dir = staging / "coverage"
        for directory in (work, receipts_dir, products_dir, previews_dir, coverage_dir):
            directory.mkdir()

        _emit(progress, ProgressStage.QUALITY_CONTROL, "started", "measuring and gating Light frames")
        qc_dir = staging / "qc"
        qc_dir.mkdir()
        qc_timings: dict[str, float] = {}
        qc_cache_stats: dict[str, int] = {}
        qc_started = perf_counter()
        measurements = measure_paths(
            [str(path) for path in lights],
            qc_dir,
            request.qc_config,
            workers=request.workers,
        )
        qc_timings["measurementSeconds"] = perf_counter() - qc_started
        _emit(progress, ProgressStage.QUALITY_CONTROL, "running", f"measured {len(measurements)} Light frames; analyzing star fields")
        qc_started = perf_counter()
        groups, frame_results = analyze_measurements(
            measurements, request.qc_config,
            cache_directory=quality_cache_directory(), cache_stats=qc_cache_stats,
        )
        qc_timings["analysisSeconds"] = perf_counter() - qc_started
        qc_started = perf_counter()
        evaluate_quality_gate(frame_results, measurements, request.gate_policy)
        qc_timings["gateSeconds"] = perf_counter() - qc_started
        qc_manifest = _qc_manifest(staging, groups, frame_results, request.qc_config, request.gate_policy)
        qc_manifest["timings"] = qc_timings
        qc_manifest["analysisCache"] = qc_cache_stats
        approved_review_paths, approval_request_digest, approval_evidence = (
            _apply_review_approvals(
                request=request,
                identities=identities,
                results=frame_results,
            )
        )
        qc_manifest["manualReviewApprovals"] = {
            "defaultDisposition": "EXCLUDED",
            "requestDigest": approval_request_digest,
            "gatePolicyDigest": request.gate_policy.canonical_digest(),
            "accepted": approval_evidence,
        }
        _write_json(qc_dir / "manifest.json", qc_manifest)
        gate_by_path = {
            str(Path(result.path).resolve(strict=True)): result.quality_gate
            for result in frame_results
        }
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
        excluded = tuple(path for path in lights if path not in passed_set)
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
            f"{len(passed)} admitted ({len(approved_review_paths)} explicitly approved REVIEW); "
            f"{len(excluded)} REVIEW/HARD_FAIL excluded",
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
        _verify_staged_pixel_inputs(staged_input_digests)
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
        registration = _register_lights(
            staged_inputs["LIGHT"],
            calibration_plan,
            detection=request.registration_detection,
            registration=request.registration_config,
            workers=request.workers,
            allow_projective=request.integration_mode is IntegrationMode.DRIZZLE,
            source_aliases=registration_source_aliases,
            source_sha256_by_path={
                str(path): staged_input_digests[str(path)]
                for path in staged_inputs["LIGHT"]
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
        pipeline_root = work / "pixel-pipeline"
        if request.integration_mode is IntegrationMode.DRIZZLE:
            # This pipeline invocation is the current public calibration-only
            # bridge.  Its registered/integrated products are not promoted;
            # exact projective matrices go to drizzle below.  Supplying a
            # complete explicit identity map prevents the pixel pipeline's
            # missing-key identity fallback from hiding a registration gap.
            identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            pipeline_transforms: Mapping[str, Sequence[Sequence[float]]] = {
                str(path): identity for path in passed
            }
        else:
            pipeline_transforms = registration.transforms
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
        )
        if (
            request.integration_mode is IntegrationMode.DRIZZLE
            and request.pipeline_parameters.local_normalization.enabled
        ):
            pipeline_parameters = replace(
                pipeline_parameters,
                local_normalization=replace(
                    request.pipeline_parameters.local_normalization,
                    enabled=False,
                ),
            )
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
        staged_pipeline_transforms = {
            str(staged): pipeline_transforms[
                str(registration_source_aliases[str(staged)].resolve(strict=True))
            ]
            for staged in staged_inputs["LIGHT"]
        }
        staged_quality_weights = {
            str(staged): registration.quality_weights[
                str(registration_source_aliases[str(staged)].resolve(strict=True))
            ]
            for staged in staged_inputs["LIGHT"]
        }
        staged_light_by_original = {
            str(registration_source_aliases[str(staged)].resolve(strict=True)): staged
            for staged in staged_inputs["LIGHT"]
        }
        staged_stellar_scale_hints: dict[str, StellarScaleHint] = {}
        for original, staged in staged_light_by_original.items():
            hint = registration.stellar_scale_hints[original]
            reference_staged = staged_light_by_original.get(
                str(Path(hint.reference_path).resolve(strict=True))
            )
            if reference_staged is None:
                raise E2EError(
                    "STELLAR_SCALE_HINT_REFERENCE_MISMATCH",
                    "registration stellar scale reference is absent from the pixel Light set",
                    path=hint.reference_path,
                )
            staged_stellar_scale_hints[str(staged)] = replace(
                hint,
                source_path=str(staged),
                reference_path=str(reference_staged),
            )
        try:
            pipeline_result = _run_portable_pipeline_fits(
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
                light_files=staged_inputs["LIGHT"],
                output_directory=pipeline_root,
                transforms=staged_pipeline_transforms,
                quality_weights=staged_quality_weights,
                stellar_scale_hints=staged_stellar_scale_hints,
                parameters=pipeline_parameters,
                _source_aliases=registration_source_aliases,
                _xisf_conversions=xisf_conversions,
                _trusted_generated_calibration=trusted_generated_calibration,
            )
        except CalibrationError as error:
            raise E2EError(error.code, str(error), path=error.path) from error
        _verify_staged_pixel_inputs(staged_input_digests)
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
                pipeline_root=pipeline_root,
                lights=passed,
                registration=registration,
                options=request.drizzle,
                sampling_evidence=drizzle_sampling,
                provider=drizzle_provider,
                local_normalization=request.pipeline_parameters.local_normalization,
                light_infos=staged_light_infos,
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
            coverage["localNormalizationEvidence"] = (
                _preserve_local_normalization_evidence(
                    pipeline_root / "local-normalization",
                    staging,
                    namespace="ordinary",
                )
                if request.pipeline_parameters.local_normalization.enabled
                else {"status": "DISABLED", "groups": {}}
            )
        _write_json(coverage_dir / "coverage.json", coverage)

        solver_records: dict[str, Any] = {}
        product_paths_staged: list[Path] = []
        solved_products: dict[str, Path] = {}
        all_solved = True
        _emit(progress, ProgressStage.ASTROMETRY, "started", "solving every filter", total=len(candidates))
        for index, (filter_name, candidate) in enumerate(sorted(candidates.items()), start=1):
            token = _safe_token(filter_name)
            filter_dir = products_dir / token
            filter_dir.mkdir(parents=True, exist_ok=False)
            solved_path = filter_dir / f"master_light_{token}_wcs.fits"
            solved, attempts = _solve_one(
                input_path=candidate,
                output_path=solved_path,
                backends=solver_backends,
                hints=solver_hints,
                min_matches=request.min_matches,
                max_rms_arcsec=request.max_rms_arcsec,
            )
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
            _emit(
                progress,
                ProgressStage.ASTROMETRY,
                "progress",
                f"{filter_name}: {'SOLVED' if solved else 'UNSOLVED'}",
                current=index,
                total=len(candidates),
            )

        cross_filter_validation = (
            _validate_cross_filter_wcs(solved_products)
            if all_solved
            else WcsValidation(
                False,
                "CROSS_FILTER_WCS_SKIPPED",
                "one or more filter solutions failed their individual gates",
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
        shutil.rmtree(work)
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
            "qualityControl": {
                "manifest": "qc/manifest.json",
                "passedLights": len(passed),
                "excludedLights": len(excluded),
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
        if not published and staging.exists():
            shutil.rmtree(staging)
        try:
            _emit(progress, ProgressStage.FAILED, "failed", "E2E execution aborted without publishing success")
        except Exception:
            pass
        raise


__all__ = [
    "DrizzleOptions",
    "E2EError",
    "E2ERequest",
    "E2EResult",
    "E2EState",
    "E2E_VERSION",
    "IntegrationMode",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressStage",
    "ReviewApproval",
    "bind_review_approval_selections",
    "run_e2e",
]
