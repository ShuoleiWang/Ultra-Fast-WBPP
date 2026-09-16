"""Portable CPU raw-to-linear-master pipeline.

The pipeline is deliberately not an astrometric solver or drizzle engine. Its
published masters are marked ``UNSOLVED_WORKING`` and contain no fabricated WCS.
All pixel operations use vertical slices and all input FITS files stay read-only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading
import ctypes
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .calibration_policy import (STRICT, MONO_STANDARD, WORKFLOWS, apply_mono_workflow, metadata_changes, same_metadata, cfa_for_workflow, resolve_dark_bias, workflow_receipt, can_omit_bias, conflicting_profile_fields, acquisition_receipt)
from .calibration import (
    CalibrationError,
    FitsFloatWriter,
    FitsFrame,
    FrameExpression,
    FrameInfo,
    IntegrationMapPaths,
    IntegrationParameters,
    PixelStatistics,
    _MemoryFrame,
    _StatsAccumulator,
    _atomic_publish_file,
    _canonical_expression,
    _expression_rows,
    _temporary_output,
    _validate_expression_shapes,
    integrate_expressions,
    read_frame_info,
    robust_location,
    write_expression,
)
from .native_kernels import WARP_KERNEL_ID, load_native_kernels
from .preview import render_auto_stretch_preview
from .hardware import detect_hardware
from .metal_integration import (
    MetalIntegrationError,
    NativeMetalExecutor,
    SUPPORTED_BACKENDS,
    integrate_registered_group,
)
from .performance_profile import select_execution_tuning
from .local_normalization import (
    LocalNormalizationParameters,
    normalize_registered_group,
)
from .global_normalization import (
    GlobalNormalizationParameters,
    StellarScaleHint,
    fit_registered_group_global_normalization,
)
from .xisf_pixels import XisfDecodePolicy, convert_xisf_to_fits


PIPELINE_VERSION = "portable-pixel-pipeline-v1"
OUTPUT_STATE = "UNSOLVED_WORKING"
REGISTRATION_RESAMPLERS = frozenset({"bilinear", "lanczos-3-clamped"})
# v2: tap weights are evaluated through exact trigonometric identities from
# three transcendental calls per axis instead of twelve; the interpolation
# contract is unchanged and results differ from v1 by at most one Float32 ulp.
LANCZOS3_REGISTRATION_ALGORITHM = (
    "normalized-lanczos-3-domain-union-support-clamp-v2"
)
NUMPY_WARP_KERNEL_ID = "numpy-lanczos3-warp-v2"
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
class AffineTransform:
    """Input-pixel to output-pixel homogeneous affine transform."""

    matrix: tuple[tuple[float, float, float], ...]

    @classmethod
    def identity(cls) -> AffineTransform:
        return cls(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    @classmethod
    def from_value(
        cls, value: AffineTransform | Sequence[Sequence[float]]
    ) -> AffineTransform:
        if isinstance(value, cls):
            result = value
        else:
            try:
                result = cls(
                    tuple(tuple(float(item) for item in row) for row in value)
                )
            except (TypeError, ValueError) as error:
                raise CalibrationError(
                    "TRANSFORM_INVALID", "transform must be a numeric 3x3 matrix"
                ) from error
        result.validated_matrix()
        return result

    def validated_matrix(self) -> NDArray[np.float64]:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise CalibrationError(
                "TRANSFORM_INVALID", "transform must be a finite 3x3 matrix"
            )
        if not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-12):
            raise CalibrationError(
                "TRANSFORM_PROJECTIVE_UNSUPPORTED",
                "portable registration currently accepts affine transforms only",
            )
        determinant = float(np.linalg.det(matrix[:2, :2]))
        if not math.isfinite(determinant) or abs(determinant) < 1e-12:
            raise CalibrationError("TRANSFORM_SINGULAR", "affine transform is singular")
        return matrix

    @property
    def is_identity(self) -> bool:
        return bool(
            np.allclose(
                self.validated_matrix(), np.eye(3), rtol=0.0, atol=1e-12
            )
        )

    def serializable(self) -> list[list[float]]:
        return [list(row) for row in self.matrix]


@dataclass(frozen=True, slots=True)
class PipelineParameters:
    calibration_workflow: str = STRICT
    integration: IntegrationParameters = field(default_factory=IntegrationParameters)
    registration_memory_bytes: int = 256 * 1024 * 1024
    registration_resampler: str = "lanczos-3-clamped"
    auto_crop: bool = True
    minimum_crop_fraction: float = 0.25
    preview_max_long_edge: int = 1600
    dark_temperature_tolerance_celsius: float = 3.0
    ordinary_integration_backend: str = "auto"
    native_library_path: str | None = None
    metal_source_path: str | None = None
    xisf_decode: XisfDecodePolicy = field(default_factory=XisfDecodePolicy)
    local_normalization: LocalNormalizationParameters = field(
        default_factory=LocalNormalizationParameters
    )
    global_normalization: GlobalNormalizationParameters = field(
        default_factory=GlobalNormalizationParameters
    )
    master_metadata_overrides: tuple[MasterMetadataOverride, ...] = ()
    raw_frame_metadata_overrides: tuple[RawFrameMetadataOverride, ...] = ()
    # Calibrated Lights are computed in memory and registered directly.  They
    # are written to disk only when a consumer (Drizzle, the public portable
    # pipeline) needs them; the ordinary E2E path keeps them transient.
    materialize_calibrated_lights: bool = True
    # Published pipeline outputs are fsynced.  An enclosing run whose whole
    # pipeline directory is transient (the E2E work tree) turns this off and
    # relies on its own fsynced promotion of the final products.
    durable_intermediates: bool = True

    def validate(self) -> None:
        if self.calibration_workflow not in WORKFLOWS:
            raise ValueError("unsupported calibration_workflow")
        if not isinstance(self.materialize_calibrated_lights, bool):
            raise ValueError("materialize_calibrated_lights must be a boolean")
        if not isinstance(self.durable_intermediates, bool):
            raise ValueError("durable_intermediates must be a boolean")
        self.integration.validate()
        if self.registration_memory_bytes < 1024:
            raise ValueError("registration_memory_bytes is too small")
        if self.registration_resampler not in REGISTRATION_RESAMPLERS:
            raise ValueError(
                "registration_resampler must be bilinear or lanczos-3-clamped"
            )
        if not 0 < self.minimum_crop_fraction <= 1:
            raise ValueError("minimum_crop_fraction must be in (0, 1]")
        if self.preview_max_long_edge < 16:
            raise ValueError("preview_max_long_edge must be at least 16")
        if (
            not math.isfinite(self.dark_temperature_tolerance_celsius)
            or self.dark_temperature_tolerance_celsius < 0
        ):
            raise ValueError("dark_temperature_tolerance_celsius must be finite and nonnegative")
        if self.ordinary_integration_backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                "ordinary_integration_backend must be auto, portable-cpu, "
                "generic-apple-metal, or m3-pro-tuned"
            )
        self.xisf_decode.validate()
        self.local_normalization.validate()
        self.global_normalization.validate()
        if self.local_normalization.enabled and self.global_normalization.enabled:
            raise ValueError(
                "local_normalization and global_normalization are mutually exclusive"
            )
        seen_overrides: set[str] = set()
        for override in self.master_metadata_overrides:
            override.validate()
            if override.source_sha256 in seen_overrides:
                raise ValueError("master_metadata_overrides contains a duplicate source digest")
            seen_overrides.add(override.source_sha256)
        seen_raw_overrides: set[str] = set()
        for override in self.raw_frame_metadata_overrides:
            override.validate()
            if override.source_sha256 in seen_raw_overrides:
                raise ValueError("raw_frame_metadata_overrides contains a duplicate source digest")
            seen_raw_overrides.add(override.source_sha256)

    def serializable(self) -> dict[str, Any]:
        return {
            "calibrationWorkflow": self.calibration_workflow,
            "calibrationPolicy": workflow_receipt(self.calibration_workflow),
            "integration": self.integration.serializable(),
            "registrationMemoryBytes": self.registration_memory_bytes,
            "registrationResampler": self.registration_resampler,
            "autoCrop": self.auto_crop,
            "minimumCropFraction": self.minimum_crop_fraction,
            "previewMaxLongEdge": self.preview_max_long_edge,
            "darkTemperatureToleranceCelsius": self.dark_temperature_tolerance_celsius,
            "ordinaryIntegrationBackend": self.ordinary_integration_backend,
            "nativeLibraryPath": self.native_library_path,
            "metalSourcePath": self.metal_source_path,
            "xisfDecode": self.xisf_decode.serializable(),
            "localNormalization": self.local_normalization.serializable(),
            "globalNormalization": self.global_normalization.serializable(),
            "masterMetadataOverrides": [
                item.serializable() for item in self.master_metadata_overrides
            ],
            "rawFrameMetadataOverrides": [
                item.serializable() for item in self.raw_frame_metadata_overrides
            ],
            "materializeCalibratedLights": self.materialize_calibrated_lights,
            "durableIntermediates": self.durable_intermediates,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_directory: str
    receipt_path: str
    state: str
    master_light_paths: tuple[str, ...]
    preview_paths: tuple[str, ...]

    def serializable(self) -> dict[str, Any]:
        return {
            "outputDirectory": self.output_directory,
            "receiptPath": self.receipt_path,
            "state": self.state,
            "masterLightPaths": list(self.master_light_paths),
            "previewPaths": list(self.preview_paths),
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


def _known_mismatch(left: Any, right: Any, unknown: Any = "UNKNOWN") -> bool:
    if left is None or right is None or left == unknown or right == unknown:
        return False
    return left != right


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
            left, right = cfa_for_workflow(left, workflow), cfa_for_workflow(right, workflow)
            if left != "NONE" or right != "NONE":
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


def _require_filter(info: FrameInfo) -> str:
    if info.filter_name == "UNKNOWN":
        raise CalibrationError(
            "FILTER_UNKNOWN", "Flat and Light frames require FILTER metadata", path=info.path
        )
    return info.filter_name


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    if not token:
        raise CalibrationError("FILENAME_TOKEN_EMPTY", f"cannot encode group name {value!r}")
    return token


def _exposure_token(value: float) -> str:
    return format(value, ".9g").replace("-", "m").replace(".", "p")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def _source_records(
    grouped: Sequence[tuple[str, tuple[Path, ...]]],
    source_aliases: Mapping[str, Path] | None = None,
    identity_cache: _SourceIdentityCache | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    records: list[dict[str, Any]] = []
    identities: dict[str, dict[str, int]] = {}
    for role, paths in grouped:
        for path in paths:
            original, digest, identity = _source_identity(
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
    identity_cache: _SourceIdentityCache,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, infos in grouped:
        for path, info in infos.items():
            original, digest, _ = _source_identity(
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
            actual = _stat_identity(path)
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
        "sha256": sha256 if sha256 is not None else _hash_file(path),
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
    trusted_generated: Mapping[str, _TrustedGeneratedMaster] | None = None,
    identity_cache: _SourceIdentityCache | None = None,
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
        original, digest, identity = _source_identity(
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


def _resolve_transforms(
    light_paths: tuple[Path, ...],
    transforms: Mapping[str, AffineTransform | Sequence[Sequence[float]]] | None,
) -> dict[Path, AffineTransform]:
    if transforms is None:
        return {path: AffineTransform.identity() for path in light_paths}
    if not all(isinstance(key, str) and key for key in transforms):
        raise CalibrationError("TRANSFORM_KEY_INVALID", "transform keys must be strings")
    basename_counts: dict[str, int] = {}
    for path in light_paths:
        basename_counts[path.name] = basename_counts.get(path.name, 0) + 1
    used: set[str] = set()
    result: dict[Path, AffineTransform] = {}
    # Callers may key transforms by any spelling of a Light path; resolve
    # every key once so symlinked temporary roots and relative paths match
    # the canonical Light list exactly as quality weights already do.
    canonical_keys: dict[str, list[str]] = {}
    for key in transforms:
        try:
            canonical = str(Path(key).expanduser().resolve(strict=True))
        except OSError:
            continue
        canonical_keys.setdefault(os.path.normcase(canonical), []).append(key)
    for path in light_paths:
        candidates = [str(path), os.path.normcase(str(path))]
        if basename_counts[path.name] == 1:
            candidates.append(path.name)
        candidates.extend(canonical_keys.get(os.path.normcase(str(path)), ()))
        matching = [key for key in candidates if key in transforms]
        matching = list(dict.fromkeys(matching))
        if len(matching) > 1:
            values = [AffineTransform.from_value(transforms[key]) for key in matching]
            matrices = [value.validated_matrix() for value in values]
            if not all(np.array_equal(matrices[0], matrix) for matrix in matrices[1:]):
                raise CalibrationError(
                    "TRANSFORM_KEY_CONFLICT", "multiple transform keys disagree", path=str(path)
                )
        if matching:
            key = matching[0]
            used.update(matching)
            result[path] = AffineTransform.from_value(transforms[key])
        else:
            result[path] = AffineTransform.identity()
    unused = sorted(set(transforms) - used)
    if unused:
        raise CalibrationError(
            "TRANSFORM_INPUT_UNKNOWN",
            f"transform keys do not match any Light: {', '.join(unused)}",
        )
    return result


def _resolve_quality_weights(
    light_paths: tuple[Path, ...],
    weights: Mapping[str, float] | None,
) -> dict[Path, float]:
    if weights is None:
        return {path: 1.0 for path in light_paths}
    if not all(isinstance(key, str) and key for key in weights):
        raise CalibrationError(
            "QUALITY_WEIGHT_KEY_INVALID", "quality weight keys must be non-empty strings"
        )
    canonical = {os.path.normcase(str(path)): path for path in light_paths}
    result: dict[Path, float] = {}
    for key, raw_value in weights.items():
        try:
            path = Path(key).expanduser().resolve(strict=True)
            value = float(raw_value)
        except (OSError, TypeError, ValueError) as error:
            raise CalibrationError(
                "QUALITY_WEIGHT_INVALID",
                "quality weights must bind existing Lights to finite positive numbers",
                path=key,
            ) from error
        canonical_key = os.path.normcase(str(path))
        if canonical_key not in canonical:
            raise CalibrationError(
                "QUALITY_WEIGHT_INPUT_UNKNOWN",
                "quality weight does not bind an integration Light",
                path=str(path),
            )
        if not math.isfinite(value) or value <= 0:
            raise CalibrationError(
                "QUALITY_WEIGHT_INVALID",
                "quality weights must be finite and positive",
                path=str(path),
            )
        bound = canonical[canonical_key]
        if bound in result:
            raise CalibrationError(
                "QUALITY_WEIGHT_KEY_CONFLICT",
                "multiple quality weight keys bind the same Light",
                path=str(path),
            )
        result[bound] = value
    missing = [str(path) for path in light_paths if path not in result]
    if missing:
        raise CalibrationError(
            "QUALITY_WEIGHT_SET_INCOMPLETE",
            "quality weights must bind every admitted Light: " + ", ".join(missing),
        )
    return result


def _resolve_stellar_scale_hints(
    light_paths: tuple[Path, ...],
    light_info: Mapping[Path, FrameInfo],
    hints: Mapping[str, StellarScaleHint] | None,
    source_aliases: Mapping[str, Path],
    identity_cache: _SourceIdentityCache,
) -> dict[Path, StellarScaleHint | None]:
    if hints is None:
        return {path: None for path in light_paths}
    if not all(isinstance(key, str) and key for key in hints):
        raise CalibrationError(
            "STELLAR_SCALE_HINT_KEY_INVALID",
            "stellar scale hint keys must be non-empty paths",
        )
    canonical = {os.path.normcase(str(path)): path for path in light_paths}
    result: dict[Path, StellarScaleHint | None] = {}
    for key, hint in hints.items():
        if not isinstance(hint, StellarScaleHint):
            raise CalibrationError(
                "STELLAR_SCALE_HINT_INVALID",
                "stellar scale hints must use the typed identity-bound contract",
                path=key,
            )
        try:
            source = Path(key).expanduser().resolve(strict=True)
            hinted_source = Path(hint.source_path).expanduser().resolve(strict=True)
            reference = Path(hint.reference_path).expanduser().resolve(strict=True)
        except OSError as error:
            raise CalibrationError(
                "STELLAR_SCALE_HINT_INVALID",
                "stellar scale hint paths must identify current Light inputs",
                path=key,
            ) from error
        source_key = os.path.normcase(str(source))
        reference_key = os.path.normcase(str(reference))
        if (
            source != hinted_source
            or source_key not in canonical
            or reference_key not in canonical
        ):
            raise CalibrationError(
                "STELLAR_SCALE_HINT_INPUT_MISMATCH",
                "stellar scale hint source/reference is outside the admitted Light set",
                path=key,
            )
        bound_source = canonical[source_key]
        bound_reference = canonical[reference_key]
        if bound_source in result:
            raise CalibrationError(
                "STELLAR_SCALE_HINT_KEY_CONFLICT",
                "multiple hints bind the same Light",
                path=key,
            )
        _, source_sha256, _ = _source_identity(
            bound_source, source_aliases, identity_cache
        )
        _, reference_sha256, _ = _source_identity(
            bound_reference, source_aliases, identity_cache
        )
        if (
            hint.source_sha256 != source_sha256
            or hint.reference_sha256 != reference_sha256
        ):
            raise CalibrationError(
                "STELLAR_SCALE_HINT_IDENTITY_MISMATCH",
                "stellar scale hint is not bound to the current source/reference bytes",
                path=key,
            )
        source_filter = light_info[bound_source].filter_name
        reference_filter = light_info[bound_reference].filter_name
        if (
            hint.filter_name != source_filter
            or source_filter != reference_filter
        ):
            raise CalibrationError(
                "STELLAR_SCALE_HINT_FILTER_MISMATCH",
                "stellar scale hints cannot cross optical filters",
                path=key,
            )
        if hint.status not in {
            "REFERENCE_IDENTITY",
            "STELLAR_SCALE_ACCEPTED",
            "STELLAR_SCALE_UNAVAILABLE",
        }:
            raise CalibrationError(
                "STELLAR_SCALE_HINT_STATUS_INVALID",
                "stellar scale hint has an unknown status",
                path=key,
            )
        if hint.status == "REFERENCE_IDENTITY":
            if bound_source != bound_reference or hint.scale != 1.0:
                raise CalibrationError(
                    "STELLAR_SCALE_HINT_REFERENCE_INVALID",
                    "reference hint must bind itself with scale 1",
                    path=key,
                )
        elif hint.status == "STELLAR_SCALE_ACCEPTED":
            if (
                hint.scale is None
                or not math.isfinite(hint.scale)
                or hint.scale <= 0
            ):
                raise CalibrationError(
                    "STELLAR_SCALE_HINT_INVALID",
                    "accepted stellar scale must be finite and positive",
                    path=key,
                )
        elif hint.scale is not None:
            raise CalibrationError(
                "STELLAR_SCALE_HINT_INVALID",
                "unavailable stellar scale must not carry a numeric scale",
                path=key,
            )
        result[bound_source] = hint
    missing = [str(path) for path in light_paths if path not in result]
    if missing:
        raise CalibrationError(
            "STELLAR_SCALE_HINT_SET_INCOMPLETE",
            "stellar scale hints must bind every admitted Light: "
            + ", ".join(missing),
        )
    return result


def _exact_half_turn_translation(
    transform: AffineTransform, shape: tuple[int, int]
) -> tuple[int, int] | None:
    """Recognize only an integer half-turn up to float64 arithmetic roundoff.

    Coefficient checks alone can hide a displacement on a long image axis.
    Bound the residual at every corner, separately for x and y; the affine
    residual between corners cannot exceed those bounds. Eight float64 epsilons
    allow numerical noise (including sin(pi)), not a fitted angular tolerance.
    """
    matrix = transform.validated_matrix()
    if not np.array_equal(matrix[2], (0.0, 0.0, 1.0)):
        return None
    linear_error = matrix[:2, :2] + np.eye(2)
    roundoff = 8.0 * np.finfo(np.float64).eps
    if np.any(np.abs(linear_error) > roundoff):
        return None
    translation = np.rint(matrix[:2, 2])
    translation_error = matrix[:2, 2] - translation
    if np.any(
        np.abs(translation_error) > roundoff * np.maximum(1.0, np.abs(translation))
    ):
        return None
    height, width = shape
    corners = np.asarray(
        ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)),
        dtype=np.float64,
    )
    residual = corners @ linear_error.T + translation_error
    coordinate_scale = np.maximum(1.0, np.maximum(corners, np.abs(translation)))
    if np.any(np.abs(residual) > roundoff * coordinate_scale):
        return None
    return int(translation[0]), int(translation[1])


def _registration_provenance(
    transform: AffineTransform, shape: tuple[int, int], resampler: str
) -> dict[str, str]:
    if transform.is_identity:
        actual, algorithm = "identity-exact", "identity-exact-copy-v1"
    elif _exact_half_turn_translation(transform, shape) is not None:
        actual, algorithm = "half-turn-exact", "half-turn-integer-copy-v1"
    else:
        actual = resampler
        algorithm = (
            LANCZOS3_REGISTRATION_ALGORITHM
            if resampler == "lanczos-3-clamped"
            else "bilinear-2x2-v1"
        )
    return {"resampler": actual, "resamplerAlgorithm": algorithm}


def _registration_metadata(
    info: FrameInfo,
    transform: AffineTransform,
    *,
    resampler: str,
    source_exposure_seconds: float | None = None,
) -> dict[str, Any]:
    actual_resampler = _registration_provenance(transform, info.shape, resampler)[
        "resampler"
    ]
    uses_lanczos = actual_resampler == "lanczos-3-clamped"
    return {
        "IMAGETYP": "Registered Light",
        "FILTER": info.filter_name,
        "OBJECT": info.target,
        "EXPTIME": info.exposure_seconds,
        "OAFSTATE": OUTPUT_STATE,
        "OAFREG": "IDENTITY" if transform.is_identity else "AFFINE",
        "OAFRSAMP": actual_resampler.upper(),
        "OAFRCLMP": "DOMAIN_UNION_SUPPORT" if uses_lanczos else None,
        "OAFRMARG": 2 if uses_lanczos else 0,
        "OAFSRCEX": source_exposure_seconds,
        **_numeric_domain_metadata(info),
    }


def _registration_bytes_per_pixel(
    transform: AffineTransform, resampler: str, shape: tuple[int, int]
) -> int:
    if transform.is_identity:
        return 24
    if _exact_half_turn_translation(transform, shape) is not None:
        # Includes the destination tile, scaled FITS conversion and previous
        # iteration's values/statistics buffers while the next tile is read.
        return 32
    return 192 if resampler == "lanczos-3-clamped" else 112


@dataclass(frozen=True, slots=True)
class _RegistrationJob:
    source_path: Path
    destination: Path
    transform: AffineTransform
    info: FrameInfo
    source_exposure_seconds: float | None = None


def _registration_worker_count(
    jobs: Sequence[_RegistrationJob],
    *,
    max_memory_bytes: int,
    resampler: str,
    cpu_workers: int,
) -> int:
    minimum_worker_bytes = max(
        (
            job.info.shape[1]
            * _registration_bytes_per_pixel(job.transform, resampler, job.info.shape)
            for job in jobs
        ),
        default=1,
    )
    # Eight simultaneous warps improve the measured M3 Pro executor workload.
    # Respect smaller hardware profiles and leave one full row per worker.
    # A budget below one row retains _register_frame's existing failure path.
    return max(1, min(8, cpu_workers, len(jobs), max_memory_bytes // minimum_worker_bytes))


def _register_frames(
    jobs: Sequence[_RegistrationJob],
    *,
    max_memory_bytes: int,
    resampler: str,
    cpu_workers: int,
    native_threads: int | None = None,
    execution_records: list[dict[str, Any]] | None = None,
) -> tuple[PixelStatistics, ...]:
    workers = _registration_worker_count(
        jobs,
        max_memory_bytes=max_memory_bytes,
        resampler=resampler,
        cpu_workers=cpu_workers,
    )
    worker_memory_bytes = max_memory_bytes // workers
    records: list[dict[str, Any]] = [{} for _ in jobs]

    def register(index: int) -> PixelStatistics:
        job = jobs[index]
        # Every worker opens its own read-only input and unique output writer.
        # Pixel buffers, writable headers, and accumulators are never shared.
        return _register_frame(
            job.source_path,
            job.destination,
            job.transform,
            job.info,
            max_memory_bytes=worker_memory_bytes,
            resampler=resampler,
            source_exposure_seconds=job.source_exposure_seconds,
            native_threads=native_threads,
            execution=records[index],
        )

    if workers == 1:
        results = tuple(register(index) for index in range(len(jobs)))
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wbpp-warp")
        try:
            futures = [executor.submit(register, index) for index in range(len(jobs))]
            # Keep provenance and artifact order independent of completion order.
            results = tuple(future.result() for future in futures)
        finally:
            # A failed warp must finish/cancel every writer before staging cleanup.
            executor.shutdown(wait=True, cancel_futures=True)
    if execution_records is not None:
        execution_records.extend(records)
    return results


def _read_half_turn_rows(
    source: Any, y0: int, y1: int, translation: tuple[int, int]
) -> NDArray[np.float32]:
    height, width = source.shape
    tx, ty = translation
    values = np.full((y1 - y0, width), np.nan, dtype=np.float32)
    left, right = max(0, tx - width + 1), min(width, tx + 1)
    top, bottom = max(y0, ty - height + 1), min(y1, ty + 1)
    if left < right and top < bottom:
        # Read at most this output tile's row count. Reversal is a view of the
        # physical Float32 values, so it neither interpolates nor clamps them.
        rows = source.read_rows(ty - bottom + 1, ty - top + 1)
        values[top - y0 : bottom - y0, left:right] = rows[
            ::-1, tx - right + 1 : tx - left + 1
        ][:, ::-1]
    return values


def _open_registration_source(source: Path | _MemoryFrame) -> Any:
    if isinstance(source, _MemoryFrame):
        return nullcontext(source)
    return FitsFrame(source)


def _register_frame(
    source: Path | _MemoryFrame,
    destination: Path,
    transform: AffineTransform,
    info: FrameInfo,
    *,
    max_memory_bytes: int,
    resampler: str,
    source_exposure_seconds: float | None = None,
    native_threads: int | None = None,
    execution: dict[str, Any] | None = None,
    durable: bool = True,
) -> PixelStatistics:
    """Resample one calibrated Light (file or in-memory) into a new FITS.

    Exact identity and integer half-turn transforms copy pixels.  General
    transforms use the native multithreaded Lanczos-3 kernel when it is
    available and the budget holds the decoded source; otherwise the NumPy
    reference resampler runs on bounded coordinate tiles.  Both produce
    value-identical output.  ``execution`` receives the backend actually used.
    """

    matrix = transform.validated_matrix()
    inverse = np.linalg.inv(matrix)
    with _open_registration_source(source) as frame:
        height, width = frame.shape
        is_identity = transform.is_identity
        half_turn = _exact_half_turn_translation(transform, frame.shape)
        kernels = None
        source_values: NDArray[np.float32] | None = None
        domain_scale = frame.info.normalized_unit_scale
        if is_identity:
            warp_backend = "identity-copy"
        elif half_turn is not None:
            warp_backend = "half-turn-copy"
        else:
            warp_backend = "numpy"
        if (
            warp_backend == "numpy"
            and resampler == "lanczos-3-clamped"
            and domain_scale is not None
            and math.isfinite(domain_scale)
            and domain_scale > 0.0
        ):
            kernels = load_native_kernels()
        if kernels is not None:
            decoded_bytes = 0 if isinstance(frame, _MemoryFrame) else height * width * 4
            native_row_bytes = width * NATIVE_WARP_BYTES_PER_PIXEL
            if decoded_bytes + native_row_bytes <= max_memory_bytes:
                warp_backend = "native-cpu"
                tile_rows = max(
                    1,
                    min(height, (max_memory_bytes - decoded_bytes) // native_row_bytes),
                )
            else:
                kernels = None
        if warp_backend != "native-cpu":
            bytes_per_pixel = _registration_bytes_per_pixel(
                transform, resampler, frame.shape
            )
            bytes_per_row = width * bytes_per_pixel
            if bytes_per_row > max_memory_bytes:
                raise CalibrationError(
                    "MEMORY_BUDGET_TOO_SMALL", "one registration row exceeds memory budget"
                )
            tile_rows = max(1, min(height, max_memory_bytes // bytes_per_row))
        temporary = destination.with_name(f".{destination.name}.partial")
        if temporary.exists() or os.path.lexists(temporary):
            raise CalibrationError(
                "OUTPUT_EXISTS", "registration temporary already exists", path=str(temporary)
            )
        stats_min = math.inf
        stats_max = -math.inf
        stats_sum = 0.0
        finite_total = 0
        invalid_total = 0
        digest: str | None = None
        try:
            with FitsFloatWriter(
                temporary,
                frame.shape,
                _registration_metadata(
                    info,
                    transform,
                    resampler=resampler,
                    source_exposure_seconds=source_exposure_seconds,
                ),
                durable=durable,
            ) as writer:
                if warp_backend == "native-cpu":
                    source_values = frame.full_values()
                for y0 in range(0, height, tile_rows):
                    y1 = min(height, y0 + tile_rows)
                    if is_identity:
                        values = frame.read_rows(y0, y1)
                    elif half_turn is not None:
                        values = _read_half_turn_rows(frame, y0, y1, half_turn)
                    elif warp_backend == "native-cpu":
                        assert kernels is not None and source_values is not None
                        values = kernels.warp_lanczos3(
                            source_values,
                            inverse[:2],
                            first_row=y0,
                            row_count=y1 - y0,
                            output_width=width,
                            domain_scale=float(domain_scale),
                            threads=native_threads,
                        )
                    else:
                        output_y = np.arange(y0, y1, dtype=np.float64)[:, None]
                        output_x = np.arange(width, dtype=np.float64)[None, :]
                        input_x = (
                            inverse[0, 0] * output_x
                            + inverse[0, 1] * output_y
                            + inverse[0, 2]
                        )
                        input_y = (
                            inverse[1, 0] * output_x
                            + inverse[1, 1] * output_y
                            + inverse[1, 2]
                        )
                        input_x = np.broadcast_to(input_x, (y1 - y0, width))
                        input_y = np.broadcast_to(input_y, (y1 - y0, width))
                        if resampler == "lanczos-3-clamped":
                            values = frame.sample_lanczos3_clamped(input_x, input_y)
                        else:
                            values = frame.sample_bilinear(input_x, input_y)
                    finite = np.isfinite(values)
                    count = int(np.count_nonzero(finite))
                    finite_total += count
                    invalid_total += int(values.size - count)
                    if count:
                        selected = values[finite]
                        stats_min = min(stats_min, float(np.min(selected)))
                        stats_max = max(stats_max, float(np.max(selected)))
                        stats_sum += float(np.sum(selected, dtype=np.float64))
                    writer.write_rows(y0, values)
            digest = writer.sha256
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise CalibrationError(
                    "OUTPUT_EXISTS",
                    "refusing to overwrite registered frame",
                    path=str(destination),
                ) from error
            temporary.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()
    if execution is not None:
        execution.update(
            {
                "warpBackend": warp_backend,
                "warpKernel": (
                    WARP_KERNEL_ID
                    if warp_backend == "native-cpu"
                    else NUMPY_WARP_KERNEL_ID
                    if warp_backend == "numpy" and resampler == "lanczos-3-clamped"
                    else warp_backend
                ),
                "tileRows": tile_rows,
                "nativeThreads": native_threads if warp_backend == "native-cpu" else None,
                "sha256": digest,
            }
        )
    return PixelStatistics(
        finite_pixels=finite_total,
        invalid_pixels=invalid_total,
        minimum=stats_min if finite_total else None,
        maximum=stats_max if finite_total else None,
        mean=stats_sum / finite_total if finite_total else None,
    )


@dataclass(frozen=True, slots=True)
class _LightJob:
    """One Light's fused calibrate-in-memory then register work item."""

    source_path: Path
    expression: FrameExpression
    calibrated_path: Path | None
    calibrated_metadata: Mapping[str, Any]
    destination: Path
    transform: AffineTransform
    info: FrameInfo
    source_exposure_seconds: float | None


@dataclass(frozen=True, slots=True)
class _LightJobResult:
    calibrated_statistics: PixelStatistics
    calibrated_sha256: str | None
    registered_statistics: PixelStatistics
    registered_sha256: str | None
    execution: dict[str, Any]


class _MasterCache:
    """Decoded Float32 masters shared read-only by every fused worker.

    Each master is converted from its FITS storage exactly once; workers
    receive copies of the rows they ask for, so the cache is never mutated.
    """

    def __init__(self) -> None:
        self._frames: dict[str, _MemoryFrame] = {}
        self._lock = threading.Lock()

    def frame(self, path: str) -> _MemoryFrame:
        key = str(Path(path).expanduser().resolve(strict=True))
        with self._lock:
            cached = self._frames.get(key)
            if cached is None:
                with FitsFrame(key) as source:
                    cached = _MemoryFrame(source.full_values(), source.info, key)
                self._frames[key] = cached
            return cached

    @property
    def decoded_bytes(self) -> int:
        with self._lock:
            return sum(frame.values.nbytes for frame in self._frames.values())


def _write_float_fits(
    values: NDArray[np.float32],
    destination: Path,
    metadata: Mapping[str, Any],
    *,
    durable: bool = True,
) -> str | None:
    """Publish one in-memory Float32 image atomically and return its digest."""

    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output", path=str(destination)
        )
    temporary = _temporary_output(destination)
    try:
        with FitsFloatWriter(temporary, values.shape, metadata, durable=durable) as writer:
            writer.write_rows(0, values)
        _atomic_publish_file(temporary, destination)
        return writer.sha256
    finally:
        if temporary.exists():
            temporary.unlink()


def _process_light_job(
    job: _LightJob,
    *,
    master_cache: _MasterCache,
    max_memory_bytes: int,
    resampler: str,
    native_threads: int,
    division_floor: float,
    durable: bool = True,
) -> _LightJobResult:
    expression = _canonical_expression(job.expression)
    sources: dict[str, Any] = {}
    with FitsFrame(expression.source_path) as light:
        sources[expression.source_path] = light
        for master_path in (
            expression.subtract_path,
            *expression.subtract_paths,
            expression.divide_path,
        ):
            if master_path is not None:
                sources[master_path] = master_cache.frame(master_path)
        height, _width = _validate_expression_shapes((expression,), sources)
        # The Light's decoded Float32 buffer becomes the calibrated image in
        # place: the same arithmetic as write_expression, without a disk trip.
        calibrated = _expression_rows(
            expression, sources, 0, height, division_floor=division_floor
        )
    statistics = _StatsAccumulator()
    statistics.update(calibrated)
    calibrated_statistics = statistics.result()
    if calibrated_statistics.finite_pixels == 0:
        raise CalibrationError(
            "NO_FINITE_OUTPUT", "calibration produced no finite pixels",
            path=str(job.source_path),
        )
    calibrated_sha256: str | None = None
    if job.calibrated_path is not None:
        calibrated_sha256 = _write_float_fits(
            calibrated, job.calibrated_path, job.calibrated_metadata, durable=durable
        )
    memory_frame = _MemoryFrame(
        calibrated, job.info, job.calibrated_path or job.source_path
    )
    execution: dict[str, Any] = {}
    # The calibrated image already occupies its share; leave the rest of the
    # worker budget to warp tiles, but always allow at least one NumPy row.
    warp_budget = max(
        max_memory_bytes - calibrated.nbytes,
        calibrated.shape[1]
        * _registration_bytes_per_pixel(job.transform, resampler, calibrated.shape),
    )
    registered_statistics = _register_frame(
        memory_frame,
        job.destination,
        job.transform,
        job.info,
        max_memory_bytes=warp_budget,
        resampler=resampler,
        source_exposure_seconds=job.source_exposure_seconds,
        native_threads=native_threads,
        execution=execution,
        durable=durable,
    )
    registered_sha256 = execution.pop("sha256", None)
    return _LightJobResult(
        calibrated_statistics=calibrated_statistics,
        calibrated_sha256=calibrated_sha256,
        registered_statistics=registered_statistics,
        registered_sha256=registered_sha256,
        execution=execution,
    )


def _fused_job_bytes(job: _LightJob, resampler: str) -> int:
    height, width = job.info.shape
    per_row = width * max(
        NATIVE_WARP_BYTES_PER_PIXEL,
        _registration_bytes_per_pixel(job.transform, resampler, job.info.shape),
    )
    return height * width * FUSED_LIGHT_BYTES_PER_PIXEL + per_row


def _fused_worker_count(
    jobs: Sequence[_LightJob],
    *,
    max_memory_bytes: int,
    resampler: str,
    cpu_workers: int,
) -> int:
    largest = max((_fused_job_bytes(job, resampler) for job in jobs), default=1)
    return max(1, min(cpu_workers, len(jobs), max_memory_bytes // max(1, largest)))


def _calibrate_and_register_frames(
    jobs: Sequence[_LightJob],
    *,
    master_cache: _MasterCache,
    max_memory_bytes: int,
    resampler: str,
    cpu_workers: int,
    division_floor: float,
    durable: bool = True,
) -> tuple[tuple[_LightJobResult, ...], dict[str, Any]]:
    """Calibrate and register every Light with one shared memory budget.

    Lights run concurrently in ``workers`` threads; each thread hands its warp
    to the native kernel with the remaining CPU share, so all cores stay busy
    whether memory allows many Lights in flight or only one.
    """

    workers = _fused_worker_count(
        jobs,
        max_memory_bytes=max_memory_bytes,
        resampler=resampler,
        cpu_workers=cpu_workers,
    )
    native_threads = max(1, cpu_workers // workers)
    worker_memory_bytes = max_memory_bytes // workers
    # Lights start in submission order, so the last ``len(jobs) % workers``
    # Lights run while the other workers are already idle; their warps take
    # the CPU share those workers would have used. Warp results do not depend
    # on the thread count.
    rounds = max(1, math.ceil(len(jobs) / workers))
    tail_start = (rounds - 1) * workers
    tail_threads = max(native_threads, cpu_workers // max(1, len(jobs) - tail_start))

    def run(index: int, job: _LightJob) -> _LightJobResult:
        return _process_light_job(
            job,
            master_cache=master_cache,
            max_memory_bytes=worker_memory_bytes,
            resampler=resampler,
            native_threads=tail_threads if index >= tail_start else native_threads,
            division_floor=division_floor,
            durable=durable,
        )

    if workers == 1:
        results = tuple(run(index, job) for index, job in enumerate(jobs))
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wbpp-light")
        try:
            futures = [executor.submit(run, index, job) for index, job in enumerate(jobs)]
            results = tuple(future.result() for future in futures)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    backends: dict[str, int] = {}
    for result in results:
        backend = str(result.execution.get("warpBackend", "unknown"))
        backends[backend] = backends.get(backend, 0) + 1
    return results, {
        "executor": "thread-pool" if workers > 1 else "serial",
        "executionModel": "fused-calibrate-warp-v1",
        "cpuWorkersUsed": workers,
        "nativeThreadsPerWorker": native_threads,
        "tailNativeThreads": tail_threads,
        "tailLights": len(jobs) - tail_start,
        "perWorkerMemoryBudgetBytes": worker_memory_bytes,
        "warpBackends": backends,
        "masterCacheBytes": master_cache.decoded_bytes,
    }


def _histogram_rectangle(
    heights: NDArray[np.int64], row_index: int
) -> tuple[int, int, int, int, int] | None:
    best: tuple[int, int, int, int, int] | None = None
    stack: list[tuple[int, int]] = []
    width = int(heights.size)
    if not width:
        return None
    # Repeating a height cannot push or pop the monotone stack. Registered
    # footprints typically leave thousands of adjacent columns at the same
    # height, so locate run boundaries in NumPy and visit only those in Python.
    # Keep the original coordinates and sentinel: candidates and their exact
    # lexicographic tie-break remain identical, including masks with holes.
    boundaries = np.flatnonzero(heights[1:] != heights[:-1]) + 1
    positions = [0, *boundaries.tolist(), width]
    levels = [int(heights[0]), *heights[boundaries].tolist(), 0]
    for x, height in zip(positions, levels, strict=True):
        start = x
        while stack and stack[-1][1] > height:
            left, previous_height = stack.pop()
            area = previous_height * (x - left)
            candidate = (
                area,
                row_index - previous_height + 1,
                left,
                row_index + 1,
                x,
            )
            if best is None or candidate > best:
                best = candidate
            start = left
        if height and (not stack or stack[-1][1] < height):
            stack.append((start, height))
    return best


def _common_valid_crop(
    shape: tuple[int, int],
    transforms: Sequence[AffineTransform],
    *,
    max_memory_bytes: int,
    resampler: str,
) -> tuple[int, int, int, int]:
    if not transforms:
        raise CalibrationError("NO_REGISTERED_INPUTS", "no registration transforms")
    height, width = shape
    bytes_per_row = width * 56
    if bytes_per_row > max_memory_bytes:
        raise CalibrationError(
            "MEMORY_BUDGET_TOO_SMALL", "one crop-mask row exceeds memory budget"
        )
    tile_rows = max(1, min(height, max_memory_bytes // bytes_per_row))
    inverses = []
    margins = []
    for transform in transforms:
        half_turn = _exact_half_turn_translation(transform, shape)
        if half_turn is not None:
            tx, ty = half_turn
            # Match the exact integer map actually used by the row copier.
            inverses.append(
                np.asarray(
                    ((-1, 0, tx), (0, -1, ty), (0, 0, 1)), dtype=np.float64
                )
            )
        else:
            inverses.append(np.linalg.inv(transform.validated_matrix()))
        margins.append(
            2
            if not transform.is_identity
            and half_turn is None
            and resampler == "lanczos-3-clamped"
            else 0
        )
    heights = np.zeros(width, dtype=np.int64)
    best: tuple[int, int, int, int, int] | None = None
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        output_y = np.arange(y0, y1, dtype=np.float64)[:, None]
        output_x = np.arange(width, dtype=np.float64)[None, :]
        common = np.ones((y1 - y0, width), dtype=bool)
        for inverse, interpolation_margin in zip(inverses, margins, strict=True):
            input_x = (
                inverse[0, 0] * output_x
                + inverse[0, 1] * output_y
                + inverse[0, 2]
            )
            input_y = (
                inverse[1, 0] * output_x
                + inverse[1, 1] * output_y
                + inverse[1, 2]
            )
            common &= (
                (input_x >= interpolation_margin)
                & (input_x <= width - 1 - interpolation_margin)
                & (input_y >= interpolation_margin)
                & (input_y <= height - 1 - interpolation_margin)
            )
        for local_y, row in enumerate(common):
            heights = np.where(row, heights + 1, 0)
            candidate = _histogram_rectangle(heights, y0 + local_y)
            if candidate is not None and (best is None or candidate > best):
                best = candidate
    if best is None or best[0] == 0:
        raise CalibrationError(
            "AUTOCROP_EMPTY", "registered inputs have no common finite rectangle"
        )
    _, top, left, bottom, right = best
    return top, left, bottom, right


def _crop_fits(
    source_path: Path,
    destination: Path,
    crop: tuple[int, int, int, int],
    metadata: Mapping[str, Any],
    *,
    max_memory_bytes: int,
    durable: bool = True,
) -> tuple[PixelStatistics, str | None]:
    """Crop one FITS into a new file; returns statistics and the writer digest."""
    top, left, bottom, right = crop
    with FitsFrame(source_path) as source:
        height, width = source.shape
        if not (0 <= top < bottom <= height and 0 <= left < right <= width):
            raise CalibrationError("AUTOCROP_INVALID", "crop is outside image bounds")
        output_shape = (bottom - top, right - left)
        bytes_per_row = output_shape[1] * 12
        if bytes_per_row > max_memory_bytes:
            raise CalibrationError(
                "MEMORY_BUDGET_TOO_SMALL", "one crop row exceeds memory budget"
            )
        tile_rows = max(1, min(output_shape[0], max_memory_bytes // bytes_per_row))
        temporary = destination.with_name(f".{destination.name}.partial")
        if temporary.exists() or os.path.lexists(temporary):
            raise CalibrationError("OUTPUT_EXISTS", "crop temporary already exists")
        finite_total = 0
        invalid_total = 0
        total = 0.0
        minimum = math.inf
        maximum = -math.inf
        try:
            with FitsFloatWriter(
                temporary, output_shape, metadata, durable=durable
            ) as writer:
                output_y = 0
                for source_y in range(top, bottom, tile_rows):
                    source_y1 = min(bottom, source_y + tile_rows)
                    values = source.read_rows(source_y, source_y1)[:, left:right]
                    finite = np.isfinite(values)
                    count = int(np.count_nonzero(finite))
                    finite_total += count
                    invalid_total += int(values.size - count)
                    if count:
                        selected = values[finite]
                        minimum = min(minimum, float(np.min(selected)))
                        maximum = max(maximum, float(np.max(selected)))
                        total += float(np.sum(selected, dtype=np.float64))
                    writer.write_rows(output_y, values)
                    output_y += values.shape[0]
            digest = writer.sha256
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise CalibrationError(
                    "OUTPUT_EXISTS", "refusing to overwrite master", path=str(destination)
                ) from error
            temporary.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()
    return PixelStatistics(
        finite_pixels=finite_total,
        invalid_pixels=invalid_total,
        minimum=minimum if finite_total else None,
        maximum=maximum if finite_total else None,
        mean=total / finite_total if finite_total else None,
    ), digest


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(destination)
        )
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as error:
            raise CalibrationError(
                "OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(destination)
            ) from error
        return
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        result = library.renamex_np(encoded_source, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        at_fdcwd = -100
        result = library.renameat2(
            at_fdcwd, encoded_source, at_fdcwd, encoded_destination, 0x00000001
        )
    else:
        raise CalibrationError(
            "ATOMIC_DIRECTORY_PUBLISH_UNSUPPORTED",
            "platform has no no-replace directory rename primitive",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise CalibrationError(
                "OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(destination)
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


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


def _run_portable_pipeline_fits(
    *,
    bias_files: Iterable[str | os.PathLike[str]] = (),
    dark_files: Iterable[str | os.PathLike[str]] = (),
    flat_files: Iterable[str | os.PathLike[str]] = (),
    master_bias_file: str | os.PathLike[str] | None = None,
    master_dark_files: Iterable[str | os.PathLike[str]] = (),
    master_flat_files: Iterable[str | os.PathLike[str]] = (),
    light_files: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    transforms: Mapping[
        str, AffineTransform | Sequence[Sequence[float]]
    ]
    | None = None,
    quality_weights: Mapping[str, float] | None = None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None = None,
    parameters: PipelineParameters | None = None,
    _source_aliases: Mapping[str, Path] | None = None,
    _xisf_conversions: Sequence[Mapping[str, Any]] = (),
    _trusted_generated_calibration: _TrustedGeneratedCalibrationSet | None = None,
    _source_identity_seed: Mapping[str, tuple[str, Mapping[str, int]]] | None = None,
) -> PipelineResult:
    """Run raw or pre-integrated calibration through unsolved linear masters.

    ``_source_identity_seed`` maps canonical original paths to a content
    digest and the stat identity that digest was captured with.  Seeded
    sources are not rehashed; a stat mismatch against the seed still fails
    closed as ``SOURCE_CHANGED``.

    A calibration profile may use raw frames or a supplied master, never both
    for the same bias/filter/exposure identity.  Supplied masters are opened
    read-only and reused directly; they are never integrated or calibrated a
    second time.
    """

    parameters = parameters or PipelineParameters()
    parameters.validate()
    biases = _canonical_inputs(bias_files, "Bias", required=False)
    darks = _canonical_inputs(dark_files, "Dark", required=False)
    flats = _canonical_inputs(flat_files, "Flat", required=False)
    master_biases = _canonical_inputs(
        (() if master_bias_file is None else (master_bias_file,)),
        "MasterBias",
        required=False,
    )
    master_darks_input = _canonical_inputs(
        master_dark_files, "MasterDark", required=False
    )
    master_flats_input = _canonical_inputs(
        master_flat_files, "MasterFlat", required=False
    )
    lights = _canonical_inputs(light_files, "Light", required=True)
    if (biases and master_biases) or (not biases and not master_biases and parameters.calibration_workflow != MONO_STANDARD):
        raise CalibrationError(
            "BIAS_SOURCE_AMBIGUOUS",
            "supply exactly one Bias source mode: raw Bias frames or one MasterBias",
        )
    all_paths = (
        *biases,
        *darks,
        *flats,
        *master_biases,
        *master_darks_input,
        *master_flats_input,
        *lights,
    )
    if len({os.path.normcase(str(path)) for path in all_paths}) != len(all_paths):
        raise CalibrationError("INPUT_ROLE_OVERLAP", "one source appears in multiple roles")

    source_aliases = dict(_source_aliases or {})
    source_groups = (
        ("BIAS", biases),
        ("DARK", darks),
        ("FLAT", flats),
        ("MASTER_BIAS", master_biases),
        ("MASTER_DARK", master_darks_input),
        ("MASTER_FLAT", master_flats_input),
        ("LIGHT", lights),
    )
    source_identity_cache: _SourceIdentityCache = {}
    for seed_path, (seed_digest, seed_identity) in (_source_identity_seed or {}).items():
        seed_key = os.path.normcase(str(Path(seed_path).expanduser().resolve(strict=True)))
        source_identity_cache[seed_key] = (str(seed_digest), dict(seed_identity))
    trusted_generated: dict[str, Any] | None = None
    if _trusted_generated_calibration is not None:
        trusted_generated = _validate_trusted_generated_calibration_set(
            _trusted_generated_calibration,
            source_groups=source_groups,
            source_aliases=source_aliases,
            identity_cache=source_identity_cache,
        )
        original_keys = {os.path.normcase(str(path)) for path in all_paths}
        if original_keys.intersection(trusted_generated["byPath"]):
            raise CalibrationError(
                "TRUSTED_GENERATED_MASTER_INPUT_OVERLAP",
                "E2E-generated masters cannot be presented as public input files",
            )

    output = Path(output_directory).expanduser().resolve(strict=False)
    if output.exists() or os.path.lexists(output):
        raise CalibrationError(
            "OUTPUT_EXISTS", "output directory must be new", path=str(output)
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    bias_info = _read_infos(biases, "BIAS")
    dark_info = _read_infos(darks, "DARK")
    flat_info = _read_infos(flats, "FLAT")
    master_bias_info = _read_infos(master_biases, "MASTER_BIAS")
    master_dark_info = _read_infos(master_darks_input, "MASTER_DARK")
    master_flat_info = _read_infos(master_flats_input, "MASTER_FLAT")
    light_info = _read_infos(lights, "LIGHT")
    (
        bias_info,
        dark_info,
        flat_info,
        master_bias_info,
        master_dark_info,
        master_flat_info,
        light_info,
    ) = _trust_private_xisf_numeric_domains(
        (
            bias_info,
            dark_info,
            flat_info,
            master_bias_info,
            master_dark_info,
            master_flat_info,
            light_info,
        ),
        source_aliases,
    )
    bias_info, dark_info, flat_info, light_info = _apply_raw_frame_metadata_overrides(
        (bias_info, dark_info, flat_info, light_info),
        parameters.raw_frame_metadata_overrides,
        dict(_source_aliases or {}),
        source_identity_cache,
    )
    (
        master_bias_info,
        master_dark_info,
        master_flat_info,
    ) = _apply_master_metadata_overrides(
        (master_bias_info, master_dark_info, master_flat_info),
        parameters.master_metadata_overrides,
        dict(_source_aliases or {}),
        source_identity_cache,
    )
    supplied_dark_bias_included = _master_dark_bias_semantics(
        master_darks_input,
        parameters.master_metadata_overrides,
        dict(_source_aliases or {}),
        source_identity_cache,
        workflow=parameters.calibration_workflow,
    )
    for group in (bias_info, dark_info, flat_info, master_bias_info, master_dark_info, master_flat_info, light_info):
        for path, info in group.items():
            group[path] = apply_mono_workflow(info, parameters.calibration_workflow)
    reference_bias = (
        bias_info[biases[0]] if biases else master_bias_info[master_biases[0]] if master_biases else light_info[lights[0]]
    )
    profile_infos = [*bias_info.values(), *master_bias_info.values(), *dark_info.values(), *flat_info.values(), *master_dark_info.values(), *master_flat_info.values(), *light_info.values()]
    conflicts = conflicting_profile_fields(profile_infos, parameters.calibration_workflow)
    if conflicts:
        raise CalibrationError("CALIBRATION_PROFILE_MISMATCH", "Conflicting known acquisition metadata: " + ", ".join(conflicts))
    if not biases and not master_biases and not can_omit_bias(
        (*light_info.values(), *flat_info.values()),
        [*((info, True) for info in dark_info.values()), *((info, supplied_dark_bias_included[path]) for path, info in master_dark_info.items())],
        parameters.calibration_workflow,
    ):
        raise CalibrationError("BIAS_REQUIRED_FOR_CALIBRATION", "Bias is required unless every Light and raw Flat has a matching Dark that includes Bias.")
    for info in (*bias_info.values(), *master_bias_info.values()):
        _assert_compatible(reference_bias, info, workflow=parameters.calibration_workflow)
    for info in (
        *dark_info.values(),
        *flat_info.values(),
        *master_dark_info.values(),
        *master_flat_info.values(),
        *light_info.values(),
    ):
        _assert_compatible(reference_bias, info, workflow=parameters.calibration_workflow)

    flat_groups: dict[str, list[Path]] = {}
    for path, info in flat_info.items():
        flat_groups.setdefault(_require_filter(info), []).append(path)
    supplied_flats: dict[str, Path] = {}
    for path, info in master_flat_info.items():
        filter_name = _require_filter(info)
        if filter_name in supplied_flats:
            raise CalibrationError(
                "MASTER_FLAT_AMBIGUOUS",
                f"multiple supplied MasterFlats match filter {filter_name}",
            )
        supplied_flats[filter_name] = path
    light_groups: dict[str, list[Path]] = {}
    for path, info in light_info.items():
        if info.exposure_seconds is None or info.exposure_seconds <= 0:
            raise CalibrationError(
                "LIGHT_EXPOSURE_UNKNOWN",
                "Light requires positive EXPTIME",
                path=str(path),
            )
        light_groups.setdefault(_require_filter(info), []).append(path)
    for filter_name, paths in light_groups.items():
        reference = light_info[paths[0]]
        for path in paths[1:]:
            _assert_compatible(
                reference,
                light_info[path],
                compare_filter=True,
                compare_target=True,
                workflow=parameters.calibration_workflow,
            )
    reference_exposures = {
        filter_name: min(
            float(light_info[path].exposure_seconds) for path in paths
        )
        for filter_name, paths in light_groups.items()
    }
    light_domain_references = {
        filter_name: light_info[paths[0]] for filter_name, paths in light_groups.items()
    }
    for filter_name, paths in flat_groups.items():
        reference = flat_info[paths[0]]
        for path in paths[1:]:
            _assert_compatible(reference, flat_info[path], compare_filter=True, workflow=parameters.calibration_workflow)
        if filter_name in supplied_flats:
            raise CalibrationError(
                "FLAT_SOURCE_AMBIGUOUS",
                f"filter {filter_name} has both raw Flats and a supplied MasterFlat",
            )
    missing_flats = sorted(set(light_groups) - set(flat_groups) - set(supplied_flats))
    if missing_flats:
        raise CalibrationError(
            "MASTER_FLAT_MISSING",
            f"no raw Flat group or MasterFlat for Light filters: {', '.join(missing_flats)}",
        )
    tokens: dict[str, str] = {}
    for filter_name in {*flat_groups, *supplied_flats, *light_groups}:
        token = _safe_token(filter_name)
        if token in tokens and tokens[token] != filter_name:
            raise CalibrationError(
                "FILTER_FILENAME_COLLISION",
                f"filters {tokens[token]!r} and {filter_name!r} share output token {token}",
            )
        tokens[token] = filter_name

    dark_groups: dict[float, list[Path]] = {}
    for path, info in dark_info.items():
        if info.exposure_seconds is None or info.exposure_seconds <= 0:
            raise CalibrationError(
                "DARK_EXPOSURE_UNKNOWN", "Dark requires positive EXPTIME", path=str(path)
            )
        dark_groups.setdefault(info.exposure_seconds, []).append(path)
    supplied_darks: dict[float, Path] = {}
    for path, info in master_dark_info.items():
        if info.exposure_seconds is None or info.exposure_seconds <= 0:
            raise CalibrationError(
                "DARK_EXPOSURE_UNKNOWN",
                "MasterDark requires positive EXPTIME",
                path=str(path),
            )
        if _find_dark(info.exposure_seconds, supplied_darks) is not None:
            raise CalibrationError(
                "MASTER_DARK_AMBIGUOUS",
                "multiple supplied MasterDarks have the same exposure",
                path=str(path),
            )
        if _find_dark(
            info.exposure_seconds, {value: Path() for value in dark_groups}
        ) is not None:
            raise CalibrationError(
                "DARK_SOURCE_AMBIGUOUS",
                "an exposure has both raw Darks and a supplied MasterDark",
                path=str(path),
            )
        supplied_darks[info.exposure_seconds] = path
    available_dark_exposures = {
        **{value: Path() for value in dark_groups},
        **supplied_darks,
    }
    if available_dark_exposures:
        for path, info in light_info.items():
            if _find_dark(info.exposure_seconds, available_dark_exposures) is None:
                raise CalibrationError(
                    "DARK_EXPOSURE_MISMATCH",
                    "no exact raw Dark or MasterDark exposure matches this Light",
                    path=str(path),
                )

    trusted_bias = trusted_generated["bias"] if trusted_generated is not None else None
    trusted_darks_by_exposure: dict[float, _TrustedGeneratedMaster] = {}
    trusted_flats_by_filter: dict[str, _TrustedGeneratedMaster] = {}
    if trusted_generated is not None:
        if bool(biases) != (trusted_bias is not None):
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "generated MasterBias coverage does not match raw Bias provenance",
            )
        for item in trusted_generated["darks"]:
            exposure = item.frame_info.exposure_seconds
            if exposure is None or exposure <= 0 or _find_dark(
                exposure, {value: Path() for value in trusted_darks_by_exposure}
            ) is not None:
                raise CalibrationError(
                    "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                    "generated MasterDark exposures are invalid or ambiguous",
                    path=item.path,
                )
            trusted_darks_by_exposure[float(exposure)] = item
        for item in trusted_generated["flats"]:
            filter_name = _require_filter(item.frame_info)
            if filter_name in trusted_flats_by_filter:
                raise CalibrationError(
                    "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                    "generated MasterFlat filters are ambiguous",
                    path=item.path,
                )
            trusted_flats_by_filter[filter_name] = item
        if set(trusted_darks_by_exposure) != set(dark_groups):
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "generated MasterDark coverage does not match raw Dark provenance",
            )
        if set(trusted_flats_by_filter) != set(flat_groups):
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "generated MasterFlat coverage does not match raw Flat provenance",
            )
        if trusted_bias is not None:
            _assert_compatible(reference_bias, trusted_bias.frame_info, workflow=parameters.calibration_workflow)
        for exposure, item in trusted_darks_by_exposure.items():
            _assert_compatible(
                dark_info[dark_groups[exposure][0]],
                item.frame_info,
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
                workflow=parameters.calibration_workflow,
            )
        for filter_name, item in trusted_flats_by_filter.items():
            _assert_compatible(
                flat_info[flat_groups[filter_name][0]],
                item.frame_info,
                compare_filter=True,
                workflow=parameters.calibration_workflow,
            )

    resolved_transforms = _resolve_transforms(lights, transforms)
    resolved_quality_weights = _resolve_quality_weights(lights, quality_weights)
    resolved_stellar_scale_hints = _resolve_stellar_scale_hints(
        lights,
        light_info,
        stellar_scale_hints,
        source_aliases,
        source_identity_cache,
    )
    display_path = lambda path: source_aliases.get(str(path), path)  # noqa: E731
    source_records, source_identities = _source_records(
        source_groups,
        source_aliases,
        source_identity_cache,
    )
    pixel_numeric_domains = _pixel_numeric_domain_records(
        (
            ("BIAS", bias_info),
            ("DARK", dark_info),
            ("FLAT", flat_info),
            ("MASTER_BIAS", master_bias_info),
            ("MASTER_DARK", master_dark_info),
            ("MASTER_FLAT", master_flat_info),
            ("LIGHT", light_info),
        ),
        source_aliases,
        source_identity_cache,
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent)
    )
    published = False
    metal_executor: NativeMetalExecutor | None = None
    metal_unavailable_reason: str | None = None
    artifacts: list[dict[str, Any]] = []
    stage_statistics: dict[str, Any] = {}
    try:
        masters_dir = staging / "masters"
        calibrated_dir = staging / "calibrated"
        registered_dir = staging / "registered"
        normalization_dir = staging / "local-normalization"
        coverage_dir = staging / "coverage"
        previews_dir = staging / "previews"
        work_dir = staging / ".work"
        for directory in (
            masters_dir,
            calibrated_dir,
            registered_dir,
            normalization_dir,
            coverage_dir,
            previews_dir,
            work_dir,
        ):
            directory.mkdir()

        if trusted_bias is not None:
            master_bias = Path(trusted_bias.path)
            stage_statistics["masterBias"] = {
                "mode": "REUSED_E2E_GENERATED_MASTER",
                "sha256": trusted_bias.sha256,
                "sizeBytes": trusted_bias.size_bytes,
                "calibrationApplied": False,
                "doubleBiasSubtraction": False,
                "numericDomain": reference_bias.numeric_domain,
                "normalizedUnitScale": reference_bias.normalized_unit_scale,
            }
        elif biases:
            master_bias = masters_dir / "master_bias.fits"
            bias_integration = integrate_expressions(
                (
                    FrameExpression(
                        str(path),
                        scale=_numeric_application_scale(
                            reference_bias,
                            bias_info[path],
                            target_label="MasterBias reference",
                            additive_label="raw Bias",
                        ),
                    )
                    for path in biases
                ),
                master_bias,
                metadata={
                    "IMAGETYP": "Master Bias",
                    "OAFSTATE": OUTPUT_STATE,
                    "OAFBIAS": "MASTER",
                    **_numeric_domain_metadata(reference_bias),
                },
                parameters=parameters.integration,
                native_threads=None,
                durable=parameters.durable_intermediates,
            )
            artifacts.append(
                _artifact_record(
                    staging,
                    master_bias,
                    "MASTER_BIAS",
                    statistics=bias_integration.statistics,
                    sha256=bias_integration.output_sha256,
                )
            )
            stage_statistics["masterBias"] = {
                "mode": "BUILT_FROM_RAW",
                "numericDomain": reference_bias.numeric_domain,
                "normalizedUnitScale": reference_bias.normalized_unit_scale,
                **_integration_record(bias_integration, staging),
            }
        elif master_biases:
            master_bias = master_biases[0]
            _, master_bias_sha256, _ = _source_identity(
                master_bias, source_aliases, source_identity_cache
            )
            stage_statistics["masterBias"] = {
                "mode": "REUSED_SUPPLIED_MASTER",
                "path": str(display_path(master_bias)),
                "sha256": master_bias_sha256,
                "calibrationApplied": False,
                "numericDomain": reference_bias.numeric_domain,
                "normalizedUnitScale": reference_bias.normalized_unit_scale,
            }

        else:
            master_bias = None
            stage_statistics["masterBias"] = {"mode": "NOT_REQUIRED_DARK_INCLUDES_BIAS"}

        master_darks: dict[float, Path] = {}
        master_dark_domain_info: dict[float, FrameInfo] = {}
        for exposure, paths in sorted(dark_groups.items()):
            dark_reference = dark_info[paths[0]]
            for path in paths:
                _assert_compatible(reference_bias, dark_info[path], workflow=parameters.calibration_workflow)
                _assert_compatible(
                    dark_reference,
                    dark_info[path],
                    compare_exposure=True,
                    compare_temperature=True,
                    temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
                    workflow=parameters.calibration_workflow,
                )
            trusted_dark = trusted_darks_by_exposure.get(exposure)
            if trusted_dark is not None:
                destination = Path(trusted_dark.path)
                master_darks[exposure] = destination
                master_dark_domain_info[exposure] = dark_reference
                stage_statistics[f"masterDark:{exposure:.9g}"] = {
                    "mode": "REUSED_E2E_GENERATED_MASTER",
                    "sha256": trusted_dark.sha256,
                    "sizeBytes": trusted_dark.size_bytes,
                    "calibrationApplied": False,
                    "doubleBiasSubtraction": False,
                    "biasIncluded": trusted_dark.bias_included,
                    "numericDomain": dark_reference.numeric_domain,
                    "normalizedUnitScale": dark_reference.normalized_unit_scale,
                    "applicationScaleToRawDarkReference": 1.0,
                }
            else:
                destination = masters_dir / f"master_dark_{_exposure_token(exposure)}s.fits"
                integration = integrate_expressions(
                    (
                        FrameExpression(
                            str(path),
                            scale=_numeric_application_scale(
                                dark_reference,
                                dark_info[path],
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
                        "OAFSTATE": OUTPUT_STATE,
                        "OAFBIAS": "INCLUDED",
                        **_numeric_domain_metadata(dark_reference),
                    },
                    parameters=parameters.integration,
                    durable=parameters.durable_intermediates,
                )
                master_darks[exposure] = destination
                master_dark_domain_info[exposure] = dark_reference
                artifacts.append(
                    _artifact_record(
                        staging,
                        destination,
                        "MASTER_DARK",
                        statistics=integration.statistics,
                        sha256=integration.output_sha256,
                        details={"exposureSeconds": exposure, "biasIncluded": True},
                    )
                )
                stage_statistics[f"masterDark:{exposure:.9g}"] = _integration_record(
                    integration, staging
                )
                stage_statistics[f"masterDark:{exposure:.9g}"]["mode"] = "BUILT_FROM_RAW"
                stage_statistics[f"masterDark:{exposure:.9g}"].update(
                    {
                        "numericDomain": dark_reference.numeric_domain,
                        "normalizedUnitScale": dark_reference.normalized_unit_scale,
                        "applicationScaleToRawDarkReference": 1.0,
                    }
                )
        for exposure, supplied in sorted(supplied_darks.items()):
            master_darks[exposure] = supplied
            master_dark_domain_info[exposure] = master_dark_info[supplied]
            _, supplied_sha256, _ = _source_identity(
                supplied, source_aliases, source_identity_cache
            )
            stage_statistics[f"masterDark:{exposure:.9g}"] = {
                "mode": "REUSED_SUPPLIED_MASTER",
                "path": str(display_path(supplied)),
                "sha256": supplied_sha256,
                "calibrationApplied": False,
                "biasIncluded": supplied_dark_bias_included[supplied],
                "numericDomain": master_dark_info[supplied].numeric_domain,
                "normalizedUnitScale": master_dark_info[
                    supplied
                ].normalized_unit_scale,
            }

        master_flats: dict[str, Path] = {}
        flat_application_scales: dict[str, float] = {}
        for filter_name, paths in sorted(flat_groups.items()):
            trusted_flat = trusted_flats_by_filter.get(filter_name)
            if trusted_flat is not None:
                destination = Path(trusted_flat.path)
                master_flats[filter_name] = destination
                flat_application_scales[filter_name] = 1.0
                stage_statistics[f"masterFlat:{filter_name}"] = {
                    "mode": "REUSED_E2E_GENERATED_MASTER",
                    "sha256": trusted_flat.sha256,
                    "sizeBytes": trusted_flat.size_bytes,
                    "calibrationApplied": False,
                    "doubleBiasSubtraction": False,
                    "applicationScale": 1.0,
                    "applicationNormalization": 1.0,
                }
                continue
            expressions: list[FrameExpression] = []
            normalizations: list[float] = []
            calibration_sources: list[dict[str, Any]] = []
            for path in paths:
                info = flat_info[path]
                _assert_compatible(
                    reference_bias, info, compare_filter=False, compare_exposure=False,
                    workflow=parameters.calibration_workflow,
                )
                if info.filter_name != filter_name:
                    raise CalibrationError("FLAT_GROUP_INVALID", "internal filter grouping error")
                flat_dark_match = _find_dark(info.exposure_seconds, master_darks)
                if flat_dark_match is not None:
                    dark_exposure, flat_subtract = flat_dark_match
                    dark_reference = (
                        dark_info[dark_groups[dark_exposure][0]]
                        if dark_exposure in dark_groups
                        else master_dark_info[supplied_darks[dark_exposure]]
                    )
                    _assert_compatible(
                        info,
                        dark_reference,
                        compare_exposure=True,
                        compare_temperature=True,
                        temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
                        workflow=parameters.calibration_workflow,
                    )
                    dark_bias_included = (
                        bool(trusted_darks_by_exposure[dark_exposure].bias_included)
                        if dark_exposure in trusted_darks_by_exposure
                        else (
                            True
                            if dark_exposure in dark_groups
                            else supplied_dark_bias_included[flat_subtract]
                        )
                    )
                    calibration_mode = (
                        "MATCHED_BIAS_INCLUDED_DARK"
                        if dark_bias_included
                        else "MATCHED_BIAS_SUBTRACTED_DARK_PLUS_MASTER_BIAS"
                    )
                    flat_subtract_info = master_dark_domain_info[dark_exposure]
                else:
                    flat_subtract = master_bias
                    dark_bias_included = True
                    calibration_mode = "BIAS"
                    flat_subtract_info = reference_bias
                flat_subtract_scale = _numeric_application_scale(
                    info,
                    flat_subtract_info,
                    target_label="raw Flat",
                    additive_label=(
                        "MasterDark" if flat_dark_match is not None else "MasterBias"
                    ),
                )
                flat_bias_scale = _numeric_application_scale(
                    info,
                    reference_bias,
                    target_label="raw Flat",
                    additive_label="MasterBias",
                )
                unnormalized = FrameExpression(
                    source_path=str(path),
                    subtract_path=str(flat_subtract),
                    subtract_scale=flat_subtract_scale,
                    subtract_paths=(str(master_bias),) if not dark_bias_included else (),
                    subtract_scales=(flat_bias_scale,) if not dark_bias_included else (),
                )
                location = robust_location(
                    unnormalized,
                    max_samples=parameters.integration.max_statistics_samples,
                    division_floor=parameters.integration.division_floor,
                    max_memory_bytes=parameters.integration.max_memory_bytes,
                )
                if location <= parameters.integration.division_floor:
                    raise CalibrationError(
                        "FLAT_SIGNAL_INVALID",
                        "Bias-subtracted Flat has non-positive robust signal",
                        path=str(path),
                    )
                normalizations.append(location)
                expressions.append(
                    FrameExpression(
                        source_path=str(path),
                        subtract_path=str(flat_subtract),
                        subtract_scale=flat_subtract_scale,
                        subtract_paths=(str(master_bias),) if not dark_bias_included else (),
                        subtract_scales=(flat_bias_scale,) if not dark_bias_included else (),
                        scale=1.0 / location,
                    )
                )
                calibration_sources.append(
                    {
                        "source": str(display_path(path)),
                        "mode": calibration_mode,
                        "subtracted": _path_receipt_reference(
                            staging,
                            flat_subtract,
                            source_aliases,
                            trusted_generated["byPath"]
                            if trusted_generated is not None
                            else None,
                            source_identity_cache,
                        ),
                        "targetNumericDomain": info.numeric_domain,
                        "additiveNumericDomain": flat_subtract_info.numeric_domain,
                        "applicationScale": flat_subtract_scale,
                        "applicationScaleSource": "normalized-unit-domain-ratio",
                    }
                )
            token = _safe_token(filter_name)
            destination = masters_dir / f"master_flat_{token}.fits"
            integration = integrate_expressions(
                expressions,
                destination,
                metadata={
                    "IMAGETYP": "Master Flat",
                    "FILTER": filter_name,
                    "OAFSTATE": OUTPUT_STATE,
                    "OAFBIAS": "SUBTRACTED",
                    "OAFNORM": "ROBUST_MEDIAN",
                    "OAFNDOM": "DIMENSIONLESS_RESPONSE",
                    "OAFNSCL": 1.0,
                },
                parameters=parameters.integration,
                durable=parameters.durable_intermediates,
            )
            master_flats[filter_name] = destination
            flat_application_scales[filter_name] = 1.0
            artifacts.append(
                _artifact_record(
                    staging,
                    destination,
                    "MASTER_FLAT",
                    statistics=integration.statistics,
                    sha256=integration.output_sha256,
                    details={
                        "filter": filter_name,
                        "normalizations": normalizations,
                        "calibrationSources": calibration_sources,
                    },
                )
            )
            stage_statistics[f"masterFlat:{filter_name}"] = _integration_record(
                integration, staging
            )
            stage_statistics[f"masterFlat:{filter_name}"]["mode"] = "BUILT_FROM_RAW"
        for filter_name, supplied in sorted(supplied_flats.items()):
            location = robust_location(
                FrameExpression(str(supplied)),
                max_samples=parameters.integration.max_statistics_samples,
                division_floor=parameters.integration.division_floor,
                max_memory_bytes=parameters.integration.max_memory_bytes,
            )
            if not math.isfinite(location) or location <= parameters.integration.division_floor:
                raise CalibrationError(
                    "FLAT_SIGNAL_INVALID",
                    "supplied MasterFlat has no positive finite robust signal",
                    path=str(supplied),
                )
            master_flats[filter_name] = supplied
            # Division uses a response normalized to unity without altering the
            # supplied master: (Light - calibration) / MasterFlat * median.
            flat_application_scales[filter_name] = location
            _, supplied_sha256, _ = _source_identity(
                supplied, source_aliases, source_identity_cache
            )
            stage_statistics[f"masterFlat:{filter_name}"] = {
                "mode": "REUSED_SUPPLIED_MASTER",
                "path": str(display_path(supplied)),
                "sha256": supplied_sha256,
                "calibrationApplied": False,
                "applicationNormalization": location,
            }

        hardware_profile = detect_hardware()
        execution_tuning = select_execution_tuning(hardware_profile)
        registered: dict[Path, Path] = {}
        registration_records: dict[str, Any] = {}
        light_jobs: list[_LightJob] = []
        calibrated_details: list[dict[str, Any]] = []
        for index, path in enumerate(lights, start=1):
            info = light_info[path]
            filter_name = info.filter_name
            flat_path = master_flats[filter_name]
            flat_reference = (
                flat_info[flat_groups[filter_name][0]]
                if filter_name in flat_groups
                else master_flat_info[supplied_flats[filter_name]]
            )
            _assert_compatible(info, flat_reference, compare_filter=True, workflow=parameters.calibration_workflow)
            dark_match = _find_dark(info.exposure_seconds, master_darks)
            if dark_match is not None:
                dark_exposure, subtract_path = dark_match
                dark_reference = (
                    dark_info[dark_groups[dark_exposure][0]]
                    if dark_exposure in dark_groups
                    else master_dark_info[supplied_darks[dark_exposure]]
                )
                _assert_compatible(
                    info,
                    dark_reference,
                    compare_exposure=True,
                    compare_temperature=True,
                    temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
                    workflow=parameters.calibration_workflow,
                )
                dark_bias_included = (
                    bool(trusted_darks_by_exposure[dark_exposure].bias_included)
                    if dark_exposure in trusted_darks_by_exposure
                    else (
                        True
                        if dark_exposure in dark_groups
                        else supplied_dark_bias_included[subtract_path]
                    )
                )
                bias_mode = (
                    "INCLUDED_IN_MASTER_DARK"
                    if dark_bias_included
                    else "MASTER_BIAS_AND_BIAS_SUBTRACTED_DARK"
                )
                subtract_info = master_dark_domain_info[dark_exposure]
            else:
                subtract_path = master_bias
                dark_bias_included = True
                bias_mode = "MASTER_BIAS_SUBTRACTED"
                subtract_info = reference_bias
            subtract_scale = _numeric_application_scale(
                info,
                subtract_info,
                target_label="raw Light",
                additive_label=(
                    "MasterDark" if dark_match is not None else "MasterBias"
                ),
            )
            bias_scale = _numeric_application_scale(
                info,
                reference_bias,
                target_label="raw Light",
                additive_label="MasterBias",
            )
            light_output_domain = light_domain_references[filter_name]
            light_domain_scale = _numeric_application_scale(
                light_output_domain,
                info,
                target_label="filter integration domain",
                additive_label="raw Light",
            )
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_") or "light"
            calibrated_path = (
                calibrated_dir / f"{index:05d}_{stem}.fits"
                if parameters.materialize_calibrated_lights
                else None
            )
            registered_path = registered_dir / f"{index:05d}_{stem}.fits"
            expression = FrameExpression(
                source_path=str(path),
                subtract_path=str(subtract_path),
                subtract_scale=subtract_scale,
                subtract_paths=(str(master_bias),) if not dark_bias_included else (),
                subtract_scales=(bias_scale,) if not dark_bias_included else (),
                divide_path=str(flat_path),
                scale=(
                    flat_application_scales[filter_name]
                    * reference_exposures[filter_name]
                    / float(info.exposure_seconds)
                    * light_domain_scale
                ),
            )
            calibrated_metadata = {
                "IMAGETYP": "Calibrated Light",
                "FILTER": filter_name,
                "OBJECT": info.target,
                "EXPTIME": reference_exposures[filter_name],
                "OAFSRCEX": info.exposure_seconds,
                "OAFEXPSC": reference_exposures[filter_name]
                / float(info.exposure_seconds),
                "OAFSTATE": OUTPUT_STATE,
                "OAFBIAS": bias_mode,
                **_numeric_domain_metadata(light_output_domain),
            }
            calibrated_details.append(
                {
                    "source": str(display_path(path)),
                    "filter": filter_name,
                    "subtractedMaster": _path_receipt_reference(
                        staging,
                        subtract_path,
                        source_aliases,
                        trusted_generated["byPath"]
                        if trusted_generated is not None
                        else None,
                        source_identity_cache,
                    ),
                    "biasMode": bias_mode,
                    "sourceNumericDomain": info.numeric_domain,
                    "additiveNumericDomain": subtract_info.numeric_domain,
                    "additiveApplicationScale": subtract_scale,
                    "additiveApplicationScaleSource": "normalized-unit-domain-ratio",
                    "biasApplicationScale": (
                        bias_scale if not dark_bias_included else None
                    ),
                    "outputNumericDomain": light_output_domain.numeric_domain,
                    "sourceToOutputDomainScale": light_domain_scale,
                    "dividedMasterFlat": _path_receipt_reference(
                        staging,
                        flat_path,
                        source_aliases,
                        trusted_generated["byPath"]
                        if trusted_generated is not None
                        else None,
                        source_identity_cache,
                    ),
                    "flatApplicationNormalization": flat_application_scales[
                        filter_name
                    ],
                    "exposureNormalization": {
                        "sourceSeconds": info.exposure_seconds,
                        "referenceSeconds": reference_exposures[filter_name],
                        "scale": reference_exposures[filter_name]
                        / float(info.exposure_seconds),
                    },
                }
            )
            light_jobs.append(
                _LightJob(
                    source_path=path,
                    expression=expression,
                    calibrated_path=calibrated_path,
                    calibrated_metadata=calibrated_metadata,
                    destination=registered_path,
                    transform=resolved_transforms[path],
                    info=replace(
                        info,
                        exposure_seconds=reference_exposures[filter_name],
                        numeric_domain=light_output_domain.numeric_domain,
                        normalized_unit_scale=light_output_domain.normalized_unit_scale,
                    ),
                    source_exposure_seconds=info.exposure_seconds,
                )
            )

        # Calibrate in memory and warp within one shared registration budget.
        # Source-identity caches and receipt construction stay on this thread.
        master_cache = _MasterCache()
        registration_started = time.perf_counter()
        light_results, fused_execution = _calibrate_and_register_frames(
            light_jobs,
            master_cache=master_cache,
            max_memory_bytes=parameters.registration_memory_bytes,
            resampler=parameters.registration_resampler,
            cpu_workers=execution_tuning.cpu_workers,
            division_floor=parameters.integration.division_floor,
            durable=parameters.durable_intermediates,
        )
        registration_wall_seconds = time.perf_counter() - registration_started
        del master_cache
        for path, job, details, result in zip(
            lights, light_jobs, calibrated_details, light_results, strict=True,
        ):
            transform = job.transform
            resampling = _registration_provenance(
                transform, job.info.shape, parameters.registration_resampler
            )
            registered[path] = job.destination
            if job.calibrated_path is not None:
                artifacts.append(
                    _artifact_record(
                        staging,
                        job.calibrated_path,
                        "CALIBRATED_LIGHT",
                        statistics=result.calibrated_statistics,
                        details=details,
                        sha256=result.calibrated_sha256,
                    )
                )
            artifacts.append(
                _artifact_record(
                    staging,
                    job.destination,
                    "REGISTERED_LIGHT",
                    statistics=result.registered_statistics,
                    details={
                        "source": str(display_path(path)),
                        "transformInputToOutput": transform.serializable(),
                        **resampling,
                        "warpBackend": result.execution.get("warpBackend"),
                        "warpKernel": result.execution.get("warpKernel"),
                    },
                    sha256=result.registered_sha256,
                )
            )
            registration_records[str(display_path(path))] = {
                "transformInputToOutput": transform.serializable(),
                "identity": transform.is_identity,
                **resampling,
                "qualityWeight": resolved_quality_weights[path],
                "calibration": {
                    "materialized": job.calibrated_path is not None,
                    "statistics": result.calibrated_statistics.serializable(),
                    **details,
                },
                "execution": dict(result.execution),
            }
        if not parameters.materialize_calibrated_lights:
            try:
                calibrated_dir.rmdir()
            except OSError:
                # Removing this unused staging directory is best effort.
                # Keep it for diagnostics if cleanup fails; artifact and
                # final-publication validation still run independently.
                pass

        registration_execution = {
            "executor": fused_execution["executor"],
            "executionModel": fused_execution["executionModel"],
            "configuredCpuWorkers": execution_tuning.cpu_workers,
            "cpuWorkersUsed": fused_execution["cpuWorkersUsed"],
            "nativeThreadsPerWorker": fused_execution["nativeThreadsPerWorker"],
            "tailNativeThreads": fused_execution["tailNativeThreads"],
            "tailLights": fused_execution["tailLights"],
            "frameCount": len(light_jobs),
            "totalMemoryBudgetBytes": parameters.registration_memory_bytes,
            "perWorkerMemoryBudgetBytes": fused_execution["perWorkerMemoryBudgetBytes"],
            "warpBackends": fused_execution["warpBackends"],
            "calibratedLightsMaterialized": parameters.materialize_calibrated_lights,
            "masterCacheBytes": fused_execution["masterCacheBytes"],
            "wallSeconds": registration_wall_seconds,
        }
        if parameters.ordinary_integration_backend != "portable-cpu":
            try:
                metal_executor = NativeMetalExecutor(
                    library_path=parameters.native_library_path,
                    metal_source_path=parameters.metal_source_path,
                )
            except MetalIntegrationError as error:
                metal_unavailable_reason = str(error)

        master_lights: list[Path] = []
        previews: list[Path] = []
        integration_groups: dict[str, Any] = {}
        for filter_name, paths in sorted(light_groups.items()):
            exposures = {
                info.exposure_seconds for path, info in light_info.items() if path in paths
            }
            reference_exposure = reference_exposures[filter_name]
            total_exposure = sum(
                float(light_info[path].exposure_seconds) for path in paths
            )
            registered_paths = [registered[path] for path in paths]
            reference_index = max(
                range(len(paths)),
                key=lambda item: resolved_quality_weights[paths[item]],
            )
            integration_expressions = [
                FrameExpression(str(path)) for path in registered_paths
            ]
            local_normalization_record: dict[str, Any] = {
                "status": "DISABLED",
                "parameters": parameters.local_normalization.serializable(),
            }
            global_normalization_record: dict[str, Any] = {
                "status": "DISABLED",
                "parameters": parameters.global_normalization.serializable(),
            }
            normalization_method = "NONE"
            if parameters.local_normalization.enabled:
                normalization_result = normalize_registered_group(
                    [str(path) for path in registered_paths],
                    normalization_dir / _safe_token(filter_name),
                    reference_index=reference_index,
                    parameters=parameters.local_normalization,
                )
                registered_paths = [Path(path) for path in normalization_result.normalized_paths]
                integration_expressions = [
                    FrameExpression(str(path)) for path in registered_paths
                ]
                local_normalization_record = {
                    "status": "APPLIED",
                    "referenceInput": str(display_path(paths[reference_index])),
                    "receipt": str(
                        Path(normalization_result.receipt_path).relative_to(staging)
                    ),
                    "receiptSha256": _hash_file(Path(normalization_result.receipt_path)),
                    "evidence": normalization_result.receipt,
                }
                for frame in normalization_result.receipt["frames"]:
                    normalized_path = (
                        normalization_dir
                        / _safe_token(filter_name)
                        / frame["normalized"]["path"]
                    )
                    artifacts.append(
                        _artifact_record(
                            staging,
                            normalized_path,
                            "LOCALLY_NORMALIZED_REGISTERED_LIGHT",
                            details={
                                "filter": filter_name,
                                "reference": frame["reference"],
                                "modelEvidence": frame["evidence"],
                            },
                        )
                    )
                global_normalization_record = {
                    "status": "SUPERSEDED_BY_LOCAL_NORMALIZATION",
                    "parameters": parameters.global_normalization.serializable(),
                    "reason": (
                        "LocalNormalization already owns per-frame scale and offset; "
                        "a second global normalization is forbidden"
                    ),
                }
                normalization_method = "LOCAL_GRID"
            elif parameters.global_normalization.enabled:
                group_hints: list[StellarScaleHint | None] = []
                expected_reference = paths[reference_index]
                registered_reference = registered_paths[reference_index]
                for source_path, registered_path in zip(
                    paths, registered_paths, strict=True
                ):
                    hint = resolved_stellar_scale_hints[source_path]
                    if hint is None:
                        group_hints.append(None)
                        continue
                    hinted_reference = Path(hint.reference_path).expanduser().resolve(
                        strict=True
                    )
                    if hinted_reference != expected_reference:
                        raise CalibrationError(
                            "STELLAR_SCALE_HINT_REFERENCE_MISMATCH",
                            "stellar scale reference differs from the integration-quality reference",
                            path=str(source_path),
                        )
                    group_hints.append(
                        replace(
                            hint,
                            source_path=str(registered_path),
                            reference_path=str(registered_reference),
                        )
                    )
                global_result = fit_registered_group_global_normalization(
                    [str(path) for path in registered_paths],
                    reference_index=reference_index,
                    parameters=parameters.global_normalization,
                    stellar_scale_hints=group_hints,
                    workers=execution_tuning.cpu_workers,
                )
                integration_expressions = [
                    FrameExpression(
                        str(path),
                        scale=coefficient.scale,
                        offset=coefficient.offset,
                        offset_grid=coefficient.offset_grid,
                        offset_grid_x=coefficient.offset_grid_x,
                        offset_grid_y=coefficient.offset_grid_y,
                    )
                    for path, coefficient in zip(
                        registered_paths,
                        global_result.coefficients,
                        strict=True,
                    )
                ]
                public_frames: list[dict[str, Any]] = []
                for index, frame in enumerate(global_result.receipt["frames"]):
                    public_frame = {**dict(frame), "source": str(display_path(paths[index]))}
                    frame_evidence = dict(public_frame["evidence"])
                    stellar = frame_evidence.get("stellarScale")
                    if isinstance(stellar, dict):
                        stellar = dict(stellar)
                        stellar["source"] = str(display_path(paths[index]))
                        stellar["reference"] = str(
                            display_path(paths[reference_index])
                        )
                        frame_evidence["stellarScale"] = stellar
                    public_frame["evidence"] = frame_evidence
                    public_frames.append(public_frame)
                public_evidence = {
                    **dict(global_result.receipt),
                    "frames": public_frames,
                }
                global_normalization_record = {
                    "status": "APPLIED",
                    "referenceInput": str(display_path(paths[reference_index])),
                    "evidence": public_evidence,
                }
                normalization_method = "GLOBAL_STELLAR"
            full_master = work_dir / f"integrated_{_safe_token(filter_name)}.fits"
            token = _safe_token(filter_name)
            full_maps = IntegrationMapPaths(
                accepted_count=work_dir / f"accepted_count_{token}.fits",
                coverage=work_dir / f"coverage_{token}.fits",
                rejection_count=work_dir / f"rejection_count_{token}.fits",
            )
            integration = integrate_registered_group(
                integration_expressions,
                full_master,
                metadata={
                    "IMAGETYP": "Master Light",
                    "FILTER": filter_name,
                    "OAFSTATE": OUTPUT_STATE,
                    "OAFWCS": "UNSOLVED",
                    "EXPTIME": reference_exposure,
                    "OAFINTTM": total_exposure,
                    "OAFNORM": normalization_method,
                    **_numeric_domain_metadata(
                        light_domain_references[filter_name]
                    ),
                },
                parameters=parameters.integration,
                requested_backend=parameters.ordinary_integration_backend,
                native_library_path=parameters.native_library_path,
                metal_source_path=parameters.metal_source_path,
                metal_executor=metal_executor,
                metal_unavailable_reason=metal_unavailable_reason,
                hardware=hardware_profile,
                tuning=execution_tuning,
                quality_weights=[resolved_quality_weights[path] for path in paths],
                map_paths=full_maps,
                durable=parameters.durable_intermediates,
            )
            fallback_reason = str(integration.execution.get("fallbackReason") or "")
            if (
                metal_executor is not None
                and integration.execution.get("selectedBackend") == "portable-cpu"
                and fallback_reason.startswith("Metal execution rejected:")
            ):
                metal_executor.close()
                metal_executor = None
                metal_unavailable_reason = fallback_reason
            if parameters.auto_crop:
                crop = _common_valid_crop(
                    integration.shape,
                    [resolved_transforms[path] for path in paths],
                    max_memory_bytes=parameters.registration_memory_bytes,
                    resampler=parameters.registration_resampler,
                )
            else:
                height, width = integration.shape
                crop = (0, 0, height, width)
            top, left, bottom, right = crop
            crop_fraction = ((bottom - top) * (right - left)) / (
                integration.shape[0] * integration.shape[1]
            )
            if crop_fraction < parameters.minimum_crop_fraction:
                raise CalibrationError(
                    "AUTOCROP_TOO_SMALL",
                    f"common crop retains only {crop_fraction:.3%} of the frame",
                )
            master_light = masters_dir / f"master_light_{token}.fits"
            master_stats, master_sha256 = _crop_fits(
                full_master,
                master_light,
                crop,
                {
                    "IMAGETYP": "Master Light",
                    "FILTER": filter_name,
                    "EXPTIME": reference_exposure,
                    "OAFINTTM": total_exposure,
                    "OAFSTATE": OUTPUT_STATE,
                    "OAFWCS": "UNSOLVED",
                    "OAFCROP": "AUTO" if parameters.auto_crop else "NONE",
                    "OAFNFRM": len(paths),
                    "OAFNORM": normalization_method,
                    **_numeric_domain_metadata(
                        light_domain_references[filter_name]
                    ),
                },
                max_memory_bytes=parameters.integration.max_memory_bytes,
                durable=parameters.durable_intermediates,
            )
            master_lights.append(master_light)
            artifacts.append(
                _artifact_record(
                    staging,
                    master_light,
                    "MASTER_LIGHT_LINEAR_UNSOLVED",
                    statistics=master_stats,
                    sha256=master_sha256,
                    details={
                        "filter": filter_name,
                        "crop": {
                            "top": top,
                            "left": left,
                            "bottomExclusive": bottom,
                            "rightExclusive": right,
                            "retainedFraction": crop_fraction,
                        },
                    },
                )
            )
            cropped_maps: dict[str, Path] = {}
            map_specs = (
                (
                    "acceptedSampleCount",
                    Path(integration.map_paths["acceptedSampleCount"]),
                    "INTEGRATION_ACCEPTED_COUNT",
                ),
                (
                    "coverageFraction",
                    Path(integration.map_paths["coverageFraction"]),
                    "INTEGRATION_COVERAGE",
                ),
                (
                    "rejectionCount",
                    Path(integration.map_paths["rejectionCount"]),
                    "INTEGRATION_REJECTION_COUNT",
                ),
            )
            map_statistics: dict[str, Any] = {}
            for map_name, full_map, artifact_kind in map_specs:
                destination = coverage_dir / f"{token}_{map_name}.fits"
                statistics, map_sha256 = _crop_fits(
                    full_map,
                    destination,
                    crop,
                    {
                        "IMAGETYP": artifact_kind.replace("_", " ").title(),
                        "FILTER": filter_name,
                        "OAFSTATE": OUTPUT_STATE,
                        "OAFMAP": map_name.upper(),
                        "OAFNFRM": len(paths),
                    },
                    max_memory_bytes=parameters.integration.max_memory_bytes,
                    durable=parameters.durable_intermediates,
                )
                cropped_maps[map_name] = destination
                map_statistics[map_name] = statistics.serializable()
                artifacts.append(
                    _artifact_record(
                        staging,
                        destination,
                        artifact_kind,
                        statistics=statistics,
                        sha256=map_sha256,
                        details={
                            "filter": filter_name,
                            "sourceIntegration": str(full_master.relative_to(staging)),
                            "usesRegistrationQualityWeights": True,
                        },
                    )
                )
            preview_path = previews_dir / f"master_light_{token}.png"
            preview_result = render_auto_stretch_preview(
                master_light,
                preview_path,
                max_long_edge=parameters.preview_max_long_edge,
                max_memory_bytes=parameters.registration_memory_bytes,
            )
            previews.append(preview_path)
            artifacts.append(
                _artifact_record(
                    staging,
                    preview_path,
                    "AUTO_STRETCH_PREVIEW",
                    details=preview_result.serializable(),
                )
            )
            artifacts[-1]["details"]["outputPath"] = str(
                preview_path.relative_to(staging)
            )
            integration_record = _integration_record(integration, staging)
            integration_record["maps"] = {
                name: str(path.relative_to(staging))
                for name, path in cropped_maps.items()
            }
            integration_groups[filter_name] = {
                "integration": integration_record,
                "localNormalization": local_normalization_record,
                "globalNormalization": global_normalization_record,
                "exposureNormalization": {
                    "sourceExposureSeconds": sorted(float(value) for value in exposures),
                    "referenceSeconds": reference_exposure,
                    "totalIntegrationSeconds": total_exposure,
                    "method": "LINEAR_REFERENCE_EXPOSURE",
                },
                "crop": [top, left, bottom, right],
                "masterStatistics": master_stats.serializable(),
                "mapStatistics": map_statistics,
            }

        shutil.rmtree(work_dir)
        _verify_source_identities(source_identities)
        if _trusted_generated_calibration is not None:
            # Recheck source stat identities and rehash the generated masters
            # and upstream receipt after all consumers finish.  The enclosing
            # E2E publication gate deliberately performs the second full hash
            # of every original source; intermediate receipt lookups reuse this
            # run's path/stat-bound digest instead of rereading large inputs.
            _validate_trusted_generated_calibration_set(
                _trusted_generated_calibration,
                source_groups=source_groups,
                source_aliases=source_aliases,
                identity_cache=source_identity_cache,
            )
        raw_source_provenance = [
            dict(item)
            for item in source_records
            if item["role"] in {"BIAS", "DARK", "FLAT"}
        ]
        trusted_reuse_receipt = (
            {
                "status": "REUSED_E2E_GENERATED_MASTER",
                "sourceContentManifestSha256": _content_lineage_sha256(
                    raw_source_provenance
                ),
                "privateExecutionManifestBound": True,
                "upstreamRegistrationCalibrationReceiptSha256": (
                    _trusted_generated_calibration.upstream_receipt_sha256
                ),
                "calibrationApplied": False,
                "doubleBiasSubtraction": False,
                "originalRawSourceProvenance": raw_source_provenance,
                "masters": [
                    {
                        "role": item.role,
                        "sha256": item.sha256,
                        "sizeBytes": item.size_bytes,
                        **(
                            {"biasIncluded": item.bias_included}
                            if item.role == "MASTER_DARK"
                            else {}
                        ),
                        **(
                            {"applicationScale": item.application_scale}
                            if item.role == "MASTER_FLAT"
                            else {}
                        ),
                    }
                    for item in (
                        _trusted_generated_calibration.master_bias,
                        *_trusted_generated_calibration.master_darks,
                        *_trusted_generated_calibration.master_flats,
                    )
                    if item is not None
                ],
            }
            if _trusted_generated_calibration is not None
            else {"status": "NOT_USED"}
        )
        receipt = {
            "schemaVersion": 1,
            "pipelineVersion": PIPELINE_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "state": OUTPUT_STATE,
            "parameters": parameters.serializable(),
            "inputs": source_records,
            "pixelNumericDomains": pixel_numeric_domains,
            "trustedGeneratedCalibration": trusted_reuse_receipt,
            "pixelInputStaging": {
                "xisfPolicy": parameters.xisf_decode.serializable(),
                "conversions": [dict(item) for item in _xisf_conversions],
                "privateStagingRemovedAfterRun": True,
            },
            "masterMetadataOverrides": [
                {
                    **item.serializable(),
                    "status": "APPLIED_CONTENT_BOUND_DECLARATION",
                }
                for item in parameters.master_metadata_overrides
            ],
            "outputs": artifacts,
            "registration": registration_records,
            "statistics": {
                "calibration": stage_statistics,
                "registration": registration_execution,
                "integrationGroups": integration_groups,
            },
            "astrometry": {
                "status": "UNSOLVED",
                "wcsValidated": False,
                "message": "No solver was invoked; every linear master remains UNSOLVED_WORKING.",
            },
            "drizzle": {
                "status": "NOT_RUN",
                "message": "This portable ordinary-integration pipeline does not run drizzle.",
            },
        }
        _write_receipt(staging / "receipt.json", receipt)
        _rename_directory_no_replace(staging, output)
        published = True
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return PipelineResult(
            output_directory=str(output),
            receipt_path=str(output / "receipt.json"),
            state=OUTPUT_STATE,
            master_light_paths=tuple(
                str(output / path.relative_to(staging)) for path in master_lights
            ),
            preview_paths=tuple(str(output / path.relative_to(staging)) for path in previews),
        )
    finally:
        if metal_executor is not None:
            metal_executor.close()
        if not published and staging.exists():
            shutil.rmtree(staging)


def _rekey_for_staged_lights(
    originals: tuple[Path, ...],
    staged_by_original: Mapping[Path, Path],
    transforms: Mapping[str, AffineTransform | Sequence[Sequence[float]]] | None,
    quality_weights: Mapping[str, float] | None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None,
) -> tuple[
    Mapping[str, AffineTransform | Sequence[Sequence[float]]] | None,
    Mapping[str, float] | None,
    Mapping[str, StellarScaleHint] | None,
]:
    resolved_transforms = _resolve_transforms(originals, transforms)
    resolved_weights = _resolve_quality_weights(originals, quality_weights)
    staged_hints: Mapping[str, StellarScaleHint] | None = None
    if stellar_scale_hints is not None:
        by_path = {
            Path(key).expanduser().resolve(strict=True): value
            for key, value in stellar_scale_hints.items()
        }
        staged_values: dict[str, StellarScaleHint] = {}
        for original in originals:
            hint = by_path.get(original)
            if hint is None:
                continue
            reference = Path(hint.reference_path).expanduser().resolve(strict=True)
            if reference not in staged_by_original:
                raise CalibrationError(
                    "STELLAR_SCALE_HINT_INPUT_MISMATCH",
                    "stellar scale reference is not an admitted Light",
                    path=str(reference),
                )
            staged_source = staged_by_original[original]
            staged_values[str(staged_source)] = replace(
                hint,
                source_path=str(staged_source),
                reference_path=str(staged_by_original[reference]),
            )
        staged_hints = staged_values
    return (
        {
            str(staged_by_original[path]): transform
            for path, transform in resolved_transforms.items()
        },
        {
            str(staged_by_original[path]): weight
            for path, weight in resolved_weights.items()
        },
        staged_hints,
    )


def run_portable_pipeline(
    *,
    bias_files: Iterable[str | os.PathLike[str]] = (),
    dark_files: Iterable[str | os.PathLike[str]] = (),
    flat_files: Iterable[str | os.PathLike[str]] = (),
    master_bias_file: str | os.PathLike[str] | None = None,
    master_dark_files: Iterable[str | os.PathLike[str]] = (),
    master_flat_files: Iterable[str | os.PathLike[str]] = (),
    light_files: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    transforms: Mapping[str, AffineTransform | Sequence[Sequence[float]]] | None = None,
    quality_weights: Mapping[str, float] | None = None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None = None,
    parameters: PipelineParameters | None = None,
) -> PipelineResult:
    """Run the portable pipeline with private, content-bound XISF staging."""

    parameters = parameters or PipelineParameters()
    parameters.validate()
    output = Path(output_directory).expanduser().resolve(strict=False)
    if output.exists() or os.path.lexists(output):
        raise CalibrationError("OUTPUT_EXISTS", "output directory must be new", path=str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    originals = {
        "BIAS": _canonical_inputs(bias_files, "Bias", required=False),
        "DARK": _canonical_inputs(dark_files, "Dark", required=False),
        "FLAT": _canonical_inputs(flat_files, "Flat", required=False),
        "MASTER_BIAS": _canonical_inputs(
            () if master_bias_file is None else (master_bias_file,),
            "MasterBias",
            required=False,
        ),
        "MASTER_DARK": _canonical_inputs(master_dark_files, "MasterDark", required=False),
        "MASTER_FLAT": _canonical_inputs(master_flat_files, "MasterFlat", required=False),
        "LIGHT": _canonical_inputs(light_files, "Light", required=True),
    }
    all_originals = [path for paths in originals.values() for path in paths]
    if len({os.path.normcase(str(path)) for path in all_originals}) != len(all_originals):
        raise CalibrationError("INPUT_ROLE_OVERLAP", "one source appears in multiple roles")
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.xisf-pixels-",
        dir=output.parent,
    ) as private_name:
        private = Path(private_name)
        staged_by_original: dict[Path, Path] = {}
        aliases: dict[str, Path] = {}
        conversions: list[dict[str, Any]] = []
        sequence = 0
        for role, paths in originals.items():
            for original in paths:
                sequence += 1
                if original.suffix.casefold() == ".xisf":
                    staged = private / f"{sequence:06d}_{role.casefold()}.fits"
                    receipt = convert_xisf_to_fits(
                        original,
                        staged,
                        policy=parameters.xisf_decode,
                    )
                    conversions.append({"role": role, **receipt.serializable()})
                else:
                    staged = original
                staged_by_original[original] = staged
                aliases[str(staged)] = original
        staged = {
            role: tuple(staged_by_original[path] for path in paths)
            for role, paths in originals.items()
        }
        staged_transforms, staged_weights, staged_stellar_scale_hints = (
            _rekey_for_staged_lights(
                originals["LIGHT"],
                staged_by_original,
                transforms,
                quality_weights,
                stellar_scale_hints,
            )
        )
        return _run_portable_pipeline_fits(
            bias_files=staged["BIAS"],
            dark_files=staged["DARK"],
            flat_files=staged["FLAT"],
            master_bias_file=(staged["MASTER_BIAS"][0] if staged["MASTER_BIAS"] else None),
            master_dark_files=staged["MASTER_DARK"],
            master_flat_files=staged["MASTER_FLAT"],
            light_files=staged["LIGHT"],
            output_directory=output,
            transforms=staged_transforms,
            quality_weights=staged_weights,
            stellar_scale_hints=staged_stellar_scale_hints,
            parameters=parameters,
            _source_aliases=aliases,
            _xisf_conversions=conversions,
        )


__all__ = [
    "AffineTransform",
    "MasterMetadataOverride",
    "OUTPUT_STATE",
    "PIPELINE_VERSION",
    "PipelineParameters",
    "PipelineResult",
    "run_portable_pipeline",
]
