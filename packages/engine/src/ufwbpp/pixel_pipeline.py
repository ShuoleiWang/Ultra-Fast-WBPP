"""Portable CPU raw-to-linear-master pipeline.

The pipeline is deliberately not an astrometric solver or drizzle engine. Its
published masters are marked ``UNSOLVED_WORKING`` and contain no fabricated WCS.
All pixel operations use vertical slices and all input FITS files stay read-only.
"""

from __future__ import annotations
from .calibration_inputs import (
    MasterMetadataOverride,
    RawFrameMetadataOverride,
    _InternalSourceIdentity,
    _TrustedCalibrationSource,
    _TrustedGeneratedMaster,
    _TrustedGeneratedCalibrationSet,
    _apply_master_metadata_overrides,
    _apply_raw_frame_metadata_overrides,
    _trust_private_xisf_numeric_domains,
    _master_dark_bias_semantics,
    _required_mismatch,
    _assert_compatible,
    _numeric_application_scale,
    _numeric_domain_metadata,
    _hash_file,
    _stat_identity,
    _trusted_source_manifest_sha256,
    _capture_trusted_generated_master,
    _capture_trusted_generated_calibration_set,
    _validate_trusted_generated_master,
    _validate_trusted_generated_calibration_set,
    _canonical_original_path,
    _source_identity,
    _find_dark,
    _SourceIdentityCache,
)


from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence, Callable

import numpy as np
from numpy.typing import NDArray

from lightframeqc.cfa import (
    CFA_PATTERNS,
    CHANNEL_NAMES,
    bilinear_debayer,
    channel_medians,
    normalize_pattern as normalize_cfa_pattern,
)
from .calibration_policy import (STRICT, MONO_STANDARD, WORKFLOWS, apply_mono_workflow, metadata_changes, same_metadata, cfa_for_workflow, resolve_dark_bias, workflow_receipt, can_omit_bias, conflicting_profile_fields, acquisition_receipt)
from .calibration import (
    cfa_metadata,
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
from .proper_coaddition import (
    PROPER_COADD_ALGORITHM_ID,
    ProperCoadditionParameters,
    proper_coadd_group,
)
from .preview import render_auto_stretch_preview
from .hardware import detect_hardware
from .metal_integration import (
    MetalIntegrationError,
    NativeMetalExecutor,
    SUPPORTED_BACKENDS,
    integrate_registered_group,
)
from . import platform as platform_services
from .platform import NoReplaceError, remove_file, remove_tree
from .path_budget import STAGING_SUFFIX, check_output_path_budget, light_stem, name_token
from .native_kernels import describe_native_kernels
from .performance_profile import select_execution_tuning
from .global_normalization import (
    GlobalNormalizationParameters,
    StellarScaleHint,
    fit_registered_group_global_normalization,
)
from .xisf_pixels import XisfDecodePolicy, convert_xisf_to_fits
from .drizzle_native import DrizzleFrame, DrizzleGroupInputs


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
class PixelTransform:
    """Input-pixel to output-pixel homogeneous transform.

    The last row is ``(0, 0, 1)`` for an affine map.  A projective map (any
    other finite last row) is accepted as well: registration against frames
    of another night or hour angle needs the two perspective terms that a
    tilt or differential refraction adds, which an affine fit leaves as a
    field-dependent misregistration of several tenths of a pixel.
    """

    matrix: tuple[tuple[float, float, float], ...]

    @classmethod
    def identity(cls) -> PixelTransform:
        return cls(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    @classmethod
    def from_value(
        cls, value: PixelTransform | Sequence[Sequence[float]]
    ) -> PixelTransform:
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
        if matrix[2, 2] == 0.0:
            raise CalibrationError(
                "TRANSFORM_INVALID", "homogeneous transform must be normalized (m22 != 0)"
            )
        determinant = float(np.linalg.det(matrix))
        linear_determinant = float(np.linalg.det(matrix[:2, :2]))
        if (
            not math.isfinite(determinant)
            or abs(determinant) < 1e-12
            or not math.isfinite(linear_determinant)
            or abs(linear_determinant) < 1e-12
        ):
            raise CalibrationError("TRANSFORM_SINGULAR", "transform is singular")
        return matrix

    @property
    def is_projective(self) -> bool:
        matrix = self.validated_matrix()
        return not bool(np.array_equal(matrix[2], (0.0, 0.0, 1.0)))

    @property
    def is_identity(self) -> bool:
        return bool(
            np.allclose(
                self.validated_matrix(), np.eye(3), rtol=0.0, atol=1e-12
            )
        )

    def serializable(self) -> list[list[float]]:
        return [list(row) for row in self.matrix]


# Compatibility import; serialized matrices and numerical evaluation are unchanged.
AffineTransform = PixelTransform


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
    global_normalization: GlobalNormalizationParameters = field(
        default_factory=GlobalNormalizationParameters
    )
    # Opt-in ZOGY proper coaddition: one ADDITIONAL linear product per group,
    # from the same registered, normalized frames and the same per-pixel
    # rejection decisions.  The ordinary master is unaffected either way.
    proper_coaddition: ProperCoadditionParameters = field(
        default_factory=ProperCoadditionParameters
    )
    master_metadata_overrides: tuple[MasterMetadataOverride, ...] = ()
    raw_frame_metadata_overrides: tuple[RawFrameMetadataOverride, ...] = ()
    # Calibrated Lights are computed in memory and registered directly.  They
    # are written to disk only when a consumer (Drizzle, the public portable
    # pipeline) needs them; the ordinary E2E path keeps them transient.
    materialize_calibrated_lights: bool = True
    # Drizzle consumes the ordinary integration's per-frame products: with
    # this flag every group also keeps its calibrated paths, transforms,
    # normalization coefficients, weights and packed rejection masks in the
    # result (``PipelineResult.drizzle_groups``).  Requires materialized
    # calibrated Lights.
    capture_drizzle_inputs: bool = False
    # Cosmetic correction of hot pixels: pixels of the subtracted master dark
    # that lie more than this many robust sigmas above its median are
    # replaced in every calibrated Light by the median of their eight
    # neighbours before registration.  Unstable hot pixels leave residuals
    # after dark subtraction that pixel rejection cannot always remove when
    # several frames share one pointing.  ``None`` disables the correction.
    cosmetic_hot_pixel_sigma: float | None = 3.0
    # Published pipeline outputs are fsynced.  An enclosing run whose whole
    # pipeline directory is transient (the E2E work tree) turns this off and
    # relies on its own fsynced promotion of the final products.
    durable_intermediates: bool = True

    def validate(self) -> None:
        if self.calibration_workflow not in WORKFLOWS:
            raise ValueError("unsupported calibration_workflow")
        if self.cosmetic_hot_pixel_sigma is not None and (
            isinstance(self.cosmetic_hot_pixel_sigma, bool)
            or not math.isfinite(self.cosmetic_hot_pixel_sigma)
            or self.cosmetic_hot_pixel_sigma < 1.0
        ):
            raise ValueError("cosmetic_hot_pixel_sigma must be None or a finite value >= 1")
        if not isinstance(self.materialize_calibrated_lights, bool):
            raise ValueError("materialize_calibrated_lights must be a boolean")
        if not isinstance(self.capture_drizzle_inputs, bool):
            raise ValueError("capture_drizzle_inputs must be a boolean")
        if self.capture_drizzle_inputs and not self.materialize_calibrated_lights:
            raise ValueError("capture_drizzle_inputs requires materialize_calibrated_lights")
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
        self.global_normalization.validate()
        self.proper_coaddition.validate()
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
            "globalNormalization": self.global_normalization.serializable(),
            "masterMetadataOverrides": [
                item.serializable() for item in self.master_metadata_overrides
            ],
            "rawFrameMetadataOverrides": [
                item.serializable() for item in self.raw_frame_metadata_overrides
            ],
            "materializeCalibratedLights": self.materialize_calibrated_lights,
            "captureDrizzleInputs": self.capture_drizzle_inputs,
            # Serialized only when asked for, so every digest built over these
            # parameters is unchanged for a run that does not use it.
            **(
                {"properCoaddition": self.proper_coaddition.serializable()}
                if self.proper_coaddition.enabled
                else {}
            ),
            "cosmeticHotPixelSigma": self.cosmetic_hot_pixel_sigma,
            "durableIntermediates": self.durable_intermediates,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_directory: str
    receipt_path: str
    state: str
    master_light_paths: tuple[str, ...]
    preview_paths: tuple[str, ...]
    # Per filter, the drizzle inputs captured by ``capture_drizzle_inputs``
    # (in-memory rejection masks; not part of the serializable receipt).
    drizzle_groups: Mapping[str, DrizzleGroupInputs] = field(default_factory=dict)
    # Per filter, the additional proper-coaddition product, when the recipe
    # asked for one.  The ordinary masters above stay the primary products.
    proper_coadd_paths: Mapping[str, str] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "outputDirectory": self.output_directory,
            "receiptPath": self.receipt_path,
            "state": self.state,
            "masterLightPaths": list(self.master_light_paths),
            "previewPaths": list(self.preview_paths),
            **(
                {"properCoaddPaths": dict(self.proper_coadd_paths)}
                if self.proper_coadd_paths
                else {}
            ),
        }


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


def _require_filter(info: FrameInfo) -> str:
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
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None,
) -> dict[Path, PixelTransform]:
    if transforms is None:
        return {path: PixelTransform.identity() for path in light_paths}
    if not all(isinstance(key, str) and key for key in transforms):
        raise CalibrationError("TRANSFORM_KEY_INVALID", "transform keys must be strings")
    basename_counts: dict[str, int] = {}
    for path in light_paths:
        basename_counts[path.name] = basename_counts.get(path.name, 0) + 1
    used: set[str] = set()
    result: dict[Path, PixelTransform] = {}
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
            values = [PixelTransform.from_value(transforms[key]) for key in matching]
            matrices = [value.validated_matrix() for value in values]
            if not all(np.array_equal(matrices[0], matrix) for matrix in matrices[1:]):
                raise CalibrationError(
                    "TRANSFORM_KEY_CONFLICT", "multiple transform keys disagree", path=str(path)
                )
        if matching:
            key = matching[0]
            used.update(matching)
            result[path] = PixelTransform.from_value(transforms[key])
        else:
            result[path] = PixelTransform.identity()
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


def _resolve_region_weight_maps(
    light_paths: tuple[Path, ...],
    maps: Mapping[str, Any] | None,
) -> dict[Path, Any]:
    """Bind optional region weight maps to admitted Lights (unknown keys fail)."""

    if not maps:
        return {}
    canonical = {os.path.normcase(str(path)): path for path in light_paths}
    result: dict[Path, Any] = {}
    for key, region_map in maps.items():
        if not isinstance(key, str) or not key:
            raise CalibrationError(
                "REGION_WEIGHT_KEY_INVALID", "region weight map keys must be non-empty strings"
            )
        try:
            path = Path(key).expanduser().resolve(strict=True)
        except OSError as error:
            raise CalibrationError(
                "REGION_WEIGHT_INPUT_UNKNOWN",
                "region weight map does not bind an existing Light",
                path=key,
            ) from error
        bound = canonical.get(os.path.normcase(str(path)))
        if bound is None:
            raise CalibrationError(
                "REGION_WEIGHT_INPUT_UNKNOWN",
                "region weight map does not bind an integration Light",
                path=str(path),
            )
        if not hasattr(region_map, "nodes") or not hasattr(region_map, "pixel_nodes"):
            raise CalibrationError(
                "REGION_WEIGHT_MAP_INVALID",
                "region weight maps must provide nodes and pixel_nodes()",
                path=str(path),
            )
        if bound in result:
            raise CalibrationError(
                "REGION_WEIGHT_KEY_CONFLICT",
                "multiple region weight map keys bind the same Light",
                path=str(path),
            )
        result[bound] = region_map
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


def _inverse_coordinates(
    inverse: NDArray[np.float64],
    output_x: NDArray[np.float64],
    output_y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map output pixel coordinates to input coordinates.

    The evaluation order ``((m00*x) + (m01*y)) + m02`` and, for a projective
    map, the division by ``((m20*x) + (m21*y)) + m22`` are the arithmetic
    contract the native warp kernel reproduces value for value.
    """

    input_x = inverse[0, 0] * output_x + inverse[0, 1] * output_y + inverse[0, 2]
    input_y = inverse[1, 0] * output_x + inverse[1, 1] * output_y + inverse[1, 2]
    if not (inverse[2, 0] == 0.0 and inverse[2, 1] == 0.0 and inverse[2, 2] == 1.0):
        denominator = inverse[2, 0] * output_x + inverse[2, 1] * output_y + inverse[2, 2]
        input_x = input_x / denominator
        input_y = input_y / denominator
    return input_x, input_y


def _exact_half_turn_translation(
    transform: PixelTransform, shape: tuple[int, int]
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
    transform: PixelTransform, shape: tuple[int, int], resampler: str
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
    transform: PixelTransform,
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
        "OAFREG": (
            "IDENTITY"
            if transform.is_identity
            else "PROJECTIVE" if transform.is_projective else "AFFINE"
        ),
        "OAFRSAMP": actual_resampler.upper(),
        "OAFRCLMP": "DOMAIN_UNION_SUPPORT" if uses_lanczos else None,
        "OAFRMARG": 2 if uses_lanczos else 0,
        "OAFSRCEX": source_exposure_seconds,
        **_numeric_domain_metadata(info),
    }


def _registration_bytes_per_pixel(
    transform: PixelTransform, resampler: str, shape: tuple[int, int]
) -> int:
    if transform.is_identity:
        return 24
    if _exact_half_turn_translation(transform, shape) is not None:
        # Includes the destination tile, scaled FITS conversion and previous
        # iteration's values/statistics buffers while the next tile is read.
        return 32
    return 192 if resampler == "lanczos-3-clamped" else 112


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
    transform: PixelTransform,
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
                            inverse,
                            first_row=y0,
                            row_count=y1 - y0,
                            output_width=width,
                            domain_scale=float(domain_scale),
                            threads=native_threads,
                        )
                    else:
                        output_y = np.arange(y0, y1, dtype=np.float64)[:, None]
                        output_x = np.arange(width, dtype=np.float64)[None, :]
                        input_x, input_y = _inverse_coordinates(inverse, output_x, output_y)
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
            remove_file(temporary, missing_ok=False)
        finally:
            remove_file(temporary)
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


class _RejectionMaskRecorder:
    """Tile observer that keeps every frame's accepted-sample mask as packed
    row bits on the reference grid, the form the drizzle stage reads."""

    def __init__(self, frame_count: int, shape: tuple[int, int]) -> None:
        height, width = shape
        self.bits = [
            np.zeros((height, (width + 7) // 8), dtype=np.uint8) for _ in range(frame_count)
        ]
        self._rows_seen = np.zeros(height, dtype=bool)

    def __call__(self, observation: Any) -> None:
        accepted = np.asarray(observation.accepted, dtype=bool)
        first_row = int(observation.first_row)
        rows = accepted.shape[1]
        for index, frame_bits in enumerate(self.bits):
            frame_bits[first_row : first_row + rows] = np.packbits(accepted[index], axis=1)
        self._rows_seen[first_row : first_row + rows] = True

    @property
    def complete(self) -> bool:
        """Every row was observed; an unobserved row would read as all rejected."""

        return bool(np.all(self._rows_seen))


def _compose_tile_observers(*observers: Any) -> Callable[[Any], None] | None:
    active = [observer for observer in observers if observer is not None]
    if not active:
        return None

    def observe(observation: Any) -> None:
        for observer in active:
            observer(observation)

    return observe


@dataclass(frozen=True, slots=True)
class _ChannelDestination:
    """One registered output of a Light: its group and, for a Bayer Light,
    the colour channel (0 R, 1 G, 2 B) debayered before the warp."""

    group: str
    channel: int | None
    path: Path


@dataclass(frozen=True, slots=True)
class _LightJob:
    """One Light's fused calibrate-in-memory then register work item."""

    source_path: Path
    expression: FrameExpression
    calibrated_path: Path | None
    calibrated_metadata: Mapping[str, Any]
    destinations: tuple[_ChannelDestination, ...]
    transform: PixelTransform
    info: FrameInfo
    source_exposure_seconds: float | None
    # Master dark whose hot-pixel map drives the cosmetic correction, and the
    # detection threshold; ``None`` leaves the calibrated pixels untouched.
    hot_pixel_master: str | None = None
    hot_pixel_sigma: float | None = None
    # Bayer pattern of the Light (None for mono): the calibrated mosaic is
    # debayered into the channels the destinations ask for.
    cfa_pattern: str | None = None

    @property
    def channel_count(self) -> int:
        return 3 if self.cfa_pattern is not None else 1


@dataclass(frozen=True, slots=True)
class _RegisteredOutput:
    group: str
    channel: int | None
    path: Path
    statistics: PixelStatistics
    sha256: str | None
    execution: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _LightJobResult:
    calibrated_statistics: PixelStatistics
    calibrated_sha256: str | None
    registered: tuple[_RegisteredOutput, ...]
    cosmetic: dict[str, Any] | None = None
    debayer: dict[str, Any] | None = None

    @property
    def execution(self) -> dict[str, Any]:
        return self.registered[0].execution if self.registered else {}


class _MasterCache:
    """Decoded Float32 masters shared read-only by every fused worker.

    Each master is converted from its FITS storage exactly once; workers
    receive copies of the rows they ask for, so the cache is never mutated.
    """

    def __init__(self) -> None:
        self._frames: dict[str, _MemoryFrame] = {}
        self._hot_pixels: dict[tuple[str, float], tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]] = {}
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

    def hot_pixels(self, path: str, sigma: float) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
        """Row/column indices of the master dark's hot pixels (cached per dark)."""

        master = self.frame(path)
        key = (str(master.path), float(sigma))
        with self._lock:
            cached = self._hot_pixels.get(key)
        if cached is None:
            cached = _hot_pixel_map(master.values, sigma)
            with self._lock:
                self._hot_pixels[key] = cached
        return cached


def _hot_pixel_map(
    values: NDArray[np.float32], sigma: float
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), {"count": 0, "threshold": None}
    median = float(np.median(finite))
    dispersion = 1.4826 * float(np.median(np.abs(finite - median)))
    if not math.isfinite(dispersion) or dispersion <= 0.0:
        # A dark without measurable dispersion (synthetic or degenerate)
        # carries no hot-pixel evidence.
        return np.empty(0, np.int64), np.empty(0, np.int64), {
            "count": 0, "fraction": 0.0, "darkMedian": median, "darkRobustSigma": dispersion,
            "threshold": None, "sigma": float(sigma),
        }
    threshold = float(median + sigma * dispersion)
    rows, columns = np.nonzero(values > np.float32(threshold))
    return rows.astype(np.int64), columns.astype(np.int64), {
        "count": int(rows.size),
        "fraction": float(rows.size / values.size),
        "darkMedian": median,
        "darkRobustSigma": dispersion,
        "threshold": threshold,
        "sigma": float(sigma),
    }


def _replace_hot_pixels(
    image: NDArray[np.float32],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    *,
    cfa: bool = False,
) -> None:
    """Replace the listed pixels in place by the median of their eight neighbours.

    On a Bayer mosaic the neighbours are the eight same-colour pixels two
    steps away, so a hot pixel never takes on another colour's value.
    """

    if rows.size == 0:
        return
    height, width = image.shape
    step = 2 if cfa else 1
    neighbours = np.empty((8, rows.size), dtype=np.float32)
    index = 0
    for dy in (-step, 0, step):
        for dx in (-step, 0, step):
            if dy == 0 and dx == 0:
                continue
            neighbours[index] = image[
                np.clip(rows + dy, 0, height - 1), np.clip(columns + dx, 0, width - 1)
            ]
            index += 1
    image[rows, columns] = np.nanmedian(neighbours, axis=0)


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
        remove_file(temporary)


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
    cosmetic: dict[str, Any] | None = None
    if job.hot_pixel_master is not None and job.hot_pixel_sigma is not None:
        rows, columns, evidence = master_cache.hot_pixels(job.hot_pixel_master, job.hot_pixel_sigma)
        _replace_hot_pixels(calibrated, rows, columns, cfa=job.cfa_pattern is not None)
        cosmetic = {
            "algorithm": (
                "master-dark-hot-pixel-same-colour-neighbour-median-v1"
                if job.cfa_pattern is not None
                else "master-dark-hot-pixel-neighbour-median-v1"
            ),
            "replacedPixels": int(rows.size),
            **evidence,
        }
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
    # A Bayer Light is debayered once; each colour plane is then registered
    # like a mono Light of its own filter group.  The mosaic itself stays the
    # materialized calibrated frame (the drizzle drops its real samples).
    planes: NDArray[np.float32] | None = None
    debayer: dict[str, Any] | None = None
    if job.cfa_pattern is not None:
        debayer_started = time.perf_counter()
        planes = bilinear_debayer(calibrated, job.cfa_pattern)
        debayer = {
            "algorithm": "bilinear-same-colour-neighbours-v1",
            "pattern": job.cfa_pattern,
            "seconds": round(time.perf_counter() - debayer_started, 3),
        }
    occupied = calibrated.nbytes + (planes.nbytes if planes is not None else 0)
    # The calibrated image already occupies its share; leave the rest of the
    # worker budget to warp tiles, but always allow at least one NumPy row.
    warp_budget = max(
        max_memory_bytes - occupied,
        calibrated.shape[1]
        * _registration_bytes_per_pixel(job.transform, resampler, calibrated.shape),
    )
    registered: list[_RegisteredOutput] = []
    for destination in job.destinations:
        if destination.channel is None:
            source_values = calibrated
        else:
            if planes is None:
                raise CalibrationError(
                    "CFA_CHANNEL_WITHOUT_PATTERN",
                    "a colour channel destination needs the Light's Bayer pattern",
                    path=str(job.source_path),
                )
            source_values = planes[destination.channel]
        memory_frame = _MemoryFrame(
            source_values, job.info, job.calibrated_path or job.source_path
        )
        execution: dict[str, Any] = {}
        registered_statistics = _register_frame(
            memory_frame,
            destination.path,
            job.transform,
            job.info,
            max_memory_bytes=warp_budget,
            resampler=resampler,
            source_exposure_seconds=job.source_exposure_seconds,
            native_threads=native_threads,
            execution=execution,
            durable=durable,
        )
        registered.append(
            _RegisteredOutput(
                group=destination.group,
                channel=destination.channel,
                path=destination.path,
                statistics=registered_statistics,
                sha256=execution.pop("sha256", None),
                execution=execution,
            )
        )
    return _LightJobResult(
        calibrated_statistics=calibrated_statistics,
        calibrated_sha256=calibrated_sha256,
        registered=tuple(registered),
        cosmetic=cosmetic,
        debayer=debayer,
    )


def _fused_job_bytes(job: _LightJob, resampler: str) -> int:
    height, width = job.info.shape
    per_row = width * max(
        NATIVE_WARP_BYTES_PER_PIXEL,
        _registration_bytes_per_pixel(job.transform, resampler, job.info.shape),
    )
    # A Bayer Light also holds its three debayered planes while it is warped.
    planes = 3 * height * width * 4 if job.cfa_pattern is not None else 0
    return height * width * FUSED_LIGHT_BYTES_PER_PIXEL + planes + per_row


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
    kernel_threads: int | None = None,
) -> tuple[tuple[_LightJobResult, ...], dict[str, Any]]:
    """Calibrate and register every Light with one shared memory budget.

    Lights run concurrently in ``workers`` threads; each thread hands its warp
    to the native kernel with the remaining CPU share, so all cores stay busy
    whether memory allows many Lights in flight or only one.  ``kernel_threads``
    is the native thread budget shared by the in-flight warps (defaults to
    ``cpu_workers``, the pre-tuning-table behaviour).
    """

    workers = _fused_worker_count(
        jobs,
        max_memory_bytes=max_memory_bytes,
        resampler=resampler,
        cpu_workers=cpu_workers,
    )
    thread_budget = cpu_workers if kernel_threads is None else max(1, int(kernel_threads))
    native_threads = max(1, thread_budget // workers)
    worker_memory_bytes = max_memory_bytes // workers
    # Lights start in submission order, so the last ``len(jobs) % workers``
    # Lights run while the other workers are already idle; their warps take
    # the CPU share those workers would have used. Warp results do not depend
    # on the thread count.
    rounds = max(1, math.ceil(len(jobs) / workers))
    tail_start = (rounds - 1) * workers
    tail_threads = max(native_threads, thread_budget // max(1, len(jobs) - tail_start))

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


def _normalization_reference_index(
    paths: Sequence[Path],
    hints: Mapping[Path, StellarScaleHint | None],
    quality_weights: Mapping[Path, float],
) -> tuple[int, dict[str, Any]]:
    """Normalization reference of one filter group.

    Registration chose the group's stellar-scale reference (the lowest-sky
    frame of acceptable quality) and bound every hint to it; the same frame
    is the additive reference so the master inherits its background.  Without
    hints the highest-quality frame is used, as before.
    """

    hinted = {
        Path(hint.reference_path).expanduser().resolve(strict=True)
        for path in paths
        if (hint := hints.get(path)) is not None
    }
    if len(hinted) == 1:
        reference = next(iter(hinted))
        for index, path in enumerate(paths):
            if path == reference:
                return index, {"rule": "stellar-scale-hint-reference", "reference": str(path)}
    index = max(range(len(paths)), key=lambda item: quality_weights[paths[item]])
    return index, {"rule": "highest-quality-weight", "reference": str(paths[index])}


def _shared_auto_crop(
    light_groups: Mapping[str, Sequence[Path]],
    light_info: Mapping[Path, FrameInfo],
    transforms: Mapping[Path, PixelTransform],
    *,
    enabled: bool,
    max_memory_bytes: int,
    resampler: str,
) -> tuple[tuple[int, int, int, int] | None, dict[str, tuple[int, int, int, int]]]:
    """Intersect the per-filter valid crops of one registration run.

    Returns ``(shared_crop, crop_by_filter)``; ``shared_crop`` is ``None`` when
    auto-crop is disabled.  All groups must share the registered geometry,
    which is the source frame shape of every Light of the run.
    """

    if not enabled or not light_groups:
        return None, {}
    shapes = {
        filter_name: light_info[paths[0]].shape for filter_name, paths in light_groups.items()
    }
    if len(set(shapes.values())) != 1:
        raise CalibrationError(
            "REGISTRATION_GEOMETRY_MISMATCH",
            "filter groups of one run must share the registered frame geometry: "
            + ", ".join(f"{name}={shape}" for name, shape in sorted(shapes.items())),
        )
    crops: dict[str, tuple[int, int, int, int]] = {}
    for filter_name, paths in sorted(light_groups.items()):
        crops[filter_name] = _common_valid_crop(
            shapes[filter_name],
            [transforms[path] for path in paths],
            max_memory_bytes=max_memory_bytes,
            resampler=resampler,
        )
    shared = (
        max(crop[0] for crop in crops.values()),
        max(crop[1] for crop in crops.values()),
        min(crop[2] for crop in crops.values()),
        min(crop[3] for crop in crops.values()),
    )
    if shared[2] <= shared[0] or shared[3] <= shared[1]:
        raise CalibrationError(
            "AUTOCROP_TOO_SMALL",
            "the filter groups of this run have no common fully covered rectangle",
        )
    return shared, crops


def _common_valid_crop(
    shape: tuple[int, int],
    transforms: Sequence[PixelTransform],
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
    # Every frame's valid pixels of a row form one run (the frame's footprint
    # is convex), so each frame contributes a per-row interval, found with
    # the exact per-pixel test by bisection; the common mask row is the
    # intersection of the intervals.  Rows the bisection cannot bracket are
    # evaluated pixel by pixel, so every mask row equals the dense one.
    heights = np.zeros(width, dtype=np.int64)
    best: tuple[int, int, int, int, int] | None = None
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        first = np.zeros(y1 - y0, dtype=np.int64)
        last = np.full(y1 - y0, width - 1, dtype=np.int64)
        for inverse, interpolation_margin in zip(inverses, margins, strict=True):
            frame_first, frame_last = _valid_row_runs(
                inverse, interpolation_margin, y0, y1, width, height
            )
            first = np.maximum(first, frame_first)
            last = np.minimum(last, frame_last)
        for local_y in range(y1 - y0):
            run_first = int(first[local_y])
            run_last = int(last[local_y])
            if run_last < run_first:
                heights.fill(0)
            else:
                heights[:run_first] = 0
                heights[run_first : run_last + 1] += 1
                heights[run_last + 1 :] = 0
            candidate = _histogram_rectangle(heights, y0 + local_y)
            if candidate is not None and (best is None or candidate > best):
                best = candidate
    if best is None or best[0] == 0:
        raise CalibrationError(
            "AUTOCROP_EMPTY", "registered inputs have no common finite rectangle"
        )
    _, top, left, bottom, right = best
    return top, left, bottom, right


def _valid_row_runs(
    inverse: NDArray[np.float64],
    interpolation_margin: int,
    y0: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Per-row ``(first, last)`` valid output columns of one frame, rows y0..y1.

    A column is valid when its inverse-mapped coordinates lie inside the
    source frame less the interpolation margin, the dense mask test.  Rows
    without a valid column return ``last < first``.
    """

    rows = y1 - y0
    output_y = np.arange(y0, y1, dtype=np.float64)
    x_low = float(interpolation_margin)
    x_high = float(width - 1 - interpolation_margin)
    y_low = float(interpolation_margin)
    y_high = float(height - 1 - interpolation_margin)

    def valid_at(columns: NDArray[np.int64]) -> NDArray[np.bool_]:
        input_x, input_y = _inverse_coordinates(
            inverse, columns.astype(np.float64), output_y
        )
        return (
            (input_x >= x_low) & (input_x <= x_high)
            & (input_y >= y_low) & (input_y <= y_high)
        )

    # Analytic run centre of each row (the geometry is affine or a
    # near-identity projective map, so the source-centre column bounds the
    # run's interior); the exact test decides whether it is inside.
    centre_x = 0.5 * (x_low + x_high)
    centre_y = 0.5 * (y_low + y_high)
    a, b, c = inverse[0]
    d, e, f = inverse[1]
    g, h, i = inverse[2]
    # Solve for x with y fixed: the column mapping to input (centre_x, *) or,
    # when the x row is degenerate, to input (*, centre_y).
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator_x = centre_x * (h * output_y + i) - (b * output_y + c)
        denominator_x = a - centre_x * g
        numerator_y = centre_y * (h * output_y + i) - (e * output_y + f)
        denominator_y = d - centre_y * g
        guess = np.where(
            np.abs(denominator_x) >= np.abs(denominator_y),
            numerator_x / denominator_x,
            numerator_y / denominator_y,
        )
    guess = np.where(np.isfinite(guess), guess, 0.5 * (width - 1))
    inside = np.clip(np.rint(guess), 0, width - 1).astype(np.int64)
    bracketed = valid_at(inside)
    first = np.zeros(rows, dtype=np.int64)
    last = np.full(rows, -1, dtype=np.int64)
    if np.any(bracketed):
        # Bisection on the single run: ``low`` is outside (or virtual -1 /
        # width), ``high`` is inside.
        low = np.full(rows, -1, dtype=np.int64)
        high = inside.copy()
        while True:
            active = high - low > 1
            if not np.any(active):
                break
            middle = (low + high) // 2
            probe = valid_at(np.clip(middle, 0, width - 1))
            high = np.where(active & probe, middle, high)
            low = np.where(active & ~probe, middle, low)
        first_bisect = high
        low = inside.copy()
        high = np.full(rows, width, dtype=np.int64)
        while True:
            active = high - low > 1
            if not np.any(active):
                break
            middle = (low + high) // 2
            probe = valid_at(np.clip(middle, 0, width - 1))
            low = np.where(active & probe, middle, low)
            high = np.where(active & ~probe, middle, high)
        last_bisect = low
        first = np.where(bracketed, first_bisect, first)
        last = np.where(bracketed, last_bisect, last)
    unbracketed = np.flatnonzero(~bracketed)
    if unbracketed.size:
        # Rows whose centre guess is outside (edge rows of a tilted frame,
        # or an unexpected geometry): the dense row test decides, a bounded
        # number of rows at a time.
        output_x = np.arange(width, dtype=np.float64)[None, :]
        chunk = max(1, (8 * 1024 * 1024) // (width * 40))
        for start in range(0, unbracketed.size, chunk):
            selected = unbracketed[start : start + chunk]
            input_x, input_y = _inverse_coordinates(
                inverse, output_x, output_y[selected][:, None]
            )
            dense = (
                (input_x >= x_low) & (input_x <= x_high)
                & (input_y >= y_low) & (input_y <= y_high)
            )
            any_valid = np.any(dense, axis=1)
            dense_first = np.argmax(dense, axis=1)
            dense_last = width - 1 - np.argmax(dense[:, ::-1], axis=1)
            first[selected] = np.where(any_valid, dense_first, 0)
            last[selected] = np.where(any_valid, dense_last, -1)
    return first, last


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
            remove_file(temporary, missing_ok=False)
        finally:
            remove_file(temporary)
    return PixelStatistics(
        finite_pixels=finite_total,
        invalid_pixels=invalid_total,
        minimum=minimum if finite_total else None,
        maximum=maximum if finite_total else None,
        mean=total / finite_total if finite_total else None,
    ), digest


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Create-only directory publication through the platform service layer."""

    try:
        platform_services.current().rename_directory_no_replace(source, destination)
    except NoReplaceError as error:
        if error.code == "OUTPUT_EXISTS":
            raise CalibrationError(
                "OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(destination)
            ) from error
        raise CalibrationError(error.code, error.message) from error


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


@dataclass(frozen=True)
class _RunPlan:
    """What a run decides before it writes a pixel.

    Canonical inputs, their frame metadata after overrides, the calibration
    and output groups, each Light's transform/weight/hint bindings and the
    provenance records the receipt reports.  Building it validates every
    input combination, so the stages after it only execute.
    """

    output: Path
    lights: tuple[Path, ...]
    biases: tuple[Path, ...]
    master_biases: tuple[Path, ...]
    source_aliases: dict[str, Path]
    source_groups: tuple[tuple[str, tuple[Path, ...]], ...]
    source_identity_cache: _SourceIdentityCache
    trusted_generated: dict[str, Any] | None
    bias_info: dict[Path, FrameInfo]
    dark_info: dict[Path, FrameInfo]
    flat_info: dict[Path, FrameInfo]
    master_dark_info: dict[Path, FrameInfo]
    master_flat_info: dict[Path, FrameInfo]
    light_info: dict[Path, FrameInfo]
    supplied_dark_bias_included: dict[Path, bool]
    reference_bias: FrameInfo
    flat_groups: dict[str, list[Path]]
    supplied_flats: dict[str, Path]
    light_groups: dict[str, list[Path]]
    reference_exposures: dict[str, float]
    light_domain_references: dict[str, FrameInfo]
    # Output (integration) groups: a mono filter is its own group; a Bayer
    # filter becomes the colour channel groups R, G and B.
    output_groups: dict[str, list[Path]]
    group_filter: dict[str, str]
    group_channel: dict[str, int | None]
    group_cfa_pattern: dict[str, str | None]
    light_cfa_pattern: dict[str, str | None]
    dark_groups: dict[float, list[Path]]
    supplied_darks: dict[float, Path]
    trusted_bias: _TrustedGeneratedMaster | None
    trusted_darks_by_exposure: dict[float, _TrustedGeneratedMaster]
    trusted_flats_by_filter: dict[str, _TrustedGeneratedMaster]
    transforms: dict[Path, PixelTransform]
    quality_weights: dict[Path, float]
    region_weight_maps: dict[Path, Any]
    stellar_scale_hints: dict[Path, StellarScaleHint | None]
    source_records: list[dict[str, Any]]
    source_identities: dict[str, dict[str, int]]
    pixel_numeric_domains: list[dict[str, Any]]

    def display_path(self, path: Path) -> Path:
        return self.source_aliases.get(str(path), path)

    @property
    def trusted_by_path(self) -> Mapping[str, _TrustedGeneratedMaster] | None:
        return self.trusted_generated["byPath"] if self.trusted_generated is not None else None

    def receipt_reference(self, staging: Path, path: Path) -> dict[str, Any]:
        return _path_receipt_reference(
            staging, path, self.source_aliases, self.trusted_by_path, self.source_identity_cache
        )

    def dark_reference(self, exposure: float) -> FrameInfo:
        """The frame metadata that stands for the master dark of ``exposure``."""

        if exposure in self.dark_groups:
            return self.dark_info[self.dark_groups[exposure][0]]
        return self.master_dark_info[self.supplied_darks[exposure]]

    def dark_bias_included(self, exposure: float, subtract_path: Path) -> bool:
        if exposure in self.trusted_darks_by_exposure:
            return bool(self.trusted_darks_by_exposure[exposure].bias_included)
        if exposure in self.dark_groups:
            return True
        return self.supplied_dark_bias_included[subtract_path]


def _plan_run(
    *,
    bias_files: Iterable[str | os.PathLike[str]],
    dark_files: Iterable[str | os.PathLike[str]],
    flat_files: Iterable[str | os.PathLike[str]],
    master_bias_file: str | os.PathLike[str] | None,
    master_dark_files: Iterable[str | os.PathLike[str]],
    master_flat_files: Iterable[str | os.PathLike[str]],
    light_files: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None,
    quality_weights: Mapping[str, float] | None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None,
    region_weight_maps: Mapping[str, Any] | None,
    parameters: PipelineParameters,
    source_aliases: Mapping[str, Path] | None,
    trusted_generated_calibration: _TrustedGeneratedCalibrationSet | None,
    source_identity_seed: Mapping[str, tuple[str, Mapping[str, int]]] | None,
) -> _RunPlan:
    workflow = parameters.calibration_workflow
    biases = _canonical_inputs(bias_files, "Bias", required=False)
    darks = _canonical_inputs(dark_files, "Dark", required=False)
    flats = _canonical_inputs(flat_files, "Flat", required=False)
    master_biases = _canonical_inputs(
        (() if master_bias_file is None else (master_bias_file,)),
        "MasterBias",
        required=False,
    )
    master_darks_input = _canonical_inputs(master_dark_files, "MasterDark", required=False)
    master_flats_input = _canonical_inputs(master_flat_files, "MasterFlat", required=False)
    lights = _canonical_inputs(light_files, "Light", required=True)
    if (biases and master_biases) or (not biases and not master_biases and workflow != MONO_STANDARD):
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

    aliases = dict(source_aliases or {})
    source_groups = (
        ("BIAS", biases),
        ("DARK", darks),
        ("FLAT", flats),
        ("MASTER_BIAS", master_biases),
        ("MASTER_DARK", master_darks_input),
        ("MASTER_FLAT", master_flats_input),
        ("LIGHT", lights),
    )
    identity_cache: _SourceIdentityCache = {}
    for seed_path, (seed_digest, seed_identity) in (source_identity_seed or {}).items():
        seed_key = os.path.normcase(str(Path(seed_path).expanduser().resolve(strict=True)))
        identity_cache[seed_key] = (str(seed_digest), dict(seed_identity))
    trusted_generated: dict[str, Any] | None = None
    if trusted_generated_calibration is not None:
        trusted_generated = _validate_trusted_generated_calibration_set(
            trusted_generated_calibration,
            source_groups=source_groups,
            source_aliases=aliases,
            identity_cache=identity_cache,
        )
        original_keys = {os.path.normcase(str(path)) for path in all_paths}
        if original_keys.intersection(trusted_generated["byPath"]):
            raise CalibrationError(
                "TRUSTED_GENERATED_MASTER_INPUT_OVERLAP",
                "E2E-generated masters cannot be presented as public input files",
            )

    output = Path(output_directory).expanduser().resolve(strict=False)
    if output.exists() or os.path.lexists(output):
        raise CalibrationError("OUTPUT_EXISTS", "output directory must be new", path=str(output))
    output.parent.mkdir(parents=True, exist_ok=True)

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
            _read_infos(biases, "BIAS"),
            _read_infos(darks, "DARK"),
            _read_infos(flats, "FLAT"),
            _read_infos(master_biases, "MASTER_BIAS"),
            _read_infos(master_darks_input, "MASTER_DARK"),
            _read_infos(master_flats_input, "MASTER_FLAT"),
            _read_infos(lights, "LIGHT"),
        ),
        aliases,
    )
    bias_info, dark_info, flat_info, light_info = _apply_raw_frame_metadata_overrides(
        (bias_info, dark_info, flat_info, light_info),
        parameters.raw_frame_metadata_overrides,
        dict(aliases),
        identity_cache,
    )
    master_bias_info, master_dark_info, master_flat_info = _apply_master_metadata_overrides(
        (master_bias_info, master_dark_info, master_flat_info),
        parameters.master_metadata_overrides,
        dict(aliases),
        identity_cache,
    )
    supplied_dark_bias_included = _master_dark_bias_semantics(
        master_darks_input,
        parameters.master_metadata_overrides,
        dict(aliases),
        identity_cache,
        workflow=workflow,
    )
    for group in (bias_info, dark_info, flat_info, master_bias_info, master_dark_info, master_flat_info, light_info):
        for path, info in group.items():
            group[path] = apply_mono_workflow(info, workflow)
    reference_bias = (
        bias_info[biases[0]] if biases else master_bias_info[master_biases[0]] if master_biases else light_info[lights[0]]
    )
    profile_infos = [*bias_info.values(), *master_bias_info.values(), *dark_info.values(), *flat_info.values(), *master_dark_info.values(), *master_flat_info.values(), *light_info.values()]
    conflicts = conflicting_profile_fields(profile_infos, workflow)
    if conflicts:
        raise CalibrationError("CALIBRATION_PROFILE_MISMATCH", "Conflicting known acquisition metadata: " + ", ".join(conflicts))
    if not biases and not master_biases and not can_omit_bias(
        (*light_info.values(), *flat_info.values()),
        [*((info, True) for info in dark_info.values()), *((info, supplied_dark_bias_included[path]) for path, info in master_dark_info.items())],
        workflow,
    ):
        raise CalibrationError("BIAS_REQUIRED_FOR_CALIBRATION", "Bias is required unless every Light and raw Flat has a matching Dark that includes Bias.")
    for info in (*bias_info.values(), *master_bias_info.values()):
        _assert_compatible(reference_bias, info, workflow=workflow)
    for info in (
        *dark_info.values(),
        *flat_info.values(),
        *master_dark_info.values(),
        *master_flat_info.values(),
        *light_info.values(),
    ):
        _assert_compatible(reference_bias, info, workflow=workflow)

    flat_groups, supplied_flats, light_groups = _group_flats_and_lights(flat_info, master_flat_info, light_info, workflow)
    reference_exposures = {
        filter_name: min(float(light_info[path].exposure_seconds) for path in paths)
        for filter_name, paths in light_groups.items()
    }
    light_domain_references = {
        filter_name: light_info[paths[0]] for filter_name, paths in light_groups.items()
    }
    output_groups, group_filter, group_channel, group_cfa_pattern, light_cfa_pattern = _output_groups(
        light_groups, light_info, workflow
    )
    for filter_name, paths in flat_groups.items():
        reference = flat_info[paths[0]]
        for path in paths[1:]:
            _assert_compatible(reference, flat_info[path], compare_filter=True, workflow=workflow)
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
    for filter_name in {*flat_groups, *supplied_flats, *light_groups, *output_groups}:
        token = _safe_token(filter_name)
        if token in tokens and tokens[token] != filter_name:
            raise CalibrationError(
                "FILTER_FILENAME_COLLISION",
                f"filters {tokens[token]!r} and {filter_name!r} share output token {token}",
            )
        tokens[token] = filter_name
    dark_groups, supplied_darks = _group_darks(dark_info, master_dark_info, light_info)
    trusted_bias, trusted_darks_by_exposure, trusted_flats_by_filter = _trusted_generated_coverage(
        trusted_generated,
        has_raw_bias=bool(biases),
        reference_bias=reference_bias,
        dark_info=dark_info,
        dark_groups=dark_groups,
        flat_info=flat_info,
        flat_groups=flat_groups,
        parameters=parameters,
    )
    resolved_transforms = _resolve_transforms(lights, transforms)
    resolved_quality_weights = _resolve_quality_weights(lights, quality_weights)
    resolved_region_weight_maps = _resolve_region_weight_maps(lights, region_weight_maps)
    resolved_stellar_scale_hints = _resolve_stellar_scale_hints(
        lights, light_info, stellar_scale_hints, aliases, identity_cache
    )
    source_records, source_identities = _source_records(source_groups, aliases, identity_cache)
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
        aliases,
        identity_cache,
    )
    return _RunPlan(
        output=output,
        lights=lights,
        biases=biases,
        master_biases=master_biases,
        source_aliases=aliases,
        source_groups=source_groups,
        source_identity_cache=identity_cache,
        trusted_generated=trusted_generated,
        bias_info=bias_info,
        dark_info=dark_info,
        flat_info=flat_info,
        master_dark_info=master_dark_info,
        master_flat_info=master_flat_info,
        light_info=light_info,
        supplied_dark_bias_included=supplied_dark_bias_included,
        reference_bias=reference_bias,
        flat_groups=flat_groups,
        supplied_flats=supplied_flats,
        light_groups=light_groups,
        reference_exposures=reference_exposures,
        light_domain_references=light_domain_references,
        output_groups=output_groups,
        group_filter=group_filter,
        group_channel=group_channel,
        group_cfa_pattern=group_cfa_pattern,
        light_cfa_pattern=light_cfa_pattern,
        dark_groups=dark_groups,
        supplied_darks=supplied_darks,
        trusted_bias=trusted_bias,
        trusted_darks_by_exposure=trusted_darks_by_exposure,
        trusted_flats_by_filter=trusted_flats_by_filter,
        transforms=resolved_transforms,
        quality_weights=resolved_quality_weights,
        region_weight_maps=resolved_region_weight_maps,
        stellar_scale_hints=resolved_stellar_scale_hints,
        source_records=source_records,
        source_identities=source_identities,
        pixel_numeric_domains=pixel_numeric_domains,
    )


def _group_flats_and_lights(
    flat_info: Mapping[Path, FrameInfo],
    master_flat_info: Mapping[Path, FrameInfo],
    light_info: Mapping[Path, FrameInfo],
    workflow: str,
) -> tuple[dict[str, list[Path]], dict[str, Path], dict[str, list[Path]]]:
    """Raw Flat groups, supplied MasterFlats and Light groups, all by filter."""

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
                workflow=workflow,
            )
    return flat_groups, supplied_flats, light_groups


def _output_groups(
    light_groups: Mapping[str, list[Path]],
    light_info: Mapping[Path, FrameInfo],
    workflow: str,
) -> tuple[
    dict[str, list[Path]],
    dict[str, str],
    dict[str, int | None],
    dict[str, str | None],
    dict[str, str | None],
]:
    """Integration groups: a mono filter group is its own output group; a
    Bayer filter group is debayered into the colour channel groups R, G and
    B, each holding every Light of the filter, so the rest of the pipeline
    treats a colour channel exactly like a filter."""

    output_groups: dict[str, list[Path]] = {}
    group_filter: dict[str, str] = {}
    group_channel: dict[str, int | None] = {}
    group_cfa_pattern: dict[str, str | None] = {}
    light_cfa_pattern: dict[str, str | None] = {}
    for filter_name, paths in light_groups.items():
        pattern = normalize_cfa_pattern(cfa_for_workflow(light_info[paths[0]].cfa_pattern, workflow))
        if pattern in CFA_PATTERNS:
            light_cfa_pattern[filter_name] = pattern
            for channel, channel_name in enumerate(CHANNEL_NAMES):
                if channel_name in output_groups:
                    raise CalibrationError(
                        "CFA_CHANNEL_GROUP_COLLISION",
                        f"colour channel {channel_name} of Bayer filter {filter_name!r} collides with "
                        f"filter or channel group {group_filter[channel_name]!r}; one run integrates one "
                        "Bayer filter and no mono R/G/B filters alongside it",
                    )
                output_groups[channel_name] = list(paths)
                group_filter[channel_name] = filter_name
                group_channel[channel_name] = channel
                group_cfa_pattern[channel_name] = pattern
        else:
            if pattern not in {"NONE", "UNKNOWN", "UNSPECIFIED", ""}:
                raise CalibrationError(
                    "CFA_PATTERN_UNSUPPORTED",
                    f"Bayer pattern {pattern!r} of filter {filter_name!r} is not supported "
                    f"(supported: {', '.join(sorted(CFA_PATTERNS))})",
                )
            light_cfa_pattern[filter_name] = None
            if filter_name in output_groups:
                raise CalibrationError(
                    "CFA_CHANNEL_GROUP_COLLISION",
                    f"filter {filter_name!r} collides with a colour channel group of Bayer filter "
                    f"{group_filter[filter_name]!r}",
                )
            output_groups[filter_name] = list(paths)
            group_filter[filter_name] = filter_name
            group_channel[filter_name] = None
            group_cfa_pattern[filter_name] = None
    return output_groups, group_filter, group_channel, group_cfa_pattern, light_cfa_pattern


def _group_darks(
    dark_info: Mapping[Path, FrameInfo],
    master_dark_info: Mapping[Path, FrameInfo],
    light_info: Mapping[Path, FrameInfo],
) -> tuple[dict[float, list[Path]], dict[float, Path]]:
    """Raw Dark groups and supplied MasterDarks by exposure; every Light
    needs an exact exposure match once any Dark is supplied."""

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
        if _find_dark(info.exposure_seconds, {value: Path() for value in dark_groups}) is not None:
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
    return dark_groups, supplied_darks


def _trusted_generated_coverage(
    trusted_generated: Mapping[str, Any] | None,
    *,
    has_raw_bias: bool,
    reference_bias: FrameInfo,
    dark_info: Mapping[Path, FrameInfo],
    dark_groups: Mapping[float, list[Path]],
    flat_info: Mapping[Path, FrameInfo],
    flat_groups: Mapping[str, list[Path]],
    parameters: PipelineParameters,
) -> tuple[
    _TrustedGeneratedMaster | None,
    dict[float, _TrustedGeneratedMaster],
    dict[str, _TrustedGeneratedMaster],
]:
    """Map E2E-generated masters onto the raw calibration groups they replace."""

    trusted_bias = trusted_generated["bias"] if trusted_generated is not None else None
    trusted_darks_by_exposure: dict[float, _TrustedGeneratedMaster] = {}
    trusted_flats_by_filter: dict[str, _TrustedGeneratedMaster] = {}
    if trusted_generated is None:
        return trusted_bias, trusted_darks_by_exposure, trusted_flats_by_filter
    workflow = parameters.calibration_workflow
    if has_raw_bias != (trusted_bias is not None):
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
        _assert_compatible(reference_bias, trusted_bias.frame_info, workflow=workflow)
    for exposure, item in trusted_darks_by_exposure.items():
        _assert_compatible(
            dark_info[dark_groups[exposure][0]],
            item.frame_info,
            compare_exposure=True,
            compare_temperature=True,
            temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
            workflow=workflow,
        )
    for filter_name, item in trusted_flats_by_filter.items():
        _assert_compatible(
            flat_info[flat_groups[filter_name][0]],
            item.frame_info,
            compare_filter=True,
            workflow=workflow,
        )
    return trusted_bias, trusted_darks_by_exposure, trusted_flats_by_filter


@dataclass(frozen=True)
class _StagingDirs:
    root: Path
    masters: Path
    calibrated: Path
    registered: Path
    coverage: Path
    previews: Path
    work: Path

    @classmethod
    def create(cls, root: Path) -> _StagingDirs:
        dirs = cls(
            root=root,
            masters=root / "masters",
            calibrated=root / "calibrated",
            registered=root / "registered",
            coverage=root / "coverage",
            previews=root / "previews",
            work=root / ".work",
        )
        for directory in (
            dirs.masters,
            dirs.calibrated,
            dirs.registered,
            dirs.coverage,
            dirs.previews,
            dirs.work,
        ):
            directory.mkdir()
        return dirs


@dataclass
class _RunLedger:
    """The evidence a run accumulates for its receipt, in production order."""

    staging: Path
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    stage_statistics: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        path: Path,
        kind: str,
        *,
        statistics: PixelStatistics | None = None,
        details: Mapping[str, Any] | None = None,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        record = _artifact_record(
            self.staging, path, kind, statistics=statistics, details=details, sha256=sha256
        )
        self.artifacts.append(record)
        return record


@dataclass(frozen=True)
class _CalibrationMasters:
    bias: Path | None
    darks: dict[float, Path]
    dark_domain_info: dict[float, FrameInfo]
    flats: dict[str, Path]
    flat_application_scales: dict[str, float]
    flat_pattern_scales: dict[str, tuple[float, float, float, float]]
    flat_channel_medians: dict[str, tuple[float, float, float]]


def _build_master_bias(
    plan: _RunPlan, dirs: _StagingDirs, parameters: PipelineParameters, ledger: _RunLedger
) -> Path | None:
    reference_bias = plan.reference_bias
    if plan.trusted_bias is not None:
        ledger.stage_statistics["masterBias"] = {
            "mode": "REUSED_E2E_GENERATED_MASTER",
            "sha256": plan.trusted_bias.sha256,
            "sizeBytes": plan.trusted_bias.size_bytes,
            "calibrationApplied": False,
            "doubleBiasSubtraction": False,
            "numericDomain": reference_bias.numeric_domain,
            "normalizedUnitScale": reference_bias.normalized_unit_scale,
        }
        return Path(plan.trusted_bias.path)
    if plan.biases:
        master_bias = dirs.masters / "master_bias.fits"
        bias_integration = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=_numeric_application_scale(
                        reference_bias,
                        plan.bias_info[path],
                        target_label="MasterBias reference",
                        additive_label="raw Bias",
                    ),
                )
                for path in plan.biases
            ),
            master_bias,
            metadata={
                "IMAGETYP": "Master Bias",
                "OAFSTATE": OUTPUT_STATE,
                "OAFBIAS": "MASTER",
                **cfa_metadata(reference_bias),
                **_numeric_domain_metadata(reference_bias),
            },
            parameters=parameters.integration,
            native_threads=None,
            durable=parameters.durable_intermediates,
        )
        ledger.record(
            master_bias,
            "MASTER_BIAS",
            statistics=bias_integration.statistics,
            sha256=bias_integration.output_sha256,
        )
        ledger.stage_statistics["masterBias"] = {
            "mode": "BUILT_FROM_RAW",
            "numericDomain": reference_bias.numeric_domain,
            "normalizedUnitScale": reference_bias.normalized_unit_scale,
            **_integration_record(bias_integration, dirs.root),
        }
        return master_bias
    if plan.master_biases:
        master_bias = plan.master_biases[0]
        _, master_bias_sha256, _ = _source_identity(
            master_bias, plan.source_aliases, plan.source_identity_cache
        )
        ledger.stage_statistics["masterBias"] = {
            "mode": "REUSED_SUPPLIED_MASTER",
            "path": str(plan.display_path(master_bias)),
            "sha256": master_bias_sha256,
            "calibrationApplied": False,
            "numericDomain": reference_bias.numeric_domain,
            "normalizedUnitScale": reference_bias.normalized_unit_scale,
        }
        return master_bias
    ledger.stage_statistics["masterBias"] = {"mode": "NOT_REQUIRED_DARK_INCLUDES_BIAS"}
    return None


def _build_master_darks(
    plan: _RunPlan, dirs: _StagingDirs, parameters: PipelineParameters, ledger: _RunLedger
) -> tuple[dict[float, Path], dict[float, FrameInfo]]:
    """Master darks by exposure (built, reused from the E2E run or supplied)
    and the frame metadata that stands for each one's numeric domain."""

    workflow = parameters.calibration_workflow
    master_darks: dict[float, Path] = {}
    domain_info: dict[float, FrameInfo] = {}
    for exposure, paths in sorted(plan.dark_groups.items()):
        dark_reference = plan.dark_info[paths[0]]
        for path in paths:
            _assert_compatible(plan.reference_bias, plan.dark_info[path], workflow=workflow)
            _assert_compatible(
                dark_reference,
                plan.dark_info[path],
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
                workflow=workflow,
            )
        key = f"masterDark:{exposure:.9g}"
        trusted_dark = plan.trusted_darks_by_exposure.get(exposure)
        if trusted_dark is not None:
            master_darks[exposure] = Path(trusted_dark.path)
            domain_info[exposure] = dark_reference
            ledger.stage_statistics[key] = {
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
            continue
        destination = dirs.masters / f"master_dark_{_exposure_token(exposure)}s.fits"
        integration = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=_numeric_application_scale(
                        dark_reference,
                        plan.dark_info[path],
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
                **cfa_metadata(dark_reference),
                **_numeric_domain_metadata(dark_reference),
            },
            parameters=parameters.integration,
            durable=parameters.durable_intermediates,
        )
        master_darks[exposure] = destination
        domain_info[exposure] = dark_reference
        ledger.record(
            destination,
            "MASTER_DARK",
            statistics=integration.statistics,
            sha256=integration.output_sha256,
            details={"exposureSeconds": exposure, "biasIncluded": True},
        )
        ledger.stage_statistics[key] = _integration_record(integration, dirs.root)
        ledger.stage_statistics[key]["mode"] = "BUILT_FROM_RAW"
        ledger.stage_statistics[key].update(
            {
                "numericDomain": dark_reference.numeric_domain,
                "normalizedUnitScale": dark_reference.normalized_unit_scale,
                "applicationScaleToRawDarkReference": 1.0,
            }
        )
    for exposure, supplied in sorted(plan.supplied_darks.items()):
        master_darks[exposure] = supplied
        domain_info[exposure] = plan.master_dark_info[supplied]
        _, supplied_sha256, _ = _source_identity(
            supplied, plan.source_aliases, plan.source_identity_cache
        )
        ledger.stage_statistics[f"masterDark:{exposure:.9g}"] = {
            "mode": "REUSED_SUPPLIED_MASTER",
            "path": str(plan.display_path(supplied)),
            "sha256": supplied_sha256,
            "calibrationApplied": False,
            "biasIncluded": plan.supplied_dark_bias_included[supplied],
            "numericDomain": plan.master_dark_info[supplied].numeric_domain,
            "normalizedUnitScale": plan.master_dark_info[supplied].normalized_unit_scale,
        }
    return master_darks, domain_info


def _build_master_flats(
    plan: _RunPlan,
    dirs: _StagingDirs,
    parameters: PipelineParameters,
    ledger: _RunLedger,
    *,
    master_bias: Path | None,
    master_darks: Mapping[float, Path],
    dark_domain_info: Mapping[float, FrameInfo],
) -> tuple[dict[str, Path], dict[str, float]]:
    """Master flats by filter and the factor each is divided with, so the
    division uses a response normalized to unity."""

    workflow = parameters.calibration_workflow
    reference_bias = plan.reference_bias
    master_flats: dict[str, Path] = {}
    application_scales: dict[str, float] = {}
    for filter_name, paths in sorted(plan.flat_groups.items()):
        key = f"masterFlat:{filter_name}"
        trusted_flat = plan.trusted_flats_by_filter.get(filter_name)
        if trusted_flat is not None:
            master_flats[filter_name] = Path(trusted_flat.path)
            application_scales[filter_name] = 1.0
            ledger.stage_statistics[key] = {
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
            info = plan.flat_info[path]
            _assert_compatible(
                reference_bias, info, compare_filter=False, compare_exposure=False,
                workflow=workflow,
            )
            if info.filter_name != filter_name:
                raise CalibrationError("FLAT_GROUP_INVALID", "internal filter grouping error")
            flat_dark_match = _find_dark(info.exposure_seconds, master_darks)
            if flat_dark_match is not None:
                dark_exposure, flat_subtract = flat_dark_match
                _assert_compatible(
                    info,
                    plan.dark_reference(dark_exposure),
                    compare_exposure=True,
                    compare_temperature=True,
                    temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
                    workflow=workflow,
                )
                dark_bias_included = plan.dark_bias_included(dark_exposure, flat_subtract)
                calibration_mode = (
                    "MATCHED_BIAS_INCLUDED_DARK"
                    if dark_bias_included
                    else "MATCHED_BIAS_SUBTRACTED_DARK_PLUS_MASTER_BIAS"
                )
                flat_subtract_info = dark_domain_info[dark_exposure]
            else:
                flat_subtract = master_bias
                dark_bias_included = True
                calibration_mode = "BIAS"
                flat_subtract_info = reference_bias
            flat_subtract_scale = _numeric_application_scale(
                info,
                flat_subtract_info,
                target_label="raw Flat",
                additive_label=("MasterDark" if flat_dark_match is not None else "MasterBias"),
            )
            flat_bias_scale = _numeric_application_scale(
                info,
                reference_bias,
                target_label="raw Flat",
                additive_label="MasterBias",
            )
            bias_terms = {
                "subtract_paths": (str(master_bias),) if not dark_bias_included else (),
                "subtract_scales": (flat_bias_scale,) if not dark_bias_included else (),
            }
            location = robust_location(
                FrameExpression(
                    source_path=str(path),
                    subtract_path=str(flat_subtract),
                    subtract_scale=flat_subtract_scale,
                    **bias_terms,
                ),
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
                    scale=1.0 / location,
                    **bias_terms,
                )
            )
            calibration_sources.append(
                {
                    "source": str(plan.display_path(path)),
                    "mode": calibration_mode,
                    "subtracted": plan.receipt_reference(dirs.root, flat_subtract),
                    "targetNumericDomain": info.numeric_domain,
                    "additiveNumericDomain": flat_subtract_info.numeric_domain,
                    "applicationScale": flat_subtract_scale,
                    "applicationScaleSource": "normalized-unit-domain-ratio",
                }
            )
        destination = dirs.masters / f"master_flat_{_safe_token(filter_name)}.fits"
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
                **cfa_metadata(plan.flat_info[paths[0]]),
            },
            parameters=parameters.integration,
            durable=parameters.durable_intermediates,
        )
        master_flats[filter_name] = destination
        application_scales[filter_name] = 1.0
        ledger.record(
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
        ledger.stage_statistics[key] = _integration_record(integration, dirs.root)
        ledger.stage_statistics[key]["mode"] = "BUILT_FROM_RAW"
    for filter_name, supplied in sorted(plan.supplied_flats.items()):
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
        application_scales[filter_name] = location
        _, supplied_sha256, _ = _source_identity(
            supplied, plan.source_aliases, plan.source_identity_cache
        )
        ledger.stage_statistics[f"masterFlat:{filter_name}"] = {
            "mode": "REUSED_SUPPLIED_MASTER",
            "path": str(plan.display_path(supplied)),
            "sha256": supplied_sha256,
            "calibrationApplied": False,
            "applicationNormalization": location,
        }
    return master_flats, application_scales


def _cfa_flat_scaling(
    plan: _RunPlan,
    master_flats: Mapping[str, Path],
    application_scales: Mapping[str, float],
    parameters: PipelineParameters,
    ledger: _RunLedger,
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, tuple[float, float, float]]]:
    """Separate flat scaling factors for the colour channels of a Bayer
    filter: each channel is divided by the master flat normalized to its own
    channel median, so the flat's colour response does not tint the
    calibrated frame (PixInsight's "separate CFA flat scaling factors")."""

    pattern_scales: dict[str, tuple[float, float, float, float]] = {}
    channel_medians_by_filter: dict[str, tuple[float, float, float]] = {}
    for filter_name, pattern in plan.light_cfa_pattern.items():
        if pattern is None:
            continue
        with FitsFrame(master_flats[filter_name]) as flat_frame:
            medians = channel_medians(flat_frame.full_values(), pattern)
        if any(not math.isfinite(value) or value <= parameters.integration.division_floor for value in medians):
            raise CalibrationError(
                "FLAT_SIGNAL_INVALID",
                f"the master flat of Bayer filter {filter_name!r} has a colour channel without positive signal",
                path=str(master_flats[filter_name]),
            )
        channel_medians_by_filter[filter_name] = medians
        layout = CFA_PATTERNS[pattern]
        reference_level = application_scales[filter_name]
        pattern_scales[filter_name] = tuple(  # type: ignore[assignment]
            medians[layout[position]] / reference_level for position in range(4)
        )
        ledger.stage_statistics[f"masterFlat:{filter_name}"]["cfaChannelMedians"] = {
            "pattern": pattern,
            "R": medians[0],
            "G": medians[1],
            "B": medians[2],
            "separateChannelScaling": True,
        }
    return pattern_scales, channel_medians_by_filter


def _build_calibration_masters(
    plan: _RunPlan, dirs: _StagingDirs, parameters: PipelineParameters, ledger: _RunLedger
) -> _CalibrationMasters:
    master_bias = _build_master_bias(plan, dirs, parameters, ledger)
    master_darks, dark_domain_info = _build_master_darks(plan, dirs, parameters, ledger)
    master_flats, application_scales = _build_master_flats(
        plan,
        dirs,
        parameters,
        ledger,
        master_bias=master_bias,
        master_darks=master_darks,
        dark_domain_info=dark_domain_info,
    )
    pattern_scales, channel_medians_by_filter = _cfa_flat_scaling(
        plan, master_flats, application_scales, parameters, ledger
    )
    return _CalibrationMasters(
        bias=master_bias,
        darks=master_darks,
        dark_domain_info=dark_domain_info,
        flats=master_flats,
        flat_application_scales=application_scales,
        flat_pattern_scales=pattern_scales,
        flat_channel_medians=channel_medians_by_filter,
    )


def _plan_light_jobs(
    plan: _RunPlan,
    dirs: _StagingDirs,
    parameters: PipelineParameters,
    masters: _CalibrationMasters,
) -> tuple[list[_LightJob], list[dict[str, Any]]]:
    """One calibrate-and-register job per Light, and the calibration
    details its receipt entry reports."""

    workflow = parameters.calibration_workflow
    reference_bias = plan.reference_bias
    groups_of_filter: dict[str, list[str]] = {}
    for group_name, source_filter in plan.group_filter.items():
        groups_of_filter.setdefault(source_filter, []).append(group_name)
    light_jobs: list[_LightJob] = []
    calibrated_details: list[dict[str, Any]] = []
    for index, path in enumerate(plan.lights, start=1):
        info = plan.light_info[path]
        filter_name = info.filter_name
        flat_path = masters.flats[filter_name]
        flat_reference = (
            plan.flat_info[plan.flat_groups[filter_name][0]]
            if filter_name in plan.flat_groups
            else plan.master_flat_info[plan.supplied_flats[filter_name]]
        )
        _assert_compatible(info, flat_reference, compare_filter=True, workflow=workflow)
        dark_match = _find_dark(info.exposure_seconds, masters.darks)
        if dark_match is not None:
            dark_exposure, subtract_path = dark_match
            _assert_compatible(
                info,
                plan.dark_reference(dark_exposure),
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
                workflow=workflow,
            )
            dark_bias_included = plan.dark_bias_included(dark_exposure, subtract_path)
            bias_mode = (
                "INCLUDED_IN_MASTER_DARK"
                if dark_bias_included
                else "MASTER_BIAS_AND_BIAS_SUBTRACTED_DARK"
            )
            subtract_info = masters.dark_domain_info[dark_exposure]
        else:
            subtract_path = masters.bias
            dark_bias_included = True
            bias_mode = "MASTER_BIAS_SUBTRACTED"
            subtract_info = reference_bias
        subtract_scale = _numeric_application_scale(
            info,
            subtract_info,
            target_label="raw Light",
            additive_label=("MasterDark" if dark_match is not None else "MasterBias"),
        )
        bias_scale = _numeric_application_scale(
            info,
            reference_bias,
            target_label="raw Light",
            additive_label="MasterBias",
        )
        light_output_domain = plan.light_domain_references[filter_name]
        light_domain_scale = _numeric_application_scale(
            light_output_domain,
            info,
            target_label="filter integration domain",
            additive_label="raw Light",
        )
        reference_exposure = plan.reference_exposures[filter_name]
        exposure_scale = reference_exposure / float(info.exposure_seconds)
        stem = light_stem(path)
        cfa_pattern = plan.light_cfa_pattern[filter_name]
        destinations = tuple(
            _ChannelDestination(
                group=group_name,
                channel=plan.group_channel[group_name],
                path=(
                    dirs.registered / f"{index:05d}_{stem}.fits"
                    if plan.group_channel[group_name] is None
                    else dirs.registered / f"{index:05d}_{stem}_{_safe_token(group_name)}.fits"
                ),
            )
            for group_name in groups_of_filter[filter_name]
        )
        expression = FrameExpression(
            source_path=str(path),
            subtract_path=str(subtract_path),
            subtract_scale=subtract_scale,
            subtract_paths=(str(masters.bias),) if not dark_bias_included else (),
            subtract_scales=(bias_scale,) if not dark_bias_included else (),
            divide_path=str(flat_path),
            scale=(
                masters.flat_application_scales[filter_name]
                * reference_exposure
                / float(info.exposure_seconds)
                * light_domain_scale
            ),
            pattern_scales=masters.flat_pattern_scales.get(filter_name, ()),
        )
        calibrated_metadata = {
            "IMAGETYP": "Calibrated Light",
            "FILTER": filter_name,
            "OBJECT": info.target,
            "EXPTIME": reference_exposure,
            "OAFSRCEX": info.exposure_seconds,
            "OAFEXPSC": exposure_scale,
            "OAFSTATE": OUTPUT_STATE,
            "OAFBIAS": bias_mode,
            **({"BAYERPAT": cfa_pattern, "OAFCFA": cfa_pattern} if cfa_pattern else {}),
            **_numeric_domain_metadata(light_output_domain),
        }
        calibrated_details.append(
            {
                "source": str(plan.display_path(path)),
                "filter": filter_name,
                "subtractedMaster": plan.receipt_reference(dirs.root, subtract_path),
                "biasMode": bias_mode,
                "sourceNumericDomain": info.numeric_domain,
                "additiveNumericDomain": subtract_info.numeric_domain,
                "additiveApplicationScale": subtract_scale,
                "additiveApplicationScaleSource": "normalized-unit-domain-ratio",
                "biasApplicationScale": bias_scale if not dark_bias_included else None,
                "outputNumericDomain": light_output_domain.numeric_domain,
                "sourceToOutputDomainScale": light_domain_scale,
                "dividedMasterFlat": plan.receipt_reference(dirs.root, flat_path),
                "flatApplicationNormalization": masters.flat_application_scales[filter_name],
                **(
                    {
                        "cfaPattern": cfa_pattern,
                        "cfaFlatChannelMedians": list(masters.flat_channel_medians[filter_name]),
                        "cfaFlatPatternScales": list(masters.flat_pattern_scales[filter_name]),
                    }
                    if cfa_pattern
                    else {}
                ),
                "exposureNormalization": {
                    "sourceSeconds": info.exposure_seconds,
                    "referenceSeconds": reference_exposure,
                    "scale": exposure_scale,
                },
            }
        )
        light_jobs.append(
            _LightJob(
                source_path=path,
                expression=expression,
                calibrated_path=(
                    dirs.calibrated / f"{index:05d}_{stem}.fits"
                    if parameters.materialize_calibrated_lights
                    else None
                ),
                calibrated_metadata=calibrated_metadata,
                destinations=destinations,
                transform=plan.transforms[path],
                info=replace(
                    info,
                    exposure_seconds=reference_exposure,
                    numeric_domain=light_output_domain.numeric_domain,
                    normalized_unit_scale=light_output_domain.normalized_unit_scale,
                ),
                source_exposure_seconds=info.exposure_seconds,
                hot_pixel_master=(
                    str(subtract_path)
                    if dark_match is not None and parameters.cosmetic_hot_pixel_sigma is not None
                    else None
                ),
                hot_pixel_sigma=parameters.cosmetic_hot_pixel_sigma,
                cfa_pattern=cfa_pattern,
            )
        )
    return light_jobs, calibrated_details


@dataclass(frozen=True)
class _RegisteredLights:
    by_group: dict[tuple[Path, str], Path]
    calibrated: dict[Path, Path]
    records: dict[str, Any]
    execution: dict[str, Any]
    wall_seconds: float


def _calibrate_and_register_lights(
    plan: _RunPlan,
    dirs: _StagingDirs,
    jobs: Sequence[_LightJob],
    calibrated_details: Sequence[dict[str, Any]],
    parameters: PipelineParameters,
    execution_tuning: Any,
    ledger: _RunLedger,
) -> _RegisteredLights:
    """Calibrate in memory and warp within one shared registration budget.
    Source-identity caches and receipt construction stay on this thread."""

    master_cache = _MasterCache()
    started = time.perf_counter()
    light_results, fused_execution = _calibrate_and_register_frames(
        jobs,
        master_cache=master_cache,
        max_memory_bytes=parameters.registration_memory_bytes,
        resampler=parameters.registration_resampler,
        cpu_workers=execution_tuning.cpu_workers,
        kernel_threads=execution_tuning.kernel_threads,
        division_floor=parameters.integration.division_floor,
        durable=parameters.durable_intermediates,
    )
    wall_seconds = time.perf_counter() - started
    del master_cache
    registered: dict[tuple[Path, str], Path] = {}
    calibrated: dict[Path, Path] = {}
    records: dict[str, Any] = {}
    for path, job, details, result in zip(
        plan.lights, jobs, calibrated_details, light_results, strict=True,
    ):
        transform = job.transform
        resampling = _registration_provenance(
            transform, job.info.shape, parameters.registration_resampler
        )
        if job.calibrated_path is not None:
            calibrated[path] = job.calibrated_path
            ledger.record(
                job.calibrated_path,
                "CALIBRATED_LIGHT",
                statistics=result.calibrated_statistics,
                details=details,
                sha256=result.calibrated_sha256,
            )
        for registered_output in result.registered:
            registered[(path, registered_output.group)] = registered_output.path
            ledger.record(
                registered_output.path,
                "REGISTERED_LIGHT",
                statistics=registered_output.statistics,
                details={
                    "source": str(plan.display_path(path)),
                    "group": registered_output.group,
                    **(
                        {"cfaChannel": CHANNEL_NAMES[registered_output.channel], "cfaPattern": job.cfa_pattern}
                        if registered_output.channel is not None
                        else {}
                    ),
                    "transformInputToOutput": transform.serializable(),
                    **resampling,
                    "warpBackend": registered_output.execution.get("warpBackend"),
                    "warpKernel": registered_output.execution.get("warpKernel"),
                },
                sha256=registered_output.sha256,
            )
        records[str(plan.display_path(path))] = {
            "transformInputToOutput": transform.serializable(),
            "identity": transform.is_identity,
            **resampling,
            "qualityWeight": plan.quality_weights[path],
            "calibration": {
                "materialized": job.calibrated_path is not None,
                "statistics": result.calibrated_statistics.serializable(),
                "cosmetic": result.cosmetic or {"algorithm": None, "replacedPixels": 0},
                **({"debayer": result.debayer} if result.debayer else {}),
                **details,
            },
            "outputs": {
                registered_output.group: {
                    "path": str(registered_output.path.relative_to(dirs.root)),
                    "channel": CHANNEL_NAMES[registered_output.channel] if registered_output.channel is not None else None,
                    "statistics": registered_output.statistics.serializable(),
                }
                for registered_output in result.registered
            },
            "execution": dict(result.execution),
        }
    if not parameters.materialize_calibrated_lights:
        try:
            dirs.calibrated.rmdir()
        except OSError:
            # Removing this unused staging directory is best effort.  Keep it
            # for diagnostics if cleanup fails; artifact and final-publication
            # validation still run independently.
            pass
    execution = {
        "executor": fused_execution["executor"],
        "executionModel": fused_execution["executionModel"],
        "configuredCpuWorkers": execution_tuning.cpu_workers,
        "cpuWorkersUsed": fused_execution["cpuWorkersUsed"],
        "nativeThreadsPerWorker": fused_execution["nativeThreadsPerWorker"],
        "tailNativeThreads": fused_execution["tailNativeThreads"],
        "tailLights": fused_execution["tailLights"],
        "frameCount": len(jobs),
        "totalMemoryBudgetBytes": parameters.registration_memory_bytes,
        "perWorkerMemoryBudgetBytes": fused_execution["perWorkerMemoryBudgetBytes"],
        "warpBackends": fused_execution["warpBackends"],
        "calibratedLightsMaterialized": parameters.materialize_calibrated_lights,
        "masterCacheBytes": fused_execution["masterCacheBytes"],
        "wallSeconds": wall_seconds,
    }
    return _RegisteredLights(registered, calibrated, records, execution, wall_seconds)


class _MetalSession:
    """The run's Metal executor, dropped for the rest of the run once Metal
    rejects a group (the CPU reference then integrates the remaining ones)."""

    def __init__(self) -> None:
        self.executor: NativeMetalExecutor | None = None
        self.unavailable_reason: str | None = None

    def open(self, parameters: PipelineParameters) -> None:
        try:
            self.executor = NativeMetalExecutor(
                library_path=parameters.native_library_path,
                metal_source_path=parameters.metal_source_path,
            )
        except MetalIntegrationError as error:
            self.unavailable_reason = str(error)

    def observe(self, execution: Mapping[str, Any]) -> None:
        fallback_reason = str(execution.get("fallbackReason") or "")
        if (
            self.executor is not None
            and execution.get("selectedBackend") == "portable-cpu"
            and fallback_reason.startswith("Metal execution rejected:")
        ):
            self.executor.close()
            self.executor = None
            self.unavailable_reason = fallback_reason

    def close(self) -> None:
        if self.executor is not None:
            self.executor.close()


class _NormalizationFits:
    """Global-normalization fits of the output groups.

    The fit of a group is a pure function of its registered frames, hints and
    transforms, so the next group's fit runs on one helper thread while this
    group integrates: the fit is mostly Python-level work whose gaps and the
    integration's I/O and Python phases overlap.  The coefficients are
    identical either way.
    """

    def __init__(
        self,
        plan: _RunPlan,
        registered: Mapping[tuple[Path, str], Path],
        parameters: PipelineParameters,
        execution_tuning: Any,
        ordered_groups: Sequence[tuple[str, list[Path]]],
        *,
        prefetch: bool = True,
    ) -> None:
        self._plan = plan
        self._registered = registered
        self._parameters = parameters
        self._cpu_workers = execution_tuning.cpu_workers
        # A fit that overlaps another group's integration gets a third of the
        # cores: the integration's kernels keep the rest, and the fit's
        # Python-level work does not scale past a few threads anyway.
        self._prefetch_workers = max(2, execution_tuning.cpu_workers // 3)
        self._ordered_groups = ordered_groups
        self._prefetch = prefetch and parameters.global_normalization.enabled and len(ordered_groups) > 1
        self._pool: ThreadPoolExecutor | None = None
        self._prefetched: dict[str, Any] = {}

    def _job(self, group_name: str, group_paths: list[Path], fit_workers: int) -> Callable[[], Any]:
        plan = self._plan
        group_registered = [self._registered[(path, group_name)] for path in group_paths]
        group_reference_index, _selection = _normalization_reference_index(
            group_paths, plan.stellar_scale_hints, plan.quality_weights
        )
        hints: list[StellarScaleHint | None] = []
        expected = group_paths[group_reference_index]
        registered_reference_path = group_registered[group_reference_index]
        for source_path, registered_path in zip(group_paths, group_registered, strict=True):
            hint = plan.stellar_scale_hints[source_path]
            if hint is None:
                hints.append(None)
                continue
            hinted_reference = Path(hint.reference_path).expanduser().resolve(strict=True)
            if hinted_reference != expected:
                raise CalibrationError(
                    "STELLAR_SCALE_HINT_REFERENCE_MISMATCH",
                    "stellar scale reference differs from the integration-quality reference",
                    path=str(source_path),
                )
            hints.append(
                replace(hint, source_path=str(registered_path), reference_path=str(registered_reference_path))
            )
        transforms = [plan.transforms[path].validated_matrix() for path in group_paths]
        parameters = self._parameters.global_normalization

        def run() -> tuple[Any, list[StellarScaleHint | None], float]:
            started = time.perf_counter()
            result = fit_registered_group_global_normalization(
                [str(path) for path in group_registered],
                reference_index=group_reference_index,
                parameters=parameters,
                stellar_scale_hints=hints,
                workers=fit_workers,
                transforms=transforms,
            )
            return result, hints, time.perf_counter() - started

        return run

    def prefetch_after(self, position: int) -> None:
        """Start the fit of the group after ``position`` on the helper thread."""

        if not self._prefetch or position + 1 >= len(self._ordered_groups):
            return
        next_name, next_paths = self._ordered_groups[position + 1]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ufwbpp-normalize-next")
        self._prefetched[next_name] = self._pool.submit(
            self._job(next_name, next_paths, self._prefetch_workers)
        )

    def prefetch_all(self, order: Sequence[int], *, concurrent: int) -> None:
        """Start the fit of every group, in ``order``, on ``concurrent``
        helper threads, so each group's integration waits only for its own
        fit while the fits of later groups overlap earlier integrations."""

        if not self._parameters.global_normalization.enabled:
            return
        self._pool = ThreadPoolExecutor(max_workers=concurrent, thread_name_prefix="ufwbpp-normalize")
        workers = max(2, self._cpu_workers // concurrent)
        for position in order:
            name, paths = self._ordered_groups[position]
            self._prefetched[name] = self._pool.submit(self._job(name, paths, workers))

    def fit(self, group_name: str, paths: list[Path]) -> tuple[Any, list[StellarScaleHint | None], float, bool]:
        """The fit of ``group_name``: its result, hints, fit seconds and
        whether it was prefetched."""

        future = self._prefetched.pop(group_name, None)
        if future is not None:
            return (*future.result(), True)
        return (*self._job(group_name, list(paths), self._cpu_workers)(), False)

    def shutdown(self, *, cancel: bool = False) -> None:
        if self._pool is not None:
            # A fit still running for a later group must finish before the
            # staging tree it reads is removed.
            if cancel:
                self._pool.shutdown(wait=True, cancel_futures=True)
            else:
                self._pool.shutdown(wait=True)


@dataclass(frozen=True)
class _GroupProducts:
    master_light: Path
    preview: Path
    drizzle: DrizzleGroupInputs | None
    record: dict[str, Any]
    timing: dict[str, float]
    proper_coadd: Path | None = None


def _public_normalization_evidence(
    plan: _RunPlan, paths: Sequence[Path], reference_index: int, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """The fit's receipt with frames named by their public source paths."""

    public_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(receipt["frames"]):
        public_frame = {**dict(frame), "source": str(plan.display_path(paths[index]))}
        frame_evidence = dict(public_frame["evidence"])
        stellar = frame_evidence.get("stellarScale")
        if isinstance(stellar, dict):
            stellar = dict(stellar)
            stellar["source"] = str(plan.display_path(paths[index]))
            stellar["reference"] = str(plan.display_path(paths[reference_index]))
            frame_evidence["stellarScale"] = stellar
        public_frame["evidence"] = frame_evidence
        public_frames.append(public_frame)
    return {**dict(receipt), "frames": public_frames}


def _with_region_weights(
    plan: _RunPlan, paths: Sequence[Path], expressions: list[FrameExpression]
) -> tuple[list[FrameExpression], list[dict[str, Any]]]:
    """Attach each Light's selection region weight map to its expression."""

    attached: list[FrameExpression] = []
    mapped: list[dict[str, Any]] = []
    for path, expression in zip(paths, expressions, strict=True):
        region_map = plan.region_weight_maps.get(path)
        if region_map is None:
            attached.append(expression)
            continue
        height, width = plan.light_info[path].shape
        x_nodes, y_nodes = region_map.pixel_nodes(height, width)
        attached.append(
            replace(
                expression,
                weight_grid=tuple(tuple(float(value) for value in row) for row in region_map.nodes),
                weight_grid_x=tuple(float(value) for value in x_nodes),
                weight_grid_y=tuple(float(value) for value in y_nodes),
            )
        )
        mapped.append(
            {
                "path": str(plan.display_path(path)),
                "frame": str(region_map.evidence.get("frame", "qc-reference")),
                "zeroFraction": float(region_map.zero_fraction),
                "minimumWeight": float(region_map.minimum_weight),
                "meanWeight": float(region_map.mean_weight),
            }
        )
    return attached, mapped


# Cores per concurrently integrated group.  A group's integration keeps
# about five cores busy on average (serial reads, writes and Python-level
# work between its multithreaded kernels), so on larger machines two groups
# side by side fill the idle cores.
_CORES_PER_CONCURRENT_GROUP = 6


def _group_concurrency(parameters: PipelineParameters, tuning: Any, groups: int) -> int:
    """How many output groups integrate at the same time.

    Only the CPU integration runs groups side by side (the Metal executor is
    one per run); the masters never depend on the count, which only changes
    the schedule.
    """

    backend = parameters.ordinary_integration_backend
    cpu = backend == "portable-cpu" or (backend == "auto" and load_native_kernels() is not None)
    if not cpu or groups < 2:
        return 1
    return max(1, min(groups, int(tuning.cpu_workers) // _CORES_PER_CONCURRENT_GROUP))


def _integrate_group(
    plan: _RunPlan,
    dirs: _StagingDirs,
    group_name: str,
    paths: list[Path],
    *,
    parameters: PipelineParameters,
    lights: _RegisteredLights,
    fits: _NormalizationFits,
    metal: _MetalSession,
    hardware_profile: Any,
    execution_tuning: Any,
    shared_crop: tuple[int, int, int, int] | None,
    group_crops: Mapping[str, Any],
    tile_observer: Callable[[Any], None] | None,
    ledger: _RunLedger,
) -> _GroupProducts:
    """Normalize, reject and integrate one output group, then crop the master
    and its maps to the run's common rectangle and render its preview.

    ``group_name`` names the output group (a filter, or a colour channel of a
    Bayer filter); ``source_filter`` is the Lights' own filter, which owns the
    flats, exposures and numeric domain.
    """

    source_filter = plan.group_filter[group_name]
    cfa_channel = plan.group_channel[group_name]
    cfa_pattern = plan.group_cfa_pattern[group_name]
    group_cfa_metadata = (
        {
            "OAFCFA": cfa_pattern,
            "OAFCFACH": CHANNEL_NAMES[cfa_channel],
            "OAFCFAF": source_filter,
        }
        if cfa_channel is not None
        else {}
    )
    domain_metadata = _numeric_domain_metadata(plan.light_domain_references[source_filter])
    timing: dict[str, float] = {}
    group_started = time.perf_counter()
    exposures = {
        info.exposure_seconds for path, info in plan.light_info.items() if path in paths
    }
    reference_exposure = plan.reference_exposures[source_filter]
    total_exposure = sum(float(plan.light_info[path].exposure_seconds) for path in paths)
    registered_paths = [lights.by_group[(path, group_name)] for path in paths]
    reference_index, reference_selection = _normalization_reference_index(
        paths, plan.stellar_scale_hints, plan.quality_weights
    )
    expressions = [FrameExpression(str(path)) for path in registered_paths]
    normalization_record: dict[str, Any] = {
        "status": "DISABLED",
        "parameters": parameters.global_normalization.serializable(),
    }
    normalization_method = "NONE"
    if parameters.global_normalization.enabled:
        normalization_started = time.perf_counter()
        global_result, _hints, fit_seconds, prefetched = fits.fit(group_name, paths)
        timing["normalizationFit"] = fit_seconds
        timing["normalizationWait"] = time.perf_counter() - normalization_started
        timing["normalizationPrefetched"] = float(prefetched)
        expressions = [
            FrameExpression(
                str(path),
                scale=coefficient.scale,
                offset=coefficient.offset,
                offset_grid=coefficient.offset_grid,
                offset_grid_x=coefficient.offset_grid_x,
                offset_grid_y=coefficient.offset_grid_y,
            )
            for path, coefficient in zip(registered_paths, global_result.coefficients, strict=True)
        ]
        normalization_record = {
            "status": "APPLIED",
            "referenceInput": str(plan.display_path(paths[reference_index])),
            "referenceSelection": reference_selection,
            "evidence": _public_normalization_evidence(plan, paths, reference_index, global_result.receipt),
        }
        normalization_method = "GLOBAL_STELLAR"
    token = _safe_token(group_name)
    full_master = dirs.work / f"integrated_{token}.fits"
    full_maps = IntegrationMapPaths(
        accepted_count=dirs.work / f"accepted_count_{token}.fits",
        coverage=dirs.work / f"coverage_{token}.fits",
        rejection_count=dirs.work / f"rejection_count_{token}.fits",
    )
    timing["normalization"] = time.perf_counter() - group_started
    region_mapped_lights: list[dict[str, Any]] = []
    if plan.region_weight_maps:
        expressions, region_mapped_lights = _with_region_weights(plan, paths, expressions)
    integration_started = time.perf_counter()
    proper = parameters.proper_coaddition
    reuse_rejection = proper.enabled and proper.outlier_handling == "reuse-rejection"
    mask_recorder = (
        _RejectionMaskRecorder(len(paths), plan.light_info[paths[0]].shape)
        if parameters.capture_drizzle_inputs or reuse_rejection
        else None
    )
    if parameters.capture_drizzle_inputs and any(
        path not in lights.calibrated for path in paths
    ):
        raise CalibrationError(
            "DRIZZLE_INPUTS_UNAVAILABLE",
            "drizzle inputs need materialized calibrated Lights",
        )
    master_metadata = {
        "IMAGETYP": "Master Light",
        "FILTER": group_name,
        "OAFSTATE": OUTPUT_STATE,
        "OAFWCS": "UNSOLVED",
        "EXPTIME": reference_exposure,
        "OAFINTTM": total_exposure,
        "OAFNORM": normalization_method,
        **group_cfa_metadata,
        **domain_metadata,
    }
    integration = integrate_registered_group(
        expressions,
        full_master,
        metadata=master_metadata,
        parameters=parameters.integration,
        requested_backend=parameters.ordinary_integration_backend,
        native_library_path=parameters.native_library_path,
        metal_source_path=parameters.metal_source_path,
        metal_executor=metal.executor,
        metal_unavailable_reason=metal.unavailable_reason,
        hardware=hardware_profile,
        tuning=execution_tuning,
        quality_weights=[plan.quality_weights[path] for path in paths],
        map_paths=full_maps,
        durable=parameters.durable_intermediates,
        tile_observer=_compose_tile_observers(tile_observer, mask_recorder),
    )
    drizzle = None
    if parameters.capture_drizzle_inputs and mask_recorder is not None:
        drizzle = DrizzleGroupInputs(
            filter_name=group_name,
            cfa_pattern=cfa_pattern,
            channel=cfa_channel,
            frames=tuple(
                DrizzleFrame(
                    calibrated_path=str(lights.calibrated[path]),
                    source_path=str(plan.display_path(path)),
                    input_to_reference=tuple(
                        tuple(float(value) for value in row)
                        for row in plan.transforms[path].validated_matrix()
                    ),
                    weight=float(weight),
                    exposure_seconds=float(plan.light_info[path].exposure_seconds),
                    normalization_scale=float(expression.scale),
                    normalization_offset=float(expression.offset),
                    offset_grid=expression.offset_grid,
                    offset_grid_x=expression.offset_grid_x,
                    offset_grid_y=expression.offset_grid_y,
                    weight_grid=expression.weight_grid,
                    weight_grid_x=expression.weight_grid_x,
                    weight_grid_y=expression.weight_grid_y,
                    accepted_mask_bits=bits,
                )
                for path, expression, weight, bits in zip(
                    paths, expressions, integration.weights, mask_recorder.bits, strict=True,
                )
            ),
            reference_shape=plan.light_info[paths[0]].shape,
            metadata={
                "IMAGETYP": "Master Light",
                "FILTER": group_name,
                "EXPTIME": reference_exposure,
                "OAFINTTM": total_exposure,
                "OAFSTATE": OUTPUT_STATE,
                "OAFWCS": "UNSOLVED",
                "OAFNORM": normalization_method,
                **group_cfa_metadata,
                **domain_metadata,
            },
        )
    metal.observe(integration.execution)
    if shared_crop is not None:
        if integration.shape != plan.light_info[paths[0]].shape:
            raise CalibrationError(
                "REGISTRATION_GEOMETRY_MISMATCH",
                f"{group_name} integrated {integration.shape} but its "
                f"Lights were registered as {plan.light_info[paths[0]].shape}",
            )
        crop = shared_crop
    else:
        height, width = integration.shape
        crop = (0, 0, height, width)
    top, left, bottom, right = crop
    crop_fraction = ((bottom - top) * (right - left)) / (integration.shape[0] * integration.shape[1])
    if crop_fraction < parameters.minimum_crop_fraction:
        raise CalibrationError(
            "AUTOCROP_TOO_SMALL",
            f"common crop retains only {crop_fraction:.3%} of the frame",
        )
    timing["integration"] = time.perf_counter() - integration_started
    crop_write_started = time.perf_counter()
    master_light = dirs.masters / f"master_light_{token}.fits"
    master_stats, master_sha256 = _crop_fits(
        full_master,
        master_light,
        crop,
        {
            "IMAGETYP": "Master Light",
            "FILTER": group_name,
            "EXPTIME": reference_exposure,
            "OAFINTTM": total_exposure,
            "OAFSTATE": OUTPUT_STATE,
            "OAFWCS": "UNSOLVED",
            "OAFCROP": "AUTO" if parameters.auto_crop else "NONE",
            "OAFNFRM": len(paths),
            "OAFNORM": normalization_method,
            **group_cfa_metadata,
            **domain_metadata,
        },
        max_memory_bytes=parameters.integration.max_memory_bytes,
        durable=parameters.durable_intermediates,
    )
    ledger.record(
        master_light,
        "MASTER_LIGHT_LINEAR_UNSOLVED",
        statistics=master_stats,
        sha256=master_sha256,
        details={
            "filter": group_name,
            "crop": {
                "top": top,
                "left": left,
                "bottomExclusive": bottom,
                "rightExclusive": right,
                "retainedFraction": crop_fraction,
            },
        },
    )
    proper_record: dict[str, Any] | None = None
    proper_light: Path | None = None
    if proper.enabled:
        proper_started = time.perf_counter()
        proper_full = dirs.work / f"proper_{token}.fits"
        # After global normalization every frame carries the reference's
        # photometric scale, so the model's per-frame flux scale is the same
        # constant for all of them: the transparency difference has moved
        # into each frame's own background sigma, which is exactly where the
        # weight F_j / sigma_j^2 needs it.  Without normalization there is no
        # measured transparency to use and equal flux scales are assumed.
        flux_scale_source = (
            "normalized-to-reference"
            if parameters.global_normalization.enabled
            else "unit-assumed-equal-transparency"
        )
        if reuse_rejection and (mask_recorder is None or not mask_recorder.complete):
            raise CalibrationError(
                "PROPER_COADD_REJECTION_UNAVAILABLE",
                f"{group_name}: the integration did not report an accepted-sample mask for every row",
            )
        proper_result = proper_coadd_group(
            expressions,
            proper_full,
            master_path=full_master,
            shape=integration.shape,
            flux_scales=[1.0] * len(expressions),
            accepted_bits=mask_recorder.bits if mask_recorder is not None else None,
            metadata={
                "IMAGETYP": "Master Light Proper Coadd",
                "FILTER": group_name,
                "OAFSTATE": OUTPUT_STATE,
                "OAFWCS": "UNSOLVED",
                "EXPTIME": reference_exposure,
                "OAFINTTM": total_exposure,
                "OAFNORM": normalization_method,
                **group_cfa_metadata,
                **domain_metadata,
            },
            parameters=proper,
            division_floor=parameters.integration.division_floor,
            max_memory_bytes=parameters.integration.max_memory_bytes,
            workers=max(1, execution_tuning.cpu_workers),
            durable=False,
        )
        proper_light = dirs.masters / f"proper_light_{token}.fits"
        proper_statistics, proper_sha256 = _crop_fits(
            proper_full,
            proper_light,
            crop,
            {
                "IMAGETYP": "Master Light Proper Coadd",
                "FILTER": group_name,
                "EXPTIME": reference_exposure,
                "OAFINTTM": total_exposure,
                "OAFSTATE": OUTPUT_STATE,
                "OAFWCS": "UNSOLVED",
                "OAFCROP": "AUTO" if parameters.auto_crop else "NONE",
                "OAFNFRM": len(paths),
                "OAFNORM": normalization_method,
                "OAFPCOAD": PROPER_COADD_ALGORITHM_ID,
                "OAFPCFR": proper_result.flux_scale_norm,
                "OAFPCFWH": proper_result.coadd_fwhm_pixels,
                "OAFPCSKY": proper_result.sky_added,
                "OAFPCAPO": proper.apodization_pixels,
                "OAFPCREP": proper_result.replaced_samples,
                "OAFPCOUT": proper.outlier_handling,
                **group_cfa_metadata,
                **domain_metadata,
            },
            max_memory_bytes=parameters.integration.max_memory_bytes,
            durable=parameters.durable_intermediates,
        )
        ledger.record(
            proper_light,
            "MASTER_LIGHT_PROPER_COADD_UNSOLVED",
            statistics=proper_statistics,
            sha256=proper_sha256,
            details={"filter": group_name, "algorithm": PROPER_COADD_ALGORITHM_ID},
        )
        remove_file(proper_full)
        proper_record = {
            **proper_result.serializable(),
            "outputPath": str(proper_light.relative_to(dirs.root)),
            "uncroppedSha256": proper_result.output_sha256,
            "croppedSha256": proper_sha256,
            "fluxScaleSource": flux_scale_source,
            # Region weight maps scale samples in the ordinary weighted mean;
            # the transform has no per-sample weight, so a group that uses
            # them coadds unweighted and the receipt says so.
            "regionWeightMapsApplied": False,
            "regionWeightMapFrames": len(region_mapped_lights),
            "statistics": proper_statistics.serializable(),
            "primaryProduct": False,
            "ordinaryMasterUnaffected": True,
        }
        timing["properCoaddition"] = time.perf_counter() - proper_started
    cropped_maps: dict[str, Path] = {}
    map_statistics: dict[str, Any] = {}
    for map_name, artifact_kind in (
        ("acceptedSampleCount", "INTEGRATION_ACCEPTED_COUNT"),
        ("coverageFraction", "INTEGRATION_COVERAGE"),
        ("rejectionCount", "INTEGRATION_REJECTION_COUNT"),
    ):
        destination = dirs.coverage / f"{token}_{map_name}.fits"
        statistics, map_sha256 = _crop_fits(
            Path(integration.map_paths[map_name]),
            destination,
            crop,
            {
                "IMAGETYP": artifact_kind.replace("_", " ").title(),
                "FILTER": group_name,
                "OAFSTATE": OUTPUT_STATE,
                "OAFMAP": map_name.upper(),
                "OAFNFRM": len(paths),
            },
            max_memory_bytes=parameters.integration.max_memory_bytes,
            durable=parameters.durable_intermediates,
        )
        cropped_maps[map_name] = destination
        map_statistics[map_name] = statistics.serializable()
        ledger.record(
            destination,
            artifact_kind,
            statistics=statistics,
            sha256=map_sha256,
            details={
                "filter": group_name,
                "sourceIntegration": str(full_master.relative_to(dirs.root)),
                "usesRegistrationQualityWeights": True,
            },
        )
    preview_path = dirs.previews / f"master_light_{token}.png"
    timing["cropAndMaps"] = time.perf_counter() - crop_write_started
    preview_started = time.perf_counter()
    preview_result = render_auto_stretch_preview(
        master_light,
        preview_path,
        max_long_edge=parameters.preview_max_long_edge,
        max_memory_bytes=parameters.registration_memory_bytes,
    )
    preview_record = ledger.record(
        preview_path, "AUTO_STRETCH_PREVIEW", details=preview_result.serializable()
    )
    preview_record["details"]["outputPath"] = str(preview_path.relative_to(dirs.root))
    integration_record = _integration_record(integration, dirs.root)
    integration_record["maps"] = {
        name: str(path.relative_to(dirs.root)) for name, path in cropped_maps.items()
    }
    timing["preview"] = time.perf_counter() - preview_started
    timing["total"] = time.perf_counter() - group_started
    record = {
        "integration": integration_record,
        "regionWeightMaps": region_mapped_lights,
        "globalNormalization": normalization_record,
        "exposureNormalization": {
            "sourceExposureSeconds": sorted(float(value) for value in exposures),
            "referenceSeconds": reference_exposure,
            "totalIntegrationSeconds": total_exposure,
            "method": "LINEAR_REFERENCE_EXPOSURE",
        },
        "crop": [top, left, bottom, right],
        "groupCrop": list(group_crops.get(source_filter, crop)),
        "cropSharedAcrossFilters": len(plan.output_groups) > 1,
        "sourceFilter": source_filter,
        "cfaChannel": CHANNEL_NAMES[cfa_channel] if cfa_channel is not None else None,
        "cfaPattern": cfa_pattern,
        "masterStatistics": master_stats.serializable(),
        "mapStatistics": map_statistics,
        **({"properCoaddition": proper_record} if proper_record is not None else {}),
    }
    return _GroupProducts(
        master_light, preview_path, drizzle, record, timing, proper_coadd=proper_light
    )


def _trusted_reuse_receipt(
    plan: _RunPlan, trusted: _TrustedGeneratedCalibrationSet | None
) -> dict[str, Any]:
    if trusted is None:
        return {"status": "NOT_USED"}
    raw_source_provenance = [
        dict(item) for item in plan.source_records if item["role"] in {"BIAS", "DARK", "FLAT"}
    ]
    return {
        "status": "REUSED_E2E_GENERATED_MASTER",
        "sourceContentManifestSha256": _content_lineage_sha256(raw_source_provenance),
        "privateExecutionManifestBound": True,
        "upstreamRegistrationCalibrationReceiptSha256": trusted.upstream_receipt_sha256,
        "calibrationApplied": False,
        "doubleBiasSubtraction": False,
        "originalRawSourceProvenance": raw_source_provenance,
        "masters": [
            {
                "role": item.role,
                "sha256": item.sha256,
                "sizeBytes": item.size_bytes,
                **({"biasIncluded": item.bias_included} if item.role == "MASTER_DARK" else {}),
                **({"applicationScale": item.application_scale} if item.role == "MASTER_FLAT" else {}),
            }
            for item in (trusted.master_bias, *trusted.master_darks, *trusted.master_flats)
            if item is not None
        ],
    }


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
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None = None,
    quality_weights: Mapping[str, float] | None = None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None = None,
    parameters: PipelineParameters | None = None,
    _source_aliases: Mapping[str, Path] | None = None,
    _xisf_conversions: Sequence[Mapping[str, Any]] = (),
    _trusted_generated_calibration: _TrustedGeneratedCalibrationSet | None = None,
    _source_identity_seed: Mapping[str, tuple[str, Mapping[str, int]]] | None = None,
    _integration_tile_observers: Callable[[str, Sequence[str]], Any] | None = None,
    region_weight_maps: Mapping[str, Any] | None = None,
    _staging_stem: str | None = None,
) -> PipelineResult:
    """Run raw or pre-integrated calibration through unsolved linear masters.

    The stages: plan (validate and group every input), calibration masters,
    per-Light calibrate-and-register jobs, a common crop, then per output
    group normalization, rejection/integration, crop, maps and preview; the
    receipt and one no-replace publication of the staging directory close
    the run.

    ``_staging_stem`` names the transient staging directory beside the output
    (``.<stem>.<8>.stage``); the E2E run passes a short stem because that
    directory is the deepest level of its layout (see ``path_budget``).

    ``_integration_tile_observers(filter_name, ordered_light_paths)`` may
    return a tile observer for that group's ordinary integration (selection
    counterfactual); observers never change the products.

    ``region_weight_maps`` binds Lights (by path) to their selection region
    weight maps (``RegionWeightMap``: node values in normalized reference
    coordinates); a bound Light's samples are weighted by the map during the
    ordinary weighted mean.  Lights without a map keep unit sample weights.

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
    plan = _plan_run(
        bias_files=bias_files,
        dark_files=dark_files,
        flat_files=flat_files,
        master_bias_file=master_bias_file,
        master_dark_files=master_dark_files,
        master_flat_files=master_flat_files,
        light_files=light_files,
        output_directory=output_directory,
        transforms=transforms,
        quality_weights=quality_weights,
        stellar_scale_hints=stellar_scale_hints,
        region_weight_maps=region_weight_maps,
        parameters=parameters,
        source_aliases=_source_aliases,
        trusted_generated_calibration=_trusted_generated_calibration,
        source_identity_seed=_source_identity_seed,
    )
    output = plan.output
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{_staging_stem or output.name}.", suffix=STAGING_SUFFIX, dir=output.parent
        )
    )
    published = False
    metal = _MetalSession()
    fits: _NormalizationFits | None = None
    try:
        dirs = _StagingDirs.create(staging)
        ledger = _RunLedger(staging)
        masters = _build_calibration_masters(plan, dirs, parameters, ledger)
        hardware_profile = detect_hardware()
        execution_tuning = select_execution_tuning(hardware_profile)
        light_jobs, calibrated_details = _plan_light_jobs(plan, dirs, parameters, masters)
        lights = _calibrate_and_register_lights(
            plan, dirs, light_jobs, calibrated_details, parameters, execution_tuning, ledger
        )
        if parameters.ordinary_integration_backend != "portable-cpu":
            metal.open(parameters)

        # Every group of this run was registered onto the same reference grid.
        # Cropping each master to the rectangle that is valid in all groups
        # keeps the masters of different filters on one identical pixel grid,
        # as WBPP's autocrop does, so LRGB composition never resamples them.
        # A single-filter run keeps exactly its own crop.
        crop_started = time.perf_counter()
        shared_crop, group_crops = _shared_auto_crop(
            plan.light_groups,
            plan.light_info,
            plan.transforms,
            enabled=parameters.auto_crop,
            max_memory_bytes=parameters.registration_memory_bytes,
            resampler=parameters.registration_resampler,
        )
        stage_timing: dict[str, Any] = {
            "fusedCalibrateWarp": round(lights.wall_seconds, 3),
            "autoCrop": round(time.perf_counter() - crop_started, 3),
            "groups": {},
        }
        ordered_groups = sorted(plan.output_groups.items())
        concurrency = _group_concurrency(parameters, execution_tuning, len(ordered_groups))
        fits = _NormalizationFits(
            plan, lights.by_group, parameters, execution_tuning, ordered_groups,
            prefetch=concurrency == 1,
        )
        # Observers are created here, in group order, so a caller that keeps
        # them sees the groups in that order however the groups are scheduled.
        observers = {
            group_name: (
                _integration_tile_observers(group_name, [str(path) for path in paths])
                if _integration_tile_observers is not None
                else None
            )
            for group_name, paths in ordered_groups
        }
        # Each group records its artifacts in its own ledger; the run ledger
        # takes them in group order, so the receipt does not depend on which
        # group finished first.
        group_ledgers = {group_name: _RunLedger(staging) for group_name, _ in ordered_groups}

        def integrate(position: int) -> _GroupProducts:
            group_name, paths = ordered_groups[position]
            if concurrency == 1:
                fits.prefetch_after(position)
            return _integrate_group(
                plan,
                dirs,
                group_name,
                paths,
                parameters=parameters,
                lights=lights,
                fits=fits,
                metal=metal,
                hardware_profile=hardware_profile,
                execution_tuning=execution_tuning,
                shared_crop=shared_crop,
                group_crops=group_crops,
                tile_observer=observers[group_name],
                ledger=group_ledgers[group_name],
            )

        if concurrency == 1:
            results = [integrate(position) for position in range(len(ordered_groups))]
        else:
            # Largest groups start first so the last group to finish is a
            # small one; every group's products are independent of the order.
            schedule = sorted(
                range(len(ordered_groups)),
                key=lambda position: (-len(ordered_groups[position][1]), position),
            )
            fits.prefetch_all(schedule, concurrent=concurrency)
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="ufwbpp-group") as pool:
                futures = {position: pool.submit(integrate, position) for position in schedule}
            results = [futures[position].result() for position in range(len(ordered_groups))]
        products: dict[str, _GroupProducts] = {}
        for (group_name, _paths), product in zip(ordered_groups, results, strict=True):
            products[group_name] = product
            ledger.artifacts.extend(group_ledgers[group_name].artifacts)
            stage_timing["groups"][group_name] = {
                key: round(value, 3) for key, value in product.timing.items()
            }
        fits.shutdown()
        remove_tree(dirs.work)
        _verify_source_identities(plan.source_identities)
        if _trusted_generated_calibration is not None:
            # Recheck source stat identities and rehash the generated masters
            # and upstream receipt after all consumers finish.  The enclosing
            # E2E publication gate deliberately performs the second full hash
            # of every original source; intermediate receipt lookups reuse this
            # run's path/stat-bound digest instead of rereading large inputs.
            _validate_trusted_generated_calibration_set(
                _trusted_generated_calibration,
                source_groups=plan.source_groups,
                source_aliases=plan.source_aliases,
                identity_cache=plan.source_identity_cache,
            )
        receipt = {
            "schemaVersion": 1,
            "pipelineVersion": PIPELINE_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "state": OUTPUT_STATE,
            "parameters": parameters.serializable(),
            "inputs": plan.source_records,
            "pixelNumericDomains": plan.pixel_numeric_domains,
            "trustedGeneratedCalibration": _trusted_reuse_receipt(plan, _trusted_generated_calibration),
            "pixelInputStaging": {
                "xisfPolicy": parameters.xisf_decode.serializable(),
                "conversions": [dict(item) for item in _xisf_conversions],
                "privateStagingRemovedAfterRun": True,
            },
            "masterMetadataOverrides": [
                {**item.serializable(), "status": "APPLIED_CONTENT_BOUND_DECLARATION"}
                for item in parameters.master_metadata_overrides
            ],
            "outputs": ledger.artifacts,
            "registration": lights.records,
            "statistics": {
                "calibration": ledger.stage_statistics,
                "registration": lights.execution,
                "integrationGroups": {name: product.record for name, product in products.items()},
                "integrationGroupConcurrency": concurrency,
                "timingSeconds": stage_timing,
            },
            # Platform facts behind the execution choices: what the machine
            # is, which tuning table row ran, and which native library (if
            # any) produced the kernel results.  None of them changes pixels.
            "platform": {
                "hardware": hardware_profile.serializable(),
                "tuning": execution_tuning.serializable(),
                "nativeKernels": describe_native_kernels(),
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
        with suppress(OSError):
            platform_services.current().fsync_directory(output.parent)

        def published_path(path: Path | str) -> str:
            return str(output / Path(path).relative_to(staging))

        return PipelineResult(
            output_directory=str(output),
            receipt_path=str(output / "receipt.json"),
            state=OUTPUT_STATE,
            master_light_paths=tuple(published_path(product.master_light) for product in products.values()),
            preview_paths=tuple(published_path(product.preview) for product in products.values()),
            drizzle_groups={
                name: replace(
                    product.drizzle,
                    frames=tuple(
                        replace(frame, calibrated_path=published_path(frame.calibrated_path))
                        for frame in product.drizzle.frames
                    ),
                )
                for name, product in products.items()
                if product.drizzle is not None
            },
            proper_coadd_paths={
                name: published_path(product.proper_coadd)
                for name, product in products.items()
                if product.proper_coadd is not None
            },
        )
    finally:
        if fits is not None:
            fits.shutdown(cancel=True)
        metal.close()
        if not published:
            remove_tree(staging)


def _rekey_for_staged_lights(
    originals: tuple[Path, ...],
    staged_by_original: Mapping[Path, Path],
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None,
    quality_weights: Mapping[str, float] | None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None,
) -> tuple[
    Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None,
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
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None = None,
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
    light_paths = tuple(light_files)
    # The runtime checks the same budget for its own layouts before a run
    # starts; a direct caller of the pipeline gets the check here, in this
    # module's error contract.  The runtime module imports this one, so its
    # error class is resolved at call time.
    from .runtime import RuntimeConfigurationError

    try:
        check_output_path_budget(
            output, light_count=len(light_paths), light_paths=light_paths, layout="pixels"
        )
    except RuntimeConfigurationError as error:
        raise CalibrationError(error.code, str(error), path=str(output)) from error
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
        "LIGHT": _canonical_inputs(light_paths, "Light", required=True),
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
    "PixelTransform",
    "AffineTransform",
    "MasterMetadataOverride",
    "OUTPUT_STATE",
    "PIPELINE_VERSION",
    "PipelineParameters",
    "PipelineResult",
    "run_portable_pipeline",
]
