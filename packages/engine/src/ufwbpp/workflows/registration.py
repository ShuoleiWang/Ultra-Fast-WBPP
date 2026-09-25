"""Registration of the admitted Lights: the calibration masters it measures on, and the transforms it produces."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lightframeqc.parallel import FrameRunner

from ..calibration.inputs import (
    InternalSourceIdentity,
    apply_master_metadata_overrides,
    assert_compatible,
    find_dark,
    master_dark_bias_semantics,
    numeric_application_scale,
    numeric_domain_metadata,
    capture_trusted_generated_calibration_set,
)
from ..calibration.policy import MONO_STANDARD, can_omit_bias, conflicting_profile_fields, workflow_receipt
from ..image_io.fits import cfa_metadata
from ..integrity import sha256_digest
from ..stacking.integration import (
    FrameExpression,
    FrameInfo,
    integrate_expressions,
    normalize_role,
    robust_location,
)
from ..stacking.normalization import StellarScaleHint
from ..stacking.parameters import PipelineParameters
from ..stacking.records import require_filter
from .common import _safe_token
from .contracts import E2EError
from .sources import _input_frame_info, _SourceIdentity


@dataclass(frozen=True, slots=True)
class _RegistrationProducts:
    transforms: dict[str, tuple[tuple[float, float, float], ...]]
    quality_weights: dict[str, float]
    stellar_scale_hints: dict[str, StellarScaleHint]
    run: Any
    receipt: dict[str, Any]
    source_aliases: dict[str, str] = field(default_factory=dict)


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
        from ufwbpp_registration import CalibrationPlan
    except ImportError as error:
        raise E2EError(
            "REGISTRATION_BACKEND_UNAVAILABLE",
            "ufwbpp-registration must be installed for E2E execution",
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
    ) = apply_master_metadata_overrides(
        (supplied_bias_infos, supplied_dark_infos, supplied_flat_infos),
        pipeline_parameters.master_metadata_overrides,
        dict(source_aliases or {}),
    )
    supplied_dark_bias_included = master_dark_bias_semantics(
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
        light_domain_scale = numeric_application_scale(
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
        assert_compatible(reference_bias, info, workflow=pipeline_parameters.calibration_workflow)
    if biases:
        master_bias = directory / "master_bias.fits"
        bias_result = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=numeric_application_scale(
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
                **numeric_domain_metadata(reference_bias),
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
            "sha256": sha256_digest(display(master_bias)),
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
            assert_compatible(reference_bias, dark_infos[path], workflow=pipeline_parameters.calibration_workflow)
            assert_compatible(
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
                    scale=numeric_application_scale(
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
                **numeric_domain_metadata(reference_dark),
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
        if find_dark(exposure, master_darks) is not None:
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
            "sha256": sha256_digest(display(path)),
            "calibrationApplied": False,
            "biasIncluded": supplied_dark_bias_included[path],
            "numericDomain": info.numeric_domain,
            "normalizedUnitScale": info.normalized_unit_scale,
        }

    flat_groups: dict[str, list[Path]] = {}
    for path in flats:
        filter_name = require_filter(flat_infos[path])
        flat_groups.setdefault(filter_name, []).append(path)
    supplied_flats_by_filter: dict[str, Path] = {}
    for path, info in supplied_flat_infos.items():
        filter_name = require_filter(info)
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
    light_filters = {require_filter(info) for info in light_infos.values()}
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
            assert_compatible(reference_bias, flat_infos[path], workflow=pipeline_parameters.calibration_workflow)
            flat_info = flat_infos[path]
            flat_dark_match = find_dark(flat_info.exposure_seconds, master_darks)
            if flat_dark_match is not None:
                dark_exposure, subtract_path = flat_dark_match
                dark_info = (
                    dark_infos[dark_groups[dark_exposure][0]]
                    if dark_exposure in dark_groups
                    else supplied_dark_infos[subtract_path]
                )
                assert_compatible(
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
            flat_subtract_scale = numeric_application_scale(
                flat_info,
                flat_subtract_info,
                target_label="raw Flat",
                additive_label=(
                    "MasterDark" if flat_dark_match is not None else "MasterBias"
                ),
            )
            flat_bias_scale = numeric_application_scale(
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
                    "subtractedSha256": sha256_digest(subtract_path),
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
            "sha256": sha256_digest(display(path)),
            "calibrationApplied": False,
        }

    for path, light_info in light_infos.items():
        filter_name = require_filter(light_info)
        flat_path = master_flats[filter_name]
        flat_info = (
            flat_infos[flat_groups[filter_name][0]]
            if filter_name in flat_groups
            else supplied_flat_infos[flat_path]
        )
        assert_compatible(light_info, flat_info, compare_filter=True, workflow=pipeline_parameters.calibration_workflow)
        dark_match = find_dark(light_info.exposure_seconds, master_darks)
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
            assert_compatible(
                light_info,
                dark_info,
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=pipeline_parameters.dark_temperature_tolerance_celsius,
                workflow=pipeline_parameters.calibration_workflow,
            )
            numeric_application_scale(
                light_info,
                master_dark_domain_info[exposure],
                target_label="registration Light",
                additive_label="MasterDark",
            )
        numeric_application_scale(
            light_info,
            reference_bias,
            target_label="registration Light",
            additive_label="MasterBias",
        )
    bias_application_scale = numeric_application_scale(
        light_numeric_reference,
        reference_bias,
        target_label="registration Light domain",
        additive_label="MasterBias",
    )
    dark_application_scales = {
        exposure: numeric_application_scale(
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
                "sha256": sha256_digest(path),
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
    internal_source_identities: Mapping[str, InternalSourceIdentity],
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
        match = find_dark(exposure, {float(key): Path(value) for key, value in plan.dark_paths.items()})
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
    for filter_name in sorted({require_filter(info) for info in raw_flat_infos.values()}):
        flat_path = Path(plan.flat_paths[filter_name])
        if not is_generated(flat_path):
            raise E2EError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "raw Flat provenance did not produce an internal MasterFlat",
                path=str(flat_path),
            )
        reference = next(
            info for info in raw_flat_infos.values() if require_filter(info) == filter_name
        )
        flat_specs.append((flat_path, reference, 1.0))

    return capture_trusted_generated_calibration_set(
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
        from ufwbpp_registration import RegistrationConfig, run_registration
        from ufwbpp_registration.quality import (
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
            digest = sha256_digest(Path(actual))
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
