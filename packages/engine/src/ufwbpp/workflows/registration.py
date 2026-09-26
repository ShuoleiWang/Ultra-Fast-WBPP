"""Registration of the admitted Lights: the calibration masters it measures on, and the transforms it produces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lightframeqc.metadata import grouping_keyword_root
from lightframeqc.parallel import FrameRunner

from ..calibration.inputs import (
    InternalSourceIdentity,
    apply_master_metadata_overrides,
    master_dark_bias_semantics,
    numeric_application_scale,
    numeric_domain_metadata,
    capture_trusted_generated_calibration_set,
)
from ..calibration.matching import BIAS, DARK, FLAT, CalibrationMatch
from ..calibration.pairing import dark_group_warnings, pair_calibration
from ..calibration.policy import MONO_STANDARD, workflow_receipt
from ..image_io.fits import cfa_metadata
from ..integrity import sha256_digest
from ..stacking.integration import (
    CalibrationError,
    FrameExpression,
    FrameInfo,
    integrate_expressions,
    normalize_role,
    robust_location,
)
from ..stacking.normalization import StellarScaleHint
from ..stacking.parameters import PipelineParameters
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


@dataclass(frozen=True)
class _RegistrationCalibration:
    """What registration calibrates with: every master by calibration group
    (``(kind, key)``), the pairing that chose them, the per-Light plan the
    registration library applies and the receipt."""

    plan: Any
    receipt: dict[str, Any]
    match: CalibrationMatch
    masters: dict[tuple[str, str], Path]


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
) -> _RegistrationCalibration:
    """Build the master of every calibration group the Lights use, paired by
    WBPP's rules exactly as the pixel pipeline pairs them."""

    try:
        from ufwbpp_registration import CalibrationPlan, LightMasters
    except ImportError as error:
        raise E2EError(
            "REGISTRATION_BACKEND_UNAVAILABLE",
            "ufwbpp-registration must be installed for E2E execution",
        ) from error

    if pipeline_parameters.grouping_keyword_root is None:
        pipeline_parameters = replace(
            pipeline_parameters,
            grouping_keyword_root=grouping_keyword_root(
                str((source_aliases or {}).get(str(path), path))
                for path in (*biases, *darks, *flats, *supplied_biases, *supplied_darks, *supplied_flats, *lights)
            ),
        )
    parameters = pipeline_parameters.integration
    workflow = pipeline_parameters.calibration_workflow
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
    infos_by_role: dict[str, dict[Path, FrameInfo]] = {}
    for role, paths in grouped_roles:
        infos_by_role[role] = {}
        for path in paths:
            info = frame_info(path)
            actual = normalize_role(info.role)
            if actual == "UNKNOWN":
                info = replace(info, role=role)
            elif actual != role:
                raise E2EError(
                    "FRAME_ROLE_MISMATCH",
                    f"expected {role}, found {actual}",
                    path=str(path),
                )
            infos_by_role[role][path] = info

    if not biases and not supplied_biases and workflow != MONO_STANDARD:
        raise E2EError(
            "BIAS_SOURCE_AMBIGUOUS",
            "the strict workflow requires raw Bias frames or a MasterBias",
        )
    bias_infos = infos_by_role["BIAS"]
    light_infos = infos_by_role["LIGHT"]
    dark_infos = infos_by_role["DARK"]
    flat_infos = infos_by_role["FLAT"]
    (
        supplied_bias_infos,
        supplied_dark_infos,
        supplied_flat_infos,
    ) = apply_master_metadata_overrides(
        (infos_by_role["MASTER_BIAS"], infos_by_role["MASTER_DARK"], infos_by_role["MASTER_FLAT"]),
        pipeline_parameters.master_metadata_overrides,
        dict(source_aliases or {}),
    )
    supplied_dark_bias_included = master_dark_bias_semantics(
        supplied_darks,
        pipeline_parameters.master_metadata_overrides,
        dict(source_aliases or {}),
        workflow=workflow,
    )
    display = lambda path: (source_aliases or {}).get(str(path), path)  # noqa: E731
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
    try:
        match = pair_calibration(
            bias_info=bias_infos,
            master_bias_info=supplied_bias_infos,
            dark_info=dark_infos,
            master_dark_info=supplied_dark_infos,
            flat_info=flat_infos,
            master_flat_info=supplied_flat_infos,
            light_info=light_infos,
            supplied_dark_bias_included=supplied_dark_bias_included,
            workflow=workflow,
            dark_temperature_tolerance_celsius=pipeline_parameters.dark_temperature_tolerance_celsius,
        )
    except CalibrationError as error:
        raise E2EError(error.code, str(error), path=error.path) from error
    info_of: dict[str, FrameInfo] = {
        str(path): info
        for group in (bias_infos, supplied_bias_infos, dark_infos, supplied_dark_infos, flat_infos, supplied_flat_infos)
        for path, info in group.items()
    }

    def reference(kind: str, key: str) -> FrameInfo:
        return info_of[match.groups[kind][key].members[0]]

    def members(kind: str, key: str) -> tuple[Path, ...]:
        return tuple(Path(member) for member in match.groups[kind][key].members)

    def label(key: str) -> str:
        return f"_{_safe_token(key.split('|', 1)[1])}" if "|" in key else ""

    def supplied_record(kind: str, path: Path, info: FrameInfo, *, used: bool) -> dict[str, Any]:
        return {
            "mode": "REUSED_SUPPLIED_MASTER",
            "path": str(display(path)),
            "sha256": sha256_digest(display(path)),
            "calibrationApplied": False,
            **({"biasIncluded": supplied_dark_bias_included[path]} if kind == DARK else {}),
            **(
                {}
                if kind == FLAT
                else {"numericDomain": info.numeric_domain, "normalizedUnitScale": info.normalized_unit_scale}
            ),
            **({} if used else {"used": False}),
        }

    masters: dict[tuple[str, str], Path] = {}
    domain: dict[tuple[str, str], FrameInfo] = {}
    records: dict[str, dict[str, Any]] = {BIAS: {}, DARK: {}, FLAT: {}}
    for key, group in sorted(match.groups[BIAS].items()):
        info = reference(BIAS, key)
        used = key in match.used(BIAS)
        if group.supplied_master:
            records[BIAS][key] = supplied_record(BIAS, Path(group.members[0]), info, used=used)
            if used:
                masters[(BIAS, key)] = Path(group.members[0])
                domain[(BIAS, key)] = info
            continue
        if not used:
            records[BIAS][key] = {"mode": "NOT_USED"}
            continue
        master_bias = directory / f"master_bias{label(key)}.fits"
        bias_result = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=numeric_application_scale(
                        info,
                        bias_infos[path],
                        target_label="MasterBias reference",
                        additive_label="raw Bias",
                    ),
                )
                for path in members(BIAS, key)
            ),
            master_bias,
            metadata={
                "IMAGETYP": "Master Bias",
                "OAFSTATE": "UNSOLVED_WORKING",
                **cfa_metadata(info),
                **numeric_domain_metadata(info),
            },
            parameters=parameters,
        )
        masters[(BIAS, key)] = master_bias
        domain[(BIAS, key)] = info
        records[BIAS][key] = {
            "mode": "BUILT_FROM_RAW",
            "numericDomain": info.numeric_domain,
            "normalizedUnitScale": info.normalized_unit_scale,
            **bias_result.serializable(),
        }

    for key, group in sorted(match.groups[DARK].items()):
        info = reference(DARK, key)
        used = key in match.used(DARK)
        if group.supplied_master:
            records[DARK][key] = supplied_record(DARK, Path(group.members[0]), info, used=used)
            if used:
                masters[(DARK, key)] = Path(group.members[0])
                domain[(DARK, key)] = info
            continue
        if not used:
            records[DARK][key] = {"mode": "NOT_USED"}
            continue
        exposure = float(info.exposure_seconds)
        destination = directory / f"master_dark_{format(exposure, '.9g').replace('.', 'p')}s{label(key)}.fits"
        result = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=numeric_application_scale(
                        info,
                        dark_infos[path],
                        target_label="MasterDark reference",
                        additive_label="raw Dark",
                    ),
                )
                for path in members(DARK, key)
            ),
            destination,
            metadata={
                "IMAGETYP": "Master Dark",
                "EXPTIME": exposure,
                "OAFSTATE": "UNSOLVED_WORKING",
                "OAFBIAS": "INCLUDED",
                **cfa_metadata(info),
                **numeric_domain_metadata(info),
            },
            parameters=parameters,
        )
        masters[(DARK, key)] = destination
        domain[(DARK, key)] = info
        records[DARK][key] = {
            "mode": "BUILT_FROM_RAW",
            "biasIncluded": True,
            "numericDomain": info.numeric_domain,
            "normalizedUnitScale": info.normalized_unit_scale,
            "applicationScaleToRawDarkReference": 1.0,
            **result.serializable(),
        }

    def dark_bias_included(key: str) -> bool:
        group = match.groups[DARK][key]
        return True if not group.supplied_master else supplied_dark_bias_included[Path(group.members[0])]

    for key, group in sorted(match.groups[FLAT].items()):
        info = reference(FLAT, key)
        used = key in match.used(FLAT)
        if group.supplied_master:
            records[FLAT][key] = supplied_record(FLAT, Path(group.members[0]), info, used=used)
            if used:
                masters[(FLAT, key)] = Path(group.members[0])
                domain[(FLAT, key)] = info
            continue
        if not used:
            records[FLAT][key] = {"mode": "NOT_USED"}
            continue
        pairing = match.flats[key]
        expressions: list[FrameExpression] = []
        normalizations: list[float] = []
        calibration_sources: list[dict[str, Any]] = []
        for path in members(FLAT, key):
            flat_info = flat_infos[path]
            subtract_path: Path | None
            if pairing.dark is not None:
                subtract_path = masters[(DARK, pairing.dark)]
                bias_included = dark_bias_included(pairing.dark)
                calibration_mode = (
                    "MATCHED_BIAS_INCLUDED_DARK"
                    if bias_included
                    else "MATCHED_BIAS_SUBTRACTED_DARK_PLUS_MASTER_BIAS"
                )
                subtract_info: FrameInfo | None = domain[(DARK, pairing.dark)]
            elif pairing.bias is not None:
                subtract_path = masters[(BIAS, pairing.bias)]
                bias_included = True
                calibration_mode = "BIAS"
                subtract_info = domain[(BIAS, pairing.bias)]
            else:
                subtract_path = None
                bias_included = True
                calibration_mode = "UNCALIBRATED"
                subtract_info = None
            subtract_scale = (
                numeric_application_scale(
                    flat_info,
                    subtract_info,
                    target_label="raw Flat",
                    additive_label=("MasterDark" if pairing.dark is not None else "MasterBias"),
                )
                if subtract_info is not None
                else 1.0
            )
            separate_bias = not bias_included and pairing.bias is not None
            bias_scale = (
                numeric_application_scale(
                    flat_info,
                    domain[(BIAS, pairing.bias)],
                    target_label="raw Flat",
                    additive_label="MasterBias",
                )
                if separate_bias
                else None
            )
            bias_terms: dict[str, Any] = (
                {
                    "subtract_paths": (str(masters[(BIAS, pairing.bias)]),),
                    "subtract_scales": (bias_scale,),
                }
                if separate_bias
                else {}
            )
            expression = FrameExpression(
                str(path),
                subtract_path=str(subtract_path) if subtract_path is not None else None,
                subtract_scale=subtract_scale,
                **bias_terms,
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
            expressions.append(replace(expression, scale=1.0 / location))
            calibration_sources.append(
                {
                    "source": str(display(path)),
                    "mode": calibration_mode,
                    "subtracted": str(subtract_path) if subtract_path is not None else None,
                    "subtractedSha256": sha256_digest(subtract_path) if subtract_path is not None else None,
                    "targetNumericDomain": flat_info.numeric_domain,
                    "additiveNumericDomain": subtract_info.numeric_domain if subtract_info is not None else None,
                    "applicationScale": subtract_scale,
                    "applicationScaleSource": "normalized-unit-domain-ratio",
                    "biasApplicationScale": bias_scale,
                }
            )
        destination = directory / f"master_flat_{_safe_token(info.filter_name)}{label(key)}.fits"
        result = integrate_expressions(
            expressions,
            destination,
            metadata={
                "IMAGETYP": "Master Flat",
                "FILTER": info.filter_name,
                "OAFSTATE": "UNSOLVED_WORKING",
                "OAFBIAS": "SUBTRACTED",
                "OAFNDOM": "DIMENSIONLESS_RESPONSE",
                "OAFNSCL": 1.0,
                **cfa_metadata(info),
            },
            parameters=parameters,
        )
        masters[(FLAT, key)] = destination
        domain[(FLAT, key)] = info
        records[FLAT][key] = {
            "mode": "BUILT_FROM_RAW",
            "applicationScale": 1.0,
            **result.serializable(),
            "normalizations": normalizations,
            "calibrationSources": calibration_sources,
        }

    light_masters: dict[str, Any] = {}
    bias_scales: dict[str, float] = {}
    dark_scales: dict[str, float] = {}
    for path, light_info in light_infos.items():
        pairing = match.lights[str(path)]
        if pairing.bias is not None:
            numeric_application_scale(
                light_info, domain[(BIAS, pairing.bias)], target_label="registration Light", additive_label="MasterBias"
            )
            bias_scales[pairing.bias] = numeric_application_scale(
                light_numeric_reference,
                domain[(BIAS, pairing.bias)],
                target_label="registration Light domain",
                additive_label="MasterBias",
            )
        if pairing.dark is not None:
            numeric_application_scale(
                light_info, domain[(DARK, pairing.dark)], target_label="registration Light", additive_label="MasterDark"
            )
            dark_scales[pairing.dark] = numeric_application_scale(
                light_numeric_reference,
                domain[(DARK, pairing.dark)],
                target_label="registration Light domain",
                additive_label="MasterDark",
            )
        light_masters[str(path)] = LightMasters(
            bias_path=str(masters[(BIAS, pairing.bias)]) if pairing.bias is not None else None,
            dark_path=str(masters[(DARK, pairing.dark)]) if pairing.dark is not None else None,
            flat_path=str(masters[(FLAT, pairing.flat)]) if pairing.flat is not None else None,
            dark_includes_bias=dark_bias_included(pairing.dark) if pairing.dark is not None else True,
            bias_application_scale=bias_scales.get(pairing.bias, 1.0) if pairing.bias is not None else 1.0,
            dark_application_scale=dark_scales.get(pairing.dark, 1.0) if pairing.dark is not None else 1.0,
        )
    plan = CalibrationPlan(
        dark_scale=1.0,
        dark_includes_bias=True,
        flat_floor_fraction=0.05,
        light_masters=light_masters,
    )
    warnings = (*match.warnings, *dark_group_warnings(match, dark_infos, pipeline_parameters.dark_temperature_tolerance_celsius))
    if not match.groups[BIAS]:
        compatible_bias = {"mode": "NOT_REQUIRED_DARK_INCLUDES_BIAS"}
    elif set(match.groups[BIAS]) == {"bias"}:
        compatible_bias = records[BIAS]["bias"]
    else:
        compatible_bias = {"mode": "GROUPED", "groups": sorted(match.groups[BIAS])}
    receipt = {
        "schemaVersion": 1,
        "stage": "registration-calibration-masters",
        "calibrationPolicy": workflow_receipt(workflow),
        "xisfConversions": [dict(item) for item in xisf_conversions],
        "calibrationMatching": {
            **match.serializable(),
            "warnings": [issue.serializable() for issue in warnings],
        },
        "masterBias": compatible_bias,
        "masterBiases": records[BIAS],
        "masterDarks": records[DARK],
        "masterFlats": records[FLAT],
        "registrationDarksByExposure": {
            key: str(masters[(DARK, key)]) for key in sorted(match.used(DARK))
        },
        "registrationNumericDomain": {
            "light": light_numeric_reference.serializable(),
            "biasApplicationScaleByGroup": dict(sorted(bias_scales.items())),
            "darkApplicationScaleByGroup": dict(sorted(dark_scales.items())),
            "applicationScaleSource": "normalized-unit-domain-ratio",
        },
        "artifacts": [
            {
                "path": str(path),
                "sha256": sha256_digest(path),
                "sizeBytes": path.stat().st_size,
            }
            for path in dict.fromkeys(masters.values())
        ],
    }
    return _RegistrationCalibration(plan=plan, receipt=receipt, match=match, masters=masters)


def _capture_single_field_generated_calibration(
    *,
    plan: _RegistrationCalibration,
    generated_directory: Path,
    upstream_receipt_path: Path,
    staged_inputs: Mapping[str, tuple[Path, ...]],
    source_aliases: Mapping[str, Path],
    pipeline_parameters: PipelineParameters,
    consumer_source_groups: Sequence[tuple[str, Sequence[Path]]],
    internal_source_identities: Mapping[str, InternalSourceIdentity],
) -> Any:
    """Capture the private single-run trust handoff for generated masters:
    one per raw calibration group registration used, named by its key."""

    del staged_inputs  # the calibration groups name their own members
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

    specs: dict[str, list[tuple[Any, ...]]] = {BIAS: [], DARK: [], FLAT: []}
    for kind in (BIAS, DARK, FLAT):
        for key in sorted(plan.match.used(kind)):
            group = plan.match.groups[kind][key]
            if group.supplied_master:
                continue
            master = plan.masters[(kind, key)]
            if not is_generated(master):
                raise E2EError(
                    "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                    f"raw {kind.title()} provenance did not produce an internal master",
                    path=str(master),
                )
            reference = source_info(Path(group.members[0]))
            if kind == BIAS:
                specs[kind].append((master, reference, key))
            elif kind == DARK:
                specs[kind].append((master, reference, True, key))
            else:
                specs[kind].append((master, reference, 1.0, key))

    return capture_trusted_generated_calibration_set(
        master_biases=specs[BIAS],
        master_darks=specs[DARK],
        master_flats=specs[FLAT],
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
