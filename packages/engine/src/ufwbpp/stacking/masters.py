"""Building or reusing the MasterBias, MasterDarks and MasterFlats of one run."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from lightframeqc.cfa import CFA_PATTERNS, channel_medians

from ..calibration.inputs import (
    assert_compatible,
    numeric_application_scale,
    numeric_domain_metadata,
    source_identity,
    find_dark,
)
from ..image_io.fits import cfa_metadata
from .integration import (
    CalibrationError,
    FitsFrame,
    FrameExpression,
    FrameInfo,
    integrate_expressions,
    robust_location,
)
from .parameters import OUTPUT_STATE, PipelineParameters
from .records import _exposure_token, _integration_record, _safe_token
from .run_plan import _RunLedger, _RunPlan, _StagingDirs


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
                    scale=numeric_application_scale(
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
                **numeric_domain_metadata(reference_bias),
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
        _, master_bias_sha256, _ = source_identity(
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
            assert_compatible(plan.reference_bias, plan.dark_info[path], workflow=workflow)
            assert_compatible(
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
                    scale=numeric_application_scale(
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
                **numeric_domain_metadata(dark_reference),
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
        _, supplied_sha256, _ = source_identity(
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
            assert_compatible(
                reference_bias, info, compare_filter=False, compare_exposure=False,
                workflow=workflow,
            )
            if info.filter_name != filter_name:
                raise CalibrationError("FLAT_GROUP_INVALID", "internal filter grouping error")
            flat_dark_match = find_dark(info.exposure_seconds, master_darks)
            if flat_dark_match is not None:
                dark_exposure, flat_subtract = flat_dark_match
                assert_compatible(
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
            flat_subtract_scale = numeric_application_scale(
                info,
                flat_subtract_info,
                target_label="raw Flat",
                additive_label=("MasterDark" if flat_dark_match is not None else "MasterBias"),
            )
            flat_bias_scale = numeric_application_scale(
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
        _, supplied_sha256, _ = source_identity(
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
