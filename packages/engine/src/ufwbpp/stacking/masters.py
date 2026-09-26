"""Building or reusing the MasterBias, MasterDark and MasterFlat of every
calibration group a run uses (see :mod:`ufwbpp.calibration.matching`)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from lightframeqc.cfa import CFA_PATTERNS, channel_medians

from ..calibration.inputs import numeric_application_scale, numeric_domain_metadata, source_identity
from ..calibration.matching import BIAS, DARK, FLAT
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
    """Master paths and the metadata standing for each, by group key."""

    biases: dict[str, Path]
    bias_domain_info: dict[str, FrameInfo]
    darks: dict[str, Path]
    dark_domain_info: dict[str, FrameInfo]
    flats: dict[str, Path]
    flat_application_scales: dict[str, float]
    flat_pattern_scales: dict[str, tuple[float, float, float, float]]
    flat_channel_medians: dict[str, tuple[float, float, float]]


def _label(key: str) -> str:
    """What sets a group apart from the others of its filter or exposure
    (``NIGHT=1`` of ``L|NIGHT=1``); empty for a plain key."""

    return key.split("|", 1)[1] if "|" in key else ""


def _statistics_key(kind: str, key: str) -> str:
    if kind == BIAS:
        return "masterBias" if key == "bias" else f"masterBias:{key}"
    return f"master{'Dark' if kind == DARK else 'Flat'}:{key}"


def _reused(
    plan: _RunPlan, kind: str, key: str, ledger: _RunLedger, reference: FrameInfo, **extra: Any
) -> Path | None:
    """The trusted E2E-generated or supplied master of a group, recorded;
    ``None`` when the group's master must be built from its raw frames."""

    statistics = _statistics_key(kind, key)
    trusted = plan.trusted_masters.get((kind, key))
    if trusted is not None:
        ledger.stage_statistics[statistics] = {
            "mode": "REUSED_E2E_GENERATED_MASTER",
            "sha256": trusted.sha256,
            "sizeBytes": trusted.size_bytes,
            "calibrationApplied": False,
            "doubleBiasSubtraction": False,
            **({"biasIncluded": trusted.bias_included} if kind == DARK else {}),
            **(
                {"applicationScale": 1.0, "applicationNormalization": 1.0}
                if kind == FLAT
                else {
                    "numericDomain": reference.numeric_domain,
                    "normalizedUnitScale": reference.normalized_unit_scale,
                }
            ),
            **({"applicationScaleToRawDarkReference": 1.0} if kind == DARK else {}),
            **extra,
        }
        return Path(trusted.path)
    group = plan.calibration.groups[kind][key]
    if not group.supplied_master:
        return None
    supplied = Path(group.members[0])
    _, sha256, _ = source_identity(supplied, plan.source_aliases, plan.source_identity_cache)
    ledger.stage_statistics[statistics] = {
        "mode": "REUSED_SUPPLIED_MASTER",
        "path": str(plan.display_path(supplied)),
        "sha256": sha256,
        "calibrationApplied": False,
        **(
            {"biasIncluded": plan.supplied_dark_bias_included[supplied]}
            if kind == DARK
            else {}
        ),
        **(
            {}
            if kind == FLAT
            else {
                "numericDomain": reference.numeric_domain,
                "normalizedUnitScale": reference.normalized_unit_scale,
            }
        ),
        **extra,
    }
    return supplied


def _not_used(plan: _RunPlan, kind: str, key: str, ledger: _RunLedger, reference: FrameInfo) -> None:
    """A group no Light or Flat is paired with: raw frames are not
    integrated; a supplied master is still identified in the receipt."""

    group = plan.calibration.groups[kind][key]
    if group.supplied_master and (kind, key) not in plan.trusted_masters:
        _reused(plan, kind, key, ledger, reference, used=False)
    else:
        ledger.stage_statistics[_statistics_key(kind, key)] = {"mode": "NOT_USED"}


def _build_master_biases(
    plan: _RunPlan, dirs: _StagingDirs, parameters: PipelineParameters, ledger: _RunLedger
) -> tuple[dict[str, Path], dict[str, FrameInfo]]:
    masters: dict[str, Path] = {}
    domain_info: dict[str, FrameInfo] = {}
    groups = plan.calibration.groups[BIAS]
    if not groups:
        missing = any(issue.code == "BIAS_MISSING" for issue in plan.warnings)
        ledger.stage_statistics["masterBias"] = {
            "mode": "NOT_SUPPLIED" if missing else "NOT_REQUIRED_DARK_INCLUDES_BIAS"
        }
        return masters, domain_info
    used = plan.calibration.used(BIAS)
    for key, group in sorted(groups.items()):
        reference = plan.group_reference(BIAS, key)
        if key not in used:
            _not_used(plan, BIAS, key, ledger, reference)
            continue
        domain_info[key] = reference
        reused = _reused(plan, BIAS, key, ledger, reference)
        if reused is not None:
            masters[key] = reused
            continue
        label = _label(key)
        master_bias = dirs.masters / (f"master_bias_{_safe_token(label)}.fits" if label else "master_bias.fits")
        bias_integration = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=numeric_application_scale(
                        reference,
                        plan.bias_info[path],
                        target_label="MasterBias reference",
                        additive_label="raw Bias",
                    ),
                )
                for path in plan.group_paths(BIAS, key)
            ),
            master_bias,
            metadata={
                "IMAGETYP": "Master Bias",
                "OAFSTATE": OUTPUT_STATE,
                "OAFBIAS": "MASTER",
                **cfa_metadata(reference),
                **numeric_domain_metadata(reference),
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
            details={"group": key} if label else None,
        )
        ledger.stage_statistics[_statistics_key(BIAS, key)] = {
            "mode": "BUILT_FROM_RAW",
            "numericDomain": reference.numeric_domain,
            "normalizedUnitScale": reference.normalized_unit_scale,
            **_integration_record(bias_integration, dirs.root),
        }
        masters[key] = master_bias
    return masters, domain_info


def _build_master_darks(
    plan: _RunPlan, dirs: _StagingDirs, parameters: PipelineParameters, ledger: _RunLedger
) -> tuple[dict[str, Path], dict[str, FrameInfo]]:
    """Master darks by group (built, reused from the E2E run or supplied)
    and the frame metadata that stands for each one's numeric domain."""

    masters: dict[str, Path] = {}
    domain_info: dict[str, FrameInfo] = {}
    used = plan.calibration.used(DARK)
    for key, group in sorted(plan.calibration.groups[DARK].items()):
        reference = plan.group_reference(DARK, key)
        if key not in used:
            _not_used(plan, DARK, key, ledger, reference)
            continue
        domain_info[key] = reference
        reused = _reused(plan, DARK, key, ledger, reference)
        if reused is not None:
            masters[key] = reused
            continue
        exposure = float(reference.exposure_seconds)
        label = _label(key)
        destination = dirs.masters / (
            f"master_dark_{_exposure_token(exposure)}s"
            + (f"_{_safe_token(label)}" if label else "")
            + ".fits"
        )
        integration = integrate_expressions(
            (
                FrameExpression(
                    str(path),
                    scale=numeric_application_scale(
                        reference,
                        plan.dark_info[path],
                        target_label="MasterDark reference",
                        additive_label="raw Dark",
                    ),
                )
                for path in plan.group_paths(DARK, key)
            ),
            destination,
            metadata={
                "IMAGETYP": "Master Dark",
                "EXPTIME": exposure,
                "OAFSTATE": OUTPUT_STATE,
                "OAFBIAS": "INCLUDED",
                **cfa_metadata(reference),
                **numeric_domain_metadata(reference),
            },
            parameters=parameters.integration,
            durable=parameters.durable_intermediates,
        )
        masters[key] = destination
        ledger.record(
            destination,
            "MASTER_DARK",
            statistics=integration.statistics,
            sha256=integration.output_sha256,
            details={"exposureSeconds": exposure, "biasIncluded": True, **({"group": key} if label else {})},
        )
        statistics = _statistics_key(DARK, key)
        ledger.stage_statistics[statistics] = _integration_record(integration, dirs.root)
        ledger.stage_statistics[statistics]["mode"] = "BUILT_FROM_RAW"
        ledger.stage_statistics[statistics].update(
            {
                "numericDomain": reference.numeric_domain,
                "normalizedUnitScale": reference.normalized_unit_scale,
                "applicationScaleToRawDarkReference": 1.0,
            }
        )
    return masters, domain_info


def _build_master_flats(
    plan: _RunPlan,
    dirs: _StagingDirs,
    parameters: PipelineParameters,
    ledger: _RunLedger,
    *,
    master_biases: dict[str, Path],
    bias_domain_info: dict[str, FrameInfo],
    master_darks: dict[str, Path],
    dark_domain_info: dict[str, FrameInfo],
) -> tuple[dict[str, Path], dict[str, float]]:
    """Master flats by group and the factor each is divided with, so the
    division uses a response normalized to unity."""

    master_flats: dict[str, Path] = {}
    application_scales: dict[str, float] = {}
    used = plan.calibration.used(FLAT)
    for key, group in sorted(plan.calibration.groups[FLAT].items()):
        statistics = _statistics_key(FLAT, key)
        reference = plan.group_reference(FLAT, key)
        if key not in used:
            _not_used(plan, FLAT, key, ledger, reference)
            continue
        if group.supplied_master:
            supplied = Path(group.members[0])
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
            # Division uses a response normalized to unity without altering the
            # supplied master: (Light - calibration) / MasterFlat * median.
            _reused(plan, FLAT, key, ledger, reference, applicationNormalization=location)
            master_flats[key] = supplied
            application_scales[key] = location
            continue
        reused = _reused(plan, FLAT, key, ledger, reference)
        if reused is not None:
            master_flats[key] = reused
            application_scales[key] = 1.0
            continue
        pairing = plan.calibration.flats[key]
        expressions: list[FrameExpression] = []
        normalizations: list[float] = []
        calibration_sources: list[dict[str, Any]] = []
        for path in plan.group_paths(FLAT, key):
            info = plan.flat_info[path]
            flat_subtract: Path | None
            if pairing.dark is not None:
                flat_subtract = master_darks[pairing.dark]
                dark_bias_included = plan.dark_bias_included(pairing.dark)
                calibration_mode = (
                    "MATCHED_BIAS_INCLUDED_DARK"
                    if dark_bias_included
                    else "MATCHED_BIAS_SUBTRACTED_DARK_PLUS_MASTER_BIAS"
                )
                flat_subtract_info: FrameInfo | None = dark_domain_info[pairing.dark]
            elif pairing.bias is not None:
                flat_subtract = master_biases[pairing.bias]
                dark_bias_included = True
                calibration_mode = "BIAS"
                flat_subtract_info = bias_domain_info[pairing.bias]
            else:
                flat_subtract = None
                dark_bias_included = True
                calibration_mode = "UNCALIBRATED"
                flat_subtract_info = None
            flat_subtract_scale = (
                numeric_application_scale(
                    info,
                    flat_subtract_info,
                    target_label="raw Flat",
                    additive_label=("MasterDark" if pairing.dark is not None else "MasterBias"),
                )
                if flat_subtract_info is not None
                else 1.0
            )
            bias_terms: dict[str, Any] = {}
            if not dark_bias_included and pairing.bias is not None:
                bias_terms = {
                    "subtract_paths": (str(master_biases[pairing.bias]),),
                    "subtract_scales": (
                        numeric_application_scale(
                            info,
                            bias_domain_info[pairing.bias],
                            target_label="raw Flat",
                            additive_label="MasterBias",
                        ),
                    ),
                }
            subtract_path = str(flat_subtract) if flat_subtract is not None else None
            location = robust_location(
                FrameExpression(
                    source_path=str(path),
                    subtract_path=subtract_path,
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
                    subtract_path=subtract_path,
                    subtract_scale=flat_subtract_scale,
                    scale=1.0 / location,
                    **bias_terms,
                )
            )
            calibration_sources.append(
                {
                    "source": str(plan.display_path(path)),
                    "mode": calibration_mode,
                    "subtracted": (
                        plan.receipt_reference(dirs.root, flat_subtract) if flat_subtract is not None else None
                    ),
                    "targetNumericDomain": info.numeric_domain,
                    "additiveNumericDomain": (
                        flat_subtract_info.numeric_domain if flat_subtract_info is not None else None
                    ),
                    "applicationScale": flat_subtract_scale,
                    "applicationScaleSource": "normalized-unit-domain-ratio",
                }
            )
        filter_name = reference.filter_name
        label = _label(key)
        destination = dirs.masters / (
            f"master_flat_{_safe_token(filter_name)}" + (f"_{_safe_token(label)}" if label else "") + ".fits"
        )
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
                **cfa_metadata(reference),
            },
            parameters=parameters.integration,
            durable=parameters.durable_intermediates,
        )
        master_flats[key] = destination
        application_scales[key] = 1.0
        ledger.record(
            destination,
            "MASTER_FLAT",
            statistics=integration.statistics,
            sha256=integration.output_sha256,
            details={
                "filter": filter_name,
                **({"group": key} if label else {}),
                "normalizations": normalizations,
                "calibrationSources": calibration_sources,
            },
        )
        ledger.stage_statistics[statistics] = _integration_record(integration, dirs.root)
        ledger.stage_statistics[statistics]["mode"] = "BUILT_FROM_RAW"
    return master_flats, application_scales


def _cfa_flat_scaling(
    plan: _RunPlan,
    master_flats: dict[str, Path],
    application_scales: dict[str, float],
    parameters: PipelineParameters,
    ledger: _RunLedger,
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, tuple[float, float, float]]]:
    """Separate flat scaling factors for the colour channels of a Bayer
    filter: each channel is divided by the master flat normalized to its own
    channel median, so the flat's colour response does not tint the
    calibrated frame (PixInsight's "separate CFA flat scaling factors")."""

    patterns: dict[str, str] = {}
    for light_path, pairing in plan.calibration.lights.items():
        pattern = plan.light_cfa_pattern[plan.light_info[Path(light_path)].filter_name]
        if pattern is not None and pairing.flat is not None:
            patterns[pairing.flat] = pattern
    pattern_scales: dict[str, tuple[float, float, float, float]] = {}
    channel_medians_by_flat: dict[str, tuple[float, float, float]] = {}
    for key, pattern in sorted(patterns.items()):
        with FitsFrame(master_flats[key]) as flat_frame:
            medians = channel_medians(flat_frame.full_values(), pattern)
        if any(not math.isfinite(value) or value <= parameters.integration.division_floor for value in medians):
            raise CalibrationError(
                "FLAT_SIGNAL_INVALID",
                f"the master flat {key!r} of a Bayer filter has a colour channel without positive signal",
                path=str(master_flats[key]),
            )
        channel_medians_by_flat[key] = medians
        layout = CFA_PATTERNS[pattern]
        reference_level = application_scales[key]
        pattern_scales[key] = tuple(  # type: ignore[assignment]
            medians[layout[position]] / reference_level for position in range(4)
        )
        ledger.stage_statistics[_statistics_key(FLAT, key)]["cfaChannelMedians"] = {
            "pattern": pattern,
            "R": medians[0],
            "G": medians[1],
            "B": medians[2],
            "separateChannelScaling": True,
        }
    return pattern_scales, channel_medians_by_flat


def _build_calibration_masters(
    plan: _RunPlan, dirs: _StagingDirs, parameters: PipelineParameters, ledger: _RunLedger
) -> _CalibrationMasters:
    master_biases, bias_domain_info = _build_master_biases(plan, dirs, parameters, ledger)
    master_darks, dark_domain_info = _build_master_darks(plan, dirs, parameters, ledger)
    master_flats, application_scales = _build_master_flats(
        plan,
        dirs,
        parameters,
        ledger,
        master_biases=master_biases,
        bias_domain_info=bias_domain_info,
        master_darks=master_darks,
        dark_domain_info=dark_domain_info,
    )
    pattern_scales, channel_medians_by_flat = _cfa_flat_scaling(
        plan, master_flats, application_scales, parameters, ledger
    )
    return _CalibrationMasters(
        biases=master_biases,
        bias_domain_info=bias_domain_info,
        darks=master_darks,
        dark_domain_info=dark_domain_info,
        flats=master_flats,
        flat_application_scales=application_scales,
        flat_pattern_scales=pattern_scales,
        flat_channel_medians=channel_medians_by_flat,
    )
