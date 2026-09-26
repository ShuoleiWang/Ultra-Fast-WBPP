from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from lightframeqc.cfa import is_cfa_pattern

from .backends import BackendDescriptor, BackendRegistry, DeviceKind, StageKind
from .calibration.matching import BIAS as MATCH_BIAS
from .calibration.matching import DARK as MATCH_DARK
from .calibration.matching import FLAT as MATCH_FLAT
from .calibration.matching import LIGHT as MATCH_LIGHT
from .calibration.matching import CalibrationMatch, FrameTraits, match_calibration
from .calibration.policy import STRICT, MONO_STANDARD, same_metadata, cfa_for_workflow, metadata_changes, unknown
from .hardware import HardwareProfile, detect_hardware
from .models import (
    AssetRole,
    AssetStatus,
    FrameAsset,
    IssueSeverity,
    ProjectInventory,
    json_value,
)
from .recipe import Recipe, Requirement, SolverPolicy
from .solvers.registry import solver_backends


class PlanIssueCategory(StrEnum):
    INPUT = "INPUT"
    RECIPE = "RECIPE"
    CAPABILITY = "CAPABILITY"
    IMPLEMENTATION = "IMPLEMENTATION"


class StageState(StrEnum):
    READY = "READY"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass(frozen=True, slots=True)
class PlanIssue:
    code: str
    severity: IssueSeverity
    category: PlanIssueCategory
    stage: StageKind | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    blocks_contract: bool = False
    blocks_execution: bool = False

    def serializable(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "category": self.category.value,
            "stage": self.stage.value if self.stage else None,
            "message": self.message,
            "details": json_value(self.details, "planIssue.details"),
            "blocksContract": self.blocks_contract,
            "blocksExecution": self.blocks_execution,
        }


@dataclass(frozen=True, slots=True)
class PlanStage:
    stage: StageKind
    state: StageState
    backend_id: str | None
    message: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()

    def serializable(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "state": self.state.value,
            "backendId": self.backend_id,
            "message": self.message,
            "consumes": list(self.consumes),
            "produces": list(self.produces),
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    plan_id: str
    project_id: str
    recipe: Recipe
    hardware: HardwareProfile
    stages: tuple[PlanStage, ...]
    issues: tuple[PlanIssue, ...]
    contract_valid: bool
    execution_ready: bool
    schema_version: int = 1

    def serializable(self) -> dict[str, Any]:
        solver_stage = next(
            (stage for stage in self.stages if stage.stage is StageKind.SOLVER),
            None,
        )
        return {
            "schemaVersion": self.schema_version,
            "planId": self.plan_id,
            "projectId": self.project_id,
            "contractValid": self.contract_valid,
            "executionReady": self.execution_ready,
            "recipe": self.recipe.serializable(),
            "hardware": self.hardware.serializable(),
            "stages": [stage.serializable() for stage in self.stages],
            "issues": [issue.serializable() for issue in self.issues],
            "claims": {
                "inventoryImplemented": True,
                "pixelExecutionImplemented": True,
                "astrometricSolutionRequired": self.recipe.solver.policy
                == SolverPolicy.REQUIRED,
                "astrometricBackendReady": bool(
                    solver_stage is not None and solver_stage.state is StageState.READY
                ),
                "note": (
                    "READY means an executable backend passed its runtime probe. "
                    "Scientific gates, source identity checks, and WCS validation still "
                    "fail closed during execution."
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class RuntimeStageBackend:
    descriptor: BackendDescriptor
    accepted_options: frozenset[str] = frozenset()
    option_validator: Any = None

    def validate_options(self, options: dict[str, Any]) -> tuple[str, ...]:
        errors = [
            f"unknown {self.descriptor.stage.value.lower()} option: {key}"
            for key in sorted(set(options) - self.accepted_options)
        ]
        if callable(self.option_validator):
            errors.extend(self.option_validator(options))
        return tuple(errors)


# Backward-compatible import name.  Instances now describe probed executors,
# never a placeholder that claims a stage is contract-only.
ContractOnlyBackend = RuntimeStageBackend


def _internal_backend(
    *,
    backend_id: str,
    stage: StageKind,
    display_name: str,
    version: str,
    devices: tuple[DeviceKind, ...],
    capabilities: tuple[str, ...],
    probe: Any,
    accepted_options: frozenset[str] = frozenset(),
) -> RuntimeStageBackend:
    try:
        ready = bool(probe())
        reason = None if ready else "required callable was not found"
    except Exception as error:  # pragma: no cover - defensive import boundary
        ready = False
        reason = f"runtime probe failed: {type(error).__name__}: {error}"
    return RuntimeStageBackend(
        BackendDescriptor(
            backend_id=backend_id,
            stage=stage,
            display_name=display_name,
            version=version,
            available=ready,
            execution_ready=ready,
            devices=devices,
            capabilities=capabilities if ready else (),
            reason=reason,
            metadata={
                "probe": {
                    "kind": "python-callable",
                    "available": ready,
                    "executionReady": ready,
                }
            },
        ),
        accepted_options=accepted_options,
    )


def _runtime_backends() -> tuple[RuntimeStageBackend, ...]:
    """Probe the in-process executors used by :func:`run_e2e`.

    These probes deliberately check the actual callable composition root.  A
    module name or a type contract on its own does not make a stage ready.
    """

    def qc_ready() -> bool:
        from lightframeqc.measure import measure_paths
        from lightframeqc.quality_gate import evaluate_quality_gate

        return callable(measure_paths) and callable(evaluate_quality_gate)

    def calibration_ready() -> bool:
        from .stacking.integration import read_frame_info
        from .stacking.pipeline import run_portable_pipeline

        return callable(read_frame_info) and callable(run_portable_pipeline)

    def registration_ready() -> bool:
        from ufwbpp_registration import run_registration

        return callable(run_registration)

    def integration_ready() -> bool:
        from .stacking.metal_integration import integrate_registered_group
        from .stacking.pipeline import run_portable_pipeline

        return callable(integrate_registered_group) and callable(run_portable_pipeline)

    return (
        _internal_backend(
            backend_id="light-frame-qc",
            stage=StageKind.QUALITY_GATE,
            display_name="Ultra-Fast WBPP Light Frame Quality Gate",
            version="quality-gate-v1",
            devices=(DeviceKind.CPU,),
            capabilities=(
                "cloud-gate",
                "occlusion-gate",
                "focus-and-trailing-gate",
                "registration-evidence",
            ),
            probe=qc_ready,
        ),
        _internal_backend(
            backend_id="native-calibration",
            stage=StageKind.CALIBRATION,
            display_name="Ultra-Fast WBPP Portable FITS Calibration",
            version="portable-pixel-pipeline-v1",
            devices=(DeviceKind.CPU,),
            capabilities=("raw-flat", "raw-dark", "raw-bias", "bounded-memory-fits"),
            probe=calibration_ready,
        ),
        _internal_backend(
            backend_id="native-registration",
            stage=StageKind.REGISTRATION,
            display_name="Ultra-Fast WBPP Full-resolution Registration",
            version="registration-v1",
            devices=(DeviceKind.CPU,),
            capabilities=(
                "star-detection",
                "full-resolution-transform",
                "common-footprint",
            ),
            probe=registration_ready,
        ),
        _internal_backend(
            backend_id="native-integration",
            stage=StageKind.INTEGRATION,
            display_name="Ultra-Fast WBPP Portable Integration",
            version="portable-pixel-pipeline-v1",
            devices=(DeviceKind.CPU, DeviceKind.METAL),
            capabilities=(
                "weighted-integration",
                "rejection",
                "portable-cpu",
                "generic-apple-metal",
                "m3-pro-tuned",
            ),
            probe=integration_ready,
        ),
    )


def _native_drizzle_backend() -> RuntimeStageBackend:
    from .stacking.drizzle_native import SUPPORTED_KERNELS, SUPPORTED_SCALES
    from .native_kernels import DRIZZLE_KERNEL_ID, describe_native_kernels, load_native_kernels

    kernels = load_native_kernels()
    ready = kernels is not None and hasattr(kernels, "drizzle_band")
    description = describe_native_kernels()

    def validate(options: dict[str, Any]) -> tuple[str, ...]:
        errors: list[str] = []
        scale = options.get("scale", 2)
        drop_shrink = options.get("dropShrink", 0.9)
        kernel = options.get("kernel", "square")
        if isinstance(scale, bool) or scale not in SUPPORTED_SCALES:
            errors.append(f"scale must be one of {list(SUPPORTED_SCALES)}")
        if (
            isinstance(drop_shrink, bool)
            or not isinstance(drop_shrink, (int, float))
            or not 0.1 <= float(drop_shrink) <= 1.0
        ):
            errors.append("dropShrink must be numeric and in [0.1, 1]")
        if not isinstance(kernel, str) or kernel not in SUPPORTED_KERNELS:
            errors.append(f"kernel must be one of {list(SUPPORTED_KERNELS)}")
        if not isinstance(options.get("cfaDrizzle", False), bool):
            errors.append("cfaDrizzle must be boolean")
        return tuple(errors)

    return RuntimeStageBackend(
        BackendDescriptor(
            backend_id="native-drizzle",
            stage=StageKind.DRIZZLE,
            display_name="Ultra-Fast WBPP Native Drizzle",
            version=DRIZZLE_KERNEL_ID,
            available=ready,
            execution_ready=ready,
            devices=(DeviceKind.CPU,),
            capabilities=(
                "square-circular-gaussian-point-kernels",
                "scale-1-to-4",
                "drop-shrink",
                "integration-rejection-masks",
                "integration-normalization-and-weights",
                "cfa-drizzle",
                "science-receipt-v3",
            )
            if ready
            else (),
            reason=None if ready else (
                str(description.get("reason") or "the native kernel library is not loaded")
            ),
            metadata={"nativeKernels": description},
        ),
        accepted_options=frozenset({"scale", "dropShrink", "kernel", "cfaDrizzle"}),
        option_validator=validate,
    )


def default_registry() -> BackendRegistry:
    # External solvers are instantiated here, so their descriptors contain the
    # result of a real bounded process probe rather than PATH guesses.
    return BackendRegistry(
        [
            *_runtime_backends(),
            _native_drizzle_backend(),
            *solver_backends(),
        ]
    )


def solver_backend_science_ready(backend: Any) -> bool:
    """Whether a solver can satisfy the E2E catalog-correspondence gate.

    A successful executable probe is insufficient when the adapter cannot
    provide recomputable match/RMS/index evidence required by ``run_e2e``.
    """

    descriptor = backend.descriptor
    if not (descriptor.available and descriptor.execution_ready):
        return False
    if "catalog-correspondence-quality-v1" in descriptor.capabilities:
        return True
    evidence = descriptor.metadata.get("scientificQualityEvidence", {})
    return isinstance(evidence, dict) and evidence.get("status") in {
        "recomputed",
        "verified",
    }


def _known_same(left: object, right: object, unknown: object = "UNKNOWN") -> bool:
    """Require positive metadata evidence; unknown values never prove a match."""

    # Inventory preserves the acquisition header's spelling, whereas trusted
    # master overrides and the pixel reader canonicalize text to uppercase.
    # Use that same convention here so planning and execution agree.
    if isinstance(left, str):
        left = left.strip().upper()
    if isinstance(right, str):
        right = right.strip().upper()
    if left == unknown or right == unknown or left is None or right is None:
        return False
    return left == right


def base_compatible(light: FrameAsset, calibration: FrameAsset, workflow: str = STRICT) -> bool:
    return (
        light.status == AssetStatus.READY
        and calibration.status == AssetStatus.READY
        and light.width == calibration.width
        and light.height == calibration.height
        and light.channels == calibration.channels
        and all(same_metadata(getattr(light, name), getattr(calibration, name), workflow) for name in ("binning_x", "binning_y", "camera", "gain", "offset", "readout_mode"))
        and same_metadata(cfa_for_workflow(light.cfa_pattern, workflow), cfa_for_workflow(calibration.cfa_pattern, workflow), workflow, required=True)
        and (
            cfa_for_workflow(light.cfa_pattern, workflow) == "NONE"
            or is_cfa_pattern(cfa_for_workflow(light.cfa_pattern, workflow))
        )
    )


def flat_compatible(light: FrameAsset, flat: FrameAsset, workflow: str = STRICT) -> bool:
    return base_compatible(light, flat, workflow) and _known_same(light.filter_name, flat.filter_name)


def dark_compatible(light: FrameAsset, dark: FrameAsset, workflow: str = STRICT) -> bool:
    temperatures_known = light.temperature_celsius is not None and dark.temperature_celsius is not None
    return (
        base_compatible(light, dark, workflow)
        and light.exposure_seconds is not None
        and dark.exposure_seconds is not None
        and math.isclose(light.exposure_seconds, dark.exposure_seconds, rel_tol=0.0, abs_tol=1e-6)
        and ((temperatures_known and abs(light.temperature_celsius - dark.temperature_celsius) <= 3.0) or (not temperatures_known and workflow == MONO_STANDARD))
    )


def _calibration_candidates(
    inventory: ProjectInventory,
    raw_role: AssetRole,
    master_role: AssetRole,
    allow_masters: bool,
    assets: tuple[FrameAsset, ...] | None = None,
) -> tuple[FrameAsset, ...]:
    accepted = {raw_role}
    if allow_masters:
        accepted.add(master_role)
    return tuple(asset for asset in (assets or inventory.assets) if asset.role in accepted)


def effective_calibration_assets(
    inventory: ProjectInventory, recipe: Recipe
) -> tuple[tuple[FrameAsset, ...], list[PlanIssue], bool]:
    overrides = recipe.calibration.master_metadata_overrides
    if not overrides:
        return inventory.assets, [], True
    masters = tuple(asset for asset in inventory.assets if asset.is_master)
    by_digest: dict[str, list[FrameAsset]] = {}
    for asset in masters:
        digest = hashlib.sha256()
        try:
            with Path(asset.path).open("rb") as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    digest.update(chunk)
        except OSError:
            continue
        by_digest.setdefault("sha256:" + digest.hexdigest(), []).append(asset)
    replacements: dict[str, FrameAsset] = {}
    issues: list[PlanIssue] = []
    valid = True
    for override in overrides:
        matches = by_digest.get(override.source_sha256, [])
        if len(matches) != 1:
            valid = False
            issues.append(
                PlanIssue(
                    code="MASTER_METADATA_OVERRIDE_SOURCE_AMBIGUOUS",
                    severity=IssueSeverity.ERROR,
                    category=PlanIssueCategory.INPUT,
                    stage=StageKind.CALIBRATION,
                    message="A master metadata override must bind exactly one current supplied master.",
                    details={"matchingMasterCount": len(matches)},
                    blocks_contract=True,
                    blocks_execution=True,
                )
            )
            continue
        asset = matches[0]
        changes = metadata_changes(override)
        if "cfa_pattern" in changes and asset.cfa_explicit and changes["cfa_pattern"] != asset.cfa_pattern.strip().upper():
            valid = False
            issues.append(PlanIssue(code="MASTER_CFA_OVERRIDE_CONFLICT", severity=IssueSeverity.ERROR, category=PlanIssueCategory.INPUT, stage=StageKind.CALIBRATION, message="A master override cannot replace explicit CFA metadata.", blocks_contract=True, blocks_execution=True))
            continue
        replacements[asset.asset_id] = replace(asset, **changes)
    return (
        tuple(replacements.get(asset.asset_id, asset) for asset in inventory.assets),
        issues,
        valid,
    )


def asset_traits(
    asset: FrameAsset, kind: str, workflow: str, *, bias_included: bool | None = None
) -> FrameTraits:
    """An inventory asset as :mod:`ufwbpp.calibration.matching` pairs it."""

    return FrameTraits(
        path=asset.path,
        kind=kind,
        supplied_master=asset.is_master,
        shape=(asset.height, asset.width, asset.channels) if asset.width and asset.height else None,
        binning=(asset.binning_x, asset.binning_y),
        color=cfa_for_workflow(asset.cfa_pattern, workflow),
        filter_name=asset.filter_name if not unknown(asset.filter_name) else "UNKNOWN",
        exposure_seconds=asset.exposure_seconds,
        temperature_celsius=asset.temperature_celsius,
        camera=asset.camera,
        gain=asset.gain,
        offset=asset.offset,
        readout_mode=asset.readout_mode,
        keywords=asset.grouping_keywords,
        bias_included=True if kind == MATCH_DARK and not asset.is_master else bias_included,
    )


def match_inventory_calibration(
    assets: Sequence[FrameAsset],
    recipe: Recipe,
    dark_bias_included: Mapping[str, bool] | None = None,
) -> CalibrationMatch:
    """Pair the READY Lights of ``assets`` with the calibration the recipe
    allows, by WBPP's rules (see :mod:`ufwbpp.calibration.matching`).
    ``dark_bias_included`` declares, by path, whether a MasterDark includes
    the Bias; an undeclared one is taken to include it."""

    workflow = recipe.calibration.workflow
    kinds = (
        (MATCH_BIAS, AssetRole.BIAS, AssetRole.MASTER_BIAS, recipe.calibration.bias),
        (MATCH_DARK, AssetRole.DARK, AssetRole.MASTER_DARK, recipe.calibration.dark),
        (MATCH_FLAT, AssetRole.FLAT, AssetRole.MASTER_FLAT, recipe.calibration.flat),
    )
    calibration = [
        asset_traits(asset, kind, workflow, bias_included=(dark_bias_included or {}).get(asset.path))
        for kind, raw_role, master_role, requirement in kinds
        if requirement != Requirement.DISABLED
        for asset in assets
        if asset.status == AssetStatus.READY
        and (asset.role is raw_role or (asset.role is master_role and recipe.calibration.allow_masters))
    ]
    lights = [
        asset_traits(asset, MATCH_LIGHT, workflow)
        for asset in assets
        if asset.role == AssetRole.LIGHT and asset.status == AssetStatus.READY
    ]
    return match_calibration(lights, calibration)


def _wbpp_calibration_issues(
    assets: Sequence[FrameAsset],
    recipe: Recipe,
    dark_bias_included: Mapping[str, bool] | None = None,
) -> tuple[list[PlanIssue], bool]:
    """The standard workflow's plan issues: WBPP's pairing, with a refused
    pairing (size, binning, colour) and a REQUIRED kind without any fitting
    frame blocking, every other difference WBPP accepts a warning."""

    match = match_inventory_calibration(assets, recipe, dark_bias_included)
    group_of = {asset.path: asset.group_id for asset in assets}
    requirements = {
        MATCH_FLAT: recipe.calibration.flat,
        MATCH_DARK: recipe.calibration.dark,
        MATCH_BIAS: recipe.calibration.bias,
    }
    missing_codes = {"FLAT_MISSING": MATCH_FLAT, "DARK_MISSING": MATCH_DARK, "BIAS_MISSING": MATCH_BIAS}
    collected: dict[tuple[str, str, IssueSeverity], set[str]] = {}

    def add(code: str, message: str, severity: IssueSeverity, light: str) -> None:
        collected.setdefault((code, message, severity), set()).add(group_of.get(light, ""))

    for light, pairing in match.lights.items():
        for error in pairing.errors:
            add(error.code, error.message, IssueSeverity.ERROR, light)
        for warning in pairing.warnings:
            kind = missing_codes.get(warning.code)
            if kind is None:
                add(warning.code, warning.message, IssueSeverity.WARNING, light)
                continue
            if requirements[kind] == Requirement.REQUIRED:
                # The matcher's message ends with what processing would do
                # without the frame; a required kind blocks instead.
                cause = warning.message.split(";", 1)[0]
                add(f"{kind}_MATCH_MISSING", f"{cause}; the recipe requires one", IssueSeverity.ERROR, light)
            else:
                add(f"{kind}_MATCH_MISSING", warning.message, IssueSeverity.WARNING, light)
        refused = {error.code.split("_", 1)[0] for error in pairing.errors}
        for kind, attribute in ((MATCH_FLAT, "flat"), (MATCH_DARK, "dark"), (MATCH_BIAS, "bias")):
            if (
                requirements[kind] == Requirement.REQUIRED
                and getattr(pairing, attribute) is None
                and kind not in refused
                and not any(missing_codes.get(item.code) == kind for item in pairing.warnings)
            ):
                add(
                    f"{kind}_MATCH_MISSING",
                    f"no {kind.lower()} fits these lights and the recipe requires one",
                    IssueSeverity.ERROR,
                    light,
                )
    for flat_key, pairing in match.flats.items():
        for warning in pairing.warnings:
            collected.setdefault((warning.code, warning.message, IssueSeverity.WARNING), set())
    issues: list[PlanIssue] = []
    valid = True
    for (code, message, severity), groups in sorted(collected.items(), key=lambda item: (item[0][0], item[0][1])):
        blocking = severity == IssueSeverity.ERROR
        valid = valid and not blocking
        issues.append(
            PlanIssue(
                code=code,
                severity=severity,
                category=PlanIssueCategory.INPUT,
                stage=StageKind.CALIBRATION,
                message=message,
                details={"lightGroups": sorted(group for group in groups if group)},
                blocks_contract=blocking,
                blocks_execution=blocking,
            )
        )
    return issues, valid


def calibration_match_issues(
    inventory: ProjectInventory,
    recipe: Recipe,
    dark_bias_included: Mapping[str, bool] | None = None,
) -> tuple[list[PlanIssue], bool]:
    effective_assets, override_issues, valid = effective_calibration_assets(
        inventory, recipe
    )
    if recipe.calibration.workflow == MONO_STANDARD:
        # WBPP-compatible pairing (grouping keywords, closest Dark, other
        # settings reported); the strict workflow below keeps its exact rules.
        issues, match_valid = _wbpp_calibration_issues(effective_assets, recipe, dark_bias_included)
        return [*override_issues, *issues], valid and match_valid
    issues: list[PlanIssue] = list(override_issues)
    lights = tuple(
        asset
        for asset in effective_assets
        if asset.role == AssetRole.LIGHT and asset.status == AssetStatus.READY
    )
    policies = (
        (
            "Flat",
            recipe.calibration.flat,
            AssetRole.FLAT,
            AssetRole.MASTER_FLAT,
            flat_compatible,
        ),
        (
            "Dark",
            recipe.calibration.dark,
            AssetRole.DARK,
            AssetRole.MASTER_DARK,
            dark_compatible,
        ),
        (
            "Bias",
            recipe.calibration.bias,
            AssetRole.BIAS,
            AssetRole.MASTER_BIAS,
            base_compatible,
        ),
    )
    for label, requirement, raw_role, master_role, compatible in policies:
        if requirement == Requirement.DISABLED:
            continue
        candidates = _calibration_candidates(
            inventory,
            raw_role,
            master_role,
            recipe.calibration.allow_masters,
            effective_assets,
        )
        if label == "Bias" and recipe.calibration.workflow == MONO_STANDARD and requirement == Requirement.OPTIONAL and not candidates:
            # Standard dependency checks determine whether a Bias is actually
            # needed; Bias-inclusive Darks do not require a separate Bias.
            continue
        unmatched_groups: dict[str, FrameAsset] = {}
        ambiguous_groups: dict[str, FrameAsset] = {}
        for light in lights:
            # Grouping keywords (NIGHT_1, SESSION_2) pair a Light with its own
            # night's frames: a different value excludes a candidate and the
            # most matching keywords win (WBPP), before ambiguity is judged.
            light_keywords = dict(light.grouping_keywords)
            matching = tuple(
                candidate
                for candidate in candidates
                if compatible(light, candidate, recipe.calibration.workflow)
                and all(light_keywords.get(name, value) == value for name, value in candidate.grouping_keywords)
            )
            if matching:
                scores = [sum(light_keywords.get(name) == value for name, value in candidate.grouping_keywords) for candidate in matching]
                matching = tuple(candidate for candidate, score in zip(matching, scores) if score == max(scores))
            if not matching:
                unmatched_groups.setdefault(light.group_id, light)
                continue
            matching_raw = tuple(
                candidate for candidate in matching if candidate.role is raw_role
            )
            matching_masters = tuple(
                candidate for candidate in matching if candidate.role is master_role
            )
            if len(matching_masters) > 1 or (matching_raw and matching_masters):
                ambiguous_groups.setdefault(light.group_id, light)
        if ambiguous_groups:
            valid = False
            issues.append(
                PlanIssue(
                    code=f"{label.upper()}_MATCH_AMBIGUOUS",
                    severity=IssueSeverity.ERROR,
                    category=PlanIssueCategory.INPUT,
                    stage=StageKind.CALIBRATION,
                    message=(
                        f"{len(ambiguous_groups)} Light group(s) match both raw and "
                        f"master {label} sources, or more than one supplied master."
                    ),
                    details={
                        "requirement": requirement.value,
                        "lightGroups": sorted(ambiguous_groups),
                        "candidateCount": len(candidates),
                    },
                    blocks_contract=True,
                    blocks_execution=True,
                )
            )
        if unmatched_groups:
            severity = (
                IssueSeverity.ERROR
                if requirement == Requirement.REQUIRED
                else IssueSeverity.WARNING
            )
            if severity == IssueSeverity.ERROR:
                valid = False
            issues.append(
                PlanIssue(
                    code=f"{label.upper()}_MATCH_MISSING",
                    severity=severity,
                    category=PlanIssueCategory.INPUT,
                    stage=StageKind.CALIBRATION,
                    message=(
                        f"{len(unmatched_groups)} Light group(s) have no metadata-compatible {label}."
                    ),
                    details={
                        "requirement": requirement.value,
                        "lightGroups": sorted(unmatched_groups),
                        "candidateCount": len(candidates),
                    },
                    blocks_contract=severity == IssueSeverity.ERROR,
                    blocks_execution=severity == IssueSeverity.ERROR,
                )
            )
    return issues, valid


def _backend_unavailable_issue(
    stage: StageKind, backend_id: str, reason: str | None
) -> PlanIssue:
    return PlanIssue(
        code="BACKEND_NOT_EXECUTION_READY",
        severity=IssueSeverity.ERROR,
        category=PlanIssueCategory.CAPABILITY,
        stage=stage,
        message=(
            f"{stage.value} backend {backend_id} did not pass its runtime probe"
            + (f": {reason}" if reason else ".")
        ),
        details={"backendId": backend_id},
        blocks_execution=True,
    )


def _stage_from_backend(
    *,
    registry: BackendRegistry,
    stage: StageKind,
    backend_id: str,
    input_valid: bool,
    ready_message: str,
    invalid_message: str,
    consumes: tuple[str, ...],
    produces: tuple[str, ...],
) -> tuple[PlanStage, PlanIssue | None]:
    backend = registry.get(backend_id)
    if not input_valid:
        return (
            PlanStage(
                stage,
                StageState.BLOCKED,
                backend_id if backend is not None else None,
                invalid_message,
                consumes,
                produces,
            ),
            None,
        )
    if backend is None:
        return (
            PlanStage(
                stage,
                StageState.BLOCKED,
                None,
                f"Required backend {backend_id} is not registered.",
                consumes,
                produces,
            ),
            _backend_unavailable_issue(stage, backend_id, "backend is not registered"),
        )
    descriptor = backend.descriptor
    if not (descriptor.available and descriptor.execution_ready):
        return (
            PlanStage(
                stage,
                StageState.BLOCKED,
                backend_id,
                descriptor.reason or "Backend did not pass its runtime probe.",
                consumes,
                produces,
            ),
            _backend_unavailable_issue(stage, backend_id, descriptor.reason),
        )
    return (
        PlanStage(
            stage,
            StageState.READY,
            backend_id,
            ready_message,
            consumes,
            produces,
        ),
        None,
    )


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_plan(
    inventory: ProjectInventory,
    recipe: Recipe | None = None,
    *,
    hardware: HardwareProfile | None = None,
    registry: BackendRegistry | None = None,
) -> ExecutionPlan:
    recipe = recipe or Recipe()
    hardware = hardware or detect_hardware()
    registry = registry or default_registry()
    issues: list[PlanIssue] = []
    stages: list[PlanStage] = []
    contract_valid = True

    input_errors = [
        issue for issue in inventory.issues if issue.severity == IssueSeverity.ERROR
    ]
    if input_errors:
        contract_valid = False
        for issue in input_errors:
            issues.append(
                PlanIssue(
                    code=issue.code,
                    severity=issue.severity,
                    category=PlanIssueCategory.INPUT,
                    stage=None,
                    message=issue.message,
                    details={"path": issue.path} if issue.path else {},
                    blocks_contract=True,
                    blocks_execution=True,
                )
            )

    unsupported_formats = sorted(
        {
            asset.format
            for asset in inventory.assets
            if asset.status == AssetStatus.READY and asset.format not in {"FITS", "XISF"}
        }
    )
    if unsupported_formats:
        contract_valid = False
        issues.append(
            PlanIssue(
                code="PIXEL_FORMAT_UNSUPPORTED",
                severity=IssueSeverity.ERROR,
                category=PlanIssueCategory.CAPABILITY,
                stage=StageKind.CALIBRATION,
                message=(
                    "The executable pixel path accepts FITS and bounded-staging XISF; remove "
                    f"these input formats: {', '.join(unsupported_formats)}."
                ),
                details={"formats": unsupported_formats},
                blocks_contract=True,
                blocks_execution=True,
            )
        )

    quality_stage, quality_issue = _stage_from_backend(
        registry=registry,
        stage=StageKind.QUALITY_GATE,
        backend_id="light-frame-qc",
        input_valid=not input_errors,
        ready_message="Light-frame QC and the production quality gate are connected.",
        invalid_message="Input inventory errors block quality gating.",
        consumes=("raw-light",),
        produces=("quality-decision",),
    )
    stages.append(quality_stage)
    if quality_issue is not None:
        issues.append(quality_issue)

    calibration_issues, calibration_valid = calibration_match_issues(inventory, recipe)
    issues.extend(calibration_issues)
    contract_valid = contract_valid and calibration_valid
    calibration_stage, calibration_backend_issue = _stage_from_backend(
        registry=registry,
        stage=StageKind.CALIBRATION,
        backend_id="native-calibration",
        input_valid=calibration_valid and not unsupported_formats,
        ready_message="Portable bounded-memory FITS calibration is ready.",
        invalid_message="Calibration input matching or pixel-format validation failed.",
        consumes=("quality-approved-light", "flat", "dark", "bias"),
        produces=("calibrated-light",),
    )
    stages.append(calibration_stage)
    if calibration_backend_issue is not None:
        issues.append(calibration_backend_issue)

    upstream_valid = calibration_valid and not unsupported_formats and not input_errors
    for stage, backend_id, ready_message, consumes, produces in (
        (
            StageKind.REGISTRATION,
            "native-registration",
            "Full-resolution star registration is ready.",
            ("calibrated-light",),
            ("registered-light", "geometric-transform"),
        ),
        (
            StageKind.INTEGRATION,
            "native-integration",
            "Portable CPU/Metal integration dispatch is ready.",
            ("registered-light",),
            ("linear-master", "rejection-map"),
        ),
    ):
        planned_stage, backend_issue = _stage_from_backend(
            registry=registry,
            stage=stage,
            backend_id=backend_id,
            input_valid=upstream_valid,
            ready_message=ready_message,
            invalid_message="An upstream input or calibration gate is blocked.",
            consumes=consumes,
            produces=produces,
        )
        stages.append(planned_stage)
        if backend_issue is not None:
            issues.append(backend_issue)

    if recipe.drizzle.enabled:
        backend = registry.choose(StageKind.DRIZZLE, recipe.drizzle.backend)
        if backend is None:
            contract_valid = False
            issues.append(
                PlanIssue(
                    code="DRIZZLE_BACKEND_UNKNOWN",
                    severity=IssueSeverity.ERROR,
                    category=PlanIssueCategory.RECIPE,
                    stage=StageKind.DRIZZLE,
                    message=f"no drizzle backend matches {recipe.drizzle.backend!r}",
                    blocks_contract=True,
                    blocks_execution=True,
                )
            )
            stages.append(
                PlanStage(
                    StageKind.DRIZZLE,
                    StageState.BLOCKED,
                    None,
                    "Requested drizzle backend is unknown.",
                )
            )
        else:
            options = {
                "scale": recipe.drizzle.scale,
                "dropShrink": recipe.drizzle.drop_shrink,
                "cfaDrizzle": recipe.drizzle.cfa_drizzle,
                "kernel": recipe.drizzle.kernel,
            }
            option_errors = backend.validate_options(options)
            if option_errors:
                contract_valid = False
                for message in option_errors:
                    issues.append(
                        PlanIssue(
                            code="DRIZZLE_CAPABILITY_MISMATCH",
                            severity=IssueSeverity.ERROR,
                            category=PlanIssueCategory.CAPABILITY,
                            stage=StageKind.DRIZZLE,
                            message=message,
                            details={"backendId": backend.descriptor.backend_id},
                            blocks_contract=True,
                            blocks_execution=True,
                        )
                    )
            ready = backend.descriptor.available and backend.descriptor.execution_ready
            stages.append(
                PlanStage(
                    StageKind.DRIZZLE,
                    StageState.READY
                    if ready and not option_errors
                    else StageState.BLOCKED
                    if option_errors
                    else StageState.BLOCKED,
                    backend.descriptor.backend_id,
                    "Drizzle capability validated."
                    if ready and not option_errors
                    else backend.descriptor.reason or "Drizzle is unavailable.",
                    consumes=("registered-light", "geometric-transform", "rejection-map"),
                    produces=("drizzled-linear-master",),
                )
            )
            if not ready:
                issues.append(
                    _backend_unavailable_issue(
                        StageKind.DRIZZLE,
                        backend.descriptor.backend_id,
                        backend.descriptor.reason,
                    )
                )
    else:
        stages.append(
            PlanStage(
                StageKind.DRIZZLE,
                StageState.SKIPPED,
                None,
                "Drizzle is disabled by recipe.",
            )
        )

    if recipe.solver.policy == SolverPolicy.DISABLED:
        stages.append(
            PlanStage(
                StageKind.SOLVER,
                StageState.SKIPPED,
                None,
                "Astrometric solving was explicitly disabled; output WCS is not guaranteed.",
            )
        )
        issues.append(
            PlanIssue(
                code="ASTROMETRY_DISABLED",
                severity=IssueSeverity.WARNING,
                category=PlanIssueCategory.RECIPE,
                stage=StageKind.SOLVER,
                message="The output may require manual astrometric solving.",
            )
        )
    else:
        if recipe.solver.backend == "auto":
            candidates = registry.for_stage(StageKind.SOLVER)
            backend = next(
                (
                    candidate
                    for candidate in candidates
                    if solver_backend_science_ready(candidate)
                ),
                None,
            )
            if backend is None:
                backend = next(iter(candidates), None)
        else:
            backend = registry.choose(StageKind.SOLVER, recipe.solver.backend)
        if backend is None:
            contract_valid = False
            issues.append(
                PlanIssue(
                    code="SOLVER_BACKEND_UNKNOWN",
                    severity=IssueSeverity.ERROR,
                    category=PlanIssueCategory.RECIPE,
                    stage=StageKind.SOLVER,
                    message=f"no solver backend matches {recipe.solver.backend!r}",
                    blocks_contract=True,
                    blocks_execution=True,
                )
            )
            stages.append(
                PlanStage(
                    StageKind.SOLVER,
                    StageState.BLOCKED,
                    None,
                    "Required solver backend is unknown.",
                )
            )
        else:
            process_ready = (
                backend.descriptor.available and backend.descriptor.execution_ready
            )
            ready = solver_backend_science_ready(backend)
            stages.append(
                PlanStage(
                    StageKind.SOLVER,
                    StageState.READY if ready else StageState.BLOCKED,
                    backend.descriptor.backend_id,
                    "Solver will be followed by fail-closed WCS validation."
                    if ready
                    else (
                        "Solver process is available but does not provide the "
                        "catalog-correspondence quality evidence required by E2E."
                        if process_ready
                        else backend.descriptor.reason or "Solver is unavailable."
                    ),
                    consumes=(("drizzled-linear-master",) if recipe.drizzle.enabled else ("linear-master",)),
                    produces=("wcs-solved-master", "wcs-validation"),
                )
            )
            if not ready:
                issues.append(
                    _backend_unavailable_issue(
                        StageKind.SOLVER,
                        backend.descriptor.backend_id,
                        (
                            "catalog-correspondence quality evidence is unavailable"
                            if process_ready
                            else backend.descriptor.reason
                        ),
                    )
                )

    execution_ready = contract_valid and all(
        stage.state in {StageState.READY, StageState.SKIPPED} for stage in stages
    )
    plan_payload = {
        "projectId": inventory.project_id,
        "recipe": recipe.serializable(),
        "hardware": hardware.serializable(),
        "stages": [stage.serializable() for stage in stages],
    }
    return ExecutionPlan(
        plan_id=_digest(plan_payload),
        project_id=inventory.project_id,
        recipe=recipe,
        hardware=hardware,
        stages=tuple(stages),
        issues=tuple(issues),
        contract_valid=contract_valid,
        execution_ready=execution_ready,
    )


__all__ = [
    "ContractOnlyBackend",
    "ExecutionPlan",
    "PlanIssue",
    "PlanIssueCategory",
    "PlanStage",
    "RuntimeStageBackend",
    "StageState",
    "build_plan",
    "default_registry",
    "solver_backend_science_ready",
]
