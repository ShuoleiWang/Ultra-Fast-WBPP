"""Composition helpers for the public headless and worker entry points.

This module is intentionally small: it turns an already inspected inventory
into the explicit :class:`~openastroflow_engine.e2e.E2ERequest` consumed by the
pixel pipeline, and it selects only solver adapters that passed their real
runtime probes.  Unsupported recipe semantics fail before any pixel work.
"""

from __future__ import annotations

from .calibration_policy import MONO_STANDARD

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .backends import BackendRegistry, StageKind
from .calibration import IntegrationParameters
from .e2e import (
    DrizzleOptions,
    E2ERequest,
    IntegrationMode,
    ReviewApproval,
)
from .hardware import HardwareProfile, detect_hardware
from .models import AssetRole, AssetStatus, IssueSeverity, ProjectInventory
from .performance_profile import select_execution_tuning
from .pixel_pipeline import (
    MasterMetadataOverride,
    PipelineParameters,
    RawFrameMetadataOverride,
)
from .local_normalization import LocalNormalizationParameters
from .global_normalization import GlobalNormalizationParameters
from .planning import (
    ExecutionPlan,
    build_plan,
    default_registry,
    solver_backend_science_ready,
)
from .recipe import Recipe, Requirement, SolverPolicy
from .project_e2e import ProjectE2ERequest
from .solver import SolverBackend


class RuntimeConfigurationError(RuntimeError):
    """Stable fail-closed error raised before E2E pixel execution starts."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def inventory_manifest_sha256(inventory: ProjectInventory) -> str:
    """Return the raw SHA-256 of the canonical inventory snapshot.

    The digest includes resolved paths, role/metadata probes, source stat
    identities, and inventory issues.  Pixel execution performs stronger full
    content hashes before publication; this manifest digest binds planning.
    """

    encoded = json.dumps(
        inventory.serializable(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _recipe_sha256(recipe: Recipe) -> str:
    value = recipe.serializable()
    # Approval evidence is verified against this digest and therefore cannot
    # be part of the science-contract digest itself.
    value.pop("reviewApprovals", None)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def pixel_backend_for_hardware_profile(profile: str | None) -> str:
    """Map one canonical wire profile to the pixel executor it authorizes."""

    if profile is None:
        return "auto"
    try:
        return {
            "portable-cpu": "portable-cpu",
            "generic-arm64-cpu": "portable-cpu",
            "windows-cpu": "portable-cpu",
            "generic-apple-metal": "generic-apple-metal",
            "m3-pro-tuned": "m3-pro-tuned",
        }[profile]
    except KeyError as error:
        raise RuntimeConfigurationError(
            "HARDWARE_PROFILE_UNMAPPABLE",
            f"canonical hardware profile has no pixel backend: {profile}",
        ) from error


def _ready_assets(inventory: ProjectInventory, role: AssetRole) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                asset.path
                for asset in inventory.assets
                if asset.role is role and asset.status is AssetStatus.READY
            ),
            key=lambda value: value.casefold(),
        )
    )


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_e2e_inventory(inventory: ProjectInventory, recipe: Recipe) -> None:
    errors = [
        issue
        for issue in inventory.issues
        if issue.severity is IssueSeverity.ERROR
    ]
    if errors:
        codes = ", ".join(sorted({issue.code for issue in errors}))
        raise RuntimeConfigurationError(
            "INVENTORY_NOT_EXECUTABLE",
            f"inventory contains blocking errors: {codes}",
        )
    not_ready = [asset.path for asset in inventory.assets if asset.status is not AssetStatus.READY]
    if not_ready:
        raise RuntimeConfigurationError(
            "INVENTORY_NOT_EXECUTABLE",
            f"{len(not_ready)} frame(s) are conflicted or unreadable",
        )
    unknown = _ready_assets(inventory, AssetRole.UNKNOWN)
    if unknown:
        raise RuntimeConfigurationError(
            "FRAME_ROLE_UNKNOWN",
            f"{len(unknown)} frame(s) have no trustworthy acquisition role",
        )
    unsupported_formats = sorted(
        {asset.format for asset in inventory.assets if asset.format not in {"FITS", "XISF"}}
    )
    if unsupported_formats:
        raise RuntimeConfigurationError(
            "PIXEL_FORMAT_UNSUPPORTED",
            "the executable pixel path accepts FITS/XISF only; found "
            + ", ".join(unsupported_formats),
        )
    masters = tuple(
        asset.path
        for asset in inventory.assets
        if asset.role
        in {AssetRole.MASTER_FLAT, AssetRole.MASTER_DARK, AssetRole.MASTER_BIAS}
    )
    if masters and not recipe.calibration.allow_masters:
        raise RuntimeConfigurationError(
            "MASTER_CALIBRATION_INPUT_DISABLED",
            "the recipe disables supplied calibration masters",
        )
    dark_overrides = {
        item.source_sha256: item.bias_included
        for item in recipe.calibration.master_metadata_overrides
    }
    for path in _ready_assets(inventory, AssetRole.MASTER_DARK):
        digest = _file_sha256(path)
        if recipe.calibration.workflow != MONO_STANDARD and (digest not in dark_overrides or dark_overrides[digest] is None):
            raise RuntimeConfigurationError(
                "MASTER_DARK_BIAS_SEMANTICS_REQUIRED",
                "every supplied MasterDark requires a hash-bound boolean biasIncluded override",
            )
    if recipe.output_format != "FITS":
        raise RuntimeConfigurationError(
            "OUTPUT_FORMAT_UNSUPPORTED", "the E2E executor currently publishes FITS"
        )
    if recipe.overwrite:
        raise RuntimeConfigurationError(
            "OVERWRITE_UNSUPPORTED",
            "E2E publication is create-only; recipe.overwrite must be false",
        )
    if recipe.solver.policy is not SolverPolicy.REQUIRED:
        raise RuntimeConfigurationError(
            "ASTROMETRY_REQUIRED",
            "the product E2E entry point requires a newly solved and verified WCS",
        )
    if recipe.calibration.flat is Requirement.DISABLED:
        raise RuntimeConfigurationError(
            "FLAT_DISABLED_UNSUPPORTED", "the raw E2E path always calibrates with Flats"
        )
    if recipe.calibration.bias is Requirement.DISABLED:
        raise RuntimeConfigurationError(
            "BIAS_DISABLED_UNSUPPORTED", "the raw E2E path always calibrates with Bias frames"
        )
    if not _ready_assets(inventory, AssetRole.LIGHT):
        raise RuntimeConfigurationError("NO_LIGHTS", "at least one Light frame is required")
    raw_assets = [
        asset
        for asset in inventory.assets
        if asset.status is AssetStatus.READY
        and asset.role in {AssetRole.LIGHT, AssetRole.FLAT, AssetRole.DARK, AssetRole.BIAS}
    ]
    raw_by_digest: dict[str, list[Any]] = {}
    for asset in raw_assets:
        raw_by_digest.setdefault(_file_sha256(asset.path), []).append(asset)
    raw_overrides = {
        item.source_sha256: item.cfa_pattern.strip().upper()
        for item in recipe.raw_frame_metadata_overrides
    }
    for digest in raw_overrides:
        if len(raw_by_digest.get(digest, ())) != 1:
            raise RuntimeConfigurationError(
                "RAW_FRAME_METADATA_OVERRIDE_SOURCE_AMBIGUOUS",
                "each raw-frame override must bind exactly one current raw source",
            )
    for asset in raw_assets:
        observed = asset.cfa_pattern.strip().upper()
        digest = _file_sha256(asset.path)
        confirmed = raw_overrides.get(digest)
        if asset.cfa_explicit:
            if observed != "NONE":
                raise RuntimeConfigurationError(
                    "CFA_PIXEL_PIPELINE_UNSUPPORTED",
                    f"explicit CFA/Bayer metadata on raw {asset.role.value} requires a phase-preserving pipeline that is not implemented",
                )
            if confirmed is not None and confirmed != observed:
                raise RuntimeConfigurationError(
                    "RAW_CFA_OVERRIDE_CONFLICT",
                    "a raw-frame override cannot replace explicit CFA metadata",
                )
        else:
            if confirmed is None and recipe.calibration.workflow == MONO_STANDARD:
                confirmed = "NONE"
            if confirmed is None:
                raise RuntimeConfigurationError(
                    "CFA_CONFIRMATION_REQUIRED",
                    f"raw {asset.role.value} CFA metadata is absent; confirm mono/CFA with a hash-bound rawFrameMetadataOverrides entry",
                )
            if confirmed != "NONE":
                raise RuntimeConfigurationError(
                    "CFA_PIXEL_PIPELINE_UNSUPPORTED",
                    f"confirmed CFA/Bayer raw {asset.role.value} is not supported by the v1 pixel pipeline",
                )
    if not (
        _ready_assets(inventory, AssetRole.FLAT)
        or _ready_assets(inventory, AssetRole.MASTER_FLAT)
    ):
        raise RuntimeConfigurationError(
            "NO_FLATS", "raw Flats or supplied MasterFlats are required"
        )
    raw_biases = _ready_assets(inventory, AssetRole.BIAS)
    master_biases = _ready_assets(inventory, AssetRole.MASTER_BIAS)
    if len(master_biases) > 1 or (raw_biases and master_biases) or (not raw_biases and not master_biases and recipe.calibration.workflow != MONO_STANDARD):
        raise RuntimeConfigurationError(
            "BIAS_SOURCE_AMBIGUOUS",
            "supply raw Bias frames or exactly one MasterBias",
        )
    if (
        recipe.calibration.dark is Requirement.REQUIRED
        and not (
            _ready_assets(inventory, AssetRole.DARK)
            or _ready_assets(inventory, AssetRole.MASTER_DARK)
        )
    ):
        raise RuntimeConfigurationError(
            "NO_DARKS", "the recipe requires raw Darks or supplied MasterDarks"
        )


def build_e2e_request(
    inventory: ProjectInventory,
    recipe: Recipe,
    output_directory: str | Path,
    *,
    workers: int | None = None,
    hardware: HardwareProfile | None = None,
    ra_hint_degrees: float | None = None,
    dec_hint_degrees: float | None = None,
    field_of_view_degrees: float | None = None,
    search_radius_degrees: float | None = None,
    requested_hardware_profile: str | None = None,
) -> E2ERequest:
    """Build one explicit, role-separated E2E request from an inventory."""

    validate_e2e_inventory(inventory, recipe)
    hardware = hardware or detect_hardware()
    tuning = select_execution_tuning(hardware)
    selected_workers = tuning.qc_workers if workers is None else workers
    if isinstance(selected_workers, bool) or not isinstance(selected_workers, int) or selected_workers < 1:
        raise RuntimeConfigurationError("WORKER_COUNT_INVALID", "workers must be positive")
    integration = replace(
        IntegrationParameters(), max_memory_bytes=tuning.integration_memory_bytes
    )
    pipeline = replace(
        PipelineParameters(),
        calibration_workflow=recipe.calibration.workflow,
        integration=integration,
        registration_memory_bytes=tuning.registration_memory_bytes,
        local_normalization=LocalNormalizationParameters(
            enabled=recipe.local_normalization.enabled,
            tile_size_pixels=recipe.local_normalization.tile_size_pixels,
        ),
        global_normalization=GlobalNormalizationParameters(
            enabled=not recipe.local_normalization.enabled,
        ),
        ordinary_integration_backend=pixel_backend_for_hardware_profile(
            requested_hardware_profile
        ),
        master_metadata_overrides=tuple(
            MasterMetadataOverride(
                source_sha256=item.source_sha256,
                camera=item.camera,
                gain=item.gain,
                offset=item.offset,
                binning_x=item.binning_x,
                binning_y=item.binning_y,
                filter_name=item.filter_name,
                cfa_pattern=item.cfa_pattern,
                readout_mode=item.readout_mode,
                temperature_celsius=item.temperature_celsius,
                exposure_seconds=item.exposure_seconds,
                bias_included=item.bias_included,
                numeric_domain=item.numeric_domain,
                normalized_unit_scale=item.normalized_unit_scale,
            )
            for item in recipe.calibration.master_metadata_overrides
        ),
        raw_frame_metadata_overrides=tuple(
            RawFrameMetadataOverride(
                source_sha256=item.source_sha256,
                cfa_pattern=item.cfa_pattern,
            )
            for item in recipe.raw_frame_metadata_overrides
        ),
    )
    drizzle = DrizzleOptions(
        scale=recipe.drizzle.scale,
        pixfrac=recipe.drizzle.drop_shrink,
        kernel="square",
        max_working_set_bytes=tuning.integration_memory_bytes,
    )
    darks = (
        ()
        if recipe.calibration.dark is Requirement.DISABLED
        else _ready_assets(inventory, AssetRole.DARK)
    )
    return E2ERequest(
        light_files=_ready_assets(inventory, AssetRole.LIGHT),
        flat_files=_ready_assets(inventory, AssetRole.FLAT),
        dark_files=darks,
        bias_files=_ready_assets(inventory, AssetRole.BIAS),
        master_bias_files=_ready_assets(inventory, AssetRole.MASTER_BIAS),
        master_dark_files=(
            ()
            if recipe.calibration.dark is Requirement.DISABLED
            else _ready_assets(inventory, AssetRole.MASTER_DARK)
        ),
        master_flat_files=_ready_assets(inventory, AssetRole.MASTER_FLAT),
        output_directory=str(Path(output_directory).expanduser()),
        integration_mode=(
            IntegrationMode.DRIZZLE
            if recipe.drizzle.enabled
            else IntegrationMode.ORDINARY
        ),
        workers=selected_workers,
        pipeline_parameters=pipeline,
        drizzle=drizzle,
        ra_hint_degrees=ra_hint_degrees,
        dec_hint_degrees=dec_hint_degrees,
        field_of_view_degrees=field_of_view_degrees,
        search_radius_degrees=(
            search_radius_degrees
            if search_radius_degrees is not None
            else recipe.solver.search_radius_degrees
        ),
        review_approvals=tuple(
            ReviewApproval(
                source_sha256=item.source_sha256,
                gate_policy_digest=item.gate_policy_digest,
                request_digest=item.request_digest,
            )
            for item in recipe.review_approvals
        ),
        recipe_digest=_recipe_sha256(recipe),
    )


def select_solver_chain(
    registry: BackendRegistry,
    requested: str | Sequence[str] = "auto",
) -> tuple[SolverBackend, ...]:
    """Resolve an ordered chain containing only execution-ready solvers."""

    if isinstance(requested, str):
        requested_ids = tuple(
            item.strip() for item in requested.split(",") if item.strip()
        )
    else:
        requested_ids = tuple(item.strip() for item in requested if item.strip())
    if not requested_ids:
        raise RuntimeConfigurationError(
            "SOLVER_CHAIN_EMPTY", "solver chain must name at least one backend"
        )
    if requested_ids == ("auto",):
        requested_ids = ("astrometry-net", "astap", "native")
        explicit = False
    elif "auto" in requested_ids:
        raise RuntimeConfigurationError(
            "SOLVER_CHAIN_INVALID", "auto cannot be mixed with explicit solver ids"
        )
    else:
        explicit = True
    selected: list[SolverBackend] = []
    problems: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for backend_id in requested_ids:
        if backend_id in seen:
            continue
        seen.add(backend_id)
        backend = registry.get(backend_id)
        if backend is None or backend.descriptor.stage is not StageKind.SOLVER:
            problems.append(f"{backend_id}: not a registered solver")
            unknown.append(backend_id)
            continue
        if not isinstance(backend, SolverBackend):
            problems.append(f"{backend_id}: adapter does not implement solve()")
            continue
        descriptor = backend.descriptor
        if not (descriptor.available and descriptor.execution_ready):
            problems.append(
                f"{backend_id}: {descriptor.reason or 'runtime probe did not pass'}"
            )
            continue
        if not solver_backend_science_ready(backend):
            problems.append(
                f"{backend_id}: catalog-correspondence quality evidence is unavailable"
            )
            continue
        selected.append(backend)
    if explicit and unknown:
        raise RuntimeConfigurationError(
            "SOLVER_UNAVAILABLE", "; ".join(problems)
        )
    if not selected:
        detail = "; ".join(problems) or "no solver adapters are registered"
        raise RuntimeConfigurationError(
            "SOLVER_UNAVAILABLE",
            "no execution-ready offline solver is available: " + detail,
        )
    return tuple(selected)


def prepare_execution(
    inventory: ProjectInventory,
    recipe: Recipe,
    output_directory: str | Path,
    *,
    registry: BackendRegistry | None = None,
    hardware: HardwareProfile | None = None,
    solver_chain: str | Sequence[str] | None = None,
    workers: int | None = None,
    ra_hint_degrees: float | None = None,
    dec_hint_degrees: float | None = None,
    field_of_view_degrees: float | None = None,
    search_radius_degrees: float | None = None,
    requested_hardware_profile: str | None = None,
) -> tuple[ExecutionPlan, E2ERequest, tuple[SolverBackend, ...]]:
    """Validate a plan and return every object needed by ``run_e2e``."""

    registry = registry or default_registry()
    hardware = hardware or detect_hardware()
    plan = build_plan(inventory, recipe, hardware=hardware, registry=registry)
    if not plan.contract_valid:
        reasons = "; ".join(
            issue.message for issue in plan.issues if issue.blocks_contract
        )
        raise RuntimeConfigurationError(
            "PLAN_CONTRACT_INVALID", reasons or "execution plan is invalid"
        )
    if not plan.execution_ready:
        reasons = "; ".join(
            issue.message for issue in plan.issues if issue.blocks_execution
        )
        raise RuntimeConfigurationError(
            "PLAN_NOT_EXECUTION_READY", reasons or "one or more stages are blocked"
        )
    requested = solver_chain if solver_chain is not None else recipe.solver.backend
    solvers = select_solver_chain(registry, requested)
    request = build_e2e_request(
        inventory,
        recipe,
        output_directory,
        workers=workers,
        hardware=hardware,
        ra_hint_degrees=ra_hint_degrees,
        dec_hint_degrees=dec_hint_degrees,
        field_of_view_degrees=field_of_view_degrees,
        search_radius_degrees=search_radius_degrees,
        requested_hardware_profile=requested_hardware_profile,
    )
    return plan, request, solvers


def prepare_project_execution(
    inventory: ProjectInventory,
    recipe: Recipe,
    output_directory: str | Path,
    *,
    registry: BackendRegistry | None = None,
    hardware: HardwareProfile | None = None,
    solver_chain: str | Sequence[str] | None = None,
    workers: int | None = None,
    ra_hint_degrees: float | None = None,
    dec_hint_degrees: float | None = None,
    field_of_view_degrees: float | None = None,
    search_radius_degrees: float | None = None,
    requested_hardware_profile: str | None = None,
) -> tuple[ExecutionPlan, ProjectE2ERequest, tuple[SolverBackend, ...]]:
    """Prepare a multi-target project using the same validated base E2E plan."""

    plan, request, solvers = prepare_execution(
        inventory,
        recipe,
        output_directory,
        registry=registry,
        hardware=hardware,
        solver_chain=solver_chain,
        workers=workers,
        ra_hint_degrees=ra_hint_degrees,
        dec_hint_degrees=dec_hint_degrees,
        field_of_view_degrees=field_of_view_degrees,
        search_radius_degrees=search_radius_degrees,
        requested_hardware_profile=requested_hardware_profile,
    )
    return (
        plan,
        ProjectE2ERequest(inventory, request, str(Path(output_directory).expanduser())),
        solvers,
    )


__all__ = [
    "RuntimeConfigurationError",
    "build_e2e_request",
    "inventory_manifest_sha256",
    "pixel_backend_for_hardware_profile",
    "prepare_execution",
    "prepare_project_execution",
    "select_solver_chain",
    "validate_e2e_inventory",
]
