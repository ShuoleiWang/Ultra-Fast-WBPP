"""Ultra-Fast WBPP's typed, headless planning engine."""

__version__ = "0.1.0"

# The public names live in submodules that are imported on first use (PEP
# 562): a worker command such as ``inventory`` must not pay for the E2E
# pipeline, the solver backends and scipy at start-up.  Every static
# ``from .module import name`` that PyInstaller needs to see is in the
# submodules themselves.
from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING

_LAZY_EXPORTS = {
    "AstrometricQuality": "solver",
    "AstrometryNetSolveProfile": "astrometry_net_backend",
    "AstrometryNetSolverBackend": "astrometry_net_backend",
    "Backend": "backends",
    "BackendDescriptor": "backends",
    "BackendRegistry": "backends",
    "DrizzleBackend": "drizzle",
    "DrizzleOptions": "e2e",
    "E2EError": "e2e",
    "E2ERequest": "e2e",
    "E2EResult": "e2e",
    "E2EState": "e2e",
    "ExecutionPlan": "planning",
    "GlobalNormalizationParameters": "global_normalization",
    "HardwareProfile": "hardware",
    "IntegrationMode": "e2e",
    "ProgressEvent": "e2e",
    "ProgressStage": "e2e",
    "Project": "models",
    "ProjectE2EError": "project_e2e",
    "ProjectE2ERequest": "project_e2e",
    "ProjectE2EResult": "project_e2e",
    "ProjectInventory": "models",
    "ProjectLayout": "project_e2e",
    "ProtocolCursor": "protocol_v1",
    "ProtocolV1Error": "protocol_v1",
    "Recipe": "recipe",
    "RuntimeConfigurationError": "runtime",
    "SciencePanel": "project_e2e",
    "SolverBackend": "solver",
    "SolverIndexArtifact": "solver",
    "WorkerEnvelope": "protocol_v1",
    "build_e2e_request": "runtime",
    "build_plan": "planning",
    "canonical_e2e_recipe": "controller",
    "canonical_wcs_sha256": "solver",
    "classify_project_layout": "project_e2e",
    "controller_plan_envelope": "controller",
    "decode_ndjson_line": "protocol_v1",
    "default_hardware_profile": "controller",
    "default_registry": "planning",
    "detect_hardware": "hardware",
    "discover_astrometry_config": "astrometry_net_backend",
    "encode_ndjson_line": "protocol_v1",
    "inventory_manifest_sha256": "runtime",
    "inventory_project": "inventory",
    "prepare_execution": "runtime",
    "prepare_project_execution": "runtime",
    "project_requires_orchestration": "project_e2e",
    "run_e2e": "e2e",
    "run_project_e2e": "project_e2e",
    "select_solver_chain": "runtime",
    "validate_solver_result": "solver",
    "validate_wcs_header": "solver",
}


def __getattr__(name: str) -> object:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if _TYPE_CHECKING:
    from .backends import Backend, BackendDescriptor, BackendRegistry
    from .astrometry_net_backend import (
        AstrometryNetSolveProfile,
        AstrometryNetSolverBackend,
        discover_astrometry_config,
    )
    from .controller import (
        canonical_e2e_recipe,
        controller_plan_envelope,
        default_hardware_profile,
    )
    from .drizzle import DrizzleBackend
    from .e2e import (
        DrizzleOptions,
        E2EError,
        E2ERequest,
        E2EResult,
        E2EState,
        IntegrationMode,
        ProgressEvent,
        ProgressStage,
        run_e2e,
    )
    from .hardware import HardwareProfile, detect_hardware
    from .global_normalization import GlobalNormalizationParameters
    from .inventory import inventory_project
    from .models import Project, ProjectInventory
    from .planning import ExecutionPlan, build_plan, default_registry
    from .protocol_v1 import (
        ProtocolCursor,
        ProtocolV1Error,
        WorkerEnvelope,
        decode_ndjson_line,
        encode_ndjson_line,
    )
    from .recipe import Recipe
    from .runtime import (
        RuntimeConfigurationError,
        build_e2e_request,
        inventory_manifest_sha256,
        prepare_execution,
        prepare_project_execution,
        select_solver_chain,
    )
    from .project_e2e import (
        ProjectE2EError,
        ProjectE2ERequest,
        ProjectE2EResult,
        ProjectLayout,
        SciencePanel,
        classify_project_layout,
        project_requires_orchestration,
        run_project_e2e,
    )
    from .solver import (
        AstrometricQuality,
        SolverBackend,
        SolverIndexArtifact,
        canonical_wcs_sha256,
        validate_solver_result,
        validate_wcs_header,
    )

__all__ = [
    "Backend",
    "BackendDescriptor",
    "BackendRegistry",
    "AstrometricQuality",
    "AstrometryNetSolveProfile",
    "AstrometryNetSolverBackend",
    "DrizzleOptions",
    "DrizzleBackend",
    "E2EError",
    "E2ERequest",
    "E2EResult",
    "E2EState",
    "ExecutionPlan",
    "HardwareProfile",
    "GlobalNormalizationParameters",
    "IntegrationMode",
    "ProgressEvent",
    "ProgressStage",
    "Project",
    "ProjectInventory",
    "ProjectE2EError",
    "ProjectE2ERequest",
    "ProjectE2EResult",
    "ProjectLayout",
    "ProtocolCursor",
    "ProtocolV1Error",
    "Recipe",
    "RuntimeConfigurationError",
    "SolverBackend",
    "SolverIndexArtifact",
    "WorkerEnvelope",
    "build_e2e_request",
    "build_plan",
    "canonical_e2e_recipe",
    "canonical_wcs_sha256",
    "controller_plan_envelope",
    "decode_ndjson_line",
    "default_hardware_profile",
    "default_registry",
    "detect_hardware",
    "discover_astrometry_config",
    "encode_ndjson_line",
    "inventory_manifest_sha256",
    "inventory_project",
    "prepare_execution",
    "prepare_project_execution",
    "project_requires_orchestration",
    "classify_project_layout",
    "run_project_e2e",
    "SciencePanel",
    "run_e2e",
    "select_solver_chain",
    "validate_solver_result",
    "validate_wcs_header",
]
