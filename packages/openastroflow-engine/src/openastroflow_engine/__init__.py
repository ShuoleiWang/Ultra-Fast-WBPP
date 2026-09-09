"""Ultra-Fast WBPP's typed, headless planning engine."""

__version__ = "0.1.0"

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
