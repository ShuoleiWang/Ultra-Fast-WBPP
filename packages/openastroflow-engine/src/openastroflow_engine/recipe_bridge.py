"""Lossless bridge from canonical ``app-core`` Project/Recipe v1 documents.

The existing Python planner predates the canonical Rust model.  This module is
the only compatibility boundary: it retains the complete canonical documents,
derives the smaller Python execution recipe only when the mapping is exact,
and carries explicit result contracts for settings that the Python recipe does
not model.  Unsupported execution semantics fail closed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
import math
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Callable, Mapping

from .inventory import inventory_project
from .models import AssetRole, ProjectInventory
from .protocol_v1 import (
    ProtocolV1Error,
    WorkerEnvelope,
    validate_project_v1,
    validate_recipe_v1,
)
from .recipe import (
    CalibrationRecipe,
    DrizzleRecipe,
    Recipe as PythonRecipe,
    Requirement,
    SolverPolicy,
    SolverRecipe,
)


_SUPPORTED_FILE_SUFFIXES = {".fit", ".fits", ".fts", ".xisf"}
_ROLE_MAP = {
    "light": AssetRole.LIGHT,
    "flat": AssetRole.FLAT,
    "dark": AssetRole.DARK,
    "bias": AssetRole.BIAS,
    "master-flat": AssetRole.MASTER_FLAT,
    "master-dark": AssetRole.MASTER_DARK,
    "master-bias": AssetRole.MASTER_BIAS,
}
_STAGE_MAP = {
    "quality-control": "QUALITY_GATE",
    "calibration": "CALIBRATION",
    "registration": "REGISTRATION",
    "integration": "INTEGRATION",
    "drizzle": "DRIZZLE",
    "astrometric-solve": "SOLVER",
}
_SOLVER_CATALOG_BACKENDS = {
    "gaia-dr3-offline": "native",
    "astrometry-net-offline": "astrometry-net",
    "astap-offline": "astap",
}
_NATIVE_SOLVER_PROJECTIONS = {"TAN"}
_NATIVE_DRIZZLE_SCALES = {1, 2, 3}
_NATIVE_DRIZZLE_KERNELS = {"square"}
_F32_TOLERANCE = 8.0 * 1.1920928955078125e-7
_SHA256 = re.compile(r"[0-9a-f]{64}")


class BridgeError(ValueError):
    """A canonical setting cannot be represented without changing meaning."""

    def __init__(self, code: str, message: str, path: str = "") -> None:
        self.code = code
        self.path = path
        self.message = message
        prefix = f"{path}: " if path else ""
        super().__init__(f"{prefix}{message}")


class ResultRequirement(StrEnum):
    DISABLED = "disabled"
    BEST_EFFORT = "best-effort"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class CanonicalStage:
    stage_id: str
    kind: str
    enabled: bool
    depends_on: tuple[str, ...]
    parameters: dict[str, Any]

    @property
    def python_stage(self) -> str | None:
        return _STAGE_MAP.get(self.kind)

    def serializable(self) -> dict[str, Any]:
        return {
            "stageId": self.stage_id,
            "kind": self.kind,
            "enabled": self.enabled,
            "dependsOn": list(self.depends_on),
            "parameters": deepcopy(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class SolverContract:
    result: ResultRequirement
    catalog: str
    projection: str
    minimum_matches: int
    maximum_rms_arcsec: float

    @property
    def required(self) -> bool:
        return self.result == ResultRequirement.REQUIRED

    def validate_receipt(self, receipt: Mapping[str, Any] | None) -> None:
        """Apply the same final-master solver thresholds as ``app-core``."""

        if self.result == ResultRequirement.DISABLED:
            return
        if receipt is None:
            raise BridgeError("SOLVER_RESULT_MISSING", "final master has no astrometry receipt", "artifact.astrometry")
        if not isinstance(receipt, Mapping):
            raise BridgeError(
                "SOLVER_RECEIPT_INVALID",
                "astrometry receipt must be an object",
                "artifact.astrometry",
            )
        try:
            reference_frame = receipt["referenceFrame"]
            projection = receipt["projection"]
            center_ra = receipt["centerRaDegrees"]
            center_dec = receipt["centerDecDegrees"]
            pixel_scale = receipt["pixelScaleArcsec"]
            rotation = receipt["rotationDegrees"]
            matched_stars = receipt["matchedStars"]
            rms_pixels = receipt["rmsPixels"]
            rms_arcsec = receipt["rmsArcsec"]
            parity = receipt["parity"]
            catalog_identity = receipt["catalogIdentity"]
            index_identities = receipt["indexIdentities"]
            correspondence_sha256 = receipt["correspondenceSha256"]
            wcs_sha256 = receipt["wcsSha256"]
        except KeyError as error:
            raise BridgeError(
                "SOLVER_RECEIPT_INCOMPLETE",
                f"missing {error.args[0]}",
                "artifact.astrometry",
            ) from error
        if not isinstance(projection, str) or projection.casefold() != self.projection.casefold():
            raise BridgeError(
                "SOLVER_PROJECTION_MISMATCH",
                f"expected {self.projection}, received {projection!r}",
                "artifact.astrometry.projection",
            )
        if not isinstance(reference_frame, str) or not reference_frame.strip():
            raise BridgeError(
                "SOLVER_RECEIPT_INVALID",
                "referenceFrame must not be blank",
                "artifact.astrometry.referenceFrame",
            )
        numeric = {
            "centerRaDegrees": center_ra,
            "centerDecDegrees": center_dec,
            "pixelScaleArcsec": pixel_scale,
            "rotationDegrees": rotation,
            "rmsPixels": rms_pixels,
            "rmsArcsec": rms_arcsec,
        }
        for field, value in numeric.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise BridgeError(
                    "SOLVER_RECEIPT_INVALID",
                    f"{field} must be finite",
                    f"artifact.astrometry.{field}",
                )
        if not 0.0 <= float(center_ra) < 360.0 or not -90.0 <= float(center_dec) <= 90.0:
            raise BridgeError(
                "SOLVER_RECEIPT_INVALID",
                "astrometric center is outside the celestial coordinate range",
                "artifact.astrometry.center",
            )
        if float(pixel_scale) <= 0 or float(rms_pixels) < 0 or float(rms_arcsec) < 0:
            raise BridgeError(
                "SOLVER_RECEIPT_INVALID",
                "pixel scale must be positive and RMS values non-negative",
                "artifact.astrometry.rms",
            )
        if isinstance(matched_stars, bool) or not isinstance(matched_stars, int):
            raise BridgeError("SOLVER_RECEIPT_INVALID", "matchedStars must be an integer", "artifact.astrometry.matchedStars")
        if matched_stars < self.minimum_matches:
            raise BridgeError(
                "SOLVER_MATCHES_BELOW_MINIMUM",
                f"expected at least {self.minimum_matches}, received {matched_stars}",
                "artifact.astrometry.matchedStars",
            )
        if float(rms_arcsec) > self.maximum_rms_arcsec:
            raise BridgeError(
                "SOLVER_RMS_ABOVE_MAXIMUM",
                f"expected at most {self.maximum_rms_arcsec}, received {rms_arcsec}",
                "artifact.astrometry.rmsArcsec",
            )
        expected_arcsec = float(rms_pixels) * float(pixel_scale)
        rms_consistent = (
            float(rms_arcsec) <= 1e-9
            if expected_arcsec == 0.0
            else 0.5 <= float(rms_arcsec) / expected_arcsec <= 2.0
        )
        if not rms_consistent:
            raise BridgeError(
                "SOLVER_RMS_INCONSISTENT",
                "rmsPixels and rmsArcsec disagree with pixelScaleArcsec",
                "artifact.astrometry.rms",
            )
        if parity not in {"POSITIVE", "NEGATIVE"}:
            raise BridgeError(
                "SOLVER_PARITY_INVALID",
                "parity must be POSITIVE or NEGATIVE",
                "artifact.astrometry.parity",
            )
        if (
            not isinstance(index_identities, list)
            or not index_identities
            or any(not isinstance(item, str) or not item.strip() for item in index_identities)
            or len(set(index_identities)) != len(index_identities)
        ):
            raise BridgeError(
                "SOLVER_INDEX_IDENTITY_INVALID",
                "indexIdentities must contain unique, non-blank strings",
                "artifact.astrometry.indexIdentities",
            )
        for field, value in (
            ("catalogIdentity", catalog_identity),
            ("correspondenceSha256", correspondence_sha256),
            ("wcsSha256", wcs_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise BridgeError(
                    "SOLVER_DIGEST_INVALID",
                    f"{field} must be a lowercase SHA-256 digest",
                    f"artifact.astrometry.{field}",
                )


@dataclass(frozen=True, slots=True)
class DrizzleContract:
    result: ResultRequirement
    scale: float
    drop_shrink: float
    kernel: str

    @property
    def required(self) -> bool:
        return self.result == ResultRequirement.REQUIRED

    def validate_receipt(self, receipt: Mapping[str, Any] | None) -> None:
        """Require exact drizzle provenance within Rust's f32 tolerance."""

        if self.result == ResultRequirement.DISABLED:
            return
        if receipt is None:
            raise BridgeError("DRIZZLE_RESULT_MISSING", "final master has no drizzle receipt", "artifact.drizzle")
        try:
            scale = receipt["scale"]
            drop_shrink = receipt["dropShrink"]
            kernel = receipt["kernel"]
        except KeyError as error:
            raise BridgeError(
                "DRIZZLE_RECEIPT_INCOMPLETE",
                f"missing {error.args[0]}",
                "artifact.drizzle",
            ) from error
        for value, field in ((scale, "scale"), (drop_shrink, "dropShrink")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise BridgeError("DRIZZLE_RECEIPT_INVALID", f"{field} must be finite", f"artifact.drizzle.{field}")
        if abs(float(scale) - self.scale) > _F32_TOLERANCE:
            raise BridgeError("DRIZZLE_SCALE_MISMATCH", f"expected {self.scale}, received {scale}", "artifact.drizzle.scale")
        if abs(float(drop_shrink) - self.drop_shrink) > _F32_TOLERANCE:
            raise BridgeError(
                "DRIZZLE_DROP_SHRINK_MISMATCH",
                f"expected {self.drop_shrink}, received {drop_shrink}",
                "artifact.drizzle.dropShrink",
            )
        if not isinstance(kernel, str) or kernel.casefold() != self.kernel.casefold():
            raise BridgeError("DRIZZLE_KERNEL_MISMATCH", f"expected {self.kernel}, received {kernel!r}", "artifact.drizzle.kernel")


@dataclass(frozen=True, slots=True)
class RecipeBridge:
    recipe_id: str
    display_name: str
    stages: tuple[CanonicalStage, ...]
    solver: SolverContract
    drizzle: DrizzleContract
    python_recipe: PythonRecipe
    _canonical: dict[str, Any]

    def canonical(self) -> dict[str, Any]:
        """Return the byte-semantics-preserving canonical document."""

        return deepcopy(self._canonical)

    def validate_final_artifact(self, artifact: Mapping[str, Any]) -> None:
        """Reject a final artifact that downgrades a required scientific result."""

        if not isinstance(artifact, Mapping):
            raise BridgeError("ARTIFACT_INVALID", "artifact must be an object", "artifact")
        self.solver.validate_receipt(artifact.get("astrometry"))
        self.drizzle.validate_receipt(artifact.get("drizzle"))


@dataclass(frozen=True, slots=True)
class ProjectSourceBinding:
    source_id: str
    role: str
    host_path: str
    recursive: bool
    filter_name: str | None

    @property
    def display_path(self) -> str:
        # Do not reinterpret a Windows path merely to display it on macOS/Linux.
        return self.host_path


@dataclass(frozen=True, slots=True)
class ProjectBridge:
    project_id: str
    display_name: str
    created_at_unix_ms: int
    sources: tuple[ProjectSourceBinding, ...]
    labels: dict[str, str]
    inventory: ProjectInventory | None
    _canonical: dict[str, Any]

    def canonical(self) -> dict[str, Any]:
        return deepcopy(self._canonical)


@dataclass(frozen=True, slots=True)
class PlanBridge:
    request_id: str
    plan_id: str
    requested_hardware_profile: str
    input_manifest_sha256: str
    project: ProjectBridge
    recipe: RecipeBridge


def _stage_enabled(stages: tuple[CanonicalStage, ...], kind: str) -> bool:
    return any(stage.enabled and stage.kind == kind for stage in stages)


def bridge_recipe(raw: Mapping[str, Any]) -> RecipeBridge:
    """Map canonical Recipe v1 to the Python recipe without silent fallback."""

    try:
        canonical = validate_recipe_v1(raw)
    except ProtocolV1Error as error:
        raise BridgeError("CANONICAL_RECIPE_INVALID", str(error), error.path) from error

    if canonical.get("parameters", {}):
        raise BridgeError(
            "RECIPE_PARAMETERS_UNMAPPABLE",
            "top-level canonical parameters have no Python v1 execution mapping",
            "recipe.parameters",
        )
    stages = tuple(
        CanonicalStage(
            stage_id=stage["stageId"],
            kind=stage["kind"],
            enabled=stage.get("enabled", True),
            depends_on=tuple(stage.get("dependsOn", [])),
            parameters=deepcopy(dict(stage.get("parameters", {}))),
        )
        for stage in canonical["stages"]
    )
    enabled_mapped_kinds: set[str] = set()
    for index, stage in enumerate(stages):
        if stage.parameters:
            raise BridgeError(
                "STAGE_PARAMETERS_UNMAPPABLE",
                "stage parameters have no Python v1 execution mapping",
                f"recipe.stages[{index}].parameters",
            )
        if not stage.enabled:
            continue
        mapped = _STAGE_MAP.get(stage.kind)
        if mapped is None:
            raise BridgeError(
                "STAGE_UNSUPPORTED",
                f"enabled stage {stage.kind!r} has no Python execution module",
                f"recipe.stages[{index}].kind",
            )
        if mapped in enabled_mapped_kinds:
            raise BridgeError(
                "STAGE_MULTIPLICITY_UNSUPPORTED",
                f"Python v1 supports only one enabled {stage.kind} stage",
                f"recipe.stages[{index}].kind",
            )
        enabled_mapped_kinds.add(mapped)

    required_e2e_kinds = {
        "quality-control",
        "calibration",
        "registration",
        "integration",
    }
    missing_e2e_kinds = sorted(
        kind for kind in required_e2e_kinds if not _stage_enabled(stages, kind)
    )
    if missing_e2e_kinds:
        raise BridgeError(
            "E2E_STAGE_REQUIRED",
            "the raw E2E executor requires enabled stages: "
            + ", ".join(missing_e2e_kinds),
            "recipe.stages",
        )

    solver_raw = canonical["solver"]
    solver_result = ResultRequirement(solver_raw["result"])
    solver = SolverContract(
        result=solver_result,
        catalog=solver_raw.get("catalog", "gaia-dr3-offline"),
        projection=solver_raw.get("projection", "TAN"),
        minimum_matches=solver_raw.get("minimumMatches", 12),
        maximum_rms_arcsec=float(solver_raw.get("maximumRmsArcsec", 2.0)),
    )
    solver_stage_enabled = _stage_enabled(stages, "astrometric-solve")
    if solver_result == ResultRequirement.BEST_EFFORT:
        raise BridgeError(
            "SOLVER_BEST_EFFORT_UNMAPPABLE",
            "the Python recipe has no best-effort solver state",
            "recipe.solver.result",
        )
    if solver_result == ResultRequirement.DISABLED and solver_stage_enabled:
        raise BridgeError(
            "SOLVER_STAGE_RESULT_UNMAPPABLE",
            "an enabled solver stage with disabled result cannot be represented",
            "recipe.solver.result",
        )
    if solver_result == ResultRequirement.REQUIRED and not solver_stage_enabled:
        raise BridgeError(
            "SOLVER_STAGE_REQUIRED",
            "a required solver result needs an enabled astrometric-solve stage",
            "recipe.solver.result",
        )
    if solver_result == ResultRequirement.REQUIRED:
        if solver.catalog not in _SOLVER_CATALOG_BACKENDS:
            raise BridgeError(
                "SOLVER_CATALOG_UNSUPPORTED",
                f"catalog {solver.catalog!r} has no exact Python provider mapping",
                "recipe.solver.catalog",
            )
        if solver.projection.upper() not in _NATIVE_SOLVER_PROJECTIONS:
            raise BridgeError(
                "SOLVER_PROJECTION_UNSUPPORTED",
                f"projection {solver.projection!r} has no validated Python mapping",
                "recipe.solver.projection",
            )
    python_solver = SolverRecipe(
        policy=SolverPolicy.REQUIRED
        if solver_result == ResultRequirement.REQUIRED
        else SolverPolicy.DISABLED,
        backend=(
            _SOLVER_CATALOG_BACKENDS[solver.catalog]
            if solver_result == ResultRequirement.REQUIRED
            else "auto"
        ),
    )

    drizzle_raw = canonical["drizzle"]
    drizzle_result = ResultRequirement(drizzle_raw["result"])
    drizzle = DrizzleContract(
        result=drizzle_result,
        scale=float(drizzle_raw.get("scale", 2.0)),
        drop_shrink=float(drizzle_raw.get("dropShrink", 0.9)),
        kernel=drizzle_raw.get("kernel", "square"),
    )
    drizzle_stage_enabled = _stage_enabled(stages, "drizzle")
    if drizzle_result == ResultRequirement.BEST_EFFORT:
        raise BridgeError(
            "DRIZZLE_BEST_EFFORT_UNMAPPABLE",
            "the Python recipe has no best-effort drizzle state",
            "recipe.drizzle.result",
        )
    if drizzle_result == ResultRequirement.DISABLED and drizzle_stage_enabled:
        raise BridgeError(
            "DRIZZLE_STAGE_RESULT_UNMAPPABLE",
            "an enabled drizzle stage with disabled result cannot be represented",
            "recipe.drizzle.result",
        )
    if drizzle_result == ResultRequirement.REQUIRED and not drizzle_stage_enabled:
        raise BridgeError(
            "DRIZZLE_STAGE_REQUIRED",
            "a required drizzle result needs an enabled drizzle stage",
            "recipe.drizzle.result",
        )
    integral_scale = int(drizzle.scale)
    if drizzle_stage_enabled:
        if drizzle.scale != integral_scale or integral_scale not in _NATIVE_DRIZZLE_SCALES:
            raise BridgeError(
                "DRIZZLE_SCALE_UNSUPPORTED",
                f"scale {drizzle.scale!r} has no exact Python backend mapping",
                "recipe.drizzle.scale",
            )
        if drizzle.kernel.casefold() not in _NATIVE_DRIZZLE_KERNELS:
            raise BridgeError(
                "DRIZZLE_KERNEL_UNSUPPORTED",
                f"kernel {drizzle.kernel!r} has no exact Python backend mapping",
                "recipe.drizzle.kernel",
            )
    python_drizzle = DrizzleRecipe(
        enabled=drizzle_stage_enabled,
        backend="stsci-drizzle-cpu" if drizzle_stage_enabled else "auto",
        scale=integral_scale,
        drop_shrink=drizzle.drop_shrink,
        cfa_drizzle=False,
    )

    calibration_enabled = _stage_enabled(stages, "calibration")
    python_calibration = (
        CalibrationRecipe()
        if calibration_enabled
        else CalibrationRecipe(
            flat=Requirement.DISABLED,
            dark=Requirement.DISABLED,
            bias=Requirement.DISABLED,
            allow_masters=True,
        )
    )
    python_recipe = PythonRecipe(
        calibration=python_calibration,
        solver=python_solver,
        drizzle=python_drizzle,
        output_format="FITS",
        overwrite=False,
        schema_version=1,
    )
    # Explicit invariants make a future Python Recipe change fail visibly.
    if solver.required and python_recipe.solver.policy != SolverPolicy.REQUIRED:
        raise BridgeError("SOLVER_REQUIRED_DOWNGRADED", "required solver was downgraded")
    if drizzle.required and not python_recipe.drizzle.enabled:
        raise BridgeError("DRIZZLE_REQUIRED_DOWNGRADED", "required drizzle was downgraded")
    return RecipeBridge(
        recipe_id=canonical["recipeId"],
        display_name=canonical["displayName"],
        stages=stages,
        solver=solver,
        drizzle=drizzle,
        python_recipe=python_recipe,
        _canonical=canonical,
    )


def _looks_like_foreign_windows_path(host_path: str) -> bool:
    return os.name != "nt" and PureWindowsPath(host_path).is_absolute()


def _inventory_inputs(source: ProjectSourceBinding) -> tuple[str, ...]:
    if _looks_like_foreign_windows_path(source.host_path):
        raise BridgeError(
            "HOST_PATH_PLATFORM_MISMATCH",
            "Windows host path can be displayed here but must be inventoried by a Windows worker",
            f"project.sources.{source.source_id}.hostPath",
        )
    path = Path(source.host_path).expanduser()
    if not path.exists():
        raise BridgeError("SOURCE_NOT_FOUND", f"source does not exist: {source.host_path}", f"project.sources.{source.source_id}.hostPath")
    if path.is_file() or source.recursive:
        return (str(path),)
    if not path.is_dir():
        raise BridgeError("SOURCE_INVALID", "source must be a file or directory", f"project.sources.{source.source_id}.hostPath")
    direct = tuple(
        str(candidate)
        for candidate in sorted(path.iterdir(), key=lambda item: os.path.normcase(item.name))
        if candidate.is_file() and candidate.suffix.casefold() in _SUPPORTED_FILE_SUFFIXES
    )
    if not direct:
        raise BridgeError("SOURCE_EMPTY", "non-recursive source contains no supported frames", f"project.sources.{source.source_id}.hostPath")
    return direct


def _asset_belongs(asset_path: str, source: ProjectSourceBinding) -> bool:
    if _looks_like_foreign_windows_path(source.host_path):
        return False
    try:
        asset = Path(asset_path).resolve(strict=True)
        root = Path(source.host_path).expanduser().resolve(strict=True)
    except OSError:
        return False
    if root.is_file():
        return asset == root
    if source.recursive:
        try:
            asset.relative_to(root)
            return True
        except ValueError:
            return False
    return asset.parent == root


def bridge_project(
    raw: Mapping[str, Any],
    *,
    build_inventory: bool = True,
    inventory_builder: Callable[..., ProjectInventory] = inventory_project,
) -> ProjectBridge:
    """Map canonical sources to the inventory boundary and verify role/filter claims.

    ``build_inventory=False`` is useful to display a project created on another
    host.  It never rewrites a Windows path into a POSIX-looking identity.
    """

    try:
        canonical = validate_project_v1(raw)
    except ProtocolV1Error as error:
        raise BridgeError("CANONICAL_PROJECT_INVALID", str(error), error.path) from error
    sources = tuple(
        ProjectSourceBinding(
            source_id=source["sourceId"],
            role=source["role"],
            host_path=source["hostPath"],
            recursive=source.get("recursive", False),
            filter_name=source.get("filter"),
        )
        for source in canonical["sources"]
    )
    inventory: ProjectInventory | None = None
    if build_inventory:
        inputs: list[str] = []
        seen: set[str] = set()
        for source in sources:
            for item in _inventory_inputs(source):
                key = os.path.normcase(os.path.abspath(item))
                if key not in seen:
                    inputs.append(item)
                    seen.add(key)
        try:
            built = inventory_builder(inputs, name=canonical["displayName"])
        except BridgeError:
            raise
        except Exception as error:
            code = getattr(error, "code", "INVENTORY_FAILED")
            raise BridgeError(str(code), str(error), "project.sources") from error
        if not isinstance(built, ProjectInventory):
            raise BridgeError("INVENTORY_INVALID", "inventory builder returned the wrong type", "project.sources")
        for asset in built.assets:
            owners = [source for source in sources if _asset_belongs(asset.path, source)]
            if len(owners) != 1:
                raise BridgeError(
                    "SOURCE_OWNERSHIP_AMBIGUOUS",
                    f"asset belongs to {len(owners)} canonical sources: {asset.path}",
                    "project.sources",
                )
            owner = owners[0]
            expected_role = _ROLE_MAP[owner.role]
            if asset.role != expected_role:
                raise BridgeError(
                    "SOURCE_ROLE_MISMATCH",
                    f"{asset.path} was declared {owner.role} but inventory found {asset.role.value}",
                    f"project.sources.{owner.source_id}.role",
                )
            if owner.filter_name is not None and asset.filter_name.casefold() != owner.filter_name.casefold():
                raise BridgeError(
                    "SOURCE_FILTER_MISMATCH",
                    f"{asset.path} has filter {asset.filter_name!r}, expected {owner.filter_name!r}",
                    f"project.sources.{owner.source_id}.filter",
                )
        inventory = replace(
            built,
            project_id=canonical["projectId"],
            name=canonical["displayName"],
            source_roots=tuple(source.host_path for source in sources),
        )
    return ProjectBridge(
        project_id=canonical["projectId"],
        display_name=canonical["displayName"],
        created_at_unix_ms=canonical["createdAtUnixMs"],
        sources=sources,
        labels=deepcopy(dict(canonical.get("labels", {}))),
        inventory=inventory,
        _canonical=canonical,
    )


def bridge_plan(
    envelope_or_payload: WorkerEnvelope | Mapping[str, Any],
    *,
    build_inventory: bool = True,
    inventory_builder: Callable[..., ProjectInventory] = inventory_project,
) -> PlanBridge:
    """Bridge a validated plan envelope (or its payload) into Python models."""

    if isinstance(envelope_or_payload, WorkerEnvelope):
        envelope = envelope_or_payload
    elif isinstance(envelope_or_payload, Mapping) and "type" in envelope_or_payload:
        envelope = WorkerEnvelope.from_mapping(envelope_or_payload)
    else:
        envelope = WorkerEnvelope.from_mapping(
            {
                "protocolVersion": 1,
                "sessionId": "bridge-session",
                "sequence": 1,
                "sentAtUnixMs": 0,
                "type": "plan",
                "payload": envelope_or_payload,
            }
        )
    if envelope.message_type != "plan":
        raise BridgeError("MESSAGE_TYPE_INVALID", "expected a plan message", "type")
    payload = envelope.payload
    return PlanBridge(
        request_id=payload["requestId"],
        plan_id=payload["planId"],
        requested_hardware_profile=payload["requestedHardwareProfile"],
        input_manifest_sha256=payload["inputManifestSha256"],
        project=bridge_project(
            payload["project"],
            build_inventory=build_inventory,
            inventory_builder=inventory_builder,
        ),
        recipe=bridge_recipe(payload["recipe"]),
    )


__all__ = [
    "BridgeError",
    "CanonicalStage",
    "DrizzleContract",
    "PlanBridge",
    "ProjectBridge",
    "ProjectSourceBinding",
    "RecipeBridge",
    "ResultRequirement",
    "SolverContract",
    "bridge_plan",
    "bridge_project",
    "bridge_recipe",
]
