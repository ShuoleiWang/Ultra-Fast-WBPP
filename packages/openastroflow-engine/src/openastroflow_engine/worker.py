"""Canonical app-core protocol-v1 Python worker.

stdin is the controller-to-worker NDJSON stream; stdout is the independent
worker-to-controller stream. Both directions are handshake-first and use the
strict codec shared with the Rust ``app-core`` schema. There is deliberately no
compatibility parser for the old unbound ``inputs`` request format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, TextIO

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np

from . import __version__
from lightframeqc.content_hash import file_sha256
from .backends import BackendRegistry, StageKind
from .e2e import E2EError, E2EResult, ProgressEvent, ProgressStage, run_e2e
from .hardware import HardwareProfile, detect_hardware
from .performance_profile import ExecutionTuning, select_execution_tuning
from .planning import (
    ExecutionPlan,
    build_plan,
    default_registry,
    solver_backend_science_ready,
)
from .protocol_v1 import (
    PROTOCOL_VERSION,
    ProtocolCursor,
    ProtocolV1Error,
    WorkerEnvelope,
    decode_ndjson_line,
    encode_ndjson_line,
)
from .recipe_bridge import BridgeError, PlanBridge, bridge_plan
from .runtime import (
    RuntimeConfigurationError,
    inventory_manifest_sha256,
    prepare_execution,
    prepare_project_execution,
)
from .project_e2e import (
    ProjectE2EError,
    ProjectE2EResult,
    project_requires_orchestration,
    run_project_e2e,
)
from .solver import canonical_wcs_sha256, wcs_parity


RegistryFactory = Callable[[], BackendRegistry]
E2ERunner = Callable[..., E2EResult]
ProjectRunner = Callable[..., ProjectE2EResult]
HardwareDetector = Callable[[], HardwareProfile]
MetalProbe = Callable[[HardwareProfile], bool]


@dataclass(frozen=True, slots=True)
class StoredPlan:
    bridge: PlanBridge
    execution_plan: ExecutionPlan
    inventory_manifest_sha256: str
    canonical_payload_sha256: str


@dataclass(slots=True)
class WorkerState:
    input_cursor: ProtocolCursor = field(default_factory=ProtocolCursor)
    session_id: str | None = None
    output_sequence: int = 0
    registry: BackendRegistry | None = None
    hardware: HardwareProfile | None = None
    supported_profiles: tuple[str, ...] = ()
    plans: dict[str, StoredPlan] = field(default_factory=dict)
    active_run_id: str | None = None


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _sha256_file(path: Path) -> str:
    return file_sha256(path)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_code(value: str) -> str:
    code = re.sub(r"[^a-z0-9._-]+", "-", value.casefold().replace("_", "-"))
    code = code.strip("-.")
    return code[:128] if code else "worker-error"


def _native_metal_available(hardware: HardwareProfile) -> bool:
    if not hardware.apple_silicon:
        return False
    try:
        from .metal_integration import NativeMetalExecutor

        with NativeMetalExecutor():
            return True
    except Exception:
        return False


def _probe_metal_without_harming_cpu(
    metal_probe: MetalProbe, hardware: HardwareProfile
) -> bool:
    """Convert every accelerator-probe failure into a CPU-only capability set.

    The worker handshake is also the CPU readiness boundary.  A missing native
    library, unavailable Metal device, ABI mismatch, or third-party probe bug
    must therefore remove Metal profiles without suppressing the otherwise
    usable portable CPU worker.
    """

    try:
        return bool(metal_probe(hardware))
    except Exception:
        return False


def _supported_hardware_profiles(
    hardware: HardwareProfile,
    *,
    metal_available: bool,
    tuning: ExecutionTuning | None = None,
) -> tuple[str, ...]:
    selected_tuning = tuning or select_execution_tuning(hardware)
    if hardware.operating_system.casefold() == "windows":
        return ("windows-cpu",)
    profiles: list[str] = []
    if hardware.architecture in {"arm64", "aarch64"}:
        profiles.append("generic-arm64-cpu")
    elif hardware.architecture in {"x86_64", "amd64"}:
        profiles.append("portable-cpu")
    if hardware.apple_silicon and metal_available:
        profiles.append("generic-apple-metal")
        if selected_tuning.profile_id == "apple-m3-pro-tuned-v1":
            profiles.append("m3-pro-tuned")
    return tuple(profiles)


def worker_capabilities(
    registry: BackendRegistry,
    hardware: HardwareProfile,
    *,
    metal_available: bool | None = None,
    tuning: ExecutionTuning | None = None,
) -> dict[str, Any]:
    if metal_available is None:
        metal_available = _native_metal_available(hardware)
    selected_tuning = tuning or select_execution_tuning(hardware)
    profiles = _supported_hardware_profiles(
        hardware,
        metal_available=metal_available,
        tuning=selected_tuning,
    )
    ready_stages: list[str] = []
    mapping = {
        StageKind.QUALITY_GATE: "quality-control",
        StageKind.CALIBRATION: "calibration",
        StageKind.REGISTRATION: "registration",
        StageKind.INTEGRATION: "integration",
        StageKind.DRIZZLE: "drizzle",
        StageKind.SOLVER: "astrometric-solve",
    }
    for stage, wire_name in mapping.items():
        candidates = registry.for_stage(stage)
        if stage is StageKind.SOLVER:
            stage_ready = any(solver_backend_science_ready(backend) for backend in candidates)
        else:
            stage_ready = any(
                backend.descriptor.available and backend.descriptor.execution_ready
                for backend in candidates
            )
        if stage_ready:
            ready_stages.append(wire_name)
    features = ["cpu-execution", "deterministic-receipts", "fits"]
    if metal_available:
        features.append("metal-execution")
    if (
        selected_tuning.profile_id == "apple-m3-pro-tuned-v1"
        and metal_available
    ):
        features.append("m3-pro-tuning")
    if "drizzle" in ready_stages:
        features.append("drizzle")
    if "astrometric-solve" in ready_stages:
        features.append("offline-astrometric-solver")
    try:
        from .mosaic import reproject_capability

        if reproject_capability().capable:
            features.append("mosaic")
    except Exception:
        pass
    return {
        "schemaVersion": 1,
        "backendId": "openastroflow-python-worker",
        "backendVersion": __version__,
        "workerBuild": os.environ.get("OPENASTROFLOW_WORKER_BUILD", "source"),
        "hardwareProfiles": list(profiles),
        "stages": ready_stages,
        "features": features,
        "maximumParallelStages": 1,
        "inputExtensions": ["fit", "fits", "fts"],
        "outputExtensions": ["fits"],
    }


def worker_handshake_envelope(
    session_id: str,
    *,
    registry: BackendRegistry | None = None,
    hardware: HardwareProfile | None = None,
    sent_at_unix_ms: int | None = None,
    metal_available: bool | None = None,
) -> WorkerEnvelope:
    registry = registry or default_registry()
    hardware = hardware or detect_hardware()
    return WorkerEnvelope.from_mapping(
        {
            "protocolVersion": PROTOCOL_VERSION,
            "sessionId": session_id,
            "sequence": 0,
            "sentAtUnixMs": _now_ms() if sent_at_unix_ms is None else sent_at_unix_ms,
            "type": "handshake",
            "payload": {
                "role": "worker",
                "implementation": "openastroflow-python-worker",
                "implementationVersion": __version__,
                "supportedProtocolVersions": [PROTOCOL_VERSION],
                "capabilities": worker_capabilities(
                    registry, hardware, metal_available=metal_available
                ),
            },
        }
    )


def _envelope(
    state: WorkerState, message_type: str, payload: Mapping[str, Any]
) -> WorkerEnvelope:
    if state.session_id is None:
        raise RuntimeError("worker output cannot precede a controller handshake")
    envelope = WorkerEnvelope.from_mapping(
        {
            "protocolVersion": PROTOCOL_VERSION,
            "sessionId": state.session_id,
            "sequence": state.output_sequence,
            "sentAtUnixMs": _now_ms(),
            "type": message_type,
            "payload": dict(payload),
        }
    )
    state.output_sequence += 1
    return envelope


def _write(output_stream: TextIO, envelope: WorkerEnvelope) -> None:
    output_stream.write(encode_ndjson_line(envelope).decode("utf-8"))
    output_stream.flush()


def _error_payload(
    *,
    code: str,
    message: str,
    request_id: str | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": _stable_code(code),
        "message": message or "worker operation failed",
        "retryable": retryable,
        "details": dict(details or {}),
    }
    if request_id is not None:
        payload["requestId"] = request_id
    if run_id is not None:
        payload["runId"] = run_id
    if stage_id is not None:
        payload["stageId"] = stage_id
    return payload


def _progress_stage_id(plan: PlanBridge, stage: ProgressStage) -> str | None:
    wire_kind = {
        ProgressStage.QUALITY_CONTROL: "quality-control",
        ProgressStage.CALIBRATION: "calibration",
        ProgressStage.REGISTRATION: "registration",
        ProgressStage.INTEGRATION: "integration",
        ProgressStage.DRIZZLE: "drizzle",
        ProgressStage.ASTROMETRY: "astrometric-solve",
    }.get(stage)
    if wire_kind is None:
        return None
    return next(
        (
            item.stage_id
            for item in plan.recipe.stages
            if item.enabled and item.kind == wire_kind
        ),
        None,
    )


def _progress_payload(
    plan: PlanBridge,
    run_id: str,
    event: ProgressEvent,
) -> dict[str, Any]:
    if event.total > 0:
        fraction = min(1.0, max(0.0, event.current / event.total))
    elif event.status == "completed":
        fraction = 1.0
    elif event.status == "started":
        fraction = 0.0
    else:
        fraction = 0.5
    if event.stage is ProgressStage.FAILED:
        state = "failed"
    elif event.stage is ProgressStage.COMPLETE:
        state = "finalizing"
    elif event.status == "completed":
        state = "succeeded"
    elif event.stage in {ProgressStage.VERIFY, ProgressStage.PUBLISH}:
        state = "finalizing"
    else:
        state = "running"
    payload: dict[str, Any] = {
        "requestId": plan.request_id,
        "runId": run_id,
        "state": state,
        "fraction": fraction,
        "message": event.message or event.stage.value,
    }
    stage_id = _progress_stage_id(plan, event.stage)
    if stage_id is not None:
        payload["stageId"] = stage_id
    if event.total > 0:
        payload["completedUnits"] = event.current
        payload["totalUnits"] = event.total
    # Protocol v1 is strict and has no project-context extension fields.
    # Its stage-less running events carry aggregate work in the existing fraction.
    serialized = event.serializable()
    if "overallFraction" in serialized:
        # Stage-less succeeded events are whole-run completion in protocol v1.
        # Project and child milestones must wait for _execute's artifact gate.
        if state != "failed":
            payload["state"] = "finalizing" if serialized.get("stage") in {"verify", "publish"} else "running"
        payload["fraction"] = min(0.99, max(0.0, serialized["overallFraction"]))
        for key in ("stageId", "completedUnits", "totalUnits"):
            payload.pop(key, None)
        if serialized.get("scope") == "panel":
            payload["message"] = (
                f"panel {serialized['panelIndex']}/{serialized['panelCount']} "
                f"{serialized['panelTarget']}/{serialized['panelFilter']}: {payload['message']}"
            )
    return payload


def _image_hdu(path: Path) -> tuple[fits.Header, tuple[int, int]]:
    with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
        hdu = next(
            (
                item
                for item in hdul
                if item.data is not None and getattr(item.data, "ndim", 0) == 2
            ),
            None,
        )
        if hdu is None:
            raise RuntimeConfigurationError(
                "FINAL_ARTIFACT_INVALID", f"no two-dimensional image in {path}"
            )
        return hdu.header.copy(), tuple(int(value) for value in hdu.data.shape)


def _rotation_degrees(wcs: WCS) -> float:
    matrix = np.asarray(wcs.celestial.pixel_scale_matrix, dtype=np.float64)
    return float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))


def _projection(header: fits.Header) -> str:
    match = re.search(r"RA---([A-Z0-9]+)", str(header.get("CTYPE1", "")).upper())
    if match is None:
        raise RuntimeConfigurationError(
            "FINAL_ASTROMETRY_INVALID", "cannot derive celestial projection"
        )
    return match.group(1)


def _astrometry_receipt(
    path: Path,
    accepted_attempt: Mapping[str, Any],
) -> dict[str, Any]:
    header, shape = _image_hdu(path)
    celestial = WCS(header, relax=False).celestial
    height, width = shape
    world = celestial.all_pix2world(
        np.asarray([[(width - 1.0) / 2.0, (height - 1.0) / 2.0]]), 0
    )[0]
    scales = np.asarray(proj_plane_pixel_scales(celestial), dtype=np.float64) * 3600.0
    result = accepted_attempt.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeConfigurationError(
            "FINAL_ASTROMETRY_EVIDENCE_MISSING", "accepted solver result is missing"
        )
    quality = result.get("astrometricQuality")
    if not isinstance(quality, Mapping):
        raise RuntimeConfigurationError(
            "FINAL_ASTROMETRY_EVIDENCE_MISSING",
            "accepted solver result has no catalog correspondence evidence",
        )
    return {
        "referenceFrame": str(header.get("RADESYS", "ICRS") or "ICRS"),
        "projection": _projection(header),
        "centerRaDegrees": float(world[0] % 360.0),
        "centerDecDegrees": float(world[1]),
        "pixelScaleArcsec": float(math.sqrt(float(scales[0] * scales[1]))),
        "rotationDegrees": _rotation_degrees(celestial),
        "rmsPixels": float(quality["rmsPixels"]),
        "rmsArcsec": float(quality["rmsArcsec"]),
        "matchedStars": int(quality["matchedStars"]),
        "parity": wcs_parity(header).value,
        "catalogIdentity": str(quality["catalogIdentity"]),
        "indexIdentities": list(quality["indexIdentities"]),
        "correspondenceSha256": str(quality["correspondenceSha256"]),
        "wcsSha256": canonical_wcs_sha256(header),
        "catalogManaged": quality["catalogManaged"],
        "installedSetIdentity": quality["installedSetIdentity"],
        "catalogManifestSha256": quality["catalogManifestSha256"],
        "indexArtifacts": list(quality["indexArtifacts"]),
    }


def _drizzle_receipt(
    coverage: Mapping[str, Any], filter_name: str
) -> dict[str, Any] | None:
    if coverage.get("mode") != "drizzle":
        return None
    raw = coverage.get("filters", {}).get(filter_name)
    if not isinstance(raw, Mapping):
        raise RuntimeConfigurationError(
            "FINAL_DRIZZLE_EVIDENCE_MISSING",
            f"drizzle receipt is missing for filter {filter_name}",
        )
    recipe = raw.get("recipe")
    geometry = raw.get("geometry")
    statistics = raw.get("statistics")
    if not all(isinstance(value, Mapping) for value in (recipe, geometry, statistics)):
        raise RuntimeConfigurationError(
            "FINAL_DRIZZLE_EVIDENCE_INVALID",
            f"drizzle receipt is incomplete for filter {filter_name}",
        )
    return {
        "scale": float(recipe["scale"]),
        "dropShrink": float(recipe["pixfrac"]),
        "kernel": str(recipe["kernel"]),
        "inputFrames": int(statistics["inputFrames"]),
        "outputWidth": int(geometry["outputWidth"]),
        "outputHeight": int(geometry["outputHeight"]),
    }


def _validate_stage_evidence(kind: str, path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeConfigurationError(
            "FINAL_STAGE_EVIDENCE_INVALID", f"cannot parse {kind} evidence: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise RuntimeConfigurationError(
            "FINAL_STAGE_EVIDENCE_INVALID", f"{kind} evidence is not an object"
        )
    valid = False
    if kind == "quality-control":
        valid = payload.get("stage") == "quality-control" and isinstance(
            payload.get("counts"), Mapping
        )
    elif kind == "calibration":
        valid = (
            payload.get("stage") == "registration-calibration-masters"
            and isinstance(payload.get("artifacts"), list)
        )
    elif kind == "registration":
        valid = payload.get("stage") == "registration" and isinstance(
            payload.get("transforms"), list
        )
    elif kind == "integration":
        valid = (
            payload.get("pipelineVersion") == "portable-pixel-pipeline-v1"
            and payload.get("state") == "UNSOLVED_WORKING"
            and isinstance(payload.get("outputs"), list)
        )
    elif kind == "drizzle":
        filters = payload.get("filters")
        valid = (
            payload.get("mode") == "drizzle"
            and isinstance(filters, Mapping)
            and bool(filters)
            and all(
                isinstance(item, Mapping) and item.get("status") == "succeeded"
                for item in filters.values()
            )
        )
    if not valid:
        raise RuntimeConfigurationError(
            "FINAL_STAGE_EVIDENCE_INVALID",
            f"{kind} evidence does not match the connected executor receipt",
        )


def _artifact_payloads(
    result: E2EResult,
    plan: PlanBridge,
    run_id: str,
    *,
    started_at_ms: int,
) -> tuple[dict[str, Any], ...]:
    if not result.success or result.output_directory is None:
        return ()
    output = Path(result.output_directory).resolve(strict=True)
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    if receipt.get("pipelineVersion") == "openastroflow-project-e2e-v1":
        return _project_artifact_payloads(
            result,
            plan,
            run_id,
            started_at_ms=started_at_ms,
            receipt=receipt,
        )
    astrometry_filters = receipt.get("astrometry", {}).get("filters", {})
    coverage = json.loads(
        (output / receipt["integration"]["coverage"]).read_text(encoding="utf-8")
    )
    solver_stage_id = next(
        (
            stage.stage_id
            for stage in plan.recipe.stages
            if stage.enabled and stage.kind == "astrometric-solve"
        ),
        None,
    )
    if solver_stage_id is None:
        raise RuntimeConfigurationError(
            "FINAL_STAGE_MISSING", "successful E2E result has no solver stage"
        )
    records: list[tuple[str, Path, Mapping[str, Any]]] = []
    for filter_name, solver_record in sorted(astrometry_filters.items()):
        if not isinstance(solver_record, Mapping) or solver_record.get("status") != "SOLVED":
            raise RuntimeConfigurationError(
                "FINAL_ASTROMETRY_INVALID", f"filter {filter_name} is not SOLVED"
            )
        relative = solver_record.get("output")
        if not isinstance(relative, str):
            raise RuntimeConfigurationError(
                "FINAL_ARTIFACT_INVALID", f"filter {filter_name} output is missing"
            )
        path = (output / relative).resolve(strict=True)
        try:
            path.relative_to(output)
        except ValueError as error:
            raise RuntimeConfigurationError(
                "FINAL_ARTIFACT_ESCAPE", "final artifact escaped the output directory"
            ) from error
        attempts = solver_record.get("attempts")
        accepted = (
            next(
                (
                    item
                    for item in reversed(attempts)
                    if isinstance(item, Mapping) and item.get("accepted") is True
                ),
                None,
            )
            if isinstance(attempts, list)
            else None
        )
        if accepted is None:
            raise RuntimeConfigurationError(
                "FINAL_ASTROMETRY_EVIDENCE_MISSING",
                f"filter {filter_name} has no accepted solver attempt",
            )
        records.append((filter_name, path, accepted))

    artifact_ids = [
        "final-" + re.sub(r"[^a-z0-9._-]+", "-", name.casefold()).strip("-.")
        for name, _, _ in records
    ]
    if len(artifact_ids) != len(set(artifact_ids)) or any(
        item == "final-" for item in artifact_ids
    ):
        artifact_ids = [
            "final-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
            for name, _, _ in records
        ]
    finished_at = _now_ms()
    payloads: list[dict[str, Any]] = []
    stage_sources = {
        "quality-control": (receipt["qualityControl"]["manifest"], "quality-report"),
        "calibration": ("receipts/registration-calibration.json", "run-log"),
        "registration": (receipt["registration"]["receipt"], "run-log"),
        "integration": (receipt["integration"]["pixelPipelineReceipt"], "run-log"),
    }
    if coverage.get("mode") == "drizzle":
        stage_sources["drizzle"] = (
            receipt["integration"]["coverage"],
            "drizzle-data",
        )
    stage_by_kind = {
        stage.kind: stage for stage in plan.recipe.stages if stage.enabled
    }
    for kind in (
        "quality-control",
        "calibration",
        "registration",
        "integration",
        "drizzle",
    ):
        if kind not in stage_sources:
            continue
        stage_spec = stage_by_kind.get(kind)
        if stage_spec is None:
            raise RuntimeConfigurationError(
                "FINAL_STAGE_MISSING", f"successful E2E result has no {kind} stage"
            )
        relative, artifact_kind = stage_sources[kind]
        evidence_path = (output / relative).resolve(strict=True)
        try:
            safe_relative = evidence_path.relative_to(output).as_posix()
        except ValueError as error:
            raise RuntimeConfigurationError(
                "FINAL_ARTIFACT_ESCAPE", f"{kind} evidence escaped the output directory"
            ) from error
        if not evidence_path.is_file():
            raise RuntimeConfigurationError(
                "FINAL_STAGE_EVIDENCE_MISSING", f"missing {kind} evidence: {safe_relative}"
            )
        _validate_stage_evidence(kind, evidence_path)
        raw_artifact_id = f"{stage_spec.stage_id}-receipt"
        diagnostic_id = (
            raw_artifact_id
            if len(raw_artifact_id) <= 128
            else f"stage-{hashlib.sha256(raw_artifact_id.encode('utf-8')).hexdigest()[:16]}"
        )
        diagnostic_artifact = {
            "schemaVersion": 1,
            "artifactId": diagnostic_id,
            "producedByStageId": stage_spec.stage_id,
            "kind": artifact_kind,
            "designation": "diagnostic",
            "relativePath": safe_relative,
            "mediaType": "application/json",
            "sha256": _sha256_file(evidence_path),
            "sizeBytes": evidence_path.stat().st_size,
            "createdAtUnixMs": finished_at,
            "attributes": {"evidenceRole": "e2e-stage-receipt"},
        }
        diagnostic_stage = {
            "schemaVersion": 1,
            "stageId": stage_spec.stage_id,
            "kind": kind,
            "status": "succeeded",
            "startedAtUnixMs": started_at_ms,
            "finishedAtUnixMs": finished_at,
            "artifactIds": [diagnostic_id],
            "metrics": {},
        }
        payloads.append(
            {
                "requestId": plan.request_id,
                "runId": run_id,
                "stage": diagnostic_stage,
                "artifact": diagnostic_artifact,
            }
        )
    prepared = [
        (
            artifact_id,
            filter_name,
            path,
            _astrometry_receipt(path, accepted),
            _drizzle_receipt(coverage, filter_name),
        )
        for artifact_id, (filter_name, path, accepted) in zip(
            artifact_ids, records, strict=True
        )
    ]
    if not prepared:
        raise RuntimeConfigurationError(
            "FINAL_ARTIFACT_MISSING", "successful E2E result contains no final master"
        )
    maximum_rms = max(float(item[3]["rmsArcsec"]) for item in prepared)
    for artifact_id, filter_name, path, astrometry, drizzle in prepared:
        artifact: dict[str, Any] = {
            "schemaVersion": 1,
            "artifactId": artifact_id,
            "producedByStageId": solver_stage_id,
            "kind": "final-master",
            "designation": "final-master",
            "relativePath": path.relative_to(output).as_posix(),
            "mediaType": "image/fits",
            "sha256": _sha256_file(path),
            "sizeBytes": path.stat().st_size,
            "createdAtUnixMs": finished_at,
            "astrometry": astrometry,
            "attributes": {
                "filter": filter_name,
                "e2eReceipt": Path(result.receipt_path).relative_to(output).as_posix(),
            },
        }
        if drizzle is not None:
            artifact["drizzle"] = drizzle
        plan.recipe.validate_final_artifact(artifact)
        stage = {
            "schemaVersion": 1,
            "stageId": solver_stage_id,
            "kind": "astrometric-solve",
            "status": "succeeded",
            "startedAtUnixMs": started_at_ms,
            "finishedAtUnixMs": finished_at,
            "artifactIds": artifact_ids,
            "metrics": {"rmsArcsec": maximum_rms},
        }
        payloads.append(
            {
                "requestId": plan.request_id,
                "runId": run_id,
                "stage": stage,
                "artifact": artifact,
            }
        )
    return tuple(payloads)


def _project_artifact_payloads(
    result: ProjectE2EResult,
    plan: PlanBridge,
    run_id: str,
    *,
    started_at_ms: int,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Map outer project products to strict canonical final-master receipts."""

    if result.output_directory is None:
        return ()
    output = Path(result.output_directory).resolve(strict=True)
    final = receipt.get("finalProducts")
    artifacts = final.get("guiArtifacts") if isinstance(final, Mapping) else None
    if not isinstance(artifacts, list):
        raise RuntimeConfigurationError(
            "PROJECT_FINAL_ARTIFACTS_MISSING", "project receipt has no guiArtifacts"
        )
    selected = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") in {"SOLVED_MONO_FITS", "LINEAR_RGB_FITS"}
    ]
    if not selected:
        raise RuntimeConfigurationError(
            "PROJECT_FINAL_ARTIFACTS_MISSING", "project produced no solved FITS artifact"
        )
    solver_stage_id = next(
        (
            stage.stage_id
            for stage in plan.recipe.stages
            if stage.enabled and stage.kind == "astrometric-solve"
        ),
        None,
    )
    if solver_stage_id is None:
        raise RuntimeConfigurationError("FINAL_STAGE_MISSING", "project has no solver stage")
    finished_at = _now_ms()
    allowed_astrometry = {
        "referenceFrame",
        "projection",
        "centerRaDegrees",
        "centerDecDegrees",
        "pixelScaleArcsec",
        "rotationDegrees",
        "rmsPixels",
        "rmsArcsec",
        "matchedStars",
        "parity",
        "catalogIdentity",
        "indexIdentities",
        "correspondenceSha256",
        "wcsSha256",
        "catalogManaged",
        "installedSetIdentity",
        "catalogManifestSha256",
        "indexArtifacts",
    }
    artifact_ids = [
        "final-project-" + hashlib.sha256(str(item["relativePath"]).encode()).hexdigest()[:16]
        for item in selected
    ]
    payloads: list[dict[str, Any]] = []
    for artifact_id, source in zip(artifact_ids, selected, strict=True):
        relative = source.get("relativePath")
        if not isinstance(relative, str):
            raise RuntimeConfigurationError("PROJECT_FINAL_ARTIFACT_INVALID", "relativePath is missing")
        path = (output / relative).resolve(strict=True)
        try:
            path.relative_to(output)
        except ValueError as error:
            raise RuntimeConfigurationError("FINAL_ARTIFACT_ESCAPE", relative) from error
        raw_astrometry = source.get("astrometry")
        if not isinstance(raw_astrometry, Mapping):
            raise RuntimeConfigurationError(
                "FINAL_ASTROMETRY_EVIDENCE_MISSING", f"{relative} has no astrometry evidence"
            )
        astrometry = {
            key: raw_astrometry[key]
            for key in allowed_astrometry
            if key in raw_astrometry
        }
        artifact: dict[str, Any] = {
            "schemaVersion": 1,
            "artifactId": artifact_id,
            "producedByStageId": solver_stage_id,
            "kind": "final-master",
            "designation": "final-master",
            "relativePath": relative,
            "mediaType": "image/fits",
            "sha256": _sha256_file(path),
            "sizeBytes": path.stat().st_size,
            "createdAtUnixMs": finished_at,
            "astrometry": astrometry,
            "attributes": {
                "projectProductKind": source.get("kind"),
                "filter": source.get("filter"),
                "e2eReceipt": Path(result.receipt_path).relative_to(output).as_posix(),
                "astrometryProvenanceType": (
                    source.get("astrometryProvenance", {}).get("type")
                    if isinstance(source.get("astrometryProvenance"), Mapping)
                    else None
                ),
                "freshSolveOnThisArtifactGrid": (
                    source.get("astrometryProvenance", {}).get(
                        "freshSolveOnThisPixelGrid",
                        source.get("astrometryProvenance", {}).get(
                            "freshSolveOnRgbCube"
                        ),
                    )
                    if isinstance(source.get("astrometryProvenance"), Mapping)
                    else None
                ),
            },
        }
        if plan.recipe.drizzle.required:
            _, shape = _image_hdu(path) if source.get("kind") == "SOLVED_MONO_FITS" else (
                fits.Header(), tuple(int(value) for value in fits.getdata(path).shape[-2:])
            )
            filter_name = source.get("filter")
            input_frames = sum(
                len(panel.get("lightFiles", []))
                for panel in receipt.get("layout", {}).get("panels", [])
                if isinstance(panel, Mapping) and panel.get("filter") == filter_name
            )
            artifact["drizzle"] = {
                "scale": plan.recipe.drizzle.scale,
                "dropShrink": plan.recipe.drizzle.drop_shrink,
                "kernel": plan.recipe.drizzle.kernel,
                "inputFrames": input_frames,
                "outputHeight": shape[0],
                "outputWidth": shape[1],
            }
        plan.recipe.validate_final_artifact(artifact)
        stage = {
            "schemaVersion": 1,
            "stageId": solver_stage_id,
            "kind": "astrometric-solve",
            "status": "succeeded",
            "startedAtUnixMs": started_at_ms,
            "finishedAtUnixMs": finished_at,
            "artifactIds": artifact_ids,
            "metrics": {"rmsArcsec": float(astrometry["rmsArcsec"])},
        }
        payloads.append(
            {
                "requestId": plan.request_id,
                "runId": run_id,
                "stage": stage,
                "artifact": artifact,
            }
        )
    return tuple(payloads)


def _store_plan(envelope: WorkerEnvelope, state: WorkerState) -> None:
    if state.registry is None or state.hardware is None:
        raise RuntimeError("worker handshake state is incomplete")
    bridged = bridge_plan(envelope, build_inventory=True)
    if bridged.requested_hardware_profile not in state.supported_profiles:
        raise RuntimeConfigurationError(
            "HARDWARE_PROFILE_UNAVAILABLE",
            f"worker does not support {bridged.requested_hardware_profile}",
        )
    assert bridged.project.inventory is not None
    manifest_sha256 = inventory_manifest_sha256(bridged.project.inventory)
    if manifest_sha256 != bridged.input_manifest_sha256:
        raise RuntimeConfigurationError(
            "INPUT_MANIFEST_MISMATCH",
            "plan inputManifestSha256 does not match the worker inventory snapshot",
        )
    execution_plan = build_plan(
        bridged.project.inventory,
        bridged.recipe.python_recipe,
        hardware=state.hardware,
        registry=state.registry,
    )
    if not execution_plan.contract_valid or not execution_plan.execution_ready:
        blocking = [
            issue.message
            for issue in execution_plan.issues
            if issue.blocks_contract or issue.blocks_execution
        ]
        raise RuntimeConfigurationError(
            "PLAN_NOT_EXECUTION_READY",
            "; ".join(blocking) or "one or more requested stages are blocked",
        )
    payload_digest = _canonical_digest(envelope.payload)
    existing = state.plans.get(bridged.plan_id)
    if existing is not None and existing.canonical_payload_sha256 != payload_digest:
        raise RuntimeConfigurationError(
            "PLAN_ID_COLLISION", "planId is already bound to different content"
        )
    state.plans[bridged.plan_id] = StoredPlan(
        bridge=bridged,
        execution_plan=execution_plan,
        inventory_manifest_sha256=manifest_sha256,
        canonical_payload_sha256=payload_digest,
    )


def _execute(
    envelope: WorkerEnvelope,
    state: WorkerState,
    output_stream: TextIO,
    *,
    e2e_runner: E2ERunner,
    project_runner: ProjectRunner,
) -> None:
    if state.registry is None or state.hardware is None:
        raise RuntimeError("worker handshake state is incomplete")
    payload = envelope.payload
    request_id = payload["requestId"]
    run_id = payload["runId"]
    stored = state.plans.get(payload["planId"])
    if stored is None:
        raise RuntimeConfigurationError(
            "PLAN_NOT_FOUND", "execute references a plan that was not stored"
        )
    if request_id != stored.bridge.request_id:
        raise RuntimeConfigurationError(
            "REQUEST_ID_MISMATCH", "execute requestId does not match the stored plan"
        )
    if payload.get("resumeFromRunId") is not None:
        raise RuntimeConfigurationError(
            "RESUME_UNSUPPORTED", "protocol v1 resume is not implemented by this worker"
        )
    output_parent = Path(payload["outputParentHostPath"]).expanduser().resolve(strict=True)
    if not output_parent.is_dir():
        raise RuntimeConfigurationError(
            "OUTPUT_PARENT_INVALID", "outputParentHostPath must be an existing directory"
        )
    output_directory = output_parent / payload["outputDirectoryName"]
    assert stored.bridge.project.inventory is not None
    use_project = project_requires_orchestration(stored.bridge.project.inventory)
    prepare = prepare_project_execution if use_project else prepare_execution
    _, request, solvers = prepare(
        stored.bridge.project.inventory,
        stored.bridge.recipe.python_recipe,
        output_directory,
        registry=state.registry,
        hardware=state.hardware,
        requested_hardware_profile=stored.bridge.requested_hardware_profile,
    )
    started_at = _now_ms()
    state.active_run_id = run_id
    _write(
        output_stream,
        _envelope(
            state,
            "progress",
            {
                "requestId": request_id,
                "runId": run_id,
                "state": "queued",
                "fraction": 0.0,
                "message": "execution accepted",
            },
        ),
    )

    def progress(event: ProgressEvent) -> None:
        _write(
            output_stream,
            _envelope(state, "progress", _progress_payload(stored.bridge, run_id, event)),
        )

    try:
        runner = project_runner if use_project else e2e_runner
        result = runner(request, solver_backends=solvers, progress=progress)
    finally:
        state.active_run_id = None
    if not result.success:
        _write(
            output_stream,
            _envelope(
                state,
                "error",
                _error_payload(
                    code=result.code,
                    message=(
                        getattr(result, "message", None)
                        or "E2E execution did not publish a solved final master"
                    ),
                    request_id=request_id,
                    run_id=run_id,
                    details={
                        "state": result.state.value,
                        "evidenceDirectory": result.evidence_directory,
                    },
                ),
            ),
        )
        return
    artifacts = _artifact_payloads(
        result, stored.bridge, run_id, started_at_ms=started_at
    )
    for artifact in artifacts:
        _write(output_stream, _envelope(state, "artifact", artifact))
    _write(
        output_stream,
        _envelope(
            state,
            "progress",
            {
                "requestId": request_id,
                "runId": run_id,
                "state": "succeeded",
                "fraction": 1.0,
                "message": f"published {len(artifacts)} verified artifact receipts",
            },
        ),
    )


def run_worker(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    registry_factory: RegistryFactory = default_registry,
    e2e_runner: E2ERunner = run_e2e,
    project_runner: ProjectRunner = run_project_e2e,
    hardware_detector: HardwareDetector = detect_hardware,
    metal_probe: MetalProbe = _native_metal_available,
) -> int:
    state = WorkerState()
    for raw_line in input_stream:
        if not raw_line.strip():
            continue
        try:
            envelope = decode_ndjson_line(raw_line)
            envelope = state.input_cursor.accept(envelope)
        except ProtocolV1Error as error:
            if state.session_id is not None:
                _write(
                    output_stream,
                    _envelope(
                        state,
                        "error",
                        _error_payload(code=error.code, message=str(error)),
                    ),
                )
            return 2

        if envelope.message_type == "handshake":
            try:
                registry = registry_factory()
                hardware = hardware_detector()
                metal_available = _probe_metal_without_harming_cpu(
                    metal_probe, hardware
                )
                profiles = _supported_hardware_profiles(
                    hardware, metal_available=metal_available
                )
                state.session_id = envelope.session_id
                state.registry = registry
                state.hardware = hardware
                state.supported_profiles = profiles
                handshake = worker_handshake_envelope(
                    envelope.session_id,
                    registry=registry,
                    hardware=hardware,
                    metal_available=metal_available,
                )
                _write(output_stream, handshake)
                state.output_sequence = 1
            except Exception:
                # No canonical worker stream exists until a valid worker
                # handshake can be constructed. Fail without a substitute.
                return 2
            continue

        request_id = envelope.payload.get("requestId")
        run_id = envelope.payload.get("runId")
        try:
            if envelope.message_type == "plan":
                _store_plan(envelope, state)
            elif envelope.message_type == "execute":
                _execute(
                    envelope,
                    state,
                    output_stream,
                    e2e_runner=e2e_runner,
                    project_runner=project_runner,
                )
            else:
                raise RuntimeConfigurationError(
                    "OPERATION_UNSUPPORTED",
                    f"controller message {envelope.message_type} is unsupported",
                )
        except (
            BridgeError,
            RuntimeConfigurationError,
            E2EError,
            ProjectE2EError,
            OSError,
            ValueError,
        ) as error:
            code = getattr(error, "code", "worker-operation-failed")
            _write(
                output_stream,
                _envelope(
                    state,
                    "error",
                    _error_payload(
                        code=str(code),
                        message=str(error),
                        request_id=request_id,
                        run_id=run_id,
                    ),
                ),
            )
        except Exception:
            if os.environ.get("OAF_DEBUG_WORKER_ERRORS") == "1":
                raise
            _write(
                output_stream,
                _envelope(
                    state,
                    "error",
                    _error_payload(
                        code="internal-error",
                        message="unexpected worker failure",
                        request_id=request_id,
                        run_id=run_id,
                    ),
                ),
            )
    return 0


def main() -> int:
    return run_worker(sys.stdin, sys.stdout)


__all__ = [
    "StoredPlan",
    "WorkerState",
    "main",
    "run_worker",
    "worker_capabilities",
    "worker_handshake_envelope",
]
