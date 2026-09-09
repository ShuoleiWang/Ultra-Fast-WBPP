"""Canonical controller-document builder for GUI and headless clients."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from .hardware import HardwareProfile, detect_hardware
from .models import AssetRole, AssetStatus, ProjectInventory
from .protocol_v1 import WorkerEnvelope
from .recipe import Recipe
from .runtime import inventory_manifest_sha256, validate_e2e_inventory


_ROLE_WIRE = {
    AssetRole.LIGHT: "light",
    AssetRole.FLAT: "flat",
    AssetRole.DARK: "dark",
    AssetRole.BIAS: "bias",
    AssetRole.MASTER_FLAT: "master-flat",
    AssetRole.MASTER_DARK: "master-dark",
    AssetRole.MASTER_BIAS: "master-bias",
}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(prefix: str, value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not token or not token[0].isalnum():
        token = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return (prefix + token)[:128]


def default_hardware_profile(hardware: HardwareProfile | None = None) -> str:
    hardware = hardware or detect_hardware()
    if hardware.operating_system.casefold() == "windows":
        return "windows-cpu"
    if hardware.architecture in {"arm64", "aarch64"}:
        return "generic-arm64-cpu"
    return "portable-cpu"


def canonical_e2e_recipe(
    *,
    mode: str = "ordinary",
    recipe_id: str | None = None,
    display_name: str | None = None,
    solver_catalog: str = "astrometry-net-offline",
    minimum_matches: int = 12,
    maximum_rms_arcsec: float = 2.0,
    drizzle_scale: int = 2,
    drop_shrink: float = 0.9,
) -> dict[str, Any]:
    if mode not in {"ordinary", "drizzle"}:
        raise ValueError("mode must be ordinary or drizzle")
    stages: list[dict[str, Any]] = [
        {
            "stageId": "quality-control",
            "kind": "quality-control",
            "enabled": True,
            "dependsOn": [],
            "parameters": {},
        },
        {
            "stageId": "calibrate",
            "kind": "calibration",
            "enabled": True,
            "dependsOn": ["quality-control"],
            "parameters": {},
        },
        {
            "stageId": "register",
            "kind": "registration",
            "enabled": True,
            "dependsOn": ["calibrate"],
            "parameters": {},
        },
        {
            "stageId": "integrate",
            "kind": "integration",
            "enabled": True,
            "dependsOn": ["register"],
            "parameters": {},
        },
    ]
    solver_dependency = "integrate"
    if mode == "drizzle":
        stages.append(
            {
                "stageId": "drizzle",
                "kind": "drizzle",
                "enabled": True,
                "dependsOn": ["integrate"],
                "parameters": {},
            }
        )
        solver_dependency = "drizzle"
    stages.append(
        {
            "stageId": "solve",
            "kind": "astrometric-solve",
            "enabled": True,
            "dependsOn": [solver_dependency],
            "parameters": {},
        }
    )
    recipe = {
        "schemaVersion": 1,
        "recipeId": recipe_id or f"raw-{mode}-solved",
        "displayName": display_name
        or ("Raw to Solved 2x Drizzle" if mode == "drizzle" else "Raw to Solved Master"),
        "stages": stages,
        "solver": {
            "result": "required",
            "catalog": solver_catalog,
            "projection": "TAN",
            "minimumMatches": minimum_matches,
            "maximumRmsArcsec": maximum_rms_arcsec,
        },
        "drizzle": {
            "result": "required" if mode == "drizzle" else "disabled",
            "scale": float(drizzle_scale),
            "dropShrink": float(drop_shrink),
            "kernel": "square",
        },
        "parameters": {},
    }
    # Reuse the strict canonical codec for all numeric and dependency checks.
    WorkerEnvelope.from_mapping(
        {
            "protocolVersion": 1,
            "sessionId": "recipe-validation",
            "sequence": 1,
            "sentAtUnixMs": 0,
            "type": "plan",
            "payload": {
                "requestId": "request-validation",
                "planId": "plan-validation",
                "project": {
                    "schemaVersion": 1,
                    "projectId": "project-validation",
                    "displayName": "Validation",
                    "createdAtUnixMs": 0,
                    "sources": [
                        {
                            "sourceId": "source-validation",
                            "role": "light",
                            "hostPath": "/validation",
                            "recursive": False,
                        }
                    ],
                    "labels": {},
                },
                "recipe": recipe,
                "requestedHardwareProfile": "generic-arm64-cpu",
                "inputManifestSha256": "0" * 64,
            },
        }
    )
    return recipe


def controller_plan_envelope(
    inventory: ProjectInventory,
    *,
    mode: str = "ordinary",
    session_id: str = "openastroflow-controller",
    sequence: int = 1,
    request_id: str | None = None,
    plan_id: str | None = None,
    project_id: str | None = None,
    requested_hardware_profile: str | None = None,
    solver_catalog: str = "astrometry-net-offline",
    minimum_matches: int = 12,
    maximum_rms_arcsec: float = 2.0,
    drizzle_scale: int = 2,
    drop_shrink: float = 0.9,
    created_at_unix_ms: int | None = None,
) -> WorkerEnvelope:
    validate_e2e_inventory(inventory, Recipe())
    assets = tuple(
        asset
        for asset in inventory.assets
        if asset.status is AssetStatus.READY and asset.role in _ROLE_WIRE
    )
    project_token = project_id or "project-" + _digest(
        {"name": inventory.name, "assets": [asset.asset_id for asset in assets]}
    )[:24]
    sources: list[dict[str, Any]] = []
    for index, asset in enumerate(assets, start=1):
        source: dict[str, Any] = {
            "sourceId": f"source-{index:06d}",
            "role": _ROLE_WIRE[asset.role],
            "hostPath": str(Path(asset.path).resolve(strict=True)),
            "recursive": False,
        }
        if asset.role in {
            AssetRole.LIGHT,
            AssetRole.FLAT,
            AssetRole.MASTER_FLAT,
        } and asset.filter_name != "UNKNOWN":
            source["filter"] = asset.filter_name
        sources.append(source)
    targets = {
        asset.target
        for asset in assets
        if asset.role is AssetRole.LIGHT and asset.target != "UNKNOWN"
    }
    labels = {"target": next(iter(targets))} if len(targets) == 1 else {}
    project = {
        "schemaVersion": 1,
        "projectId": _identifier("", project_token),
        "displayName": inventory.name,
        "createdAtUnixMs": (
            time.time_ns() // 1_000_000
            if created_at_unix_ms is None
            else created_at_unix_ms
        ),
        "sources": sources,
        "labels": labels,
    }
    recipe = canonical_e2e_recipe(
        mode=mode,
        solver_catalog=solver_catalog,
        minimum_matches=minimum_matches,
        maximum_rms_arcsec=maximum_rms_arcsec,
        drizzle_scale=drizzle_scale,
        drop_shrink=drop_shrink,
    )
    # Canonical schema validity is broader than this executable's capability
    # set (for example schema v1 allows drizzle scale 4).  Apply the exact
    # bridge now so the GUI never emits a plan the worker can only reject.
    from .recipe_bridge import bridge_recipe

    bridge_recipe(recipe)
    bound_inventory = replace(
        inventory,
        project_id=project["projectId"],
        name=project["displayName"],
        source_roots=tuple(source["hostPath"] for source in sources),
    )
    manifest_sha256 = inventory_manifest_sha256(bound_inventory)
    identity = _digest(
        {
            "project": project,
            "recipe": recipe,
            "inputManifestSha256": manifest_sha256,
            "hardware": requested_hardware_profile or default_hardware_profile(),
        }
    )
    payload = {
        "requestId": request_id or f"request-{identity[:20]}",
        "planId": plan_id or f"plan-{identity[:24]}",
        "project": project,
        "recipe": recipe,
        "requestedHardwareProfile": requested_hardware_profile
        or default_hardware_profile(),
        "inputManifestSha256": manifest_sha256,
    }
    return WorkerEnvelope.from_mapping(
        {
            "protocolVersion": 1,
            "sessionId": session_id,
            "sequence": sequence,
            "sentAtUnixMs": time.time_ns() // 1_000_000,
            "type": "plan",
            "payload": payload,
        }
    )


__all__ = [
    "canonical_e2e_recipe",
    "controller_plan_envelope",
    "default_hardware_profile",
]
