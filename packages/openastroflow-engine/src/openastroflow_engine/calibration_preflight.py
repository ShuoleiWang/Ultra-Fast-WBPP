"""Read-only calibration checks shared by the desktop and public planner.

This checks acquisition metadata and the executor's supported library shape.
It cannot establish that an old Flat still describes the current optical train.
"""

from __future__ import annotations

from .calibration_policy import MONO_STANDARD, workflow_receipt, resolve_dark_bias, bias_from_header, can_omit_bias, conflicting_profile_fields

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .inventory import inventory_project
from .models import AssetRole, AssetStatus, FrameAsset
from .planning import (
    _base_compatible,
    _calibration_issues,
    _dark_compatible,
    _effective_calibration_assets,
    _flat_compatible,
)
from .recipe import Recipe, Requirement


class CalibrationPreflightError(ValueError):
    code = "CALIBRATION_REQUEST_INVALID"


def load_calibration_request(path: str) -> tuple[list[str], Recipe]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationPreflightError(str(error)) from error
    if (
        not isinstance(raw, Mapping)
        or set(raw) - {"schemaVersion", "paths", "recipe"}
        or raw.get("schemaVersion") != 1
    ):
        raise CalibrationPreflightError("expected schemaVersion=1, paths and optional recipe")
    paths = raw.get("paths")
    if (
        not isinstance(paths, list)
        or not 1 <= len(paths) <= 10_000
        or any(not isinstance(path, str) or not path.strip() for path in paths)
    ):
        raise CalibrationPreflightError("paths must contain between 1 and 10000 file or directory names")
    return paths, Recipe.from_dict(raw.get("recipe"))


def inspect_calibration(paths: list[str], recipe: Recipe | None = None) -> dict[str, Any]:
    recipe = recipe or Recipe()
    workflow = recipe.calibration.workflow
    inventory = inventory_project(paths)
    issues: list[dict[str, Any]] = []
    digests: dict[str, str] = {}

    def digest(path: str) -> str:
        if path not in digests:
            value = hashlib.sha256()
            with Path(path).open("rb") as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    value.update(chunk)
            digests[path] = "sha256:" + value.hexdigest()
        return digests[path]

    def issue(code: str, message: str, *, severity: str = "ERROR", affected: tuple[FrameAsset, ...] = (), groups: tuple[str, ...] = ()) -> None:
        issues.append({
            "code": code, "severity": severity, "message": message,
            "paths": sorted({asset.path for asset in affected}),
            "lightGroups": sorted(set(groups)),
        })

    for item in inventory.issues:
        issues.append({"code": item.code, "severity": item.severity.value, "message": item.message, "paths": [item.path] if item.path else [], "lightGroups": []})

    assets, override_issues, _ = _effective_calibration_assets(inventory, recipe)
    for item in override_issues:
        issue(item.code, item.message)

    # Raw overrides confirm only CFA identity; explicit header values retain
    # authority. Hash only when confirmations actually need to be bound.
    raw_assets = tuple(asset for asset in assets if asset.role in {AssetRole.LIGHT, AssetRole.FLAT, AssetRole.DARK, AssetRole.BIAS})
    confirmations: dict[str, str] = {}
    if recipe.raw_frame_metadata_overrides:
        by_digest: dict[str, list[FrameAsset]] = {}
        for asset in raw_assets:
            if asset.status is AssetStatus.READY:
                by_digest.setdefault(digest(asset.path), []).append(asset)
        for override in recipe.raw_frame_metadata_overrides:
            matched = by_digest.get(override.source_sha256, [])
            if len(matched) != 1:
                issue("RAW_FRAME_METADATA_OVERRIDE_SOURCE_AMBIGUOUS", "A raw-frame confirmation must bind exactly one current source.")
                continue
            asset = matched[0]
            cfa = override.cfa_pattern.strip().upper()
            if asset.cfa_explicit and cfa != asset.cfa_pattern.strip().upper():
                issue("RAW_CFA_OVERRIDE_CONFLICT", "A confirmation cannot replace explicit CFA header metadata.", affected=(asset,))
                continue
            confirmations[asset.path] = cfa
        assets = tuple(replace(asset, cfa_pattern=confirmations[asset.path]) if asset.path in confirmations else asset for asset in assets)
    for asset in raw_assets:
        cfa = confirmations.get(asset.path, asset.cfa_pattern.strip().upper())
        if not asset.cfa_explicit and asset.path not in confirmations and workflow != MONO_STANDARD:
            issue("CFA_CONFIRMATION_REQUIRED", "Missing raw CFA metadata requires an explicit mono/CFA confirmation.", affected=(asset,))
        elif cfa not in ({"NONE", "UNKNOWN", "UNSPECIFIED", ""} if workflow == MONO_STANDARD else {"NONE"}):
            issue("CFA_PIXEL_PIPELINE_UNSUPPORTED", "Bayer/CFA raw processing is not supported by this pixel pipeline.", affected=(asset,))

    effective_inventory = replace(inventory, assets=assets)
    # Overrides have already been applied and verified above.
    effective_recipe = replace(recipe, calibration=replace(recipe.calibration, master_metadata_overrides=()))
    planned_issues, _ = _calibration_issues(effective_inventory, effective_recipe)
    for item in planned_issues:
        issue(item.code, item.message, severity=item.severity.value, groups=tuple(item.details.get("lightGroups", ())))

    ready = tuple(asset for asset in assets if asset.status is AssetStatus.READY)
    lights = tuple(asset for asset in ready if asset.role is AssetRole.LIGHT)
    calibration = tuple(asset for asset in ready if asset.role in {AssetRole.BIAS, AssetRole.MASTER_BIAS, AssetRole.DARK, AssetRole.MASTER_DARK, AssetRole.FLAT, AssetRole.MASTER_FLAT})
    if not recipe.calibration.allow_masters and any(asset.is_master for asset in calibration):
        issue("MASTER_CALIBRATION_INPUT_DISABLED", "The recipe disables supplied calibration masters.")
    if recipe.calibration.flat is Requirement.DISABLED:
        issue("FLAT_DISABLED_UNSUPPORTED", "The public pixel executor requires Flat calibration.")
    if recipe.calibration.bias is Requirement.DISABLED:
        issue("BIAS_DISABLED_UNSUPPORTED", "The public pixel executor requires Bias calibration.")
    if not lights:
        issue("NO_LIGHTS", "At least one readable Light is required.")
    for light in lights:
        if light.exposure_seconds is None or light.exposure_seconds <= 0:
            issue("LIGHT_EXPOSURE_UNKNOWN", "Light requires a known positive exposure.", affected=(light,))
    unknown_roles = tuple(asset for asset in ready if asset.role is AssetRole.UNKNOWN)
    if unknown_roles:
        issue("FRAME_ROLE_UNKNOWN", "Every source requires a trustworthy acquisition role.", affected=unknown_roles)
    biases = tuple(asset for asset in calibration if asset.role is AssetRole.BIAS)
    master_biases = tuple(asset for asset in calibration if asset.role is AssetRole.MASTER_BIAS)
    if len(master_biases) > 1 or (biases and master_biases):
        issue("BIAS_SOURCE_AMBIGUOUS", "Supply raw Bias frames or exactly one MasterBias; this executor uses one calibration profile.", affected=(*biases, *master_biases))
    reference = (biases or master_biases or lights)
    conflicts = conflicting_profile_fields((*calibration, *lights), workflow)
    if conflicts:
        issue("CALIBRATION_PROFILE_UNSUPPORTED", "Known acquisition metadata conflicts: " + ", ".join(conflicts), affected=(*calibration, *lights))
    if reference:
        incompatible = tuple(asset for asset in (*calibration, *lights) if not _base_compatible(reference[0], asset, workflow))
        if incompatible:
            issue("CALIBRATION_PROFILE_UNSUPPORTED", "The current library requires known matching camera, gain, offset, dimensions, binning, CFA and readout mode. Separate incompatible acquisition profiles.", affected=incompatible)

    master_semantics = {item.source_sha256: item.bias_included for item in recipe.calibration.master_metadata_overrides}
    for asset in calibration:
        if asset.role is AssetRole.MASTER_DARK and workflow != MONO_STANDARD and master_semantics.get(digest(asset.path)) is None:
            issue("MASTER_DARK_BIAS_SEMANTICS_REQUIRED", "MasterDark must explicitly declare whether it already includes Bias; this cannot be inferred safely from pixels or its filename.", affected=(asset,))

    darks = tuple(asset for asset in calibration if asset.role in {AssetRole.DARK, AssetRole.MASTER_DARK})
    if workflow == MONO_STANDARD and not biases and not master_biases:
        from lightframeqc.readers import probe_frame_metadata
        semantics = []
        for dark in darks:
            included = True if not dark.is_master else resolve_dark_bias(master_semantics.get(digest(dark.path)), bias_from_header(probe_frame_metadata(Path(dark.path)).header), workflow)
            semantics.append((dark, included))
        raw_targets = (*lights, *(asset for asset in calibration if asset.role is AssetRole.FLAT))
        if not can_omit_bias(raw_targets, semantics, workflow):
            issue("BIAS_REQUIRED_FOR_CALIBRATION", "Bias is required unless every Light and raw Flat has a matching Dark that includes Bias.", affected=raw_targets)
    if darks:
        unmatched = tuple(light for light in lights if not any(_dark_compatible(light, dark, workflow) for dark in darks))
        if unmatched:
            issue("DARK_EXPOSURE_MISMATCH", "Imported Darks require an exact exposure and a temperature within 3 C for every Light; unmatched library entries are not selected automatically.", affected=unmatched, groups=tuple(asset.group_id for asset in unmatched))
        for dark in darks:
            if dark.exposure_seconds is None or dark.exposure_seconds <= 0:
                issue("DARK_EXPOSURE_UNKNOWN", "Dark requires a known positive exposure.", affected=(dark,))
        for index, dark in enumerate(darks):
            for other in darks[index + 1:]:
                if dark.exposure_seconds is None or other.exposure_seconds is None or not math.isclose(dark.exposure_seconds, other.exposure_seconds, rel_tol=0.0, abs_tol=1e-6):
                    continue
                if dark.role is AssetRole.MASTER_DARK or other.role is AssetRole.MASTER_DARK:
                    issue("DARK_SOURCE_AMBIGUOUS", "One exposure cannot contain multiple MasterDarks or both raw Darks and a MasterDark.", affected=(dark, other))
                    break
                if ((dark.temperature_celsius is None or other.temperature_celsius is None) and workflow != MONO_STANDARD) or (dark.temperature_celsius is not None and other.temperature_celsius is not None and abs(dark.temperature_celsius - other.temperature_celsius) > 3.0):
                    issue("DARK_TEMPERATURE_MISMATCH", "Raw Darks with one exposure require known temperatures within 3 C; separate temperature libraries.", affected=(dark, other))
                    break

    flat_sets: dict[str, list[FrameAsset]] = {}
    for asset in calibration:
        if asset.role in {AssetRole.FLAT, AssetRole.MASTER_FLAT}:
            flat_sets.setdefault(asset.filter_name.strip().upper(), []).append(asset)
    for filter_name, flat_set in flat_sets.items():
        master_count = sum(asset.role is AssetRole.MASTER_FLAT for asset in flat_set)
        if master_count > 1 or (master_count and master_count != len(flat_set)):
            issue("FLAT_SOURCE_AMBIGUOUS", f"Filter {filter_name} has multiple MasterFlats or both raw Flats and a MasterFlat.", affected=tuple(flat_set))
        raw_flats = tuple(asset for asset in flat_set if asset.role is AssetRole.FLAT)
        for flat in raw_flats:
            exact_darks = tuple(dark for dark in darks if flat.exposure_seconds is not None and dark.exposure_seconds is not None and math.isclose(flat.exposure_seconds, dark.exposure_seconds, rel_tol=0.0, abs_tol=1e-6))
            if exact_darks and not all(_dark_compatible(flat, dark, workflow) for dark in exact_darks):
                issue("FLAT_DARK_MISMATCH", "The exact-exposure FlatDark requires matching acquisition metadata and a temperature within 3 C of its Flat.", affected=(flat, *exact_darks))
        flat_dates = {asset.observed_at[:10] for asset in flat_set if asset.observed_at}
        light_dates = {asset.observed_at[:10] for asset in lights if asset.observed_at}
        if master_count or len(flat_dates | light_dates) > 1 or any(not asset.observed_at for asset in flat_set):
            issue("CAPTURE_SESSION_NOT_VERIFIED", f"Filter {filter_name}: dates do not prove the same dust, rotation or optical setup. Matching raw Flats are combined into one master; confirm that they describe the same optical setup.", severity="WARNING", affected=tuple(flat_set))
    # The project composition root publishes hash-bound overrides for its
    # generated masters. That current contract requires a known temperature,
    # including Bias/Flat masters even when no temperature subtraction is used.
    generated_references = [min(biases, key=lambda item: item.path.casefold())] if biases else []
    generated_references.extend(min(raw, key=lambda item: item.path.casefold()) for flat_set in flat_sets.values() if (raw := [asset for asset in flat_set if asset.role is AssetRole.FLAT]))
    for asset in generated_references:
        if asset.temperature_celsius is None and workflow != MONO_STANDARD:
            issue("RAW_MASTER_METADATA_MISSING", "Shared master generation currently requires temperature metadata on its raw Bias/Flat reference. No temperature value will be guessed.", affected=(asset,))

    if workflow == MONO_STANDARD:
        for dark in darks:
            if dark.temperature_celsius is None or any(light.temperature_celsius is None for light in lights):
                issue("DARK_TEMPERATURE_UNRECORDED", "Dark temperature compatibility is unverified because a source temperature was not recorded.", severity="WARNING", affected=(dark,))

    grouped: dict[str, list[FrameAsset]] = {}
    for light in lights:
        grouped.setdefault(light.group_id, []).append(light)
    report_groups = []
    for group_id, members in sorted(grouped.items()):
        first = members[0]
        matches = {}
        for label, raw_role, master_role, compatible in (
            ("FLAT", AssetRole.FLAT, AssetRole.MASTER_FLAT, _flat_compatible),
            ("DARK", AssetRole.DARK, AssetRole.MASTER_DARK, _dark_compatible),
            ("BIAS", AssetRole.BIAS, AssetRole.MASTER_BIAS, _base_compatible),
        ):
            matching = tuple(asset for asset in calibration if asset.role in {raw_role, master_role} and compatible(first, asset, workflow))
            matches[label] = {"rawCount": sum(asset.role is raw_role for asset in matching), "masterCount": sum(asset.role is master_role for asset in matching)}
        blocked = any(item["severity"] == "ERROR" and (not item["lightGroups"] or group_id in item["lightGroups"]) for item in issues)
        report_groups.append({"groupId": group_id, "target": first.target, "filter": first.filter_name, "lightCount": len(members), "observedDates": sorted({asset.observed_at[:10] for asset in members if asset.observed_at}), "status": "BLOCKED" if blocked else "READY", "matches": matches})
    blocked = any(item["severity"] == "ERROR" for item in issues)
    return {"schemaVersion": 1, "calibrationPolicy": workflow_receipt(workflow), "status": "BLOCKED" if blocked else "READY", "calibrationReady": not blocked, "groups": report_groups, "issues": issues}
