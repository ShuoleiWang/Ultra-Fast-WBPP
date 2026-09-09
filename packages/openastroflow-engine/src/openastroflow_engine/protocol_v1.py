"""Strict Python codec for the Rust ``app-core`` worker protocol v1.

The module deliberately owns no transport.  It validates and frames one NDJSON
record at a time and exposes :class:`ProtocolCursor` for the state that belongs
to one ordered, unidirectional stream.  Native host paths are opaque strings;
artifact relative paths use the stricter portable identity contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re
import unicodedata
from typing import Any, Mapping


PROTOCOL_VERSION = 1
MAX_NDJSON_LINE_BYTES = 8 * 1024 * 1024

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MESSAGE_TYPES = {
    "handshake",
    "plan",
    "execute",
    "progress",
    "artifact",
    "error",
}
_CONTROLLER_MESSAGES = {"plan", "execute"}
_WORKER_MESSAGES = {"progress", "artifact", "error"}
_HARDWARE_PROFILES = {
    "portable-cpu",
    "generic-arm64-cpu",
    "generic-apple-metal",
    "m3-pro-tuned",
    "windows-cpu",
}
_STAGE_KINDS = {
    "ingest",
    "quality-control",
    "calibration",
    "cosmetic-correction",
    "debayer",
    "registration",
    "local-normalization",
    "integration",
    "drizzle",
    "astrometric-solve",
    "mosaic",
    "export",
}
_RESULT_REQUIREMENTS = {"disabled", "best-effort", "required"}
_INPUT_ROLES = {
    "light",
    "flat",
    "dark",
    "bias",
    "master-flat",
    "master-dark",
    "master-bias",
}
_BACKEND_FEATURES = {
    "cpu-execution",
    "metal-execution",
    "m3-pro-tuning",
    "checkpoint-resume",
    "deterministic-receipts",
    "offline-astrometric-solver",
    "drizzle",
    "mosaic",
    "fits",
    "xisf",
}
_PROGRESS_STATES = {
    "queued",
    "running",
    "finalizing",
    "succeeded",
    "failed",
    "cancelled",
}
_STAGE_STATUSES = {
    "pending",
    "running",
    "succeeded",
    "failed",
    "skipped",
    "cancelled",
}
_ARTIFACT_KINDS = {
    "frame-manifest",
    "quality-report",
    "master-bias",
    "master-dark",
    "master-flat",
    "calibrated-light",
    "registered-light",
    "local-normalization-model",
    "drizzle-data",
    "integration-master",
    "drizzled-master",
    "solved-master",
    "mosaic-master",
    "final-master",
    "run-log",
}
_ARTIFACT_DESIGNATIONS = {"intermediate", "diagnostic", "final-master"}
_ASTROMETRIC_PARITIES = {"POSITIVE", "NEGATIVE"}


class ProtocolV1Error(ValueError):
    """Stable, path-aware protocol failure."""

    def __init__(self, code: str, message: str, path: str = "") -> None:
        self.code = code
        self.path = path
        self.message = message
        prefix = f"{path}: " if path else ""
        super().__init__(f"{prefix}{message}")


def _fail(code: str, path: str, message: str) -> None:
    raise ProtocolV1Error(code, message, path)


def _strict_object(
    value: Any,
    path: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail("invalid-type", path, "must be an object with string keys")
    missing = sorted(required - set(value))
    if missing:
        _fail("missing-field", path, f"missing required fields: {', '.join(missing)}")
    unknown = sorted(set(value) - required - optional)
    if unknown:
        _fail("unknown-field", path, f"unknown fields: {', '.join(unknown)}")
    return value


def _string(value: Any, path: str, *, nonblank: bool = False) -> str:
    if not isinstance(value, str):
        _fail("invalid-type", path, "must be a string")
    if nonblank and not value.strip():
        _fail("invalid-value", path, "must not be blank")
    return value


def _identifier(value: Any, path: str) -> str:
    value = _string(value, path)
    if not _IDENTIFIER.fullmatch(value):
        _fail(
            "invalid-identifier",
            path,
            "must be 1-128 ASCII letters, digits, dot, underscore, or hyphen and begin alphanumeric",
        )
    return value


def _integer(value: Any, path: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("invalid-type", path, "must be an integer")
    if value < 0 or value > maximum:
        _fail("invalid-value", path, f"must be in [0, {maximum}]")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("invalid-type", path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        _fail("non-finite-number", path, "must be finite")
    return result


def _enum(value: Any, path: str, choices: set[str]) -> str:
    value = _string(value, path)
    if value not in choices:
        _fail("invalid-enum", path, f"unsupported value {value!r}")
    return value


def _string_array(value: Any, path: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list):
        _fail("invalid-type", path, "must be an array")
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if unique and len(set(result)) != len(result):
        _fail("duplicate-value", path, "must contain unique values")
    return result


def _finite_json(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("non-finite-number", path, "must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _finite_json(item, f"{path}.{key}")
        return
    _fail("invalid-json-value", path, f"unsupported JSON value {type(value).__name__}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_project(project: Any, path: str) -> None:
    value = _strict_object(
        project,
        path,
        required={"schemaVersion", "projectId", "displayName", "createdAtUnixMs", "sources"},
        optional={"labels"},
    )
    if _integer(value["schemaVersion"], f"{path}.schemaVersion", 65535) != 1:
        _fail("unsupported-schema-version", f"{path}.schemaVersion", "expected 1")
    _identifier(value["projectId"], f"{path}.projectId")
    _string(value["displayName"], f"{path}.displayName", nonblank=True)
    _integer(value["createdAtUnixMs"], f"{path}.createdAtUnixMs", 2**64 - 1)
    sources = value["sources"]
    if not isinstance(sources, list) or not sources:
        _fail("invalid-value", f"{path}.sources", "must be a non-empty array")
    source_ids: set[str] = set()
    for index, raw_source in enumerate(sources):
        source_path = f"{path}.sources[{index}]"
        source = _strict_object(
            raw_source,
            source_path,
            required={"sourceId", "role", "hostPath"},
            optional={"recursive", "filter"},
        )
        source_id = _identifier(source["sourceId"], f"{source_path}.sourceId")
        if source_id in source_ids:
            _fail("duplicate-value", f"{source_path}.sourceId", "duplicate source identifier")
        source_ids.add(source_id)
        _enum(source["role"], f"{source_path}.role", _INPUT_ROLES)
        host_path = _string(source["hostPath"], f"{source_path}.hostPath", nonblank=True)
        if "\0" in host_path:
            _fail("invalid-host-path", f"{source_path}.hostPath", "must not contain NUL")
        if "recursive" in source and not isinstance(source["recursive"], bool):
            _fail("invalid-type", f"{source_path}.recursive", "must be boolean")
        if "filter" in source and source["filter"] is not None:
            _string(source["filter"], f"{source_path}.filter")
    labels = value.get("labels", {})
    if not isinstance(labels, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in labels.items()
    ):
        _fail("invalid-type", f"{path}.labels", "must map strings to strings")


def _validate_recipe(recipe: Any, path: str) -> None:
    value = _strict_object(
        recipe,
        path,
        required={"schemaVersion", "recipeId", "displayName", "stages", "solver", "drizzle"},
        optional={"parameters"},
    )
    if _integer(value["schemaVersion"], f"{path}.schemaVersion", 65535) != 1:
        _fail("unsupported-schema-version", f"{path}.schemaVersion", "expected 1")
    _identifier(value["recipeId"], f"{path}.recipeId")
    _string(value["displayName"], f"{path}.displayName", nonblank=True)
    stages = value["stages"]
    if not isinstance(stages, list) or not stages:
        _fail("invalid-value", f"{path}.stages", "must be a non-empty array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_stage in enumerate(stages):
        stage_path = f"{path}.stages[{index}]"
        stage = _strict_object(
            raw_stage,
            stage_path,
            required={"stageId", "kind"},
            optional={"enabled", "dependsOn", "parameters"},
        )
        stage_id = _identifier(stage["stageId"], f"{stage_path}.stageId")
        if stage_id in by_id:
            _fail("duplicate-value", f"{stage_path}.stageId", "duplicate stage identifier")
        by_id[stage_id] = stage
        _enum(stage["kind"], f"{stage_path}.kind", _STAGE_KINDS)
        if "enabled" in stage and not isinstance(stage["enabled"], bool):
            _fail("invalid-type", f"{stage_path}.enabled", "must be boolean")
        dependencies = _string_array(stage.get("dependsOn", []), f"{stage_path}.dependsOn", unique=True)
        for dependency in dependencies:
            _identifier(dependency, f"{stage_path}.dependsOn")
        parameters = stage.get("parameters", {})
        if not isinstance(parameters, Mapping) or not all(isinstance(key, str) for key in parameters):
            _fail("invalid-type", f"{stage_path}.parameters", "must be an object")
        _finite_json(parameters, f"{stage_path}.parameters")
    for index, stage in enumerate(stages):
        enabled = stage.get("enabled", True)
        for dependency in stage.get("dependsOn", []):
            if dependency not in by_id:
                _fail("unknown-dependency", f"{path}.stages[{index}].dependsOn", dependency)
            if enabled and not by_id[dependency].get("enabled", True):
                _fail(
                    "disabled-dependency",
                    f"{path}.stages[{index}].dependsOn",
                    f"enabled stage depends on disabled stage {dependency}",
                )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        if stage_id in visiting:
            _fail("dependency-cycle", f"{path}.stages", f"cycle includes {stage_id}")
        visiting.add(stage_id)
        for dependency in by_id[stage_id].get("dependsOn", []):
            visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id in by_id:
        visit(stage_id)

    solver = _strict_object(
        value["solver"],
        f"{path}.solver",
        required={"result"},
        optional={"catalog", "projection", "minimumMatches", "maximumRmsArcsec"},
    )
    solver_result = _enum(solver["result"], f"{path}.solver.result", _RESULT_REQUIREMENTS)
    catalog = _string(solver.get("catalog", "gaia-dr3-offline"), f"{path}.solver.catalog")
    projection = _string(solver.get("projection", "TAN"), f"{path}.solver.projection")
    minimum_matches = _integer(solver.get("minimumMatches", 12), f"{path}.solver.minimumMatches", 2**32 - 1)
    maximum_rms = _number(solver.get("maximumRmsArcsec", 2.0), f"{path}.solver.maximumRmsArcsec")
    if solver_result != "disabled":
        if not catalog.strip() or not projection.strip():
            _fail("invalid-value", f"{path}.solver", "catalog and projection must not be blank")
        if minimum_matches < 3:
            _fail("invalid-value", f"{path}.solver.minimumMatches", "must be at least 3")
        if maximum_rms <= 0:
            _fail("invalid-value", f"{path}.solver.maximumRmsArcsec", "must be positive")

    drizzle = _strict_object(
        value["drizzle"],
        f"{path}.drizzle",
        required={"result"},
        optional={"scale", "dropShrink", "kernel"},
    )
    drizzle_result = _enum(drizzle["result"], f"{path}.drizzle.result", _RESULT_REQUIREMENTS)
    scale = _number(drizzle.get("scale", 2.0), f"{path}.drizzle.scale")
    drop_shrink = _number(drizzle.get("dropShrink", 0.9), f"{path}.drizzle.dropShrink")
    kernel = _string(drizzle.get("kernel", "square"), f"{path}.drizzle.kernel")
    if drizzle_result != "disabled":
        if not 1 <= scale <= 4:
            _fail("invalid-value", f"{path}.drizzle.scale", "must be in [1, 4]")
        if not 0 < drop_shrink <= 1:
            _fail("invalid-value", f"{path}.drizzle.dropShrink", "must be in (0, 1]")
        if not kernel.strip():
            _fail("invalid-value", f"{path}.drizzle.kernel", "must not be blank")

    enabled_kinds = {stage["kind"] for stage in stages if stage.get("enabled", True)}
    if solver_result != "disabled" and "astrometric-solve" not in enabled_kinds:
        _fail("required-stage-missing", f"{path}.solver.result", "requires astrometric-solve")
    if drizzle_result != "disabled" and "drizzle" not in enabled_kinds:
        _fail("required-stage-missing", f"{path}.drizzle.result", "requires drizzle")
    parameters = value.get("parameters", {})
    if not isinstance(parameters, Mapping) or not all(isinstance(key, str) for key in parameters):
        _fail("invalid-type", f"{path}.parameters", "must be an object")
    _finite_json(parameters, f"{path}.parameters")


def _validate_capabilities(raw: Any, path: str) -> None:
    value = _strict_object(
        raw,
        path,
        required={
            "schemaVersion",
            "backendId",
            "backendVersion",
            "workerBuild",
            "hardwareProfiles",
            "stages",
            "features",
            "maximumParallelStages",
        },
        optional={"inputExtensions", "outputExtensions"},
    )
    if _integer(value["schemaVersion"], f"{path}.schemaVersion", 65535) != 1:
        _fail("unsupported-schema-version", f"{path}.schemaVersion", "expected 1")
    _identifier(value["backendId"], f"{path}.backendId")
    _string(value["backendVersion"], f"{path}.backendVersion", nonblank=True)
    _string(value["workerBuild"], f"{path}.workerBuild", nonblank=True)
    profiles = _string_array(value["hardwareProfiles"], f"{path}.hardwareProfiles", unique=True)
    stages = _string_array(value["stages"], f"{path}.stages", unique=True)
    features = _string_array(value["features"], f"{path}.features", unique=True)
    if not profiles or not stages:
        _fail("invalid-value", path, "at least one hardware profile and stage are required")
    for index, profile in enumerate(profiles):
        _enum(profile, f"{path}.hardwareProfiles[{index}]", _HARDWARE_PROFILES)
    for index, stage in enumerate(stages):
        _enum(stage, f"{path}.stages[{index}]", _STAGE_KINDS)
    for index, feature in enumerate(features):
        _enum(feature, f"{path}.features[{index}]", _BACKEND_FEATURES)
    if _integer(value["maximumParallelStages"], f"{path}.maximumParallelStages", 65535) == 0:
        _fail("invalid-value", f"{path}.maximumParallelStages", "must be positive")
    for field in ("inputExtensions", "outputExtensions"):
        extensions = _string_array(value.get(field, []), f"{path}.{field}", unique=True)
        if any(
            not extension
            or not all(
                character.isascii()
                and (character.isdigit() or character.islower())
                for character in extension
            )
            for extension in extensions
        ):
            _fail("invalid-value", f"{path}.{field}", "extensions must be lowercase alphanumeric")
    feature_set = set(features)
    for profile in profiles:
        required = {
            "portable-cpu": "cpu-execution",
            "generic-arm64-cpu": "cpu-execution",
            "windows-cpu": "cpu-execution",
            "generic-apple-metal": "metal-execution",
            "m3-pro-tuned": "m3-pro-tuning",
        }[profile]
        if required not in feature_set or (profile == "m3-pro-tuned" and "metal-execution" not in feature_set):
            _fail("capability-mismatch", f"{path}.hardwareProfiles", f"{profile} lacks required feature")


def _validate_receipt_error(raw: Any, path: str) -> None:
    value = _strict_object(raw, path, required={"code", "message", "retryable"})
    _string(value["code"], f"{path}.code")
    _string(value["message"], f"{path}.message")
    if not isinstance(value["retryable"], bool):
        _fail("invalid-type", f"{path}.retryable", "must be boolean")


def _validate_stage_receipt(raw: Any, path: str) -> Mapping[str, Any]:
    value = _strict_object(
        raw,
        path,
        required={"schemaVersion", "stageId", "kind", "status", "startedAtUnixMs"},
        optional={"finishedAtUnixMs", "artifactIds", "metrics", "error"},
    )
    if _integer(value["schemaVersion"], f"{path}.schemaVersion", 65535) != 1:
        _fail("unsupported-schema-version", f"{path}.schemaVersion", "expected 1")
    _identifier(value["stageId"], f"{path}.stageId")
    _enum(value["kind"], f"{path}.kind", _STAGE_KINDS)
    status = _enum(value["status"], f"{path}.status", _STAGE_STATUSES)
    started = _integer(value["startedAtUnixMs"], f"{path}.startedAtUnixMs", 2**64 - 1)
    finished = value.get("finishedAtUnixMs")
    if finished is not None:
        finished = _integer(finished, f"{path}.finishedAtUnixMs", 2**64 - 1)
        if finished < started:
            _fail("invalid-timing", f"{path}.finishedAtUnixMs", "must not precede start")
    terminal = status in {"succeeded", "failed", "skipped", "cancelled"}
    if terminal != (finished is not None):
        _fail("invalid-timing", f"{path}.finishedAtUnixMs", "must be present exactly for terminal status")
    _string_array(value.get("artifactIds", []), f"{path}.artifactIds")
    metrics = value.get("metrics", {})
    if not isinstance(metrics, Mapping) or not all(isinstance(key, str) for key in metrics):
        _fail("invalid-type", f"{path}.metrics", "must be an object")
    for key, metric in metrics.items():
        _number(metric, f"{path}.metrics.{key}")
    error = value.get("error")
    if status == "failed" and error is None:
        _fail("missing-field", f"{path}.error", "failed stage requires an error")
    if status != "failed" and error is not None:
        _fail("invalid-value", f"{path}.error", "only failed stage may include an error")
    if error is not None:
        _validate_receipt_error(error, f"{path}.error")
    return value


def _portable_relative_path(value: Any, path: str, *, single_component: bool = False) -> str:
    value = _string(value, path)
    if not 1 <= len(value.encode("utf-8")) <= 4096:
        _fail("invalid-relative-path", path, "must contain 1-4096 UTF-8 bytes")
    if value.startswith("/") or "\\" in value or "\0" in value:
        _fail("invalid-relative-path", path, "must be relative and use forward slashes")
    components = value.split("/")
    if single_component and len(components) != 1:
        _fail("invalid-relative-path", path, "must be one path component")
    for component in components:
        if not component or component in {".", ".."} or len(component.encode("utf-8")) > 255:
            _fail("invalid-relative-path", path, "contains an invalid component")
        if component.endswith((".", " ")):
            _fail("invalid-relative-path", path, "component ends in dot or space")
        if any(
            unicodedata.category(character) == "Cc" or character in ':*?"<>|'
            for character in component
        ):
            _fail("invalid-relative-path", path, "contains a cross-platform forbidden character")
        basename = component.split(".", 1)[0].upper()
        if basename in {"CON", "PRN", "AUX", "NUL"} or (
            len(basename) == 4
            and basename[:3] in {"COM", "LPT"}
            and basename[3] in "123456789"
        ):
            _fail("invalid-relative-path", path, "contains a Windows-reserved device name")
    return value


def _validate_astrometry(raw: Any, path: str) -> None:
    value = _strict_object(
        raw,
        path,
        required={
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
            "catalogManaged",
            "installedSetIdentity",
            "catalogManifestSha256",
            "indexArtifacts",
            "wcsSha256",
        },
    )
    _string(value["referenceFrame"], f"{path}.referenceFrame", nonblank=True)
    _string(value["projection"], f"{path}.projection", nonblank=True)
    ra = _number(value["centerRaDegrees"], f"{path}.centerRaDegrees")
    dec = _number(value["centerDecDegrees"], f"{path}.centerDecDegrees")
    if not 0 <= ra < 360 or not -90 <= dec <= 90:
        _fail("invalid-value", path, "RA/declination are outside their valid ranges")
    pixel_scale_arcsec = _number(value["pixelScaleArcsec"], f"{path}.pixelScaleArcsec")
    if pixel_scale_arcsec <= 0:
        _fail("invalid-value", f"{path}.pixelScaleArcsec", "must be positive")
    _number(value["rotationDegrees"], f"{path}.rotationDegrees")
    rms_pixels = _number(value["rmsPixels"], f"{path}.rmsPixels")
    rms_arcsec = _number(value["rmsArcsec"], f"{path}.rmsArcsec")
    if rms_pixels < 0 or rms_arcsec < 0:
        _fail("invalid-value", f"{path}.rms", "pixel and angular RMS must be non-negative")
    expected_arcsec = rms_pixels * pixel_scale_arcsec
    consistent = rms_arcsec <= 1e-9 if expected_arcsec == 0 else 0.5 <= rms_arcsec / expected_arcsec <= 2.0
    if not consistent:
        _fail("invalid-value", f"{path}.rms", "pixel and angular RMS disagree with pixel scale")
    if _integer(value["matchedStars"], f"{path}.matchedStars", 2**32 - 1) < 3:
        _fail("invalid-value", f"{path}.matchedStars", "must be at least 3")
    _enum(value["parity"], f"{path}.parity", _ASTROMETRIC_PARITIES)
    index_identities = _string_array(
        value["indexIdentities"], f"{path}.indexIdentities", unique=True
    )
    if not index_identities or any(not identity.strip() for identity in index_identities):
        _fail("invalid-value", f"{path}.indexIdentities", "must contain non-blank identities")
    if value["catalogManaged"] is not True:
        _fail(
            "invalid-value",
            f"{path}.catalogManaged",
            "a publishable astrometric receipt requires a managed catalog",
        )
    for field in (
        "catalogIdentity",
        "correspondenceSha256",
        "installedSetIdentity",
        "catalogManifestSha256",
        "wcsSha256",
    ):
        if not _SHA256.fullmatch(_string(value[field], f"{path}.{field}")):
            _fail("invalid-sha256", f"{path}.{field}", "must be lowercase SHA-256")
    index_artifacts = value["indexArtifacts"]
    if not isinstance(index_artifacts, list) or not index_artifacts:
        _fail("invalid-value", f"{path}.indexArtifacts", "must contain at least one managed index")
    seen_index_ids: set[str] = set()
    logical_index_ids: set[str] = set()
    for identity in index_identities:
        match = re.fullmatch(
            r"astrometry\.net:index:([0-9]+):healpix:[^:]+:hpnside:[^:]+",
            identity,
        )
        if match is None:
            _fail("invalid-value", f"{path}.indexIdentities", "contains an unsupported index identity")
        logical_index_ids.add(match.group(1))
    for index, raw_artifact in enumerate(index_artifacts):
        artifact_path = f"{path}.indexArtifacts[{index}]"
        artifact = _strict_object(
            raw_artifact,
            artifact_path,
            required={
                "indexId",
                "relativeName",
                "sizeBytes",
                "sha256",
                "manifestSha256",
                "installedSetIdentity",
            },
        )
        index_id = _string(artifact["indexId"], f"{artifact_path}.indexId")
        if not re.fullmatch(r"[0-9]+", index_id) or index_id in seen_index_ids:
            _fail("invalid-value", f"{artifact_path}.indexId", "must be a unique decimal INDEXID")
        seen_index_ids.add(index_id)
        if artifact["relativeName"] != f"index-{index_id}.fits":
            _fail("invalid-value", f"{artifact_path}.relativeName", "must be the managed index filename")
        if _integer(artifact["sizeBytes"], f"{artifact_path}.sizeBytes", 2**64 - 1) == 0:
            _fail("invalid-value", f"{artifact_path}.sizeBytes", "must be positive")
        for field in ("sha256", "manifestSha256", "installedSetIdentity"):
            if not _SHA256.fullmatch(_string(artifact[field], f"{artifact_path}.{field}")):
                _fail("invalid-sha256", f"{artifact_path}.{field}", "must be lowercase SHA-256")
        if artifact["manifestSha256"] != value["catalogManifestSha256"]:
            _fail("invalid-value", f"{artifact_path}.manifestSha256", "disagrees with catalogManifestSha256")
        if artifact["installedSetIdentity"] != value["installedSetIdentity"]:
            _fail("invalid-value", f"{artifact_path}.installedSetIdentity", "disagrees with installedSetIdentity")
    if seen_index_ids != logical_index_ids:
        _fail("invalid-value", f"{path}.indexArtifacts", "does not match indexIdentities")


def _validate_drizzle_receipt(raw: Any, path: str) -> None:
    value = _strict_object(
        raw,
        path,
        required={"scale", "dropShrink", "kernel", "inputFrames", "outputWidth", "outputHeight"},
    )
    scale = _number(value["scale"], f"{path}.scale")
    drop = _number(value["dropShrink"], f"{path}.dropShrink")
    if not 1 <= scale <= 4 or not 0 < drop <= 1:
        _fail("invalid-value", path, "scale or dropShrink is outside its valid range")
    if not _string(value["kernel"], f"{path}.kernel").strip():
        _fail("invalid-value", f"{path}.kernel", "must not be blank")
    for field in ("inputFrames", "outputWidth", "outputHeight"):
        if _integer(value[field], f"{path}.{field}", 2**32 - 1) == 0:
            _fail("invalid-value", f"{path}.{field}", "must be positive")


def _validate_artifact_receipt(raw: Any, path: str) -> Mapping[str, Any]:
    value = _strict_object(
        raw,
        path,
        required={
            "schemaVersion",
            "artifactId",
            "producedByStageId",
            "kind",
            "designation",
            "relativePath",
            "mediaType",
            "sha256",
            "sizeBytes",
            "createdAtUnixMs",
        },
        optional={"astrometry", "drizzle", "attributes"},
    )
    if _integer(value["schemaVersion"], f"{path}.schemaVersion", 65535) != 1:
        _fail("unsupported-schema-version", f"{path}.schemaVersion", "expected 1")
    _identifier(value["artifactId"], f"{path}.artifactId")
    _identifier(value["producedByStageId"], f"{path}.producedByStageId")
    _enum(value["kind"], f"{path}.kind", _ARTIFACT_KINDS)
    _enum(value["designation"], f"{path}.designation", _ARTIFACT_DESIGNATIONS)
    _portable_relative_path(value["relativePath"], f"{path}.relativePath")
    media_type = _string(value["mediaType"], f"{path}.mediaType", nonblank=True)
    if "/" not in media_type:
        _fail("invalid-value", f"{path}.mediaType", "must be a media type")
    if not _SHA256.fullmatch(_string(value["sha256"], f"{path}.sha256")):
        _fail("invalid-sha256", f"{path}.sha256", "must be lowercase SHA-256")
    _integer(value["sizeBytes"], f"{path}.sizeBytes", 2**64 - 1)
    _integer(value["createdAtUnixMs"], f"{path}.createdAtUnixMs", 2**64 - 1)
    if value.get("astrometry") is not None:
        _validate_astrometry(value["astrometry"], f"{path}.astrometry")
    if value.get("drizzle") is not None:
        _validate_drizzle_receipt(value["drizzle"], f"{path}.drizzle")
    attributes = value.get("attributes", {})
    if not isinstance(attributes, Mapping) or not all(isinstance(key, str) for key in attributes):
        _fail("invalid-type", f"{path}.attributes", "must be an object")
    _finite_json(attributes, f"{path}.attributes")
    return value


def _validate_payload(message_type: str, raw: Any) -> None:
    path = f"{message_type}.payload"
    if message_type == "handshake":
        value = _strict_object(
            raw,
            path,
            required={"role", "implementation", "implementationVersion", "supportedProtocolVersions"},
            optional={"capabilities"},
        )
        role = _enum(value["role"], f"{path}.role", {"controller", "worker"})
        _identifier(value["implementation"], f"{path}.implementation")
        _string(value["implementationVersion"], f"{path}.implementationVersion", nonblank=True)
        versions = value["supportedProtocolVersions"]
        if not isinstance(versions, list):
            _fail("invalid-type", f"{path}.supportedProtocolVersions", "must be an array")
        parsed_versions = [
            _integer(item, f"{path}.supportedProtocolVersions[{index}]", 65535)
            for index, item in enumerate(versions)
        ]
        if PROTOCOL_VERSION not in parsed_versions:
            _fail("unsupported-protocol-version", f"{path}.supportedProtocolVersions", "must include 1")
        capabilities = value.get("capabilities")
        if role == "worker" and capabilities is None:
            _fail("missing-field", f"{path}.capabilities", "worker must advertise capabilities")
        if role == "controller" and capabilities is not None:
            _fail("invalid-value", f"{path}.capabilities", "controller must not advertise worker capabilities")
        if capabilities is not None:
            _validate_capabilities(capabilities, f"{path}.capabilities")
        return
    if message_type == "plan":
        value = _strict_object(
            raw,
            path,
            required={"requestId", "planId", "project", "recipe", "requestedHardwareProfile", "inputManifestSha256"},
        )
        _identifier(value["requestId"], f"{path}.requestId")
        _identifier(value["planId"], f"{path}.planId")
        _validate_project(value["project"], f"{path}.project")
        _validate_recipe(value["recipe"], f"{path}.recipe")
        _enum(value["requestedHardwareProfile"], f"{path}.requestedHardwareProfile", _HARDWARE_PROFILES)
        if not _SHA256.fullmatch(_string(value["inputManifestSha256"], f"{path}.inputManifestSha256")):
            _fail("invalid-sha256", f"{path}.inputManifestSha256", "must be lowercase SHA-256")
        return
    if message_type == "execute":
        value = _strict_object(
            raw,
            path,
            required={"requestId", "planId", "runId", "outputParentHostPath", "outputDirectoryName"},
            optional={"resumeFromRunId"},
        )
        for field in ("requestId", "planId", "runId"):
            _identifier(value[field], f"{path}.{field}")
        host_path = _string(value["outputParentHostPath"], f"{path}.outputParentHostPath", nonblank=True)
        if "\0" in host_path:
            _fail("invalid-host-path", f"{path}.outputParentHostPath", "must not contain NUL")
        _portable_relative_path(value["outputDirectoryName"], f"{path}.outputDirectoryName", single_component=True)
        resume = value.get("resumeFromRunId")
        if resume is not None:
            _identifier(resume, f"{path}.resumeFromRunId")
            if resume == value["runId"]:
                _fail("invalid-value", f"{path}.resumeFromRunId", "must differ from runId")
        return
    if message_type == "progress":
        value = _strict_object(
            raw,
            path,
            required={"requestId", "runId", "state", "fraction", "message"},
            optional={"stageId", "completedUnits", "totalUnits"},
        )
        _identifier(value["requestId"], f"{path}.requestId")
        _identifier(value["runId"], f"{path}.runId")
        if value.get("stageId") is not None:
            _identifier(value["stageId"], f"{path}.stageId")
        _enum(value["state"], f"{path}.state", _PROGRESS_STATES)
        fraction = _number(value["fraction"], f"{path}.fraction")
        if not 0 <= fraction <= 1:
            _fail("invalid-value", f"{path}.fraction", "must be in [0, 1]")
        completed = value.get("completedUnits")
        total = value.get("totalUnits")
        if (completed is None) != (total is None):
            _fail("invalid-value", f"{path}.completedUnits", "completed and total must both be absent or present")
        if completed is not None:
            completed = _integer(completed, f"{path}.completedUnits", 2**64 - 1)
            total = _integer(total, f"{path}.totalUnits", 2**64 - 1)
            if total == 0 or completed > total:
                _fail("invalid-value", f"{path}.completedUnits", "must satisfy completed <= total and total > 0")
        _string(value["message"], f"{path}.message", nonblank=True)
        return
    if message_type == "artifact":
        value = _strict_object(raw, path, required={"requestId", "runId", "stage", "artifact"})
        _identifier(value["requestId"], f"{path}.requestId")
        _identifier(value["runId"], f"{path}.runId")
        stage = _validate_stage_receipt(value["stage"], f"{path}.stage")
        artifact = _validate_artifact_receipt(value["artifact"], f"{path}.artifact")
        if stage["stageId"] != artifact["producedByStageId"]:
            _fail("receipt-link-mismatch", f"{path}.artifact.producedByStageId", "must match stage receipt")
        if artifact["artifactId"] not in stage.get("artifactIds", []):
            _fail("receipt-link-mismatch", f"{path}.stage.artifactIds", "must contain artifact id")
        return
    if message_type == "error":
        value = _strict_object(
            raw,
            path,
            required={"code", "message", "retryable"},
            optional={"requestId", "runId", "stageId", "details"},
        )
        for field in ("requestId", "runId", "stageId"):
            if value.get(field) is not None:
                _identifier(value[field], f"{path}.{field}")
        _identifier(value["code"], f"{path}.code")
        _string(value["message"], f"{path}.message", nonblank=True)
        if not isinstance(value["retryable"], bool):
            _fail("invalid-type", f"{path}.retryable", "must be boolean")
        details = value.get("details", {})
        if not isinstance(details, Mapping) or not all(isinstance(key, str) for key in details):
            _fail("invalid-type", f"{path}.details", "must be an object")
        _finite_json(details, f"{path}.details")
        return
    _fail("unknown-message-type", "type", f"unsupported message type {message_type!r}")


@dataclass(frozen=True, slots=True)
class WorkerEnvelope:
    protocol_version: int
    session_id: str
    sequence: int
    sent_at_unix_ms: int
    message_type: str
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Any) -> "WorkerEnvelope":
        value = _strict_object(
            raw,
            "envelope",
            required={"protocolVersion", "sessionId", "sequence", "sentAtUnixMs", "type", "payload"},
        )
        protocol_version = _integer(value["protocolVersion"], "protocolVersion", 65535)
        if protocol_version != PROTOCOL_VERSION:
            _fail("unsupported-protocol-version", "protocolVersion", "expected 1")
        session_id = _identifier(value["sessionId"], "sessionId")
        sequence = _integer(value["sequence"], "sequence", 2**64 - 1)
        sent_at = _integer(value["sentAtUnixMs"], "sentAtUnixMs", 2**64 - 1)
        message_type = _enum(value["type"], "type", _MESSAGE_TYPES)
        _validate_payload(message_type, value["payload"])
        return cls(
            protocol_version=protocol_version,
            session_id=session_id,
            sequence=sequence,
            sent_at_unix_ms=sent_at,
            message_type=message_type,
            payload=deepcopy(dict(value["payload"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocolVersion": self.protocol_version,
            "sessionId": self.session_id,
            "sequence": self.sequence,
            "sentAtUnixMs": self.sent_at_unix_ms,
            "type": self.message_type,
            "payload": deepcopy(self.payload),
        }


def decode_ndjson_line(line: bytes | str) -> WorkerEnvelope:
    """Decode and validate exactly one v1 NDJSON record."""

    if isinstance(line, str):
        try:
            encoded = line.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ProtocolV1Error("invalid-utf8", str(error)) from error
    elif isinstance(line, bytes):
        encoded = line
    else:
        raise TypeError("line must be bytes or str")
    if len(encoded) > MAX_NDJSON_LINE_BYTES:
        raise ProtocolV1Error("line-too-long", f"record exceeds {MAX_NDJSON_LINE_BYTES} bytes")
    if encoded.endswith(b"\r\n"):
        record = encoded[:-2]
    elif encoded.endswith(b"\n"):
        record = encoded[:-1]
    else:
        record = encoded
    if not record:
        raise ProtocolV1Error("empty-line", "empty NDJSON record")
    if b"\n" in record or b"\r" in record:
        raise ProtocolV1Error("multiple-records", "input contains more than one NDJSON record")
    try:
        text = record.decode("utf-8")
        raw = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ProtocolV1Error("invalid-json", str(error)) from error
    return WorkerEnvelope.from_mapping(raw)


def encode_ndjson_line(envelope: WorkerEnvelope | Mapping[str, Any]) -> bytes:
    """Validate and encode one envelope, including its trailing newline."""

    parsed = envelope if isinstance(envelope, WorkerEnvelope) else WorkerEnvelope.from_mapping(envelope)
    # Revalidate dataclass instances too; callers can pass mutable payload dictionaries.
    parsed = WorkerEnvelope.from_mapping(parsed.to_dict())
    try:
        encoded = json.dumps(
            parsed.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ProtocolV1Error("invalid-json", str(error)) from error
    if len(encoded) > MAX_NDJSON_LINE_BYTES:
        raise ProtocolV1Error("line-too-long", f"record exceeds {MAX_NDJSON_LINE_BYTES} bytes")
    return encoded


def validate_project_v1(project: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a canonical Project v1 document and return a defensive copy."""

    _validate_project(project, "project")
    return deepcopy(dict(project))


def validate_recipe_v1(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a canonical Recipe v1 document and return a defensive copy."""

    _validate_recipe(recipe, "recipe")
    return deepcopy(dict(recipe))


class ProtocolCursor:
    """Enforce handshake-first, peer direction, session, and contiguous sequence."""

    def __init__(self, *, enforce_direction: bool = True) -> None:
        self.session_id: str | None = None
        self.next_sequence = 0
        self.peer_role: str | None = None
        self.enforce_direction = enforce_direction

    def accept(self, envelope: WorkerEnvelope | Mapping[str, Any]) -> WorkerEnvelope:
        parsed = envelope if isinstance(envelope, WorkerEnvelope) else WorkerEnvelope.from_mapping(envelope)
        parsed = WorkerEnvelope.from_mapping(parsed.to_dict())
        if self.session_id is None:
            if parsed.sequence != 0 or parsed.message_type != "handshake":
                raise ProtocolV1Error(
                    "handshake-required",
                    "the first session record must be handshake sequence 0",
                )
            self.session_id = parsed.session_id
            self.peer_role = parsed.payload["role"]
            self.next_sequence = 1
            return parsed
        if parsed.session_id != self.session_id:
            raise ProtocolV1Error("session-mismatch", "session identifier changed")
        if parsed.sequence != self.next_sequence:
            raise ProtocolV1Error(
                "sequence-mismatch",
                f"expected sequence {self.next_sequence}, received {parsed.sequence}",
                "sequence",
            )
        if parsed.message_type == "handshake":
            raise ProtocolV1Error("duplicate-handshake", "a session may contain only one handshake")
        if self.enforce_direction:
            allowed = _CONTROLLER_MESSAGES if self.peer_role == "controller" else _WORKER_MESSAGES
            if parsed.message_type not in allowed:
                raise ProtocolV1Error(
                    "direction-mismatch",
                    f"{self.peer_role} stream cannot send {parsed.message_type}",
                    "type",
                )
        if self.next_sequence == 2**64 - 1:
            raise ProtocolV1Error("sequence-exhausted", "protocol sequence number exhausted")
        self.next_sequence += 1
        return parsed


__all__ = [
    "MAX_NDJSON_LINE_BYTES",
    "PROTOCOL_VERSION",
    "ProtocolCursor",
    "ProtocolV1Error",
    "WorkerEnvelope",
    "decode_ndjson_line",
    "encode_ndjson_line",
    "validate_project_v1",
    "validate_recipe_v1",
]
