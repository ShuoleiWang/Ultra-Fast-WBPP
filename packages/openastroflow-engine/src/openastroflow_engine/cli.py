from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

from . import __version__
from .backends import StageKind
from .catalogs import (
    catalog_doctor,
    catalog_list,
    install_catalog,
    remove_catalog_plan,
    verify_catalog,
)
from . import platform as platform_services
from .hardware import detect_hardware
from .native_kernels import describe_native_kernels
from .performance_profile import select_execution_tuning
from .inventory import inventory_manifest_sha256, inventory_project
from .planning import build_plan, default_registry, solver_backend_science_ready
from .recipe import Recipe, RecipeError
from .calibration_preflight import inspect_calibration, load_calibration_request

# The E2E pipeline, the solver backends, the quality gate's analysis stack and
# the controller cost about a second to import; the interface's short
# commands (``inventory``, ``calibration-check``, ``doctor``, ``catalog``)
# would pay it on every launch, so those modules are imported by the
# commands that run them.


def _json(payload: object, *, compact: bool = False) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
        allow_nan=False,
    )


def _emit(payload: object, output: str | None, *, compact: bool, force: bool) -> None:
    rendered = _json(payload, compact=compact) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    destination = Path(output).expanduser()
    if destination.exists() and not force:
        raise FileExistsError(
            f"refusing to replace existing output: {destination}; pass --force to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, destination)
        else:
            # A hard-link publication is atomic and refuses a destination that
            # appeared after the initial existence check.
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def doctor_payload() -> dict[str, Any]:
    hardware = detect_hardware()
    tuning = select_execution_tuning(hardware)
    registry = default_registry()
    descriptors = registry.serializable()
    # A solver's probe says whether its process runs; ``scienceReady`` says
    # whether its solutions can pass the E2E catalog-correspondence gate.  The
    # GUI needs both to tell "install the solver" from "solver is diagnostic
    # only" without re-deriving the planner's rule.
    for backend in descriptors:
        if backend["stage"] == StageKind.SOLVER.value:
            solver = registry.get(backend["backendId"])
            backend["scienceReady"] = solver is not None and solver_backend_science_ready(solver)

    def ready(stage: StageKind) -> bool:
        return any(
            backend["stage"] == stage.value
            and backend["available"]
            and backend["executionReady"]
            for backend in descriptors
        )

    core_pixel_stages = (
        StageKind.QUALITY_GATE,
        StageKind.CALIBRATION,
        StageKind.REGISTRATION,
        StageKind.INTEGRATION,
    )
    core_ready = all(ready(stage) for stage in core_pixel_stages)
    solver_executable_ready = ready(StageKind.SOLVER)
    solver_ready = any(
        solver_backend_science_ready(backend)
        for backend in registry.for_stage(StageKind.SOLVER)
    )
    drizzle_ready = ready(StageKind.DRIZZLE)
    siril = next(
        (backend for backend in descriptors if backend["backendId"] == "siril-cli"),
        None,
    )
    # Binary/API probes cannot prove that installed astrometry indexes cover an
    # arbitrary future field.  That evidence is established and recorded only
    # by each solve attempt.
    catalog_coverage_verified = False
    # The installed SEP build decides whether two runs of the same project
    # agree bit for bit (Windows builds without the patched extractor do
    # not); the doctor states it so the receipts' warning has a home.
    from lightframeqc.source_extraction import cached_extraction_self_test

    try:
        source_extraction: dict[str, Any] = cached_extraction_self_test()
    except Exception as error:  # a broken extractor is reported, not hidden
        source_extraction = {"version": None, "deterministic": None, "error": str(error)[:200]}
    return {
        "schemaVersion": 1,
        "engineVersion": __version__,
        "hardware": hardware.serializable(),
        "tuning": tuning.serializable(),
        "nativeKernels": describe_native_kernels(),
        "sourceExtraction": source_extraction,
        "backends": descriptors,
        "status": {
            "inventoryReady": True,
            "workerProtocolReady": True,
            "qualityGateReady": ready(StageKind.QUALITY_GATE),
            "pixelExecutionReady": core_ready,
            "drizzleReady": drizzle_ready,
            "solverReady": solver_ready,
            "solverExecutableReady": solver_executable_ready,
            "catalogCoverageVerified": catalog_coverage_verified,
            "sirilReady": bool(siril and siril["executionReady"]),
            "endToEndExecutableReady": core_ready and solver_ready,
            "endToEndReady": core_ready
            and solver_ready
            and catalog_coverage_verified,
            "message": (
                "Raw FITS executors and a solver binary are ready; catalog/index "
                "coverage is proven fail-closed for each field only when it solves."
                if core_ready and solver_ready
                else "Core FITS execution is ready, but a required external solver "
                "did not pass its runtime probe."
                if core_ready
                else "One or more required in-process execution modules failed their probe."
            ),
        },
    }


def _load_recipe(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path).expanduser()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecipeError(f"cannot read recipe {source}: {error}") from error
    if not isinstance(value, dict):
        raise RecipeError("recipe file must contain one JSON object")
    return value


def _load_project_request(path: str) -> tuple[list[str], str | None, str, Recipe, dict[str, Any]]:
    from .runtime import RuntimeConfigurationError

    try:
        source = Path(path).expanduser().resolve(strict=True)
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeConfigurationError("PROJECT_REQUEST_INVALID", str(error)) from error
    if not isinstance(raw, Mapping):
        raise RuntimeConfigurationError("PROJECT_REQUEST_INVALID", "request JSON must be an object")
    allowed = {
        "schemaVersion",
        "sources",
        "outputDirectory",
        "projectName",
        "recipe",
        "solverHints",
        "execution",
        "reviewSelections",
        "selection",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown or raw.get("schemaVersion") != 1:
        raise RuntimeConfigurationError(
            "PROJECT_REQUEST_INVALID",
            "request JSON has an unsupported schema or fields: " + ", ".join(unknown),
        )
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeConfigurationError("PROJECT_REQUEST_INVALID", "sources must be a non-empty array")
    paths: list[str] = []
    expected: dict[str, str] = {}
    allowed_roles = {"LIGHT", "FLAT", "DARK", "BIAS", "MASTER_FLAT", "MASTER_DARK", "MASTER_BIAS"}
    for index, item in enumerate(sources):
        if not isinstance(item, Mapping) or set(item) - {"sourceId", "hostPath", "expectedRole", "recursive"}:
            raise RuntimeConfigurationError("PROJECT_REQUEST_INVALID", f"sources[{index}] is invalid")
        host = item.get("hostPath")
        role = item.get("expectedRole")
        recursive = item.get("recursive", True)
        if not isinstance(host, str) or not host.strip() or role not in allowed_roles or not isinstance(recursive, bool):
            raise RuntimeConfigurationError("PROJECT_REQUEST_INVALID", f"sources[{index}] has invalid fields")
        resolved = Path(host).expanduser().resolve(strict=True)
        if resolved.is_dir() and not recursive:
            raise RuntimeConfigurationError(
                "PROJECT_REQUEST_INVALID",
                f"sources[{index}] is a directory but recursive is false",
            )
        paths.append(str(resolved))
        expected[str(resolved)] = str(role)
    output = raw.get("outputDirectory")
    if not isinstance(output, str) or not output.strip():
        raise RuntimeConfigurationError("PROJECT_REQUEST_INVALID", "outputDirectory is required")
    project_name = raw.get("projectName")
    if project_name is not None and not isinstance(project_name, str):
        raise RuntimeConfigurationError("PROJECT_REQUEST_INVALID", "projectName must be a string")
    recipe = Recipe.from_dict(raw.get("recipe"))
    options = {
        "expectedRoles": expected,
        "solverHints": raw.get("solverHints", {}),
        "execution": raw.get("execution", {}),
        "reviewSelections": raw.get("reviewSelections", []),
    }
    if not isinstance(options["solverHints"], Mapping) or not isinstance(options["execution"], Mapping):
        raise RuntimeConfigurationError("PROJECT_REQUEST_INVALID", "solverHints/execution must be objects")
    selections = options["reviewSelections"]
    if not isinstance(selections, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"sourceSha256", "gatePolicyDigest"}
        for item in selections
    ):
        raise RuntimeConfigurationError(
            "PROJECT_REQUEST_INVALID",
            "reviewSelections must be an array of exact source/policy digest objects",
        )
    # The blink review's decisions (schema selection-v1); they replace the
    # legacy REVIEW selections rather than adding to them.
    options["selection"] = None
    if raw.get("selection") is not None:
        from .e2e import parse_explicit_selection

        if selections:
            raise RuntimeConfigurationError(
                "SELECTION_POLICY_CONFLICT",
                "a request carries either selection or reviewSelections, not both",
            )
        options["selection"] = parse_explicit_selection(raw["selection"])
    return paths, project_name, output, recipe, options


def _load_selection_file(path: str | None) -> Any:
    """The ``--selection`` file as an ``ExplicitSelection`` (or ``None``)."""

    if path is None:
        return None
    from .e2e import E2EError, parse_explicit_selection

    source = Path(path).expanduser()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise E2EError("SELECTION_INVALID", f"cannot read selection {source}: {error}") from error
    return parse_explicit_selection(raw)


def _load_quality_request(path: str) -> list[str]:
    from .quality_preflight import QualityPreflightError

    try:
        source = Path(path).expanduser().resolve(strict=True)
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualityPreflightError("QUALITY_REQUEST_INVALID", str(error)) from error
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schemaVersion", "lightPaths"}
        or raw.get("schemaVersion") != 1
    ):
        raise QualityPreflightError(
            "QUALITY_REQUEST_INVALID",
            "quality request must contain only schemaVersion=1 and lightPaths",
        )
    paths = raw.get("lightPaths")
    if (
        not isinstance(paths, list)
        or not paths
        or len(paths) > 10_000
        or not all(isinstance(item, str) and item.strip() for item in paths)
    ):
        raise QualityPreflightError(
            "QUALITY_REQUEST_INVALID",
            "lightPaths must contain between 1 and 10000 non-empty paths",
        )
    return paths


def _load_blink_request(path: str) -> Any:
    from .blink_session import BlinkRequest, BlinkSessionError

    try:
        source = Path(path).expanduser().resolve(strict=True)
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlinkSessionError("BLINK_REQUEST_INVALID", str(error)) from error
    return BlinkRequest.from_mapping(raw)


def _verify_expected_source_roles(inventory: Any, expected: Mapping[str, str]) -> None:
    from .runtime import RuntimeConfigurationError

    for asset in inventory.assets:
        asset_path = Path(asset.path)
        matches = [
            (Path(root), role)
            for root, role in expected.items()
            if asset_path == Path(root) or Path(root) in asset_path.parents
        ]
        if not matches:
            raise RuntimeConfigurationError("PROJECT_SOURCE_UNBOUND", "inventoried asset is not bound to a request source")
        root, role = max(matches, key=lambda item: len(item[0].parts))
        del root
        if asset.role.value != role:
            raise RuntimeConfigurationError(
                "PROJECT_SOURCE_ROLE_MISMATCH",
                f"{asset.path} expected {role}, header/path evidence found {asset.role.value}",
            )


def _recipe_from_args(args: argparse.Namespace) -> Recipe:
    # Validate the file before merging CLI overrides so malformed nested values
    # become stable RecipeError diagnostics instead of raw dict-conversion errors.
    value = Recipe.from_dict(_load_recipe(args.recipe)).serializable()
    solver = dict(value.get("solver", {}))
    drizzle = dict(value.get("drizzle", {}))
    if getattr(args, "solver_backend", None) is not None:
        solver["backend"] = args.solver_backend
    if getattr(args, "solver_policy", None) is not None:
        solver["policy"] = args.solver_policy
    if getattr(args, "drizzle", False):
        drizzle["enabled"] = True
    if getattr(args, "drizzle_backend", None) is not None:
        drizzle["backend"] = args.drizzle_backend
    if getattr(args, "drizzle_scale", None) is not None:
        drizzle["scale"] = args.drizzle_scale
    if getattr(args, "drop_shrink", None) is not None:
        drizzle["dropShrink"] = args.drop_shrink
    if getattr(args, "drizzle_kernel", None) is not None:
        drizzle["kernel"] = args.drizzle_kernel
    if getattr(args, "cfa_drizzle", False):
        drizzle["cfaDrizzle"] = True
    mode = getattr(args, "mode", None)
    if mode is not None:
        drizzle["enabled"] = mode == "drizzle"
    solver_chain = getattr(args, "solver_chain", None)
    if solver_chain:
        first = next(
            (item.strip() for item in solver_chain.split(",") if item.strip()),
            None,
        )
        if first is not None:
            solver["backend"] = first
    if solver:
        value["solver"] = solver
    if drizzle:
        value["drizzle"] = drizzle
    return Recipe.from_dict(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ultra-fast-wbpp",
        description="Inspect and process Ultra-Fast WBPP projects without mutating source frames.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect hardware and backend seams")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    inventory = subparsers.add_parser(
        "inventory", help="recursively inventory NINA FITS/XISF files"
    )
    inventory.add_argument("inputs", nargs="+", help="frame files or directories")
    inventory.add_argument("--name", help="project display name")
    inventory.add_argument("--output", help="write JSON to a new file")
    inventory.add_argument("--compact", action="store_true")
    inventory.add_argument("--force", action="store_true", help="replace --output")

    quality = subparsers.add_parser(
        "quality-check",
        help="run the real read-only Light Quality Gate for native GUI review",
    )
    quality.add_argument(
        "--request-json", required=True, help="private schema-1 Light path request"
    )
    quality.add_argument("--workers", type=int)
    quality.add_argument("--compact", action="store_true")

    blink = subparsers.add_parser(
        "blink-measure",
        help="measure Lights, flag the obvious junk, pick a reference per channel and render normalized previews for blinking",
    )
    blink.add_argument(
        "--request-json", required=True, help="private schema-1 blink request (Light paths, new session directory)"
    )
    blink.add_argument("--compact", action="store_true")
    blink.add_argument(
        "--progress-json",
        action="store_true",
        help="emit one JSON progress record per line on stderr",
    )

    calibration = subparsers.add_parser(
        "calibration-check", help="read-only calibration metadata and library checks"
    )
    calibration.add_argument("--request-json", required=True)
    calibration.add_argument("--compact", action="store_true")

    controller_plan = subparsers.add_parser(
        "controller-plan",
        help="inventory inputs and emit one canonical app-core plan envelope for the GUI worker",
    )
    controller_plan.add_argument("inputs", nargs="+", help="frame files or recursive directories")
    controller_plan.add_argument("--name", help="project display name")
    controller_plan.add_argument("--mode", choices=("ordinary", "drizzle"), default="ordinary")
    controller_plan.add_argument("--session-id", default="openastroflow-controller")
    controller_plan.add_argument("--sequence", type=int, default=1)
    controller_plan.add_argument("--request-id")
    controller_plan.add_argument("--plan-id")
    controller_plan.add_argument("--project-id")
    controller_plan.add_argument("--hardware-profile")
    controller_plan.add_argument("--solver-catalog", default="astrometry-net-offline")
    controller_plan.add_argument("--minimum-matches", type=int, default=12)
    controller_plan.add_argument("--maximum-rms-arcsec", type=float, default=2.0)
    controller_plan.add_argument("--drizzle-scale", type=int, default=2)
    controller_plan.add_argument("--drop-shrink", type=float, default=0.9)
    controller_plan.add_argument("--output", help="write envelope JSON to a new file")
    controller_plan.add_argument("--compact", action="store_true")
    controller_plan.add_argument("--force", action="store_true", help="replace --output")

    plan = subparsers.add_parser("plan", help="inventory inputs and validate a recipe")
    plan.add_argument("inputs", nargs="+", help="frame files or directories")
    plan.add_argument("--name", help="project display name")
    plan.add_argument("--recipe", help="recipe JSON file")
    plan.add_argument("--solver-backend")
    plan.add_argument("--solver-policy", choices=("REQUIRED", "DISABLED"))
    plan.add_argument("--drizzle", action="store_true")
    plan.add_argument("--drizzle-backend")
    plan.add_argument("--drizzle-scale", type=int)
    plan.add_argument("--drop-shrink", type=float)
    plan.add_argument("--drizzle-kernel", choices=("square", "circular", "gaussian", "point"))
    plan.add_argument("--cfa-drizzle", action="store_true")
    plan.add_argument("--output", help="write JSON to a new file")
    plan.add_argument("--compact", action="store_true")
    plan.add_argument("--force", action="store_true", help="replace --output")

    run = subparsers.add_parser(
        "run",
        help="run raw NINA FITS folders through QC, calibration, registration, integration, and WCS solving",
    )
    run.add_argument("inputs", nargs="+", help="frame files or recursively scanned directories")
    run.add_argument("--output", required=True, help="new output directory; replacement is never allowed")
    run.add_argument("--name", help="project display name")
    run.add_argument("--recipe", help="Python recipe JSON file")
    run.add_argument(
        "--mode",
        choices=("ordinary", "drizzle"),
        help="override the recipe integration mode (default recipe: ordinary)",
    )
    run.add_argument(
        "--solver-chain",
        help="ordered comma-separated solver ids (default recipe: automatic ready chain)",
    )
    run.add_argument("--workers", type=int, help="QC worker count; defaults to the hardware profile")
    run.add_argument(
        "--selection",
        help="selection-v1 JSON (the blink review's KEEP/DROP per Light); replaces the automatic screening",
    )
    run.add_argument("--ra", type=float, dest="ra_hint_degrees", help="optional center RA in degrees")
    run.add_argument("--dec", type=float, dest="dec_hint_degrees", help="optional center Dec in degrees")
    run.add_argument("--fov", type=float, dest="field_of_view_degrees", help="optional field width in degrees")
    run.add_argument("--search-radius", type=float, dest="search_radius_degrees")
    run.add_argument(
        "--progress-json",
        action="store_true",
        help="emit one JSON progress record per line on stderr",
    )
    # Recipe-compatible overrides that remain executable.  Planning can model
    # disabled astrometry and CFA Drizzle, but product runs deliberately do not
    # advertise flags that the fail-closed E2E executor must reject.
    run.add_argument("--solver-backend")
    run.add_argument("--drizzle", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--drizzle-backend")
    run.add_argument("--drizzle-scale", type=int)
    run.add_argument("--drop-shrink", type=float)
    run.add_argument("--drizzle-kernel", choices=("square", "circular", "gaussian", "point"))

    run_project = subparsers.add_parser(
        "run-project",
        help="run target/filter panels through solved mosaics and optional RGB/LRGB publication",
    )
    run_project.add_argument("inputs", nargs="*", help="frame files or role-specific recursive directories")
    run_project.add_argument("--request-json", help="strict project request JSON; mutually exclusive with positional inputs")
    run_project.add_argument("--output", help="new output directory (required without --request-json)")
    run_project.add_argument("--name", help="project display name")
    run_project.add_argument("--recipe", help="Python recipe JSON file")
    run_project.add_argument("--mode", choices=("ordinary", "drizzle"))
    run_project.add_argument("--solver-chain")
    run_project.add_argument("--workers", type=int)
    run_project.add_argument("--selection", help="selection-v1 JSON; positional mode only")
    run_project.add_argument("--ra", type=float, dest="ra_hint_degrees")
    run_project.add_argument("--dec", type=float, dest="dec_hint_degrees")
    run_project.add_argument("--fov", type=float, dest="field_of_view_degrees")
    run_project.add_argument("--search-radius", type=float, dest="search_radius_degrees")
    run_project.add_argument("--progress-json", action="store_true")
    run_project.add_argument("--solver-backend")
    run_project.add_argument("--drizzle", action="store_true", help=argparse.SUPPRESS)
    run_project.add_argument("--drizzle-backend")
    run_project.add_argument("--drizzle-scale", type=int)
    run_project.add_argument("--drop-shrink", type=float)
    run_project.add_argument("--drizzle-kernel", choices=("square", "circular", "gaussian", "point"))

    catalog = subparsers.add_parser(
        "catalog",
        help="inspect and explicitly install offline plate-solver indexes",
    )
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)

    def catalog_paths(command: argparse.ArgumentParser) -> None:
        command.add_argument("--catalog-dir", help="managed astrometry.net index directory")
        command.add_argument("--manifest-dir", help="checked catalog manifest directory")

    catalog_list_parser = catalog_commands.add_parser(
        "list", help="show checked catalogs, scale/FOV coverage, terms, and storage"
    )
    catalog_paths(catalog_list_parser)
    catalog_list_parser.add_argument("--json", action="store_true")

    catalog_doctor_parser = catalog_commands.add_parser(
        "doctor", help="hash installed indexes and inspect generated solver configuration"
    )
    catalog_paths(catalog_doctor_parser)
    catalog_doctor_parser.add_argument("--json", action="store_true")

    catalog_install_parser = catalog_commands.add_parser(
        "install", help="download from the provider after versioned, explicit terms acceptance"
    )
    catalog_install_parser.add_argument("catalog_id")
    catalog_paths(catalog_install_parser)
    catalog_install_parser.add_argument(
        "--accept-provider-terms",
        metavar="ACCEPTANCE_ID",
        dest="accepted_terms_id",
        help="exact versioned acceptance ID printed by 'catalog list' (required for download)",
    )
    catalog_install_parser.add_argument(
        "--artifact",
        action="append",
        dest="artifact_ids",
        help="install one checked artifact; repeat for more than one",
    )
    catalog_install_parser.add_argument(
        "--field-of-view",
        type=float,
        dest="field_of_view_degrees",
        help="install scales whose quads match this image width in degrees",
    )
    catalog_install_parser.add_argument("--timeout", type=float, default=60.0)
    catalog_install_parser.add_argument("--progress-json", action="store_true")

    catalog_verify_parser = catalog_commands.add_parser(
        "verify", help="hash-check installed indexes without changing them"
    )
    catalog_verify_parser.add_argument("catalog_id")
    catalog_paths(catalog_verify_parser)
    catalog_verify_parser.add_argument("--artifact", action="append", dest="artifact_ids")
    catalog_verify_parser.add_argument("--field-of-view", type=float, dest="field_of_view_degrees")
    catalog_verify_parser.add_argument(
        "--configure",
        action="store_true",
        help="after successful verification, create astrometry.cfg and an immutable installed-set receipt",
    )

    catalog_remove_parser = catalog_commands.add_parser(
        "remove-plan", help="print exact removable paths and bytes; never delete anything"
    )
    catalog_remove_parser.add_argument("catalog_id")
    catalog_paths(catalog_remove_parser)

    subparsers.add_parser("worker", help="run the stdin/stdout NDJSON worker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Receipts, progress and the worker protocol are UTF-8 on every platform;
    # a Windows console code page must not decide how they are encoded.
    platform_services.reconfigure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            payload = doctor_payload()
            if args.json:
                sys.stdout.write(_json(payload) + "\n")
            else:
                hardware = payload["hardware"]
                status = payload["status"]
                sys.stdout.write(
                    f"Ultra-Fast WBPP Engine {payload['engineVersion']}\n"
                    f"Hardware: {hardware['cpuBrand']} ({hardware['architecture']})\n"
                    f"Profile: {hardware['optimizationProfile']}\n"
                    f"Devices: {', '.join(hardware['devices'])}\n"
                    f"Pixel execution ready: {'yes' if status['pixelExecutionReady'] else 'no'}\n"
                    f"Offline solver ready: {'yes' if status['solverReady'] else 'no'}\n"
                    f"E2E executors ready: {'yes' if status['endToEndExecutableReady'] else 'no'}\n"
                    f"Catalog coverage preverified: {'yes' if status['catalogCoverageVerified'] else 'no'}\n"
                    f"{status['message']}\n"
                )
            return 0
        if args.command == "inventory":
            inventory = inventory_project(args.inputs, name=args.name)
            payload = inventory.serializable()
            payload["inputManifestSha256"] = inventory_manifest_sha256(inventory)
            _emit(
                payload,
                args.output,
                compact=args.compact,
                force=args.force,
            )
            return 0
        if args.command == "calibration-check":
            paths, recipe = load_calibration_request(args.request_json)
            payload = inspect_calibration(paths, recipe)
            sys.stdout.write(_json(payload, compact=args.compact) + "\n")
            return 0
        if args.command == "quality-check":
            from .quality_preflight import inspect_light_quality

            payload = inspect_light_quality(
                _load_quality_request(args.request_json), workers=args.workers
            )
            sys.stdout.write(_json(payload, compact=args.compact) + "\n")
            return 0
        if args.command == "blink-measure":
            from .blink_session import run_blink_session

            def blink_progress(stage: str, message: str) -> None:
                if args.progress_json:
                    sys.stderr.write(
                        _json({"type": "progress", "stage": stage, "message": message}, compact=True) + "\n"
                    )
                    sys.stderr.flush()

            with platform_services.current().keep_awake():
                manifest = run_blink_session(
                    _load_blink_request(args.request_json),
                    progress=blink_progress if args.progress_json else None,
                )
            sys.stdout.write(_json(manifest, compact=args.compact) + "\n")
            return 0
        if args.command == "controller-plan":
            from .controller import controller_plan_envelope

            inventory = inventory_project(args.inputs, name=args.name)
            envelope = controller_plan_envelope(
                inventory,
                mode=args.mode,
                session_id=args.session_id,
                sequence=args.sequence,
                request_id=args.request_id,
                plan_id=args.plan_id,
                project_id=args.project_id,
                requested_hardware_profile=args.hardware_profile,
                solver_catalog=args.solver_catalog,
                minimum_matches=args.minimum_matches,
                maximum_rms_arcsec=args.maximum_rms_arcsec,
                drizzle_scale=args.drizzle_scale,
                drop_shrink=args.drop_shrink,
            )
            _emit(
                envelope.to_dict(),
                args.output,
                compact=args.compact,
                force=args.force,
            )
            return 0
        if args.command == "plan":
            inventory = inventory_project(args.inputs, name=args.name)
            recipe = _recipe_from_args(args)
            plan = build_plan(inventory, recipe)
            _emit(
                plan.serializable(),
                args.output,
                compact=args.compact,
                force=args.force,
            )
            return 0
        if args.command == "run":
            from .e2e import ProgressEvent, run_e2e
            from .project_e2e import project_requires_orchestration, run_project_e2e
            from .runtime import prepare_execution, prepare_project_execution

            inventory = inventory_project(args.inputs, name=args.name)
            recipe = _recipe_from_args(args)

            def progress(event: ProgressEvent) -> None:
                if args.progress_json:
                    sys.stderr.write(
                        _json(
                            {"type": "progress", "event": event.serializable()},
                            compact=True,
                        )
                        + "\n"
                    )
                    sys.stderr.flush()
            common = {
                "solver_chain": args.solver_chain,
                "workers": args.workers,
                "ra_hint_degrees": args.ra_hint_degrees,
                "dec_hint_degrees": args.dec_hint_degrees,
                "field_of_view_degrees": args.field_of_view_degrees,
                "search_radius_degrees": args.search_radius_degrees,
                "explicit_selection": _load_selection_file(args.selection),
            }
            if project_requires_orchestration(inventory):
                _, request, solvers = prepare_project_execution(
                    inventory, recipe, args.output, **common
                )
                with platform_services.current().keep_awake():
                    result = run_project_e2e(
                        request,
                        solver_backends=solvers,
                        progress=progress if args.progress_json else None,
                    )
            else:
                _, request, solvers = prepare_execution(
                    inventory, recipe, args.output, **common
                )
                with platform_services.current().keep_awake():
                    result = run_e2e(
                        request,
                        solver_backends=solvers,
                        progress=progress if args.progress_json else None,
                    )
            sys.stdout.write(_json(result.serializable()) + "\n")
            return 0 if result.success else 3
        if args.command == "run-project":
            from .e2e import ProgressEvent
            from .project_e2e import run_project_e2e
            from .runtime import RuntimeConfigurationError, prepare_project_execution

            review_selections: Sequence[Mapping[str, str]] = ()
            if args.request_json:
                if args.inputs or args.output or args.recipe or args.selection:
                    raise RuntimeConfigurationError(
                        "PROJECT_REQUEST_AMBIGUOUS",
                        "--request-json cannot be mixed with positional inputs, --output, --recipe or --selection",
                    )
                inputs, name, output, recipe, options = _load_project_request(args.request_json)
                inventory = inventory_project(inputs, name=name)
                _verify_expected_source_roles(inventory, options["expectedRoles"])
                hints = options["solverHints"]
                execution = options["execution"]
                review_selections = options["reviewSelections"]
                workers = execution.get("workers")
                kwargs = {
                    "workers": workers,
                    "ra_hint_degrees": hints.get("raDegrees"),
                    "dec_hint_degrees": hints.get("decDegrees"),
                    "field_of_view_degrees": hints.get("fieldOfViewDegrees"),
                    "search_radius_degrees": hints.get("searchRadiusDegrees"),
                    "solver_chain": recipe.solver.backend,
                    "explicit_selection": options["selection"],
                }
            else:
                if not args.inputs or not args.output:
                    raise RuntimeConfigurationError(
                        "PROJECT_REQUEST_INVALID",
                        "run-project requires positional inputs and --output, or --request-json",
                    )
                inventory = inventory_project(args.inputs, name=args.name)
                recipe = _recipe_from_args(args)
                output = args.output
                kwargs = {
                    "workers": args.workers,
                    "ra_hint_degrees": args.ra_hint_degrees,
                    "dec_hint_degrees": args.dec_hint_degrees,
                    "field_of_view_degrees": args.field_of_view_degrees,
                    "search_radius_degrees": args.search_radius_degrees,
                    "solver_chain": args.solver_chain,
                    "explicit_selection": _load_selection_file(args.selection),
                }
            _, request, solvers = prepare_project_execution(
                inventory, recipe, output, **kwargs
            )
            if review_selections:
                # Bound per target run inside run_project_e2e, so projects
                # with several targets or filters admit reviewed Lights too.
                request = replace(
                    request,
                    review_selections=tuple(dict(selection) for selection in review_selections),
                )

            def project_progress(event: ProgressEvent) -> None:
                if args.progress_json:
                    sys.stderr.write(
                        _json({"type": "progress", "event": event.serializable()}, compact=True)
                        + "\n"
                    )
                    sys.stderr.flush()

            with platform_services.current().keep_awake():
                result = run_project_e2e(
                    request,
                    solver_backends=solvers,
                    progress=project_progress if args.progress_json else None,
                )
            sys.stdout.write(_json(result.serializable()) + "\n")
            return 0 if result.success else 3
        if args.command == "catalog":
            if args.catalog_command == "list":
                payload = catalog_list(
                    catalog_root=args.catalog_dir,
                    manifest_dir=args.manifest_dir,
                )
                if args.json:
                    sys.stdout.write(_json(payload) + "\n")
                else:
                    sys.stdout.write(f"Catalog directory: {payload['catalogRoot']}\n")
                    for item in payload["catalogs"]:
                        terms = item["providerTerms"]
                        mib = item["totalSizeBytes"] / (1024 * 1024)
                        sys.stdout.write(
                            f"{item['catalogId']}: {len(item['artifacts'])} artifact(s), "
                            f"{mib:.1f} MiB, installed-by-size "
                            f"{item['installedArtifactsBySize']}/{item['artifactCount']}\n"
                            f"  terms: {terms['licenseStatus']} — {terms['url']}\n"
                            f"  acceptance ID: {terms['acceptanceId']}\n"
                        )
                return 0
            if args.catalog_command == "doctor":
                payload = catalog_doctor(
                    catalog_root=args.catalog_dir,
                    manifest_dir=args.manifest_dir,
                )
                if args.json:
                    sys.stdout.write(_json(payload) + "\n")
                else:
                    sys.stdout.write(
                        f"Catalog directory: {payload['catalogRoot']}\n"
                        f"Ready: {'yes' if payload['ok'] else 'no'}\n"
                        f"{payload['message']}\n"
                    )
                return 0 if payload["ok"] else 3
            if args.catalog_command == "install":

                def catalog_progress(event: Mapping[str, Any]) -> None:
                    if args.progress_json:
                        sys.stderr.write(_json(event, compact=True) + "\n")
                        sys.stderr.flush()

                payload = install_catalog(
                    args.catalog_id,
                    accepted_terms_id=args.accepted_terms_id,
                    catalog_root=args.catalog_dir,
                    manifest_dir=args.manifest_dir,
                    artifact_ids=args.artifact_ids,
                    field_of_view_degrees=args.field_of_view_degrees,
                    timeout_seconds=args.timeout,
                    progress=catalog_progress if args.progress_json else None,
                )
                sys.stdout.write(_json(payload) + "\n")
                return 0
            if args.catalog_command == "verify":
                payload = verify_catalog(
                    args.catalog_id,
                    catalog_root=args.catalog_dir,
                    manifest_dir=args.manifest_dir,
                    artifact_ids=args.artifact_ids,
                    field_of_view_degrees=args.field_of_view_degrees,
                    write_configuration=args.configure,
                )
                sys.stdout.write(_json(payload) + "\n")
                return 0 if payload["ok"] else 3
            if args.catalog_command == "remove-plan":
                payload = remove_catalog_plan(
                    args.catalog_id,
                    catalog_root=args.catalog_dir,
                    manifest_dir=args.manifest_dir,
                )
                sys.stdout.write(_json(payload) + "\n")
                return 0
        if args.command == "worker":
            from .worker import run_worker

            return run_worker(sys.stdin, sys.stdout)
    except Exception as error:
        # Every engine boundary error (inventory, recipe, runtime, E2E,
        # project, calibration, catalog and preflight errors) carries a stable
        # ``code``; matching on it keeps their modules out of this module's
        # import path.  Anything else is a defect and keeps its traceback.
        if isinstance(error, FileExistsError):
            code = "OUTPUT_EXISTS"
        elif isinstance(getattr(error, "code", None), str) and getattr(error, "code"):
            code = error.code
        elif isinstance(error, OSError):
            code = "IO_ERROR"
        else:
            raise
        sys.stderr.write(
            _json({"ok": False, "error": {"code": code, "message": str(error)}})
            + "\n"
        )
        return 2
    parser.error("unknown command")
    return 2


__all__ = ["doctor_payload", "main"]
