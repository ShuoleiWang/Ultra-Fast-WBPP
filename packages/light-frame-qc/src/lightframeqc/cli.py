from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence
import webbrowser

from . import __version__
from .config import DEFAULT_CONFIG, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="light-frame-qc",
        description=(
            "Conservative cloud and hard-obstruction screening for raw astronomical "
            "light frames. Inputs are never modified, moved, or deleted."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    analyze = subcommands.add_parser("analyze", help="scan, measure, classify, and report")
    analyze.add_argument("inputs", nargs="+", help="FITS/XISF files or folders")
    analyze.add_argument(
        "-o", "--output", required=True, help="new or existing report directory"
    )
    analyze.add_argument("--config", help="JSON settings overriding the built-in defaults")
    analyze.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel frame readers (default 1; memory use scales with this value)",
    )
    analyze.add_argument(
        "--no-thumbnails", action="store_true", help="skip PNG review thumbnails"
    )
    analyze.add_argument(
        "--pixinsight-measurements",
        help="optional existing SubframeSelector v3 JSON; imported as auxiliary metrics only",
    )
    analyze.add_argument(
        "--open-report", action="store_true", help="open report.html after a successful run"
    )

    prepare = subcommands.add_parser(
        "prepare-wbpp",
        help="screen downloads and build one WBPP-ready LIGHT+FLAT tree per target",
    )
    prepare.add_argument("inputs", nargs="+", help="downloaded FITS/XISF files or folders")
    prepare.add_argument(
        "-o", "--output", required=True, help="WBPP-ready destination tree"
    )
    prepare.add_argument(
        "--report-output",
        help="QC/plan report directory (default: <output>-qc)",
    )
    prepare.add_argument("--config", help="JSON quality-control settings")
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument("--no-thumbnails", action="store_true")
    prepare.add_argument(
        "--flat-library",
        action="append",
        default=[],
        help="folder/file to scan for authoritative master flats; repeatable",
    )
    prepare.add_argument(
        "--master-flat",
        action="append",
        default=[],
        help="explicit master-flat file; repeatable",
    )
    prepare.add_argument(
        "--adjudication",
        help="optional hash-bound APPROVE/REJECT adjudication JSON",
    )
    prepare.add_argument(
        "--apply",
        action="store_true",
        help="publish the planned WBPP tree; omission is read-only plan mode",
    )

    apply_plan = subcommands.add_parser(
        "apply-wbpp-plan",
        help="apply an existing prepare-plan.json without remeasuring frames",
    )
    apply_plan.add_argument("plan", help="strict prepare-plan.json path")
    apply_plan.add_argument(
        "-o",
        "--output",
        help="optional destination assertion; must equal the destination bound in the plan",
    )

    doctor = subcommands.add_parser("doctor", help="check imports and supported readers")
    doctor.add_argument("paths", nargs="*", help="optional files/folders to inventory")
    subcommands.add_parser(
        "show-config", help="print the built-in JSON configuration"
    )
    subcommands.add_parser(
        "show-gate-policy", help="print the independent Quality Gate policy"
    )
    return parser


def _merge_pixinsight(measurements, path: str | None) -> list[str]:
    if path is None:
        return []
    from .metadata import airmass_from_altitude
    from .pixinsight import load_subframe_selector_v3_json

    imported = load_subframe_selector_v3_json(path)
    by_path = {row.path: row for row in imported.measurements}
    unmatched: list[str] = []
    for measurement in measurements:
        row = by_path.get(measurement.metadata.path)
        if row is None:
            unmatched.append(measurement.metadata.path)
            continue
        measurement.pixinsight = row.as_features()
        if measurement.metadata.altitude_degrees is None and row.altitude > 0:
            measurement.metadata.altitude_degrees = row.altitude
            if measurement.metadata.airmass is None:
                measurement.metadata.airmass = airmass_from_altitude(row.altitude)
        if measurement.metadata.azimuth_degrees is None and row.azimuth >= 0:
            measurement.metadata.azimuth_degrees = row.azimuth
    imported_paths = set(imported.request_paths)
    measured_paths = {measurement.metadata.path for measurement in measurements}
    extras = sorted(imported_paths - measured_paths)
    warnings: list[str] = []
    if unmatched:
        warnings.append(
            f"PIXINSIGHT_ROWS_MISSING_FOR_INPUTS: {len(unmatched)} input frame(s)"
        )
    if extras:
        warnings.append(
            f"PIXINSIGHT_ROWS_NOT_REQUESTED: {len(extras)} imported row(s)"
        )
    return warnings


def _run_analyze(arguments: argparse.Namespace) -> int:
    from .analysis import analyze_measurements
    from .measure import measure_paths
    from .models import RunResult
    from .quality_gate import evaluate_quality_gate
    from .readers import discover_paths
    from .report import write_reports

    config = load_config(arguments.config)
    if arguments.no_thumbnails:
        config = replace(config, make_thumbnails=False)
    if arguments.workers < 1:
        raise ValueError("--workers must be positive")

    inputs = [str(Path(value).expanduser()) for value in arguments.inputs]
    output = Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = discover_paths(inputs)
    print(f"Discovered {len(paths)} supported frame(s).", flush=True)
    print("Measuring read-only previews...", flush=True)
    measurements = measure_paths(paths, output, config, workers=arguments.workers)
    failed = sum(item.status != "MEASURED" for item in measurements)
    if failed:
        print(f"Measurement completed with {failed} per-frame failure(s).", flush=True)
    warnings = _merge_pixinsight(measurements, arguments.pixinsight_measurements)

    print("Building per-filter references and evidence scores...", flush=True)
    groups, frames = analyze_measurements(measurements, config)
    evaluate_quality_gate(frames, measurements, config)
    run = RunResult(
        schema_version=3,
        algorithm_version=__version__,
        generated_at=datetime.now(timezone.utc),
        inputs=[str(Path(value).expanduser().resolve()) for value in inputs],
        output_directory=str(output),
        config=config.serializable(),
        groups=groups,
        frames=frames,
        warnings=warnings,
    )
    written = write_reports(run)
    counts: dict[str, int] = {}
    for frame in frames:
        counts[frame.decision.value] = counts.get(frame.decision.value, 0) + 1
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"HTML report: {written['html']}", flush=True)
    print(f"CSV decisions: {written['csv']}", flush=True)
    print("No input frame was modified, moved, or deleted.", flush=True)
    if arguments.open_report:
        webbrowser.open(written["html"].resolve().as_uri())
    return 0


def _atomic_json(path: Path, value: object) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _nearby_master_flat_candidates(inputs: list[str]) -> list[Path]:
    """Find top-level masterFlat files no more than two ancestors away."""

    from .readers import is_supported_frame_path

    found: dict[str, Path] = {}
    for raw_value in inputs:
        requested = Path(raw_value).expanduser()
        try:
            resolved = requested.resolve(strict=True)
        except OSError:
            continue
        directory = resolved.parent if resolved.is_file() else resolved
        for _depth in range(3):
            try:
                children = list(directory.iterdir())
            except OSError:
                children = []
            for child in children:
                compact = re.sub(r"[^a-z0-9]+", "", child.stem.casefold())
                if (
                    compact.startswith("masterflat")
                    and child.is_file()
                    and not child.is_symlink()
                    and is_supported_frame_path(child)
                ):
                    canonical = child.resolve(strict=True)
                    found[os.path.normcase(str(canonical))] = canonical
            parent = directory.parent
            if parent == directory:
                break
            directory = parent
    return sorted(found.values(), key=lambda path: os.path.normcase(str(path)))


def _collect_master_flat_paths(arguments: argparse.Namespace) -> list[Path]:
    from .models import FrameRole
    from .readers import discover_paths, probe_frame_metadata

    candidates: dict[str, Path] = {
        os.path.normcase(str(path)): path
        for path in _nearby_master_flat_candidates(arguments.inputs)
    }
    for raw_library in arguments.flat_library:
        requested = Path(raw_library).expanduser()
        discovered = discover_paths([requested])
        explicit_file = requested.is_file()
        matched = 0
        for path in discovered:
            metadata = probe_frame_metadata(path)
            if metadata.role_conflicts:
                raise ValueError(
                    f"METADATA_ROLE_CONFLICT: {path}: "
                    + "; ".join(metadata.role_conflicts)
                )
            if metadata.role is FrameRole.MASTER_FLAT:
                candidates[os.path.normcase(str(path))] = path
                matched += 1
            elif explicit_file:
                raise ValueError(f"NOT_MASTER_FLAT: {path}: {metadata.role.value}")
        if matched == 0 and not explicit_file:
            raise ValueError(f"NO_MASTER_FLATS: {requested}")
    explicit: list[tuple[Path, tuple[object, ...]]] = []

    def override_key(metadata: object) -> tuple[object, ...]:
        camera = str(getattr(metadata, "camera", "UNKNOWN")).strip().casefold()
        readout = str(getattr(metadata, "readout_mode", "UNKNOWN")).strip().casefold()
        return (
            getattr(metadata, "width"),
            getattr(metadata, "height"),
            getattr(metadata, "channels"),
            getattr(metadata, "binning_x"),
            getattr(metadata, "binning_y"),
            str(getattr(metadata, "filter_name")).casefold(),
            str(getattr(metadata, "cfa_pattern")).casefold(),
            None if camera in {"", "unknown"} else camera,
            getattr(metadata, "gain", None),
            getattr(metadata, "offset", None),
            None if readout in {"", "unknown"} else readout,
        )

    for raw_path in arguments.master_flat:
        path = Path(raw_path).expanduser()
        metadata = probe_frame_metadata(path)
        if metadata.role is not FrameRole.MASTER_FLAT or metadata.role_conflicts:
            raise ValueError(
                f"NOT_MASTER_FLAT: {path}: {metadata.role.value}"
            )
        canonical = path.resolve(strict=True)
        explicit.append((canonical, override_key(metadata)))
    if explicit:
        override_keys = {key for _, key in explicit}
        for normalized, candidate in list(candidates.items()):
            if override_key(probe_frame_metadata(candidate)) in override_keys:
                del candidates[normalized]
        for canonical, _ in explicit:
            candidates[os.path.normcase(str(canonical))] = canonical
    return sorted(candidates.values(), key=lambda path: os.path.normcase(str(path)))


def _run_prepare_wbpp(arguments: argparse.Namespace) -> int:
    from .analysis import analyze_measurements
    from .identity import compute_file_identity
    from .measure import measure_paths
    from .models import FrameRole, RunResult
    from .prepare import apply_prepare_plan, build_prepare_plan
    from .quality_gate import evaluate_quality_gate
    from .readers import discover_paths, probe_frame_metadata
    from .report import write_reports

    if arguments.workers < 1:
        raise ValueError("--workers must be positive")
    destination = Path(arguments.output).expanduser().resolve(strict=False)
    report_output = (
        Path(arguments.report_output).expanduser().resolve(strict=False)
        if arguments.report_output
        else destination.with_name(destination.name + "-qc")
    )
    try:
        report_output.relative_to(destination)
    except ValueError:
        pass
    else:
        raise ValueError("REPORT_INSIDE_DESTINATION: choose a sibling report directory")

    inputs = [str(Path(value).expanduser()) for value in arguments.inputs]
    discovered = discover_paths(inputs)
    light_paths: list[Path] = []
    role_counts: dict[str, int] = {}
    for path in discovered:
        metadata = probe_frame_metadata(path)
        if metadata.role_conflicts:
            raise ValueError(
                f"METADATA_ROLE_CONFLICT: {path}: "
                + "; ".join(metadata.role_conflicts)
            )
        role_counts[metadata.role.value] = role_counts.get(metadata.role.value, 0) + 1
        if metadata.role is FrameRole.LIGHT:
            if any(
                str(value).startswith(("FITS:", "XISF:"))
                for value in metadata.role_evidence
            ):
                light_paths.append(path)
            else:
                role_counts["NON_AUTHORITATIVE_LIGHT"] = (
                    role_counts.get("NON_AUTHORITATIVE_LIGHT", 0) + 1
                )
    if not light_paths:
        raise ValueError("NO_LIGHT_FRAMES: no authoritative LIGHT frames were found")

    for raw_input in inputs:
        source = Path(raw_input).expanduser().resolve(strict=True)
        source_root = source.parent if source.is_file() else source
        if destination == source_root or destination.is_relative_to(source_root):
            raise ValueError(
                f"DESTINATION_INSIDE_INPUT: {destination} is inside {source_root}"
            )

    config = load_config(arguments.config)
    if arguments.no_thumbnails:
        config = replace(config, make_thumbnails=False)
    report_output.mkdir(parents=True, exist_ok=True)
    print(
        f"Discovered {len(discovered)} supported image(s); "
        f"screening {len(light_paths)} authoritative LIGHT frame(s).",
        flush=True,
    )
    measurements = measure_paths(
        light_paths, report_output, config, workers=arguments.workers
    )
    groups, frames = analyze_measurements(measurements, config)
    evaluate_quality_gate(frames, measurements, config)
    run = RunResult(
        schema_version=3,
        algorithm_version=__version__,
        generated_at=datetime.now(timezone.utc),
        inputs=[str(Path(value).expanduser().resolve()) for value in inputs],
        output_directory=str(report_output),
        config=config.serializable(),
        groups=groups,
        frames=frames,
        warnings=[
            "IGNORED_NON_LIGHT_COUNTS: "
            + json.dumps(
                {
                    role: count
                    for role, count in sorted(role_counts.items())
                    if role != FrameRole.LIGHT.value
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        ],
    )
    written = write_reports(run)
    results_identity = compute_file_identity(written["json"])
    master_flats = _collect_master_flat_paths(arguments)
    plan = build_prepare_plan(
        frames,
        master_flats,
        destination,
        adjudication=arguments.adjudication,
        qc_report={
            "path": str(written["json"].resolve()),
            "sha256": results_identity.sha256,
            "sizeBytes": results_identity.size_bytes,
        },
    )
    plan_path = report_output / "prepare-plan.json"
    _atomic_json(plan_path, plan)

    result: dict[str, object] = {
        "status": "PLANNED",
        "planId": plan["planId"],
        "destination": str(destination),
    }
    if arguments.apply:
        result = apply_prepare_plan(plan, destination)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"QC report: {written['html']}", flush=True)
    print(f"Prepare plan: {plan_path}", flush=True)
    print("No input frame was modified, moved, or deleted.", flush=True)
    return 0


def _run_apply_wbpp_plan(arguments: argparse.Namespace) -> int:
    from .prepare import apply_prepare_plan, load_prepare_plan

    plan = load_prepare_plan(arguments.plan, require_qc_report=True)
    destination = arguments.output or str(plan["destination"])
    result = apply_prepare_plan(plan, destination)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    print("No input frame was modified, moved, or deleted.", flush=True)
    return 0


def _run_doctor(arguments: argparse.Namespace) -> int:
    versions = {
        "light-frame-qc": __version__,
        "python": sys.version.split()[0],
    }
    modules = {
        "numpy": "numpy",
        "scipy": "scipy",
        "astropy": "astropy",
        "sep": "sep",
        "scikit-image": "skimage",
        "astroalign": "astroalign",
        "lz4": "lz4",
        "zstandard": "zstandard",
    }
    missing = False
    for label, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            versions[label] = f"MISSING: {type(error).__name__}: {error}"
            missing = True
        else:
            versions[label] = getattr(module, "__version__", "unknown")
    print(json.dumps(versions, indent=2, sort_keys=True))
    if arguments.paths:
        if missing:
            print("supportedFrames=UNAVAILABLE")
            return 1
        from .readers import discover_paths

        paths = discover_paths(arguments.paths)
        print(f"supportedFrames={len(paths)}")
        for path in paths[:20]:
            print(path)
        if len(paths) > 20:
            print(f"... and {len(paths) - 20} more")
    return 1 if missing else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "analyze":
            return _run_analyze(arguments)
        if arguments.command == "doctor":
            return _run_doctor(arguments)
        if arguments.command == "prepare-wbpp":
            return _run_prepare_wbpp(arguments)
        if arguments.command == "apply-wbpp-plan":
            return _run_apply_wbpp_plan(arguments)
        if arguments.command == "show-config":
            print(
                json.dumps(
                    DEFAULT_CONFIG.serializable(),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        if arguments.command == "show-gate-policy":
            from .quality_gate import GatePolicy

            policy = GatePolicy.from_qc_config(DEFAULT_CONFIG)
            print(
                json.dumps(
                    {
                        "policyDigest": policy.canonical_digest(),
                        "policy": policy.serializable(),
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        parser.error("unknown command")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"light-frame-qc: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
