from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
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
