"""Stage-by-stage wall-time profile of one ordinary mono E2E run.

Runs ``run_e2e`` (quality control, registration calibration, registration,
fused calibrate+warp, normalization, integration, previews, verification and
publication) on a synthetic dithered 6252x4176 dataset with the repository's
fake catalog solver, so every stage except the external ``solve-field`` is
measured on this machine.  The report records per-stage wall time from the
progress events plus the engine's own receipt timings and, optionally, a
cProfile summary of the hottest functions.

    .venv/bin/python benchmarks/e2e_stage_timings.py \
        --output build/e2e-stages.json --frames 12 --backend auto \
        --profile build/e2e-profile.txt

Reports are local evidence and are not checked in.

Astrometry is faked; the desktop, XISF inputs and Drizzle are outside this
boundary.  Both the fake solver and the synthetic frames are deterministic.
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import io
import json
from pathlib import Path
import platform
import pstats
import sys
import tempfile
import time
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
for relative in (
    "packages/openastroflow-engine/src",
    "packages/light-frame-qc/src",
    "packages/openastroflow-registration",
    "packages/openastroflow-engine/tests",
    "benchmarks",
):
    candidate = REPOSITORY / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from native_kernels_pipeline import _git_commit, _portable_path, _synthetic_dataset  # noqa: E402
from openastroflow_engine import native_kernels  # noqa: E402
from openastroflow_engine.calibration import IntegrationParameters  # noqa: E402
from openastroflow_engine.e2e import E2ERequest, IntegrationMode, run_e2e  # noqa: E402
from openastroflow_engine.hardware import detect_hardware  # noqa: E402
from openastroflow_engine.performance_profile import select_execution_tuning  # noqa: E402
from openastroflow_engine.pixel_pipeline import PipelineParameters  # noqa: E402
from test_e2e import FakeSolver  # noqa: E402


def _stage_durations(events: list[tuple[float, str, str, str]]) -> dict[str, Any]:
    """Derive per-stage wall seconds from ordered progress events."""

    durations: dict[str, float] = {}
    started: dict[str, float] = {}
    order: list[str] = []
    for timestamp, stage, status, _message in events:
        if status == "started":
            started[stage] = timestamp
            if stage not in order:
                order.append(stage)
        elif status in {"completed", "failed"} and stage in started:
            durations[stage] = timestamp - started.pop(stage)
    return {"order": order, "seconds": durations}


def run_once(
    dataset: dict[str, list[Path]],
    output: Path,
    *,
    backend: str,
    workers: int,
    profile_path: Path | None,
) -> dict[str, Any]:
    tuning = select_execution_tuning(detect_hardware())
    parameters = replace(
        PipelineParameters(),
        integration=replace(
            IntegrationParameters(), max_memory_bytes=tuning.integration_memory_bytes
        ),
        registration_memory_bytes=tuning.registration_memory_bytes,
        ordinary_integration_backend=backend,
    )
    request = E2ERequest(
        light_files=tuple(str(path) for path in dataset["light"]),
        flat_files=tuple(str(path) for path in dataset["flat"]),
        dark_files=tuple(str(path) for path in dataset["dark"]),
        bias_files=tuple(str(path) for path in dataset["bias"]),
        output_directory=str(output),
        integration_mode=IntegrationMode.ORDINARY,
        workers=workers,
        pipeline_parameters=parameters,
        ra_hint_degrees=150.0,
        dec_hint_degrees=20.0,
        field_of_view_degrees=3.0,
        search_radius_degrees=5.0,
    )
    events: list[tuple[float, str, str, str]] = []

    def progress(event: Any) -> None:
        events.append((time.perf_counter(), str(event.stage.value), str(event.status), str(event.message)))

    profiler = cProfile.Profile() if profile_path is not None else None
    started = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    result = run_e2e(request, solver_backends=(FakeSolver(),), progress=progress)
    if profiler is not None:
        profiler.disable()
    wall = time.perf_counter() - started
    report: dict[str, Any] = {
        "success": result.success,
        "code": result.code,
        "wallSeconds": wall,
        "stages": _stage_durations(events),
        "workers": workers,
        "backend": backend,
        "tuning": tuning.serializable(),
    }
    if not result.success:
        report["message"] = result.message
        return report
    root = Path(result.output_directory)
    qc = json.loads((root / "qc" / "manifest.json").read_text())
    report["qualityControl"] = {
        "timings": qc.get("timings"),
        "analysisCache": qc.get("analysisCache"),
        "passed": len(result.passed_light_paths),
        "excluded": len(result.excluded_light_paths),
    }
    registration = json.loads((root / "receipts" / "registration.json").read_text())
    report["registration"] = {
        "timingSeconds": registration.get("timingSeconds"),
        "fullRefineSecondsSum": sum(
            float(item.get("fullRefineSeconds", 0.0)) for item in registration.get("transforms", [])
        ),
        "transformModels": sorted({item.get("transformModel") for item in registration.get("transforms", [])}),
        "maximumRmsFullPixels": max(
            (float(item.get("rmsFullPixels", 0.0)) for item in registration.get("transforms", [])),
            default=None,
        ),
    }
    pixel = json.loads((root / "receipts" / "pixel-pipeline.json").read_text())
    statistics = pixel["statistics"]
    report["pixelPipeline"] = {"registration": statistics.get("registration")}
    groups = {}
    for filter_name, group in statistics.get("integrationGroups", {}).items():
        execution = group["integration"]["execution"]
        groups[filter_name] = {
            key: execution.get(key)
            for key in (
                "selectedBackend", "tileRows", "cpuWorkersUsed", "kernelThreadsPerWorker",
                "tilesSubmitted", "gpuWallSecondsSum", "gpuSecondsSum", "reducer", "nativeThreads",
            )
        }
        groups[filter_name]["rejectionKernel"] = execution.get("rejectionMask", {}).get("kernel")
        groups[filter_name]["transientStatus"] = (
            execution.get("rejectionMask", {}).get("spatialTransients", {}).get("status")
        )
        groups[filter_name]["rejectedSamples"] = group["integration"].get("rejectedSamples")
        normalization = group.get("globalNormalization", {})
        modes: dict[str, int] = {}
        for frame in normalization.get("evidence", {}).get("frames", []):
            modes[frame.get("mode")] = modes.get(frame.get("mode"), 0) + 1
        groups[filter_name]["globalNormalizationModes"] = modes
    report["pixelPipeline"]["integrationGroups"] = groups
    if profiler is not None:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs().sort_stats("cumulative").print_stats(45)
        stats.sort_stats("tottime").print_stats(45)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(stream.getvalue())
        report["profileText"] = _portable_path(profile_path)
        # Raw statistics allow caller/callee attribution without rerunning.
        raw_path = profile_path.with_suffix(".prof")
        profiler.dump_stats(str(raw_path))
        report["profileStats"] = _portable_path(raw_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="new JSON report path (must not exist)")
    parser.add_argument("--scratch", default=None, help="scratch directory (default: temporary)")
    parser.add_argument("--shape", default="4176x6252", help="HEIGHTxWIDTH of synthetic frames")
    parser.add_argument("--frames", type=int, default=12, help="Light frame count")
    parser.add_argument("--backend", default="auto", help="ordinary integration backend: auto, portable-cpu, generic-apple-metal")
    parser.add_argument("--workers", type=int, default=None, help="QC/registration workers (default: hardware profile)")
    parser.add_argument("--profile", default=None, help="write a cProfile text summary to this path")
    parser.add_argument("--disable-native", action="store_true", help="force the NumPy kernels")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists")
    height, width = (int(value) for value in args.shape.lower().split("x"))
    if args.disable_native:
        import os

        os.environ[native_kernels.DISABLE_ENVIRONMENT_VARIABLE] = "1"
        native_kernels.reset_native_kernel_cache()
    tuning = select_execution_tuning(detect_hardware())
    workers = args.workers or tuning.cpu_workers
    kernels = native_kernels.load_native_kernels()
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "ultra-fast-wbpp-e2e-stage-timings-v1",
        "boundary": "synthetic dithered frames, fake catalog solver; no XISF, Drizzle, or desktop",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {
            "platform": platform.platform(),
            "hardware": detect_hardware().serializable(),
            "tuning": tuning.serializable(),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "nativeKernels": _portable_path(kernels.library_path) if kernels is not None else None,
        "source": _git_commit(),
        "shape": [height, width],
        "frames": args.frames,
    }
    with tempfile.TemporaryDirectory(prefix="oaf-e2e-bench-") as temporary:
        scratch = (Path(args.scratch) if args.scratch else Path(temporary)).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        dataset_started = time.perf_counter()
        dataset = _synthetic_dataset(scratch / "dataset", (height, width), args.frames, seed=11, dither=True)
        report["datasetSeconds"] = time.perf_counter() - dataset_started
        report["run"] = run_once(
            dataset,
            scratch / "e2e-output",
            backend=args.backend,
            workers=workers,
            profile_path=Path(args.profile) if args.profile else None,
        )
        print(json.dumps({key: value for key, value in report["run"].items() if key != "tuning"}, indent=1, default=str), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
