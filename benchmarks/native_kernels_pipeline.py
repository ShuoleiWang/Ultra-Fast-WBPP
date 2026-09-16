"""Benchmark the native CPU kernels against the NumPy reference on one machine.

Three boundaries are measured on synthetic 6252x4176 mono data so the report
can be compared across Apple M-series machines:

* ``warp``: one Lanczos-3 registration of a 26 MP calibrated frame
  (``_register_frame``), NumPy reference versus native kernel;
* ``integration``: the complete ``integrate_expressions`` reduction of a
  registered stack (rejection floor, transient preparation, median/MAD
  rejection, weighted mean, maps) with and without the native kernels;
* ``pipeline``: ``run_portable_pipeline`` from raw calibration frames to an
  unsolved master, i.e. fused calibrate+warp plus integration, both ways.

Both paths must publish identical pixels; the report records that check.
Astrometry, QC, XISF and the desktop are outside this boundary.

    .venv/bin/python benchmarks/native_kernels_pipeline.py --output build/native-kernels.json

Reports are local evidence and are not checked in.

Pass ``--frames`` and ``--shape`` to scale the fixture; the default 12 frames
at 4176x6252 need about 4 GB of scratch disk under ``--scratch``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

from astropy.io import fits
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
for relative in ("packages/openastroflow-engine/src", "packages/light-frame-qc/src", "engine/native/python"):
    candidate = REPOSITORY / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from openastroflow_engine import native_kernels  # noqa: E402
from openastroflow_engine.calibration import (  # noqa: E402
    FrameExpression,
    IntegrationMapPaths,
    IntegrationParameters,
    read_frame_info,
    integrate_expressions,
)
from openastroflow_engine.hardware import detect_hardware  # noqa: E402
from openastroflow_engine.performance_profile import select_execution_tuning  # noqa: E402
from openastroflow_engine import pixel_pipeline as pipeline  # noqa: E402
from openastroflow_engine.pixel_pipeline import AffineTransform, PipelineParameters  # noqa: E402


def _portable_path(path: Path) -> str:
    """Record a path relative to the repository so reports carry no private prefix."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY).as_posix()
    except ValueError:
        return resolved.name


def _git_commit() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPOSITORY, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": True}


def _header(role: str, *, exposure: float, filter_name: str = "B") -> fits.Header:
    header = fits.Header()
    header["IMAGETYP"] = role
    header["FILTER"] = filter_name
    header["OBJECT"] = "BENCH"
    header["INSTRUME"] = "BENCH-CAM"
    header["EXPTIME"] = exposure
    header["GAIN"] = 26
    header["OFFSET"] = 30
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "MODE-1"
    header["BAYERPAT"] = "NONE"
    header["CCD-TEMP"] = -10.0
    return header


def _render_stars(
    shape: tuple[int, int],
    sx: np.ndarray,
    sy: np.ndarray,
    amplitude: np.ndarray,
    *,
    sigma: float = 1.6,
) -> np.ndarray:
    """Render Gaussian stars at fractional positions (sub-pixel accurate)."""

    height, width = shape
    stars = np.zeros(shape, dtype=np.float64)
    yy, xx = np.mgrid[-7:8, -7:8]
    for cx, cy, a in zip(sx, sy, amplitude, strict=True):
        ix, iy = int(round(cx)), int(round(cy))
        if ix < 7 or iy < 7 or ix >= width - 8 or iy >= height - 8:
            continue
        fx, fy = cx - ix, cy - iy
        stamp = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * sigma**2))
        stars[iy - 7 : iy + 8, ix - 7 : ix + 8] += a * stamp
    return stars


def _dither(index: int, shape: tuple[int, int]) -> tuple[float, float, float]:
    """Deterministic per-frame dither: pixels of shift and a small rotation."""

    if index == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(1000 + index)
    dx = float(rng.uniform(-12.0, 12.0))
    dy = float(rng.uniform(-12.0, 12.0))
    theta = float(np.deg2rad(rng.uniform(-0.15, 0.15)))
    return dx, dy, theta


def _synthetic_dataset(
    root: Path,
    shape: tuple[int, int],
    frames: int,
    seed: int,
    *,
    dither: bool = False,
) -> dict[str, list[Path]]:
    height, width = shape
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width]
    vignetting = 1.0 - 0.25 * (((x - width / 2) / width) ** 2 + ((y - height / 2) / height) ** 2)
    sky = 1200.0 + 0.02 * x + 0.015 * y
    star_count = 2500
    sx = rng.uniform(20, width - 21, star_count)
    sy = rng.uniform(20, height - 21, star_count)
    amplitude = rng.lognormal(6.5, 1.0, star_count)
    reference_stars = _render_stars(shape, sx, sy, amplitude)
    paths: dict[str, list[Path]] = {"bias": [], "dark": [], "flat": [], "light": []}
    for index in range(3):
        bias = np.full(shape, 500.0) + rng.normal(0, 4, shape)
        path = root / "bias" / f"bias_{index}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(path, np.rint(bias).astype(np.uint16), _header("Bias", exposure=0.001))
        paths["bias"].append(path)
        dark = np.full(shape, 512.0) + rng.normal(0, 5, shape)
        path = root / "dark" / f"dark_{index}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(path, np.rint(dark).astype(np.uint16), _header("Dark", exposure=300.0))
        paths["dark"].append(path)
        flat = 500.0 + 20000.0 * vignetting + rng.normal(0, 30, shape)
        path = root / "flat" / f"flat_{index}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(path, np.rint(flat).astype(np.uint16), _header("Flat", exposure=2.0))
        paths["flat"].append(path)
    for index in range(frames):
        if dither and index > 0:
            dx, dy, theta = _dither(index, shape)
            cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
            rx = np.cos(theta) * (sx - cx) - np.sin(theta) * (sy - cy) + cx + dx
            ry = np.sin(theta) * (sx - cx) + np.cos(theta) * (sy - cy) + cy + dy
            stars = _render_stars(shape, rx, ry, amplitude)
        else:
            stars = reference_stars
        signal = (sky + stars) * vignetting
        light = 512.0 + signal * (1.0 + 0.02 * index) + rng.normal(0, 25, shape)
        if index == frames - 1:
            light[height // 3, :] += 30000.0  # satellite-like trail
        path = root / "light" / f"light_{index:02d}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = _header("Light", exposure=300.0)
        header["DATE-OBS"] = f"2026-09-01T0{index % 10}:{(index * 7) % 60:02d}:00"
        fits.writeto(path, np.clip(np.rint(light), 0, 65535).astype(np.uint16), header)
        paths["light"].append(path)
    return paths


def _transforms(paths: list[Path]) -> dict[str, AffineTransform]:
    result = {}
    for index, path in enumerate(paths):
        angle = np.deg2rad(0.08 * index)
        result[str(path)] = AffineTransform.from_value(
            (
                (np.cos(angle), -np.sin(angle), 1.37 * index),
                (np.sin(angle), np.cos(angle), -0.83 * index),
                (0.0, 0.0, 1.0),
            )
        )
    return result


def _timed(function, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return samples


def _with_native(enabled: bool):
    if enabled:
        os.environ.pop(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, None)
    else:
        os.environ[native_kernels.DISABLE_ENVIRONMENT_VARIABLE] = "1"
    native_kernels.reset_native_kernel_cache()


def benchmark_warp(scratch: Path, shape: tuple[int, int], repeats: int, threads: int) -> dict[str, Any]:
    height, width = shape
    rng = np.random.default_rng(1)
    calibrated = rng.normal(1000, 30, shape).astype(np.float32)
    header = fits.Header()
    header["IMAGETYP"] = "Calibrated Light"
    header["OAFNDOM"] = "INTEGER_16_PHYSICAL_0_BASED"
    header["OAFNSCL"] = 65535.0
    source = scratch / "warp-source.fits"
    fits.writeto(source, calibrated, header)
    info = replace(
        read_frame_info(source), normalized_unit_scale=65535.0,
        numeric_domain_authority="CONTENT_BOUND_OVERRIDE",
    )
    angle = np.deg2rad(0.7)
    transform = AffineTransform.from_value(
        ((np.cos(angle), -np.sin(angle), 12.3), (np.sin(angle), np.cos(angle), -7.6), (0.0, 0.0, 1.0))
    )
    budget = 2 * 1024**3
    report: dict[str, Any] = {"shape": list(shape), "transformDegrees": 0.7, "repeats": repeats}
    outputs = {}
    for label, enabled, kwargs in (
        ("numpy", False, {}),
        ("native-1-thread", True, {"native_threads": 1}),
        (f"native-{threads}-threads", True, {"native_threads": threads}),
    ):
        _with_native(enabled)
        samples = []
        for index in range(repeats):
            destination = scratch / f"warp-{label}-{index}.fits"
            execution: dict[str, Any] = {}
            started = time.perf_counter()
            pipeline._register_frame(
                source, destination, transform, info, max_memory_bytes=budget,
                resampler="lanczos-3-clamped", execution=execution, **kwargs,
            )
            samples.append(time.perf_counter() - started)
            outputs[label] = destination
        report[label] = {
            "seconds": samples,
            "medianSeconds": statistics.median(samples),
            "warpBackend": execution.get("warpBackend"),
        }
    reference = fits.getdata(outputs["numpy"])
    report["nativeMatchesNumpy"] = all(
        np.array_equal(fits.getdata(path), reference, equal_nan=True)
        for label, path in outputs.items() if label != "numpy"
    )
    report["speedupMultiThread"] = report["numpy"]["medianSeconds"] / report[f"native-{threads}-threads"]["medianSeconds"]
    _with_native(True)
    return report


def benchmark_integration(scratch: Path, shape: tuple[int, int], frames: int, repeats: int, threads: int) -> dict[str, Any]:
    rng = np.random.default_rng(2)
    paths = []
    base = rng.normal(1000, 30, shape).astype(np.float32)
    for index in range(frames):
        values = base + rng.normal(0, 25, shape).astype(np.float32)
        values[:, : 3 + index] = np.nan
        if index == frames - 1:
            values[shape[0] // 2, :] += 20000.0
        path = scratch / f"registered-{index:02d}.fits"
        fits.writeto(path, values, fits.Header({"IMAGETYP": "Registered Light"}))
        paths.append(path)
    tuning = select_execution_tuning(detect_hardware())
    parameters = IntegrationParameters(max_memory_bytes=tuning.integration_memory_bytes)
    report: dict[str, Any] = {"shape": list(shape), "frames": frames, "repeats": repeats,
                              "integrationMemoryBytes": tuning.integration_memory_bytes}
    masters = {}
    for label, enabled in (("numpy", False), ("native", True)):
        _with_native(enabled)
        samples = []
        for index in range(repeats):
            output = scratch / f"master-{label}-{index}.fits"
            maps = IntegrationMapPaths(
                scratch / f"accepted-{label}-{index}.fits",
                scratch / f"coverage-{label}-{index}.fits",
                scratch / f"rejected-{label}-{index}.fits",
            )
            started = time.perf_counter()
            result = integrate_expressions(
                [FrameExpression(str(path)) for path in paths], output,
                parameters=parameters, map_paths=maps, native_threads=threads,
            )
            samples.append(time.perf_counter() - started)
            masters[label] = output
        report[label] = {
            "seconds": samples,
            "medianSeconds": statistics.median(samples),
            "rejectionKernel": result.execution["rejectionMask"]["kernel"],
            "reducer": result.execution["reducer"],
            "rejectedSamples": result.rejected_samples,
        }
    report["nativeMatchesNumpy"] = bool(
        np.array_equal(fits.getdata(masters["native"]), fits.getdata(masters["numpy"]), equal_nan=True)
    )
    report["speedup"] = report["numpy"]["medianSeconds"] / report["native"]["medianSeconds"]
    _with_native(True)
    return report


def benchmark_pipeline(scratch: Path, dataset: dict[str, list[Path]], repeats: int) -> dict[str, Any]:
    tuning = select_execution_tuning(detect_hardware())
    parameters = replace(
        PipelineParameters(),
        integration=IntegrationParameters(max_memory_bytes=tuning.integration_memory_bytes),
        registration_memory_bytes=tuning.registration_memory_bytes,
        ordinary_integration_backend="portable-cpu",
        materialize_calibrated_lights=False,
    )
    transforms = _transforms(dataset["light"])
    report: dict[str, Any] = {"frames": len(dataset["light"]), "repeats": repeats,
                              "tuning": tuning.serializable()}
    masters = {}
    for label, enabled in (("numpy", False), ("native", True)):
        _with_native(enabled)
        samples = []
        for index in range(repeats):
            output = scratch / f"pipeline-{label}-{index}"
            started = time.perf_counter()
            result = pipeline.run_portable_pipeline(
                bias_files=dataset["bias"], dark_files=dataset["dark"], flat_files=dataset["flat"],
                light_files=dataset["light"], output_directory=output,
                transforms=transforms, parameters=parameters,
            )
            samples.append(time.perf_counter() - started)
            receipt = json.loads(Path(result.receipt_path).read_text())
            masters[label] = Path(result.master_light_paths[0])
            if index < repeats - 1:
                shutil.rmtree(output)
        registration = receipt["statistics"]["registration"]
        report[label] = {
            "seconds": samples,
            "medianSeconds": statistics.median(samples),
            "registrationWallSeconds": registration["wallSeconds"],
            "warpBackends": registration["warpBackends"],
            "cpuWorkersUsed": registration["cpuWorkersUsed"],
            "nativeThreadsPerWorker": registration["nativeThreadsPerWorker"],
        }
    report["nativeMatchesNumpy"] = bool(
        np.array_equal(fits.getdata(masters["native"]), fits.getdata(masters["numpy"]), equal_nan=True)
    )
    report["speedup"] = report["numpy"]["medianSeconds"] / report["native"]["medianSeconds"]
    _with_native(True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="new JSON report path (must not exist)")
    parser.add_argument("--scratch", default=None, help="scratch directory (default: temporary)")
    parser.add_argument("--shape", default="4176x6252", help="HEIGHTxWIDTH of synthetic frames")
    parser.add_argument("--frames", type=int, default=12, help="Light frame count for integration/pipeline")
    parser.add_argument("--repeats", type=int, default=3, help="measured repetitions per boundary")
    parser.add_argument("--skip", default="", help="comma list of boundaries to skip: warp,integration,pipeline")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists")
    height, width = (int(value) for value in args.shape.lower().split("x"))
    shape = (height, width)
    skip = {item.strip() for item in args.skip.split(",") if item.strip()}
    kernels = native_kernels.load_native_kernels()
    if kernels is None:
        parser.error("native kernels are unavailable; build engine/native first")
    tuning = select_execution_tuning(detect_hardware())
    threads = tuning.cpu_workers
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "ultra-fast-wbpp-native-kernel-benchmark-v1",
        "boundary": "synthetic frames; no QC, XISF, astrometry, or desktop",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hardware": detect_hardware().serializable(),
            "tuning": tuning.serializable(),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "nativeLibrary": _portable_path(kernels.library_path),
        "source": _git_commit(),
        "results": {},
    }
    with tempfile.TemporaryDirectory(prefix="oaf-bench-") as temporary:
        scratch = Path(args.scratch) if args.scratch else Path(temporary)
        scratch.mkdir(parents=True, exist_ok=True)
        # macOS temporary roots are symlinks; keep every generated path canonical.
        scratch = scratch.resolve(strict=True)
        if "warp" not in skip:
            report["results"]["warp"] = benchmark_warp(scratch, shape, args.repeats, threads)
            print("warp:", json.dumps({k: v for k, v in report["results"]["warp"].items() if k != "seconds"}, default=str), flush=True)
        if "integration" not in skip:
            report["results"]["integration"] = benchmark_integration(scratch, shape, args.frames, args.repeats, threads)
            print("integration:", json.dumps(report["results"]["integration"], default=str), flush=True)
        if "pipeline" not in skip:
            dataset = _synthetic_dataset(scratch / "dataset", shape, args.frames, seed=7)
            report["results"]["pipeline"] = benchmark_pipeline(scratch, dataset, max(1, args.repeats - 1))
            print("pipeline:", json.dumps(report["results"]["pipeline"], default=str), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
