"""Trace a real Ultra-Fast WBPP run: stage timeline, hot functions, CPU and memory.

Runs any engine CLI command in this process with a curated set of timers
around the stages, the native kernels, the FITS/XISF I/O and the numerical
steps, samples the process's CPU time and resident memory, and writes

- a Chrome trace-event file (open it in https://ui.perfetto.dev or
  chrome://tracing): one row per thread with the nested timed calls, one row
  per pipeline stage, counters for CPU utilisation and RSS, and the events of
  the spawned worker processes (quality control, registration analysis) on
  their own rows;
- a summary JSON and a table on stderr: per stage wall time, CPU time of this
  process and of its children, parallel efficiency; per timer the inclusive
  and exclusive time, calls, mean and maximum, sorted by exclusive time.

    .venv/bin/python benchmarks/trace_run.py --trace build/trace.json \\
        --summary build/trace-summary.json -- \\
        run-project ~/Astro/Target ~/Astro/Masters/masterDark.xisf \\
        --recipe recipe.json --output ~/Astro/Results/target-trace --workers 8

Everything after ``--`` is passed to ``ufwbpp.cli.main``;
``--progress-json`` is added when absent because the stage rows come from the
progress events.  The run's own result JSON still goes to stdout.  Timers add
a few microseconds per call and never change pixels; the worker processes
find this file again through the ``spawn`` start method and install the same
timers when ``UFWBPP_TRACE_DIR`` is set.  Traces are local evidence and are not
checked in.
"""

from __future__ import annotations

import argparse
import atexit
import functools
import json
import os
from pathlib import Path
import sys

try:  # POSIX only; Windows falls back to os.times() and has no children CPU.
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
import threading
import time
from typing import Any, Callable

REPOSITORY = Path(__file__).resolve().parents[1]
for relative in (
    "packages/engine/src",
    "packages/light-frame-qc/src",
    "packages/registration",
):
    candidate = str(REPOSITORY / relative)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

TRACE_DIR_VARIABLE = "UFWBPP_TRACE_DIR"
TRACE_EPOCH_VARIABLE = "UFWBPP_TRACE_EPOCH_NS"

# (module, attribute or Class.method, label, category).  Missing attributes
# are reported and skipped, so the list may name symbols of newer code.
TARGETS: tuple[tuple[str, str, str, str], ...] = (
    # quality control (mostly in worker processes)
    ("lightframeqc.measure", "measure_frame", "qc.measure_frame", "qc"),
    ("lightframeqc.readers", "read_frame_preview", "io.read_frame_preview", "io"),
    ("lightframeqc.native_psf", "measure_native_psf", "qc.native_psf", "qc"),
    ("lightframeqc.content_hash", "file_sha256", "io.file_sha256", "io"),
    # registration
    ("ufwbpp_registration.pipeline", "analyze_frames", "registration.analyze_frames", "registration"),
    ("ufwbpp_registration.pipeline", "analyze_frame", "registration.analyze_frame", "registration"),
    ("ufwbpp_registration.pipeline", "detect_stars", "registration.detect_stars", "registration"),
    ("ufwbpp_registration.pipeline", "_read_calibrated_preview", "registration.read_calibrated_preview", "registration"),
    ("ufwbpp_registration.pipeline", "register_analyses", "registration.register_analyses", "registration"),
    ("ufwbpp_registration.pipeline", "_refine_full_resolution", "registration.refine_full_resolution", "registration"),
    ("ufwbpp_registration.pipeline", "_estimate_one", "registration.estimate_one", "registration"),
    ("ufwbpp_registration.pipeline", "_estimate_via_bridge", "registration.estimate_via_bridge", "registration"),
    ("ufwbpp_registration.pipeline", "_local_centroids", "registration.local_centroids", "registration"),
    ("ufwbpp_registration.pipeline", "warp_image", "registration.validate_warp", "registration"),
    ("ufwbpp_registration.pipeline", "_robust_warp_similarity", "registration.warp_similarity", "registration"),
    ("ufwbpp_registration.pipeline", "read_full_image", "io.read_full_image", "io"),
    # fused calibrate + warp
    ("ufwbpp.pixel_pipeline", "_calibrate_and_register_frames", "fused.all_lights", "fused"),
    ("ufwbpp.pixel_pipeline", "_process_light_job", "fused.light_job", "fused"),
    ("ufwbpp.pixel_pipeline", "_expression_rows", "fused.calibrate_rows", "fused"),
    ("ufwbpp.pixel_pipeline", "bilinear_debayer", "fused.debayer", "fused"),
    ("ufwbpp.pixel_pipeline", "_register_frame", "fused.register_frame", "fused"),
    ("ufwbpp.pixel_pipeline", "_write_float_fits", "io.write_calibrated", "io"),
    ("ufwbpp.pixel_pipeline", "_shared_auto_crop", "crop.shared_auto_crop", "crop"),
    ("ufwbpp.pixel_pipeline", "_crop_fits", "crop.crop_fits", "crop"),
    ("ufwbpp.native_kernels", "NativeKernels.warp_lanczos3", "native.warp_lanczos3", "native"),
    # normalization
    ("ufwbpp.global_normalization", "fit_registered_group_global_normalization", "normalization.fit_group", "normalization"),
    ("ufwbpp.global_normalization", "_group_tile_levels", "normalization.group_tile_levels", "normalization"),
    ("ufwbpp.global_normalization", "_fit_sky_response", "normalization.fit_sky_response", "normalization"),
    ("ufwbpp.global_normalization", "_fit_coefficient", "normalization.fit_coefficient", "normalization"),
    ("ufwbpp.global_normalization", "_paired_samples", "normalization.paired_samples", "normalization"),
    ("ufwbpp.global_normalization", "_fit_additive_offset_grid", "normalization.fit_additive_offset_grid", "normalization"),
    ("ufwbpp.global_normalization", "_tile_backgrounds", "normalization.tile_backgrounds", "normalization"),
    ("ufwbpp.global_normalization", "_smooth_offset_grid", "normalization.smooth_offset_grid", "normalization"),
    ("ufwbpp.global_normalization", "_ReferenceSampleCache.sample", "normalization.reference_sample", "normalization"),
    ("ufwbpp.global_normalization", "_ReferenceSampleCache.rows", "normalization.reference_rows", "normalization"),
    ("ufwbpp.calibration", "_add_offset_grid_rows", "integration.add_offset_grid_rows", "integration"),
    ("ufwbpp.transient_rejection", "fast_radon_levels", "transients.fast_radon_levels", "integration"),
    ("ufwbpp.transient_rejection", "_measure_line", "transients.measure_line", "integration"),
    ("ufwbpp.native_kernels", "NativeKernels.tile_offsets", "native.tile_offsets", "native"),
    # integration
    ("ufwbpp.metal_integration", "integrate_registered_group", "integration.group", "integration"),
    ("ufwbpp.calibration", "integrate_expressions", "integration.integrate_expressions", "integration"),
    ("ufwbpp.calibration", "_expression_rows", "integration.expression_rows", "integration"),
    ("ufwbpp.calibration", "_expression_sampled_rows", "integration.expression_sampled_rows", "integration"),
    ("ufwbpp.calibration", "_ordinary_integration_tile", "integration.rejection_tile", "integration"),
    ("ufwbpp.calibration", "_prepare_transient_rejection", "integration.prepare_transients", "integration"),
    ("ufwbpp.calibration", "_combined_integration_weights", "integration.weights", "integration"),
    ("ufwbpp.calibration", "_estimate_rejection_sigma_floor", "integration.sigma_floor", "integration"),
    ("ufwbpp.calibration", "robust_location", "integration.robust_location", "integration"),
    ("ufwbpp.calibration", "fit_residual_background", "transients.fit_residual_background", "integration"),
    ("ufwbpp.transient_rejection", "detect_transient_trails", "transients.detect", "integration"),
    ("ufwbpp.native_kernels", "NativeKernels.mad_rejection", "native.mad_rejection", "native"),
    ("ufwbpp.native_kernels", "NativeKernels.masked_weighted_mean", "native.masked_weighted_mean", "native"),
    ("ufwbpp.native_kernels", "NativeKernels.radon_line_peaks", "native.radon_line_peaks", "native"),
    # FITS I/O of the integration
    ("ufwbpp.calibration", "FitsFrame.read_rows", "io.fits_read_rows", "io"),
    ("ufwbpp.calibration", "FitsFrame.read_sampled_rows", "io.fits_read_sampled_rows", "io"),
    ("ufwbpp.calibration", "FitsFrame.full_values", "io.fits_full_values", "io"),
    ("ufwbpp.calibration", "FitsFloatWriter.write_rows", "io.fits_write_rows", "io"),
    ("ufwbpp.calibration", "FitsFloatWriter.__exit__", "io.fits_writer_close", "io"),
    # drizzle
    ("ufwbpp.drizzle_native", "drizzle_group", "drizzle.group", "drizzle"),
    ("ufwbpp.drizzle_native", "_sha256_of", "drizzle.sha256", "drizzle"),
    ("ufwbpp.native_kernels", "NativeKernels.drizzle_band", "native.drizzle_band", "native"),
    # astrometry, products, publication
    ("ufwbpp.astrometry_net_backend", "AstrometryNetBackend.solve", "astrometry.solve_field", "astrometry"),
    ("ufwbpp.workflows.single_target", "_unify_same_grid_solutions", "astrometry.unify_same_grid", "astrometry"),
    ("ufwbpp.workflows.single_target", "_validate_cross_filter_wcs", "astrometry.validate_cross_filter", "astrometry"),
    ("ufwbpp.workflows.single_target", "_promote_solved_state", "astrometry.promote_solved_state", "astrometry"),
    ("ufwbpp.workflows.single_target", "_ordinary_candidates", "products.ordinary_candidates", "products"),
    ("ufwbpp.workflows.single_target", "_drizzle_candidates", "products.drizzle_candidates", "products"),
    ("ufwbpp.workflows.single_target", "_verify_sources", "verify.sources", "verify"),
    ("ufwbpp.workflows.single_target", "sha256_digest", "verify.sha256", "verify"),
    ("ufwbpp.preview", "render_auto_stretch_preview", "products.preview", "products"),
    ("ufwbpp.color_product", "build_color_product", "products.color_product", "products"),
    ("ufwbpp.workflows.project", "_align_channel", "products.align_channel", "products"),
    ("ufwbpp.workflows.project", "_build_shared_calibration", "calibration.shared_library", "calibration"),
)


class Tracer:
    """Thread-safe recorder of nested timed calls, stage markers and samples.

    Timestamps come from ``time.perf_counter_ns``: system-wide on every
    platform (mach_absolute_time, CLOCK_MONOTONIC, QueryPerformanceCounter),
    so worker-process events line up with the main process, and fine enough
    on Windows, where ``time.monotonic`` ticks every 15.6 ms.
    """

    def __init__(self, epoch_ns: int, profile_labels: frozenset[str] = frozenset()) -> None:
        self.epoch_ns = epoch_ns
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.local = threading.local()
        self.pid = os.getpid()
        self.installed: list[str] = []
        self.missing: list[str] = []
        self.stage_open: dict[str, tuple[float, int]] = {}
        self.stage_records: list[dict[str, Any]] = []
        self._rusage_at_stage: dict[str, tuple[float, float]] = {}
        # Timers whose calls run under cProfile (one profiler per call, so
        # concurrent threads never share one); statistics merge per label.
        self.profile_labels = profile_labels
        self.profiles: dict[str, Any] = {}
        # One interpreter-wide profiler at a time (Python 3.12 refuses a
        # second active tool): a call is profiled only when no other is.
        self.profile_lock = threading.Lock()

    def now_us(self) -> float:
        return (time.perf_counter_ns() - self.epoch_ns) / 1000.0

    # ---- timers
    def wrap(self, function: Callable[..., Any], label: str, category: str) -> Callable[..., Any]:
        tracer = self

        profiled = label in self.profile_labels

        @functools.wraps(function)
        def traced(*args: Any, **kwargs: Any) -> Any:
            start = tracer.now_us()
            stack = getattr(tracer.local, "stack", None)
            if stack is None:
                stack = tracer.local.stack = []
            stack.append(0.0)  # children's inclusive time
            profiler = None
            if profiled and tracer.profile_lock.acquire(blocking=False):
                import cProfile

                try:
                    profiler = cProfile.Profile()
                    profiler.enable()
                except Exception:
                    profiler = None
                    tracer.profile_lock.release()
            try:
                return function(*args, **kwargs)
            finally:
                if profiler is not None:
                    try:
                        profiler.disable()
                        tracer.add_profile(label, profiler)
                    finally:
                        tracer.profile_lock.release()
                end = tracer.now_us()
                children = stack.pop()
                duration = end - start
                if stack:
                    stack[-1] += duration
                with tracer.lock:
                    tracer.events.append(
                        {
                            "name": label,
                            "cat": category,
                            "ph": "X",
                            "ts": start,
                            "dur": duration,
                            "pid": tracer.pid,
                            "tid": threading.get_ident(),
                            "args": {"exclusive_us": duration - children},
                        }
                    )

        traced.__ufwbpp_traced__ = True  # type: ignore[attr-defined]
        return traced

    def add_profile(self, label: str, profiler: Any) -> None:
        import pstats

        with self.lock:
            stats = self.profiles.get(label)
            if stats is None:
                self.profiles[label] = pstats.Stats(profiler)
            else:
                stats.add(profiler)

    def profile_report(self, lines: int = 25) -> str:
        import io

        chunks = []
        for label, stats in self.profiles.items():
            buffer = io.StringIO()
            stats.stream = buffer
            stats.sort_stats("cumulative").print_stats(lines)
            chunks.append(f"== cProfile of {label} (all calls merged)\n{buffer.getvalue()}")
        return "\n".join(chunks)

    def install(self, targets: tuple[tuple[str, str, str, str], ...] = TARGETS) -> None:
        import importlib

        for module_name, attribute, label, category in targets:
            try:
                module = importlib.import_module(module_name)
            except Exception as error:  # pragma: no cover - depends on the environment
                self.missing.append(f"{module_name}: {error}")
                continue
            owner: Any = module
            parts = attribute.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part, None)
                if owner is None:
                    break
            if owner is None or not hasattr(owner, parts[-1]):
                self.missing.append(f"{module_name}.{attribute}")
                continue
            original = getattr(owner, parts[-1])
            if getattr(original, "__ufwbpp_traced__", False):
                continue
            setattr(owner, parts[-1], self.wrap(original, label, category))
            self.installed.append(label)

    # ---- stages
    def install_stage_hook(self) -> None:
        try:
            from ufwbpp.workflows import single_target as e2e
        except Exception as error:  # pragma: no cover
            self.missing.append(f"stage hook: {error}")
            return
        tracer = self
        original = e2e.ProgressEvent.serializable

        @functools.wraps(original)
        def serializable(self_event: Any) -> dict[str, Any]:
            value = original(self_event)
            # Project-level events carry their own stage and are recorded by
            # the project hook below, once.
            if getattr(self_event, "project_stage", None) is None:
                tracer.record_stage(value)
            return value

        e2e.ProgressEvent.serializable = serializable  # type: ignore[method-assign]
        try:
            from ufwbpp.workflows import project as project_e2e

            project_original = project_e2e.ProjectProgressEvent.serializable

            @functools.wraps(project_original)
            def project_serializable(self_event: Any) -> dict[str, Any]:
                value = project_original(self_event)
                if self_event.project_stage is not None:
                    tracer.record_stage(value)
                return value

            project_e2e.ProjectProgressEvent.serializable = project_serializable  # type: ignore[method-assign]
        except Exception as error:  # pragma: no cover
            self.missing.append(f"project stage hook: {error}")

    def record_stage(self, event: dict[str, Any]) -> None:
        now = self.now_us()
        stage = str(event.get("stage"))
        status = str(event.get("status"))
        message = str(event.get("message", ""))
        usage = self_cpu_seconds(), children_cpu_seconds()
        with self.lock:
            self.events.append(
                {
                    "name": f"{stage}: {message}"[:120],
                    "cat": "progress",
                    "ph": "i",
                    "s": "p",
                    "ts": now,
                    "pid": self.pid,
                    "tid": 1,
                    "args": {"status": status, "current": event.get("current"), "total": event.get("total")},
                }
            )
            if status == "started" or (status == "running" and stage not in self.stage_open):
                self.stage_open[stage] = (now, len(self.stage_records))
                self._rusage_at_stage[stage] = usage
            elif status in {"completed", "failed"} and stage in self.stage_open:
                start, _index = self.stage_open.pop(stage)
                cpu0, children0 = self._rusage_at_stage.pop(stage, usage)
                self.stage_records.append(
                    {
                        "stage": stage,
                        "startSeconds": start / 1e6,
                        "wallSeconds": (now - start) / 1e6,
                        "selfCpuSeconds": usage[0] - cpu0,
                        "childrenCpuSeconds": usage[1] - children0,
                        "message": message,
                    }
                )
                self.events.append(
                    {
                        "name": stage,
                        "cat": "stage",
                        "ph": "X",
                        "ts": start,
                        "dur": now - start,
                        "pid": self.pid,
                        "tid": 1,
                        "args": {"message": message},
                    }
                )

    # ---- samples
    def start_sampler(self, interval_seconds: float = 0.25) -> threading.Event:
        stop = threading.Event()
        tracer = self

        def sample() -> None:
            last_wall = time.perf_counter()
            last_cpu = self_cpu_seconds() + children_cpu_seconds()
            while not stop.wait(interval_seconds):
                wall = time.perf_counter()
                cpu = self_cpu_seconds() + children_cpu_seconds()
                utilisation = (cpu - last_cpu) / max(wall - last_wall, 1e-9)
                last_wall, last_cpu = wall, cpu
                rss = current_rss_bytes()
                with tracer.lock:
                    tracer.events.append(
                        {
                            "name": "cpu (cores busy, self+children)",
                            "cat": "sample",
                            "ph": "C",
                            "ts": tracer.now_us(),
                            "pid": tracer.pid,
                            "tid": 0,
                            "args": {"cores": round(utilisation, 2)},
                        }
                    )
                    if rss is not None:
                        tracer.events.append(
                            {
                                "name": "rss (MB)",
                                "cat": "sample",
                                "ph": "C",
                                "ts": tracer.now_us(),
                                "pid": tracer.pid,
                                "tid": 0,
                                "args": {"rss_mb": round(rss / 1e6, 1)},
                            }
                        )

        threading.Thread(target=sample, name="ufwbpp-trace-sampler", daemon=True).start()
        return stop

    # ---- output
    def dump(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        with self.lock:
            events = list(self.events)
        events.insert(
            0,
            {"name": "process_name", "ph": "M", "pid": self.pid, "tid": 0, "args": {"name": f"ultra-fast-wbpp pid {self.pid}"}},
        )
        events.insert(1, {"name": "thread_name", "ph": "M", "pid": self.pid, "tid": 1, "args": {"name": "stages"}})
        events.insert(2, {"name": "thread_name", "ph": "M", "pid": self.pid, "tid": 0, "args": {"name": "samples"}})
        payload = {"traceEvents": events, "displayTimeUnit": "ms", "metadata": metadata or {}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def self_cpu_seconds() -> float:
    if resource is None:
        times = os.times()
        return times.user + times.system
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def children_cpu_seconds() -> float:
    if resource is None:
        times = os.times()
        return times.children_user + times.children_system
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def current_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    if sys.platform == "linux":
        try:
            with open("/proc/self/statm", encoding="ascii") as stream:
                pages = int(stream.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return None
    return None


def peak_rss_bytes() -> int | None:
    if resource is None:
        try:
            import psutil  # type: ignore[import-not-found]

            return int(psutil.Process().memory_info().peak_wset)  # type: ignore[attr-defined]
        except Exception:
            return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


# ---------------------------------------------------------------- summaries
def summarize(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Inclusive/exclusive seconds, calls, mean and max per timer label."""

    table: dict[str, dict[str, float]] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") in {"stage"}:
            continue
        row = table.setdefault(
            event["name"],
            {"inclusiveSeconds": 0.0, "exclusiveSeconds": 0.0, "calls": 0, "maxSeconds": 0.0, "workerProcesses": 0},
        )
        duration = event["dur"] / 1e6
        row["inclusiveSeconds"] += duration
        row["exclusiveSeconds"] += event.get("args", {}).get("exclusive_us", event["dur"]) / 1e6
        row["calls"] += 1
        row["maxSeconds"] = max(row["maxSeconds"], duration)
        if event.get("args", {}).get("worker"):
            row["workerProcesses"] += 1
    for row in table.values():
        row["meanSeconds"] = row["inclusiveSeconds"] / row["calls"] if row["calls"] else 0.0
    return dict(sorted(table.items(), key=lambda item: -item[1]["exclusiveSeconds"]))


def format_table(stages: list[dict[str, Any]], timers: dict[str, dict[str, float]], total_wall: float, limit: int = 40) -> str:
    lines = [f"total wall {total_wall:.2f} s", "", f"{'stage':<22}{'wall s':>9}{'self cpu':>10}{'children':>10}{'cores':>7}"]
    for record in stages:
        cores = (record["selfCpuSeconds"] + record["childrenCpuSeconds"]) / max(record["wallSeconds"], 1e-9)
        lines.append(
            f"{record['stage']:<22}{record['wallSeconds']:>9.2f}{record['selfCpuSeconds']:>10.2f}"
            f"{record['childrenCpuSeconds']:>10.2f}{cores:>7.1f}"
        )
    lines += ["", f"{'timer':<44}{'excl s':>9}{'incl s':>9}{'calls':>7}{'mean s':>8}{'max s':>8}"]
    for name, row in list(timers.items())[:limit]:
        lines.append(
            f"{name:<44}{row['exclusiveSeconds']:>9.2f}{row['inclusiveSeconds']:>9.2f}{int(row['calls']):>7d}"
            f"{row['meanSeconds']:>8.3f}{row['maxSeconds']:>8.2f}"
        )
    return "\n".join(lines)


def collect_worker_events(trace_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("worker-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = payload.get("pid")
        for event in payload.get("events", []):
            event.setdefault("args", {})["worker"] = True
            events.append(event)
        events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": f"worker pid {pid}"}})
    return events


# ---------------------------------------------------------------- entry points
def _install_in_worker() -> None:
    """Called when a ``spawn`` worker re-imports this file as ``__mp_main__``."""

    trace_dir = os.environ.get(TRACE_DIR_VARIABLE)
    epoch = os.environ.get(TRACE_EPOCH_VARIABLE)
    if not trace_dir or not epoch:
        return
    tracer = Tracer(int(epoch))
    tracer.install()

    def flush() -> None:
        with tracer.lock:
            events = list(tracer.events)
        if not events:
            return
        path = Path(trace_dir) / f"worker-{os.getpid()}.json"
        try:
            path.write_text(json.dumps({"pid": os.getpid(), "events": events}, separators=(",", ":")), encoding="utf-8")
        except Exception:
            pass

    # Pool workers leave through ``os._exit`` after multiprocessing's own
    # finalizers, so register there as well as with atexit.
    atexit.register(flush)
    try:
        from multiprocessing import util as multiprocessing_util

        multiprocessing_util.Finalize(None, flush, exitpriority=100)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", required=True, help="Chrome trace-event JSON to write")
    parser.add_argument("--summary", help="summary JSON to write (stages, timers, environment)")
    parser.add_argument("--sample-interval", type=float, default=0.25, help="CPU/RSS sample interval in seconds")
    parser.add_argument("--top", type=int, default=40, help="timers shown in the stderr table")
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="TIMER",
        help=(
            "run this timer's calls under cProfile and print the merged statistics (repeatable); "
            "only one call is profiled at a time, so concurrent calls are sampled"
        ),
    )
    parser.add_argument("--profile-lines", type=int, default=25, help="functions listed per profiled timer")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="engine CLI command and arguments (after --)")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an engine CLI command is required after --")
    if "--progress-json" not in command and command[0] in {"run", "run-project"}:
        command.append("--progress-json")

    trace_path = Path(args.trace).expanduser().resolve()
    trace_dir = trace_path.parent / (trace_path.stem + ".workers")
    trace_dir.mkdir(parents=True, exist_ok=True)
    epoch_ns = time.perf_counter_ns()
    os.environ[TRACE_DIR_VARIABLE] = str(trace_dir)
    os.environ[TRACE_EPOCH_VARIABLE] = str(epoch_ns)

    tracer = Tracer(epoch_ns, frozenset(args.profile))
    tracer.install()
    tracer.install_stage_hook()
    stop_sampler = tracer.start_sampler(args.sample_interval)

    from ufwbpp import cli
    from ufwbpp.native_kernels import describe_native_kernels
    from ufwbpp.hardware import detect_hardware

    started = time.perf_counter()
    code = 1
    failure: BaseException | None = None
    try:
        code = cli.main(command)
    except SystemExit as exit_request:
        code = exit_request.code if isinstance(exit_request.code, int) else (0 if exit_request.code is None else 1)
    except BaseException as error:  # the trace is written even when the run crashes
        failure = error
    total_wall = time.perf_counter() - started
    stop_sampler.set()

    worker_events = collect_worker_events(trace_dir)
    with tracer.lock:
        tracer.events.extend(worker_events)
    timers = summarize(tracer.events)
    metadata = {
        "command": command,
        "exitCode": code,
        "failure": repr(failure) if failure is not None else None,
        "totalWallSeconds": round(total_wall, 3),
        "selfCpuSeconds": round(self_cpu_seconds(), 3),
        "childrenCpuSeconds": round(children_cpu_seconds(), 3),
        "peakRssBytes": peak_rss_bytes(),
        "installedTimers": tracer.installed,
        "missingTimers": tracer.missing,
        "workerTraceFiles": len(list(trace_dir.glob("worker-*.json"))),
        "nativeKernels": describe_native_kernels(),
        "hardware": detect_hardware().serializable(),
    }
    tracer.dump(trace_path, metadata)
    summary = {"metadata": metadata, "stages": tracer.stage_records, "timers": timers}
    if args.summary:
        summary_path = Path(args.summary).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sys.stderr.write("\n" + format_table(tracer.stage_records, timers, total_wall, args.top) + "\n")
    sys.stderr.write(
        f"\ntrace: {trace_path}  (open in https://ui.perfetto.dev)\n"
        f"peak rss {(metadata['peakRssBytes'] or 0) / 1e9:.2f} GB; self cpu {metadata['selfCpuSeconds']:.1f} s; "
        f"children cpu {metadata['childrenCpuSeconds']:.1f} s; worker trace files {metadata['workerTraceFiles']}\n"
    )
    if tracer.missing:
        sys.stderr.write("timers not installed: " + ", ".join(tracer.missing) + "\n")
    if tracer.profiles:
        sys.stderr.write(tracer.profile_report(args.profile_lines) + "\n")
    if failure is not None:
        raise failure
    return code


if __name__ == "__main__":
    sys.exit(main())
elif __name__ == "__mp_main__":
    _install_in_worker()
