"""Calibrating and registering every Light in one fused, memory-bounded parallel pass."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from lightframeqc.cfa import CHANNEL_NAMES, bilinear_debayer

from ..calibration.inputs import (
    assert_compatible,
    numeric_application_scale,
    numeric_domain_metadata,
    find_dark,
)
from ..path_budget import light_stem
from ..platform import remove_file
from .integration import (
    CalibrationError,
    FitsFloatWriter,
    FitsFrame,
    FrameExpression,
    FrameInfo,
    PixelStatistics,
    MemoryFrame,
    _StatsAccumulator,
    atomic_publish_file,
    _canonical_expression,
    _expression_rows,
    temporary_output,
    _validate_expression_shapes,
)
from .masters import _CalibrationMasters
from .parameters import OUTPUT_STATE, PipelineParameters, PixelTransform
from .records import _safe_token
from .run_plan import _RunLedger, _RunPlan, _StagingDirs
from .warp import (
    _register_frame,
    _registration_bytes_per_pixel,
    _registration_provenance,
    NATIVE_WARP_BYTES_PER_PIXEL,
)


# Fused calibrate+register working set per Light: the Float32 result, one
# master temporary during subtraction/division, and masks/temporaries.
FUSED_LIGHT_BYTES_PER_PIXEL = 12


@dataclass(frozen=True, slots=True)
class _ChannelDestination:
    """One registered output of a Light: its group and, for a Bayer Light,
    the colour channel (0 R, 1 G, 2 B) debayered before the warp."""

    group: str
    channel: int | None
    path: Path


@dataclass(frozen=True, slots=True)
class _LightJob:
    """One Light's fused calibrate-in-memory then register work item."""

    source_path: Path
    expression: FrameExpression
    calibrated_path: Path | None
    calibrated_metadata: Mapping[str, Any]
    destinations: tuple[_ChannelDestination, ...]
    transform: PixelTransform
    info: FrameInfo
    source_exposure_seconds: float | None
    # Master dark whose hot-pixel map drives the cosmetic correction, and the
    # detection threshold; ``None`` leaves the calibrated pixels untouched.
    hot_pixel_master: str | None = None
    hot_pixel_sigma: float | None = None
    # Bayer pattern of the Light (None for mono): the calibrated mosaic is
    # debayered into the channels the destinations ask for.
    cfa_pattern: str | None = None

    @property
    def channel_count(self) -> int:
        return 3 if self.cfa_pattern is not None else 1


@dataclass(frozen=True, slots=True)
class _RegisteredOutput:
    group: str
    channel: int | None
    path: Path
    statistics: PixelStatistics
    sha256: str | None
    execution: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _LightJobResult:
    calibrated_statistics: PixelStatistics
    calibrated_sha256: str | None
    registered: tuple[_RegisteredOutput, ...]
    cosmetic: dict[str, Any] | None = None
    debayer: dict[str, Any] | None = None

    @property
    def execution(self) -> dict[str, Any]:
        return self.registered[0].execution if self.registered else {}


class _MasterCache:
    """Decoded Float32 masters shared read-only by every fused worker.

    Each master is converted from its FITS storage exactly once; workers
    receive copies of the rows they ask for, so the cache is never mutated.
    """

    def __init__(self) -> None:
        self._frames: dict[str, MemoryFrame] = {}
        self._hot_pixels: dict[tuple[str, float], tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def frame(self, path: str) -> MemoryFrame:
        key = str(Path(path).expanduser().resolve(strict=True))
        with self._lock:
            cached = self._frames.get(key)
            if cached is None:
                with FitsFrame(key) as source:
                    cached = MemoryFrame(source.full_values(), source.info, key)
                self._frames[key] = cached
            return cached

    @property
    def decoded_bytes(self) -> int:
        with self._lock:
            return sum(frame.values.nbytes for frame in self._frames.values())

    def hot_pixels(self, path: str, sigma: float) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
        """Row/column indices of the master dark's hot pixels (cached per dark)."""

        master = self.frame(path)
        key = (str(master.path), float(sigma))
        with self._lock:
            cached = self._hot_pixels.get(key)
        if cached is None:
            cached = _hot_pixel_map(master.values, sigma)
            with self._lock:
                self._hot_pixels[key] = cached
        return cached


def _hot_pixel_map(
    values: NDArray[np.float32], sigma: float
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), {"count": 0, "threshold": None}
    median = float(np.median(finite))
    dispersion = 1.4826 * float(np.median(np.abs(finite - median)))
    if not math.isfinite(dispersion) or dispersion <= 0.0:
        # A dark without measurable dispersion (synthetic or degenerate)
        # carries no hot-pixel evidence.
        return np.empty(0, np.int64), np.empty(0, np.int64), {
            "count": 0, "fraction": 0.0, "darkMedian": median, "darkRobustSigma": dispersion,
            "threshold": None, "sigma": float(sigma),
        }
    threshold = float(median + sigma * dispersion)
    rows, columns = np.nonzero(values > np.float32(threshold))
    return rows.astype(np.int64), columns.astype(np.int64), {
        "count": int(rows.size),
        "fraction": float(rows.size / values.size),
        "darkMedian": median,
        "darkRobustSigma": dispersion,
        "threshold": threshold,
        "sigma": float(sigma),
    }


def _replace_hot_pixels(
    image: NDArray[np.float32],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    *,
    cfa: bool = False,
) -> None:
    """Replace the listed pixels in place by the median of their eight neighbours.

    On a Bayer mosaic the neighbours are the eight same-colour pixels two
    steps away, so a hot pixel never takes on another colour's value.
    """

    if rows.size == 0:
        return
    height, width = image.shape
    step = 2 if cfa else 1
    neighbours = np.empty((8, rows.size), dtype=np.float32)
    index = 0
    for dy in (-step, 0, step):
        for dx in (-step, 0, step):
            if dy == 0 and dx == 0:
                continue
            neighbours[index] = image[
                np.clip(rows + dy, 0, height - 1), np.clip(columns + dx, 0, width - 1)
            ]
            index += 1
    image[rows, columns] = np.nanmedian(neighbours, axis=0)


def _write_float_fits(
    values: NDArray[np.float32],
    destination: Path,
    metadata: Mapping[str, Any],
    *,
    durable: bool = True,
) -> str | None:
    """Publish one in-memory Float32 image atomically and return its digest."""

    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output", path=str(destination)
        )
    temporary = temporary_output(destination)
    try:
        with FitsFloatWriter(temporary, values.shape, metadata, durable=durable) as writer:
            writer.write_rows(0, values)
        atomic_publish_file(temporary, destination)
        return writer.sha256
    finally:
        remove_file(temporary)


def _process_light_job(
    job: _LightJob,
    *,
    master_cache: _MasterCache,
    max_memory_bytes: int,
    resampler: str,
    native_threads: int,
    division_floor: float,
    durable: bool = True,
) -> _LightJobResult:
    expression = _canonical_expression(job.expression)
    sources: dict[str, Any] = {}
    with FitsFrame(expression.source_path) as light:
        sources[expression.source_path] = light
        for master_path in (
            expression.subtract_path,
            *expression.subtract_paths,
            expression.divide_path,
        ):
            if master_path is not None:
                sources[master_path] = master_cache.frame(master_path)
        height, _width = _validate_expression_shapes((expression,), sources)
        # The Light's decoded Float32 buffer becomes the calibrated image in
        # place: the same arithmetic as write_expression, without a disk trip.
        calibrated = _expression_rows(
            expression, sources, 0, height, division_floor=division_floor
        )
    cosmetic: dict[str, Any] | None = None
    if job.hot_pixel_master is not None and job.hot_pixel_sigma is not None:
        rows, columns, evidence = master_cache.hot_pixels(job.hot_pixel_master, job.hot_pixel_sigma)
        _replace_hot_pixels(calibrated, rows, columns, cfa=job.cfa_pattern is not None)
        cosmetic = {
            "algorithm": (
                "master-dark-hot-pixel-same-colour-neighbour-median-v1"
                if job.cfa_pattern is not None
                else "master-dark-hot-pixel-neighbour-median-v1"
            ),
            "replacedPixels": int(rows.size),
            **evidence,
        }
    statistics = _StatsAccumulator()
    statistics.update(calibrated)
    calibrated_statistics = statistics.result()
    if calibrated_statistics.finite_pixels == 0:
        raise CalibrationError(
            "NO_FINITE_OUTPUT", "calibration produced no finite pixels",
            path=str(job.source_path),
        )
    calibrated_sha256: str | None = None
    if job.calibrated_path is not None:
        calibrated_sha256 = _write_float_fits(
            calibrated, job.calibrated_path, job.calibrated_metadata, durable=durable
        )
    # A Bayer Light is debayered once; each colour plane is then registered
    # like a mono Light of its own filter group.  The mosaic itself stays the
    # materialized calibrated frame (the drizzle drops its real samples).
    planes: NDArray[np.float32] | None = None
    debayer: dict[str, Any] | None = None
    if job.cfa_pattern is not None:
        debayer_started = time.perf_counter()
        planes = bilinear_debayer(calibrated, job.cfa_pattern)
        debayer = {
            "algorithm": "bilinear-same-colour-neighbours-v1",
            "pattern": job.cfa_pattern,
            "seconds": round(time.perf_counter() - debayer_started, 3),
        }
    occupied = calibrated.nbytes + (planes.nbytes if planes is not None else 0)
    # The calibrated image already occupies its share; leave the rest of the
    # worker budget to warp tiles, but always allow at least one NumPy row.
    warp_budget = max(
        max_memory_bytes - occupied,
        calibrated.shape[1]
        * _registration_bytes_per_pixel(job.transform, resampler, calibrated.shape),
    )
    registered: list[_RegisteredOutput] = []
    for destination in job.destinations:
        if destination.channel is None:
            source_values = calibrated
        else:
            if planes is None:
                raise CalibrationError(
                    "CFA_CHANNEL_WITHOUT_PATTERN",
                    "a colour channel destination needs the Light's Bayer pattern",
                    path=str(job.source_path),
                )
            source_values = planes[destination.channel]
        memory_frame = MemoryFrame(
            source_values, job.info, job.calibrated_path or job.source_path
        )
        execution: dict[str, Any] = {}
        registered_statistics = _register_frame(
            memory_frame,
            destination.path,
            job.transform,
            job.info,
            max_memory_bytes=warp_budget,
            resampler=resampler,
            source_exposure_seconds=job.source_exposure_seconds,
            native_threads=native_threads,
            execution=execution,
            durable=durable,
        )
        registered.append(
            _RegisteredOutput(
                group=destination.group,
                channel=destination.channel,
                path=destination.path,
                statistics=registered_statistics,
                sha256=execution.pop("sha256", None),
                execution=execution,
            )
        )
    return _LightJobResult(
        calibrated_statistics=calibrated_statistics,
        calibrated_sha256=calibrated_sha256,
        registered=tuple(registered),
        cosmetic=cosmetic,
        debayer=debayer,
    )


def _fused_job_bytes(job: _LightJob, resampler: str) -> int:
    height, width = job.info.shape
    per_row = width * max(
        NATIVE_WARP_BYTES_PER_PIXEL,
        _registration_bytes_per_pixel(job.transform, resampler, job.info.shape),
    )
    # A Bayer Light also holds its three debayered planes while it is warped.
    planes = 3 * height * width * 4 if job.cfa_pattern is not None else 0
    return height * width * FUSED_LIGHT_BYTES_PER_PIXEL + planes + per_row


def _fused_worker_count(
    jobs: Sequence[_LightJob],
    *,
    max_memory_bytes: int,
    resampler: str,
    cpu_workers: int,
) -> int:
    largest = max((_fused_job_bytes(job, resampler) for job in jobs), default=1)
    return max(1, min(cpu_workers, len(jobs), max_memory_bytes // max(1, largest)))


def _calibrate_and_register_frames(
    jobs: Sequence[_LightJob],
    *,
    master_cache: _MasterCache,
    max_memory_bytes: int,
    resampler: str,
    cpu_workers: int,
    division_floor: float,
    durable: bool = True,
    kernel_threads: int | None = None,
) -> tuple[tuple[_LightJobResult, ...], dict[str, Any]]:
    """Calibrate and register every Light with one shared memory budget.

    Lights run concurrently in ``workers`` threads; each thread hands its warp
    to the native kernel with the remaining CPU share, so all cores stay busy
    whether memory allows many Lights in flight or only one.  ``kernel_threads``
    is the native thread budget shared by the in-flight warps (defaults to
    ``cpu_workers``, the pre-tuning-table behaviour).
    """

    workers = _fused_worker_count(
        jobs,
        max_memory_bytes=max_memory_bytes,
        resampler=resampler,
        cpu_workers=cpu_workers,
    )
    thread_budget = cpu_workers if kernel_threads is None else max(1, int(kernel_threads))
    native_threads = max(1, thread_budget // workers)
    worker_memory_bytes = max_memory_bytes // workers
    # Lights start in submission order, so the last ``len(jobs) % workers``
    # Lights run while the other workers are already idle; their warps take
    # the CPU share those workers would have used. Warp results do not depend
    # on the thread count.
    rounds = max(1, math.ceil(len(jobs) / workers))
    tail_start = (rounds - 1) * workers
    tail_threads = max(native_threads, thread_budget // max(1, len(jobs) - tail_start))

    def run(index: int, job: _LightJob) -> _LightJobResult:
        return _process_light_job(
            job,
            master_cache=master_cache,
            max_memory_bytes=worker_memory_bytes,
            resampler=resampler,
            native_threads=tail_threads if index >= tail_start else native_threads,
            division_floor=division_floor,
            durable=durable,
        )

    if workers == 1:
        results = tuple(run(index, job) for index, job in enumerate(jobs))
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wbpp-light")
        try:
            futures = [executor.submit(run, index, job) for index, job in enumerate(jobs)]
            results = tuple(future.result() for future in futures)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    backends: dict[str, int] = {}
    for result in results:
        backend = str(result.execution.get("warpBackend", "unknown"))
        backends[backend] = backends.get(backend, 0) + 1
    return results, {
        "executor": "thread-pool" if workers > 1 else "serial",
        "executionModel": "fused-calibrate-warp-v1",
        "cpuWorkersUsed": workers,
        "nativeThreadsPerWorker": native_threads,
        "tailNativeThreads": tail_threads,
        "tailLights": len(jobs) - tail_start,
        "perWorkerMemoryBudgetBytes": worker_memory_bytes,
        "warpBackends": backends,
        "masterCacheBytes": master_cache.decoded_bytes,
    }


def _plan_light_jobs(
    plan: _RunPlan,
    dirs: _StagingDirs,
    parameters: PipelineParameters,
    masters: _CalibrationMasters,
) -> tuple[list[_LightJob], list[dict[str, Any]]]:
    """One calibrate-and-register job per Light, and the calibration
    details its receipt entry reports."""

    workflow = parameters.calibration_workflow
    reference_bias = plan.reference_bias
    groups_of_filter: dict[str, list[str]] = {}
    for group_name, source_filter in plan.group_filter.items():
        groups_of_filter.setdefault(source_filter, []).append(group_name)
    light_jobs: list[_LightJob] = []
    calibrated_details: list[dict[str, Any]] = []
    for index, path in enumerate(plan.lights, start=1):
        info = plan.light_info[path]
        filter_name = info.filter_name
        flat_path = masters.flats[filter_name]
        flat_reference = (
            plan.flat_info[plan.flat_groups[filter_name][0]]
            if filter_name in plan.flat_groups
            else plan.master_flat_info[plan.supplied_flats[filter_name]]
        )
        assert_compatible(info, flat_reference, compare_filter=True, workflow=workflow)
        dark_match = find_dark(info.exposure_seconds, masters.darks)
        if dark_match is not None:
            dark_exposure, subtract_path = dark_match
            assert_compatible(
                info,
                plan.dark_reference(dark_exposure),
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
                workflow=workflow,
            )
            dark_bias_included = plan.dark_bias_included(dark_exposure, subtract_path)
            bias_mode = (
                "INCLUDED_IN_MASTER_DARK"
                if dark_bias_included
                else "MASTER_BIAS_AND_BIAS_SUBTRACTED_DARK"
            )
            subtract_info = masters.dark_domain_info[dark_exposure]
        else:
            subtract_path = masters.bias
            dark_bias_included = True
            bias_mode = "MASTER_BIAS_SUBTRACTED"
            subtract_info = reference_bias
        subtract_scale = numeric_application_scale(
            info,
            subtract_info,
            target_label="raw Light",
            additive_label=("MasterDark" if dark_match is not None else "MasterBias"),
        )
        bias_scale = numeric_application_scale(
            info,
            reference_bias,
            target_label="raw Light",
            additive_label="MasterBias",
        )
        light_output_domain = plan.light_domain_references[filter_name]
        light_domain_scale = numeric_application_scale(
            light_output_domain,
            info,
            target_label="filter integration domain",
            additive_label="raw Light",
        )
        reference_exposure = plan.reference_exposures[filter_name]
        exposure_scale = reference_exposure / float(info.exposure_seconds)
        stem = light_stem(path)
        cfa_pattern = plan.light_cfa_pattern[filter_name]
        destinations = tuple(
            _ChannelDestination(
                group=group_name,
                channel=plan.group_channel[group_name],
                path=(
                    dirs.registered / f"{index:05d}_{stem}.fits"
                    if plan.group_channel[group_name] is None
                    else dirs.registered / f"{index:05d}_{stem}_{_safe_token(group_name)}.fits"
                ),
            )
            for group_name in groups_of_filter[filter_name]
        )
        expression = FrameExpression(
            source_path=str(path),
            subtract_path=str(subtract_path),
            subtract_scale=subtract_scale,
            subtract_paths=(str(masters.bias),) if not dark_bias_included else (),
            subtract_scales=(bias_scale,) if not dark_bias_included else (),
            divide_path=str(flat_path),
            scale=(
                masters.flat_application_scales[filter_name]
                * reference_exposure
                / float(info.exposure_seconds)
                * light_domain_scale
            ),
            pattern_scales=masters.flat_pattern_scales.get(filter_name, ()),
        )
        calibrated_metadata = {
            "IMAGETYP": "Calibrated Light",
            "FILTER": filter_name,
            "OBJECT": info.target,
            "EXPTIME": reference_exposure,
            "OAFSRCEX": info.exposure_seconds,
            "OAFEXPSC": exposure_scale,
            "OAFSTATE": OUTPUT_STATE,
            "OAFBIAS": bias_mode,
            **({"BAYERPAT": cfa_pattern, "OAFCFA": cfa_pattern} if cfa_pattern else {}),
            **numeric_domain_metadata(light_output_domain),
        }
        calibrated_details.append(
            {
                "source": str(plan.display_path(path)),
                "filter": filter_name,
                "subtractedMaster": plan.receipt_reference(dirs.root, subtract_path),
                "biasMode": bias_mode,
                "sourceNumericDomain": info.numeric_domain,
                "additiveNumericDomain": subtract_info.numeric_domain,
                "additiveApplicationScale": subtract_scale,
                "additiveApplicationScaleSource": "normalized-unit-domain-ratio",
                "biasApplicationScale": bias_scale if not dark_bias_included else None,
                "outputNumericDomain": light_output_domain.numeric_domain,
                "sourceToOutputDomainScale": light_domain_scale,
                "dividedMasterFlat": plan.receipt_reference(dirs.root, flat_path),
                "flatApplicationNormalization": masters.flat_application_scales[filter_name],
                **(
                    {
                        "cfaPattern": cfa_pattern,
                        "cfaFlatChannelMedians": list(masters.flat_channel_medians[filter_name]),
                        "cfaFlatPatternScales": list(masters.flat_pattern_scales[filter_name]),
                    }
                    if cfa_pattern
                    else {}
                ),
                "exposureNormalization": {
                    "sourceSeconds": info.exposure_seconds,
                    "referenceSeconds": reference_exposure,
                    "scale": exposure_scale,
                },
            }
        )
        light_jobs.append(
            _LightJob(
                source_path=path,
                expression=expression,
                calibrated_path=(
                    dirs.calibrated / f"{index:05d}_{stem}.fits"
                    if parameters.materialize_calibrated_lights
                    else None
                ),
                calibrated_metadata=calibrated_metadata,
                destinations=destinations,
                transform=plan.transforms[path],
                info=replace(
                    info,
                    exposure_seconds=reference_exposure,
                    numeric_domain=light_output_domain.numeric_domain,
                    normalized_unit_scale=light_output_domain.normalized_unit_scale,
                ),
                source_exposure_seconds=info.exposure_seconds,
                hot_pixel_master=(
                    str(subtract_path)
                    if dark_match is not None and parameters.cosmetic_hot_pixel_sigma is not None
                    else None
                ),
                hot_pixel_sigma=parameters.cosmetic_hot_pixel_sigma,
                cfa_pattern=cfa_pattern,
            )
        )
    return light_jobs, calibrated_details


@dataclass(frozen=True)
class _RegisteredLights:
    by_group: dict[tuple[Path, str], Path]
    calibrated: dict[Path, Path]
    records: dict[str, Any]
    execution: dict[str, Any]
    wall_seconds: float


def _calibrate_and_register_lights(
    plan: _RunPlan,
    dirs: _StagingDirs,
    jobs: Sequence[_LightJob],
    calibrated_details: Sequence[dict[str, Any]],
    parameters: PipelineParameters,
    execution_tuning: Any,
    ledger: _RunLedger,
) -> _RegisteredLights:
    """Calibrate in memory and warp within one shared registration budget.
    Source-identity caches and receipt construction stay on this thread."""

    master_cache = _MasterCache()
    started = time.perf_counter()
    light_results, fused_execution = _calibrate_and_register_frames(
        jobs,
        master_cache=master_cache,
        max_memory_bytes=parameters.registration_memory_bytes,
        resampler=parameters.registration_resampler,
        cpu_workers=execution_tuning.cpu_workers,
        kernel_threads=execution_tuning.kernel_threads,
        division_floor=parameters.integration.division_floor,
        durable=parameters.durable_intermediates,
    )
    wall_seconds = time.perf_counter() - started
    del master_cache
    registered: dict[tuple[Path, str], Path] = {}
    calibrated: dict[Path, Path] = {}
    records: dict[str, Any] = {}
    for path, job, details, result in zip(
        plan.lights, jobs, calibrated_details, light_results, strict=True,
    ):
        transform = job.transform
        resampling = _registration_provenance(
            transform, job.info.shape, parameters.registration_resampler
        )
        if job.calibrated_path is not None:
            calibrated[path] = job.calibrated_path
            ledger.record(
                job.calibrated_path,
                "CALIBRATED_LIGHT",
                statistics=result.calibrated_statistics,
                details=details,
                sha256=result.calibrated_sha256,
            )
        for registered_output in result.registered:
            registered[(path, registered_output.group)] = registered_output.path
            ledger.record(
                registered_output.path,
                "REGISTERED_LIGHT",
                statistics=registered_output.statistics,
                details={
                    "source": str(plan.display_path(path)),
                    "group": registered_output.group,
                    **(
                        {"cfaChannel": CHANNEL_NAMES[registered_output.channel], "cfaPattern": job.cfa_pattern}
                        if registered_output.channel is not None
                        else {}
                    ),
                    "transformInputToOutput": transform.serializable(),
                    **resampling,
                    "warpBackend": registered_output.execution.get("warpBackend"),
                    "warpKernel": registered_output.execution.get("warpKernel"),
                },
                sha256=registered_output.sha256,
            )
        records[str(plan.display_path(path))] = {
            "transformInputToOutput": transform.serializable(),
            "identity": transform.is_identity,
            **resampling,
            "qualityWeight": plan.quality_weights[path],
            "calibration": {
                "materialized": job.calibrated_path is not None,
                "statistics": result.calibrated_statistics.serializable(),
                "cosmetic": result.cosmetic or {"algorithm": None, "replacedPixels": 0},
                **({"debayer": result.debayer} if result.debayer else {}),
                **details,
            },
            "outputs": {
                registered_output.group: {
                    "path": str(registered_output.path.relative_to(dirs.root)),
                    "channel": CHANNEL_NAMES[registered_output.channel] if registered_output.channel is not None else None,
                    "statistics": registered_output.statistics.serializable(),
                }
                for registered_output in result.registered
            },
            "execution": dict(result.execution),
        }
    if not parameters.materialize_calibrated_lights:
        try:
            dirs.calibrated.rmdir()
        except OSError:
            # Removing this unused staging directory is best effort.  Keep it
            # for diagnostics if cleanup fails; artifact and final-publication
            # validation still run independently.
            pass
    execution = {
        "executor": fused_execution["executor"],
        "executionModel": fused_execution["executionModel"],
        "configuredCpuWorkers": execution_tuning.cpu_workers,
        "cpuWorkersUsed": fused_execution["cpuWorkersUsed"],
        "nativeThreadsPerWorker": fused_execution["nativeThreadsPerWorker"],
        "tailNativeThreads": fused_execution["tailNativeThreads"],
        "tailLights": fused_execution["tailLights"],
        "frameCount": len(jobs),
        "totalMemoryBudgetBytes": parameters.registration_memory_bytes,
        "perWorkerMemoryBudgetBytes": fused_execution["perWorkerMemoryBudgetBytes"],
        "warpBackends": fused_execution["warpBackends"],
        "calibratedLightsMaterialized": parameters.materialize_calibrated_lights,
        "masterCacheBytes": fused_execution["masterCacheBytes"],
        "wallSeconds": wall_seconds,
    }
    return _RegisteredLights(registered, calibrated, records, execution, wall_seconds)
