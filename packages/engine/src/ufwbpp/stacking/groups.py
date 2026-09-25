"""Normalizing, rejecting and integrating the registered Lights of one output group."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any, Mapping, Sequence, Callable

import numpy as np

from lightframeqc.cfa import CHANNEL_NAMES

from ..calibration.inputs import numeric_domain_metadata
from ..native_kernels import load_native_kernels
from ..platform import remove_file
from ..products.preview import render_auto_stretch_preview
from .crop import _crop_fits
from .drizzle_native import DrizzleFrame, DrizzleGroupInputs
from .integration import CalibrationError, FrameExpression, IntegrationMapPaths
from .lights import _RegisteredLights
from .metal_integration import MetalIntegrationError, NativeMetalExecutor, integrate_registered_group
from .normalization import StellarScaleHint, fit_registered_group_global_normalization
from .parameters import OUTPUT_STATE, PipelineParameters
from .proper_coaddition import PROPER_COADD_ALGORITHM_ID, proper_coadd_group
from .records import _integration_record, _safe_token
from .run_plan import _RunLedger, _RunPlan, _StagingDirs


class _RejectionMaskRecorder:
    """Tile observer that keeps every frame's accepted-sample mask as packed
    row bits on the reference grid, the form the drizzle stage reads."""

    def __init__(self, frame_count: int, shape: tuple[int, int]) -> None:
        height, width = shape
        self.bits = [
            np.zeros((height, (width + 7) // 8), dtype=np.uint8) for _ in range(frame_count)
        ]
        self._rows_seen = np.zeros(height, dtype=bool)

    def __call__(self, observation: Any) -> None:
        accepted = np.asarray(observation.accepted, dtype=bool)
        first_row = int(observation.first_row)
        rows = accepted.shape[1]
        for index, frame_bits in enumerate(self.bits):
            frame_bits[first_row : first_row + rows] = np.packbits(accepted[index], axis=1)
        self._rows_seen[first_row : first_row + rows] = True

    @property
    def complete(self) -> bool:
        """Every row was observed; an unobserved row would read as all rejected."""

        return bool(np.all(self._rows_seen))


def _compose_tile_observers(*observers: Any) -> Callable[[Any], None] | None:
    active = [observer for observer in observers if observer is not None]
    if not active:
        return None

    def observe(observation: Any) -> None:
        for observer in active:
            observer(observation)

    return observe


def _normalization_reference_index(
    paths: Sequence[Path],
    hints: Mapping[Path, StellarScaleHint | None],
    quality_weights: Mapping[Path, float],
) -> tuple[int, dict[str, Any]]:
    """Normalization reference of one filter group.

    Registration chose the group's stellar-scale reference (the lowest-sky
    frame of acceptable quality) and bound every hint to it; the same frame
    is the additive reference so the master inherits its background.  Without
    hints the highest-quality frame is used, as before.
    """

    hinted = {
        Path(hint.reference_path).expanduser().resolve(strict=True)
        for path in paths
        if (hint := hints.get(path)) is not None
    }
    if len(hinted) == 1:
        reference = next(iter(hinted))
        for index, path in enumerate(paths):
            if path == reference:
                return index, {"rule": "stellar-scale-hint-reference", "reference": str(path)}
    index = max(range(len(paths)), key=lambda item: quality_weights[paths[item]])
    return index, {"rule": "highest-quality-weight", "reference": str(paths[index])}


class _MetalSession:
    """The run's Metal executor, dropped for the rest of the run once Metal
    rejects a group (the CPU reference then integrates the remaining ones)."""

    def __init__(self) -> None:
        self.executor: NativeMetalExecutor | None = None
        self.unavailable_reason: str | None = None

    def open(self, parameters: PipelineParameters) -> None:
        try:
            self.executor = NativeMetalExecutor(
                library_path=parameters.native_library_path,
                metal_source_path=parameters.metal_source_path,
            )
        except MetalIntegrationError as error:
            self.unavailable_reason = str(error)

    def observe(self, execution: Mapping[str, Any]) -> None:
        fallback_reason = str(execution.get("fallbackReason") or "")
        if (
            self.executor is not None
            and execution.get("selectedBackend") == "portable-cpu"
            and fallback_reason.startswith("Metal execution rejected:")
        ):
            self.executor.close()
            self.executor = None
            self.unavailable_reason = fallback_reason

    def close(self) -> None:
        if self.executor is not None:
            self.executor.close()


class _NormalizationFits:
    """Global-normalization fits of the output groups.

    The fit of a group is a pure function of its registered frames, hints and
    transforms, so the next group's fit runs on one helper thread while this
    group integrates: the fit is mostly Python-level work whose gaps and the
    integration's I/O and Python phases overlap.  The coefficients are
    identical either way.
    """

    def __init__(
        self,
        plan: _RunPlan,
        registered: Mapping[tuple[Path, str], Path],
        parameters: PipelineParameters,
        execution_tuning: Any,
        ordered_groups: Sequence[tuple[str, list[Path]]],
        *,
        prefetch: bool = True,
    ) -> None:
        self._plan = plan
        self._registered = registered
        self._parameters = parameters
        self._cpu_workers = execution_tuning.cpu_workers
        # A fit that overlaps another group's integration gets a third of the
        # cores: the integration's kernels keep the rest, and the fit's
        # Python-level work does not scale past a few threads anyway.
        self._prefetch_workers = max(2, execution_tuning.cpu_workers // 3)
        self._ordered_groups = ordered_groups
        self._prefetch = prefetch and parameters.global_normalization.enabled and len(ordered_groups) > 1
        self._pool: ThreadPoolExecutor | None = None
        self._prefetched: dict[str, Any] = {}

    def _job(self, group_name: str, group_paths: list[Path], fit_workers: int) -> Callable[[], Any]:
        plan = self._plan
        group_registered = [self._registered[(path, group_name)] for path in group_paths]
        group_reference_index, _selection = _normalization_reference_index(
            group_paths, plan.stellar_scale_hints, plan.quality_weights
        )
        hints: list[StellarScaleHint | None] = []
        expected = group_paths[group_reference_index]
        registered_reference_path = group_registered[group_reference_index]
        for source_path, registered_path in zip(group_paths, group_registered, strict=True):
            hint = plan.stellar_scale_hints[source_path]
            if hint is None:
                hints.append(None)
                continue
            hinted_reference = Path(hint.reference_path).expanduser().resolve(strict=True)
            if hinted_reference != expected:
                raise CalibrationError(
                    "STELLAR_SCALE_HINT_REFERENCE_MISMATCH",
                    "stellar scale reference differs from the integration-quality reference",
                    path=str(source_path),
                )
            hints.append(
                replace(hint, source_path=str(registered_path), reference_path=str(registered_reference_path))
            )
        transforms = [plan.transforms[path].validated_matrix() for path in group_paths]
        parameters = self._parameters.global_normalization

        def run() -> tuple[Any, list[StellarScaleHint | None], float]:
            started = time.perf_counter()
            result = fit_registered_group_global_normalization(
                [str(path) for path in group_registered],
                reference_index=group_reference_index,
                parameters=parameters,
                stellar_scale_hints=hints,
                workers=fit_workers,
                transforms=transforms,
            )
            return result, hints, time.perf_counter() - started

        return run

    def prefetch_after(self, position: int) -> None:
        """Start the fit of the group after ``position`` on the helper thread."""

        if not self._prefetch or position + 1 >= len(self._ordered_groups):
            return
        next_name, next_paths = self._ordered_groups[position + 1]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ufwbpp-normalize-next")
        self._prefetched[next_name] = self._pool.submit(
            self._job(next_name, next_paths, self._prefetch_workers)
        )

    def prefetch_all(self, order: Sequence[int], *, concurrent: int) -> None:
        """Start the fit of every group, in ``order``, on ``concurrent``
        helper threads, so each group's integration waits only for its own
        fit while the fits of later groups overlap earlier integrations."""

        if not self._parameters.global_normalization.enabled:
            return
        self._pool = ThreadPoolExecutor(max_workers=concurrent, thread_name_prefix="ufwbpp-normalize")
        workers = max(2, self._cpu_workers // concurrent)
        for position in order:
            name, paths = self._ordered_groups[position]
            self._prefetched[name] = self._pool.submit(self._job(name, paths, workers))

    def fit(self, group_name: str, paths: list[Path]) -> tuple[Any, list[StellarScaleHint | None], float, bool]:
        """The fit of ``group_name``: its result, hints, fit seconds and
        whether it was prefetched."""

        future = self._prefetched.pop(group_name, None)
        if future is not None:
            return (*future.result(), True)
        return (*self._job(group_name, list(paths), self._cpu_workers)(), False)

    def shutdown(self, *, cancel: bool = False) -> None:
        if self._pool is not None:
            # A fit still running for a later group must finish before the
            # staging tree it reads is removed.
            if cancel:
                self._pool.shutdown(wait=True, cancel_futures=True)
            else:
                self._pool.shutdown(wait=True)


@dataclass(frozen=True)
class _GroupProducts:
    master_light: Path
    preview: Path
    drizzle: DrizzleGroupInputs | None
    record: dict[str, Any]
    timing: dict[str, float]
    proper_coadd: Path | None = None


def _public_normalization_evidence(
    plan: _RunPlan, paths: Sequence[Path], reference_index: int, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """The fit's receipt with frames named by their public source paths."""

    public_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(receipt["frames"]):
        public_frame = {**dict(frame), "source": str(plan.display_path(paths[index]))}
        frame_evidence = dict(public_frame["evidence"])
        stellar = frame_evidence.get("stellarScale")
        if isinstance(stellar, dict):
            stellar = dict(stellar)
            stellar["source"] = str(plan.display_path(paths[index]))
            stellar["reference"] = str(plan.display_path(paths[reference_index]))
            frame_evidence["stellarScale"] = stellar
        public_frame["evidence"] = frame_evidence
        public_frames.append(public_frame)
    return {**dict(receipt), "frames": public_frames}


def _with_region_weights(
    plan: _RunPlan, paths: Sequence[Path], expressions: list[FrameExpression]
) -> tuple[list[FrameExpression], list[dict[str, Any]]]:
    """Attach each Light's selection region weight map to its expression."""

    attached: list[FrameExpression] = []
    mapped: list[dict[str, Any]] = []
    for path, expression in zip(paths, expressions, strict=True):
        region_map = plan.region_weight_maps.get(path)
        if region_map is None:
            attached.append(expression)
            continue
        height, width = plan.light_info[path].shape
        x_nodes, y_nodes = region_map.pixel_nodes(height, width)
        attached.append(
            replace(
                expression,
                weight_grid=tuple(tuple(float(value) for value in row) for row in region_map.nodes),
                weight_grid_x=tuple(float(value) for value in x_nodes),
                weight_grid_y=tuple(float(value) for value in y_nodes),
            )
        )
        mapped.append(
            {
                "path": str(plan.display_path(path)),
                "frame": str(region_map.evidence.get("frame", "qc-reference")),
                "zeroFraction": float(region_map.zero_fraction),
                "minimumWeight": float(region_map.minimum_weight),
                "meanWeight": float(region_map.mean_weight),
            }
        )
    return attached, mapped


# Cores per concurrently integrated group.  A group's integration keeps
# about five cores busy on average (serial reads, writes and Python-level
# work between its multithreaded kernels), so on larger machines two groups
# side by side fill the idle cores.
_CORES_PER_CONCURRENT_GROUP = 6


def _group_concurrency(parameters: PipelineParameters, tuning: Any, groups: int) -> int:
    """How many output groups integrate at the same time.

    Only the CPU integration runs groups side by side (the Metal executor is
    one per run); the masters never depend on the count, which only changes
    the schedule.
    """

    backend = parameters.ordinary_integration_backend
    cpu = backend == "portable-cpu" or (backend == "auto" and load_native_kernels() is not None)
    if not cpu or groups < 2:
        return 1
    return max(1, min(groups, int(tuning.cpu_workers) // _CORES_PER_CONCURRENT_GROUP))


def _integrate_group(
    plan: _RunPlan,
    dirs: _StagingDirs,
    group_name: str,
    paths: list[Path],
    *,
    parameters: PipelineParameters,
    lights: _RegisteredLights,
    fits: _NormalizationFits,
    metal: _MetalSession,
    hardware_profile: Any,
    execution_tuning: Any,
    shared_crop: tuple[int, int, int, int] | None,
    group_crops: Mapping[str, Any],
    tile_observer: Callable[[Any], None] | None,
    ledger: _RunLedger,
) -> _GroupProducts:
    """Normalize, reject and integrate one output group, then crop the master
    and its maps to the run's common rectangle and render its preview.

    ``group_name`` names the output group (a filter, or a colour channel of a
    Bayer filter); ``source_filter`` is the Lights' own filter, which owns the
    flats, exposures and numeric domain.
    """

    source_filter = plan.group_filter[group_name]
    cfa_channel = plan.group_channel[group_name]
    cfa_pattern = plan.group_cfa_pattern[group_name]
    group_cfa_metadata = (
        {
            "OAFCFA": cfa_pattern,
            "OAFCFACH": CHANNEL_NAMES[cfa_channel],
            "OAFCFAF": source_filter,
        }
        if cfa_channel is not None
        else {}
    )
    domain_metadata = numeric_domain_metadata(plan.light_domain_references[source_filter])
    timing: dict[str, float] = {}
    group_started = time.perf_counter()
    exposures = {
        info.exposure_seconds for path, info in plan.light_info.items() if path in paths
    }
    reference_exposure = plan.reference_exposures[source_filter]
    total_exposure = sum(float(plan.light_info[path].exposure_seconds) for path in paths)
    registered_paths = [lights.by_group[(path, group_name)] for path in paths]
    reference_index, reference_selection = _normalization_reference_index(
        paths, plan.stellar_scale_hints, plan.quality_weights
    )
    expressions = [FrameExpression(str(path)) for path in registered_paths]
    normalization_record: dict[str, Any] = {
        "status": "DISABLED",
        "parameters": parameters.global_normalization.serializable(),
    }
    normalization_method = "NONE"
    if parameters.global_normalization.enabled:
        normalization_started = time.perf_counter()
        global_result, _hints, fit_seconds, prefetched = fits.fit(group_name, paths)
        timing["normalizationFit"] = fit_seconds
        timing["normalizationWait"] = time.perf_counter() - normalization_started
        timing["normalizationPrefetched"] = float(prefetched)
        expressions = [
            FrameExpression(
                str(path),
                scale=coefficient.scale,
                offset=coefficient.offset,
                offset_grid=coefficient.offset_grid,
                offset_grid_x=coefficient.offset_grid_x,
                offset_grid_y=coefficient.offset_grid_y,
            )
            for path, coefficient in zip(registered_paths, global_result.coefficients, strict=True)
        ]
        normalization_record = {
            "status": "APPLIED",
            "referenceInput": str(plan.display_path(paths[reference_index])),
            "referenceSelection": reference_selection,
            "evidence": _public_normalization_evidence(plan, paths, reference_index, global_result.receipt),
        }
        normalization_method = "GLOBAL_STELLAR"
    token = _safe_token(group_name)
    full_master = dirs.work / f"integrated_{token}.fits"
    full_maps = IntegrationMapPaths(
        accepted_count=dirs.work / f"accepted_count_{token}.fits",
        coverage=dirs.work / f"coverage_{token}.fits",
        rejection_count=dirs.work / f"rejection_count_{token}.fits",
    )
    timing["normalization"] = time.perf_counter() - group_started
    region_mapped_lights: list[dict[str, Any]] = []
    if plan.region_weight_maps:
        expressions, region_mapped_lights = _with_region_weights(plan, paths, expressions)
    integration_started = time.perf_counter()
    proper = parameters.proper_coaddition
    reuse_rejection = proper.enabled and proper.outlier_handling == "reuse-rejection"
    mask_recorder = (
        _RejectionMaskRecorder(len(paths), plan.light_info[paths[0]].shape)
        if parameters.capture_drizzle_inputs or reuse_rejection
        else None
    )
    if parameters.capture_drizzle_inputs and any(
        path not in lights.calibrated for path in paths
    ):
        raise CalibrationError(
            "DRIZZLE_INPUTS_UNAVAILABLE",
            "drizzle inputs need materialized calibrated Lights",
        )
    master_metadata = {
        "IMAGETYP": "Master Light",
        "FILTER": group_name,
        "OAFSTATE": OUTPUT_STATE,
        "OAFWCS": "UNSOLVED",
        "EXPTIME": reference_exposure,
        "OAFINTTM": total_exposure,
        "OAFNORM": normalization_method,
        **group_cfa_metadata,
        **domain_metadata,
    }
    integration = integrate_registered_group(
        expressions,
        full_master,
        metadata=master_metadata,
        parameters=parameters.integration,
        requested_backend=parameters.ordinary_integration_backend,
        native_library_path=parameters.native_library_path,
        metal_source_path=parameters.metal_source_path,
        metal_executor=metal.executor,
        metal_unavailable_reason=metal.unavailable_reason,
        hardware=hardware_profile,
        tuning=execution_tuning,
        quality_weights=[plan.quality_weights[path] for path in paths],
        map_paths=full_maps,
        durable=parameters.durable_intermediates,
        tile_observer=_compose_tile_observers(tile_observer, mask_recorder),
    )
    drizzle = None
    if parameters.capture_drizzle_inputs and mask_recorder is not None:
        drizzle = DrizzleGroupInputs(
            filter_name=group_name,
            cfa_pattern=cfa_pattern,
            channel=cfa_channel,
            frames=tuple(
                DrizzleFrame(
                    calibrated_path=str(lights.calibrated[path]),
                    source_path=str(plan.display_path(path)),
                    input_to_reference=tuple(
                        tuple(float(value) for value in row)
                        for row in plan.transforms[path].validated_matrix()
                    ),
                    weight=float(weight),
                    exposure_seconds=float(plan.light_info[path].exposure_seconds),
                    normalization_scale=float(expression.scale),
                    normalization_offset=float(expression.offset),
                    offset_grid=expression.offset_grid,
                    offset_grid_x=expression.offset_grid_x,
                    offset_grid_y=expression.offset_grid_y,
                    weight_grid=expression.weight_grid,
                    weight_grid_x=expression.weight_grid_x,
                    weight_grid_y=expression.weight_grid_y,
                    accepted_mask_bits=bits,
                )
                for path, expression, weight, bits in zip(
                    paths, expressions, integration.weights, mask_recorder.bits, strict=True,
                )
            ),
            reference_shape=plan.light_info[paths[0]].shape,
            metadata={
                "IMAGETYP": "Master Light",
                "FILTER": group_name,
                "EXPTIME": reference_exposure,
                "OAFINTTM": total_exposure,
                "OAFSTATE": OUTPUT_STATE,
                "OAFWCS": "UNSOLVED",
                "OAFNORM": normalization_method,
                **group_cfa_metadata,
                **domain_metadata,
            },
        )
    metal.observe(integration.execution)
    if shared_crop is not None:
        if integration.shape != plan.light_info[paths[0]].shape:
            raise CalibrationError(
                "REGISTRATION_GEOMETRY_MISMATCH",
                f"{group_name} integrated {integration.shape} but its "
                f"Lights were registered as {plan.light_info[paths[0]].shape}",
            )
        crop = shared_crop
    else:
        height, width = integration.shape
        crop = (0, 0, height, width)
    top, left, bottom, right = crop
    crop_fraction = ((bottom - top) * (right - left)) / (integration.shape[0] * integration.shape[1])
    if crop_fraction < parameters.minimum_crop_fraction:
        raise CalibrationError(
            "AUTOCROP_TOO_SMALL",
            f"common crop retains only {crop_fraction:.3%} of the frame",
        )
    timing["integration"] = time.perf_counter() - integration_started
    crop_write_started = time.perf_counter()
    master_light = dirs.masters / f"master_light_{token}.fits"
    master_stats, master_sha256 = _crop_fits(
        full_master,
        master_light,
        crop,
        {
            "IMAGETYP": "Master Light",
            "FILTER": group_name,
            "EXPTIME": reference_exposure,
            "OAFINTTM": total_exposure,
            "OAFSTATE": OUTPUT_STATE,
            "OAFWCS": "UNSOLVED",
            "OAFCROP": "AUTO" if parameters.auto_crop else "NONE",
            "OAFNFRM": len(paths),
            "OAFNORM": normalization_method,
            **group_cfa_metadata,
            **domain_metadata,
        },
        max_memory_bytes=parameters.integration.max_memory_bytes,
        durable=parameters.durable_intermediates,
    )
    ledger.record(
        master_light,
        "MASTER_LIGHT_LINEAR_UNSOLVED",
        statistics=master_stats,
        sha256=master_sha256,
        details={
            "filter": group_name,
            "crop": {
                "top": top,
                "left": left,
                "bottomExclusive": bottom,
                "rightExclusive": right,
                "retainedFraction": crop_fraction,
            },
        },
    )
    proper_record: dict[str, Any] | None = None
    proper_light: Path | None = None
    if proper.enabled:
        proper_started = time.perf_counter()
        proper_full = dirs.work / f"proper_{token}.fits"
        # After global normalization every frame carries the reference's
        # photometric scale, so the model's per-frame flux scale is the same
        # constant for all of them: the transparency difference has moved
        # into each frame's own background sigma, which is exactly where the
        # weight F_j / sigma_j^2 needs it.  Without normalization there is no
        # measured transparency to use and equal flux scales are assumed.
        flux_scale_source = (
            "normalized-to-reference"
            if parameters.global_normalization.enabled
            else "unit-assumed-equal-transparency"
        )
        if reuse_rejection and (mask_recorder is None or not mask_recorder.complete):
            raise CalibrationError(
                "PROPER_COADD_REJECTION_UNAVAILABLE",
                f"{group_name}: the integration did not report an accepted-sample mask for every row",
            )
        proper_result = proper_coadd_group(
            expressions,
            proper_full,
            master_path=full_master,
            shape=integration.shape,
            flux_scales=[1.0] * len(expressions),
            accepted_bits=mask_recorder.bits if mask_recorder is not None else None,
            metadata={
                "IMAGETYP": "Master Light Proper Coadd",
                "FILTER": group_name,
                "OAFSTATE": OUTPUT_STATE,
                "OAFWCS": "UNSOLVED",
                "EXPTIME": reference_exposure,
                "OAFINTTM": total_exposure,
                "OAFNORM": normalization_method,
                **group_cfa_metadata,
                **domain_metadata,
            },
            parameters=proper,
            division_floor=parameters.integration.division_floor,
            max_memory_bytes=parameters.integration.max_memory_bytes,
            workers=max(1, execution_tuning.cpu_workers),
            durable=False,
        )
        proper_light = dirs.masters / f"proper_light_{token}.fits"
        proper_statistics, proper_sha256 = _crop_fits(
            proper_full,
            proper_light,
            crop,
            {
                "IMAGETYP": "Master Light Proper Coadd",
                "FILTER": group_name,
                "EXPTIME": reference_exposure,
                "OAFINTTM": total_exposure,
                "OAFSTATE": OUTPUT_STATE,
                "OAFWCS": "UNSOLVED",
                "OAFCROP": "AUTO" if parameters.auto_crop else "NONE",
                "OAFNFRM": len(paths),
                "OAFNORM": normalization_method,
                "OAFPCOAD": PROPER_COADD_ALGORITHM_ID,
                "OAFPCFR": proper_result.flux_scale_norm,
                "OAFPCFWH": proper_result.coadd_fwhm_pixels,
                "OAFPCSKY": proper_result.sky_added,
                "OAFPCAPO": proper.apodization_pixels,
                "OAFPCREP": proper_result.replaced_samples,
                "OAFPCOUT": proper.outlier_handling,
                **group_cfa_metadata,
                **domain_metadata,
            },
            max_memory_bytes=parameters.integration.max_memory_bytes,
            durable=parameters.durable_intermediates,
        )
        ledger.record(
            proper_light,
            "MASTER_LIGHT_PROPER_COADD_UNSOLVED",
            statistics=proper_statistics,
            sha256=proper_sha256,
            details={"filter": group_name, "algorithm": PROPER_COADD_ALGORITHM_ID},
        )
        remove_file(proper_full)
        proper_record = {
            **proper_result.serializable(),
            "outputPath": str(proper_light.relative_to(dirs.root)),
            "uncroppedSha256": proper_result.output_sha256,
            "croppedSha256": proper_sha256,
            "fluxScaleSource": flux_scale_source,
            # Region weight maps scale samples in the ordinary weighted mean;
            # the transform has no per-sample weight, so a group that uses
            # them coadds unweighted and the receipt says so.
            "regionWeightMapsApplied": False,
            "regionWeightMapFrames": len(region_mapped_lights),
            "statistics": proper_statistics.serializable(),
            "primaryProduct": False,
            "ordinaryMasterUnaffected": True,
        }
        timing["properCoaddition"] = time.perf_counter() - proper_started
    cropped_maps: dict[str, Path] = {}
    map_statistics: dict[str, Any] = {}
    for map_name, artifact_kind in (
        ("acceptedSampleCount", "INTEGRATION_ACCEPTED_COUNT"),
        ("coverageFraction", "INTEGRATION_COVERAGE"),
        ("rejectionCount", "INTEGRATION_REJECTION_COUNT"),
    ):
        destination = dirs.coverage / f"{token}_{map_name}.fits"
        statistics, map_sha256 = _crop_fits(
            Path(integration.map_paths[map_name]),
            destination,
            crop,
            {
                "IMAGETYP": artifact_kind.replace("_", " ").title(),
                "FILTER": group_name,
                "OAFSTATE": OUTPUT_STATE,
                "OAFMAP": map_name.upper(),
                "OAFNFRM": len(paths),
            },
            max_memory_bytes=parameters.integration.max_memory_bytes,
            durable=parameters.durable_intermediates,
        )
        cropped_maps[map_name] = destination
        map_statistics[map_name] = statistics.serializable()
        ledger.record(
            destination,
            artifact_kind,
            statistics=statistics,
            sha256=map_sha256,
            details={
                "filter": group_name,
                "sourceIntegration": str(full_master.relative_to(dirs.root)),
                "usesRegistrationQualityWeights": True,
            },
        )
    preview_path = dirs.previews / f"master_light_{token}.png"
    timing["cropAndMaps"] = time.perf_counter() - crop_write_started
    preview_started = time.perf_counter()
    preview_result = render_auto_stretch_preview(
        master_light,
        preview_path,
        max_long_edge=parameters.preview_max_long_edge,
        max_memory_bytes=parameters.registration_memory_bytes,
    )
    preview_record = ledger.record(
        preview_path, "AUTO_STRETCH_PREVIEW", details=preview_result.serializable()
    )
    preview_record["details"]["outputPath"] = str(preview_path.relative_to(dirs.root))
    integration_record = _integration_record(integration, dirs.root)
    integration_record["maps"] = {
        name: str(path.relative_to(dirs.root)) for name, path in cropped_maps.items()
    }
    timing["preview"] = time.perf_counter() - preview_started
    timing["total"] = time.perf_counter() - group_started
    record = {
        "integration": integration_record,
        "regionWeightMaps": region_mapped_lights,
        "globalNormalization": normalization_record,
        "exposureNormalization": {
            "sourceExposureSeconds": sorted(float(value) for value in exposures),
            "referenceSeconds": reference_exposure,
            "totalIntegrationSeconds": total_exposure,
            "method": "LINEAR_REFERENCE_EXPOSURE",
        },
        "crop": [top, left, bottom, right],
        "groupCrop": list(group_crops.get(source_filter, crop)),
        "cropSharedAcrossFilters": len(plan.output_groups) > 1,
        "sourceFilter": source_filter,
        "cfaChannel": CHANNEL_NAMES[cfa_channel] if cfa_channel is not None else None,
        "cfaPattern": cfa_pattern,
        "masterStatistics": master_stats.serializable(),
        "mapStatistics": map_statistics,
        **({"properCoaddition": proper_record} if proper_record is not None else {}),
    }
    return _GroupProducts(
        master_light, preview_path, drizzle, record, timing, proper_coadd=proper_light
    )
