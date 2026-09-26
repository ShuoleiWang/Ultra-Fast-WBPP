"""The integration stage of one target.

Calibrate, register and integrate the admitted Lights in one pixel-pipeline
run.  Under the unattended selection a frame the leave-one-out counterfactual
confirms harmful is removed and its groups are integrated again without it,
at most ``max_integration_passes`` times; the selection receipt records every
pass.  Drizzle mode runs the same integration because its normalization,
weights and rejection masks are what the drizzle applies.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, NamedTuple, Sequence

from ..path_budget import PIXEL_PIPELINE_DIRECTORY, PIXEL_PIPELINE_STAGING_STEM
from ..platform import remove_tree
from ..selection import (
    CounterfactualReport,
    LeaveOneOutAccumulator,
    annotate_with_counterfactual,
    confirmed_harmful,
    exclude_confirmed,
)
from ..selection.policy import selection_receipt
from ..stacking.integration import CalibrationError
from ..stacking.parameters import PipelineResult
from ..stacking.pipeline import run_portable_pipeline_fits
from .common import _emit, _write_json
from .contracts import E2EError, E2ERequest, IntegrationMode, ProgressCallback, ProgressStage
from .registration import _capture_single_field_generated_calibration, _RegistrationProducts
from .screening import (
    _counterfactual_exclusions,
    _registered_region_maps,
    _Screening,
    _screening_summary,
    _staged_pixel_maps,
)
from .sources import _E2ESources, _trusted_source_identity_bindings


class _IntegrationProducts(NamedTuple):
    drizzle_mode: bool
    pipeline_result: PipelineResult
    pipeline_root: Path
    pixel_pipeline_receipt: dict[str, Any]
    ordinary_executions: dict[str, Any]
    selection_receipt_path: str | None


def _integrate_admitted_lights(
    request: E2ERequest,
    sources: _E2ESources,
    screening: _Screening,
    registration: _RegistrationProducts,
    *,
    staged_inputs: Mapping[str, tuple[Path, ...]],
    registration_source_aliases: Mapping[str, Path],
    xisf_conversions: Sequence[Mapping[str, Any]],
    calibration_plan: Any,
    registration_calibration_receipt_path: Path,
    work: Path,
    receipts_dir: Path,
    progress: ProgressCallback | None,
) -> _IntegrationProducts:
    """Integrate the admitted Lights; ``screening`` is updated in place with
    every frame a counterfactual pass removes."""

    identity_by_path = sources.identity_by_path
    lights = sources.lights
    _emit(progress, ProgressStage.INTEGRATION, "started", "calibrating, registering, and integrating PASS frames")
    # Drizzle mode runs the same registered integration: its per-frame
    # normalization, weights and rejection masks are what the drizzle
    # applies to the calibrated Lights on the finer grid.
    pipeline_transforms: Mapping[str, Sequence[Sequence[float]]] = registration.transforms
    if set(pipeline_transforms) != {str(path) for path in screening.passed}:
        raise E2EError(
            "REGISTRATION_TRANSFORM_SET_INCOMPLETE",
            "transform map must bind every and only admitted Light",
        )
    pixel_source_paths = tuple(
        path for _role, paths in sources.pixel_groups(screening.passed) for path in paths
    )
    selected_source_digests = {
        identity_by_path[str(path.resolve(strict=True))].sha256 for path in pixel_source_paths
    }
    drizzle_mode = request.integration_mode is IntegrationMode.DRIZZLE
    pipeline_parameters = replace(
        request.pipeline_parameters,
        raw_frame_metadata_overrides=tuple(
            override
            for override in request.pipeline_parameters.raw_frame_metadata_overrides
            if override.source_sha256 in selected_source_digests
        ),
        # Ordinary integration consumes calibrated Lights in memory; only
        # Drizzle reads them back from the pipeline directory.
        materialize_calibrated_lights=drizzle_mode,
        capture_drizzle_inputs=drizzle_mode,
        # The whole pipeline directory lives in the transient work tree;
        # the promoted products below are fsynced by this run.
        durable_intermediates=False,
    )
    # The inventory already hashed every original through one read; hand
    # those path/stat-bound digests to the pixel pipeline so it never
    # rereads a source only to recompute a digest it must then verify.
    pixel_identity_seed = {}
    for path in pixel_source_paths:
        identity = identity_by_path[str(path.resolve(strict=True))]
        pixel_identity_seed[str(path.resolve(strict=True))] = (
            identity.sha256,
            {
                "sizeBytes": identity.size_bytes,
                "mtimeNs": identity.mtime_ns,
                "device": identity.device,
                "inode": identity.inode,
            },
        )
    trusted_generated_calibration = None
    trusted_source_identities = None
    if request.integration_mode is IntegrationMode.ORDINARY and bool(
        sources.biases or sources.darks or sources.flats
    ):
        trusted_source_identities = _trusted_source_identity_bindings(identity_by_path, pixel_source_paths)
        trusted_generated_calibration = _capture_single_field_generated_calibration(
            plan=calibration_plan,
            generated_directory=work / "registration-calibration",
            upstream_receipt_path=registration_calibration_receipt_path,
            staged_inputs=staged_inputs,
            source_aliases=registration_source_aliases,
            pipeline_parameters=request.pipeline_parameters,
            consumer_source_groups=sources.pixel_groups(screening.passed),
            internal_source_identities=trusted_source_identities,
        )

    selection_observers: dict[str, LeaveOneOutAccumulator] = {}

    def _selection_observer_factory(
        filter_name: str, ordered_paths: Sequence[str]
    ) -> LeaveOneOutAccumulator | None:
        if not request.selection.unattended or request.selection.counterfactual != "analytic":
            return None
        accumulator = LeaveOneOutAccumulator(
            [str(registration_source_aliases[str(staged)].resolve(strict=True)) for staged in ordered_paths]
        )
        selection_observers[filter_name] = accumulator
        return accumulator

    # The unattended selection may integrate more than once: a frame the
    # counterfactual confirms harmful is removed and its groups are
    # integrated again without it (at most ``max_integration_passes``).
    light_subset: list[Path] = list(staged_inputs["LIGHT"])
    selection_reports: dict[str, CounterfactualReport] = {}
    selection_receipt_path: str | None = None
    selection_reintegration: dict[str, Any] | None = None
    reintegration_passes: list[dict[str, Any]] = []
    admitted_at_start = len(light_subset)
    pass_index = 0
    while True:
        pass_index += 1
        pass_root = work / (
            PIXEL_PIPELINE_DIRECTORY if pass_index == 1 else f"{PIXEL_PIPELINE_DIRECTORY}-pass{pass_index}"
        )
        staged_transforms, staged_weights, staged_hints = _staged_pixel_maps(
            light_subset, registration_source_aliases, pipeline_transforms, registration
        )
        selection_observers.clear()
        try:
            pipeline_result = run_portable_pipeline_fits(
                # The deepest level of the run's layout; see path_budget.
                _staging_stem=PIXEL_PIPELINE_STAGING_STEM,
                bias_files=staged_inputs["BIAS"],
                dark_files=staged_inputs["DARK"],
                flat_files=staged_inputs["FLAT"],
                master_bias_files=staged_inputs["MASTER_BIAS"],
                master_dark_files=staged_inputs["MASTER_DARK"],
                master_flat_files=staged_inputs["MASTER_FLAT"],
                light_files=light_subset,
                output_directory=pass_root,
                transforms=staged_transforms,
                quality_weights=staged_weights,
                stellar_scale_hints=staged_hints,
                parameters=pipeline_parameters,
                _source_aliases=registration_source_aliases,
                _xisf_conversions=xisf_conversions,
                _trusted_generated_calibration=trusted_generated_calibration,
                _source_identity_seed=pixel_identity_seed,
                _integration_tile_observers=(
                    _selection_observer_factory if request.selection.unattended else None
                ),
                region_weight_maps=(
                    _registered_region_maps(light_subset, registration_source_aliases, screening, pipeline_transforms) or None
                    if screening.selection_region_maps
                    else None
                ),
            )
        except CalibrationError as error:
            raise E2EError(error.code, str(error), path=error.path) from error
        if not request.selection.unattended:
            break
        fwhm_by_path = {item.path: item.fwhm_native for item in screening.selection_features}
        selection_reports = {
            group_name: accumulator.finalize(
                fwhm_by_frame=[fwhm_by_path.get(path) for path in accumulator.paths]
            )
            for group_name, accumulator in selection_observers.items()
        }
        annotated = list(screening.selection_decisions)
        for report in selection_reports.values():
            annotated = annotate_with_counterfactual(annotated, report, request.selection)
        screening.selection_decisions = annotated
        harmful = confirmed_harmful(screening.selection_decisions)
        if (
            pass_index >= request.selection.max_integration_passes
            or request.selection.counterfactual_action != "exclude"
            or not harmful
        ):
            break
        removable, kept_reasons = _counterfactual_exclusions(
            harmful,
            light_subset=light_subset,
            source_aliases=registration_source_aliases,
            frame_results=screening.frame_results,
            registration=registration,
            admitted_at_start=admitted_at_start,
            soft_exclusion_fraction_guard=request.selection.soft_exclusion_fraction_guard,
        )
        # The confirming counterfactual numbers travel with the record:
        # the next pass measures only the remaining frames.
        harmful_evidence = {
            item.path: item.counterfactual.serializable()
            for item in harmful
            if item.counterfactual is not None
        }
        if not removable:
            selection_reintegration = {
                "status": "NOT_APPLIED",
                "pass": pass_index,
                "confirmedHarmful": [item.path for item in harmful],
                "keptBecause": kept_reasons,
                "evidence": harmful_evidence,
            }
            reintegration_passes.append(selection_reintegration)
            break
        screening.selection_decisions = exclude_confirmed(screening.selection_decisions, removable)
        removed_set = set(removable)
        light_subset = [
            staged
            for staged in light_subset
            if str(registration_source_aliases[str(staged)].resolve(strict=True)) not in removed_set
        ]
        screening.passed = tuple(path for path in screening.passed if str(path) not in removed_set)
        passed_set = set(screening.passed)
        screening.excluded = tuple(path for path in lights if path not in passed_set)
        if trusted_generated_calibration is not None:
            # The generated calibration set is bound to the exact pixel
            # input manifest, so it is captured again for the reduced set.
            trusted_generated_calibration = _capture_single_field_generated_calibration(
                plan=calibration_plan,
                generated_directory=work / "registration-calibration",
                upstream_receipt_path=registration_calibration_receipt_path,
                staged_inputs={**staged_inputs, "LIGHT": tuple(light_subset)},
                source_aliases=registration_source_aliases,
                pipeline_parameters=request.pipeline_parameters,
                consumer_source_groups=sources.pixel_groups(screening.passed),
                internal_source_identities=trusted_source_identities,
            )
        screening.screening = _screening_summary(
            screening.frame_results, screening.passed, screening.approved_review_paths, screening.review_previews
        )
        selection_reintegration = {
            "status": "APPLIED",
            "pass": pass_index,
            "excluded": sorted(removable),
            "keptBecause": kept_reasons,
            "evidence": harmful_evidence,
            "discardedPass": pass_root.name,
        }
        reintegration_passes.append(selection_reintegration)
        _emit(
            progress,
            ProgressStage.INTEGRATION,
            "running",
            f"{len(removable)} frame(s) measured harmful by the counterfactual; "
            "integrating again without them",
        )
        remove_tree(pass_root, ignore_errors=True)
    pipeline_root = pass_root
    if request.selection.unattended:
        selection_block = selection_receipt(
            request.selection,
            screening.selection_features,
            screening.selection_decisions,
            selection_reports,
            region_maps=screening.selection_region_maps,
        )
        if selection_reintegration is not None:
            applied = [item for item in reintegration_passes if item.get("status") == "APPLIED"]
            kept: dict[str, str] = {}
            for item in reintegration_passes:
                kept.update(item.get("keptBecause", {}))
            selection_block["reintegration"] = {
                "status": "APPLIED" if applied else selection_reintegration["status"],
                "excluded": sorted(path for item in applied for path in item.get("excluded", [])),
                "keptBecause": kept,
                "passes": reintegration_passes,
            }
        else:
            selection_block["reintegration"] = None
        selection_block["integrationPasses"] = pass_index
        _write_json(screening.qc_dir / "selection.json", selection_block)
        selection_receipt_path = "qc/selection.json"
    shutil.copyfile(pipeline_result.receipt_path, receipts_dir / "pixel-pipeline.json")
    pixel_pipeline_receipt = json.loads(Path(pipeline_result.receipt_path).read_text(encoding="utf-8"))
    ordinary_executions = (
        {
            filter_name: group.get("integration", {}).get("execution", {})
            for filter_name, group in pixel_pipeline_receipt.get("statistics", {})
            .get("integrationGroups", {})
            .items()
        }
        if request.integration_mode is IntegrationMode.ORDINARY
        else {}
    )
    _emit(progress, ProgressStage.INTEGRATION, "completed", "linear UNSOLVED masters created")
    return _IntegrationProducts(
        drizzle_mode,
        pipeline_result,
        pipeline_root,
        pixel_pipeline_receipt,
        ordinary_executions,
        selection_receipt_path,
    )
