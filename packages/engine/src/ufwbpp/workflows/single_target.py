"""Fail-closed end-to-end astrophotography orchestration.

The public pixel pipeline deliberately publishes unsolved working masters.  This
module composes the existing QC, calibration, registration, integration,
optional drizzle, and solver boundaries into one transaction.  A requested
output directory is published only after every filter has a newly solved and
independently verified celestial WCS.  Solver failure publishes a separate
``.unsolved`` evidence directory and returns an unsuccessful result.

All caller-owned files are opened read-only.  Outputs are built on the same
filesystem as their destination and committed with a no-replace directory
rename, so a successful receipt can never describe a partial output tree.
"""


from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from lightframeqc.parallel import FrameRunner

from ..blink.session import BlinkEvidence
from ..calibration.policy import workflow_receipt
from ..integrity import canonical_json_document
from ..path_budget import STAGING_SUFFIX, WORK_DIRECTORY
from ..platform import remove_tree
from ..products.preview import render_auto_stretch_preview
from ..solvers.base import SolverBackend, WcsValidation
from ..stacking.integration import CalibrationError
from ..stacking.proper_coaddition import PROPER_COADD_ALGORITHM_ID
from .common import (
    _artifact_records,
    _emit,
    _fsync_directory,
    _rename_directory_no_replace,
    _safe_token,
    _write_json,
)
from .contracts import (
    E2EError,
    E2ERequest,
    E2EResult,
    E2EState,
    ExplicitSelection,
    IntegrationMode,
    ProgressCallback,
    ProgressStage,
    ReviewApproval,
)
from .integration import _integrate_admitted_lights
from .products import _drizzle_candidates, _ordinary_candidates, _promote_proper_coadds, _solve_candidates
from .registration import _build_registration_masters, _register_lights
from .screening import _panels_below_registration_minimum, _registration_anchor_exclusions, _screen_lights
from .sharing import _sanitize_shareable_tree, _share_safe_receipt_core, _share_safe_value
from .solve import _same_grid_signatures, _unify_same_grid_solutions, _validate_cross_filter_wcs
from .sources import (
    _SourceIdentity,
    _stage_e2e_xisf_inputs,
    _validated_sources,
    _verify_sources,
    _verify_staged_pixel_inputs,
)


E2E_VERSION = "ultra-fast-wbpp-e2e-v1"


def _validate_request(request: E2ERequest) -> None:
    if not isinstance(request, E2ERequest):
        raise E2EError("REQUEST_INVALID", "request has the wrong type")
    if isinstance(request.workers, bool) or not isinstance(request.workers, int) or request.workers < 1:
        raise E2EError("WORKER_COUNT_INVALID", "workers must be a positive integer")
    request.qc_config.validate()
    request.gate_policy.validate()
    request.pipeline_parameters.validate()
    request.drizzle.validate()
    digest_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    if request.recipe_digest is not None and (
        not isinstance(request.recipe_digest, str)
        or digest_pattern.fullmatch(request.recipe_digest) is None
    ):
        raise E2EError(
            "RECIPE_DIGEST_INVALID", "recipe_digest must be a lowercase sha256: digest"
        )
    for approval in request.review_approvals:
        if not isinstance(approval, ReviewApproval):
            raise E2EError(
                "REVIEW_APPROVAL_INVALID", "review approvals have the wrong type"
            )
        for name, value in approval.serializable().items():
            if not isinstance(value, str) or digest_pattern.fullmatch(value) is None:
                raise E2EError(
                    "REVIEW_APPROVAL_INVALID",
                    f"{name} must be a lowercase sha256: digest",
                )
    if not isinstance(request.integration_mode, IntegrationMode):
        raise E2EError("INTEGRATION_MODE_INVALID", "unknown integration mode")
    request.selection.validate()
    request.blink_flags_policy.validate()
    if request.explicit_selection is not None and not isinstance(
        request.explicit_selection, ExplicitSelection
    ):
        raise E2EError("SELECTION_INVALID", "explicit selection has the wrong type")
    if request.selection.explicit != (request.explicit_selection is not None):
        raise E2EError(
            "SELECTION_POLICY_CONFLICT",
            "selection policy explicit-v1 and a supplied selection go together",
        )
    if request.explicit_selection is not None and request.review_approvals:
        raise E2EError(
            "SELECTION_POLICY_CONFLICT",
            "an explicit selection cannot be combined with REVIEW approvals",
        )
    numeric_hints = {
        "ra_hint_degrees": request.ra_hint_degrees,
        "dec_hint_degrees": request.dec_hint_degrees,
        "field_of_view_degrees": request.field_of_view_degrees,
        "search_radius_degrees": request.search_radius_degrees,
    }
    for name, value in numeric_hints.items():
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise E2EError("SOLVER_HINT_INVALID", f"{name} must be finite")
    if (request.ra_hint_degrees is None) != (request.dec_hint_degrees is None):
        raise E2EError("SOLVER_HINT_INCOMPLETE", "RA and Dec hints must be supplied together")
    if request.ra_hint_degrees is not None and not 0 <= request.ra_hint_degrees < 360:
        raise E2EError("SOLVER_HINT_INVALID", "RA hint must be in [0, 360)")
    if request.dec_hint_degrees is not None and not -90 <= request.dec_hint_degrees <= 90:
        raise E2EError("SOLVER_HINT_INVALID", "Dec hint must be in [-90, 90]")
    for name in ("field_of_view_degrees", "search_radius_degrees"):
        value = getattr(request, name)
        if value is not None and value <= 0:
            raise E2EError("SOLVER_HINT_INVALID", f"{name} must be positive")
    if (
        isinstance(request.min_matches, bool)
        or not isinstance(request.min_matches, int)
        or request.min_matches < 12
    ):
        raise E2EError(
            "SOLVER_QUALITY_POLICY_INVALID",
            "min_matches must be an integer of at least 12",
        )
    if (
        isinstance(request.max_rms_arcsec, bool)
        or not isinstance(request.max_rms_arcsec, (int, float))
        or not math.isfinite(float(request.max_rms_arcsec))
        or float(request.max_rms_arcsec) <= 0
        or float(request.max_rms_arcsec) > 2.0
    ):
        raise E2EError(
            "SOLVER_QUALITY_POLICY_INVALID",
            "max_rms_arcsec must be finite, positive, and no greater than 2.0",
        )
    tolerance = request.same_grid_wcs_tolerance_pixels
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0
        or float(tolerance) > 5.0
    ):
        raise E2EError(
            "SOLVER_QUALITY_POLICY_INVALID",
            "same_grid_wcs_tolerance_pixels must be finite, positive, and no greater than 5.0",
        )


def _failure_result(
    *,
    staging: Path,
    failure_directory: Path,
    code: str,
    message: str,
    passed: tuple[Path, ...],
    excluded: tuple[Path, ...],
    solver: Mapping[str, Any],
    sources: Sequence[_SourceIdentity],
    callback: ProgressCallback | None,
    screening: Mapping[str, Any] | None = None,
    selection_policy: str | None = None,
    selection: Mapping[str, Any] | None = None,
) -> E2EResult:
    private_work = staging / WORK_DIRECTORY
    remove_tree(private_work)
    _sanitize_shareable_tree(staging, sources)
    evidence_artifacts = _artifact_records(
        staging,
        tuple(
            path
            for path in (
                staging / "products",
                staging / "previews",
                staging / "qc",
                staging / "coverage",
                staging / "receipts",
            )
            if path.exists()
        ),
    )
    receipt_core = {
        "schemaVersion": 1,
        "pipelineVersion": E2E_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "state": E2EState.UNSOLVED_WORKING.value,
        "code": code,
        "message": message,
        "sources": [item.serializable() for item in sources],
        "qualityControl": {
            "manifest": "qc/manifest.json",
            "passedLights": len(passed),
            "excludedLights": len(excluded),
            **({"screening": dict(screening)} if screening is not None else {}),
            **({"selectionPolicy": selection_policy} if selection_policy is not None else {}),
            **({"selection": dict(selection)} if selection is not None else {}),
        },
        "astrometry": {"status": "UNSOLVED", **dict(solver)},
        "artifacts": evidence_artifacts,
        "publication": {
            "kind": "failure-evidence-only",
            "requestedOutputPublished": False,
            "noReplace": True,
        },
    }
    receipt_core = _share_safe_receipt_core(
        receipt_core, staging=staging, identities=sources
    )
    receipt_id = "sha256:" + hashlib.sha256(canonical_json_document(receipt_core)).hexdigest()
    _write_json(staging / "receipt.json", {"receiptId": receipt_id, **receipt_core})
    _emit(callback, ProgressStage.PUBLISH, "started", "publishing UNSOLVED evidence")
    _rename_directory_no_replace(staging, failure_directory)
    _fsync_directory(failure_directory.parent)
    try:
        _emit(callback, ProgressStage.FAILED, "completed", message)
    except Exception:
        pass
    return E2EResult(
        success=False,
        code=code,
        state=E2EState.UNSOLVED_WORKING,
        output_directory=None,
        evidence_directory=str(failure_directory),
        receipt_path=str(failure_directory / "receipt.json"),
        product_paths=(),
        preview_paths=(),
        passed_light_paths=tuple(str(path) for path in passed),
        excluded_light_paths=tuple(str(path) for path in excluded),
        message=message,
    )


def _blink_receipt(
    blink: BlinkEvidence, registration: Any, pixel_pipeline_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """The run's blink evidence next to the references the pipeline used."""

    return {
        "manifest": "qc/blink.json",
        "flagsPolicyDigest": blink.flags_policy.canonical_digest(),
        "referenceRule": next(iter(blink.references.values())).rule if blink.references else None,
        "referenceBySource": {
            channel_id: {
                "sourceSha256": reference.source_sha256,
                "path": reference.path,
                "candidacy": reference.candidacy,
            }
            for channel_id, reference in blink.references.items()
        },
        "pipelineReferences": {
            "registration": registration.receipt.get("referencePath"),
            "normalization": {
                filter_name: (group.get("globalNormalization") or {}).get("referenceInput")
                for filter_name, group in pixel_pipeline_receipt.get("statistics", {})
                .get("integrationGroups", {})
                .items()
            },
        },
    }


def run_e2e(
    request: E2ERequest,
    *,
    solver_backends: Sequence[SolverBackend],
    progress: ProgressCallback | None = None,
) -> E2EResult:
    """Run the complete workflow and atomically publish only verified WCS products.

    The phases: validate and hash the sources, screen the Lights, build the
    registration calibration, register, integrate (again without frames the
    selection counterfactual confirms harmful), optionally drizzle, solve
    every filter, verify the solutions against each other, then publish.

    ``solver_backends`` is an ordered fallback chain.  Each backend result must
    pass :func:`validate_solver_result`, execution-receipt verification, and an
    independent validation of the WCS actually written to ``output_path``.
    """

    _validate_request(request)
    if not isinstance(solver_backends, Sequence) or not solver_backends:
        raise E2EError("SOLVER_CHAIN_EMPTY", "at least one solver backend is required")
    sources = _validated_sources(request, progress)
    output = sources.output
    identities = sources.identities
    identity_by_path = sources.identity_by_path
    lights = sources.lights

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=STAGING_SUFFIX, dir=output.parent))
    published = False
    # One spawned worker pool serves quality-gate measurement, star-field
    # analysis and registration: each spawned worker imports NumPy, astropy,
    # SEP and the engine (about 1.7 s per worker on the Windows laptop), so
    # three pools per run cost several seconds of wall time for nothing.
    # Every frame runs the same function whichever pool executes it, so the
    # values never depend on the sharing.  Closed as soon as registration is
    # done so its processes do not sit on memory during integration.
    frame_runner = FrameRunner(request.workers, len(lights))
    try:
        work = staging / WORK_DIRECTORY
        receipts_dir = staging / "receipts"
        products_dir = staging / "products"
        previews_dir = staging / "previews"
        coverage_dir = staging / "coverage"
        for directory in (work, receipts_dir, products_dir, previews_dir, coverage_dir):
            directory.mkdir()

        screening = _screen_lights(request, sources, staging, frame_runner, progress)
        insufficient = _panels_below_registration_minimum(screening.frame_results, screening.passed)
        if insufficient:
            _verify_sources(identities)
            result = _failure_result(
                staging=staging,
                failure_directory=sources.failure_directory,
                code="QC_INSUFFICIENT_LIGHTS" if screening.passed else "NO_PASS_LIGHTS",
                message="; ".join(insufficient) + "; at least 2 admitted Light frames per target/filter are required for registration. Review the screening evidence before processing.",
                passed=screening.passed,
                excluded=screening.excluded,
                solver={"attempts": {}},
                sources=identities,
                callback=progress,
                screening=screening.screening,
                selection_policy=request.selection.policy,
                selection=screening.explicit_block,
            )
            published = True
            return result

        (
            staged_inputs,
            registration_source_aliases,
            xisf_conversions,
            staged_input_digests,
        ) = _stage_e2e_xisf_inputs(
            (
                ("LIGHT", screening.passed),
                ("FLAT", sources.flats),
                ("DARK", sources.darks),
                ("BIAS", sources.biases),
                ("MASTER_BIAS", sources.master_biases),
                ("MASTER_DARK", sources.master_darks),
                ("MASTER_FLAT", sources.master_flats),
            ),
            work / "pixel-inputs",
            request.pipeline_parameters,
            identity_by_path,
        )

        _emit(progress, ProgressStage.CALIBRATION, "started", "building registration calibration masters")
        try:
            calibration_plan, calibration_receipt = _build_registration_masters(
                biases=staged_inputs["BIAS"],
                darks=staged_inputs["DARK"],
                flats=staged_inputs["FLAT"],
                supplied_biases=staged_inputs["MASTER_BIAS"],
                supplied_darks=staged_inputs["MASTER_DARK"],
                supplied_flats=staged_inputs["MASTER_FLAT"],
                lights=staged_inputs["LIGHT"],
                directory=work / "registration-calibration",
                pipeline_parameters=request.pipeline_parameters,
                source_aliases=registration_source_aliases,
                source_identities=identity_by_path,
                xisf_conversions=xisf_conversions,
            )
        except CalibrationError as error:
            raise E2EError(error.code, str(error), path=error.path) from error
        _verify_staged_pixel_inputs(staged_input_digests, identities)
        registration_calibration_receipt_path = receipts_dir / "registration-calibration.json"
        # This receipt becomes the content-bound trust anchor for generated
        # masters.  Write its final share-safe representation now so the SHA
        # recorded by the pixel receipt remains verifiable after publication.
        _write_json(
            registration_calibration_receipt_path,
            _share_safe_value(
                calibration_receipt,
                staging=staging,
                source_tokens={
                    identity.path: f"source/{identity.source_id}/{Path(identity.path).name}"
                    for identity in identities
                },
            ),
        )
        _emit(progress, ProgressStage.CALIBRATION, "completed", "calibration masters verified")

        _emit(progress, ProgressStage.REGISTRATION, "started", "measuring full-resolution transforms")
        anchor_excluded = _registration_anchor_exclusions(request, screening)
        gate_passed_lights = [
            path for path in staged_inputs["LIGHT"]
            if str(registration_source_aliases.get(str(path), path)) not in anchor_excluded
            and str(path) not in anchor_excluded
        ]
        registration = _register_lights(
            staged_inputs["LIGHT"],
            calibration_plan,
            detection=request.registration_detection,
            registration=request.registration_config,
            workers=request.workers,
            allow_projective=True,
            reference_candidates=gate_passed_lights if gate_passed_lights and len(gate_passed_lights) < len(staged_inputs["LIGHT"]) else None,
            source_aliases=registration_source_aliases,
            source_sha256_by_path={
                str(path): (
                    staged_input_digests[str(path)]
                    if str(path) in staged_input_digests
                    else identity_by_path[str(registration_source_aliases[str(path)].resolve(strict=True))].sha256
                )
                for path in staged_inputs["LIGHT"]
            },
            runner=frame_runner,
        )
        frame_runner.close()
        if screening.selection_confidence:
            # Reduced-confidence frames keep their registration quality weight
            # scaled by the selection confidence; excluded frames never reach here.
            registration = replace(
                registration,
                quality_weights={
                    path: float(weight) * float(screening.selection_confidence.get(path, 1.0))
                    for path, weight in registration.quality_weights.items()
                },
            )
        _write_json(receipts_dir / "registration.json", registration.receipt)
        _emit(progress, ProgressStage.REGISTRATION, "completed", f"accepted {len(registration.transforms)} full matrices")

        (
            drizzle_mode,
            pipeline_result,
            pipeline_root,
            pixel_pipeline_receipt,
            ordinary_executions,
            selection_receipt_path,
        ) = _integrate_admitted_lights(
            request,
            sources,
            screening,
            registration,
            staged_inputs=staged_inputs,
            registration_source_aliases=registration_source_aliases,
            xisf_conversions=xisf_conversions,
            calibration_plan=calibration_plan,
            registration_calibration_receipt_path=registration_calibration_receipt_path,
            work=work,
            receipts_dir=receipts_dir,
            progress=progress,
        )

        if drizzle_mode:
            _emit(progress, ProgressStage.DRIZZLE, "started", "executing per-filter drizzle")
            candidates, coverage = _drizzle_candidates(
                staging=staging,
                work=work,
                pipeline_result=pipeline_result,
                options=request.drizzle,
                sampling_evidence=screening.drizzle_sampling,
                threads=None,
            )
            for filter_name in sorted(candidates):
                source_receipt = work / "drizzle" / _safe_token(filter_name) / "receipt.json"
                shutil.copyfile(source_receipt, receipts_dir / f"drizzle_{_safe_token(filter_name)}.json")
            _emit(progress, ProgressStage.DRIZZLE, "completed", f"drizzled {len(candidates)} filters")
        else:
            candidates, coverage = _ordinary_candidates(
                pipeline_result.master_light_paths,
                pipeline_root,
                staging,
                minimum_coverage_fraction=request.drizzle.minimum_coverage_fraction,
                maximum_null_fraction=request.drizzle.maximum_null_fraction,
            )
        _write_json(coverage_dir / "coverage.json", coverage)

        _emit(progress, ProgressStage.ASTROMETRY, "started", "solving every filter", total=len(candidates))
        solver_records, product_paths_staged, solved_products, all_solved = _solve_candidates(
            candidates,
            products_dir=products_dir,
            staging=staging,
            backends=solver_backends,
            hints=screening.solver_hints,
            request=request,
            progress=progress,
        )
        # WCS tolerances are expressed in native pixels; drizzled masters have
        # ``scale`` pixels per native pixel, so the same angular agreement is
        # ``scale`` times as many of their own pixels.
        grid_pixels_per_native = float(request.drizzle.scale) if drizzle_mode else 1.0
        same_grid_unification: dict[str, Any] = {
            "status": "NOT_APPLICABLE",
            "reason": "single filter",
        }
        if all_solved and len(solved_products) > 1:
            same_grid_unification = _unify_same_grid_solutions(
                solved_products,
                solver_records=solver_records,
                tolerance_pixels=float(request.same_grid_wcs_tolerance_pixels) * grid_pixels_per_native,
                grid_signatures=_same_grid_signatures(
                    integration_mode=request.integration_mode,
                    pixel_pipeline_receipt=pixel_pipeline_receipt,
                    coverage=coverage,
                ),
            )
            if same_grid_unification["status"] == "MISMATCH":
                all_solved = False
        cross_filter_validation = (
            _validate_cross_filter_wcs(solved_products, tolerance_pixels=0.05 * grid_pixels_per_native)
            if all_solved
            else WcsValidation(
                False,
                str(same_grid_unification.get("code") or "CROSS_FILTER_WCS_SKIPPED"),
                str(
                    same_grid_unification.get("message")
                    or "one or more filter solutions failed their individual gates"
                ),
                {"sameGridUnification": same_grid_unification},
            )
        )
        all_solved = all_solved and cross_filter_validation.valid
        astrometry_record = {
            "hints": screening.solver_hints.serializable(),
            "qualityPolicy": {
                "minMatches": request.min_matches,
                "maxRmsArcsec": request.max_rms_arcsec,
            },
            "crossFilterValidation": cross_filter_validation.serializable(),
            "sameGridUnification": same_grid_unification,
            "filters": solver_records,
        }
        if not all_solved:
            _write_json(receipts_dir / "solver-attempts.json", {"filters": solver_records})
            _verify_sources(identities)
            result = _failure_result(
                staging=staging,
                failure_directory=sources.failure_directory,
                code="ASTROMETRY_REQUIRED",
                message="one or more filters did not produce a verified new WCS solution",
                passed=screening.passed,
                excluded=screening.excluded,
                solver=astrometry_record,
                sources=identities,
                callback=progress,
            )
            published = True
            return result
        _emit(progress, ProgressStage.ASTROMETRY, "completed", "all filters have verified WCS")

        proper_coadd_paths = dict(getattr(pipeline_result, "proper_coadd_paths", {}) or {})
        proper_coadd_record: dict[str, Any] = {"status": "NOT_REQUESTED"}
        if proper_coadd_paths:
            promoted_proper, proper_filters = _promote_proper_coadds(
                proper_paths=proper_coadd_paths,
                solved_products=solved_products,
                products_dir=products_dir,
                staging=staging,
            )
            proper_coadd_record = {
                "status": "PUBLISHED",
                "algorithm": PROPER_COADD_ALGORITHM_ID,
                "options": request.pipeline_parameters.proper_coaddition.serializable(),
                "additionalProduct": True,
                "primaryProductUnchanged": True,
                "filters": proper_filters,
            }

        _emit(progress, ProgressStage.PREVIEW, "started", "rendering solved-master previews")
        preview_paths_staged: list[Path] = []
        for product in product_paths_staged:
            preview_path = previews_dir / f"master_light_{product.parent.name}.png"
            render_auto_stretch_preview(
                product,
                preview_path,
                max_long_edge=request.pipeline_parameters.preview_max_long_edge,
                max_memory_bytes=request.pipeline_parameters.registration_memory_bytes,
            )
            preview_paths_staged.append(preview_path)
        _emit(progress, ProgressStage.PREVIEW, "completed", f"rendered {len(preview_paths_staged)} previews")

        _emit(progress, ProgressStage.VERIFY, "started", "verifying sources and final artifact identities")
        _verify_sources(identities)
        remove_tree(work)
        _sanitize_shareable_tree(staging, identities)
        artifacts = _artifact_records(
            staging,
            (products_dir, previews_dir, screening.qc_dir, coverage_dir, receipts_dir),
        )
        receipt_core: dict[str, Any] = {
            "schemaVersion": 1,
            "pipelineVersion": E2E_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "success": True,
            "state": E2EState.SOLVED.value,
            "code": "E2E_SUCCEEDED",
            "calibrationPolicy": workflow_receipt(request.pipeline_parameters.calibration_workflow),
            "integrationMode": request.integration_mode.value,
            "sources": [item.serializable() for item in identities],
            "execution": {"sourceExtraction": dict(screening.source_extraction)},
            "qualityControl": {
                "manifest": "qc/manifest.json",
                "passedLights": len(screening.passed),
                "excludedLights": len(screening.excluded),
                "screening": screening.screening,
                "selectionPolicy": request.selection.policy,
                "selection": (
                    screening.explicit_block if screening.explicit_block is not None else selection_receipt_path
                ),
                **(
                    {"blink": _blink_receipt(screening.blink, registration, pixel_pipeline_receipt)}
                    if screening.blink is not None
                    else {}
                ),
            },
            "registration": {
                "receipt": "receipts/registration.json",
                "fullMatrixConvention": "INPUT_TO_OUTPUT",
            },
            "integration": {
                "pixelPipelineReceipt": "receipts/pixel-pipeline.json",
                "coverage": "coverage/coverage.json",
                "ordinaryExecutions": ordinary_executions,
                "properCoaddition": proper_coadd_record,
            },
            "astrometry": {"status": "SOLVED", "requiredForSuccess": True, **astrometry_record},
            "artifacts": artifacts,
            "publication": {"atomic": True, "noReplace": True, "sourceMutation": False},
        }
        receipt_core = _share_safe_receipt_core(receipt_core, staging=staging, identities=identities)
        receipt_id = "sha256:" + hashlib.sha256(canonical_json_document(receipt_core)).hexdigest()
        _write_json(staging / "receipt.json", {"receiptId": receipt_id, **receipt_core})
        _emit(progress, ProgressStage.VERIFY, "completed", f"verified {len(artifacts)} artifacts")
        _emit(progress, ProgressStage.PUBLISH, "started", "atomically publishing solved project")
        _rename_directory_no_replace(staging, output)
        _fsync_directory(output.parent)
        published = True
        # A UI callback must not be able to turn an already committed success
        # into an apparent processing failure.
        try:
            _emit(progress, ProgressStage.COMPLETE, "completed", "E2E project published")
        except Exception:
            pass
        return E2EResult(
            success=True,
            code="E2E_SUCCEEDED",
            state=E2EState.SOLVED,
            output_directory=str(output),
            evidence_directory=None,
            receipt_path=str(output / "receipt.json"),
            product_paths=tuple(str(output / path.relative_to(staging)) for path in product_paths_staged),
            preview_paths=tuple(str(output / path.relative_to(staging)) for path in preview_paths_staged),
            passed_light_paths=tuple(str(path) for path in screening.passed),
            excluded_light_paths=tuple(str(path) for path in screening.excluded),
        )
    except Exception:
        if not published:
            remove_tree(staging)
        try:
            _emit(progress, ProgressStage.FAILED, "failed", "E2E execution aborted without publishing success")
        except Exception:
            pass
        raise
    finally:
        # Idempotent: the success path closed the pool after registration;
        # a gate failure returned early with the pool still open.
        frame_runner.close()


__all__ = [
    "DrizzleOptions",
    "E2EError",
    "E2ERequest",
    "E2EResult",
    "E2EState",
    "E2E_VERSION",
    "ExplicitDecision",
    "ExplicitSelection",
    "IntegrationMode",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressStage",
    "ReviewApproval",
    "bind_review_approval_selections",
    "parse_explicit_selection",
    "run_e2e",
]
