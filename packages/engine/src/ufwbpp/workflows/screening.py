"""Screening the Lights before pixel work: quality gate evidence, solver hints, drizzle sampling, admission, registration anchors and region weight maps."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from lightframeqc.analysis import analyze_measurements
from lightframeqc.config import QcConfig
from lightframeqc.measure import measure_paths
from lightframeqc.models import FrameResult, GateDisposition
from lightframeqc.parallel import FrameRunner
from lightframeqc.quality_gate import GatePolicy, evaluate_quality_gate
from lightframeqc.source_extraction import NONDETERMINISTIC_WARNING, cached_extraction_self_test

from ..blink.session import BlinkEvidence, blink_evidence
from ..quality.cache import quality_cache_directory
from ..quality.review_preview import (
    MAX_REVIEW_PREVIEWS,
    MAX_TOTAL_PREVIEW_BYTES,
    REVIEW_DIRECTORY,
    bounded_review_preview,
    review_preview_name,
)
from ..selection import FrameSelectionFeatures, SelectionDecision, decide, extract_features
from ..selection.policy import selection_receipt
from ..selection.region import RegionWeightMap, region_weight_maps
from ..stacking.normalization import StellarScaleHint
from .common import _emit, _write_json
from .contracts import IntegrationMode, ProgressStage, E2EError, DrizzleOptions, E2ERequest, ProgressCallback
from .review import _apply_explicit_selection, _apply_review_approvals
from .solve import _sky_separation_degrees, _SolverHints
from .sources import _E2ESources


def _write_review_previews(
    staging: Path, qc_dir: Path, results: Sequence[FrameResult]
) -> dict[str, str]:
    """Bounded previews of every frame the gate did not pass, keyed by path.

    The desktop shows them with the run's result so an excluded frame can be
    judged without opening the raw file; values are paths relative to the run.
    """

    previews: dict[str, str] = {}
    total = 0
    for index, result in enumerate(results):
        gate = result.quality_gate
        if gate is None or gate.disposition is GateDisposition.PASS or result.thumbnail_path is None:
            continue
        if len(previews) >= MAX_REVIEW_PREVIEWS:
            break
        data = bounded_review_preview(result.thumbnail_path)
        if data is None or total + len(data) > MAX_TOTAL_PREVIEW_BYTES:
            continue
        destination = qc_dir / REVIEW_DIRECTORY / review_preview_name(index, result.path)
        destination.parent.mkdir(exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        total += len(data)
        previews[result.path] = destination.relative_to(staging).as_posix()
    return previews


def _screening_summary(
    results: Sequence[FrameResult],
    passed: Sequence[Path],
    approved_review_paths: Sequence[str],
    review_previews: Mapping[str, str],
    explicit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Counts plus the frames that needed a decision, for receipts and the desktop.

    With an explicit selection every user drop is listed (``reason``
    ``USER_DROP`` plus its blink flag codes) and so is every keep that
    overrode an EXCLUDE flag (``USER_KEEP_OVERRIDE``), next to the gate's own
    non-PASS frames, so the result page shows both kinds of decision.
    """

    counts = {disposition.value: 0 for disposition in GateDisposition}
    frames: list[dict[str, Any]] = []
    passed_set = {str(path) for path in passed}
    explicit_by_path = {
        item["path"]: item for item in (explicit or {}).get("frames", []) if isinstance(item, Mapping)
    }
    for result in sorted(results, key=lambda item: item.path):
        gate = result.quality_gate
        disposition = gate.disposition.value if gate is not None else "HARD_FAIL"
        counts[disposition] += 1
        resolved = str(Path(result.path).resolve(strict=True))
        decision = explicit_by_path.get(resolved)
        reason: str | None = None
        if decision is not None:
            if decision["decision"] == "DROP":
                reason = "USER_DROP"
            elif "EXCLUDE_FLAGS" in decision.get("overrode", ()):
                reason = "USER_KEEP_OVERRIDE"
        if gate is not None and gate.disposition is GateDisposition.PASS and reason is None:
            continue
        frames.append(
            {
                "path": resolved,
                "disposition": disposition,
                "admitted": resolved in passed_set or resolved in approved_review_paths,
                "summary": gate.summary if gate is not None else "; ".join(result.reasons) or "not measured",
                "evidence": [item.message for item in gate.evidence][:8] if gate is not None else list(result.reasons)[:8],
                "starCount": result.star_count,
                "reviewPreview": review_previews.get(result.path),
                **({"reason": reason, "flags": list(decision["flags"])} if reason is not None and decision is not None else {}),
            }
        )
    return {
        "counts": counts,
        "admitted": len(passed_set),
        "excluded": len(results) - len(passed_set),
        "frames": frames,
    }


def _qc_manifest(
    staging: Path,
    groups: Sequence[Mapping[str, Any]],
    results: Sequence[FrameResult],
    config: QcConfig,
    policy: GatePolicy,
    review_previews: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    frame_payloads: list[dict[str, Any]] = []
    for result in results:
        payload = result.serializable()
        thumbnail = payload.get("thumbnailPath")
        if isinstance(thumbnail, str):
            try:
                payload["thumbnailPath"] = Path(thumbnail).relative_to(staging).as_posix()
            except ValueError:
                payload["thumbnailPath"] = None
        payload["reviewPreviewPath"] = (review_previews or {}).get(result.path)
        frame_payloads.append(payload)
    counts = {
        disposition.value: sum(
            result.quality_gate is not None
            and result.quality_gate.disposition is disposition
            for result in results
        )
        for disposition in GateDisposition
    }
    return {
        "schemaVersion": 1,
        "stage": "quality-control",
        "gatePolicy": policy.serializable(),
        "gatePolicyDigest": policy.canonical_digest(),
        "qcConfig": config.serializable(),
        "counts": counts,
        "groups": list(groups),
        "frames": frame_payloads,
    }


def _inferred_solver_hints(
    request: E2ERequest,
    passed_results: Sequence[FrameResult],
) -> _SolverHints:
    explicit_coordinates = request.ra_hint_degrees is not None
    coordinate_pairs = [
        (result.metadata.ra_degrees, result.metadata.dec_degrees)
        for result in passed_results
        if result.metadata.ra_degrees is not None
        and result.metadata.dec_degrees is not None
        and math.isfinite(result.metadata.ra_degrees)
        and math.isfinite(result.metadata.dec_degrees)
    ]
    metadata_center: tuple[float, float] | None = None
    coordinate_spread_degrees: float | None = None
    if coordinate_pairs:
        ra_radians = np.radians([item[0] for item in coordinate_pairs])
        metadata_ra = float(
            np.degrees(
                math.atan2(float(np.mean(np.sin(ra_radians))), float(np.mean(np.cos(ra_radians))))
            )
            % 360.0
        )
        metadata_dec = float(np.median([item[1] for item in coordinate_pairs]))
        metadata_center = (metadata_ra, metadata_dec)
        coordinate_spread_degrees = max(
            _sky_separation_degrees(metadata_center, item) for item in coordinate_pairs
        )

    coordinate_evidence: dict[str, Any] = {
        "explicit": (
            {
                "raDegrees": request.ra_hint_degrees,
                "decDegrees": request.dec_hint_degrees,
            }
            if explicit_coordinates
            else None
        ),
        "metadata": (
            {
                "raDegrees": metadata_center[0],
                "decDegrees": metadata_center[1],
                "sampleCount": len(coordinate_pairs),
                "maximumSeparationFromConsensusDegrees": coordinate_spread_degrees,
            }
            if metadata_center is not None
            else None
        ),
        "consistency": "NOT_COMPARABLE",
    }
    if explicit_coordinates and metadata_center is not None:
        assert request.ra_hint_degrees is not None and request.dec_hint_degrees is not None
        explicit_to_metadata = _sky_separation_degrees(
            (request.ra_hint_degrees, request.dec_hint_degrees), metadata_center
        )
        consistency_radius = request.search_radius_degrees or 15.0
        coordinate_evidence["explicitToMetadataDegrees"] = explicit_to_metadata
        coordinate_evidence["consistencyRadiusDegrees"] = consistency_radius
        if explicit_to_metadata <= consistency_radius:
            ra = request.ra_hint_degrees
            dec = request.dec_hint_degrees
            provenance = "explicit+NINA/FITS-consistent"
            coordinate_evidence["consistency"] = "CONSISTENT"
            coordinate_evidence["selection"] = "explicit"
        else:
            # Conflicting pointing hints are never allowed to lock the solver.
            # A consensus from the actual acquired frames is a better seed; the
            # astrometry.net adapter still has an unconstrained fallback.
            ra, dec = metadata_center
            provenance = "NINA/FITS-header;conflicting-explicit-coordinate-ignored"
            coordinate_evidence["consistency"] = "CONFLICT"
            coordinate_evidence["selection"] = "metadata-consensus"
    elif explicit_coordinates:
        ra = request.ra_hint_degrees
        dec = request.dec_hint_degrees
        provenance = "explicit"
        coordinate_evidence["selection"] = "explicit"
    elif metadata_center is not None:
        ra, dec = metadata_center
        provenance = "NINA/FITS-header"
        coordinate_evidence["selection"] = "metadata-consensus"
    else:
        ra = None
        dec = None
        provenance = "blind"
        coordinate_evidence["selection"] = "blind"

    fov_samples: list[dict[str, Any]] = []
    for result in passed_results:
        header = result.metadata.header
        try:
            focal_length = float(
                header.get(
                    "FOCALLEN",
                    header.get("FOCAL", header.get("FOCALLENGTH", 0.0)),
                )
            )
            pixel_size_x_um = float(
                header.get(
                    "XPIXSZ",
                    header.get("PIXSIZE1", header.get("PIXSIZE", 0.0)),
                )
            )
            pixel_size_y_um = float(
                header.get(
                    "YPIXSZ",
                    header.get("PIXSIZE2", header.get("PIXSIZE", pixel_size_x_um)),
                )
            )
        except (TypeError, ValueError):
            continue
        width = result.metadata.width
        height = result.metadata.height
        values = (focal_length, pixel_size_x_um, pixel_size_y_um)
        if (
            focal_length <= 0
            or pixel_size_x_um <= 0
            or pixel_size_y_um <= 0
            or width <= 0
            or height <= 0
            or not all(math.isfinite(value) for value in values)
        ):
            continue
        width_degrees = math.degrees(
            2.0 * math.atan((width * pixel_size_x_um / 1000.0) / (2.0 * focal_length))
        )
        height_degrees = math.degrees(
            2.0 * math.atan((height * pixel_size_y_um / 1000.0) / (2.0 * focal_length))
        )
        if not (
            math.isfinite(width_degrees)
            and math.isfinite(height_degrees)
            and 0 < width_degrees <= 180
            and 0 < height_degrees <= 180
        ):
            continue
        fov_samples.append(
            {
                "widthDegrees": width_degrees,
                "heightDegrees": height_degrees,
                "widthPixels": width,
                "heightPixels": height,
                "focalLengthMm": focal_length,
                "pixelSizeXMicrons": pixel_size_x_um,
                "pixelSizeYMicrons": pixel_size_y_um,
            }
        )

    derived_fov: float | None = None
    fov_spread_ratio: float | None = None
    if fov_samples:
        width_estimates = np.asarray(
            [sample["widthDegrees"] for sample in fov_samples], dtype=np.float64
        )
        median_width = float(np.median(width_estimates))
        consistent = width_estimates[
            (width_estimates >= median_width * 0.8)
            & (width_estimates <= median_width * 1.2)
        ]
        if consistent.size:
            derived_fov = float(np.median(consistent))
            fov_spread_ratio = float(np.max(consistent) / np.min(consistent))

    explicit_fov = request.field_of_view_degrees
    fov_evidence: dict[str, Any] = {
        "explicitWidthDegrees": explicit_fov,
        "derivedWidthDegrees": derived_fov,
        "derivedSampleCount": len(fov_samples),
        "derivedSpreadRatio": fov_spread_ratio,
        "derivation": "2*atan((imagePixels*effectivePixelMicrons/1000)/(2*focalLengthMm))",
        "binningAssumption": "FITS XPIXSZ/YPIXSZ describe effective image pixels",
        "consistency": "NOT_COMPARABLE",
    }
    if explicit_fov is not None and derived_fov is not None:
        explicit_to_derived_ratio = explicit_fov / derived_fov
        fov_evidence["explicitToDerivedRatio"] = explicit_to_derived_ratio
        if 0.7 <= explicit_to_derived_ratio <= 1.3:
            field_of_view = explicit_fov
            fov_evidence["consistency"] = "CONSISTENT"
            fov_evidence["selection"] = "explicit"
        else:
            # A wrong scale hint is much more damaging than a missing one.  Use
            # image geometry and acquisition optics, and retain the conflict as
            # auditable evidence rather than constraining solve-field to it.
            field_of_view = derived_fov
            provenance += "+derived-FOV;conflicting-explicit-FOV-ignored"
            fov_evidence["consistency"] = "CONFLICT"
            fov_evidence["selection"] = "derived"
    elif explicit_fov is not None:
        field_of_view = explicit_fov
        fov_evidence["selection"] = "explicit-unverified"
    elif derived_fov is not None:
        field_of_view = derived_fov
        provenance += "+derived-FOV"
        fov_evidence["selection"] = "derived"
    else:
        field_of_view = None
        fov_evidence["selection"] = "unconstrained"
    return _SolverHints(
        ra,
        dec,
        field_of_view,
        request.search_radius_degrees,
        provenance,
        {
            "coordinates": coordinate_evidence,
            "fieldOfView": fov_evidence,
            "fallback": "astrometry.net removes all hints on its bounded fallback attempt",
        },
    )


def _drizzle_sampling_evidence(
    results: Sequence[FrameResult], options: DrizzleOptions
) -> dict[str, Any]:
    """Turn QC morphology into a fail-closed drizzle sampling decision."""

    direct_fwhm: list[float] = []
    hfr_fwhm: list[float] = []
    pixel_scales: list[float] = []
    for result in results:
        fwhm = result.features.median_fwhm_native_pixels
        if fwhm is not None and math.isfinite(fwhm) and fwhm > 0:
            direct_fwhm.append(float(fwhm))
        hfr = result.features.nina_hfr_pixels
        if hfr is not None and math.isfinite(hfr) and hfr > 0:
            # For a circular Gaussian, FWHM is exactly twice the half-flux
            # radius.  This is fallback evidence only; measured PSF FWHM wins.
            hfr_fwhm.append(float(2.0 * hfr))
        header = result.metadata.header
        try:
            focal_length_mm = float(header.get("FOCALLEN", header.get("FOCAL")))
            pixel_size_um = float(
                header.get("XPIXSZ", header.get("PIXSIZE1", header.get("PIXSIZE")))
            )
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(focal_length_mm)
            and focal_length_mm > 0
            and math.isfinite(pixel_size_um)
            and pixel_size_um > 0
        ):
            pixel_scales.append(206.265 * pixel_size_um / focal_length_mm)

    direct_median = float(np.median(direct_fwhm)) if direct_fwhm else None
    hfr_median = float(np.median(hfr_fwhm)) if hfr_fwhm else None
    if direct_median is not None and hfr_median is not None:
        fwhm_values = [direct_median, hfr_median]
        provenance = "QC_NATIVE_PSF_FWHM+NINA_HFR_GAUSSIAN_EQUIVALENT"
    elif direct_median is not None:
        fwhm_values = [direct_median]
        provenance = "QC_NATIVE_PSF_FWHM"
    elif hfr_median is not None:
        fwhm_values = [hfr_median]
        provenance = "NINA_HFR_GAUSSIAN_EQUIVALENT"
    else:
        fwhm_values = []
        provenance = "UNKNOWN"
    median_fwhm = float(np.median(fwhm_values)) if fwhm_values else None
    pixel_scale = float(np.median(pixel_scales)) if pixel_scales else None
    evidence: dict[str, Any] = {
        "medianNativeFwhmPixels": median_fwhm,
        "qcMedianNativeFwhmPixels": direct_median,
        "ninaHfrEquivalentFwhmPixels": hfr_median,
        "pixelScaleArcsec": pixel_scale,
        "seeingFwhmArcsec": (
            median_fwhm * pixel_scale
            if median_fwhm is not None and pixel_scale is not None
            else None
        ),
        "sampleCount": max(len(direct_fwhm), len(hfr_fwhm)),
        "qcFwhmSampleCount": len(direct_fwhm),
        "ninaHfrSampleCount": len(hfr_fwhm),
        "provenance": provenance,
        "maximumFwhmForUpsamplingPixels": options.maximum_fwhm_for_upsampling_pixels,
    }
    # The sampling evidence is advisory, as in WBPP: the user chose the scale;
    # the receipt records whether upsampling is expected to gain resolution.
    evidence["advisory"] = True
    if options.scale == 1:
        evidence["status"] = "NOT_APPLICABLE"
        return evidence
    if median_fwhm is None:
        evidence["status"] = "UNKNOWN_SAMPLING"
        evidence["recommendation"] = (
            "QC could not establish the native PSF sampling; the benefit of "
            f"{options.scale}x drizzle is unknown"
        )
        return evidence
    if (
        direct_median is not None
        and hfr_median is not None
        and (direct_median >= options.maximum_fwhm_for_upsampling_pixels)
        != (hfr_median >= options.maximum_fwhm_for_upsampling_pixels)
    ):
        evidence["status"] = "CONFLICTING_SAMPLING"
        evidence["recommendation"] = (
            "QC PSF FWHM and NINA HFR disagree across the adequately-sampled boundary"
        )
        return evidence
    if median_fwhm >= options.maximum_fwhm_for_upsampling_pixels:
        evidence["status"] = "WELL_SAMPLED"
        evidence["recommendation"] = (
            f"QC median native FWHM is {median_fwhm:.3f} px, at or above the "
            f"{options.maximum_fwhm_for_upsampling_pixels:.3f} px threshold: "
            f"{options.scale}x drizzle mainly gains sub-pixel sampling, not resolution"
        )
        return evidence
    evidence["status"] = "PASS_UNDERSAMPLED"
    return evidence


@dataclass
class _Screening:
    """The Light admission of a run and the evidence behind it.

    ``passed``/``excluded``/``screening`` and the selection decisions change
    when a counterfactual integration pass removes a harmful frame.
    """

    source_extraction: dict[str, Any]
    qc_dir: Path
    measurements: list[Any]
    frame_results: list[Any]
    review_previews: dict[str, str]
    approved_review_paths: set[str]
    blink: BlinkEvidence | None
    explicit_block: dict[str, Any] | None
    selection_features: list[FrameSelectionFeatures]
    selection_decisions: list[SelectionDecision]
    selection_confidence: dict[str, float]
    selection_region_maps: dict[str, RegionWeightMap]
    gate_by_path: dict[str, Any]
    passed: tuple[Path, ...]
    excluded: tuple[Path, ...]
    screening: dict[str, Any]
    passed_results: list[Any]
    solver_hints: _SolverHints
    drizzle_sampling: dict[str, Any]


def _admitted_lights(
    request: E2ERequest,
    lights: Sequence[Path],
    gate_by_path: Mapping[str, Any],
    approved_review_paths: set[str],
    explicit_admitted: set[str],
    selection_decisions: Sequence[SelectionDecision],
) -> tuple[Path, ...]:
    if request.selection.explicit:
        return tuple(path for path in lights if str(path) in explicit_admitted)
    if request.selection.unattended:
        decision_by_path = {item.path: item for item in selection_decisions}
        return tuple(
            path
            for path in lights
            if decision_by_path.get(str(path)) is not None and decision_by_path[str(path)].admitted
        )
    return tuple(
        path
        for path in lights
        if gate_by_path.get(str(path)) is not None
        and (
            gate_by_path[str(path)].disposition is GateDisposition.PASS
            or str(path) in approved_review_paths
        )
    )


def _screen_lights(
    request: E2ERequest,
    sources: _E2ESources,
    staging: Path,
    frame_runner: FrameRunner,
    progress: ProgressCallback | None,
) -> _Screening:
    """Measure and gate every Light, then admit frames by the request's
    route: the legacy gate (PASS plus approved REVIEW), an explicit (blink)
    selection or the unattended selection policy."""

    lights = sources.lights
    _emit(progress, ProgressStage.QUALITY_CONTROL, "started", "measuring and gating Light frames")
    # Star counts, quality weights and therefore the masters depend on SEP's
    # extraction being reproducible; the verdict of its self-test is part of
    # the receipt and a failing build is called out here.
    source_extraction = cached_extraction_self_test()
    if not source_extraction["deterministic"]:
        notice = (
            f"{NONDETERMINISTIC_WARNING}: the installed sep {source_extraction['sepVersion']} "
            "returns different objects for identical input; run-to-run identical products "
            "cannot be claimed on this machine"
        )
        warnings.warn(notice, RuntimeWarning, stacklevel=3)
        _emit(progress, ProgressStage.QUALITY_CONTROL, "running", notice)
    qc_dir = staging / "qc"
    qc_dir.mkdir()
    qc_timings: dict[str, float] = {}
    qc_cache_stats: dict[str, int] = {}
    qc_measurement_stats: dict[str, Any] = {}
    qc_measurement_cache_stats: dict[str, int] = {}
    qc_started = perf_counter()
    measurements = measure_paths(
        [str(path) for path in lights],
        qc_dir,
        request.qc_config,
        workers=request.workers,
        stats=qc_measurement_stats,
        runner=frame_runner,
        cache_directory=quality_cache_directory(),
        cache_stats=qc_measurement_cache_stats,
    )
    qc_timings["measurementSeconds"] = perf_counter() - qc_started
    _emit(progress, ProgressStage.QUALITY_CONTROL, "running", f"measured {len(measurements)} Light frames; analyzing star fields")
    qc_started = perf_counter()
    qc_analysis_stats: dict[str, Any] = {}
    groups, frame_results = analyze_measurements(
        measurements, request.qc_config,
        cache_directory=quality_cache_directory(), cache_stats=qc_cache_stats,
        workers=request.workers, stats=qc_analysis_stats, runner=frame_runner,
    )
    qc_timings["analysisSeconds"] = perf_counter() - qc_started
    qc_started = perf_counter()
    evaluate_quality_gate(frame_results, measurements, request.gate_policy)
    qc_timings["gateSeconds"] = perf_counter() - qc_started
    review_previews = _write_review_previews(staging, qc_dir, frame_results)
    qc_manifest = _qc_manifest(
        staging, groups, frame_results, request.qc_config, request.gate_policy, review_previews
    )
    qc_manifest["timings"] = qc_timings
    qc_manifest["measurement"] = qc_measurement_stats
    qc_manifest["analysis"] = qc_analysis_stats
    qc_manifest["analysisCache"] = qc_cache_stats
    qc_manifest["measurementCache"] = qc_measurement_cache_stats
    approved_review_paths, approval_request_digest, approval_evidence = _apply_review_approvals(
        request=request,
        identities=sources.identities,
        results=frame_results,
    )
    qc_manifest["manualReviewApprovals"] = {
        "defaultDisposition": (
            "DECIDED_BY_EXPLICIT_SELECTION"
            if request.selection.explicit
            else "DECIDED_BY_SELECTION_POLICY"
            if request.selection.unattended
            else "EXCLUDED"
        ),
        "requestDigest": approval_request_digest,
        "gatePolicyDigest": request.gate_policy.canonical_digest(),
        "accepted": approval_evidence,
    }
    # The blink evidence (flags, reference, scores) is recomputed from this
    # run's own quality pass, so a run with an explicit selection is
    # self-describing without the blink session that produced it.
    blink: BlinkEvidence | None = None
    explicit_block: dict[str, Any] | None = None
    explicit_admitted: set[str] = set()
    if request.selection.explicit:
        qc_started = perf_counter()
        blink = blink_evidence(
            frame_results,
            measurements,
            flags_policy=request.blink_flags_policy,
            gate_policy=request.gate_policy,
            qc_config=request.qc_config,
        )
        explicit_admitted, explicit_block = _apply_explicit_selection(
            request=request,
            identities=sources.identities,
            results=frame_results,
            evidence=blink,
        )
        qc_timings["blinkSeconds"] = perf_counter() - qc_started
        qc_manifest["explicitSelection"] = {
            "selectionDigest": explicit_block["selectionDigest"],
            "flagsPolicyDigest": explicit_block["flagsPolicyDigest"],
            "counts": dict(explicit_block["counts"]),
        }
    selection_features: list[FrameSelectionFeatures] = []
    selection_decisions: list[SelectionDecision] = []
    selection_confidence: dict[str, float] = {}
    selection_region_maps: dict[str, RegionWeightMap] = {}
    if request.selection.unattended:
        selection_features = extract_features(
            frame_results,
            measurements,
            night_boundary_hours=request.gate_policy.night_boundary_hours,
            observing_timezone=request.qc_config.observing_timezone,
        )
        if request.selection.region_weights:
            # Region weight maps come from the QC grid alone, so they are
            # built before the decisions that depend on them.
            selection_region_maps = region_weight_maps(frame_results)
        selection_decisions = decide(
            selection_features,
            request.selection,
            approved_paths=approved_review_paths,
            region_maps=selection_region_maps,
        )
        selection_confidence = {
            item.path: item.weight_multiplier for item in selection_decisions if item.admitted
        }
        qc_manifest["selection"] = selection_receipt(
            request.selection,
            selection_features,
            selection_decisions,
            region_maps=selection_region_maps,
        )
    _write_json(qc_dir / "manifest.json", qc_manifest)
    if blink is not None:
        _write_json(qc_dir / "blink.json", blink.serializable())
    gate_by_path = {
        str(Path(result.path).resolve(strict=True)): result.quality_gate
        for result in frame_results
    }
    passed = _admitted_lights(
        request, lights, gate_by_path, approved_review_paths, explicit_admitted, selection_decisions
    )
    passed_set = set(passed)
    excluded = tuple(path for path in lights if path not in passed_set)
    screening = _screening_summary(
        frame_results, passed, approved_review_paths, review_previews, explicit_block
    )
    passed_results = [
        result for result in frame_results if Path(result.path).resolve(strict=True) in passed_set
    ]
    solver_hints = _inferred_solver_hints(request, passed_results)
    drizzle_sampling = (
        _drizzle_sampling_evidence(passed_results, request.drizzle)
        if request.integration_mode is IntegrationMode.DRIZZLE
        else {}
    )
    _emit(
        progress,
        ProgressStage.QUALITY_CONTROL,
        "completed",
        (
            f"{len(passed)} kept by the explicit selection "
            f"({explicit_block['counts']['overriddenExcludeFlags']} kept against an EXCLUDE flag); "
            f"{len(excluded)} dropped"
            if request.selection.explicit and explicit_block is not None
            else f"{len(passed)} admitted by selection policy {request.selection.policy} "
            f"({sum(1 for item in selection_decisions if item.admitted and item.confidence < 1.0)} "
            f"with reduced weight); {len(excluded)} excluded"
            if request.selection.unattended
            else f"{len(passed)} admitted ({len(approved_review_paths)} explicitly approved REVIEW); "
            f"{len(excluded)} REVIEW/HARD_FAIL excluded"
        ),
    )
    return _Screening(
        source_extraction=source_extraction,
        qc_dir=qc_dir,
        measurements=measurements,
        frame_results=frame_results,
        review_previews=review_previews,
        approved_review_paths=approved_review_paths,
        blink=blink,
        explicit_block=explicit_block,
        selection_features=selection_features,
        selection_decisions=selection_decisions,
        selection_confidence=selection_confidence,
        selection_region_maps=selection_region_maps,
        gate_by_path=gate_by_path,
        passed=passed,
        excluded=excluded,
        screening=screening,
        passed_results=passed_results,
        solver_hints=solver_hints,
        drizzle_sampling=drizzle_sampling,
    )


def _panels_below_registration_minimum(
    frame_results: Sequence[Any], passed: Sequence[Path]
) -> list[str]:
    """Target/filter panels with fewer than two admitted Lights."""

    passed_set = set(passed)
    panel_counts: dict[tuple[str, str], list[int]] = {}
    for frame in frame_results:
        panel_key = (frame.metadata.target.strip().upper(), frame.metadata.filter_name.strip().upper())
        counts = panel_counts.setdefault(panel_key, [0, 0])
        counts[1] += 1
        counts[0] += int(Path(frame.path).resolve(strict=True) in passed_set)
    return [
        f"{target} / {filter_name}: {admitted} of {total} Light frames admitted"
        for (target, filter_name), (admitted, total) in sorted(panel_counts.items())
        if admitted < 2
    ]


def _registration_anchor_exclusions(
    request: E2ERequest, screening: _Screening
) -> set[str]:
    """Lights that may be integrated but must never anchor the registration.

    A manually admitted REVIEW frame never anchors: a cloud-covered frame
    scores well on sharp noise blobs and would leave every real frame
    unregistered.  The same holds for a Light the blink reviewer kept
    against the gate.  A frame with a blink flag (moonlit or hazy sky, low
    star retention, extinction, atypical background shape, ...) never
    anchors either: the master inherits the reference's background, and the
    previous rule's preference for the lowest sky picked exactly such
    cloud-dimmed frames (NGC 6822, 2026-09-22).  Only when nothing else
    remains do the excluded frames compete.
    """

    excluded = set(screening.approved_review_paths)
    if screening.blink is None:
        screening.blink = blink_evidence(
            screening.frame_results,
            screening.measurements,
            flags_policy=request.blink_flags_policy,
            gate_policy=request.gate_policy,
            qc_config=request.qc_config,
        )
    excluded.update(
        str(Path(frame.path).resolve(strict=True)) for frame in screening.blink.flags if frame.flags
    )
    if request.selection.explicit:
        passed_text = {str(path) for path in screening.passed}
        excluded.update(
            path
            for path, gate in screening.gate_by_path.items()
            if path in passed_text and (gate is None or gate.disposition is not GateDisposition.PASS)
        )
    return excluded


def _staged_pixel_maps(
    light_subset: Sequence[Path],
    source_aliases: Mapping[str, Path],
    transforms: Mapping[str, Sequence[Sequence[float]]],
    registration: Any,
) -> tuple[dict[str, Any], dict[str, float], dict[str, StellarScaleHint]]:
    """Transforms, quality weights and stellar-scale hints keyed by the staged
    pixel inputs instead of the original Lights."""

    def original_of(staged: Path) -> str:
        return str(source_aliases[str(staged)].resolve(strict=True))

    transforms_map = {str(staged): transforms[original_of(staged)] for staged in light_subset}
    weights_map = {
        str(staged): registration.quality_weights[original_of(staged)] for staged in light_subset
    }
    light_by_original = {original_of(staged): staged for staged in light_subset}
    hints_map: dict[str, StellarScaleHint] = {}
    for original, staged in light_by_original.items():
        hint = registration.stellar_scale_hints[original]
        reference_staged = light_by_original.get(str(Path(hint.reference_path).resolve(strict=True)))
        if reference_staged is None:
            raise E2EError(
                "STELLAR_SCALE_HINT_REFERENCE_MISMATCH",
                "registration stellar scale reference is absent from the pixel Light set",
                path=hint.reference_path,
            )
        hints_map[str(staged)] = replace(
            hint,
            source_path=str(staged),
            reference_path=str(reference_staged),
        )
    return transforms_map, weights_map, hints_map


def _registered_region_maps(
    light_subset: Sequence[Path],
    source_aliases: Mapping[str, Path],
    screening: _Screening,
    transforms: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, RegionWeightMap]:
    """Region maps resampled from the QC reference frame into the pipeline's.

    The QC grid lives in the QC reference's preview frame; registered Lights
    live in the pixel pipeline's reference frame.  For a Light ``f`` a
    registered pixel maps to the QC reference through ``S Q_f S^-1 T_f^-1``
    (``T_f``: source to pipeline reference in native pixels, ``Q_f``: source
    to QC reference in preview pixels, ``S``: preview to native scale), so
    each frame's own matrices carry it across, including a meridian flip
    between nights.
    """

    result_by_original = {
        str(Path(frame.path).resolve(strict=True)): frame for frame in screening.frame_results
    }
    measurement_by_original = {
        str(Path(item.metadata.path).resolve(strict=True)): item for item in screening.measurements
    }
    registered: dict[str, RegionWeightMap] = {}
    for staged in light_subset:
        original = str(source_aliases[str(staged)].resolve(strict=True))
        qc_map = screening.selection_region_maps.get(original)
        if qc_map is None:
            continue
        frame = result_by_original.get(original)
        measurement = measurement_by_original.get(original)
        transform = transforms.get(original)
        if (
            frame is None
            or measurement is None
            or transform is None
            or frame.registration.matrix is None
            or measurement.preview_scale_x is None
            or measurement.preview_scale_y is None
            or not measurement.metadata.width
            or not measurement.metadata.height
        ):
            continue
        try:
            scale = np.diag([float(measurement.preview_scale_x), float(measurement.preview_scale_y), 1.0])
            qc_matrix = np.asarray(frame.registration.matrix, dtype=np.float64)
            pipeline_matrix = np.asarray(transform, dtype=np.float64)
            if qc_matrix.shape != (3, 3) or pipeline_matrix.shape != (3, 3):
                continue
            composite = scale @ qc_matrix @ np.linalg.inv(scale) @ np.linalg.inv(pipeline_matrix)
            registered[str(staged)] = qc_map.transformed(
                composite,
                int(measurement.metadata.height),
                int(measurement.metadata.width),
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    return registered


def _counterfactual_exclusions(
    harmful: Sequence[SelectionDecision],
    *,
    light_subset: Sequence[Path],
    source_aliases: Mapping[str, Path],
    frame_results: Sequence[Any],
    registration: Any,
    admitted_at_start: int,
    soft_exclusion_fraction_guard: float,
) -> tuple[list[str], dict[str, str]]:
    """Which confirmed-harmful frames an integration pass may remove.

    Frames that other frames' normalization hints reference, and frames whose
    removal would leave a panel below the registration minimum, stay; the
    cumulative exclusions stay under the soft-exclusion guard.  Returns the
    removable originals and why each other harmful frame was kept.
    """

    reference_originals = {
        str(Path(hint.reference_path).resolve(strict=True))
        for hint in registration.stellar_scale_hints.values()
    }
    removable: list[str] = []
    kept_reasons: dict[str, str] = {}
    for item in harmful:
        if item.path in reference_originals:
            kept_reasons[item.path] = "normalization reference of its group"
            continue
        removable.append(item.path)

    def panel_of(original: str) -> tuple[str, str] | None:
        result_for = next(
            (frame for frame in frame_results if str(Path(frame.path).resolve(strict=True)) == original),
            None,
        )
        if result_for is None:
            return None
        return (
            result_for.metadata.target.strip().upper(),
            result_for.metadata.filter_name.strip().upper(),
        )

    remaining_by_panel: dict[tuple[str, str], int] = {}
    for staged in light_subset:
        original = str(source_aliases[str(staged)].resolve(strict=True))
        key = panel_of(original)
        if key is None:
            continue
        remaining_by_panel[key] = remaining_by_panel.get(key, 0) + (0 if original in removable else 1)
    for item in harmful:
        if item.path not in removable:
            continue
        key = panel_of(item.path)
        if remaining_by_panel.get(key, 0) < 2:
            removable.remove(item.path)
            remaining_by_panel[key] = remaining_by_panel.get(key, 0) + 1
            kept_reasons[item.path] = "fewer than 2 Lights would remain in its panel"
    already_removed = admitted_at_start - len(light_subset)
    allowed = int(soft_exclusion_fraction_guard * admitted_at_start) - already_removed
    if len(removable) > max(0, allowed):
        for path in removable[max(0, allowed):]:
            kept_reasons[path] = "cumulative exclusions would exceed the soft-exclusion guard"
        removable = removable[: max(0, allowed)]
    return removable, kept_reasons
