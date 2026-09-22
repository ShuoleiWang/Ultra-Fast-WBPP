"""Unattended selection: parameters, features, policy and the counterfactual oracle."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from lightframeqc.models import (
    Confidence,
    Decision,
    EvidenceFamily,
    EvidenceSeverity,
    FrameFeatures,
    FrameMetadata,
    FrameResult,
    GateDisposition,
    QualityEvidence,
    QualityGateResult,
    RegistrationMetrics,
)

from openastroflow_engine.recipe import Recipe, RecipeError
from openastroflow_engine.reference.selection_oracle_numpy import leave_one_out_depths
from openastroflow_engine.selection import (
    CounterfactualReport,
    FrameCounterfactual,
    FrameSelectionFeatures,
    LeaveOneOutAccumulator,
    SelectionParameters,
    TileObservation,
    annotate_with_counterfactual,
    confirmed_harmful,
    decide,
    exclude_confirmed,
    extract_features,
)
from openastroflow_engine.selection.policy import selection_receipt


def _features(path: str, **overrides) -> FrameSelectionFeatures:
    base = dict(
        path=path,
        group_id="group-a",
        filter_name="L",
        target="NGC 7331",
        night_id="2026-09-12",
        is_reference=False,
        star_count=1200,
        registration_ok=True,
        registration_estimated=True,
        registration_rms=0.4,
        matched_stars=400,
        transparency=0.98,
        extra_extinction_mag=0.02,
        fwhm_native=4.0,
        ellipticity=0.1,
        orientation_coherence=0.2,
        spatial_dimming_p90_mag=0.05,
        star_completeness=0.98,
        detected_source_ratio=0.99,
        overlap_fraction=0.99,
        occlusion_area=0.0,
        occlusion_density=1.0,
        boundary_support=0.0,
        background_support=0.0,
        background_z=0.3,
        noise_z=0.2,
        gate_disposition="PASS",
        hard_fail_codes=(),
        review_codes=(),
        fwhm_night_ratio=1.0,
        fwhm_group_ratio=1.0,
    )
    base.update(overrides)
    return FrameSelectionFeatures(**base)


def test_parameters_validate_serialize_and_map_from_recipe() -> None:
    parameters = SelectionParameters.from_mapping(
        {"policy": "unattended-v1", "priority": "resolution", "aggressiveness": "conservative"}
    )
    assert parameters.unattended and parameters.profile.psf_exponent == 2.0
    serialized = parameters.serializable()
    # The recipe form round-trips exactly; derived thresholds live in describe().
    assert SelectionParameters.from_mapping(serialized) == parameters
    assert "fwhmNightExclusionRatio" not in serialized
    assert parameters.describe()["fwhmNightExclusionRatio"] == pytest.approx(1.3 * 1.15)
    json.dumps(parameters.describe())
    with pytest.raises(ValueError):
        SelectionParameters.from_mapping({"policy": "whatever"})
    with pytest.raises(ValueError):
        SelectionParameters.from_mapping({"unknownKey": 1})
    assert SelectionParameters.from_mapping(None).policy == "legacy-gate"
    recipe = Recipe.from_dict({"selection": {"policy": "include-all"}})
    assert recipe.selection.present and recipe.selection.parameters.policy == "include-all"
    assert "selection" in recipe.serializable()
    # The CLI round-trips recipes through serializable(); it must parse again.
    assert Recipe.from_dict(recipe.serializable()).selection.parameters.policy == "include-all"
    assert "selection" not in Recipe.from_dict({}).serializable()
    with pytest.raises(RecipeError):
        Recipe.from_dict({"selection": {"priority": "sharp"}})


def test_policy_decisions_follow_guards_gray_zone_and_priority() -> None:
    unattended = SelectionParameters(policy="unattended-v1", priority="balanced")
    frames = [
        _features("/a/pass.fits"),
        _features(
            "/a/thin-evidence.fits",
            gate_disposition="REVIEW",
            review_codes=("GATE_INSUFFICIENT_NIGHT_BASELINE",),
        ),
        _features(
            "/a/soft-psf.fits",
            gate_disposition="REVIEW",
            review_codes=("GATE_FOCUS_SEEING_REVIEW",),
            fwhm_native=5.6,
            fwhm_night_ratio=1.4,
            fwhm_group_ratio=1.4,
        ),
        _features(
            "/a/opaque.fits",
            gate_disposition="REVIEW",
            review_codes=("GATE_TEMPORAL_EXTINCTION_STRONG",),
            transparency=0.40,
        ),
        _features(
            "/a/trail.fits",
            gate_disposition="HARD_FAIL",
            hard_fail_codes=("GATE_COHERENT_TRAILING_HARD",),
        ),
        _features(
            "/a/cloud-review.fits",
            gate_disposition="REVIEW",
            review_codes=("GATE_SPATIAL_DIMMING_REVIEW",),
            spatial_dimming_p90_mag=0.25,
        ),
    ]
    by_path = {item.path: item for item in decide(frames, unattended)}
    assert by_path["/a/pass.fits"].action == "KEEP" and by_path["/a/pass.fits"].confidence == 1.0
    thin = by_path["/a/thin-evidence.fits"]
    assert thin.action == "KEEP" and thin.confidence == 0.5
    assert thin.reasons[0][0] == "SEL_KEEP_INSUFFICIENT_EVIDENCE_GATE_INSUFFICIENT_NIGHT_BASELINE"
    # Balanced priority tolerates a 1.4x FWHM (cut-off 1.6); resolution does not.
    assert by_path["/a/soft-psf.fits"].action == "KEEP"
    assert by_path["/a/soft-psf.fits"].confidence == pytest.approx(0.6)
    resolution = {
        item.path: item
        for item in decide(frames, replace(unattended, priority="resolution"))
    }
    assert resolution["/a/soft-psf.fits"].action == "EXCLUDE"
    assert resolution["/a/soft-psf.fits"].reasons[0][0] == "SEL_PSF_BEYOND_NIGHT_CUTOFF"
    assert resolution["/a/soft-psf.fits"].soft is True
    opaque = by_path["/a/opaque.fits"]
    assert opaque.action == "EXCLUDE" and not opaque.soft
    assert opaque.reasons[0][0] == "SEL_GUARD_TRANSPARENCY_BELOW_NORMALIZATION_FLOOR"
    assert by_path["/a/trail.fits"].action == "EXCLUDE"
    assert by_path["/a/trail.fits"].reasons[0][0] == "SEL_GUARD_GATE_COHERENT_TRAILING_HARD"
    cloud = by_path["/a/cloud-review.fits"]
    assert cloud.action == "KEEP" and cloud.confidence == pytest.approx(0.6)
    # Balanced priority: the soft frame keeps 0.6 confidence times a PSF factor
    # (5.6 / median 4.0)^-2 = 0.51, so its weight multiplier is ~0.31.
    assert by_path["/a/soft-psf.fits"].psf_factor == pytest.approx((5.6 / 4.0) ** -2)
    assert by_path["/a/soft-psf.fits"].weight_multiplier == pytest.approx(0.6 * (5.6 / 4.0) ** -2)
    assert by_path["/a/pass.fits"].psf_factor == pytest.approx(1.0)
    # An approval promotes a review frame to full weight; guards still win.
    approved = {
        item.path: item
        for item in decide(frames, unattended, approved_paths=["/a/cloud-review.fits", "/a/trail.fits"])
    }
    assert approved["/a/cloud-review.fits"].confidence == 1.0
    assert approved["/a/trail.fits"].action == "EXCLUDE"
    # Diagnostic policy keeps everything the guards allow at full weight.
    everything = {item.path: item for item in decide(frames, replace(unattended, policy="include-all"))}
    assert everything["/a/soft-psf.fits"].confidence == 1.0
    assert everything["/a/opaque.fits"].action == "EXCLUDE"
    with pytest.raises(ValueError):
        decide(frames, SelectionParameters())


def test_uniform_cloud_and_soft_pass_frames_follow_the_policy_not_the_gate() -> None:
    balanced = SelectionParameters(policy="unattended-v1", priority="balanced")
    frames = [
        _features("/a/pass.fits"),
        _features("/a/pass2.fits"),
        # A uniform 0.65 transparency loss: the gate hard-fails it as a
        # multi-family cloud, the policy keeps it at reduced weight because
        # the normalization scale models it.
        _features(
            "/a/thin-cloud.fits",
            gate_disposition="HARD_FAIL",
            hard_fail_codes=("GATE_MULTI_FAMILY_CLOUD_HARD",),
            review_codes=("GATE_SOURCE_RETENTION_REVIEW",),
            transparency=0.64,
            extra_extinction_mag=0.56,
        ),
        # A PASS frame (the gate saw nothing) whose PSF is 1.7x the night's best
        # is excluded by the priority cut-off (balanced: 1.6).
        _features("/a/soft-pass.fits", fwhm_native=6.8, fwhm_night_ratio=1.7, fwhm_group_ratio=1.7),
        # Slightly soft PASS frame: kept, weighted down by the PSF factor only.
        _features("/a/dew.fits", fwhm_native=5.2, fwhm_night_ratio=1.3, fwhm_group_ratio=1.3),
    ]
    by_path = {item.path: item for item in decide(frames, balanced)}
    cloud = by_path["/a/thin-cloud.fits"]
    assert cloud.action == "KEEP" and cloud.confidence == 0.5
    assert cloud.reasons[0][0] == "SEL_KEEP_DOWNWEIGHTED_GATE_SOURCE_RETENTION_REVIEW"
    assert any(code == "SEL_KEEP_DOWNWEIGHTED_GATE_MULTI_FAMILY_CLOUD_HARD" for code, _ in cloud.reasons)
    soft = by_path["/a/soft-pass.fits"]
    assert soft.action == "EXCLUDE" and soft.soft and soft.reasons[0][0] == "SEL_PSF_BEYOND_NIGHT_CUTOFF"
    dew = by_path["/a/dew.fits"]
    assert dew.action == "KEEP" and dew.confidence == 1.0
    assert dew.psf_factor == pytest.approx((5.2 / 4.0) ** -2)
    # Depth priority never excludes for PSF and applies no PSF factor.
    depth = {item.path: item for item in decide(frames, replace(balanced, priority="depth"))}
    assert depth["/a/soft-pass.fits"].action == "KEEP" and depth["/a/soft-pass.fits"].psf_factor == 1.0
    # A real opaque cloud still trips the relative transparency guard.
    opaque = _features(
        "/a/opaque.fits",
        gate_disposition="HARD_FAIL",
        hard_fail_codes=("GATE_MULTI_FAMILY_CLOUD_HARD",),
        transparency=0.40,
        extra_extinction_mag=0.9,
    )
    decided = {item.path: item for item in decide(frames + [opaque], balanced)}
    assert decided["/a/opaque.fits"].action == "EXCLUDE"
    codes = {code for code, _ in decided["/a/opaque.fits"].reasons}
    assert {"SEL_GUARD_TRANSPARENCY_BELOW_NORMALIZATION_FLOOR", "SEL_GUARD_EXTRA_EXTINCTION"} <= codes


def test_soft_exclusion_guard_downgrades_mass_rule_exclusions() -> None:
    parameters = SelectionParameters(policy="unattended-v1", priority="resolution")
    frames = [_features("/a/good.fits")] + [
        _features(
            f"/a/soft-{index}.fits",
            gate_disposition="REVIEW",
            review_codes=("GATE_FOCUS_SEEING_REVIEW",),
            fwhm_night_ratio=1.5,
            fwhm_group_ratio=1.5,
        )
        for index in range(3)
    ]
    decisions = decide(frames, parameters)
    soft = [item for item in decisions if item.path != "/a/good.fits"]
    assert all(item.action == "KEEP" and item.confidence == 0.5 for item in soft)
    assert all(item.reasons[-1][0] == "SEL_SOFT_EXCLUSION_GUARD" for item in soft)
    receipt = selection_receipt(parameters, frames, decisions)
    assert receipt["counts"] == {"KEEP": 1, "EXCLUDE": 0, "KEEP_REDUCED_WEIGHT": 3}
    json.dumps(receipt)


def _result(
    path: str,
    *,
    observed: str,
    fwhm: float,
    disposition: GateDisposition = GateDisposition.PASS,
    evidence: tuple[QualityEvidence, ...] = (),
    reference: str | None = None,
) -> FrameResult:
    return FrameResult(
        path=path,
        group_id="g1",
        reference_path=reference,
        decision=Decision.KEEP,
        confidence=Confidence.HIGH,
        reasons=[],
        warnings=[],
        registration=RegistrationMetrics(ok=True, matched_stars=300, match_fraction=0.8, rms_pixels=0.5, matrix=[[1, 0, 0], [0, 1, 0]]),
        features=FrameFeatures(
            transparency_ratio=0.97,
            median_fwhm_native_pixels=fwhm,
            nightly_extinction_residual=0.01,
        ),
        metadata=FrameMetadata(path=path, filter_name="L", target="M31", observed_at=datetime.fromisoformat(observed).replace(tzinfo=timezone.utc)),
        star_count=900,
        quality_gate=QualityGateResult(disposition=disposition, evidence=list(evidence), summary="x"),
    )


def test_extract_features_assigns_nights_and_fwhm_ratios(tmp_path: Path) -> None:
    paths = [str(tmp_path / f"f{index}.fits") for index in range(4)]
    for path in paths:
        Path(path).write_bytes(b"x")
    review = QualityEvidence(
        code="GATE_FOCUS_SEEING_REVIEW",
        family=EvidenceFamily.MORPHOLOGY,
        severity=EvidenceSeverity.REVIEW,
        message="soft",
    )
    results = [
        _result(paths[0], observed="2026-09-12T22:00:00", fwhm=4.0, reference=paths[0]),
        _result(paths[1], observed="2026-09-13T01:00:00", fwhm=4.2),
        _result(paths[2], observed="2026-09-13T22:30:00", fwhm=6.0, disposition=GateDisposition.REVIEW, evidence=(review,)),
        _result(paths[3], observed="2026-09-13T23:30:00", fwhm=4.1),
    ]
    features = extract_features(results, night_boundary_hours=12.0)
    assert [item.night_id for item in features][:2] == [features[0].night_id] * 2
    assert features[2].night_id == features[3].night_id != features[0].night_id
    assert features[0].is_reference is True
    assert features[2].fwhm_night_ratio == pytest.approx(6.0 / 4.1)
    # Four or more frames: the robust best is the 10th percentile floored at the minimum.
    assert features[2].fwhm_group_ratio == pytest.approx(6.0 / max(4.0, np.percentile([4.0, 4.2, 6.0, 4.1], 10)))
    assert features[2].review_codes == ("GATE_FOCUS_SEEING_REVIEW",)
    assert features[2].gate_disposition == "REVIEW"
    json.dumps([item.serializable() for item in features])


def _synthetic_stack(seed: int = 3, *, cloud: bool = False):
    rng = np.random.default_rng(seed)
    frames, rows, width = 6, 128, 96
    sigma = np.array([1.0, 1.0, 1.0, 4.0, 1.0, 1.0])
    samples = np.empty((frames, rows, width), dtype=np.float32)
    yy, xx = np.indices((rows, width), dtype=np.float64)
    for index in range(frames):
        frame = 100.0 + rng.normal(0.0, sigma[index], (rows, width))
        if cloud and index == 5:
            # A cloud-like patch: non-polynomial background structure that only
            # this frame carries (a linear gradient would be absorbed by the
            # second-order background fit by design).
            frame += 8.0 * np.exp(-(((yy - 64.0) ** 2 + (xx - 48.0) ** 2) / (2 * 15.0**2)))
        samples[index] = frame
    # Stars shared by every frame, at the same positions.
    for cy, cx in ((20, 30), (70, 60), (100, 15)):
        samples[:, cy - 1 : cy + 2, cx - 1 : cx + 2] += 400.0
    accepted = np.ones(samples.shape, dtype=bool)
    accepted[2, 40:44, 10:20] = False  # a rejected patch
    # Equal weights on purpose: under inverse-variance weights a noisy frame is
    # merely useless, never harmful, so the sign test below needs an
    # over-weighted noisy frame.
    weights = np.full(frames, 1.0 / frames)
    return samples, accepted, weights


def _integrate(samples, accepted, weights):
    numerator = np.sum(np.where(accepted, samples, 0.0) * weights[:, None, None], axis=0)
    denominator = np.sum(accepted * weights[:, None, None], axis=0)
    return (numerator / denominator).astype(np.float32)


def _accumulate(samples, accepted, weights):
    integrated = _integrate(samples, accepted, weights)
    accumulator = LeaveOneOutAccumulator(
        [f"/f/{index}.fits" for index in range(6)], minimum_blocks=12, background_segment_blocks=2,
        statistics_rows=32,
    )
    tile_starts = (0, 32, 64, 96)
    for first_row in tile_starts:
        accumulator(
            TileObservation(
                first_row=first_row,
                samples=samples[:, first_row : first_row + 32],
                accepted=accepted[:, first_row : first_row + 32],
                weights=weights,
                integrated=integrated[first_row : first_row + 32],
            )
        )
    assert accumulator.tiles_used == 4
    return accumulator, integrated, tile_starts


def test_leave_one_out_accumulator_matches_naive_reference_and_signs() -> None:
    samples, accepted, weights = _synthetic_stack()
    accumulator, integrated, tile_starts = _accumulate(samples, accepted, weights)
    for tile_index, first_row in enumerate(tile_starts):
        expected = leave_one_out_depths(
            samples[:, first_row : first_row + 32],
            accepted[:, first_row : first_row + 32],
            weights,
            integrated[first_row : first_row + 32],
        )
        np.testing.assert_allclose(accumulator._depth[tile_index], expected, rtol=1e-9, atol=1e-12)
    report = accumulator.finalize(fwhm_by_frame=[4.0, 4.0, 4.0, 4.0, 5.0, 4.0], bootstrap=50)
    assert report.tiles_used == 4 and len(report.frames) == 6
    noisy = report.frames[3]
    clean = report.frames[0]
    assert noisy.delta_depth_mag is not None and clean.delta_depth_mag is not None
    # With equal weights, removing the 4x noisier frame lowers the block noise;
    # removing a clean one raises it.
    assert noisy.delta_depth_mag > 0 > clean.delta_depth_mag
    assert report.frames[4].delta_fwhm_px is not None and report.frames[4].delta_fwhm_px > 0
    assert report.frames[0].delta_fwhm_px is not None and report.frames[0].delta_fwhm_px < 0
    json.dumps(report.serializable())


def test_leave_one_out_background_proxy_flags_a_cloud_patch() -> None:
    samples, accepted, weights = _synthetic_stack(cloud=True)
    accumulator, _, _ = _accumulate(samples, accepted, weights)
    report = accumulator.finalize(bootstrap=50)
    patch = report.frames[5]
    assert patch.delta_background_sigma is not None and patch.delta_background_sigma > 0
    assert patch.delta_depth_mag is not None and patch.delta_depth_mag > 0
    assert all(
        frame.delta_background_sigma is not None
        and frame.delta_background_sigma < patch.delta_background_sigma
        for frame in report.frames[:5]
    )


def test_counterfactual_annotation_suggests_confirmed_exclusions() -> None:
    parameters = SelectionParameters(policy="unattended-v1")
    frames = [_features("/a/harm.fits"), _features("/a/good.fits", gate_disposition="REVIEW", review_codes=("GATE_INSUFFICIENT_COHORT",))]
    decisions = decide(frames, parameters)
    report = CounterfactualReport(
        mode="test",
        frames=(
            FrameCounterfactual(0, "/a/harm.fits", 0.2, 0.02, (0.01, 0.03), 0.0, (-0.01, 0.01), 0.0),
            FrameCounterfactual(1, "/a/good.fits", 0.2, -0.03, (-0.04, -0.02), -0.01, (-0.02, -0.001), -0.01),
        ),
        tiles_observed=40,
        tiles_used=40,
        block=8,
        sigma_block_all=1.0,
        background_rms_all_sigma=0.1,
        fwhm_all_px=4.0,
        bootstrap=10,
    )
    # Two frames cannot define an outlier threshold: nothing is confirmed.
    annotated = {item.path: item for item in annotate_with_counterfactual(decisions, report, parameters)}
    assert annotated["/a/harm.fits"].suggestion is None
    # With a homogeneous group behind it, the harmful frame stands out.
    peers = tuple(
        FrameCounterfactual(index, f"/a/peer{index}.fits", 0.2, -0.03, (-0.04, -0.02), -0.01, (-0.02, -0.001), -0.01)
        for index in range(2, 6)
    )
    report = replace(report, frames=report.frames + peers)
    annotated = {item.path: item for item in annotate_with_counterfactual(decisions, report, parameters)}
    assert annotated["/a/harm.fits"].suggestion == "EXCLUDE_CONFIRMED_HARMFUL"
    # Too few statistics tiles: report only.
    few = {item.path: item for item in annotate_with_counterfactual(decisions, replace(report, tiles_used=4), parameters)}
    assert few["/a/harm.fits"].suggestion is None
    assert annotated["/a/harm.fits"].action == "KEEP"  # report-only in this version
    assert annotated["/a/good.fits"].suggestion == "RESTORE_FULL_WEIGHT"
    assert annotate_with_counterfactual(decisions, None, parameters) == decisions


def test_occlusion_area_is_excluded_until_region_maps_exist() -> None:
    parameters = SelectionParameters(policy="unattended-v1")
    blocked = _features(
        "/a/blocked.fits",
        gate_disposition="REVIEW",
        review_codes=("GATE_OCCLUSION_REVIEW", "GATE_SOURCE_RETENTION_REVIEW"),
        occlusion_area=0.25,
        occlusion_density=0.10,
    )
    clean = [_features(f"/a/p{index}.fits") for index in range(6)]
    decided = {item.path: item for item in decide([*clean, blocked], parameters)}
    assert decided["/a/blocked.fits"].action == "EXCLUDE" and decided["/a/blocked.fits"].soft
    assert decided["/a/blocked.fits"].reasons[0][0] == "SEL_OCCLUSION_AREA_EXCLUDED"
    with_maps = {item.path: item for item in decide([blocked], replace(parameters, region_weights=True))}
    assert with_maps["/a/blocked.fits"].action == "KEEP" and with_maps["/a/blocked.fits"].confidence == 0.5


def test_confirmed_harmful_frames_are_excluded_by_the_counterfactual() -> None:
    parameters = SelectionParameters(policy="unattended-v1", priority="resolution")
    frames = [_features("/a/good.fits"), _features("/a/soft.fits"), _features("/a/blocked.fits")] + [
        _features(f"/a/peer{index}.fits") for index in range(4)
    ]
    decisions = decide(frames, parameters)
    report = CounterfactualReport(
        mode="test",
        frames=(
            FrameCounterfactual(0, "/a/good.fits", 0.3, -0.03, (-0.04, -0.02), -0.01, (-0.02, -0.005), -0.01),
            # Broadens the PSF: harmful under resolution priority even though it adds depth.
            FrameCounterfactual(1, "/a/soft.fits", 0.3, -0.02, (-0.03, -0.01), -0.01, (-0.02, -0.005), 0.25),
            FrameCounterfactual(2, "/a/blocked.fits", 0.4, 0.2, (0.05, 0.3), 0.4, (0.3, 0.5), -0.1),
            *(
                FrameCounterfactual(3 + index, f"/a/peer{index}.fits", 0.3, -0.03, (-0.04, -0.02), -0.01, (-0.02, -0.005), -0.01)
                for index in range(4)
            ),
        ),
        tiles_observed=60, tiles_used=60, block=8, sigma_block_all=1.0, background_rms_all_sigma=0.2, fwhm_all_px=4.0, bootstrap=10,
    )
    annotated = annotate_with_counterfactual(decisions, report, parameters)
    harmful = {item.path for item in confirmed_harmful(annotated)}
    assert harmful == {"/a/soft.fits", "/a/blocked.fits"}
    balanced = annotate_with_counterfactual(decisions, report, replace(parameters, priority="balanced"))
    assert {item.path for item in confirmed_harmful(balanced)} == {"/a/blocked.fits"}
    excluded = {item.path: item for item in exclude_confirmed(annotated, ["/a/blocked.fits"])}
    assert excluded["/a/blocked.fits"].action == "EXCLUDE"
    assert excluded["/a/blocked.fits"].reasons[-1][0] == "SEL_COUNTERFACTUAL_CONFIRMED_HARMFUL"
    assert excluded["/a/soft.fits"].action == "KEEP"


def test_normalization_modelled_codes_cost_no_confidence() -> None:
    """A dim night or mild extra extinction is modelled by normalization, so the
    frame keeps full weight; a genuine defect code still enters the gray zone."""

    from openastroflow_engine.selection.policy import decide
    from openastroflow_engine.selection import SelectionParameters

    features = [
        _features(
            f"/tmp/night{i}.fits",
            transparency=0.7,
            gate_disposition="REVIEW",
            review_codes=("GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW",),
        )
        for i in range(4)
    ]
    features.append(
        _features(
            "/tmp/hazy.fits",
            transparency=0.6,
            extra_extinction_mag=0.2,
            gate_disposition="REVIEW",
            review_codes=("GATE_TEMPORAL_EXTINCTION_REVIEW", "GATE_NOISE_REVIEW"),
        )
    )
    decisions = {item.path: item for item in decide(features, SelectionParameters(policy="unattended-v1"))}
    for i in range(4):
        item = decisions[f"/tmp/night{i}.fits"]
        assert item.action == "KEEP" and item.confidence == 1.0
        assert item.reasons[0][0] == "SEL_KEEP_NORMALIZED_GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW"
    hazy = decisions["/tmp/hazy.fits"]
    assert hazy.action == "KEEP" and hazy.confidence == 0.6
    assert [code for code, _ in hazy.reasons] == [
        "SEL_KEEP_NORMALIZED_GATE_TEMPORAL_EXTINCTION_REVIEW",
        "SEL_KEEP_DOWNWEIGHTED_GATE_NOISE_REVIEW",
    ]


def test_background_harm_does_not_remove_a_frame_that_deepens_the_master() -> None:
    from openastroflow_engine.selection import SelectionParameters
    from openastroflow_engine.selection.counterfactual import CounterfactualReport, FrameCounterfactual
    from openastroflow_engine.selection.policy import SelectionDecision, annotate_with_counterfactual

    parameters = SelectionParameters(policy="unattended-v1")
    paths = [f"/tmp/f{i}.fits" for i in range(8)]
    decisions = [SelectionDecision(path, "KEEP", 1.0, ()) for path in paths]

    def frame(index: int, depth: float, background: float) -> FrameCounterfactual:
        return FrameCounterfactual(
            index=index,
            path=paths[index],
            weight=0.125,
            delta_depth_mag=depth,
            delta_depth_ci=(depth - 0.004, depth + 0.004),
            delta_background_sigma=background,
            delta_background_ci=(background - 0.005, background + 0.005),
            delta_fwhm_px=0.0,
        )

    frames = [frame(i, -0.03, 0.0) for i in range(6)]
    frames.append(frame(6, -0.03, 0.40))  # deeper master, but background structure: kept
    frames.append(frame(7, 0.0, 0.40))  # no depth gain, background structure: harmful
    report = CounterfactualReport(
        mode="test", frames=tuple(frames), tiles_observed=40, tiles_used=40, block=8,
        sigma_block_all=1.0, background_rms_all_sigma=0.2, fwhm_all_px=4.0, bootstrap=10,
    )
    annotated = {item.path: item for item in annotate_with_counterfactual(decisions, report, parameters)}
    assert annotated[paths[6]].suggestion is None
    assert annotated[paths[7]].suggestion == "EXCLUDE_CONFIRMED_HARMFUL"


@pytest.mark.parametrize("band_rows", [1, 7, 20, 32, 65])
def test_counterfactual_is_invariant_to_io_bands_and_masks_nan_samples(band_rows: int) -> None:
    """A unit region map and arbitrary I/O bands must preserve the oracle."""
    rng = np.random.default_rng(7)
    samples = rng.normal(100, 1, (6, 1027, 96)).astype(np.float32)
    samples[0] = 100 + (samples[0] - 100) * 4
    samples[1, ::8, ::8] = np.nan
    accepted = np.isfinite(samples)
    weights = np.ones(6)
    integrated = np.nanmean(samples, axis=0)
    paths = [f"frame-{i}" for i in range(6)]

    def run(rows: int, region: bool):
        accumulator = LeaveOneOutAccumulator(paths)
        for y in range(0, samples.shape[1], rows):
            tile = samples[:, y:y + rows]
            accumulator(TileObservation(
                y, tile, accepted[:, y:y + rows], weights, integrated[y:y + rows],
                np.ones_like(tile) if region else None,
            ))
        report = accumulator.finalize(bootstrap=30)
        assert report.tiles_used == 16
        decisions = decide([_features(path) for path in paths], SelectionParameters(policy="unattended-v1"))
        harmful = confirmed_harmful(annotate_with_counterfactual(decisions, report, SelectionParameters(policy="unattended-v1")))
        assert [item.path for item in harmful] == [paths[0]]
        return report.serializable()

    assert run(band_rows, True) == run(64, False)
