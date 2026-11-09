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

from ufwbpp.recipe import Recipe, RecipeError
from ufwbpp.reference.selection_oracle_numpy import leave_one_out_depths
from ufwbpp.selection import (
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
from ufwbpp.selection.policy import selection_receipt


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
        _features(
            "/a/off-target.fits",
            gate_disposition="HARD_FAIL",
            hard_fail_codes=("GATE_FIELD_OFFSET_HARD",),
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
    off_target = by_path["/a/off-target.fits"]
    assert off_target.action == "EXCLUDE" and not off_target.soft
    assert off_target.reasons[0][0] == "SEL_GUARD_GATE_FIELD_OFFSET_HARD"
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
        for item in decide(
            frames, unattended, approved_paths=["/a/cloud-review.fits", "/a/trail.fits", "/a/off-target.fits"]
        )
    }
    assert approved["/a/cloud-review.fits"].confidence == 1.0
    assert approved["/a/trail.fits"].action == "EXCLUDE"
    assert approved["/a/off-target.fits"].action == "EXCLUDE"
    # Diagnostic policy keeps everything the guards allow at full weight.
    everything = {item.path: item for item in decide(frames, replace(unattended, policy="include-all"))}
    assert everything["/a/soft-psf.fits"].confidence == 1.0
    assert everything["/a/opaque.fits"].action == "EXCLUDE"
    assert everything["/a/off-target.fits"].action == "EXCLUDE"
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

    from ufwbpp.selection.policy import decide
    from ufwbpp.selection import SelectionParameters

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
    from ufwbpp.selection import SelectionParameters
    from ufwbpp.selection.counterfactual import CounterfactualReport, FrameCounterfactual
    from ufwbpp.selection.policy import SelectionDecision, annotate_with_counterfactual

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


# --- unattended-v2 ---------------------------------------------------------

from lightframeqc.blink_flags import (  # noqa: E402
    BlinkFlagPolicy,
    BlinkFrameInput,
    channel_statistics,
    compute_flags,
)

from ufwbpp.selection import (  # noqa: E402
    V2_THRESHOLDS,
    blink_context,
    confirmed_harmful_blocks,
    decide_v2,
)
from ufwbpp.selection.unattended_v2 import suspect_channels  # noqa: E402

NGC6822_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "light-frame-qc"
    / "tests"
    / "fixtures"
    / "blink"
    / "ngc6822-features.json"
)


def _blink_input(item: FrameSelectionFeatures, **overrides) -> BlinkFrameInput:
    base = dict(
        path=item.path,
        channel_id=item.group_id,
        target=item.target,
        filter_name=item.filter_name,
        night=item.night_id,
        observed_at=None,
        airmass=1.2,
        sky=800.0,
        sky_mad=20.0,
        star_count=item.star_count,
        transparency=item.transparency,
        extinction_mag=item.extra_extinction_mag,
        fwhm_native=item.fwhm_native,
        ellipticity=item.ellipticity,
        eccentricity=None,
        registration_ok=item.registration_ok,
        matched_stars=item.matched_stars,
        registration_rms=item.registration_rms,
        overlap=item.overlap_fraction,
        background_shape=0.05,
        gate_disposition=item.gate_disposition,
        gate_codes=(*item.hard_fail_codes, *item.review_codes),
    )
    base.update(overrides)
    return BlinkFrameInput(**base)


def _v2(
    features: list[FrameSelectionFeatures],
    blink_overrides: dict[str, dict] | None = None,
    priority: str = "balanced",
    **kwargs,
):
    inputs = [_blink_input(item, **(blink_overrides or {}).get(item.path, {})) for item in features]
    flags = compute_flags(inputs, BlinkFlagPolicy())
    context = blink_context(inputs, flags, channel_statistics(inputs))
    parameters = SelectionParameters(policy="unattended-v2", priority=priority)
    return decide_v2(features, parameters, context, **kwargs)


# Feature paths in a run are resolved paths, the form the blink context keys
# by; bare names are already that form on every OS ("/v2/x" is not on Windows).
def _clean_group(count: int = 10, night: str = "2026-09-12") -> list[FrameSelectionFeatures]:
    return [_features(f"v2-clean-{index}.fits", night_id=night) for index in range(count)]


def test_v2_keeps_uniform_extinction_down_to_a_quarter_of_the_clean_transparency() -> None:
    frames = _clean_group() + [
        _features("v2-thin.fits", transparency=0.40, extra_extinction_mag=0.98, detected_source_ratio=0.6,
                  gate_disposition="REVIEW", review_codes=("GATE_SOURCE_RETENTION_REVIEW",)),
        _features("v2-quarter.fits", transparency=0.26, extra_extinction_mag=1.45),
        _features("v2-thick.fits", transparency=0.20, extra_extinction_mag=1.75),
    ]
    decisions, summaries, _ = _v2(frames)
    by_path = {item.path: item for item in decisions}
    thin = by_path["v2-thin.fits"]
    # 0.40x: kept at full confidence; the noise weight carries the loss.
    assert thin.admitted and thin.confidence == 1.0 and thin.firm is False
    assert any(code == "SEL_KEEP_RETENTION_EXPLAINED" for code, _ in thin.reasons)
    assert by_path["v2-quarter.fits"].admitted
    thick = by_path["v2-thick.fits"]
    assert not thick.admitted and thick.firm
    assert thick.reasons[0][0] == "SEL_TRANSPARENCY_FLOOR"
    assert summaries["group-a"].transparency_median == pytest.approx(0.98)
    # Every v2 decision carries its value and the components the receipt shows.
    assert all(item.components is not None and item.block == "2026-09-12" for item in decisions)
    assert thin.value is not None and thin.value < by_path["v2-clean-0.fits"].value
    assert "LOW_VALUE" in thin.suspect
    record = thin.serializable()
    assert record["firm"] is False and record["components"]["transparencyRelative"] == pytest.approx(0.40 / 0.98)


def test_v2_small_cohort_is_kept_at_full_confidence() -> None:
    frames = [
        _features(f"v2-small-{index}.fits", gate_disposition="REVIEW",
                  review_codes=("GATE_INSUFFICIENT_COHORT", "GATE_INSUFFICIENT_NIGHT_BASELINE"))
        for index in range(6)
    ]
    decisions, summaries, blocks = _v2(frames)
    assert all(item.admitted and item.confidence == 1.0 for item in decisions)
    assert all(
        any(code == "SEL_KEEP_INSUFFICIENT_EVIDENCE_GATE_INSUFFICIENT_COHORT" for code, _ in item.reasons)
        for item in decisions
    )
    # Six equal frames: the completeness factor is 1 and nothing is suspect.
    assert all(item.weight_multiplier == 1.0 for item in decisions)
    assert blocks == []
    channels = suspect_channels(decisions, frames, summaries, blocks, {})
    assert channels == []


def test_v2_soft_exclusions_stop_at_the_minimum_count_guard() -> None:
    # Five good frames, three with a wide PSF and three of negligible value:
    # six soft exclusions against a guard of max(2, ceil(11 / 2)) = 6 kept, so
    # the five lowest-valued frames go and the best remaining wide frame is
    # kept at half confidence.
    frames = (
        [_features(f"v2-good-{index}.fits") for index in range(5)]
        + [_features(f"v2-wide-{index}.fits", fwhm_native=6.0) for index in range(3)]
        + [
            _features(f"v2-faint-{index}.fits", transparency=0.26, extra_extinction_mag=1.46, star_count=700)
            for index in range(3)
        ]
    )
    decisions, _, _ = _v2(frames)
    by_path = {item.path: item for item in decisions}
    excluded = [item for item in decisions if not item.admitted]
    assert len(excluded) == 5 and all(not item.firm and item.soft for item in excluded)
    assert {item.reasons[0][0] for item in excluded if "faint" in item.path} == {"SEL_VALUE_NEGLIGIBLE"}
    assert {item.reasons[0][0] for item in excluded if "wide" in item.path} == {"SEL_PSF_BEYOND_MEDIAN"}
    guarded = [item for item in decisions if any(code == "SEL_SOFT_GUARD_KEPT" for code, _ in item.reasons)]
    assert len(guarded) == 1 and "wide" in guarded[0].path
    assert guarded[0].admitted and guarded[0].confidence == 0.5
    assert all(by_path[f"v2-good-{index}.fits"].admitted for index in range(5))
    # A two-frame group never loses a frame to a soft rule.
    pair = [_features("v2-p0.fits"), _features("v2-p1.fits", fwhm_native=7.0)]
    pair_decisions, _, _ = _v2(pair)
    assert all(item.admitted for item in pair_decisions)


def test_v2_overlap_and_psf_cut_offs_exclude_firmly_and_softly() -> None:
    frames = _clean_group() + [
        _features("v2-offset.fits", overlap_fraction=0.88),
        _features("v2-wide.fits", fwhm_native=4.0 * 1.35),
        _features("v2-wide-ok.fits", fwhm_native=4.0 * 1.25),
        _features("v2-moon.fits", star_count=600),
    ]
    overrides = {"v2-moon.fits": {"sky": 800.0 * 1.9}}
    decisions, _, _ = _v2(frames, overrides)
    by_path = {item.path: item for item in decisions}
    offset = by_path["v2-offset.fits"]
    assert not offset.admitted and offset.firm and offset.reasons[0][0] == "SEL_GUARD_OVERLAP"
    wide = by_path["v2-wide.fits"]
    assert not wide.admitted and not wide.firm and wide.reasons[0][0] == "SEL_PSF_BEYOND_MEDIAN"
    # Inside the cut-off ``balanced`` weights by noise alone; ``resolution``
    # adds (F_median / F) ** 2 below its own 1.2 cut-off.
    ok = by_path["v2-wide-ok.fits"]
    assert ok.admitted and ok.psf_factor == 1.0
    sharp_decisions, _, _ = _v2(
        frames + [_features("v2-wide-slight.fits", fwhm_native=4.0 * 1.15)], overrides, priority="resolution"
    )
    sharp = {item.path: item for item in sharp_decisions}
    assert not sharp["v2-wide-ok.fits"].admitted
    assert sharp["v2-wide-slight.fits"].psf_factor == pytest.approx(1.15**-2)
    # The blink combined rule (sky >= 1.6x and source ratio <= 0.60) is firm.
    moon = by_path["v2-moon.fits"]
    assert not moon.admitted and moon.firm and moon.reasons[0][0] == "SEL_NIGHT_SKY_BRIGHT"
    assert moon.components["skyRatio"] == pytest.approx(1.9)
    # The completeness factor cancels the registration score's reward for lost stars.
    assert by_path["v2-clean-0.fits"].completeness_factor == 1.0
    assert by_path["v2-clean-0.fits"].weight_multiplier == 1.0


def test_v2_multi_family_cloud_and_patchy_cloud_without_a_map_are_firm() -> None:
    frames = _clean_group() + [
        _features("v2-two-families.fits", gate_disposition="HARD_FAIL",
                  hard_fail_codes=("GATE_MULTI_FAMILY_CLOUD_HARD",), review_codes=("GATE_SPATIAL_DIMMING_REVIEW",)),
        _features("v2-patch.fits", gate_disposition="REVIEW", review_codes=("GATE_SPATIAL_DIMMING_REVIEW",),
                  spatial_dimming_p90_mag=0.3),
    ]
    decisions, _, _ = _v2(frames, region_maps_unavailable="region weight maps are not applied when drizzling onto a mosaic canvas")
    by_path = {item.path: item for item in decisions}
    assert by_path["v2-two-families.fits"].reasons[0][0] == "SEL_GUARD_GATE_MULTI_FAMILY_CLOUD_HARD"
    patch = by_path["v2-patch.fits"]
    assert not patch.admitted and patch.firm and patch.reasons[0][0] == "SEL_PATCHY_CLOUD"
    assert "mosaic canvas" in patch.reasons[0][1]


def test_v2_reproduces_the_ngc6822_screening_from_the_blink_fixture() -> None:
    """All 23 moonlit L frames firm-excluded, no dark-night frame excluded, the
    light-cloud R/G/B frames the maintainer kept are kept (one known exception)."""

    raw = json.loads(NGC6822_FIXTURE.read_text(encoding="utf-8"))
    frames = raw["frames"]
    inputs = [
        BlinkFrameInput.from_mapping({key: value for key, value in frame.items() if key not in {"id", "userDecision"}})
        for frame in frames
    ]
    flags = compute_flags(inputs, BlinkFlagPolicy())

    def features_of(item: BlinkFrameInput) -> FrameSelectionFeatures:
        hard = tuple(code for code in item.gate_codes if code.endswith("_HARD"))
        review = tuple(code for code in item.gate_codes if not code.endswith(("_HARD", "_INFO")))
        return _features(
            item.path,
            group_id=item.channel_id,
            filter_name=item.filter_name,
            target=item.target,
            night_id=item.night,
            is_reference=item.is_qc_reference,
            star_count=item.star_count,
            registration_ok=item.registration_ok,
            registration_estimated=item.transform is not None,
            registration_rms=item.registration_rms,
            matched_stars=item.matched_stars,
            transparency=item.transparency,
            extra_extinction_mag=item.extinction_mag,
            fwhm_native=item.fwhm_native,
            ellipticity=item.ellipticity,
            orientation_coherence=None,
            spatial_dimming_p90_mag=None,
            detected_source_ratio=None,
            overlap_fraction=item.overlap,
            occlusion_area=None,
            occlusion_density=None,
            gate_disposition=item.gate_disposition or "HARD_FAIL",
            hard_fail_codes=hard,
            review_codes=review,
            fwhm_night_ratio=None,
            fwhm_group_ratio=None,
        )

    features = [features_of(item) for item in inputs]
    context = blink_context(inputs, flags, channel_statistics(inputs))
    decisions, summaries, blocks = decide_v2(features, SelectionParameters(policy="unattended-v2"), context)
    by_path = {item.path: item for item in decisions}
    user = {frame["path"]: frame["userDecision"] for frame in frames}

    moonlit = [frame["path"] for frame in frames if frame["filter_name"] == "L" and frame["night"] == "2026-08-20"]
    assert len(moonlit) == 23
    assert all(not by_path[path].admitted and by_path[path].firm for path in moonlit)
    assert sum("SEL_NIGHT_SKY_BRIGHT" in {code for code, _ in by_path[path].reasons} for path in moonlit) >= 20
    dark = [frame["path"] for frame in frames if frame["filter_name"] == "L" and frame["night"] == "2026-08-17"]
    assert len(dark) == 18 and all(by_path[path].admitted for path in dark)
    kept_rgb = [
        frame["path"] for frame in frames if frame["filter_name"] in {"R", "G", "B"} and user[frame["path"]] == "KEEP"
    ]
    excluded_kept = [path for path in kept_rgb if not by_path[path].admitted]
    assert len(excluded_kept) == 1 and "G_0_CAA0.00°_2026-09-06_23-43-53" in excluded_kept[0]
    assert by_path[excluded_kept[0]].reasons[0][0] == "SEL_SOURCES_LOW"
    # The clean L baseline is the dark night; the moonlit night never forms a block.
    assert summaries["L"].clean_count == 17 and summaries["L"].sky_clean is not None
    assert not any(block.group_id == "L" for block in blocks)
    receipt = selection_receipt(SelectionParameters(policy="unattended-v2"), features, decisions)
    json.dumps(receipt)


def _night_stack(seed: int = 5):
    """Nine frames: six clean and a three-frame night with a shared bright, non-polynomial sky."""

    rng = np.random.default_rng(seed)
    frames, rows, width = 9, 128, 96
    samples = np.empty((frames, rows, width), dtype=np.float32)
    yy, xx = np.indices((rows, width), dtype=np.float64)
    structure = 6.0 * np.exp(-(((yy - 40.0) ** 2 + (xx - 70.0) ** 2) / (2 * 18.0**2))) + 3.0 * np.sin(yy / 9.0)
    for index in range(frames):
        frame = 100.0 + rng.normal(0.0, 1.0, (rows, width))
        if index >= 6:
            frame += structure
        samples[index] = frame
    for cy, cx in ((20, 30), (70, 60), (100, 15)):
        samples[:, cy - 1 : cy + 2, cx - 1 : cx + 2] += 400.0
    accepted = np.ones(samples.shape, dtype=bool)
    weights = np.full(frames, 1.0 / frames)
    return samples, accepted, weights


def test_leave_block_out_measures_a_bright_night_and_leaves_frame_results_unchanged() -> None:
    samples, accepted, weights = _night_stack()
    integrated = _integrate(samples, accepted, weights)
    paths = [f"/n/{index}.fits" for index in range(9)]

    def run(blocks):
        accumulator = LeaveOneOutAccumulator(
            paths, minimum_blocks=12, background_segment_blocks=2, statistics_rows=32, blocks=blocks
        )
        for first_row in (0, 32, 64, 96):
            accumulator(
                TileObservation(
                    first_row=first_row,
                    samples=samples[:, first_row : first_row + 32],
                    accepted=accepted[:, first_row : first_row + 32],
                    weights=weights,
                    integrated=integrated[first_row : first_row + 32],
                )
            )
        return accumulator.finalize(bootstrap=50)

    plain = run(())
    with_block = run([("g/nightB", (6, 7, 8)), ("g/nightA", (0, 1, 2))])
    # The frame results do not change when blocks are measured alongside.
    assert [frame.serializable() for frame in plain.frames] == [frame.serializable() for frame in with_block.frames]
    assert "blocks" not in plain.serializable()
    record = with_block.serializable()
    assert record["blockMode"] == "analytic-leave-block-out-block8-v1" and len(record["blocks"]) == 2
    night_b = next(block for block in with_block.blocks if block.name == "g/nightB")
    night_a = next(block for block in with_block.blocks if block.name == "g/nightA")
    assert night_b.paths == ("/n/6.fits", "/n/7.fits", "/n/8.fits") and night_b.weight == pytest.approx(3 / 9)
    # Removing the bright night flattens the background and removes more
    # structure than any one of its frames alone; removing a clean night does not.
    assert night_b.delta_background_sigma is not None and night_b.delta_background_sigma > 0
    assert night_b.delta_background_sigma > max(
        frame.delta_background_sigma for frame in with_block.frames[6:]
    )
    assert night_a.delta_background_sigma is not None and night_a.delta_background_sigma < night_b.delta_background_sigma
    thresholds = replace(V2_THRESHOLDS, minimum_counterfactual_tiles=4)
    harmful = confirmed_harmful_blocks(with_block, thresholds)
    assert set(harmful) == {"g/nightB"} and harmful["g/nightB"]["because"] == "background"
    # Too few tiles: nothing is confirmed.
    assert confirmed_harmful_blocks(with_block, V2_THRESHOLDS) == {}
    with pytest.raises(ValueError):
        LeaveOneOutAccumulator(paths, blocks=[("bad", (9,))])
