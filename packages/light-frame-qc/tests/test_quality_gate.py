from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from lightframeqc.models import (
    Confidence,
    Decision,
    EvidenceFamily,
    FileIdentity,
    FrameFeatures,
    FrameMeasurement,
    FrameMetadata,
    FrameResult,
    FrameRole,
    GateDisposition,
    RegistrationMetrics,
    Star,
)
from lightframeqc.quality_gate import GatePolicy, evaluate_quality_gate


def _identity(index: int) -> FileIdentity:
    return FileIdentity(
        sha256=f"{index + 1:064x}",
        size_bytes=10_000 + index,
        mtime_ns=1_800_000_000_000_000_000 + index,
        device=1,
        inode=100 + index,
    )


def _stars(
    *, fwhm: float = 2.0, ellipticity: float = 0.10, coherent: bool = False
) -> list[Star]:
    stars = []
    for index in range(40):
        major = fwhm / 2.3548
        minor = major * (1.0 - ellipticity)
        stars.append(
            Star(
                x=20.0 + index,
                y=30.0 + index * 0.5,
                flux=1_000.0 + index,
                peak=300.0,
                a=major,
                b=minor,
                theta=0.25 if coherent else index * 0.37,
                fwhm=fwhm,
                ellipticity=ellipticity,
                flags=0,
            )
        )
    return stars


def _cohort(count: int = 8) -> tuple[list[FrameResult], list[FrameMeasurement]]:
    results: list[FrameResult] = []
    measurements: list[FrameMeasurement] = []
    start = datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc)
    reference_path = "/gate/frame-00.fits"
    for index in range(count):
        path = f"/gate/frame-{index:02d}.fits"
        airmass = 1.0 + index * 0.06
        metadata = FrameMetadata(
            path=path,
            width=4_000,
            height=3_000,
            channels=1,
            filter_name="R",
            exposure_seconds=300.0,
            gain=0.0,
            offset=30.0,
            camera="QHY268M",
            target="NGC 7000",
            airmass=airmass,
            observed_at=start + timedelta(minutes=index * 10),
            role=FrameRole.LIGHT,
            cfa_pattern="NONE",
            binning_known=True,
        )
        identity = _identity(index)
        measurement = FrameMeasurement(
            metadata=metadata,
            stars=_stars(),
            detected_source_count=100,
            preview_width=2_000,
            preview_height=1_500,
            preview_scale_x=2.0,
            preview_scale_y=2.0,
            finite_fraction=1.0,
            dynamic_range=500.0,
            image_median=1_000.0,
            image_mad=5.0,
            status="MEASURED",
            identity=identity,
        )
        # A normal atmospheric law changes whole-frame brightness by more than
        # many naive quality filters tolerate.
        transparency = 10 ** (-0.4 * 0.55 * (airmass - 1.0))
        features = FrameFeatures(
            transparency_ratio=transparency,
            star_completeness=1.0,
            detected_source_ratio=1.0,
            spatial_dimming_p90_mag=0.0,
            overlap_fraction=1.0,
            nina_hfr_pixels=2.0,
        )
        registration = RegistrationMetrics(
            ok=True,
            matched_stars=40,
            match_fraction=0.80,
            rms_pixels=0.40,
        )
        result = FrameResult(
            path=path,
            group_id="cohort-r",
            reference_path=path if index == 0 else reference_path,
            decision=Decision.KEEP,
            confidence=Confidence.HIGH,
            reasons=[],
            warnings=[],
            registration=registration,
            features=features,
            metadata=metadata,
            star_count=100,
            identity=identity,
        )
        measurements.append(measurement)
        results.append(result)
    return results, measurements


def _codes(result: FrameResult) -> set[str]:
    assert result.quality_gate is not None
    return {item.code for item in result.quality_gate.evidence}


def test_airmass_explained_global_brightness_change_passes() -> None:
    results, measurements = _cohort(8)

    gates = evaluate_quality_gate(results, measurements)

    assert all(gate.disposition is GateDisposition.PASS for gate in gates)
    assert all(result.quality_gate is gate for result, gate in zip(results, gates))


def test_global_brightness_without_airmass_is_never_a_hard_failure_by_itself() -> None:
    results, measurements = _cohort(8)
    for result in results:
        result.metadata.airmass = None
        result.features.transparency_ratio = 1.0
    results[4].features.transparency_ratio = 0.20

    evaluate_quality_gate(results, measurements)

    assert results[4].quality_gate is not None
    assert results[4].quality_gate.disposition is not GateDisposition.HARD_FAIL
    assert "GATE_MULTI_FAMILY_CLOUD_HARD" not in _codes(results[4])


@pytest.mark.parametrize("count", [4, 7])
def test_groups_smaller_than_eight_never_pass(count: int) -> None:
    results, measurements = _cohort(count)

    gates = evaluate_quality_gate(results, measurements)

    assert all(gate.disposition is GateDisposition.REVIEW for gate in gates)
    assert all("GATE_INSUFFICIENT_COHORT" in _codes(result) for result in results)


def test_eight_healthy_frames_can_pass() -> None:
    results, measurements = _cohort(8)
    assert all(
        gate.disposition is GateDisposition.PASS
        for gate in evaluate_quality_gate(results, measurements)
    )


def test_eight_single_frame_nights_do_not_form_a_nightly_pass_baseline() -> None:
    results, measurements = _cohort(8)
    for index, result in enumerate(results):
        result.metadata.observed_at += timedelta(days=index)

    gates = evaluate_quality_gate(results, measurements)

    assert all(gate.disposition is GateDisposition.REVIEW for gate in gates)
    assert all(
        "GATE_INSUFFICIENT_NIGHT_BASELINE" in _codes(result)
        for result in results
    )


def test_uniformly_dim_second_night_is_reviewed_as_zeropoint_shift() -> None:
    results, measurements = _cohort(16)
    for index in range(8, 16):
        results[index].metadata.observed_at += timedelta(days=1)
        results[index].features.transparency_ratio *= 0.30

    evaluate_quality_gate(results, measurements)

    assert all(
        result.quality_gate is not None
        and result.quality_gate.disposition is GateDisposition.REVIEW
        and "GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW" in _codes(result)
        for result in results[8:]
    )


def test_moderate_uniform_night_shift_without_other_anomaly_passes() -> None:
    results, measurements = _cohort(16)
    shift_magnitude = 0.39427
    throughput = 10 ** (-0.4 * shift_magnitude)
    for index in range(8, 16):
        results[index].metadata.observed_at += timedelta(days=1)
        results[index].features.transparency_ratio *= throughput

    evaluate_quality_gate(results, measurements)

    assert all(
        result.quality_gate is not None
        and result.quality_gate.disposition is GateDisposition.PASS
        and "GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW" not in _codes(result)
        for result in results[8:]
    )


def test_moonlit_second_night_with_fewer_faint_detections_is_not_reviewed() -> None:
    """A clear night under a bright moon detects fewer faint sources than the
    dark reference night, although its transparency and its bright-star
    completeness are normal.  The raw count against the other night's
    reference must not review the whole night (NGC 6822, 2026-08-20)."""

    results, measurements = _cohort(16)
    for result in results:
        # No usable extinction model (small airmass span in real campaigns), so
        # nothing "explains" the count away: the same-night rule has to.
        result.metadata.airmass = None
        result.features.transparency_ratio = 1.0
    for index in range(8, 16):
        results[index].metadata.observed_at += timedelta(days=3)
        results[index].features.transparency_ratio = 0.92
        results[index].features.detected_source_ratio = 0.52
        results[index].features.star_completeness = 0.95
        measurements[index].detected_source_count = 52
        measurements[index].image_median = 3_000.0

    evaluate_quality_gate(results, measurements)

    assert all(
        result.quality_gate is not None
        and result.quality_gate.disposition is GateDisposition.PASS
        and "GATE_SOURCE_RETENTION_REVIEW" not in _codes(result)
        for result in results[8:]
    ), [(result.quality_gate.disposition, sorted(_codes(result))) for result in results[8:]]


def test_same_night_detection_drop_is_still_reviewed() -> None:
    results, measurements = _cohort(8)
    for result in results:
        result.metadata.airmass = None
        result.features.transparency_ratio = 1.0
    results[5].features.detected_source_ratio = 0.55
    measurements[5].detected_source_count = 55

    evaluate_quality_gate(results, measurements)

    assert results[5].quality_gate is not None
    assert "GATE_SOURCE_RETENTION_REVIEW" in _codes(results[5])


def test_uniformly_defocused_second_night_is_reviewed_against_cohort_best() -> None:
    results, measurements = _cohort(16)
    for index in range(8, 16):
        results[index].metadata.observed_at += timedelta(days=1)
        results[index].features.nina_hfr_pixels = 4.0
        results[index].features.median_fwhm_native_pixels = 8.0

    evaluate_quality_gate(results, measurements)

    assert all(
        result.quality_gate is not None
        and result.quality_gate.disposition is GateDisposition.REVIEW
        and "GATE_NIGHT_FOCUS_SHIFT_REVIEW" in _codes(result)
        for result in results[8:]
    )


def test_strong_cloud_requires_multiple_independent_families() -> None:
    results, measurements = _cohort(8)
    cloudy = results[4]
    cloudy.features.transparency_ratio *= 0.25
    cloudy.features.spatial_dimming_p90_mag = 0.60
    cloudy.features.star_completeness = 0.35
    cloudy.features.detected_source_ratio = 0.35
    measurements[4].detected_source_count = 35

    evaluate_quality_gate(results, measurements)

    assert cloudy.quality_gate is not None
    assert cloudy.quality_gate.disposition is GateDisposition.HARD_FAIL
    assert "GATE_MULTI_FAMILY_CLOUD_HARD" in _codes(cloudy)


def test_one_clear_frame_anchors_a_cloudy_majority() -> None:
    results, measurements = _cohort(8)
    for index in range(1, 8):
        results[index].features.transparency_ratio *= 0.20
        results[index].features.star_completeness = 0.35
        results[index].features.detected_source_ratio = 0.35
        measurements[index].detected_source_count = 35

    gates = evaluate_quality_gate(results, measurements)

    assert gates[0].disposition is GateDisposition.PASS
    assert all(
        gate.disposition is GateDisposition.HARD_FAIL for gate in gates[1:]
    )


def test_correlated_source_count_metrics_form_one_family_and_only_review() -> None:
    results, measurements = _cohort(8)
    suspect = results[3]
    suspect.features.star_completeness = 0.30
    suspect.features.detected_source_ratio = 0.25
    measurements[3].detected_source_count = 25

    evaluate_quality_gate(results, measurements)

    assert suspect.quality_gate is not None
    assert suspect.quality_gate.disposition is GateDisposition.REVIEW
    families = [item.family for item in suspect.quality_gate.evidence]
    assert len(families) == len(set(families))
    assert families.count(EvidenceFamily.CONSENSUS) == 1


def test_hard_occlusion_blocks_but_focus_outlier_only_reviews() -> None:
    results, measurements = _cohort(8)
    wall = results[2]
    wall.features.largest_missing_region = 0.35
    wall.features.missing_inside_outside_ratio = 0.10
    wall.features.boundary_support = 0.80
    wall.features.background_support = 0.80
    focus = results[5]
    focus.features.nina_hfr_pixels = 3.0
    focus.features.median_fwhm_native_pixels = 6.0

    evaluate_quality_gate(results, measurements)

    assert wall.quality_gate is not None
    assert wall.quality_gate.disposition is GateDisposition.HARD_FAIL
    assert "GATE_OCCLUSION_HARD" in _codes(wall)
    assert focus.quality_gate is not None
    assert focus.quality_gate.disposition is GateDisposition.REVIEW
    assert "GATE_FOCUS_SEEING_REVIEW" in _codes(focus)


def test_background_signal_outlier_alone_requires_review_not_hard_fail() -> None:
    results, measurements = _cohort(8)
    measurements[4].image_median = 1_700.0
    measurements[4].image_mad = 12.0

    evaluate_quality_gate(results, measurements)

    assert results[4].quality_gate is not None
    assert results[4].quality_gate.disposition is GateDisposition.REVIEW
    assert "GATE_BACKGROUND_STRONG" in _codes(results[4])
    assert "GATE_MULTI_FAMILY_CLOUD_HARD" not in _codes(results[4])


def test_historical_known_bad_metric_profiles_never_pass_gate() -> None:
    dispositions: list[GateDisposition] = []

    # Thin uniform cloud: temporal family alone is sufficient to withhold PASS.
    results, measurements = _cohort(8)
    results[4].features.transparency_ratio *= 10 ** (-0.4 * 0.531)
    dispositions.append(evaluate_quality_gate(results, measurements)[4].disposition)

    # Strong cloud with source loss.
    results, measurements = _cohort(8)
    results[4].features.transparency_ratio *= 10 ** (-0.4 * 0.898)
    results[4].features.detected_source_ratio = 0.60
    measurements[4].detected_source_count = 60
    dispositions.append(evaluate_quality_gate(results, measurements)[4].disposition)

    # Cloud/background glow.
    results, measurements = _cohort(8)
    results[4].features.transparency_ratio *= 10 ** (-0.4 * 0.804)
    measurements[4].image_median = 1_700.0
    dispositions.append(evaluate_quality_gate(results, measurements)[4].disposition)

    # Defocus/poor seeing: HFR and native FWHM agree.
    results, measurements = _cohort(8)
    results[4].features.nina_hfr_pixels = 3.14
    results[4].features.median_fwhm_native_pixels = 6.0
    dispositions.append(evaluate_quality_gate(results, measurements)[4].disposition)

    # Patchy cloud: temporal and spatial evidence agree.
    results, measurements = _cohort(8)
    results[4].features.transparency_ratio *= 10 ** (-0.4 * 0.519)
    results[4].features.spatial_dimming_p90_mag = 0.224
    dispositions.append(evaluate_quality_gate(results, measurements)[4].disposition)

    assert all(disposition is not GateDisposition.PASS for disposition in dispositions)


def test_strong_coherent_full_field_trailing_is_hard_fail() -> None:
    results, measurements = _cohort(8)
    trailed = results[6]
    measurements[6].stars = _stars(ellipticity=0.55, coherent=True)
    trailed.features.median_ellipticity = 0.55
    trailed.features.p90_ellipticity = 0.60
    trailed.features.elongated_fraction = 0.90
    trailed.features.orientation_coherence = 0.95
    trailed.features.valid_morphology_star_count = 40

    evaluate_quality_gate(results, measurements)

    assert trailed.quality_gate is not None
    assert trailed.quality_gate.disposition is GateDisposition.HARD_FAIL
    assert "GATE_COHERENT_TRAILING_HARD" in _codes(trailed)


def test_round_fragments_in_raw_catalog_cannot_pass_as_good_star_shapes() -> None:
    results, measurements = _cohort(8)
    fragments = [
        Star(
            x=x + step * 7, y=y + step * 2, flux=10_000 - step,
            peak=2_500, a=1, b=1, theta=0, fwhm=2.355,
            ellipticity=0, flags=1,
        )
        for x in (200, 1000, 1800)
        for y in (200, 750, 1300)
        for step in range(-3, 4)
    ]
    # The retained PSF sample is round and unflagged, while discarded or
    # deblended components still contain distributed tracking evidence.
    measurements[6].raw_stars = fragments + measurements[6].stars

    evaluate_quality_gate(results, measurements)

    assert results[6].quality_gate.disposition is GateDisposition.HARD_FAIL
    assert "GATE_FRAGMENTED_TRAILING_HARD" in _codes(results[6])
    assert results[6].features.fragmented_trail_chain_count >= 6
    assert all(
        item.quality_gate.disposition is GateDisposition.PASS
        for index, item in enumerate(results) if index != 6
    )


def test_measurement_error_is_hard_fail() -> None:
    results, measurements = _cohort(8)
    measurements[4].status = "ERROR"
    measurements[4].error_code = "SEP_EXTRACTION_FAILED"

    evaluate_quality_gate(results, measurements)

    assert results[4].quality_gate is not None
    assert results[4].quality_gate.disposition is GateDisposition.HARD_FAIL
    assert "GATE_MEASUREMENT_FAILED" in _codes(results[4])


def test_reference_requires_an_independent_registration_edge() -> None:
    results, measurements = _cohort(8)
    for result in results[1:]:
        result.registration.ok = False

    evaluate_quality_gate(results, measurements)

    reference = results[0]
    assert reference.quality_gate is not None
    assert reference.quality_gate.disposition is GateDisposition.REVIEW
    assert "GATE_REFERENCE_NOT_CONNECTED" in _codes(reference)


def test_policy_digest_is_canonical_and_changes_with_thresholds() -> None:
    first = GatePolicy()
    same = GatePolicy()
    changed = replace(first, minimum_detected_sources=21)

    assert first.canonical_digest() == same.canonical_digest()
    assert first.canonical_digest() != changed.canonical_digest()


def test_evidence_revision_is_fixed_and_invalidates_previous_policy_digest() -> None:
    policy = GatePolicy()
    legacy = policy.serializable()
    assert legacy.pop("evidence_revision") == 3
    previous_digest = "sha256:" + hashlib.sha256(
        json.dumps(legacy, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert policy.canonical_digest() != previous_digest
    with pytest.raises(TypeError):
        GatePolicy(evidence_revision=1)
