from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from lightframeqc.blink_flags import (
    BlinkFlagPolicy,
    BlinkFrameInput,
    EVIDENCE_INSUFFICIENCY_CODES,
    GATE_FLAG_MAPPING,
    SEVERITY_ATTENTION,
    SEVERITY_EXCLUDE,
    background_shape_statistic,
    background_shapes,
    blink_inputs,
    channel_statistics,
    compute_flags,
    night_summaries,
)
from lightframeqc.models import (
    Confidence,
    Decision,
    EvidenceFamily,
    EvidenceSeverity,
    FileIdentity,
    FrameFeatures,
    FrameMeasurement,
    FrameMetadata,
    FrameResult,
    FrameRole,
    GateDisposition,
    QualityEvidence,
    QualityGateResult,
    RegistrationMetrics,
)


def _frame(index: int, **overrides) -> BlinkFrameInput:
    night = overrides.pop("night", "2026-08-17")
    values = dict(
        path=f"/blink/{night}/frame-{index:02d}.fits",
        channel_id="L",
        target="NGC 6822",
        filter_name="L",
        night=night,
        observed_at=f"{night}T21:{index:02d}:00",
        airmass=1.2,
        sky=1000.0,
        sky_mad=30.0,
        star_count=5000,
        transparency=1.0,
        extinction_mag=0.05,
        fwhm_native=4.2,
        ellipticity=0.08,
        eccentricity=0.4,
        registration_ok=True,
        matched_stars=1500,
        registration_rms=0.3,
        overlap=1.0,
        background_shape=0.1,
        gate_disposition="PASS",
        gate_codes=(),
        transform=((1.0, 0.0, float(index)), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    values.update(overrides)
    return BlinkFrameInput(**values)


def _clean_channel(count: int = 10) -> list[BlinkFrameInput]:
    return [_frame(index) for index in range(count)]


def _codes(record) -> dict[str, str]:
    return {item.code: item.severity for item in record.flags}


def test_policy_digest_is_stable_and_changes_with_any_threshold() -> None:
    policy = BlinkFlagPolicy()
    assert policy.version == "blink-flags-v1"
    assert policy.canonical_digest() == BlinkFlagPolicy().canonical_digest()
    assert policy.canonical_digest().startswith("sha256:")
    changed = replace(policy, sky_bright_ratio=1.7)
    assert changed.canonical_digest() != policy.canonical_digest()
    serialized = policy.serializable()
    assert serialized["sky_bright_ratio"] == 1.6 and serialized["few_stars_minimum"] == 20


def test_policy_validation_rejects_inconsistent_thresholds() -> None:
    with pytest.raises(ValueError):
        replace(BlinkFlagPolicy(), extinction_exclude_mag=0.4).validate()
    with pytest.raises(ValueError):
        replace(BlinkFlagPolicy(), few_stars_minimum=0).validate()
    with pytest.raises(ValueError):
        replace(BlinkFlagPolicy(), sources_low_exclude_ratio=0.7).validate()
    with pytest.raises(ValueError):
        replace(BlinkFlagPolicy(), version="").validate()


def test_clean_channel_has_no_flags_and_keep_defaults() -> None:
    records = compute_flags(_clean_channel())
    assert all(not record.flags and record.default_decision == "KEEP" for record in records)
    stats = channel_statistics(_clean_channel())["L"]
    assert stats.clean_count == 10 and stats.sky_clean == 1000.0
    assert stats.sources_best == 5000.0 and stats.fwhm_best == pytest.approx(4.2)


def test_sky_bright_is_attention_alone_and_exclude_with_low_sources() -> None:
    frames = _clean_channel()
    bright_transparent = _frame(20, night="2026-08-20", sky=1700.0)
    moonlit = _frame(21, night="2026-08-20", sky=2400.0, star_count=2800)
    just_below = _frame(22, night="2026-08-20", sky=1590.0, star_count=2800)
    records = compute_flags([*frames, bright_transparent, moonlit, just_below])
    transparent, moon, below = records[-3:]
    assert _codes(transparent) == {"BLINK_SKY_BRIGHT": SEVERITY_ATTENTION}
    assert transparent.default_decision == "KEEP"
    assert _codes(moon) == {
        "BLINK_SKY_BRIGHT": SEVERITY_EXCLUDE,
        "BLINK_SOURCES_LOW": SEVERITY_ATTENTION,
    }
    assert all(item.combined for item in moon.flags)
    assert moon.default_decision == "DROP"
    assert moon.metrics["skyRatio"] == pytest.approx(2.4)
    # Below the sky ratio only the source ratio speaks, and it is attention.
    assert _codes(below) == {"BLINK_SOURCES_LOW": SEVERITY_ATTENTION}
    assert not any(item.combined for item in below.flags)


def test_sources_low_thresholds_on_both_sides() -> None:
    frames = _clean_channel()
    records = compute_flags(
        [*frames, _frame(30, star_count=3050), _frame(31, star_count=2990), _frame(32, star_count=2240)]
    )
    assert "BLINK_SOURCES_LOW" not in _codes(records[-3])
    assert _codes(records[-2])["BLINK_SOURCES_LOW"] == SEVERITY_ATTENTION
    assert _codes(records[-1])["BLINK_SOURCES_LOW"] == SEVERITY_EXCLUDE
    assert records[-1].default_decision == "DROP"


def test_sources_best_uses_the_maximum_on_small_channels_and_p90_on_large() -> None:
    small = [_frame(index, star_count=1000 + 100 * index) for index in range(5)]
    assert channel_statistics(small)["L"].sources_best == 1400.0
    large = [_frame(index, star_count=1000 + 100 * index) for index in range(20)]
    assert 2500.0 < channel_statistics(large)["L"].sources_best < 2900.0


def test_extinction_flags_on_both_sides() -> None:
    frames = _clean_channel()
    records = compute_flags(
        [*frames, _frame(40, extinction_mag=0.49), _frame(41, extinction_mag=0.5), _frame(42, extinction_mag=1.0)]
    )
    assert "BLINK_EXTINCTION" not in _codes(records[-3])
    assert _codes(records[-2])["BLINK_EXTINCTION"] == SEVERITY_ATTENTION
    assert _codes(records[-1])["BLINK_EXTINCTION"] == SEVERITY_EXCLUDE


def test_fwhm_ellipticity_background_shape_and_gradient() -> None:
    frames = [_frame(index, gradient_adu=30.0) for index in range(10)]
    records = compute_flags(
        [
            *frames,
            _frame(50, fwhm_native=4.2 * 1.31, gradient_adu=30.0),
            _frame(51, fwhm_native=4.2 * 1.61, gradient_adu=30.0),
            _frame(52, ellipticity=0.31, gradient_adu=30.0),
            _frame(53, background_shape=0.51, gradient_adu=30.0),
            _frame(54, gradient_adu=61.0),
            _frame(55, gradient_adu=91.0, transparency=1.0),
            _frame(56, gradient_adu=45.0, transparency=0.5),
        ]
    )
    assert _codes(records[-7]) == {"BLINK_FWHM_WIDE": SEVERITY_ATTENTION}
    assert _codes(records[-6]) == {"BLINK_FWHM_WIDE": SEVERITY_EXCLUDE}
    assert _codes(records[-5]) == {"BLINK_STARS_ELONGATED": SEVERITY_ATTENTION}
    assert _codes(records[-4]) == {"BLINK_BACKGROUND_SHAPE": SEVERITY_ATTENTION}
    assert _codes(records[-3]) == {"BLINK_GRADIENT_AMPLITUDE": SEVERITY_ATTENTION}
    assert _codes(records[-2]) == {"BLINK_GRADIENT_AMPLITUDE": SEVERITY_EXCLUDE}
    # The gradient is scaled to the reference flux: 45 ADU at half the flux
    # is a 3x gradient after normalization.
    assert _codes(records[-1]) == {"BLINK_GRADIENT_AMPLITUDE": SEVERITY_EXCLUDE}
    assert records[-1].metrics["gradientRatio"] == pytest.approx(3.0)
    assert records[0].metrics["gradientRatio"] == pytest.approx(1.0)
    # Without calibrated gradients the metric is absent, never a flag.
    plain = compute_flags(_clean_channel())
    assert all(record.metrics["gradientRatio"] is None for record in plain)


def test_unregistrable_and_few_stars_are_always_exclude() -> None:
    frames = _clean_channel()
    records = compute_flags(
        [*frames, _frame(60, registration_ok=False, matched_stars=0, registration_rms=None), _frame(61, star_count=19)]
    )
    assert _codes(records[-2])["BLINK_UNREGISTRABLE"] == SEVERITY_EXCLUDE
    assert _codes(records[-1])["BLINK_FEW_STARS"] == SEVERITY_EXCLUDE
    assert records[-1].default_decision == "DROP"


def test_no_sky_flag_without_three_clean_frames_or_a_darker_night() -> None:
    # A single hazy night: nothing is darker to compare with.
    hazy = [_frame(index, night="2026-08-20", sky=2400.0, star_count=2800) for index in range(8)]
    records = compute_flags(hazy)
    assert not any("BLINK_SKY_BRIGHT" in _codes(record) for record in records)
    assert channel_statistics(hazy)["L"].sky_clean == 2400.0
    assert all(record.metrics["skyRatio"] == pytest.approx(1.0) for record in records)
    # Two clean frames are not enough for a clean-sky level either.
    few_clean = [*hazy, _frame(90, sky=1000.0), _frame(91, sky=1000.0)]
    assert channel_statistics(few_clean)["L"].sky_clean is None
    assert not any("BLINK_SKY_BRIGHT" in _codes(record) for record in compute_flags(few_clean))
    # A third clean frame establishes it and the hazy night is pre-dropped.
    enough = [*few_clean, _frame(92, sky=1000.0)]
    records = compute_flags(enough)
    assert all(record.default_decision == "DROP" for record in records[:8])
    assert all(record.default_decision == "KEEP" for record in records[8:])


def test_gate_codes_map_to_flags_notes_and_never_duplicate() -> None:
    frames = _clean_channel()
    trailing = _frame(70, gate_codes=("GATE_COHERENT_TRAILING_HARD",), gate_disposition="HARD_FAIL")
    review = _frame(
        71,
        gate_codes=(
            "GATE_TRAILING_REVIEW",
            "GATE_INSUFFICIENT_NIGHT_BASELINE",
            "GATE_TEMPORAL_EXTINCTION_REVIEW",
            "GATE_SOURCE_RETENTION_REVIEW",
        ),
        gate_disposition="REVIEW",
    )
    # A source-retention code does not duplicate an absolute BLINK_SOURCES_LOW.
    low = _frame(72, star_count=2500, gate_codes=("GATE_SOURCE_RETENTION_STRONG",))
    # Registration review on an unregistrable frame is covered by UNREGISTRABLE.
    broken = _frame(
        73, registration_ok=False, gate_codes=("GATE_REGISTRATION_REVIEW",), gate_disposition="REVIEW"
    )
    weak = _frame(74, gate_codes=("GATE_REGISTRATION_REVIEW",), gate_disposition="REVIEW")
    records = compute_flags([*frames, trailing, review, low, broken, weak])
    assert _codes(records[-5]) == {"BLINK_TRAILING": SEVERITY_EXCLUDE}
    assert records[-5].default_decision == "DROP"
    assert _codes(records[-4]) == {
        "BLINK_TRAILING": SEVERITY_ATTENTION,
        "BLINK_SOURCES_LOW": SEVERITY_ATTENTION,
    }
    assert records[-4].notes == ("GATE_INSUFFICIENT_NIGHT_BASELINE",)
    assert records[-4].default_decision == "KEEP"
    assert [item.code for item in records[-3].flags] == ["BLINK_SOURCES_LOW"]
    assert _codes(records[-2]) == {"BLINK_UNREGISTRABLE": SEVERITY_EXCLUDE}
    assert _codes(records[-1]) == {"BLINK_REGISTRATION_WEAK": SEVERITY_ATTENTION}
    for code in EVIDENCE_INSUFFICIENCY_CODES:
        assert code not in GATE_FLAG_MAPPING
    for code in (
        "GATE_OCCLUSION_HARD",
        "GATE_SPATIAL_DIMMING_STRONG",
        "GATE_MULTI_FAMILY_CLOUD_HARD",
        "GATE_MEASUREMENT_FAILED",
        "GATE_NOT_A_LIGHT_FRAME",
    ):
        assert GATE_FLAG_MAPPING[code][1] == SEVERITY_EXCLUDE
    for code in (
        "GATE_FOCUS_SEEING_REVIEW",
        "GATE_COMMON_FOOTPRINT_REVIEW",
        "GATE_BACKGROUND_STRONG",
        "GATE_NOISE_REVIEW",
    ):
        assert GATE_FLAG_MAPPING[code][1] == SEVERITY_ATTENTION


def test_channels_are_independent() -> None:
    l_frames = _clean_channel()
    r_frames = [
        _frame(index, channel_id="R", filter_name="R", sky=600.0, star_count=3500, fwhm_native=3.9)
        for index in range(10)
    ]
    # A sky of 1000 is normal for L but bright for R.
    odd = _frame(99, channel_id="R", filter_name="R", sky=1000.0, star_count=3500, fwhm_native=3.9)
    records = compute_flags([*l_frames, *r_frames, odd])
    assert _codes(records[-1]) == {"BLINK_SKY_BRIGHT": SEVERITY_ATTENTION}
    assert all(not record.flags for record in records[:-1])


def test_night_summaries_mark_a_fully_excluded_night() -> None:
    frames = [*_clean_channel(), *[_frame(index, night="2026-08-20", sky=2400.0, star_count=2800) for index in range(5)]]
    frames.append(_frame(9, night="2026-08-20", sky=1000.0))
    records = compute_flags(frames)
    summaries = {item["night"]: item for item in night_summaries(frames, records)}
    assert summaries["2026-08-17"]["exclude"] == 0
    assert summaries["2026-08-17"]["defaultDropNight"] is False
    assert summaries["2026-08-20"]["frameCount"] == 6
    assert summaries["2026-08-20"]["exclude"] == 5
    assert summaries["2026-08-20"]["defaultDropNight"] is False
    assert summaries["2026-08-20"]["skyRatio"] == pytest.approx(2.4)
    only_bad = compute_flags(frames[:-1])
    assert {item["night"]: item["defaultDropNight"] for item in night_summaries(frames[:-1], only_bad)} == {
        "2026-08-17": False,
        "2026-08-20": True,
    }
    with pytest.raises(ValueError):
        night_summaries(frames, records[:-1])


def test_background_shape_statistic_excludes_the_core() -> None:
    grid = [[0.0] * 16 for _ in range(16)]
    for row in range(5, 11):
        for column in range(5, 11):
            grid[row][column] = 9.0  # the target's nebulosity
    assert background_shape_statistic(grid) == 0.0
    # A single outlier cell does not move the 5-95 percentile spread; a
    # band across the top does.
    grid[0][0] = 3.0
    assert background_shape_statistic(grid) == 0.0
    for row in range(3):
        grid[row] = [3.0] * 16
    assert background_shape_statistic(grid) == pytest.approx(3.0)
    assert background_shape_statistic(None) is None
    assert background_shape_statistic([[None] * 16] * 16) is None


def _result(
    index: int,
    *,
    flipped: bool = False,
    grid_value: float = 0.0,
    extinction: float = 0.05,
    disposition: GateDisposition = GateDisposition.PASS,
    ok: bool = True,
    codes: tuple[str, ...] = (),
    reference: bool = False,
) -> tuple[FrameResult, FrameMeasurement]:
    path = f"/blink/light-{index:02d}.fits"
    metadata = FrameMetadata(
        path=path,
        filter_name="L",
        target="NGC 6822",
        airmass=1.3,
        observed_at=datetime(2026, 8, 17, 21, index, tzinfo=timezone.utc) + timedelta(hours=0),
        role=FrameRole.LIGHT,
    )
    sign = -1.0 if flipped else 1.0
    matrix = [[sign, 0.0, 2.0 * index], [0.0, sign, 1.0], [0.0, 0.0, 1.0]] if ok else None
    # A tilted background whose orientation follows the camera.
    grid = [[grid_value + sign * 0.01 * column for column in range(16)] for _ in range(16)]
    gate = QualityGateResult(
        disposition=disposition,
        evidence=[
            QualityEvidence(code=code, family=EvidenceFamily.CONSENSUS, severity=EvidenceSeverity.REVIEW, message=code)
            for code in codes
        ],
    )
    result = FrameResult(
        path=path,
        group_id="group-l",
        reference_path="/blink/light-00.fits",
        decision=Decision.KEEP,
        confidence=Confidence.HIGH,
        reasons=[],
        warnings=[],
        registration=RegistrationMetrics(ok=ok, matched_stars=1200 if ok else 0, match_fraction=0.8, rms_pixels=0.3 if ok else None, matrix=matrix),
        features=FrameFeatures(
            transparency_ratio=0.98,
            extra_extinction_mag=extinction,
            median_ellipticity=0.09,
            median_eccentricity=0.4,
            overlap_fraction=1.0,
            psf_fwhm_native_pixels=4.3,
            median_fwhm_native_pixels=4.5,
        ),
        metadata=metadata,
        star_count=5000,
        grid={"backgroundDeltaRobustSigma": grid},
        identity=FileIdentity(sha256=f"{index + 1:064x}", size_bytes=10, mtime_ns=1, device=1, inode=index),
        quality_gate=gate,
    )
    measurement = FrameMeasurement(metadata=metadata, image_median=1000.0 + index, image_mad=30.0, preview_scale_x=4.0, preview_scale_y=4.0)
    return result, measurement


def test_blink_inputs_project_results_and_use_flip_families_for_the_shape() -> None:
    pairs = [_result(index, flipped=index >= 6) for index in range(10)]
    pairs.append(_result(10, flipped=True, grid_value=2.0))  # a flipped frame with a real shape change
    pairs.append(_result(11, ok=False, codes=("GATE_INSUFFICIENT_NIGHT_BASELINE",), disposition=GateDisposition.REVIEW))
    results = [result for result, _ in pairs]
    measurements = [measurement for _, measurement in pairs]
    inputs = blink_inputs(results, measurements)
    assert [item.path for item in inputs] == [result.path for result in results]
    first = inputs[0]
    assert first.channel_id == "group-l" and first.night == "2026-08-17"
    assert first.sky == 1000.0 and first.sky_mad == 30.0 and first.star_count == 5000
    assert first.fwhm_native == 4.3 and first.extinction_mag == 0.05
    assert first.source_sha256 == "sha256:" + "1".rjust(64, "0")
    assert first.is_qc_reference is True and inputs[1].is_qc_reference is False
    assert first.transform is not None and first.transform[0][0] == 1.0
    # The flipped family shares the reference's shape up to its rotation, so
    # no flipped frame is a shape outlier; the frame with a changed grid is.
    shapes = {item.path: item.background_shape for item in inputs}
    assert all(shapes[item.path] == pytest.approx(0.0, abs=1e-9) for item in inputs[:10])
    assert shapes[inputs[10].path] == pytest.approx(0.0, abs=1e-9)  # a constant offset is not a shape
    assert shapes[inputs[11].path] is None  # no transform, no family
    unregistrable = inputs[11]
    assert unregistrable.registration_ok is False and unregistrable.transform is None
    assert unregistrable.gate_disposition == "REVIEW"
    assert unregistrable.gate_codes == ("GATE_INSUFFICIENT_NIGHT_BASELINE",)
    # The reference's family needs three members; a lone flipped frame has no statistic.
    lone = [_result(index) for index in range(4)] + [_result(4, flipped=True)]
    lone_shapes = background_shapes([result for result, _ in lone])
    assert lone_shapes["/blink/light-04.fits"] is None
    assert all(value == pytest.approx(0.0, abs=1e-9) for path, value in lone_shapes.items() if path != "/blink/light-04.fits")


def test_blink_inputs_round_trip_through_json() -> None:
    frame = _frame(3, gate_codes=("GATE_TRAILING_REVIEW",))
    restored = BlinkFrameInput.from_mapping(frame.serializable())
    assert restored == frame
    with pytest.raises(ValueError):
        BlinkFrameInput.from_mapping({**frame.serializable(), "unexpected": 1})
