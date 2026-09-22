from __future__ import annotations

from lightframeqc.blink_flags import BlinkFrameInput, compute_flags
from lightframeqc.blink_reference import (
    BLINK_REFERENCE_RULE,
    CANDIDACY_ANY_FLAGS,
    CANDIDACY_ATTENTION_ALLOWED,
    CANDIDACY_CLEAN,
    choose_reference,
    choose_references,
    score_frames,
)


def _frame(index: int, **overrides) -> BlinkFrameInput:
    values = dict(
        path=f"/blink/frame-{index:02d}.fits",
        channel_id="L",
        target="NGC 6822",
        filter_name="L",
        night="2026-08-17",
        observed_at=f"2026-08-17T21:{index:02d}:00",
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
        transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        source_sha256=f"sha256:{index + 1:064x}",
    )
    values.update(overrides)
    return BlinkFrameInput(**values)


def _score_of(scores, path: str) -> float:
    value = next(item.score for item in scores if item.path == path)
    assert value is not None
    return value


def test_score_is_monotonic_in_sky_noise_psf_roundness_and_star_count() -> None:
    frames = [
        _frame(0),
        _frame(1, sky=2400.0, sky_mad=110.0, transparency=0.84),  # moonlit: brighter, noisier
        _frame(2, fwhm_native=5.0),
        _frame(3, ellipticity=0.25),
        _frame(4, star_count=2600),
        *[_frame(index) for index in range(5, 10)],
    ]
    scores = score_frames(frames)
    base = _score_of(scores, frames[0].path)
    assert _score_of(scores, frames[1].path) < base / 10
    assert _score_of(scores, frames[2].path) < base
    assert _score_of(scores, frames[3].path) < base
    assert _score_of(scores, frames[4].path) < base
    ranks = {item.path: item.rank for item in scores}
    assert ranks[frames[0].path] == 1
    assert ranks[frames[1].path] == len(frames)
    zs = {item.path: item.score_z for item in scores}
    assert zs[frames[1].path] < zs[frames[0].path]


def test_candidacy_excludes_flagged_weak_or_partial_frames() -> None:
    frames = [
        *[_frame(index) for index in range(8)],
        _frame(8, registration_ok=False, matched_stars=0, registration_rms=None),
        _frame(9, matched_stars=20),
        _frame(10, registration_rms=1.6),
        _frame(11, overlap=0.85),
        _frame(12, transparency=None),
        _frame(13, sky=2400.0, star_count=2600, transparency=1.4, sky_mad=10.0),  # moonlit, EXCLUDE flag
        _frame(14, extinction_mag=0.6, transparency=1.4, sky_mad=10.0),  # ATTENTION flag
    ]
    flags = compute_flags(frames)
    scores = score_frames(frames, flags)
    by_path = {item.path: item for item in scores}
    for index, reason in ((8, "UNREGISTRABLE"), (9, "FEW_MATCHES"), (10, "REGISTRATION_RMS"), (11, "LOW_OVERLAP"), (12, "MEASUREMENT_MISSING"), (13, "EXCLUDE_FLAG"), (14, "ATTENTION_FLAG")):
        score = by_path[frames[index].path]
        assert score.candidate is False and reason in score.reasons, (index, score.reasons)
        assert score.score is None and score.rank is None
    reference = choose_reference(scores, flags, channel_id="L", inputs=frames)
    assert reference is not None
    assert reference.candidacy == CANDIDACY_CLEAN
    assert reference.candidate_count == 8
    assert reference.rule == BLINK_REFERENCE_RULE
    assert reference.source_sha256 == frames[0].source_sha256 or reference.path in {frame.path for frame in frames[:8]}
    # A flagged frame with a tempting score never becomes the reference.
    assert reference.path not in {frames[13].path, frames[14].path}


def test_candidacy_relaxes_to_attention_then_to_every_frame() -> None:
    attention_only = [_frame(index, extinction_mag=0.6) for index in range(6)]
    flags = compute_flags(attention_only)
    assert all(record.attention for record in flags)
    reference = choose_reference(score_frames(attention_only, flags), flags, channel_id="L")
    assert reference is not None and reference.candidacy == CANDIDACY_ATTENTION_ALLOWED
    all_excluded = [_frame(index, extinction_mag=1.2) for index in range(6)]
    flags = compute_flags(all_excluded)
    assert all(record.exclude for record in flags)
    reference = choose_reference(score_frames(all_excluded, flags), flags, channel_id="L")
    assert reference is not None and reference.candidacy == CANDIDACY_ANY_FLAGS
    nothing = [_frame(index, registration_ok=False) for index in range(3)]
    assert choose_reference(score_frames(nothing), None, channel_id="L") is None


def test_ties_go_to_the_central_dither_then_the_earlier_frame() -> None:
    # Identical quality; translations spread on a line so the median is the
    # middle frame, which should win; among equal distances the earlier one.
    frames = [
        _frame(index, transform=((1.0, 0.0, 10.0 * index), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        for index in range(7)
    ]
    scores = score_frames(frames)
    reference = choose_reference(scores, None, channel_id="L", inputs=frames)
    assert reference is not None and reference.path == frames[3].path
    symmetric = [
        _frame(0, transform=((1.0, 0.0, -5.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        _frame(1, transform=((1.0, 0.0, 5.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        _frame(2, transform=((1.0, 0.0, 30.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        _frame(3, transform=((1.0, 0.0, -30.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ]
    reference = choose_reference(score_frames(symmetric), None, channel_id="L")
    assert reference is not None and reference.path == symmetric[0].path
    later_first = [symmetric[1], symmetric[0], symmetric[2], symmetric[3]]
    reference = choose_reference(score_frames(later_first), None, channel_id="L")
    assert reference is not None and reference.path == symmetric[0].path


def test_scores_are_deterministic_and_per_channel() -> None:
    frames = [
        *[_frame(index) for index in range(6)],
        *[
            _frame(index, channel_id="R", filter_name="R", sky=600.0, sky_mad=15.0, star_count=3500)
            for index in range(6, 12)
        ],
    ]
    flags = compute_flags(frames)
    first = score_frames(frames, flags)
    second = score_frames(list(frames), compute_flags(list(frames)))
    assert [item.serializable() for item in first] == [item.serializable() for item in second]
    references = choose_references(first, flags, inputs=frames)
    assert set(references) == {"L", "R"}
    assert references["L"].path.startswith("/blink/frame-0")
    assert references["R"].channel_id == "R"
    ranks_r = sorted(item.rank for item in first if item.channel_id == "R")
    assert ranks_r == [1, 2, 3, 4, 5, 6]
    assert all(item.serializable()["log10"] is not None for item in first)
