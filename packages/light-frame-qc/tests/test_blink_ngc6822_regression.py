"""NGC 6822 regression: the blink flags must reproduce the user's screening.

The fixture holds the per-frame blink inputs of 97 raw Lights over six
nights (built once from a real run's qc/manifest.json; basenames only, no
digests, paths or coordinates) together with the decision the user made by
blinking the frames in PixInsight.  The moonlit 2026-08-20 L night is the
case that motivated the flags: the legacy gate admitted 18 of its 22 frames
and the L master came out with an edge gradient.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from lightframeqc.blink_flags import (
    BlinkFlagPolicy,
    BlinkFrameInput,
    SEVERITY_ATTENTION,
    SEVERITY_EXCLUDE,
    compute_flags,
    night_summaries,
)
from lightframeqc.blink_reference import BLINK_REFERENCE_RULE, choose_references, score_frames

FIXTURE = Path(__file__).parent / "fixtures" / "blink" / "ngc6822-features.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frames = raw["frames"]
    inputs = [
        BlinkFrameInput.from_mapping(
            {key: value for key, value in frame.items() if key not in {"id", "userDecision"}}
        )
        for frame in frames
    ]
    flags = compute_flags(inputs, BlinkFlagPolicy())
    scores = score_frames(inputs, flags)
    references = choose_references(scores, flags, inputs=inputs)
    return {
        "frames": frames,
        "inputs": inputs,
        "flags": flags,
        "scores": scores,
        "references": references,
        "user": {frame["path"]: frame["userDecision"] for frame in frames},
        "by_stamp": {frame["path"]: record for frame, record in zip(frames, flags, strict=True)},
    }


def _record(fixture: dict, filter_name: str, stamp: str):
    matches = [
        (frame, record)
        for frame, record in zip(fixture["frames"], fixture["flags"], strict=True)
        if frame["filter_name"] == filter_name and stamp in frame["path"]
    ]
    assert len(matches) == 1, (filter_name, stamp)
    return matches[0]


def _severities(record) -> set[str]:
    return {item.severity for item in record.flags}


def test_fixture_is_share_safe(fixture: dict) -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    assert "/Users/" not in text and "C:\\" not in text
    assert all(frame["source_sha256"] is None for frame in fixture["frames"])
    assert all("/" not in frame["path"] for frame in fixture["frames"])
    # NINA names with a degree sign and spaces survive the round trip.
    assert any("°" in frame["path"] and " " in frame["path"] for frame in fixture["frames"])
    assert len(fixture["frames"]) == 97


def test_every_08_20_l_frame_is_pre_marked_drop(fixture: dict) -> None:
    night = [
        record
        for frame, record in zip(fixture["frames"], fixture["flags"], strict=True)
        if frame["filter_name"] == "L" and frame["night"] == "2026-08-20"
    ]
    assert len(night) == 23
    assert all(record.default_decision == "DROP" for record in night)
    assert all("BLINK_SKY_BRIGHT" in record.codes for record in night)
    summaries = {
        (item["channelId"], item["night"]): item
        for item in night_summaries(fixture["inputs"], fixture["flags"])
    }
    assert summaries[("L", "2026-08-20")]["defaultDropNight"] is True
    assert summaries[("L", "2026-08-20")]["exclude"] == 23
    assert summaries[("L", "2026-08-17")]["defaultDropNight"] is False


def test_clearly_bad_l_frames_carry_an_exclude_flag_beyond_the_combined_rule(
    fixture: dict,
) -> None:
    for stamp in ("23-10-49", "23-19-33", "23-28-04", "23-37-28"):
        _frame, record = _record(fixture, "L", f"2026-08-20_{stamp}")
        standalone = [
            item
            for item in record.flags
            if item.severity == SEVERITY_EXCLUDE and not item.combined
        ]
        assert standalone, stamp


def test_no_08_17_frame_is_excluded_and_only_one_needs_attention(fixture: dict) -> None:
    night = [
        (frame, record)
        for frame, record in zip(fixture["frames"], fixture["flags"], strict=True)
        if frame["filter_name"] == "L" and frame["night"] == "2026-08-17"
    ]
    assert len(night) == 18
    assert not any(SEVERITY_EXCLUDE in _severities(record) for _, record in night)
    attention = [frame["path"] for frame, record in night if SEVERITY_ATTENTION in _severities(record)]
    assert len(attention) <= 1
    assert all("2026-08-17_22-36-34" in path for path in attention)


def test_user_kept_rgb_frames_have_no_exclude_flag_except_one(fixture: dict) -> None:
    excluded_kept = {
        frame["path"]
        for frame, record in zip(fixture["frames"], fixture["flags"], strict=True)
        if frame["filter_name"] in {"R", "G", "B"}
        and fixture["user"][frame["path"]] == "KEEP"
        and record.exclude
    }
    assert len(excluded_kept) == 1
    assert "G_0_CAA0.00°_2026-09-06_23-43-53" in next(iter(excluded_kept))


def test_thick_cloud_and_broken_rgb_frames_are_excluded(fixture: dict) -> None:
    for filter_name, stamp in (
        ("R", "2026-09-06_23-22-50"),
        ("R", "2026-09-06_23-27-52"),
        ("R", "2026-09-06_23-32-53"),
        ("G", "2026-09-06_23-38-51"),
        ("G", "2026-09-08_21-57-05"),
        ("G", "2026-09-08_22-02-07"),
    ):
        _frame, record = _record(fixture, filter_name, stamp)
        assert record.exclude, (filter_name, stamp, record.codes)
    for filter_name, stamp in (("R", "2026-09-06_23-32-53"), ("G", "2026-09-08_22-02-07")):
        _frame, record = _record(fixture, filter_name, stamp)
        assert "BLINK_UNREGISTRABLE" in record.codes


def test_l_reference_is_a_clear_08_17_frame(fixture: dict) -> None:
    reference = fixture["references"]["L"]
    assert reference.rule == BLINK_REFERENCE_RULE
    assert "2026-08-17" in reference.path
    frame = next(item for item in fixture["inputs"] if item.path == reference.path)
    assert frame.extinction_mag is not None and frame.extinction_mag < 0.1
    assert reference.candidacy == "clean"
    ranks = {score.path: score.rank for score in fixture["scores"] if score.channel_id == "L"}
    assert ranks[reference.path] == 1
    # Every channel has a reference and every scored frame a rank and a z.
    assert set(fixture["references"]) == {"L", "R", "G", "B"}
    scored = [score for score in fixture["scores"] if score.score is not None]
    assert scored and all(score.rank is not None and score.score_z is not None for score in scored)


def test_evidence_insufficiency_becomes_a_note_not_a_flag(fixture: dict) -> None:
    _frame, record = _record(fixture, "B", "2026-09-07_23-30-20")
    assert "GATE_INSUFFICIENT_NIGHT_BASELINE" in record.notes
    assert record.default_decision == "KEEP"
    assert not any(item.code == "BLINK_NIGHT_OUTLIER" for item in record.flags)


def test_confusion_counts_against_the_user_for_information(fixture: dict, capsys) -> None:
    counts: Counter[tuple[str, str]] = Counter()
    per_filter: dict[str, Counter[str]] = {}
    for frame, record in zip(fixture["frames"], fixture["flags"], strict=True):
        pair = (fixture["user"][frame["path"]], record.default_decision)
        counts[pair] += 1
        per_filter.setdefault(frame["filter_name"], Counter())[f"{pair[0]}/{pair[1]}"] += 1
    with capsys.disabled():
        print(
            "\nblink defaults vs user (user/default): "
            + ", ".join(f"{user}/{default}={count}" for (user, default), count in sorted(counts.items()))
        )
        for filter_name, values in sorted(per_filter.items()):
            print(f"  {filter_name}: " + ", ".join(f"{key}={value}" for key, value in sorted(values.items())))
    # Agreement on drops: everything the flags pre-drop the user also dropped,
    # except the one "not yet removed" cloud frame.
    assert counts[("KEEP", "DROP")] == 1
    # The flags never pre-drop the clean nights; the user's extra drops are
    # the light-cloud frames the flags mark ATTENTION for the user's call.
    assert counts[("DROP", "DROP")] >= 29
