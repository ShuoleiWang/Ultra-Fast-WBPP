"""Blink reference per channel: a PSF-signal-weight analogue from QC numbers.

PixInsight's PSF signal weight ranks frames by the signal power of a unit
flux star over the background noise power.  Every ingredient of that idea is
already measured per frame by the quality analysis, so the blink reference
is chosen without reading a pixel again::

    S = (T / (1.4826 * MAD))^2 * (F_best / F)^2 * (1 - e) * min(1, N / N_rich)

T is the flux ratio to the QC reference, MAD the preview's median absolute
deviation (raw ADU, sky photon noise included), F the native PSF FWHM,
F_best the channel's robust best FWHM, e the median ellipticity and N the
uncapped detection count over the channel's robust richest count.  A wider
PSF spreads the same flux, a rounder star concentrates it, a poorer
detection count says the frame is less transparent than its flux ratio
alone admits.

The reference anchors the blink previews and the compare mode only; the
run's registration and normalization references are chosen by the pipeline
as before, so an identical admitted set gives bit-identical products.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .blink_flags import BlinkFrameFlags, BlinkFrameInput

BLINK_REFERENCE_RULE = "psf-signal-weight-proxy-v1"

# Candidacy: enough matched stars, a tight transform and (nearly) the whole
# common field, so every frame of the channel registers to the reference.
CANDIDATE_MINIMUM_MATCHED_STARS = 30
CANDIDATE_MAXIMUM_RMS_PIXELS = 1.5
CANDIDATE_MINIMUM_OVERLAP = 0.90

CANDIDACY_CLEAN = "clean"
CANDIDACY_ATTENTION_ALLOWED = "attention-allowed"
CANDIDACY_ANY_FLAGS = "any-flags"


@dataclass(frozen=True, slots=True)
class BlinkScore:
    path: str
    channel_id: str
    score: float | None
    score_log10: float | None
    score_z: float | None
    rank: int | None
    candidate: bool
    # Why the frame is not a candidate (empty when it is one).
    reasons: tuple[str, ...]
    # Tie-break data: distance of the frame's registration translation to
    # the channel's median translation (preview pixels) and the time stamp.
    translation_distance: float | None
    observed_at: str | None

    def serializable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "channelId": self.channel_id,
            "score": self.score,
            "log10": None if self.score_log10 is None else round(self.score_log10, 4),
            "z": None if self.score_z is None else round(self.score_z, 3),
            "rank": self.rank,
            "candidate": self.candidate,
            "reasons": list(self.reasons),
            "translationDistance": (
                None if self.translation_distance is None else round(self.translation_distance, 3)
            ),
        }


@dataclass(frozen=True, slots=True)
class BlinkReference:
    channel_id: str
    path: str
    rule: str
    candidacy: str
    candidate_count: int
    score: float | None
    source_sha256: str | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "channelId": self.channel_id,
            "path": self.path,
            "sourceSha256": self.source_sha256,
            "rule": self.rule,
            "candidacy": self.candidacy,
            "candidateCount": self.candidate_count,
            "score": self.score,
        }


def _raw_score(frame: BlinkFrameInput, fwhm_best: float, richest: float) -> float | None:
    transparency = frame.transparency
    noise = frame.sky_mad
    fwhm = frame.fwhm_native
    if (
        transparency is None
        or noise is None
        or fwhm is None
        or transparency <= 0
        or noise <= 0
        or fwhm <= 0
        or richest <= 0
    ):
        return None
    ellipticity = frame.ellipticity if frame.ellipticity is not None else 0.0
    roundness = max(0.0, min(1.0, 1.0 - ellipticity))
    completeness = min(1.0, frame.star_count / richest)
    value = (
        (transparency / (1.4826 * noise)) ** 2
        * (fwhm_best / fwhm) ** 2
        * roundness
        * completeness
    )
    return value if math.isfinite(value) and value > 0 else None


def _registrable(frame: BlinkFrameInput) -> list[str]:
    reasons: list[str] = []
    if not frame.registration_ok:
        reasons.append("UNREGISTRABLE")
    if frame.matched_stars < CANDIDATE_MINIMUM_MATCHED_STARS:
        reasons.append("FEW_MATCHES")
    if frame.registration_rms is None or frame.registration_rms > CANDIDATE_MAXIMUM_RMS_PIXELS:
        reasons.append("REGISTRATION_RMS")
    # A field too sparse for the overlap grid leaves the overlap unknown;
    # the registration evidence above then stands alone.
    if frame.overlap is not None and frame.overlap < CANDIDATE_MINIMUM_OVERLAP:
        reasons.append("LOW_OVERLAP")
    if frame.transparency is None or frame.fwhm_native is None:
        reasons.append("MEASUREMENT_MISSING")
    return reasons


def _translation(frame: BlinkFrameInput) -> tuple[float, float] | None:
    if frame.transform is None or len(frame.transform) < 2:
        return None
    try:
        return float(frame.transform[0][2]), float(frame.transform[1][2])
    except (IndexError, TypeError, ValueError):
        return None


def _candidacy(
    frames: Sequence[BlinkFrameInput], flags_by_path: Mapping[str, BlinkFrameFlags]
) -> tuple[str, dict[str, list[str]]]:
    """Registrable frames without flags; relaxed when that leaves nothing."""

    reasons: dict[str, list[str]] = {frame.path: _registrable(frame) for frame in frames}
    for level in (CANDIDACY_CLEAN, CANDIDACY_ATTENTION_ALLOWED, CANDIDACY_ANY_FLAGS):
        level_reasons: dict[str, list[str]] = {}
        for frame in frames:
            items = list(reasons[frame.path])
            record = flags_by_path.get(frame.path)
            if record is not None:
                if record.exclude and level != CANDIDACY_ANY_FLAGS:
                    items.append("EXCLUDE_FLAG")
                elif record.attention and level == CANDIDACY_CLEAN:
                    items.append("ATTENTION_FLAG")
            level_reasons[frame.path] = items
        if any(not items for items in level_reasons.values()):
            return level, level_reasons
    return CANDIDACY_ANY_FLAGS, {frame.path: reasons[frame.path] for frame in frames}


def score_frames(
    inputs: Sequence[BlinkFrameInput],
    flags: Sequence[BlinkFrameFlags] | None = None,
) -> list[BlinkScore]:
    """Score every frame of every channel; the result is in input order.

    ``flags`` (from :func:`~lightframeqc.blink_flags.compute_flags`) decide
    candidacy and the robust channel references; without them every frame
    is treated as unflagged.
    """

    flags_by_path = {record.path: record for record in flags} if flags else {}
    by_channel: dict[str, list[BlinkFrameInput]] = defaultdict(list)
    for frame in inputs:
        by_channel[frame.channel_id].append(frame)
    scores_by_path: dict[str, BlinkScore] = {}
    for channel_id, frames in by_channel.items():
        _level, reasons = _candidacy(frames, flags_by_path)
        candidates = [frame for frame in frames if not reasons[frame.path]]
        # Robust best FWHM over the candidates, never below their minimum;
        # robust richest count over the frames that are not EXCLUDE-flagged.
        fwhm_values = [frame.fwhm_native for frame in candidates if frame.fwhm_native]
        fwhm_best = (
            float(max(np.min(fwhm_values), np.percentile(fwhm_values, 10)))
            if fwhm_values
            else None
        )
        counted = [
            float(frame.star_count)
            for frame in frames
            if frame.star_count > 0
            and not (frame.path in flags_by_path and flags_by_path[frame.path].exclude)
        ] or [float(frame.star_count) for frame in frames if frame.star_count > 0]
        richest = float(np.percentile(counted, 90)) if counted else 0.0
        translations = [_translation(frame) for frame in candidates]
        known = [item for item in translations if item is not None]
        center = (
            (float(np.median([item[0] for item in known])), float(np.median([item[1] for item in known])))
            if known
            else None
        )
        raw: dict[str, float | None] = {}
        for frame in frames:
            raw[frame.path] = (
                _raw_score(frame, fwhm_best, richest)
                if fwhm_best is not None and not reasons[frame.path]
                else None
            )
        logs = {path: math.log10(value) for path, value in raw.items() if value is not None}
        if logs:
            values = np.asarray(list(logs.values()), dtype=np.float64)
            median = float(np.median(values))
            # A floor of 0.05 dex (about 12 % in score) keeps z finite when
            # most frames of a channel score alike.
            madn = max(1.4826 * float(np.median(np.abs(values - median))), 0.05)
        else:
            median, madn = 0.0, 0.0
        ordered = sorted(
            (frame for frame in frames if raw[frame.path] is not None),
            key=lambda frame: (-raw[frame.path], frame.path),  # type: ignore[operator]
        )
        ranks = {frame.path: position for position, frame in enumerate(ordered, start=1)}
        for frame in frames:
            value = raw[frame.path]
            translation = _translation(frame)
            distance = (
                math.hypot(translation[0] - center[0], translation[1] - center[1])
                if translation is not None and center is not None
                else None
            )
            log10 = logs.get(frame.path)
            scores_by_path[frame.path] = BlinkScore(
                path=frame.path,
                channel_id=channel_id,
                score=value,
                score_log10=log10,
                score_z=(log10 - median) / madn if log10 is not None else None,
                rank=ranks.get(frame.path),
                candidate=not reasons[frame.path],
                reasons=tuple(reasons[frame.path]),
                translation_distance=distance,
                observed_at=frame.observed_at,
            )
    return [scores_by_path[frame.path] for frame in inputs]


def choose_reference(
    scores: Sequence[BlinkScore],
    flags: Sequence[BlinkFrameFlags] | None = None,
    *,
    channel_id: str,
    inputs: Sequence[BlinkFrameInput] | None = None,
) -> BlinkReference | None:
    """The best candidate of one channel; ties go to the central, earlier frame.

    The candidate set is the one :func:`score_frames` used (no flags, then
    ATTENTION allowed, then every frame); ``flags`` names that level for the
    receipt.  ``None`` when no frame of the channel has a score.
    """

    channel_scores = [score for score in scores if score.channel_id == channel_id]
    candidates = [score for score in channel_scores if score.candidate and score.score is not None]
    if not candidates:
        return None
    flags_by_path = {record.path: record for record in flags} if flags else {}
    candidacy = CANDIDACY_CLEAN
    for score in candidates:
        record = flags_by_path.get(score.path)
        if record is not None and record.exclude:
            candidacy = CANDIDACY_ANY_FLAGS
            break
        if record is not None and record.attention:
            candidacy = CANDIDACY_ATTENTION_ALLOWED
    best = min(
        candidates,
        key=lambda score: (
            -(score.score or 0.0),
            math.inf if score.translation_distance is None else score.translation_distance,
            score.observed_at or "￿",
            score.path,
        ),
    )
    source_sha256 = None
    if inputs is not None:
        source_sha256 = next(
            (frame.source_sha256 for frame in inputs if frame.path == best.path), None
        )
    return BlinkReference(
        channel_id=channel_id,
        path=best.path,
        rule=BLINK_REFERENCE_RULE,
        candidacy=candidacy,
        candidate_count=len(candidates),
        score=best.score,
        source_sha256=source_sha256,
    )


def choose_references(
    scores: Sequence[BlinkScore],
    flags: Sequence[BlinkFrameFlags] | None = None,
    *,
    inputs: Sequence[BlinkFrameInput] | None = None,
) -> dict[str, BlinkReference]:
    """One reference per channel, keyed by channel id (channels without a
    scorable frame are absent)."""

    references: dict[str, BlinkReference] = {}
    for channel_id in dict.fromkeys(score.channel_id for score in scores):
        reference = choose_reference(scores, flags, channel_id=channel_id, inputs=inputs)
        if reference is not None:
            references[channel_id] = reference
    return references


__all__ = [
    "BLINK_REFERENCE_RULE",
    "BlinkReference",
    "BlinkScore",
    "CANDIDACY_ANY_FLAGS",
    "CANDIDACY_ATTENTION_ALLOWED",
    "CANDIDACY_CLEAN",
    "choose_reference",
    "choose_references",
    "score_frames",
]
