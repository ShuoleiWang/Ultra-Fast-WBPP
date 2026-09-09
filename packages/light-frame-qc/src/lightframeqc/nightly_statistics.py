"""Robust night-aware baselines for light-frame quality measurements.

The routines in this module are deliberately independent of the classifier.
They expose both estimates and reliability diagnostics so a caller cannot
silently treat a small or geometrically degenerate sample as a fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
import math
import re
from typing import Hashable, Iterable, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np


Number = float | int | np.floating | np.integer


@dataclass(frozen=True, slots=True)
class NightlyExtinctionDiagnostics:
    """Evidence supporting a nightly extinction-envelope fit."""

    total_count: int
    valid_count: int
    pair_count: int
    airmass_span: float | None
    slope: float | None
    night_counts: dict[str, int]
    night_airmass_spans: dict[str, float | None]
    night_offsets: dict[str, float]
    reliable: bool
    reliability_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NightlyExtinctionFit:
    """Per-frame residuals and diagnostics for an extinction model."""

    residuals: tuple[float | None, ...]
    diagnostics: NightlyExtinctionDiagnostics


def _observing_timezone(value: str | None) -> tzinfo | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("observing_timezone must be None, an IANA zone, or a UTC offset")
    text = value.strip()
    if text.upper() in {"UTC", "Z"}:
        return timezone.utc
    offset = re.fullmatch(r"([+-])(\d{2}):(\d{2})", text)
    if offset is not None:
        hours = int(offset.group(2))
        minutes = int(offset.group(3))
        if hours > 14 or minutes > 59 or (hours == 14 and minutes != 0):
            raise ValueError("observing_timezone UTC offset is outside +/-14:00")
        delta = timedelta(hours=hours, minutes=minutes)
        if offset.group(1) == "-":
            delta = -delta
        return timezone(delta)
    try:
        return ZoneInfo(text)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"unknown observing_timezone {value!r}; use an IANA zone such as Asia/Shanghai"
        ) from error


def observing_night(
    observed_at: datetime | str | None,
    boundary_hours: float = 12,
    observing_timezone: str | None = None,
) -> str | None:
    """Return an observing-night identifier using a local daytime boundary.

    With the default noon boundary, timestamps before local noon belong to the
    preceding observing night.  ``observing_timezone`` may be an IANA name or a
    fixed ``+HH:MM``/``-HH:MM`` offset.  An aware input is converted into that
    zone; a naive N.I.N.A. DATE-LOC value is attached without changing its wall
    clock.  Without an explicit zone, aware inputs retain their source offset
    and naive inputs are treated as already local. ``None`` remains unresolved.
    """

    if not isinstance(boundary_hours, (int, float)) or isinstance(
        boundary_hours, bool
    ):
        raise ValueError("boundary_hours must be a number in [0, 24)")
    boundary = float(boundary_hours)
    if not math.isfinite(boundary) or not 0 <= boundary < 24:
        raise ValueError("boundary_hours must be a number in [0, 24)")
    if observed_at is None:
        return None
    if isinstance(observed_at, str):
        text = observed_at.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"invalid observation timestamp: {observed_at!r}") from error
    elif isinstance(observed_at, datetime):
        value = observed_at
    else:
        raise TypeError("observed_at must be datetime, ISO string, or None")
    selected_timezone = _observing_timezone(observing_timezone)
    if selected_timezone is not None:
        value = (
            value.replace(tzinfo=selected_timezone)
            if value.tzinfo is None
            else value.astimezone(selected_timezone)
        )
    return (value - timedelta(hours=boundary)).date().isoformat()


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _night(value: Hashable | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lower_or_upper_envelope(values: Sequence[float], *, high: bool) -> float:
    ordered = sorted(values, reverse=high)
    # "10% clear envelope" means the median of the clearest tenth, not a
    # linearly interpolated quantile.  With eight frames this deliberately uses
    # the single clear frame instead of blending it with seven cloudy frames.
    count = max(1, int(math.ceil(0.10 * len(ordered))))
    return float(np.median(np.asarray(ordered[:count], dtype=np.float64)))


def night_robust_baseline(
    values: Iterable[Number | None],
    night_ids: Iterable[Hashable | None],
    direction: str = "median",
) -> tuple[float | None, ...]:
    """Return a leave-one-out baseline for every valid frame.

    ``low`` is suitable for HFR/FWHM/background-like metrics where the clear
    envelope is small.  ``high`` is suitable for source-count or transparency
    metrics.  ``median`` represents a typical nightly value.  A one-frame
    night returns ``None``: a frame is never allowed to establish its own
    baseline.
    """

    if direction not in {"low", "median", "high"}:
        raise ValueError("direction must be 'low', 'median', or 'high'")
    raw_values = list(values)
    raw_nights = list(night_ids)
    if len(raw_values) != len(raw_nights):
        raise ValueError("values and night_ids must have the same length")

    finite_values = [_finite(value) for value in raw_values]
    nights = [_night(value) for value in raw_nights]
    result: list[float | None] = []
    for index, (current, current_night) in enumerate(zip(finite_values, nights)):
        if current is None or current_night is None:
            result.append(None)
            continue
        peers = [
            value
            for peer_index, (value, night_id) in enumerate(
                zip(finite_values, nights)
            )
            if peer_index != index
            and night_id == current_night
            and value is not None
        ]
        if not peers:
            result.append(None)
        elif direction == "median":
            result.append(float(np.median(np.asarray(peers, dtype=np.float64))))
        else:
            result.append(
                _lower_or_upper_envelope(peers, high=direction == "high")
            )
    return tuple(result)


def fit_nightly_extinction_envelope(
    airmass: Iterable[Number | None],
    extinction: Iterable[Number | None],
    night_ids: Iterable[Hashable | None],
) -> NightlyExtinctionFit:
    """Fit a robust common airmass slope with nightly clear envelopes.

    Only within-night pairs contribute to the Theil-Sen-style slope estimate,
    so different nightly zero points cannot masquerade as atmospheric
    extinction.  After removing the slope, the offset for each night is the
    median of its clearest 10 percent.  Consequently a short cloud excursion,
    and even a seven-cloud/one-clear bad majority, remains in the residuals.

    Rows lacking finite airmass, extinction, or night identity receive a
    ``None`` residual.  A fit with fewer than four valid rows, fewer than three
    useful same-night pairs, or insufficient within-night airmass span is
    explicitly marked unreliable.
    """

    raw_airmass = list(airmass)
    raw_extinction = list(extinction)
    raw_nights = list(night_ids)
    if not (len(raw_airmass) == len(raw_extinction) == len(raw_nights)):
        raise ValueError("airmass, extinction, and night_ids must have equal length")

    air = [_finite(value) for value in raw_airmass]
    ext = [_finite(value) for value in raw_extinction]
    nights = [_night(value) for value in raw_nights]
    valid_indices = [
        index
        for index, (x, y, night_id) in enumerate(zip(air, ext, nights))
        if x is not None and x > 0 and y is not None and night_id is not None
    ]

    by_night: dict[str, list[int]] = {}
    for index in valid_indices:
        assert nights[index] is not None
        by_night.setdefault(nights[index], []).append(index)

    night_counts = {key: len(indices) for key, indices in sorted(by_night.items())}
    night_spans: dict[str, float | None] = {}
    for key, indices in sorted(by_night.items()):
        values = [air[index] for index in indices]
        assert all(value is not None for value in values)
        night_spans[key] = (
            float(max(values) - min(values)) if len(values) >= 2 else None
        )
    finite_spans = [value for value in night_spans.values() if value is not None]
    airmass_span = max(finite_spans) if finite_spans else None

    pair_slopes: list[float] = []
    minimum_pair_span = 0.05
    for indices in by_night.values():
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                assert air[left] is not None and air[right] is not None
                assert ext[left] is not None and ext[right] is not None
                delta_airmass = air[right] - air[left]
                if abs(delta_airmass) < minimum_pair_span:
                    continue
                slope = (ext[right] - ext[left]) / delta_airmass
                if math.isfinite(slope):
                    pair_slopes.append(float(slope))

    slope = (
        float(np.median(np.asarray(pair_slopes, dtype=np.float64)))
        if pair_slopes
        else None
    )
    offsets: dict[str, float] = {}
    residuals: list[float | None] = [None] * len(raw_airmass)
    if slope is not None:
        for key, indices in sorted(by_night.items()):
            corrected = []
            for index in indices:
                assert air[index] is not None and ext[index] is not None
                corrected.append(ext[index] - slope * (air[index] - 1.0))
            offset = _lower_or_upper_envelope(corrected, high=False)
            offsets[key] = offset
            for index in indices:
                assert air[index] is not None and ext[index] is not None
                residuals[index] = float(
                    ext[index] - slope * (air[index] - 1.0) - offset
                )

    reasons: list[str] = []
    if len(valid_indices) < 4:
        reasons.append("TOO_FEW_VALID_FRAMES")
    if len(pair_slopes) < 3:
        reasons.append("TOO_FEW_WITHIN_NIGHT_PAIRS")
    if airmass_span is None or airmass_span < 0.20:
        reasons.append("INSUFFICIENT_WITHIN_NIGHT_AIRMASS_SPAN")
    if slope is None:
        reasons.append("SLOPE_UNRESOLVED")

    diagnostics = NightlyExtinctionDiagnostics(
        total_count=len(raw_airmass),
        valid_count=len(valid_indices),
        pair_count=len(pair_slopes),
        airmass_span=airmass_span,
        slope=slope,
        night_counts=night_counts,
        night_airmass_spans=night_spans,
        night_offsets=offsets,
        reliable=not reasons,
        reliability_reasons=tuple(reasons),
    )
    return NightlyExtinctionFit(tuple(residuals), diagnostics)


__all__ = [
    "NightlyExtinctionDiagnostics",
    "NightlyExtinctionFit",
    "fit_nightly_extinction_envelope",
    "night_robust_baseline",
    "observing_night",
]
