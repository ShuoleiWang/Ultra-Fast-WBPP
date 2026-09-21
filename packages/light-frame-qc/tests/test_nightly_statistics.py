from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from lightframeqc.nightly_statistics import (
    fit_nightly_extinction_envelope,
    night_robust_baseline,
    observing_night,
)


def test_explicit_observing_timezone_keeps_one_chinese_evening_together() -> None:
    before_twenty_local = "2026-05-17T11:30:00+00:00"
    after_twenty_local = "2026-05-17T12:30:00+00:00"

    assert observing_night(
        before_twenty_local, observing_timezone="Asia/Shanghai"
    ) == observing_night(
        after_twenty_local, observing_timezone="Asia/Shanghai"
    ) == "2026-05-17"


def test_naive_nina_local_times_use_local_wall_clock_by_default() -> None:
    assert observing_night("2026-05-17T19:30:00") == "2026-05-17"
    assert observing_night("2026-05-17T20:30:00") == "2026-05-17"


@pytest.mark.parametrize("value", ["", "Mars/Olympus", "+15:00", "+08:99"])
def test_invalid_observing_timezone_fails_closed(value: str) -> None:
    with pytest.raises(ValueError):
        observing_night("2026-05-17T12:00:00+00:00", observing_timezone=value)


def test_observing_night_uses_noon_boundary_without_discarding_timezone() -> None:
    assert observing_night(datetime(2026, 8, 14, 1, 30)) == "2026-08-13"
    assert observing_night(datetime(2026, 8, 14, 12, 0)) == "2026-08-14"
    assert (
        observing_night("2026-08-14T03:00:00+08:00", boundary_hours=12)
        == "2026-08-13"
    )
    assert observing_night(None) is None
    with pytest.raises(ValueError):
        observing_night(datetime.now(timezone.utc), boundary_hours=24)


def test_two_nights_with_different_zeropoints_share_one_airmass_slope() -> None:
    airmass = [1.0, 1.5, 2.0, 2.5] * 2
    nights = ["night-a"] * 4 + ["night-b"] * 4
    expected_slope = 0.24
    extinction = [
        (0.10 if night == "night-a" else 0.65)
        + expected_slope * (value - 1.0)
        for value, night in zip(airmass, nights)
    ]

    result = fit_nightly_extinction_envelope(airmass, extinction, nights)

    assert result.diagnostics.reliable is True
    assert result.diagnostics.slope == pytest.approx(expected_slope, abs=1e-12)
    assert result.diagnostics.night_counts == {"night-a": 4, "night-b": 4}
    assert result.diagnostics.night_airmass_spans == {
        "night-a": pytest.approx(1.5),
        "night-b": pytest.approx(1.5),
    }
    assert result.diagnostics.night_offsets == {
        "night-a": pytest.approx(0.10),
        "night-b": pytest.approx(0.65),
    }
    assert result.residuals == pytest.approx([0.0] * 8, abs=1e-12)


def test_global_airmass_change_from_one_to_two_point_five_is_removed() -> None:
    airmass = np.linspace(1.0, 2.5, 10).tolist()
    extinction = [0.07 + 0.31 * (value - 1.0) for value in airmass]

    result = fit_nightly_extinction_envelope(
        airmass, extinction, ["night-a"] * len(airmass)
    )

    assert result.diagnostics.reliable
    assert result.diagnostics.airmass_span == pytest.approx(1.5)
    assert result.diagnostics.slope == pytest.approx(0.31, abs=1e-12)
    assert max(abs(value or 0.0) for value in result.residuals) < 1e-12


def test_short_same_night_cloud_excursion_remains_in_residuals() -> None:
    airmass = np.linspace(1.0, 1.7, 8).tolist()
    clear = [0.12 + 0.20 * (value - 1.0) for value in airmass]
    extinction = clear.copy()
    extinction[3] += 0.40
    extinction[4] += 0.40

    result = fit_nightly_extinction_envelope(
        airmass, extinction, ["night-a"] * 8
    )

    assert result.diagnostics.reliable
    assert result.diagnostics.slope == pytest.approx(0.20, abs=1e-12)
    assert result.residuals[3] == pytest.approx(0.40, abs=1e-12)
    assert result.residuals[4] == pytest.approx(0.40, abs=1e-12)
    assert max(
        abs(value or 0.0)
        for index, value in enumerate(result.residuals)
        if index not in {3, 4}
    ) < 1e-12


def test_one_clear_and_seven_cloudy_frames_do_not_define_cloud_as_baseline() -> None:
    airmass = np.linspace(1.0, 1.7, 8).tolist()
    clear = [0.05 + 0.18 * (value - 1.0) for value in airmass]
    extinction = [clear[0], *[value + 0.40 for value in clear[1:]]]

    result = fit_nightly_extinction_envelope(
        airmass, extinction, ["night-a"] * 8
    )

    assert result.diagnostics.reliable
    assert result.diagnostics.slope == pytest.approx(0.18, abs=1e-12)
    assert result.residuals[0] == pytest.approx(0.0, abs=1e-12)
    assert result.residuals[1:] == pytest.approx([0.40] * 7, abs=1e-12)


def test_missing_airmass_is_none_and_is_not_imputed_from_other_frames() -> None:
    result = fit_nightly_extinction_envelope(
        [1.0, 1.4, None, 1.8, 2.2],
        [0.1, 0.18, 0.30, 0.26, 0.34],
        ["night-a"] * 5,
    )

    assert result.diagnostics.valid_count == 4
    assert result.diagnostics.reliable
    assert result.residuals[2] is None

    unresolved = fit_nightly_extinction_envelope(
        [None, None, None, None],
        [0.1, 0.2, 0.3, 0.4],
        ["night-a"] * 4,
    )
    assert unresolved.residuals == (None, None, None, None)
    assert unresolved.diagnostics.slope is None
    assert unresolved.diagnostics.reliable is False
    assert "SLOPE_UNRESOLVED" in unresolved.diagnostics.reliability_reasons


def test_fewer_than_four_frames_is_explicitly_unreliable() -> None:
    result = fit_nightly_extinction_envelope(
        [1.0, 1.5, 2.0],
        [0.1, 0.2, 0.3],
        ["night-a"] * 3,
    )

    assert result.diagnostics.valid_count == 3
    assert result.diagnostics.reliable is False
    assert "TOO_FEW_VALID_FRAMES" in result.diagnostics.reliability_reasons
    # Estimates may be exposed for inspection, but the reliability flag is the
    # admission boundary for any automatic decision.
    assert result.diagnostics.slope == pytest.approx(0.20)


def test_night_baselines_are_leave_one_out_and_never_cross_nights() -> None:
    values = [1.0, 1.1, 9.0, 20.0, 21.0]
    nights = ["a", "a", "a", "b", "b"]

    low = night_robust_baseline(values, nights, direction="low")
    median = night_robust_baseline(values, nights, direction="median")
    high = night_robust_baseline(values, nights, direction="high")

    assert low == pytest.approx((1.1, 1.0, 1.0, 21.0, 20.0))
    assert median == pytest.approx((5.05, 5.0, 1.05, 21.0, 20.0))
    assert high == pytest.approx((9.0, 9.0, 1.1, 21.0, 20.0))

    singleton = night_robust_baseline([3.0], ["only"], direction="low")
    assert singleton == (None,)
    with pytest.raises(ValueError):
        night_robust_baseline([1.0], ["a"], direction="sideways")


def test_cloud_frames_inside_a_short_airmass_span_do_not_define_the_slope() -> None:
    """NGC 6822, Sept 2026: twelve R frames on one night over airmass 1.32-1.52,
    nine clear (0.03-0.2 mag) and three under cloud (1.8-2.7 mag); three
    clear frames on the next night.  The cloud pairs have slopes of tens of
    magnitudes per airmass and used to give a median slope of 14 mag/airmass,
    nightly offsets of -5.6 and -8.0 mag, and a "2.4 mag dimmer night" review
    of every clear frame of the first night."""

    airmass = [1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.40, 1.51, 1.53, 1.56]
    nights = ["night-a"] * 12 + ["night-b"] * 3
    slope = 0.20
    extinction = [0.04 + slope * (value - 1.0) for value in airmass]
    for index, cloud in ((3, 2.7), (7, 1.8), (10, 2.2)):
        extinction[index] += cloud
    for index in range(12, 15):
        extinction[index] += 0.05  # the second night is 0.05 mag dimmer, not magnitudes

    result = fit_nightly_extinction_envelope(airmass, extinction, nights)

    assert result.diagnostics.slope == pytest.approx(slope, abs=1e-9)
    assert result.diagnostics.discarded_pair_count > 0
    assert result.diagnostics.night_offsets["night-b"] - result.diagnostics.night_offsets["night-a"] == pytest.approx(0.05, abs=1e-9)
    clear = [value for index, value in enumerate(result.residuals) if index not in {3, 7, 10}]
    assert max(abs(value or 0.0) for value in clear) < 1e-9
    assert result.residuals[3] == pytest.approx(2.7, abs=1e-9)


def test_a_cloud_dominated_slope_is_clamped_and_marks_the_fit_unreliable() -> None:
    airmass = [1.30, 1.36, 1.42, 1.48, 1.30, 1.36, 1.42, 1.48]
    nights = ["night-a"] * 4 + ["night-b"] * 4
    # Every same-night pair rises by 0.09 mag per 0.06 airmass: 1.5 mag/airmass,
    # inside the pair bound but outside the physical range.
    extinction = [0.05 + 1.5 * (value - 1.3) for value in airmass]

    result = fit_nightly_extinction_envelope(airmass, extinction, nights)

    assert result.diagnostics.unbounded_slope == pytest.approx(1.5, abs=1e-9)
    assert result.diagnostics.slope == pytest.approx(1.0, abs=1e-9)
    assert result.diagnostics.reliable is False
    assert "SLOPE_OUT_OF_RANGE" in result.diagnostics.reliability_reasons
