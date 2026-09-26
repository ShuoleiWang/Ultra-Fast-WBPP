from __future__ import annotations

from dataclasses import replace

from ufwbpp.calibration.matching import (
    BIAS,
    DARK,
    FLAT,
    LIGHT,
    FrameTraits,
    group_calibration,
    match_calibration,
)


def frame(
    path: str,
    kind: str,
    *,
    filter_name: str = "L",
    exposure: float | None = 300.0,
    keywords: dict[str, str] | None = None,
    supplied: bool = False,
    binning: int = 1,
    shape: tuple[int, int] = (100, 150),
    color: str = "NONE",
    gain: float | None = 100.0,
    temperature: float | None = -10.0,
    bias_included: bool | None = None,
) -> FrameTraits:
    return FrameTraits(
        path=path,
        kind=kind,
        supplied_master=supplied,
        shape=shape,
        binning=(binning, binning),
        color=color,
        filter_name=filter_name if kind in {FLAT, LIGHT} else "UNKNOWN",
        exposure_seconds=exposure if kind != BIAS else 0.0,
        temperature_celsius=temperature,
        camera="ASI2600MM",
        gain=gain,
        offset=50.0,
        readout_mode="MODE 0",
        keywords=tuple(sorted((keywords or {}).items())),
        bias_included=(True if kind == DARK and not supplied and bias_included is None else bias_included),
    )


def test_flats_are_paired_night_by_night_through_grouping_keywords() -> None:
    lights = [frame(f"/d/NIGHT_{night}/L/l{night}.fits", LIGHT, keywords={"NIGHT": str(night)}) for night in (1, 2)]
    flats = [
        frame(f"/d/NIGHT_{night}/Flats/f{night}_{index}.fits", FLAT, exposure=2.0, keywords={"NIGHT": str(night)})
        for night in (1, 2)
        for index in range(3)
    ]
    bias = [frame(f"/d/Bias/b{index}.fits", BIAS) for index in range(3)]
    match = match_calibration(lights, [*flats, *bias])
    assert sorted(match.groups[FLAT]) == ["L|NIGHT=1", "L|NIGHT=2"]
    assert match.lights["/d/NIGHT_1/L/l1.fits"].flat == "L|NIGHT=1"
    assert match.lights["/d/NIGHT_2/L/l2.fits"].flat == "L|NIGHT=2"
    assert all(pairing.bias == "bias" for pairing in match.lights.values())
    assert {pairing.bias for pairing in match.flats.values()} == {"bias"}
    assert match.errors == ()
    assert match.warnings == ()


def test_a_flat_without_the_keyword_fits_every_night_and_the_closer_one_wins() -> None:
    lights = [frame(f"/d/NIGHT_{night}/l.fits", LIGHT, keywords={"NIGHT": str(night)}) for night in (1, 2)]
    flats = [
        frame("/d/NIGHT_1/masterFlat_L.xisf", FLAT, keywords={"NIGHT": "1"}, supplied=True),
        frame("/d/masterFlat_L.xisf", FLAT, supplied=True),
    ]
    match = match_calibration(lights, [*flats, frame("/d/masterBias.xisf", BIAS, supplied=True)])
    by_path = {group.members[0]: key for key, group in match.groups[FLAT].items()}
    assert match.lights["/d/NIGHT_1/l.fits"].flat == by_path["/d/NIGHT_1/masterFlat_L.xisf"]
    assert match.lights["/d/NIGHT_2/l.fits"].flat == by_path["/d/masterFlat_L.xisf"]
    assert match.warnings == () and match.errors == ()


def test_flats_of_other_nights_only_leave_the_lights_unflattened_with_a_warning() -> None:
    lights = [frame("/d/NIGHT_3/l.fits", LIGHT, keywords={"NIGHT": "3"})]
    flats = [frame("/d/NIGHT_1/masterFlat.xisf", FLAT, keywords={"NIGHT": "1"}, supplied=True)]
    match = match_calibration(lights, [*flats, frame("/d/masterBias.xisf", BIAS, supplied=True)])
    assert match.lights["/d/NIGHT_3/l.fits"].flat is None
    assert [issue.code for issue in match.warnings] == ["FLAT_MISSING"]
    assert match.errors == ()


def test_calibration_of_another_size_binning_or_colour_is_refused() -> None:
    lights = [frame("/d/l.fits", LIGHT)]
    for changed in (
        {"binning": 2},
        {"shape": (200, 300)},
        {"color": "RGGB"},
    ):
        flats = [frame("/d/masterFlat.xisf", FLAT, supplied=True, **changed)]
        match = match_calibration(lights, [*flats, frame("/d/masterBias.xisf", BIAS, supplied=True)])
        assert [issue.code for issue in match.errors] == ["FLAT_INCOMPATIBLE"], changed
        assert match.lights["/d/l.fits"].flat is None


def test_darks_follow_wbpp_closest_exposure_with_warnings() -> None:
    light = frame("/d/l.fits", LIGHT, exposure=300.0)
    darks = [
        frame("/d/masterDark_240s.xisf", DARK, exposure=240.0, supplied=True, bias_included=True),
        frame("/d/masterDark_280s.xisf", DARK, exposure=280.0, supplied=True, bias_included=True, temperature=0.0),
    ]
    match = match_calibration([light], [*darks, frame("/d/masterFlat.xisf", FLAT, supplied=True)])
    pairing = match.lights["/d/l.fits"]
    assert pairing.dark == "280"
    assert pairing.needs_bias is False  # the dark includes the bias
    assert {issue.code for issue in match.warnings} == {"DARK_EXPOSURE_DIFFERS", "DARK_TEMPERATURE_DIFFERS"}
    exact = replace(darks[0], path="/d/masterDark_300s.xisf", exposure_seconds=300.0)
    match = match_calibration([light], [*darks, exact, frame("/d/masterFlat.xisf", FLAT, supplied=True)])
    assert match.lights["/d/l.fits"].dark == "300"
    assert match.warnings == ()


def test_raw_darks_of_two_gains_become_two_masters_and_the_light_takes_its_gain() -> None:
    light = frame("/d/l.fits", LIGHT, gain=100.0)
    darks = [frame(f"/d/darks/d{gain:g}_{index}.fits", DARK, gain=gain) for gain in (0.0, 100.0) for index in range(2)]
    match = match_calibration([light], [*darks, frame("/d/masterFlat.xisf", FLAT, supplied=True)])
    assert sorted(match.groups[DARK]) == ["300|gain=0", "300|gain=100"]
    assert match.lights["/d/l.fits"].dark == "300|gain=100"
    assert match.warnings == ()
    # Only the other gain: paired as WBPP would, and reported.
    other = [dark for dark in darks if dark.gain == 0.0]
    match = match_calibration([light], [*other, frame("/d/masterFlat.xisf", FLAT, supplied=True)])
    assert match.lights["/d/l.fits"].dark == "300"
    assert [issue.code for issue in match.warnings] == ["DARK_SETTINGS_DIFFER"]


def test_raw_flats_take_a_dark_only_within_half_a_second_when_a_bias_exists() -> None:
    flats = [frame(f"/d/flats/f{index}.fits", FLAT, exposure=2.2) for index in range(3)]
    near = [frame(f"/d/flatdarks/d{index}.fits", DARK, exposure=2.0) for index in range(3)]
    far = [frame(f"/d/darks/d{index}.fits", DARK, exposure=5.0) for index in range(3)]
    bias = [frame(f"/d/bias/b{index}.fits", BIAS) for index in range(3)]
    light = frame("/d/l.fits", LIGHT, exposure=5.0)
    match = match_calibration([light], [*flats, *near, *far, *bias])
    assert match.flats["L"].dark == "2"
    assert match.flats["L"].needs_bias is False  # raw darks include the bias
    match = match_calibration([light], [*flats, *far, *bias])
    assert match.flats["L"].dark is None and match.flats["L"].bias == "bias"
    assert match.warnings == ()


def test_equal_candidates_are_reported_and_raw_frames_come_first() -> None:
    light = frame("/d/l.fits", LIGHT)
    masters = [frame(f"/d/{name}.xisf", FLAT, supplied=True) for name in ("masterFlat_a", "masterFlat_b")]
    match = match_calibration([light], [*masters, frame("/d/masterBias.xisf", BIAS, supplied=True)])
    assert [issue.code for issue in match.warnings] == ["FLAT_MATCH_AMBIGUOUS"]
    raw = [frame(f"/d/flats/f{index}.fits", FLAT) for index in range(2)]
    match = match_calibration([light], [masters[0], *raw, frame("/d/masterBias.xisf", BIAS, supplied=True)])
    assert match.groups[FLAT][match.lights["/d/l.fits"].flat].supplied_master is False


def test_one_group_per_filter_exposure_and_bias_keeps_the_plain_keys() -> None:
    frames = [
        frame("/s/DATE_0322/masterFlat_R.xisf", FLAT, filter_name="R", keywords={"DATE": "0322"}, supplied=True),
        frame("/s/DATE_0327/masterFlat_L.xisf", FLAT, keywords={"DATE": "0327"}, supplied=True),
        frame("/m/masterDark_300s.xisf", DARK, supplied=True, bias_included=True),
        frame("/m/masterBias.xisf", BIAS, supplied=True),
    ]
    groups = group_calibration(frames)
    assert sorted(groups[FLAT]) == ["L", "R"]
    assert list(groups[DARK]) == ["300"] and list(groups[BIAS]) == ["bias"]
    lights = [
        frame("/s/DATE_0322/R/l.fits", LIGHT, filter_name="R", keywords={"DATE": "0322"}),
        frame("/s/DATE_0327/L/l.fits", LIGHT, keywords={"DATE": "0327"}),
    ]
    match = match_calibration(lights, frames)
    assert match.lights["/s/DATE_0322/R/l.fits"].flat == "R"
    assert match.lights["/s/DATE_0327/L/l.fits"].flat == "L"
    assert {pairing.dark for pairing in match.lights.values()} == {"300"}
    assert match.warnings == () and match.errors == ()


def test_missing_bias_and_dark_are_warnings_not_refusals() -> None:
    light = frame("/d/l.fits", LIGHT)
    match = match_calibration([light], [frame("/d/masterFlat.xisf", FLAT, supplied=True)])
    pairing = match.lights["/d/l.fits"]
    assert (pairing.bias, pairing.dark, pairing.flat) == (None, None, "L")
    assert [issue.code for issue in match.warnings] == ["BIAS_MISSING"]
    assert match.errors == ()
