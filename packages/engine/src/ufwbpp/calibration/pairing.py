"""WBPP-compatible calibration pairing of frame metadata (``FrameInfo``).

The frame-level side of :mod:`ufwbpp.calibration.matching`: the pixel
pipeline and the registration calibration both pair their inputs here, so a
run's calibration groups and pairings are the same at every stage.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Mapping

from ..stacking.integration import CalibrationError, FrameInfo
from .inputs import assert_compatible, frame_traits
from .matching import (
    BIAS,
    DARK,
    EXPOSURE_EPSILON_SECONDS,
    FLAT,
    FLAT_DARK_EXPOSURE_TOLERANCE_SECONDS,
    LIGHT,
    CalibrationMatch,
    MatchIssue,
    match_calibration,
)
from .policy import MONO_STANDARD, unknown


def pair_calibration(
    *,
    bias_info: Mapping[Path, FrameInfo],
    master_bias_info: Mapping[Path, FrameInfo],
    dark_info: Mapping[Path, FrameInfo],
    master_dark_info: Mapping[Path, FrameInfo],
    flat_info: Mapping[Path, FrameInfo],
    master_flat_info: Mapping[Path, FrameInfo],
    light_info: Mapping[Path, FrameInfo],
    supplied_dark_bias_included: Mapping[Path, bool],
    workflow: str,
    dark_temperature_tolerance_celsius: float,
) -> CalibrationMatch:
    """Group the calibration frames and pair every Light and raw Flat group
    by WBPP's rules; a pairing no frame can serve is refused.

    The strict workflow additionally refuses every pairing WBPP would make
    despite unknown or differing metadata, as it always has.
    """

    def traits(path: Path, info: FrameInfo, kind: str, supplied: bool, bias_included: bool | None = None) -> Any:
        return replace(
            frame_traits(info, kind, supplied_master=supplied, workflow=workflow, bias_included=bias_included),
            path=str(path),
        )

    for path, info in (*dark_info.items(), *master_dark_info.items()):
        if info.exposure_seconds is None or info.exposure_seconds <= 0:
            raise CalibrationError("DARK_EXPOSURE_UNKNOWN", "Dark requires positive EXPTIME", path=str(path))
    for path, info in (*flat_info.items(), *master_flat_info.items()):
        if unknown(info.filter_name):
            raise CalibrationError(
                "FILTER_UNKNOWN", "Flat and Light frames require FILTER metadata", path=str(path)
            )
    calibration = [
        *(traits(path, info, BIAS, False) for path, info in bias_info.items()),
        *(traits(path, info, BIAS, True) for path, info in master_bias_info.items()),
        *(traits(path, info, DARK, False, True) for path, info in dark_info.items()),
        *(
            traits(path, info, DARK, True, supplied_dark_bias_included[path])
            for path, info in master_dark_info.items()
        ),
        *(traits(path, info, FLAT, False) for path, info in flat_info.items()),
        *(traits(path, info, FLAT, True) for path, info in master_flat_info.items()),
    ]
    lights = [traits(path, info, LIGHT, False) for path, info in light_info.items()]
    strict = workflow != MONO_STANDARD
    match = match_calibration(
        lights,
        calibration,
        dark_temperature_tolerance_celsius=dark_temperature_tolerance_celsius,
        flat_dark_exposure_tolerance_seconds=EXPOSURE_EPSILON_SECONDS if strict else FLAT_DARK_EXPOSURE_TOLERANCE_SECONDS,
    )
    if match.errors:
        first = match.errors[0]
        raise CalibrationError(
            first.code,
            "; ".join(issue.message for issue in match.errors),
        )
    if strict:
        infos = {
            str(path): info
            for group in (bias_info, master_bias_info, dark_info, master_dark_info, flat_info, master_flat_info)
            for path, info in group.items()
        }
        _assert_strict_pairings(match, infos, light_info, workflow, dark_temperature_tolerance_celsius)
    return match


_STRICT_AMBIGUITY_CODES = {
    "BIAS_MATCH_AMBIGUOUS": "BIAS_SOURCE_AMBIGUOUS",
    "DARK_MATCH_AMBIGUOUS": "MASTER_DARK_AMBIGUOUS",
    "FLAT_MATCH_AMBIGUOUS": "MASTER_FLAT_AMBIGUOUS",
}


def _assert_strict_pairings(
    match: CalibrationMatch,
    infos: Mapping[str, FrameInfo],
    light_info: Mapping[Path, FrameInfo],
    workflow: str,
    tolerance: float,
) -> None:
    """The strict workflow pairs only frames whose metadata is known and
    equal, with exact Dark exposures: every WBPP relaxation fails closed."""
    # Every input shares the known acquisition profile of the first Bias
    # (or Light): an unknown or differing value is refused.
    biases = [info for key in sorted(match.groups[BIAS]) for info in (infos[match.groups[BIAS][key].members[0]],)]
    profile = biases[0] if biases else next(iter(light_info.values()))
    for info in (*infos.values(), *light_info.values()):
        assert_compatible(profile, info, workflow=workflow)
    for issue in match.warnings:
        if issue.code in _STRICT_AMBIGUITY_CODES:
            raise CalibrationError(_STRICT_AMBIGUITY_CODES[issue.code], issue.message)
    for kind in (BIAS, DARK, FLAT):
        sources: dict[str, set[bool]] = {}
        for key, group in match.groups[kind].items():
            sources.setdefault(key.split("|", 1)[0], set()).add(group.supplied_master)
        for base, kinds in sources.items():
            if kinds == {True, False}:
                raise CalibrationError(
                    f"{kind}_SOURCE_AMBIGUOUS",
                    f"{kind.lower()} {base} is supplied as raw frames and as a master",
                )

    def reference(kind: str, key: str) -> FrameInfo:
        return infos[match.groups[kind][key].members[0]]

    for path, pairing in match.lights.items():
        light = light_info[Path(path)]
        if pairing.flat is None:
            raise CalibrationError(
                "MASTER_FLAT_MISSING",
                f"no raw Flat group or MasterFlat for Light filters: {light.filter_name}",
            )
        assert_compatible(light, reference(FLAT, pairing.flat), compare_filter=True, workflow=workflow)
        if match.groups[DARK] and (
            pairing.dark is None
            or not math.isclose(
                float(light.exposure_seconds),
                float(reference(DARK, pairing.dark).exposure_seconds),
                rel_tol=0.0,
                abs_tol=EXPOSURE_EPSILON_SECONDS,
            )
        ):
            raise CalibrationError(
                "DARK_EXPOSURE_MISMATCH",
                "no exact raw Dark or MasterDark exposure matches this Light",
                path=path,
            )
        if pairing.dark is not None:
            assert_compatible(
                light,
                reference(DARK, pairing.dark),
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=tolerance,
                workflow=workflow,
            )
        needs_bias = pairing.dark is None or match.groups[DARK][pairing.dark].traits.bias_included is False
        if needs_bias and pairing.bias is None:
            raise CalibrationError(
                "BIAS_REQUIRED_FOR_CALIBRATION",
                "Bias is required unless every Light and raw Flat has a matching Dark that includes Bias.",
                path=path,
            )
        if pairing.bias is not None:
            assert_compatible(reference(BIAS, pairing.bias), light, workflow=workflow)
    for key, pairing in match.flats.items():
        flat = reference(FLAT, key)
        if pairing.dark is not None:
            assert_compatible(
                flat,
                reference(DARK, pairing.dark),
                compare_exposure=True,
                compare_temperature=True,
                temperature_tolerance_celsius=tolerance,
                workflow=workflow,
            )
        elif pairing.bias is None:
            raise CalibrationError(
                "BIAS_REQUIRED_FOR_CALIBRATION",
                "Bias is required unless every Light and raw Flat has a matching Dark that includes Bias.",
            )
        if pairing.bias is not None:
            assert_compatible(reference(BIAS, pairing.bias), flat, workflow=workflow)


def light_group_warnings(
    light_groups: Mapping[str, list[Path]],
    light_info: Mapping[Path, FrameInfo],
) -> tuple[MatchIssue, ...]:
    """Lights of one filter taken with other acquisition settings are
    stacked together, as WBPP does; the difference is reported."""

    issues: list[MatchIssue] = []
    for filter_name, paths in sorted(light_groups.items()):
        differing = {}
        for name, label in (("camera", "camera"), ("gain", "gain"), ("offset", "offset"), ("readout_mode", "readoutMode")):
            values = sorted({str(getattr(light_info[path], name)) for path in paths if not unknown(getattr(light_info[path], name))})
            if len(values) > 1:
                differing[label] = values
        if differing:
            issues.append(
                MatchIssue(
                    "LIGHT_SETTINGS_DIFFER",
                    f"the Lights of filter {filter_name} were taken with other acquisition settings",
                    (("fields", differing), ("filter", filter_name)),
                )
            )
    return tuple(issues)


def dark_group_warnings(
    calibration: CalibrationMatch,
    dark_info: Mapping[Path, FrameInfo],
    tolerance: float,
) -> tuple[MatchIssue, ...]:
    """Raw Darks of one group whose temperatures spread beyond tolerance are
    integrated together, as WBPP does; the spread is reported."""
    issues: list[MatchIssue] = []
    for key, group in sorted(calibration.groups[DARK].items()):
        if group.supplied_master:
            continue
        temperatures = [
            dark_info[Path(member)].temperature_celsius
            for member in group.members
            if dark_info[Path(member)].temperature_celsius is not None
        ]
        if temperatures and max(temperatures) - min(temperatures) > tolerance:
            issues.append(
                MatchIssue(
                    "DARK_GROUP_TEMPERATURE_SPREAD",
                    f"the raw darks of group {key} span more than {tolerance:g} °C",
                    (("group", key), ("temperatureCelsius", [min(temperatures), max(temperatures)])),
                )
            )
    return tuple(issues)


__all__ = ["dark_group_warnings", "light_group_warnings", "pair_calibration"]
