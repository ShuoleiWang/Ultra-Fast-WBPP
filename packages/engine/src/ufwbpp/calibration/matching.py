"""WBPP-compatible calibration grouping and pairing.

Raw calibration frames are grouped the way WBPP groups them (type, size,
binning, colour, the filter of a Flat, the exposure of a Dark and the
grouping keywords) and, beyond WBPP, by the acquisition settings WBPP merges
unless told otherwise (camera, gain, offset, readout mode): merging those
would build a master no frame was taken with. Every supplied master is a
group of its own.

Each Light, and each raw Flat group, is then paired with a Bias, a Dark and
(for a Light) a Flat by WBPP's rules:

* compatible: the same size, binning and colour (mono or one Bayer pattern),
  the same filter for a Flat, and no grouping keyword with a different value;
  a frame without a keyword is compatible with every value of it;
* preferred: the most matching keywords (WBPP's "high-compatible" group),
  then the most matching acquisition settings, then for a Dark the closest
  exposure and a temperature within tolerance, then raw frames before a
  supplied master and the lexically first group;
* a raw Flat group takes a Dark only within 0.5 s of its exposure when a Bias
  exists, and the Bias otherwise.

A pairing WBPP would make although the frames differ (a Dark of another
exposure or temperature, another gain or offset, an ambiguous choice, or no
Bias, Dark or Flat at all) is made and reported as a warning. Frames that
cannot calibrate each other (size, binning, colour) are never paired: when
every Bias, Dark or Flat that a Light's keywords (and filter) admit differs
that way, the Light is refused, as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

from .policy import canonical, unknown

BIAS = "BIAS"
DARK = "DARK"
FLAT = "FLAT"
LIGHT = "LIGHT"
KINDS = (BIAS, DARK, FLAT)
# WBPP pairs a raw Flat with a Dark only this close to its exposure when a
# Bias exists (without dark optimization).
FLAT_DARK_EXPOSURE_TOLERANCE_SECONDS = 0.5
# Exposures within this are one exposure (FITS stores them as decimals).
EXPOSURE_EPSILON_SECONDS = 1e-6
SETTINGS = ("camera", "gain", "offset", "readout_mode")
_SETTING_LABELS = {"camera": "camera", "gain": "gain", "offset": "offset", "readout_mode": "readoutMode"}
MATCHING_ALGORITHM = "wbpp-compatible-calibration-matching-v1"


@dataclass(frozen=True, slots=True)
class FrameTraits:
    """What pairing reads of one frame or supplied master."""

    path: str
    kind: str
    supplied_master: bool
    shape: tuple[int, int] | None
    binning: tuple[int | None, int | None]
    color: str
    filter_name: str
    exposure_seconds: float | None
    temperature_celsius: float | None
    camera: Any
    gain: Any
    offset: Any
    readout_mode: Any
    keywords: tuple[tuple[str, str], ...] = ()
    # A Dark's master includes the Bias (raw Darks always do).
    bias_included: bool | None = None

    @property
    def keyword_map(self) -> dict[str, str]:
        return dict(self.keywords)


@dataclass(frozen=True, slots=True)
class CalibrationGroup:
    kind: str
    key: str
    members: tuple[str, ...]
    supplied_master: bool
    traits: FrameTraits

    def serializable(self) -> dict[str, Any]:
        traits = self.traits
        return {
            "kind": self.kind,
            "key": self.key,
            "suppliedMaster": self.supplied_master,
            "memberCount": len(self.members),
            "filter": traits.filter_name if self.kind == FLAT else None,
            "exposureSeconds": traits.exposure_seconds if self.kind == DARK else None,
            "keywords": dict(traits.keywords),
        }


@dataclass(frozen=True, slots=True)
class MatchIssue:
    code: str
    message: str
    details: tuple[tuple[str, Any], ...] = ()

    def serializable(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True, slots=True)
class Pairing:
    """The groups one Light or raw Flat group is calibrated with.

    ``bias`` is the best Bias whenever one fits, also when the Dark already
    includes it (registration previews subtract it on both sides);
    ``needs_bias`` says whether the pixels must be Bias-subtracted.
    """

    bias: str | None = None
    dark: str | None = None
    flat: str | None = None
    warnings: tuple[MatchIssue, ...] = ()
    needs_bias: bool = True
    errors: tuple[MatchIssue, ...] = ()


@dataclass(frozen=True)
class CalibrationMatch:
    groups: dict[str, dict[str, CalibrationGroup]]
    lights: dict[str, Pairing]
    flats: dict[str, Pairing]
    warnings: tuple[MatchIssue, ...]
    errors: tuple[MatchIssue, ...]
    group_of: dict[str, str] = field(default_factory=dict)

    def group(self, kind: str, key: str | None) -> CalibrationGroup | None:
        return None if key is None else self.groups[kind][key]

    def used(self, kind: str) -> set[str]:
        """The groups of ``kind`` a Light or raw Flat group is paired with."""

        attribute = {BIAS: "bias", DARK: "dark", FLAT: "flat"}[kind]
        return {
            key
            for pairing in (*self.lights.values(), *self.flats.values())
            if (key := getattr(pairing, attribute)) is not None
        }

    def serializable(self) -> dict[str, Any]:
        """The receipt form: groups, the pairing of every Light group and
        raw Flat group, and every warning."""

        pairings: dict[tuple[str | None, str | None, str | None], list[str]] = {}
        for path, pairing in sorted(self.lights.items()):
            pairings.setdefault((pairing.bias, pairing.dark, pairing.flat), []).append(path)
        return {
            "algorithm": MATCHING_ALGORITHM,
            "groups": {
                kind: [group.serializable() for _, group in sorted(groups.items())]
                for kind, groups in self.groups.items()
            },
            "lightPairings": [
                {"bias": bias, "dark": dark, "flat": flat, "lightCount": len(paths)}
                for (bias, dark, flat), paths in sorted(pairings.items(), key=lambda item: str(item[0]))
            ],
            "flatPairings": {
                key: {"bias": pairing.bias, "dark": pairing.dark}
                for key, pairing in sorted(self.flats.items())
            },
            "warnings": [issue.serializable() for issue in self.warnings],
            "errors": [issue.serializable() for issue in self.errors],
        }


def _known(value: Any) -> bool:
    return not unknown(value)


def _same_exposure(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and math.isclose(
        left, right, rel_tol=0.0, abs_tol=EXPOSURE_EPSILON_SECONDS
    )


def _color(traits: FrameTraits) -> str:
    value = str(traits.color or "").strip().upper()
    return "UNKNOWN" if value in {"", "UNKNOWN", "UNSPECIFIED"} else value


def physically_compatible(light: FrameTraits, calibration: FrameTraits) -> list[str]:
    """What prevents ``calibration`` from calibrating ``light`` at all: size,
    binning or colour. Empty when nothing does."""

    reasons: list[str] = []
    if light.shape is not None and calibration.shape is not None and light.shape != calibration.shape:
        reasons.append("size")
    for left, right in zip(light.binning, calibration.binning):
        if left is not None and right is not None and left != right:
            reasons.append("binning")
            break
    left_color, right_color = _color(light), _color(calibration)
    if "UNKNOWN" not in {left_color, right_color} and left_color != right_color:
        reasons.append("colour")
    return reasons


def keywords_compatible(left: FrameTraits, right: FrameTraits) -> bool:
    """No keyword carries a different value; a missing one matches any."""

    theirs = right.keyword_map
    return all(theirs.get(name, value) == value for name, value in left.keywords)


def _keyword_matches(left: FrameTraits, right: FrameTraits) -> int:
    theirs = right.keyword_map
    return sum(1 for name, value in left.keywords if theirs.get(name) == value)


def _setting_differences(left: FrameTraits, right: FrameTraits) -> list[str]:
    return [
        name
        for name in SETTINGS
        if _known(getattr(left, name))
        and _known(getattr(right, name))
        and canonical(getattr(left, name)) != canonical(getattr(right, name))
    ]


def _setting_matches(left: FrameTraits, right: FrameTraits) -> int:
    return sum(
        1
        for name in SETTINGS
        if _known(getattr(left, name))
        and _known(getattr(right, name))
        and canonical(getattr(left, name)) == canonical(getattr(right, name))
    )


def _temperature_differs(left: FrameTraits, right: FrameTraits, tolerance: float) -> bool:
    a, b = left.temperature_celsius, right.temperature_celsius
    return (
        a is not None
        and b is not None
        and math.isfinite(a)
        and math.isfinite(b)
        and abs(a - b) > tolerance
    )


def _exposure_distance(left: FrameTraits, right: FrameTraits) -> float:
    if left.exposure_seconds is None or right.exposure_seconds is None:
        return math.inf
    return abs(float(left.exposure_seconds) - float(right.exposure_seconds))


def _group_identity(traits: FrameTraits) -> tuple[Any, ...]:
    """Raw frames with equal identities are integrated into one master."""

    return (
        traits.kind,
        traits.shape,
        traits.binning,
        _color(traits),
        traits.filter_name if traits.kind == FLAT else None,
        round(float(traits.exposure_seconds), 6)
        if traits.kind == DARK and traits.exposure_seconds is not None
        else None,
        traits.keywords,
        *(canonical(getattr(traits, name)) if _known(getattr(traits, name)) else None for name in SETTINGS),
    )


def _base_key(traits: FrameTraits) -> str:
    if traits.kind == BIAS:
        return "bias"
    if traits.kind == DARK:
        exposure = traits.exposure_seconds
        return f"{exposure:.9g}" if exposure is not None else "unknown-exposure"
    return str(traits.filter_name)


def _label_value(value: Any) -> str:
    if value is None:
        return "*"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _labels(group_traits: Sequence[FrameTraits], supplied: Sequence[bool], paths: Sequence[str]) -> list[str]:
    """Short distinguishing labels for groups that share a base key."""

    labels = [[] for _ in group_traits]
    names = sorted({name for traits in group_traits for name, _ in traits.keywords})
    for name in names:
        values = [traits.keyword_map.get(name) for traits in group_traits]
        if len(set(values)) > 1:
            for index, value in enumerate(values):
                labels[index].append(f"{name}={_label_value(value)}")
    for name in SETTINGS:
        values = [
            canonical(getattr(traits, name)) if _known(getattr(traits, name)) else None
            for traits in group_traits
        ]
        if len(set(values)) > 1:
            for index, value in enumerate(values):
                labels[index].append(f"{_SETTING_LABELS[name]}={_label_value(value)}")
    if len(set(supplied)) > 1:
        for index, is_supplied in enumerate(supplied):
            labels[index].append("master" if is_supplied else "raw")
    result = [",".join(parts) for parts in labels]
    if len(set(result)) < len(result):
        stems = [path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for path in paths]
        result = [f"{label},{stem}" if label else stem for label, stem in zip(result, stems)]
    return result


def group_calibration(frames: Iterable[FrameTraits]) -> dict[str, dict[str, CalibrationGroup]]:
    """Calibration groups by kind and stable key.

    A kind with one group per filter (Flat) or exposure (Dark), or one Bias
    group, keeps the plain key (``L``, ``300``, ``bias``); groups that share
    one get a label of what tells them apart (``L|NIGHT=1``).
    """

    raw: dict[tuple[Any, ...], list[FrameTraits]] = {}
    supplied: list[FrameTraits] = []
    for traits in frames:
        if traits.kind not in KINDS:
            raise ValueError(f"not a calibration frame: {traits.kind}")
        if traits.supplied_master:
            supplied.append(traits)
        else:
            raw.setdefault(_group_identity(traits), []).append(traits)
    # Members keep the order they were given in: integration sums frames in
    # order, so the order is part of a master's exact value.
    candidates: list[tuple[FrameTraits, tuple[str, ...], bool]] = [
        (members[0], tuple(member.path for member in members), False)
        for members in raw.values()
    ]
    candidates.extend((traits, (traits.path,), True) for traits in supplied)
    by_base: dict[tuple[str, str], list[tuple[FrameTraits, tuple[str, ...], bool]]] = {}
    for traits, members, is_supplied in candidates:
        by_base.setdefault((traits.kind, _base_key(traits)), []).append((traits, members, is_supplied))
    groups: dict[str, dict[str, CalibrationGroup]] = {kind: {} for kind in KINDS}
    for (kind, base), items in sorted(by_base.items()):
        items.sort(key=lambda item: (item[2], sorted(item[1])))
        if len(items) == 1:
            keys = [base]
        else:
            labels = _labels([item[0] for item in items], [item[2] for item in items], [item[1][0] for item in items])
            keys = [f"{base}|{label}" for label in labels]
        for key, (traits, members, is_supplied) in zip(keys, items):
            groups[kind][key] = CalibrationGroup(kind, key, members, is_supplied, traits)
    return groups


def _issue(code: str, message: str, **details: Any) -> MatchIssue:
    return MatchIssue(code, message, tuple(sorted(details.items())))


def _rank(
    target: FrameTraits,
    groups: Mapping[str, CalibrationGroup],
    *,
    compare_filter: bool,
    dark: bool,
    temperature_tolerance: float,
) -> tuple[list[tuple[tuple[Any, ...], CalibrationGroup]], list[tuple[CalibrationGroup, list[str]]]]:
    ranked: list[tuple[tuple[Any, ...], CalibrationGroup]] = []
    impossible: list[tuple[CalibrationGroup, list[str]]] = []
    for group in groups.values():
        candidate = group.traits
        if compare_filter and candidate.filter_name.casefold() != target.filter_name.casefold():
            continue
        if not keywords_compatible(target, candidate):
            continue
        reasons = physically_compatible(target, candidate)
        if reasons:
            impossible.append((group, reasons))
            continue
        score = (
            -_keyword_matches(target, candidate),
            -_setting_matches(target, candidate),
            _exposure_distance(target, candidate) if dark else 0.0,
            int(_temperature_differs(target, candidate, temperature_tolerance)) if dark else 0,
            int(group.supplied_master),
        )
        ranked.append(((*score, group.key), group))
    ranked.sort(key=lambda item: item[0])
    return ranked, impossible


def _choose(
    target: FrameTraits,
    kind: str,
    groups: Mapping[str, CalibrationGroup],
    *,
    temperature_tolerance: float,
    exposure_limit: float | None = None,
) -> tuple[CalibrationGroup | None, list[MatchIssue], list[tuple[CalibrationGroup, list[str]]]]:
    ranked, impossible = _rank(
        target,
        groups,
        compare_filter=kind == FLAT,
        dark=kind == DARK,
        temperature_tolerance=temperature_tolerance,
    )
    if exposure_limit is not None:
        ranked = [item for item in ranked if _exposure_distance(target, item[1].traits) <= exposure_limit]
    if not ranked:
        return None, [], impossible
    score, chosen = ranked[0]
    warnings: list[MatchIssue] = []
    if len(ranked) > 1 and ranked[1][0][:-1] == score[:-1]:
        warnings.append(
            _issue(
                f"{kind}_MATCH_AMBIGUOUS",
                f"several {kind.lower()} groups fit equally well; the first is used",
                chosen=chosen.key,
                alternatives=[item[1].key for item in ranked[1:] if item[0][:-1] == score[:-1]],
            )
        )
    differences = _setting_differences(target, chosen.traits)
    if differences:
        warnings.append(
            _issue(
                f"{kind}_SETTINGS_DIFFER",
                f"the {kind.lower()} was taken with other acquisition settings ("
                + ", ".join(_SETTING_LABELS[name] for name in differences)
                + ")",
                group=chosen.key,
                fields={
                    _SETTING_LABELS[name]: [canonical(getattr(target, name)), canonical(getattr(chosen.traits, name))]
                    for name in differences
                },
            )
        )
    if kind == DARK:
        if not _same_exposure(target.exposure_seconds, chosen.traits.exposure_seconds):
            warnings.append(
                _issue(
                    "DARK_EXPOSURE_DIFFERS",
                    "no dark of this exposure; the closest one is subtracted unscaled, "
                    "as WBPP does without dark optimization",
                    group=chosen.key,
                    exposureSeconds=[target.exposure_seconds, chosen.traits.exposure_seconds],
                )
            )
        if _temperature_differs(target, chosen.traits, temperature_tolerance):
            warnings.append(
                _issue(
                    "DARK_TEMPERATURE_DIFFERS",
                    f"the dark differs by more than {temperature_tolerance:g} °C",
                    group=chosen.key,
                    temperatureCelsius=[target.temperature_celsius, chosen.traits.temperature_celsius],
                )
            )
    return chosen, warnings, impossible


def match_calibration(
    lights: Sequence[FrameTraits],
    calibration: Sequence[FrameTraits],
    *,
    dark_temperature_tolerance_celsius: float = 3.0,
    groups: Mapping[str, Mapping[str, CalibrationGroup]] | None = None,
    flat_dark_exposure_tolerance_seconds: float = FLAT_DARK_EXPOSURE_TOLERANCE_SECONDS,
) -> CalibrationMatch:
    """Group ``calibration`` and pair every Light and raw Flat group.

    ``flat_dark_exposure_tolerance_seconds`` is how far a raw Flat group's
    Dark may be from its exposure when a Bias exists (WBPP: 0.5 s).
    """

    grouped = {kind: dict(value) for kind, value in (groups or group_calibration(calibration)).items()}
    for kind in KINDS:
        grouped.setdefault(kind, {})
    group_of = {
        member: group.key
        for kind_groups in grouped.values()
        for group in kind_groups.values()
        for member in group.members
    }
    tolerance = float(dark_temperature_tolerance_celsius)
    warnings: dict[tuple[Any, ...], tuple[MatchIssue, list[str]]] = {}
    errors: dict[tuple[Any, ...], tuple[MatchIssue, list[str]]] = {}

    def note(target: dict, issue: MatchIssue, subject: str) -> None:
        key = (issue.code, issue.message, repr(issue.details))
        target.setdefault(key, (issue, []))[1].append(subject)

    flat_pairings: dict[str, Pairing] = {}
    has_bias = bool(grouped[BIAS])
    for key, group in sorted(grouped[FLAT].items()):
        if group.supplied_master:
            continue
        flat = group.traits
        bias, bias_warnings, _ = _choose(flat, BIAS, grouped[BIAS], temperature_tolerance=tolerance)
        dark, dark_warnings, _ = _choose(
            flat,
            DARK,
            grouped[DARK],
            temperature_tolerance=tolerance,
            exposure_limit=flat_dark_exposure_tolerance_seconds if has_bias else None,
        )
        issues = list(bias_warnings)
        if dark is not None:
            issues.extend(item for item in dark_warnings if item.code != "DARK_EXPOSURE_DIFFERS" or not has_bias)
            if dark.traits.bias_included is False and bias is None:
                issues.append(_issue("BIAS_MISSING", "the flat dark excludes the bias and no bias fits", flat=key))
        elif bias is None:
            issues.append(
                _issue(
                    "FLAT_CALIBRATION_MISSING",
                    "no bias or dark fits these flats; they are integrated uncalibrated",
                    flat=key,
                )
            )
        needs_bias = dark is None or dark.traits.bias_included is False
        pairing = Pairing(
            bias=bias.key if bias is not None else None,
            dark=dark.key if dark is not None else None,
            warnings=tuple(issues),
            needs_bias=needs_bias,
        )
        flat_pairings[key] = pairing
        for issue in issues:
            note(warnings, issue, f"flat:{key}")

    light_pairings: dict[str, Pairing] = {}
    flat_filters = {group.traits.filter_name.casefold() for group in grouped[FLAT].values()}

    def refuse(
        kind: str, impossible: list[tuple[CalibrationGroup, list[str]]], light: FrameTraits, refusals: list[MatchIssue]
    ) -> None:
        reasons = sorted({reason for _, found in impossible for reason in found})
        issue = _issue(
            f"{kind}_INCOMPATIBLE",
            f"no {kind.lower()} can calibrate these lights: the " + ", ".join(reasons) + " differ",
            groups=sorted(group.key for group, _ in impossible),
            **({"filter": light.filter_name} if kind == FLAT else {}),
        )
        refusals.append(issue)
        note(errors, issue, light.path)

    for light in lights:
        issues: list[MatchIssue] = []
        refusals: list[MatchIssue] = []
        flat, flat_warnings, flat_impossible = _choose(light, FLAT, grouped[FLAT], temperature_tolerance=tolerance)
        issues.extend(flat_warnings)
        if flat is None and flat_impossible:
            refuse(FLAT, flat_impossible, light, refusals)
        elif flat is None:
            issues.append(
                _issue(
                    "FLAT_MISSING",
                    f"no flat fits filter {light.filter_name}"
                    + (" (the flats of this filter carry other keyword values)" if light.filter_name.casefold() in flat_filters else "")
                    + "; the lights are not flat-fielded",
                    filter=light.filter_name,
                )
            )
        dark, dark_warnings, dark_impossible = _choose(light, DARK, grouped[DARK], temperature_tolerance=tolerance)
        issues.extend(dark_warnings)
        if dark is None and dark_impossible:
            refuse(DARK, dark_impossible, light, refusals)
        elif dark is None and grouped[DARK]:
            issues.append(_issue("DARK_MISSING", "no dark fits these lights; only the bias is subtracted"))
        needs_bias = dark is None or dark.traits.bias_included is False
        bias, bias_warnings, bias_impossible = _choose(light, BIAS, grouped[BIAS], temperature_tolerance=tolerance)
        if needs_bias:
            issues.extend(bias_warnings)
            if bias is None and bias_impossible:
                refuse(BIAS, bias_impossible, light, refusals)
            elif bias is None:
                issues.append(
                    _issue(
                        "BIAS_MISSING",
                        "no bias fits these lights and no dark includes it; the pedestal is not subtracted",
                    )
                )
        light_pairings[light.path] = Pairing(
            bias=bias.key if bias is not None else None,
            dark=dark.key if dark is not None else None,
            flat=flat.key if flat is not None else None,
            warnings=tuple(issues),
            needs_bias=needs_bias,
            errors=tuple(refusals),
        )
        for issue in issues:
            note(warnings, issue, light.path)

    def collected(items: Mapping[tuple[Any, ...], tuple[MatchIssue, list[str]]]) -> tuple[MatchIssue, ...]:
        result = []
        for issue, subjects in items.values():
            details = dict(issue.details)
            details["frameCount"] = len(subjects)
            result.append(MatchIssue(issue.code, issue.message, tuple(sorted(details.items()))))
        return tuple(sorted(result, key=lambda issue: (issue.code, repr(issue.details))))

    return CalibrationMatch(
        groups=grouped,
        lights=light_pairings,
        flats=flat_pairings,
        warnings=collected(warnings),
        errors=collected(errors),
        group_of=group_of,
    )


__all__ = [
    "BIAS",
    "DARK",
    "EXPOSURE_EPSILON_SECONDS",
    "FLAT",
    "FLAT_DARK_EXPOSURE_TOLERANCE_SECONDS",
    "LIGHT",
    "MATCHING_ALGORITHM",
    "CalibrationGroup",
    "CalibrationMatch",
    "FrameTraits",
    "MatchIssue",
    "Pairing",
    "group_calibration",
    "keywords_compatible",
    "match_calibration",
    "physically_compatible",
]
