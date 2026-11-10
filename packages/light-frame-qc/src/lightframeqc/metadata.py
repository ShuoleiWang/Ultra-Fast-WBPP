from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from astropy.coordinates import Angle
import astropy.units as u

from .models import FrameMetadata, FrameRole
from .cfa import normalize_pattern as normalize_cfa_pattern


# The channel of Lights and Flats that name no filter (WBPP's "NoFilter"):
# one-shot colour without a wheel, a mono camera without one.
NO_FILTER = "NOFILTER"
# The one filter identity. Inventory, plan, preflight, QC, the pixel reader and
# the project layout all name a filter through ``normalize_filter``, so Lights
# written "Red" and Flats written "R" are one filter at every stage.
_FILTER_ALIASES = {
    "L": "L",
    "LUM": "L",
    "LUMINANCE": "L",
    "R": "R",
    "RED": "R",
    "G": "G",
    "GREEN": "G",
    "B": "B",
    "BLUE": "B",
    "HA": "HA",
    "HALPHA": "HA",
    "HΑ": "HA",
    "HYDROGENALPHA": "HA",
    "O3": "OIII",
    "OIII": "OIII",
    "OXYGENIII": "OIII",
    "S2": "SII",
    "SII": "SII",
    "SULFURII": "SII",
    "SULPHURII": "SII",
    "NOFILTER": NO_FILTER,
}
# What a header writes when no filter is in the light path.
_FILTERLESS_NAMES = frozenset({"UNKNOWN", "NONE", "UNSPECIFIED"})


# The one frame-role table: IMAGETYP / XISF imageType values and the folder
# or file-name words that name a role, compared without case, spaces or
# punctuation. Inventory, QC and the pixel reader all read it, so a spelling
# is the same role at every stage. It holds what acquisition software writes
# (N.I.N.A. LIGHT/FLAT/DARK/BIAS/DARKFLAT; MaxIm DL and INDI/Ekos "Light
# Frame", "Dark Frame", "Bias Frame", "Flat Frame"; SBIG's FITS extensions
# "Flat Field"; ASIAIR "Light", "Dark", "Flat", "Bias"; IRAF "object",
# "zero") and PixInsight's words ("Science", "Master Dark"). A type that is
# not a calibration role (SNAPSHOT, FOCUS, PREVIEW, "Tricolor Image") is
# deliberately absent: it stays unknown and is never taken for a Light.
_ROLE_ALIASES = {
    "LIGHT": FrameRole.LIGHT,
    "LIGHTS": FrameRole.LIGHT,
    "LIGHTFRAME": FrameRole.LIGHT,
    "LIGHTFRAMES": FrameRole.LIGHT,
    "SCIENCE": FrameRole.LIGHT,
    "SCIENCEFRAME": FrameRole.LIGHT,
    "SCIENCEFRAMES": FrameRole.LIGHT,
    "OBJECT": FrameRole.LIGHT,
    "FLAT": FrameRole.RAW_FLAT,
    "FLATS": FrameRole.RAW_FLAT,
    "FLATFRAME": FrameRole.RAW_FLAT,
    "FLATFRAMES": FrameRole.RAW_FLAT,
    "FLATFIELD": FrameRole.RAW_FLAT,
    "FLATFIELDS": FrameRole.RAW_FLAT,
    "RAWFLAT": FrameRole.RAW_FLAT,
    "SKYFLAT": FrameRole.RAW_FLAT,
    "DOMEFLAT": FrameRole.RAW_FLAT,
    "TWILIGHTFLAT": FrameRole.RAW_FLAT,
    "MASTERFLAT": FrameRole.MASTER_FLAT,
    "MASTERFLATFRAME": FrameRole.MASTER_FLAT,
    "MASTERFLATFIELD": FrameRole.MASTER_FLAT,
    "DARK": FrameRole.DARK,
    "DARKS": FrameRole.DARK,
    "DARKFRAME": FrameRole.DARK,
    "DARKFRAMES": FrameRole.DARK,
    # A Flat-Dark (N.I.N.A. IMAGETYP DARKFLAT, WBPP "Dark Flat") is a Dark at
    # the Flats' exposure: a Light takes the Dark of its own exposure, so it
    # calibrates the Flats.
    "DARKFLAT": FrameRole.DARK,
    "DARKFLATS": FrameRole.DARK,
    "FLATDARK": FrameRole.DARK,
    "FLATDARKS": FrameRole.DARK,
    "MASTERDARK": FrameRole.MASTER_DARK,
    "MASTERDARKFRAME": FrameRole.MASTER_DARK,
    "MASTERDARKFLAT": FrameRole.MASTER_DARK,
    "MASTERFLATDARK": FrameRole.MASTER_DARK,
    "BIAS": FrameRole.BIAS,
    "BIASES": FrameRole.BIAS,
    "BIASFRAME": FrameRole.BIAS,
    "BIASFRAMES": FrameRole.BIAS,
    "ZERO": FrameRole.BIAS,
    "ZEROS": FrameRole.BIAS,
    "ZEROFRAME": FrameRole.BIAS,
    "MASTERBIAS": FrameRole.MASTER_BIAS,
    "MASTERBIASFRAME": FrameRole.MASTER_BIAS,
    "MASTERZERO": FrameRole.MASTER_BIAS,
    "MASTERLIGHT": FrameRole.MASTER_LIGHT,
}
# Role evidence of a frame whose header and path name no type.
ROLE_DEFAULT_EVIDENCE = "DEFAULT=LIGHT"

# Folders WBPP writes its intermediate frames to, and the suffixes it appends
# (_c calibrated, _cc cosmetized, _d debayered, _r registered).
_WBPP_PROCESSED_FOLDERS = frozenset({"CALIBRATED", "COSMETIZED", "DEBAYERED", "REGISTERED"})
_WBPP_PROCESSED_SUFFIX = re.compile(r"(?:_c(?:_cc)?(?:_d)?(?:_r)?|_cc(?:_d)?(?:_r)?|_d(?:_r)?|_r)$", re.IGNORECASE)
# WBPP grouping keywords recognised without configuration. Each groups the
# calibration: a calibration frame and a Light are compatible unless they
# carry different values of one keyword (a frame without the keyword matches
# every value). PANEL also separates the mosaic panels of one target.
DEFAULT_GROUPING_KEYWORDS = ("DATE", "NIGHT", "PANEL", "SESSION")
SESSION_KEYWORDS = ("DATE", "NIGHT", "SESSION")
# WBPP's syntax for a keyword in a folder or file name: the name after a
# non-alphanumeric character, then "_", "-" or a space, then the value in the
# characters WBPP accepts; an underscore ends the value (NIGHT_2, DATE-0322).
_KEYWORD_VALUE = r"[_\- ]([-():. A-Za-z0-9]+)"
_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {}
# Smart naming: properties a path supplies when the header lacks them.
_SMART_FILTER = re.compile(r"(?:^|[^A-Za-z0-9])(?:FILTER|INSFLNAM)[_\- ]([A-Za-z0-9+]+)", re.IGNORECASE)
_SMART_BINNING = re.compile(r"(?:^|[^A-Za-z0-9])(?:XBINNING|BINNING|CCDBINX|BIN)[_\- ]?([1-9])(?:x[1-9])?(?![0-9])", re.IGNORECASE)
_SMART_EXPOSURE = re.compile(r"(?:^|[^A-Za-z0-9])(?:EXPTIME|EXPOSURE)[_\- ]([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_BARE_EXPOSURE = re.compile(r"(?:^|[_\- ])([0-9]+(?:\.[0-9]+)?)s(?:[_\- .]|$)", re.IGNORECASE)
_SMART_BAYER = re.compile(r"(?:^|[^A-Za-z0-9])BAYERPAT[_\- ]([A-Za-z]{4})(?![A-Za-z])", re.IGNORECASE)


def _header_lookup(header: dict[str, Any], names: Iterable[str]) -> Any | None:
    folded = {str(key).upper(): value for key, value in header.items()}
    for name in names:
        value = folded.get(name.upper())
        if value is not None and str(value).strip() != "":
            return value
    return None


def _clean_text(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("'").strip('"').strip()
    return text or None


def _number(value: Any | None) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_filter(value: Any | None) -> str:
    """The canonical name of a filter: an alias spelling (Lum, Red, H-alpha,
    O3 ...) names its filter, any other text is the filter in upper case, and
    no text is ``UNKNOWN``."""

    text = _clean_text(value)
    if text is None:
        return "UNKNOWN"
    compact = re.sub(r"[\W_]+", "", text.upper())
    return _FILTER_ALIASES.get(compact, text.upper())


def filter_or_filterless(name: str) -> str:
    """The filter of a Light or Flat: one that names none belongs to the
    filterless channel ``NO_FILTER``, as in WBPP."""

    return NO_FILTER if name in _FILTERLESS_NAMES else name


def parse_frame_role(value: Any | None) -> FrameRole:
    """Normalize FITS/XISF image-type vocabulary without guessing.

    The compact representation accepts common spacing, underscore and hyphen
    variants (for example ``"Master Flat"`` and ``"MasterFlat"``) while
    unknown explicit values remain unknown.
    """

    text = _clean_text(value)
    if text is None:
        return FrameRole.UNKNOWN
    compact = re.sub(r"[^A-Z0-9]+", "", text.upper())
    return _ROLE_ALIASES.get(compact, FrameRole.UNKNOWN)


def infer_frame_role_from_path(path: str) -> FrameRole:
    """Use conservative smart-name hints only when headers provide no role."""

    file_compact = re.sub(r"[^A-Z0-9]+", "", Path(path).stem.upper())
    for marker, role in (
        ("MASTERFLATDARK", FrameRole.MASTER_DARK),
        ("MASTERDARKFLAT", FrameRole.MASTER_DARK),
        ("MASTERFLAT", FrameRole.MASTER_FLAT),
        ("MASTERDARK", FrameRole.MASTER_DARK),
        ("MASTERBIAS", FrameRole.MASTER_BIAS),
        ("MASTERZERO", FrameRole.MASTER_BIAS),
        ("MASTERLIGHT", FrameRole.MASTER_LIGHT),
    ):
        if marker in file_compact:
            return role

    # Exact tokens avoid treating an unrelated parent such as
    # ``light-frame-qc`` as evidence that every contained file is a Light.
    tokens: list[str] = []
    for component in (*Path(path).parts[-4:-1], Path(path).stem):
        tokens.extend(
            token
            for token in re.split(r"[^A-Z0-9]+", component.upper())
            if token
        )
    for token in reversed(tokens):
        role = _ROLE_ALIASES.get(token, FrameRole.UNKNOWN)
        if role != FrameRole.UNKNOWN:
            return role
    return FrameRole.UNKNOWN


def _normalize_role(metadata: FrameMetadata) -> None:
    xisf_value = _header_lookup(
        metadata.header, ("XISF:IMAGETYPE", "XISF:IMAGE_TYPE")
    )
    fits_value = _header_lookup(metadata.header, ("IMAGETYP", "IMAGETYPE"))
    xisf_role = parse_frame_role(xisf_value)
    fits_role = parse_frame_role(fits_value)
    xisf_explicit = _clean_text(xisf_value) is not None
    fits_explicit = _clean_text(fits_value) is not None

    evidence: list[str] = []
    if _clean_text(xisf_value) is not None:
        evidence.append(f"XISF:imageType={_clean_text(xisf_value)}")
    if _clean_text(fits_value) is not None:
        evidence.append(f"FITS:IMAGETYP={_clean_text(fits_value)}")

    conflicts: list[str] = []
    if (
        xisf_role != FrameRole.UNKNOWN
        and fits_role != FrameRole.UNKNOWN
        and xisf_role != fits_role
    ):
        conflicts.append(
            "role disagreement: "
            f"XISF:imageType={_clean_text(xisf_value)} -> {xisf_role.value}; "
            f"FITS:IMAGETYP={_clean_text(fits_value)} -> {fits_role.value}"
        )
        role = FrameRole.UNKNOWN
    elif xisf_role != FrameRole.UNKNOWN:
        role = xisf_role
    elif fits_role != FrameRole.UNKNOWN:
        role = fits_role
    elif xisf_explicit or fits_explicit:
        # An explicit but unknown image type (for example FOCUS or PREVIEW)
        # must not be promoted by a LIGHT directory or a masterFlat filename.
        role = FrameRole.UNKNOWN
    else:
        role = infer_frame_role_from_path(metadata.path)
        if role != FrameRole.UNKNOWN:
            evidence.append(f"PATH={role.value}")
        else:
            # WBPP takes a frame whose header and path name no image type
            # for a Light; inventory, QC and the run all read this role.
            role = FrameRole.LIGHT
            evidence.append(ROLE_DEFAULT_EVIDENCE)

    metadata.role = role
    metadata.role_evidence = evidence
    metadata.role_conflicts = conflicts


def _normalize_cfa_pattern(value: Any | None) -> str:
    return normalize_cfa_pattern(_clean_text(value))


def _positive_binning(value: Any | None) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer() or number <= 0:
        return None
    return int(number)


def parse_datetime(
    value: Any | None,
    path: str,
    *,
    assume_naive_utc: bool = True,
) -> datetime | None:
    text = _clean_text(value)
    if text:
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None and assume_naive_utc:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    match = re.search(
        r"(20\d\d)-(\d\d)-(\d\d)[_T](\d\d)-([0-5]\d)-([0-5]\d)",
        Path(path).name,
    )
    if not match:
        return None
    try:
        parsed = datetime(*(int(part) for part in match.groups()))
        return parsed.replace(tzinfo=timezone.utc) if assume_naive_utc else parsed
    except ValueError:
        return None


_SEXAGESIMAL = re.compile(r"[:\s°'\"hmsd]", re.IGNORECASE)
# How far two readings of one pointing may differ and still be the same field.
_POINTING_AGREEMENT_DEGREES = 2.0
# Header key the FITS/XISF readers add when the comment of a numeric RA names
# hours (``RA = 0.7123 / [h] ...``); degrees need no evidence.
RA_UNIT_EVIDENCE_KEY = "RA:UNIT"
_HOUR_COMMENT = re.compile(r"\bh(?:ou)?rs?\b|\[h\]|\bhour\s*angle\b", re.IGNORECASE)


def ra_comment_names_hours(comment: Any | None) -> bool:
    text = _clean_text(comment)
    return bool(text) and _HOUR_COMMENT.search(text) is not None and "deg" not in text.lower()


def _sexagesimal_degrees(text: str, *, hours: bool) -> float | None:
    try:
        return float(Angle(text, unit=u.hourangle if hours else u.deg).degree)
    except (TypeError, ValueError, u.UnitsError):
        return None


def _circular_difference(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _right_ascension_degrees(header: dict[str, Any]) -> float | None:
    """Right ascension in degrees, by the unit each keyword establishes.

    ``OBJCTRA`` is sexagesimal hours. ``RA`` written as sexagesimal text is
    hours; a number is decimal degrees (N.I.N.A., SGP, INDI, ASIAIR) unless
    its comment names hours or it is fifteen times smaller than the WCS
    value. ``CRVAL1`` is degrees when ``CTYPE1`` is a right-ascension axis.
    A plain number in ``OBJCTRA`` may be hours or degrees: it is read as the
    one that agrees with ``RA``/``CRVAL1``, and without them it is used only
    above 24, where it can only be degrees. No evidence gives no value: a
    solver without a seed searches the sky, one with a wrong seed rejects
    the right solution.
    """

    reference: float | None = None
    crval = _number(_header_lookup(header, ("CRVAL1",)))
    ctype = _clean_text(_header_lookup(header, ("CTYPE1",)))
    if crval is not None and ctype is not None and ctype.upper().startswith("RA"):
        reference = crval

    def resolved(number: float) -> float | None:
        """A number that is hours or degrees, decided by ``reference``."""

        if reference is None:
            return None
        readings = [value for value in (number, number * 15.0) if _circular_difference(value, reference) <= _POINTING_AGREEMENT_DEGREES]
        return readings[0] if len(readings) == 1 else None

    telescope: float | None = None
    text = _clean_text(_header_lookup(header, ("RA",)))
    if text is not None:
        if _SEXAGESIMAL.search(text):
            telescope = _sexagesimal_degrees(text, hours=True)
        else:
            number = _number(text)
            if number is not None:
                if _clean_text(header.get(RA_UNIT_EVIDENCE_KEY)) is not None:
                    telescope = number * 15.0
                else:
                    agreed = resolved(number)
                    telescope = agreed if agreed is not None else number

    text = _clean_text(_header_lookup(header, ("OBJCTRA",)))
    if text is not None:
        if _SEXAGESIMAL.search(text):
            value = _sexagesimal_degrees(text, hours=True)
            if value is not None:
                return value
        else:
            number = _number(text)
            if number is not None:
                if reference is None:
                    reference = telescope
                agreed = resolved(number)
                if agreed is not None:
                    return agreed
                if telescope is None and reference is None and 24.0 < number < 360.0:
                    return number
    return telescope if telescope is not None else reference


def _declination_degrees(header: dict[str, Any]) -> float | None:
    """Declination in degrees: ``OBJCTDEC``, then ``DEC`` (sexagesimal or
    decimal degrees), then ``CRVAL2`` when ``CTYPE2`` is a declination axis."""

    for name in ("OBJCTDEC", "DEC"):
        text = _clean_text(_header_lookup(header, (name,)))
        if text is None:
            continue
        value = _sexagesimal_degrees(text, hours=False) if _SEXAGESIMAL.search(text) else _number(text)
        if value is not None:
            return value
    crval = _number(_header_lookup(header, ("CRVAL2",)))
    ctype = _clean_text(_header_lookup(header, ("CTYPE2",)))
    if crval is not None and ctype is not None and ctype.upper().startswith("DEC"):
        return crval
    return None


def airmass_from_altitude(altitude_degrees: float | None) -> float | None:
    if altitude_degrees is None or not 3.0 < altitude_degrees <= 90.0:
        return None
    altitude = math.radians(altitude_degrees)
    denominator = math.sin(altitude) + 0.50572 * (
        altitude_degrees + 6.07995
    ) ** -1.6364
    if denominator <= 0:
        return None
    return 1.0 / denominator


def _path_components(path: str) -> tuple[str, ...]:
    """The folder names and the file stem of ``path``, outermost first."""

    return (*Path(path).parts[:-1], Path(path).stem)


def _innermost(pattern: re.Pattern[str], path: str) -> str | None:
    """The last match of ``pattern`` in the path: the file name before its
    folders, an inner folder before an outer one."""

    for component in reversed(_path_components(path)):
        matches = list(pattern.finditer(component))
        if matches:
            return matches[-1].group(1)
    return None


def infer_filter_from_path(path: str) -> str:
    # WBPP smart naming (FILTER_Ha, ..._FILTER-L_mono, INSFLNAM-OIII) anywhere
    # in the path, then a bare filter token of the file name (M31_L_300s).
    explicit = _innermost(_SMART_FILTER, path)
    if explicit is not None:
        return normalize_filter(explicit)
    match = re.search(
        r"(?:^|[_\- ])(H(?:A|ALPHA)|O(?:III|3)|S(?:II|2)|L|R|G|B)(?:[_\- .]|$)",
        Path(path).name.upper(),
    )
    return normalize_filter(match.group(1)) if match else "UNKNOWN"


def infer_binning_from_path(path: str) -> int | None:
    value = _innermost(_SMART_BINNING, path)
    return int(value) if value is not None else None


def infer_bayer_pattern_from_path(path: str) -> str | None:
    value = _innermost(_SMART_BAYER, path)
    if value is None:
        return None
    pattern = normalize_cfa_pattern(value)
    return pattern if pattern not in {"UNKNOWN", "NONE"} else None


def _keyword_pattern(name: str) -> re.Pattern[str]:
    pattern = _KEYWORD_PATTERNS.get(name)
    if pattern is None:
        pattern = re.compile(
            rf"(?:^|[^A-Za-z0-9])(?:{re.escape(name)}){_KEYWORD_VALUE}", re.IGNORECASE
        )
        _KEYWORD_PATTERNS[name] = pattern
    return pattern


def grouping_keywords_from_path(
    path: str,
    names: Iterable[str] = DEFAULT_GROUPING_KEYWORDS,
    *,
    root: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """WBPP grouping keywords written in the path, as sorted (NAME, value).

    Like WBPP, every folder and the file name are read and the innermost
    value of a keyword wins; names are matched without regard to case and
    reported in upper case, values are kept as written. ``root`` limits the
    reading to the folders below it (for messages that compare frames).
    """

    parts = _path_components(path)
    if root is not None:
        root_parts = Path(root).parts
        if parts[: len(root_parts)] == root_parts:
            parts = parts[len(root_parts):]
    values: dict[str, str] = {}
    for component in parts:
        for name in names:
            for match in _keyword_pattern(name).finditer(component):
                value = match.group(1).strip()
                if value:
                    values[name.upper()] = value
    return tuple(sorted(values.items()))


def session_keywords_from_path(path: str, *, root: str | None = None) -> str | None:
    """The night or session a path names, e.g. ``DATE=0322``, or ``None``.

    Only the components below ``root`` are read, so a shared ancestor such as
    ``one_night_trip/`` is not mistaken for a keyword.
    """

    values = grouping_keywords_from_path(path, SESSION_KEYWORDS, root=root)
    return ";".join(f"{key}={value}" for key, value in values) or None


def session_keywords_of(paths: Sequence[str]) -> list[str | None]:
    """:func:`session_keywords_from_path` for frames compared with each other:
    only the folders below their common ancestor can tell them apart."""

    try:
        root = os.path.commonpath(list(paths)) if len(paths) > 1 else None
    except ValueError:  # different drives on Windows
        root = None
    return [session_keywords_from_path(path, root=root) for path in paths]


def processing_markers(header: dict[str, Any], path: str) -> list[str]:
    """What PixInsight already did to this frame: CALIBRATED and/or REGISTERED.

    XISF output carries ``PCL:Calibration:*`` and ``PCL:AlignmentMatrix``
    properties (recorded by the reader as ``XISF:PROCESSING``). FITS output
    has no such properties, so WBPP's folder layout (``calibrated/``,
    ``registered/``, …) together with its file suffix is used instead.
    """

    markers = {
        item.strip().upper()
        for item in str(_header_lookup(header, ("XISF:PROCESSING",)) or "").split(",")
        if item.strip()
    }
    folders = {part.upper() for part in Path(path).parts[-4:-1]}
    suffix = _WBPP_PROCESSED_SUFFIX.search(Path(path).stem)
    if suffix and folders & _WBPP_PROCESSED_FOLDERS:
        steps = suffix.group(0).lower().split("_")
        if "c" in steps:
            markers.add("CALIBRATED")
        if "r" in steps:
            markers.add("REGISTERED")
        if not markers:
            markers.add("CALIBRATED")
    return sorted(markers)


def infer_exposure_from_path(path: str) -> float | None:
    # The file name first (M31_L_300s, EXPOSURE-300.00s), then its folders
    # (lights_60s/), as WBPP's smart naming reads both.
    name = Path(path).name
    for pattern in (_BARE_EXPOSURE, _SMART_EXPOSURE):
        match = pattern.search(name)
        if match:
            return _number(match.group(1))
    for component in reversed(Path(path).parts[:-1]):
        for pattern in (_SMART_EXPOSURE, _BARE_EXPOSURE):
            match = pattern.search(component)
            if match:
                return _number(match.group(1))
    return None


@dataclass(frozen=True)
class PathMetadata:
    """What a path says about its frame, in WBPP's conventions.

    Header values take precedence; these fill the ones a header lacks, and
    ``keywords`` group the calibration (see ``DEFAULT_GROUPING_KEYWORDS``).
    """

    role: FrameRole
    filter_name: str
    exposure_seconds: float | None
    binning: int | None
    cfa_pattern: str | None
    keywords: tuple[tuple[str, str], ...]


def path_metadata(
    path: str,
    keyword_names: Iterable[str] = DEFAULT_GROUPING_KEYWORDS,
    *,
    keyword_root: str | None = None,
) -> PathMetadata:
    """``keyword_root`` limits the grouping keywords to the folders below it
    (see :func:`grouping_keyword_root`)."""

    return PathMetadata(
        role=infer_frame_role_from_path(path),
        filter_name=infer_filter_from_path(path),
        exposure_seconds=infer_exposure_from_path(path),
        binning=infer_binning_from_path(path),
        cfa_pattern=infer_bayer_pattern_from_path(path),
        keywords=grouping_keywords_from_path(path, keyword_names, root=keyword_root),
    )


def grouping_keyword_root(paths: Iterable[str]) -> str | None:
    """The folder grouping keywords of a set of frames are read below: the
    common ancestor of their folders.

    A keyword-like folder above every frame (``Night_Sky/``, a test's
    ``..._night_c0/``) would otherwise name a "night" for the frames that no
    inner folder overrides, and keep them from the Lights of ``NIGHT_1/``.
    """

    folders = [os.path.dirname(os.path.abspath(path)) for path in paths]
    if not folders:
        return None
    try:
        return os.path.commonpath(folders)
    except ValueError:  # different drives on Windows
        return None


_DATE_IN_NAME = re.compile(r"(?<!\d)(?:19|20)\d\d[-_.]?[01]\d[-_.]?[0-3]\d(?!\d)")
_TIME_IN_NAME = re.compile(r"(?<!\d)[0-2]\d[-_.:][0-5]\d[-_.:][0-5]\d(?!\d)")
_EXPOSURE_WORD = re.compile(r"[0-9]+(?:\.[0-9]+)?S|BIN[0-9]", re.IGNORECASE)


def header_target(header: dict[str, Any]) -> str | None:
    """The target a header names (``OBJECT``, ``OBJNAME``, ``TARGET``), else ``None``."""

    return _clean_text(_header_lookup(header, ("OBJECT", "OBJNAME", "TARGET")))


def infer_target_from_path(path: str) -> str:
    """The target a file name states before its exposure (``M31_300s_001``).

    A prefix that holds a date or a time (N.I.N.A.'s default file pattern)
    differs from frame to frame and names no target.
    """

    name = Path(path).stem
    match = re.match(r"(.+?)[_\- ]+[0-9]+(?:\.[0-9]+)?s(?:[_\- ]|$)", name, re.I)
    if not match or _DATE_IN_NAME.search(match.group(1)) or _TIME_IN_NAME.search(match.group(1)):
        return "UNKNOWN"
    target = re.sub(r"\s+", " ", match.group(1).replace("_", " ")).strip()
    return target.upper() if target else "UNKNOWN"


def infer_target_from_folders(path: str, *, root: str | None, filter_name: str = "UNKNOWN") -> str | None:
    """The target the folders of a Light without ``OBJECT`` name, else ``None``.

    The nearest folder below ``root`` (the folder that holds every input)
    whose name, without a date, is more than frame-type words (``LIGHT``),
    the Light's filter, exposure or binning words and grouping keywords
    (``NIGHT_1``): ``M31/2026-03-22/LIGHT/L/x.fits`` is ``M31``, and so is
    ``M31_2026-03-22/LIGHT/x.fits``.
    """

    parts = Path(path).parts[:-1]
    if root is not None:
        root_parts = Path(root).parts
        if parts[: len(root_parts)] != root_parts:
            return None
        parts = parts[len(root_parts):]
    for component in reversed(parts):
        if grouping_keywords_from_path(str(Path(component) / "x")):
            continue
        name = _TIME_IN_NAME.sub(" ", _DATE_IN_NAME.sub(" ", component))
        name = re.sub(r"[\s_]+", " ", name).strip(" -.,")
        filters = {filter_name, *_FILTER_ALIASES.values()}
        words = [word for word in re.split(r"[^\w+]+", name.upper()) if word]
        if not any(character.isalnum() for character in name) or normalize_filter(name) in filters:
            continue
        if all(
            word in _ROLE_ALIASES or word.isdigit() or _EXPOSURE_WORD.fullmatch(word) or normalize_filter(word) in filters
            for word in words
        ):
            continue
        return name.upper()
    return None


def normalize_metadata(metadata: FrameMetadata) -> FrameMetadata:
    header = metadata.header
    _normalize_role(metadata)
    metadata.processing = processing_markers(header, metadata.path)
    filter_value = _header_lookup(header, ("FILTER", "INSFLNAM", "FILTERID"))
    metadata.filter_name = normalize_filter(filter_value)
    if metadata.filter_name == "UNKNOWN":
        metadata.filter_name = infer_filter_from_path(metadata.path)
    if metadata.role in {FrameRole.LIGHT, FrameRole.RAW_FLAT, FrameRole.MASTER_FLAT}:
        metadata.filter_name = filter_or_filterless(metadata.filter_name)

    exposure = _number(
        _header_lookup(header, ("EXPTIME", "EXPOSURE", "EXPOSURETIME"))
    )
    metadata.exposure_seconds = (
        exposure if exposure is not None else infer_exposure_from_path(metadata.path)
    )
    metadata.gain = _number(_header_lookup(header, ("GAIN", "EGAIN", "CAMGAIN")))
    metadata.offset = _number(_header_lookup(header, ("OFFSET", "CAMOFFSET")))
    shared_binning = _positive_binning(_header_lookup(header, ("BINNING",)))
    binning_x = _positive_binning(
        _header_lookup(header, ("XBINNING", "CCDBINX"))
    )
    binning_y = _positive_binning(
        _header_lookup(header, ("YBINNING", "CCDBINY"))
    )
    metadata.binning_known = any(
        value is not None for value in (shared_binning, binning_x, binning_y)
    )
    metadata.binning_x = binning_x or shared_binning or binning_y or 1
    metadata.binning_y = binning_y or shared_binning or binning_x or 1
    if not metadata.binning_known:
        path_binning = infer_binning_from_path(metadata.path)
        if path_binning is not None:
            metadata.binning_x = metadata.binning_y = path_binning
            metadata.binning_known = True
    metadata.camera = (
        _clean_text(_header_lookup(header, ("INSTRUME", "CAMERA", "DETECTOR")))
        or "UNKNOWN"
    ).upper()
    metadata.readout_mode = (
        _clean_text(
            _header_lookup(
                header,
                ("READOUTM", "READOUT", "READMODE", "READOUTMODE", "CAMMODE"),
            )
        )
        or "UNKNOWN"
    )
    metadata.cfa_pattern = _normalize_cfa_pattern(
        _header_lookup(
            header,
            (
                "BAYERPAT",
                "BAYERPATN",
                "CFAPAT",
                "CFAPATTERN",
                "PCL:CFASourcePattern",
            ),
        )
    )
    if metadata.cfa_pattern == "UNKNOWN":
        metadata.cfa_pattern = infer_bayer_pattern_from_path(metadata.path) or "UNKNOWN"
    if metadata.cfa_pattern == "UNKNOWN":
        if metadata.channels > 1:
            metadata.cfa_pattern = "NONE"
        elif re.search(
            r"(?:QHY\s*\d+[A-Z0-9-]*M\b|ASI\s*\d+[A-Z0-9-]*MM\b|\bMONO\b)",
            metadata.camera,
            re.IGNORECASE,
        ):
            # Camera model strings with an explicit monochrome suffix are
            # authoritative enough to distinguish mono masters from CFA data.
            metadata.cfa_pattern = "NONE"
    metadata.target = (header_target(header) or infer_target_from_path(metadata.path)).upper()
    metadata.grouping_keywords = dict(grouping_keywords_from_path(metadata.path))

    metadata.ra_degrees = _right_ascension_degrees(header)
    metadata.dec_degrees = _declination_degrees(header)
    metadata.altitude_degrees = _number(
        _header_lookup(header, ("ALTITUDE", "OBJCTALT", "CENTALT"))
    )
    metadata.azimuth_degrees = _number(
        _header_lookup(header, ("AZIMUTH", "OBJCTAZ", "CENTAZ"))
    )
    metadata.airmass = _number(_header_lookup(header, ("AIRMASS",)))
    if metadata.airmass is None:
        metadata.airmass = airmass_from_altitude(metadata.altitude_degrees)
    # N.I.N.A. writes DATE-OBS in UTC and DATE-LOC in the observatory's local
    # wall clock.  Observing-night grouping needs the latter: using UTC with a
    # local-noon boundary can split one Chinese session at 20:00 local time.
    # A timezone-less DATE-LOC deliberately remains naive, which the nightly
    # model treats as an already-local civil timestamp.  DATE-OBS remains the
    # standards-compliant fallback when no local timestamp can be recovered.
    local_observed = _header_lookup(
        header, ("DATE-LOC", "DATE_LOCAL", "DATE-LOCAL")
    )
    metadata.observed_at = (
        parse_datetime(
            local_observed,
            metadata.path,
            assume_naive_utc=False,
        )
        if local_observed is not None
        else None
    )
    if metadata.observed_at is None:
        metadata.observed_at = parse_datetime(
            _header_lookup(header, ("DATE-OBS", "DATE_OBS", "DATE")),
            metadata.path,
        )
    return metadata


def field_token(metadata: FrameMetadata) -> str:
    if metadata.target != "UNKNOWN":
        return "OBJECT:" + re.sub(r"\s+", " ", metadata.target.strip())
    if metadata.ra_degrees is not None and metadata.dec_degrees is not None:
        return f"SKY:{metadata.ra_degrees:.2f}:{metadata.dec_degrees:.2f}"
    return "AUTO"
