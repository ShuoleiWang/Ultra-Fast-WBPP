from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from pathlib import Path
from typing import Any, Iterable

from astropy.coordinates import Angle
import astropy.units as u

from .models import FrameMetadata, FrameRole


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
    "H-ALPHA": "HA",
    "HΑ": "HA",
    "O3": "OIII",
    "OIII": "OIII",
    "OXYGENIII": "OIII",
    "S2": "SII",
    "SII": "SII",
}


_ROLE_ALIASES = {
    "LIGHT": FrameRole.LIGHT,
    "LIGHTFRAME": FrameRole.LIGHT,
    "LIGHTFRAMES": FrameRole.LIGHT,
    "FLAT": FrameRole.RAW_FLAT,
    "FLATFRAME": FrameRole.RAW_FLAT,
    "FLATFRAMES": FrameRole.RAW_FLAT,
    "RAWFLAT": FrameRole.RAW_FLAT,
    "MASTERFLAT": FrameRole.MASTER_FLAT,
    "DARK": FrameRole.DARK,
    "DARKFRAME": FrameRole.DARK,
    "DARKFRAMES": FrameRole.DARK,
    "MASTERDARK": FrameRole.MASTER_DARK,
    "BIAS": FrameRole.BIAS,
    "BIASFRAME": FrameRole.BIAS,
    "BIASFRAMES": FrameRole.BIAS,
    "ZERO": FrameRole.BIAS,
    "ZEROFRAME": FrameRole.BIAS,
    "MASTERBIAS": FrameRole.MASTER_BIAS,
    "MASTERZERO": FrameRole.MASTER_BIAS,
    "MASTERLIGHT": FrameRole.MASTER_LIGHT,
}


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
    text = _clean_text(value)
    if text is None:
        return "UNKNOWN"
    compact = re.sub(r"[ _]+", "", text.upper())
    return _FILTER_ALIASES.get(compact, text.upper())


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

    metadata.role = role
    metadata.role_evidence = evidence
    metadata.role_conflicts = conflicts


def _normalize_cfa_pattern(value: Any | None) -> str:
    text = _clean_text(value)
    if text is None:
        return "UNKNOWN"
    compact = re.sub(r"[^A-Z0-9]+", "", text.upper())
    if compact in {"", "UNKNOWN", "AUTO"}:
        return "UNKNOWN"
    if compact in {"NONE", "NOCFA", "MONO", "FALSE", "0"}:
        return "NONE"
    return compact


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


def _angle_degrees(value: Any | None, hour_angle: bool) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        if ":" in text or " " in text:
            return float(Angle(text, unit=u.hourangle if hour_angle else u.deg).degree)
        numeric = float(text)
        if hour_angle and abs(numeric) <= 24:
            return numeric * 15.0
        return numeric
    except (TypeError, ValueError, u.UnitsError):
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


def infer_filter_from_path(path: str) -> str:
    name = Path(path).name.upper()
    patterns = (
        r"(?:^|[_\- ])(H(?:A|ALPHA)|O(?:III|3)|S(?:II|2)|L|R|G|B)(?:[_\- .]|$)",
        r"FILTER[_\- ]([A-Z0-9+\-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return normalize_filter(match.group(1))
    return "UNKNOWN"


def infer_exposure_from_path(path: str) -> float | None:
    name = Path(path).name
    for pattern in (
        r"(?:^|[_\- ])([0-9]+(?:\.[0-9]+)?)s(?:[_\- .]|$)",
        r"(?:EXPTIME|EXPOSURE)[_\- ]([0-9]+(?:\.[0-9]+)?)",
    ):
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return _number(match.group(1))
    return None


def infer_target_from_path(path: str) -> str:
    name = Path(path).stem
    match = re.match(r"(.+?)[_\- ]+[0-9]+(?:\.[0-9]+)?s(?:[_\- ]|$)", name, re.I)
    if not match:
        return "UNKNOWN"
    target = re.sub(r"\s+", " ", match.group(1).replace("_", " ")).strip()
    return target.upper() if target else "UNKNOWN"


def normalize_metadata(metadata: FrameMetadata) -> FrameMetadata:
    header = metadata.header
    _normalize_role(metadata)
    filter_value = _header_lookup(header, ("FILTER", "INSFLNAM", "FILTERID"))
    metadata.filter_name = normalize_filter(filter_value)
    if metadata.filter_name == "UNKNOWN":
        metadata.filter_name = infer_filter_from_path(metadata.path)

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
    metadata.target = (
        _clean_text(_header_lookup(header, ("OBJECT", "OBJNAME", "TARGET")))
        or infer_target_from_path(metadata.path)
    ).upper()

    metadata.ra_degrees = _angle_degrees(
        _header_lookup(header, ("OBJCTRA", "RA", "CRVAL1")), hour_angle=True
    )
    metadata.dec_degrees = _angle_degrees(
        _header_lookup(header, ("OBJCTDEC", "DEC", "CRVAL2")), hour_angle=False
    )
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
