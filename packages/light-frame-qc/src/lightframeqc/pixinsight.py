"""Strict import of user-produced PixInsight SubframeSelector v3 JSON.

This module is an interoperability boundary, not a PixInsight implementation.
It only reads an existing JSON result, validates the observed v3 table schema,
and exposes a conservative allow-list of auxiliary measurements.  It does not
load, execute, copy, or derive algorithms from PixInsight source or binaries.

Paths are treated as opaque source identities.  They are never opened by this
module, and a measurement row must bind to its ``requestPaths`` entry exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping, NoReturn, TypeAlias


PathLike: TypeAlias = str | os.PathLike[str]

SUBFRAME_SELECTOR_V3_COLUMNS: tuple[str, ...] = (
    "index",
    "enabled",
    "locked",
    "filePath",
    "weight",
    "FWHM",
    "eccentricity",
    "PSFSignalWeight",
    "unused01",
    "SNRWeight",
    "median",
    "medianMeanDev",
    "noise",
    "noiseRatio",
    "stars",
    "starResidual",
    "FWHMMeanDev",
    "eccentricityMeanDev",
    "starResidualMeanDev",
    "azimuth",
    "altitude",
    "PSFFlux",
    "PSFFluxPower",
    "PSFTotalMeanFlux",
    "PSFTotalMeanPowerFlux",
    "PSFCount",
    "MStar",
    "NStar",
    "PSFSNR",
    "PSFScale",
    "PSFScaleSNR",
)

_COLUMN_COUNT = len(SUBFRAME_SELECTOR_V3_COLUMNS)
_UINT32_MAX = 2**32 - 1
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_MEASUREMENT_ROWS = 1_000_000


class PixInsightImportError(ValueError):
    """Raised when a purported SubframeSelector result is not trustworthy."""


@dataclass(frozen=True, slots=True)
class PixInsightMeasurement:
    """Safe auxiliary fields from one validated SubframeSelector v3 row.

    The process ``weight`` and undocumented ``unused01`` cells are validated
    but deliberately not exposed: neither is a stable cloud/occlusion score.
    ``path`` is an opaque identity and is not opened or resolved.
    """

    index: int
    path: str
    enabled: bool
    locked: bool
    fwhm: float
    eccentricity: float
    psf_signal_weight: float
    snr_weight: float
    median: float
    median_mean_dev: float
    noise: float
    noise_ratio: float
    stars: int
    star_residual: float
    fwhm_mean_dev: float
    eccentricity_mean_dev: float
    star_residual_mean_dev: float
    azimuth: float
    altitude: float
    psf_flux: float
    psf_flux_power: float
    psf_total_mean_flux: float
    psf_total_mean_power_flux: float
    psf_count: int
    m_star: float
    n_star: float
    psf_snr: float
    psf_scale: float
    psf_scale_snr: float

    def as_features(self) -> dict[str, float | int | bool]:
        """Return only auxiliary controls and measurements, never the path."""

        excluded = {"index", "path"}
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in excluded
        }


@dataclass(frozen=True, slots=True)
class PixInsightMeasurementSet:
    """A complete, request-order-preserving SubframeSelector v3 import."""

    schema_version: int
    process_version: int
    request_paths: tuple[str, ...]
    measurements: tuple[PixInsightMeasurement, ...]

    def __len__(self) -> int:
        return len(self.measurements)

    def by_path(self, path: str) -> PixInsightMeasurement:
        """Return the uniquely bound row for ``path`` or raise ``KeyError``."""

        try:
            index = self.request_paths.index(path)
        except ValueError as error:
            raise KeyError(path) from error
        return self.measurements[index]


def parse_subframe_selector_v3(
    payload: Mapping[str, Any],
) -> PixInsightMeasurementSet:
    """Validate and import a decoded SubframeSelector v3 JSON object.

    Unknown top-level metadata is ignored, but the versioned identity, row
    count, complete 31-column rows, zero-based row order, and exact path binding
    are mandatory.  Every numeric cell must be finite and have its declared
    integer/real type.
    """

    if not isinstance(payload, Mapping):
        _fail("top-level JSON value must be an object")

    schema_version = _required_uint(payload, "schemaVersion")
    if schema_version != 1:
        _fail("unsupported JSON schemaVersion")
    if _required_string(payload, "processId") != "SubframeSelector":
        _fail("processId is not SubframeSelector")
    process_version = _required_uint(payload, "processVersion")
    if process_version != 3:
        _fail("unsupported SubframeSelector processVersion")
    if _required_string(payload, "routine") != "MeasureSubframes":
        _fail("routine is not MeasureSubframes")

    if "ok" in payload:
        ok = payload["ok"]
        if not isinstance(ok, bool) or not ok:
            _fail("producer did not report a successful measurement run")
    if "fileCache" in payload and not isinstance(payload["fileCache"], bool):
        _fail("fileCache must be Boolean when present")
    if "wallSeconds" in payload:
        wall_seconds = _real(payload["wallSeconds"], "wallSeconds")
        if wall_seconds < 0.0:
            _fail("wallSeconds must not be negative")

    request_paths_value = payload.get("requestPaths")
    if not isinstance(request_paths_value, list):
        _fail("requestPaths must be an array")
    if not request_paths_value:
        _fail("requestPaths must not be empty")
    if len(request_paths_value) > _MAX_MEASUREMENT_ROWS:
        _fail("requestPaths exceeds the safety row limit")

    request_paths: list[str] = []
    seen_paths: set[str] = set()
    for index, value in enumerate(request_paths_value):
        path = _source_path(value, f"requestPaths[{index}]")
        if path in seen_paths:
            _fail("requestPaths contains a duplicate path")
        seen_paths.add(path)
        request_paths.append(path)

    row_count = _required_uint(payload, "measurementRowCount")
    measurements_value = payload.get("measurements")
    if not isinstance(measurements_value, list):
        _fail("measurements must be an array")
    if row_count != len(measurements_value):
        _fail("measurementRowCount does not match measurements")
    if row_count != len(request_paths):
        _fail("measurements do not cover requestPaths exactly")

    measurements: list[PixInsightMeasurement] = []
    for expected_index, row in enumerate(measurements_value):
        measurements.append(
            _parse_measurement_row(row, expected_index, request_paths[expected_index])
        )

    return PixInsightMeasurementSet(
        schema_version=schema_version,
        process_version=process_version,
        request_paths=tuple(request_paths),
        measurements=tuple(measurements),
    )


def loads_subframe_selector_v3_json(
    document: str | bytes | bytearray,
) -> PixInsightMeasurementSet:
    """Parse a UTF-8 JSON document and import its validated v3 measurements."""

    if isinstance(document, str):
        try:
            encoded_size = len(document.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise PixInsightImportError("JSON document is not valid UTF-8") from error
        text = document
    elif isinstance(document, (bytes, bytearray)):
        raw = bytes(document)
        encoded_size = len(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PixInsightImportError("JSON document is not valid UTF-8") from error
    else:
        raise TypeError("document must be str, bytes, or bytearray")
    if encoded_size == 0:
        _fail("JSON document is empty")
    if encoded_size > _MAX_JSON_BYTES:
        _fail("JSON document exceeds the safety size limit")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_json_number,
        )
    except PixInsightImportError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise PixInsightImportError("invalid JSON document") from error
    return parse_subframe_selector_v3(payload)


def load_subframe_selector_v3_json(path: PathLike) -> PixInsightMeasurementSet:
    """Read one JSON file without modifying it and import its measurements."""

    source = Path(path)
    try:
        with source.open("rb") as stream:
            document = stream.read(_MAX_JSON_BYTES + 1)
    except OSError as error:
        raise PixInsightImportError("unable to read measurement JSON") from error
    if len(document) > _MAX_JSON_BYTES:
        _fail("JSON document exceeds the safety size limit")
    return loads_subframe_selector_v3_json(document)


def _parse_measurement_row(
    value: Any,
    expected_index: int,
    expected_path: str,
) -> PixInsightMeasurement:
    label = f"measurements[{expected_index}]"
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    if len(value) != _COLUMN_COUNT:
        _fail(f"{label} must contain exactly {_COLUMN_COUNT} columns")

    index = _uint32(value[0], f"{label}.index")
    if index != expected_index:
        _fail(f"{label} has a noncanonical row index")
    enabled = _boolean(value[1], f"{label}.enabled")
    locked = _boolean(value[2], f"{label}.locked")
    path = _source_path(value[3], f"{label}.filePath")
    if path != expected_path:
        _fail(f"{label} filePath does not match requestPaths")

    # Validate every cell, including fields intentionally not exposed below.
    doubles = {
        column: _real(value[column], f"{label}.{SUBFRAME_SELECTOR_V3_COLUMNS[column]}")
        for column in range(_COLUMN_COUNT)
        if column not in {0, 1, 2, 3, 14, 25}
    }
    stars = _uint32(value[14], f"{label}.stars")
    if stars == 0:
        _fail(f"{label}.stars must be positive for a usable measurement row")
    psf_count = _uint32(value[25], f"{label}.PSFCount")

    return PixInsightMeasurement(
        index=index,
        path=path,
        enabled=enabled,
        locked=locked,
        fwhm=doubles[5],
        eccentricity=doubles[6],
        psf_signal_weight=doubles[7],
        snr_weight=doubles[9],
        median=doubles[10],
        median_mean_dev=doubles[11],
        noise=doubles[12],
        noise_ratio=doubles[13],
        stars=stars,
        star_residual=doubles[15],
        fwhm_mean_dev=doubles[16],
        eccentricity_mean_dev=doubles[17],
        star_residual_mean_dev=doubles[18],
        azimuth=doubles[19],
        altitude=doubles[20],
        psf_flux=doubles[21],
        psf_flux_power=doubles[22],
        psf_total_mean_flux=doubles[23],
        psf_total_mean_power_flux=doubles[24],
        psf_count=psf_count,
        m_star=doubles[26],
        n_star=doubles[27],
        psf_snr=doubles[28],
        psf_scale=doubles[29],
        psf_scale_snr=doubles[30],
    )


def _required_value(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        _fail(f"missing required field {key}")
    return payload[key]


def _required_uint(payload: Mapping[str, Any], key: str) -> int:
    return _uint32(_required_value(payload, key), key)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = _required_value(payload, key)
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail(f"{key} must be a nonempty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be Boolean")
    return value


def _uint32(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an unsigned integer")
    if value < 0 or value > _UINT32_MAX:
        _fail(f"{label} is outside the UInt32 range")
    return value


def _real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError:
        _fail(f"{label} is outside the finite numeric range")
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _source_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail(f"{label} must be a nonempty path string")

    # Accept results produced on POSIX or Windows, but never reinterpret or
    # resolve them on the importing host.  Parent traversal is rejected before
    # the path can be handed to another component.
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    is_absolute = posix.is_absolute() or windows.is_absolute()
    raw_parts = re.split(r"[\\/]", value)
    if not is_absolute or any(part in {".", ".."} for part in raw_parts):
        _fail(f"{label} must be an absolute path without parent traversal")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_nonstandard_json_number(value: str) -> Any:
    _fail(f"nonstandard JSON number {value} is forbidden")


def _fail(message: str) -> NoReturn:
    raise PixInsightImportError(message)


__all__ = [
    "PixInsightImportError",
    "PixInsightMeasurement",
    "PixInsightMeasurementSet",
    "SUBFRAME_SELECTOR_V3_COLUMNS",
    "load_subframe_selector_v3_json",
    "loads_subframe_selector_v3_json",
    "parse_subframe_selector_v3",
]
