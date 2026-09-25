"""Engine-side catalog-correspondence evidence for external plate solutions.

ASTAP's documented ``.ini``/``.wcs`` contract confirms that a solve happened,
but it does not say which catalog stars were matched or with what residuals.
The E2E gate (``validate_astrometric_quality``) and the desktop's publication
check only accept a solution whose match count, residual RMS, parity and
index identity were recomputed from actual image/catalog correspondences and
whose catalog bytes are app-managed.  ``solve-field`` provides that through
its ``.corr``/``.match`` tables; this module provides the same evidence for
any solver by recomputing it inside the engine:

1. the stars of the densest managed Astrometry.net index that suits the field
   are decoded from the index's star kd-tree (``kdtree_data_stars``);
2. stars are detected on the solved image with the registration detector;
3. the catalog stars are projected through the solver's WCS and matched
   one-to-one (mutual nearest neighbour) against the detections;
4. residuals are measured exactly as the astrometry.net adapter measures its
   ``.corr`` rows, and the correspondence table is written as a canonical
   JSON artifact whose digest becomes ``correspondenceSha256``.

Nothing here weakens the gate: a WCS that does not put catalog stars on the
detected stars yields no mutual matches and the solve fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from ..integrity import canonical_json
from .base import AstrometricQuality, SolverIndexArtifact, wcs_parity
from .catalogs import (
    CatalogError,
    installed_set_identity_for_solver_indexes,
    verify_installed_set_snapshot,
)


VERIFICATION_METHOD = "engine-catalog-correspondence-v1"
CORRESPONDENCE_ARTIFACT_NAME = "correspondence.json"
MATCH_ARTIFACT_NAME = "match.json"

_INDEX_FILE_NAME = re.compile(r"^index-(\d+)\.fits$")
_ENDIAN_MARKERS = {"04:03:02:01": "<", "01:02:03:04": ">"}
_DECODE_CHUNK_ROWS = 262144
_MATCH_CHUNK_ROWS = 256
_STAR_TABLE_COLUMNS = ("kdtree_header_stars", "kdtree_range_stars", "kdtree_data_stars")


class CatalogCorrespondenceError(RuntimeError):
    """Stable failure code for the solver adapter's receipt."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CorrespondenceParameters:
    """Detection and matching policy; every value is recorded in the receipt."""

    detection_sigma: float = 4.5
    minimum_source_pixels: int = 5
    maximum_stars: int = 2000
    background_box: int = 64
    # Catalog stars considered for matching after projection into the image;
    # the brightest are kept so a dense index cannot flood the matcher.
    maximum_catalog_stars: int = 8000
    minimum_match_radius_pixels: float = 3.0
    match_radius_fwhm_factor: float = 2.0
    search_radius_diagonal_fraction: float = 0.6
    minimum_matches: int = 3

    def validate(self) -> None:
        if (
            isinstance(self.detection_sigma, bool)
            or not math.isfinite(self.detection_sigma)
            or not 1.0 <= self.detection_sigma <= 20.0
        ):
            raise ValueError("detection_sigma must be in [1, 20]")
        for name, value, floor in (
            ("minimum_source_pixels", self.minimum_source_pixels, 1),
            ("maximum_stars", self.maximum_stars, 20),
            ("maximum_catalog_stars", self.maximum_catalog_stars, 20),
            ("minimum_matches", self.minimum_matches, 3),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"{name} must be an integer of at least {floor}")
        if isinstance(self.background_box, bool) or not isinstance(self.background_box, int) or self.background_box < 16:
            raise ValueError("background_box must be an integer of at least 16")
        for name, value in (
            ("minimum_match_radius_pixels", self.minimum_match_radius_pixels),
            ("match_radius_fwhm_factor", self.match_radius_fwhm_factor),
            ("search_radius_diagonal_fraction", self.search_radius_diagonal_fraction),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def serializable(self) -> dict[str, Any]:
        return {
            "detectionSigma": self.detection_sigma,
            "minimumSourcePixels": self.minimum_source_pixels,
            "maximumStars": self.maximum_stars,
            "backgroundBox": self.background_box,
            "maximumCatalogStars": self.maximum_catalog_stars,
            "minimumMatchRadiusPixels": self.minimum_match_radius_pixels,
            "matchRadiusFwhmFactor": self.match_radius_fwhm_factor,
            "searchRadiusDiagonalFraction": self.search_radius_diagonal_fraction,
            "minimumMatches": self.minimum_matches,
        }


@dataclass(frozen=True, slots=True)
class IndexSummary:
    """Primary-header facts of one Astrometry.net index file."""

    relative_name: str
    index_id: str
    healpix: int
    hpnside: int
    scale_lower_radians: float
    scale_upper_radians: float
    star_count: int | None

    @property
    def identity(self) -> str:
        # The same tuple astrometry.net writes to its .match table (INDEXID,
        # HEALPIX, HPNSIDE), so managed-catalog binding treats both routes alike.
        return f"astrometry.net:index:{self.index_id}:healpix:{self.healpix}:hpnside:{self.hpnside}"

    def serializable(self) -> dict[str, Any]:
        return {
            "relativeName": self.relative_name,
            "indexId": self.index_id,
            "identity": self.identity,
            "healpix": self.healpix,
            "hpnside": self.hpnside,
            "scaleLowerRadians": self.scale_lower_radians,
            "scaleUpperRadians": self.scale_upper_radians,
            "quadScaleArcminutes": [
                math.degrees(self.scale_lower_radians) * 60.0,
                math.degrees(self.scale_upper_radians) * 60.0,
            ],
            "starCount": self.star_count,
        }


@dataclass(frozen=True, slots=True)
class IndexStars:
    """Catalog stars of one index within a search cone."""

    summary: IndexSummary
    rows: np.ndarray
    ra_degrees: np.ndarray
    dec_degrees: np.ndarray
    magnitudes: np.ndarray | None
    table_rows: int
    endian: str

    @property
    def count(self) -> int:
        return int(self.rows.shape[0])


@dataclass(frozen=True, slots=True)
class FieldGeometry:
    width: int
    height: int
    centre_ra_degrees: float
    centre_dec_degrees: float
    width_degrees: float
    height_degrees: float
    diagonal_degrees: float

    def serializable(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "centreRaDegrees": self.centre_ra_degrees,
            "centreDecDegrees": self.centre_dec_degrees,
            "widthDegrees": self.width_degrees,
            "heightDegrees": self.height_degrees,
            "diagonalDegrees": self.diagonal_degrees,
        }


@dataclass(frozen=True, slots=True)
class CorrespondenceVerification:
    """Recomputed quality plus the artifacts and diagnostics behind it."""

    quality: AstrometricQuality
    correspondence_path: Path
    match_path: Path
    correspondence_diagnostics: dict[str, Any]
    match_diagnostics: dict[str, Any]


def _unit_vectors(ra_degrees: np.ndarray, dec_degrees: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_degrees, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_degrees, dtype=np.float64))
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def _angular_separation_arcsec(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Chord-based separation, the same measure the astrometry.net adapter
    # applies to its .corr rows, so both routes report comparable sky RMS.
    left_vectors = _unit_vectors(left[:, 0], left[:, 1])
    right_vectors = _unit_vectors(right[:, 0], right[:, 1])
    chord = np.linalg.norm(left_vectors - right_vectors, axis=1)
    return np.rad2deg(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))) * 3600.0


def _separation_degrees(left: Sequence[float], right: Sequence[float]) -> float:
    value = _angular_separation_arcsec(
        np.asarray([left], dtype=np.float64), np.asarray([right], dtype=np.float64)
    )
    return float(value[0]) / 3600.0


def field_geometry(wcs_header: Mapping[str, Any] | fits.Header, image_shape: tuple[int, int]) -> FieldGeometry:
    """Sky centre and extents of the image under the solver's WCS."""

    height, width = image_shape
    try:
        celestial = WCS(wcs_header, relax=False).celestial
        # FITS one-based pixel centres: the image spans 0.5 .. N + 0.5.
        mid_x = (width + 1) / 2.0
        mid_y = (height + 1) / 2.0
        points = np.asarray(
            [
                [mid_x, mid_y],
                [0.5, 0.5],
                [width + 0.5, height + 0.5],
                [0.5, height + 0.5],
                [width + 0.5, 0.5],
                [0.5, mid_y],
                [width + 0.5, mid_y],
                [mid_x, 0.5],
                [mid_x, height + 0.5],
            ],
            dtype=np.float64,
        )
        world = celestial.all_pix2world(points, 1)
    except Exception as error:
        raise CatalogCorrespondenceError("CORRESPONDENCE_WCS_FAILED", str(error)) from error
    if not np.all(np.isfinite(world)):
        raise CatalogCorrespondenceError("CORRESPONDENCE_WCS_FAILED", "field corners are non-finite on the sky")
    diagonal = max(_separation_degrees(world[1], world[2]), _separation_degrees(world[3], world[4]))
    return FieldGeometry(
        width=int(width),
        height=int(height),
        centre_ra_degrees=float(world[0, 0] % 360.0),
        centre_dec_degrees=float(world[0, 1]),
        width_degrees=_separation_degrees(world[5], world[6]),
        height_degrees=_separation_degrees(world[7], world[8]),
        diagonal_degrees=diagonal,
    )


def _header_int(header: fits.Header, key: str, *, context: str) -> int:
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{context}: {key} is missing or not an integer")
    return int(value)


def _header_float(header: fits.Header, key: str, *, context: str) -> float:
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
        raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{context}: {key} is missing or not finite")
    return float(value)


def read_index_summary(path: Path) -> IndexSummary:
    """Read the identity and quad-scale range from an index's primary header."""

    name = path.name
    match = _INDEX_FILE_NAME.fullmatch(name)
    if match is None:
        raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"not a managed index file name: {name}")
    try:
        header = fits.getheader(path, ext=0, memmap=False)
    except Exception as error:
        raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: cannot read the primary header: {error}") from error
    index_id = _header_int(header, "INDEXID", context=name)
    if str(index_id) != match.group(1):
        raise CatalogCorrespondenceError(
            "CATALOG_INDEX_INVALID", f"{name}: INDEXID {index_id} disagrees with the managed file name"
        )
    lower = _header_float(header, "SCALE_L", context=name)
    upper = _header_float(header, "SCALE_U", context=name)
    if not 0.0 < lower < upper:
        raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: SCALE_L/SCALE_U are not an increasing positive range")
    star_count = header.get("NSTARS")
    return IndexSummary(
        relative_name=name,
        index_id=str(index_id),
        healpix=_header_int(header, "HEALPIX", context=name),
        hpnside=_header_int(header, "HPNSIDE", context=name),
        scale_lower_radians=lower,
        scale_upper_radians=upper,
        star_count=int(star_count) if isinstance(star_count, (int, np.integer)) and not isinstance(star_count, bool) else None,
    )


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mtimeNs": int(value.st_mtime_ns),
        "sizeBytes": int(value.st_size),
    }


def _require_identity(actual: os.stat_result, expected: Mapping[str, Any] | None, *, name: str) -> None:
    if expected is None:
        return
    current = _stat_identity(actual)
    if any(current[key] != expected.get(key) for key in current):
        raise CatalogCorrespondenceError(
            "CATALOG_FILE_CHANGED", f"{name} is not the file described by the pre-solve catalog snapshot"
        )


def _raw_table_bytes(handle: Any, hdu: fits.BinTableHDU, *, name: str) -> bytes:
    # libkd stores the tree arrays as opaque fixed-width byte columns; read
    # the exact NAXIS1 x NAXIS2 bytes rather than letting the FITS table
    # layer decode them as text.
    info = hdu.fileinfo()
    if info is None or info.get("datLoc") is None:
        raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: table has no data location")
    size = int(hdu.header["NAXIS1"]) * int(hdu.header["NAXIS2"])
    handle.seek(int(info["datLoc"]))
    buffer = handle.read(size)
    if len(buffer) != size:
        raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: truncated kd-tree table")
    return buffer


def read_index_stars(
    path: Path,
    *,
    centre_ra_degrees: float,
    centre_dec_degrees: float,
    radius_degrees: float,
    expected_stat: Mapping[str, Any] | None = None,
) -> IndexStars:
    """Decode the stars of one index that lie within ``radius_degrees`` of a centre.

    The star kd-tree keeps unit vectors as ``u32`` triples scaled into the
    ``[minval, maxval]`` box recorded in ``kdtree_range_stars``; decoding is
    ``xyz = u32 / scale + minval``.  The file's ``ENDIAN`` marker decides the
    byte order of both tables.  ``expected_stat`` binds the bytes read here to
    the file identity captured by the pre-solve installed-set snapshot.
    """

    name = path.name
    try:
        before = path.lstat()
    except OSError as error:
        raise CatalogCorrespondenceError("CATALOG_ARTIFACT_MISSING", f"{name}: {error}") from error
    # The identity is checked before any byte is read and again after the
    # last one, so summary and stars are known to come from the same file.
    _require_identity(before, expected_stat, name=name)
    summary = read_index_summary(path)
    centre = _unit_vectors(np.asarray([centre_ra_degrees]), np.asarray([centre_dec_degrees]))[0]
    threshold = math.cos(math.radians(min(max(radius_degrees, 0.0), 180.0)))
    selected_rows: list[np.ndarray] = []
    selected_xyz: list[np.ndarray] = []
    try:
        with path.open("rb") as handle, fits.open(path, mode="readonly", memmap=False, lazy_load_hdus=True) as hdul:
            _require_identity(os.fstat(handle.fileno()), expected_stat, name=name)
            tables: dict[str, fits.BinTableHDU] = {}
            tag_along: fits.BinTableHDU | None = None
            for hdu in hdul[1:]:
                if not isinstance(hdu, fits.BinTableHDU):
                    continue
                names = tuple(str(item) for item in hdu.columns.names)
                if len(names) == 1 and names[0] in _STAR_TABLE_COLUMNS:
                    tables.setdefault(names[0], hdu)
                elif "MAG" in {item.upper() for item in names} and tag_along is None:
                    tag_along = hdu
            missing = [item for item in _STAR_TABLE_COLUMNS if item not in tables]
            if missing:
                raise CatalogCorrespondenceError(
                    "CATALOG_INDEX_INVALID", f"{name}: star kd-tree tables are missing: {', '.join(missing)}"
                )
            tree_header = tables["kdtree_header_stars"].header
            endian = _ENDIAN_MARKERS.get(str(tree_header.get("ENDIAN", "")).strip())
            if endian is None:
                raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: unknown kd-tree ENDIAN marker")
            data_type = str(tree_header.get("KDT_DATA", "")).strip().lower()
            dimensions = _header_int(tree_header, "KDT_NDIM", context=name)
            declared_rows = _header_int(tree_header, "KDT_NDAT", context=name)
            data_hdu = tables["kdtree_data_stars"]
            row_bytes = int(data_hdu.header["NAXIS1"])
            table_rows = int(data_hdu.header["NAXIS2"])
            if dimensions != 3 or table_rows != declared_rows or table_rows < 1:
                raise CatalogCorrespondenceError(
                    "CATALOG_INDEX_INVALID", f"{name}: star kd-tree geometry is inconsistent"
                )
            if data_type == "u32":
                if row_bytes != 12:
                    raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: u32 star rows must be 12 bytes")
                range_hdu = tables["kdtree_range_stars"]
                if int(range_hdu.header["NAXIS1"]) * int(range_hdu.header["NAXIS2"]) != 56:
                    raise CatalogCorrespondenceError(
                        "CATALOG_INDEX_INVALID", f"{name}: kdtree_range_stars must hold 7 doubles"
                    )
                scaling = np.frombuffer(_raw_table_bytes(handle, range_hdu, name=name), dtype=f"{endian}f8")
                minimum = scaling[:3].astype(np.float64)
                scale = float(scaling[6])
                if not np.all(np.isfinite(scaling)) or scale <= 0:
                    raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: kd-tree range is not finite")
                raw = np.frombuffer(_raw_table_bytes(handle, data_hdu, name=name), dtype=f"{endian}u4").reshape(table_rows, 3)

                def decode(block: np.ndarray) -> np.ndarray:
                    return block.astype(np.float64) / scale + minimum

            elif data_type in {"double", "d", "f64"}:
                if row_bytes != 24:
                    raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: double star rows must be 24 bytes")
                raw = np.frombuffer(_raw_table_bytes(handle, data_hdu, name=name), dtype=f"{endian}f8").reshape(table_rows, 3)

                def decode(block: np.ndarray) -> np.ndarray:
                    return block.astype(np.float64)

            else:
                raise CatalogCorrespondenceError(
                    "CATALOG_INDEX_UNSUPPORTED", f"{name}: unsupported star kd-tree data type {data_type!r}"
                )
            for start in range(0, table_rows, _DECODE_CHUNK_ROWS):
                xyz = decode(raw[start : start + _DECODE_CHUNK_ROWS])
                norms = np.linalg.norm(xyz, axis=1)
                valid = np.isfinite(norms) & (norms > 0)
                dots = np.where(valid, (xyz @ centre) / np.where(valid, norms, 1.0), -2.0)
                chosen = np.flatnonzero(dots >= threshold)
                if chosen.size:
                    selected_rows.append(chosen + start)
                    selected_xyz.append(xyz[chosen] / norms[chosen, None])
            rows = np.concatenate(selected_rows) if selected_rows else np.empty(0, dtype=np.int64)
            xyz_selected = np.concatenate(selected_xyz) if selected_xyz else np.empty((0, 3), dtype=np.float64)
            magnitudes: np.ndarray | None = None
            if tag_along is not None and int(tag_along.header["NAXIS2"]) == table_rows and rows.size:
                try:
                    column = next(item for item in tag_along.columns.names if str(item).upper() == "MAG")
                    magnitudes = np.asarray(tag_along.data[column], dtype=np.float64)[rows]
                except Exception:
                    # Magnitudes are diagnostic only; positions are the evidence.
                    magnitudes = None
            _require_identity(os.fstat(handle.fileno()), expected_stat, name=name)
        after = path.lstat()
    except CatalogCorrespondenceError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise CatalogCorrespondenceError("CATALOG_INDEX_INVALID", f"{name}: {type(error).__name__}: {error}") from error
    if _stat_identity(before) != _stat_identity(after):
        raise CatalogCorrespondenceError("CATALOG_FILE_CHANGED", f"{name} changed while its stars were read")
    ra = np.rad2deg(np.arctan2(xyz_selected[:, 1], xyz_selected[:, 0])) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(xyz_selected[:, 2], -1.0, 1.0)))
    return IndexStars(
        summary=summary,
        rows=rows.astype(np.int64),
        ra_degrees=ra,
        dec_degrees=dec,
        magnitudes=magnitudes,
        table_rows=table_rows,
        endian="little" if endian == "<" else "big",
    )


def rank_indexes(summaries: Sequence[IndexSummary], field: FieldGeometry) -> tuple[IndexSummary, ...]:
    """Order indexes densest first, preferring quads that fit inside the field.

    Astrometry.net builds each index so its stars are uniform at its quad
    scale: the smaller the quads, the denser the stars.  An index whose
    smallest quads do not fit across the image is what the catalog manager
    would not recommend for solving; it is still usable for verification and
    is only tried after the fitting ones.
    """

    width_radians = math.radians(field.width_degrees)

    def key(item: IndexSummary) -> tuple[int, float, int, int]:
        fits_field = item.scale_lower_radians < width_radians
        return (0 if fits_field else 1, item.scale_lower_radians, -(item.star_count or 0), int(item.index_id))

    return tuple(sorted(summaries, key=key))


def detect_field_stars(
    image: np.ndarray,
    parameters: CorrespondenceParameters,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Detect stars on the solved image; centroids are returned FITS one-based."""

    from ufwbpp_registration.pipeline import DetectionConfig, detect_stars

    if image.ndim != 2:
        raise CatalogCorrespondenceError(
            "CORRESPONDENCE_INPUT_UNSUPPORTED", "catalog verification needs a two-dimensional image"
        )
    config = DetectionConfig(
        detection_sigma=parameters.detection_sigma,
        minimum_source_pixels=parameters.minimum_source_pixels,
        maximum_stars=parameters.maximum_stars,
        background_box=parameters.background_box,
    )
    try:
        catalog = detect_stars(image, config)
    except Exception as error:
        raise CatalogCorrespondenceError(
            "CORRESPONDENCE_DETECTION_FAILED", f"{type(error).__name__}: {error}"
        ) from error
    # SEP centroids use the array convention (the centre of the first pixel
    # is 0, 0); the WCS calls below use FITS one-based pixels like the
    # astrometry.net .corr table, so the same offset is applied here.
    points = np.asarray(catalog.points, dtype=np.float64) + 1.0
    fwhm = np.asarray(catalog.fwhm, dtype=np.float64)
    diagnostics = {
        "detectedStars": int(catalog.detected_count),
        "rankedStars": int(points.shape[0]),
        "background": float(catalog.background),
        "noise": float(catalog.noise),
        "medianFwhmPixels": float(np.median(fwhm)) if fwhm.size else None,
    }
    return points, fwhm, diagnostics


def match_mutual_nearest(
    detected: np.ndarray,
    predicted: np.ndarray,
    radius_pixels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one pairs where each is the other's nearest neighbour within the radius.

    Returns ``(detection_rows, catalog_rows, distances)``.  Distances are
    evaluated exhaustively in bounded blocks, so the result is deterministic
    and independent of any spatial-index library.
    """

    detected = np.asarray(detected, dtype=np.float64).reshape(-1, 2)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    detection_count = detected.shape[0]
    catalog_count = predicted.shape[0]
    empty = np.empty(0, dtype=np.int64)
    if detection_count == 0 or catalog_count == 0:
        return empty, empty, np.empty(0, dtype=np.float64)
    nearest_catalog = np.full(detection_count, -1, dtype=np.int64)
    nearest_catalog_d2 = np.full(detection_count, np.inf, dtype=np.float64)
    nearest_detection = np.full(catalog_count, -1, dtype=np.int64)
    nearest_detection_d2 = np.full(catalog_count, np.inf, dtype=np.float64)
    for start in range(0, detection_count, _MATCH_CHUNK_ROWS):
        block = detected[start : start + _MATCH_CHUNK_ROWS]
        delta = block[:, None, :] - predicted[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", delta, delta)
        catalog_index = np.argmin(d2, axis=1)
        block_rows = np.arange(block.shape[0])
        nearest_catalog[start : start + block.shape[0]] = catalog_index
        nearest_catalog_d2[start : start + block.shape[0]] = d2[block_rows, catalog_index]
        detection_index = np.argmin(d2, axis=0)
        detection_d2 = d2[detection_index, np.arange(catalog_count)]
        # Strict comparison keeps the earlier (brighter) detection on ties.
        better = detection_d2 < nearest_detection_d2
        nearest_detection[better] = detection_index[better] + start
        nearest_detection_d2[better] = detection_d2[better]
    candidates = np.arange(detection_count)
    mutual = candidates[nearest_detection[nearest_catalog] == candidates]
    within = mutual[nearest_catalog_d2[mutual] <= float(radius_pixels) ** 2]
    return within.astype(np.int64), nearest_catalog[within].astype(np.int64), np.sqrt(nearest_catalog_d2[within])



def _write_artifact(path: Path, payload: bytes) -> str:
    body = payload + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise CatalogCorrespondenceError("CORRESPONDENCE_ARTIFACT_WRITE_FAILED", f"{path.name}: {error}") from error
    return hashlib.sha256(body).hexdigest()


def verify_solution(
    *,
    image: np.ndarray,
    wcs_header: Mapping[str, Any] | fits.Header,
    image_shape: tuple[int, int],
    catalog_root: Path,
    index_artifacts: Sequence[Mapping[str, Any]],
    artifact_dir: Path,
    parameters: CorrespondenceParameters | None = None,
) -> CorrespondenceVerification:
    """Recompute catalog-correspondence quality for a WCS from managed indexes.

    ``index_artifacts`` are the ``artifacts`` of a pre-solve installed-set
    snapshot; only ``index-NNNN.fits`` entries take part and each read is
    bound to that snapshot's file identity.  The returned quality is not yet
    catalog-managed; ``bind_installed_set`` performs that binding.
    """

    parameters = parameters or CorrespondenceParameters()
    parameters.validate()
    height, width = image_shape
    field = field_geometry(wcs_header, image_shape)
    summaries: list[tuple[IndexSummary, Mapping[str, Any]]] = []
    for artifact in index_artifacts:
        relative_name = str(artifact.get("relativeName") or artifact.get("artifactId") or "")
        if _INDEX_FILE_NAME.fullmatch(relative_name) is None:
            continue
        summaries.append((read_index_summary(catalog_root / relative_name), artifact))
    if not summaries:
        raise CatalogCorrespondenceError(
            "CATALOG_INSTALLED_SET_UNBOUND", "the managed installed set contains no index-NNNN.fits artifact"
        )
    by_name = {summary.relative_name: artifact for summary, artifact in summaries}
    ranked = rank_indexes([summary for summary, _ in summaries], field)
    search_radius = parameters.search_radius_diagonal_fraction * field.diagonal_degrees
    selection: list[dict[str, Any]] = []
    stars: IndexStars | None = None
    for summary in ranked:
        candidate = read_index_stars(
            catalog_root / summary.relative_name,
            centre_ra_degrees=field.centre_ra_degrees,
            centre_dec_degrees=field.centre_dec_degrees,
            radius_degrees=search_radius,
            expected_stat=by_name[summary.relative_name].get("statIdentity"),
        )
        record = {**summary.serializable(), "starsWithinSearchRadius": candidate.count}
        eligible = candidate.count >= parameters.minimum_matches
        record["selected"] = eligible
        selection.append(record)
        if eligible:
            stars = candidate
            break
    if stars is None:
        raise CatalogCorrespondenceError(
            "CATALOG_FIELD_UNCOVERED",
            f"no managed index holds {parameters.minimum_matches} stars within {search_radius:.4g} degrees of the solved centre",
        )
    detected, fwhm, detection_diagnostics = detect_field_stars(image, parameters)
    if detected.shape[0] < parameters.minimum_matches:
        raise CatalogCorrespondenceError(
            "CORRESPONDENCE_DETECTION_INSUFFICIENT",
            f"only {detected.shape[0]} stars were detected on the image; at least {parameters.minimum_matches} are needed",
        )
    catalog_world = np.column_stack((stars.ra_degrees % 360.0, stars.dec_degrees))
    try:
        celestial = WCS(wcs_header, relax=False).celestial
        predicted = celestial.all_world2pix(catalog_world, 1)
    except Exception as error:
        raise CatalogCorrespondenceError("CORRESPONDENCE_WCS_FAILED", str(error)) from error
    finite = np.all(np.isfinite(predicted), axis=1)
    inside = finite & (predicted[:, 0] >= 0.5) & (predicted[:, 0] <= width + 0.5) & (predicted[:, 1] >= 0.5) & (predicted[:, 1] <= height + 0.5)
    inside_rows = np.flatnonzero(inside)
    if stars.magnitudes is not None and inside_rows.size > parameters.maximum_catalog_stars:
        magnitudes = stars.magnitudes[inside_rows]
        order = np.lexsort((inside_rows, np.where(np.isfinite(magnitudes), magnitudes, np.inf)))
        inside_rows = np.sort(inside_rows[order[: parameters.maximum_catalog_stars]])
    else:
        inside_rows = inside_rows[: parameters.maximum_catalog_stars]
    median_fwhm = float(np.median(fwhm)) if fwhm.size else 0.0
    radius = max(parameters.minimum_match_radius_pixels, parameters.match_radius_fwhm_factor * median_fwhm)
    detection_rows, catalog_rows, _ = match_mutual_nearest(detected, predicted[inside_rows], radius)
    catalog_rows = inside_rows[catalog_rows]
    if detection_rows.size < parameters.minimum_matches:
        raise CatalogCorrespondenceError(
            "CORRESPONDENCE_UNIQUE_MATCHES_MISSING",
            f"{detection_rows.size} mutual image/catalog matches within {radius:.3g} px; at least {parameters.minimum_matches} are needed",
        )
    # Residuals exactly as the astrometry.net adapter computes them from
    # .corr rows: pixel offsets in the image and spherical offsets on the sky.
    field_pixels = detected[detection_rows]
    matched_world = catalog_world[catalog_rows]
    try:
        predicted_pixels = celestial.all_world2pix(matched_world, 1)
        field_world = celestial.all_pix2world(field_pixels, 1)
    except Exception as error:
        raise CatalogCorrespondenceError("CORRESPONDENCE_WCS_FAILED", str(error)) from error
    if not np.all(np.isfinite(predicted_pixels)) or not np.all(np.isfinite(field_world)):
        raise CatalogCorrespondenceError("CORRESPONDENCE_WCS_FAILED", "correspondence transforms are non-finite")
    pixel_residuals = np.linalg.norm(field_pixels - predicted_pixels, axis=1)
    angular_residuals = _angular_separation_arcsec(field_world, matched_world)
    if not np.all(np.isfinite(pixel_residuals)) or not np.all(np.isfinite(angular_residuals)):
        raise CatalogCorrespondenceError("CORRESPONDENCE_RMS_INVALID", "correspondence residuals are non-finite")
    rms_pixels = float(math.sqrt(float(np.mean(np.square(pixel_residuals)))))
    rms_arcsec = float(math.sqrt(float(np.mean(np.square(angular_residuals)))))
    try:
        parity = wcs_parity(wcs_header)
    except Exception as error:
        raise CatalogCorrespondenceError("CORRESPONDENCE_WCS_FAILED", str(error)) from error

    index_identities = (stars.summary.identity,)
    catalog_ids = stars.rows[catalog_rows]
    catalog_row_records = sorted(
        (
            str(int(catalog_ids[position])),
            format(float(matched_world[position, 0]), ".17g"),
            format(float(matched_world[position, 1]), ".17g"),
        )
        for position in range(len(catalog_rows))
    )
    catalog_identity = hashlib.sha256(
        canonical_json({"indexes": list(index_identities), "matchedCatalogRows": catalog_row_records})
    ).hexdigest()
    order = np.argsort(catalog_ids, kind="stable")
    matches = [
        {
            "fieldId": int(detection_rows[position]),
            "fieldX": float(field_pixels[position, 0]),
            "fieldY": float(field_pixels[position, 1]),
            "indexId": int(catalog_ids[position]),
            "indexRa": float(matched_world[position, 0]),
            "indexDec": float(matched_world[position, 1]),
            "indexMag": (
                float(stars.magnitudes[catalog_rows[position]])
                if stars.magnitudes is not None and math.isfinite(float(stars.magnitudes[catalog_rows[position]]))
                else None
            ),
            "residualPixels": float(pixel_residuals[position]),
            "residualArcsec": float(angular_residuals[position]),
        }
        for position in order
    ]
    correspondence_payload = {
        "schemaVersion": 1,
        "method": VERIFICATION_METHOD,
        "coordinateOrigin": 1,
        "indexIdentities": list(index_identities),
        "parameters": parameters.serializable(),
        "matchRadiusPixels": radius,
        "matches": matches,
    }
    match_payload = {
        "schemaVersion": 1,
        "method": VERIFICATION_METHOD,
        "indexIdentities": list(index_identities),
        "index": stars.summary.serializable(),
        "field": field.serializable(),
        "searchRadiusDegrees": search_radius,
        "indexSelection": selection,
        "nmatch": int(detection_rows.size),
        "parity": parity.value,
    }
    correspondence_path = artifact_dir / CORRESPONDENCE_ARTIFACT_NAME
    match_path = artifact_dir / MATCH_ARTIFACT_NAME
    correspondence_sha256 = _write_artifact(correspondence_path, canonical_json(correspondence_payload))
    _write_artifact(match_path, canonical_json(match_payload))
    quality = AstrometricQuality(
        matched_stars=int(detection_rows.size),
        rms_pixels=rms_pixels,
        rms_arcsec=rms_arcsec,
        parity=parity,
        catalog_identity=catalog_identity,
        index_identities=index_identities,
        correspondence_sha256=correspondence_sha256,
    )
    correspondence_diagnostics = {
        **detection_diagnostics,
        "catalogStarsWithinSearchRadius": stars.count,
        "catalogStarsInsideImage": int(np.count_nonzero(inside)),
        "catalogStarsConsidered": int(inside_rows.size),
        "matchRadiusPixels": radius,
        "uniqueOneToOneMatches": int(detection_rows.size),
        "coordinateOrigin": 1,
        "residualMethod": "WCS(index_ra,index_dec)->pixel vs detected centroid; spherical sky RMS",
        "verificationMethod": VERIFICATION_METHOD,
        "parameters": parameters.serializable(),
    }
    match_diagnostics = {
        "indexIdentities": list(index_identities),
        "indexSelection": selection,
        "indexEndian": stars.endian,
        "field": field.serializable(),
        "searchRadiusDegrees": search_radius,
        "validatedParity": parity.value,
        "backendNmatch": int(detection_rows.size),
        "verificationMethod": VERIFICATION_METHOD,
    }
    return CorrespondenceVerification(
        quality=quality,
        correspondence_path=correspondence_path,
        match_path=match_path,
        correspondence_diagnostics=correspondence_diagnostics,
        match_diagnostics=match_diagnostics,
    )


def bind_installed_set(
    quality: AstrometricQuality,
    snapshot: Mapping[str, Any],
    *,
    manifest_dir: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
) -> tuple[AstrometricQuality, dict[str, Any]]:
    """Bind the index identities to the pre-solve installed-set bytes.

    Mirrors the astrometry.net adapter: the matched index must have been in
    the pre-solve snapshot with the same size and digest, and re-hashing the
    whole set afterwards must show no drift.  ``CatalogError`` propagates so
    the adapter can decide between failing closed and diagnostic use.
    """

    binding = installed_set_identity_for_solver_indexes(
        quality.index_identities,
        catalog_root=Path(str(snapshot["catalogRoot"])),
        manifest_dir=manifest_dir,
        environment=environment,
    )
    pre_receipts = {item["installedSetIdentity"]: item for item in snapshot["receipts"]}
    if binding["installedSetIdentity"] not in pre_receipts:
        raise CatalogError(
            "CATALOG_INSTALLED_SET_DRIFT",
            "the matched index was not present in the pre-solve installed set",
        )
    pre_artifacts = {item["artifactId"]: item for item in snapshot["artifacts"]}
    for artifact in binding["indexArtifacts"]:
        before = pre_artifacts.get(artifact["relativeName"])
        if before is None or any(before[key] != artifact[key] for key in ("sizeBytes", "sha256")):
            raise CatalogError(
                "CATALOG_INSTALLED_SET_DRIFT",
                "the matched index differs from its pre-solve byte identity",
            )
    verify_installed_set_snapshot(snapshot, manifest_dir=manifest_dir, environment=environment)
    index_artifacts = tuple(
        SolverIndexArtifact(
            index_id=item["indexId"],
            relative_name=item["relativeName"],
            size_bytes=item["sizeBytes"],
            sha256=item["sha256"],
            manifest_sha256=item["manifestSha256"],
            installed_set_identity=item["installedSetIdentity"],
        )
        for item in binding["indexArtifacts"]
    )
    bound = replace(
        quality,
        catalog_managed=True,
        installed_set_identity=binding["installedSetIdentity"],
        catalog_manifest_sha256=binding["manifestSha256"],
        index_artifacts=index_artifacts,
    )
    catalog_binding = {
        "catalogManaged": True,
        "catalogId": binding["catalogId"],
        "installedSetIdentity": binding["installedSetIdentity"],
        "manifestSha256": binding["manifestSha256"],
        "config": {
            "relativeName": "astrometry.cfg",
            "sizeBytes": binding["config"]["sizeBytes"],
            "sha256": binding["config"]["sha256"],
        },
        "indexArtifacts": [item.serializable() for item in index_artifacts],
    }
    return bound, catalog_binding


__all__ = [
    "CORRESPONDENCE_ARTIFACT_NAME",
    "MATCH_ARTIFACT_NAME",
    "VERIFICATION_METHOD",
    "CatalogCorrespondenceError",
    "CorrespondenceParameters",
    "CorrespondenceVerification",
    "FieldGeometry",
    "IndexStars",
    "IndexSummary",
    "bind_installed_set",
    "detect_field_stars",
    "field_geometry",
    "match_mutual_nearest",
    "rank_indexes",
    "read_index_stars",
    "read_index_summary",
    "verify_solution",
]
