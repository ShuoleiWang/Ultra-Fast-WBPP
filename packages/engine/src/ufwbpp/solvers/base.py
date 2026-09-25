from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import math
import re
from typing import Any, Mapping, Protocol, runtime_checkable

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np

from ..backends import Backend, BackendDescriptor
from ..models import json_value


class SolverStatus(StrEnum):
    SOLVED = "SOLVED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class SolutionKind(StrEnum):
    SOLVED = "SOLVED"
    EXISTING = "EXISTING"
    SEED = "SEED"
    NONE = "NONE"


class WcsParity(StrEnum):
    """Handedness of the celestial pixel-to-world linear transform."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


@dataclass(frozen=True, slots=True)
class SolverIndexArtifact:
    """Content identity of one managed astrometry.net index file."""

    index_id: str
    relative_name: str
    size_bytes: int
    sha256: str
    manifest_sha256: str
    installed_set_identity: str

    def serializable(self) -> dict[str, Any]:
        return {
            "indexId": self.index_id,
            "relativeName": self.relative_name,
            "sizeBytes": self.size_bytes,
            "sha256": self.sha256,
            "manifestSha256": self.manifest_sha256,
            "installedSetIdentity": self.installed_set_identity,
        }


@dataclass(frozen=True, slots=True)
class AstrometricQuality:
    """Catalog-correspondence evidence for one plate solution.

    A mathematically invertible WCS is not proof that the image was solved at
    the right place on the sky.  These values must therefore come from actual
    image/catalog correspondences and are kept separate from the WCS header.
    """

    matched_stars: int
    rms_pixels: float
    rms_arcsec: float
    parity: WcsParity
    catalog_identity: str
    index_identities: tuple[str, ...]
    correspondence_sha256: str
    catalog_managed: bool = False
    installed_set_identity: str | None = None
    catalog_manifest_sha256: str | None = None
    index_artifacts: tuple[SolverIndexArtifact, ...] = ()

    def serializable(self) -> dict[str, Any]:
        return {
            "matchedStars": self.matched_stars,
            "rmsPixels": self.rms_pixels,
            "rmsArcsec": self.rms_arcsec,
            "parity": self.parity.value,
            "catalogIdentity": self.catalog_identity,
            "indexIdentities": list(self.index_identities),
            "correspondenceSha256": self.correspondence_sha256,
            "catalogManaged": self.catalog_managed,
            "installedSetIdentity": self.installed_set_identity,
            "catalogManifestSha256": self.catalog_manifest_sha256,
            "indexArtifacts": [item.serializable() for item in self.index_artifacts],
        }


@dataclass(frozen=True, slots=True)
class SolveRequest:
    input_path: str
    output_path: str
    ra_hint_degrees: float | None = None
    dec_hint_degrees: float | None = None
    field_of_view_degrees: float | None = None
    search_radius_degrees: float | None = None


@dataclass(frozen=True, slots=True)
class SolverResult:
    backend_id: str
    status: SolverStatus
    solution_kind: SolutionKind
    backend_confirmed: bool
    header: Mapping[str, Any] = field(default_factory=dict)
    image_shape: tuple[int, int] | None = None
    output_path: str | None = None
    astrometric_quality: AstrometricQuality | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "backendId": self.backend_id,
            "status": self.status.value,
            "solutionKind": self.solution_kind.value,
            "backendConfirmed": self.backend_confirmed,
            "imageShape": list(self.image_shape) if self.image_shape else None,
            "outputPath": self.output_path,
            "astrometricQuality": (
                self.astrometric_quality.serializable()
                if self.astrometric_quality is not None
                else None
            ),
            "evidence": json_value(self.evidence, "solver.evidence"),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class WcsValidation:
    valid: bool
    code: str
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "code": self.code,
            "message": self.message,
            "diagnostics": json_value(self.diagnostics, "wcs.diagnostics"),
        }


@runtime_checkable
class SolverBackend(Backend, Protocol):
    def solve(self, request: SolveRequest) -> SolverResult: ...


def _header(value: Mapping[str, Any] | fits.Header) -> fits.Header:
    if isinstance(value, fits.Header):
        return value.copy()
    header = fits.Header()
    for key, item in value.items():
        if item is not None:
            header[str(key)] = item
    return header


def validate_wcs_header(
    value: Mapping[str, Any] | fits.Header,
    *,
    image_shape: tuple[int, int] | None = None,
) -> WcsValidation:
    """Validate a celestial WCS numerically; pointing seeds fail closed."""

    try:
        header = _header(value)
    except Exception as error:
        return WcsValidation(False, "WCS_HEADER_INVALID", str(error))

    ctype1 = str(header.get("CTYPE1", "")).upper()
    ctype2 = str(header.get("CTYPE2", "")).upper()
    if "RA" not in ctype1 or "DEC" not in ctype2:
        return WcsValidation(
            False,
            "WCS_MISSING_CELESTIAL_AXES",
            "RA/Dec pointing metadata is only a seed; CTYPE1/CTYPE2 celestial axes are required.",
        )
    required = ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2")
    missing = [key for key in required if key not in header]
    if missing:
        return WcsValidation(
            False,
            "WCS_INCOMPLETE_REFERENCE",
            "the WCS reference point is incomplete",
            {"missing": missing},
        )

    if image_shape is None:
        try:
            width = int(header["NAXIS1"])
            height = int(header["NAXIS2"])
        except (KeyError, TypeError, ValueError):
            return WcsValidation(
                False,
                "WCS_IMAGE_GEOMETRY_MISSING",
                "NAXIS1/NAXIS2 or an explicit image_shape is required for validation.",
            )
    else:
        if (
            not isinstance(image_shape, tuple)
            or len(image_shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in image_shape)
        ):
            return WcsValidation(
                False,
                "WCS_IMAGE_GEOMETRY_INVALID",
                "image_shape must be a (height, width) integer tuple",
            )
        height, width = image_shape
    if width < 2 or height < 2:
        return WcsValidation(False, "WCS_IMAGE_GEOMETRY_INVALID", "image geometry is invalid")

    try:
        if all(key in header for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
            matrix = np.asarray(
                [
                    [header["CD1_1"], header["CD1_2"]],
                    [header["CD2_1"], header["CD2_2"]],
                ],
                dtype=np.float64,
            )
        elif "CDELT1" in header and "CDELT2" in header:
            pc = np.asarray(
                [
                    [header.get("PC1_1", 1.0), header.get("PC1_2", 0.0)],
                    [header.get("PC2_1", 0.0), header.get("PC2_2", 1.0)],
                ],
                dtype=np.float64,
            )
            matrix = np.diag(
                np.asarray([header["CDELT1"], header["CDELT2"]], dtype=np.float64)
            ) @ pc
        else:
            return WcsValidation(
                False,
                "WCS_LINEAR_TRANSFORM_MISSING",
                "a CD matrix or CDELT plus optional PC matrix is required",
            )
        determinant = float(np.linalg.det(matrix))
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        return WcsValidation(False, "WCS_LINEAR_TRANSFORM_INVALID", str(error))
    if not np.all(np.isfinite(matrix)) or not math.isfinite(determinant) or abs(determinant) < 1e-18:
        return WcsValidation(
            False,
            "WCS_LINEAR_TRANSFORM_SINGULAR",
            "the WCS linear transform is non-finite or singular",
            {"determinant": determinant if math.isfinite(determinant) else None},
        )

    try:
        wcs = WCS(header, relax=False)
        if not wcs.has_celestial:
            return WcsValidation(False, "WCS_NOT_CELESTIAL", "Astropy found no celestial WCS")
        celestial = wcs.celestial
        points = np.asarray(
            [
                [0.0, 0.0],
                [(width - 1) / 2.0, (height - 1) / 2.0],
                [float(width - 1), float(height - 1)],
            ],
            dtype=np.float64,
        )
        world = celestial.all_pix2world(points, 0)
        returned = celestial.all_world2pix(world, 0)
        if not np.all(np.isfinite(world)) or not np.all(np.isfinite(returned)):
            raise ValueError("the pixel/world transform produced non-finite coordinates")
        roundtrip_error = float(np.max(np.abs(returned - points)))
        if roundtrip_error > 0.05:
            raise ValueError(f"pixel/world round-trip error is {roundtrip_error:.6g} px")
        scales = np.asarray(proj_plane_pixel_scales(celestial), dtype=np.float64)
        if scales.size < 2 or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("the projected pixel scale is invalid")
    except Exception as error:
        return WcsValidation(False, "WCS_NUMERIC_VALIDATION_FAILED", str(error))

    return WcsValidation(
        True,
        "WCS_VALID",
        "celestial WCS passed structural and numerical validation",
        {
            "width": width,
            "height": height,
            "determinant": determinant,
            "roundtripMaxPixels": roundtrip_error,
            "pixelScaleDegrees": [float(value) for value in scales[:2]],
        },
    )


def wcs_parity(value: Mapping[str, Any] | fits.Header) -> WcsParity:
    """Return WCS handedness from Astropy's celestial pixel scale matrix."""

    celestial = WCS(_header(value), relax=False).celestial
    matrix = np.asarray(celestial.pixel_scale_matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("the celestial WCS does not have a finite 2x2 pixel scale matrix")
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or abs(determinant) < 1e-18:
        raise ValueError("the celestial WCS parity is undefined")
    return WcsParity.POSITIVE if determinant > 0 else WcsParity.NEGATIVE


def canonical_wcs_sha256(value: Mapping[str, Any] | fits.Header) -> str:
    """Return the raw lowercase SHA-256 of Astropy's canonical celestial cards."""

    celestial_header = WCS(_header(value), relax=True).celestial.to_header(relax=True)
    serialized = celestial_header.tostring(
        sep="\n",
        endcard=True,
        padding=False,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def validate_astrometric_quality(
    result: SolverResult,
    *,
    min_matches: int = 1,
    max_rms_arcsec: float | None = None,
    require_managed_catalog: bool = False,
) -> WcsValidation:
    """Validate catalog evidence and its agreement with the returned WCS."""

    if isinstance(min_matches, bool) or not isinstance(min_matches, int) or min_matches < 1:
        return WcsValidation(False, "SOLVER_QUALITY_POLICY_INVALID", "min_matches must be a positive integer")
    if max_rms_arcsec is not None and (
        isinstance(max_rms_arcsec, bool)
        or not isinstance(max_rms_arcsec, (int, float))
        or not math.isfinite(float(max_rms_arcsec))
        or float(max_rms_arcsec) <= 0
    ):
        return WcsValidation(False, "SOLVER_QUALITY_POLICY_INVALID", "max_rms_arcsec must be finite and positive")
    quality = result.astrometric_quality
    if quality is None:
        return WcsValidation(
            False,
            "SOLVER_QUALITY_EVIDENCE_MISSING",
            "the backend did not provide independently measured catalog correspondences",
        )
    if (
        isinstance(quality.matched_stars, bool)
        or not isinstance(quality.matched_stars, int)
        or quality.matched_stars < 1
    ):
        return WcsValidation(False, "SOLVER_MATCH_COUNT_INVALID", "matchedStars must be a positive integer")
    for name, value in (("rmsPixels", quality.rms_pixels), ("rmsArcsec", quality.rms_arcsec)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            return WcsValidation(False, "SOLVER_RMS_INVALID", f"{name} must be finite and non-negative")
    if not isinstance(quality.parity, WcsParity):
        return WcsValidation(False, "SOLVER_PARITY_INVALID", "parity must be a WcsParity value")
    if (
        not isinstance(quality.catalog_identity, str)
        or not re.fullmatch(r"[0-9a-f]{64}", quality.catalog_identity)
    ):
        return WcsValidation(
            False,
            "SOLVER_CATALOG_IDENTITY_INVALID",
            "catalogIdentity must be a SHA-256 identity of the matched catalog rows",
        )
    if (
        not isinstance(quality.index_identities, tuple)
        or not quality.index_identities
        or any(not isinstance(item, str) or not item.strip() for item in quality.index_identities)
        or len(set(quality.index_identities)) != len(quality.index_identities)
    ):
        return WcsValidation(False, "SOLVER_INDEX_IDENTITY_MISSING", "one or more unique indexIdentities are required")
    if (
        not isinstance(quality.correspondence_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", quality.correspondence_sha256)
    ):
        return WcsValidation(False, "SOLVER_CORRESPONDENCE_IDENTITY_INVALID", "correspondenceSha256 must be a SHA-256 identity")
    if not isinstance(quality.catalog_managed, bool):
        return WcsValidation(False, "SOLVER_CATALOG_MANAGEMENT_INVALID", "catalogManaged must be a boolean")
    if not quality.catalog_managed:
        if require_managed_catalog:
            return WcsValidation(
                False,
                "SOLVER_MANAGED_CATALOG_REQUIRED",
                "a required final solution must bind the exact managed index bytes",
            )
        if (
            quality.installed_set_identity is not None
            or quality.catalog_manifest_sha256 is not None
            or quality.index_artifacts
        ):
            return WcsValidation(
                False,
                "SOLVER_CATALOG_BINDING_INVALID",
                "unmanaged catalog evidence must not claim managed index identities",
            )
    else:
        if (
            not isinstance(quality.installed_set_identity, str)
            or not re.fullmatch(r"[0-9a-f]{64}", quality.installed_set_identity)
            or not isinstance(quality.catalog_manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", quality.catalog_manifest_sha256)
            or not isinstance(quality.index_artifacts, tuple)
            or not quality.index_artifacts
        ):
            return WcsValidation(
                False,
                "SOLVER_MANAGED_CATALOG_IDENTITY_INVALID",
                "managed catalog evidence needs installed-set, manifest, and index artifact identities",
            )
        artifact_index_ids: set[str] = set()
        for artifact in quality.index_artifacts:
            if not isinstance(artifact, SolverIndexArtifact):
                return WcsValidation(False, "SOLVER_INDEX_ARTIFACT_INVALID", "indexArtifacts contains an invalid value")
            if (
                not re.fullmatch(r"[0-9]+", artifact.index_id)
                or artifact.relative_name != f"index-{artifact.index_id}.fits"
                or isinstance(artifact.size_bytes, bool)
                or not isinstance(artifact.size_bytes, int)
                or artifact.size_bytes < 1
                or not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
                or artifact.manifest_sha256 != quality.catalog_manifest_sha256
                or artifact.installed_set_identity != quality.installed_set_identity
                or artifact.index_id in artifact_index_ids
            ):
                return WcsValidation(
                    False,
                    "SOLVER_INDEX_ARTIFACT_INVALID",
                    "index artifact identity is malformed, duplicated, or disagrees with the installed set",
                )
            artifact_index_ids.add(artifact.index_id)
        logical_index_ids = {
            match.group(1)
            for identity in quality.index_identities
            if (match := re.fullmatch(
                r"astrometry\.net:index:([0-9]+):healpix:[^:]+:hpnside:[^:]+",
                identity,
            ))
        }
        if len(logical_index_ids) != len(quality.index_identities) or logical_index_ids != artifact_index_ids:
            return WcsValidation(
                False,
                "SOLVER_INDEX_ARTIFACT_MISMATCH",
                "backend INDEXID evidence does not match the managed index artifacts",
            )

    header_validation = validate_wcs_header(result.header, image_shape=result.image_shape)
    if not header_validation.valid:
        return header_validation
    try:
        expected_parity = wcs_parity(result.header)
    except Exception as error:
        return WcsValidation(False, "SOLVER_PARITY_INVALID", str(error))
    if quality.parity is not expected_parity:
        return WcsValidation(
            False,
            "SOLVER_PARITY_MISMATCH",
            "reported parity disagrees with the returned WCS",
            {"reported": quality.parity.value, "expected": expected_parity.value},
        )

    scales_arcsec = np.asarray(header_validation.diagnostics["pixelScaleDegrees"], dtype=np.float64) * 3600.0
    geometric_scale = float(math.sqrt(float(scales_arcsec[0] * scales_arcsec[1])))
    expected_arcsec = float(quality.rms_pixels) * geometric_scale
    if expected_arcsec == 0.0:
        consistent = float(quality.rms_arcsec) <= 1e-9
        ratio = 1.0 if consistent else math.inf
    else:
        ratio = float(quality.rms_arcsec) / expected_arcsec
        # Spherical and tangent-plane residuals need not be bit-identical, but
        # they must describe the same order of error.
        consistent = 0.5 <= ratio <= 2.0
    if not consistent:
        return WcsValidation(
            False,
            "SOLVER_RMS_UNITS_INCONSISTENT",
            "pixel and angular RMS do not agree with the solved pixel scale",
            {
                "rmsPixels": quality.rms_pixels,
                "rmsArcsec": quality.rms_arcsec,
                "pixelScaleArcsec": geometric_scale,
                "angularToPixelRmsRatio": ratio if math.isfinite(ratio) else None,
            },
        )
    if quality.matched_stars < min_matches:
        return WcsValidation(
            False,
            "SOLVER_MATCH_COUNT_BELOW_MINIMUM",
            f"{quality.matched_stars} unique matches are below the required {min_matches}",
            {"matchedStars": quality.matched_stars, "minMatches": min_matches},
        )
    if max_rms_arcsec is not None and quality.rms_arcsec > float(max_rms_arcsec):
        return WcsValidation(
            False,
            "SOLVER_RMS_ABOVE_MAXIMUM",
            f"RMS {quality.rms_arcsec:.6g} arcsec exceeds the {float(max_rms_arcsec):.6g} arcsec limit",
            {"rmsArcsec": quality.rms_arcsec, "maxRmsArcsec": float(max_rms_arcsec)},
        )
    return WcsValidation(
        True,
        "SOLVER_QUALITY_VALID",
        "catalog correspondence evidence passed the scientific quality gate",
        {
            **quality.serializable(),
            "minMatches": min_matches,
            "maxRmsArcsec": max_rms_arcsec,
            "pixelScaleArcsec": geometric_scale,
        },
    )


def validate_solver_result(
    result: SolverResult,
    *,
    require_scientific_evidence: bool = False,
    min_matches: int = 1,
    max_rms_arcsec: float | None = None,
    require_managed_catalog: bool | None = None,
) -> WcsValidation:
    """Accept only a newly solved, backend-confirmed, numerically valid WCS."""

    if result.status != SolverStatus.SOLVED:
        return WcsValidation(
            False,
            "SOLVER_DID_NOT_SOLVE",
            result.error or f"backend returned {result.status.value}",
        )
    if result.solution_kind != SolutionKind.SOLVED:
        return WcsValidation(
            False,
            "SOLVER_RESULT_IS_NOT_NEW_SOLUTION",
            "seeded or inherited coordinates do not count as a solved output",
            {"solutionKind": result.solution_kind.value},
        )
    if not result.backend_confirmed:
        return WcsValidation(
            False,
            "SOLVER_CONFIRMATION_MISSING",
            "a zero exit code or WCS-looking header is insufficient without backend solve confirmation",
        )
    validation = validate_wcs_header(result.header, image_shape=result.image_shape)
    if not validation.valid:
        return validation
    if result.astrometric_quality is not None or require_scientific_evidence:
        managed_required = (
            require_scientific_evidence
            if require_managed_catalog is None
            else require_managed_catalog
        )
        quality_validation = validate_astrometric_quality(
            result,
            min_matches=min_matches,
            max_rms_arcsec=max_rms_arcsec,
            require_managed_catalog=managed_required,
        )
        if not quality_validation.valid:
            return quality_validation
    diagnostics = dict(validation.diagnostics)
    diagnostics["backendId"] = result.backend_id
    diagnostics["backendEvidence"] = json_value(result.evidence, "solver.evidence")
    if result.astrometric_quality is not None:
        diagnostics["astrometricQuality"] = result.astrometric_quality.serializable()
    return WcsValidation(True, "SOLVER_RESULT_VALID", validation.message, diagnostics)


@dataclass(frozen=True, slots=True)
class DeclarativeSolverBackend:
    """Published provider seam; execution stays disabled until its adapter ships."""

    descriptor: BackendDescriptor

    def validate_options(self, options: dict[str, Any]) -> tuple[str, ...]:
        allowed = {
            "raHintDegrees",
            "decHintDegrees",
            "fieldOfViewDegrees",
            "searchRadiusDegrees",
        }
        unknown = sorted(set(options) - allowed)
        return tuple(f"unknown solver option: {key}" for key in unknown)

    def solve(self, request: SolveRequest) -> SolverResult:
        return SolverResult(
            backend_id=self.descriptor.backend_id,
            status=SolverStatus.UNAVAILABLE,
            solution_kind=SolutionKind.NONE,
            backend_confirmed=False,
            error=self.descriptor.reason or "solver execution adapter is not available",
        )


__all__ = [
    "AstrometricQuality",
    "DeclarativeSolverBackend",
    "SolutionKind",
    "SolveRequest",
    "SolverBackend",
    "SolverIndexArtifact",
    "SolverResult",
    "SolverStatus",
    "WcsParity",
    "WcsValidation",
    "canonical_wcs_sha256",
    "validate_astrometric_quality",
    "validate_solver_result",
    "validate_wcs_header",
    "wcs_parity",
]
