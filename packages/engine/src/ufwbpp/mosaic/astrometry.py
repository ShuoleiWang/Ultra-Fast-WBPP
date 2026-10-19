"""A panel's TAN+SIP solution, refined on catalog stars.

A solver's solution is enough to find a frame on the sky, but a mosaic needs
more: its overlaps are the corners of the frames, exactly where a pure TAN
solution (astrometry.net's ``.match`` table) leaves the optical distortion,
and two panels disagreeing by a pixel there double every star of the
overlap.  :func:`refine_solution` therefore matches the stars detected on
the frame to the managed catalog through the solver's solution and fits a
gnomonic projection with SIP polynomials (the forward ``A``/``B`` and the
inverse ``AP``/``BP``), choosing the polynomial order by the Bayesian
information criterion so that a sparse catalog never buys a wiggle it cannot
support.  The fit is a robust least-squares in the tangent plane of its own
centre, iterated until that centre is the CRVAL of the solution.

The panels of a mosaic are fitted at once (:func:`solve_mosaic`): with a
few dozen catalog stars per field each panel alone is off by a pixel or more
at its edges, where the panels overlap, while the hundreds of stars two
neighbours both measure there tie their solutions to a tenth of a pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray

from ..stacking.parameters import SHIFT_FIELD_MAXIMUM_ORDER, ShiftField
from .canvas import _plane, celestial_wcs


SIP_FIT_ALGORITHM = "tan-sip-catalog-fit-v1"
MAXIMUM_SIP_ORDER = 4
# Each fitted polynomial term needs this many stars on average.
STARS_PER_TERM = 3
# A first match tolerates a solver solution that ignores the distortion.
FIRST_MATCH_RADIUS_PIXELS = 6.0
FIRST_MATCH_RADIUS_FWHM = 2.5
_CLIP_ITERATIONS = 3
_CLIP_SIGMA = 4.0
_CLIP_FLOOR_PIXELS = 0.25
# A star with another detection closer than this many FWHM is left out of
# the fit: a blend pulls its centroid by up to a pixel.
ISOLATION_FWHM = 3.0
JOINT_FIT_ALGORITHM = "mosaic-joint-tan-sip-v1"
# Two panels constrain each other once this many of their stars match in
# the overlap; the difference of a star's two measured positions starts at
# this error per axis (re-estimated from the residuals).
MINIMUM_OVERLAP_MATCHES = 20
OVERLAP_SIGMA_PIXELS = 0.07
# Overlap stars are first matched through the panels' own catalog fits,
# which may disagree there by a pixel or two: within this many FWHM.
OVERLAP_MATCH_FWHM = 1.5
_JOINT_ITERATIONS = 20
_REMATCH_ROUNDS = 3
_INVERSE_GRID = 33


class MosaicAstrometryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class CatalogStars:
    ra_degrees: NDArray[np.float64]
    dec_degrees: NDArray[np.float64]
    magnitudes: NDArray[np.float64] | None
    identity: str


class CatalogStarSource(Protocol):
    """Catalog stars around a field (the managed index catalog in a run;
    the true star list in a synthetic test)."""

    def stars(self, centre_ra_degrees: float, centre_dec_degrees: float, radius_degrees: float) -> CatalogStars: ...


@dataclass(frozen=True)
class ManagedCatalogStars:
    """Stars of the managed astrometry.net indexes behind a solver config."""

    config_path: Path
    manifest_dir: Path | None = None
    environment: Mapping[str, str] | None = None

    def stars(self, centre_ra_degrees: float, centre_dec_degrees: float, radius_degrees: float) -> CatalogStars:
        from ..solvers.catalog_correspondence import (
            FieldGeometry,
            read_index_stars,
            read_index_summary,
            rank_indexes,
        )
        from ..solvers.catalogs import CatalogError, installed_set_snapshot_for_solver_config

        try:
            snapshot = installed_set_snapshot_for_solver_config(
                self.config_path, manifest_dir=self.manifest_dir, environment=self.environment
            )
        except CatalogError as error:
            raise MosaicAstrometryError(error.code, str(error)) from error
        root = Path(self.config_path).expanduser().resolve(strict=True).parent
        artifacts = {
            str(item["relativeName"]): item
            for item in snapshot["artifacts"]
            if str(item.get("relativeName", "")).startswith("index-")
        }
        summaries = [read_index_summary(root / name) for name in sorted(artifacts)]
        diameter = 2.0 * radius_degrees
        field_geometry = FieldGeometry(
            width=1,
            height=1,
            centre_ra_degrees=centre_ra_degrees,
            centre_dec_degrees=centre_dec_degrees,
            width_degrees=diameter / math.sqrt(2.0),
            height_degrees=diameter / math.sqrt(2.0),
            diagonal_degrees=diameter,
        )
        # Densest index first; a HEALPix-tiled index is read as the union of
        # its tiles, so a field on a tile boundary keeps its stars on both sides.
        ordered: dict[str, list[Any]] = {}
        for summary in rank_indexes(summaries, field_geometry):
            ordered.setdefault(summary.index_id, []).append(summary)
        for index_id, members in ordered.items():
            ra: list[NDArray[np.float64]] = []
            dec: list[NDArray[np.float64]] = []
            magnitudes: list[NDArray[np.float64]] | None = []
            read: list[Any] = []
            # Tiles in HEALPix order, so the star list does not depend on the ranking.
            for summary in sorted(members, key=lambda item: item.healpix):
                found = read_index_stars(
                    root / summary.relative_name,
                    centre_ra_degrees=centre_ra_degrees,
                    centre_dec_degrees=centre_dec_degrees,
                    radius_degrees=radius_degrees,
                    expected_stat=artifacts[summary.relative_name].get("statIdentity"),
                )
                if not found.count:
                    continue
                read.append(summary)
                ra.append(np.asarray(found.ra_degrees, dtype=np.float64) % 360.0)
                dec.append(np.asarray(found.dec_degrees, dtype=np.float64))
                if magnitudes is not None and found.magnitudes is not None:
                    magnitudes.append(np.asarray(found.magnitudes, dtype=np.float64))
                else:
                    magnitudes = None
            if not read:
                continue
            all_ra, all_dec = np.concatenate(ra), np.concatenate(dec)
            # A tile also holds the stars in a margin around it, so neighbouring
            # tiles repeat those; positions equal to 0.01" are one star.
            _, unique = np.unique(np.round(np.column_stack((all_ra, all_dec)) * 3.6e5), axis=0, return_index=True)
            unique = np.sort(unique)
            if unique.size >= 3 * STARS_PER_TERM * 3:
                identity = (
                    read[0].identity
                    if len(read) == 1
                    else f"astrometry.net:index:{index_id}:healpix-union:"
                    f"{'+'.join(str(item.healpix) for item in read)}:hpnside:{read[0].hpnside}"
                )
                return CatalogStars(
                    all_ra[unique],
                    all_dec[unique],
                    None if not magnitudes else np.concatenate(magnitudes)[unique],
                    identity,
                )
        raise MosaicAstrometryError(
            "MOSAIC_CATALOG_FIELD_UNCOVERED", "no managed index holds enough stars for this panel"
        )


def catalog_source_from_backends(backends: Sequence[Any]) -> CatalogStarSource | None:
    """The catalog stars of the first solver backend that has them: its own
    ``catalog_star_source``, else the managed catalog behind its solver
    config."""

    for backend in backends:
        source = getattr(backend, "catalog_star_source", None)
        if source is not None:
            return source
        config_path = getattr(backend, "config_path", None)
        if config_path is not None:
            return ManagedCatalogStars(
                Path(config_path),
                manifest_dir=getattr(backend, "catalog_manifest_dir", None),
                environment=dict(os.environ) | dict(getattr(backend, "environment", None) or {}),
            )
    return None


def _terms(order: int, *, minimum: int = 0) -> list[tuple[int, int]]:
    return [(p, total - p) for total in range(minimum, order + 1) for p in range(total, -1, -1)]


def _design(u: NDArray[np.float64], v: NDArray[np.float64], terms: Sequence[tuple[int, int]]) -> NDArray[np.float64]:
    return np.column_stack([u**p * v**q for p, q in terms])


@dataclass(frozen=True)
class SipFit:
    """A fitted solution and the evidence behind it."""

    header: dict[str, Any]
    order: int
    matched: int
    rms_pixels: float
    rms_arcsec: float
    max_residual_pixels: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "algorithm": SIP_FIT_ALGORITHM,
            "order": self.order,
            "matchedStars": self.matched,
            "rmsPixels": self.rms_pixels,
            "rmsArcsec": self.rms_arcsec,
            "maxResidualPixels": self.max_residual_pixels,
            **self.evidence,
        }


def _sip_header(
    crval: tuple[float, float],
    crpix: tuple[float, float],
    cd: NDArray[np.float64],
    sip_a: Mapping[tuple[int, int], float],
    sip_b: Mapping[tuple[int, int], float],
    order: int,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """The TAN(+SIP) header of a solution; the inverse polynomials (one
    order more) are fitted on a grid slightly larger than the frame."""

    height, width = shape
    length = max(width, height) / 2.0
    header: dict[str, Any] = {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN-SIP" if order >= 2 else "RA---TAN",
        "CTYPE2": "DEC--TAN-SIP" if order >= 2 else "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CRPIX1": crpix[0],
        "CRPIX2": crpix[1],
        "CD1_1": float(cd[0, 0]),
        "CD1_2": float(cd[0, 1]),
        "CD2_1": float(cd[1, 0]),
        "CD2_2": float(cd[1, 1]),
        "RADESYS": "ICRS",
    }
    if order >= 2:
        header["A_ORDER"] = order
        header["B_ORDER"] = order
        for term in _terms(order, minimum=2):
            header[f"A_{term[0]}_{term[1]}"] = sip_a[term]
            header[f"B_{term[0]}_{term[1]}"] = sip_b[term]
        # AP/BP map the distorted offsets back.
        samples = np.linspace(-0.02, 1.02, _INVERSE_GRID)
        grid_u, grid_v = np.broadcast_arrays(
            (samples * (width - 1))[None, :] + 1.0 - crpix[0],
            (samples * (height - 1))[:, None] + 1.0 - crpix[1],
        )
        grid_u, grid_v = grid_u.ravel(), grid_v.ravel()
        forward_u = grid_u + sum(value * grid_u**p * grid_v**q for (p, q), value in sip_a.items())
        forward_v = grid_v + sum(value * grid_u**p * grid_v**q for (p, q), value in sip_b.items())
        inverse_order = order + 1
        inverse_terms = _terms(inverse_order, minimum=1)
        inverse_design = _design(forward_u / length, forward_v / length, inverse_terms)
        inverse_scale = np.asarray([length ** (p + q) for p, q in inverse_terms], dtype=np.float64)
        ap, *_ = np.linalg.lstsq(inverse_design, grid_u - forward_u, rcond=None)
        bp, *_ = np.linalg.lstsq(inverse_design, grid_v - forward_v, rcond=None)
        header["AP_ORDER"] = inverse_order
        header["BP_ORDER"] = inverse_order
        for term, value_ap, value_bp in zip(inverse_terms, ap / inverse_scale, bp / inverse_scale):
            header[f"AP_{term[0]}_{term[1]}"] = float(value_ap)
            header[f"BP_{term[0]}_{term[1]}"] = float(value_bp)
    return header


def _fit_order(
    pixels: NDArray[np.float64],
    world: NDArray[np.float64],
    shape: tuple[int, int],
    order: int,
    crval: tuple[float, float],
) -> dict[str, Any]:
    """Least-squares TAN+SIP of one polynomial order on fixed stars."""

    height, width = shape
    crpix = ((width + 1) / 2.0, (height + 1) / 2.0)
    # FITS pixel offsets from CRPIX; zero-based pixel x is FITS x + 1.
    u = pixels[:, 0] + 1.0 - crpix[0]
    v = pixels[:, 1] + 1.0 - crpix[1]
    length = max(width, height) / 2.0
    terms = _terms(order)
    design = _design(u / length, v / length, terms)
    for _ in range(8):
        plane = _plane(crval)
        xi, eta = plane.wcs_world2pix(world[:, 0], world[:, 1], 1)
        a, *_ = np.linalg.lstsq(design, xi, rcond=None)
        b, *_ = np.linalg.lstsq(design, eta, rcond=None)
        shifted = plane.wcs_pix2world([a[0]], [b[0]], 1)
        crval = (float(shifted[0][0]) % 360.0, float(shifted[1][0]))
        if abs(a[0]) < 1e-11 and abs(b[0]) < 1e-11:
            break
    # Back to unnormalized pixel offsets.
    scale = np.asarray([length ** (p + q) for p, q in terms], dtype=np.float64)
    a = a / scale
    b = b / scale
    index = {term: position for position, term in enumerate(terms)}
    cd = np.asarray(((a[index[(1, 0)]], a[index[(0, 1)]]), (b[index[(1, 0)]], b[index[(0, 1)]])), dtype=np.float64)
    inverse_cd = np.linalg.inv(cd)
    sip_a: dict[tuple[int, int], float] = {}
    sip_b: dict[tuple[int, int], float] = {}
    for term in _terms(order, minimum=2):
        coefficient = inverse_cd @ np.asarray((a[index[term]], b[index[term]]))
        sip_a[term] = float(coefficient[0])
        sip_b[term] = float(coefficient[1])
    header = _sip_header(crval, crpix, cd, sip_a, sip_b, order, shape)
    wcs = celestial_wcs(header)
    predicted = np.column_stack(wcs.all_world2pix(world[:, 0], world[:, 1], 0, tolerance=1e-9, maxiter=100, quiet=True))
    residual = pixels - predicted
    return {"header": header, "residual": residual, "terms": len(terms), "wcs": wcs, "crval": crval}


def _clipped_fit(
    pixels: NDArray[np.float64],
    world: NDArray[np.float64],
    shape: tuple[int, int],
    order: int,
    crval: tuple[float, float],
) -> tuple[dict[str, Any], NDArray[np.bool_]]:
    keep = np.ones(len(pixels), dtype=bool)
    fit = _fit_order(pixels[keep], world[keep], shape, order, crval)
    for _ in range(_CLIP_ITERATIONS):
        wcs = fit["wcs"]
        predicted = np.column_stack(wcs.all_world2pix(world[:, 0], world[:, 1], 0, tolerance=1e-9, maxiter=100, quiet=True))
        radius = np.hypot(*(pixels - predicted).T)
        sigma = float(np.median(radius[keep])) / 1.1774 if keep.any() else 0.0
        updated = radius <= max(_CLIP_SIGMA * sigma, _CLIP_FLOOR_PIXELS)
        if np.array_equal(updated, keep) or np.count_nonzero(updated) < 0.7 * len(pixels):
            break
        keep = updated
        fit = _fit_order(pixels[keep], world[keep], shape, order, fit["crval"])
    return fit, keep


def fit_tan_sip(
    pixels: NDArray[np.float64],
    world: NDArray[np.float64],
    shape: tuple[int, int],
    *,
    crval: tuple[float, float],
    maximum_order: int = 3,
) -> tuple[dict[str, Any], int, NDArray[np.bool_]]:
    """Fit TAN+SIP to zero-based ``pixels`` of stars at ``world`` (RA, Dec
    in degrees); the order (1 = no SIP) is chosen by the BIC on the inliers
    of the most flexible admissible fit.  Returns (header, order, inliers)."""

    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    world = np.asarray(world, dtype=np.float64).reshape(-1, 2)
    admissible = [
        order
        for order in range(1, min(maximum_order, MAXIMUM_SIP_ORDER) + 1)
        if len(pixels) >= STARS_PER_TERM * len(_terms(order))
    ]
    if not admissible:
        raise MosaicAstrometryError(
            "MOSAIC_REFERENCE_STARS_INSUFFICIENT",
            f"{len(pixels)} catalog matches cannot constrain even a linear solution",
        )
    _, inliers = _clipped_fit(pixels, world, shape, admissible[-1], crval)
    best: tuple[float, int] | None = None
    observations = 2 * int(np.count_nonzero(inliers))
    for order in admissible:
        fit = _fit_order(pixels[inliers], world[inliers], shape, order, crval)
        sum_squares = float(np.sum(np.square(fit["residual"])))
        bic = observations * math.log(max(sum_squares, 1e-30) / observations) + 2 * fit["terms"] * math.log(observations)
        if best is None or bic < best[0]:
            best = (bic, order)
    assert best is not None
    order = best[1]
    fit, keep = _clipped_fit(pixels, world, shape, order, crval)
    return fit["header"], order, keep


def _holdout_rms_arcsec(
    pixels: NDArray[np.float64],
    world: NDArray[np.float64],
    shape: tuple[int, int],
    order: int,
    crval: tuple[float, float],
) -> float:
    """Two-fold cross-validated residual: each half of the stars against the
    solution of ``order`` fitted on the other half."""

    first = np.arange(len(pixels)) % 2 == 0
    residuals = []
    for train in (first, ~first):
        try:
            fit, _ = _clipped_fit(pixels[train], world[train], shape, order, crval)
        except (np.linalg.LinAlgError, ValueError):
            return math.inf
        residuals.append(_arcsec_residuals(fit["wcs"], pixels[~train], world[~train]))
    values = np.concatenate(residuals)
    return float(np.sqrt(np.mean(np.square(values)))) if np.all(np.isfinite(values)) else math.inf


def _arcsec_residuals(wcs: WCS, pixels: NDArray[np.float64], world: NDArray[np.float64]) -> NDArray[np.float64]:
    ra, dec = wcs.all_pix2world(pixels[:, 0], pixels[:, 1], 0)
    a = np.deg2rad(np.column_stack((ra, dec)))
    b = np.deg2rad(world)
    sin_dlat = np.sin((b[:, 1] - a[:, 1]) / 2.0)
    sin_dlon = np.sin((b[:, 0] - a[:, 0]) / 2.0)
    h = sin_dlat**2 + np.cos(a[:, 1]) * np.cos(b[:, 1]) * sin_dlon**2
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))) * 3600.0


def _isolated_windowed_centroids(
    image: NDArray[np.float32], points: NDArray[np.float64], fwhm: float
) -> NDArray[np.float64]:
    """Windowed centroids (SEP ``winpos`` on the background-subtracted
    frame) of the stars with no other detection within ``ISOLATION_FWHM``
    FWHM; a detection's isophotal barycentre is pulled by its neighbours."""

    import sep

    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64)
    finite = np.isfinite(image)
    work = np.ascontiguousarray(np.where(finite, image, np.nanmedian(image)), dtype=np.float32)
    background = sep.Background(work, mask=~finite, bw=64, bh=64)
    residual = np.ascontiguousarray(work - background.back(), dtype=np.float64)
    separation = np.sqrt(np.sum(np.square(points[:, None, :] - points[None, :, :]), axis=2))
    np.fill_diagonal(separation, np.inf)
    isolated = separation.min(axis=1) >= ISOLATION_FWHM * fwhm
    candidates = points[isolated]
    x, y, flags = sep.winpos(residual, candidates[:, 0], candidates[:, 1], max(fwhm / 2.3548, 0.6), mask=~finite)
    good = (np.asarray(flags) == 0) & np.isfinite(x) & np.isfinite(y) & (np.hypot(x - candidates[:, 0], y - candidates[:, 1]) < 1.0)
    return np.column_stack((x[good], y[good])).astype(np.float64)


@dataclass(frozen=True)
class PanelStars:
    """A frame measured for a solution: its isolated star centroids
    (zero-based pixels), the median FWHM, its catalog matches and its own
    catalog fit."""

    shape: tuple[int, int]
    fwhm: float
    detected: NDArray[np.float64]
    catalog_pixels: NDArray[np.float64]
    catalog_world: NDArray[np.float64]
    fit: SipFit


def measure_panel(
    image: NDArray[np.float32],
    header: Mapping[str, Any],
    catalog: CatalogStarSource,
    *,
    minimum_matches: int = 30,
    maximum_order: int = 3,
    minimum_catalog_stars: int = 0,
) -> PanelStars:
    """Detect the stars of ``image``, match them to catalog stars through a
    solver's solution and refine it into TAN+SIP; a field holding fewer than
    ``minimum_catalog_stars`` catalog stars is refused before any star
    detection."""

    from ..solvers.catalog_correspondence import (
        CorrespondenceParameters,
        detect_field_stars,
        match_mutual_nearest,
    )

    height, width = image.shape
    initial = celestial_wcs(header)
    centre = initial.all_pix2world([(width - 1) / 2.0], [(height - 1) / 2.0], 0)
    corner = initial.all_pix2world([0.0], [0.0], 0)
    radius = 1.05 * _arcsec_residuals(
        initial, np.asarray([[(width - 1) / 2.0, (height - 1) / 2.0]]), np.asarray([[float(corner[0][0]), float(corner[1][0])]])
    )[0] / 3600.0
    found = catalog.stars(float(centre[0][0]) % 360.0, float(centre[1][0]), float(radius))
    world = np.column_stack((found.ra_degrees, found.dec_degrees))
    predicted = np.column_stack(initial.all_world2pix(world[:, 0], world[:, 1], 0, quiet=True))
    inside = (
        np.all(np.isfinite(predicted), axis=1)
        & (predicted[:, 0] >= -0.5)
        & (predicted[:, 0] <= width - 0.5)
        & (predicted[:, 1] >= -0.5)
        & (predicted[:, 1] <= height - 0.5)
    )
    world = world[inside]
    predicted = predicted[inside]
    if len(world) < minimum_catalog_stars:
        raise MosaicAstrometryError(
            "MOSAIC_CATALOG_TOO_SPARSE",
            f"{len(world)} catalog stars in the field; a refit needs at least {minimum_catalog_stars}",
        )
    detected_one_based, fwhm, detection = detect_field_stars(
        np.asarray(image, dtype=np.float32), CorrespondenceParameters(maximum_stars=4000)
    )
    detected = _isolated_windowed_centroids(
        np.asarray(image, dtype=np.float32),
        np.asarray(detected_one_based, dtype=np.float64) - 1.0,
        float(np.median(fwhm)) if len(fwhm) else 3.0,
    )
    median_fwhm = float(np.median(fwhm)) if len(fwhm) else 0.0
    first_radius = max(FIRST_MATCH_RADIUS_PIXELS, FIRST_MATCH_RADIUS_FWHM * median_fwhm)
    rows, catalog_rows, _ = match_mutual_nearest(detected, predicted, first_radius)
    if rows.size < minimum_matches:
        raise MosaicAstrometryError(
            "MOSAIC_REFERENCE_STARS_INSUFFICIENT",
            f"{rows.size} catalog matches on the reference frame; at least {minimum_matches} are needed",
        )
    crval = (float(centre[0][0]) % 360.0, float(centre[1][0]))
    fitted, order, keep = fit_tan_sip(detected[rows], world[catalog_rows], (height, width), crval=crval, maximum_order=maximum_order)
    # Match again through the fitted solution with a radius its residuals justify.
    wcs = celestial_wcs(fitted)
    first_radius_rms = float(np.sqrt(np.mean(np.sum(np.square(
        detected[rows][keep] - np.column_stack(wcs.all_world2pix(world[catalog_rows][keep, 0], world[catalog_rows][keep, 1], 0, quiet=True))
    ), axis=1))))
    predicted = np.column_stack(wcs.all_world2pix(world[:, 0], world[:, 1], 0, tolerance=1e-9, maxiter=100, quiet=True))
    second_radius = max(1.5, 4.0 * first_radius_rms)
    rows, catalog_rows, _ = match_mutual_nearest(detected, predicted, second_radius)
    if rows.size < minimum_matches:
        raise MosaicAstrometryError(
            "MOSAIC_REFERENCE_STARS_INSUFFICIENT",
            f"{rows.size} catalog matches through the fitted solution; at least {minimum_matches} are needed",
        )
    fitted, order, keep = fit_tan_sip(
        detected[rows], world[catalog_rows], (height, width), crval=(float(fitted["CRVAL1"]), float(fitted["CRVAL2"])), maximum_order=maximum_order
    )
    matched_pixels = detected[rows][keep]
    matched_world = world[catalog_rows][keep]
    wcs = celestial_wcs(fitted)
    residual = matched_pixels - np.column_stack(
        wcs.all_world2pix(matched_world[:, 0], matched_world[:, 1], 0, tolerance=1e-9, maxiter=100, quiet=True)
    )
    radius_pixels = np.hypot(residual[:, 0], residual[:, 1])
    arcsec = _arcsec_residuals(wcs, matched_pixels, matched_world)
    # The solver's own solution on the same stars, for the comparison.
    initial_arcsec = _arcsec_residuals(initial, matched_pixels, matched_world)
    holdout_arcsec = _holdout_rms_arcsec(
        matched_pixels, matched_world, (height, width), order, (float(fitted["CRVAL1"]), float(fitted["CRVAL2"]))
    )
    # Residuals of the frame's outer ring (where mosaic overlaps lie) against its middle.
    edge = np.minimum.reduce(
        (matched_pixels[:, 0], width - 1 - matched_pixels[:, 0], matched_pixels[:, 1], height - 1 - matched_pixels[:, 1])
    ) < 0.15 * min(width, height)
    fit = SipFit(
        header=fitted,
        order=order,
        matched=int(len(matched_pixels)),
        rms_pixels=float(np.sqrt(np.mean(np.square(radius_pixels)))),
        rms_arcsec=float(np.sqrt(np.mean(np.square(arcsec)))),
        max_residual_pixels=float(radius_pixels.max()),
        evidence={
            "catalogIdentity": found.identity,
            "detectedStars": int(detection.get("detectedStars", len(detected))),
            "catalogStarsInField": int(len(world)),
            "matchRadiusPixels": [first_radius, second_radius],
            "clippedMatches": int(rows.size - keep.sum()),
            "edgeRmsPixels": float(np.sqrt(np.mean(np.square(radius_pixels[edge])))) if edge.any() else None,
            "centreRmsPixels": float(np.sqrt(np.mean(np.square(radius_pixels[~edge])))) if (~edge).any() else None,
            "edgeMatches": int(edge.sum()),
            "initialRmsArcsec": float(np.sqrt(np.mean(np.square(initial_arcsec)))),
            "holdoutRmsArcsec": holdout_arcsec,
        },
    )
    return PanelStars((height, width), median_fwhm, detected, matched_pixels, matched_world, fit)


def refine_solution(
    image: NDArray[np.float32],
    header: Mapping[str, Any],
    catalog: CatalogStarSource,
    *,
    minimum_matches: int = 30,
    maximum_order: int = 3,
    minimum_catalog_stars: int = 0,
) -> SipFit:
    """A solver's solution of ``image`` refined into TAN+SIP on catalog stars
    (see :func:`measure_panel`)."""

    return measure_panel(
        image,
        header,
        catalog,
        minimum_matches=minimum_matches,
        maximum_order=maximum_order,
        minimum_catalog_stars=minimum_catalog_stars,
    ).fit


def _tangent_basis(crval: tuple[float, float]) -> NDArray[np.float64]:
    """Rows: the east and north unit vectors of the tangent plane at
    ``crval`` and the unit vector of ``crval`` itself."""

    alpha, delta = math.radians(crval[0]), math.radians(crval[1])
    return np.asarray(
        (
            (-math.sin(alpha), math.cos(alpha), 0.0),
            (-math.sin(delta) * math.cos(alpha), -math.sin(delta) * math.sin(alpha), math.cos(delta)),
            (math.cos(delta) * math.cos(alpha), math.cos(delta) * math.sin(alpha), math.sin(delta)),
        ),
        dtype=np.float64,
    )


def _directions(world: NDArray[np.float64]) -> NDArray[np.float64]:
    alpha, delta = np.radians(world[:, 0]), np.radians(world[:, 1])
    return np.column_stack((np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)))


def _in_plane(basis: NDArray[np.float64], directions: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    """Gnomonic coordinates of ``directions`` in units of ``scale`` radians."""

    local = directions @ basis.T
    return local[:, :2] / local[:, 2:3] / scale


def _transfer(matrix: NDArray[np.float64], z: NDArray[np.float64], scale: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Points of one tangent plane in another (``matrix`` = target basis @
    source basis.T, a projective map) and its Jacobian per point."""

    h = np.column_stack((z * scale, np.ones(len(z)))) @ matrix.T
    a, b, c = h[:, 0], h[:, 1], h[:, 2]
    jacobian = np.empty((len(z), 2, 2))
    square = c * c
    jacobian[:, 0, 0] = (matrix[0, 0] * c - a * matrix[2, 0]) / square
    jacobian[:, 0, 1] = (matrix[0, 1] * c - a * matrix[2, 1]) / square
    jacobian[:, 1, 0] = (matrix[1, 0] * c - b * matrix[2, 0]) / square
    jacobian[:, 1, 1] = (matrix[1, 1] * c - b * matrix[2, 1]) / square
    return np.column_stack((a / c, b / c)) / scale, jacobian


@dataclass(frozen=True)
class _JointModel:
    """Per panel a CD matrix and an offset in its tangent plane; the SIP
    distortion of the given order shared by all panels or one per panel.
    Parameters: six per panel, then the distortion coefficients."""

    order: int
    shared: bool
    panels: int

    @property
    def terms(self) -> list[tuple[int, int]]:
        return _terms(self.order, minimum=2)

    @property
    def size(self) -> int:
        return 6 * self.panels + 2 * len(self.terms) * (1 if self.shared else self.panels)

    def local_size(self) -> int:
        """The parameters a panel owns alone."""

        return 6 + (0 if self.shared else 2 * len(self.terms))

    def indices(self, panel: int) -> NDArray[np.int64]:
        count = 2 * len(self.terms)
        base = 6 * self.panels + (0 if self.shared else count * panel)
        return np.concatenate((np.arange(6 * panel, 6 * panel + 6), np.arange(base, base + count)))

    def serializable(self) -> dict[str, Any]:
        return {"order": self.order, "distortion": "shared" if self.shared or self.order < 2 else "perPanel"}


def _panel_model(
    local: NDArray[np.float64], q: NDArray[np.float64], length: float, terms: Sequence[tuple[int, int]]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A panel's tangent-plane points at pixel offsets ``q`` and their
    derivatives by the panel's parameters (CD, offset, distortion)."""

    count = len(terms)
    cd = local[:4].reshape(2, 2)
    phi = _design(q[:, 0] / length, q[:, 1] / length, terms) * length if count else np.zeros((len(q), 0))
    distorted = q + np.column_stack((phi @ local[6 : 6 + count], phi @ local[6 + count :])) if count else q
    z = distorted @ cd.T + local[4:6]
    jacobian = np.zeros((len(q), 2, 6 + 2 * count))
    jacobian[:, 0, 0] = distorted[:, 0]
    jacobian[:, 0, 1] = distorted[:, 1]
    jacobian[:, 1, 2] = distorted[:, 0]
    jacobian[:, 1, 3] = distorted[:, 1]
    jacobian[:, 0, 4] = 1.0
    jacobian[:, 1, 5] = 1.0
    if count:
        jacobian[:, 0, 6 : 6 + count] = cd[0, 0] * phi
        jacobian[:, 1, 6 : 6 + count] = cd[1, 0] * phi
        jacobian[:, 0, 6 + count :] = cd[0, 1] * phi
        jacobian[:, 1, 6 + count :] = cd[1, 1] * phi
    return z, jacobian


@dataclass
class _Overlap:
    """Stars two panels both measured: pixel offsets in each."""

    first: int
    second: int
    q_first: NDArray[np.float64]
    q_second: NDArray[np.float64]
    keep: NDArray[np.bool_]
    sigma: float = OVERLAP_SIGMA_PIXELS


@dataclass
class _Catalog:
    q: NDArray[np.float64]
    directions: NDArray[np.float64]
    keep: NDArray[np.bool_]
    sigma: float


class _JointProblem:
    """The panels' stars, catalog matches and overlaps, and the state of a fit."""

    def __init__(self, panels: Sequence[PanelStars]) -> None:
        self.panels = list(panels)
        self.crpix = [((stars.shape[1] + 1) / 2.0, (stars.shape[0] + 1) / 2.0) for stars in self.panels]
        self.length = [max(stars.shape) / 2.0 for stars in self.panels]
        scales = [math.sqrt(abs(np.linalg.det(_cd(stars.fit.header)))) for stars in self.panels]
        # One unit for every tangent plane: a panel pixel, radians.
        self.scale = math.radians(float(np.median(scales)))
        self.catalogs = [
            _Catalog(
                self.offsets(index, stars.catalog_pixels),
                _directions(stars.catalog_world),
                np.ones(len(stars.catalog_pixels), dtype=bool),
                max(stars.fit.rms_pixels / math.sqrt(2.0), 0.05),
            )
            for index, stars in enumerate(self.panels)
        ]
        self.overlaps: list[_Overlap] = []

    def offsets(self, panel: int, pixels: NDArray[np.float64]) -> NDArray[np.float64]:
        crpix = self.crpix[panel]
        return np.column_stack((pixels[:, 0] + 1.0 - crpix[0], pixels[:, 1] + 1.0 - crpix[1]))

    def start(self, model: _JointModel) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """Parameters from the panels' own catalog fits (a shared distortion
        starts at their mean) and their tangent points."""

        theta = np.zeros(model.size)
        distortions = []
        for index, stars in enumerate(self.panels):
            header = stars.fit.header
            indices = model.indices(index)
            theta[indices[:4]] = (np.radians(_cd(header)) / self.scale).ravel()
            length = self.length[index]
            coefficients = [
                float(header.get(f"{axis}_{p}_{q}", 0.0)) * length ** (p + q - 1)
                for axis in ("A", "B")
                for p, q in model.terms
            ]
            distortions.append(coefficients)
            if not model.shared:
                theta[indices[6:]] = coefficients
        if model.shared and model.terms:
            weights = np.asarray([len(stars.catalog_pixels) for stars in self.panels], dtype=np.float64)
            theta[6 * model.panels :] = np.average(np.asarray(distortions), axis=0, weights=weights)
        centres = [(float(stars.fit.header["CRVAL1"]), float(stars.fit.header["CRVAL2"])) for stars in self.panels]
        return theta, centres

    def residuals(
        self, model: _JointModel, theta: NDArray[np.float64], centres: Sequence[tuple[float, float]]
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        """Residual vectors (plane pixels) of every catalog match and overlap star, kept or not."""

        bases = [_tangent_basis(centre) for centre in centres]
        catalog = []
        for index, item in enumerate(self.catalogs):
            z, _ = _panel_model(theta[model.indices(index)], item.q, self.length[index], model.terms)
            catalog.append(_in_plane(bases[index], item.directions, self.scale) - z)
        overlap = []
        for item in self.overlaps:
            z_first, _ = _panel_model(theta[model.indices(item.first)], item.q_first, self.length[item.first], model.terms)
            z_second, _ = _panel_model(theta[model.indices(item.second)], item.q_second, self.length[item.second], model.terms)
            moved, _ = _transfer(bases[item.first] @ bases[item.second].T, z_second, self.scale)
            overlap.append(moved - z_first)
        return catalog, overlap

    def _normal_equations(
        self, model: _JointModel, theta: NDArray[np.float64], centres: Sequence[tuple[float, float]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        bases = [_tangent_basis(centre) for centre in centres]
        normal = np.zeros((model.size, model.size))
        gradient = np.zeros(model.size)
        chi_square = 0.0
        for index, item in enumerate(self.catalogs):
            indices = model.indices(index)
            z, jacobian = _panel_model(theta[indices], item.q[item.keep], self.length[index], model.terms)
            residual = (_in_plane(bases[index], item.directions[item.keep], self.scale) - z).ravel()
            design = jacobian.reshape(-1, len(indices))
            weight = 1.0 / item.sigma**2
            normal[np.ix_(indices, indices)] += weight * (design.T @ design)
            gradient[indices] += weight * (design.T @ residual)
            chi_square += weight * float(residual @ residual)
        for item in self.overlaps:
            first, second = model.indices(item.first), model.indices(item.second)
            z_first, jacobian_first = _panel_model(theta[first], item.q_first[item.keep], self.length[item.first], model.terms)
            z_second, jacobian_second = _panel_model(theta[second], item.q_second[item.keep], self.length[item.second], model.terms)
            moved, transfer = _transfer(bases[item.first] @ bases[item.second].T, z_second, self.scale)
            residual = (moved - z_first).ravel()
            design_first = jacobian_first.reshape(-1, len(first))
            design_second = -np.einsum("nij,njk->nik", transfer, jacobian_second).reshape(-1, len(second))
            weight = 1.0 / item.sigma**2
            normal[np.ix_(first, first)] += weight * (design_first.T @ design_first)
            normal[np.ix_(second, second)] += weight * (design_second.T @ design_second)
            normal[np.ix_(first, second)] += weight * (design_first.T @ design_second)
            normal[np.ix_(second, first)] += weight * (design_second.T @ design_first)
            gradient[first] += weight * (design_first.T @ residual)
            gradient[second] += weight * (design_second.T @ residual)
            chi_square += weight * float(residual @ residual)
        return normal, gradient, chi_square

    def solve(
        self, model: _JointModel, theta: NDArray[np.float64], centres: Sequence[tuple[float, float]]
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]], float]:
        """Gauss-Newton on the kept stars; every panel's tangent point moves
        to its fitted centre after each step, so its offset ends at zero
        and the solution is a plain TAN+SIP header."""

        theta = theta.copy()
        centres = list(centres)
        chi_square = math.inf
        for _ in range(_JOINT_ITERATIONS):
            normal, gradient, chi_square = self._normal_equations(model, theta, centres)
            diagonal = np.sqrt(np.diag(normal))
            if np.any(diagonal <= 0.0):
                raise MosaicAstrometryError("MOSAIC_PANELS_UNCONSTRAINED", "a mosaic panel has no stars to place it")
            scaled = normal / np.outer(diagonal, diagonal)
            eigenvalues = np.linalg.eigvalsh(scaled)
            if eigenvalues[0] < 1e-12 * eigenvalues[-1]:
                raise MosaicAstrometryError(
                    "MOSAIC_PANELS_UNCONSTRAINED",
                    "the catalog and overlap stars do not determine every panel's solution",
                )
            step = np.linalg.solve(scaled, gradient / diagonal) / diagonal
            theta += step
            moved = 0.0
            for index in range(model.panels):
                indices = model.indices(index)
                corners = np.asarray(
                    [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
                ) * np.asarray(self.panels[index].shape[::-1]) / 2.0
                _, jacobian = _panel_model(theta[indices], corners, self.length[index], model.terms)
                moved = max(moved, float(np.max(np.abs(jacobian @ step[indices]))))
                offset = theta[indices[4:6]]
                direction = _tangent_basis(centres[index]).T @ np.asarray((offset[0] * self.scale, offset[1] * self.scale, 1.0))
                direction /= np.linalg.norm(direction)
                centres[index] = (
                    math.degrees(math.atan2(direction[1], direction[0])) % 360.0,
                    math.degrees(math.asin(max(-1.0, min(1.0, direction[2])))),
                )
                theta[indices[4:6]] = 0.0
            if moved < 1e-7:
                break
        else:
            if moved > 1e-3:
                raise MosaicAstrometryError("MOSAIC_JOINT_FIT_DIVERGED", "the joint astrometric fit did not converge")
        return theta, centres, chi_square

    def clipped(
        self, model: _JointModel, theta: NDArray[np.float64], centres: Sequence[tuple[float, float]]
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """A fit with outliers clipped and each class's error re-estimated:
        per panel for its catalog matches, per overlap for its stars."""

        for _ in range(_CLIP_ITERATIONS + 1):
            theta, centres, _ = self.solve(model, theta, centres)
            catalog, overlap = self.residuals(model, theta, centres)
            changed = False
            for item, residual in [*zip(self.catalogs, catalog), *zip(self.overlaps, overlap)]:
                radius = np.hypot(residual[:, 0], residual[:, 1])
                floor = 0.05 if isinstance(item, _Catalog) else 0.02
                item.sigma = max(float(np.median(radius[item.keep])) / 1.1774, floor) if item.keep.any() else item.sigma
                updated = radius <= max(_CLIP_SIGMA * item.sigma, _CLIP_FLOOR_PIXELS)
                if np.count_nonzero(updated) < 0.7 * len(radius):
                    # More outliers than one clip may take: keep the nearest 70 %.
                    updated = radius <= np.quantile(radius, 0.7)
                if not np.array_equal(updated, item.keep):
                    item.keep = updated
                    changed = True
            if not changed:
                break
        return theta, centres

    def admissible(self, model: _JointModel) -> bool:
        """Three stars per term in all and for every panel's own
        parameters, and enough catalog stars to hold the mosaic on the sky."""

        catalog = [int(item.keep.sum()) for item in self.catalogs]
        local = list(catalog)
        for item in self.overlaps:
            local[item.first] += int(item.keep.sum())
            local[item.second] += int(item.keep.sum())
        stars = sum(catalog) + sum(int(item.keep.sum()) for item in self.overlaps)
        return (
            2 * stars >= STARS_PER_TERM * model.size
            and all(2 * count >= STARS_PER_TERM * model.local_size() for count in local)
            and sum(catalog) >= STARS_PER_TERM * 3
        )

    def header(self, model: _JointModel, theta: NDArray[np.float64], centre: tuple[float, float], panel: int) -> dict[str, Any]:
        local = theta[model.indices(panel)]
        cd = np.degrees(local[:4].reshape(2, 2) * self.scale)
        length = self.length[panel]
        count = len(model.terms)
        sip_a = {term: float(local[6 + position]) * length ** (1 - sum(term)) for position, term in enumerate(model.terms)}
        sip_b = {term: float(local[6 + count + position]) * length ** (1 - sum(term)) for position, term in enumerate(model.terms)}
        return _sip_header(centre, self.crpix[panel], cd, sip_a, sip_b, model.order, self.panels[panel].shape)

    def match_overlaps(self, headers: Sequence[Mapping[str, Any]], radii: Sequence[Sequence[float]]) -> None:
        """Stars of each pair of panels that are one star on the sky through
        ``headers``, within ``radii[first][second]`` pixels."""

        from ..solvers.catalog_correspondence import match_mutual_nearest

        wcs = [celestial_wcs(header) for header in headers]
        self.overlaps = []
        for first in range(len(self.panels)):
            sky = wcs[first].all_pix2world(self.panels[first].detected, 0)
            for second in range(first + 1, len(self.panels)):
                height, width = self.panels[second].shape
                projected = _project(wcs[second], sky)
                inside = np.flatnonzero(
                    np.all(np.isfinite(projected), axis=1)
                    & (projected[:, 0] >= 0.0)
                    & (projected[:, 0] <= width - 1.0)
                    & (projected[:, 1] >= 0.0)
                    & (projected[:, 1] <= height - 1.0)
                )
                if inside.size < MINIMUM_OVERLAP_MATCHES:
                    continue
                rows, second_rows, _ = match_mutual_nearest(projected[inside], self.panels[second].detected, radii[first][second])
                if rows.size < MINIMUM_OVERLAP_MATCHES:
                    continue
                self.overlaps.append(
                    _Overlap(
                        first,
                        second,
                        self.offsets(first, self.panels[first].detected[inside[rows]]),
                        self.offsets(second, self.panels[second].detected[second_rows]),
                        np.ones(rows.size, dtype=bool),
                    )
                )


def _cd(header: Mapping[str, Any]) -> NDArray[np.float64]:
    return np.asarray(((header["CD1_1"], header["CD1_2"]), (header["CD2_1"], header["CD2_2"])), dtype=np.float64)


def _project(wcs: WCS, world: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack(wcs.all_world2pix(world[:, 0], world[:, 1], 0, tolerance=1e-9, maxiter=100, quiet=True))


def _clip_flexible(
    problem: _JointProblem, admissible: Sequence[_JointModel]
) -> tuple[_JointModel, NDArray[np.float64], list[tuple[float, float]]]:
    """The most flexible admissible model the stars determine, fitted with
    its outliers clipped; a model they cannot determine gives way to the next."""

    items = [*problem.catalogs, *problem.overlaps]
    for model in sorted(admissible, key=lambda item: item.size, reverse=True):
        state = [(item.keep.copy(), item.sigma) for item in items]
        try:
            return (model, *problem.clipped(model, *problem.start(model)))
        except MosaicAstrometryError:
            for item, (keep, sigma) in zip(items, state):
                item.keep, item.sigma = keep, sigma
    raise MosaicAstrometryError(
        "MOSAIC_PANELS_UNCONSTRAINED", "the catalog and overlap stars do not determine every panel's solution"
    )


def solve_mosaic(panels: Mapping[str, PanelStars]) -> tuple[dict[str, SipFit], dict[str, Any]]:
    """One TAN+SIP solution per panel, fitted on all panels at once.

    A panel's catalog matches place it on the sky; the stars it shares with
    a neighbour in their overlap tie the two solutions together, so the
    overlaps agree and no panel's sparse-catalog error passes to another
    unchecked.  The distortion is one SIP polynomial for all panels (the
    same optics) or one per panel, whichever, with its order, the BIC
    prefers on the inliers of the most flexible admissible model.  Returns
    the solution per panel and the record of the fit."""

    keys = list(panels)
    problem = _JointProblem([panels[key] for key in keys])
    count = len(keys)
    fwhm = [max(panels[key].fwhm, 1.0) for key in keys]
    problem.match_overlaps(
        [panels[key].fit.header for key in keys],
        [[max(2.0, OVERLAP_MATCH_FWHM * max(fwhm[first], fwhm[second])) for second in range(count)] for first in range(count)],
    )
    candidates = [_JointModel(1, True, count)] + [
        _JointModel(order, shared, count) for order in (2, 3) for shared in ((True, False) if count > 1 else (True,))
    ]
    admissible = [model for model in candidates if problem.admissible(model)]
    if not admissible:
        raise MosaicAstrometryError(
            "MOSAIC_REFERENCE_STARS_INSUFFICIENT", "the panels' catalog and overlap stars cannot constrain even linear solutions"
        )
    flexible, theta, centres = _clip_flexible(problem, admissible)
    # Again through the joint solution, within what its residuals justify,
    # until no overlap gains stars: a pair the panels' own fits put too far
    # apart links once its neighbours have pulled them into place.
    linked = sum(int(item.keep.sum()) for item in problem.overlaps)
    for _ in range(_REMATCH_ROUNDS):
        headers = [problem.header(flexible, theta, centres[index], index) for index in range(count)]
        sigma = max([item.sigma for item in problem.overlaps] or [OVERLAP_SIGMA_PIXELS])
        problem.match_overlaps(headers, [[max(1.0, 5.0 * sigma)] * count for _ in range(count)])
        admissible = [model for model in admissible if problem.admissible(model)]
        flexible, theta, centres = _clip_flexible(problem, admissible)
        now = sum(int(item.keep.sum()) for item in problem.overlaps)
        if now <= linked:
            break
        linked = now
    observations = 2 * (
        sum(int(item.keep.sum()) for item in problem.catalogs) + sum(int(item.keep.sum()) for item in problem.overlaps)
    )
    best: tuple[float, _JointModel, NDArray[np.float64], list[tuple[float, float]]] | None = None
    table = []
    for model in admissible:
        try:
            fitted, fitted_centres, chi_square = problem.solve(model, *problem.start(model))
        except MosaicAstrometryError as error:
            table.append({**model.serializable(), "parameters": model.size, "failed": error.code})
            continue
        bic = chi_square + model.size * math.log(observations)
        table.append({**model.serializable(), "parameters": model.size, "bic": bic})
        if best is None or bic < best[0]:
            best = (bic, model, fitted, fitted_centres)
    if best is None:
        raise MosaicAstrometryError(
            "MOSAIC_PANELS_UNCONSTRAINED", "no candidate model determines every panel's solution"
        )
    _, model, theta, centres = best
    theta, centres = problem.clipped(model, theta, centres)
    catalog_residuals, overlap_residuals = problem.residuals(model, theta, centres)
    overlaps = []
    for item, residual in zip(problem.overlaps, overlap_residuals):
        radius = np.hypot(residual[:, 0], residual[:, 1])[item.keep]
        overlaps.append(
            {
                "panels": [keys[item.first], keys[item.second]],
                "matches": int(item.keep.size),
                "inliers": int(item.keep.sum()),
                "rmsPixels": float(np.sqrt(np.mean(np.square(radius)))) if radius.size else None,
                "sigmaPixels": item.sigma,
            }
        )
    record = {
        "algorithm": JOINT_FIT_ALGORITHM,
        **model.serializable(),
        "panels": count,
        "catalogStars": sum(int(item.keep.sum()) for item in problem.catalogs),
        "overlapStars": sum(int(item.keep.sum()) for item in problem.overlaps),
        "candidates": table,
        "overlaps": overlaps,
    }
    solutions = {}
    for index, key in enumerate(keys):
        stars = panels[key]
        item = problem.catalogs[index]
        header = problem.header(model, theta, centres[index], index)
        wcs = celestial_wcs(header)
        pixels, world = stars.catalog_pixels[item.keep], stars.catalog_world[item.keep]
        residual = pixels - _project(wcs, world)
        radius_pixels = np.hypot(residual[:, 0], residual[:, 1])
        arcsec = _arcsec_residuals(wcs, pixels, world)
        height, width = stars.shape
        edge = np.minimum.reduce((pixels[:, 0], width - 1 - pixels[:, 0], pixels[:, 1], height - 1 - pixels[:, 1])) < 0.15 * min(width, height)
        shared_stars = sum(int(overlap.keep.sum()) for overlap in problem.overlaps if index in (overlap.first, overlap.second))
        evidence = {
            key_: value
            for key_, value in stars.fit.evidence.items()
            if key_ in ("catalogIdentity", "detectedStars", "catalogStarsInField", "matchRadiusPixels", "initialRmsArcsec")
        }
        solutions[key] = SipFit(
            header=header,
            order=model.order,
            matched=int(pixels.shape[0]),
            rms_pixels=float(np.sqrt(np.mean(np.square(radius_pixels)))) if radius_pixels.size else 0.0,
            rms_arcsec=float(np.sqrt(np.mean(np.square(arcsec)))) if arcsec.size else 0.0,
            max_residual_pixels=float(radius_pixels.max()) if radius_pixels.size else 0.0,
            evidence={
                **evidence,
                "algorithm": JOINT_FIT_ALGORITHM,
                "clippedMatches": int(item.keep.size - item.keep.sum()),
                "catalogSigmaPixels": item.sigma,
                "edgeRmsPixels": float(np.sqrt(np.mean(np.square(radius_pixels[edge])))) if edge.any() else None,
                "centreRmsPixels": float(np.sqrt(np.mean(np.square(radius_pixels[~edge])))) if (~edge).any() else None,
                "edgeMatches": int(edge.sum()),
                "overlapStars": shared_stars,
                "ownFit": {
                    "order": stars.fit.order,
                    "matchedStars": stars.fit.matched,
                    "rmsPixels": stars.fit.rms_pixels,
                    "rmsArcsec": stars.fit.rms_arcsec,
                },
            },
        )
    return solutions, record


LATERAL_COLOUR_ALGORITHM = "registration-residual-lateral-colour-v1"
FRAME_FIELD_ALGORITHM = "registration-residual-frame-field-v1"
# A shift field is fitted on at least this many star pairs.
_FIELD_MINIMUM_MATCHES = 50
# An order is admissible only where its stars determine it: the standard
# error of its predicted shift stays within this at every node where the
# field is applied (a cloud can leave a corner without stars).
_FIELD_MAXIMUM_STANDARD_ERROR_PIXELS = 0.1
_LATERAL_MATCH_RADIUS_PIXELS = 1.5


def frame_stars(image: NDArray[np.float32]) -> NDArray[np.float64]:
    """Zero-based windowed centroids of a frame's isolated stars."""

    from ..solvers.catalog_correspondence import CorrespondenceParameters, detect_field_stars

    image = np.asarray(image, dtype=np.float32)
    detected, fwhm, _ = detect_field_stars(image, CorrespondenceParameters(maximum_stars=4000))
    return _isolated_windowed_centroids(image, np.asarray(detected, dtype=np.float64) - 1.0, float(np.median(fwhm)) if len(fwhm) else 3.0)


def fit_shift_field(
    points: NDArray[np.float64],
    shifts: NDArray[np.float64],
    shape: tuple[int, int],
    nodes: NDArray[np.float64],
    *,
    algorithm: str = FRAME_FIELD_ALGORITHM,
) -> ShiftField:
    """The shift field ``shifts`` measured at reference pixels ``points`` of
    a ``shape`` frame and applied at reference pixels ``nodes``: polynomial
    of the order (none, 0 to 3) the BIC chooses, on the inliers of the most
    flexible fit, among the orders the stars determine at every node."""

    height, width = shape
    centre, length = ((width - 1) / 2.0, (height - 1) / 2.0), max(width, height) / 2.0
    evidence: dict[str, Any] = {"algorithm": algorithm, "matches": int(len(points))}
    if len(points) < _FIELD_MINIMUM_MATCHES:
        return ShiftField(None, (), (), centre, length, {**evidence, "applied": False})
    u, v = (points[:, 0] - centre[0]) / length, (points[:, 1] - centre[1]) / length
    node_x, node_y = np.asarray(nodes, dtype=np.float64).T

    def fit(order: int, keep: NDArray[np.bool_]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        design = ShiftField.design(u, v, order)
        cx, *_ = np.linalg.lstsq(design[keep], shifts[keep, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(design[keep], shifts[keep, 1], rcond=None)
        return cx, cy, shifts - np.column_stack((design @ cx, design @ cy))

    keep = np.ones(len(points), dtype=bool)
    for _ in range(3):
        *_, residual = fit(SHIFT_FIELD_MAXIMUM_ORDER, keep)
        radius = np.hypot(residual[:, 0], residual[:, 1])
        sigma = 1.4826 * float(np.median(radius[keep])) + 1e-6
        updated = radius <= 3.5 * sigma
        if np.array_equal(updated, keep) or updated.sum() < _FIELD_MINIMUM_MATCHES:
            break
        keep = updated
    observations = 2 * int(keep.sum())
    best = (observations * math.log(max(float(np.sum(np.square(shifts[keep]))), 1e-30) / observations), None, (), (), 0.0)
    for order in range(SHIFT_FIELD_MAXIMUM_ORDER + 1):
        cx, cy, residual = fit(order, keep)
        squares = max(float(np.sum(np.square(residual[keep]))), 1e-30)
        bic = observations * math.log(squares / observations) + 2 * len(cx) * math.log(observations)
        if bic >= best[0]:
            continue
        design = ShiftField.design(u[keep], v[keep], order)
        at_nodes = ShiftField.design((node_x - centre[0]) / length, (node_y - centre[1]) / length, order)
        leverage = np.einsum("ij,jk,ik->i", at_nodes, np.linalg.pinv(design.T @ design), at_nodes)
        standard_error = math.sqrt(squares / max(observations - 2 * len(cx), 1) * float(leverage.max()))
        if standard_error <= _FIELD_MAXIMUM_STANDARD_ERROR_PIXELS:
            best = (bic, order, tuple(float(value) for value in cx), tuple(float(value) for value in cy), standard_error)
    _, order, cx, cy, standard_error = best
    field_ = ShiftField(order, cx, cy, centre, length)
    sx, sy = field_.shift(node_x, node_y)
    after = shifts[keep] - np.column_stack(field_.shift(points[keep, 0], points[keep, 1]))
    evidence.update(
        applied=order is not None,
        order=order,
        inliers=int(keep.sum()),
        rmsBeforePixels=float(np.sqrt(np.mean(np.sum(np.square(shifts[keep]), axis=1)))),
        rmsAfterPixels=float(np.sqrt(np.mean(np.sum(np.square(after), axis=1)))),
        maxShiftPixels=float(np.max(np.hypot(sx, sy))),
        standardErrorPixels=standard_error,
    )
    return ShiftField(order, cx, cy, centre, length, evidence)


def _mapped(transform: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    mapped = np.column_stack((points, np.ones(len(points)))) @ np.asarray(transform, dtype=np.float64).T
    return mapped[:, :2] / mapped[:, 2:3]


def measure_lateral_colour(
    reference_stars: NDArray[np.float64],
    image: NDArray[np.float32],
    transform: NDArray[np.float64],
    shape: tuple[int, int],
) -> ShiftField:
    """The shift field of ``image`` (a Light of another filter, registered to
    the reference by ``transform``, a 3x3 frame-to-reference matrix) against
    the reference's ``reference_stars``: its filter's lateral colour."""

    from ..solvers.catalog_correspondence import match_mutual_nearest

    mapped = _mapped(transform, frame_stars(image))
    rows, reference_rows, _ = match_mutual_nearest(mapped, reference_stars, _LATERAL_MATCH_RADIUS_PIXELS)
    points = reference_stars[reference_rows]
    # One lattice per filter holds over the whole reference frame.
    height, width = shape
    grid = np.stack(np.meshgrid(np.linspace(0, width - 1, 17), np.linspace(0, height - 1, 13)), axis=-1).reshape(-1, 2)
    return fit_shift_field(points, mapped[rows] - points, shape, grid, algorithm=LATERAL_COLOUR_ALGORITHM)


def measure_frame_field(
    pairs: tuple[NDArray[np.float64], NDArray[np.float64]],
    transform: NDArray[np.float64],
    shape: tuple[int, int],
) -> ShiftField:
    """A registered Light's own shift field (a meridian flip, another night,
    its filter's lateral colour) from its registration's inlier centroid
    ``pairs`` (source, reference) and its 3x3 matrix ``transform``; it must
    hold on a 17x13 grid over the Light's frame, mapped into the reference."""

    source, reference = (np.asarray(item, dtype=np.float64) for item in pairs)
    height, width = shape
    grid = np.stack(np.meshgrid(np.linspace(0, width - 1, 17), np.linspace(0, height - 1, 13)), axis=-1).reshape(-1, 2)
    return fit_shift_field(reference, _mapped(transform, source) - reference, shape, _mapped(transform, grid))


__all__ = [
    "FRAME_FIELD_ALGORITHM",
    "JOINT_FIT_ALGORITHM",
    "LATERAL_COLOUR_ALGORITHM",
    "CatalogStarSource",
    "CatalogStars",
    "ManagedCatalogStars",
    "MosaicAstrometryError",
    "PanelStars",
    "SIP_FIT_ALGORITHM",
    "SipFit",
    "catalog_source_from_backends",
    "fit_shift_field",
    "fit_tan_sip",
    "frame_stars",
    "measure_frame_field",
    "measure_lateral_colour",
    "measure_panel",
    "refine_solution",
    "solve_mosaic",
]
