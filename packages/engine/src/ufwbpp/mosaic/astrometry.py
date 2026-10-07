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
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray

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
        for summary in rank_indexes(summaries, field_geometry):
            found = read_index_stars(
                root / summary.relative_name,
                centre_ra_degrees=centre_ra_degrees,
                centre_dec_degrees=centre_dec_degrees,
                radius_degrees=radius_degrees,
                expected_stat=artifacts[summary.relative_name].get("statIdentity"),
            )
            if found.count >= 3 * STARS_PER_TERM * 3:
                return CatalogStars(
                    np.asarray(found.ra_degrees, dtype=np.float64) % 360.0,
                    np.asarray(found.dec_degrees, dtype=np.float64),
                    None if found.magnitudes is None else np.asarray(found.magnitudes, dtype=np.float64),
                    found.summary.identity,
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
        sip_a: dict[tuple[int, int], float] = {}
        sip_b: dict[tuple[int, int], float] = {}
        for term in _terms(order, minimum=2):
            coefficient = inverse_cd @ np.asarray((a[index[term]], b[index[term]]))
            sip_a[term] = float(coefficient[0])
            sip_b[term] = float(coefficient[1])
            header[f"A_{term[0]}_{term[1]}"] = sip_a[term]
            header[f"B_{term[0]}_{term[1]}"] = sip_b[term]
        # The inverse polynomials (one order more) on a grid slightly
        # larger than the frame: AP/BP map the distorted offsets back.
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


def refine_solution(
    image: NDArray[np.float32],
    header: Mapping[str, Any],
    catalog: CatalogStarSource,
    *,
    minimum_matches: int = 30,
    maximum_order: int = 3,
) -> SipFit:
    """Refine a solver's solution of ``image`` into TAN+SIP on catalog stars."""

    from ..solvers.catalog_correspondence import (
        CorrespondenceParameters,
        detect_field_stars,
        match_mutual_nearest,
    )

    height, width = image.shape
    initial = celestial_wcs(header)
    detected_one_based, fwhm, detection = detect_field_stars(
        np.asarray(image, dtype=np.float32), CorrespondenceParameters(maximum_stars=4000)
    )
    detected = _isolated_windowed_centroids(
        np.asarray(image, dtype=np.float32),
        np.asarray(detected_one_based, dtype=np.float64) - 1.0,
        float(np.median(fwhm)) if len(fwhm) else 3.0,
    )
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
    wcs = celestial_wcs(fitted)
    matched_pixels = detected[rows][keep]
    matched_world = world[catalog_rows][keep]
    residual = matched_pixels - np.column_stack(
        wcs.all_world2pix(matched_world[:, 0], matched_world[:, 1], 0, tolerance=1e-9, maxiter=100, quiet=True)
    )
    radius_pixels = np.hypot(residual[:, 0], residual[:, 1])
    arcsec = _arcsec_residuals(wcs, matched_pixels, matched_world)
    # Residuals of the frame's outer ring (where mosaic overlaps lie) against its middle.
    edge = np.minimum.reduce(
        (matched_pixels[:, 0], width - 1 - matched_pixels[:, 0], matched_pixels[:, 1], height - 1 - matched_pixels[:, 1])
    ) < 0.15 * min(width, height)
    return SipFit(
        header=fitted,
        order=order,
        matched=int(keep.sum()),
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
        },
    )


__all__ = [
    "CatalogStarSource",
    "CatalogStars",
    "ManagedCatalogStars",
    "MosaicAstrometryError",
    "SIP_FIT_ALGORITHM",
    "SipFit",
    "catalog_source_from_backends",
    "fit_tan_sip",
    "refine_solution",
]
