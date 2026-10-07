"""A synthetic sky for mosaic tests: frames rendered through their true WCS.

Stars have sky positions; every frame is rendered by projecting them through
the frame's own TAN+SIP solution, so the geometry of every pixel is known
exactly, distortion included.  The frame headers carry their true solution
(``T``-prefixed cards); :class:`TruthSolver` "solves" a frame by reading them
back as a pure TAN solution (as astrometry.net's ``.match`` table does,
without the distortion), serves the true star list as the managed catalog
and verifies a WCS by matching detected stars to it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from ufwbpp.mosaic.astrometry import CatalogStars
from ufwbpp.mosaic.canvas import celestial_wcs
from ufwbpp.solvers.base import (
    AstrometricQuality,
    SolutionKind,
    SolverIndexArtifact,
    SolverResult,
    SolverStatus,
    WcsParity,
)


@dataclass(frozen=True)
class Sky:
    ra: np.ndarray
    dec: np.ndarray
    flux: np.ndarray


def make_sky(
    centre: tuple[float, float],
    radius_degrees: float,
    count: int,
    seed: int = 11,
    *,
    flux_range: tuple[float, float] = (4.2, 5.2),
) -> Sky:
    rng = np.random.default_rng(seed)
    # Uniform on a small cap around the centre.
    distance = radius_degrees * np.sqrt(rng.uniform(0.0, 1.0, count))
    angle = rng.uniform(0.0, 2.0 * math.pi, count)
    dec = centre[1] + distance * np.sin(angle)
    ra = centre[0] + distance * np.cos(angle) / math.cos(math.radians(centre[1]))
    flux = 10 ** rng.uniform(flux_range[0], flux_range[1], count)
    return Sky(ra, dec, flux)


def true_header(
    centre: tuple[float, float],
    shape: tuple[int, int],
    *,
    scale_arcsec: float,
    rotation_degrees: float = 0.0,
    distortion: float = 0.0,
) -> dict[str, Any]:
    """TAN+SIP of a frame: ``distortion`` pixels of cubic barrel at the corners."""

    height, width = shape
    s = scale_arcsec / 3600.0
    a = math.radians(rotation_degrees)
    cd = s * np.asarray(((math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a)))) @ np.diag((-1.0, 1.0))
    header: dict[str, Any] = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": float(centre[0]),
        "CRVAL2": float(centre[1]),
        "CRPIX1": (width + 1) / 2.0,
        "CRPIX2": (height + 1) / 2.0,
        "CD1_1": float(cd[0, 0]),
        "CD1_2": float(cd[0, 1]),
        "CD2_1": float(cd[1, 0]),
        "CD2_2": float(cd[1, 1]),
    }
    if distortion:
        corner = math.hypot(width / 2.0, height / 2.0)
        k = distortion / corner**3
        header.update(
            {
                "CTYPE1": "RA---TAN-SIP",
                "CTYPE2": "DEC--TAN-SIP",
                "A_ORDER": 3,
                "B_ORDER": 3,
                "A_3_0": k,
                "A_1_2": k,
                "B_2_1": k,
                "B_0_3": k,
            }
        )
    return header


_TRUTH_KEYS = {
    "CTYPE1": "TCTYPE1",
    "CTYPE2": "TCTYPE2",
    "CRVAL1": "TCRVAL1",
    "CRVAL2": "TCRVAL2",
    "CRPIX1": "TCRPIX1",
    "CRPIX2": "TCRPIX2",
    "CD1_1": "TCD1_1",
    "CD1_2": "TCD1_2",
    "CD2_1": "TCD2_1",
    "CD2_2": "TCD2_2",
    "A_ORDER": "TA_ORDER",
    "B_ORDER": "TB_ORDER",
    "A_3_0": "TA_3_0",
    "A_1_2": "TA_1_2",
    "B_2_1": "TB_2_1",
    "B_0_3": "TB_0_3",
}


def truth_cards(header: Mapping[str, Any]) -> dict[str, Any]:
    return {_TRUTH_KEYS[key]: value for key, value in header.items() if key in _TRUTH_KEYS}


def render(
    sky: Sky,
    header: Mapping[str, Any],
    shape: tuple[int, int],
    *,
    fwhm: float,
    background: float,
    noise: float,
    rng: np.random.Generator,
    gain: float = 1.0,
) -> np.ndarray:
    """The frame: every star of the sky drawn where the frame's WCS puts it."""

    height, width = shape
    wcs = celestial_wcs(header)
    x, y = wcs.all_world2pix(sky.ra, sky.dec, 0)
    image = np.full(shape, background, dtype=np.float64)
    sigma = fwhm / 2.3548
    reach = int(math.ceil(5 * sigma))
    for sx, sy, flux in zip(x, y, sky.flux):
        if not (-reach <= sx < width + reach and -reach <= sy < height + reach):
            continue
        x0, x1 = max(0, int(sx) - reach), min(width, int(sx) + reach + 2)
        y0, y1 = max(0, int(sy) - reach), min(height, int(sy) + reach + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        image[y0:y1, x0:x1] += gain * flux / (2 * math.pi * sigma**2) * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * sigma**2))
    return image + rng.normal(0.0, noise, shape)


def _quality(matched: int, rms_pixels: float, scale_arcsec: float) -> AstrometricQuality:
    return AstrometricQuality(
        matched_stars=matched,
        rms_pixels=rms_pixels,
        rms_arcsec=rms_pixels * scale_arcsec,
        parity=WcsParity.NEGATIVE,
        catalog_identity="1" * 64,
        index_identities=("astrometry.net:index:4200:healpix:1:hpnside:1",),
        correspondence_sha256="2" * 64,
        catalog_managed=True,
        installed_set_identity="3" * 64,
        catalog_manifest_sha256="4" * 64,
        index_artifacts=(
            SolverIndexArtifact(
                index_id="4200",
                relative_name="index-4200.fits",
                size_bytes=4096,
                sha256="5" * 64,
                manifest_sha256="4" * 64,
                installed_set_identity="3" * 64,
            ),
        ),
    )


class TruthCatalog:
    def __init__(self, sky: Sky) -> None:
        self.sky = sky

    def stars(self, centre_ra_degrees: float, centre_dec_degrees: float, radius_degrees: float) -> CatalogStars:
        dra = (self.sky.ra - centre_ra_degrees) * math.cos(math.radians(centre_dec_degrees))
        ddec = self.sky.dec - centre_dec_degrees
        inside = np.hypot(dra, ddec) <= radius_degrees
        return CatalogStars(self.sky.ra[inside], self.sky.dec[inside], -np.log10(self.sky.flux[inside]), "truth-catalog")


class TruthSolver:
    """A solver that reads a frame's true solution from its header (as a
    pure TAN, the way astrometry.net's ``.match`` table reports it)."""

    backend_id = "truth-solver"

    def __init__(self, sky: Sky, *, scale_arcsec: float) -> None:
        self.sky = sky
        self.scale_arcsec = scale_arcsec
        self.catalog_star_source = TruthCatalog(sky)
        self.inputs: list[str] = []

    def solve(self, request: Any) -> SolverResult:
        self.inputs.append(request.input_path)
        with fits.open(request.input_path, mode="readonly", memmap=False) as hdul:
            hdu = next(item for item in hdul if item.data is not None and item.data.ndim == 2)
            height, width = hdu.data.shape
            source = hdul[0].header
            if "TCRVAL1" not in source:
                return SolverResult(self.backend_id, SolverStatus.FAILED, SolutionKind.NONE, False, error="no truth")
            primary = hdul[0].header
            for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"):
                primary[key] = source["T" + key]
            primary["CTYPE1"] = "RA---TAN"
            primary["CTYPE2"] = "DEC--TAN"
            primary["OAFSTATE"] = "SOLVED"
            hdul.writeto(request.output_path, overwrite=False, checksum=True)
            solved = primary.copy()
        return SolverResult(
            backend_id=self.backend_id,
            status=SolverStatus.SOLVED,
            solution_kind=SolutionKind.SOLVED,
            backend_confirmed=True,
            header=solved,
            image_shape=(height, width),
            output_path=request.output_path,
            astrometric_quality=_quality(40, 0.2, self.scale_arcsec),
            evidence={
                "fakeReceiptVerified": True,
                "astrometricQuality": _quality(40, 0.2, self.scale_arcsec).serializable(),
            },
        )

    def verify_result(self, result: SolverResult) -> bool:
        return (
            bool(result.evidence.get("fakeReceiptVerified"))
            and result.astrometric_quality is not None
            and result.evidence.get("astrometricQuality") == result.astrometric_quality.serializable()
            and Path(result.output_path or "").is_file()
        )

    def canvas_verifier(self, image: np.ndarray, header: Mapping[str, Any], artifact_dir: Path) -> Mapping[str, Any]:
        """Match the detected stars to the true star list through ``header``."""

        from ufwbpp.solvers.catalog_correspondence import (
            CorrespondenceParameters,
            detect_field_stars,
            match_mutual_nearest,
        )

        detected, _, _ = detect_field_stars(np.nan_to_num(np.asarray(image, dtype=np.float32), nan=float(np.nanmedian(image))), CorrespondenceParameters())
        wcs = celestial_wcs(header)
        predicted = np.column_stack(wcs.all_world2pix(self.sky.ra, self.sky.dec, 1))
        height, width = image.shape
        inside = (predicted[:, 0] > 3) & (predicted[:, 0] < width - 3) & (predicted[:, 1] > 3) & (predicted[:, 1] < height - 3)
        rows, catalog_rows, distances = match_mutual_nearest(detected, predicted[inside], 2.0)
        rms = float(np.sqrt(np.mean(np.square(distances)))) if distances.size else math.inf
        # The geometry is judged on isolated stars away from the edges: a
        # blended pair pulls a detector's centroid by up to a pixel.
        candidates = predicted[inside]
        separation = np.sqrt(np.sum(np.square(candidates[:, None, :] - candidates[None, :, :]), axis=2))
        np.fill_diagonal(separation, np.inf)
        matched = candidates[catalog_rows]
        isolated = (separation[catalog_rows].min(axis=1) >= 6.0) & (matched[:, 0] > 7) & (matched[:, 0] < width - 7) & (matched[:, 1] > 7) & (matched[:, 1] < height - 7)
        clean = distances[isolated]
        (artifact_dir / "verification.json").write_text("{}", encoding="utf-8")
        return {
            **_quality(int(rows.size), rms, self.scale_arcsec).serializable(),
            "isolatedStars": int(clean.size),
            "offsetMedianPixels": float(np.median(clean)) if clean.size else math.inf,
            "offsetP90Pixels": float(np.percentile(clean, 90)) if clean.size else math.inf,
        }


def light_header(
    target: str,
    filter_name: str,
    *,
    exposure: float,
    observed_at: str,
    truth: Mapping[str, Any],
    pointing: tuple[float, float],
) -> fits.Header:
    header = fits.Header()
    header["IMAGETYP"] = "Light"
    header["FILTER"] = filter_name
    header["OBJECT"] = target
    header["INSTRUME"] = "SYNTHETIC-CAMERA"
    header["EXPTIME"] = exposure
    header["GAIN"] = 100
    header["OFFSET"] = 20
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "MODE-1"
    header["CCD-TEMP"] = -10.0
    header["BAYERPAT"] = "NONE"
    header["DATE-OBS"] = observed_at
    header["RA"] = float(pointing[0])
    header["DEC"] = float(pointing[1])
    for key, value in truth_cards(truth).items():
        header[key] = value
    return header


def calibration_header(role: str, *, exposure: float, filter_name: str = "R") -> fits.Header:
    header = fits.Header()
    header["IMAGETYP"] = role
    header["FILTER"] = filter_name
    header["OBJECT"] = "CALIBRATION"
    header["INSTRUME"] = "SYNTHETIC-CAMERA"
    header["EXPTIME"] = exposure
    header["GAIN"] = 100
    header["OFFSET"] = 20
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "MODE-1"
    header["CCD-TEMP"] = -10.0
    header["BAYERPAT"] = "NONE"
    header["DATE-OBS"] = "2026-01-01T18:00:00Z"
    return header


def write_uint16(path: Path, data: np.ndarray, header: fits.Header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.clip(np.round(data), 0, 65535).astype(np.uint16), header, overwrite=False)
    return path


__all__ = [
    "Sky",
    "TruthCatalog",
    "TruthSolver",
    "calibration_header",
    "light_header",
    "make_sky",
    "render",
    "true_header",
    "write_uint16",
]
