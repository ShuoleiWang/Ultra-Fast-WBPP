"""A mosaic panel's place on the canvas: the stage between registration and
integration of a panel run.

The registration reference is calibrated at full resolution and solved, the
solution is refined into TAN+SIP on catalog stars, and the window of the
canvas that holds every admitted Light becomes the output grid of the pixel
pipeline: each Light is then resampled once, from its calibrated pixels
straight onto the canvas (see :class:`ufwbpp.stacking.parameters.OutputGrid`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

from astropy.io import fits
import numpy as np

from ..mosaic.astrometry import MosaicAstrometryError, catalog_source_from_backends, refine_solution
from ..mosaic.canvas import CanvasError, canvas_window, celestial_wcs, frame_boundary, window_grid
from ..stacking.parameters import OutputGrid
from ..solvers.base import SolverBackend
from .common import _emit, _relativize_solver_attempts, _safe_token
from .contracts import E2EError, E2ERequest, ProgressCallback, ProgressStage
from .solve import _SolverHints, _read_image_header, _solve_one


CANVAS_VERIFICATION_BACKEND = "canvas-catalog-verification"


# Catalog matches a panel reference needs before its distortion is trusted.
MINIMUM_REFERENCE_MATCHES = 30
_STRUCTURAL_KEYS = frozenset(
    {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE", "CHECKSUM", "DATASUM"}
)


# The Lanczos-3 warp leaves this many pixels at every frame edge undefined.
_WARP_MARGIN_PIXELS = 2.5


@dataclass(frozen=True)
class _CanvasPlacement:
    grid: OutputGrid
    record: dict[str, Any]
    # The fraction of the window the reference frame's footprint fills: a
    # window is a rectangle on the canvas, a rotated frame is not.
    footprint_fill: float


def _light_header(path: str) -> fits.Header:
    """The Light's own cards (pointing, optics, exposure) without its data
    layout, for the solver of its calibrated copy."""

    header = fits.Header()
    if not path.casefold().endswith((".fit", ".fits", ".fts")):
        return header
    with fits.open(path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdul:
        source = next((item for item in hdul if int(item.header.get("NAXIS", 0) or 0) >= 2), hdul[0])
        for card in source.header.cards:
            if card.keyword and card.keyword not in _STRUCTURAL_KEYS and card.keyword not in ("COMMENT", "HISTORY", ""):
                header[card.keyword] = card.value
    return header


def _place_on_canvas(
    request: E2ERequest,
    registration: Any,
    calibration_plan: Any,
    *,
    work: Path,
    backends: Sequence[Any],
    hints: _SolverHints,
) -> _CanvasPlacement:
    from ufwbpp_registration import read_calibrated_full_image

    canvas = request.mosaic_canvas
    assert canvas is not None
    run = registration.run
    reference_path = str(Path(run.analyses[run.reference_index].path))
    directory = work / "canvas"
    directory.mkdir()
    try:
        image = read_calibrated_full_image(reference_path, calibration_plan)
    except (OSError, ValueError) as error:
        raise E2EError("MOSAIC_REFERENCE_UNREADABLE", str(error), path=reference_path) from error
    reference_fits = directory / "reference.fits"
    fits.PrimaryHDU(np.asarray(image, dtype=np.float32), _light_header(reference_path)).writeto(reference_fits)
    solved_path = directory / "reference-solved.fits"
    solved, attempts = _solve_one(
        input_path=reference_fits,
        output_path=solved_path,
        backends=backends,
        hints=hints,
        min_matches=request.min_matches,
        max_rms_arcsec=request.max_rms_arcsec,
    )
    if not solved:
        raise E2EError(
            "MOSAIC_REFERENCE_UNSOLVED",
            "the panel's registration reference did not solve; its place on the mosaic canvas is unknown",
        )
    solved_header, shape = _read_image_header(solved_path)
    catalog = catalog_source_from_backends(backends)
    if catalog is None:
        raise E2EError(
            "MOSAIC_CATALOG_UNAVAILABLE",
            "a mosaic panel needs the managed star catalog to measure its distortion",
        )
    try:
        fit = refine_solution(
            image,
            solved_header,
            catalog,
            minimum_matches=max(request.min_matches, MINIMUM_REFERENCE_MATCHES),
        )
    except MosaicAstrometryError as error:
        raise E2EError(error.code, str(error)) from error
    if fit.rms_arcsec > request.max_rms_arcsec:
        raise E2EError(
            "MOSAIC_REFERENCE_ASTROMETRY_FAILED",
            f"the panel reference's catalog residual is {fit.rms_arcsec:.3g} arcsec; at most "
            f"{request.max_rms_arcsec:.3g} is accepted",
        )
    reference_wcs = celestial_wcs(fit.header)
    # The part of a Light the warp can sample: inside its interpolation margin.
    boundary = frame_boundary(shape, inset=_WARP_MARGIN_PIXELS)
    homogeneous = np.column_stack((boundary, np.ones(len(boundary))))
    points_x: list[np.ndarray] = []
    points_y: list[np.ndarray] = []
    for matrix in registration.transforms.values():
        mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
        points_x.append(mapped[:, 0] / mapped[:, 2])
        points_y.append(mapped[:, 1] / mapped[:, 2])
    ra, dec = reference_wcs.all_pix2world(np.concatenate(points_x), np.concatenate(points_y), 0)
    try:
        origin, window_shape = canvas_window(canvas, np.asarray(ra), np.asarray(dec), margin=1)
        grid = window_grid(canvas, origin, window_shape, fit.header, shape)
    except (CanvasError, ValueError) as error:
        raise E2EError(getattr(error, "code", "MOSAIC_CANVAS_WINDOW_INVALID"), str(error)) from error
    accepted = next(item for item in reversed(attempts) if item.get("accepted") is True)
    record = {
        "schemaVersion": 1,
        "panel": request.mosaic_panel,
        "canvas": canvas.serializable(),
        "referenceLight": Path(reference_path).name,
        "referenceSolve": {
            "backendId": accepted.get("result", {}).get("backendId"),
            "astrometricQuality": accepted.get("result", {}).get("astrometricQuality"),
        },
        "distortion": fit.serializable(),
        "referenceWcs": {key: fit.header[key] for key in sorted(fit.header)},
        "window": grid.serializable(),
        "lightsInWindow": len(registration.transforms),
    }
    fill = min(1.0, (shape[0] * shape[1]) / float(window_shape[0] * window_shape[1]))
    record["footprintFill"] = fill
    return _CanvasPlacement(grid, record, fill)


def _catalog_verifier(
    backends: Sequence[SolverBackend],
) -> Callable[[np.ndarray, Mapping[str, Any], Path], Mapping[str, Any]] | None:
    """Verify a known WCS against catalog stars with the first backend that
    can: its own ``canvas_verifier`` (as ``verify_result`` replaces the
    generic check of a solve), else the managed catalog behind its solver
    config, which yields the correspondence evidence a solve must carry."""

    import os

    from ..solvers.catalog_correspondence import bind_installed_set, verify_solution
    from ..solvers.catalogs import installed_set_snapshot_for_solver_config

    for backend in backends:
        supplied = getattr(backend, "canvas_verifier", None)
        if supplied is not None:
            return supplied
        config_path = getattr(backend, "config_path", None)
        if config_path is None:
            continue
        manifest_dir = getattr(backend, "catalog_manifest_dir", None)
        environment = dict(os.environ) | dict(getattr(backend, "environment", None) or {})

        def verify(image: np.ndarray, header: Mapping[str, Any], artifact_dir: Path) -> Mapping[str, Any]:
            snapshot = installed_set_snapshot_for_solver_config(config_path, manifest_dir=manifest_dir, environment=environment)
            verification = verify_solution(
                image=np.asarray(image, dtype=np.float32),
                wcs_header=dict(header),
                image_shape=tuple(image.shape),
                catalog_root=Path(str(snapshot["catalogRoot"])),
                index_artifacts=snapshot["artifacts"],
                artifact_dir=artifact_dir,
            )
            quality, _ = bind_installed_set(verification.quality, snapshot, manifest_dir=manifest_dir, environment=environment)
            return quality.serializable()

        return verify
    return None


def _verify_canvas_candidates(
    candidates: Mapping[str, Path],
    *,
    grid: OutputGrid,
    products_dir: Path,
    staging: Path,
    verifier: Callable[[np.ndarray, Mapping[str, Any], Path], Mapping[str, Any]] | None,
    request: E2ERequest,
    progress: ProgressCallback | None,
) -> tuple[dict[str, Any], list[Path], dict[str, Path], bool]:
    """The filter masters of a mosaic panel carry the canvas window's WCS:
    it is verified on catalog stars instead of being solved blind.  Returns
    what :func:`ufwbpp.workflows.products._solve_candidates` returns."""

    records: dict[str, Any] = {}
    staged: list[Path] = []
    solved: dict[str, Path] = {}
    all_verified = True
    for position, (filter_name, candidate) in enumerate(sorted(candidates.items()), start=1):
        token = _safe_token(filter_name)
        directory = products_dir / token
        directory.mkdir(parents=True, exist_ok=False)
        output = directory / f"master_light_{token}_wcs.fits"
        artifacts = directory / "canvas-verification"
        artifacts.mkdir()
        attempt: dict[str, Any] = {"backendId": CANVAS_VERIFICATION_BACKEND, "accepted": False}
        with fits.open(candidate, mode="readonly", memmap=True) as hdul:
            # A copy: no mapping of the staged master may outlive this block.
            data = np.array(hdul[0].data, dtype=np.float32)
            header = hdul[0].header.copy()
        try:
            if verifier is None:
                raise E2EError("MOSAIC_CATALOG_UNAVAILABLE", "no managed catalog verifies the canvas WCS")
            quality = dict(verifier(data, dict(grid.wcs), artifacts))
            matched = int(quality.get("matchedStars", 0) or 0)
            rms = float(quality.get("rmsArcsec", math.inf) or math.inf)
            accepted = (
                quality.get("catalogManaged") is True
                and matched >= request.min_matches
                and rms <= request.max_rms_arcsec
            )
            attempt.update(result={"backendId": CANVAS_VERIFICATION_BACKEND, "astrometricQuality": quality}, accepted=accepted)
            if not accepted:
                attempt["code"] = "CANVAS_WCS_VERIFICATION_FAILED"
        except Exception as error:
            attempt.update(code=getattr(error, "code", "CANVAS_WCS_VERIFICATION_FAILED"), message=str(error))
        if attempt["accepted"]:
            for key, value in grid.wcs.items():
                header[key] = value
            header["OAFSTATE"] = "SOLVED"
            header["OAFWCS"] = "SOLVED"
            header["OAFWCSPR"] = ("CANVAS_CATALOG_VERIFIED", "WCS provenance of this grid")
            fits.PrimaryHDU(data, header).writeto(output, checksum=True)
            staged.append(output)
            solved[filter_name] = output
        else:
            shutil.rmtree(artifacts, ignore_errors=True)
            all_verified = False
        records[filter_name] = {
            "status": "SOLVED" if attempt["accepted"] else "UNSOLVED",
            "input": str(candidate.relative_to(staging)),
            "output": str(output.relative_to(staging)) if attempt["accepted"] else None,
            "attempts": _relativize_solver_attempts([attempt], staging),
        }
        _emit(
            progress,
            ProgressStage.ASTROMETRY,
            "progress",
            f"{filter_name}: canvas WCS {'verified' if attempt['accepted'] else 'NOT verified'}",
            current=position,
            total=len(candidates),
        )
    return records, staged, solved, all_verified


__all__ = ["CANVAS_VERIFICATION_BACKEND", "MINIMUM_REFERENCE_MATCHES"]
