"""A mosaic panel's place on the canvas: the stage between registration and
integration of a panel run.

The registration reference is calibrated at full resolution, solved, and its
stars are measured against the catalog; the project then fits every panel's
TAN+SIP solution at once (:func:`ufwbpp.mosaic.astrometry.solve_mosaic`), and
the window of the canvas that holds every admitted Light becomes the output
grid of the pixel pipeline: each Light is then resampled once, from its
calibrated pixels straight onto the canvas, or, in a drizzle run, dropped
through the window's forward lattice onto a copy of the window the drizzle
scale times finer (see :class:`ufwbpp.stacking.parameters.OutputGrid`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

from astropy.io import fits
from lightframeqc.cfa import is_cfa_pattern
import numpy as np

from ..mosaic.astrometry import (
    FRAME_FIELD_ALGORITHM,
    MosaicAstrometryError,
    PanelStars,
    SipFit,
    catalog_source_from_backends,
    frame_stars,
    measure_frame_field,
    measure_lateral_colour,
    measure_panel,
    solve_mosaic,
)
from ..mosaic.canvas import (
    CanvasError,
    CanvasProjection,
    canvas_window,
    celestial_wcs,
    forward_lattice,
    frame_boundary,
    window_grid,
)
from ..stacking.parameters import OutputGrid, ShiftField
from ..solvers.base import SolverBackend
from .common import _emit, _relativize_solver_attempts, _safe_token
from .contracts import E2EError, E2ERequest, IntegrationMode, ProgressCallback, ProgressStage
from .solve import _SolverHints, _read_image_header, _solve_one
from .sources import _input_frame_info


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
    # Per registered Light (by its registration key), its own residual
    # shift against the reference; none in a drizzle run.
    frame_fields: Mapping[str, ShiftField]
    # (filter, None, lattice) for each mono filter with its own lateral
    # colour (a drizzle run).
    channel_grids: tuple[tuple[str, int | None, OutputGrid], ...] = ()


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


@dataclass(frozen=True)
class _MeasuredReference:
    """A panel's registration reference, solved and measured on catalog stars."""

    stars: PanelStars
    path: str
    # The calibrated reference as solved (the lateral colour is measured on it).
    image_path: Path
    attempts: list[dict[str, Any]]


def _measure_reference(
    request: E2ERequest,
    registration: Any,
    calibration_plan: Any,
    *,
    work: Path,
    backends: Sequence[Any],
    hints: _SolverHints,
) -> _MeasuredReference:
    from ufwbpp_registration import read_calibrated_full_image

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
    solved_header, _ = _read_image_header(solved_path)
    catalog = catalog_source_from_backends(backends)
    if catalog is None:
        raise E2EError(
            "MOSAIC_CATALOG_UNAVAILABLE",
            "a mosaic panel needs the managed star catalog to measure its distortion",
        )
    try:
        stars = measure_panel(
            image, solved_header, catalog, minimum_matches=max(request.min_matches, MINIMUM_REFERENCE_MATCHES)
        )
    except MosaicAstrometryError as error:
        raise E2EError(error.code, str(error)) from error
    return _MeasuredReference(stars, reference_path, reference_fits, attempts)


def _solve_placements(
    measured: Mapping[str, PanelStars], *, max_rms_arcsec: float
) -> dict[str, tuple[SipFit, dict[str, Any]]]:
    """Every panel's solution, fitted on all panels' catalog and overlap stars
    at once; each must pass the catalog-residual gate before any panel integrates."""

    try:
        solutions, record = solve_mosaic(measured)
    except MosaicAstrometryError as error:
        raise E2EError(error.code, str(error)) from error
    for key, solution in solutions.items():
        if solution.rms_arcsec > max_rms_arcsec:
            raise E2EError(
                "MOSAIC_REFERENCE_ASTROMETRY_FAILED",
                f"{key}: the panel reference's catalog residual is {solution.rms_arcsec:.3g} arcsec; at most "
                f"{max_rms_arcsec:.3g} is accepted",
            )
    return {key: (solution, record) for key, solution in solutions.items()}


def _place_on_canvas(
    request: E2ERequest,
    registration: Any,
    calibration_plan: Any,
    measured: _MeasuredReference,
    solution: SipFit,
    joint: Mapping[str, Any],
) -> _CanvasPlacement:
    canvas = request.mosaic_canvas
    assert canvas is not None
    shape = measured.stars.shape
    reference_wcs = celestial_wcs(solution.header)
    # The part of a Light the warp can sample: inside its interpolation margin.
    boundary = frame_boundary(shape, inset=_WARP_MARGIN_PIXELS)
    homogeneous = np.column_stack((boundary, np.ones(len(boundary))))
    points_x: list[np.ndarray] = []
    points_y: list[np.ndarray] = []
    for matrix in registration.transforms.values():
        mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
        points_x.append(mapped[:, 0] / mapped[:, 2])
        points_y.append(mapped[:, 1] / mapped[:, 2])
    reach_x, reach_y = np.concatenate(points_x), np.concatenate(points_y)
    ra, dec = reference_wcs.all_pix2world(reach_x, reach_y, 0)
    # A drizzle drops every pixel through the forward map, over every point
    # the Lights reach in the reference.
    forward_domain = (
        (float(reach_x.min()), float(reach_y.min()), float(reach_x.max()), float(reach_y.max()))
        if request.integration_mode is IntegrationMode.DRIZZLE
        else None
    )
    try:
        origin, window_shape = canvas_window(canvas, np.asarray(ra), np.asarray(dec), margin=1)
        grid = window_grid(canvas, origin, window_shape, solution.header, shape, forward_domain=forward_domain)
    except (CanvasError, ValueError) as error:
        raise E2EError(getattr(error, "code", "MOSAIC_CANVAS_WINDOW_INVALID"), str(error)) from error
    accepted = next(item for item in reversed(measured.attempts) if item.get("accepted") is True)
    record = {
        "schemaVersion": 1,
        "panel": request.mosaic_panel,
        "canvas": canvas.serializable(),
        "referenceLight": Path(measured.path).name,
        "referenceSolve": {
            "backendId": accepted.get("result", {}).get("backendId"),
            "astrometricQuality": accepted.get("result", {}).get("astrometricQuality"),
        },
        "distortion": solution.serializable(),
        "jointFit": dict(joint),
        "referenceWcs": {key: solution.header[key] for key in sorted(solution.header)},
        "window": grid.serializable(),
        "lightsInWindow": len(registration.transforms),
    }
    fill = min(1.0, (shape[0] * shape[1]) / float(window_shape[0] * window_shape[1]))
    record["footprintFill"] = fill
    if request.integration_mode is IntegrationMode.DRIZZLE:
        image = np.asarray(fits.getdata(measured.image_path), dtype=np.float32)
        channel_grids, record["lateralColour"] = _lateral_colour_grids(
            request, registration, calibration_plan, measured.path, image, grid, canvas=canvas, reference_header=solution.header
        )
        return _CanvasPlacement(grid, record, fill, frame_fields={}, channel_grids=channel_grids)
    frame_fields, record["frameFields"] = _frame_fields(registration, shape)
    return _CanvasPlacement(grid, record, fill, frame_fields=frame_fields)


def _frame_fields(registration: Any, shape: tuple[int, int]) -> tuple[dict[str, ShiftField], dict[str, Any]]:
    """Each registered Light's own residual shift against the reference,
    fitted on its registration's inlier centroids: a meridian flip, another
    night's optics and the Light's filter (lateral colour) each leave one
    that its matrix cannot absorb, and a panel that stacks them unshifted
    places a star where none of its Lights does.  Per filter, the receipt's
    summary."""

    filters = {item["path"]: str(item.get("filter")) for item in registration.receipt.get("transforms", ())}
    fields: dict[str, ShiftField] = {}
    measured: dict[str, list[ShiftField]] = {}
    for path, pairs in registration.star_pairs.items():
        shift_field = measure_frame_field(pairs, np.asarray(registration.transforms[path], dtype=np.float64), shape)
        measured.setdefault(filters.get(path, "None"), []).append(shift_field)
        if shift_field.order is not None:
            fields[path] = shift_field
    summary: dict[str, Any] = {}
    for filter_name, items in sorted(measured.items()):
        applied = [item.evidence for item in items if item.order is not None]
        orders: dict[str, int] = {}
        for item in items:
            key = "none" if item.order is None else str(item.order)
            orders[key] = orders.get(key, 0) + 1
        summary[filter_name] = {
            "lights": len(items),
            "applied": len(applied),
            "orders": orders,
            **(
                {
                    "maxShiftPixels": {
                        "median": float(np.median([item["maxShiftPixels"] for item in applied])),
                        "max": float(np.max([item["maxShiftPixels"] for item in applied])),
                    },
                    "rmsBeforePixels": float(np.median([item["rmsBeforePixels"] for item in applied])),
                    "rmsAfterPixels": float(np.median([item["rmsAfterPixels"] for item in applied])),
                }
                if applied
                else {}
            ),
        }
    return fields, {"algorithm": FRAME_FIELD_ALGORITHM, "filters": summary}


def _lateral_colour_grids(
    request: E2ERequest,
    registration: Any,
    calibration_plan: Any,
    reference_path: str,
    reference_image: np.ndarray,
    grid: OutputGrid,
    *,
    canvas: CanvasProjection,
    reference_header: Mapping[str, Any],
) -> tuple[tuple[tuple[str, int | None, OutputGrid], ...], dict[str, Any]]:
    """Each mono filter other than the reference's gets its own lattice on the
    window when its stars, registered to the reference, keep a smooth
    lateral-colour field (measured on its best-weighted Light): the lattice
    node at reference pixel ``p`` samples that filter at ``p + shift(p)``,
    and the forward lattice of a drizzle run drops the filter's pixel at
    ``p + shift(p)`` where the window's lattice drops ``p``.  One lattice
    per panel would place a filter's stars by the reference colour's
    distortion, and the overlapping panels would disagree."""

    from ufwbpp_registration import read_calibrated_full_image

    def info(path: str) -> Any:
        try:
            return _input_frame_info(Path(path), request.pipeline_parameters)
        except Exception:
            return None

    reference = info(reference_path)
    if reference is None or is_cfa_pattern(reference.cfa_pattern):
        return (), {}
    best: dict[str, tuple[float, str]] = {}
    for path in registration.transforms:
        frame = info(path)
        if frame is None or is_cfa_pattern(frame.cfa_pattern) or frame.filter_name == reference.filter_name:
            continue
        weight = float(registration.quality_weights.get(path, 0.0))
        if frame.filter_name not in best or weight > best[frame.filter_name][0]:
            best[frame.filter_name] = (weight, path)
    if not best:
        return (), {}
    reference_stars = frame_stars(reference_image)
    grids: list[tuple[str, int | None, OutputGrid]] = []
    records: dict[str, Any] = {}
    for filter_name, (_, path) in sorted(best.items()):
        try:
            image = read_calibrated_full_image(path, calibration_plan)
        except (OSError, ValueError) as error:
            records[filter_name] = {"applied": False, "message": str(error)}
            continue
        lateral = measure_lateral_colour(
            reference_stars, image, np.asarray(registration.transforms[path], dtype=np.float64), reference_image.shape
        )
        records[filter_name] = {"frame": Path(path).name, **lateral.evidence}
        if lateral.order is None:
            continue
        forward = (
            forward_lattice(
                canvas,
                grid.canvas_origin,
                reference_header,
                grid.forward_origin,
                grid.forward_x.shape,
                spacing=grid.spacing,
                shift=lateral.shift,
            )
            if grid.has_forward_lattice
            else {}
        )
        shifted = replace(grid.shifted(lateral), **forward)
        shifted.validate()
        grids.append((filter_name, None, shifted))
        records[filter_name]["sha256"] = shifted.digest
    return tuple(grids), records


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
    wcs: Mapping[str, Any],
    products_dir: Path,
    staging: Path,
    verifier: Callable[[np.ndarray, Mapping[str, Any], Path], Mapping[str, Any]] | None,
    request: E2ERequest,
    progress: ProgressCallback | None,
) -> tuple[dict[str, Any], list[Path], dict[str, Path], bool]:
    """The filter masters of a mosaic panel carry the canvas window's WCS
    (``wcs``; a drizzle's is the window's that many times finer): it is
    verified on catalog stars instead of being solved blind.  Returns what
    :func:`ufwbpp.workflows.products._solve_candidates` returns."""

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
        try:
            if verifier is None:
                raise E2EError("MOSAIC_CATALOG_UNAVAILABLE", "no managed catalog verifies the canvas WCS")
            quality = dict(verifier(data, dict(wcs), artifacts))
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
        del data
        if attempt["accepted"]:
            # The whole master, extensions (a drizzle's weight and coverage)
            # included, with the verified WCS in its primary header.
            shutil.copyfile(candidate, output)
            with fits.open(output, mode="update", memmap=False) as hdul:
                header = hdul[0].header
                for key, value in wcs.items():
                    header[key] = value
                header["OAFSTATE"] = "SOLVED"
                header["OAFWCS"] = "SOLVED"
                header["OAFWCSPR"] = ("CANVAS_CATALOG_VERIFIED", "WCS provenance of this grid")
                for hdu in hdul:
                    hdu.add_checksum()
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
