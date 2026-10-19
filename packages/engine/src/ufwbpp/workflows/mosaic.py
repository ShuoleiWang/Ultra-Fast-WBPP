"""The mosaic stages of a project run.

Before the panels run, :func:`_plan_mosaic_canvas` solves one Light of every
panel, checks that the catalog holds enough stars in each panel to place it,
and plans the canvas they all share; every panel run then integrates (or
drizzles) straight onto its window of that canvas
(``E2ERequest.mosaic_canvas``).  Afterwards :func:`_assemble_canvas_mosaics`
matches and blends each filter's panel masters, which already share the
canvas lattice (drizzled: the canvas the drizzle scale times finer), and
verifies the canvas WCS of every mosaic against catalog stars.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from astropy.coordinates import angular_separation
from astropy.io import fits
import numpy as np

from ..mosaic.assemble import CANVAS_MOSAIC_VERSION, MosaicGateError, assemble_filter
from ..mosaic.astrometry import CatalogStarSource, MosaicAstrometryError, catalog_source_from_backends
from ..mosaic.canvas import CanvasError, CanvasPlan, PanelFootprint, frame_boundary, panels_connected, plan_canvas
from ..mosaic.photometry import PanelImage
from ..platform import remove_tree
from ..solvers.base import SolverBackend
from .canvas import MINIMUM_REFERENCE_MATCHES, _light_header
from .common import _write_json
from .contracts import E2ERequest
from .screening import _inferred_solver_hints
from .solve import _read_image_header, _solve_one


# Lights tried per panel to plan the canvas (middle, first, last).
SURVEY_CANDIDATES = 3
# A canvas larger than this is refused before any panel runs.
MAXIMUM_CANVAS_PIXELS = 1 << 30
DENSER_CATALOG_COMMAND = (
    "ultra-fast-wbpp catalog install astrometry-net-5205-lite --accept-provider-terms astrometry-net-index-data-2026-09"
)


class MosaicStageError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        # The canvas survey so far, for the failure receipt of a refused plan.
        self.survey: list[dict[str, Any]] | None = None
        super().__init__(f"{code}: {message}")


def _survey_candidates(light_files: Sequence[str]) -> list[str]:
    ordered = list(light_files)
    picks = [ordered[len(ordered) // 2], ordered[0], ordered[-1]]
    return list(dict.fromkeys(picks))[:SURVEY_CANDIDATES]


def _survey_panel(
    key: str,
    light_files: Sequence[str],
    request: E2ERequest,
    backends: Sequence[SolverBackend],
    work: Path,
) -> tuple[PanelFootprint | None, dict[str, Any]]:
    """Solve one Light of a panel: where the panel lies on the sky."""

    from lightframeqc.readers import read_frame_preview
    from ufwbpp_registration import read_full_image

    attempts: list[dict[str, Any]] = []
    for index, path in enumerate(_survey_candidates(light_files)):
        try:
            metadata = read_frame_preview(path, max_long_edge=256).metadata
            image = read_full_image(path)
        except (OSError, ValueError) as error:
            attempts.append({"light": Path(path).name, "accepted": False, "code": "SURVEY_LIGHT_UNREADABLE", "message": str(error)})
            continue
        hints = _inferred_solver_hints(request, [metadata])
        frame = work / f"{key}-{index}.fits"
        fits.PrimaryHDU(np.asarray(image, dtype=np.float32), _light_header(path)).writeto(frame)
        solved = work / f"{key}-{index}-solved.fits"
        accepted, solve_attempts = _solve_one(
            input_path=frame,
            output_path=solved,
            backends=backends,
            hints=hints,
            min_matches=request.min_matches,
            max_rms_arcsec=request.max_rms_arcsec,
        )
        attempts.append({"light": Path(path).name, "accepted": bool(accepted), "hints": hints.serializable()})
        if accepted:
            header, shape = _read_image_header(solved)
            celestial = {
                card.keyword: card.value
                for card in header.cards
                if card.keyword
                and (
                    card.keyword.startswith(("CTYPE", "CRVAL", "CRPIX", "CD1_", "CD2_", "CDELT", "PC1_", "PC2_", "CUNIT"))
                    or card.keyword in ("RADESYS", "EQUINOX", "LONPOLE", "LATPOLE")
                    or card.keyword.split("_")[0] in ("A", "B", "AP", "BP")
                )
            }
            return PanelFootprint(key, celestial, shape), {"panel": key, "attempts": attempts, "solvedLight": Path(path).name}
    return None, {"panel": key, "attempts": attempts}


def _catalog_stars_in_frame(footprint: PanelFootprint, catalog: CatalogStarSource) -> tuple[int, str]:
    """The catalog stars inside a panel's solved frame, and the catalog's identity."""

    wcs = footprint.wcs()
    corners = frame_boundary(footprint.shape, 1)
    ra, dec = wcs.all_pix2world(corners[:, 0], corners[:, 1], 0)
    centre = footprint.centre()
    radius = float(np.degrees(np.max(angular_separation(np.radians(ra), np.radians(dec), *np.radians(centre)))))
    found = catalog.stars(centre[0], centre[1], 1.05 * radius)
    x, y = wcs.all_world2pix(np.asarray(found.ra_degrees), np.asarray(found.dec_degrees), 0, quiet=True)
    height, width = footprint.shape
    inside = np.isfinite(x) & np.isfinite(y) & (x >= -0.5) & (x <= width - 0.5) & (y >= -0.5) & (y <= height - 0.5)
    return int(inside.sum()), found.identity


def _check_catalog_density(
    footprints: Sequence[PanelFootprint], request: E2ERequest, backends: Sequence[SolverBackend], records: Sequence[dict[str, Any]]
) -> None:
    """A panel is placed on its catalog stars: refuse a mosaic before any
    panel runs without a managed catalog, or when it holds fewer stars in a
    panel's frame than the placement needs (a long focal length on the
    Tycho-2 set)."""

    catalog = catalog_source_from_backends(backends)
    if catalog is None:
        raise MosaicStageError(
            "MOSAIC_CATALOG_UNAVAILABLE", "a mosaic needs the managed star catalog to place its panels"
        )
    needed = max(request.min_matches, MINIMUM_REFERENCE_MATCHES)
    sparse = []
    for footprint, record in zip(footprints, records):
        try:
            count, record["catalogIdentity"] = _catalog_stars_in_frame(footprint, catalog)
        except MosaicAstrometryError as error:
            if error.code != "MOSAIC_CATALOG_FIELD_UNCOVERED":
                raise MosaicStageError(error.code, str(error).removeprefix(f"{error.code}: ")) from error
            count = 0
        record["catalogStarsInFrame"] = count
        if count < needed:
            sparse.append(f"{footprint.key} ({count})")
    if sparse:
        raise MosaicStageError(
            "MOSAIC_CATALOG_TOO_SPARSE",
            f"the installed catalog holds fewer than {needed} stars in the frame of panel {', '.join(sparse)}, "
            f"too few to place it on the mosaic; install a denser catalog: {DENSER_CATALOG_COMMAND}",
        )


def _plan_mosaic_canvas(
    panels: Sequence[tuple[str, Sequence[str]]],
    request: E2ERequest,
    backends: Sequence[SolverBackend],
    work_root: Path,
) -> tuple[CanvasPlan, dict[str, Any]]:
    """Plan the canvas from one solved Light of every panel."""

    work = Path(tempfile.mkdtemp(prefix="survey-", dir=work_root))
    records: list[dict[str, Any]] = []
    try:
        footprints: list[PanelFootprint] = []
        for key, light_files in panels:
            footprint, record = _survey_panel(key, light_files, request, backends, work)
            records.append(record)
            if footprint is None:
                raise MosaicStageError(
                    "MOSAIC_SURVEY_UNSOLVED",
                    f"no Light of panel {key} solved; its place on the mosaic canvas is unknown",
                )
            footprints.append(footprint)
        _check_catalog_density(footprints, request, backends, records)
        try:
            plan = plan_canvas(footprints)
        except CanvasError as error:
            raise MosaicStageError(error.code, str(error)) from error
    except MosaicStageError as error:
        error.survey = records
        raise
    finally:
        remove_tree(work)
    if plan.width * plan.height > MAXIMUM_CANVAS_PIXELS:
        raise MosaicStageError(
            "MOSAIC_CANVAS_TOO_LARGE",
            f"the panels span a {plan.width}x{plan.height} canvas; at most {MAXIMUM_CANVAS_PIXELS} pixels are supported",
        )
    if not panels_connected({panel["key"]: np.asarray(panel["canvasPolygon"], dtype=np.float64) for panel in plan.panels}):
        raise MosaicStageError(
            "MOSAIC_PANELS_DISCONNECTED",
            "the panels do not overlap into one mosaic (separate targets belong in separate projects)",
        )
    return plan, {"schemaVersion": 1, "canvas": plan.serializable(), "survey": records}


def _panel_product(panel: str, filter_name: str, master: Path, token: str, scale: int = 1) -> PanelImage:
    """Bind a panel run's master to its canvas window (the run's receipt);
    a master drizzled at ``scale`` lies on the window that many times finer."""

    run_root = master.parents[2]
    canvas = json.loads((run_root / "receipts" / "canvas.json").read_text(encoding="utf-8"))
    window = canvas["window"]
    header, shape = _read_image_header(master)
    # A filter with its own lateral colour was resampled onto its own lattice of the window.
    lattices = {window["sha256"][:32]}
    lateral = canvas.get("lateralColour", {}).get(filter_name, {})
    if lateral.get("applied") is True and "sha256" in lateral:
        lattices.add(lateral["sha256"][:32])
    height, width = scale * int(window["height"]), scale * int(window["width"])
    drizzled = header.get("OAFDRZ") == "NATIVE"
    if (
        header.get("OAFGRID") not in lattices
        or (int(header.get("OAFDRZSC", 1) or 1) if drizzled else 1) != scale
        or tuple(shape) != (height, width)
    ):
        raise MosaicStageError(
            "MOSAIC_PANEL_WINDOW_MISMATCH",
            f"the {filter_name} master of panel {panel} is not on the canvas window its run recorded",
        )
    exposure = header.get("EXPTIME")
    if drizzled:
        # A drizzle: the frames that reached each pixel are its COVERAGE
        # extension, and it is valid wherever a drop landed.
        planes: dict[str, Any] = {"count_path": master, "count_hdu": "COVERAGE"}
    else:
        coverage = run_root / "coverage" / f"{token}_coverageFraction.fits"
        counts = run_root / "coverage" / f"{token}_acceptedSampleCount.fits"
        planes = {
            "coverage_path": coverage if coverage.is_file() else None,
            "count_path": counts if counts.is_file() else None,
        }
    return PanelImage(
        key=panel,
        path=master,
        origin=(scale * int(window["canvasOrigin"][0]), scale * int(window["canvasOrigin"][1])),
        shape=(height, width),
        exposure_seconds=float(exposure) if isinstance(exposure, (int, float)) and exposure > 0 else None,
        **planes,
    )


def _assemble_canvas_mosaics(
    plan: CanvasPlan,
    panels_by_filter: Mapping[str, tuple[str, Sequence[PanelImage]]],
    output_root: Path,
    *,
    verifier: Callable[[np.ndarray, Mapping[str, Any], Path], Mapping[str, Any]] | None,
    minimum_matches: int,
    maximum_rms_arcsec: float,
    scale: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, Path], dict[str, dict[str, Any] | None], dict[str, dict[str, Any]], dict[str, Any]]:
    """Every filter's mosaic on one canvas box (the union of all windows);
    ``panels_by_filter`` maps a filter key to its name and panel masters,
    which lie on the canvas ``scale`` times finer when they were drizzled.
    Every filter is assembled before a failure is raised, so the receipt
    holds the gates of all of them.  A filter whose panels fall into
    several photometric components publishes its largest as the mosaic and
    the others under ``<filter>/component<k>/`` with their receipts; the
    summary lists them under ``photometricComponents``."""

    projection = plan.projection.scaled(scale) if scale > 1 else plan.projection
    boxes = [panel.box for _, panels in panels_by_filter.values() for panel in panels]
    canvas_box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    paths: dict[str, Path] = {}
    qualities: dict[str, dict[str, Any] | None] = {}
    receipts: dict[str, dict[str, Any]] = {}
    components: dict[str, list[dict[str, Any]]] = {}
    failures: dict[str, MosaicGateError] = {}
    total = len(panels_by_filter)
    for position, (filter_key, (filter_name, panels)) in enumerate(sorted(panels_by_filter.items()), start=1):
        directory = output_root / filter_key
        directory.mkdir(parents=True)
        try:
            result = assemble_filter(
                filter_name,
                panels,
                projection=projection,
                canvas_box=canvas_box,
                output_directory=directory,
                verifier=verifier,
                minimum_matches=minimum_matches,
                maximum_rms_arcsec=maximum_rms_arcsec,
            )
        except MosaicGateError as error:
            failures[filter_key] = error
            if progress is not None:
                progress(position, total, f"{filter_name}: mosaic failed its gates")
            continue
        paths[filter_key] = result.science_path
        qualities[filter_key] = result.quality
        receipts[filter_key] = result.receipt
        for component in result.components:
            _write_json(component.science_path.parent / "receipt.json", component.receipt)
            components.setdefault(filter_key, []).append(
                {
                    "component": component.receipt["component"],
                    "panels": [panel["key"] for panel in component.receipt["panels"]],
                    "verdict": component.receipt["verdict"],
                    "gates": component.receipt["gates"],
                    "directory": component.science_path.parent.relative_to(output_root).as_posix(),
                    "science": component.science_path.name,
                    "sha256": component.receipt["blend"]["sha256"]["science"],
                }
            )
        if progress is not None:
            progress(
                position,
                total,
                f"{filter_name}: mosaic assembled and verified"
                + (f" ({len(result.components) + 1} photometric components)" if result.components else ""),
            )
    if failures:
        codes = {error.code for error in failures.values()}
        raise MosaicGateError(
            codes.pop() if len(codes) == 1 else "MOSAIC_GATE_FAILED",
            "; ".join(str(error).removeprefix(f"{error.code}: ") for error in failures.values()),
            {
                "filters": {key: {"code": error.code, **error.receipt} for key, error in failures.items()},
                "passedFilters": receipts,
            },
        )
    return paths, qualities, receipts, {
        "canvasBox": list(canvas_box),
        "scale": scale,
        "version": CANVAS_MOSAIC_VERSION,
        "photometricComponents": components,
    }


__all__ = ["MAXIMUM_CANVAS_PIXELS", "MosaicGateError", "MosaicStageError"]
