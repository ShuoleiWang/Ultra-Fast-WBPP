"""The mosaic stages of a project run.

Before the panels run, :func:`_plan_mosaic_canvas` solves one Light of every
panel and plans the canvas they all share; every panel run then integrates
straight onto its window of that canvas (``E2ERequest.mosaic_canvas``).
Afterwards :func:`_assemble_canvas_mosaics` matches and blends each filter's
panel masters, which already share the canvas lattice, and verifies the
canvas WCS of every mosaic against catalog stars.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from astropy.io import fits
import numpy as np

from ..mosaic.assemble import CANVAS_MOSAIC_VERSION, MosaicGateError, assemble_filter
from ..mosaic.canvas import CanvasError, CanvasPlan, PanelFootprint, plan_canvas
from ..mosaic.photometry import PanelImage
from ..platform import remove_tree
from ..solvers.base import SolverBackend
from .canvas import _light_header
from .contracts import E2ERequest
from .screening import _inferred_solver_hints
from .solve import _read_image_header, _solve_one


# Lights tried per panel to plan the canvas (middle, first, last).
SURVEY_CANDIDATES = 3
# A canvas larger than this is refused before any panel runs.
MAXIMUM_CANVAS_PIXELS = 1 << 30


class MosaicStageError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
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


def _plan_mosaic_canvas(
    panels: Sequence[tuple[str, Sequence[str]]],
    request: E2ERequest,
    backends: Sequence[SolverBackend],
    work_root: Path,
) -> tuple[CanvasPlan, dict[str, Any]]:
    """Plan the canvas from one solved Light of every panel."""

    work = Path(tempfile.mkdtemp(prefix="survey-", dir=work_root))
    try:
        footprints: list[PanelFootprint] = []
        records = []
        for key, light_files in panels:
            footprint, record = _survey_panel(key, light_files, request, backends, work)
            records.append(record)
            if footprint is None:
                raise MosaicStageError(
                    "MOSAIC_SURVEY_UNSOLVED",
                    f"no Light of panel {key} solved; its place on the mosaic canvas is unknown",
                )
            footprints.append(footprint)
        try:
            plan = plan_canvas(footprints)
        except CanvasError as error:
            raise MosaicStageError(error.code, str(error)) from error
    finally:
        remove_tree(work)
    if plan.width * plan.height > MAXIMUM_CANVAS_PIXELS:
        raise MosaicStageError(
            "MOSAIC_CANVAS_TOO_LARGE",
            f"the panels span a {plan.width}x{plan.height} canvas; at most {MAXIMUM_CANVAS_PIXELS} pixels are supported",
        )
    boxes = {panel["key"]: panel["canvasBox"] for panel in plan.panels}
    if not _boxes_connected(boxes):
        raise MosaicStageError(
            "MOSAIC_PANELS_DISCONNECTED",
            "the panels do not overlap into one mosaic (separate targets belong in separate projects)",
        )
    return plan, {"schemaVersion": 1, "canvas": plan.serializable(), "survey": records}


def _boxes_connected(boxes: Mapping[str, Sequence[float]]) -> bool:
    keys = sorted(boxes)
    if len(keys) <= 1:
        return True
    linked = {keys[0]}
    frontier = [keys[0]]
    while frontier:
        current = boxes[frontier.pop()]
        for key in keys:
            if key in linked:
                continue
            other = boxes[key]
            if min(current[2], other[2]) > max(current[0], other[0]) and min(current[3], other[3]) > max(current[1], other[1]):
                linked.add(key)
                frontier.append(key)
    return len(linked) == len(keys)


def _panel_product(panel: str, filter_name: str, master: Path, token: str) -> PanelImage:
    """Bind a panel run's master to its canvas window (the run's receipt)."""

    run_root = master.parents[2]
    canvas = json.loads((run_root / "receipts" / "canvas.json").read_text(encoding="utf-8"))
    window = canvas["window"]
    header, shape = _read_image_header(master)
    if header.get("OAFGRID") != window["sha256"][:32] or tuple(shape) != (window["height"], window["width"]):
        raise MosaicStageError(
            "MOSAIC_PANEL_WINDOW_MISMATCH",
            f"the {filter_name} master of panel {panel} is not on the canvas window its run recorded",
        )
    coverage = run_root / "coverage" / f"{token}_coverageFraction.fits"
    counts = run_root / "coverage" / f"{token}_acceptedSampleCount.fits"
    exposure = header.get("EXPTIME")
    return PanelImage(
        key=panel,
        path=master,
        origin=(int(window["canvasOrigin"][0]), int(window["canvasOrigin"][1])),
        shape=(int(window["height"]), int(window["width"])),
        exposure_seconds=float(exposure) if isinstance(exposure, (int, float)) and exposure > 0 else None,
        coverage_path=coverage if coverage.is_file() else None,
        count_path=counts if counts.is_file() else None,
    )


def _assemble_canvas_mosaics(
    plan: CanvasPlan,
    panels_by_filter: Mapping[str, tuple[str, Sequence[PanelImage]]],
    output_root: Path,
    *,
    verifier: Callable[[np.ndarray, Mapping[str, Any], Path], Mapping[str, Any]] | None,
    minimum_matches: int,
    maximum_rms_arcsec: float,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, Path], dict[str, dict[str, Any] | None], dict[str, dict[str, Any]], dict[str, Any]]:
    """Every filter's mosaic on one canvas box (the union of all windows);
    ``panels_by_filter`` maps a filter key to its name and panel masters."""

    boxes = [panel.box for _, panels in panels_by_filter.values() for panel in panels]
    canvas_box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    paths: dict[str, Path] = {}
    qualities: dict[str, dict[str, Any] | None] = {}
    receipts: dict[str, dict[str, Any]] = {}
    total = len(panels_by_filter)
    for position, (filter_key, (filter_name, panels)) in enumerate(sorted(panels_by_filter.items()), start=1):
        directory = output_root / filter_key
        directory.mkdir(parents=True)
        result = assemble_filter(
            filter_name,
            panels,
            projection=plan.projection,
            canvas_box=canvas_box,
            output_directory=directory,
            verifier=verifier,
            minimum_matches=minimum_matches,
            maximum_rms_arcsec=maximum_rms_arcsec,
        )
        paths[filter_key] = result.science_path
        qualities[filter_key] = result.quality
        receipts[filter_key] = result.receipt
        if progress is not None:
            progress(position, total, f"{filter_name}: mosaic assembled and verified")
    return paths, qualities, receipts, {"canvasBox": list(canvas_box), "version": CANVAS_MOSAIC_VERSION}


__all__ = ["MAXIMUM_CANVAS_PIXELS", "MosaicGateError", "MosaicStageError"]
