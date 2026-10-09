"""One filter's mosaic from panel masters that already share the canvas.

The panels are matched (:mod:`.photometry`), blended (:mod:`.blend`) and
then judged.  Every gate is PASS, WARN or FAIL with its numbers in the
receipt; a FAIL stops the mosaic (``MosaicGateError``):

* photometry: after correction, the stars of every overlap agree in flux
  (median ratio within 0.3 % PASS, 1 % WARN);
* astrometry of the overlaps: the same stars sit on the same canvas pixel in
  both panels (offset RMS at most 0.10 px PASS, 0.25 px WARN);
* the residual background difference of every overlap after its planes;
* flux conservation: the stars of the overlaps have the same flux in the
  mosaic as in the corrected panels (0.5 %);
* the canvas WCS, verified against catalog stars on every panel's window
  of the finished mosaic (the mosaic is never blind-solved: its WCS is the
  canvas the panels were resampled onto, and the catalog decides).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray

from ..integrity import sha256_digest
from .blend import blend_filter, panel_noise
from .canvas import CanvasProjection
from .photometry import (
    BACKGROUND_ALGORITHM,
    PHOTOMETRY_ALGORITHM,
    EdgeMeasurement,
    MosaicPhotometryError,
    PanelImage,
    PlaneSolution,
    ScaleSolution,
    measure_edge,
    solve_planes,
    solve_scales,
)


CANVAS_MOSAIC_VERSION = "ultra-fast-wbpp-canvas-mosaic-v3"
PHOTOMETRY_PASS = 0.003
PHOTOMETRY_WARN = 0.01
ASTROMETRY_PASS_PIXELS = 0.10
ASTROMETRY_WARN_PIXELS = 0.25
RESIDUAL_PASS_SIGMA = 0.1
RESIDUAL_WARN_SIGMA = 0.3
FLUX_PASS = 0.005
FLUX_WARN = 0.02
# Seeing assumed for a panel with no measured overlap star.
DEFAULT_FWHM_PIXELS = 3.0

Verifier = Callable[[np.ndarray, Mapping[str, Any], Path], Mapping[str, Any]]


class MosaicGateError(RuntimeError):
    def __init__(self, code: str, message: str, receipt: Mapping[str, Any] | None = None) -> None:
        self.code = code
        self.receipt = dict(receipt or {})
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class FilterMosaic:
    science_path: Path
    noise_path: Path
    coverage_path: Path
    mask_path: Path
    receipt: dict[str, Any]
    quality: dict[str, Any] | None


def _verdict(value: float, passing: float, warning: float) -> str:
    if not math.isfinite(value):
        return "FAIL"
    return "PASS" if value <= passing else "WARN" if value <= warning else "FAIL"


def _worst(verdicts: Sequence[str]) -> str:
    return "FAIL" if "FAIL" in verdicts else "WARN" if "WARN" in verdicts else "PASS"


def _residual_records(
    edges: Sequence[EdgeMeasurement], scales: ScaleSolution, planes: PlaneSolution
) -> list[dict[str, Any]]:
    """Per overlap: the binned difference left after scales and planes."""

    records = []
    for edge in edges:
        if edge.bin_x.size < 8:
            continue
        ci, cj = scales.scales[edge.left], scales.scales[edge.right]
        difference = ci * edge.bin_left - cj * edge.bin_right
        difference -= planes.evaluate(edge.left, edge.bin_x, edge.bin_y) - planes.evaluate(edge.right, edge.bin_x, edge.bin_y)
        noise = np.hypot(ci * edge.bin_noise, cj * edge.bin_noise)
        structure = 0.5 * (ci * edge.bin_left + cj * edge.bin_right)
        faint = np.abs(structure - np.median(structure)) * scales.ratio_error(edge.left, edge.right) <= 0.1 * noise
        if faint.sum() < 8:
            continue
        weight = 1.0 / np.square(noise[faint])
        mean = float(np.sum(weight * difference[faint]) / np.sum(weight))
        typical = float(np.median(noise[faint]))
        # Pixel noise of a bin, to state the level against one pixel's noise.
        records.append(
            {
                "left": edge.left,
                "right": edge.right,
                "bins": int(faint.sum()),
                "meanDifference": mean,
                "meanDifferenceError": float(1.0 / math.sqrt(float(np.sum(weight)))),
                "levelBinSigma": abs(mean) / typical if typical > 0 else math.inf,
                "verdict": _verdict(abs(mean) / typical if typical > 0 else math.inf, RESIDUAL_PASS_SIGMA, RESIDUAL_WARN_SIGMA),
            }
        )
    return records


def _canvas_crop(path: Path, canvas_box: tuple[int, int, int, int], box: tuple[int, int, int, int]) -> NDArray[np.float32]:
    """A copy of ``box`` (canvas coordinates) of a plane covering
    ``canvas_box``.  No reference to the memory map outlives the call:
    Windows cannot rename a directory that holds a mapped file, and a gate
    failure's traceback would keep a local view alive."""

    x0, y0, x1, y1 = box
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        return np.array(
            hdul[0].data[y0 - canvas_box[1] : y1 - canvas_box[1], x0 - canvas_box[0] : x1 - canvas_box[0]],
            dtype=np.float32,
        )


def _flux_conservation(
    science_path: Path,
    canvas_box: tuple[int, int, int, int],
    edges: Sequence[EdgeMeasurement],
    scales: ScaleSolution,
    planes: PlaneSolution,
) -> dict[str, Any]:
    """The overlap stars' aperture flux in the mosaic against the mean of
    their corrected panel fluxes."""

    import sep

    ratios = []
    for edge in edges:
        if edge.flux_left.size == 0:
            continue
        x0, y0, _, _ = edge.box
        crop = _canvas_crop(science_path, canvas_box, edge.box)
        invalid = np.ascontiguousarray(~np.isfinite(crop))
        work = np.ascontiguousarray(np.where(invalid, np.nanmedian(crop), crop), dtype=np.float32)
        background = sep.Background(work, mask=invalid, bw=64, bh=64, fw=3, fh=3)
        residual = np.ascontiguousarray(work - background.back(), dtype=np.float64)
        radius = edge.aperture_radius
        flux, _, flags = sep.sum_circle(
            residual, edge.star_x - x0, edge.star_y - y0, radius, mask=invalid, subpix=5
        )
        expected = 0.5 * (scales.scales[edge.left] * edge.flux_left + scales.scales[edge.right] * edge.flux_right)
        usable = (np.asarray(flags) == 0) & (expected > 0) & np.isfinite(flux)
        ratios.extend((flux[usable] / expected[usable]).tolist())
    if not ratios:
        return {"stars": 0, "verdict": "PASS", "note": "no overlap stars (single panel)"}
    deviation = float(np.median(ratios)) - 1.0
    return {
        "stars": len(ratios),
        "medianRatioMinusOne": deviation,
        "verdict": _verdict(abs(deviation), FLUX_PASS, FLUX_WARN),
    }


def assemble_filter(
    filter_name: str,
    panels: Sequence[PanelImage],
    *,
    projection: CanvasProjection,
    canvas_box: tuple[int, int, int, int],
    output_directory: Path,
    verifier: Verifier | None,
    minimum_matches: int = 12,
    maximum_rms_arcsec: float = 2.0,
    durable: bool = True,
) -> FilterMosaic:
    """Match, blend and verify one filter's panels on the canvas box."""

    edges: list[EdgeMeasurement] = []
    for left_index in range(len(panels)):
        for right_index in range(left_index + 1, len(panels)):
            edge = measure_edge(panels[left_index], panels[right_index])
            if edge is not None:
                edges.append(edge)
    try:
        scales = solve_scales(panels, edges)
    except MosaicPhotometryError as error:
        raise MosaicGateError(error.code, str(error)) from error
    x0, y0, x1, y1 = canvas_box
    planes = solve_planes(panels, edges, scales, (y1 - y0, x1 - x0))
    noises = {panel.key: panel_noise(panel) for panel in panels}
    widths: dict[str, float] = {}
    for edge in edges:
        if edge.valid_pixels <= 0:
            continue
        for key in (edge.left, edge.right):
            widths[key] = min(widths.get(key, math.inf), float(edge.width))
    seeing: dict[str, list[float]] = {}
    for edge in edges:
        for key, value in ((edge.left, edge.fwhm_left), (edge.right, edge.fwhm_right)):
            if value > 0:
                seeing.setdefault(key, []).append(value)
    # A panel without measured overlap stars takes the widest seeing seen.
    widest = max((max(values) for values in seeing.values()), default=DEFAULT_FWHM_PIXELS)
    fwhm = {panel.key: float(np.median(seeing[panel.key])) if panel.key in seeing else widest for panel in panels}
    reference = next(panel for panel in panels if panel.key == scales.reference)
    header: dict[str, Any] = {
        **projection.header((x0, y0)),
        "IMAGETYP": "Master Light",
        "FILTER": filter_name,
        "EXPTIME": reference.exposure_seconds,
        "OAFSTATE": "UNSOLVED_WORKING",
        "OAFWCS": "CANVAS",
        "OAFMOSAI": CANVAS_MOSAIC_VERSION,
        "OAFNPANL": len(panels),
    }
    science_path = output_directory / f"mosaic_{filter_name}.fits"
    noise_path = output_directory / f"mosaic_{filter_name}_noise.fits"
    coverage_path = output_directory / f"mosaic_{filter_name}_coverage.fits"
    mask_path = output_directory / f"mosaic_{filter_name}_mask.fits"
    blend = blend_filter(
        panels,
        scales,
        planes,
        noises=noises,
        fwhm=fwhm,
        overlap_widths=widths,
        canvas_box=canvas_box,
        header=header,
        science_path=science_path,
        noise_path=noise_path,
        coverage_path=coverage_path,
        mask_path=mask_path,
        durable=durable,
    )
    photometry_records = []
    for record in scales.edges:
        deviation = abs(record["correctedRatioMinusOne"])
        photometry_records.append(
            {
                **record,
                "photometryVerdict": _verdict(deviation, PHOTOMETRY_PASS, PHOTOMETRY_WARN),
                "astrometryVerdict": _verdict(record["astrometricOffsetRmsPixels"], ASTROMETRY_PASS_PIXELS, ASTROMETRY_WARN_PIXELS),
            }
        )
    residuals = _residual_records(edges, scales, planes)
    flux = _flux_conservation(science_path, canvas_box, edges, scales, planes)
    unmeasured = [
        {"left": edge.left, "right": edge.right, "stars": int(edge.flux_left.size)}
        for edge in edges
        if edge.valid_pixels > 0 and edge.flux_left.size < 8
    ]
    verification: list[dict[str, Any]] = []
    quality: dict[str, Any] | None = None
    if verifier is not None:
        for panel in panels:
            crop = _canvas_crop(science_path, canvas_box, panel.box)
            artifacts = output_directory / f"verification-{panel.key}"
            artifacts.mkdir()
            try:
                result = dict(verifier(crop, projection.header(panel.origin), artifacts))
            except Exception as error:  # a verifier failure is a failed gate, with its reason
                result = {"error": f"{getattr(error, 'code', type(error).__name__)}: {error}"}
            matched = int(result.get("matchedStars", 0) or 0)
            rms = float(result.get("rmsArcsec", math.inf) or math.inf)
            result["verdict"] = (
                "PASS"
                if "error" not in result and matched >= minimum_matches and rms <= maximum_rms_arcsec
                else "FAIL"
            )
            verification.append({"panel": panel.key, **result})
        verified = [item for item in verification if item["verdict"] == "PASS"]
        if verified:
            worst = max(verified, key=lambda item: float(item.get("rmsArcsec", math.inf)))
            quality = {key: value for key, value in worst.items() if key not in ("panel", "verdict")}
    gates = {
        "photometry": _worst([item["photometryVerdict"] for item in photometry_records]) if photometry_records else "PASS",
        "overlapAstrometry": _worst([item["astrometryVerdict"] for item in photometry_records]) if photometry_records else "PASS",
        "residualBackground": _worst([item["verdict"] for item in residuals]) if residuals else "PASS",
        "fluxConservation": flux["verdict"],
        "photometricNetwork": "WARN" if (scales.chi2_per_dof or 0.0) > 9.0 else "PASS",
        "unmeasuredOverlaps": "WARN" if unmeasured else "PASS",
        "canvasWcs": (
            _worst([item["verdict"] for item in verification]) if verifier is not None else "NOT_MEASURED"
        ),
    }
    receipt = {
        "schemaVersion": 1,
        "version": CANVAS_MOSAIC_VERSION,
        "filter": filter_name,
        "canvasBox": list(canvas_box),
        "panels": [
            {
                "key": panel.key,
                "origin": list(panel.origin),
                "shape": list(panel.shape),
                "scale": scales.scales[panel.key],
                "logScaleError": scales.log_scale_errors[panel.key],
                "plane": list(planes.coefficients[panel.key]),
                "noise": noises[panel.key],
                "fwhmPixels": fwhm[panel.key],
                "exposureSeconds": panel.exposure_seconds,
            }
            for panel in panels
        ],
        "photometry": {
            "algorithm": PHOTOMETRY_ALGORITHM,
            "scaleReference": scales.reference,
            "chi2PerDof": scales.chi2_per_dof,
            "edges": photometry_records,
            "unmeasuredOverlaps": unmeasured,
        },
        "background": {"algorithm": BACKGROUND_ALGORITHM, **planes.diagnostics, "residuals": residuals},
        "blend": blend,
        "fluxConservation": flux,
        "canvasWcsVerification": verification,
        "gates": gates,
        "verdict": _worst([value for value in gates.values() if value != "NOT_MEASURED"]),
    }
    failed = [name for name, value in gates.items() if value == "FAIL"]
    if failed:
        raise MosaicGateError(
            "MOSAIC_GATE_FAILED",
            f"{filter_name} mosaic failed: {', '.join(failed)}",
            receipt,
        )
    if verifier is not None:
        # Verified against the catalog: the canvas WCS is the solution.
        with fits.open(science_path, mode="update", memmap=False) as hdul:
            hdul[0].header["OAFSTATE"] = "SOLVED"
            hdul[0].header["OAFWCS"] = "SOLVED"
            hdul[0].header["OAFWCSPR"] = ("CANVAS_CATALOG_VERIFIED", "WCS provenance of this grid")
        receipt["blend"]["sha256"]["science"] = sha256_digest(science_path)
    return FilterMosaic(science_path, noise_path, coverage_path, mask_path, receipt, quality)


__all__ = ["CANVAS_MOSAIC_VERSION", "FilterMosaic", "MosaicGateError", "assemble_filter"]
