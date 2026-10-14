"""One filter's mosaic from panel masters that already share the canvas.

The panels are matched (:mod:`.photometry`), blended (:mod:`.blend`) and
then judged.  Every gate is PASS, WARN, FAIL or NOT_MEASURED with its
numbers in the receipt; a FAIL stops the mosaic (``MosaicGateError``):

* photometry: after correction, every overlap's star flux ratio is within
  0.3 % and known to be within 1 % at two standard errors (PASS); it fails
  only beyond 1 % by more than two standard errors;
* astrometry of the overlaps: the systematic star offset between the two
  panels (binned along the overlap, less the centroid noise) in units of
  the larger FWHM (0.05 PASS, 0.10 WARN);
* the residual background difference of every overlap after the panels'
  polynomials, against the per-pixel noise of the blend (0.1 PASS, 0.3 WARN);
* flux conservation: the stars of the overlaps have the same flux in the
  mosaic as in the corrected panels (0.5 % PASS, 2 % WARN);
* the photometric network's chi-square, and overlaps without a scale (WARN);
* the canvas WCS, verified against catalog stars on every panel's window
  of the finished mosaic (the mosaic is never blind-solved: its WCS is the
  canvas the panels were resampled onto, and the catalog decides).

Panels whose overlaps give no scale (too few common stars) fall into
separate photometric components.  Each component is matched, blended and
gated on its own with its own gauge: the largest is the filter's mosaic on
the whole canvas box, the others are published beside it as
``component<k>`` on their own bounding box with their own receipts, and the
filter is WARN (``MOSAIC_PHOTOMETRIC_COMPONENTS``).
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
    MINIMUM_OVERLAP_STARS,
    PHOTOMETRY_ALGORITHM,
    SCALE_LEAK_FRACTION,
    EdgeMeasurement,
    MosaicPhotometryError,
    PanelImage,
    PlaneSolution,
    ScaleObservation,
    ScaleSolution,
    measure_edge,
    photometric_components,
    scale_observations,
    solve_planes,
    solve_scales,
)


CANVAS_MOSAIC_VERSION = "ultra-fast-wbpp-canvas-mosaic-v4"
PHOTOMETRY_PASS = 0.003
PHOTOMETRY_WARN = 0.01
# An overlap fails photometry only when its corrected ratio lies beyond the
# WARN limit by more than this many standard errors: a corner overlap of 35
# stars measures the ratio to about 0.7 %, so its point estimate alone
# cannot tell 1 % from noise.
PHOTOMETRY_FAIL_SIGNIFICANCE = 2.0
# Systematic overlap misregistration in units of the overlap's larger FWHM:
# two equal Gaussians offset by 0.1 FWHM blend into one 0.7 % wider along
# the offset.
ASTROMETRY_PASS_FWHM = 0.05
ASTROMETRY_WARN_FWHM = 0.10
# Residual sky difference of an overlap against the per-pixel noise of the
# blended mosaic there.  The blend fades a panel in over up to 256 px, so a
# remaining difference becomes a smooth ramp; at a tenth of a pixel's noise
# it is invisible in any stretch that keeps the sky noise visible.
RESIDUAL_PASS_PIXEL_SIGMA = 0.1
RESIDUAL_WARN_PIXEL_SIGMA = 0.3
# An overlap's residual background is measured on at least this many bins
# free of bright structure.
MINIMUM_RESIDUAL_BINS = 8
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
    # The filter's other photometric components, largest first: each a
    # mosaic of its own panels on the same canvas box.
    components: tuple["FilterMosaic", ...] = ()


def _verdict(value: float, passing: float, warning: float) -> str:
    if not math.isfinite(value):
        return "FAIL"
    return "PASS" if value <= passing else "WARN" if value <= warning else "FAIL"


def _photometry_verdict(deviation: float | None, error: float | None) -> str:
    """PASS when the corrected ratio is within PHOTOMETRY_PASS and known to
    be within PHOTOMETRY_WARN at PHOTOMETRY_FAIL_SIGNIFICANCE standard
    errors; FAIL when it lies beyond PHOTOMETRY_WARN by that many."""

    if deviation is None or error is None or not (math.isfinite(deviation) and math.isfinite(error)):
        return "NOT_MEASURED"
    if deviation - PHOTOMETRY_FAIL_SIGNIFICANCE * error > PHOTOMETRY_WARN:
        return "FAIL"
    if deviation <= PHOTOMETRY_PASS and deviation + PHOTOMETRY_FAIL_SIGNIFICANCE * error <= PHOTOMETRY_WARN:
        return "PASS"
    return "WARN"


def _worst(verdicts: Sequence[str]) -> str:
    return "FAIL" if "FAIL" in verdicts else "WARN" if "WARN" in verdicts else "PASS"


def _measured(verdicts: Sequence[str]) -> str:
    """The worst of the measured verdicts; NOT_MEASURED when none was."""

    measured = [item for item in verdicts if item != "NOT_MEASURED"]
    return _worst(measured) if measured else "NOT_MEASURED"


def _residual_records(
    edges: Sequence[EdgeMeasurement],
    scales: ScaleSolution,
    planes: PlaneSolution,
    noises: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Per overlap: the binned difference left after gains and planes;
    ``noises`` is each panel's per-pixel background noise."""

    records = []
    for edge in edges:
        if edge.bin_x.size < MINIMUM_RESIDUAL_BINS:
            records.append(
                {
                    "left": edge.left,
                    "right": edge.right,
                    "bins": int(edge.bin_x.size),
                    "verdict": "NOT_MEASURED",
                    "reason": f"fewer than {MINIMUM_RESIDUAL_BINS} background bins",
                }
            )
            continue
        ci = scales.gain(edge.left, edge.bin_x, edge.bin_y)
        cj = scales.gain(edge.right, edge.bin_x, edge.bin_y)
        difference = ci * edge.bin_left - cj * edge.bin_right
        difference -= planes.evaluate(edge.left, edge.bin_x, edge.bin_y) - planes.evaluate(edge.right, edge.bin_x, edge.bin_y)
        noise = np.hypot(ci * edge.bin_noise, cj * edge.bin_noise)
        structure = 0.5 * (ci * edge.bin_left + cj * edge.bin_right)
        relative_error = scales.ratio_error(edge.left, edge.right, float(np.mean(edge.bin_x)), float(np.mean(edge.bin_y)))
        faint = np.abs(structure - np.median(structure)) * relative_error <= SCALE_LEAK_FRACTION * noise
        if faint.sum() < MINIMUM_RESIDUAL_BINS:
            records.append(
                {
                    "left": edge.left,
                    "right": edge.right,
                    "bins": int(faint.sum()),
                    "verdict": "NOT_MEASURED",
                    "reason": f"fewer than {MINIMUM_RESIDUAL_BINS} bins free of bright structure",
                }
            )
            continue
        weight = 1.0 / np.square(noise[faint])
        mean = float(np.sum(weight * difference[faint]) / np.sum(weight))
        typical = float(np.median(noise[faint]))
        # The per-pixel noise of the inverse-variance blend of both panels.
        left_noise = float(np.median(ci)) * noises[edge.left]
        right_noise = float(np.median(cj)) * noises[edge.right]
        pixel = 1.0 / math.sqrt(left_noise**-2 + right_noise**-2) if left_noise > 0 and right_noise > 0 else 0.0
        level = abs(mean) / pixel if pixel > 0 else math.inf
        sky = float(np.median(structure[faint]))
        records.append(
            {
                "left": edge.left,
                "right": edge.right,
                "bins": int(faint.sum()),
                "meanDifference": mean,
                "meanDifferenceError": float(1.0 / math.sqrt(float(np.sum(weight)))),
                "levelBinSigma": abs(mean) / typical if typical > 0 else math.inf,
                "levelPixelSigma": level,
                "levelSkyFraction": abs(mean) / sky if sky > 0 else None,
                "verdict": _verdict(level, RESIDUAL_PASS_PIXEL_SIGMA, RESIDUAL_WARN_PIXEL_SIGMA),
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
        expected = 0.5 * (
            scales.gain(edge.left, edge.star_x, edge.star_y) * edge.flux_left
            + scales.gain(edge.right, edge.star_x, edge.star_y) * edge.flux_right
        )
        usable = (np.asarray(flags) == 0) & (expected > 0) & np.isfinite(flux)
        ratios.extend((flux[usable] / expected[usable]).tolist())
    if not ratios:
        return {"stars": 0, "verdict": "NOT_MEASURED", "reason": "no overlap stars"}
    deviation = float(np.median(ratios)) - 1.0
    return {
        "stars": len(ratios),
        "medianRatioMinusOne": deviation,
        "verdict": _verdict(abs(deviation), FLUX_PASS, FLUX_WARN),
    }


def _mark_solved(science_path: Path, receipt: dict[str, Any]) -> None:
    """Verified against the catalog: the canvas WCS is the solution."""

    with fits.open(science_path, mode="update", memmap=False) as hdul:
        hdul[0].header["OAFSTATE"] = "SOLVED"
        hdul[0].header["OAFWCS"] = "SOLVED"
        hdul[0].header["OAFWCSPR"] = ("CANVAS_CATALOG_VERIFIED", "WCS provenance of this grid")
    receipt["blend"]["sha256"]["science"] = sha256_digest(science_path)


def _assemble_component(
    filter_name: str,
    panels: Sequence[PanelImage],
    edges: Sequence[EdgeMeasurement],
    observations: Mapping[tuple[str, str], ScaleObservation],
    *,
    component: int,
    component_count: int,
    projection: CanvasProjection,
    canvas_box: tuple[int, int, int, int],
    directory: Path,
    stem: str,
    verifier: Verifier | None,
    minimum_matches: int,
    maximum_rms_arcsec: float,
    durable: bool,
) -> FilterMosaic:
    """Match, blend and gate one photometric component; the receipt holds
    its gates and verdict, nothing is raised."""

    try:
        scales = solve_scales(panels, observations)
    except MosaicPhotometryError as error:
        raise MosaicGateError(error.code, f"{filter_name}: " + str(error).removeprefix(f"{error.code}: ")) from error
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
        "OAFCOMP": component,
        "OAFNCOMP": component_count,
    }
    science_path = directory / f"{stem}.fits"
    noise_path = directory / f"{stem}_noise.fits"
    coverage_path = directory / f"{stem}_coverage.fits"
    mask_path = directory / f"{stem}_mask.fits"
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
        deviation = record["correctedRatioMinusOne"]
        photometry_records.append(
            {
                **record,
                "photometryVerdict": _photometry_verdict(
                    None if deviation is None else abs(deviation), record["correctedRatioError"]
                ),
                "astrometryVerdict": _verdict(
                    record["astrometricSystematicPixels"] / (record["fwhmPixels"] or DEFAULT_FWHM_PIXELS),
                    ASTROMETRY_PASS_FWHM,
                    ASTROMETRY_WARN_FWHM,
                ),
            }
        )
    residuals = _residual_records(edges, scales, planes, noises)
    flux = _flux_conservation(science_path, canvas_box, edges, scales)
    # Overlaps that give no scale, or whose corrected ratio has no error.
    unmeasured = [
        {
            "left": edge.left,
            "right": edge.right,
            "stars": int(edge.flux_left.size),
            "reason": (
                f"fewer than {MINIMUM_OVERLAP_STARS} common stars"
                if edge.flux_left.size < MINIMUM_OVERLAP_STARS
                else "no finite flux ratio"
            ),
        }
        for edge in edges
        if edge.valid_pixels > 0 and (edge.left, edge.right) not in observations
    ] + [
        {"left": item["left"], "right": item["right"], "stars": item["stars"], "reason": "corrected ratio not measured"}
        for item in photometry_records
        if item["photometryVerdict"] == "NOT_MEASURED"
    ]
    verification: list[dict[str, Any]] = []
    quality: dict[str, Any] | None = None
    if verifier is not None:
        for panel in panels:
            crop = _canvas_crop(science_path, canvas_box, panel.box)
            artifacts = directory / f"verification-{panel.key}"
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
    overlaps = [edge for edge in edges if edge.valid_pixels > 0]
    gates = {
        "photometry": _measured([item["photometryVerdict"] for item in photometry_records]),
        "overlapAstrometry": _measured([item["astrometryVerdict"] for item in photometry_records]),
        "residualBackground": _measured([item["verdict"] for item in residuals]),
        "fluxConservation": flux["verdict"],
        "photometricNetwork": (
            "NOT_MEASURED" if scales.chi2_per_dof is None else "WARN" if scales.chi2_per_dof > 9.0 else "PASS"
        ),
        "unmeasuredOverlaps": "NOT_MEASURED" if not overlaps else "WARN" if unmeasured else "PASS",
        "canvasWcs": (
            _worst([item["verdict"] for item in verification]) if verifier is not None else "NOT_MEASURED"
        ),
    }
    receipt = {
        "schemaVersion": 1,
        "version": CANVAS_MOSAIC_VERSION,
        "filter": filter_name,
        "component": component,
        "componentCount": component_count,
        "canvasBox": list(canvas_box),
        "panels": [
            {
                "key": panel.key,
                "origin": list(panel.origin),
                "shape": list(panel.shape),
                "scale": scales.scales[panel.key],
                "gainGradient": list(scales.gradients[panel.key]),
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
            "gainModel": scales.model,
            "gainModelBic": scales.criterion,
            "gainCentre": list(scales.centre),
            "gainLength": scales.length,
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
    return FilterMosaic(science_path, noise_path, coverage_path, mask_path, receipt, quality)


def _failed_gates(receipt: Mapping[str, Any]) -> list[str]:
    return [name for name, value in receipt["gates"].items() if value == "FAIL"]


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
    """Match, blend and verify one filter's panels on the canvas box.

    The largest photometric component is the filter's mosaic
    (``mosaic_<F>.fits`` in ``output_directory``); every other component is
    assembled in ``component<k>/`` as ``mosaic_<F>_component<k>.fits`` with
    its own receipt.  A FAIL in any component fails the filter, after every
    component was gated.
    """

    edges: list[EdgeMeasurement] = []
    for left_index in range(len(panels)):
        for right_index in range(left_index + 1, len(panels)):
            edge = measure_edge(panels[left_index], panels[right_index])
            if edge is not None:
                edges.append(edge)
    observations = scale_observations(edges)
    components = photometric_components(panels, observations)
    results: list[FilterMosaic] = []
    for number, members in enumerate(components, start=1):
        keys = {panel.key for panel in members}
        inner = [edge for edge in edges if edge.left in keys and edge.right in keys]
        if number == 1:
            directory, stem, box = output_directory, f"mosaic_{filter_name}", canvas_box
        else:
            # Another component covers only its own panels: an integer
            # sub-window of the same canvas lattice.
            directory, stem = output_directory / f"component{number}", f"mosaic_{filter_name}_component{number}"
            directory.mkdir()
            boxes = [panel.box for panel in members]
            box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        results.append(
            _assemble_component(
                filter_name,
                members,
                inner,
                observations,
                component=number,
                component_count=len(components),
                projection=projection,
                canvas_box=box,
                directory=directory,
                stem=stem,
                verifier=verifier,
                minimum_matches=minimum_matches,
                maximum_rms_arcsec=maximum_rms_arcsec,
                durable=durable,
            )
        )
    main, others = results[0], results[1:]
    receipt = main.receipt
    if others:
        membership = {panel.key: number for number, members in enumerate(components, start=1) for panel in members}
        receipt["components"] = [other.receipt for other in others]
        receipt["gates"]["photometricComponents"] = "WARN"
        receipt["warnings"] = [
            {
                "code": "MOSAIC_PHOTOMETRIC_COMPONENTS",
                "message": (
                    f"the {filter_name} panels fall into {len(components)} photometric components; the largest is "
                    f"the mosaic, the others are published as components"
                ),
                "components": [[panel.key for panel in members] for members in components],
                "unmatchedOverlaps": [
                    {"left": edge.left, "right": edge.right, "stars": int(edge.flux_left.size), "bins": int(edge.bin_x.size)}
                    for edge in edges
                    if membership[edge.left] != membership[edge.right]
                ],
            }
        ]
        receipt["verdict"] = _worst([result.receipt["verdict"] for result in results] + ["WARN"])
    failed = {result.receipt["component"]: _failed_gates(result.receipt) for result in results}
    failed = {number: gates for number, gates in failed.items() if gates}
    if failed:
        raise MosaicGateError(
            "MOSAIC_GATE_FAILED",
            f"{filter_name} mosaic failed: "
            + "; ".join(
                ", ".join(gates) + (f" (component {number})" if len(results) > 1 else "") for number, gates in failed.items()
            ),
            receipt,
        )
    if verifier is not None:
        for result in results:
            _mark_solved(result.science_path, result.receipt)
    return FilterMosaic(
        main.science_path,
        main.noise_path,
        main.coverage_path,
        main.mask_path,
        receipt,
        main.quality,
        tuple(others),
    )


__all__ = ["CANVAS_MOSAIC_VERSION", "FilterMosaic", "MosaicGateError", "assemble_filter"]
