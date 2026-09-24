"""Final plate-solving coordination, WCS agreement and solved-state verification."""

from __future__ import annotations
from ..integrity import sha256_digest
from .contracts import IntegrationMode, E2EError


from ..platform import remove_file

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


from ..solvers.process import (
    verify_solver_execution_result,
)
from ..solver import (
    SolveRequest,
    SolverBackend,
    SolverResult,
    WcsValidation,
    validate_solver_result,
    validate_wcs_header,
    wcs_parity,
)




@dataclass(frozen=True, slots=True)
class _SolverHints:
    ra_degrees: float | None
    dec_degrees: float | None
    field_of_view_degrees: float | None
    search_radius_degrees: float | None
    provenance: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "raDegrees": self.ra_degrees,
            "decDegrees": self.dec_degrees,
            "fieldOfViewDegrees": self.field_of_view_degrees,
            "searchRadiusDegrees": self.search_radius_degrees,
            "provenance": self.provenance,
            "evidence": self.evidence,
        }


def _read_image_header(path: Path) -> tuple[fits.Header, tuple[int, int]]:
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            lazy_load_hdus=True,
            do_not_scale_image_data=True,
            uint=False,
            checksum=False,
        ) as hdul:
            hdu = next(
                (
                    item
                    for item in hdul
                    if int(item.header.get("NAXIS", 0) or 0) == 2 and item.data is not None
                ),
                None,
            )
            if hdu is None or hdu.data is None:
                raise E2EError("SOLVED_OUTPUT_INVALID", "no two-dimensional FITS image", path=str(path))
            shape = tuple(int(value) for value in hdu.data.shape)
            if len(shape) != 2:
                raise E2EError("SOLVED_OUTPUT_INVALID", "image is not two-dimensional", path=str(path))
            return hdu.header.copy(), (shape[0], shape[1])
    except E2EError:
        raise
    except Exception as error:
        raise E2EError("SOLVED_OUTPUT_INVALID", str(error), path=str(path)) from error


def _wcs_headers_agree(
    result_header: Mapping[str, Any] | fits.Header,
    output_header: fits.Header,
    shape: tuple[int, int],
) -> bool:
    try:
        left = WCS(result_header).celestial
        right = WCS(output_header).celestial
        height, width = shape
        pixels = np.asarray(
            [[0.0, 0.0], [(width - 1) / 2.0, (height - 1) / 2.0], [width - 1.0, height - 1.0]],
            dtype=np.float64,
        )
        left_world = left.all_pix2world(pixels, 0)
        right_pixels = right.all_world2pix(left_world, 0)
        return bool(np.all(np.isfinite(right_pixels)) and np.max(np.abs(right_pixels - pixels)) <= 0.05)
    except Exception:
        return False


def _unit_vectors(world_degrees: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(world_degrees[:, 0])
    dec = np.deg2rad(world_degrees[:, 1])
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def _sky_separation_degrees(left: tuple[float, float], right: tuple[float, float]) -> float:
    vectors = _unit_vectors(np.asarray((left, right), dtype=np.float64))
    chord = float(np.linalg.norm(vectors[0] - vectors[1]))
    return float(np.rad2deg(2.0 * math.asin(min(1.0, max(0.0, chord / 2.0)))))


def _solution_geometry(
    header: Mapping[str, Any] | fits.Header,
    shape: tuple[int, int],
) -> dict[str, Any]:
    height, width = shape
    celestial = WCS(header, relax=False).celestial
    center_pixel = np.asarray([[(width - 1.0) / 2.0, (height - 1.0) / 2.0]])
    horizontal = np.asarray(
        [[0.0, (height - 1.0) / 2.0], [width - 1.0, (height - 1.0) / 2.0]],
        dtype=np.float64,
    )
    vertical = np.asarray(
        [[(width - 1.0) / 2.0, 0.0], [(width - 1.0) / 2.0, height - 1.0]],
        dtype=np.float64,
    )
    center_world = celestial.all_pix2world(center_pixel, 0)[0]
    horizontal_world = celestial.all_pix2world(horizontal, 0)
    vertical_world = celestial.all_pix2world(vertical, 0)
    matrix = np.asarray(celestial.pixel_scale_matrix, dtype=np.float64)
    column_scales = np.sqrt(np.sum(np.square(matrix), axis=0)) * 3600.0
    if (
        not np.all(np.isfinite(center_world))
        or not np.all(np.isfinite(horizontal_world))
        or not np.all(np.isfinite(vertical_world))
        or not np.all(np.isfinite(column_scales))
        or np.any(column_scales <= 0)
    ):
        raise ValueError("the WCS geometry is non-finite")
    return {
        "centerRaDegrees": float(center_world[0] % 360.0),
        "centerDecDegrees": float(center_world[1]),
        "fieldWidthDegrees": _sky_separation_degrees(
            tuple(float(value) for value in horizontal_world[0]),
            tuple(float(value) for value in horizontal_world[1]),
        ),
        "fieldHeightDegrees": _sky_separation_degrees(
            tuple(float(value) for value in vertical_world[0]),
            tuple(float(value) for value in vertical_world[1]),
        ),
        "pixelScaleArcsec": float(math.sqrt(float(column_scales[0] * column_scales[1]))),
        "axisPixelScalesArcsec": [float(column_scales[0]), float(column_scales[1])],
        "pixelAspectRatio": float(max(column_scales) / min(column_scales)),
        "rotationDegrees": float(
            math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
        ),
        "parity": wcs_parity(header).value,
        "imageShape": [height, width],
    }


def _validate_solution_against_hints(
    result: SolverResult,
    hints: _SolverHints,
) -> WcsValidation:
    try:
        if result.image_shape is None:
            raise ValueError("solver result omitted image geometry")
        geometry = _solution_geometry(result.header, result.image_shape)
    except Exception as error:
        return WcsValidation(False, "SOLVER_GEOMETRY_INVALID", str(error))
    diagnostics: dict[str, Any] = {"solution": geometry, "hints": hints.serializable()}
    if hints.ra_degrees is not None and hints.dec_degrees is not None:
        separation = _sky_separation_degrees(
            (geometry["centerRaDegrees"], geometry["centerDecDegrees"]),
            (hints.ra_degrees, hints.dec_degrees),
        )
        effective_radius = hints.search_radius_degrees if hints.search_radius_degrees is not None else 15.0
        diagnostics["centerToHintDegrees"] = separation
        diagnostics["effectiveSearchRadiusDegrees"] = effective_radius
        if separation > effective_radius:
            return WcsValidation(
                False,
                "SOLVER_CENTER_OUTSIDE_HINT_RADIUS",
                f"solved center is {separation:.6g} deg from the NINA/request hint, outside {effective_radius:.6g} deg",
                diagnostics,
            )
    if hints.field_of_view_degrees is not None:
        solved_width = float(geometry["fieldWidthDegrees"])
        height, width = (int(value) for value in geometry["imageShape"])
        ratio = solved_width / hints.field_of_view_degrees
        expected_height = hints.field_of_view_degrees * max(height - 1, 1) / max(width - 1, 1)
        height_ratio = float(geometry["fieldHeightDegrees"]) / expected_height
        hinted_scale = hints.field_of_view_degrees * 3600.0 / max(width - 1, 1)
        scale_ratio = float(geometry["pixelScaleArcsec"]) / hinted_scale
        diagnostics.update(
            {
                "fieldWidthToHintRatio": ratio,
                "fieldHeightToHintRatio": height_ratio,
                "hintDerivedPixelScaleArcsec": hinted_scale,
                "pixelScaleToHintRatio": scale_ratio,
            }
        )
        # solve-field itself is constrained to 0.8--1.2 of this hint.  The
        # slightly wider independent gate allows projection/corner-definition
        # differences without permitting a different scale solution.
        if not (
            0.7 <= ratio <= 1.3
            and 0.7 <= height_ratio <= 1.3
            and 0.7 <= scale_ratio <= 1.3
            and geometry["pixelAspectRatio"] <= 1.2
        ):
            return WcsValidation(
                False,
                "SOLVER_SCALE_OUTSIDE_HINT",
                "solved field width/pixel scale is inconsistent with the NINA or derived FOV hint",
                diagnostics,
            )
    return WcsValidation(
        True,
        "SOLVER_HINTS_VALID",
        "solved center, field of view, and pixel scale agree with acquisition hints",
        diagnostics,
    )


_WCS_CARD_PATTERN = re.compile(
    r"^(WCSAXES|CRPIX\d+|CRVAL\d+|CDELT\d+|CUNIT\d+|CTYPE\d+|CD\d+_\d+|PC\d+_\d+"
    r"|CROTA\d+|LONPOLE|LATPOLE|RADESYS|EQUINOX|MJDREF|MJDREFI|MJDREFF"
    r"|A_ORDER|B_ORDER|AP_ORDER|BP_ORDER|A_\d+_\d+|B_\d+_\d+|AP_\d+_\d+|BP_\d+_\d+)$"
)


def _grid_sample_points(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
            [(width - 1.0) / 2.0, 0.0],
            [(width - 1.0) / 2.0, height - 1.0],
            [0.0, (height - 1.0) / 2.0],
            [width - 1.0, (height - 1.0) / 2.0],
            [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
        ],
        dtype=np.float64,
    )


def _wcs_grid_disagreement(
    left: fits.Header, right: fits.Header, shape: tuple[int, int]
) -> float | None:
    """Maximum pixel disagreement of two solutions of one grid, both ways."""

    try:
        left_wcs = WCS(left, relax=False).celestial
        right_wcs = WCS(right, relax=False).celestial
    except Exception:
        return None
    samples = _grid_sample_points(shape)
    in_right = right_wcs.all_world2pix(left_wcs.all_pix2world(samples, 0), 0)
    in_left = left_wcs.all_world2pix(right_wcs.all_pix2world(samples, 0), 0)
    if not (np.all(np.isfinite(in_right)) and np.all(np.isfinite(in_left))):
        return None
    return float(
        max(np.max(np.abs(in_right - samples)), np.max(np.abs(in_left - samples)))
    )


def _accepted_solve_rms_pixels(record: Mapping[str, Any]) -> float:
    attempts = record.get("attempts")
    if not isinstance(attempts, list):
        return math.inf
    for attempt in reversed(attempts):
        if isinstance(attempt, Mapping) and attempt.get("accepted") is True:
            quality = attempt.get("result", {}).get("astrometricQuality", {})
            value = quality.get("rmsPixels") if isinstance(quality, Mapping) else None
            try:
                rms = float(value)
            except (TypeError, ValueError):
                return math.inf
            return rms if math.isfinite(rms) else math.inf
    return math.inf


def _same_grid_signatures(
    *,
    integration_mode: IntegrationMode,
    pixel_pipeline_receipt: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, list[int]]:
    """Describe the pixel grid each filter master occupies, by construction.

    Ordinary masters share their grid when they were cropped to one rectangle
    of the shared reference frame; drizzled masters share it when the drizzle
    placed them on the same scaled reference grid.  Filters without evidence
    are left out, which keeps the unification from touching them.
    """

    signatures: dict[str, list[int]] = {}
    if integration_mode is IntegrationMode.DRIZZLE:
        filters = coverage.get("filters", {})
        if not isinstance(filters, Mapping):
            return signatures
        for filter_name, receipt in filters.items():
            if not isinstance(receipt, Mapping):
                continue
            geometry = receipt.get("geometry", {})
            recipe = receipt.get("recipe", {})
            if not isinstance(geometry, Mapping) or not isinstance(recipe, Mapping):
                continue
            values = [
                recipe.get("scale"),
                geometry.get("referenceHeight"),
                geometry.get("referenceWidth"),
                geometry.get("outputHeight"),
                geometry.get("outputWidth"),
            ]
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                signatures[str(filter_name)] = [int(value) for value in values]
        return signatures
    groups = pixel_pipeline_receipt.get("statistics", {}).get("integrationGroups", {})
    if not isinstance(groups, Mapping):
        return signatures
    for filter_name, group in groups.items():
        crop = group.get("crop") if isinstance(group, Mapping) else None
        if isinstance(crop, list) and len(crop) == 4:
            signatures[str(filter_name)] = [int(value) for value in crop]
    return signatures


def _unify_same_grid_solutions(
    products: Mapping[str, Path],
    *,
    solver_records: Mapping[str, Any],
    tolerance_pixels: float,
    pixel_pipeline_receipt: Mapping[str, Any] | None = None,
    grid_signatures: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Share one fresh solve between masters that occupy one pixel grid.

    The pixel pipeline registers every filter of a run onto the same reference
    frame and crops all ordinary masters to one common rectangle, and the
    drizzle places every filter on the same scaled reference grid, so the
    masters are the same grid by construction.  Each still received its own
    fresh solve; those independent solutions verify the shared grid at solver
    precision, and the lowest-RMS solution is then written to every master so
    the products describe one sky mapping exactly.  Pixel values are untouched.
    Runs whose masters do not share their grid are left unchanged.  The grid
    evidence comes from ``grid_signatures`` (see ``_same_grid_signatures``) or,
    for ordinary masters, from the pixel pipeline receipt's crops.
    """

    if grid_signatures is None:
        grid_signatures = _same_grid_signatures(
            integration_mode=IntegrationMode.ORDINARY,
            pixel_pipeline_receipt=pixel_pipeline_receipt or {},
            coverage={},
        )
    signatures: dict[str, tuple[int, ...]] = {}
    headers: dict[str, fits.Header] = {}
    shapes: dict[str, tuple[int, int]] = {}
    for filter_name, path in sorted(products.items()):
        signature = grid_signatures.get(filter_name)
        if not isinstance(signature, (list, tuple)) or not signature:
            return {"status": "NOT_APPLICABLE", "reason": f"{filter_name} has no grid evidence"}
        signatures[filter_name] = tuple(int(value) for value in signature)
        header, shape = _read_image_header(path)
        headers[filter_name] = header
        shapes[filter_name] = shape
    if len(set(signatures.values())) != 1 or len(set(shapes.values())) != 1:
        return {
            "status": "NOT_APPLICABLE",
            "reason": "filter masters do not share one registration grid",
            "grids": {name: list(value) for name, value in signatures.items()},
            "shapes": {name: list(value) for name, value in shapes.items()},
        }
    shape = next(iter(shapes.values()))
    adopted = min(
        sorted(products),
        key=lambda name: (
            _accepted_solve_rms_pixels(solver_records.get(name, {})),
            name != "L",
            name,
        ),
    )
    adopted_header = headers[adopted]
    adopted_cards = [card for card in adopted_header.cards if _WCS_CARD_PATTERN.match(card.keyword)]
    adopted_rms = _accepted_solve_rms_pixels(solver_records.get(adopted, {}))
    record: dict[str, Any] = {
        "status": "APPLIED",
        "adoptedFilter": adopted,
        "adoptedRmsPixels": adopted_rms,
        "tolerancePixels": tolerance_pixels,
        "grid": list(signatures[adopted]),
        "imageShape": list(shape),
        "filters": {},
    }
    for filter_name, path in sorted(products.items()):
        own_header = headers[filter_name]
        disagreement = _wcs_grid_disagreement(own_header, adopted_header, shape)
        own_rms = _accepted_solve_rms_pixels(solver_records.get(filter_name, {}))
        # Two fresh solves of one grid differ by a fraction of their own
        # catalogue RMS; the gate only has to reject a genuinely different
        # grid (dither-scale offsets, rotations, mirrored axes).
        effective_tolerance = max(
            tolerance_pixels,
            2.0 * max(
                rms for rms in (own_rms, adopted_rms) if math.isfinite(rms)
            ) if any(math.isfinite(rms) for rms in (own_rms, adopted_rms)) else tolerance_pixels,
        )
        entry: dict[str, Any] = {
            "ownRmsPixels": own_rms,
            "ownVersusAdoptedMaximumPixels": disagreement,
            "effectiveTolerancePixels": effective_tolerance,
            "rewritten": False,
        }
        record["filters"][filter_name] = entry
        if disagreement is None or disagreement > effective_tolerance:
            record["status"] = "MISMATCH"
            record["code"] = "SAME_GRID_WCS_MISMATCH"
            record["message"] = (
                f"{filter_name} solved {disagreement} px away from {adopted} on a shared grid"
                if disagreement is not None
                else f"{filter_name} and {adopted} solutions do not map the shared grid"
            )
            return record
    for filter_name, path in sorted(products.items()):
        if filter_name == adopted:
            continue
        entry = record["filters"][filter_name]
        before_sha256 = sha256_digest(path)
        try:
            with fits.open(
                path,
                mode="update",
                memmap=True,
                do_not_scale_image_data=True,
                uint=False,
                checksum=False,
            ) as hdul:
                header = hdul[0].header
                for keyword in [card.keyword for card in header.cards if _WCS_CARD_PATTERN.match(card.keyword)]:
                    del header[keyword]
                # Insert after the mandatory cards so the layout stays FITS-legal.
                insert_at = header.index("EXTEND") + 1 if "EXTEND" in header else 5
                for offset, card in enumerate(adopted_cards):
                    header.insert(insert_at + offset, card)
                header["OAFWCSSG"] = (adopted, "Filter whose same-grid solve is shared")
                header["OAFWCSVP"] = (
                    float(entry["ownVersusAdoptedMaximumPixels"]),
                    "Own solve vs shared solve, max px",
                )
                header.add_history(
                    f"Ultra-Fast WBPP: same-grid {adopted} solve adopted; own solve agreed within "
                    f"{entry['ownVersusAdoptedMaximumPixels']:.4f} px"
                )
                for hdu in hdul:
                    if "CHECKSUM" in hdu.header or "DATASUM" in hdu.header:
                        hdu.add_checksum(override_datasum=True)
                hdul.flush(output_verify="exception")
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
        except Exception as error:
            raise E2EError("SAME_GRID_WCS_REWRITE_FAILED", str(error), path=str(path)) from error
        after_header, after_shape = _read_image_header(path)
        validation = validate_wcs_header(after_header, image_shape=after_shape)
        residual = _wcs_grid_disagreement(adopted_header, after_header, shape)
        if (
            not validation.valid
            or after_shape != shape
            or residual is None
            or residual > 1e-6
            or after_header.get("OAFSTATE") != "SOLVED"
            or after_header.get("OAFWCS") != "SOLVED"
        ):
            raise E2EError(
                "SAME_GRID_WCS_REWRITE_INVALID",
                "rewritten master does not carry the adopted solution exactly",
                path=str(path),
            )
        entry.update(
            rewritten=True,
            sha256Before=before_sha256,
            sha256After=sha256_digest(path),
            wcsValidation=validation.serializable(),
        )
        accepted = next(
            (
                item
                for item in reversed(solver_records.get(filter_name, {}).get("attempts", []))
                if isinstance(item, Mapping) and item.get("accepted") is True
            ),
            None,
        )
        if isinstance(accepted, dict) and isinstance(accepted.get("artifact"), dict):
            accepted["artifact"]["sha256"] = entry["sha256After"]
            accepted["artifact"]["sizeBytes"] = path.stat().st_size
            accepted["sameGridSolveAdopted"] = adopted
    return record


def _validate_cross_filter_wcs(
    products: Mapping[str, Path], *, tolerance_pixels: float = 0.05
) -> WcsValidation:
    """Require direct full-field pixel agreement between solved filter WCSes.

    Centre/scale/parity summaries cannot detect a 90-degree rotation or
    edge-only SIP drift.  Registered filter masters are required to describe
    the same pixel grid, so this gate compares centre, corners, and edge
    midpoints through each celestial transform in both directions.  The
    tolerance is in the masters' own pixels; callers scale it for drizzled
    grids, whose pixels are a fraction of a native pixel.
    """

    if not math.isfinite(tolerance_pixels) or tolerance_pixels <= 0:
        return WcsValidation(
            False,
            "CROSS_FILTER_POLICY_INVALID",
            "cross-filter WCS tolerance must be finite and positive",
        )
    geometries: dict[str, dict[str, Any]] = {}
    headers: dict[str, fits.Header] = {}
    shapes: dict[str, tuple[int, int]] = {}
    try:
        for filter_name, path in sorted(products.items()):
            header, shape = _read_image_header(path)
            validation = validate_wcs_header(header, image_shape=shape)
            if not validation.valid:
                return validation
            geometries[filter_name] = _solution_geometry(header, shape)
            headers[filter_name] = header
            shapes[filter_name] = shape
    except Exception as error:
        return WcsValidation(False, "CROSS_FILTER_WCS_INVALID", str(error))
    if len(geometries) < 2:
        return WcsValidation(
            True,
            "CROSS_FILTER_WCS_NOT_APPLICABLE",
            "only one solved filter is present",
            {"filters": geometries},
        )
    reference_name = next(iter(geometries))
    reference = geometries[reference_name]
    reference_shape = shapes[reference_name]
    height, width = reference_shape
    samples = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
            [(width - 1.0) / 2.0, 0.0],
            [(width - 1.0) / 2.0, height - 1.0],
            [0.0, (height - 1.0) / 2.0],
            [width - 1.0, (height - 1.0) / 2.0],
            [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
        ],
        dtype=np.float64,
    )
    reference_wcs = WCS(headers[reference_name], relax=False).celestial
    comparisons: dict[str, Any] = {}
    for filter_name, geometry in geometries.items():
        if filter_name == reference_name:
            continue
        shape_agrees = shapes[filter_name] == reference_shape
        candidate_wcs = WCS(headers[filter_name], relax=False).celestial
        if shape_agrees:
            candidate_world = candidate_wcs.all_pix2world(samples, 0)
            in_reference = reference_wcs.all_world2pix(candidate_world, 0)
            reference_world = reference_wcs.all_pix2world(samples, 0)
            in_candidate = candidate_wcs.all_world2pix(reference_world, 0)
            finite = bool(
                np.all(np.isfinite(in_reference)) and np.all(np.isfinite(in_candidate))
            )
            candidate_to_reference = (
                float(np.max(np.abs(in_reference - samples))) if finite else None
            )
            reference_to_candidate = (
                float(np.max(np.abs(in_candidate - samples))) if finite else None
            )
            per_sample = (
                np.maximum(
                    np.max(np.abs(in_reference - samples), axis=1),
                    np.max(np.abs(in_candidate - samples), axis=1),
                ).tolist()
                if finite
                else [None] * len(samples)
            )
        else:
            finite = False
            candidate_to_reference = None
            reference_to_candidate = None
            per_sample = [None] * len(samples)
        rotation_delta = abs(
            ((float(geometry["rotationDegrees"]) - float(reference["rotationDegrees"]) + 180.0) % 360.0)
            - 180.0
        )
        comparisons[filter_name] = {
            "referenceFilter": reference_name,
            "shapeAgrees": shape_agrees,
            "tolerancePixels": tolerance_pixels,
            "candidateToReferenceMaximumResidualPixels": candidate_to_reference,
            "referenceToCandidateMaximumResidualPixels": reference_to_candidate,
            "sampleResidualPixels": per_sample,
            "samplePixels": samples.tolist(),
            "rotationDeltaDegrees": rotation_delta,
            "parityAgrees": reference["parity"] == geometry["parity"],
        }
        if (
            not shape_agrees
            or not finite
            or candidate_to_reference is None
            or reference_to_candidate is None
            or candidate_to_reference > tolerance_pixels
            or reference_to_candidate > tolerance_pixels
            or reference["parity"] != geometry["parity"]
        ):
            return WcsValidation(
                False,
                "CROSS_FILTER_WCS_MISMATCH",
                f"{filter_name} does not describe the same registered sky footprint as {reference_name}",
                {"filters": geometries, "comparisons": comparisons},
            )
    return WcsValidation(
        True,
        "CROSS_FILTER_WCS_VALID",
        "all filter WCSes agree directly at centre, corners, and edge midpoints",
        {
            "filters": geometries,
            "comparisons": comparisons,
            "policy": {
                "maximumResidualPixels": tolerance_pixels,
                "sampleCount": len(samples),
                "bidirectional": True,
            },
        },
    )


def _verify_backend_result(
    backend: SolverBackend,
    result: SolverResult,
) -> bool:
    backend_verifier = getattr(backend, "verify_result", None)
    if callable(backend_verifier):
        try:
            return bool(backend_verifier(result))
        except Exception:
            return False
    return bool(verify_solver_execution_result(result))


def _discard_owned_solver_output(path: Path) -> bool:
    """Remove only the exact, staging-owned output requested from a backend."""

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISDIR(mode):
        return False
    try:
        remove_file(path, missing_ok=False)
        return True
    except OSError:
        return False


def _promote_solved_state(
    path: Path,
    *,
    backend_id: str,
    solver_verified_sha256: str,
) -> dict[str, Any]:
    """Make the E2E state cards agree with an already verified fresh WCS.

    External adapters intentionally preserve non-WCS input cards, including
    the pixel pipeline's ``UNSOLVED_WORKING`` marker.  The adapter receipt is
    verified first; this E2E-owned promotion then changes only state/history
    cards, revalidates WCS, and binds the resulting file in the outer receipt.
    """

    before_stat = path.lstat()
    if not stat.S_ISREG(before_stat.st_mode) or path.is_symlink():
        raise E2EError("SOLVED_OUTPUT_INVALID", "solver output is not a regular file", path=str(path))
    before_sha256 = sha256_digest(path)
    if before_sha256 != solver_verified_sha256:
        raise E2EError("SOLVED_OUTPUT_DRIFT", "solver output changed before E2E promotion", path=str(path))
    before_header, before_shape = _read_image_header(path)
    try:
        with fits.open(
            path,
            mode="update",
            memmap=True,
            do_not_scale_image_data=True,
            uint=False,
            checksum=False,
        ) as hdul:
            hdul[0].header["OAFSTATE"] = ("SOLVED", "Ultra-Fast WBPP E2E product state")
            hdul[0].header["OAFWCS"] = ("SOLVED", "Fresh WCS independently verified")
            hdul[0].header["OAFSOLVR"] = (backend_id, "Plate solver backend")
            hdul[0].header.add_history(
                "Ultra-Fast WBPP: solver evidence verified before E2E state promotion"
            )
            for hdu in hdul:
                if "CHECKSUM" in hdu.header or "DATASUM" in hdu.header:
                    hdu.add_checksum(override_datasum=True)
            hdul.flush(output_verify="exception")
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
    except Exception as error:
        raise E2EError("SOLVED_STATE_PROMOTION_FAILED", str(error), path=str(path)) from error
    after_header, after_shape = _read_image_header(path)
    validation = validate_wcs_header(after_header, image_shape=after_shape)
    if (
        not validation.valid
        or before_shape != after_shape
        or not _wcs_headers_agree(before_header, after_header, after_shape)
        or after_header.get("OAFSTATE") != "SOLVED"
        or after_header.get("OAFWCS") != "SOLVED"
    ):
        raise E2EError(
            "SOLVED_STATE_PROMOTION_INVALID",
            "state promotion changed geometry/WCS or did not persist SOLVED markers",
            path=str(path),
        )
    try:
        with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
            for hdu in hdul:
                if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
                    raise E2EError("SOLVED_OUTPUT_CHECKSUM_INVALID", "FITS checksum failed", path=str(path))
                if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
                    raise E2EError("SOLVED_OUTPUT_CHECKSUM_INVALID", "FITS datasum failed", path=str(path))
    except E2EError:
        raise
    except Exception as error:
        raise E2EError("SOLVED_OUTPUT_CHECKSUM_INVALID", str(error), path=str(path)) from error
    return {
        "kind": "E2E_SOLVED_STATE_PROMOTION",
        "solverVerifiedSha256": before_sha256,
        "finalSha256": sha256_digest(path),
        "sizeBytes": path.stat().st_size,
        "wcsValidation": validation.serializable(),
        "stateCards": {"OAFSTATE": "SOLVED", "OAFWCS": "SOLVED"},
    }


def _solve_one(
    *,
    input_path: Path,
    output_path: Path,
    backends: Sequence[SolverBackend],
    hints: _SolverHints,
    min_matches: int,
    max_rms_arcsec: float,
) -> tuple[bool, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for backend in backends:
        if os.path.lexists(output_path):
            attempts.append(
                {
                    "backendId": getattr(backend, "backend_id", type(backend).__name__),
                    "accepted": False,
                    "code": "UNEXPECTED_OUTPUT_EXISTS",
                }
            )
            break
        try:
            result = backend.solve(
                SolveRequest(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    ra_hint_degrees=hints.ra_degrees,
                    dec_hint_degrees=hints.dec_degrees,
                    field_of_view_degrees=hints.field_of_view_degrees,
                    search_radius_degrees=hints.search_radius_degrees,
                )
            )
        except Exception as error:
            attempts.append(
                {
                    "backendId": getattr(backend, "backend_id", type(backend).__name__),
                    "accepted": False,
                    "code": "BACKEND_EXCEPTION",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            if not _discard_owned_solver_output(output_path):
                attempts[-1]["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        if not isinstance(result, SolverResult):
            attempts.append(
                {
                    "backendId": getattr(backend, "backend_id", type(backend).__name__),
                    "accepted": False,
                    "code": "BACKEND_PROTOCOL_ERROR",
                    "message": "solver returned the wrong result type",
                }
            )
            if not _discard_owned_solver_output(output_path):
                attempts[-1]["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        try:
            validation = validate_solver_result(
                result,
                require_scientific_evidence=True,
                min_matches=min_matches,
                max_rms_arcsec=max_rms_arcsec,
            )
            hint_validation = (
                _validate_solution_against_hints(result, hints)
                if validation.valid
                else WcsValidation(
                    False,
                    "SOLVER_HINT_VALIDATION_SKIPPED",
                    "catalog quality validation failed first",
                )
            )
            verified = _verify_backend_result(backend, result)
            serialized_result = result.serializable()
        except Exception as error:
            attempts.append(
                {
                    "backendId": getattr(backend, "backend_id", type(backend).__name__),
                    "accepted": False,
                    "code": "BACKEND_PROTOCOL_ERROR",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            if not _discard_owned_solver_output(output_path):
                attempts[-1]["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        attempt = {
            "result": serialized_result,
            "validation": validation.serializable(),
            "hintValidation": hint_validation.serializable(),
            "executionVerified": verified,
            "accepted": False,
        }
        if not validation.valid or not hint_validation.valid or not verified:
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        if result.output_path is None:
            attempt["code"] = "SOLVED_OUTPUT_MISSING"
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        try:
            actual_output = Path(result.output_path).resolve(strict=True)
        except OSError:
            attempt["code"] = "SOLVED_OUTPUT_MISSING"
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        if actual_output != output_path.resolve(strict=False):
            attempt["code"] = "SOLVED_OUTPUT_PATH_MISMATCH"
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        try:
            output_header, output_shape = _read_image_header(actual_output)
        except E2EError as error:
            attempt["code"] = error.code
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        output_validation = validate_wcs_header(output_header, image_shape=output_shape)
        attempt["publishedWcsValidation"] = output_validation.serializable()
        if (
            not output_validation.valid
            or result.image_shape != output_shape
            or not _wcs_headers_agree(result.header, output_header, output_shape)
        ):
            attempt["code"] = "SOLVER_RESULT_OUTPUT_MISMATCH"
            attempts.append(attempt)
            if not _discard_owned_solver_output(output_path):
                attempt["code"] = "UNEXPECTED_OUTPUT_CANNOT_DISCARD"
                break
            continue
        attempt["accepted"] = True
        attempt["artifact"] = {
            "path": str(actual_output),
            "sha256": sha256_digest(actual_output),
            "sizeBytes": actual_output.stat().st_size,
        }
        try:
            attempt["promotion"] = _promote_solved_state(
                actual_output,
                backend_id=result.backend_id,
                solver_verified_sha256=attempt["artifact"]["sha256"],
            )
            attempt["artifact"] = {
                "path": str(actual_output),
                "sha256": attempt["promotion"]["finalSha256"],
                "sizeBytes": attempt["promotion"]["sizeBytes"],
            }
        except E2EError as error:
            attempt["accepted"] = False
            attempt["code"] = error.code
            attempt["message"] = str(error)
            attempts.append(attempt)
            return False, attempts
        attempts.append(attempt)
        return True, attempts
    return False, attempts

