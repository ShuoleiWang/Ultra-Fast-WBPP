"""Lateral colour on a two-panel L+B mosaic through the real pipeline.

The optics image B with more barrel distortion than L (a corrector's
lateral colour).  A B Light's homography to the panel's reference absorbs
the linear part only; each panel must resample every B Light by its own
shift field, or the two panels place the same B star differently in their
overlap.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from astropy.io import fits
import numpy as np

from mosaic_synthetic import TruthSolver, calibration_header, light_header, make_sky, render, true_header, write_uint16
from ufwbpp.inventory import inventory_project
from ufwbpp.workflows.project import ProjectE2ERequest, run_project_e2e
from ufwbpp.workflows.single_target import E2ERequest


SCALE = 2.0
SHAPE = (192, 256)
BIAS = 1000.0
# Barrel distortion at the frame corners, in pixels, per filter.
DISTORTION = {"L": 1.5, "B": 2.1}


def _project(root: Path) -> tuple[E2ERequest, TruthSolver]:
    sky = make_sky((150.0, 20.0), 0.15, 900, flux_range=(3.6, 4.8))
    rng = np.random.default_rng(12)
    lights: list[Path] = []
    panels = {"MOSAIC-P1": -80.0, "MOSAIC-P2": 80.0}
    for target, offset in panels.items():
        centre = (150.0 + offset * SCALE / 3600.0 / math.cos(math.radians(20.0)), 20.0)
        for filter_name, distortion in DISTORTION.items():
            for index in range(8):
                dither = rng.uniform(-3.0, 3.0, 2) * SCALE / 3600.0
                truth = true_header(
                    (centre[0] + dither[0], centre[1] + dither[1]), SHAPE, scale_arcsec=SCALE, rotation_degrees=0.4, distortion=distortion
                )
                frame = BIAS + render(sky, truth, SHAPE, fwhm=2.4, background=300.0, noise=3.0, rng=rng)
                minute = index + (10 if target.endswith("2") else 0) + (20 if filter_name == "B" else 0)
                lights.append(
                    write_uint16(
                        root / target / "LIGHT" / f"light_{filter_name}_{index:02d}.fits",
                        frame,
                        light_header(target, filter_name, exposure=60.0, observed_at=f"2026-01-01T20:{minute:02d}:00Z", truth=truth, pointing=centre),
                    )
                )
    biases = [
        write_uint16(root / "BIAS" / f"bias_{index:02d}.fits", BIAS + rng.normal(0.0, 1.0, SHAPE), calibration_header("Bias", exposure=0.001))
        for index in range(3)
    ]
    flats = [
        write_uint16(
            root / "FLAT" / f"flat_{filter_name}_{index:02d}.fits",
            BIAS + 24_000.0 + rng.normal(0.0, 3.0, SHAPE),
            calibration_header("Flat", exposure=2.0, filter_name=filter_name),
        )
        for filter_name in DISTORTION
        for index in range(3)
    ]
    request = E2ERequest(
        light_files=tuple(str(path) for path in lights),
        flat_files=tuple(str(path) for path in flats),
        bias_files=tuple(str(path) for path in biases),
        output_directory=str(root.parent / "product"),
        workers=2,
        ra_hint_degrees=150.0,
        dec_hint_degrees=20.0,
        field_of_view_degrees=SHAPE[1] * SCALE / 3600.0,
        search_radius_degrees=2.0,
    )
    return request, TruthSolver(sky, scale_arcsec=SCALE)


def test_each_light_is_resampled_with_its_own_lateral_colour(tmp_path: Path) -> None:
    request, solver = _project(tmp_path / "input")
    inventory = inventory_project([tmp_path / "input"])
    result = run_project_e2e(ProjectE2ERequest(inventory, request, request.output_directory), solver_backends=(solver,))
    assert result.success is True, result
    output = Path(result.output_directory)
    for run in (output / "details" / "runs").iterdir():
        canvas = json.loads((run / "receipts" / "canvas.json").read_text(encoding="utf-8"))
        # Every Light of the filter that is not the reference's carries its own shift field.
        fields = canvas["frameFields"]["filters"]
        pipeline = json.loads((run / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))
        shifted = sum("shiftField" in item for item in pipeline["registration"].values())
        assert shifted == sum(item["applied"] for item in fields.values()), (shifted, fields)
        filter_name, record = max(fields.items(), key=lambda item: item[1].get("maxShiftPixels", {}).get("median", 0.0))
        assert record["applied"] == record["lights"] >= 7, record
        assert record["maxShiftPixels"]["median"] > 0.15 and record["rmsAfterPixels"] < 0.6 * record["rmsBeforePixels"], record
        master = next(path for path in (run / "products").rglob("master_light_*_wcs.fits") if fits.getheader(path)["FILTER"] == filter_name)
        assert fits.getheader(master)["OAFGRID"] == canvas["window"]["sha256"][:32]
    # In both mosaics every star sits where the true sky puts it.
    for name in ("L", "B"):
        path = output / f"{name}.fits"
        header = fits.getheader(path)
        verification = solver.canvas_verifier(fits.getdata(path).astype(np.float32), dict(header), tmp_path)
        assert verification["matchedStars"] >= 60, (name, verification)
        assert verification["offsetMedianPixels"] < 0.05, (name, verification)
        assert verification["offsetP90Pixels"] < 0.15, (name, verification)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    for key, value in receipt["execution"]["mosaics"].items():
        if isinstance(value, dict) and value.get("mode") == "CANVAS_MOSAIC":
            assert value["gates"]["overlapAstrometry"] == "PASS", (key, value["gates"])
