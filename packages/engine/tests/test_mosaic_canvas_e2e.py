"""A two-panel mosaic through the real pipeline, on the shared canvas.

The panels differ in transparency, sky level, seeing and pointing, and the
optics have a barrel distortion the solver's pure TAN solution ignores.  The
project must plan the canvas from one solved Light per panel, integrate
every panel straight onto its canvas window, and blend a mosaic whose stars
sit where the true sky puts them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from astropy.io import fits
import numpy as np

from mosaic_synthetic import (
    TruthSolver,
    calibration_header,
    light_header,
    make_sky,
    render,
    true_header,
    write_uint16,
)
from ufwbpp.inventory import inventory_project
from ufwbpp.workflows.project import ProjectE2ERequest, run_project_e2e
from ufwbpp.workflows.single_target import E2ERequest


SCALE = 2.0
SHAPE = (192, 256)
BIAS = 1000.0


def _panel_centre(offset_pixels: float) -> tuple[float, float]:
    return 150.0 + offset_pixels * SCALE / 3600.0 / math.cos(math.radians(20.0)), 20.0


def _project(root: Path) -> tuple[E2ERequest, TruthSolver]:
    sky = make_sky((150.0, 20.0), 0.15, 600, flux_range=(3.6, 4.8))
    rng = np.random.default_rng(5)
    lights: list[Path] = []
    # Two panels 160 px apart: a 96 px overlap of 256 px wide frames.
    panels = {
        "MOSAIC-P1": {"offset": -80.0, "fwhm": 2.2, "gain": 1.0, "sky": 300.0},
        "MOSAIC-P2": {"offset": 80.0, "fwhm": 2.8, "gain": 0.85, "sky": 360.0},
    }
    for target, panel in panels.items():
        centre = _panel_centre(panel["offset"])
        for index in range(8):
            dither = rng.uniform(-3.0, 3.0, 2) * SCALE / 3600.0
            truth = true_header(
                (centre[0] + dither[0], centre[1] + dither[1]),
                SHAPE,
                scale_arcsec=SCALE,
                rotation_degrees=0.4 + 0.1 * index,
                distortion=1.5,
            )
            frame = BIAS + render(
                sky, truth, SHAPE, fwhm=panel["fwhm"], background=panel["sky"], noise=3.0, rng=rng, gain=panel["gain"]
            )
            lights.append(
                write_uint16(
                    root / target / "LIGHT" / f"light_R_{index:02d}.fits",
                    frame,
                    light_header(
                        target,
                        "R",
                        exposure=60.0,
                        observed_at=f"2026-01-01T20:{index + (10 if target.endswith('2') else 0):02d}:00Z",
                        truth=truth,
                        pointing=centre,
                    ),
                )
            )
    biases = [
        write_uint16(root / "BIAS" / f"bias_{index:02d}.fits", BIAS + rng.normal(0.0, 1.0, SHAPE), calibration_header("Bias", exposure=0.001))
        for index in range(3)
    ]
    flats = [
        write_uint16(
            root / "FLAT" / f"flat_R_{index:02d}.fits",
            BIAS + 24_000.0 + rng.normal(0.0, 3.0, SHAPE),
            calibration_header("Flat", exposure=2.0),
        )
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


def test_two_panels_integrate_onto_one_canvas_and_blend_where_the_sky_is(tmp_path: Path) -> None:
    request, solver = _project(tmp_path / "input")
    inventory = inventory_project([tmp_path / "input"])
    result = run_project_e2e(
        ProjectE2ERequest(inventory, request, request.output_directory),
        solver_backends=(solver,),
    )
    assert result.success is True, result
    output = Path(result.output_directory)
    mosaic_path = output / "R.fits"
    header = fits.getheader(mosaic_path)
    assert header["OAFWCSPR"] == "CANVAS_CATALOG_VERIFIED"
    mosaic = fits.getdata(mosaic_path).astype(np.float64)
    # The mosaic spans both panels: wider than one frame.
    assert mosaic.shape[1] > SHAPE[1] + 40
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    mosaics = receipt["execution"]["mosaics"]
    assert mosaics["r"]["mode"] == "CANVAS_MOSAIC"
    assert mosaics["r"]["gates"]["photometry"] in {"PASS", "WARN"}
    assert mosaics["r"]["gates"]["overlapAstrometry"] == "PASS"
    # Every star of the true sky inside the mosaic sits on its canvas pixel.
    verification = solver.canvas_verifier(mosaic.astype(np.float32), dict(header), tmp_path)
    assert verification["matchedStars"] >= 60 and verification["isolatedStars"] >= 30
    assert verification["offsetMedianPixels"] < 0.05
    assert verification["offsetP90Pixels"] < 0.15
    # Each panel was integrated straight onto its window of the canvas.
    for run in (output / "details" / "runs").iterdir():
        canvas = json.loads((run / "receipts" / "canvas.json").read_text(encoding="utf-8"))
        assert canvas["distortion"]["order"] >= 2
        assert canvas["distortion"]["rmsPixels"] < 0.15
        master = next((run / "products").rglob("master_light_*_wcs.fits"))
        assert fits.getheader(master)["OAFGRID"] == canvas["window"]["sha256"][:32]
