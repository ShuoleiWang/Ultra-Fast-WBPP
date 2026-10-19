"""A two-panel one-shot-colour (RGGB) mosaic through the real pipeline.

Each panel's Bayer Lights must be placed on the canvas from their
debayered luminance, integrated (or Bayer-drizzled from the mosaics' own
samples) into three colour-channel masters on the panel's window, and every
channel must join its own mosaic with the stars where the true sky puts
them.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from mosaic_synthetic import TruthSolver, calibration_header, light_header, make_sky, render, true_header, write_uint16
from ufwbpp.inventory import inventory_project
from ufwbpp.native_kernels import load_native_kernels
from ufwbpp.workflows.contracts import DrizzleOptions, IntegrationMode
from ufwbpp.workflows.project import ProjectE2ERequest, run_project_e2e
from ufwbpp.workflows.single_target import E2ERequest


SCALE = 2.0
SHAPE = (192, 256)
BIAS = 1000.0
# The camera's channel response behind the filter (the flat carries it too).
RESPONSE = {"R": 0.8, "G": 1.0, "B": 0.6}


def _mosaic(planes: dict[str, np.ndarray]) -> np.ndarray:
    """RGGB: R at (even row, even column), B at (odd, odd), G elsewhere."""

    mosaic = np.empty(SHAPE, dtype=np.float64)
    mosaic[0::2, 0::2] = planes["R"][0::2, 0::2]
    mosaic[0::2, 1::2] = planes["G"][0::2, 1::2]
    mosaic[1::2, 0::2] = planes["G"][1::2, 0::2]
    mosaic[1::2, 1::2] = planes["B"][1::2, 1::2]
    return mosaic


def _project(root: Path) -> tuple[E2ERequest, TruthSolver]:
    sky = make_sky((150.0, 20.0), 0.15, 600, flux_range=(3.6, 4.8))
    rng = np.random.default_rng(8)
    colour = {"R": rng.uniform(0.7, 1.3, sky.ra.size), "G": np.ones(sky.ra.size), "B": rng.uniform(0.6, 1.3, sky.ra.size)}
    lights: list[Path] = []
    panels = {"MOSAIC-P1": {"offset": -80.0, "sky": 300.0}, "MOSAIC-P2": {"offset": 80.0, "sky": 340.0}}
    for target, panel in panels.items():
        centre = (150.0 + panel["offset"] * SCALE / 3600.0 / math.cos(math.radians(20.0)), 20.0)
        for index in range(8):
            dither = rng.uniform(-3.0, 3.0, 2) * SCALE / 3600.0
            truth = true_header(
                (centre[0] + dither[0], centre[1] + dither[1]), SHAPE, scale_arcsec=SCALE, rotation_degrees=0.4, distortion=1.5
            )
            planes = {}
            for name, factor in colour.items():
                channel_sky = type(sky)(sky.ra, sky.dec, sky.flux * factor)
                planes[name] = RESPONSE[name] * render(
                    channel_sky, truth, SHAPE, fwhm=2.4, background=panel["sky"], noise=3.0, rng=rng
                )
            header = light_header(
                target,
                "LP",
                exposure=60.0,
                observed_at=f"2026-01-01T20:{index + (10 if target.endswith('2') else 0):02d}:00Z",
                truth=truth,
                pointing=centre,
            )
            header["BAYERPAT"] = "RGGB"
            lights.append(write_uint16(root / target / "LIGHT" / f"light_LP_{index:02d}.fits", BIAS + _mosaic(planes), header))
    biases = []
    for index in range(3):
        header = calibration_header("Bias", exposure=0.001)
        header["BAYERPAT"] = "RGGB"
        biases.append(write_uint16(root / "BIAS" / f"bias_{index:02d}.fits", BIAS + rng.normal(0.0, 1.0, SHAPE), header))
    flats = []
    for index in range(3):
        header = calibration_header("Flat", exposure=2.0, filter_name="LP")
        header["BAYERPAT"] = "RGGB"
        level = _mosaic({name: np.full(SHAPE, 24_000.0 * response) for name, response in RESPONSE.items()})
        flats.append(write_uint16(root / "FLAT" / f"flat_LP_{index:02d}.fits", BIAS + level + rng.normal(0.0, 3.0, SHAPE), header))
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


@pytest.mark.parametrize("scale", [1, 2], ids=["integrated", "bayer-drizzled-2x"])
def test_bayer_panels_join_three_colour_mosaics_on_one_canvas(tmp_path: Path, scale: int) -> None:
    request, solver = _project(tmp_path / "input")
    if scale > 1:
        if not getattr(load_native_kernels(), "has_lattice_drizzle", False):
            pytest.skip("the native lattice drizzle is not installed")
        request = replace(request, integration_mode=IntegrationMode.DRIZZLE, drizzle=DrizzleOptions(scale=scale, pixfrac=1.0))
    inventory = inventory_project([tmp_path / "input"])
    result = run_project_e2e(ProjectE2ERequest(inventory, request, request.output_directory), solver_backends=(solver,))
    assert result.success is True, result
    output = Path(result.output_directory)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    mosaics = receipt["execution"]["mosaics"]
    canvas_mosaics = {key: value for key, value in mosaics.items() if isinstance(value, dict) and value.get("mode") == "CANVAS_MOSAIC"}
    assert len(canvas_mosaics) == 3, mosaics
    assert mosaics["canvas"]["scale"] == scale
    for key, value in canvas_mosaics.items():
        assert value["gates"]["overlapAstrometry"] == "PASS", (key, value["gates"])
        assert value["gates"]["photometry"] in {"PASS", "WARN"}, (key, value["gates"])
    # Every channel's stars sit where the true sky puts them on the canvas
    # (a drizzled pixel is 1/scale of a canvas pixel).
    for name in ("R", "G", "B"):
        path = output / f"{name}.fits"
        header = fits.getheader(path)
        verification = solver.canvas_verifier(fits.getdata(path).astype(np.float32), dict(header), tmp_path)
        assert verification["matchedStars"] >= 60, (path.name, verification)
        assert verification["offsetMedianPixels"] < 0.08 * scale, (path.name, verification)
    # Both panels were placed by one fit on their debayered stars; drizzled,
    # each channel dropped the mosaics' own samples of its colour.
    for run in (output / "details" / "runs").iterdir():
        canvas = json.loads((run / "receipts" / "canvas.json").read_text(encoding="utf-8"))
        assert canvas["jointFit"]["panels"] == 2, canvas["jointFit"]
        if scale > 1:
            for name in ("R", "G", "B"):
                drizzle = json.loads((run / "receipts" / f"drizzle_{name}.json").read_text(encoding="utf-8"))
                assert drizzle["backend"]["id"] == "native-cpu-drizzle-v2-lattice"
                assert drizzle["recipe"]["cfaPattern"] == "RGGB" and drizzle["recipe"]["cfaChannel"] == name
                assert drizzle["geometry"]["canvasWindow"]["sha256"] == canvas["window"]["sha256"]
