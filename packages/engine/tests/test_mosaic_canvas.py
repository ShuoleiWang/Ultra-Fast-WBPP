"""The mosaic canvas, the panel distortion fit, and the photometric network."""

from __future__ import annotations

import math
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest
from scipy import ndimage

from mosaic_synthetic import true_header
from ufwbpp.mosaic import protect
from ufwbpp.mosaic.assemble import assemble_filter
from ufwbpp.mosaic.astrometry import fit_tan_sip
from ufwbpp.mosaic.canvas import (
    CanvasProjection,
    PanelFootprint,
    canvas_window,
    celestial_wcs,
    plan_canvas,
    window_grid,
)
from ufwbpp.mosaic.photometry import PanelImage
from ufwbpp.stacking.warp import _lattice_coordinates


def _grid_footprints(*, step_degrees: float, rotations: tuple[float, ...], shape=(600, 400)) -> list[PanelFootprint]:
    footprints = []
    for index, rotation in enumerate(rotations):
        column, row = index % 3, index // 3
        dec = 41.0 + (row - 0.5) * step_degrees
        ra = 10.0 + (column - 1) * step_degrees / math.cos(math.radians(dec))
        footprints.append(
            PanelFootprint(f"P{index}", true_header((ra, dec), shape, scale_arcsec=10.0, rotation_degrees=rotation, distortion=1.0), shape)
        )
    return footprints


def test_canvas_takes_the_union_centroid_and_the_median_panel_rotation() -> None:
    footprints = _grid_footprints(step_degrees=0.9, rotations=(1.0, 1.1, 1.2, 1.0, 1.1, 5.0))
    plan = plan_canvas(footprints)
    summary = plan.serializable()
    assert summary["projection"] == "TAN"
    # The outlier panel rotated by 5 degrees does not turn the canvas.
    assert summary["rotationDegrees"] == pytest.approx(1.1, abs=0.02)
    assert summary["pixelScaleArcsec"] == pytest.approx(10.0, rel=1e-3)
    assert summary["crval"][0] == pytest.approx(10.0, abs=0.05) and summary["crval"][1] == pytest.approx(41.0, abs=0.05)
    # Every panel lies inside the canvas.
    for panel in summary["panels"]:
        x0, y0, x1, y1 = panel["canvasBox"]
        assert 0 <= x0 < x1 <= plan.width and 0 <= y0 < y1 <= plan.height


def test_a_wide_mosaic_switches_to_a_stereographic_canvas() -> None:
    footprints = _grid_footprints(step_degrees=5.0, rotations=(0.0,) * 6, shape=(2000, 2000))
    assert plan_canvas(footprints).projection.projection == "STG"


def test_the_window_lattice_maps_canvas_pixels_into_the_distorted_reference() -> None:
    footprint = _grid_footprints(step_degrees=0.9, rotations=(1.0,))[0]
    plan = plan_canvas([footprint])
    ra, dec = footprint.boundary_world()
    origin, shape = canvas_window(plan.projection, ra, dec)
    grid = window_grid(plan.projection, origin, shape, footprint.header, footprint.shape)
    reference = celestial_wcs(footprint.header)
    rng = np.random.default_rng(3)
    for row in rng.integers(0, shape[0], 20):
        lattice_x, lattice_y = _lattice_coordinates(grid.reference_x, grid.reference_y, grid.spacing, int(row), 1, shape[1])
        columns = np.arange(shape[1], dtype=np.float64)
        world_ra, world_dec = plan.projection.canvas_to_world(columns + origin[0], np.full(shape[1], float(row + origin[1])))
        exact_x, exact_y = reference.all_world2pix(world_ra, world_dec, 0, tolerance=1e-10, maxiter=200)
        inside = (exact_x > 0) & (exact_x < footprint.shape[1] - 1) & (exact_y > 0) & (exact_y < footprint.shape[0] - 1)
        assert np.max(np.abs(lattice_x[0, inside] - exact_x[inside])) < 2e-3
        assert np.max(np.abs(lattice_y[0, inside] - exact_y[inside])) < 2e-3


def test_tan_sip_fit_recovers_the_distortion_a_pure_tan_misses() -> None:
    shape = (4176, 6252)
    truth = true_header((10.68, 41.27), shape, scale_arcsec=1.0, rotation_degrees=1.2, distortion=2.0)
    truth_wcs = celestial_wcs(truth)
    rng = np.random.default_rng(1)
    pixels = np.column_stack((rng.uniform(0, shape[1] - 1, 150), rng.uniform(0, shape[0] - 1, 150)))
    world = np.column_stack(truth_wcs.all_pix2world(pixels[:, 0], pixels[:, 1], 0))
    measured = pixels + rng.normal(0.0, 0.1, pixels.shape)
    measured[:3] += 3.0  # mismatched stars
    grid_x, grid_y = np.meshgrid(np.linspace(0, shape[1] - 1, 21), np.linspace(0, shape[0] - 1, 15))
    sky = truth_wcs.all_pix2world(grid_x.ravel(), grid_y.ravel(), 0)

    def worst(header: dict) -> float:
        x, y = celestial_wcs(header).all_world2pix(sky[0], sky[1], 0)
        return float(np.max(np.hypot(x - grid_x.ravel(), y - grid_y.ravel())))

    fitted, order, inliers = fit_tan_sip(measured, world, shape, crval=(10.7, 41.25), maximum_order=4)
    assert order >= 3 and inliers.sum() == 147
    assert worst(fitted) < 0.2
    pure_tan, _, _ = fit_tan_sip(measured, world, shape, crval=(10.7, 41.25), maximum_order=1)
    assert worst(pure_tan) > 1.0


def _six_panels(root: Path) -> tuple[list[PanelImage], dict[str, float], tuple[int, int], np.ndarray, np.ndarray]:
    """Six panels of one sky with a bright galaxy whose core sits in the
    middle overlap: each panel with its own transparency, sky plane, seeing
    and noise."""

    rng = np.random.default_rng(7)
    width, height, step_x, step_y = 640, 480, 540, 380
    canvas_w, canvas_h = 2 * step_x + width, step_y + height
    yy, xx = np.mgrid[:canvas_h, :canvas_w].astype(float)
    r = np.hypot(xx - canvas_w / 2, (yy - canvas_h / 2 - 10) / 0.5)
    galaxy = 300 * np.exp(-r / 160.0) + 30000 * np.exp(-r / 5.0)
    count = int(canvas_w * canvas_h / 1200)
    sx, sy = rng.uniform(5, canvas_w - 5, count), rng.uniform(5, canvas_h - 5, count)
    flux = 10 ** rng.uniform(3.0, 5.0, count)

    def render(fwhm: float) -> np.ndarray:
        image = np.zeros((canvas_h, canvas_w))
        np.add.at(image, (np.round(sy).astype(int), np.round(sx).astype(int)), flux)
        sigma = fwhm / 2.3548
        return ndimage.gaussian_filter(image, sigma) + ndimage.gaussian_filter(galaxy, sigma) + 100.0

    panels, scales = [], {}
    index = 0
    for column in range(3):
        for row in range(2):
            x0, y0 = column * step_x, row * step_y
            scale = 0.85 + 0.05 * index
            plane = rng.uniform(-8, 8, 3)
            u = (xx[y0 : y0 + height, x0 : x0 + width] - (canvas_w - 1) / 2) / (max(canvas_w, canvas_h) / 2)
            v = (yy[y0 : y0 + height, x0 : x0 + width] - (canvas_h - 1) / 2) / (max(canvas_w, canvas_h) / 2)
            image = render(2.5 + 0.4 * index)[y0 : y0 + height, x0 : x0 + width] * scale
            image += plane[0] + plane[1] * u + plane[2] * v + rng.normal(0, 2.0, (height, width))
            path = root / f"P{index}.fits"
            fits.writeto(path, image.astype(np.float32))
            coverage = root / f"P{index}_coverage.fits"
            fits.writeto(coverage, np.ones((height, width), np.float32))
            panels.append(PanelImage(f"P{index}", path, (x0, y0), (height, width), 300.0, coverage, None))
            scales[f"P{index}"] = scale
            index += 1
    return panels, scales, (canvas_w, canvas_h), render(3.5), galaxy


def test_six_panels_are_matched_by_one_network_and_blended_without_a_background_step(tmp_path: Path) -> None:
    panels, true_scales, (width, height), sky, galaxy = _six_panels(tmp_path)
    projection = CanvasProjection("TAN", (10.0, 41.0), (1.0, 1.0), ((-1 / 3600, 0.0), (0.0, 1 / 3600)))
    output = tmp_path / "mosaic"
    output.mkdir()
    result = assemble_filter("L", panels, projection=projection, canvas_box=(0, 0, width, height), output_directory=output, verifier=None)
    receipt = result.receipt
    reference = receipt["photometry"]["scaleReference"]
    # Scaled panels agree with the truth to a few parts in ten thousand.
    for panel in receipt["panels"]:
        assert panel["scale"] * true_scales[panel["key"]] / true_scales[reference] == pytest.approx(1.0, abs=2e-3)
    assert receipt["gates"]["photometry"] == "PASS"
    assert receipt["gates"]["overlapAstrometry"] == "PASS"
    assert receipt["gates"]["fluxConservation"] == "PASS"
    # The galaxy scale of every overlap it crosses agrees with the stars'.
    checks = receipt["background"]["extendedStructureScaleChecks"]
    assert checks and all(item["agreesWithStars"] for item in checks)
    # After the unobservable common plane, the sky differs from the truth by
    # a small fraction of the noise: no panel-shaped steps are left.
    mosaic = fits.getdata(result.science_path).astype(float)
    difference = mosaic - sky * true_scales[reference]
    yy, xx = np.mgrid[:height, :width].astype(float)
    design = np.column_stack((np.ones(difference.size), xx.ravel(), yy.ravel()))
    faint = np.isfinite(difference.ravel()) & (galaxy.ravel() < 30)
    common, *_ = np.linalg.lstsq(design[faint], difference.ravel()[faint], rcond=None)
    residual = difference - (design @ common).reshape(difference.shape)
    stars = ndimage.gaussian_filter(np.abs(sky - ndimage.median_filter(sky, 9)), 2.0) > 0.2
    usable = ~stars & (galaxy < 30)
    binned = [
        np.median(residual[i : i + 64, j : j + 64][usable[i : i + 64, j : j + 64]])
        for i in range(0, height - 63, 64)
        for j in range(0, width - 63, 64)
        if usable[i : i + 64, j : j + 64].sum() > 500
    ]
    assert np.sqrt(np.mean(np.square(binned))) < 0.25  # pixel noise 2.0; the injected planes reach 10


def _two_seeings(root: Path) -> tuple[list[PanelImage], list[tuple[int, int]], tuple[int, int]]:
    """Two panels of one sky that overlap by 300 px, one seen at FWHM 2.5 px
    and one at 4.5 px: bright stars and a galaxy core in the overlap, a
    bright star outside it in each panel, fainter stars everywhere."""

    rng = np.random.default_rng(13)
    height, width, step = 360, 640, 340
    canvas_w = step + width
    points = np.zeros((height, canvas_w))
    faint = 400
    sx, sy = rng.uniform(5, canvas_w - 5, faint), rng.uniform(5, height - 5, faint)
    np.add.at(points, (np.round(sy).astype(int), np.round(sx).astype(int)), 10 ** rng.uniform(2.0, 2.8, faint))
    # Stars across two decades of flux for the overlap photometry.
    medium = 150
    mx, my = rng.uniform(5, canvas_w - 5, medium), rng.uniform(5, height - 5, medium)
    np.add.at(points, (np.round(my).astype(int), np.round(mx).astype(int)), 10 ** rng.uniform(2.8, 4.2, medium))
    bright = [(x, y) for y in (60, 150, 240) for x in (400, 490, 580)]
    for x, y in (*bright, (150, 180), (830, 180)):
        points[y, x] += 60000.0
    yy, xx = np.mgrid[:height, :canvas_w].astype(float)
    r = np.hypot(xx - 490, yy - 310)
    sky = points + 20000.0 * np.exp(-r / 4.0)
    panels = []
    for key, x0, fwhm, scale, offset in (("A", 0, 2.5, 1.0, 3.0), ("B", step, 4.5, 0.9, -2.0)):
        image = (ndimage.gaussian_filter(sky, fwhm / 2.3548) + 100.0)[:, x0 : x0 + width] * scale + offset
        image += rng.normal(0.0, 2.0, image.shape)
        path, coverage = root / f"{key}.fits", root / f"{key}_coverage.fits"
        fits.writeto(path, image.astype(np.float32))
        fits.writeto(coverage, np.ones((height, width), np.float32))
        panels.append(PanelImage(key, path, (x0, 0), (height, width), 300.0, coverage, None))
    return panels, [*bright, (490, 310)], (canvas_w, height)


def _second_moment_fwhm(image: np.ndarray, x: int, y: int) -> float:
    """FWHM from the second moment within 8 px, above the median of the
    9-12 px annulus."""

    cut = image[y - 12 : y + 13, x - 12 : x + 13].astype(float)
    yy, xx = np.mgrid[-12:13, -12:13]
    r2 = xx**2 + yy**2
    light = cut - np.median(cut[(r2 >= 81) & (r2 <= 144)])
    inner = r2 <= 64
    return 2.3548 * math.sqrt(float(np.sum(light[inner] * r2[inner]) / (2.0 * np.sum(light[inner]))))


def test_bright_stars_and_cores_in_an_overlap_keep_one_panels_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panels, sources, (width, height) = _two_seeings(tmp_path)
    projection = CanvasProjection("TAN", (10.0, 41.0), (1.0, 1.0), ((-1 / 3600, 0.0), (0.0, 1 / 3600)))

    def assemble(name: str):
        output = tmp_path / name
        output.mkdir()
        return assemble_filter("L", panels, projection=projection, canvas_box=(0, 0, width, height), output_directory=output, verifier=None)

    protected = assemble("protected")
    monkeypatch.setattr(protect, "PROTECT_PEAK_SIGMA", math.inf)
    mean = assemble("mean")
    receipt = protected.receipt
    assert receipt["gates"]["photometry"] == "PASS" and receipt["gates"]["fluxConservation"] == "PASS"
    assert receipt["blend"]["protectedSet"]["blobsLeftToTheMean"] == 0
    science, noise = fits.getdata(protected.science_path).astype(float), fits.getdata(protected.noise_path).astype(float)
    mask = fits.getdata(protected.mask_path).astype(int)
    plain, plain_noise = fits.getdata(mean.science_path).astype(float), fits.getdata(mean.noise_path).astype(float)
    # Off the protected set the blend is the inverse-variance mean, bit for
    # bit, and the bright stars outside the overlap are not protected.
    outside = (mask & 4) == 0
    assert np.array_equal(science[outside], plain[outside], equal_nan=True)
    assert np.array_equal(noise[outside], plain_noise[outside], equal_nan=True)
    assert not mask[180, 150] & 4 and not mask[180, 830] & 4
    raw = {panel.key: fits.getdata(panel.path) for panel in panels}
    corrected_noise = {item["key"]: item["scale"] * item["noise"] for item in receipt["panels"]}
    for x, y in sources:
        own = {"A": _second_moment_fwhm(raw["A"], x, y), "B": _second_moment_fwhm(raw["B"], x - 340, y)}
        kept, mixed = _second_moment_fwhm(science, x, y), _second_moment_fwhm(plain, x, y)
        chosen = min(own, key=lambda key: abs(kept / own[key] - 1.0))
        # The mosaic holds one panel's profile (the stars' FWHM 2.5 or 4.5
        # px, the core's width with either seeing); the plain mean is a mix.
        assert kept == pytest.approx(own[chosen], rel=0.005)
        assert min(own.values()) < mixed < max(own.values()) and abs(mixed / own[chosen] - 1.0) > 0.008
        assert mask[y, x] & 4
        assert noise[y, x] == pytest.approx(corrected_noise[chosen], rel=0.02)
        # The panel with the larger weight at the peak: in the middle row, A
        # where B fades in and B where A fades out.
        if (x, y) == (400, 150):
            assert chosen == "A"
        if (x, y) == (580, 150):
            assert chosen == "B"

