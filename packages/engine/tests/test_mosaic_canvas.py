"""The mosaic canvas, the panel distortion fit, and the photometric network."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from astropy.io import fits
import numpy as np
import pytest
from scipy import ndimage

from mosaic_synthetic import true_header
from ufwbpp.mosaic import protect
from ufwbpp.mosaic.assemble import MosaicGateError, assemble_filter
from ufwbpp.mosaic.astrometry import fit_tan_sip
from ufwbpp.mosaic.canvas import (
    CanvasProjection,
    PanelFootprint,
    canvas_window,
    celestial_wcs,
    forward_lattice,
    panels_connected,
    plan_canvas,
    polygons_overlap,
    window_grid,
)
from ufwbpp.mosaic.assemble import _photometry_verdict
from ufwbpp.mosaic.photometry import (
    BIN_PIXELS,
    HUBER_THRESHOLD,
    EdgeMeasurement,
    PanelImage,
    _Design,
    _robust_solve,
    _systematic_offset,
    measure_edge,
    photometric_components,
    scale_observations,
    solve_planes,
    solve_scales,
)
from ufwbpp.integrity import canonical_json
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


def test_meridian_flipped_panels_share_a_north_up_canvas_that_they_fill(tmp_path: Path) -> None:
    """A 2x2 mosaic whose second night was shot after a meridian flip: two
    panels lie 180 degrees from the others.  The orientation is taken
    modulo a half turn, so the canvas follows the camera's axes and every
    panel fills its window; the panels overlap by 3 % and are connected on
    their footprint polygons."""

    shape = (400, 600)
    step_x, step_y = 0.97 * shape[1] * 10.0 / 3600.0, 0.97 * shape[0] * 10.0 / 3600.0
    footprints = []
    for index, rotation in enumerate((0.3, 180.3, 0.3, 180.3)):
        column, row = index % 2, index // 2
        dec = 41.0 + (row - 0.5) * step_y
        ra = 10.0 + (column - 0.5) * step_x / math.cos(math.radians(dec))
        footprints.append(PanelFootprint(f"P{index}", true_header((ra, dec), shape, scale_arcsec=10.0, rotation_degrees=rotation), shape))
    plan = plan_canvas(footprints)
    summary = plan.serializable()
    assert abs(summary["rotationDegrees"]) < 1.0 and abs(summary["rotationDegrees"] - 0.3) < 0.05
    for panel in summary["panels"]:
        x0, y0, x1, y1 = panel["canvasBox"]
        # A frame on a canvas of its own orientation is a rectangle: the window holds little else.
        assert shape[0] * shape[1] / ((x1 - x0) * (y1 - y0)) > 0.95
        expected = 0.3 if panel["key"] in ("P0", "P2") else 180.3
        assert math.remainder(panel["rotationDegrees"] - expected, 360.0) == pytest.approx(0.0, abs=0.05)
    polygons = {panel["key"]: np.asarray(panel["canvasPolygon"]) for panel in summary["panels"]}
    assert panels_connected(polygons)
    # The 3 % overlaps are real: neighbours share area, the diagonal pair does too.
    assert polygons_overlap(polygons["P0"], polygons["P1"]) and polygons_overlap(polygons["P0"], polygons["P3"])
    # Taking the circular median of the raw angles instead would turn the canvas by 90 degrees.
    assert plan.width < 1.2 * (2 * shape[1]) and plan.height < 1.2 * (2 * shape[0])


def test_panel_overlap_is_decided_on_footprint_polygons_not_bounding_boxes() -> None:
    square = np.asarray([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    diamond = np.asarray([(0.707, 0.0), (0.0, 0.707), (-0.707, 0.0), (0.0, -0.707)])
    # Two diamonds whose bounding boxes overlap in a corner while the shapes stay apart.
    assert not polygons_overlap(diamond, diamond + 1.3)
    assert polygons_overlap(diamond, diamond + (0.5, 0.3))
    # Squares that only share an edge do not overlap; shifted ones do.
    assert not polygons_overlap(square, square + (1.0, 0.0))
    assert polygons_overlap(square, square + (0.9, 0.0))
    assert polygons_overlap(square, 0.5 * square + 0.25)  # one inside the other
    assert panels_connected({"A": square, "B": square + (0.9, 0.0), "C": square + (1.8, 0.0)})
    assert not panels_connected({"A": diamond, "B": diamond + 1.3})


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


def test_the_forward_lattice_drops_reference_points_where_the_window_lattice_samples_them() -> None:
    """The forward map (a drizzle) and the window lattice (the warp) are
    inverse: a reference point lands on the window pixel that samples it,
    and a filter's lateral colour moves the landing of ``p + shift(p)`` to
    where ``p`` lands."""

    footprint = _grid_footprints(step_degrees=0.9, rotations=(1.0,))[0]
    plan = plan_canvas([footprint])
    ra, dec = footprint.boundary_world()
    origin, shape = canvas_window(plan.projection, ra, dec)
    height, width = footprint.shape
    domain = (-3.0, -3.0, width + 2.0, height + 2.0)
    grid = window_grid(plan.projection, origin, shape, footprint.header, footprint.shape, forward_domain=domain)
    assert grid.has_forward_lattice and grid.forward_origin == (-32.0, -32.0)
    rows, columns = grid.forward_x.shape
    assert grid.forward_origin[0] + (columns - 1) * grid.spacing >= domain[2] + grid.spacing
    assert grid.forward_origin[1] + (rows - 1) * grid.spacing >= domain[3] + grid.spacing
    assert "forwardLattice" in grid.serializable() and grid.digest != window_grid(
        plan.projection, origin, shape, footprint.header, footprint.shape
    ).digest

    def forward(x: np.ndarray, y: np.ndarray, nodes_x: np.ndarray, nodes_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The kernel's bilinear arithmetic on the forward lattice.
        gx = (x - grid.forward_origin[0]) / grid.spacing
        gy = (y - grid.forward_origin[1]) / grid.spacing
        cx = np.clip(np.floor(gx), 0, columns - 2).astype(int)
        cy = np.clip(np.floor(gy), 0, rows - 2).astype(int)
        fx, fy = gx - cx, gy - cy
        out = []
        for nodes in (nodes_x, nodes_y):
            top = nodes[cy, cx] + fx * (nodes[cy, cx + 1] - nodes[cy, cx])
            bottom = nodes[cy + 1, cx] + fx * (nodes[cy + 1, cx + 1] - nodes[cy + 1, cx])
            out.append(top + fy * (bottom - top))
        return out[0], out[1]

    # Window nodes with a finite reference position: their reference point
    # must come back onto them.
    valid = np.isfinite(grid.reference_x)
    node_y, node_x = np.mgrid[: grid.node_shape[0], : grid.node_shape[1]].astype(np.float64) * grid.spacing
    back_x, back_y = forward(grid.reference_x[valid], grid.reference_y[valid], grid.forward_x, grid.forward_y)
    assert np.max(np.abs(back_x - node_x[valid])) < 2e-3 and np.max(np.abs(back_y - node_y[valid])) < 2e-3

    # A lateral colour of up to 1.5 px: the filter's point q = p + shift(p)
    # lands where p lands.
    def shift(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u, v = (x - width / 2.0) / width, (y - height / 2.0) / height
        return 1.5 * u**3 + 0.3 * u, -1.2 * v**3 + 0.2 * v

    shifted = forward_lattice(
        plan.projection, origin, footprint.header, grid.forward_origin, grid.forward_x.shape, spacing=grid.spacing, shift=shift
    )
    rng = np.random.default_rng(7)
    px = rng.uniform(0.0, width - 1.0, 200)
    py = rng.uniform(0.0, height - 1.0, 200)
    dx, dy = shift(px, py)
    landing_x, landing_y = forward(px, py, grid.forward_x, grid.forward_y)
    filter_x, filter_y = forward(px + dx, py + dy, shifted["forward_x"], shifted["forward_y"])
    assert np.max(np.hypot(filter_x - landing_x, filter_y - landing_y)) < 2e-3


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


@pytest.mark.parametrize("shift", [0.0, 0.5])
def test_overlap_misregistration_is_the_binned_offset_less_centroid_noise(shift: float) -> None:
    """300 stars along a strip with 0.2 px centroid noise and 5 % blends
    several pixels off: the plain RMS reads the noise and the blends, the
    systematic offset reads only the panels' misregistration."""

    rng = np.random.default_rng(3)
    count = 300
    dx, dy = rng.normal(shift, 0.2, count), rng.normal(0.0, 0.2, count)
    blends = rng.choice(count, count // 20, replace=False)
    dx[blends] += rng.uniform(-6.0, 6.0, blends.size)
    edge = EdgeMeasurement("A", "B", (0, 0, 4000, 600), 4000 * 600)
    edge.star_x, edge.star_y = rng.uniform(0, 4000, count), rng.uniform(0, 600, count)
    edge.offset_x, edge.offset_y = dx, dy
    systematic, scatter, bins = _systematic_offset(edge)
    assert bins == 8 and scatter == pytest.approx(0.3, abs=0.05)
    assert systematic == pytest.approx(shift, abs=0.05)
    assert math.sqrt(np.mean(dx**2 + dy**2)) > 0.7


def test_a_misregistration_varying_along_the_overlap_is_read_without_bias() -> None:
    """80 stars at 0.2 px centroid noise on an overlap whose offset runs
    from -0.5 to +0.5 px along it, 200 times: the noise of the bin medians
    comes from the scatter within the bins, so the variation between them
    is not taken for noise (a scatter over the whole overlap read 12 % low)."""

    rng = np.random.default_rng(5)
    ratios = []
    for _ in range(200):
        x = rng.uniform(0, 4000, 80)
        trend = 0.5 * (2.0 * x / 4000 - 1.0)
        edge = EdgeMeasurement("A", "B", (0, 0, 4000, 600), 4000 * 600)
        edge.star_x, edge.star_y = x, rng.uniform(0, 600, 80)
        edge.offset_x, edge.offset_y = trend + rng.normal(0.0, 0.2, 80), rng.normal(0.0, 0.2, 80)
        systematic, _, bins = _systematic_offset(edge)
        groups = np.array_split(np.argsort(x, kind="stable"), bins)
        ratios.append(systematic / math.sqrt(np.mean([np.mean(trend[group]) ** 2 for group in groups])))
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("gradient", [0.0, 0.03])
def test_panel_gain_gradients_are_fitted_only_when_the_overlaps_carry_them(gradient: float) -> None:
    """A 2x2 mosaic whose masters carry opposite gain gradients of 3 % per
    half canvas (the masters of frames behind a thin-cloud gradient): the
    gain plane recovers them and closes every overlap.  Without gradients
    the BIC keeps one scale per panel."""

    rng = np.random.default_rng(11)
    width, height, step_x, step_y = 1000, 700, 850, 550
    panels = [
        PanelImage(f"P{column}{row}", Path(f"P{column}{row}.fits"), (column * step_x, row * step_y), (height, width), 300.0)
        for column in range(2)
        for row in range(2)
    ]
    centre, length = ((step_x + width - 1) / 2.0, (step_y + height - 1) / 2.0), (step_x + width) / 2.0
    truth = {
        panel.key: (1.0 + 0.1 * position, gradient * (1 if position % 2 else -1), gradient * (1 if position < 2 else -1))
        for position, panel in enumerate(panels)
    }

    def gain(key: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        scale, a, b = truth[key]
        return scale * np.exp(a * (x - centre[0]) / length + b * (y - centre[1]) / length)

    edges = []
    for left_index in range(len(panels)):
        for right_index in range(left_index + 1, len(panels)):
            left, right = panels[left_index], panels[right_index]
            box = (max(left.box[0], right.box[0]), max(left.box[1], right.box[1]), min(left.box[2], right.box[2]), min(left.box[3], right.box[3]))
            count = (box[2] - box[0]) * (box[3] - box[1]) // 400
            edge = EdgeMeasurement(left.key, right.key, box, (box[2] - box[0]) * (box[3] - box[1]))
            edge.star_x, edge.star_y = rng.uniform(box[0], box[2], count), rng.uniform(box[1], box[3], count)
            flux = 10 ** rng.uniform(3.0, 5.0, count)
            edge.flux_left = flux / gain(left.key, edge.star_x, edge.star_y) * rng.normal(1.0, 0.005, count)
            edge.flux_right = flux / gain(right.key, edge.star_x, edge.star_y) * rng.normal(1.0, 0.005, count)
            edge.offset_x, edge.offset_y = rng.normal(0.0, 0.05, count), rng.normal(0.0, 0.05, count)
            edges.append(edge)
    solution = solve_scales(panels, scale_observations(edges))
    assert solution.model == ("gainPlane" if gradient else "scale")
    for record in solution.edges:
        assert abs(record["correctedRatioMinusOne"]) < 0.003
    # Only gains relative to the overlapping panel are observable: they match
    # the truth at every overlap star to 0.5 %, where the truth varies by
    # 1.6-6.5 %.
    for edge in edges:
        fitted = solution.gain(edge.left, edge.star_x, edge.star_y) / solution.gain(edge.right, edge.star_x, edge.star_y)
        true = gain(edge.left, edge.star_x, edge.star_y) / gain(edge.right, edge.star_x, edge.star_y)
        relative = true / np.median(true)
        np.testing.assert_allclose(fitted / np.median(fitted), relative, atol=0.005)
        if gradient:
            assert np.ptp(relative) > (0.005 if {edge.left, edge.right} in ({"P00", "P11"}, {"P01", "P10"}) else 0.02)


@pytest.mark.parametrize(
    ("deviation", "error", "verdict"),
    [
        (0.002, 0.001, "PASS"),
        # Within 0.3 % but too poorly measured to be known within 1 %.
        (0.001, 0.006, "WARN"),
        (0.008, 0.001, "WARN"),
        (0.0178, 0.0074, "WARN"),
        (0.03, 0.005, "FAIL"),
        (None, None, "NOT_MEASURED"),
        (0.002, math.inf, "NOT_MEASURED"),
    ],
)
def test_overlap_photometry_passes_and_fails_only_beyond_its_measurement_error(
    deviation: float | None, error: float | None, verdict: str
) -> None:
    assert _photometry_verdict(deviation, error) == verdict


def _six_panels(
    root: Path, *, curvature: float = 0.0
) -> tuple[list[PanelImage], dict[str, float], tuple[int, int], np.ndarray, np.ndarray]:
    """Six panels of one sky with a bright galaxy whose core sits in the
    middle overlap: each panel with its own transparency, sky plane, seeing
    and noise.  With ``curvature`` every panel's sky also bends by that
    many ADU per canvas unit squared, in u² with the sign of its row and in
    v² with the sign of its column: curvature the overlaps can see (a
    strip along a panel's edge spans it in one direction only)."""

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
            image += curvature * ((1 if row else -1) * u * u + (1 if column % 2 else -1) * v * v)
            path = root / f"P{index}.fits"
            fits.writeto(path, image.astype(np.float32))
            coverage = root / f"P{index}_coverage.fits"
            fits.writeto(coverage, np.ones((height, width), np.float32))
            panels.append(PanelImage(f"P{index}", path, (x0, y0), (height, width), 300.0, coverage, None))
            scales[f"P{index}"] = scale
            index += 1
    return panels, scales, (canvas_w, canvas_h), render(3.5), galaxy


def _sky_residual_rms(mosaic: np.ndarray, truth: np.ndarray, galaxy: np.ndarray) -> float:
    """RMS of the 64 px binned sky residual (stars and the galaxy masked)
    after the unobservable common quadratic, which every panel could carry."""

    height, width = mosaic.shape
    difference = mosaic - truth
    yy, xx = np.mgrid[:height, :width].astype(float)
    u = (xx.ravel() - (width - 1) / 2) / (max(width, height) / 2)
    v = (yy.ravel() - (height - 1) / 2) / (max(width, height) / 2)
    design = np.column_stack((np.ones(u.size), u, v, u * u, u * v, v * v))
    faint = np.isfinite(difference.ravel()) & (galaxy.ravel() < 30)
    common, *_ = np.linalg.lstsq(design[faint], difference.ravel()[faint], rcond=None)
    residual = difference - (design @ common).reshape(difference.shape)
    stars = ndimage.gaussian_filter(np.abs(truth - ndimage.median_filter(truth, 9)), 2.0) > 0.2
    usable = ~stars & (galaxy < 30)
    binned = [
        np.median(residual[i : i + 64, j : j + 64][usable[i : i + 64, j : j + 64]])
        for i in range(0, height - 63, 64)
        for j in range(0, width - 63, 64)
        if usable[i : i + 64, j : j + 64].sum() > 500
    ]
    return float(np.sqrt(np.mean(np.square(binned))))


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
    # Every overlap closes within two standard errors; one of 9 to 24 stars
    # measures its ratio only to 0.5-1.5 %, so it is WARN however close it comes.
    edges = receipt["photometry"]["edges"]
    assert all(abs(item["correctedRatioMinusOne"]) <= 2.0 * item["correctedRatioError"] for item in edges)
    assert {item["photometryVerdict"] for item in edges} == {"PASS", "WARN"}
    assert all(item["correctedRatioError"] > 0.0035 for item in edges if item["photometryVerdict"] == "WARN")
    assert receipt["gates"]["photometry"] == "WARN"
    assert receipt["gates"]["overlapAstrometry"] == "PASS"
    assert receipt["gates"]["fluxConservation"] == "PASS"
    assert receipt["componentCount"] == 1 and "components" not in receipt
    # The galaxy scale of every overlap it crosses agrees with the stars'.
    checks = receipt["background"]["extendedStructureScaleChecks"]
    assert checks and all(item["agreesWithStars"] for item in checks)
    # After the unobservable common sky, the mosaic differs from the truth
    # by a small fraction of the noise: no panel-shaped steps are left.
    mosaic = fits.getdata(result.science_path).astype(float)
    assert _sky_residual_rms(mosaic, sky * true_scales[reference], galaxy) < 0.25  # pixel noise 2.0; the injected planes reach 10


def test_quadratic_sky_per_panel_is_removed_where_the_overlaps_see_its_curvature(tmp_path: Path) -> None:
    """Each panel's sky bends by 8 ADU per canvas unit squared (about half
    the pixel noise across a panel), in u² with the sign of its row and in
    v² with the sign of its column.  The strips along the panels' edges
    span them in one direction, so the network sees these terms: the
    mosaic's sky is within 0.1 pixel noise of the truth, and the fitted
    curvature differences between rows and columns have the injected sign
    and 40-115 % of its size (the prior shrinks what a strip sees weakly)."""

    panels, true_scales, (width, height), sky, galaxy = _six_panels(tmp_path, curvature=8.0)
    projection = CanvasProjection("TAN", (10.0, 41.0), (1.0, 1.0), ((-1 / 3600, 0.0), (0.0, 1 / 3600)))
    output = tmp_path / "mosaic"
    output.mkdir()
    result = assemble_filter("L", panels, projection=projection, canvas_box=(0, 0, width, height), output_directory=output, verifier=None)
    receipt = result.receipt
    assert receipt["gates"]["residualBackground"] == "PASS" and receipt["background"]["terms"] == 6
    reference = receipt["photometry"]["scaleReference"]
    mosaic = fits.getdata(result.science_path).astype(float)
    assert _sky_residual_rms(mosaic, sky * true_scales[reference], galaxy) < 0.2  # 0.1 of the pixel noise 2.0
    planes = {panel["key"]: panel["plane"] for panel in receipt["panels"]}
    gains = {panel["key"]: panel["scale"] for panel in receipt["panels"]}
    # The additive terms are fitted on the scaled panels, so a difference of
    # 8 per panel reads 8 times the sum of the two gains.  Panels P0, P2, P4
    # are the top row and P1, P3, P5 the bottom row; the columns are (P0,
    # P1), (P2, P3), (P4, P5).  The curvature prior (a tenth of the pixel
    # noise across a panel) shrinks a difference the strip measures weakly:
    # the sign is always right and most of the curvature is recovered.
    for top, bottom in (("P0", "P1"), ("P2", "P3"), ("P4", "P5")):
        expected = 8.0 * (gains[top] + gains[bottom])
        assert 0.6 * expected < planes[bottom][3] - planes[top][3] < 1.15 * expected
    for left, right in (("P0", "P2"), ("P2", "P4"), ("P1", "P3"), ("P3", "P5")):
        expected = 8.0 * (gains[left] + gains[right]) * (1.0 if left in ("P0", "P1") else -1.0)
        recovered = planes[right][5] - planes[left][5]
        assert 0.4 * abs(expected) < recovered * np.sign(expected) < 1.15 * abs(expected)


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



def _three_panels(root: Path, *, galaxy: bool) -> tuple[list[PanelImage], dict[str, float], tuple[int, int]]:
    """Three panels in a row with 200 px overlaps.  The A-B overlap is full
    of stars; the B-C overlap holds none, and with ``galaxy`` a bright
    galaxy crosses it instead."""

    rng = np.random.default_rng(23)
    width, height, step = 640, 480, 440
    canvas_w = 2 * step + width
    count = 700
    sx, sy = rng.uniform(5, canvas_w - 5, count), rng.uniform(5, height - 5, count)
    keep = (sx < 850) | (sx > 1110)
    sx, sy = sx[keep], sy[keep]
    flux = 10 ** rng.uniform(3.0, 5.0, sx.size)
    points = np.zeros((height, canvas_w))
    np.add.at(points, (np.round(sy).astype(int), np.round(sx).astype(int)), flux)
    yy, xx = np.mgrid[:height, :canvas_w].astype(float)
    structure = 400.0 * np.exp(-np.hypot(xx - 980.0, yy - 240.0) / 300.0) if galaxy else 0.0
    panels, scales = [], {}
    for index, (key, scale) in enumerate((("A", 1.0), ("B", 0.9), ("C", 1.1))):
        x0 = index * step
        image = (ndimage.gaussian_filter(points, (2.5 + 0.5 * index) / 2.3548) + structure + 100.0)[:, x0 : x0 + width] * scale
        u = (xx[:, x0 : x0 + width] - (canvas_w - 1) / 2) / (canvas_w / 2)
        image += rng.uniform(-8, 8) + rng.uniform(-8, 8) * u + rng.normal(0.0, 2.0, image.shape)
        path, coverage = root / f"{key}.fits", root / f"{key}_coverage.fits"
        fits.writeto(path, image.astype(np.float32))
        fits.writeto(coverage, np.ones((height, width), np.float32))
        panels.append(PanelImage(key, path, (x0, 0), (height, width), 300.0, coverage, None))
        scales[key] = scale
    return panels, scales, (canvas_w, height)


@pytest.mark.parametrize("galaxy", [False, True], ids=["empty", "galaxy"])
def test_panels_without_a_photometric_link_are_published_as_components(tmp_path: Path, galaxy: bool) -> None:
    """The B-C overlap has no common stars (a bright galaxy crossing it does
    not link it either): C is a photometric component of its own.  A and B
    are the filter's mosaic on the canvas box, C is published beside it on
    its own box with its own receipt, and the filter is WARN.  A FAIL in C
    fails the filter after both components were gated."""

    panels, true_scales, (width, height) = _three_panels(tmp_path, galaxy=galaxy)
    projection = CanvasProjection("TAN", (10.0, 41.0), (1.0, 1.0), ((-1 / 3600, 0.0), (0.0, 1 / 3600)))
    output = tmp_path / "mosaic"
    output.mkdir()
    result = assemble_filter("L", panels, projection=projection, canvas_box=(0, 0, width, height), output_directory=output, verifier=None)
    receipt = result.receipt
    canonical_json(receipt)  # no NaN or infinity in a receipt
    assert receipt["componentCount"] == 2 and receipt["component"] == 1
    assert [panel["key"] for panel in receipt["panels"]] == ["A", "B"]
    assert receipt["gates"]["photometricComponents"] == "WARN" and receipt["verdict"] == "WARN"
    assert receipt["warnings"][0]["code"] == "MOSAIC_PHOTOMETRIC_COMPONENTS"
    assert receipt["warnings"][0]["components"] == [["A", "B"], ["C"]]
    (unmatched,) = receipt["warnings"][0]["unmatchedOverlaps"]
    assert (unmatched["left"], unmatched["right"]) == ("B", "C") and unmatched["stars"] < 8
    assert receipt["photometry"]["scaleReference"] in ("A", "B")
    (component,) = result.components
    assert component.science_path == output / "component2" / "mosaic_L_component2.fits"
    assert component.receipt["component"] == 2 and [panel["key"] for panel in component.receipt["panels"]] == ["C"]
    assert receipt["components"] == [component.receipt]
    # A single panel has no overlap to measure.
    assert {name: component.receipt["gates"][name] for name in ("photometry", "overlapAstrometry", "residualBackground", "fluxConservation", "photometricNetwork", "unmeasuredOverlaps")} == dict.fromkeys(
        ("photometry", "overlapAstrometry", "residualBackground", "fluxConservation", "photometricNetwork", "unmeasuredOverlaps"), "NOT_MEASURED"
    )
    # The mosaic covers the canvas box, C's component only C's window of it.
    main, other = fits.getdata(result.science_path), fits.getdata(component.science_path)
    assert main.shape == (height, width) and other.shape == (height, 640)
    assert np.isfinite(main[:, :1080]).all() and not np.isfinite(main[:, 1080:]).any()
    assert np.isfinite(other).all()
    assert fits.getheader(component.science_path)["CRPIX1"] == fits.getheader(result.science_path)["CRPIX1"] - 880
    assert fits.getheader(component.science_path)["OAFCOMP"] == 2
    assert receipt["panels"][1]["scale"] * true_scales["B"] / true_scales["A"] == pytest.approx(receipt["panels"][0]["scale"], rel=3e-3)

    corner = fits.getheader(result.science_path)["CRPIX1"]

    def verifier(image: np.ndarray, header: Mapping[str, Any], artifacts: Path) -> Mapping[str, Any]:
        # Panel C's window starts 880 px into the canvas, A's and B's at 0 and 440.
        return {"matchedStars": 0 if header["CRPIX1"] < corner - 600 else 50, "rmsArcsec": 0.5}

    failing = tmp_path / "failing"
    failing.mkdir()
    with pytest.raises(MosaicGateError, match=r"canvasWcs \(component 2\)") as failure:
        assemble_filter("L", panels, projection=projection, canvas_box=(0, 0, width, height), output_directory=failing, verifier=verifier)
    assert failure.value.receipt["gates"]["canvasWcs"] == "PASS"
    assert failure.value.receipt["components"][0]["gates"]["canvasWcs"] == "FAIL"


def _lattice_edges(
    panels: Sequence[PanelImage], rng: np.random.Generator, *, stars: int, truth: dict[str, np.ndarray], gains: dict[str, float]
) -> list[EdgeMeasurement]:
    """Edges of every overlapping pair: stars with fluxes in each panel's
    gain, and bins on the BIN_PIXELS lattice carrying each panel's additive
    polynomial (``truth``, canvas-normalized terms) and noise."""

    keys = [panel.key for panel in panels]
    boxes = {panel.key: panel.box for panel in panels}
    bx1 = max(box[2] for box in boxes.values())
    by1 = max(box[3] for box in boxes.values())
    centre, length = ((bx1 - 1) / 2.0, (by1 - 1) / 2.0), max(bx1, by1) / 2.0
    edges = []
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            a, b = boxes[left], boxes[right]
            box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
            if box[2] - box[0] < BIN_PIXELS or box[3] - box[1] < BIN_PIXELS:
                continue
            edge = EdgeMeasurement(left, right, box, (box[2] - box[0]) * (box[3] - box[1]))
            edge.star_x, edge.star_y = rng.uniform(box[0], box[2], stars), rng.uniform(box[1], box[3], stars)
            flux = 10 ** rng.uniform(3.0, 5.0, stars)
            edge.flux_left = flux / gains[left] * rng.normal(1.0, 0.005, stars)
            edge.flux_right = flux / gains[right] * rng.normal(1.0, 0.005, stars)
            edge.offset_x, edge.offset_y = rng.normal(0.0, 0.05, stars), rng.normal(0.0, 0.05, stars)
            columns, rows = (box[2] - box[0]) // BIN_PIXELS, (box[3] - box[1]) // BIN_PIXELS
            by, bx = np.mgrid[:rows, :columns]
            edge.bin_x = box[0] + (bx.ravel() + 0.5) * BIN_PIXELS
            edge.bin_y = box[1] + (by.ravel() + 0.5) * BIN_PIXELS
            u, v = (edge.bin_x - centre[0]) / length, (edge.bin_y - centre[1]) / length
            terms = np.column_stack((np.ones_like(u), u, v, u * u, u * v, v * v))
            noise = 0.11
            edge.bin_left = (100.0 + terms @ truth[left]) / gains[left] + rng.normal(0.0, noise, u.size)
            edge.bin_right = (100.0 + terms @ truth[right]) / gains[right] + rng.normal(0.0, noise, u.size)
            edge.bin_noise = np.full(u.size, math.sqrt(2.0) * noise)
            edges.append(edge)
    return edges


def test_six_by_six_networks_solve_through_sparse_normal_equations() -> None:
    """A 6x6 mosaic of 1600x1200 panels with 300 px overlaps: about 24 000
    background bins over 216 unknowns.  A dense design would take 41 MB per
    copy and its SVD several times that; the normal equations keep the
    solve within the bins' own memory and a few seconds, and the gains and
    planes come out right."""

    import time
    import tracemalloc

    rng = np.random.default_rng(29)
    width, height, step_x, step_y = 1600, 1200, 1300, 950
    panels = [
        PanelImage(f"P{column}{row}", Path(f"P{column}{row}.fits"), (column * step_x, row * step_y), (height, width), 300.0)
        for column in range(6)
        for row in range(6)
    ]
    gains = {panel.key: float(rng.uniform(0.8, 1.2)) for panel in panels}
    truth = {panel.key: np.concatenate((rng.uniform(-8, 8, 3), np.zeros(3))) for panel in panels}
    edges = _lattice_edges(panels, rng, stars=150, truth=truth, gains=gains)
    assert sum(edge.bin_x.size for edge in edges) > 20000
    tracemalloc.start()
    started = time.perf_counter()
    scales = solve_scales(panels, scale_observations(edges))
    planes = solve_planes(panels, edges, scales, (5 * step_y + height, 5 * step_x + width))
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert elapsed < 20.0
    assert peak < 24 * 1024 * 1024
    assert len(scales.edges) == len(edges)
    reference = scales.reference
    for panel in panels:
        assert scales.scales[panel.key] * gains[reference] / gains[panel.key] == pytest.approx(1.0, abs=3e-3)
    # The fitted planes match the injected ones (in the reference panel's
    # gain units) up to the unobservable common polynomial: every pairwise
    # difference agrees to the bin noise.
    fitted = {key: np.asarray(value) * gains[reference] for key, value in planes.coefficients.items()}
    for edge in edges:
        difference = (fitted[edge.left] - fitted[edge.right]) - (truth[edge.left] - truth[edge.right])
        model = np.column_stack((np.ones(edge.bin_x.size), *(
            ((edge.bin_x - planes.centre[0]) / planes.length) ** p * ((edge.bin_y - planes.centre[1]) / planes.length) ** q
            for p, q in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
        ))) @ difference
        assert np.max(np.abs(model)) < 0.3


def test_the_sparse_robust_solver_equals_its_dense_counterpart() -> None:
    """The normal-equation solver returns the minimum-norm Huber-weighted
    solution a dense least-squares fit of the same rows gives."""

    rng = np.random.default_rng(31)
    unknowns, rows = 12, 400
    design = _Design(unknowns)
    # Rows that difference two blocks of three unknowns: a common vector
    # across the blocks is the null space.
    left, right = rng.integers(0, 4, rows), rng.integers(0, 4, rows)
    left, right = np.where(left == right, (left + 1) % 4, left), right
    u, v = rng.uniform(-1, 1, rows), rng.uniform(-1, 1, rows)
    columns = np.column_stack((3 * left, 3 * left + 1, 3 * left + 2, 3 * right, 3 * right + 1, 3 * right + 2))
    values = np.column_stack((np.ones(rows), u, v, -np.ones(rows), -u, -v))
    truth = rng.uniform(-5, 5, unknowns)
    sigma = rng.uniform(0.05, 0.2, rows)
    target = (values * truth[columns]).sum(axis=1) + rng.normal(0.0, sigma)
    target[:10] += 3.0  # outliers the Huber weights must take down
    design.add(columns, values, target, sigma)
    design.add(np.asarray([[1], [2]]), np.ones((2, 1)), np.zeros(2), np.full(2, 0.05))
    solution, weights, residual, covariance = _robust_solve(design, robust_rows=rows, covariance=True)
    dense = design.matrix().toarray()
    full_target, full_sigma = np.concatenate(design.targets), np.concatenate(design.sigmas)
    reference_weights = np.ones(dense.shape[0])
    for _ in range(4):
        scaled = dense * (reference_weights / full_sigma)[:, None]
        reference = np.linalg.lstsq(scaled, full_target * reference_weights / full_sigma, rcond=None)[0]
        normalized = (full_target - dense @ reference) / full_sigma
        normalized[rows:] = 0.0
        size = np.abs(normalized)
        reference_weights = np.where(size <= HUBER_THRESHOLD, 1.0, np.sqrt(HUBER_THRESHOLD / np.maximum(size, 1e-12)))
    np.testing.assert_allclose(solution, reference, atol=1e-8)
    np.testing.assert_allclose(weights, reference_weights, atol=1e-8)
    assert np.count_nonzero(weights[:10] < 1.0) == 10
    scaled = dense * (reference_weights / full_sigma)[:, None]
    np.testing.assert_allclose(covariance, np.linalg.pinv(scaled.T @ scaled), atol=1e-8)
    # The null space the priors leave (a common constant across the blocks) stays at zero.
    assert abs(solution[0::3].sum()) < 1e-8
