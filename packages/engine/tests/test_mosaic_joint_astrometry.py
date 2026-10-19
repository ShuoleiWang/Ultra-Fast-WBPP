"""Mosaic panels' solutions fitted at once on catalog and overlap stars, and
the shift fields a panel's Lights keep against its reference."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mosaic_synthetic import Sky, TruthCatalog, make_sky, render, true_header
from ufwbpp.mosaic.astrometry import measure_frame_field, measure_panel, solve_mosaic
from ufwbpp.mosaic.canvas import celestial_wcs
from ufwbpp.stacking.integration import CalibrationError
from ufwbpp.stacking.parameters import OutputGrid, PixelTransform, ShiftField
from ufwbpp.stacking.run_plan import _resolve_frame_fields


SHAPE = (600, 900)
SCALE_ARCSEC = 2.0


def _panel(sky: Sky, centre: tuple[float, float], seed: int) -> tuple[np.ndarray, dict, dict]:
    """A frame with 3 px of cubic barrel at the corners, and the solver's pure-TAN view of it."""

    truth = true_header(centre, SHAPE, scale_arcsec=SCALE_ARCSEC, distortion=3.0)
    image = render(sky, truth, SHAPE, fwhm=3.0, background=100.0, noise=2.0, rng=np.random.default_rng(seed))
    tan = {key: value for key, value in truth.items() if not key.startswith(("A_", "B_"))}
    tan.update(CTYPE1="RA---TAN", CTYPE2="DEC--TAN")
    return image.astype(np.float32), truth, tan


def _sparse_catalog(sky: Sky, count: int, seed: int) -> TruthCatalog:
    """The ``count`` brightest stars, each off by 0.3 arcsec."""

    bright = np.argsort(-sky.flux)[:count]
    noise = np.random.default_rng(seed).normal(0.0, 0.3 / 3600.0, (bright.size, 2))
    dec = sky.dec[bright]
    return TruthCatalog(Sky(sky.ra[bright] + noise[:, 0] / np.cos(np.radians(dec)), dec + noise[:, 1], sky.flux[bright]))


def _errors(fit: dict, truth: dict, overlap_with: dict | None = None) -> np.ndarray:
    """Where ``fit`` puts sky points across the frame (or its overlap with
    ``overlap_with``) minus where they truly are, in pixels."""

    grid_x, grid_y = np.meshgrid(np.linspace(0, SHAPE[1] - 1, 31), np.linspace(0, SHAPE[0] - 1, 21))
    ra, dec = celestial_wcs(truth).all_pix2world(grid_x.ravel(), grid_y.ravel(), 0)
    if overlap_with is not None:
        other = np.column_stack(celestial_wcs(overlap_with).all_world2pix(ra, dec, 0))
        inside = np.all((other >= 0.0) & (other <= np.asarray(SHAPE[::-1]) - 1.0), axis=1)
        ra, dec = ra[inside], dec[inside]
    fitted = np.column_stack(celestial_wcs(fit).all_world2pix(ra, dec, 0, quiet=True))
    true = np.column_stack(celestial_wcs(truth).all_world2pix(ra, dec, 0, quiet=True))
    return fitted - true


def _rms(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(vectors), axis=1))))


def test_a_sparse_catalog_is_tied_together_by_the_overlap_stars() -> None:
    """Two panels overlapping by a fifth and a catalog of about 30 stars per
    frame with 0.3 arcsec errors: fitted alone the panels disagree by half a
    pixel in their overlap (what a seam shows); fitted at once they agree,
    and each stays as close to the sky as its own fit."""

    first_centre = (150.0, 2.0)
    shift = 0.8 * SHAPE[1] * SCALE_ARCSEC / 3600.0
    second_centre = (first_centre[0] + shift / math.cos(math.radians(first_centre[1])), first_centre[1])
    sky = make_sky((first_centre[0] + shift / 2.0, first_centre[1]), 0.6, 2400, seed=7, flux_range=(3.4, 5.0))
    catalog = _sparse_catalog(sky, 270, 9)
    image_a, truth_a, tan_a = _panel(sky, first_centre, 1)
    image_b, truth_b, tan_b = _panel(sky, second_centre, 2)
    first = measure_panel(image_a, tan_a, catalog, minimum_matches=20)
    second = measure_panel(image_b, tan_b, catalog, minimum_matches=20)

    solutions, record = solve_mosaic({"a": first, "b": second})

    (overlap,) = record["overlaps"]
    assert overlap["panels"] == ["a", "b"] and overlap["inliers"] >= 30 and overlap["rmsPixels"] < 0.1
    separate = _rms(_errors(first.fit.header, truth_a, truth_b) - _errors(second.fit.header, truth_b, truth_a))
    joint = _rms(_errors(solutions["a"].header, truth_a, truth_b) - _errors(solutions["b"].header, truth_b, truth_a))
    assert separate > 0.3
    assert joint < 0.05 and joint < 0.1 * separate
    for key, truth in (("a", truth_a), ("b", truth_b)):
        assert _rms(_errors(solutions[key].header, truth)) < 0.15
        assert solutions[key].evidence["overlapStars"] == overlap["inliers"]


def test_panels_of_one_optics_share_their_distortion() -> None:
    """Three panels in a row through the same optics: one distortion for
    all is what the BIC keeps, and only neighbours overlap."""

    centre = (80.0, 30.0)
    step = 0.75 * SHAPE[1] * SCALE_ARCSEC / 3600.0 / math.cos(math.radians(centre[1]))
    sky = make_sky(centre, 0.7, 3600, seed=11, flux_range=(3.4, 5.0))
    catalog = _sparse_catalog(sky, 500, 12)
    panels, truths = {}, {}
    for index, offset in enumerate((-1, 0, 1)):
        image, truth, tan = _panel(sky, (centre[0] + offset * step, centre[1]), 20 + index)
        panels[f"P{index}"] = measure_panel(image, tan, catalog, minimum_matches=20)
        truths[f"P{index}"] = truth

    solutions, record = solve_mosaic(panels)

    assert record["distortion"] == "shared" and record["order"] == 3
    assert sorted(tuple(item["panels"]) for item in record["overlaps"]) == [("P0", "P1"), ("P1", "P2")]
    for key, solution in solutions.items():
        assert _rms(_errors(solution.header, truths[key])) < 0.2


@pytest.mark.parametrize("covered", [1.0, 0.4], ids=["whole-frame", "cloud-cut"])
def test_a_shift_field_holds_only_where_its_stars_determine_it(covered: float) -> None:
    """A Light registered by a dithered, rotated matrix keeps a residual
    shift (cubic, 0.67 px at the corners) on stars measured to 0.1 px: over
    the whole frame the cubic is recovered and the lattice node at each
    star's reference position samples that star in the Light; with stars on
    the left 40 % only, a cubic would put the right side pixels off, so the
    fit keeps an order its stars determine there."""

    height, width = shape = (4000, 6000)
    centre, length = ((width - 1) / 2.0, (height - 1) / 2.0), width / 2.0

    def truth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u, v = (x - centre[0]) / length, (y - centre[1]) / length
        return np.stack((0.35 * u * (u * u + v * v) + 0.05 * v, 0.35 * v * (u * u + v * v) - 0.08 * u * v))

    angle = math.radians(0.1)
    matrix = np.asarray(
        [[math.cos(angle), -math.sin(angle), 12.5], [math.sin(angle), math.cos(angle), -7.25], [0.0, 0.0, 1.0]]
    )

    def in_light(points: np.ndarray) -> np.ndarray:
        """The Light pixel that shows reference pixel ``p`` there: H^-1(p + shift(p))."""

        shown = points + truth(points[:, 0], points[:, 1]).T
        mapped = np.column_stack((shown, np.ones(len(shown)))) @ np.linalg.inv(matrix).T
        return mapped[:, :2] / mapped[:, 2:3]

    rng = np.random.default_rng(3)
    count = int(1000 * covered)
    reference = np.column_stack((rng.uniform(0.0, covered * (width - 1), count), rng.uniform(0.0, height - 1, count)))
    source = in_light(reference) + rng.normal(0.0, 0.1, (count, 2))

    field = measure_frame_field((source, reference), matrix, shape)

    grid_x, grid_y = np.meshgrid(np.linspace(0, width - 1, 61), np.linspace(0, height - 1, 41))
    error = np.hypot(*(np.asarray(field.shift(grid_x, grid_y)) - truth(grid_x, grid_y)))
    assert field.evidence["standardErrorPixels"] <= 0.1
    if covered == 1.0:
        assert field.order == 3 and error.max() < 0.08
        # The shifted lattice sends each node onto the Light's star; unshifted it misses by up to 0.67 px.
        nodes = reference[:36]
        lattice = OutputGrid(6, 6, 1, nodes[:, 0].reshape(6, 6).copy(), nodes[:, 1].reshape(6, 6).copy(), (0, 0), {})
        stars = in_light(nodes)
        transform = PixelTransform.from_value(matrix)
        sampled = np.column_stack([item.ravel() for item in lattice.shifted(field).frame_lattice(transform)])
        unshifted = np.column_stack([item.ravel() for item in lattice.frame_lattice(transform)])
        assert np.max(np.hypot(*(sampled - stars).T)) < 0.05
        assert np.max(np.hypot(*(unshifted - stars).T)) > 0.2
    else:
        assert field.order is not None and field.order < 3 and error.max() < 0.5


@pytest.mark.parametrize("case", ["no-canvas", "unknown-light", "same-light-twice", "invalid-field"])
def test_shift_fields_bind_only_the_admitted_lights_of_a_canvas_panel(tmp_path: Path, case: str) -> None:
    light, other = tmp_path / "light.fits", tmp_path / "other.fits"
    light.write_bytes(b"")
    other.write_bytes(b"")
    field = ShiftField(0, (0.1,), (0.2,), (10.0, 10.0), 10.0)
    fields, code = {
        "no-canvas": ({str(light): field}, "FRAME_FIELDS_UNSUPPORTED"),
        "unknown-light": ({str(other): field}, "FRAME_FIELD_INPUT_UNKNOWN"),
        "same-light-twice": ({str(light): field, os.path.join(tmp_path, ".", "light.fits"): field}, "FRAME_FIELD_KEY_CONFLICT"),
        "invalid-field": ({str(light): ShiftField(5, (), (), (10.0, 10.0), 10.0)}, "FRAME_FIELD_INVALID"),
    }[case]
    parameters = SimpleNamespace(
        output_grid=None if case == "no-canvas" else object(), channel_grids=(), capture_drizzle_inputs=False
    )
    with pytest.raises(CalibrationError) as error:
        _resolve_frame_fields((light.resolve(),), fields, parameters)
    assert error.value.code == code
