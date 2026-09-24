"""Region weight maps built from the Light Frame QC spatial grid."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ufwbpp.selection.region import (
    NEGLIGIBLE_FLOOR,
    RegionWeightMap,
    evaluate_weight_grid_rows,
    region_weight_map,
    region_weight_maps,
)

ROWS = COLUMNS = 16


def _grid(**overrides):
    expected = [[5] * COLUMNS for _ in range(ROWS)]
    grid = {
        "rows": ROWS,
        "columns": COLUMNS,
        "expectedStars": expected,
        "matchedStars": [[5] * COLUMNS for _ in range(ROWS)],
        "transparencyResidualMag": [[0.0] * COLUMNS for _ in range(ROWS)],
        "backgroundDeltaRobustSigma": [[0.0] * COLUMNS for _ in range(ROWS)],
        "textureRatio": [[1.0] * COLUMNS for _ in range(ROWS)],
    }
    grid.update(overrides)
    return grid


def test_clean_grid_needs_no_map() -> None:
    assert region_weight_map(_grid(), "clean") is None
    assert region_weight_map(None, "missing") is None
    assert region_weight_map({"rows": 2, "columns": 2}, "tiny") is None


def test_missing_star_component_is_zeroed_with_a_margin() -> None:
    matched = [[5] * COLUMNS for _ in range(ROWS)]
    for row in range(4, 8):
        for column in range(2, 6):
            matched[row][column] = 0
    built = region_weight_map(_grid(matchedStars=matched), "occluded")
    assert built is not None
    nodes = built.as_array()
    # The component and its one-cell margin are exactly zero.
    assert np.all(nodes[3:9, 1:7] == 0.0)
    assert built.zero_cells == 6 * 6
    assert built.zero_fraction == pytest.approx(36 / 256)
    # The ramp outside the margin rises towards one and far cells are untouched.
    assert 0.0 < nodes[2, 3] < 1.0
    assert np.all(nodes[12:, 10:] == 1.0)
    assert built.evidence["missingCells"] == 16
    assert built.evidence["zeroCellsWithMargin"] == 36
    serialized = built.serializable()
    assert serialized["rows"] == ROWS and len(serialized["nodes"]) == ROWS
    assert serialized["nodes"][5][3] == 0.0 and serialized["nodes"][15][15] == 1.0


def test_single_missing_cell_is_not_an_occluder() -> None:
    matched = [[5] * COLUMNS for _ in range(ROWS)]
    matched[8][8] = 0
    assert region_weight_map(_grid(matchedStars=matched), "speck") is None
    # Two diagonal missing cells are one small occluder with a margin.
    matched[9][9] = 0
    built = region_weight_map(_grid(matchedStars=matched), "pair")
    assert built is not None
    assert built.evidence["missingCells"] == 2
    nodes = built.as_array()
    # Each missing cell and its eight neighbours are blanked (two 3x3 blocks, four cells shared).
    assert built.zero_cells == 14
    assert nodes[7, 7] == 0.0 and nodes[10, 10] == 0.0 and nodes[7, 10] > 0.0


def _patch(value: float, cells) -> list[list[float]]:
    residual = [[0.0] * COLUMNS for _ in range(ROWS)]
    for row, column in cells:
        residual[row][column] = value
    return residual


def test_dimming_needs_a_coherent_patch_beyond_the_dead_band() -> None:
    patch = [(10, 10), (10, 11), (11, 10), (11, 11)]
    residual = _patch(0.08 + 0.45, patch)  # full loss at the dead band plus the ramp
    residual[2][2] = 0.07  # inside the dead band: untouched
    residual[5][13] = 0.40  # a lone faint-star outlier: untouched
    built = region_weight_map(_grid(transparencyResidualMag=residual), "cloudy")
    assert built is not None
    nodes = built.as_array()
    # The dimming term is not smoothed: the patch is fully lost, its neighbours untouched.
    assert np.all(nodes[10:12, 10:12] == 0.0)
    assert nodes[10, 9] == 1.0 and nodes[9, 10] == 1.0
    assert nodes[2, 2] == 1.0 and nodes[5, 13] == 1.0
    assert built.evidence["dimmedCells"] == 4
    assert built.evidence["dimmedCandidateCells"] == 5
    assert built.evidence["residualSource"] == "reference"
    # Fully dimmed cells are zero cells, but not margin-dilated ones.
    assert built.zero_cells == 4
    assert built.evidence["zeroCellsWithMargin"] == 0
    # Half the ramp times the transmission of a 0.305 mag loss (a 3x3 patch:
    # a 2x2 one at this level would be cosmetic and dropped).
    half = _patch(0.08 + 0.45 / 2, [(r, c) for r in range(9, 12) for c in range(9, 12)])
    assert region_weight_map(_grid(transparencyResidualMag=half), "half").as_array()[10, 10] == pytest.approx(
        0.5 * 10 ** (-0.4 * (0.08 + 0.45 / 2))
    )
    # Four diagonal cells are one patch; three are not enough, and a shallow
    # patch whose median barely clears the dead band does not count.
    diagonal = _patch(0.5, [(3, 3), (4, 4), (5, 5), (6, 6)])
    assert region_weight_map(_grid(transparencyResidualMag=diagonal), "diag").evidence["dimmedCells"] == 4
    assert region_weight_map(_grid(transparencyResidualMag=_patch(0.5, [(3, 3), (4, 4), (5, 5)])), "three") is None
    shallow = _patch(0.10, [(3, 3), (3, 4), (4, 3), (4, 4), (5, 5)])
    assert region_weight_map(_grid(transparencyResidualMag=shallow), "shallow") is None


def test_consensus_residual_is_preferred_over_the_reference_residual() -> None:
    patch = [(6, 6), (6, 7), (7, 6), (7, 7)]
    shared = _patch(0.6, patch)  # the reference residual shows a patch every frame shares
    flat = [[0.0] * COLUMNS for _ in range(ROWS)]
    assert region_weight_map(
        _grid(transparencyResidualMag=shared, consensusDimmingResidualMag=flat), "shared"
    ) is None
    own = region_weight_map(
        _grid(transparencyResidualMag=flat, consensusDimmingResidualMag=shared), "own"
    )
    assert own is not None and own.evidence["residualSource"] == "consensus"
    assert np.all(own.as_array()[6:8, 6:8] == 0.0)


def test_background_anomaly_needs_a_texture_loss_to_zero_a_cell() -> None:
    background = [[0.0] * COLUMNS for _ in range(ROWS)]
    texture = [[1.0] * COLUMNS for _ in range(ROWS)]
    background[3][3] = 5.0  # bright but textured: a gradient, kept
    background[9][9] = -4.0
    texture[9][9] = 0.2  # dark and featureless: an opaque blocker
    built = region_weight_map(
        _grid(backgroundDeltaRobustSigma=background, textureRatio=texture), "blocked"
    )
    assert built is not None
    nodes = built.as_array()
    assert nodes[9, 9] == 0.0
    assert nodes[3, 3] == 1.0
    assert built.evidence["backgroundCells"] == 1


def test_negligible_maps_are_dropped() -> None:
    residual = _patch(0.085, [(5, 5), (5, 6), (6, 5), (6, 6)])  # a patch a hair beyond the dead band
    assert region_weight_map(_grid(transparencyResidualMag=residual), "hair") is None
    # A small reduction with no zero cell is cosmetic (mean above 0.99) and dropped too.
    tiny = _patch(0.20, [(5, 5), (5, 6), (6, 5), (6, 6)])
    assert region_weight_map(_grid(transparencyResidualMag=tiny), "tiny") is None
    wider = _patch(0.20, [(r, c) for r in range(5, 8) for c in range(5, 8)])
    assert region_weight_map(_grid(transparencyResidualMag=wider), "wider") is not None
    assert NEGLIGIBLE_FLOOR > 0.9


def test_pixel_nodes_are_cell_centres_and_evaluation_clamps() -> None:
    nodes = np.ones((4, 4))
    nodes[:, 0] = 0.0
    built = RegionWeightMap("p", 4, 4, tuple(map(tuple, nodes)), 4, 0.0, 0.75, {})
    x, y = built.pixel_nodes(height=40, width=80)
    assert x == (10.0, 30.0, 50.0, 70.0)
    assert y == (5.0, 15.0, 25.0, 35.0)
    rows = evaluate_weight_grid_rows(built.nodes, x, y, 0, 40, 80)
    assert rows.shape == (40, 80) and rows.dtype == np.float32
    # Left of the first node the value clamps to the node; between the first
    # two nodes it ramps linearly; beyond the second node it is one.
    assert np.all(rows[:, :10] == 0.0)
    assert rows[7, 20] == pytest.approx(0.5)
    assert np.all(rows[:, 30:] == 1.0)
    with pytest.raises(ValueError):
        evaluate_weight_grid_rows(built.nodes, x[:3], y, 0, 40, 80)


def test_region_weight_maps_filter_by_path(tmp_path) -> None:
    matched = [[5] * COLUMNS for _ in range(ROWS)]
    for row in range(0, 3):
        for column in range(0, 3):
            matched[row][column] = 0
    occluded = tmp_path / "occluded.fits"
    clean = tmp_path / "clean.fits"
    occluded.write_bytes(b"")
    clean.write_bytes(b"")
    results = [
        SimpleNamespace(path=str(occluded), grid=_grid(matchedStars=matched)),
        SimpleNamespace(path=str(clean), grid=_grid()),
    ]
    maps = region_weight_maps(results)
    assert set(maps) == {str(occluded.resolve())}
    assert region_weight_maps(results, paths=[str(clean)]) == {}


def test_transformed_map_follows_a_meridian_flip_and_a_dither() -> None:
    nodes = np.ones((16, 16))
    nodes[0:4, 0:4] = 0.0  # a blocked top-left corner in the QC reference frame
    built = RegionWeightMap("f", 16, 16, tuple(map(tuple, nodes)), 16, 0.0, float(nodes.mean()), {"frame": "qc-reference"})
    height, width = 400, 640
    identity = built.transformed(np.eye(3), height, width)
    np.testing.assert_array_equal(identity.as_array(), nodes)
    assert identity.evidence["frame"] == "registered"
    # A 180-degree rotation (registered pixel -> QC pixel) moves the corner to the bottom right.
    flip = np.array([[-1.0, 0.0, width], [0.0, -1.0, height], [0.0, 0.0, 1.0]])
    flipped = built.transformed(flip, height, width).as_array()
    assert np.all(flipped[12:16, 12:16] == 0.0)
    assert np.all(flipped[0:4, 0:4] == 1.0)
    assert flipped.sum() == pytest.approx(nodes.sum())
    # A dither of one cell to the right shifts the corner by one cell, and
    # target pixels that fall outside the QC frame carry weight 1.
    shift = np.array([[1.0, 0.0, -width / 16.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    shifted = built.transformed(shift, height, width).as_array()
    assert np.all(shifted[0:4, 1:5] == 0.0)
    assert np.all(shifted[:, 0] == 1.0)
    # A sub-cell shift keeps the blanked area's size: the nearest cell decides.
    half = np.array([[1.0, 0.0, -0.4 * width / 16.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    half_shifted = built.transformed(half, height, width)
    assert half_shifted.zero_cells == 16
    with pytest.raises(ValueError):
        built.transformed(np.zeros((2, 2)), height, width)
