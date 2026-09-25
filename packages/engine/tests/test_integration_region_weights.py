"""Per-sample region weights in the ordinary weighted-mean integration."""

from __future__ import annotations

import hashlib
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from ufwbpp.stacking import integration as calibration
from ufwbpp.stacking.integration import (
    CalibrationError,
    FrameExpression,
    IntegrationMapPaths,
    IntegrationParameters,
    evaluate_weight_grid_rows,
    integrate_expressions,
)

HEIGHT, WIDTH = 40, 64


def _frames(tmp_path: Path) -> list[Path]:
    rng = np.random.default_rng(3)
    paths = []
    for index in range(3):
        data = rng.normal(100.0 + 10.0 * index, 2.0, (HEIGHT, WIDTH)).astype(np.float32)
        path = tmp_path / f"frame{index}.fits"
        fits.PrimaryHDU(data=data).writeto(path)
        paths.append(path)
    return paths


def _weight_grid() -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    nodes = np.ones((4, 4))
    nodes[:, 0] = 0.0
    nodes[1:3, 1] = 0.5
    x_nodes = tuple((column + 0.5) / 4 * WIDTH for column in range(4))
    y_nodes = tuple((row + 0.5) / 4 * HEIGHT for row in range(4))
    return tuple(tuple(map(float, row)) for row in nodes), x_nodes, y_nodes


def _integrate(paths: list[Path], output: Path, *, with_grid: bool) -> tuple[calibration.IntegrationResult, dict[str, np.ndarray]]:
    grid, x_nodes, y_nodes = _weight_grid()
    expressions = [FrameExpression(str(path)) for path in paths]
    if with_grid:
        expressions[2] = FrameExpression(
            str(paths[2]), weight_grid=grid, weight_grid_x=x_nodes, weight_grid_y=y_nodes
        )
    maps = IntegrationMapPaths(
        accepted_count=output.with_name(output.stem + "_accepted.fits"),
        coverage=output.with_name(output.stem + "_coverage.fits"),
        rejection_count=output.with_name(output.stem + "_rejected.fits"),
    )
    result = integrate_expressions(
        expressions,
        output,
        parameters=IntegrationParameters(sigma_clip=50.0, transient_rejection=False),
        quality_weights=[1.0, 1.0, 1.0],
        map_paths=maps,
        durable=False,
    )
    read = {name: np.asarray(fits.getdata(path), dtype=np.float32) for name, path in result.map_paths.items()}
    return result, read


def test_region_weight_grid_scales_samples_and_coverage(tmp_path: Path) -> None:
    paths = _frames(tmp_path)
    result, maps = _integrate(paths, tmp_path / "weighted.fits", with_grid=True)
    integrated = np.asarray(fits.getdata(result.output_path), dtype=np.float32)
    assert result.execution["regionWeights"]["frames"] == 1
    assert result.execution["regionWeights"]["coverage"] == "effective-weight-fraction"
    stack = np.stack([np.asarray(fits.getdata(path), dtype=np.float32) for path in paths])
    weights = np.asarray(result.weights, dtype=np.float64)
    grid, x_nodes, y_nodes = _weight_grid()
    sample_weights = np.ones(stack.shape, dtype=np.float32)
    sample_weights[2] = evaluate_weight_grid_rows(grid, x_nodes, y_nodes, 0, HEIGHT, WIDTH)
    accepted = maps["acceptedSampleCount"] == 3
    assert accepted.all(), "a 50-sigma clip must accept every sample"
    effective = weights[:, None, None] * sample_weights
    expected = np.asarray(
        np.sum(stack * effective, axis=0, dtype=np.float64) / np.sum(effective, axis=0, dtype=np.float64),
        dtype=np.float32,
    )
    np.testing.assert_allclose(integrated, expected, rtol=2e-6)
    # The left quarter ignores frame 2 (mean 120, weight 0): its mean is near
    # 105; the right half uses all three frames and sits near 110.
    assert 104.0 < float(integrated[:, :8].mean()) < 106.5
    assert 108.5 < float(integrated[:, 40:].mean()) < 111.5
    np.testing.assert_allclose(
        maps["coverageFraction"], np.sum(sample_weights, axis=0) / 3.0, rtol=1e-6
    )
    np.testing.assert_array_equal(maps["rejectionCount"], 0.0)
    plain, plain_maps = _integrate(paths, tmp_path / "plain.fits", with_grid=False)
    assert "regionWeights" not in plain.execution
    np.testing.assert_array_equal(plain_maps["coverageFraction"], 1.0)
    assert plain.output_sha256 != result.output_sha256


def test_numpy_fallback_matches_native_region_weighting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _frames(tmp_path)
    native, _ = _integrate(paths, tmp_path / "native.fits", with_grid=True)
    monkeypatch.setattr(calibration, "load_native_kernels", lambda *args, **kwargs: None)
    fallback, _ = _integrate(paths, tmp_path / "fallback.fits", with_grid=True)
    assert fallback.execution["regionWeights"]["reducer"] == "numpy-reference"
    assert hashlib.sha256(Path(fallback.output_path).read_bytes()).hexdigest() == hashlib.sha256(
        Path(native.output_path).read_bytes()
    ).hexdigest()


def test_weight_grid_validation_and_receipt_keys(tmp_path: Path) -> None:
    grid, x_nodes, y_nodes = _weight_grid()
    path = _frames(tmp_path)[0]
    plain = FrameExpression(str(path))
    assert "weightGrid" not in plain.serializable()
    expression = FrameExpression(str(path), weight_grid=grid, weight_grid_x=x_nodes, weight_grid_y=y_nodes)
    record = expression.serializable()
    assert record["weightGrid"][0][0] == 0.0 and record["weightGridX"] == list(x_nodes)
    bad = tuple(tuple(1.5 if value == 0.0 else value for value in row) for row in grid)
    with pytest.raises(CalibrationError) as error:
        integrate_expressions(
            [FrameExpression(str(path), weight_grid=bad, weight_grid_x=x_nodes, weight_grid_y=y_nodes)],
            tmp_path / "bad.fits",
            parameters=IntegrationParameters(transient_rejection=False),
        )
    assert error.value.code == "FRAME_EXPRESSION_WEIGHT_GRID_INVALID"


def test_noise_estimate_ignores_the_blanked_region(tmp_path: Path) -> None:
    """A blocked, low-noise area must not inflate the frame's noise weight."""

    from ufwbpp.stacking.integration import _frame_noise_estimates, _open_expression_sources
    from contextlib import ExitStack

    rng = np.random.default_rng(5)
    data = rng.normal(100.0, 8.0, (HEIGHT, WIDTH)).astype(np.float32)
    data[:, : WIDTH // 2] = rng.normal(20.0, 0.5, (HEIGHT, WIDTH // 2)).astype(np.float32)
    path = tmp_path / "blocked.fits"
    fits.PrimaryHDU(data=data).writeto(path)
    nodes = np.ones((4, 4))
    nodes[:, :2] = 0.0
    x_nodes = tuple((column + 0.5) / 4 * WIDTH for column in range(4))
    y_nodes = tuple((row + 0.5) / 4 * HEIGHT for row in range(4))
    plain = FrameExpression(str(path))
    mapped = FrameExpression(
        str(path),
        weight_grid=tuple(tuple(map(float, row)) for row in nodes),
        weight_grid_x=x_nodes,
        weight_grid_y=y_nodes,
    )
    parameters = IntegrationParameters(transient_rejection=False)
    with ExitStack() as stack:
        sources = _open_expression_sources(stack, (plain, mapped))
        estimates = _frame_noise_estimates((plain, mapped), sources, (HEIGHT, WIDTH), parameters)
    whole, usable = estimates.sigma_pixel
    assert whole < 6.0, "the blocked half drags the whole-frame estimate down"
    assert 6.5 < usable < 9.5, usable
