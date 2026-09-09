from __future__ import annotations

import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine.calibration import CalibrationError
from openastroflow_engine.local_normalization import (
    LocalNormalizationParameters,
    normalize_registered_group,
)


def _synthetic_scene(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.indices(shape, dtype=np.float64)
    scene = (
        1000.0
        + 65.0 * np.sin(x / 19.0)
        + 55.0 * np.cos(y / 23.0)
        + 30.0 * np.sin((x + y) / 31.0)
    )
    for center_x, center_y, amplitude, sigma in (
        (90, 80, 800, 2.0),
        (270, 120, 1200, 2.8),
        (180, 330, 650, 3.2),
        (390, 390, 1000, 2.5),
    ):
        scene += amplitude * np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma**2)
        )
    # Broad, real target structure shared by every exposure.
    scene += 140.0 * np.exp(-((x - 280) ** 2 + (y - 250) ** 2) / (2 * 95.0**2))
    return scene.astype(np.float32)


def test_local_normalization_removes_smooth_gradient_without_erasing_scene(
    tmp_path: Path,
) -> None:
    shape = (512, 512)
    reference = _synthetic_scene(shape)
    y, x = np.indices(shape, dtype=np.float64)
    paths: list[Path] = []
    for index, strength in enumerate((0.0, 1.0, -0.7)):
        scale = 1.0 + strength * (0.018 * (x / shape[1]) - 0.012 * (y / shape[0]))
        offset = strength * (42.0 * x / shape[1] - 31.0 * y / shape[0] + 18.0)
        values = ((reference - offset) / scale).astype(np.float32)
        path = tmp_path / f"registered-{index}.fits"
        fits.writeto(path, values, overwrite=False)
        paths.append(path)

    parameters = LocalNormalizationParameters(
        enabled=True,
        tile_size_pixels=128,
        minimum_samples_per_tile=256,
        minimum_tile_correlation=0.1,
        minimum_median_correlation=0.35,
        minimum_residual_improvement=0.05,
    )
    result = normalize_registered_group(
        paths,
        tmp_path / "normalized",
        reference_index=0,
        parameters=parameters,
    )

    with fits.open(paths[1], memmap=True) as before_hdul, fits.open(
        result.normalized_paths[1], memmap=True
    ) as after_hdul:
        before_rmse = float(np.sqrt(np.mean((before_hdul[0].data - reference) ** 2)))
        after_rmse = float(np.sqrt(np.mean((after_hdul[0].data - reference) ** 2)))
        assert after_rmse < before_rmse * 0.35
        # Bright-scene contrast remains, rather than being flattened into a background map.
        assert float(np.max(after_hdul[0].data) - np.median(after_hdul[0].data)) > 500
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["status"] == "APPLIED"
    assert receipt["pixInsightEquivalent"] is False
    assert receipt["frames"][1]["evidence"]["validTiles"] >= 12
    assert receipt["frames"][1]["evidence"]["residualImprovement"] >= 0.05


def test_local_normalization_fails_closed_when_model_is_underconstrained(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"noise-{index}.fits"
        fits.writeto(path, np.random.default_rng(index).normal(size=(128, 128)).astype(np.float32))
        paths.append(path)
    with pytest.raises(CalibrationError) as raised:
        normalize_registered_group(
            paths,
            tmp_path / "failed",
            parameters=LocalNormalizationParameters(enabled=True, tile_size_pixels=64),
        )
    assert raised.value.code in {
        "LOCAL_NORMALIZATION_MODEL_INSUFFICIENT",
        "LOCAL_NORMALIZATION_CORRELATION_GATE",
    }
    assert not (tmp_path / "failed").exists()
