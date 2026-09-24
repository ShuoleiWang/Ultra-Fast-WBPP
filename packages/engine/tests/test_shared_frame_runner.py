"""One spawned worker pool for quality-gate measurement, analysis and registration.

The E2E run shares a ``FrameRunner`` across the three stages so a run pays
one pool start-up instead of three.  Every frame runs the same function in
whichever pool executes it, so the shared pool must reproduce the results
of three separate pools exactly; this test forces the process executor and
compares.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from lightframeqc.analysis import analyze_measurements
from lightframeqc.config import DEFAULT_CONFIG
from lightframeqc.measure import measure_paths
from lightframeqc.parallel import FrameRunner
from ufwbpp_registration import run_registration

from test_e2e import _stars, _subpixel_shift


def _lights(root: Path, count: int) -> list[str]:
    rng = np.random.default_rng(5)
    base = _stars((128, 128))
    paths = []
    for index in range(count):
        image = 1000.0 + _subpixel_shift(base, 0.1 * index, 0.07 * index) + rng.normal(0.0, 1.0, base.shape)
        header = fits.Header()
        header["IMAGETYP"] = "Light"
        header["FILTER"] = "R"
        header["OBJECT"] = "FIELD"
        header["EXPTIME"] = 60.0
        header["INSTRUME"] = "CAM"
        header["GAIN"] = 100
        header["XBINNING"] = 1
        header["YBINNING"] = 1
        header["DATE-OBS"] = f"2026-01-01T20:{index:02d}:00Z"
        path = root / f"light_{index:02d}.fits"
        fits.writeto(path, image.astype(np.float32), header)
        paths.append(str(path))
    return paths


def _stages(paths: list[str], output: Path, runner: FrameRunner | None):
    config = replace(DEFAULT_CONFIG, preview_long_edge=256, make_thumbnails=False)
    measurement_stats: dict[str, object] = {}
    measurements = measure_paths(paths, output / "qc", config, workers=4, stats=measurement_stats, runner=runner)
    analysis_stats: dict[str, object] = {}
    groups, results = analyze_measurements(measurements, config, workers=4, stats=analysis_stats, runner=runner)
    run = run_registration(paths, workers=4, runner=runner)
    transforms = [
        (item.path, item.accepted, None if item.full_matrix is None else np.asarray(item.full_matrix).tolist())
        for item in run.transforms
    ]
    return (
        [(item.status, item.stars, item.background_grid, item.texture_grid) for item in measurements],
        [(item.decision.value, item.features) for item in results],
        (run.reference_index, transforms),
        measurement_stats,
        analysis_stats,
    )


def test_shared_process_pool_reproduces_separate_pools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Twelve frames is the process-pool threshold; force processes so the
    # comparison covers spawned workers serving all three stages.
    monkeypatch.setenv("LIGHTFRAMEQC_PARALLELISM", "processes")
    paths = _lights(tmp_path, 12)
    separate = _stages(paths, tmp_path / "separate", None)
    with FrameRunner(4, len(paths)) as shared:
        combined = _stages(paths, tmp_path / "shared", shared)
        assert shared.parallelism == "processes"
        assert shared._executor is not None, "the pool served every stage without being reopened"
    assert combined[0] == separate[0]
    assert combined[1] == separate[1]
    assert combined[2] == separate[2]
    assert combined[3] == separate[3] == {"parallelism": "processes", "workers": 4, "fallbackReason": None}
    assert combined[4]["parallelism"] == "processes"
    assert all(accepted for _path, accepted, _matrix in combined[2][1])
