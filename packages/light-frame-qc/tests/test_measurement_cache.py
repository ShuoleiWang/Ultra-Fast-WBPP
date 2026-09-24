from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from lightframeqc.analysis import analyze_measurements
from lightframeqc.config import DEFAULT_CONFIG, QcConfig
from lightframeqc.measure import measure_frame, measure_paths
from lightframeqc.measurement_cache import FrameMeasurementCache
from lightframeqc.quality_gate import GatePolicy, evaluate_quality_gate

CONFIG = replace(DEFAULT_CONFIG, preview_long_edge=256, make_thumbnails=False)


def _write_star_field(path: Path, *, seed: int = 7, filter_name: str = "L") -> None:
    generator = np.random.default_rng(seed)
    image = generator.normal(1000.0, 8.0, size=(256, 256)).astype(np.float32)
    rows, columns = np.mgrid[0:256, 0:256]
    for index in range(24):
        y = float(generator.uniform(16, 240))
        x = float(generator.uniform(16, 240))
        amplitude = float(generator.uniform(400.0, 3000.0))
        image += amplitude * np.exp(-(((rows - y) ** 2 + (columns - x) ** 2) / (2.0 * 2.1**2)))
    hdu = fits.PrimaryHDU(image)
    hdu.header["FILTER"] = filter_name
    hdu.header["EXPTIME"] = 60.0
    hdu.header["OBJECT"] = "NGC 6822"
    hdu.header["INSTRUME"] = "SYNTHETIC"
    hdu.header["DATE-OBS"] = f"2026-09-2{seed % 10}T22:0{seed % 6}:11.500"
    hdu.header["IMAGETYP"] = "LIGHT"
    hdu.writeto(path)


def test_restored_measurement_equals_the_measured_one(tmp_path: Path) -> None:
    source = tmp_path / "frame.fits"
    _write_star_field(source)
    measured = measure_frame(source, config=CONFIG)
    cache = FrameMeasurementCache(tmp_path / "cache", CONFIG)
    assert measured.identity is not None

    key = cache.key(source, measured.identity)
    assert key is not None
    assert cache.store(key, measured) is True
    restored = cache.load(key)

    assert restored is not None
    # Every field, including the star catalogs, the grids and the
    # provenance the gate reads.
    assert restored == replace(measured, thumbnail_path=None)
    assert restored.stars == measured.stars
    assert restored.raw_stars == measured.raw_stars
    assert restored.native_psf == measured.native_psf
    assert restored.metadata == measured.metadata
    assert restored.background_grid == measured.background_grid
    assert restored.texture_grid == measured.texture_grid


def test_key_changes_with_configuration_content_and_implementation(tmp_path: Path) -> None:
    source = tmp_path / "frame.fits"
    _write_star_field(source)
    measured = measure_frame(source, config=CONFIG)
    assert measured.identity is not None
    cache = FrameMeasurementCache(tmp_path / "cache", CONFIG)
    key = cache.key(source, measured.identity)

    other_config = FrameMeasurementCache(tmp_path / "cache", replace(CONFIG, detection_sigma=5.0))
    other_index = FrameMeasurementCache(tmp_path / "cache", CONFIG, image_index=1)
    other_decode = FrameMeasurementCache(tmp_path / "cache", CONFIG, max_full_decode_bytes=1024)
    changed_bytes = replace(measured.identity, sha256="0" * 64)

    assert key is not None
    assert other_config.key(source, measured.identity) != key
    assert other_index.key(source, measured.identity) != key
    assert other_decode.key(source, measured.identity) != key
    assert cache.key(source, changed_bytes) != key
    # A thumbnail is a side effect of measuring, never a measured value.
    assert cache.key(source, measured.identity) == FrameMeasurementCache(
        tmp_path / "cache", replace(CONFIG, make_thumbnails=True)
    ).key(source, measured.identity)


def test_measure_paths_reuses_cached_measurements_and_counts_them(tmp_path: Path) -> None:
    frames = []
    for index in range(3):
        frame = tmp_path / f"frame-{index}.fits"
        _write_star_field(frame, seed=index + 1)
        frames.append(frame)
    cache_directory = tmp_path / "cache"

    cold_stats: dict[str, int] = {}
    cold = measure_paths(
        frames, tmp_path / "cold", CONFIG, workers=1,
        cache_directory=cache_directory, cache_stats=cold_stats,
    )
    warm_stats: dict[str, int] = {}
    warm = measure_paths(
        frames, tmp_path / "warm", CONFIG, workers=1,
        cache_directory=cache_directory, cache_stats=warm_stats,
    )

    assert cold_stats == {"misses": 3, "writes": 3}
    assert warm_stats == {"hits": 3}
    assert warm == cold
    assert all(measurement.status == "MEASURED" for measurement in warm)


def test_cached_measurements_produce_identical_analysis_and_gate(tmp_path: Path) -> None:
    frames = []
    for index in range(5):
        frame = tmp_path / f"frame-{index}.fits"
        _write_star_field(frame, seed=index + 1)
        frames.append(frame)
    cache_directory = tmp_path / "cache"
    config = replace(CONFIG, minimum_group_frames=2)

    fresh = measure_paths(frames, tmp_path / "cold", config, workers=1, cache_directory=cache_directory)
    reused = measure_paths(frames, tmp_path / "warm", config, workers=1, cache_directory=cache_directory)
    fresh_groups, fresh_results = analyze_measurements(fresh, config)
    reused_groups, reused_results = analyze_measurements(reused, config)
    evaluate_quality_gate(fresh_results, fresh, GatePolicy())
    evaluate_quality_gate(reused_results, reused, GatePolicy())

    assert reused_groups == fresh_groups
    assert [item.serializable() for item in reused_results] == [
        item.serializable() for item in fresh_results
    ]


def test_cache_hit_still_writes_the_requested_side_effects(tmp_path: Path) -> None:
    frames = []
    for index in range(2):
        frame = tmp_path / f"frame-{index}.fits"
        _write_star_field(frame, seed=index + 1)
        frames.append(frame)
    cache_directory = tmp_path / "cache"
    config = replace(CONFIG, make_thumbnails=True)

    cold = measure_paths(
        frames, tmp_path / "cold", config, workers=1,
        cache_directory=cache_directory, linear_directory=tmp_path / "cold-linear",
    )
    warm_stats: dict[str, int] = {}
    warm = measure_paths(
        frames, tmp_path / "warm", config, workers=1,
        cache_directory=cache_directory, cache_stats=warm_stats,
        linear_directory=tmp_path / "warm-linear",
    )

    assert warm_stats == {"hits": 2}
    assert sorted(path.name for path in (tmp_path / "warm-linear").iterdir()) == ["0000.npy", "0001.npy"]
    for index in range(2):
        assert np.array_equal(
            np.load(tmp_path / "cold-linear" / f"{index:04d}.npy"),
            np.load(tmp_path / "warm-linear" / f"{index:04d}.npy"),
        )
    assert all(measurement.thumbnail_path is not None for measurement in warm)
    assert [Path(item.thumbnail_path).exists() for item in warm] == [True, True]
    assert [replace(item, thumbnail_path=None) for item in warm] == [
        replace(item, thumbnail_path=None) for item in cold
    ]


def test_a_rewritten_file_is_never_answered_from_the_cache(tmp_path: Path) -> None:
    source = tmp_path / "frame.fits"
    _write_star_field(source, seed=1)
    cache_directory = tmp_path / "cache"
    first = measure_paths([source], tmp_path / "first", CONFIG, workers=1, cache_directory=cache_directory)

    source.unlink()
    _write_star_field(source, seed=9)
    stats: dict[str, int] = {}
    second = measure_paths(
        [source], tmp_path / "second", CONFIG, workers=1,
        cache_directory=cache_directory, cache_stats=stats,
    )

    assert stats == {"misses": 1, "writes": 1}
    assert second[0].stars != first[0].stars


def test_unavailable_cache_directory_is_a_miss_not_a_failure(tmp_path: Path) -> None:
    source = tmp_path / "frame.fits"
    _write_star_field(source)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    stats: dict[str, int] = {}
    results = measure_paths(
        [source], tmp_path / "out", CONFIG, workers=1,
        cache_directory=blocked, cache_stats=stats,
    )

    assert results[0].status == "MEASURED"
    assert stats == {"misses": 1}


def test_failed_measurements_are_not_cached(tmp_path: Path) -> None:
    broken = tmp_path / "broken.txt"
    broken.write_text("not a frame", encoding="utf-8")
    cache_directory = tmp_path / "cache"

    stats: dict[str, int] = {}
    results = measure_paths(
        [broken], tmp_path / "out", CONFIG, workers=1,
        cache_directory=cache_directory, cache_stats=stats,
    )

    assert results[0].status == "ERROR"
    assert stats == {"misses": 1}
    assert not list((cache_directory / "frame-measurement-v1").glob("*.json"))


@pytest.mark.parametrize("config", [CONFIG, QcConfig(preview_long_edge=256)])
def test_a_corrupted_entry_is_a_miss(tmp_path: Path, config: QcConfig) -> None:
    source = tmp_path / "frame.fits"
    _write_star_field(source)
    measured = measure_frame(source, config=config)
    assert measured.identity is not None
    cache = FrameMeasurementCache(tmp_path / "cache", config)
    key = cache.key(source, measured.identity)
    assert cache.store(key, measured) is True

    entry = tmp_path / "cache" / "frame-measurement-v1" / f"{key}.json"
    payload = entry.read_text(encoding="utf-8").replace('"MEASURED"', '"IMPORTED"')
    entry.write_text(payload, encoding="utf-8")

    assert cache.load(key) is None
