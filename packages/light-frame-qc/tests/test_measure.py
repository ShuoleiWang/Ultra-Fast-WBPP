from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from astropy.io import fits
import numpy as np
from PIL import Image
import pytest

from lightframeqc.config import DEFAULT_CONFIG
from lightframeqc.models import FrameRole
import lightframeqc.measure as measure_module
from lightframeqc.measure import (
    FrameMeasurementError,
    MeasurementSettings,
    _stars_from_sep,
    measure_frame,
    measure_frame_safe,
    measure_paths,
    measure_preview,
    measurement_star_catalog,
    save_thumbnail_png,
)
from lightframeqc.models import FrameMetadata
from lightframeqc.readers import ImagePreview, read_frame_preview


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_preview(image: np.ndarray) -> ImagePreview:
    height, width = image.shape
    return ImagePreview(
        path=Path("synthetic-support.fits"),
        data=image.astype(np.float32),
        metadata=FrameMetadata(path="synthetic-support.fits", width=width, height=height),
        source_width=width,
        source_height=height,
        source_channels=1,
        block_size=1,
        reader_backend="synthetic",
    )


def test_single_hot_pixel_is_retained_only_in_raw_catalog() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 32] = 100.0

    measured = measure_preview(_array_preview(image))

    assert measured.stars == []
    assert measured.detected_source_count == 0
    assert measured.raw_detected_source_count == 1
    assert measured.raw_stars is not None
    hot = measured.raw_stars[0]
    assert hot.support_pixels == 1
    assert hot.detection_pixels == 9
    assert hot.a == pytest.approx(np.sqrt(0.5))
    assert hot.b == pytest.approx(np.sqrt(0.5))


@pytest.mark.parametrize("sigma,peak", [(0.6, 40.0), (1.0, 12.0)])
def test_narrow_and_marginal_gaussians_retain_actual_pixel_support(
    sigma: float, peak: float
) -> None:
    y, x = np.mgrid[:64, :64]
    image = np.random.default_rng(311).normal(0.0, 1.0, (64, 64))
    image += peak * np.exp(-((x - 32.25) ** 2 + (y - 32.25) ** 2) / (2 * sigma**2))

    measured = measure_preview(_array_preview(image))

    assert measured.raw_stars is not None
    raw_near = [star for star in measured.raw_stars if np.hypot(star.x - 32.25, star.y - 32.25) < 2]
    supported_near = [star for star in measured.stars if np.hypot(star.x - 32.25, star.y - 32.25) < 2]
    assert len(raw_near) == len(supported_near) == 1
    assert supported_near[0].support_pixels >= 3


def test_source_support_is_applied_before_catalog_caps_with_one_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = [(name, "f8") for name in ("x", "y", "flux", "peak", "a", "b", "theta")]
    fields += [("flag", "i4"), ("npix", "i4"), ("tnpix", "i4")]
    objects = np.zeros(6, dtype=fields)
    objects["x"] = np.arange(6) * 8 + 5
    objects["y"] = 20
    objects["flux"] = [1000, 900, 500, 400, 300, 200]
    objects["peak"] = objects["flux"] / 4
    objects["a"] = objects["b"] = 1
    objects["npix"] = 9
    objects["tnpix"] = np.arange(1, 7)
    calls = 0

    def extract(image: np.ndarray, *args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        assert kwargs["minarea"] == 5
        return objects, np.zeros(image.shape, dtype=np.int32)

    monkeypatch.setattr(measure_module.sep, "extract", extract)
    measured = measure_preview(
        _array_preview(np.zeros((64, 64), dtype=np.float32)),
        settings=MeasurementSettings(maximum_stars=2),
    )

    assert calls == 1
    assert measured.detected_source_count == 4
    assert measured.raw_detected_source_count == 6
    assert [star.flux for star in measured.stars] == [500, 400]
    assert measured.raw_stars is not None
    assert [star.flux for star in measured.raw_stars] == [1000, 900]
    assert [star.support_pixels for star in measured.stars] == [3, 4]


def test_sep_rows_without_support_columns_keep_unknown_provenance() -> None:
    fields = [(name, "f8") for name in ("x", "y", "flux", "peak", "a", "b", "theta")]
    fields += [("flag", "i4")]
    objects = np.zeros(1, dtype=fields)
    objects["flux"] = objects["peak"] = 100
    objects["a"] = objects["b"] = 1

    stars, count = _stars_from_sep(objects, 10)

    assert count == 1
    assert stars[0].support_pixels is None
    assert stars[0].detection_pixels is None


@pytest.mark.parametrize("value", [True, 0, -1, 2.5, float("nan")])
def test_measurement_support_threshold_requires_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="minimum_source_support_pixels"):
        MeasurementSettings(minimum_source_support_pixels=value).validate()


def test_measurement_settings_honor_configured_support_threshold() -> None:
    configured = replace(DEFAULT_CONFIG, minimum_source_support_pixels=4)
    assert MeasurementSettings.from_config(configured).minimum_source_support_pixels == 4


def _synthetic_star_field(
    height: int = 256,
    width: int = 256,
    *,
    seed: int = 7,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    image = (
        1_000.0
        + 0.08 * xx
        + 0.04 * yy
        + rng.normal(0.0, 2.0, size=(height, width))
    )
    centers = [
        (28.5, 31.0),
        (75.0, 48.5),
        (129.0, 63.0),
        (210.0, 36.0),
        (48.0, 112.0),
        (101.5, 133.0),
        (168.0, 118.0),
        (224.0, 151.0),
        (33.0, 207.0),
        (88.0, 229.0),
        (151.0, 196.0),
        (205.0, 218.0),
    ]
    for index, (x, y) in enumerate(centers):
        sigma = 1.6 + 0.15 * (index % 3)
        amplitude = 350.0 + 45.0 * index
        image += amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
        )
    return image.astype(np.float32), centers


def _write_star_field(path: Path, *, seed: int = 7) -> list[tuple[float, float]]:
    image, centers = _synthetic_star_field(seed=seed)
    hdu = fits.PrimaryHDU(image)
    hdu.header["FILTER"] = "L"
    hdu.header["EXPTIME"] = 60.0
    hdu.writeto(path)
    return centers


def _assert_grid_shape(grid: list[list[float | None]], rows: int, columns: int) -> None:
    assert len(grid) == rows
    assert all(len(row) == columns for row in grid)


def test_measures_sep_stars_background_texture_and_preview_coordinates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stars.fits"
    expected_centers = _write_star_field(source)
    before_hash = _sha256(source)
    before_stat = source.stat()
    config = replace(
        DEFAULT_CONFIG,
        preview_long_edge=256,
        detection_sigma=4.0,
        make_thumbnails=False,
    )

    measurement = measure_frame(source, config=config)

    assert measurement.status == "MEASURED"
    assert measurement.error_code is None
    assert measurement.preview_width == 256
    assert measurement.preview_height == 256
    assert measurement.metadata.filter_name == "L"
    assert measurement.metadata.exposure_seconds == pytest.approx(60.0)
    assert measurement.image_median == pytest.approx(1_015.3, abs=3.0)
    assert measurement.image_mad is not None and measurement.image_mad > 0
    assert len(measurement.stars) >= len(expected_centers) - 1
    assert measurement.detected_source_count >= len(measurement.stars)
    _assert_grid_shape(measurement.background_grid, 16, 16)
    _assert_grid_shape(measurement.texture_grid, 16, 16)
    assert all(
        value is None or np.isfinite(value)
        for grid in (measurement.background_grid, measurement.texture_grid)
        for row in grid
        for value in row
    )

    detected = np.asarray([(star.x, star.y) for star in measurement.stars])
    for expected in expected_centers:
        distance = np.linalg.norm(detected - expected, axis=1)
        assert float(distance.min()) < 1.0
    assert all(0 <= star.x < 256 and 0 <= star.y < 256 for star in measurement.stars)
    assert _sha256(source) == before_hash
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_star_catalog_contract_is_xy_flux_sorted_by_brightness(tmp_path: Path) -> None:
    source = tmp_path / "catalog.fit"
    _write_star_field(source)
    config = replace(DEFAULT_CONFIG, preview_long_edge=256, make_thumbnails=False)

    measurement = measure_frame(source, config=config)
    catalog = measurement_star_catalog(measurement)

    assert set(catalog) == {"x", "y", "flux"}
    assert catalog["x"].shape == catalog["y"].shape == catalog["flux"].shape
    assert catalog["x"].dtype == np.float64
    assert catalog["flux"].size == len(measurement.stars)
    assert np.all(np.diff(catalog["flux"]) <= 0)


def test_optional_thumbnail_is_png_at_preview_size_and_never_overwrites(
    tmp_path: Path,
) -> None:
    source = tmp_path / "thumbnail-source.fits"
    _write_star_field(source)
    before_hash = _sha256(source)
    before_stat = source.stat()
    thumbnail = tmp_path / "report" / "preview.png"
    config = replace(DEFAULT_CONFIG, preview_long_edge=256, make_thumbnails=True)

    measurement = measure_frame(source, config=config, thumbnail_path=thumbnail)

    assert measurement.thumbnail_path == str(thumbnail.resolve())
    with Image.open(thumbnail) as image:
        assert image.format == "PNG"
        assert image.mode == "L"
        assert image.size == (measurement.preview_width, measurement.preview_height)
    assert _sha256(source) == before_hash
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns

    preview = read_frame_preview(source, max_long_edge=256)
    with pytest.raises(FrameMeasurementError) as raised:
        save_thumbnail_png(preview, thumbnail)
    assert raised.value.code == "THUMBNAIL_EXISTS"
    assert _sha256(source) == before_hash


def test_blank_frame_has_no_stars_but_remains_measurable(tmp_path: Path) -> None:
    source = tmp_path / "blank.fits"
    fits.writeto(source, np.full((256, 256), 1_000.0, dtype=np.float32))
    config = replace(DEFAULT_CONFIG, preview_long_edge=256, make_thumbnails=False)

    measurement = measure_frame(source, config=config)

    assert measurement.status == "MEASURED"
    assert measurement.stars == []
    _assert_grid_shape(measurement.background_grid, 16, 16)
    _assert_grid_shape(measurement.texture_grid, 16, 16)


def test_nonfinite_frame_reports_a_specific_error_without_aborting_batch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonfinite.fits"
    fits.writeto(source, np.full((256, 256), np.nan, dtype=np.float32))
    config = replace(DEFAULT_CONFIG, preview_long_edge=256, make_thumbnails=False)

    with pytest.raises(FrameMeasurementError) as raised:
        measure_frame(source, config=config)
    assert raised.value.code == "NO_FINITE_PIXELS"

    safe = measure_frame_safe(source, config=config)
    assert safe.status == "ERROR"
    assert safe.error_code == "NO_FINITE_PIXELS"
    assert safe.metadata.path == str(source.resolve())


def test_failed_light_measurement_preserves_role_metadata_and_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny-light.fits"
    hdu = fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32))
    hdu.header["IMAGETYP"] = "LIGHT"
    hdu.header["FILTER"] = "R"
    hdu.header["OBJECT"] = "NGC 7000"
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    hdu.writeto(source)

    result = measure_frame_safe(
        source,
        config=replace(DEFAULT_CONFIG, make_thumbnails=False),
    )

    assert result.status == "ERROR"
    assert result.error_code == "PREVIEW_TOO_SMALL"
    assert result.metadata.role is FrameRole.LIGHT
    assert result.metadata.filter_name == "R"
    assert result.metadata.target == "NGC 7000"
    assert result.identity is not None
    assert result.identity.sha256 == _sha256(source)


def test_measure_paths_preserves_order_supports_workers_and_audits_failures(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.fits"
    second = tmp_path / "second.fits"
    unsupported = tmp_path / "not-a-frame.txt"
    _write_star_field(first, seed=1)
    _write_star_field(second, seed=2)
    unsupported.write_text("bad input", encoding="utf-8")
    config = replace(DEFAULT_CONFIG, preview_long_edge=256, make_thumbnails=False)

    results = measure_paths(
        [second, unsupported, first],
        tmp_path / "output",
        config,
        workers=2,
    )

    assert [Path(result.metadata.path) for result in results] == [
        second.resolve(),
        unsupported.resolve(),
        first.resolve(),
    ]
    assert [result.status for result in results] == ["MEASURED", "ERROR", "MEASURED"]
    assert results[1].error_code == "UNSUPPORTED_FORMAT"
    assert not (tmp_path / "output").exists()


def test_measure_paths_writes_unique_thumbnails_under_output_directory(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "same-name.fits"
    second = second_dir / "same-name.fits"
    _write_star_field(first, seed=3)
    _write_star_field(second, seed=4)
    config = replace(DEFAULT_CONFIG, preview_long_edge=256, make_thumbnails=True)
    output = tmp_path / "qc-output"

    results = measure_paths([first, second], output, config, workers=1)

    paths = [Path(result.thumbnail_path or "") for result in results]
    assert all(result.status == "MEASURED" for result in results)
    assert len(set(paths)) == 2
    assert all(path.is_file() for path in paths)
    assert all(path.parent == output / "thumbnails" for path in paths)


def test_sep_calls_are_serialized_off_macos() -> None:
    """The Windows sep wheel races under concurrent extraction; see measure.py."""

    import sys
    from contextlib import nullcontext
    import threading

    from lightframeqc import measure as measure_module

    guard = measure_module._sep_guard()
    if sys.platform == "darwin":
        assert measure_module.SEP_CALLS_SERIALIZED is False
        assert isinstance(guard, nullcontext)
    else:
        assert measure_module.SEP_CALLS_SERIALIZED is True
        assert isinstance(guard, type(threading.Lock()))
        assert guard is measure_module._sep_guard()
