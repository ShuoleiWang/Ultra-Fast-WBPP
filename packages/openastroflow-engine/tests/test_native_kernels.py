"""Differential tests: native kernels versus the NumPy reference arithmetic.

Every kernel must reproduce the NumPy implementation value for value (the
sign of an exact zero is the only tolerated difference).  The tests skip when
the native library is not built, because the NumPy path is the portable
fallback and the differential contract is what a native build must satisfy.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import warnings

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine import native_kernels
from openastroflow_engine.calibration import (
    REJECTION_FLOOR_ABSOLUTE,
    REJECTION_FLOOR_EPSILON_FACTOR,
    FitsFrame,
    FrameExpression,
    IntegrationMapPaths,
    IntegrationParameters,
    _MemoryFrame,
    _RejectionSigmaFloor,
    _ordinary_mad_rejection_decision,
    integrate_expressions,
    read_frame_info,
)
from openastroflow_engine.global_normalization import GlobalNormalizationParameters
from openastroflow_engine import pixel_pipeline as pipeline
from openastroflow_engine.pixel_pipeline import AffineTransform, PipelineParameters


KERNELS = native_kernels.load_native_kernels()
requires_native = pytest.mark.skipif(
    KERNELS is None, reason="native kernel library is not built in this checkout"
)


def _numpy_mad_decision(
    samples: np.ndarray, parameters: IntegrationParameters, group_floor: float
):
    """Run the NumPy reference decision with the native kernels disabled."""

    floor = _RejectionSigmaFloor(
        True, group_floor, None, 200_000, 65_536, 1, 1, 1, 1,
        parameters.minimum_rejection_frames, "sha256:test",
    )
    original = native_kernels.load_native_kernels
    native_kernels_module_cache = dict(native_kernels._CACHE)
    try:
        os.environ[native_kernels.DISABLE_ENVIRONMENT_VARIABLE] = "1"
        finite, center, accepted = _ordinary_mad_rejection_decision(
            samples, parameters, floor
        )
    finally:
        os.environ.pop(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, None)
        native_kernels._CACHE.clear()
        native_kernels._CACHE.update(native_kernels_module_cache)
    assert native_kernels.load_native_kernels is original
    return floor, finite, center, accepted


def _random_stack(seed: int, frames: int, rows: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stack = rng.normal(1000.0, 30.0, (frames, rows, width)).astype(np.float32)
    # Sparse NaN/inf coverage, exact ties, an isolated outlier and a pixel
    # with fewer finite samples than the rejection minimum.
    stack[rng.random(stack.shape) < 0.02] = np.nan
    stack[rng.random(stack.shape) < 0.005] = np.inf
    stack[:, 1, 2] = np.float32(1000.0)
    stack[min(3, frames - 1), 1, 2] = np.float32(1000.5)
    stack[min(5, frames - 1), 2, 3] += np.float32(5000.0)
    stack[:, 0, 0] = np.nan
    stack[:2, 0, 0] = [np.float32(999.0), np.float32(1001.0)]
    stack[0, 0, 1] = np.float32(-0.0)
    return stack


@requires_native
@pytest.mark.parametrize("frames", [3, 4, 9, 38])
def test_native_mad_rejection_matches_numpy_decisions_bitwise(frames: int) -> None:
    assert KERNELS is not None
    samples = _random_stack(11 + frames, frames, 7, 13)
    parameters = IntegrationParameters(sigma_clip=4.0, minimum_rejection_frames=3)
    group_floor = 0.75
    floor, finite, center, accepted = _numpy_mad_decision(samples, parameters, group_floor)
    native_accepted, native_center = KERNELS.mad_rejection(
        samples,
        sigma_clip=parameters.sigma_clip,
        minimum_rejection_frames=parameters.minimum_rejection_frames,
        group_sigma_floor=group_floor,
        absolute_floor=REJECTION_FLOOR_ABSOLUTE,
        epsilon_floor=float(np.float32(REJECTION_FLOOR_EPSILON_FACTOR * np.finfo(np.float32).eps)),
        threads=3,
    )
    np.testing.assert_array_equal(native_accepted, accepted)
    assert np.array_equal(native_center, center, equal_nan=True)
    assert native_accepted.dtype == np.bool_
    if frames >= 5:
        assert not native_accepted[5, 2, 3]
    # The production entry point selects the kernel and must agree too.
    production_finite, production_center, production_accepted = (
        _ordinary_mad_rejection_decision(samples, parameters, floor)
    )
    np.testing.assert_array_equal(production_accepted, accepted)
    np.testing.assert_array_equal(production_finite, finite)
    assert np.array_equal(production_center, center, equal_nan=True)


def _numpy_scaled_decision(
    samples: np.ndarray,
    parameters: IntegrationParameters,
    group_floor: float,
    frame_scales: tuple[float, ...],
    pool_half_width: int,
):
    floor = _RejectionSigmaFloor(
        True, group_floor, None, 200_000, 65_536, 1, 1, 1, 1,
        parameters.minimum_rejection_frames, "sha256:test",
        frame_scales, pool_half_width,
    )
    native_kernels_module_cache = dict(native_kernels._CACHE)
    try:
        os.environ[native_kernels.DISABLE_ENVIRONMENT_VARIABLE] = "1"
        native_kernels._CACHE.clear()
        finite, center, accepted = _ordinary_mad_rejection_decision(
            samples, parameters, floor
        )
    finally:
        os.environ.pop(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, None)
        native_kernels._CACHE.clear()
        native_kernels._CACHE.update(native_kernels_module_cache)
    return floor, finite, center, accepted


@requires_native
@pytest.mark.parametrize(
    ("frames", "rows", "width", "half_width"),
    [(3, 5, 9, 12), (4, 6, 40, 12), (9, 7, 61, 3), (13, 4, 200, 12), (38, 4, 30, 1)],
)
def test_native_scaled_mad_rejection_matches_numpy_decisions_bitwise(
    frames: int, rows: int, width: int, half_width: int
) -> None:
    """The v2 scale model (row-pooled MAD, per-frame factors) is value-identical."""

    assert KERNELS is not None
    rng = np.random.default_rng(100 + frames)
    samples = _random_stack(41 + frames, frames, rows, width)
    # Frames of two noise levels, a narrow stack next to normal ones, a
    # saturated core, a coverage edge and a pixel column with no MAD.
    samples[: frames // 2] += rng.normal(0.0, 45.0, (frames // 2, rows, width)).astype(np.float32)
    samples[:, 2, 4] = np.float32(1000.0) + np.float32(0.01) * np.arange(frames, dtype=np.float32)
    samples[min(1, frames - 1), 2, 4] = np.float32(1900.0)
    samples[:, 1, 5] = np.float32(60000.0)
    samples[:, 3, width - 1] = np.nan
    samples[: min(2, frames), 3, width - 1] = np.float32(1000.0)
    parameters = IntegrationParameters(sigma_clip=4.0, minimum_rejection_frames=3)
    scales = tuple(
        float(np.float32(value))
        for value in np.concatenate(
            (np.full(frames // 2, 1.6), np.full(frames - frames // 2, 0.9))
        )
    )
    group_floor = 0.5
    floor, finite, center, accepted = _numpy_scaled_decision(
        samples, parameters, group_floor, scales, half_width
    )
    assert floor.pooled
    native_accepted, native_center = KERNELS.mad_rejection(
        samples,
        sigma_clip=parameters.sigma_clip,
        minimum_rejection_frames=parameters.minimum_rejection_frames,
        group_sigma_floor=group_floor,
        absolute_floor=REJECTION_FLOOR_ABSOLUTE,
        epsilon_floor=float(np.float32(REJECTION_FLOOR_EPSILON_FACTOR * np.finfo(np.float32).eps)),
        frame_scales=scales,
        pool_half_width=half_width,
        threads=3,
    )
    np.testing.assert_array_equal(native_accepted, accepted)
    assert np.array_equal(native_center, center, equal_nan=True)
    production_finite, production_center, production_accepted = (
        _ordinary_mad_rejection_decision(samples, parameters, floor, native_threads=2)
    )
    np.testing.assert_array_equal(production_accepted, accepted)
    np.testing.assert_array_equal(production_finite, finite)
    assert np.array_equal(production_center, center, equal_nan=True)
    # The narrow stack's 900-unit excursion is a real outlier at every scale
    # (the row noise is ~40) while the coverage-edge pixel keeps every finite
    # sample.
    if frames >= 4:
        assert not accepted[1, 2, 4]
    np.testing.assert_array_equal(accepted[:, 3, width - 1], finite[:, 3, width - 1])
    # Unit scales without pooling reproduce the v1 decisions bit for bit.
    _, _, _, legacy = _numpy_mad_decision(samples, parameters, group_floor)
    _, _, _, unit = _numpy_scaled_decision(
        samples, parameters, group_floor, (1.0,) * frames, 0
    )
    np.testing.assert_array_equal(unit, legacy)
    native_unit, _ = KERNELS.mad_rejection(
        samples,
        sigma_clip=parameters.sigma_clip,
        minimum_rejection_frames=parameters.minimum_rejection_frames,
        group_sigma_floor=group_floor,
        absolute_floor=REJECTION_FLOOR_ABSOLUTE,
        epsilon_floor=float(np.float32(REJECTION_FLOOR_EPSILON_FACTOR * np.finfo(np.float32).eps)),
        frame_scales=(1.0,) * frames,
        pool_half_width=0,
        threads=2,
    )
    np.testing.assert_array_equal(native_unit, legacy)


@requires_native
def test_native_scaled_mad_rejection_validates_scales() -> None:
    assert KERNELS is not None
    samples = _random_stack(3, 5, 4, 6)
    common = dict(
        sigma_clip=4.0, minimum_rejection_frames=3, group_sigma_floor=0.1,
        absolute_floor=REJECTION_FLOOR_ABSOLUTE, epsilon_floor=1e-6,
    )
    with pytest.raises(ValueError):
        KERNELS.mad_rejection(samples, frame_scales=(1.0, 1.0), **common)
    with pytest.raises(ValueError):
        KERNELS.mad_rejection(samples, frame_scales=(1.0, 0.0, 1.0, 1.0, 1.0), **common)
    with pytest.raises(ValueError):
        KERNELS.mad_rejection(samples, pool_half_width=-1, **common)


@requires_native
def test_native_masked_mean_matches_numpy_float64_accumulation_bitwise() -> None:
    assert KERNELS is not None
    rng = np.random.default_rng(5)
    samples = _random_stack(21, 12, 6, 9)
    accepted = np.isfinite(samples) & (rng.random(samples.shape) > 0.15)
    weights = rng.random(12) / 12.0
    integrated, accepted_count, rejected_count = KERNELS.masked_weighted_mean(
        samples, accepted, weights, threads=4
    )
    numerator = np.sum(
        np.where(accepted, samples, 0.0) * weights[:, None, None], axis=0, dtype=np.float64
    )
    denominator = np.sum(accepted * weights[:, None, None], axis=0, dtype=np.float64)
    expected = np.full(numerator.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=expected, where=denominator > 0, casting="unsafe")
    assert np.array_equal(integrated, expected, equal_nan=True)
    np.testing.assert_array_equal(accepted_count, np.sum(accepted, axis=0))
    np.testing.assert_array_equal(
        rejected_count, np.sum(np.isfinite(samples) & ~accepted, axis=0)
    )
    single_thread = KERNELS.masked_weighted_mean(samples, accepted, weights, threads=1)[0]
    assert np.array_equal(single_thread, integrated, equal_nan=True)


@requires_native
def test_integrate_expressions_native_and_numpy_paths_publish_identical_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _random_stack(31, 9, 40, 24)
    paths = []
    for index, values in enumerate(stack):
        path = tmp_path / f"frame-{index:02d}.fits"
        fits.writeto(path, values, overwrite=False)
        paths.append(path)
    parameters = IntegrationParameters(
        max_memory_bytes=1024 * 1024, max_statistics_samples=100, minimum_rejection_frames=3
    )
    outputs = {}
    for label in ("native", "numpy"):
        if label == "numpy":
            monkeypatch.setenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, "1")
        native_kernels.reset_native_kernel_cache()
        maps = IntegrationMapPaths(
            tmp_path / f"{label}-accepted.fits",
            tmp_path / f"{label}-coverage.fits",
            tmp_path / f"{label}-rejected.fits",
        )
        result = integrate_expressions(
            [FrameExpression(str(path)) for path in paths],
            tmp_path / f"{label}-master.fits",
            parameters=parameters,
            map_paths=maps,
            native_threads=3,
        )
        outputs[label] = (result, maps)
        monkeypatch.delenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, raising=False)
        native_kernels.reset_native_kernel_cache()
    native_result, native_maps = outputs["native"]
    numpy_result, numpy_maps = outputs["numpy"]
    assert native_result.execution["rejectionMask"]["kernel"] == native_kernels.MAD_KERNEL_ID
    assert native_result.execution["reducer"] == native_kernels.MEAN_KERNEL_ID
    assert numpy_result.execution["rejectionMask"]["kernel"] == "numpy-nanmedian-pooled-mad-v2"
    assert numpy_result.execution["reducer"] == "numpy-float64-weighted-mean-v1"
    assert native_result.accepted_samples == numpy_result.accepted_samples
    assert native_result.rejected_samples == numpy_result.rejected_samples
    assert fits.getdata(native_result.output_path).tobytes() == fits.getdata(
        numpy_result.output_path
    ).tobytes()
    for native_map, numpy_map in zip(
        (native_maps.accepted_count, native_maps.coverage, native_maps.rejection_count),
        (numpy_maps.accepted_count, numpy_maps.coverage, numpy_maps.rejection_count),
        strict=True,
    ):
        np.testing.assert_array_equal(fits.getdata(native_map), fits.getdata(numpy_map))


def _declared_frame(path: Path, pixels: np.ndarray, unit_scale: float) -> Path:
    header = fits.Header()
    header["IMAGETYP"] = "Calibrated Light"
    header["OAFNDOM"] = "NORMALIZED_TEST"
    header["OAFNSCL"] = unit_scale
    fits.writeto(path, np.asarray(pixels, dtype=np.float32), header, overwrite=False)
    return path


def _warp_transforms(shape: tuple[int, int]) -> dict[str, AffineTransform]:
    height, width = shape
    angle = np.deg2rad(0.44)
    return {
        "fractional-translation": AffineTransform.from_value(
            ((1.0, 0.0, 3.37), (0.0, 1.0, -2.61), (0.0, 0.0, 1.0))
        ),
        "small-rotation": AffineTransform.from_value(
            (
                (np.cos(angle), -np.sin(angle), 1.25),
                (np.sin(angle), np.cos(angle), 0.75),
                (0.0, 0.0, 1.0),
            )
        ),
        "near-half-turn": AffineTransform.from_value(
            (
                (np.cos(np.pi - angle), -np.sin(np.pi - angle), width - 1.0),
                (np.sin(np.pi - angle), np.cos(np.pi - angle), height - 1.0),
                (0.0, 0.0, 1.0),
            )
        ),
        "shear-scale": AffineTransform.from_value(
            ((1.002, 0.0015, -0.4), (-0.0011, 0.998, 0.9), (0.0, 0.0, 1.0))
        ),
        "integer-translation": AffineTransform.from_value(
            ((1.0, 0.0, 4.0), (0.0, 1.0, -3.0), (0.0, 0.0, 1.0))
        ),
        # Perspective terms of the size a tilt or differential refraction
        # leaves between nights: a few tenths of a pixel across the field.
        "projective": AffineTransform.from_value(
            (
                (1.0004, -0.0005, 1.7),
                (0.0006, 1.0003, -2.2),
                (3.0e-6, -2.0e-6, 1.0),
            )
        ),
    }


@requires_native
@pytest.mark.parametrize("case", list(_warp_transforms((1, 1))))
def test_native_lanczos_warp_matches_numpy_resampler_bitwise(
    tmp_path: Path, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = (37, 53)
    rng = np.random.default_rng(3)
    pixels = rng.normal(0.4, 0.05, shape).astype(np.float32)
    pixels[:, 20] = 2.5  # above the declared domain scale
    pixels[7, 9] = -0.3  # below zero
    pixels[15, 30] = np.nan
    pixels[28, 41] = np.inf
    pixels[3, 3] = np.float32(-0.0)
    source = _declared_frame(tmp_path / "source.fits", pixels, 1.0)
    info = replace(
        read_frame_info(source),
        normalized_unit_scale=1.0,
        numeric_domain_authority="CONTENT_BOUND_OVERRIDE",
    )
    transform = _warp_transforms(shape)[case]
    budget = shape[0] * shape[1] * 192
    numpy_destination = tmp_path / "numpy.fits"
    native_destination = tmp_path / "native.fits"
    numpy_execution: dict = {}
    monkeypatch.setattr(pipeline, "load_native_kernels", lambda *_a, **_k: None)
    numpy_stats = pipeline._register_frame(
        source, numpy_destination, transform, info,
        max_memory_bytes=budget, resampler="lanczos-3-clamped",
        execution=numpy_execution,
    )
    monkeypatch.undo()
    native_execution: dict = {}
    native_stats = pipeline._register_frame(
        source, native_destination, transform, info,
        max_memory_bytes=budget, resampler="lanczos-3-clamped",
        native_threads=3, execution=native_execution,
    )
    assert numpy_execution["warpBackend"] == "numpy"
    assert native_execution["warpBackend"] == "native-cpu"
    assert native_execution["warpKernel"] == native_kernels.WARP_KERNEL_ID
    numpy_values = fits.getdata(numpy_destination)
    native_values = fits.getdata(native_destination)
    assert np.array_equal(np.isnan(numpy_values), np.isnan(native_values))
    assert np.array_equal(numpy_values, native_values, equal_nan=True)
    assert native_stats == numpy_stats
    assert np.count_nonzero(np.isfinite(native_values)) > shape[0] * shape[1] // 2
    # Streaming digest equals the published file's digest.
    assert native_execution["sha256"] == pipeline._hash_file(native_destination)
    assert fits.getheader(native_destination)["OAFRSAMP"] == "LANCZOS-3-CLAMPED"


@requires_native
def test_native_projective_warp_places_stars_where_the_homography_says() -> None:
    kernels = native_kernels.load_native_kernels()
    assert kernels is not None
    height, width = 160, 240
    rng = np.random.default_rng(11)
    yy, xx = np.indices((height, width), dtype=np.float64)
    # Isolated stars on a jittered grid so the box centroids never blend.
    grid_y, grid_x = np.mgrid[24:height - 24:24, 24:width - 24:24]
    truth = np.column_stack((grid_x.ravel(), grid_y.ravel())) + rng.uniform(-3.0, 3.0, (grid_x.size, 2))
    image = np.full((height, width), 100.0)
    for x, y in truth:
        image += 3000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5**2))
    forward = np.asarray(
        [[1.0006, -0.0008, 2.3], [0.0007, 0.9995, -1.6], [4.0e-5, -3.0e-5, 1.0]]
    )
    inverse = np.linalg.inv(forward)
    warped = kernels.warp_lanczos3(
        image.astype(np.float32), inverse, first_row=0, row_count=height,
        output_width=width, domain_scale=65535.0, threads=2,
    )
    warped = np.nan_to_num(warped, nan=100.0).astype(np.float64)
    expected = (forward @ np.column_stack((truth, np.ones(len(truth)))).T).T
    expected = expected[:, :2] / expected[:, 2:3]
    inside = (expected[:, 0] > 8) & (expected[:, 0] < width - 8) & (expected[:, 1] > 8) & (expected[:, 1] < height - 8)
    assert np.count_nonzero(inside) > 25
    errors = []
    for x, y in expected[inside]:
        cx, cy = int(round(x)), int(round(y))
        patch = warped[cy - 4 : cy + 5, cx - 4 : cx + 5] - 100.0
        py, px = np.indices(patch.shape, dtype=np.float64)
        total = patch.sum()
        errors.append(((patch * px).sum() / total + cx - 4 - x, (patch * py).sum() / total + cy - 4 - y))
    errors = np.asarray(errors)
    # Perspective terms move the far corner by more than a pixel; the warp
    # must follow the homography, not its affine part.
    affine_only = (forward[:2] @ np.column_stack((truth, np.ones(len(truth)))).T).T
    assert np.max(np.abs(affine_only[inside] - expected[inside])) > 0.8
    assert np.max(np.abs(errors)) < 0.05


@requires_native
def test_memory_frame_register_matches_file_backed_register(tmp_path: Path) -> None:
    shape = (24, 31)
    rng = np.random.default_rng(9)
    pixels = rng.normal(500.0, 20.0, shape).astype(np.float32)
    pixels[4, 5] = np.nan
    source = _declared_frame(tmp_path / "source.fits", pixels, 65535.0)
    info = replace(
        read_frame_info(source),
        normalized_unit_scale=65535.0,
        numeric_domain_authority="CONTENT_BOUND_OVERRIDE",
    )
    transform = _warp_transforms(shape)["small-rotation"]
    with FitsFrame(source) as frame:
        memory = _MemoryFrame(frame.full_values(), info, source)
    for label, candidate in (("file", source), ("memory", memory)):
        pipeline._register_frame(
            candidate, tmp_path / f"{label}.fits", transform, info,
            max_memory_bytes=shape[0] * shape[1] * 192, resampler="lanczos-3-clamped",
        )
    assert fits.getdata(tmp_path / "file.fits").tobytes() == fits.getdata(
        tmp_path / "memory.fits"
    ).tobytes()


@requires_native
def test_native_warp_falls_back_to_numpy_when_the_budget_cannot_hold_the_source(
    tmp_path: Path,
) -> None:
    shape = (64, 20)
    pixels = np.full(shape, 0.25, dtype=np.float32)
    source = _declared_frame(tmp_path / "small.fits", pixels, 1.0)
    info = replace(
        read_frame_info(source),
        normalized_unit_scale=1.0,
        numeric_domain_authority="CONTENT_BOUND_OVERRIDE",
    )
    transform = _warp_transforms(shape)["fractional-translation"]
    execution: dict = {}
    pipeline._register_frame(
        source, tmp_path / "tiny-budget.fits", transform, info,
        max_memory_bytes=shape[1] * 192, resampler="lanczos-3-clamped",
        execution=execution,
    )
    assert execution["warpBackend"] == "numpy"
    assert execution["tileRows"] == 1


@requires_native
def test_disable_environment_variable_forces_numpy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, "1")
    native_kernels.reset_native_kernel_cache()
    try:
        assert native_kernels.load_native_kernels() is None
    finally:
        monkeypatch.delenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, raising=False)
        native_kernels.reset_native_kernel_cache()
    assert native_kernels.load_native_kernels() is not None


def _write_frame(path: Path, role: str, data: np.ndarray, *, exposure: float = 30.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["IMAGETYP"] = role
    header["FILTER"] = "R"
    header["OBJECT"] = "SYNTHETIC"
    header["INSTRUME"] = "SYNTH-CAM"
    header["EXPTIME"] = exposure
    header["GAIN"] = 100
    header["OFFSET"] = 50
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "MODE-1"
    header["BAYERPAT"] = "NONE"
    header["CCD-TEMP"] = -10.0
    fits.writeto(path, np.asarray(np.rint(data), dtype=np.uint16), header, overwrite=False)
    return path


@requires_native
@pytest.mark.parametrize("materialize", [True, False])
def test_fused_pipeline_matches_numpy_pipeline_and_optionally_skips_calibrated_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, materialize: bool
) -> None:
    height, width = 48, 64
    rng = np.random.default_rng(17)
    y, x = np.mgrid[:height, :width]
    response = np.linspace(0.85, 1.15, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    signal = (600.0 + 2.0 * x + 1.5 * y).astype(np.float32)
    root = tmp_path / "raw"
    biases = [_write_frame(root / "bias" / f"bias_{i}.fits", "Bias", np.full((height, width), 100.0 + i), exposure=0.001) for i in range(3)]
    darks = [_write_frame(root / "dark" / f"dark_{i}.fits", "Dark", np.full((height, width), 120.0 + i)) for i in range(3)]
    flats = [_write_frame(root / "flat" / f"flat_{i}.fits", "Flat", 100.0 + 1000.0 * response, exposure=2.0) for i in range(3)]
    lights = []
    for index in range(6):
        raw = 120.0 + signal * response + rng.normal(0.0, 3.0, (height, width))
        if index == 5:
            raw[20, 30] += 9000.0
        lights.append(_write_frame(root / "light" / f"light_{index}.fits", "Light", raw))
    angle = np.deg2rad(0.3)
    transforms = {
        str(path): AffineTransform.from_value(
            (
                (np.cos(angle * index), -np.sin(angle * index), 0.37 * index),
                (np.sin(angle * index), np.cos(angle * index), -0.21 * index),
                (0.0, 0.0, 1.0),
            )
        )
        for index, path in enumerate(lights)
    }
    parameters = PipelineParameters(
        integration=IntegrationParameters(
            sigma_clip=4.0, minimum_rejection_frames=3,
            max_memory_bytes=512 * 1024, max_statistics_samples=200,
        ),
        registration_memory_bytes=4 * 1024 * 1024,
        preview_max_long_edge=64,
        ordinary_integration_backend="portable-cpu",
        materialize_calibrated_lights=materialize,
        global_normalization=GlobalNormalizationParameters(enabled=False),
    )
    results = {}
    for label in ("native", "numpy"):
        if label == "numpy":
            monkeypatch.setenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, "1")
        native_kernels.reset_native_kernel_cache()
        result = pipeline.run_portable_pipeline(
            bias_files=biases, dark_files=darks, flat_files=flats, light_files=lights,
            output_directory=tmp_path / f"{label}-{materialize}",
            parameters=parameters, transforms=transforms,
        )
        receipt = json.loads(Path(result.receipt_path).read_text())
        results[label] = (result, receipt)
        monkeypatch.delenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, raising=False)
        native_kernels.reset_native_kernel_cache()
    native_result, native_receipt = results["native"]
    numpy_result, numpy_receipt = results["numpy"]
    assert fits.getdata(native_result.master_light_paths[0]).tobytes() == fits.getdata(
        numpy_result.master_light_paths[0]
    ).tobytes()
    native_registration = native_receipt["statistics"]["registration"]
    assert native_registration["executionModel"] == "fused-calibrate-warp-v1"
    assert native_registration["warpBackends"] == {"identity-copy": 1, "native-cpu": 5}
    assert numpy_receipt["statistics"]["registration"]["warpBackends"] == {
        "identity-copy": 1, "numpy": 5,
    }
    assert native_registration["calibratedLightsMaterialized"] is materialize
    kinds = [item["kind"] for item in native_receipt["outputs"]]
    assert kinds.count("REGISTERED_LIGHT") == 6
    assert kinds.count("CALIBRATED_LIGHT") == (6 if materialize else 0)
    assert (Path(native_result.output_directory) / "calibrated").exists() is materialize
    # Every published artifact digest must equal the file's digest, including
    # the ones computed by the streaming writer.
    for item in native_receipt["outputs"]:
        assert item["sha256"] == pipeline._hash_file(
            Path(native_result.output_directory) / item["path"]
        )
    # Registered pixels are identical across backends and the receipts agree on
    # every scientific field.
    native_registered = sorted((Path(native_result.output_directory) / "registered").glob("*.fits"))
    numpy_registered = sorted((Path(numpy_result.output_directory) / "registered").glob("*.fits"))
    for left, right in zip(native_registered, numpy_registered, strict=True):
        assert fits.getdata(left).tobytes() == fits.getdata(right).tobytes()
    group = native_receipt["statistics"]["integrationGroups"]["R"]["integration"]
    assert group["rejectedSamples"] >= 1
    assert group["execution"]["rejectionMask"]["kernel"] == native_kernels.MAD_KERNEL_ID


def _random_tile_pairs(seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    tiles: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(40):
        size = int(rng.integers(0, 2600))
        target = rng.normal(900.0, 40.0, size)
        reference = 1.1 * target + 35.0 + rng.normal(0.0, 3.0, size)
        if index % 5 == 0 and size:
            reference[rng.random(size) < 0.03] += 4000.0  # outliers
        if index % 7 == 0 and size:
            target[rng.random(size) < 0.05] = np.nan
            reference[rng.random(size) < 0.02] = np.inf
        if index % 11 == 0 and size:
            reference[:] = 1.1 * target + 12.0  # zero residual dispersion
        tiles.append((np.asarray(target, dtype=np.float64), np.asarray(reference, dtype=np.float64)))
    tiles.append((np.full(700, 5.0), np.full(700, 9.0)))  # constant tile
    tiles.append((np.array([], dtype=np.float64), np.array([], dtype=np.float64)))
    return tiles


@requires_native
def test_native_tile_offsets_match_numpy_tile_offset_bitwise() -> None:
    from openastroflow_engine.global_normalization import _tile_offset

    assert KERNELS is not None
    tiles = _random_tile_pairs(23)
    parameters = GlobalNormalizationParameters(minimum_samples_per_offset_tile=200)
    scale = 1.1
    boundaries = np.concatenate(([0], np.cumsum([target.size for target, _ in tiles]))).astype(np.uint64)
    offsets, counts, mads, valid = KERNELS.tile_offsets(
        np.concatenate([target for target, _ in tiles]),
        np.concatenate([reference for _, reference in tiles]),
        boundaries,
        scale=scale,
        lower_quantile=parameters.lower_quantile,
        upper_quantile=parameters.upper_quantile,
        minimum_samples=parameters.minimum_samples_per_offset_tile,
        residual_clip_sigma=parameters.residual_clip_sigma,
        threads=3,
    )
    valid_count = 0
    for index, (target, reference) in enumerate(tiles):
        expected = _tile_offset(target, reference, scale, parameters)
        if expected is None:
            assert not valid[index]
            assert np.isnan(offsets[index]) and np.isnan(mads[index])
            continue
        valid_count += 1
        assert valid[index]
        assert offsets[index] == expected[0]
        assert int(counts[index]) == expected[1]
        assert mads[index] == expected[2]
    assert valid_count >= 20


def _registered_group(tmp_path: Path, seed: int = 5) -> list[Path]:
    rng = np.random.default_rng(seed)
    height, width = 640, 768
    y, x = np.mgrid[:height, :width]
    base = 1200.0 + 0.05 * x + 0.02 * y
    paths = []
    for index in range(3):
        values = base * (1.0 + 0.03 * index) + 15.0 * index + 0.01 * index * x
        values = values + rng.normal(0.0, 4.0, (height, width))
        values[:, :5] = np.nan
        header = fits.Header()
        header["IMAGETYP"] = "Registered Light"
        header["OAFNDOM"] = "INTEGER_16_PHYSICAL_0_BASED"
        header["OAFNSCL"] = 65535.0
        path = tmp_path / f"registered-{index}.fits"
        fits.writeto(path, values.astype(np.float32), header, overwrite=False)
        paths.append(path)
    return paths


@requires_native
def test_global_normalization_native_and_numpy_coefficients_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openastroflow_engine.global_normalization import (
        fit_registered_group_global_normalization,
    )

    paths = _registered_group(tmp_path)
    parameters = GlobalNormalizationParameters(minimum_valid_offset_tiles=4)
    results = {}
    for label in ("native", "numpy"):
        if label == "numpy":
            monkeypatch.setenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, "1")
        native_kernels.reset_native_kernel_cache()
        results[label] = fit_registered_group_global_normalization(
            [str(path) for path in paths], reference_index=0, parameters=parameters, workers=3
        )
        monkeypatch.delenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, raising=False)
        native_kernels.reset_native_kernel_cache()
    native, numpy_result = results["native"], results["numpy"]
    for left, right in zip(native.coefficients, numpy_result.coefficients, strict=True):
        assert left.scale == right.scale
        assert left.offset == right.offset
        assert left.mode == right.mode
        assert left.offset_grid == right.offset_grid
        assert left.offset_grid_x == right.offset_grid_x
        assert left.offset_grid_y == right.offset_grid_y
        left_model = dict(left.evidence["additiveModel"])
        right_model = dict(right.evidence["additiveModel"])
        left_model.pop("tileStatisticsKernel", None)
        right_model.pop("tileStatisticsKernel", None)
        assert left_model == right_model
    modes = {item.mode for item in native.coefficients}
    assert modes & {"STELLAR_SCALE_ADDITIVE_GRID", "UNIT_SCALE_ADDITIVE_GRID_STELLAR_UNAVAILABLE",
                    "UNIT_SCALE_SCALAR_OFFSET_STELLAR_UNAVAILABLE", "STELLAR_SCALE_SCALAR_OFFSET"}
    assert native.coefficients[1].evidence["additiveModel"].get("tileStatisticsKernel") in (
        native_kernels.TILE_OFFSET_KERNEL_ID, None
    )


def test_trail_background_normalization_row_runs_match_full_tile_reference() -> None:
    from openastroflow_engine.residual_background import ResidualBackgroundAlignment
    from openastroflow_engine.transient_rejection import TransientRejectionModel

    rng = np.random.default_rng(3)
    frames, rows, width = 6, 40, 32
    x_nodes = np.linspace(0.0, width - 1, 4)
    y_nodes = np.linspace(0.0, 200.0, 5)
    corrections = rng.normal(0.0, 2.0, (frames, y_nodes.size, x_nodes.size))
    alignment = ResidualBackgroundAlignment(
        x_nodes=x_nodes, y_nodes=y_nodes, corrections=corrections,
        evidence=tuple({} for _ in range(frames)), weights=tuple(rng.random(frames) + 0.1),
    )
    values = rng.normal(500.0, 10.0, (frames, rows, width)).astype(np.float32)
    original = rng.random((frames, rows, width)) > 0.1
    accepted = original.copy()
    for row in (3, 4, 5, 17, 30, 31):
        accepted[2, row, rng.random(width) > 0.5] = False
    model = TransientRejectionModel(4, (), "APPLIED", alignment)

    # Reference: the previous whole-tile formulation.
    reference = values.copy()
    changed = np.any(original & ~accepted, axis=0)
    correction = np.zeros_like(reference)
    alignment.apply_rows(correction, 120)
    weights = np.asarray(alignment.weights)[:, None, None]
    denominator = np.sum(original * weights, axis=0)
    anchor = np.divide(np.sum(np.where(original, correction, 0) * weights, axis=0),
                       denominator, out=np.zeros_like(denominator), where=denominator > 0)
    for index in range(frames):
        reference[index, changed] += (correction[index, changed] - anchor[changed]).astype(np.float32)

    candidate = values.copy()
    model.normalize_rejected_rows(candidate, 120, original, accepted)
    np.testing.assert_array_equal(candidate, reference)
    untouched = values.copy()
    model.normalize_rejected_rows(untouched, 120, original, original)
    np.testing.assert_array_equal(untouched, values)


def test_sampled_expression_rows_match_per_row_evaluation(tmp_path: Path) -> None:
    from openastroflow_engine.calibration import (
        _canonical_expression,
        _expression_rows,
        _expression_sampled_rows,
        _open_expression_sources,
        _sample_expression,
    )
    from contextlib import ExitStack

    rng = np.random.default_rng(29)
    height, width = 37, 41
    light = rng.normal(1200.0, 40.0, (height, width)).astype(np.float32)
    dark = rng.normal(300.0, 3.0, (height, width)).astype(np.float32)
    flat = rng.normal(1.0, 0.02, (height, width)).astype(np.float32)
    flat[5, 7] = 0.0
    flat[9, 3] = np.nan
    light[12, 12] = np.inf
    paths = {}
    for name, values in (("light", light), ("dark", dark), ("flat", flat)):
        path = tmp_path / f"{name}.fits"
        fits.writeto(path, values, fits.Header({"IMAGETYP": name}), overwrite=False)
        paths[name] = path
    grid = ((-2.0, 1.0, 3.0), (0.5, -1.5, 2.0), (4.0, 0.0, -3.0))
    expression = _canonical_expression(
        FrameExpression(
            str(paths["light"]), subtract_path=str(paths["dark"]), subtract_scale=1.5,
            divide_path=str(paths["flat"]), scale=0.75, offset=2.5,
            offset_grid=grid, offset_grid_x=(0.0, 20.0, 40.0), offset_grid_y=(0.0, 18.0, 36.0),
        )
    )
    y_indices = np.asarray([0, 3, 5, 9, 12, 20, 36], dtype=np.int64)
    with ExitStack() as stack:
        sources = _open_expression_sources(stack, (expression,))
        sampled = _expression_sampled_rows(expression, sources, y_indices, division_floor=1e-12)
        for position, y in enumerate(y_indices):
            row = _expression_rows(expression, sources, int(y), int(y) + 1, division_floor=1e-12)[0]
            assert np.array_equal(sampled[position], row, equal_nan=True)
        # The sampled statistics helper must equal a per-row reference.
        from openastroflow_engine.calibration import _sample_coordinates

        rows, columns = _sample_coordinates((height, width), 120)
        expected = []
        for y in rows:
            values = _expression_rows(expression, sources, int(y), int(y) + 1, division_floor=1e-12)[0, columns]
            expected.append(values[np.isfinite(values)])
        expected_samples = np.concatenate(expected)[:120]
        actual = _sample_expression(expression, sources, (height, width), max_samples=120, division_floor=1e-12)
        np.testing.assert_array_equal(actual, expected_samples)


def test_nanmedian_frames_matches_numpy_nanmedian() -> None:
    from openastroflow_engine.robust_statistics import nanmedian_frames

    rng = np.random.default_rng(5)
    for frames in (1, 2, 3, 4, 7, 12, 13):
        values = (rng.random((frames, 3000)) ** 6 * 65535.0).astype(np.float32)
        values[rng.random(values.shape) < 0.2] = np.nan
        values[:, 11] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmedian(values, axis=0)
        actual = nanmedian_frames(values)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected.astype(np.float32))
        assert np.isnan(actual[11])


def test_rejection_sigma_floor_matches_per_row_reference(tmp_path: Path) -> None:
    from contextlib import ExitStack

    from openastroflow_engine.calibration import (
        REJECTION_FLOOR_ABSOLUTE,
        REJECTION_FLOOR_GROUP_FRACTION,
        REJECTION_FLOOR_MAX_SAMPLES,
        IntegrationParameters,
        _canonical_expression,
        _estimate_rejection_sigma_floor,
        _expression_rows,
        _open_expression_sources,
        _sample_coordinates,
    )

    rng = np.random.default_rng(23)
    height, width = 96, 128
    expressions = []
    for index in range(6):
        values = rng.normal(500.0 + 10.0 * index, 8.0, (height, width)).astype(np.float32)
        values[rng.random(values.shape) < 0.03] = np.nan
        values[10:14, :] = np.inf if index == 2 else values[10:14, :]
        values[:, 40:44] = np.nan  # columns with too few finite samples
        path = tmp_path / f"frame-{index}.fits"
        fits.writeto(path, values, fits.Header({"IMAGETYP": "LIGHT"}), overwrite=False)
        expressions.append(_canonical_expression(FrameExpression(str(path))))
    expressions = tuple(expressions)
    parameters = IntegrationParameters(max_statistics_samples=5000)

    with ExitStack() as stack:
        sources = _open_expression_sources(stack, expressions)
        actual = _estimate_rejection_sigma_floor(expressions, sources, (height, width), parameters)
        # One-row-at-a-time reference (the previous implementation).
        sample_limit = min(int(parameters.max_statistics_samples), REJECTION_FLOOR_MAX_SAMPLES)
        y_indices, x_indices = _sample_coordinates((height, width), sample_limit)
        chunks = []
        for y in y_indices:
            row_values = np.empty((len(expressions), x_indices.size), dtype=np.float32)
            for frame_index, expression in enumerate(expressions):
                row = _expression_rows(expression, sources, int(y), int(y) + 1, division_floor=parameters.division_floor)
                row_values[frame_index] = row[0, x_indices]
            finite_count = np.count_nonzero(np.isfinite(row_values), axis=0)
            eligible = finite_count >= parameters.minimum_rejection_frames
            if not np.any(eligible):
                continue
            selected = row_values[:, eligible]
            selected[~np.isfinite(selected)] = np.nan
            center = np.nanmedian(selected, axis=0)
            mad = np.nanmedian(np.abs(selected - center[None, :]), axis=0)
            robust_sigma = np.asarray(np.float32(1.4826) * mad, dtype=np.float32)
            usable = np.isfinite(robust_sigma) & (robust_sigma > 0)
            if np.any(usable):
                chunks.append(robust_sigma[usable])
        sampled_sigma = np.concatenate(chunks)
        expected_median = float(np.median(sampled_sigma))
    assert actual.applicable
    assert actual.usable_sigma_count == int(sampled_sigma.size)
    assert actual.sampled_sigma_median == expected_median
    assert actual.group_sigma_floor == max(REJECTION_FLOOR_ABSOLUTE, expected_median * REJECTION_FLOOR_GROUP_FRACTION)


def _trail_preview(seed: int, height: int, width: int, frames: int = 7) -> np.ndarray:
    """Block-mean previews with a faint diagonal trail in one frame."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    sky = 100.0 + 0.01 * xx + 0.02 * yy
    values = sky[None] + rng.normal(0.0, 1.0, (frames, height, width))
    stars = rng.random((height, width)) < 0.002
    values[:, stars] += 40.0
    values[3][np.abs(yy - 0.6 * xx - 8) < 1.2] += 1.5
    values[:, :, :3] = np.nan
    return values.astype(np.float32)


@requires_native
@pytest.mark.parametrize("seed,height,width", [(3, 96, 150), (11, 130, 90)])
def test_native_radon_peaks_match_numpy_candidate_lines_bitwise(
    seed: int, height: int, width: int
) -> None:
    from openastroflow_engine import transient_rejection as tr

    assert KERNELS is not None
    values = _trail_preview(seed, height, width)
    finite = np.isfinite(values)
    common = np.sum(finite, axis=0) >= 5
    reference = np.zeros((height, width), dtype=np.float32)
    reference[common] = np.nanmedian(values[:, common], axis=0)
    for index in (0, 3):
        detect = common & finite[index]
        residual = np.where(detect, values[index] - reference, 0.0)
        sigma = float(1.4826 * np.median(np.abs(residual[detect])))
        residual_z = np.where(detect, np.clip(residual / sigma, -5.0, 5.0), 0.0).astype(np.float32)
        native = tr._candidate_lines(residual_z, detect, 16, kernels=KERNELS, threads=3)
        numpy_path = tr._candidate_lines(residual_z, detect, 16, kernels=None)
        assert len(native) == len(numpy_path)
        for (z_native, p0_native, p1_native), (z_numpy, p0_numpy, p1_numpy) in zip(native, numpy_path):
            assert z_native == z_numpy
            assert np.array_equal(p0_native, p0_numpy)
            assert np.array_equal(p1_native, p1_numpy)
        if index == 3:
            assert len(native) >= 1


@requires_native
def test_native_radon_peaks_reject_bad_geometry() -> None:
    assert KERNELS is not None
    image = np.zeros((20, 30), dtype=np.float32)
    weight = np.ones((20, 30), dtype=np.uint8)
    common = dict(minimum_rows=16, detection_z=6.5, minimum_coverage=0.6,
                  minimum_count=8.0, minimum_scale_samples=64, threads=1)
    with pytest.raises(native_kernels.NativeKernelError):
        KERNELS.radon_line_peaks(image, weight, size=24, **common)  # not a power of two
    with pytest.raises(native_kernels.NativeKernelError):
        KERNELS.radon_line_peaks(image, weight, size=16, **common)  # smaller than the image
    with pytest.raises(ValueError):
        KERNELS.radon_line_peaks(image, weight[:10], size=32, **common)


@requires_native
def test_detect_transient_trails_native_and_numpy_models_match(monkeypatch: pytest.MonkeyPatch) -> None:
    from openastroflow_engine import transient_rejection as tr

    values = _trail_preview(5, 110, 160)
    native = tr.detect_transient_trails(values, 4, workers=3)
    assert native.line_kernel == native_kernels.RADON_KERNEL_ID
    monkeypatch.setenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE, "1")
    native_kernels.reset_native_kernel_cache()
    try:
        reference = tr.detect_transient_trails(values, 4, workers=1)
    finally:
        monkeypatch.delenv(native_kernels.DISABLE_ENVIRONMENT_VARIABLE)
        native_kernels.reset_native_kernel_cache()
    assert reference.line_kernel == tr.NUMPY_RADON_KERNEL_ID
    assert native.trails == reference.trails
    assert len(native.trails) >= 1
    assert native.serializable()["lineKernel"] == native_kernels.RADON_KERNEL_ID
