"""Rejection scale model (row-pooled MAD, per-frame noise factors) and the
block-mean noise weights: exactness of the helpers and the behaviour they
were introduced for, on inputs that differ from any one real data set."""

from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from ufwbpp import native_kernels
from ufwbpp.stacking.integration import (
    NOISE_WEIGHT_BLOCK_SIZE,
    REJECTION_POOL_HALF_WIDTH,
    FrameExpression,
    IntegrationMapPaths,
    IntegrationParameters,
    _estimate_rejection_sigma_floor,
    _frame_noise_estimates,
    _noise_weights_from_estimates,
    _open_expression_sources,
    _ordinary_mad_rejection_decision,
    _pooled_row_mad,
    _rejection_frame_scales,
    _validate_expression_shapes,
    integrate_expressions,
)


def _naive_pooled_row_mad(mad: np.ndarray, half_width: int) -> np.ndarray:
    rows, width = mad.shape
    result = np.full(mad.shape, np.nan, dtype=np.float32)
    for y in range(rows):
        for x in range(width):
            window = mad[y, max(0, x - half_width) : min(width, x + half_width + 1)]
            finite = window[np.isfinite(window)]
            if finite.size:
                result[y, x] = np.float32(np.median(finite))
    return result


@pytest.mark.parametrize("half_width", [0, 1, 3, 12, 40])
def test_pooled_row_mad_matches_naive_nanmedian(half_width: int) -> None:
    rng = np.random.default_rng(half_width)
    mad = rng.gamma(2.0, 1.5, (5, 37)).astype(np.float32)
    mad[rng.random(mad.shape) < 0.2] = np.nan
    mad[2, :] = np.nan
    mad[3, :5] = np.float32(0.0)
    pooled = _pooled_row_mad(mad, half_width)
    assert pooled.dtype == np.float32
    np.testing.assert_array_equal(pooled, _naive_pooled_row_mad(mad, half_width))
    if half_width == 0:
        np.testing.assert_array_equal(pooled, mad)


def test_rejection_frame_scales_are_monotone_and_anchored_on_the_mixture() -> None:
    # One noise level: every frame sits at the sampled mixture scale or above
    # it by the small-sample bias of the per-pixel MAD, never below 1.
    scales, mixture = _rejection_frame_scales([2.0, 2.0, 2.0, 2.0], 2.0)
    assert scales == (1.0, 1.0, 1.0, 1.0) and mixture == 2.0
    scales, mixture = _rejection_frame_scales([2.0, 2.0], 1.85)
    assert scales[0] == scales[1] > 1.0
    scales, mixture = _rejection_frame_scales([1.0, 1.0, float("nan"), 2.0], 1.25)
    assert scales == (1.0, 1.0, 1.0, float(np.float32(1.6)))
    assert all(float(np.float32(value)) == value for value in scales)
    # Without a mixture scale (or an unusable one) every factor is 1.
    assert _rejection_frame_scales([1.0, 2.0], None) == ((1.0, 1.0), None)
    assert _rejection_frame_scales([1.0, 2.0], float("nan")) == ((1.0, 1.0), None)
    assert _rejection_frame_scales([1.0, 2.0], 0.0) == ((1.0, 1.0), None)
    # Empirically: the per-pixel MAD of a two-level stack sits between the
    # levels, so the noisy frames get their own noise back and the quiet ones
    # are held at the mixture scale.
    rng = np.random.default_rng(3)
    frame_sigmas = np.array([1.0] * 8 + [1.7] * 5)
    stack = rng.standard_normal((13, 200_000)) * frame_sigmas[:, None]
    center = np.median(stack, axis=0)
    mad_sigma = 1.4826 * float(np.median(np.median(np.abs(stack - center), axis=0)))
    assert 1.0 < mad_sigma < 1.7
    scales, mixture = _rejection_frame_scales(frame_sigmas.tolist(), mad_sigma)
    assert scales[0] == 1.0
    assert abs(scales[-1] * mad_sigma - 1.7) < 1e-4


def _lanczos3_shift(image: np.ndarray, phase: float) -> np.ndarray:
    from scipy.ndimage import convolve1d

    xs = np.arange(-2, 4) - phase
    taps = np.sinc(xs) * np.sinc(xs / 3.0)
    taps[np.abs(xs) >= 3.0] = 0.0
    taps /= taps.sum()
    return convolve1d(convolve1d(image, taps, axis=1, mode="wrap"), taps, axis=0, mode="wrap")


def test_block_noise_is_insensitive_to_resampling_phase(tmp_path: Path) -> None:
    """Per-pixel noise drops by the kernel's sum of squares after a half-pixel
    Lanczos-3 shift; the block-mean noise, which weights the frames, does not."""

    rng = np.random.default_rng(11)
    base = (1000.0 + rng.normal(0.0, 10.0, (256, 512))).astype(np.float64)
    shifted = _lanczos3_shift(base, 0.5)
    paths = []
    for name, data in (("integer", base), ("half", shifted)):
        path = tmp_path / f"{name}.fits"
        fits.writeto(path, data.astype(np.float32))
        paths.append(path)
    expressions = tuple(FrameExpression(str(path)) for path in paths)
    with ExitStack() as stack:
        sources = _open_expression_sources(stack, expressions)
        shape = _validate_expression_shapes(expressions, sources)
        estimates = _frame_noise_estimates(
            expressions, sources, shape, IntegrationParameters(max_statistics_samples=60_000)
        )
    assert estimates.block_size == NOISE_WEIGHT_BLOCK_SIZE
    pixel_ratio = estimates.sigma_pixel[1] / estimates.sigma_pixel[0]
    block_ratio = estimates.sigma_block[1] / estimates.sigma_block[0]
    assert 0.72 < pixel_ratio < 0.82  # variance x0.62 (0.786 per axis), sigma x0.79
    assert 0.94 < block_ratio < 1.03
    weights, _ = _noise_weights_from_estimates(estimates)
    assert abs(weights[0] / weights[1] - 1.0) < 0.12
    assert estimates.pixel_difference_counts[0] > 1000
    assert estimates.block_difference_counts[0] > 1000
    serial = estimates.serializable()
    assert serial["algorithm"] == "block-mean-effective-noise-v2"
    assert len(serial["frameSigmaBlock"]) == 2


def test_block_noise_falls_back_on_tiny_images(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    paths = []
    for index in range(3):
        path = tmp_path / f"tiny-{index}.fits"
        fits.writeto(path, (100.0 + rng.normal(0.0, 3.0, (3, 40))).astype(np.float32))
        paths.append(path)
    expressions = tuple(FrameExpression(str(path)) for path in paths)
    with ExitStack() as stack:
        sources = _open_expression_sources(stack, expressions)
        shape = _validate_expression_shapes(expressions, sources)
        estimates = _frame_noise_estimates(
            expressions, sources, shape, IntegrationParameters(max_statistics_samples=1000)
        )
    assert all(math.isnan(value) for value in estimates.sigma_block)
    weights, serialized = _noise_weights_from_estimates(estimates)
    assert abs(sum(weights) - 1.0) < 1e-12 and len(serialized) == 3


def _decisions(stack: np.ndarray, parameters: IntegrationParameters, pooled: bool):
    from ufwbpp.stacking.integration import _RejectionSigmaFloor

    if pooled:
        frame_sigmas = [
            1.4826 * float(np.median(np.abs(np.diff(frame, axis=1)))) / math.sqrt(2.0)
            for frame in stack
        ]
        center = np.median(stack, axis=0)
        sampled = 1.4826 * float(np.median(np.median(np.abs(stack - center), axis=0)))
        scales, mixture = _rejection_frame_scales(frame_sigmas, sampled)
        floor = _RejectionSigmaFloor(
            True, 1e-7, sampled, 1, 1, 1, 1, 1, 1, 3, "sha256:test",
            scales, parameters.rejection_pool_half_width, tuple(frame_sigmas), mixture,
        )
    else:
        floor = _RejectionSigmaFloor(True, 1e-7, None, 1, 1, 1, 1, 1, 1, 3, "sha256:test")
    return _ordinary_mad_rejection_decision(stack, parameters, floor)


@pytest.mark.parametrize("frames", [3, 5, 11, 13])
def test_pooled_scale_rejects_far_fewer_good_sky_samples(frames: int) -> None:
    rng = np.random.default_rng(frames)
    stack = (1000.0 + rng.normal(0.0, 8.0, (frames, 16, 600))).astype(np.float32)
    parameters = IntegrationParameters()
    _, _, legacy = _decisions(stack, parameters, pooled=False)
    _, _, pooled = _decisions(stack, parameters, pooled=True)
    legacy_rate = 1.0 - legacy.mean()
    pooled_rate = 1.0 - pooled.mean()
    # Pure Gaussian stacks: the per-pixel MAD clips 0.2-2% of good samples,
    # the pooled scale well under a tenth of that.
    assert pooled_rate < 0.25 * legacy_rate
    assert pooled_rate < 0.002


def test_pooled_scale_still_rejects_real_outliers_and_keeps_star_cores() -> None:
    rng = np.random.default_rng(9)
    frames = 11
    stack = (1000.0 + rng.normal(0.0, 8.0, (frames, 12, 400))).astype(np.float32)
    # Hot pixel column in one frame (10 sigma), a satellite-like streak in
    # another (15 sigma), and a 6 sigma streak that only partly clears the
    # threshold under either rule.
    stack[3, :, 100] += np.float32(80.0)
    stack[7, 5, 150:260] += np.float32(120.0)
    stack[9, 8, 150:260] += np.float32(48.0)
    # Two "star cores" whose brightness varies with seeing (frame 2 sharpest).
    seeing = np.linspace(1.0, 1.6, frames).astype(np.float32)
    stack[:, 6, 300] = np.float32(20000.0) / seeing**2
    stack[:, 6, 301] = np.float32(9000.0) / seeing**2
    parameters = IntegrationParameters()
    _, _, legacy = _decisions(stack, parameters, pooled=False)
    _, _, pooled = _decisions(stack, parameters, pooled=True)
    assert not pooled[3, :, 100].any()
    assert not pooled[7, 5, 150:260].any()
    # The marginal streak: one outlier per pixel inflates an 11-sample MAD,
    # so both rules keep a minority of its samples; the pooled scale
    # (bias-corrected, hence slightly wider) keeps a few more.
    assert pooled[9, 8, 150:260].mean() < 0.5
    assert pooled[9, 8, 150:260].mean() <= legacy[9, 8, 150:260].mean() + 0.15
    assert pooled[9, 8, 150:260].sum() >= legacy[9, 8, 150:260].sum()
    # Star cores: the excess per-pixel variance is kept, so the pooled model
    # never rejects more core samples than the per-pixel rule.
    assert pooled[:, 6, 300:302].sum() >= legacy[:, 6, 300:302].sum()
    # Everywhere the pooled model only removes rejections.
    assert not np.any(legacy & ~pooled)


def test_frame_noise_scaling_gives_every_frame_its_own_threshold() -> None:
    """A noisy night mixed with a quiet one: the mixture MAD used to clip the
    noisy frames at ~3 of their own sigma; the frame factors restore 4."""

    rng = np.random.default_rng(21)
    frames = 13
    sigmas = np.array([1.7] * 5 + [1.0] * 8, dtype=np.float64) * 6.0
    stack = (1000.0 + rng.standard_normal((frames, 12, 800)) * sigmas[:, None, None]).astype(np.float32)
    parameters = IntegrationParameters()
    _, _, legacy = _decisions(stack, parameters, pooled=False)
    _, _, pooled = _decisions(stack, parameters, pooled=True)
    noisy_legacy = 1.0 - legacy[:5].mean()
    noisy_pooled = 1.0 - pooled[:5].mean()
    quiet_pooled = 1.0 - pooled[5:].mean()
    assert noisy_legacy > 0.01
    assert noisy_pooled < 0.002
    assert quiet_pooled < 0.002
    # Monotone: the scale model never rejects a sample the per-pixel rule kept.
    assert not np.any(legacy & ~pooled)


def test_integration_receipt_records_scale_model_and_noise_evidence(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    paths = []
    for index in range(6):
        sigma = 4.0 if index < 3 else 7.0
        data = (500.0 + rng.normal(0.0, sigma, (192, 256))).astype(np.float32)
        path = tmp_path / f"frame-{index}.fits"
        fits.writeto(path, data)
        paths.append(path)
    result = integrate_expressions(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "master.fits",
        parameters=IntegrationParameters(max_statistics_samples=20_000),
        map_paths=IntegrationMapPaths(
            tmp_path / "accepted.fits", tmp_path / "coverage.fits", tmp_path / "rejected.fits"
        ),
    )
    execution = result.execution
    assert execution["rejectionMask"]["method"] == "median-pooled-mad-sigma-v2"
    model = execution["rejectionMask"]["sigmaFloor"]["scaleModel"]
    assert model["status"] == "APPLIED"
    assert model["poolHalfWidth"] == REJECTION_POOL_HALF_WIDTH
    assert model["frameNoiseScaling"] is True
    frame_model = execution["rejectionMask"]["frameScaleModel"]
    assert len(frame_model["frameScales"]) == 6
    assert max(frame_model["frameScales"][:3]) < min(frame_model["frameScales"][3:])
    assert min(frame_model["frameScales"]) >= 1.0
    assert model["mixtureSigma"] == execution["rejectionMask"]["sigmaFloor"]["sampledSigmaMedian"]
    noise = execution["noiseWeights"]
    assert noise["algorithm"] == "block-mean-effective-noise-v2"
    assert len(noise["frameSigmaBlock"]) == 6
    # Quiet frames carry roughly (7/4)^2 = 3.1 times the weight of noisy ones.
    weights = result.noise_weights
    assert 2.3 < weights[0] / weights[5] < 4.0
    # The legacy behaviour stays selectable and is recorded as such.
    legacy = integrate_expressions(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "legacy.fits",
        parameters=IntegrationParameters(
            max_statistics_samples=20_000,
            rejection_pool_half_width=0,
            rejection_frame_noise_scaling=False,
        ),
    )
    assert legacy.execution["rejectionMask"]["method"] == "median-mad-sigma"
    assert legacy.execution["rejectionMask"]["sigmaFloor"]["scaleModel"]["status"] == "PER_PIXEL_MAD"
    assert legacy.rejected_samples >= result.rejected_samples


def test_integration_parameters_validate_scale_model_fields() -> None:
    with pytest.raises(ValueError):
        IntegrationParameters(rejection_pool_half_width=-1).validate()
    with pytest.raises(ValueError):
        IntegrationParameters(rejection_pool_half_width=1000).validate()
    with pytest.raises(ValueError):
        IntegrationParameters(rejection_frame_noise_scaling="yes").validate()  # type: ignore[arg-type]
    serial = IntegrationParameters().serializable()
    assert serial["rejectionPoolHalfWidth"] == REJECTION_POOL_HALF_WIDTH
    assert serial["rejectionFrameNoiseScaling"] is True


def test_sigma_floor_without_noise_estimates_keeps_unit_scales(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    paths = []
    for index in range(4):
        path = tmp_path / f"f{index}.fits"
        fits.writeto(path, (10.0 + rng.normal(0.0, 1.0, (32, 32))).astype(np.float32))
        paths.append(path)
    expressions = tuple(FrameExpression(str(path)) for path in paths)
    with ExitStack() as stack:
        sources = _open_expression_sources(stack, expressions)
        shape = _validate_expression_shapes(expressions, sources)
        floor = _estimate_rejection_sigma_floor(
            expressions, sources, shape, IntegrationParameters(max_statistics_samples=500)
        )
        assert floor.frame_scales == ()
        assert floor.pool_half_width == REJECTION_POOL_HALF_WIDTH
        assert floor.pooled
        estimates = _frame_noise_estimates(
            expressions, sources, shape, IntegrationParameters(max_statistics_samples=500)
        )
        scaled = _estimate_rejection_sigma_floor(
            expressions, sources, shape, IntegrationParameters(max_statistics_samples=500),
            frame_noise=estimates,
        )
        assert len(scaled.frame_scales) == 4
        assert scaled.mixture_sigma is not None
        # Too few frames for rejection: no scale model at all.
        few = _estimate_rejection_sigma_floor(
            expressions[:2], sources, shape, IntegrationParameters(max_statistics_samples=500),
            frame_noise=estimates,
        )
        assert not few.applicable and not few.pooled


@pytest.mark.skipif(
    native_kernels.load_native_kernels() is None,
    reason="native kernel library is not built in this checkout",
)
def test_scale_model_is_tile_invariant(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    paths = []
    for index in range(7):
        data = (300.0 + rng.normal(0.0, 5.0, (64, 128))).astype(np.float32)
        if index == 2:
            data[10:20, 40] += np.float32(60.0)
        path = tmp_path / f"t{index}.fits"
        fits.writeto(path, data)
        paths.append(path)
    outputs = []
    for label, budget in (("small", 64 * 1024), ("large", 64 * 1024 * 1024)):
        result = integrate_expressions(
            [FrameExpression(str(path)) for path in paths],
            tmp_path / f"{label}.fits",
            parameters=IntegrationParameters(max_memory_bytes=budget, max_statistics_samples=3000),
        )
        outputs.append((result, fits.getdata(result.output_path)))
    assert outputs[0][0].tile_rows < outputs[1][0].tile_rows
    assert outputs[0][0].rejected_samples == outputs[1][0].rejected_samples
    assert outputs[0][1].tobytes() == outputs[1][1].tobytes()


def _integrate_maps(tmp_path: Path, stem: str, paths: list[Path], **overrides):
    maps = IntegrationMapPaths(
        tmp_path / f"{stem}-accepted.fits",
        tmp_path / f"{stem}-coverage.fits",
        tmp_path / f"{stem}-rejected.fits",
    )
    result = integrate_expressions(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / f"{stem}.fits",
        parameters=IntegrationParameters(max_statistics_samples=5000, **overrides),
        map_paths=maps,
    )
    return result, fits.getdata(result.output_path), fits.getdata(maps.rejection_count)


@pytest.mark.parametrize("frames", [3, 6])
def test_scale_model_is_monotone_on_awkward_inputs(tmp_path: Path, frames: int) -> None:
    """Coverage borders, a frame with an unmodelled gradient, a noisy frame and
    hot pixels: v2 never rejects a sample v1 kept, and strong outliers go."""

    rng = np.random.default_rng(frames)
    shape = (96, 160)
    y, x = np.indices(shape, dtype=np.float64)
    paths = []
    for index in range(frames):
        data = 200.0 + rng.normal(0.0, 6.0 if index != 1 else 11.0, shape)
        if index == 0:
            data += 15.0 * x / shape[1]  # residual gradient nobody removed
        data = data.astype(np.float32)
        data[:, : 4 + 3 * index] = np.nan  # dithered coverage border
        data[10 + index, 40 + 2 * index] += np.float32(120.0)  # 20 sigma hit
        path = tmp_path / f"awk-{index}.fits"
        fits.writeto(path, data)
        paths.append(path)
    v1, m1, r1 = _integrate_maps(
        tmp_path, "v1", paths, rejection_pool_half_width=0, rejection_frame_noise_scaling=False
    )
    v2, m2, r2 = _integrate_maps(tmp_path, "v2", paths)
    assert not np.any(r2 > r1)
    assert v2.rejected_samples <= v1.rejected_samples
    for index in range(frames):
        # Every hit the per-pixel rule caught is still caught (at N = 3 a
        # three-sample MAD misses some hits under either rule).
        assert r2[10 + index, 40 + 2 * index] == r1[10 + index, 40 + 2 * index]
        if frames >= 6:
            assert r2[10 + index, 40 + 2 * index] >= 1
    assert np.array_equal(np.isfinite(m1), np.isfinite(m2))
    assert v2.execution["rejectionMask"]["method"] == "median-pooled-mad-sigma-v2"
