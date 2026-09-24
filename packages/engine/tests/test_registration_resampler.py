from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from ufwbpp.calibration import (
    CalibrationError, FitsFrame, FrameInfo, read_frame_info,
)
from ufwbpp import pixel_pipeline as pipeline
from ufwbpp.pixel_pipeline import AffineTransform


def _write_declared_frame(
    path: Path, pixels: np.ndarray, *, unit_scale: float = 1.0
) -> Path:
    header = fits.Header()
    header["IMAGETYP"] = "Calibrated Light"
    header["OAFNDOM"] = "NORMALIZED_TEST"
    header["OAFNSCL"] = unit_scale
    fits.writeto(path, np.asarray(pixels, dtype=np.float32), header, overwrite=False)
    return path


def _coordinates(
    shape: tuple[int, int], *, dx: float, dy: float, margin: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    y, x = np.mgrid[margin : height - margin, margin : width - margin]
    return np.asarray(x - dx, dtype=np.float64), np.asarray(y - dy, dtype=np.float64)


def test_lanczos3_preserves_constant_field_and_is_tile_invariant(
    tmp_path: Path,
) -> None:
    pixels = np.full((48, 64), 0.125, dtype=np.float32)
    source = _write_declared_frame(tmp_path / "constant.fits", pixels)
    x, y = _coordinates(pixels.shape, dx=0.37, dy=0.61)

    with FitsFrame(source) as frame:
        complete = frame.sample_lanczos3_clamped(x, y)
        split = np.concatenate(
            (
                frame.sample_lanczos3_clamped(x[:13], y[:13]),
                frame.sample_lanczos3_clamped(x[13:], y[13:]),
            ),
            axis=0,
        )

    np.testing.assert_array_equal(split, complete)
    np.testing.assert_allclose(complete, 0.125, rtol=0.0, atol=2.0e-7)


def test_lanczos3_preserves_gaussian_flux_and_psf_without_bilinear_blur(
    tmp_path: Path,
) -> None:
    size = 65
    y, x = np.indices((size, size), dtype=np.float64)
    sigma = 1.3
    source_pixels = np.exp(
        -((x - 32.0) ** 2 + (y - 32.0) ** 2) / (2.0 * sigma**2)
    ).astype(np.float32)
    source = _write_declared_frame(tmp_path / "gaussian.fits", source_pixels)
    sample_x, sample_y = _coordinates(source_pixels.shape, dx=0.37, dy=0.49)
    truth = np.exp(
        -(
            (sample_x - 32.0) ** 2
            + (sample_y - 32.0) ** 2
        )
        / (2.0 * sigma**2)
    )

    with FitsFrame(source) as frame:
        sampled = frame.sample_lanczos3_clamped(sample_x, sample_y)

    normalized_rmse = float(
        np.sqrt(np.mean(np.square(sampled - truth))) / np.std(truth)
    )
    relative_flux_error = abs(float(np.sum(sampled) / np.sum(truth) - 1.0))
    assert normalized_rmse < 0.012
    assert relative_flux_error < 2.0e-5

    def second_moment(image: np.ndarray) -> float:
        weights = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
        total = float(np.sum(weights))
        center_x = float(np.sum(weights * sample_x) / total)
        center_y = float(np.sum(weights * sample_y) / total)
        return float(
            np.sum(
                weights
                * ((sample_x - center_x) ** 2 + (sample_y - center_y) ** 2)
            )
            / total
        )

    assert second_moment(sampled) / second_moment(truth) < 1.01


def test_lanczos3_retains_mid_high_frequency_response(tmp_path: Path) -> None:
    height, width = 16, 1024
    frequency = 0.3
    source_x = np.arange(width, dtype=np.float64)
    pixels = np.broadcast_to(
        0.5 + 0.2 * np.sin(2.0 * np.pi * frequency * source_x),
        (height, width),
    ).astype(np.float32)
    source = _write_declared_frame(tmp_path / "sinusoid.fits", pixels)
    y, output_x = np.mgrid[5:10, 10:1010]
    sample_x = np.asarray(output_x + 0.5, dtype=np.float64)
    with FitsFrame(source) as frame:
        sampled = frame.sample_lanczos3_clamped(
            sample_x, np.asarray(y, dtype=np.float64)
        )

    phase = 2.0 * np.pi * frequency * sample_x.ravel()
    sine, cosine, _offset = np.linalg.lstsq(
        np.column_stack(
            (np.sin(phase), np.cos(phase), np.ones(phase.shape, dtype=np.float64))
        ),
        sampled.ravel(),
        rcond=None,
    )[0]
    recovered_amplitude = float(np.hypot(sine, cosine))
    assert recovered_amplitude / 0.2 > 0.95


def test_lanczos3_domain_clamp_suppresses_new_ringing_but_keeps_input_extrema(
    tmp_path: Path,
) -> None:
    nonnegative = np.zeros((33, 33), dtype=np.float32)
    nonnegative[16, 16] = 1.0
    nonnegative_path = _write_declared_frame(
        tmp_path / "nonnegative-impulse.fits", nonnegative
    )
    y, x = np.mgrid[7:26, 7:26]
    sample_x = np.asarray(x - 0.43, dtype=np.float64)
    sample_y = np.asarray(y - 0.37, dtype=np.float64)
    with FitsFrame(nonnegative_path) as frame:
        bounded = frame.sample_lanczos3_clamped(sample_x, sample_y)
    assert float(np.min(bounded)) >= 0.0
    assert float(np.max(bounded)) <= 1.0

    existing_negative = nonnegative.copy()
    existing_negative[16, 16] = -0.25
    negative_path = _write_declared_frame(
        tmp_path / "negative-impulse.fits", existing_negative
    )
    with FitsFrame(negative_path) as frame:
        preserved = frame.sample_lanczos3_clamped(sample_x, sample_y)
    assert float(np.min(preserved)) < 0.0
    assert float(np.min(preserved)) >= -0.25


def test_lanczos3_nan_coverage_ignores_zero_weight_but_not_active_support(
    tmp_path: Path,
) -> None:
    pixels = np.arange(24 * 24, dtype=np.float32).reshape(24, 24) / 600.0
    pixels[6, 6] = np.nan
    source = _write_declared_frame(tmp_path / "nan.fits", pixels)
    with FitsFrame(source) as frame:
        # At an integer coordinate all noncentral Lanczos coefficients are
        # exact zero, so the distant NaN cannot poison this exact sample.
        exact = frame.sample_lanczos3_clamped(
            np.asarray([[8.0]]), np.asarray([[8.0]])
        )
        active = frame.sample_lanczos3_clamped(
            np.asarray([[7.25]]), np.asarray([[7.25]])
        )

    assert exact[0, 0] == pixels[8, 8]
    assert np.isnan(active[0, 0])


def test_lanczos3_requires_full_support_and_a_declared_domain(tmp_path: Path) -> None:
    pixels = np.ones((16, 16), dtype=np.float32)
    declared = _write_declared_frame(tmp_path / "declared.fits", pixels)
    with FitsFrame(declared) as frame:
        sampled = frame.sample_lanczos3_clamped(
            np.asarray([[1.99, 2.0, 13.0, 13.01]]),
            np.asarray([[8.0, 8.0, 8.0, 8.0]]),
        )
    assert np.isnan(sampled[0, 0])
    assert sampled[0, 1] == 1.0
    assert sampled[0, 2] == 1.0
    assert np.isnan(sampled[0, 3])

    ambiguous = tmp_path / "ambiguous.fits"
    fits.writeto(ambiguous, pixels, overwrite=False)
    with FitsFrame(ambiguous) as frame:
        try:
            frame.sample_lanczos3_clamped(
                np.asarray([[8.0]]), np.asarray([[8.0]])
            )
        except ValueError as error:
            assert "declared finite numeric domain" in str(error)
        else:  # pragma: no cover - documents the fail-closed contract
            raise AssertionError("ambiguous numeric domain was accepted")


def test_lanczos3_mixed_integer_corners_and_fractional_support(tmp_path: Path) -> None:
    pixels = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) / 256.0
    pixels[2, 3] = np.nan
    source = _write_declared_frame(tmp_path / "mixed-corners.fits", pixels)
    with FitsFrame(source) as frame:
        sampled = frame.sample_lanczos3_clamped(
            np.asarray([[2.0, 13.0, 2.0, 13.0, 7.25, 13.01]]),
            np.asarray([[2.0, 2.0, 13.0, 13.0, 7.75, 7.0]]),
        )
    # Fractional coordinates activate all six taps in the same tile as the
    # integer corners, whose off-image taps and nearby NaN carry zero weight.
    np.testing.assert_array_equal(
        sampled[0, :4], pixels[[2, 2, 13, 13], [2, 13, 2, 13]]
    )
    assert np.isfinite(sampled[0, 4])
    assert np.isnan(sampled[0, 5])


def test_lanczos3_rotated_tiles_preserve_source_and_nonfinite_support(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7319)
    pixels = rng.normal(0.25, 0.2, (32, 48)).astype(np.float32)
    pixels[9, 13], pixels[17, 23], pixels[24, 37] = np.nan, np.inf, -np.inf
    source = _write_declared_frame(tmp_path / "rotated.fits", pixels)
    original_bytes = source.read_bytes()
    y, x = np.mgrid[3:29, 3:45].astype(np.float64)
    angle = np.deg2rad(179.56)
    sample_x = np.cos(angle) * x - np.sin(angle) * y + 46.7
    sample_y = np.sin(angle) * x + np.cos(angle) * y + 30.4
    with FitsFrame(source) as frame:
        complete = frame.sample_lanczos3_clamped(sample_x, sample_y)
        split = np.concatenate([
            frame.sample_lanczos3_clamped(sample_x[start:start + 5], sample_y[start:start + 5])
            for start in range(0, sample_x.shape[0], 5)
        ])
        # Inactive inf * 0 is discarded by the support mask in both kernels.
        with np.errstate(invalid="ignore"):
            nonfinite_support = frame.sample_lanczos3_clamped(
                np.asarray([[25.0, 24.25, 39.0, 38.25]]),
                np.asarray([[19.0, 18.25, 26.0, 25.25]]),
            )
    np.testing.assert_array_equal(split.view(np.uint32), complete.view(np.uint32))
    np.testing.assert_array_equal(nonfinite_support[0, [0, 2]], pixels[[19, 26], [25, 39]])
    assert np.all(np.isnan(nonfinite_support[0, [1, 3]]))
    assert source.read_bytes() == original_bytes


def _half_turn(shape: tuple[int, int], dx: int = 0, dy: int = 0) -> AffineTransform:
    height, width = shape
    return AffineTransform.from_value(
        ((-1, 0, width - 1 + dx), (0, -1, height - 1 + dy), (0, 0, 1))
    )


def _registration_info(path: Path) -> FrameInfo:
    # Isolated registration receives the already-resolved domain from the
    # calibration pipeline; an arbitrary FITS declaration alone is untrusted.
    return replace(
        read_frame_info(path), normalized_unit_scale=1.0,
        numeric_domain_authority="CONTENT_BOUND_OVERRIDE",
    )


@pytest.mark.parametrize("dx,dy", [(0, 0), (-2, 1), (2, -1), (30, 0)])
def test_exact_half_turn_preserves_pixels_and_integer_offset_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dx: int, dy: int
) -> None:
    pixels = (np.arange(8 * 11, dtype=np.float32).reshape(8, 11) - 25) / 16
    pixels[2, 3] = np.nan
    pixels[4, 6] = -0.0
    source = _write_declared_frame(tmp_path / "input.fits", pixels)
    destination = tmp_path / "registered.fits"
    info = _registration_info(source)
    transform = _half_turn(pixels.shape, dx, dy)
    read_intervals = []
    read_rows = FitsFrame.read_rows

    def read_chunk(frame, y0, y1):
        read_intervals.append((y0, y1))
        return read_rows(frame, y0, y1)

    def unexpected_interpolation(*_args):
        raise AssertionError("exact half-turn must bypass interpolation")

    monkeypatch.setattr(FitsFrame, "read_rows", read_chunk)
    monkeypatch.setattr(FitsFrame, "sample_lanczos3_clamped", unexpected_interpolation)
    monkeypatch.setattr(FitsFrame, "sample_bilinear", unexpected_interpolation)
    stats = pipeline._register_frame(
        source, destination, transform, info,
        max_memory_bytes=pixels.shape[1] * 32 * 2,
        resampler="lanczos-3-clamped", source_exposure_seconds=120.0,
    )
    expected = np.full_like(pixels, np.nan)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            sy, sx = pixels.shape[0] - 1 + dy - y, pixels.shape[1] - 1 + dx - x
            if 0 <= sy < pixels.shape[0] and 0 <= sx < pixels.shape[1]:
                expected[y, x] = pixels[sy, sx]
    actual = fits.getdata(destination)
    np.testing.assert_array_equal(actual, expected)
    # Includes the sign bit of zero, which a weighted sum need not preserve.
    np.testing.assert_array_equal(
        np.signbit(actual[np.isfinite(actual)]),
        np.signbit(expected[np.isfinite(expected)]),
    )
    assert all(
        0 <= y0 < y1 <= pixels.shape[0] and y1 - y0 <= 2
        for y0, y1 in read_intervals
    )
    assert stats.finite_pixels == int(np.count_nonzero(np.isfinite(expected)))
    assert stats.invalid_pixels == int(np.count_nonzero(~np.isfinite(expected)))
    header = fits.getheader(destination)
    assert header["OAFRSAMP"] == "HALF-TURN-EXACT"
    assert header["OAFREG"] == "AFFINE"
    assert header["OAFRMARG"] == 0
    assert header.get("OAFRCLMP") is None
    assert header["OAFNDOM"] == info.numeric_domain
    assert header["OAFSRCEX"] == 120.0
    assert pipeline._registration_provenance(
        transform, pixels.shape, "lanczos-3-clamped"
    ) == {
        "resampler": "half-turn-exact",
        "resamplerAlgorithm": "half-turn-integer-copy-v1",
    }
    if dx == 30:
        assert not read_intervals
        with pytest.raises(CalibrationError, match="AUTOCROP_EMPTY"):
            pipeline._common_valid_crop(
                pixels.shape, [transform], max_memory_bytes=4096,
                resampler="lanczos-3-clamped",
            )
    else:
        assert pipeline._common_valid_crop(
            pixels.shape, [AffineTransform.identity(), transform],
            max_memory_bytes=4096, resampler="lanczos-3-clamped",
        ) == (max(0, dy), max(0, dx), min(8, 8 + dy), min(11, 11 + dx))


@pytest.mark.parametrize("storage", ["scaled-int16", "uint32", "uint64"])
def test_exact_half_turn_copies_fits_physical_values(
    tmp_path: Path, storage: str
) -> None:
    source = tmp_path / "scaled.fits"
    header = fits.Header({
        "IMAGETYP": "Calibrated Light",
        "OAFNDOM": "NORMALIZED_TEST", "OAFNSCL": 1.0,
    })
    if storage == "scaled-int16":
        raw = np.arange(6 * 9, dtype=np.int16).reshape(6, 9)
        raw[2, 5] = -32768
        header["BLANK"] = -32768
        expected = (raw.astype(np.float64) * 0.125 - 2.0).astype(np.float32)
        expected[2, 5] = np.nan
    else:
        raw = np.arange(6 * 9, dtype=storage).reshape(6, 9)
        raw[4, 2] = np.iinfo(raw.dtype).max
        expected = raw.astype(np.float32)
    hdu = fits.PrimaryHDU(raw, header)
    if storage == "scaled-int16":
        hdu.header["BSCALE"], hdu.header["BZERO"] = 0.125, -2.0
    hdu.writeto(source)
    destination = tmp_path / "registered.fits"
    pipeline._register_frame(
        source, destination, _half_turn(raw.shape), _registration_info(source),
        max_memory_bytes=raw.shape[1] * 32,
        resampler="lanczos-3-clamped",
    )
    np.testing.assert_array_equal(fits.getdata(destination), expected[::-1, ::-1])


def test_half_turn_roundoff_is_bounded_over_the_whole_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = (8, 11)
    matrix = _half_turn(shape).validated_matrix()
    matrix[0, 1], matrix[1, 0] = -np.sin(np.pi), np.sin(np.pi)
    matrix[0, 2] = np.nextafter(matrix[0, 2], np.inf)
    transform = AffineTransform.from_value(matrix)
    assert pipeline._exact_half_turn_translation(transform, shape) == (10, 7)
    assert pipeline._common_valid_crop(
        shape, [transform], max_memory_bytes=4096, resampler="lanczos-3-clamped"
    ) == (0, 0, 8, 11)
    pixels = np.arange(88, dtype=np.float32).reshape(shape)
    source = _write_declared_frame(tmp_path / "roundoff.fits", pixels)

    def unexpected_interpolation(*_args):
        raise AssertionError("numerical roundoff should retain the integer map")

    monkeypatch.setattr(FitsFrame, "sample_lanczos3_clamped", unexpected_interpolation)
    destination = tmp_path / "registered.fits"
    pipeline._register_frame(
        source, destination, transform, _registration_info(source),
        max_memory_bytes=4096, resampler="lanczos-3-clamped",
    )
    np.testing.assert_array_equal(fits.getdata(destination), pixels[::-1, ::-1])
    # An almost zero off-axis coefficient still matters over a very long axis.
    long_shape = (4, 100_000)
    matrix = _half_turn(long_shape).validated_matrix()
    matrix[1, 0] = np.sin(np.pi)
    assert pipeline._exact_half_turn_translation(
        AffineTransform.from_value(matrix), long_shape
    ) is None


@pytest.mark.parametrize(
    "deviation", ["small-angle", "179.56-degrees", "subpixel", "tiny-dither", "shear"]
)
def test_noninteger_half_turn_uses_one_lanczos_warp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deviation: str
) -> None:
    shape = (16, 24)
    matrix = _half_turn(shape).validated_matrix()
    if deviation in {"small-angle", "179.56-degrees"}:
        angle = np.pi - (1e-10 if deviation == "small-angle" else np.deg2rad(0.44))
        matrix[:2, :2] = (
            (np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))
        )
    elif deviation in {"subpixel", "tiny-dither"}:
        matrix[0, 2] += 0.25 if deviation == "subpixel" else 1e-10
    else:
        matrix[0, 1] = 1e-10
    transform = AffineTransform.from_value(matrix)
    assert pipeline._exact_half_turn_translation(transform, shape) is None
    source = _write_declared_frame(tmp_path / "input.fits", np.full(shape, 0.125))
    destination = tmp_path / "registered.fits"
    sample = FitsFrame.sample_lanczos3_clamped
    observed = []

    def sample_once(frame, x, y):
        observed.append((x.shape, y.shape))
        return sample(frame, x, y)

    # This contract describes the NumPy reference resampler; the native kernel
    # is covered by its own differential tests.
    monkeypatch.setattr(pipeline, "load_native_kernels", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(FitsFrame, "sample_lanczos3_clamped", sample_once)
    execution: dict = {}
    pipeline._register_frame(
        source, destination, transform, _registration_info(source),
        max_memory_bytes=shape[0] * shape[1] * 192, resampler="lanczos-3-clamped",
        execution=execution,
    )
    assert observed == [(shape, shape)]
    assert execution["warpBackend"] == "numpy"
    assert fits.getheader(destination)["OAFRSAMP"] == "LANCZOS-3-CLAMPED"
    assert fits.getheader(destination)["OAFRMARG"] == 2
    assert pipeline._registration_provenance(
        transform, shape, "lanczos-3-clamped"
    )["resampler"] == "lanczos-3-clamped"


def test_exact_half_turn_worker_budget_includes_scaled_copy_buffers(tmp_path: Path) -> None:
    shape = (8, 11)
    source = _write_declared_frame(tmp_path / "input.fits", np.ones(shape))
    info = _registration_info(source)
    transform = _half_turn(shape)
    assert pipeline._registration_bytes_per_pixel(transform, "lanczos-3-clamped", shape) == 32
    with pytest.raises(CalibrationError, match="MEMORY_BUDGET_TOO_SMALL"):
        pipeline._register_frame(
            source, tmp_path / "too-small.fits", transform, info,
            max_memory_bytes=shape[1] * 32 - 1, resampler="lanczos-3-clamped",
        )
