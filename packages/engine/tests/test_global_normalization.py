from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from ufwbpp.calibration import (
    CalibrationError,
    FrameExpression,
    write_expression,
)
from ufwbpp.global_normalization import (
    GlobalNormalizationParameters,
    StellarScaleHint,
    fit_registered_group_global_normalization,
)


def _write(path: Path, values: np.ndarray) -> Path:
    fits.writeto(path, np.asarray(values, dtype=np.float32), overwrite=False)
    return path


def _scene(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    values = (
        900.0
        + 0.18 * x
        - 0.11 * y
        + 34.0 * np.sin(x / 31.0)
        + 28.0 * np.cos(y / 37.0)
    )
    # Broad astrophysical structure is deliberately not a compact-star-only
    # oracle. A global affine correction must preserve its morphology exactly.
    values += 180.0 * np.exp(
        -((x - 142.0) ** 2 + (y - 119.0) ** 2) / (2.0 * 61.0**2)
    )
    for center_x, center_y, amplitude in ((42, 51, 1200), (190, 80, 900), (95, 205, 1500)):
        values += amplitude * np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * 2.0**2)
        )
    return values.astype(np.float32)


def test_global_affine_fit_excludes_bright_outlier_and_preserves_extended_scene(
    tmp_path: Path,
) -> None:
    reference = _scene()
    expected_scale = 1.18
    expected_offset = 23.0
    target = ((reference - expected_offset) / expected_scale).astype(np.float32)
    target[24:29, 210:215] += 50_000.0
    paths = (
        _write(tmp_path / "reference.fits", reference),
        _write(tmp_path / "target.fits", target),
    )

    result = fit_registered_group_global_normalization(
        paths,
        reference_index=0,
        parameters=GlobalNormalizationParameters(minimum_samples=4_096),
        stellar_scale_hints=(
            StellarScaleHint(
                str(paths[0]),
                str(paths[0]),
                "R",
                "a" * 64,
                "a" * 64,
                1.0,
                "REFERENCE_IDENTITY",
                {},
            ),
            StellarScaleHint(
                str(paths[1]),
                str(paths[0]),
                "R",
                "b" * 64,
                "a" * 64,
                expected_scale,
                "STELLAR_SCALE_ACCEPTED",
                {"acceptedScaleStars": 40},
            ),
        ),
    )

    coefficient = result.coefficients[1]
    assert coefficient.mode == "STELLAR_SCALE_SCALAR_OFFSET"
    assert coefficient.scale == pytest.approx(expected_scale, rel=2e-4)
    assert coefficient.offset == pytest.approx(expected_offset, abs=0.2)
    corrected = target.astype(np.float64) * coefficient.scale + coefficient.offset
    retained = np.ones(reference.shape, dtype=bool)
    retained[24:29, 210:215] = False
    np.testing.assert_allclose(corrected[retained], reference[retained], rtol=2e-4, atol=0.2)
    # A global affine model cannot flatten the broad target's spatial contrast.
    before_contrast = float(reference[119, 142] - reference[20, 20])
    after_contrast = float(corrected[119, 142] - corrected[20, 20])
    assert after_contrast == pytest.approx(before_contrast, rel=2e-4, abs=0.2)
    assert result.receipt["pixInsightEquivalent"] is False
    assert result.receipt["scienceGuard"]["multiplicativeScaleSpatiallyConstant"] is True


def test_unidentifiable_scale_falls_back_to_audited_offset_only(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260905)
    reference = rng.normal(1000.0, 3.0, (192, 192)).astype(np.float32)
    target = rng.normal(1042.0, 3.0, (192, 192)).astype(np.float32)
    paths = (
        _write(tmp_path / "reference.fits", reference),
        _write(tmp_path / "target.fits", target),
    )

    result = fit_registered_group_global_normalization(
        paths,
        reference_index=0,
        parameters=GlobalNormalizationParameters(),
    )

    coefficient = result.coefficients[1]
    assert coefficient.mode == "UNIT_SCALE_SCALAR_OFFSET_STELLAR_UNAVAILABLE"
    assert coefficient.scale == 1.0
    assert coefficient.evidence["fallbackReason"] == "STELLAR_SCALE_UNAVAILABLE_OR_UNSAFE"
    assert coefficient.offset == pytest.approx(-42.0, abs=0.2)


def test_unidentifiable_scale_fails_closed_when_fallback_is_forbidden(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    paths = (
        _write(tmp_path / "reference.fits", rng.normal(size=(192, 192))),
        _write(tmp_path / "target.fits", rng.normal(size=(192, 192))),
    )
    with pytest.raises(CalibrationError) as captured:
        fit_registered_group_global_normalization(
            paths,
            reference_index=0,
            parameters=GlobalNormalizationParameters(allow_offset_only_fallback=False),
        )
    assert captured.value.code == "GLOBAL_NORMALIZATION_STELLAR_SCALE_UNAVAILABLE"


def test_stellar_scale_hint_must_bind_exact_registered_source_and_reference(
    tmp_path: Path,
) -> None:
    reference = _write(tmp_path / "reference.fits", _scene((128, 128)))
    target = _write(tmp_path / "target.fits", _scene((128, 128)))
    wrong = _write(tmp_path / "wrong.fits", _scene((128, 128)))
    with pytest.raises(CalibrationError) as captured:
        fit_registered_group_global_normalization(
            (reference, target),
            reference_index=0,
            parameters=GlobalNormalizationParameters(
                maximum_samples=16_384,
                minimum_samples=1_024,
            ),
            stellar_scale_hints=(
                None,
                StellarScaleHint(
                    str(target),
                    str(wrong),
                    "R",
                    "b" * 64,
                    "c" * 64,
                    1.0,
                    "STELLAR_SCALE_ACCEPTED",
                    {},
                ),
            ),
        )
    assert captured.value.code == "GLOBAL_NORMALIZATION_HINT_IDENTITY_MISMATCH"


def test_smoothed_additive_grid_restores_gradient_and_preserves_broad_nebula(
    tmp_path: Path,
) -> None:
    shape = (1024, 1024)
    y, x = np.indices(shape, dtype=np.float64)
    reference = _scene(shape).astype(np.float64)
    reference += 210.0 * np.exp(
        -((x - 560.0) ** 2 + (y - 510.0) ** 2) / (2.0 * 260.0**2)
    )
    reference = reference.astype(np.float32)
    stellar_scale = 1.12
    additive = 24.0 + 120.0 * (x / (shape[1] - 1)) - 100.0 * (y / (shape[0] - 1))
    target = ((reference - additive) / stellar_scale).astype(np.float32)
    target[70:78, 800:808] += 30_000.0
    reference_path = _write(tmp_path / "gradient-reference.fits", reference)
    target_path = _write(tmp_path / "gradient-target.fits", target)
    hints = (
        StellarScaleHint(
            str(reference_path),
            str(reference_path),
            "R",
            "a" * 64,
            "a" * 64,
            1.0,
            "REFERENCE_IDENTITY",
            {},
        ),
        StellarScaleHint(
            str(target_path),
            str(reference_path),
            "R",
            "b" * 64,
            "a" * 64,
            stellar_scale,
            "STELLAR_SCALE_ACCEPTED",
            {"acceptedScaleStars": 64},
        ),
    )

    result = fit_registered_group_global_normalization(
        (reference_path, target_path),
        reference_index=0,
        stellar_scale_hints=hints,
        parameters=GlobalNormalizationParameters(
            offset_tile_size_pixels=64,
            offset_smoothing_sigma_nodes=3.0,
        ),
    )
    coefficient = result.coefficients[1]
    assert coefficient.mode == "STELLAR_SCALE_ADDITIVE_GRID"
    additive_evidence = coefficient.evidence["additiveModel"]
    assert additive_evidence["effectiveSmoothingSigmaPixels"] == 192.0
    defaults = GlobalNormalizationParameters()
    assert defaults.offset_tile_size_pixels * defaults.offset_smoothing_sigma_nodes == 896.0
    assert additive_evidence["checkerboardHoldout"]["holdoutTiles"] >= 4
    assert additive_evidence["sha256"].startswith("sha256:")
    assert coefficient.serializable()["offsetGrid"]["sha256"] == additive_evidence["sha256"]

    corrected_path = tmp_path / "corrected.fits"
    write_expression(
        FrameExpression(
            str(target_path),
            scale=coefficient.scale,
            offset=coefficient.offset,
            offset_grid=coefficient.offset_grid,
            offset_grid_x=coefficient.offset_grid_x,
            offset_grid_y=coefficient.offset_grid_y,
        ),
        corrected_path,
        max_memory_bytes=2 * 1024 * 1024,
    )
    with fits.open(corrected_path, memmap=False) as hdul:
        corrected = np.asarray(hdul[0].data, dtype=np.float64)
    retained = np.ones(shape, dtype=bool)
    retained[70:78, 800:808] = False
    before_rmse = float(
        np.sqrt(np.mean((target[retained] * stellar_scale - reference[retained]) ** 2))
    )
    after_rmse = float(np.sqrt(np.mean((corrected[retained] - reference[retained]) ** 2)))
    assert after_rmse < before_rmse * 0.35
    before_contrast = float(reference[520, 520] - reference[40, 40])
    after_contrast = float(corrected[520, 520] - corrected[40, 40])
    assert after_contrast == pytest.approx(before_contrast, rel=0.03, abs=2.0)


def test_additive_grid_application_is_bit_exact_across_row_tile_sizes(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "source.fits", _scene((256, 320)))
    grid = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    expression = FrameExpression(
        str(source),
        scale=1.03,
        offset_grid=grid,
        offset_grid_x=(31.5, 159.5, 287.5),
        offset_grid_y=(31.5, 127.5, 223.5),
    )
    small = tmp_path / "small-tiles.fits"
    large = tmp_path / "large-tiles.fits"
    write_expression(expression, small, max_memory_bytes=320 * 20 * 7)
    write_expression(expression, large, max_memory_bytes=320 * 20 * 512)
    with fits.open(small, memmap=False) as left, fits.open(large, memmap=False) as right:
        np.testing.assert_array_equal(left[0].data, right[0].data)


def test_excessive_additive_grid_keeps_safe_scalar_model_and_records_rejection(
    tmp_path: Path,
) -> None:
    shape = (1024, 1024)
    y, x = np.indices(shape, dtype=np.float64)
    reference = _scene(shape)
    stellar_scale = 1.12
    additive = 24.0 + 700.0 * x / (shape[1] - 1) - 550.0 * y / (shape[0] - 1)
    target = ((reference - additive) / stellar_scale).astype(np.float32)
    paths = (
        _write(tmp_path / "reference.fits", reference),
        _write(tmp_path / "target.fits", target),
    )
    parameters = GlobalNormalizationParameters(
        offset_tile_size_pixels=64,
        offset_smoothing_sigma_nodes=3.0,
    )
    result = fit_registered_group_global_normalization(
        paths,
        reference_index=0,
        parameters=parameters,
        stellar_scale_hints=(None, StellarScaleHint(
            str(paths[1]), str(paths[0]), "G", "b" * 64, "a" * 64,
            stellar_scale, "STELLAR_SCALE_ACCEPTED", {"acceptedScaleStars": 64},
        )),
    )
    coefficient = result.coefficients[1]
    assert coefficient.mode == "STELLAR_SCALE_SCALAR_OFFSET"
    assert coefficient.scale == stellar_scale
    assert np.isfinite(coefficient.offset)
    assert coefficient.offset_grid == coefficient.offset_grid_x == coefficient.offset_grid_y == ()
    assert coefficient.serializable()["offsetGrid"] is None
    evidence = result.receipt["frames"][1]["evidence"]["additiveModel"]
    assert evidence["status"] == "FALLBACK"
    assert evidence["reasonCode"] == "GLOBAL_NORMALIZATION_OFFSET_GRID_SPAN_UNSAFE"
    rejected = evidence["rejectedGrid"]
    assert rejected["status"] == "REJECTED"
    assert rejected["applied"] is False
    limits = rejected["limits"]
    assert limits == {
        "maximumOffsetP05P95Fraction": 0.25,
        "maximumOffsetSpanFraction": 0.35,
        "maximumOffsetNeighborSigmaFraction": 0.002,
    }
    assert any((
        rejected["offsetP05P95Fraction"] > limits["maximumOffsetP05P95Fraction"],
        rejected["offsetSpanFraction"] > limits["maximumOffsetSpanFraction"],
        rejected["neighborDifferenceSigmaFraction"] > limits["maximumOffsetNeighborSigmaFraction"],
    ))
    for ratio, label in (("offsetP05P95Fraction", "p05-p95"), ("offsetSpanFraction", "span"), ("neighborDifferenceSigmaFraction", "neighbor-sigma")):
        assert f"{label}={rejected[ratio]:.6g}" in evidence["reason"]
    assert rejected["offsetSpanFraction"] == pytest.approx(rejected["offsetSpan"] / rejected["referenceSky"])
    corrected_path = tmp_path / "scalar-corrected.fits"
    write_expression(FrameExpression(str(paths[1]), scale=coefficient.scale, offset=coefficient.offset), corrected_path)
    with fits.open(corrected_path, memmap=False) as hdul:
        corrected = np.asarray(hdul[0].data, dtype=np.float64)
    assert np.isfinite(corrected).all()
    expected = target.astype(np.float64) * stellar_scale + coefficient.offset
    np.testing.assert_allclose(corrected, expected, rtol=1e-6)
    # Rejected spatial correction cannot silently flatten the input scene.
    assert corrected[520, 520] - corrected[40, 40] == pytest.approx(
        stellar_scale * (float(target[520, 520]) - float(target[40, 40])), abs=2e-4,
    )


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 3.0])
def test_unsafe_stellar_scale_still_fails_when_offset_only_is_forbidden(
    tmp_path: Path, scale: float,
) -> None:
    paths = (_write(tmp_path / "reference.fits", _scene()), _write(tmp_path / "target.fits", _scene()))
    with pytest.raises(CalibrationError) as captured:
        fit_registered_group_global_normalization(
            paths,
            reference_index=0,
            parameters=GlobalNormalizationParameters(allow_offset_only_fallback=False),
            stellar_scale_hints=(None, StellarScaleHint(
                str(paths[1]), str(paths[0]), "G", "b" * 64, "a" * 64,
                scale, "STELLAR_SCALE_ACCEPTED", {},
            )),
        )
    assert captured.value.code == "GLOBAL_NORMALIZATION_STELLAR_SCALE_UNAVAILABLE"


def test_insufficient_global_samples_do_not_use_grid_rejection_fallback(
    tmp_path: Path,
) -> None:
    paths = (
        _write(tmp_path / "reference.fits", _scene((64, 64))),
        _write(tmp_path / "target.fits", _scene((64, 64)) + 42.0),
    )
    with pytest.raises(CalibrationError) as captured:
        fit_registered_group_global_normalization(paths, reference_index=0)
    assert captured.value.code == "GLOBAL_NORMALIZATION_SAMPLES_INSUFFICIENT"


def _sky_response_group(
    tmp_path: Path, *, sky_levels: tuple[float, ...], response_depth: float = 0.03
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Frames sharing an object and a sky-proportional edge roll-off."""

    shape = (1024, 640)
    y, x = np.indices(shape, dtype=np.float64)
    # Residual flat error: a roll-off over the bottom 12% and a mild
    # left-right slope, both proportional to the sky.
    response = -response_depth * np.clip((y - 0.88 * shape[0]) / (0.12 * shape[0]), 0.0, 1.0)
    response += 0.004 * (x / (shape[1] - 1) - 0.5)
    galaxy = 120.0 * np.exp(-((x - 300.0) ** 2 + (y - 400.0) ** 2) / (2.0 * 45.0**2))
    rng = np.random.default_rng(7)
    stars = np.zeros(shape)
    for _ in range(120):
        sx, sy = rng.uniform(8, shape[1] - 8), rng.uniform(8, shape[0] - 8)
        stars += rng.uniform(300, 4000) * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * 1.4**2))
    paths = []
    for index, sky in enumerate(sky_levels):
        frame = (sky * (1.0 + response) + galaxy + stars + rng.normal(0.0, 4.0, shape)).astype(np.float32)
        paths.append(_write(tmp_path / f"frame-{index}.fits", frame))
    return paths, response, galaxy


def test_sky_proportional_response_is_removed_from_every_frame(tmp_path: Path) -> None:
    skies = (720.0, 680.0, 640.0, 590.0, 540.0, 490.0, 440.0, 390.0, 350.0, 320.0)
    paths, response, _galaxy = _sky_response_group(tmp_path, sky_levels=skies)
    reference_index = len(skies) - 1  # lowest sky, as registration chooses it
    parameters = GlobalNormalizationParameters(
        offset_tile_size_pixels=64, offset_smoothing_sigma_nodes=3.0
    )

    result = fit_registered_group_global_normalization(
        paths, reference_index=reference_index, parameters=parameters, workers=2
    )

    evidence = result.receipt["skyResponse"]
    assert evidence["status"] == "APPLIED"
    assert evidence["skyRatio"] == pytest.approx(720.0 / 320.0, rel=0.05)
    assert evidence["halvesCorrelation"] > 0.85
    rows = evidence["rowMedianFractions"]
    # The roll-off occupies the bottom quarter; the response keeps only what
    # the additive grid's smoothing cannot follow, so compare the last row
    # band with the middle of the frame.
    assert rows[-1] < rows[len(rows) // 2] - 0.008
    assert result.coefficients[reference_index].mode.startswith("REFERENCE_IDENTITY+SKY_RESPONSE")
    assert all("+SKY_RESPONSE" in item.mode for item in result.coefficients)

    # Integrate (plain mean) with and without the coefficients and compare
    # the bottom band of the master against its centre.
    shape = (1024, 640)
    plain = np.zeros(shape)
    corrected = np.zeros(shape)
    for path, coefficient in zip(paths, result.coefficients, strict=True):
        out = tmp_path / f"{path.stem}-normalized.fits"
        write_expression(
            FrameExpression(
                str(path),
                scale=coefficient.scale,
                offset=coefficient.offset,
                offset_grid=coefficient.offset_grid,
                offset_grid_x=coefficient.offset_grid_x,
                offset_grid_y=coefficient.offset_grid_y,
            ),
            out,
            max_memory_bytes=4 * 1024 * 1024,
        )
        with fits.open(out, memmap=False) as hdul:
            corrected += np.asarray(hdul[0].data, dtype=np.float64)
        with fits.open(path, memmap=False) as hdul:
            plain += np.asarray(hdul[0].data, dtype=np.float64)
    plain /= len(paths)
    corrected /= len(paths)
    band = slice(int(0.93 * shape[0]), int(0.99 * shape[0]))
    centre = slice(int(0.45 * shape[0]), int(0.55 * shape[0]))
    columns = slice(0, 200)  # away from the galaxy
    plain_band = np.median(plain[band, columns]) - np.median(plain[centre, columns])
    corrected_band = np.median(corrected[band, columns]) - np.median(corrected[centre, columns])
    assert plain_band < -6.0  # the mean of the frames keeps the roll-off
    # The sky-proportional part of the roll-off is gone; what remains is the
    # low-order part left to the additive grid and the reference frame.
    assert abs(corrected_band) < 0.4 * abs(plain_band)
    # The object is untouched: the galaxy peak above its surroundings agrees.
    plain_peak = plain[400, 300] - np.median(plain[350:370, 300:320])
    corrected_peak = corrected[400, 300] - np.median(corrected[350:370, 300:320])
    assert corrected_peak == pytest.approx(plain_peak, rel=0.05, abs=3.0)


def test_sky_response_needs_enough_sky_variation(tmp_path: Path) -> None:
    skies = (500.0, 505.0, 495.0, 502.0, 498.0, 501.0)
    paths, _response, _galaxy = _sky_response_group(tmp_path, sky_levels=skies)
    result = fit_registered_group_global_normalization(
        paths,
        reference_index=0,
        parameters=GlobalNormalizationParameters(offset_tile_size_pixels=64),
    )
    assert result.receipt["skyResponse"]["status"] == "NOT_APPLICABLE"
    assert result.coefficients[0].mode.startswith("REFERENCE_IDENTITY")
    assert all("SKY_RESPONSE" not in item.mode for item in result.coefficients)
    # Six frames still receive the group's flattest low-order target.
    assert result.receipt["lowOrderTarget"]["status"] == "APPLIED"


def _tilted_levels(gradients: list[tuple[float, float]], sky: float = 500.0) -> np.ndarray:
    """Tile-level maps of frames whose backgrounds are planes with the given
    (x, y) spans in ADU across the frame, plus a common roll-off at the edges."""

    y, x = np.indices((17, 25), dtype=np.float64)
    x = x / 24.0 - 0.5
    y = y / 16.0 - 0.5
    roll_off = -3.0 * np.clip(np.hypot(x, y) - 0.4, 0.0, None)
    return np.stack([sky + gx * x + gy * y + roll_off for gx, gy in gradients])


def test_low_order_target_cancels_opposing_night_gradients() -> None:
    from ufwbpp.global_normalization import _fit_low_order_target

    # Two nights, one of them observed after a meridian flip: opposite tilts.
    gradients = [(-3.0, -0.6)] * 4 + [(-2.4, 0.6)] * 3 + [(2.6, -0.4)] * 3
    levels = _tilted_levels(gradients)
    skies = np.median(levels.reshape(len(gradients), -1), axis=1)
    result = _fit_low_order_target(levels, skies, [1.0] * len(gradients), 0, 6)
    evidence = result.evidence
    assert evidence["status"] == "APPLIED"
    assert evidence["rule"] == "flattest-convex-combination-of-frame-tilts-v1"
    # A mix of both nights is flatter than any single frame and than the mean.
    assert evidence["targetPlaneAmplitude"] < 0.15
    assert evidence["flattestFramePlaneAmplitude"] > 2.0
    assert evidence["groupMeanPlaneAmplitude"] > 0.5
    assert all(index in range(len(gradients)) for index in evidence["supportFrames"])
    assert sum(evidence["weights"]) == pytest.approx(1.0, abs=1e-3)
    # The correction removes the reference's own tilt: reference plus
    # correction is flat to first order, while the common roll-off is not
    # touched (it is not a plane).
    y, x = np.indices(levels.shape[1:], dtype=np.float64)
    x = x / 24.0 - 0.5
    y = y / 16.0 - 0.5
    corrected = levels[0] + result.correction
    design = np.column_stack((np.ones(x.size), x.ravel(), y.ravel()))
    coefficients, *_ = np.linalg.lstsq(design, corrected.ravel(), rcond=None)
    assert abs(coefficients[1]) < 0.2 and abs(coefficients[2]) < 0.2


def test_low_order_target_keeps_a_single_night_common_tilt() -> None:
    from ufwbpp.global_normalization import _fit_low_order_target

    gradients = [(-3.0, -0.6), (-2.8, -0.5), (-3.1, -0.7), (-2.9, -0.4), (-3.0, -0.6), (-2.7, -0.5)]
    levels = _tilted_levels(gradients)
    skies = np.median(levels.reshape(len(gradients), -1), axis=1)
    result = _fit_low_order_target(levels, skies, [1.0] * len(gradients), 2, 6)
    evidence = result.evidence
    assert evidence["status"] == "APPLIED"
    # All frames tilt the same way: the flattest mix is essentially the
    # flattest frame; the master keeps the night's tilt and is not flattened.
    assert evidence["targetPlaneAmplitude"] >= evidence["flattestFramePlaneAmplitude"] - 1e-6
    assert evidence["targetPlaneAmplitude"] > 2.5
    # The correction only moves the reference to the flattest frame of the night.
    assert evidence["correctionAmplitude"] < 1.0


def _flipped_sky_response_group(
    tmp_path: Path, *, sky_levels: tuple[float, ...], flipped: tuple[bool, ...]
) -> tuple[list[Path], list[np.ndarray]]:
    """Registered frames of two nights sharing a sensor-fixed roll-off; the
    second night was taken after a meridian flip, so its registered frames
    are the sensor image rotated by a half-turn and the roll-off sits on the
    opposite edge of the registered geometry."""

    shape = (1024, 640)
    y, x = np.indices(shape, dtype=np.float64)
    # Roll-off over the bottom 12% of the sensor, as a flat-field residual.
    response = -0.03 * np.clip((y - 0.88 * shape[0]) / (0.12 * shape[0]), 0.0, 1.0)
    galaxy = 120.0 * np.exp(-((x - 300.0) ** 2 + (y - 400.0) ** 2) / (2.0 * 45.0**2))
    rng = np.random.default_rng(11)
    stars = np.zeros(shape)
    for _ in range(120):
        sx, sy = rng.uniform(8, shape[1] - 8), rng.uniform(8, shape[0] - 8)
        stars += rng.uniform(300, 4000) * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * 1.4**2))
    half_turn = np.array(
        [[-1.0, 0.0, shape[1] - 1.0], [0.0, -1.0, shape[0] - 1.0], [0.0, 0.0, 1.0]]
    )
    paths, transforms = [], []
    for index, (sky, flip) in enumerate(zip(sky_levels, flipped, strict=True)):
        # The object is fixed on the sky; the flat error is fixed on the sensor.
        sensor_response = response[::-1, ::-1] if flip else response
        frame = sky * (1.0 + sensor_response) + galaxy + stars + rng.normal(0.0, 4.0, shape)
        paths.append(_write(tmp_path / f"frame-{index}.fits", frame.astype(np.float32)))
        transforms.append(half_turn if flip else np.eye(3))
    return paths, transforms


def test_sky_response_is_regressed_in_the_sensor_frame_across_a_meridian_flip(tmp_path: Path) -> None:
    skies = (720.0, 660.0, 600.0, 540.0, 480.0, 430.0, 390.0, 350.0, 320.0, 300.0)
    flipped = (False, True, False, True, False, True, False, True, False, True)
    paths, transforms = _flipped_sky_response_group(tmp_path, sky_levels=skies, flipped=flipped)
    parameters = GlobalNormalizationParameters(offset_tile_size_pixels=64, offset_smoothing_sigma_nodes=3.0)

    # In registered coordinates the roll-off changes edges from frame to
    # frame; the halves of the group do not agree and nothing is applied.
    plain = fit_registered_group_global_normalization(
        paths, reference_index=len(skies) - 1, parameters=parameters, workers=2
    )
    assert plain.receipt["skyResponse"]["status"] != "APPLIED"

    result = fit_registered_group_global_normalization(
        paths, reference_index=len(skies) - 1, parameters=parameters, workers=2, transforms=transforms
    )
    evidence = result.receipt["skyResponse"]
    assert evidence["status"] == "APPLIED", evidence
    assert evidence["regressionFrame"] == "sensor"
    assert evidence["halvesCorrelation"] > 0.85
    rows = evidence["rowMedianFractions"]  # sensor frame: roll-off at the bottom only
    assert rows[-1] < rows[len(rows) // 2] - 0.008
    assert abs(rows[0] - rows[len(rows) // 2]) < 0.5 * abs(rows[-1] - rows[len(rows) // 2])

    shape = (1024, 640)
    corrected = np.zeros(shape)
    plain_mean = np.zeros(shape)
    for path, coefficient in zip(paths, result.coefficients, strict=True):
        out = tmp_path / f"{path.stem}-normalized.fits"
        write_expression(
            FrameExpression(
                str(path),
                scale=coefficient.scale,
                offset=coefficient.offset,
                offset_grid=coefficient.offset_grid,
                offset_grid_x=coefficient.offset_grid_x,
                offset_grid_y=coefficient.offset_grid_y,
            ),
            out,
            max_memory_bytes=4 * 1024 * 1024,
        )
        with fits.open(out, memmap=False) as hdul:
            corrected += np.asarray(hdul[0].data, dtype=np.float64)
        with fits.open(path, memmap=False) as hdul:
            plain_mean += np.asarray(hdul[0].data, dtype=np.float64)
    corrected /= len(paths)
    plain_mean /= len(paths)
    columns = slice(0, 200)
    band = slice(int(0.93 * shape[0]), int(0.99 * shape[0]))
    centre = np.median(corrected[int(0.45 * shape[0]) : int(0.55 * shape[0]), columns])
    top = np.median(corrected[int(0.01 * shape[0]) : int(0.07 * shape[0]), columns])
    bottom = np.median(corrected[band, columns])
    plain_centre = np.median(plain_mean[int(0.45 * shape[0]) : int(0.55 * shape[0]), columns])
    plain_bottom = np.median(plain_mean[band, columns])
    # A plain mean keeps half of the roll-off on each edge; the corrected
    # master is flat on both edges.
    assert plain_bottom - plain_centre < -4.0
    assert abs(bottom - centre) < 0.35 * abs(plain_bottom - plain_centre)
    assert abs(top - centre) < 0.35 * abs(plain_bottom - plain_centre)
