from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine.calibration import (
    CalibrationError,
    FrameExpression,
    write_expression,
)
from openastroflow_engine.global_normalization import (
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
