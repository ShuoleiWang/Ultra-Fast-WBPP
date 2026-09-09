from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from skimage.transform import SimilarityTransform

from lightframeqc.registration import (
    RegistrationThresholds,
    match_star_catalogs,
    register_star_catalogs,
)


def _stars(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform((60.0, 50.0), (900.0, 700.0), size=(count, 2))


def test_registers_similarity_rotation_and_dither() -> None:
    source = _stars(1, 48)
    truth = SimilarityTransform(
        scale=1.004,
        rotation=np.deg2rad(3.5),
        translation=(18.0, -11.0),
    )
    rng = np.random.default_rng(2)
    reference = truth(source) + rng.normal(0.0, 0.10, size=source.shape)

    result = register_star_catalogs(source, reference)

    assert result.estimated
    assert result.accepted
    assert result.reason_codes == ()
    assert result.inlier_count >= 46
    assert result.inlier_ratio >= 0.95
    assert result.rms_px < 0.25
    np.testing.assert_allclose(result.apply(source), truth(source), atol=0.25)


@dataclass
class ObjectCatalog:
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray


def test_registers_180_degree_meridian_flip_with_dict_object_and_outliers() -> None:
    rng = np.random.default_rng(10)
    core_source = rng.uniform((40.0, 30.0), (520.0, 390.0), size=(42, 2))
    truth = SimilarityTransform(
        scale=0.997,
        rotation=np.pi,
        translation=(620.0, 480.0),
    )
    core_reference = truth(core_source) + rng.normal(0.0, 0.08, size=core_source.shape)

    source_outliers = rng.uniform((0.0, 0.0), (620.0, 480.0), size=(9, 2))
    reference_outliers = rng.uniform((0.0, 0.0), (620.0, 480.0), size=(13, 2))
    source_points = np.vstack((core_source, source_outliers))
    reference_points = np.vstack((core_reference, reference_outliers))

    # Flux makes the input contract explicit and ensures true stars are the
    # astroalign control points even when a catalog contains spurious detections.
    source_flux = np.concatenate((np.linspace(50_000.0, 10_000.0, 42), np.ones(9)))
    reference_flux = np.concatenate((np.linspace(49_000.0, 9_000.0, 42), np.ones(13)))
    source = ObjectCatalog(source_points[:, 0], source_points[:, 1], source_flux)
    reference = {"points": reference_points, "flux": reference_flux}

    result = register_star_catalogs(source, reference)

    assert result.accepted
    assert result.inlier_count >= 40
    assert result.inlier_ratio >= 0.75
    assert result.rms_px < 0.25
    np.testing.assert_allclose(result.apply(core_source), truth(core_source), atol=0.25)


def test_matching_is_one_to_one_and_preserves_input_indices() -> None:
    source = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
    reference = np.array([[0.0, 0.0], [10.0, 10.0]])

    matches = match_star_catalogs(
        source,
        reference,
        SimilarityTransform(),
        max_distance_px=0.5,
    )

    assert matches.count == 2
    assert matches.source_indices.tolist() == [0, 2]
    assert matches.reference_indices.tolist() == [0, 1]
    assert len(set(matches.reference_indices.tolist())) == matches.count


def test_default_minimum_of_twelve_inliers_is_enforced() -> None:
    source = _stars(20, 11)
    truth = SimilarityTransform(rotation=0.02, translation=(7.0, 13.0))
    reference = truth(source)

    result = register_star_catalogs(source, reference)

    assert result.estimated
    assert not result.accepted
    assert result.inlier_count == 11
    assert "INSUFFICIENT_INLIERS" in result.reason_codes


def test_minimum_inlier_ratio_uses_smaller_catalog_as_denominator() -> None:
    rng = np.random.default_rng(30)
    core = _stars(31, 20)
    truth = SimilarityTransform(rotation=-0.04, translation=(-8.0, 17.0))
    source = np.vstack((core, rng.uniform((0.0, 0.0), (950.0, 750.0), size=(10, 2))))
    reference = np.vstack(
        (truth(core), rng.uniform((0.0, 0.0), (950.0, 750.0), size=(10, 2)))
    )
    flux = np.concatenate((np.linspace(20_000.0, 5_000.0, 20), np.ones(10)))

    result = register_star_catalogs(
        {"points": source, "flux": flux},
        {"points": reference, "flux": flux},
        thresholds=RegistrationThresholds(min_inlier_ratio=0.90),
    )

    assert result.estimated
    assert not result.accepted
    assert 0.60 <= result.inlier_ratio <= 0.75
    assert "LOW_INLIER_RATIO" in result.reason_codes


def test_rms_threshold_is_evaluated_independently_of_inlier_cutoff() -> None:
    source = _stars(35, 30)
    truth = SimilarityTransform(rotation=0.03, translation=(9.0, -4.0))
    noise = np.random.default_rng(36).normal(0.0, 0.30, size=source.shape)
    reference = truth(source) + noise

    result = register_star_catalogs(
        source,
        reference,
        thresholds=RegistrationThresholds(max_rms_px=0.10),
    )

    assert result.estimated
    assert not result.accepted
    assert result.inlier_count >= 28
    assert result.rms_px > 0.10
    assert "HIGH_RMS" in result.reason_codes


def test_non_finite_and_duplicate_coordinates_are_ignored() -> None:
    source_core = _stars(40, 14)
    truth = SimilarityTransform(rotation=0.01, translation=(3.0, -5.0))
    reference_core = truth(source_core)
    source = np.vstack((source_core, source_core[0], [np.nan, 4.0]))
    reference = np.vstack((reference_core, reference_core[0], [8.0, np.inf]))

    result = register_star_catalogs(source, reference)

    assert result.accepted
    assert result.source_count == 14
    assert result.reference_count == 14
    assert result.inlier_count == 14


def test_valid_but_too_small_catalog_degrades_without_exception() -> None:
    result = register_star_catalogs(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    assert not result.estimated
    assert not result.accepted
    assert result.reason_codes == ("INSUFFICIENT_STARS_FOR_ESTIMATE",)
    assert np.isinf(result.rms_px)


def test_invalid_catalog_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        register_star_catalogs(np.ones((5, 3)), np.ones((5, 2)))
