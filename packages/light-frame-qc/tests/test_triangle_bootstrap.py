from __future__ import annotations

import astroalign
import numpy as np
import pytest
from skimage.transform import SimilarityTransform

from lightframeqc.triangle_bootstrap import (
    _TriangleMatchTransform,
    _geometry,
    find_catalog_transform,
)


@pytest.fixture
def seeded_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.random.default_rng
    monkeypatch.setattr(
        np.random, "default_rng", lambda seed=None: original(923 if seed is None else seed)
    )


@pytest.mark.parametrize("count,rotation", [(3, 0.05), (48, 0.061), (64, np.pi), (100, -0.04)])
def test_bootstrap_matches_upstream_model_exactly(
    count: int, rotation: float, seeded_bootstrap: None
) -> None:
    rng = np.random.default_rng(19)
    source = rng.uniform(0, 1000, (count, 2))
    truth = SimilarityTransform(scale=1.002, rotation=rotation, translation=(18, -11))
    reference = truth(source) + rng.normal(0, 0.1, source.shape)

    expected, _ = astroalign.find_transform(source, reference, max_control_points=100)
    actual = find_catalog_transform(source, reference, max_control_points=100)

    np.testing.assert_array_equal(actual.params, expected.params)


def test_unrelated_catalog_exhaustion_matches_upstream(seeded_bootstrap: None) -> None:
    rng = np.random.default_rng(127)
    source = rng.uniform(0, 1500, (24, 2))
    target = rng.uniform(0, 1500, (24, 2))
    with pytest.raises(astroalign.MaxIterError) as expected:
        astroalign.find_transform(source, target, max_control_points=100)
    with pytest.raises(astroalign.MaxIterError) as actual:
        find_catalog_transform(source, target, max_control_points=100)
    assert str(actual.value) == str(expected.value)


def test_repeated_vertex_errors_and_threshold_boundaries_are_exact() -> None:
    rng = np.random.default_rng(81)
    source = rng.uniform(-1500, 1500, (100, 2))
    transform = SimilarityTransform(rotation=np.pi)
    target = transform(source)
    tolerance = astroalign.PIXEL_TOL
    source[:3] = 0
    target[:3] = transform(source[:3])
    target[:3, 0] += [np.nextafter(tolerance, 0), tolerance, np.nextafter(tolerance, np.inf)]
    matches = rng.integers(0, 100, (7000, 3, 2))
    matches[:3] = np.array([[[0, 0]] * 3, [[1, 1]] * 3, [[2, 2]] * 3])
    optimized = _TriangleMatchTransform(source, target, matches)
    original = astroalign._MatchTransform(source, target)
    assert (original.get_error(matches[:3], transform) < tolerance).tolist() == [True, False, False]
    # RANSAC asks for reordered subsets and finally for the complete match set.
    for subset in (matches, matches[::-3], matches[:3]):
        expected = original.get_error(subset, transform)
        actual = optimized.get_error(subset, transform)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual < tolerance, expected < tolerance)


def test_other_astroalign_versions_use_public_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = SimilarityTransform(translation=(4, 7))
    source = np.zeros((3, 2))
    target = np.ones((3, 2))
    calls = []

    def public(s, t, *, max_control_points):
        calls.append((s, t, max_control_points))
        return expected, (s, t)

    monkeypatch.setattr(astroalign, "__version__", "future")
    monkeypatch.setattr(astroalign, "find_transform", public)
    assert find_catalog_transform(source, target, max_control_points=32) is expected
    assert calls == [(source, target, 32)]


def test_geometry_cache_owns_coordinates_and_keys_neighbor_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _geometry.cache_clear()
    points = np.random.default_rng(14).uniform(0, 1000, (32, 2))
    coordinates = points.tobytes()
    cached = _geometry(coordinates, astroalign.NUM_NEAREST_NEIGHBORS)
    points[0] += 1
    assert _geometry(coordinates, astroalign.NUM_NEAREST_NEIGHBORS) is cached
    changed = _geometry(points.tobytes(), astroalign.NUM_NEAREST_NEIGHBORS)
    assert changed is not cached
    assert not cached[0].flags.writeable
    assert not cached[1].flags.writeable
    assert _geometry.cache_info().hits == 1
    monkeypatch.setattr(astroalign, "NUM_NEAREST_NEIGHBORS", astroalign.NUM_NEAREST_NEIGHBORS + 1)
    assert _geometry(coordinates, astroalign.NUM_NEAREST_NEIGHBORS) is not cached
    _geometry.cache_clear()
