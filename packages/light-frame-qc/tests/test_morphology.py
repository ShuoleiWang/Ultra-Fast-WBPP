from __future__ import annotations

from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from lightframeqc.models import Star
from lightframeqc.morphology import measure_fragmented_trails


def _star(x: float, y: float, flux: float = 1000.0, flags: int = 1) -> Star:
    # The component PSFs are round: ordinary ellipticity cannot detect these
    # deliberately deblended tracks.
    return Star(x, y, flux, flux / 4, 1.0, 1.0, 0.0, 2.355, 0.0, flags)


def _chains(
    centers: list[tuple[float, float]] | None = None,
    *,
    angles: list[float] | None = None,
    fragments: int = 7,
    flags: int = 1,
) -> list[Star]:
    if centers is None:
        centers = [(x, y) for y in (150.0, 500.0, 850.0) for x in (150.0, 500.0, 850.0)]
    if angles is None:
        angles = [math.radians(115.0)] * len(centers)
    stars = []
    for index, ((x, y), angle) in enumerate(zip(centers, angles, strict=True)):
        for step in range(fragments):
            along = (step - (fragments - 1) / 2) * 7.0
            across = 0.4 * math.sin(step)
            stars.append(
                _star(
                    x + along * math.cos(angle) - across * math.sin(angle),
                    y + along * math.sin(angle) + across * math.cos(angle),
                    1000.0 - index - step / 10,
                    flags,
                )
            )
    return stars


def _background(count: int = 300, seed: int = 14) -> list[Star]:
    rng = np.random.default_rng(seed)
    return [_star(x, y, 1.0, 0) for x, y in rng.uniform(20.0, 980.0, (count, 2))]


def test_distributed_round_blended_fragments_detect_tracking_failure() -> None:
    result = measure_fragmented_trails(_chains() + _background(), 1000, 1000)

    assert result.available and result.detected
    assert result.chain_count >= 6
    assert result.fragment_fraction >= 0.10
    assert result.orientation_coherence is not None
    assert result.orientation_coherence > 0.99
    assert result.occupied_cells >= 6
    assert result.spatial_minor_fraction > 0.10


def test_curved_tracks_can_have_secondary_short_fragment_direction() -> None:
    primary = _chains()
    secondary = _chains(
        [(170.0, 200.0), (520.0, 200.0), (870.0, 200.0), (170.0, 550.0)],
        angles=[0.0] * 4,
        fragments=3,
    )
    result = measure_fragmented_trails(primary + secondary + _background(), 1000, 1000)

    assert result.detected
    assert result.candidate_chain_count > result.chain_count
    assert 0.5 < result.consensus_fraction < 1


def test_axial_consensus_wraps_at_pi() -> None:
    result = measure_fragmented_trails(
        _chains(angles=[math.radians(178 if index % 2 else 2) for index in range(9)])
        + _background(),
        1000,
        1000,
    )
    assert result.detected
    assert result.consensus_fraction == 1.0


def test_parallel_binary_stars_are_not_fragment_chains() -> None:
    centers = [(float(x), float(y)) for x in range(60, 1000, 80) for y in range(60, 1000, 80)]
    result = measure_fragmented_trails(_chains(centers, fragments=2), 1000, 1000)
    assert result.available
    assert result.chain_count == 0
    assert not result.detected


def test_one_satellite_is_not_distributed_tracking_failure() -> None:
    satellite = [_star(100.0 + 6 * index, 450.0 + 0.5 * index) for index in range(130)]
    result = measure_fragmented_trails(satellite + _background(), 1000, 1000)
    assert result.chain_count <= 1
    assert not result.detected


def test_interrupted_satellite_requires_two_dimensional_footprint() -> None:
    centers = [(float(x), float(x)) for x in range(80, 1000, 110)]
    result = measure_fragmented_trails(
        _chains(centers, angles=[math.pi / 4] * len(centers)) + _background(),
        1000,
        1000,
    )
    assert result.chain_count >= 6
    assert result.spatial_minor_fraction < 0.10
    assert not result.detected


def test_small_spatial_patch_of_parallel_knots_is_not_global_tracking_failure() -> None:
    centers = [(float(x), float(y)) for x in (150, 200, 250) for y in (150, 210, 270)]
    result = measure_fragmented_trails(_chains(centers) + _background(), 1000, 1000)
    assert result.chain_count >= 6
    assert not result.detected


def test_handful_of_elongated_galaxy_knots_is_insufficient() -> None:
    result = measure_fragmented_trails(
        _chains([(150.0, 150.0), (800.0, 150.0), (500.0, 800.0)]) + _background(),
        1000,
        1000,
    )
    assert result.chain_count <= 3
    assert not result.detected


def test_many_galaxy_knots_with_random_directions_lack_consensus() -> None:
    result = measure_fragmented_trails(
        _chains(angles=np.linspace(0.0, math.pi, 9, endpoint=False).tolist()) + _background(),
        1000,
        1000,
    )
    assert result.candidate_chain_count >= 6
    assert result.consensus_fraction <= 0.5
    assert not result.detected


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_crowded_blended_field_without_coherent_chains_is_negative(seed: int) -> None:
    rng = np.random.default_rng(seed)
    stars = [_star(x, y, float(flux)) for x, y, flux in rng.uniform(20, 980, (2500, 3))]
    result = measure_fragmented_trails(stars, 1000, 1000)
    assert result.available
    assert result.considered_stars == 500
    assert not result.detected


def test_round_dense_galaxy_knots_fail_chain_geometry() -> None:
    stars = []
    for x, y in [(x, y) for x in (150, 500, 850) for y in (150, 500, 850)]:
        for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            stars.append(_star(x + 8 * math.cos(angle), y + 8 * math.sin(angle)))
    result = measure_fragmented_trails(stars + _background(), 1000, 1000)
    assert result.candidate_chain_count == 0
    assert not result.detected


def test_unblended_point_chains_alone_are_not_enough_evidence() -> None:
    result = measure_fragmented_trails(_chains(flags=0) + _background(), 1000, 1000)
    assert result.available and not result.detected
    assert result.candidate_chain_count == 0


def test_invalid_or_truncated_sources_do_not_supply_evidence() -> None:
    invalid = [replace(star, x=float("nan")) for star in _chains()]
    invalid += [replace(star, b=-1.0) for star in _chains()]
    invalid += [replace(star, x=-1.0) for star in _chains()]
    invalid += _chains(flags=3)
    result = measure_fragmented_trails(invalid + _background(), 1000, 1000)
    assert result.considered_stars == 300
    assert not result.detected


def test_missing_geometry_or_sparse_catalog_is_unavailable() -> None:
    assert not measure_fragmented_trails(_chains(), 0, 1000).available
    assert not measure_fragmented_trails(_chains(), 1000, float("nan")).available
    assert not measure_fragmented_trails(_chains()[:29], 1000, 1000).available


def test_permutation_scale_and_flux_units_do_not_change_detection() -> None:
    stars = _chains() + _background()
    originals = [asdict(star) for star in stars]
    result = measure_fragmented_trails(stars, 1000, 1000)
    permuted = measure_fragmented_trails(stars[::-1], 1000, 1000)
    scaled = measure_fragmented_trails(
        [replace(star, x=star.x * 3, y=star.y * 3, a=star.a * 3, b=star.b * 3,
                 flux=star.flux * 7) for star in stars],
        3000,
        3000,
    )
    assert permuted == result
    assert scaled.detected == result.detected
    assert scaled.chain_count == result.chain_count
    assert scaled.fragment_fraction == result.fragment_fraction
    assert scaled.orientation_coherence == pytest.approx(result.orientation_coherence)
    assert scaled.spatial_minor_fraction == pytest.approx(result.spatial_minor_fraction)
    assert scaled.link_radius_pixels == pytest.approx(result.link_radius_pixels * 3)
    assert [asdict(star) for star in stars] == originals


def test_large_faint_catalog_does_not_dilute_bright_trails() -> None:
    result = measure_fragmented_trails(_chains() + _background(10_000), 1000, 1000)
    assert result.considered_stars == 500
    assert result.detected
