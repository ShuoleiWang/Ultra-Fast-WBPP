from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone

import numpy as np

from lightframeqc.analysis import analyze_measurements
from lightframeqc.config import DEFAULT_CONFIG
from lightframeqc.models import (
    Decision,
    FrameMeasurement,
    FrameMetadata,
    FrameResult,
    FrameRole,
    Star,
)


GRID_SIZE = 16
PREVIEW_SIZE = 1_024
STARS_PER_CELL = 4


def test_fragmented_source_count_cannot_win_reference_selection() -> None:
    from lightframeqc.analysis import _supported_reference

    frames = [_measurement(f"reference-{index:02d}") for index in range(8)]
    bad = frames[-1]
    bad.detected_source_count = 5000
    bad.stars = [
        Star(
            x=x + step * 7, y=y + step * 2, flux=10_000 - step,
            peak=2_500, a=1, b=1, theta=0, fwhm=2.355,
            ellipticity=0, flags=1,
        )
        for x in (150, 500, 850)
        for y in (150, 500, 850)
        for step in range(-3, 4)
    ]
    bad.raw_stars = bad.stars

    reference, _ = _supported_reference(frames, DEFAULT_CONFIG)

    assert reference is not bad
    assert reference in frames[:-1]


def test_reference_recovery_uses_ordinary_pair_acceptance(monkeypatch) -> None:
    from lightframeqc import analysis
    from lightframeqc.registration import _failure_result

    frames = [_measurement(f"reference-pair-{index:02d}") for index in range(8)]
    disconnected = frames[-1]
    disconnected.detected_source_count = 5000

    def register(source, reference, **kwargs):
        if reference is disconnected:
            return _failure_result(1024, 5000, code="TRANSFORM_NOT_FOUND")
        # Full-catalog evidence passes the normal 25% gate even though stars
        # missing from the two catalogs prevent 75% overlap.
        return replace(analysis._identity_registration(source), inlier_ratio=0.50)

    monkeypatch.setattr(analysis, "_catalog", lambda frame: frame)
    monkeypatch.setattr(analysis, "register_star_catalogs", register)

    reference, cached = analysis._supported_reference(frames, DEFAULT_CONFIG)

    assert reference is frames[0]
    assert not cached[disconnected.metadata.path].accepted


def _base_catalog() -> tuple[np.ndarray, np.ndarray]:
    """Return four irregularly placed stars in every 16x16 analysis cell."""

    rng = np.random.default_rng(20260809)
    offsets = np.asarray(
        ((0.22, 0.24), (0.72, 0.27), (0.28, 0.73), (0.75, 0.69)),
        dtype=np.float64,
    )
    points: list[tuple[float, float]] = []
    cell_size = PREVIEW_SIZE / GRID_SIZE
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            for offset_x, offset_y in offsets:
                jitter_x, jitter_y = rng.uniform(-0.035, 0.035, size=2)
                points.append(
                    (
                        (column + offset_x + jitter_x) * cell_size,
                        (row + offset_y + jitter_y) * cell_size,
                    )
                )
    coordinates = np.asarray(points, dtype=np.float64)
    # Unique fluxes prevent brightness-order ties in astroalign control points.
    flux = rng.lognormal(mean=9.0, sigma=0.45, size=coordinates.shape[0])
    flux += np.linspace(0.0, 1.0, coordinates.shape[0], endpoint=False)
    return coordinates, flux


BASE_POINTS, BASE_FLUX = _base_catalog()


def _base_grids() -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    background = 1_000.0 + rows * 1.7 + columns * 0.8 + (rows * columns) * 0.013
    texture = 5.0 + rows * 0.031 + columns * 0.047
    return background, texture


BASE_BACKGROUND, BASE_TEXTURE = _base_grids()


def _stars(
    *,
    mask: np.ndarray | None = None,
    global_scale: float = 1.0,
    local_scale: np.ndarray | None = None,
) -> list[Star]:
    selected = np.ones(BASE_POINTS.shape[0], dtype=bool) if mask is None else mask
    point_indices = np.flatnonzero(selected)
    multipliers = (
        np.ones(BASE_POINTS.shape[0], dtype=np.float64)
        if local_scale is None
        else np.asarray(local_scale, dtype=np.float64)
    )
    return [
        Star(
            x=float(BASE_POINTS[index, 0]),
            y=float(BASE_POINTS[index, 1]),
            flux=float(BASE_FLUX[index] * global_scale * multipliers[index]),
            peak=float(BASE_FLUX[index] * global_scale * multipliers[index] / 8.0),
            a=1.8,
            b=1.7,
            theta=0.0,
            fwhm=4.1,
            ellipticity=1.0 - 1.7 / 1.8,
        )
        for index in point_indices
    ]


def _measurement(
    name: str,
    *,
    filter_name: str = "L",
    global_scale: float = 1.0,
    airmass: float = 1.2,
    mask: np.ndarray | None = None,
    local_scale: np.ndarray | None = None,
    background: np.ndarray | None = None,
    texture: np.ndarray | None = None,
    image_median: float = 1_015.0,
    minute: int = 0,
    reference_weight: float = 0.0,
) -> FrameMeasurement:
    observed_at = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc) + timedelta(
        minutes=minute
    )
    metadata = FrameMetadata(
        path=f"/synthetic/{name}.fits",
        width=PREVIEW_SIZE,
        height=PREVIEW_SIZE,
        channels=1,
        filter_name=filter_name,
        exposure_seconds=60.0,
        gain=100.0,
        offset=10.0,
        camera="SYNTHETIC-CAMERA",
        target="M42",
        airmass=airmass,
        observed_at=observed_at,
        role=FrameRole.LIGHT,
    )
    return FrameMeasurement(
        metadata=metadata,
        stars=_stars(mask=mask, global_scale=global_scale, local_scale=local_scale),
        preview_width=PREVIEW_SIZE,
        preview_height=PREVIEW_SIZE,
        image_median=image_median,
        image_mad=6.0,
        background_grid=np.asarray(
            BASE_BACKGROUND if background is None else background,
            dtype=np.float64,
        ).tolist(),
        texture_grid=np.asarray(
            BASE_TEXTURE if texture is None else texture,
            dtype=np.float64,
        ).tolist(),
        reader_backend="synthetic-test",
        pixinsight={"psf_signal_weight": reference_weight},
    )


def _snapshot(result: FrameResult) -> dict[str, object]:
    return {
        "path": result.path,
        "decision": result.decision.value,
        "confidence": result.confidence.value,
        "reasons": result.reasons,
        "warnings": result.warnings,
        "registration": asdict(result.registration),
        "features": asdict(result.features),
    }


def _airmass_for_normal_scale(scale: float) -> float:
    # This is an exact, ordinary extinction law across airmass.  The 1.1 frame
    # defines the clear high-altitude endpoint; 0.6 occurs near airmass 1.94.
    brightest_extinction = -2.5 * np.log10(1.1)
    extinction = -2.5 * np.log10(scale)
    return float(1.0 + (extinction - brightest_extinction) / 0.70)


def test_eight_normal_frames_keep_global_starlight_scaling_from_point_six_to_one_one() -> None:
    scales = np.linspace(1.1, 0.6, 8)
    measurements = [
        _measurement(
            f"normal_{index:02d}",
            global_scale=float(scale),
            airmass=_airmass_for_normal_scale(float(scale)),
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index, scale in enumerate(scales)
    ]

    summaries, results = analyze_measurements(measurements, DEFAULT_CONFIG)

    unexpected = [_snapshot(result) for result in results if result.decision is not Decision.KEEP]
    assert len(summaries) == 1
    assert len(results) == 8
    assert not unexpected, unexpected
    assert min(result.features.transparency_ratio or 0.0 for result in results) <= 0.6 / 1.1
    assert max(result.features.transparency_ratio or 0.0 for result in results) == 1.0


def _severe_cloud_mask() -> np.ndarray:
    rng = np.random.default_rng(86)
    target_count = 300
    brightest = np.argsort(-BASE_FLUX)[:64]
    remaining = np.setdiff1d(np.arange(BASE_POINTS.shape[0]), brightest)
    selected = rng.choice(remaining, size=target_count - brightest.size, replace=False)
    mask = np.zeros(BASE_POINTS.shape[0], dtype=bool)
    mask[np.concatenate((brightest, selected))] = True
    return mask


def test_severe_cloud_with_star_loss_and_extra_extinction_is_rejected() -> None:
    scales = np.linspace(1.1, 0.72, 8)
    cloudy_index = 4
    measurements: list[FrameMeasurement] = []
    for index, scale in enumerate(scales):
        cloudy = index == cloudy_index
        measurements.append(
            _measurement(
                f"cloud_{index:02d}",
                global_scale=float(scale) * (0.18 if cloudy else 1.0),
                airmass=_airmass_for_normal_scale(float(scale)),
                mask=_severe_cloud_mask() if cloudy else None,
                minute=index * 10,
                reference_weight=100.0 if index == 0 else 0.0,
            )
        )

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    by_path = {result.path: result for result in results}
    cloudy_result = by_path[f"/synthetic/cloud_{cloudy_index:02d}.fits"]

    assert cloudy_result.decision is Decision.REJECT_CLOUD, _snapshot(cloudy_result)
    assert cloudy_result.features.star_completeness is not None
    assert cloudy_result.features.star_completeness <= 0.45, _snapshot(cloudy_result)
    assert cloudy_result.features.extra_extinction_mag is not None
    assert cloudy_result.features.extra_extinction_mag >= 0.60, _snapshot(cloudy_result)
    assert cloudy_result.features.cloud_score >= 4, _snapshot(cloudy_result)
    clear_failures = [
        _snapshot(result)
        for result in results
        if result is not cloudy_result and result.decision is not Decision.KEEP
    ]
    assert not clear_failures, clear_failures


def test_local_cloud_covering_less_than_half_the_frame_is_sent_to_review() -> None:
    local_scale = np.ones(BASE_POINTS.shape[0], dtype=np.float64)
    local_scale[BASE_POINTS[:, 0] < PREVIEW_SIZE * 0.30] = 0.20
    measurements = [
        _measurement(
            f"patch_cloud_{index:02d}",
            local_scale=local_scale if index == 5 else None,
            airmass=1.1 + index * 0.08,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(8)
    ]

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    cloudy = next(result for result in results if result.path.endswith("patch_cloud_05.fits"))

    assert cloudy.decision is Decision.REVIEW, _snapshot(cloudy)
    assert cloudy.features.spatial_transparency_p90_mag is not None
    assert cloudy.features.spatial_transparency_p90_mag >= 0.45, _snapshot(cloudy)
    assert "CLOUD_SPATIAL_TRANSPARENCY_STRONG" in cloudy.reasons
    assert cloudy.features.cloud_score == 2, _snapshot(cloudy)


def test_large_pointing_shift_uses_only_common_field_and_stays_keep() -> None:
    """A changed footprint is not a wall when the overlap itself is healthy."""

    measurements = [
        _measurement(
            f"dither_{index:02d}",
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(8)
    ]
    shifted = measurements[5]
    shift_x = PREVIEW_SIZE * 0.25

    # Source -> reference is x + shift_x.  Reference stars left of shift_x
    # are outside the candidate footprint, while the candidate's right quarter
    # contains a new piece of sky.  Keep the total detected-source count equal
    # so this test isolates common-footprint handling from global count changes.
    overlap = [replace(star, x=star.x - shift_x) for star in shifted.stars if star.x >= shift_x]
    rng = np.random.default_rng(20260810)
    entering: list[Star] = []
    for _ in range(len(shifted.stars) - len(overlap)):
        flux = float(rng.lognormal(mean=9.0, sigma=0.45))
        entering.append(
            Star(
                x=float(rng.uniform(PREVIEW_SIZE * 0.75, PREVIEW_SIZE - 1.0)),
                y=float(rng.uniform(1.0, PREVIEW_SIZE - 1.0)),
                flux=flux,
                peak=flux / 8.0,
                a=1.8,
                b=1.7,
                theta=0.0,
                fwhm=4.1,
                ellipticity=1.0 - 1.7 / 1.8,
            )
        )
    shifted.stars = sorted(overlap + entering, key=lambda star: -star.flux)
    for measurement in measurements:
        measurement.detected_source_count = len(measurement.stars)

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    result = next(item for item in results if item.path.endswith("dither_05.fits"))

    assert result.registration.ok, _snapshot(result)
    assert result.decision is Decision.KEEP, _snapshot(result)
    assert result.features.star_completeness == 1.0, _snapshot(result)
    assert result.features.detected_source_ratio == 1.0, _snapshot(result)
    assert result.features.overlap_fraction == 0.75, _snapshot(result)
    assert result.features.largest_missing_region == 0.0, _snapshot(result)
    assert result.features.occlusion_score == 0, _snapshot(result)


def test_opaque_local_cloud_without_background_evidence_is_not_called_a_wall() -> None:
    """A sharp star-loss mask alone is not independent hard-obstruction evidence."""

    visible = BASE_POINTS[:, 0] >= PREVIEW_SIZE * 0.30
    measurements = [
        _measurement(
            f"opaque_cloud_{index:02d}",
            mask=visible if index == 5 else None,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(8)
    ]

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    result = next(item for item in results if item.path.endswith("opaque_cloud_05.fits"))

    assert result.registration.ok, _snapshot(result)
    assert result.features.largest_missing_region is not None
    assert result.features.largest_missing_region >= 0.30, _snapshot(result)
    assert result.features.missing_inside_outside_ratio == 0.0, _snapshot(result)
    assert result.features.boundary_support == 1.0, _snapshot(result)
    assert result.features.background_support == 0.0, _snapshot(result)
    assert result.features.spatial_transparency_p90_mag == 0.0, _snapshot(result)
    assert result.decision is Decision.REVIEW, _snapshot(result)
    assert result.decision is not Decision.REJECT_OCCLUSION


def test_one_sparse_grid_cell_is_not_promoted_to_occlusion_review() -> None:
    visible = np.ones(BASE_POINTS.shape[0], dtype=bool)
    visible[:STARS_PER_CELL] = False
    measurements = [
        _measurement(
            f"single_cell_{index:02d}",
            mask=visible if index == 5 else None,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(8)
    ]

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    result = next(item for item in results if item.path.endswith("single_cell_05.fits"))

    assert result.features.largest_missing_region is not None
    assert result.features.largest_missing_region < DEFAULT_CONFIG.weak_occlusion_area
    assert result.features.occlusion_score == 0, _snapshot(result)
    assert result.decision is Decision.KEEP, _snapshot(result)


def test_straight_edge_wall_is_rejected_as_hard_occlusion() -> None:
    wall_start_column = 11
    wall_x = wall_start_column / GRID_SIZE * PREVIEW_SIZE
    visible_mask = BASE_POINTS[:, 0] < wall_x
    wall_background = BASE_BACKGROUND.copy()
    wall_background[:, wall_start_column:] = 430.0
    wall_texture = BASE_TEXTURE.copy()
    wall_texture[:, wall_start_column:] *= 0.08

    measurements = [
        _measurement(
            f"wall_{index:02d}",
            mask=visible_mask if index == 5 else None,
            background=wall_background if index == 5 else None,
            texture=wall_texture if index == 5 else None,
            airmass=1.05 + index * 0.12,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(8)
    ]

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    wall_result = next(result for result in results if result.path.endswith("wall_05.fits"))

    assert wall_result.decision is Decision.REJECT_OCCLUSION, _snapshot(wall_result)
    assert wall_result.features.largest_missing_region is not None
    assert wall_result.features.largest_missing_region >= 0.30, _snapshot(wall_result)
    assert wall_result.features.missing_inside_outside_ratio is not None
    assert wall_result.features.missing_inside_outside_ratio <= 0.15, _snapshot(wall_result)
    assert wall_result.features.boundary_support is not None
    assert wall_result.features.boundary_support >= 0.65, _snapshot(wall_result)
    assert wall_result.features.occlusion_score >= 5, _snapshot(wall_result)


def test_group_with_fewer_than_four_frames_is_unassessable() -> None:
    measurements = [
        _measurement(
            f"small_group_{index:02d}",
            global_scale=1.0 - index * 0.03,
            airmass=1.1 + index * 0.1,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(3)
    ]

    summaries, results = analyze_measurements(measurements, DEFAULT_CONFIG)

    assert len(summaries) == 1
    assert len(results) == 3
    assert all(result.decision is Decision.UNASSESSABLE for result in results), [
        _snapshot(result) for result in results
    ]
    assert all("INSUFFICIENT_GROUP_FRAMES" in result.warnings for result in results)


def test_different_filters_are_never_compared_to_each_other() -> None:
    narrowband_mask = np.zeros(BASE_POINTS.shape[0], dtype=bool)
    narrowband_mask[::4] = True
    measurements: list[FrameMeasurement] = []
    for index in range(4):
        measurements.append(
            _measurement(
                f"luminance_{index:02d}",
                filter_name="L",
                global_scale=1.0,
                airmass=1.05 + index * 0.08,
                minute=index * 10,
                reference_weight=100.0 if index == 0 else 0.0,
            )
        )
        measurements.append(
            _measurement(
                f"halpha_{index:02d}",
                filter_name="HA",
                global_scale=0.03,
                airmass=1.05 + index * 0.08,
                mask=narrowband_mask,
                image_median=120.0,
                minute=index * 10 + 5,
                reference_weight=100.0 if index == 0 else 0.0,
            )
        )

    summaries, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    result_by_path = {result.path: result for result in results}

    assert len(summaries) == 2, summaries
    assert {summary["filter"] for summary in summaries} == {"L", "HA"}
    assert all(result.decision is Decision.KEEP for result in results), [
        _snapshot(result) for result in results
    ]
    assert {result.group_id for result in results if result.metadata.filter_name == "L"}.isdisjoint(
        {result.group_id for result in results if result.metadata.filter_name == "HA"}
    )
    for result in results:
        assert result.reference_path is not None
        reference = result_by_path[result.reference_path]
        assert reference.metadata.filter_name == result.metadata.filter_name


def test_patchy_selected_reference_is_reviewed_without_contaminating_clear_frames() -> None:
    """A cloudy reference must not reverse the label of seven clear frames."""

    local_scale = np.ones(BASE_POINTS.shape[0], dtype=np.float64)
    local_scale[BASE_POINTS[:, 0] < PREVIEW_SIZE * 0.30] = 0.20
    measurements = [
        _measurement(
            f"patchy_reference_{index:02d}",
            local_scale=local_scale if index == 0 else None,
            airmass=1.05 + index * 0.08,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(8)
    ]
    for measurement in measurements:
        # Keep the uncapped source-count authority tied so PixInsight weight
        # deliberately selects the locally cloudy first frame as the reference.
        measurement.detected_source_count = BASE_POINTS.shape[0]

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    bad_reference_path = "/synthetic/patchy_reference_00.fits"
    bad_reference = next(result for result in results if result.path == bad_reference_path)
    clear_results = [result for result in results if result.path != bad_reference_path]

    assert all(result.reference_path == bad_reference_path for result in results)
    assert bad_reference.decision is Decision.REVIEW, _snapshot(bad_reference)
    assert bad_reference.features.spatial_dimming_p90_mag is not None
    assert bad_reference.features.spatial_dimming_p90_mag >= 0.45, _snapshot(
        bad_reference
    )
    assert "CLOUD_SPATIAL_TRANSPARENCY_STRONG" in bad_reference.reasons
    assert all(result.decision is Decision.KEEP for result in clear_results), [
        _snapshot(result) for result in clear_results
    ]
    assert all(
        (result.features.spatial_dimming_p90_mag or 0.0) < 0.18
        for result in clear_results
    ), [_snapshot(result) for result in clear_results]


def test_faint_uneven_frames_do_not_make_clear_sky_look_locally_dim() -> None:
    # Every part of the two poor frames is fainter than clear sky. Removing
    # their global scales before building a cross-frame envelope would make
    # their relatively brighter sides falsely brighter than the clear frames.
    gradient = 0.3 + 0.7 * BASE_POINTS[:, 0] / PREVIEW_SIZE
    measurements = [
        _measurement(
            f"common_extinction_{index:02d}",
            global_scale=0.4 if index >= 6 else 1.0,
            local_scale=gradient if index >= 6 else None,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(8)
    ]
    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    for index, result in enumerate(results):
        dimming = result.features.spatial_dimming_p90_mag
        assert dimming is not None, _snapshot(result)
        if index < 6:
            assert dimming < 0.01, _snapshot(result)
            assert result.decision is Decision.KEEP, _snapshot(result)
        else:
            assert dimming > 0.45, _snapshot(result)


def test_coarse_extinction_envelope_uses_common_flux_scale() -> None:
    from lightframeqc.analysis import _apply_spatial_consensus
    from lightframeqc.models import Confidence, FrameFeatures, RegistrationMetrics

    results = []
    for index in range(8):
        uneven = index >= 6
        grid = np.tile(np.linspace(0.5, -0.5, 4), (4, 1)) if uneven else np.zeros((4, 4))
        results.append(FrameResult(
            path=str(index), group_id="group", reference_path="0",
            decision=Decision.KEEP, confidence=Confidence.HIGH,
            reasons=[], warnings=[], registration=RegistrationMetrics(),
            features=FrameFeatures(transparency_ratio=0.3 if uneven else 1.0),
            metadata=FrameMetadata(path=str(index)), star_count=100,
            grid={"rows": 16, "columns": 16,
                  "transparencyResidualMag": [[None] * 16 for _ in range(16)],
                  "coarseTransparencyResidualMag": grid.tolist()},
        ))
    _apply_spatial_consensus(results)
    assert all(result.features.spatial_dimming_p90_mag == 0 for result in results[:6])
    assert all(abs(result.features.spatial_dimming_p90_mag - 0.5) < 1e-12 for result in results[6:])


def test_airmass_explained_global_dimming_keeps_even_when_detected_counts_collapse() -> None:
    """Thresholded source counts cannot override a fully explained airmass law."""

    scales = np.linspace(1.1, 0.6, 8)
    detected_counts = (1_200, 1_100, 1_000, 850, 650, 450, 250, 120)
    # Registration uses a stable high-SNR subset while detected_source_count
    # represents the larger uncapped SEP population, including marginal stars.
    stable_mask = np.zeros(BASE_POINTS.shape[0], dtype=bool)
    stable_mask[np.argsort(-BASE_FLUX)[:100]] = True
    measurements: list[FrameMeasurement] = []
    for index, (scale, detected_count) in enumerate(
        zip(scales, detected_counts, strict=True)
    ):
        measurement = _measurement(
            f"airmass_count_{index:02d}",
            global_scale=float(scale),
            airmass=_airmass_for_normal_scale(float(scale)),
            mask=stable_mask,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        measurement.detected_source_count = detected_count
        measurements.append(measurement)

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)

    assert all(result.decision is Decision.KEEP for result in results), [
        _snapshot(result) for result in results
    ]
    assert min(result.features.detected_source_ratio or 1.0 for result in results) == 0.1
    assert all(result.features.extra_extinction_mag is not None for result in results)
    assert all(
        abs(result.features.extra_extinction_mag or 0.0) < 1.0e-9
        for result in results
    ), [_snapshot(result) for result in results]


def _uniform_severe_cloud_mask() -> np.ndarray:
    """Keep 400 stars across every cell, including all bright control stars."""

    rng = np.random.default_rng(20260811)
    selected = set(np.argsort(-BASE_FLUX)[:100].tolist())
    for cell in range(GRID_SIZE * GRID_SIZE):
        selected.add(cell * STARS_PER_CELL + int(rng.integers(0, STARS_PER_CELL)))
    remaining = np.asarray(sorted(set(range(BASE_POINTS.shape[0])) - selected))
    selected.update(
        rng.choice(remaining, size=400 - len(selected), replace=False).tolist()
    )
    mask = np.zeros(BASE_POINTS.shape[0], dtype=bool)
    mask[list(selected)] = True
    return mask


def test_one_clear_and_seven_severe_clouds_rejects_the_cloud_majority() -> None:
    """The clear envelope must not learn a seven-frame cloudy majority as normal."""

    cloud_mask = _uniform_severe_cloud_mask()
    measurements: list[FrameMeasurement] = []
    for index in range(8):
        clear = index == 0
        measurement = _measurement(
            f"cloud_majority_{index:02d}",
            global_scale=1.0 if clear else 0.20,
            airmass=1.05 + index * 0.12,
            mask=None if clear else cloud_mask,
            minute=index * 10,
            reference_weight=100.0 if clear else 0.0,
        )
        measurement.detected_source_count = (
            BASE_POINTS.shape[0] if clear else int(np.count_nonzero(cloud_mask))
        )
        measurements.append(measurement)

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    clear_result = next(result for result in results if result.path.endswith("_00.fits"))
    cloud_results = [result for result in results if result is not clear_result]

    assert clear_result.decision is Decision.KEEP, _snapshot(clear_result)
    assert len(cloud_results) == 7
    assert all(result.decision is Decision.REJECT_CLOUD for result in cloud_results), [
        _snapshot(result) for result in cloud_results
    ]
    assert all(
        result.features.star_completeness is not None
        and result.features.star_completeness <= 0.45
        for result in cloud_results
    ), [_snapshot(result) for result in cloud_results]
    assert all(
        result.features.extra_extinction_mag is not None
        and result.features.extra_extinction_mag >= 0.60
        for result in cloud_results
    ), [_snapshot(result) for result in cloud_results]


def test_hfr_and_fwhm_outlier_is_reviewed_without_being_called_cloud() -> None:
    measurements = [
        _measurement(
            f"focus_{index:02d}",
            airmass=1.1 + index * 0.05,
            minute=index * 10,
            reference_weight=100.0 if index == 0 else 0.0,
        )
        for index in range(8)
    ]
    for measurement in measurements:
        measurement.metadata.header["HFR"] = 2.50
    bad = measurements[5]
    bad.metadata.header["HFR"] = 3.20
    bad.stars = [replace(star, fwhm=star.fwhm * 1.35) for star in bad.stars]

    _, results = analyze_measurements(measurements, DEFAULT_CONFIG)
    bad_result = next(result for result in results if result.path.endswith("focus_05.fits"))
    clear = [result for result in results if result is not bad_result]

    assert bad_result.decision is Decision.REVIEW, _snapshot(bad_result)
    assert bad_result.features.shape_score >= 2
    assert bad_result.features.nina_hfr_pixels == 3.20
    assert "SHAPE_FOCUS_OR_SEEING_OUTLIER" in bad_result.reasons
    assert bad_result.features.cloud_score == 0
    assert all(result.decision is Decision.KEEP for result in clear), [
        _snapshot(result) for result in clear
    ]
