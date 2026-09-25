from dataclasses import replace

import numpy as np

from ufwbpp.stacking.integration import (
    IntegrationParameters, _RejectionSigmaFloor,
    _ordinary_mad_rejection_decision, _ordinary_mad_rejection_mask,
    _ordinary_integration_tile,
)
from ufwbpp.stacking.transient_rejection import detect_transient_trails
from ufwbpp.stacking.residual_background import fit_residual_background


def _scene(trail=True):
    rng = np.random.default_rng(298)
    yy, xx = np.mgrid[:640, :640]
    common = 100 + 0.02 * xx
    # Real shared linear nebulosity and extended galaxy must survive.
    common = common + 80 * np.exp(-((yy - .45 * xx - 160) / 5) ** 2 / 2)
    common += 40 * np.exp(-((xx - 370) ** 2 / 65**2 + (yy - 410) ** 2 / 35**2) / 2)
    for x, y in [(70, 420), (180, 105), (320, 320), (460, 560)]:
        common += 1500 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 8)
    values = common[None] + rng.normal(0, 10, (15, 640, 640))
    # Different per-frame smooth backgrounds are detection nuisances only.
    values += np.linspace(-9, 9, 15)[:, None, None] * (xx / 640)[None]
    distance = (yy - .8 * xx - 45) / np.sqrt(1 + .8**2)
    if trail:
        values[2] += 25 * np.exp(-distance**2 / (2 * 1.5**2))
        values[2] += 5 * np.exp(-distance**2 / (2 * 4**2))
    return values.astype(np.float32), distance


def _model(values):
    return detect_transient_trails(values.reshape(15, 160, 4, 160, 4).mean((2, 4)), 4)


def _floor():
    return _RejectionSigmaFloor(True, .5, 10, 0, 0, 0, 0, 0, 0, 3, "test")


def test_faint_trail_wings_rejected_without_touching_common_structure():
    values, distance = _scene()
    model = _model(values)
    assert {trail.frame for trail in model.trails} == {2}
    baseline = _ordinary_mad_rejection_decision(values, IntegrationParameters(), _floor())[2]
    accepted = _ordinary_mad_rejection_decision(
        values, IntegrationParameters(), _floor(), transient_model=model)[2]
    inside = (np.abs(distance) < 7)
    inside[:40] = inside[-40:] = False
    assert np.mean(accepted[2, inside]) < .03
    assert np.mean(baseline[2, inside]) > .85  # reproduces the missed wings
    np.testing.assert_array_equal(accepted[np.arange(15) != 2], baseline[np.arange(15) != 2])
    outside = np.abs(distance) > 16
    np.testing.assert_array_equal(accepted[:, outside], baseline[:, outside])
    # The trail-free shared line, stars, and galaxy produce no spatial masks.
    clean, _ = _scene(trail=False)
    assert _model(clean).trails == ()


def test_spatial_masks_are_tile_and_frame_order_invariant_and_shared_by_reducers():
    values, _ = _scene()
    model = _model(values)
    expected = _ordinary_mad_rejection_mask(
        values, IntegrationParameters(), _floor(), transient_model=model)
    pieces = []
    for start in range(0, 640, 37):
        stop = min(640, start + 37)
        pieces.append(_ordinary_mad_rejection_mask(
            values[:, start:stop], IntegrationParameters(), _floor(),
            transient_model=model, first_row=start))
    np.testing.assert_array_equal(np.concatenate(pieces, axis=1), expected)
    _, _, cpu_accepted = _ordinary_mad_rejection_decision(
        values, IntegrationParameters(), _floor(), transient_model=model)
    np.testing.assert_array_equal(expected, ~cpu_accepted)
    reversed_model = _model(values[::-1])
    reversed_mask = _ordinary_mad_rejection_mask(
        values[::-1], IntegrationParameters(), _floor(), transient_model=reversed_model)
    np.testing.assert_array_equal(reversed_mask[::-1], expected)


def test_spatial_rejection_respects_per_pixel_minimum_and_missing_coverage():
    values, _ = _scene()
    model = _model(values)
    # Keep the trail sample but only one other observation in a bounded patch.
    values[np.arange(15) != 2, 285:295, 305:315] = np.nan
    values[0, 285:295, 305:315] = 100
    mask = _ordinary_mad_rejection_mask(
        values, IntegrationParameters(), _floor(), transient_model=model)
    assert not mask[:, 285:295, 305:315].any()


def test_group_sky_alignment_preserves_full_stack_mean_and_prevents_mask_stripes():
    values, distance = _scene(trail=False)
    yy, xx = np.mgrid[:640, :640]
    # One otherwise valid exposure has an additive sky difference, exactly the
    # trigger for a dark stripe when that exposure is masked along a trail.
    values[2] += 24 + .02 * xx
    weights = np.arange(1,16,dtype=np.float64)
    weights /= weights.sum()
    preview = values.reshape(15,160,4,160,4).mean((2,4))
    alignment = fit_residual_background(preview,4,weights)
    assert alignment is not None
    normalized=values.copy()
    alignment.apply_rows(normalized,0)
    before=np.einsum('i,iyx->yx',weights,values)
    after=np.einsum('i,iyx->yx',weights,normalized)
    np.testing.assert_allclose(after,before,rtol=0,atol=3e-5)
    # Identical science pixels are compared with/without one sample's sky;
    # background-subtracted astrophysical flux and all-weight mean are intact.
    stripe=abs(distance)<7
    desired=before
    old_without=(before-weights[2]*values[2])/(1-weights[2])
    new_without=(after-weights[2]*normalized[2])/(1-weights[2])
    assert abs(np.mean((new_without-desired)[stripe])) < .08
    assert abs(np.mean((old_without-desired)[stripe])) > .6
    # Row interpolation, as used by both CPU and Metal, is tile invariant.
    tiled=values.copy()
    for y0 in range(0,640,37): alignment.apply_rows(tiled[:,y0:y0+37],y0)
    np.testing.assert_array_equal(tiled,normalized)
    assert alignment.serializable()['maximumWeightedCorrection'] < 1e-12


def test_long_trail_detection_reaches_observed_image_endpoints():
    values,_=_scene(trail=False)
    values[1,:,300:304]+=100
    model=_model(values)
    matching=[t for t in model.trails if t.frame==1 and abs(t.normal_x)>.99]
    assert matching
    mask=np.ones_like(values,dtype=bool)
    model.reject_rows(mask,0,np.ones((640,640),bool))
    assert not mask[1,20,301] and not mask[1,619,301]


def test_only_added_rejections_receive_sky_alignment_and_original_mad_stays_fixed():
    values,_=_scene()
    values[2]+=24
    weights=np.full(15,1/15)
    preview=values.reshape(15,160,4,160,4).mean((2,4))
    alignment=fit_residual_background(preview,4,weights)
    assert alignment is not None
    alignment.apply_coordinates(preview,np.arange(160)*4+1.5,np.arange(160)*4+1.5)
    model=replace(detect_transient_trails(preview,4),background_alignment=alignment)
    original=_ordinary_mad_rejection_decision(values,IntegrationParameters(),_floor())[2]
    corrected=values.copy()
    _,_,accepted=_ordinary_integration_tile(corrected,IntegrationParameters(),_floor(),model,0)
    added=np.any(original&~accepted,axis=0)
    assert np.count_nonzero(added)>1000
    assert np.all(~accepted | original)
    np.testing.assert_array_equal(corrected[:,~added],values[:,~added])
    np.testing.assert_array_equal(accepted[:,~added],original[:,~added])
    # Existing point rejection remains untouched; neither original source array
    # nor any control pixel is subjected to background subtraction/re-clipping.
    assert np.max(abs(corrected[:,added]-values[:,added]))>10
    tiled=values.copy();pieces=[]
    for y0 in range(0,640,37):
        pieces.append(_ordinary_integration_tile(tiled[:,y0:y0+37],IntegrationParameters(),_floor(),model,y0)[2])
    np.testing.assert_array_equal(tiled,corrected)
    np.testing.assert_array_equal(np.concatenate(pieces,axis=1),accepted)


def _reference_fit_residual_background(values, bin_factor, weights):
    """The previous one-frame-at-a-time cell statistics, kept as the reference."""
    from ufwbpp.stacking import residual_background as module

    n, height, width = values.shape
    cell = max(4, int(np.ceil(128 / bin_factor)))
    ys = list(range(0, height, cell)); xs = list(range(0, width, cell))
    if n < 5 or len(xs) < 4 or len(ys) < 4:
        return None
    finite = np.isfinite(values)
    common = np.sum(finite, axis=0) >= max(5, (n + 1) // 2)
    reference = np.full((height, width), np.nan, np.float32)
    reference[common] = np.nanmedian(np.where(finite, values, np.nan)[:, common], axis=0)
    nodes = np.full((n, len(ys), len(xs)), np.nan, np.float64)
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            ref = reference[y0:y0 + cell, x0:x0 + cell]
            valid_ref = np.isfinite(ref)
            if np.count_nonzero(valid_ref) < 16:
                continue
            limit = float(np.quantile(ref[valid_ref], .85))
            background = valid_ref & (ref <= limit)
            for frame in range(n):
                sample = values[frame, y0:y0 + cell, x0:x0 + cell] - ref
                selected = sample[background & np.isfinite(sample)]
                if selected.size < 16:
                    continue
                center = float(np.median(selected)); sigma = float(1.4826 * np.median(abs(selected - center)))
                if sigma > 0:
                    selected = selected[abs(selected - center) < 3.5 * sigma]
                if selected.size >= 16:
                    nodes[frame, iy, ix] = float(np.median(selected))
    return nodes


def test_vectorized_cell_statistics_match_per_frame_reference(monkeypatch) -> None:
    from ufwbpp.stacking import residual_background as module

    rng = np.random.default_rng(31)
    n, height, width = 7, 150, 190
    yy, xx = np.indices((height, width), dtype=np.float64)
    preview = np.empty((n, height, width), dtype=np.float32)
    for index in range(n):
        gradient = 0.02 * index * xx - 0.01 * index * yy
        preview[index] = (1000.0 + gradient + rng.normal(0.0, 4.0, (height, width))).astype(np.float32)
    preview[:, 40:50, 60:70] += 300.0  # shared structure, cancels in the residual
    preview[rng.random(preview.shape) < 0.02] = np.nan
    preview[2, :30, :] = np.nan
    preview[3, 70:90, 100:120] = np.inf
    preview[5, 100:118, 20:38] = 1000.0  # constant cell: zero dispersion path
    expected_nodes = _reference_fit_residual_background(preview, 9, np.ones(n))
    # Recover the vectorized nodes from the per-frame robust fits (each frame
    # is fitted once for the holdout trial and, when beneficial, once more).
    real_robust_fit = module._robust_fit
    seen: list[np.ndarray] = []

    def spy_robust_fit(nodes, valid, sigma_nodes):
        if not seen or not np.array_equal(seen[-1], nodes, equal_nan=True):
            seen.append(np.array(nodes, copy=True))
        return real_robust_fit(nodes, valid, sigma_nodes)

    monkeypatch.setattr(module, "_robust_fit", spy_robust_fit)
    module.fit_residual_background(preview, 9, np.ones(n))
    assert len(seen) == n
    actual_nodes = np.stack(seen)
    assert actual_nodes.shape == expected_nodes.shape
    assert np.isfinite(expected_nodes).sum() > 0.5 * expected_nodes.size
    assert np.array_equal(actual_nodes, expected_nodes, equal_nan=True)


# --- v2 detector: fast Radon transform and multi-scale line detection ---


def _dyadic_path(n, s):
    if n == 1:
        return [0]
    h = int(np.trunc(s / 2))
    d = s - h
    top = _dyadic_path(n // 2, h)
    return top + [o + d for o in _dyadic_path(n // 2, h)]


def test_fast_radon_levels_are_exact_dyadic_line_sums():
    from ufwbpp.stacking.transient_rejection import fast_radon_levels

    rng = np.random.default_rng(0)
    height, width = 8, 11
    image = rng.standard_normal((height, width)).astype(np.float32)
    levels = fast_radon_levels(image, 2)
    assert [n for n, _ in levels] == [2, 4, 8]
    for n, sums in levels:
        assert sums.shape == (height // n, 2 * n - 1, width + 2 * height)
        for b in range(height // n):
            for s in range(-(n - 1), n):
                offsets = _dyadic_path(n, s)
                for x in range(-height, width + height):
                    expected = sum(
                        float(image[b * n + r, x + o]) for r, o in enumerate(offsets) if 0 <= x + o < width
                    )
                    assert abs(float(sums[b, s + n - 1, x + height]) - expected) < 1e-4
    # The steepest dyadic paths are the exact diagonal, shallow ones are staircases.
    assert _dyadic_path(8, 7) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert _dyadic_path(8, -5) == [0, -1, -1, -2, -3, -4, -4, -5]


def _clean_stack(rng, frames=12, size=(320, 480), sigma=10.0):
    yy, xx = np.mgrid[: size[0], : size[1]]
    common = 100 + 0.01 * xx + 0.02 * yy
    for x, y in [(60, 70), (250, 200), (400, 300), (150, 260), (430, 60)]:
        common = common + 3000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 6)
    # An elongated galaxy-like object.
    common = common + 400 * np.exp(-(((xx - 300) * 0.8 + (yy - 120) * 0.6) ** 2 / 60**2
                                    + ((xx - 300) * -0.6 + (yy - 120) * 0.8) ** 2 / 20**2) / 2)
    values = common[None] + rng.normal(0.0, sigma, (frames,) + size)
    return values.astype(np.float32), xx, yy


def _detect(values, factor=4):
    frames, height, width = values.shape
    preview = values.reshape(frames, height // factor, factor, width // factor, factor).mean((2, 4))
    return detect_transient_trails(preview.astype(np.float32), factor, workers=3)


def test_faint_full_length_trail_is_found_from_its_line_integral():
    """0.35 sigma per pixel, 2 px wide, across the frame: per-pixel rules see nothing."""

    rng = np.random.default_rng(11)
    values, xx, yy = _clean_stack(rng)
    nx, ny = np.cos(np.radians(50.0)), np.sin(np.radians(50.0))
    cross = xx * nx + yy * ny - 260.0
    values[5] += (3.5 * np.exp(-cross**2 / (2 * 1.0**2))).astype(np.float32)
    model = _detect(values)
    assert [t.frame for t in model.trails] == [5]
    trail = model.trails[0]
    assert abs(abs(trail.normal_x * nx + trail.normal_y * ny) - 1.0) < 2e-3
    assert abs(trail.distance * np.sign(trail.normal_x * nx + trail.normal_y * ny) - 260.0) < 4.0
    assert trail.stop - trail.start > 0.6 * 400
    assert _detect(np.delete(values, 5, axis=0)).trails == ()


def test_trail_that_fades_and_blinks_keeps_its_running_extent():
    rng = np.random.default_rng(12)
    values, xx, yy = _clean_stack(rng)
    nx, ny = np.cos(np.radians(-35.0)), np.sin(np.radians(-35.0))
    cross = xx * nx + yy * ny + 40.0
    along = -xx * ny + yy * nx
    # Bright in the first half, faint in the second half, with blinking gaps.
    amplitude = np.where(along < 200, 25.0, 6.0) * (np.sin(along / 12.0) > -0.6)
    values[3] += (amplitude * np.exp(-cross**2 / (2 * 1.2**2))).astype(np.float32)
    model = _detect(values)
    assert {t.frame for t in model.trails} == {3}
    covered = np.zeros(values.shape[1:], dtype=bool)
    for trail in model.trails:
        mask = np.ones(values.shape, dtype=bool)
        model.reject_rows(mask, 0, np.ones(values.shape[1:], dtype=bool))
        covered |= ~mask[3]
    line = np.abs(cross) < 1.0
    assert np.mean(covered[line]) > 0.85


def test_static_galaxy_and_star_halos_under_a_scale_error_are_not_trails():
    """A frame 2% brighter than the group leaves the galaxy's major axis and
    every halo positive in the residual; none of them is a corridor."""

    rng = np.random.default_rng(13)
    values, xx, yy = _clean_stack(rng)
    values[4] *= np.float32(1.02)
    values[7] = (values[7] - 100.0) * np.float32(0.97) + 100.0
    assert _detect(values).trails == ()


def test_short_bright_trail_is_found_at_a_lower_level():
    rng = np.random.default_rng(14)
    values, xx, yy = _clean_stack(rng)
    nx, ny = np.cos(np.radians(80.0)), np.sin(np.radians(80.0))
    cross = xx * nx + yy * ny - 150.0
    along = -xx * ny + yy * nx
    segment = (along > -80) & (along < 220)  # ~300 px of a 2 px wide, 3 sigma trail
    values[8] += (30.0 * segment * np.exp(-cross**2 / (2 * 1.0**2))).astype(np.float32)
    model = _detect(values)
    assert [t.frame for t in model.trails] == [8]
    trail = model.trails[0]
    assert 250 <= trail.stop - trail.start <= 420
