from dataclasses import replace

import numpy as np

from openastroflow_engine.calibration import (
    IntegrationParameters, _RejectionSigmaFloor,
    _ordinary_mad_rejection_decision, _ordinary_mad_rejection_mask,
    _ordinary_integration_tile,
)
from openastroflow_engine.transient_rejection import detect_transient_trails
from openastroflow_engine.residual_background import fit_residual_background


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
