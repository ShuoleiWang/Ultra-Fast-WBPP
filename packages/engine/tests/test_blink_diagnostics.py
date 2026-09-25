"""Display contracts, not scientific integration or classifier acceptance."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from ufwbpp.blink.diagnostics import DisplayReference, background_grid, diagnostic_preview, detail_transfer, local_noise


def star_field(seed=7):
    rng = np.random.default_rng(seed)
    stars = np.zeros((384, 384), np.float32)
    for y, x in rng.integers(24, 360, size=(100, 2)):
        stars[y, x] += rng.uniform(200, 800)
    return gaussian_filter(stars, 1.3), rng


def test_local_noise_does_not_confuse_a_sky_plane_with_noise():
    stars, rng = star_field()
    image = rng.normal(1000, 5, stars.shape).astype(np.float32)
    y, x = np.indices(image.shape)
    assert local_noise(image + stars + 2*x + 3*y) == pytest.approx(local_noise(image + stars), rel=0.001)
    assert local_noise(image) == pytest.approx(5, rel=0.025)
    with pytest.raises(ValueError):
        local_noise(np.ones((64, 64)))


def test_uniform_cloud_stays_dimmer_and_noisier_in_the_detail_view():
    stars, rng = star_field()
    clean = 1000 + stars + rng.normal(0, 2, stars.shape)
    cloud = 1800 + 0.3 * stars + rng.normal(0, 4, stars.shape)
    reference = DisplayReference.from_image(clean)
    a = diagnostic_preview(clean, reference, source_noise=local_noise(clean), flux_scale=1, registered=True)
    b = diagnostic_preview(cloud, reference, source_noise=local_noise(cloud), flux_scale=1/0.3, registered=True)
    # Reference star cores visibly dim; the main image does not multiply them
    # by 3.33 to make thick cloud resemble a clean exposure.
    cores = stars > 20
    assert np.mean(b.detail[cores]) < np.mean(a.detail[cores]) - 0.08
    assert b.relative_noise > 1.8 and b.matched_signal_noise > 6
    assert np.mean(b.detail[stars < 0.05] < 0.08) > np.mean(a.detail[stars < 0.05] < 0.08)


def test_background_gradient_survives_and_fixed_nebulosity_cancels():
    stars, rng = star_field()
    y, x = np.indices(stars.shape)
    nebula = 30 * np.exp(-((x-192)**2+(y-192)**2)/(2*60**2))
    ref = 1000 + stars + nebula + rng.normal(0, 2, stars.shape)
    gradient = 0.15*(x-192)
    reference = DisplayReference.from_image(ref)
    frame = 1600 + 0.65*(ref-1000) + gradient
    p = diagnostic_preview(frame, reference, source_noise=local_noise(frame), flux_scale=1/0.65, registered=True)
    expected, covered = background_grid(gradient/0.65)
    expected -= np.median(expected[covered])
    np.testing.assert_allclose(p.background_difference, expected, atol=0.001)
    assert np.ptp(p.background_difference) > 70
    assert p.background_rgb is not None
    # Equal global offset/transparency with no gradient must be neutral: it
    # cannot turn a real nebula into a claimed cloud patch.
    q = diagnostic_preview(1600+0.65*(ref-1000), reference, source_noise=local_noise(frame), flux_scale=1/0.65, registered=True)
    assert np.nanmax(np.abs(q.background_difference)) < 0.001


def test_no_difference_map_without_alignment_or_photometry_and_no_fake_coverage():
    stars, rng = star_field()
    ref = 1000+stars+rng.normal(0,2,stars.shape)
    reference = DisplayReference.from_image(ref)
    frame = ref.copy(); frame[:, :96] = np.nan
    p = diagnostic_preview(frame, reference, source_noise=2, flux_scale=1, registered=True)
    assert np.all(np.isnan(p.background_difference[:, :2]))
    assert np.all(p.detail[:, :96] == 0)
    for gain, registered in [(None,True),(float('nan'),True),(1,False)]:
        q = diagnostic_preview(frame, reference, source_noise=2, flux_scale=gain, registered=registered)
        assert q.background_difference is None and q.background_rgb is None


def test_display_curve_keeps_highlights_ordered_without_a_hard_white_clip():
    values=np.array([-1e4,-10,0,1,10,100,1000,10000],np.float32)
    shown=detail_transfer(values,2)
    assert np.all(np.diff(shown)>0)
    assert shown[2] == pytest.approx(0.22)
    assert shown[-1] < 1


def test_reference_uses_only_registered_measured_candidates():
    from ufwbpp.blink.diagnostics import choose_display_reference
    frames = [{"index": i, "score": {"candidate": True}, "normalization": {"registered": True}, "metrics": {"transparency": 1, "fwhmNative": 4, "ellipticity": 0.1, "sourceRatio": 1}} for i in range(3)]
    frames[2]["normalization"]["registered"] = False
    assert choose_display_reference(frames, {0: 8, 1: 4, 2: 1}) == 1
    frames[1]["score"]["candidate"] = False
    assert choose_display_reference(frames, {0: 8, 1: 4, 2: 1}) == 0
