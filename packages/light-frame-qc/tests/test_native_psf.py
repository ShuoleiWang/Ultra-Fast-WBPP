from __future__ import annotations

import numpy as np

from lightframeqc.models import Star
from lightframeqc.native_psf import measure_native_psf_from_array


def _field(sigma: float, halo: float = 0.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    height, width = 300, 360
    image = rng.normal(1000.0, 3.0, (height, width))
    yy, xx = np.indices((height, width), dtype=np.float64)
    stars = []
    positions = [(40 + 45 * i, 50 + 37 * j) for i in range(5) for j in range(8)]
    for cy, cx in positions:
        flux = rng.uniform(20000.0, 60000.0)
        core = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        image += flux * core / (2 * np.pi * sigma**2)
        if halo:
            wide = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (4 * sigma) ** 2))
            image += halo * flux * wide / (2 * np.pi * (4 * sigma) ** 2)
        # Preview coordinates at scale 3 (block-mean preview convention).
        stars.append(Star(x=(cx + 0.5) / 3 - 0.5, y=(cy + 0.5) / 3 - 0.5, flux=flux, peak=flux, a=1, b=1, theta=0, fwhm=2.355 * sigma / 3, ellipticity=0.0))
    return image, stars


def test_half_flux_radius_tracks_gaussian_width_and_wings_track_halos() -> None:
    sharp, stars = _field(1.6)
    summary = measure_native_psf_from_array(sharp, stars, 3.0, 3.0, minimum_stars=8)
    assert summary.star_count >= 30
    assert summary.r50_pixels is not None
    # Half-flux radius of a Gaussian is 1.1774 sigma.
    assert abs(summary.r50_pixels - 1.1774 * 1.6) < 0.15
    assert summary.fwhm_pixels == 2 * summary.r50_pixels
    assert summary.wing_fraction is not None and 0.02 < summary.wing_fraction < 0.12
    soft, soft_stars = _field(2.4)
    softer = measure_native_psf_from_array(soft, soft_stars, 3.0, 3.0, minimum_stars=8)
    assert softer.r50_pixels is not None and softer.r50_pixels / summary.r50_pixels > 1.4
    dewy, dew_stars = _field(1.6, halo=0.6)
    dew = measure_native_psf_from_array(dewy, dew_stars, 3.0, 3.0, minimum_stars=8)
    assert dew.wing_fraction is not None and dew.wing_fraction > 2.5 * summary.wing_fraction


def test_too_few_usable_stars_reports_none() -> None:
    image, stars = _field(1.6)
    summary = measure_native_psf_from_array(image, stars[:3], 3.0, 3.0)
    assert summary.r50_pixels is None and summary.star_count == 3
    assert summary.serializable()["candidates"] == 3


def test_scaled_integer_fits_is_measured_without_decoding(tmp_path) -> None:
    """16-bit Lights (BZERO = 32768) are the common case; the stamps are scaled on read."""

    from astropy.io import fits

    from lightframeqc.native_psf import measure_native_psf, open_native_image

    image, stars = _field(1.6)
    reference = measure_native_psf_from_array(image, stars, 3.0, 3.0, minimum_stars=8)
    quantized = np.clip(np.round(image), 0, 65535).astype(np.uint16)
    path = tmp_path / "light.fits"
    fits.PrimaryHDU(data=quantized).writeto(path)  # astropy stores BZERO=32768 int16
    header = fits.getheader(path)
    assert int(header["BZERO"]) == 32768 and int(header["BITPIX"]) == 16
    with open_native_image(str(path)) as native:
        assert native is not None and native.shape == image.shape
        stamp = native[10:14, 20:25]
        assert stamp.dtype == np.float64
        np.testing.assert_allclose(stamp, quantized[10:14, 20:25].astype(np.float64))
    summary = measure_native_psf(str(path), stars, 3.0, 3.0, minimum_stars=8)
    assert "error" not in summary.serializable()
    assert summary.r50_pixels is not None and reference.r50_pixels is not None
    assert abs(summary.r50_pixels - reference.r50_pixels) < 0.05
    assert summary.star_count == reference.star_count


def test_blank_and_channel_layouts_are_handled(tmp_path) -> None:
    from astropy.io import fits

    from lightframeqc.native_psf import open_native_image

    data = np.arange(24, dtype=np.int16).reshape(4, 6)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["BLANK"] = 5
    hdu.header["BSCALE"] = 2.0
    hdu.header["BZERO"] = 100.0
    path = tmp_path / "blank.fits"
    hdu.writeto(path)
    with open_native_image(str(path)) as native:
        assert native is not None
        stamp = native[0:2, 0:6]
        assert np.isnan(stamp[0, 5])
        assert stamp[0, 1] == 2.0 * 1 + 100.0
    colour = np.stack([np.full((4, 6), 10.0), np.full((4, 6), 20.0), np.full((4, 6), 30.0)]).astype(np.float32)
    colour_path = tmp_path / "colour.fits"
    fits.PrimaryHDU(data=colour).writeto(colour_path)
    with open_native_image(str(colour_path)) as native:
        assert native is not None and native.shape == (4, 6)
        np.testing.assert_allclose(native[1:3, 2:4], 20.0)
