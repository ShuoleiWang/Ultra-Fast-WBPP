"""Original-pixel crop coordinate contract for the read-only display study."""
from pathlib import Path

from astropy.io import fits
import numpy as np

from ufwbpp.blink_diagnostics import DisplayReference
from ufwbpp import blink_native_crops as study


def test_native_crop_uses_preview_block_centers_and_preserves_star_pixels(tmp_path: Path, monkeypatch):
    # Non-square geometry catches accidentally using source height as width.
    rng=np.random.default_rng(8)
    raw=rng.normal(1000,2,(256,512)).astype(np.float32)
    raw[119:122,199:202]=3000
    raw[120,200]=5000
    path=tmp_path/'light.fits';fits.PrimaryHDU(raw).writeto(path)
    flat=np.ones_like(raw);dark=np.zeros_like(raw)
    monkeypatch.setattr(study,'native_master',lambda path: flat if path=='flat' else dark)
    reference=DisplayReference.from_image(rng.normal(1000,0.5,(64,128)).astype(np.float32))
    # Pixel centers map from preview to native by (p+.5)*block_size-.5.
    points=[((200+.5)/4-.5,(120+.5)/4-.5)]*9
    atlas=study.native_atlas({'path':str(path)},np.eye(3),points,reference,'flat','dark',4)
    assert atlas is not None
    for y in range(3):
        for x in range(3):
            crop=atlas.signal[y*84:y*84+80,x*84:x*84+80]
            assert np.unravel_index(np.argmax(crop),crop.shape)==(40,40)
    # Source-to-reference translation must be inverted for the crop lookup.
    transform=np.eye(3);transform[0,2]=5;transform[1,2]=-3
    moved=[(x+5,y-3) for x,y in points]
    other=study.native_atlas({'path':str(path)},transform,moved,reference,'flat','dark',4)
    np.testing.assert_array_equal(atlas.signal,other.signal)


def test_morphology_amplitude_matching_does_not_restore_unmeasurable_stars():
    y,x=np.indices((80,80));rng=np.random.default_rng(9)
    sharp=100*np.exp(-((x-40)**2+(y-40)**2)/(2*2**2))
    blurred=50*np.exp(-((x-40)**2+(y-40)**2)/(2*4**2))
    a=study.shape_crop(sharp+rng.normal(0,0.2,sharp.shape))
    b=study.shape_crop(blurred+rng.normal(0,0.2,sharp.shape))
    assert a is not None and b is not None
    assert np.sum(b>0.5)>2*np.sum(a>0.5)
    assert study.shape_crop(0.005*sharp+rng.normal(0,2,sharp.shape)) is None
