from __future__ import annotations

import hashlib
from pathlib import Path

from astropy.io import fits
import numpy as np
from PIL import Image
import pytest

from ufwbpp.stacking.integration import CalibrationError
from ufwbpp.products.preview import render_auto_stretch_preview


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_auto_stretch_preview_is_bounded_and_does_not_change_linear_master(
    tmp_path: Path,
) -> None:
    y, x = np.mgrid[:60, :90]
    linear = (100.0 + 0.2 * x + 0.1 * y).astype(np.float32)
    linear[25:28, 42:45] += 300
    master = tmp_path / "master.fits"
    fits.writeto(master, linear, overwrite=False)
    before = _sha(master)
    preview = tmp_path / "master.png"

    result = render_auto_stretch_preview(master, preview, max_long_edge=32)

    assert result.block_size == 3
    assert max(result.preview_shape) <= 32
    assert result.white_point > result.black_point
    assert _sha(master) == before
    with Image.open(preview) as image:
        assert image.mode == "L"
        assert image.size == (result.preview_shape[1], result.preview_shape[0])
        assert np.asarray(image).max() > np.asarray(image).min()

    with pytest.raises(CalibrationError) as captured:
        render_auto_stretch_preview(master, preview, max_long_edge=32)
    assert captured.value.code == "OUTPUT_EXISTS"
