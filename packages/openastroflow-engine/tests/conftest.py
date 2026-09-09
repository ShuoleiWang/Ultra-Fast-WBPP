from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest


def write_frame(
    path: Path,
    role: str,
    *,
    filter_name: str = "R",
    exposure: float = 120.0,
    target: str = "M16",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["IMAGETYP"] = role
    header["FILTER"] = filter_name
    header["OBJECT"] = target
    header["INSTRUME"] = "ASI2600MM Pro"
    header["EXPTIME"] = exposure
    header["GAIN"] = 100
    header["OFFSET"] = 50
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "Mode 1"
    header["BAYERPAT"] = "NONE"
    header["DATE-OBS"] = "2026-08-31T12:00:00Z"
    data = np.arange(256, dtype=np.uint16).reshape(16, 16)
    fits.writeto(path, data, header, overwrite=False)
    return path


@pytest.fixture
def nina_project(tmp_path: Path) -> Path:
    root = tmp_path / "NINA M16"
    write_frame(root / "LIGHT" / "M16_120s_R_001.fits", "Light")
    write_frame(root / "FLAT" / "Flat_2s_R_001.fits", "Flat", exposure=2.0)
    write_frame(root / "DARK" / "Dark_120s_001.fit", "Dark", exposure=120.0)
    write_frame(root / "BIAS" / "Bias_001.fts", "Bias", exposure=0.001)
    write_frame(
        root / "MASTERS" / "MasterFlat_R.xisf-placeholder.fits",
        "Master Flat",
        exposure=2.0,
    )
    return root
