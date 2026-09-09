from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

from astropy.io import fits
import numpy as np
import pytest
from xisf import XISF

from lightframeqc.config import DEFAULT_CONFIG
from lightframeqc.identity import (
    FileIdentityError,
    compute_file_identity,
    verify_file_identity_stat,
)
from lightframeqc.measure import FrameMeasurementError, measure_frame
from lightframeqc.models import FrameRole
from lightframeqc.readers import probe_frame_metadata


def _write_fits(path: Path, image_type: str, *, exposure: float = 30.0) -> None:
    hdu = fits.PrimaryHDU(np.zeros((32, 48), dtype=np.float32))
    hdu.header["IMAGETYP"] = image_type
    hdu.header["EXPTIME"] = exposure
    hdu.header["FILTER"] = "R"
    hdu.header["INSTRUME"] = "QHY268M"
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    hdu.header["READOUTM"] = "High Gain 2CMS"
    hdu.header["BAYERPAT"] = "RGGB"
    hdu.writeto(path)


def _set_xisf_image_type(path: Path, image_type: str) -> None:
    raw = path.read_bytes()
    assert raw[:8] == b"XISF0100"
    header_length = struct.unpack("<I", raw[8:12])[0]
    root = ET.fromstring(raw[16 : 16 + header_length])
    image = next(element for element in root.iter() if element.tag.endswith("Image"))
    image.set("imageType", image_type)
    header = ET.tostring(root, encoding="utf-8")
    path.write_bytes(
        raw[:8]
        + struct.pack("<I", len(header))
        + raw[12:16]
        + header
        + raw[16 + header_length :]
    )


@pytest.mark.parametrize(
    ("image_type", "expected"),
    [
        ("LIGHT", FrameRole.LIGHT),
        ("Flat", FrameRole.RAW_FLAT),
        ("Master Flat", FrameRole.MASTER_FLAT),
        ("Dark", FrameRole.DARK),
        ("Master Dark", FrameRole.MASTER_DARK),
        ("Bias", FrameRole.BIAS),
        ("Master Bias", FrameRole.MASTER_BIAS),
        ("Master Light", FrameRole.MASTER_LIGHT),
    ],
)
def test_fits_header_only_probe_resolves_roles_and_acquisition_identity(
    tmp_path: Path, image_type: str, expected: FrameRole
) -> None:
    source = tmp_path / f"{expected.value}.fits"
    _write_fits(source, image_type)

    metadata = probe_frame_metadata(source)

    assert metadata.role == expected
    assert metadata.role_conflicts == []
    assert metadata.role_evidence == [f"FITS:IMAGETYP={image_type}"]
    assert (metadata.width, metadata.height, metadata.channels) == (48, 32, 1)
    assert metadata.image_count == 1
    assert metadata.binning_known is True
    assert (metadata.binning_x, metadata.binning_y) == (1, 1)
    assert metadata.cfa_pattern == "RGGB"
    assert metadata.readout_mode == "High Gain 2CMS"


def test_zero_second_exposure_does_not_fall_back_to_filename(tmp_path: Path) -> None:
    source = tmp_path / "bias_300s.fits"
    _write_fits(source, "BIAS", exposure=0.0)

    metadata = probe_frame_metadata(source)

    assert metadata.role == FrameRole.BIAS
    assert metadata.exposure_seconds == 0.0


def test_nina_date_loc_is_preferred_for_observing_night_wall_clock(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nina-local-time.fits"
    _write_fits(source, "LIGHT")
    with fits.open(source, mode="update") as hdul:
        hdul[0].header["DATE-LOC"] = "2026-05-17T19:30:00.000"
        hdul[0].header["DATE-OBS"] = "2026-05-17T11:30:00.000"
        hdul.flush()

    metadata = probe_frame_metadata(source)

    assert metadata.observed_at is not None
    assert metadata.observed_at.isoformat() == "2026-05-17T19:30:00"
    assert metadata.observed_at.tzinfo is None


def test_explicit_unknown_image_type_is_not_promoted_by_light_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "LIGHT" / "focus-frame.fits"
    source.parent.mkdir()
    _write_fits(source, "FOCUS")

    metadata = probe_frame_metadata(source)

    assert metadata.role is FrameRole.UNKNOWN
    assert metadata.role_evidence == ["FITS:IMAGETYP=FOCUS"]
    assert metadata.role_conflicts == []


def test_xisf_image_type_is_read_without_pixel_decode_and_conflicts_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "conflict.xisf"
    image_metadata = {
        "FITSKeywords": {
            "IMAGETYP": [{"value": "'LIGHT'", "comment": ""}],
            "FILTER": [{"value": "'R'", "comment": ""}],
        }
    }
    XISF.write(
        str(source),
        np.zeros((32, 48, 1), dtype=np.float32),
        image_metadata=image_metadata,
        codec=None,
    )
    _set_xisf_image_type(source, "MasterFlat")

    def forbid_decode(*args: object, **kwargs: object) -> None:
        raise AssertionError("probe_frame_metadata decoded XISF pixels")

    monkeypatch.setattr(XISF, "read_image", forbid_decode)
    metadata = probe_frame_metadata(source)

    assert metadata.role == FrameRole.UNKNOWN
    assert metadata.image_count == 1
    assert metadata.role_evidence == [
        "XISF:imageType=MasterFlat",
        "FITS:IMAGETYP=LIGHT",
    ]
    assert len(metadata.role_conflicts) == 1
    assert "MASTER_FLAT" in metadata.role_conflicts[0]
    assert "LIGHT" in metadata.role_conflicts[0]


def test_fits_probe_reports_all_image_hdus(tmp_path: Path) -> None:
    source = tmp_path / "two-images.fits"
    primary = fits.PrimaryHDU(np.zeros((20, 30), dtype=np.uint16))
    primary.header["IMAGETYP"] = "LIGHT"
    extension = fits.ImageHDU(np.zeros((12, 18), dtype=np.uint16))
    extension.header["IMAGETYP"] = "MASTER DARK"
    fits.HDUList([primary, extension]).writeto(source)

    first = probe_frame_metadata(source, image_index=0)
    second = probe_frame_metadata(source, image_index=1)

    assert first.image_count == second.image_count == 2
    assert first.role == FrameRole.LIGHT
    assert second.role == FrameRole.MASTER_DARK
    assert (second.width, second.height) == (18, 12)


def test_file_identity_hashes_bytes_and_stat_verification_detects_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "identity.bin"
    source.write_bytes(b"immutable input bytes")
    before = source.stat()

    identity = compute_file_identity(source, chunk_size=3)

    assert identity.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert identity.size_bytes == before.st_size
    assert identity.mtime_ns == before.st_mtime_ns
    assert identity.device == before.st_dev
    assert identity.inode == before.st_ino
    verify_file_identity_stat(source, identity)

    source.write_bytes(b"changed input bytes")
    with pytest.raises(FileIdentityError) as raised:
        verify_file_identity_stat(source, identity)
    assert raised.value.code == "FILE_CHANGED_SINCE_IDENTITY"


def test_measurement_binds_identity_and_rejects_mid_measurement_stat_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "light.fits"
    _write_fits(source, "LIGHT")
    original_bytes = source.read_bytes()

    measurement = measure_frame(source, config=DEFAULT_CONFIG)
    assert measurement.identity is not None
    assert measurement.identity.sha256 == hashlib.sha256(original_bytes).hexdigest()

    import lightframeqc.measure as measure_module

    original_reader = measure_module.read_frame_preview

    def read_then_touch(*args: object, **kwargs: object):
        preview = original_reader(*args, **kwargs)
        current = source.stat()
        os.utime(
            source,
            ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
        )
        return preview

    monkeypatch.setattr(measure_module, "read_frame_preview", read_then_touch)
    with pytest.raises(FrameMeasurementError) as raised:
        measure_frame(source, config=DEFAULT_CONFIG)
    assert raised.value.code == "FILE_CHANGED_DURING_MEASUREMENT"
