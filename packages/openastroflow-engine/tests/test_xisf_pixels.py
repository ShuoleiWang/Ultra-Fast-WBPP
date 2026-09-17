from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

from astropy.io import fits
import numpy as np
import pytest
from lightframeqc.xisf import XISF

from openastroflow_engine.calibration import CalibrationError, read_frame_info
from openastroflow_engine.global_normalization import GlobalNormalizationParameters
from openastroflow_engine.pixel_pipeline import (
    MasterMetadataOverride,
    PipelineParameters,
    run_portable_pipeline,
)
from openastroflow_engine.xisf_pixels import (
    XisfDecodePolicy,
    _xisf_numeric_domain,
    convert_xisf_to_fits,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, codec: str | None = None) -> np.ndarray:
    values = np.linspace(0.8, 1.2, 96 * 64, dtype=np.float32).reshape(64, 96, 1)
    XISF.write(
        str(path),
        values,
        codec=codec,
        image_metadata={
            "id": "integration",
            "imageType": "MasterFlat",
            "FITSKeywords": {
                "IMAGETYP": [{"value": "'Master Flat'", "comment": ""}],
                "FILTER": [{"value": "'R'", "comment": ""}],
                "INSTRUME": [{"value": "'QHY268M'", "comment": ""}],
                "GAIN": [{"value": "100", "comment": ""}],
                "OFFSET": [{"value": "50", "comment": ""}],
                "XBINNING": [{"value": "1", "comment": ""}],
                "YBINNING": [{"value": "1", "comment": ""}],
                "READOUTM": [{"value": "'Mode 1'", "comment": ""}],
                "BAYERPAT": [{"value": "'NONE'", "comment": ""}],
                "COMMENT": [{"value": "private /Users/example/raw", "comment": ""}],
            },
            "XISFProperties": {
                "PixInsight:ProcessingHistory": {
                    "id": "PixInsight:ProcessingHistory",
                    "type": "String",
                    "value": "private /Users/example/raw",
                }
            },
        },
    )
    return values[:, :, 0]


def _clone_image(path: Path, *, image_id: str, image_type: str) -> None:
    raw = path.read_bytes()
    header_length = struct.unpack("<I", raw[8:12])[0]
    root = ET.fromstring(raw[16 : 16 + header_length])
    image = next(element for element in root.iter() if element.tag.endswith("Image"))
    parent = next(element for element in root.iter() if image in list(element))
    clone = ET.fromstring(ET.tostring(image, encoding="utf-8"))
    clone.set("id", image_id)
    clone.set("imageType", image_type)
    parent.append(clone)
    header = ET.tostring(root, encoding="utf-8")
    path.write_bytes(
        raw[:8] + struct.pack("<I", len(header)) + raw[12:16] + header + raw[16 + header_length :]
    )


def _write_master_xisf(
    path: Path,
    values: np.ndarray,
    *,
    role: str,
    exposure: float,
) -> Path:
    XISF.write(
        str(path),
        np.asarray(values, dtype=np.float32)[:, :, None],
        image_metadata={
            "id": "integration",
            "imageType": role.replace(" ", ""),
            "FITSKeywords": {
                "IMAGETYP": [{"value": repr(role), "comment": ""}],
                "FILTER": [{"value": "'R'", "comment": ""}],
                "OBJECT": [{"value": "'TEST'", "comment": ""}],
                "INSTRUME": [{"value": "'QHY268M'", "comment": ""}],
                "EXPTIME": [{"value": str(exposure), "comment": ""}],
                "GAIN": [{"value": "100", "comment": ""}],
                "OFFSET": [{"value": "50", "comment": ""}],
                "XBINNING": [{"value": "1", "comment": ""}],
                "YBINNING": [{"value": "1", "comment": ""}],
                "READOUTM": [{"value": "'Mode 1'", "comment": ""}],
                "BAYERPAT": [{"value": "'NONE'", "comment": ""}],
                "CCD-TEMP": [{"value": "-10", "comment": ""}],
            },
        },
    )
    return path


@pytest.mark.parametrize("codec", [None, "zlib", "lz4", "zstd"])
def test_xisf_pixel_bridge_decodes_supported_attachments_read_only(
    tmp_path: Path, codec: str | None
) -> None:
    source = tmp_path / f"master-{codec or 'none'}.xisf"
    expected = _write(source, codec)
    before_hash = _hash(source)
    before_stat = source.stat()

    destination = tmp_path / f"converted-{codec or 'none'}.fits"
    receipt = convert_xisf_to_fits(source, destination)

    with fits.open(destination, memmap=True) as hdul:
        np.testing.assert_allclose(hdul[0].data, expected, rtol=0, atol=0)
        assert "private" not in repr(hdul[0].header).casefold()
    assert read_frame_info(destination).role == "MASTER_FLAT"
    assert receipt.source_sha256 == "sha256:" + before_hash
    assert receipt.converted_sha256.startswith("sha256:")
    assert receipt.serializable()["bridgeVersion"] == "xisf-private-fits-v2"
    assert receipt.source_sample_format == "Float32"
    assert receipt.source_bounds_parsed == (0.0, 1.0)
    assert receipt.numeric_domain == "NORMALIZED_UNIT"
    assert receipt.normalized_unit_scale == 1.0
    assert receipt.peak_working_set_bound_bytes > 0
    assert _hash(source) == before_hash
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_xisf_bounds_are_strictly_parsed_and_nonunit_float_bounds_fail_closed() -> None:
    domain = _xisf_numeric_domain(
        {"bounds": "0.0:1.0"}, np.dtype(np.float32)
    )
    assert domain == ("NORMALIZED_UNIT", 1.0, "0.0:1.0", (0.0, 1.0))
    for bounds in ("0:1:2", "0", "nan:1", "1:0", "0:0", ":1"):
        with pytest.raises(CalibrationError) as captured:
            _xisf_numeric_domain({"bounds": bounds}, np.dtype(np.float32))
        assert captured.value.code == "XISF_BOUNDS_INVALID"
    with pytest.raises(CalibrationError) as captured:
        _xisf_numeric_domain({"bounds": "0:65535"}, np.dtype(np.float32))
    assert captured.value.code == "XISF_FLOAT_BOUNDS_UNSUPPORTED"


def test_unique_science_primary_allows_explicit_rejection_auxiliaries(tmp_path: Path) -> None:
    source = tmp_path / "multi.xisf"
    _write(source)
    _clone_image(source, image_id="rejection_low", image_type="RejectionMapLow")
    _clone_image(source, image_id="rejection_high", image_type="RejectionMapHigh")

    receipt = convert_xisf_to_fits(source, tmp_path / "out.fits")

    assert receipt.image_id == "integration"
    assert receipt.image_type in {"MasterFlat", "Master Flat"}
    assert receipt.container_image_count == 3
    assert [item["type"] for item in receipt.ignored_auxiliary_images] == [
        "RejectionMapLow",
        "RejectionMapHigh",
    ]


def test_multiple_science_images_and_decode_budget_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.xisf"
    _write(source)
    _clone_image(source, image_id="second", image_type="MasterFlat")
    with pytest.raises(CalibrationError) as raised:
        convert_xisf_to_fits(source, tmp_path / "ambiguous.fits")
    assert raised.value.code == "XISF_SCIENCE_IMAGE_AMBIGUOUS"
    assert not (tmp_path / "ambiguous.fits").exists()

    compressed = tmp_path / "budget.xisf"
    _write(compressed, "zlib")
    with pytest.raises(CalibrationError) as raised:
        convert_xisf_to_fits(
            compressed,
            tmp_path / "budget.fits",
            policy=XisfDecodePolicy(max_decoded_image_bytes=1024),
        )
    assert raised.value.code == "XISF_DECODE_BUDGET_EXCEEDED"
    assert not (tmp_path / "budget.fits").exists()


def test_xml_header_budget_and_dtd_are_rejected_before_parser(tmp_path: Path) -> None:
    source = tmp_path / "malicious.xisf"
    _write(source)
    raw = source.read_bytes()
    header_length = struct.unpack("<I", raw[8:12])[0]
    header = raw[16 : 16 + header_length]
    declaration_end = header.find(b"?>") + 2
    malicious = (
        header[:declaration_end]
        + b'<!DOCTYPE XISF [<!ENTITY boom "expansion">]>'
        + header[declaration_end:]
    )
    source.write_bytes(
        raw[:8]
        + struct.pack("<I", len(malicious))
        + raw[12:16]
        + malicious
        + raw[16 + header_length :]
    )
    with pytest.raises(CalibrationError) as raised:
        convert_xisf_to_fits(source, tmp_path / "dtd.fits")
    assert raised.value.code == "XISF_XML_DTD_FORBIDDEN"

    oversized = tmp_path / "oversized.xisf"
    oversized.write_bytes(
        b"XISF0100" + struct.pack("<I", 16 * 1024**2 + 1) + b"\0\0\0\0" + b"<XISF/>"
    )
    with pytest.raises(CalibrationError) as raised:
        convert_xisf_to_fits(oversized, tmp_path / "oversized.fits")
    assert raised.value.code == "XISF_XML_HEADER_BUDGET"


def test_pipeline_uses_content_bound_override_for_legacy_xisf_master(
    tmp_path: Path,
) -> None:
    header = fits.Header()
    header["INSTRUME"] = "QHY268M"
    header["GAIN"] = 100
    header["OFFSET"] = 50
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "Mode 1"
    header["BAYERPAT"] = "NONE"
    header["FILTER"] = "R"
    header["OBJECT"] = "TEST"
    bias = tmp_path / "bias.fits"
    light = tmp_path / "light.fits"
    bias_header = header.copy()
    bias_header["IMAGETYP"] = "Bias"
    bias_header["EXPTIME"] = 0.001
    fits.writeto(bias, np.full((16, 16), 100, dtype=np.uint16), bias_header)
    light_header = header.copy()
    light_header["IMAGETYP"] = "Light"
    light_header["EXPTIME"] = 120.0
    fits.writeto(light, np.arange(256, dtype=np.uint16).reshape(16, 16) + 1000, light_header)
    master_flat = tmp_path / "legacy-master-flat.xisf"
    XISF.write(
        str(master_flat),
        np.ones((16, 16, 1), dtype=np.float32),
        image_metadata={
            "id": "integration",
            "FITSKeywords": {
                "IMAGETYP": [{"value": "'Master Flat'", "comment": ""}],
                "FILTER": [{"value": "'R'", "comment": ""}],
                "INSTRUME": [{"value": "'QHY268M'", "comment": ""}],
                "XBINNING": [{"value": "1", "comment": ""}],
                "YBINNING": [{"value": "1", "comment": ""}],
            },
        },
    )
    digest = "sha256:" + _hash(master_flat)
    parameters = PipelineParameters(
        master_metadata_overrides=(
            MasterMetadataOverride(
                source_sha256=digest,
                camera="QHY268M",
                gain=100,
                offset=50,
                binning_x=1,
                binning_y=1,
                filter_name="R",
                cfa_pattern="NONE",
                readout_mode="Mode 1",
                temperature_celsius=-10,
                exposure_seconds=1,
            ),
        )
    )

    result = run_portable_pipeline(
        bias_files=(bias,),
        master_flat_files=(master_flat,),
        light_files=(light,),
        output_directory=tmp_path / "pipeline",
        parameters=parameters,
    )

    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["pixelInputStaging"]["conversions"][0]["sourceSha256"] == digest
    assert receipt["masterMetadataOverrides"][0]["sourceSha256"] == digest
    assert receipt["masterMetadataOverrides"][0]["status"] == "APPLIED_CONTENT_BOUND_DECLARATION"
    assert Path(result.master_light_paths[0]).is_file()


def test_uint16_light_uses_explicit_xisf_unit_domain_for_additive_masters(
    tmp_path: Path,
) -> None:
    shape = (16, 16)
    common = {
        "FILTER": "R",
        "OBJECT": "TEST",
        "INSTRUME": "QHY268M",
        "GAIN": 100,
        "OFFSET": 50,
        "XBINNING": 1,
        "YBINNING": 1,
        "READOUTM": "Mode 1",
        "BAYERPAT": "NONE",
        "CCD-TEMP": -10,
    }
    light_header = fits.Header(common)
    light_header["IMAGETYP"] = "Light"
    light_header["EXPTIME"] = 120.0
    light = tmp_path / "light.fits"
    expected_signal = (800 + np.arange(np.prod(shape))).reshape(shape).astype(np.float32)
    fits.writeto(
        light,
        np.asarray(120.0 + expected_signal, dtype=np.uint16),
        light_header,
    )
    master_bias = _write_master_xisf(
        tmp_path / "master-bias.xisf",
        np.full(shape, 100.0 / 65535.0),
        role="Master Bias",
        exposure=0.001,
    )
    master_dark = _write_master_xisf(
        tmp_path / "master-dark.xisf",
        np.full(shape, 120.0 / 65535.0),
        role="Master Dark",
        exposure=120.0,
    )
    master_flat = _write_master_xisf(
        tmp_path / "master-flat.xisf",
        np.ones(shape),
        role="Master Flat",
        exposure=2.0,
    )
    dark_override = MasterMetadataOverride(
        source_sha256="sha256:" + _hash(master_dark),
        camera="QHY268M",
        gain=100,
        offset=50,
        binning_x=1,
        binning_y=1,
        filter_name="R",
        cfa_pattern="NONE",
        readout_mode="Mode 1",
        temperature_celsius=-10,
        exposure_seconds=120,
        bias_included=True,
    )

    result = run_portable_pipeline(
        master_bias_file=master_bias,
        master_dark_files=(master_dark,),
        master_flat_files=(master_flat,),
        light_files=(light,),
        output_directory=tmp_path / "mixed-domain-output",
        parameters=PipelineParameters(
            master_metadata_overrides=(dark_override,),
            global_normalization=GlobalNormalizationParameters(enabled=False),
        ),
    )

    with fits.open(result.master_light_paths[0], memmap=False) as hdul:
        np.testing.assert_allclose(
            hdul[0].data, expected_signal, rtol=0.0, atol=2e-4
        )
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    calibrated = next(
        item for item in receipt["outputs"] if item["kind"] == "CALIBRATED_LIGHT"
    )
    assert calibrated["details"]["sourceNumericDomain"] == "INTEGER_16_PHYSICAL_0_BASED"
    assert calibrated["details"]["additiveNumericDomain"] == "NORMALIZED_UNIT"
    assert calibrated["details"]["additiveApplicationScale"] == 65535.0
    converted = receipt["pixelInputStaging"]["conversions"]
    assert {item["numericDomain"] for item in converted} == {"NORMALIZED_UNIT"}
    assert all(item["sourceSampleFormat"] == "Float32" for item in converted)
    domains = {item["role"]: item for item in receipt["pixelNumericDomains"]}
    assert domains["LIGHT"]["storageEvidence"]["BITPIX"] == 16
    assert domains["LIGHT"]["storageEvidence"]["derivedPhysicalHigh"] == 65535.0
    assert domains["LIGHT"]["canonicalApplicationScale"] == 1.0 / 65535.0
    assert domains["MASTER_DARK"]["storageEvidence"]["sourceSampleFormat"] == "Float32"
    assert domains["MASTER_DARK"]["storageEvidence"]["sourceBounds"] == "0:1"
    assert domains["MASTER_DARK"]["canonicalApplicationScale"] == 1.0

    with pytest.raises(CalibrationError) as captured:
        run_portable_pipeline(
            master_bias_file=master_bias,
            master_dark_files=(master_dark,),
            master_flat_files=(master_flat,),
            light_files=(light,),
            output_directory=tmp_path / "conflicting-domain-output",
            parameters=PipelineParameters(
                master_metadata_overrides=(
                    replace(
                        dark_override,
                        numeric_domain="SENSOR_CODE",
                        normalized_unit_scale=65535.0,
                    ),
                ),
                global_normalization=GlobalNormalizationParameters(enabled=False),
            ),
        )
    assert captured.value.code == "MASTER_NUMERIC_DOMAIN_OVERRIDE_CONFLICT"
