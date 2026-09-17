from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from lightframeqc.xisf import BLOCK_ALIGNMENT, SIGNATURE, XISF, XisfError


def _image(dtype: type, channels: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.uniform(0.0, 1000.0, size=(37, 53, channels))
    if np.issubdtype(dtype, np.integer):
        return values.astype(dtype)
    return (values / 1000.0).astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.float32, np.float64])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize(
    ("codec", "shuffle"),
    [(None, False), ("zlib", False), ("zlib", True), ("lz4", False), ("lz4", True), ("lz4hc", True), ("zstd", False), ("zstd", True)],
)
def test_round_trip_every_codec_and_sample_format(
    tmp_path: Path, dtype: type, channels: int, codec: str | None, shuffle: bool
) -> None:
    image = _image(dtype, channels)
    path = tmp_path / "image.xisf"
    written, used = XISF.write(
        path,
        image,
        codec=codec,
        shuffle=shuffle,
        image_metadata={
            "id": "integration",
            "imageType": "Light",
            "FITSKeywords": {"OBJECT": [{"value": "'M31'", "comment": "target"}], "EXPTIME": [{"value": "300", "comment": ""}]},
            "XISFProperties": {"Observation:Object:Name": {"type": "String", "value": "M31"}},
        },
        xisf_metadata={"Instrument:Camera:Gain": {"type": "Float32", "value": 1.5}},
    )
    assert written == path.stat().st_size
    if codec is None:
        assert used is None
    else:
        assert used == (f"{codec}+sh" if shuffle and image.dtype.itemsize > 1 else codec)

    document = XISF(path)
    (metadata,) = document.get_images_metadata()
    assert metadata["geometry"] == (53, 37, channels)
    assert metadata["dtype"] == image.dtype
    assert metadata["location"][0] == "attachment"
    assert metadata["location"][1] % BLOCK_ALIGNMENT == 0
    if codec is None:
        assert "compression" not in metadata
    else:
        assert metadata["compression"][0] == used
        assert metadata["compression"][1] == image.nbytes
    # One pair of quotes is removed from keyword values; numeric properties are typed.
    assert metadata["FITSKeywords"]["OBJECT"] == [{"value": "M31", "comment": "target"}]
    assert metadata["XISFProperties"]["Observation:Object:Name"]["value"] == "M31"
    assert document.get_file_metadata()["Instrument:Camera:Gain"]["value"] == 1.5
    decoded = document.read_image(0)
    assert decoded.dtype == image.dtype and decoded.shape == image.shape
    assert np.array_equal(decoded, image)
    assert np.array_equal(document.read_image(0, data_format="channels_first"), np.moveaxis(image, -1, 0))


def _rewrite_header(path: Path, transform, payload: bytes | None = None) -> None:
    data = path.read_bytes()
    (length,) = struct.unpack("<I", data[8:12])
    header = data[16 : 16 + length]
    position = int(XISF(path).get_images_metadata()[0]["location"][1])
    block = data[position:] if payload is None else payload
    header = transform(header)
    new_position = -(-(16 + len(header)) // BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT
    header = header.replace(b"attachment:%020d" % position, b"attachment:%020d" % new_position)
    path.write_bytes(
        SIGNATURE + struct.pack("<I", len(header)) + b"\0\0\0\0" + header + b"\0" * (new_position - 16 - len(header)) + block
    )


def test_big_endian_samples_and_interleaved_storage(tmp_path: Path) -> None:
    image = ((np.arange(20 * 30 * 2, dtype=np.uint32).reshape(20, 30, 2) * 37) % 65535).astype(np.uint16)
    path = tmp_path / "be.xisf"
    XISF.write(path, image)
    _rewrite_header(
        path,
        lambda header: header.replace(b'sampleFormat="UInt16"', b'sampleFormat="UInt16" byteOrder="big"'),
        payload=np.ascontiguousarray(image.transpose(2, 0, 1)).astype(">u2").tobytes(),
    )
    decoded = XISF(path).read_image(0)
    assert decoded.dtype == np.dtype(">u2")
    assert np.array_equal(decoded, image)

    path = tmp_path / "normal.xisf"
    XISF.write(path, image)
    _rewrite_header(
        path,
        lambda header: header.replace(b"geometry=", b'pixelStorage="normal" geometry='),
        payload=np.ascontiguousarray(image).astype("<u2").tobytes(),
    )
    assert np.array_equal(XISF(path).read_image(0), image)


def test_inline_vector_properties_are_decoded(tmp_path: Path) -> None:
    image = _image(np.float32, 1)
    path = tmp_path / "vector.xisf"
    XISF.write(path, image, image_metadata={"FITSKeywords": {"OBJECT": [{"value": "M31", "comment": ""}]}})
    payload = np.asarray([0.02, 300.0], dtype="<f8").tobytes()
    import base64

    encoded = base64.b64encode(payload).decode("ascii").encode("ascii")
    _rewrite_header(
        path,
        lambda header: header.replace(
            b"</Image>",
            b'<Property id="PCL:TotalExposureTime" type="F64Vector" length="2" location="inline:base64">' + encoded + b"</Property></Image>",
        ),
    )
    entry = XISF(path).get_images_metadata()[0]["XISFProperties"]["PCL:TotalExposureTime"]
    assert entry["length"] == 2 and entry["location"] == ["inline", "base64"]
    assert np.array_equal(entry["value"], [0.02, 300.0])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"NOTXISF!" + b"\0" * 24, "not a monolithic"),
        (SIGNATURE + struct.pack("<I", 40) + b"\0\0\0\0" + b"<xisf version=\"1.0\">", "truncated"),
        (SIGNATURE + struct.pack("<I", 70) + b"\0\0\0\0" + b'<!DOCTYPE x [<!ENTITY e "x">]><xisf version="1.0"></xisf>'.ljust(70), "DTD"),
        (SIGNATURE + struct.pack("<I", 24) + b"\0\0\0\0" + b'<xisf version="1.0"></x>', "well-formed"),
    ],
)
def test_malformed_headers_fail_closed(tmp_path: Path, payload: bytes, message: str) -> None:
    path = tmp_path / "bad.xisf"
    path.write_bytes(payload)
    with pytest.raises(XisfError, match=message):
        XISF(path)


def test_declared_size_mismatches_are_rejected(tmp_path: Path) -> None:
    image = _image(np.float32, 1)
    path = tmp_path / "short.xisf"
    XISF.write(path, image, codec="zlib")
    _rewrite_header(path, lambda header: header.replace(b"compression=\"zlib:%d\"" % image.nbytes, b"compression=\"zlib:%d\"" % (image.nbytes - 4)))
    with pytest.raises(XisfError):
        XISF(path).read_image(0)
