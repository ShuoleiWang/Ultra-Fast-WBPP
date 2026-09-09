from __future__ import annotations

import json
import math

import pytest

from lightframeqc.pixinsight import (
    PixInsightImportError,
    SUBFRAME_SELECTOR_V3_COLUMNS,
    load_subframe_selector_v3_json,
    loads_subframe_selector_v3_json,
    parse_subframe_selector_v3,
)


def _row(index: int, path: str) -> list[object]:
    row: list[object] = [0.0] * len(SUBFRAME_SELECTOR_V3_COLUMNS)
    row[0] = index
    row[1] = True
    row[2] = False
    row[3] = path
    row[4] = 0.0  # process weight: validated, intentionally not exposed
    row[5] = 3.25
    row[6] = 0.42
    row[7] = 17.5
    row[8] = 0.0  # unused01: validated, intentionally not exposed
    row[9] = 12.75
    row[10] = 0.0018
    row[11] = 0.0008
    row[12] = 0.0002
    row[13] = 0.031
    row[14] = 2_345
    row[15] = 0.0003
    row[16] = 0.25
    row[17] = 0.10
    row[18] = 0.00007
    row[19] = 115.0
    row[20] = 31.0
    row[21] = 12_368.0
    row[22] = 99_975.0
    row[23] = 101.0
    row[24] = 0.58
    row[25] = 2_000
    row[26] = 0.00019
    row[27] = 0.00026
    row[28] = 99.2
    row[29] = 0.0
    row[30] = 0.0
    return row


def _payload(paths: list[str] | None = None) -> dict[str, object]:
    if paths is None:
        paths = ["/data/light-0001.xisf", "/data/light-0002.xisf"]
    return {
        "schemaVersion": 1,
        "processId": "SubframeSelector",
        "processVersion": 3,
        "routine": "MeasureSubframes",
        "fileCache": False,
        "ok": True,
        "wallSeconds": 1.25,
        "requestPaths": paths,
        "measurementRowCount": len(paths),
        "measurements": [_row(index, path) for index, path in enumerate(paths)],
    }


def test_imports_complete_v3_result_and_exposes_only_safe_fields() -> None:
    imported = parse_subframe_selector_v3(_payload())

    assert imported.schema_version == 1
    assert imported.process_version == 3
    assert imported.request_paths == (
        "/data/light-0001.xisf",
        "/data/light-0002.xisf",
    )
    assert len(imported) == 2

    first = imported.measurements[0]
    assert first.index == 0
    assert first.path == "/data/light-0001.xisf"
    assert first.fwhm == pytest.approx(3.25)
    assert first.eccentricity == pytest.approx(0.42)
    assert first.psf_signal_weight == pytest.approx(17.5)
    assert first.snr_weight == pytest.approx(12.75)
    assert first.median == pytest.approx(0.0018)
    assert first.noise == pytest.approx(0.0002)
    assert first.stars == 2_345
    assert first.altitude == pytest.approx(31.0)
    assert first.azimuth == pytest.approx(115.0)
    assert first.psf_flux == pytest.approx(12_368.0)
    assert first.psf_snr == pytest.approx(99.2)
    assert imported.by_path("/data/light-0002.xisf").index == 1

    features = first.as_features()
    assert "path" not in features
    assert "index" not in features
    assert "weight" not in features
    assert "unused01" not in features
    assert features["stars"] == 2_345


def test_loads_bytes_and_reads_file_without_modifying_it(tmp_path) -> None:
    document = json.dumps(_payload(), separators=(",", ":")).encode("utf-8")
    source = tmp_path / "subframes.json"
    source.write_bytes(document)
    before = source.stat()

    loaded_from_bytes = loads_subframe_selector_v3_json(document)
    loaded_from_file = load_subframe_selector_v3_json(source)

    after = source.stat()
    assert loaded_from_file == loaded_from_bytes
    assert source.read_bytes() == document
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schemaVersion", 2),
        ("processId", "OtherProcess"),
        ("processVersion", 4),
        ("routine", "OutputSubframes"),
        ("ok", False),
    ],
)
def test_rejects_wrong_or_failed_producer_identity(key: str, value: object) -> None:
    payload = _payload()
    payload[key] = value

    with pytest.raises(PixInsightImportError):
        parse_subframe_selector_v3(payload)


def test_rejects_missing_required_schema_field() -> None:
    payload = _payload()
    del payload["processVersion"]

    with pytest.raises(PixInsightImportError, match="missing required field"):
        parse_subframe_selector_v3(payload)


def test_rejects_incomplete_or_extra_measurement_columns() -> None:
    for delta in (-1, 1):
        payload = _payload(["/data/light.xisf"])
        row = payload["measurements"][0]  # type: ignore[index]
        if delta < 0:
            row.pop()  # type: ignore[union-attr]
        else:
            row.append(0.0)  # type: ignore[union-attr]

        with pytest.raises(PixInsightImportError, match="exactly 31 columns"):
            parse_subframe_selector_v3(payload)


def test_rejects_noncanonical_index_and_path_binding() -> None:
    wrong_index = _payload()
    wrong_index["measurements"][1][0] = 0  # type: ignore[index]
    with pytest.raises(PixInsightImportError, match="row index"):
        parse_subframe_selector_v3(wrong_index)

    wrong_path = _payload()
    wrong_path["measurements"][0][3] = "/data/other.xisf"  # type: ignore[index]
    with pytest.raises(PixInsightImportError, match="does not match"):
        parse_subframe_selector_v3(wrong_path)

    missing_row = _payload()
    missing_row["measurements"].pop()  # type: ignore[union-attr]
    missing_row["measurementRowCount"] = 1
    with pytest.raises(PixInsightImportError, match="cover requestPaths exactly"):
        parse_subframe_selector_v3(missing_row)


@pytest.mark.parametrize(
    "paths",
    [
        ["relative/light.xisf"],
        ["/data/../private/light.xisf"],
        ["/data/light.xisf", "/data/light.xisf"],
        ["/data/light\x00.xisf"],
    ],
)
def test_rejects_unsafe_or_duplicate_paths(paths: list[str]) -> None:
    with pytest.raises(PixInsightImportError):
        parse_subframe_selector_v3(_payload(paths))


def test_accepts_windows_absolute_paths_as_opaque_identities() -> None:
    path = r"C:\data\light-0001.xisf"
    imported = parse_subframe_selector_v3(_payload([path]))

    assert imported.measurements[0].path == path


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        (0, True),
        (1, 1),
        (5, "3.2"),
        (5, math.nan),
        (12, math.inf),
        (12, 10**1_000),
        (14, 0),
        (14, 3.5),
        (25, -1),
    ],
)
def test_rejects_type_confusion_nonfinite_and_invalid_integer_cells(
    column: int,
    bad_value: object,
) -> None:
    payload = _payload(["/data/light.xisf"])
    payload["measurements"][0][column] = bad_value  # type: ignore[index]

    with pytest.raises(PixInsightImportError):
        parse_subframe_selector_v3(payload)


def test_json_decoder_rejects_duplicate_keys_and_nonstandard_numbers() -> None:
    with pytest.raises(PixInsightImportError, match="duplicate key"):
        loads_subframe_selector_v3_json('{"schemaVersion":1,"schemaVersion":1}')

    document = json.dumps(_payload(["/data/light.xisf"])).replace("3.25", "NaN", 1)
    with pytest.raises(PixInsightImportError, match="nonstandard JSON number"):
        loads_subframe_selector_v3_json(document)


def test_row_count_metadata_must_match() -> None:
    payload = _payload()
    payload["measurementRowCount"] = 99

    with pytest.raises(PixInsightImportError, match="measurementRowCount"):
        parse_subframe_selector_v3(payload)
