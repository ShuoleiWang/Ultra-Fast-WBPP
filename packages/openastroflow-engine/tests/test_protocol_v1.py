from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from openastroflow_engine.protocol_v1 import (
    ProtocolCursor,
    ProtocolV1Error,
    WorkerEnvelope,
    decode_ndjson_line,
    encode_ndjson_line,
)


REPOSITORY = Path(__file__).resolve().parents[3]
PROTOCOL = REPOSITORY / "protocol"


def _records(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (PROTOCOL / "examples" / name).read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_checked_in_schema_and_all_protocol_examples_round_trip() -> None:
    schema = json.loads(
        (PROTOCOL / "schema" / "worker-envelope-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["x-openastroflow-schema-version"] == 1
    assert "additionalProperties" not in schema
    assert set(schema["propertyNames"]["enum"]) == {
        "protocolVersion",
        "sessionId",
        "sequence",
        "sentAtUnixMs",
        "type",
        "payload",
    }
    assert {variant["properties"]["type"]["enum"][0] for variant in schema["oneOf"]} == {
        "handshake",
        "plan",
        "execute",
        "progress",
        "artifact",
        "error",
    }
    astrometry = schema["definitions"]["AstrometricSolutionReceipt"]
    assert set(astrometry["required"]) >= {
        "matchedStars",
        "rmsPixels",
        "rmsArcsec",
        "parity",
        "catalogIdentity",
        "indexIdentities",
        "correspondenceSha256",
        "catalogManaged",
        "installedSetIdentity",
        "catalogManifestSha256",
        "indexArtifacts",
    }

    for fixture in (
        "controller-to-worker-v1.ndjson",
        "worker-to-controller-v1.ndjson",
    ):
        for raw in _records(fixture):
            decoded = decode_ndjson_line(json.dumps(raw, separators=(",", ":")) + "\n")
            assert decoded.to_dict() == raw
            assert json.loads(encode_ndjson_line(decoded)) == raw


def test_cursor_requires_handshake_session_direction_and_contiguous_sequence() -> None:
    controller = _records("controller-to-worker-v1.ndjson")
    cursor = ProtocolCursor()
    with pytest.raises(ProtocolV1Error) as error:
        cursor.accept(controller[1])
    assert error.value.code == "handshake-required"

    cursor.accept(controller[0])
    skipped = deepcopy(controller[1])
    skipped["sequence"] = 2
    with pytest.raises(ProtocolV1Error) as error:
        cursor.accept(skipped)
    assert error.value.code == "sequence-mismatch"

    cursor.accept(controller[1])
    wrong_session = deepcopy(controller[2])
    wrong_session["sessionId"] = "different-session"
    with pytest.raises(ProtocolV1Error) as error:
        cursor.accept(wrong_session)
    assert error.value.code == "session-mismatch"

    worker_progress = _records("worker-to-controller-v1.ndjson")[1]
    wrong_direction = deepcopy(worker_progress)
    wrong_direction["sessionId"] = "session-1"
    wrong_direction["sequence"] = 2
    with pytest.raises(ProtocolV1Error) as error:
        cursor.accept(wrong_direction)
    assert error.value.code == "direction-mismatch"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update({"surprise": True}), "unknown-field"),
        (
            lambda value: value["payload"].update({"surprise": True}),
            "unknown-field",
        ),
        (
            lambda value: value["payload"]["project"].update({"surprise": True}),
            "unknown-field",
        ),
        (
            lambda value: value["payload"]["recipe"]["solver"].update(
                {"surprise": True}
            ),
            "unknown-field",
        ),
    ],
)
def test_unknown_typed_fields_fail_closed(mutation, code: str) -> None:
    value = deepcopy(_records("controller-to-worker-v1.ndjson")[1])
    mutation(value)
    with pytest.raises(ProtocolV1Error) as error:
        WorkerEnvelope.from_mapping(value)
    assert error.value.code == code


def test_progress_nonfinite_and_unit_invariants_fail_closed() -> None:
    progress = deepcopy(_records("worker-to-controller-v1.ndjson")[1])
    progress["payload"]["fraction"] = float("nan")
    with pytest.raises(ProtocolV1Error) as error:
        WorkerEnvelope.from_mapping(progress)
    assert error.value.code == "non-finite-number"

    progress = deepcopy(_records("worker-to-controller-v1.ndjson")[1])
    progress["payload"]["completedUnits"] = 21
    progress["payload"]["totalUnits"] = 20
    with pytest.raises(ProtocolV1Error, match="completed <= total"):
        WorkerEnvelope.from_mapping(progress)


def test_native_windows_paths_are_opaque_but_artifact_identity_is_portable() -> None:
    windows_home = "C:" + r"\Users\Alice"
    execute = deepcopy(_records("controller-to-worker-v1.ndjson")[2])
    execute["payload"]["outputParentHostPath"] = windows_home + r"\OpenAstroFlow"
    assert WorkerEnvelope.from_mapping(execute).payload["outputParentHostPath"].startswith(
        "C:"
    )

    artifact = deepcopy(_records("worker-to-controller-v1.ndjson")[2])
    artifact["payload"]["artifact"]["relativePath"] = windows_home + r"\master.fits"
    with pytest.raises(ProtocolV1Error) as error:
        WorkerEnvelope.from_mapping(artifact)
    assert error.value.code == "invalid-relative-path"

    artifact["payload"]["artifact"]["relativePath"] = "/tmp/master.fits"
    with pytest.raises(ProtocolV1Error) as error:
        WorkerEnvelope.from_mapping(artifact)
    assert error.value.code == "invalid-relative-path"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.pop("rmsPixels"), "missing-field"),
        (lambda value: value.update({"rmsPixels": float("nan")}), "non-finite-number"),
        (lambda value: value.update({"rmsPixels": 20.0}), "invalid-value"),
        (lambda value: value.update({"parity": "FLIPPED"}), "invalid-enum"),
        (lambda value: value.update({"catalogIdentity": "bad"}), "invalid-sha256"),
        (lambda value: value.update({"indexIdentities": []}), "invalid-value"),
        (
            lambda value: value.update({"indexIdentities": ["same", "same"]}),
            "duplicate-value",
        ),
        (
            lambda value: value.update({"correspondenceSha256": "bad"}),
            "invalid-sha256",
        ),
        (lambda value: value.update({"catalogManaged": False}), "invalid-value"),
        (lambda value: value.update({"installedSetIdentity": "bad"}), "invalid-sha256"),
        (lambda value: value.update({"indexArtifacts": []}), "invalid-value"),
    ],
)
def test_astrometric_quality_receipt_fields_fail_closed(mutation, code: str) -> None:
    artifact = deepcopy(_records("worker-to-controller-v1.ndjson")[2])
    mutation(artifact["payload"]["artifact"]["astrometry"])
    with pytest.raises(ProtocolV1Error) as error:
        WorkerEnvelope.from_mapping(artifact)
    assert error.value.code == code


def test_decoder_rejects_multiple_records_and_nonstandard_json_numbers() -> None:
    handshake = _records("controller-to-worker-v1.ndjson")[0]
    line = json.dumps(handshake)
    with pytest.raises(ProtocolV1Error) as error:
        decode_ndjson_line(line + "\n" + line)
    assert error.value.code == "multiple-records"

    invalid = line.replace("1788220800000", "NaN")
    with pytest.raises(ProtocolV1Error) as error:
        decode_ndjson_line(invalid)
    assert error.value.code == "invalid-json"

    duplicate = line.replace('"protocolVersion": 1', '"protocolVersion": 1, "protocolVersion": 1')
    with pytest.raises(ProtocolV1Error, match="duplicate JSON object key") as error:
        decode_ndjson_line(duplicate)
    assert error.value.code == "invalid-json"
