use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use openastroflow_app_core::decode_ndjson_line;
use openastroflow_app_core::protocol::ProtocolCursor;
use tempfile::TempDir;

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives at crates/app-core")
        .to_path_buf()
}

#[test]
fn checked_in_schemas_match_the_rust_models() {
    let temporary = TempDir::new().expect("temporary schema directory");
    let status = Command::new(env!("CARGO_BIN_EXE_export-openastroflow-schemas"))
        .arg(temporary.path())
        .status()
        .expect("run schema exporter");
    assert!(status.success());

    for name in [
        "worker-envelope-v1.schema.json",
        "project-v1.schema.json",
        "recipe-v1.schema.json",
        "project-receipt-v1.schema.json",
        "recipe-receipt-v1.schema.json",
        "stage-receipt-v1.schema.json",
        "artifact-receipt-v1.schema.json",
        "run-receipt-v1.schema.json",
        "backend-capabilities-v1.schema.json",
        "publication-v1.schema.json",
        "result-gate-v1.schema.json",
    ] {
        let generated = std::fs::read(temporary.path().join(name)).expect("generated schema");
        let checked_in = std::fs::read(repository_root().join("protocol/schema").join(name))
            .expect("checked-in schema");
        assert_eq!(generated, checked_in, "schema drift: {name}");
    }
}

#[test]
fn worker_schema_models_the_flattened_wire_keys_strictly() {
    let path = repository_root().join("protocol/schema/worker-envelope-v1.schema.json");
    let schema: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("worker schema"))
            .expect("valid worker schema JSON");
    assert!(schema.get("additionalProperties").is_none());
    assert_eq!(
        schema["propertyNames"]["enum"],
        serde_json::json!([
            "protocolVersion",
            "sessionId",
            "sequence",
            "sentAtUnixMs",
            "type",
            "payload"
        ])
    );
}

#[test]
fn astrometry_schema_requires_catalog_correspondence_quality_evidence() {
    let path = repository_root().join("protocol/schema/artifact-receipt-v1.schema.json");
    let schema: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("artifact schema"))
            .expect("valid artifact schema JSON");
    let required = schema["definitions"]["AstrometricSolutionReceipt"]["required"]
        .as_array()
        .expect("required fields");
    for field in [
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
    ] {
        assert!(
            required.iter().any(|value| value.as_str() == Some(field)),
            "missing {field}"
        );
    }
    assert_eq!(
        schema["definitions"]["AstrometricParity"]["enum"],
        serde_json::json!(["POSITIVE", "NEGATIVE"])
    );
    assert_eq!(
        schema["definitions"]["AstrometricSolutionReceipt"]["properties"]["catalogManaged"]["const"],
        serde_json::json!(true)
    );
}

#[test]
fn ndjson_examples_are_valid_ordered_streams_and_cover_every_message() {
    let mut kinds = BTreeSet::new();
    for name in [
        "controller-to-worker-v1.ndjson",
        "worker-to-controller-v1.ndjson",
    ] {
        let content =
            std::fs::read_to_string(repository_root().join("protocol/examples").join(name))
                .expect("protocol example");
        let mut cursor = ProtocolCursor::default();
        for line in content.lines() {
            let envelope = decode_ndjson_line(line.as_bytes()).expect("valid example line");
            kinds.insert(envelope.message.kind());
            cursor.accept(&envelope).expect("valid example sequence");
        }
    }
    assert_eq!(
        kinds,
        BTreeSet::from([
            "artifact",
            "error",
            "execute",
            "handshake",
            "plan",
            "progress",
        ])
    );
}
