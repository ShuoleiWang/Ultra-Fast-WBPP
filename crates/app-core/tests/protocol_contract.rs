mod common;

use std::collections::BTreeMap;

use openastroflow_app_core::protocol::{
    ErrorMessage, PeerRole, PlanMessage, ProgressMessage, ProgressState, ProtocolCursor,
    ProtocolError,
};
use openastroflow_app_core::{
    HandshakeMessage, HardwareProfile, Project, ProjectSource, ResultRequirement, Validate,
    WorkerEnvelope, WorkerMessage, decode_ndjson_line, encode_ndjson_line,
};
use openastroflow_app_core::{InputRole, WORKER_PROTOCOL_VERSION};

fn controller_handshake(sequence: u64) -> WorkerEnvelope {
    WorkerEnvelope {
        protocol_version: WORKER_PROTOCOL_VERSION,
        session_id: "session-1".to_owned(),
        sequence,
        sent_at_unix_ms: 1,
        message: WorkerMessage::Handshake(HandshakeMessage {
            role: PeerRole::Controller,
            implementation: "openastroflow-gui".to_owned(),
            implementation_version: "0.1.0".to_owned(),
            supported_protocol_versions: vec![WORKER_PROTOCOL_VERSION],
            capabilities: None,
        }),
    }
}

fn plan(sequence: u64) -> WorkerEnvelope {
    WorkerEnvelope {
        protocol_version: WORKER_PROTOCOL_VERSION,
        session_id: "session-1".to_owned(),
        sequence,
        sent_at_unix_ms: 2,
        message: WorkerMessage::Plan(PlanMessage {
            request_id: "request-1".to_owned(),
            plan_id: "plan-1".to_owned(),
            project: Project {
                schema_version: 1,
                project_id: "project-1".to_owned(),
                display_name: "Project One".to_owned(),
                created_at_unix_ms: 1,
                sources: vec![ProjectSource {
                    source_id: "lights".to_owned(),
                    role: InputRole::Light,
                    host_path: "/data/lights".to_owned(),
                    recursive: true,
                    filter: None,
                }],
                labels: BTreeMap::new(),
            },
            recipe: common::recipe(ResultRequirement::Required, ResultRequirement::Disabled),
            requested_hardware_profile: HardwareProfile::GenericArm64Cpu,
            input_manifest_sha256: "a".repeat(64),
        }),
    }
}

fn progress(sequence: u64) -> WorkerEnvelope {
    WorkerEnvelope {
        protocol_version: WORKER_PROTOCOL_VERSION,
        session_id: "session-1".to_owned(),
        sequence,
        sent_at_unix_ms: 3,
        message: WorkerMessage::Progress(ProgressMessage {
            request_id: "request-1".to_owned(),
            run_id: "run-1".to_owned(),
            stage_id: Some("integrate".to_owned()),
            state: ProgressState::Running,
            fraction: 0.5,
            completed_units: Some(1),
            total_units: Some(2),
            message: "working".to_owned(),
        }),
    }
}

#[test]
fn each_message_is_exactly_one_ndjson_record() {
    let envelope = controller_handshake(0);
    let encoded = encode_ndjson_line(&envelope).expect("encode");
    assert_eq!(encoded.last(), Some(&b'\n'));
    assert!(!encoded[..encoded.len() - 1].contains(&b'\n'));
    assert_eq!(decode_ndjson_line(&encoded).expect("decode"), envelope);
}

#[test]
fn decoder_rejects_wrong_protocol_version() {
    let mut envelope = controller_handshake(0);
    envelope.protocol_version = 2;
    let bytes = serde_json::to_vec(&envelope).expect("serialize fixture");
    assert!(matches!(
        decode_ndjson_line(&bytes),
        Err(ProtocolError::InvalidMessage(_))
    ));
}

#[test]
fn decoder_rejects_multiple_records() {
    let record = serde_json::to_vec(&controller_handshake(0)).expect("serialize fixture");
    let mut doubled = record.clone();
    doubled.push(b'\n');
    doubled.extend(record);
    assert!(matches!(
        decode_ndjson_line(&doubled),
        Err(ProtocolError::MultipleRecords)
    ));
}

#[test]
fn cursor_requires_handshake_and_contiguous_sequence() {
    let mut cursor = ProtocolCursor::default();
    assert!(matches!(
        cursor.accept(&plan(0)),
        Err(ProtocolError::HandshakeRequired)
    ));
    cursor
        .accept(&controller_handshake(0))
        .expect("first handshake");
    assert!(matches!(
        cursor.accept(&plan(2)),
        Err(ProtocolError::SequenceMismatch {
            expected: 1,
            actual: 2
        })
    ));
    cursor.accept(&plan(1)).expect("next plan");
}

#[test]
fn cursor_rejects_messages_in_the_wrong_peer_direction() {
    let mut cursor = ProtocolCursor::default();
    cursor
        .accept(&controller_handshake(0))
        .expect("controller handshake");
    assert!(matches!(
        cursor.accept(&progress(1)),
        Err(ProtocolError::MessageDirection {
            role: PeerRole::Controller,
            kind: "progress"
        })
    ));
}

#[test]
fn worker_handshake_requires_capabilities() {
    let handshake = HandshakeMessage {
        role: PeerRole::Worker,
        implementation: "worker".to_owned(),
        implementation_version: "0.1.0".to_owned(),
        supported_protocol_versions: vec![1],
        capabilities: None,
    };
    assert!(handshake.validate().is_err());
}

#[test]
fn progress_units_and_fraction_are_validated() {
    let message = ProgressMessage {
        request_id: "request-1".to_owned(),
        run_id: "run-1".to_owned(),
        stage_id: Some("integrate".to_owned()),
        state: ProgressState::Running,
        fraction: 1.2,
        completed_units: Some(3),
        total_units: Some(2),
        message: "working".to_owned(),
    };
    assert!(message.validate().is_err());
}

#[test]
fn unknown_fields_are_rejected() {
    let mut value = serde_json::to_value(controller_handshake(0)).expect("serialize");
    value
        .as_object_mut()
        .expect("object")
        .insert("surprise".to_owned(), serde_json::json!(true));
    let bytes = serde_json::to_vec(&value).expect("serialize");
    assert!(decode_ndjson_line(&bytes).is_err());
}

#[test]
fn error_code_is_a_stable_identifier() {
    let error = ErrorMessage {
        request_id: None,
        run_id: None,
        stage_id: None,
        code: "path escaped".to_owned(),
        message: "bad path".to_owned(),
        retryable: false,
        details: BTreeMap::new(),
    };
    assert!(error.validate().is_err());
}
