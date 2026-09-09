use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend::{BackendCapabilities, HardwareProfile};
use crate::fs::SafeFileName;
use crate::model::{ArtifactReceipt, Project, Recipe, StageReceipt};
use crate::validation::{Validate, ValidationError, validate_identifier, validate_sha256};

pub const WORKER_PROTOCOL_VERSION: u16 = 1;
pub const MAX_NDJSON_LINE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerEnvelope {
    pub protocol_version: u16,
    pub session_id: String,
    pub sequence: u64,
    pub sent_at_unix_ms: u64,
    #[serde(flatten)]
    pub message: WorkerMessage,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "type", content = "payload", rename_all = "kebab-case")]
// Keeping payloads inline preserves the straightforward public construction API
// while envelopes are short-lived at the NDJSON boundary; scientific image data
// never enters this enum.
#[allow(clippy::large_enum_variant)]
pub enum WorkerMessage {
    Handshake(HandshakeMessage),
    Plan(PlanMessage),
    Execute(ExecuteMessage),
    Progress(ProgressMessage),
    Artifact(ArtifactMessage),
    Error(ErrorMessage),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PeerRole {
    Controller,
    Worker,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HandshakeMessage {
    pub role: PeerRole,
    pub implementation: String,
    pub implementation_version: String,
    pub supported_protocol_versions: Vec<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<BackendCapabilities>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlanMessage {
    pub request_id: String,
    pub plan_id: String,
    pub project: Project,
    pub recipe: Recipe,
    pub requested_hardware_profile: HardwareProfile,
    pub input_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecuteMessage {
    pub request_id: String,
    pub plan_id: String,
    pub run_id: String,
    /// Existing native host directory in which a new result directory is claimed.
    pub output_parent_host_path: String,
    pub output_directory_name: SafeFileName,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_from_run_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProgressState {
    Queued,
    Running,
    Finalizing,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProgressMessage {
    pub request_id: String,
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_id: Option<String>,
    pub state: ProgressState,
    pub fraction: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_units: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_units: Option<u64>,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactMessage {
    pub request_id: String,
    pub run_id: String,
    pub stage: StageReceipt,
    pub artifact: ArtifactReceipt,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ErrorMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_id: Option<String>,
    pub code: String,
    pub message: String,
    pub retryable: bool,
    #[serde(default)]
    pub details: BTreeMap<String, serde_json::Value>,
}

impl WorkerMessage {
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Handshake(_) => "handshake",
            Self::Plan(_) => "plan",
            Self::Execute(_) => "execute",
            Self::Progress(_) => "progress",
            Self::Artifact(_) => "artifact",
            Self::Error(_) => "error",
        }
    }
}

impl Validate for WorkerEnvelope {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.protocol_version != WORKER_PROTOCOL_VERSION {
            return Err(ValidationError::new(
                "protocolVersion",
                format!(
                    "unsupported worker protocol {}; expected {WORKER_PROTOCOL_VERSION}",
                    self.protocol_version
                ),
            ));
        }
        validate_identifier("sessionId", &self.session_id)?;
        self.message.validate()
    }
}

impl Validate for WorkerMessage {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Handshake(message) => message.validate(),
            Self::Plan(message) => message.validate(),
            Self::Execute(message) => message.validate(),
            Self::Progress(message) => message.validate(),
            Self::Artifact(message) => message.validate(),
            Self::Error(message) => message.validate(),
        }
    }
}

impl Validate for HandshakeMessage {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_identifier("handshake.implementation", &self.implementation)?;
        if self.implementation_version.trim().is_empty() {
            return Err(ValidationError::new(
                "handshake.implementationVersion",
                "must not be blank",
            ));
        }
        if !self
            .supported_protocol_versions
            .contains(&WORKER_PROTOCOL_VERSION)
        {
            return Err(ValidationError::new(
                "handshake.supportedProtocolVersions",
                format!("must include {WORKER_PROTOCOL_VERSION}"),
            ));
        }
        match (self.role, &self.capabilities) {
            (PeerRole::Worker, Some(capabilities)) => capabilities.validate(),
            (PeerRole::Worker, None) => Err(ValidationError::new(
                "handshake.capabilities",
                "worker handshake must advertise backend capabilities",
            )),
            (PeerRole::Controller, Some(_)) => Err(ValidationError::new(
                "handshake.capabilities",
                "controller handshake must not claim worker capabilities",
            )),
            (PeerRole::Controller, None) => Ok(()),
        }
    }
}

impl Validate for PlanMessage {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_identifier("plan.requestId", &self.request_id)?;
        validate_identifier("plan.planId", &self.plan_id)?;
        validate_sha256("plan.inputManifestSha256", &self.input_manifest_sha256)?;
        self.project.validate()?;
        self.recipe.validate()
    }
}

impl Validate for ExecuteMessage {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_identifier("execute.requestId", &self.request_id)?;
        validate_identifier("execute.planId", &self.plan_id)?;
        validate_identifier("execute.runId", &self.run_id)?;
        if let Some(resume) = &self.resume_from_run_id {
            validate_identifier("execute.resumeFromRunId", resume)?;
            if resume == &self.run_id {
                return Err(ValidationError::new(
                    "execute.resumeFromRunId",
                    "must differ from the new run identifier",
                ));
            }
        }
        if self.output_parent_host_path.trim().is_empty()
            || self.output_parent_host_path.contains('\0')
        {
            return Err(ValidationError::new(
                "execute.outputParentHostPath",
                "must be a non-empty native path without NUL characters",
            ));
        }
        Ok(())
    }
}

impl Validate for ProgressMessage {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_identifier("progress.requestId", &self.request_id)?;
        validate_identifier("progress.runId", &self.run_id)?;
        if let Some(stage_id) = &self.stage_id {
            validate_identifier("progress.stageId", stage_id)?;
        }
        if !self.fraction.is_finite() || !(0.0..=1.0).contains(&self.fraction) {
            return Err(ValidationError::new(
                "progress.fraction",
                "must be finite and in [0, 1]",
            ));
        }
        match (self.completed_units, self.total_units) {
            (None, None) => {}
            (Some(completed), Some(total)) if total > 0 && completed <= total => {}
            _ => {
                return Err(ValidationError::new(
                    "progress.completedUnits",
                    "completed/total units must be both absent or satisfy 0 <= completed <= total",
                ));
            }
        }
        if self.message.trim().is_empty() {
            return Err(ValidationError::new(
                "progress.message",
                "must not be blank",
            ));
        }
        Ok(())
    }
}

impl Validate for ArtifactMessage {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_identifier("artifact.requestId", &self.request_id)?;
        validate_identifier("artifact.runId", &self.run_id)?;
        self.stage.validate()?;
        self.artifact.validate()?;
        if self.stage.stage_id != self.artifact.produced_by_stage_id {
            return Err(ValidationError::new(
                "artifact.artifact.producedByStageId",
                "must match the attached stage receipt",
            ));
        }
        if !self.stage.artifact_ids.contains(&self.artifact.artifact_id) {
            return Err(ValidationError::new(
                "artifact.stage.artifactIds",
                "must include the attached artifact identifier",
            ));
        }
        Ok(())
    }
}

impl Validate for ErrorMessage {
    fn validate(&self) -> Result<(), ValidationError> {
        if let Some(request_id) = &self.request_id {
            validate_identifier("error.requestId", request_id)?;
        }
        if let Some(run_id) = &self.run_id {
            validate_identifier("error.runId", run_id)?;
        }
        if let Some(stage_id) = &self.stage_id {
            validate_identifier("error.stageId", stage_id)?;
        }
        validate_identifier("error.code", &self.code)?;
        if self.message.trim().is_empty() {
            return Err(ValidationError::new("error.message", "must not be blank"));
        }
        Ok(())
    }
}

/// Validate and encode one envelope, including its trailing newline.
///
/// # Errors
///
/// Returns [`ProtocolError`] for an invalid envelope, serialization failure, or
/// an oversized record.
pub fn encode_ndjson_line(envelope: &WorkerEnvelope) -> Result<Vec<u8>, ProtocolError> {
    envelope.validate().map_err(ProtocolError::InvalidMessage)?;
    let mut encoded = serde_json::to_vec(envelope).map_err(ProtocolError::Json)?;
    if encoded.len() + 1 > MAX_NDJSON_LINE_BYTES {
        return Err(ProtocolError::LineTooLong(encoded.len() + 1));
    }
    encoded.push(b'\n');
    Ok(encoded)
}

/// Decode and validate exactly one NDJSON record.
///
/// # Errors
///
/// Returns [`ProtocolError`] for invalid framing, JSON, version, or message
/// invariants.
pub fn decode_ndjson_line(line: &[u8]) -> Result<WorkerEnvelope, ProtocolError> {
    if line.len() > MAX_NDJSON_LINE_BYTES {
        return Err(ProtocolError::LineTooLong(line.len()));
    }
    let record = line
        .strip_suffix(b"\r\n")
        .or_else(|| line.strip_suffix(b"\n"))
        .unwrap_or(line);
    if record.is_empty() {
        return Err(ProtocolError::EmptyLine);
    }
    if record.contains(&b'\n') || record.contains(&b'\r') {
        return Err(ProtocolError::MultipleRecords);
    }
    let envelope: WorkerEnvelope = serde_json::from_slice(record).map_err(ProtocolError::Json)?;
    envelope.validate().map_err(ProtocolError::InvalidMessage)?;
    Ok(envelope)
}

/// Enforces session identity, handshake-first ordering, and contiguous sequence
/// numbers above the per-message codec.
#[derive(Clone, Debug, Default)]
pub struct ProtocolCursor {
    session_id: Option<String>,
    peer_role: Option<PeerRole>,
    next_sequence: u64,
}

impl ProtocolCursor {
    /// Accept the next envelope in one unidirectional protocol stream.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] when validation fails, the first record is not
    /// a handshake at sequence zero, or session/sequence continuity is broken.
    pub fn accept(&mut self, envelope: &WorkerEnvelope) -> Result<(), ProtocolError> {
        envelope.validate().map_err(ProtocolError::InvalidMessage)?;
        if self.session_id.is_none() {
            if envelope.sequence != 0 || !matches!(envelope.message, WorkerMessage::Handshake(_)) {
                return Err(ProtocolError::HandshakeRequired);
            }
            let WorkerMessage::Handshake(handshake) = &envelope.message else {
                return Err(ProtocolError::HandshakeRequired);
            };
            self.session_id = Some(envelope.session_id.clone());
            self.peer_role = Some(handshake.role);
            self.next_sequence = 1;
            return Ok(());
        }
        if self.session_id.as_deref() != Some(envelope.session_id.as_str()) {
            return Err(ProtocolError::SessionMismatch);
        }
        if envelope.sequence != self.next_sequence {
            return Err(ProtocolError::SequenceMismatch {
                expected: self.next_sequence,
                actual: envelope.sequence,
            });
        }
        if matches!(envelope.message, WorkerMessage::Handshake(_)) {
            return Err(ProtocolError::DuplicateHandshake);
        }
        let role = self.peer_role.ok_or(ProtocolError::HandshakeRequired)?;
        let direction_ok = match role {
            PeerRole::Controller => matches!(
                envelope.message,
                WorkerMessage::Plan(_) | WorkerMessage::Execute(_)
            ),
            PeerRole::Worker => matches!(
                envelope.message,
                WorkerMessage::Progress(_) | WorkerMessage::Artifact(_) | WorkerMessage::Error(_)
            ),
        };
        if !direction_ok {
            return Err(ProtocolError::MessageDirection {
                role,
                kind: envelope.message.kind(),
            });
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(ProtocolError::SequenceExhausted)?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("empty NDJSON record")]
    EmptyLine,
    #[error("input contains more than one NDJSON record")]
    MultipleRecords,
    #[error("NDJSON record is too long: {0} bytes")]
    LineTooLong(usize),
    #[error("invalid JSON: {0}")]
    Json(serde_json::Error),
    #[error("invalid protocol message: {0}")]
    InvalidMessage(ValidationError),
    #[error("the first session record must be handshake sequence 0")]
    HandshakeRequired,
    #[error("session identifier changed")]
    SessionMismatch,
    #[error("expected sequence {expected}, received {actual}")]
    SequenceMismatch { expected: u64, actual: u64 },
    #[error("a session may contain only one handshake")]
    DuplicateHandshake,
    #[error("message type {kind} is invalid for a {role:?} stream")]
    MessageDirection { role: PeerRole, kind: &'static str },
    #[error("protocol sequence number exhausted")]
    SequenceExhausted,
}
