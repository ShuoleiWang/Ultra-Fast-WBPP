//! Product-level project execution through the strict local `run-project` CLI.
//!
//! The webview submits typed roles and recipe choices. This controller writes a
//! mode-0600 request file, launches the local sidecar without a shell, streams
//! bounded progress, removes the request file, and independently revalidates
//! the published receipt and every GUI artifact before emitting completion.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use openastroflow_app_core::{AstrometricSolutionReceipt, Validate};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Runtime};

use crate::platform;
use crate::sidecar::{discover_engine, new_public_identifier};

const PROGRESS_EVENT: &str = "openastroflow://pipeline-progress";
const COMPLETE_EVENT: &str = "openastroflow://pipeline-complete";
const ERROR_EVENT: &str = "openastroflow://pipeline-error";
const MAX_RESULT_BYTES: usize = 16 * 1024 * 1024;
const MAX_DIAGNOSTIC_BYTES: usize = 8 * 1024;

fn diagnostic_tail(value: &str, limit: usize) -> &str {
    let mut start = value.len().saturating_sub(limit);
    while !value.is_char_boundary(start) {
        start += 1;
    }
    &value[start..]
}

fn append_diagnostic(diagnostics: &mut String, line: &str) {
    // Tracebacks end with the actionable exception. Retain that tail even
    // when earlier warnings or a single long line exhaust the byte budget.
    diagnostics.push_str(diagnostic_tail(line, MAX_DIAGNOSTIC_BYTES));
    diagnostics.push('\n');
    let keep = diagnostic_tail(diagnostics, MAX_DIAGNOSTIC_BYTES).to_owned();
    *diagnostics = keep;
}

fn response_failure_detail(context: &str, diagnostics: &str) -> String {
    let context = diagnostic_tail(context, MAX_DIAGNOSTIC_BYTES);
    let diagnostics = diagnostics.trim();
    if diagnostics.is_empty() || context.len() >= MAX_DIAGNOSTIC_BYTES - 1 {
        return context.to_owned();
    }
    format!(
        "{context}\n{}",
        diagnostic_tail(diagnostics, MAX_DIAGNOSTIC_BYTES - context.len() - 1)
    )
}

fn execution_failure_detail(result: &serde_json::Value, diagnostics: &str) -> String {
    let detail = result
        .get("message")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .or_else(|| {
            let detail = diagnostics.trim();
            (!detail.is_empty()).then_some(detail)
        })
        .unwrap_or("project execution failed closed");
    diagnostic_tail(detail, MAX_DIAGNOSTIC_BYTES).to_owned()
}

fn decode_project_response(
    bytes: &[u8],
    status: Option<ExitStatus>,
    diagnostics: &str,
) -> Result<serde_json::Value, (String, String)> {
    let failed = status.is_none_or(|value| !value.success());
    let result: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        let exit = status
            .map(|value| value.to_string())
            .unwrap_or_else(|| "exit status unavailable".to_owned());
        let response = if bytes.iter().all(u8::is_ascii_whitespace) {
            "no JSON result"
        } else {
            "an invalid JSON result"
        };
        // Never echo malformed stdout: it can contain a partial request or
        // private source data. The parser location and stderr explain failure.
        (
            if failed {
                "PROJECT_EXECUTION_FAILED"
            } else {
                "PROJECT_RESPONSE_INVALID"
            }
            .to_owned(),
            response_failure_detail(
                &format!("project worker returned {response} ({exit}; {error})"),
                diagnostics,
            ),
        )
    })?;
    if failed {
        return Err((
            result
                .get("code")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("PROJECT_EXECUTION_FAILED")
                .to_owned(),
            execution_failure_detail(&result, diagnostics),
        ));
    }
    Ok(result)
}

#[derive(Default)]
pub(crate) struct ProjectRegistry {
    jobs: Mutex<HashMap<String, ProjectJob>>,
    shutting_down: AtomicBool,
}

#[derive(Clone)]
struct ProjectJob {
    child: Arc<Mutex<Child>>,
    request_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiRunSource {
    source_id: String,
    role: String,
    paths: Vec<String>,
    recursive: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiRecipeOptions {
    balanced: bool,
    drizzle_enabled: bool,
    local_normalization_enabled: bool,
    solver_required: bool,
    #[serde(default = "default_calibration_workflow")]
    calibration_workflow: String,
}

fn default_calibration_workflow() -> String {
    "strict-v1".to_owned()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiMasterOverride {
    source_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    camera: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gain: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    binning: Option<[u32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cfa_pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    readout_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature_celsius: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exposure_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bias_included: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    numeric_domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    normalized_unit_scale: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiRawFrameOverride {
    source_sha256: String,
    cfa_pattern: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiReviewSelection {
    source_sha256: String,
    gate_policy_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ProjectRunRequest {
    sources: Vec<UiRunSource>,
    project_name: String,
    /// Label the interface derives from the target names (for example
    /// `NGC 7331`); it names the output folder.  Empty falls back to the
    /// project name.
    #[serde(default)]
    run_label: String,
    recipe: UiRecipeOptions,
    master_metadata_overrides: Vec<UiMasterOverride>,
    raw_frame_metadata_overrides: Vec<UiRawFrameOverride>,
    review_selections: Vec<UiReviewSelection>,
    output_parent_directory: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProjectRunReceipt {
    job_id: String,
    accepted: bool,
    execution_mode: &'static str,
    output_directory: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProgressEvent {
    job_id: String,
    stage_id: Option<String>,
    state: &'static str,
    fraction: f64,
    completed_units: Option<u64>,
    total_units: Option<u64>,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    overall_fraction: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    panel_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    panel_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    panel_filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    panel_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    panel_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorEvent {
    job_id: String,
    code: String,
    message: String,
    retryable: bool,
    details: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiArtifactReceipt {
    artifact_id: String,
    relative_path: String,
    sha256: String,
    size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    astrometry: Option<AstrometricSolutionReceipt>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiArtifact {
    kind: String,
    name: String,
    path: String,
    detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<String>,
    receipt: UiArtifactReceipt,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GateCheck {
    code: &'static str,
    required: bool,
    passed: bool,
    artifact_ids: Vec<String>,
    message: &'static str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GateReport {
    decision: &'static str,
    checks: Vec<GateCheck>,
}

/// One Light the quality gate did not pass, as the result page shows it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiScreeningFrame {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<String>,
    disposition: String,
    admitted: bool,
    summary: String,
    evidence: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    star_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    preview_data_url: Option<String>,
}

/// The run's Light screening: counts plus every frame that needed a decision.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiScreening {
    admitted: u64,
    excluded: u64,
    counts: BTreeMap<String, u64>,
    frames: Vec<UiScreeningFrame>,
}

struct Completion {
    artifacts: Vec<UiArtifact>,
    screening: Option<UiScreening>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompleteEvent {
    job_id: String,
    output_directory: String,
    artifacts: Vec<UiArtifact>,
    gate: GateReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    screening: Option<UiScreening>,
}

const MAX_SCREENING_FRAMES: usize = 512;
const MAX_SCREENING_PREVIEWS: usize = 128;
const MAX_SCREENING_PREVIEW_BYTES: u64 = 512 * 1024;
const MAX_SCREENING_PREVIEW_TOTAL_BYTES: usize = 24 * 1024 * 1024;
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// Standard base64 with padding; the previews are small, so no crate.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let value = chunk.iter().enumerate().fold(0_u32, |acc, (index, byte)| {
            acc | (u32::from(*byte) << (16 - 8 * index))
        });
        for position in 0..4 {
            if position <= chunk.len() {
                let index = (value >> (18 - 6 * position)) & 0x3f;
                encoded.push(ALPHABET[index as usize] as char);
            } else {
                encoded.push('=');
            }
        }
    }
    encoded
}

/// Load one review preview the worker published under `root` as a data URL.
fn screening_preview(root: &Path, relative: &str, budget: &mut usize) -> Option<String> {
    let (_, resolved) = relative_artifact(root, relative).ok()?;
    let size = resolved.metadata().ok()?.len();
    if size == 0 || size > MAX_SCREENING_PREVIEW_BYTES {
        return None;
    }
    let mut bytes = Vec::with_capacity(size as usize);
    File::open(&resolved)
        .ok()?
        .take(MAX_SCREENING_PREVIEW_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 != size || !bytes.starts_with(&PNG_SIGNATURE) {
        return None;
    }
    let encoded = format!("data:image/png;base64,{}", base64_encode(&bytes));
    if *budget < encoded.len() {
        return None;
    }
    *budget -= encoded.len();
    Some(encoded)
}

/// The receipt's `execution.screening`, with previews loaded; `None` when a
/// receipt predates screening records.  A malformed record is an error: the
/// result page must not show a partial screening as if it were complete.
fn screening_summary(
    root: &Path,
    receipt: &serde_json::Value,
) -> Result<Option<UiScreening>, String> {
    let Some(record) = receipt
        .get("execution")
        .and_then(|item| item.get("screening"))
    else {
        return Ok(None);
    };
    let count = |field: &str| -> Result<u64, String> {
        record
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("screening record has no {field}"))
    };
    let counts = record
        .get("counts")
        .and_then(serde_json::Value::as_object)
        .ok_or("screening record has no counts")?
        .iter()
        .map(|(key, value)| {
            value
                .as_u64()
                .map(|count| (key.clone(), count))
                .ok_or_else(|| format!("screening count {key} is not a number"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let records = record
        .get("frames")
        .and_then(serde_json::Value::as_array)
        .ok_or("screening record has no frames")?;
    if records.len() > MAX_SCREENING_FRAMES {
        return Err("screening record lists too many frames".to_owned());
    }
    let mut budget = MAX_SCREENING_PREVIEW_TOTAL_BYTES;
    let mut previews = 0_usize;
    let mut frames = Vec::with_capacity(records.len());
    for item in records {
        let path = value_string(item, "path")?;
        let disposition = value_string(item, "disposition")?;
        if !matches!(disposition, "PASS" | "REVIEW" | "HARD_FAIL") {
            return Err(format!(
                "screening frame has an unknown disposition {disposition}"
            ));
        }
        let evidence = item
            .get("evidence")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .take(8)
                    .map(|text| text.chars().take(500).collect::<String>())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let preview_data_url = item
            .get("reviewPreview")
            .and_then(serde_json::Value::as_str)
            .filter(|_| previews < MAX_SCREENING_PREVIEWS)
            .and_then(|relative| screening_preview(root, relative, &mut budget));
        previews += usize::from(preview_data_url.is_some());
        frames.push(UiScreeningFrame {
            name: path
                .rsplit(['/', '\\'])
                .next()
                .filter(|name| !name.is_empty())
                .unwrap_or(path)
                .to_owned(),
            target: item
                .get("target")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            disposition: disposition.to_owned(),
            admitted: item
                .get("admitted")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            summary: item
                .get("summary")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .chars()
                .take(2000)
                .collect(),
            evidence,
            star_count: item.get("starCount").and_then(serde_json::Value::as_u64),
            preview_data_url,
        });
    }
    Ok(Some(UiScreening {
        admitted: count("admitted")?,
        excluded: count("excluded")?,
        counts,
        frames,
    }))
}

fn checked_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn artifact_sha256_hex(value: &str) -> Result<&str, String> {
    // Python artifact identities are tagged; GUI typed receipts use bare hex.
    // Accept exactly these two established forms, without relaxing the digest.
    let hex = value.strip_prefix("sha256:").unwrap_or(value);
    if checked_sha256(hex) {
        Ok(hex)
    } else {
        Err("GUI artifact SHA-256 identity is malformed".to_owned())
    }
}

fn checked_source_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(checked_sha256)
}

fn checked_role(value: &str) -> bool {
    matches!(
        value,
        "LIGHT" | "FLAT" | "DARK" | "BIAS" | "MASTER_FLAT" | "MASTER_DARK" | "MASTER_BIAS"
    )
}

fn checked_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn canonical_output_parent(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value)
        .canonicalize()
        .map_err(|error| format!("cannot resolve output parent: {error}"))?;
    if !path.is_dir() {
        return Err("output parent must be an existing directory".to_owned());
    }
    Ok(path)
}

fn validate_master_override(value: &UiMasterOverride) -> Result<(), String> {
    if !checked_source_sha256(&value.source_sha256) {
        return Err("masterMetadataOverrides sourceSha256 is invalid".to_owned());
    }
    if [
        value.camera.as_deref(),
        value.filter.as_deref(),
        value.cfa_pattern.as_deref(),
        value.readout_mode.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|item| {
        item.trim().is_empty()
            || matches!(
                item.trim().to_ascii_uppercase().as_str(),
                "UNKNOWN" | "UNSPECIFIED"
            )
    }) {
        return Err("masterMetadataOverrides must omit unrecorded fields instead of replacing metadata with unknown values".to_owned());
    }
    if [
        value.gain,
        value.offset,
        value.temperature_celsius,
        value.exposure_seconds,
    ]
    .into_iter()
    .flatten()
    .any(|item| !item.is_finite())
        || value.exposure_seconds.is_some_and(|value| value < 0.0)
        || value.binning.is_some_and(|value| value.contains(&0))
    {
        return Err("masterMetadataOverrides contains invalid numeric metadata".to_owned());
    }
    if value.cfa_pattern.as_ref().is_some_and(|value| {
        !matches!(
            value.trim().to_ascii_uppercase().as_str(),
            "NONE" | "MONO" | "MONOCHROME"
        )
    }) {
        return Err("CFA/OSC master metadata is unsupported in this release".to_owned());
    }
    if value.numeric_domain.is_some() != value.normalized_unit_scale.is_some() {
        return Err(
            "masterMetadataOverrides numericDomain and normalizedUnitScale must be declared together"
                .to_owned(),
        );
    }
    if let (Some(domain), Some(scale)) =
        (value.numeric_domain.as_deref(), value.normalized_unit_scale)
    {
        if !matches!(
            domain.trim().to_ascii_uppercase().as_str(),
            "NORMALIZED_UNIT" | "SENSOR_CODE"
        ) || !scale.is_finite()
            || scale <= 0.0
        {
            return Err("masterMetadataOverrides pixel numeric domain is invalid".to_owned());
        }
    }
    Ok(())
}

fn project_request_json(
    request: &ProjectRunRequest,
    output: &Path,
) -> Result<serde_json::Value, String> {
    if request.sources.is_empty() || request.project_name.trim().is_empty() {
        return Err("project sources and projectName are required".to_owned());
    }
    if !request.recipe.balanced || !request.recipe.solver_required {
        return Err(
            "the product GUI requires Balanced processing and a required solver".to_owned(),
        );
    }
    if !matches!(
        request.recipe.calibration_workflow.as_str(),
        "strict-v1" | "mono-standard-v1"
    ) {
        return Err("unsupported calibration workflow".to_owned());
    }
    for item in &request.master_metadata_overrides {
        validate_master_override(item)?;
    }
    let mut raw_override_digests = HashSet::new();
    for item in &request.raw_frame_metadata_overrides {
        if !checked_source_sha256(&item.source_sha256)
            || !item.cfa_pattern.trim().eq_ignore_ascii_case("NONE")
            || !raw_override_digests.insert(item.source_sha256.as_str())
        {
            return Err(
                "rawFrameMetadataOverrides must contain unique SHA-bound NONE declarations"
                    .to_owned(),
            );
        }
    }
    let mut review_digests = HashSet::new();
    for item in &request.review_selections {
        if !checked_source_sha256(&item.source_sha256)
            || !checked_source_sha256(&item.gate_policy_digest)
            || !review_digests.insert(item.source_sha256.as_str())
        {
            return Err(
                "reviewSelections must contain unique content and policy SHA-256 bindings"
                    .to_owned(),
            );
        }
    }
    let mut seen_paths = HashSet::new();
    let mut sources = Vec::new();
    let mut role_by_path = HashMap::new();
    for group in &request.sources {
        if !checked_id(&group.source_id) || !checked_role(&group.role) || group.paths.is_empty() {
            return Err("one or more source records are invalid".to_owned());
        }
        for (index, path) in group.paths.iter().enumerate() {
            let canonical = Path::new(path)
                .canonicalize()
                .map_err(|error| format!("cannot resolve input source: {error}"))?;
            if canonical.is_dir() && !group.recursive {
                return Err(
                    "a directory source must explicitly enable recursive inventory".to_owned(),
                );
            }
            if !canonical.is_file() && !canonical.is_dir() {
                return Err("input source is not a file or directory".to_owned());
            }
            let key = canonical.to_string_lossy().into_owned();
            if !seen_paths.insert(key.clone()) {
                return Err("one input path appears more than once".to_owned());
            }
            role_by_path.insert(key.clone(), group.role.as_str());
            sources.push(serde_json::json!({
                "sourceId": if group.paths.len() == 1 { group.source_id.clone() } else { format!("{}-{}", group.source_id, index + 1) },
                "hostPath": key,
                "expectedRole": group.role,
                "recursive": group.recursive,
            }));
        }
    }
    for item in request.sources.iter().filter(|item| {
        item.role == "MASTER_DARK" && request.recipe.calibration_workflow == "strict-v1"
    }) {
        for path in &item.paths {
            let digest = format!("sha256:{}", sha256_file(Path::new(path))?);
            let declaration = request
                .master_metadata_overrides
                .iter()
                .find(|candidate| candidate.source_sha256 == digest)
                .ok_or("every MasterDark requires one content-bound metadata override")?;
            if declaration.bias_included.is_none() {
                return Err("MasterDark biasIncluded must be explicitly true or false".to_owned());
            }
        }
    }
    Ok(serde_json::json!({
        "schemaVersion": 1,
        "sources": sources,
        "outputDirectory": output,
        "projectName": request.project_name.trim(),
        "recipe": {
            "schemaVersion": 1,
            "calibration": {
                "workflow": request.recipe.calibration_workflow,
                "flat": "REQUIRED", "dark": "OPTIONAL",
                "bias": if request.recipe.calibration_workflow == "mono-standard-v1" { "OPTIONAL" } else { "REQUIRED" },
                "allowMasters": true,
                "masterMetadataOverrides": request.master_metadata_overrides,
            },
            "solver": { "policy": "REQUIRED", "backend": "astrometry-net", "searchRadiusDegrees": 15.0 },
            "drizzle": { "enabled": request.recipe.drizzle_enabled, "backend": "auto", "scale": 2, "dropShrink": 0.9, "cfaDrizzle": false },
            "localNormalization": { "enabled": request.recipe.local_normalization_enabled, "tileSizePixels": 256 },
            "outputFormat": "FITS", "overwrite": false, "reviewApprovals": [],
            "rawFrameMetadataOverrides": request.raw_frame_metadata_overrides,
        },
        "solverHints": {},
        "execution": {},
        "reviewSelections": request.review_selections,
    }))
}

fn create_private_request(value: &serde_json::Value) -> Result<PathBuf, String> {
    let path = std::env::temp_dir().join(format!(
        "{}.json",
        new_public_identifier("openastroflow-project-request")?
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .map_err(|error| format!("cannot create private project request: {error}"))?;
    let mut bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    file.write_all(&bytes).map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    Ok(path)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn relative_artifact(root: &Path, value: &str) -> Result<(String, PathBuf), String> {
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|item| !matches!(item, Component::Normal(_)))
    {
        return Err("published artifact has an unsafe relative path".to_owned());
    }
    let resolved = root
        .join(relative)
        .canonicalize()
        .map_err(|error| format!("cannot resolve published artifact: {error}"))?;
    if !resolved.starts_with(root) || !resolved.is_file() {
        return Err("published artifact escaped its output directory".to_owned());
    }
    Ok((relative.to_string_lossy().replace('\\', "/"), resolved))
}

fn value_string<'a>(value: &'a serde_json::Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("published artifact has no {field}"))
}

fn validate_completion(output: &Path, result: &serde_json::Value) -> Result<Completion, String> {
    if result.get("success").and_then(serde_json::Value::as_bool) != Some(true)
        || result.get("state").and_then(serde_json::Value::as_str) != Some("SOLVED")
    {
        return Err(format!(
            "{}: project did not publish a solved result",
            result
                .get("code")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("PROJECT_FAILED")
        ));
    }
    let reported_output = Path::new(
        result
            .get("outputDirectory")
            .and_then(serde_json::Value::as_str)
            .ok_or("project result has no outputDirectory")?,
    )
    .canonicalize()
    .map_err(|error| error.to_string())?;
    let root = output.canonicalize().map_err(|error| error.to_string())?;
    if reported_output != root {
        return Err("project result names a different output directory".to_owned());
    }
    let receipt_path = Path::new(
        result
            .get("receiptPath")
            .and_then(serde_json::Value::as_str)
            .ok_or("project result has no receiptPath")?,
    )
    .canonicalize()
    .map_err(|error| error.to_string())?;
    if !receipt_path.starts_with(&root) || !receipt_path.is_file() {
        return Err("project receipt escaped its output directory".to_owned());
    }
    let mut receipt_bytes = Vec::new();
    File::open(&receipt_path)
        .map_err(|error| error.to_string())?
        .take((MAX_RESULT_BYTES + 1) as u64)
        .read_to_end(&mut receipt_bytes)
        .map_err(|error| error.to_string())?;
    if receipt_bytes.len() > MAX_RESULT_BYTES {
        return Err("project receipt is too large".to_owned());
    }
    let receipt: serde_json::Value =
        serde_json::from_slice(&receipt_bytes).map_err(|error| error.to_string())?;
    if receipt.get("success").and_then(serde_json::Value::as_bool) != Some(true)
        || receipt.get("state").and_then(serde_json::Value::as_str) != Some("SOLVED")
    {
        return Err("project receipt is not a solved success receipt".to_owned());
    }
    let final_products = receipt
        .get("finalProducts")
        .ok_or("project receipt has no finalProducts")?;
    let gate = final_products
        .get("resultGate")
        .ok_or("project receipt has no resultGate")?;
    for field in [
        "allMonoProductsSolved",
        "managedCatalogEvidenceRequired",
        "sourceIdentityVerifiedAtCommit",
        "mosaicCoverageOverlapSeamPassed",
    ] {
        if gate.get(field).and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(format!("project result gate did not pass {field}"));
        }
    }
    if gate.get("status").and_then(serde_json::Value::as_str) != Some("PASS") {
        return Err("project result gate is not PASS".to_owned());
    }
    let records = final_products
        .get("guiArtifacts")
        .and_then(serde_json::Value::as_array)
        .ok_or("project receipt has no guiArtifacts")?;
    let mut artifacts = Vec::new();
    let mut solved_count = 0_u32;
    for (index, record) in records.iter().enumerate() {
        let kind = value_string(record, "kind")?;
        if !matches!(
            kind,
            "SOLVED_MONO_FITS"
                | "LINEAR_RGB_FITS"
                | "RGB_PREVIEW_TIFF_16"
                | "RGB_PREVIEW_PNG_16"
                | "MONO_PREVIEW_PNG"
        ) {
            return Err(format!("unsupported GUI artifact kind: {kind}"));
        }
        let relative_value = record
            .get("relativePath")
            .or_else(|| record.get("path"))
            .and_then(serde_json::Value::as_str)
            .ok_or("GUI artifact has no relativePath")?;
        let (relative, resolved) = relative_artifact(&root, relative_value)?;
        let expected_size = record
            .get("sizeBytes")
            .and_then(serde_json::Value::as_u64)
            .ok_or("GUI artifact has no sizeBytes")?;
        let expected_sha = artifact_sha256_hex(value_string(record, "sha256")?)?;
        if resolved
            .metadata()
            .map_err(|error| error.to_string())?
            .len()
            != expected_size
            || sha256_file(&resolved)? != expected_sha
        {
            return Err("GUI artifact content identity changed before publication".to_owned());
        }
        let astrometry = if matches!(kind, "SOLVED_MONO_FITS" | "LINEAR_RGB_FITS") {
            let mut value = record
                .get("astrometry")
                .and_then(serde_json::Value::as_object)
                .cloned()
                .ok_or("final FITS artifact has no astrometry evidence")?;
            value.remove("imageShape");
            value.remove("state");
            let parsed: AstrometricSolutionReceipt = serde_json::from_value(value.into())
                .map_err(|error| format!("astrometry receipt is invalid: {error}"))?;
            parsed.validate().map_err(|error| error.to_string())?;
            if record
                .get("finalGate")
                .and_then(|item| item.get("status"))
                .and_then(serde_json::Value::as_str)
                != Some("PASS")
            {
                return Err("final FITS artifact gate is not PASS".to_owned());
            }
            solved_count += u32::from(kind == "SOLVED_MONO_FITS");
            Some(parsed)
        } else {
            None
        };
        let detail = astrometry.as_ref().map_or_else(
            || match kind {
                "RGB_PREVIEW_TIFF_16" => "16-bit display preview".to_owned(),
                "RGB_PREVIEW_PNG_16" => "16-bit display preview".to_owned(),
                "MONO_PREVIEW_PNG" => "auto-stretched mono preview".to_owned(),
                _ => "verified product".to_owned(),
            },
            |value| {
                format!(
                    "WCS embedded · {:.3} arcsec RMS · {} matched stars",
                    value.rms_arcsec, value.matched_stars
                )
            },
        );
        let name = resolved
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("artifact")
            .to_owned();
        artifacts.push(UiArtifact {
            kind: kind.to_owned(),
            name,
            path: resolved.to_string_lossy().into_owned(),
            detail,
            filter: record
                .get("filter")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            target: record
                .get("target")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            receipt: UiArtifactReceipt {
                artifact_id: format!("project-artifact-{}", index + 1),
                relative_path: relative,
                sha256: expected_sha.to_owned(),
                size_bytes: expected_size,
                astrometry,
            },
        });
    }
    if solved_count == 0 {
        return Err("project receipt has no solved mono product".to_owned());
    }
    let receipt_relative = receipt_path
        .strip_prefix(&root)
        .map_err(|_| "receipt path escaped output")?
        .to_string_lossy()
        .trim_start_matches(std::path::MAIN_SEPARATOR)
        .replace('\\', "/");
    artifacts.push(UiArtifact {
        kind: "RECEIPT".to_owned(),
        name: receipt_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("receipt.json")
            .to_owned(),
        path: receipt_path.to_string_lossy().into_owned(),
        detail: "outer project receipt · content verified".to_owned(),
        filter: None,
        target: None,
        receipt: UiArtifactReceipt {
            artifact_id: "project-receipt".to_owned(),
            relative_path: receipt_relative,
            sha256: sha256_file(&receipt_path)?,
            size_bytes: receipt_path
                .metadata()
                .map_err(|error| error.to_string())?
                .len(),
            astrometry: None,
        },
    });
    let screening = screening_summary(&root, &receipt)?;
    Ok(Completion {
        artifacts,
        screening,
    })
}

fn stage_id(value: &str) -> Option<String> {
    Some(
        match value {
            "quality-control" => "quality-control",
            "calibration" => "calibrate",
            "registration" => "register",
            "integration" => "integrate",
            "drizzle" => "drizzle",
            "astrometry" => "solve",
            "prepare" | "inventory" => "prepare",
            "mosaic" => "mosaic",
            "alignment" => "alignment",
            "color" => "color",
            "verify" => "verify",
            "preview" => "preview",
            "publish" | "complete" => "publish",
            _ => return None,
        }
        .to_owned(),
    )
}

fn normalize_progress(job_id: &str, event: &serde_json::Value) -> ProgressEvent {
    let current = event
        .get("current")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total = event
        .get("total")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let status = event
        .get("status")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("running")
        .to_ascii_lowercase();
    let stage = event.get("stage").and_then(serde_json::Value::as_str);
    let done = matches!(
        status.as_str(),
        "complete" | "completed" | "succeeded" | "pass"
    );
    let failed = stage == Some("failed") || matches!(status.as_str(), "failed" | "error");
    let text = |key| {
        event
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    };
    ProgressEvent {
        job_id: job_id.to_owned(),
        stage_id: stage.and_then(stage_id),
        state: if failed {
            "failed"
        } else if done {
            "succeeded"
        } else {
            "running"
        },
        fraction: if total > 0 {
            (current as f64 / total as f64).clamp(0.0, 1.0)
        } else if done && !failed {
            1.0
        } else {
            0.0
        },
        completed_units: (total > 0).then_some(current.min(total)),
        total_units: (total > 0).then_some(total),
        message: text("message").unwrap_or_default(),
        // Only the separately revalidated completion event can finish the job.
        overall_fraction: event
            .get("overallFraction")
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite())
            .map(|value| value.clamp(0.0, 0.99)),
        scope: text("scope").filter(|value| matches!(value.as_str(), "panel" | "project")),
        panel_id: text("panelId"),
        panel_target: text("panelTarget"),
        panel_filter: text("panelFilter"),
        panel_index: event.get("panelIndex").and_then(serde_json::Value::as_u64),
        panel_count: event.get("panelCount").and_then(serde_json::Value::as_u64),
    }
}

fn stream_progress<R: Runtime>(
    app: AppHandle<R>,
    job_id: String,
    stderr: std::process::ChildStderr,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut diagnostics = String::new();
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            let value = (line.len() <= 256 * 1024)
                .then(|| serde_json::from_str::<serde_json::Value>(&line).ok())
                .flatten();
            if let Some(event) = value
                .as_ref()
                .filter(|item| {
                    item.get("type").and_then(serde_json::Value::as_str) == Some("progress")
                })
                .and_then(|item| item.get("event"))
            {
                let _ = app.emit(PROGRESS_EVENT, normalize_progress(&job_id, event));
                continue;
            }
            append_diagnostic(&mut diagnostics, &line);
        }
        diagnostics
    })
}

fn emit_error<R: Runtime>(app: &AppHandle<R>, job_id: &str, code: &str, message: &str) {
    let _ = app.emit(
        ERROR_EVENT,
        ErrorEvent {
            job_id: job_id.to_owned(),
            code: code.to_owned(),
            message: message.to_owned(),
            retryable: false,
            details: BTreeMap::new(),
        },
    );
}

pub(crate) fn start<R: Runtime>(
    app: AppHandle<R>,
    registry: Arc<ProjectRegistry>,
    request: ProjectRunRequest,
) -> Result<ProjectRunReceipt, String> {
    if registry.shutting_down.load(Ordering::Acquire) {
        return Err("application shutdown is already in progress".to_owned());
    }
    let executable = discover_engine(&app)?;
    start_with(app, registry, request, executable)
}

/// Folder-name token of a run label: `NGC 7331` -> `NGC7331`, `盾牌座 马赛克`
/// -> `盾牌座-马赛克`.  Whitespace between a letter run and a digit run (a
/// catalogue designation) is removed, other whitespace becomes `-`, and only
/// alphanumerics, `-` and `_` survive; the result is at most 48 characters.
pub(crate) fn output_directory_label(raw: &str) -> String {
    let words: Vec<String> = raw
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect();
    let mut label = String::new();
    for word in &words {
        if !label.is_empty() {
            let previous_alpha = label.chars().last().is_some_and(char::is_alphabetic);
            let next_digit = word.chars().next().is_some_and(|c| c.is_ascii_digit());
            if !(previous_alpha && next_digit) {
                label.push('-');
            }
        }
        label.push_str(word);
    }
    let mut compact = String::new();
    for c in label.chars() {
        if c == '-' && compact.ends_with('-') {
            continue;
        }
        compact.push(c);
    }
    let compact: String = compact.trim_matches(['-', '_']).chars().take(48).collect();
    if compact.is_empty() {
        "wbpp".to_owned()
    } else {
        compact
    }
}

/// `<label>_<YYYY-MM-DD_HHMM>` inside `parent`, with `_2`, `_3`, ... when
/// that name (or its `.unsolved` twin) already exists.
fn unique_output_directory(
    parent: &Path,
    label: &str,
    now: chrono::DateTime<chrono::Local>,
) -> Result<PathBuf, String> {
    let stem = format!(
        "{}_{}",
        output_directory_label(label),
        now.format("%Y-%m-%d_%H%M")
    );
    for attempt in 1..=99_u32 {
        let name = if attempt == 1 {
            stem.clone()
        } else {
            format!("{stem}_{attempt}")
        };
        let candidate = parent.join(&name);
        if !candidate.exists() && !candidate.with_extension("unsolved").exists() {
            return Ok(candidate);
        }
    }
    Err("create-only output destination already exists".to_owned())
}

fn start_with<R: Runtime>(
    app: AppHandle<R>,
    registry: Arc<ProjectRegistry>,
    request: ProjectRunRequest,
    executable: crate::sidecar::EngineExecutable,
) -> Result<ProjectRunReceipt, String> {
    if registry.shutting_down.load(Ordering::Acquire) {
        return Err("application shutdown is already in progress".to_owned());
    }
    let parent = canonical_output_parent(&request.output_parent_directory)?;
    let job_id = new_public_identifier("project")?;
    let label = if request.run_label.trim().is_empty() {
        request.project_name.as_str()
    } else {
        request.run_label.as_str()
    };
    let output = unique_output_directory(&parent, label, chrono::Local::now())?;
    let request_value = project_request_json(&request, &output)?;
    let request_path = create_private_request(&request_value)?;
    let mut command = executable.command("run-project");
    command
        .arg("--request-json")
        .arg(&request_path)
        .arg("--progress-json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match crate::sidecar::spawn_sidecar(&mut command) {
        Ok(value) => value,
        Err(error) => {
            let _ = std::fs::remove_file(&request_path);
            return Err(format!("cannot launch project sidecar: {error}"));
        }
    };
    let stdout = child.stdout.take().ok_or("project stdout is unavailable")?;
    let stderr = child.stderr.take().ok_or("project stderr is unavailable")?;
    let shared = Arc::new(Mutex::new(child));
    let mut jobs = registry
        .jobs
        .lock()
        .map_err(|_| "project registry lock was poisoned".to_owned())?;
    if registry.shutting_down.load(Ordering::Acquire) {
        drop(jobs);
        if let Ok(mut child) = shared.lock() {
            let _ = platform::terminate_process_tree(&mut child);
        }
        let _ = std::fs::remove_file(&request_path);
        return Err("application shutdown started before project registration".to_owned());
    }
    jobs.insert(
        job_id.clone(),
        ProjectJob {
            child: shared.clone(),
            request_path: request_path.clone(),
        },
    );
    drop(jobs);
    let thread_app = app.clone();
    let thread_registry = registry.clone();
    let thread_job = job_id.clone();
    let thread_output = output.clone();
    std::thread::spawn(move || {
        let diagnostics_thread = stream_progress(thread_app.clone(), thread_job.clone(), stderr);
        let mut bytes = Vec::new();
        let read_result = stdout
            .take((MAX_RESULT_BYTES + 1) as u64)
            .read_to_end(&mut bytes);
        let status = shared.lock().ok().and_then(|mut child| child.wait().ok());
        let diagnostics = diagnostics_thread.join().unwrap_or_default();
        let _ = std::fs::remove_file(&request_path);
        thread_registry
            .jobs
            .lock()
            .ok()
            .map(|mut jobs| jobs.remove(&thread_job));
        if read_result.is_err() || bytes.len() > MAX_RESULT_BYTES {
            emit_error(
                &thread_app,
                &thread_job,
                "PROJECT_RESPONSE_INVALID",
                &response_failure_detail("project result could not be read safely", &diagnostics),
            );
            return;
        }
        let result = match decode_project_response(&bytes, status, &diagnostics) {
            Ok(value) => value,
            Err((code, detail)) => {
                emit_error(&thread_app, &thread_job, &code, &detail);
                return;
            }
        };
        match validate_completion(&thread_output, &result) {
            Ok(Completion {
                artifacts,
                screening,
            }) => {
                let ids = artifacts
                    .iter()
                    .map(|item| item.receipt.artifact_id.clone())
                    .collect::<Vec<_>>();
                let gate = GateReport {
                    decision: "ready",
                    checks: vec![
                        GateCheck {
                            code: "final-project-products-present",
                            required: true,
                            passed: true,
                            artifact_ids: ids.clone(),
                            message: "one or more solved mono products are present",
                        },
                        GateCheck {
                            code: "final-project-receipts-valid",
                            required: true,
                            passed: true,
                            artifact_ids: ids.clone(),
                            message: "all GUI artifacts match their receipt identities",
                        },
                        GateCheck {
                            code: "final-project-astrometry-validated",
                            required: true,
                            passed: true,
                            artifact_ids: ids,
                            message: "managed catalog WCS evidence passed the outer result gate",
                        },
                    ],
                };
                let _ = thread_app.emit(
                    COMPLETE_EVENT,
                    CompleteEvent {
                        job_id: thread_job,
                        output_directory: thread_output.to_string_lossy().into_owned(),
                        artifacts,
                        gate,
                        screening,
                    },
                );
            }
            Err(error) => emit_error(
                &thread_app,
                &thread_job,
                "PROJECT_RESULT_GATE_BLOCKED",
                &error,
            ),
        }
    });
    Ok(ProjectRunReceipt {
        job_id,
        accepted: true,
        execution_mode: "native",
        output_directory: output.to_string_lossy().into_owned(),
    })
}

pub(crate) fn cancel(registry: &ProjectRegistry, job_id: &str) -> Result<(), String> {
    let child = registry
        .jobs
        .lock()
        .map_err(|_| "project registry lock was poisoned".to_owned())?
        .get(job_id)
        .map(|job| job.child.clone())
        .ok_or_else(|| format!("project is not running: {job_id}"))?;
    let mut child = child
        .lock()
        .map_err(|_| "project child lock was poisoned".to_owned())?;
    platform::terminate_process_tree(&mut child)
}

pub(crate) fn terminate_all(registry: &ProjectRegistry) -> Result<(), String> {
    registry.shutting_down.store(true, Ordering::Release);
    let jobs = registry
        .jobs
        .lock()
        .map_err(|_| "project registry lock was poisoned".to_owned())?
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    for job in jobs {
        match job.child.lock() {
            Ok(mut child) => {
                if let Err(error) = platform::terminate_process_tree(&mut child) {
                    failures.push(error);
                }
            }
            Err(_) => failures.push("project child lock was poisoned".to_owned()),
        }
        if let Err(error) = std::fs::remove_file(&job.request_path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                failures.push(format!("cannot remove private project request: {error}"));
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "one or more project process trees could not be terminated: {}",
            failures.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_directory_label_keeps_designations_and_readable_names() {
        assert_eq!(output_directory_label("NGC 7331"), "NGC7331");
        assert_eq!(output_directory_label("  M 31  "), "M31");
        assert_eq!(output_directory_label("Sh2-155 / Cave"), "Sh2-155-Cave");
        assert_eq!(output_directory_label("盾牌座 马赛克"), "盾牌座-马赛克");
        assert_eq!(
            output_directory_label("Ultra-Fast WBPP project"),
            "Ultra-Fast-WBPP-project"
        );
        assert_eq!(output_directory_label("../..//"), "wbpp");
        assert_eq!(output_directory_label(""), "wbpp");
        assert_eq!(output_directory_label(&"x".repeat(80)).chars().count(), 48);
    }

    #[test]
    fn unique_output_directory_adds_a_counter_on_collision() {
        let root = std::env::temp_dir().join(new_public_identifier("output-name-test").unwrap());
        std::fs::create_dir_all(&root).unwrap();
        let now = chrono::Local::now();
        let first = unique_output_directory(&root, "NGC 7331", now).unwrap();
        let expected = format!("NGC7331_{}", now.format("%Y-%m-%d_%H%M"));
        assert_eq!(first.file_name().unwrap().to_string_lossy(), expected);
        std::fs::create_dir_all(&first).unwrap();
        let second = unique_output_directory(&root, "NGC 7331", now).unwrap();
        assert_eq!(
            second.file_name().unwrap().to_string_lossy(),
            format!("{expected}_2")
        );
        std::fs::create_dir_all(second.with_extension("unsolved")).unwrap();
        let third = unique_output_directory(&root, "NGC 7331", now).unwrap();
        assert_eq!(
            third.file_name().unwrap().to_string_lossy(),
            format!("{expected}_3")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn base64_encodes_the_reference_vectors() {
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(input.as_bytes()), expected);
        }
        assert_eq!(base64_encode(&[0xff, 0xee, 0xdd, 0x00]), "/+7dAA==");
    }

    #[test]
    fn screening_summary_loads_bounded_previews_and_rejects_malformed_records() {
        let root = std::env::temp_dir().join(new_public_identifier("screening-test").unwrap());
        let review = root.join("details/runs/NGC7331/qc/review");
        std::fs::create_dir_all(&review).unwrap();
        // Previews resolve against the canonical output root, as in production.
        let root = root.canonicalize().unwrap();
        let png: Vec<u8> = PNG_SIGNATURE.iter().copied().chain([1_u8; 32]).collect();
        std::fs::write(review.join("0001-cloudy.png"), &png).unwrap();
        std::fs::write(review.join("0002-not-a-png.png"), b"plain text").unwrap();
        let receipt = serde_json::json!({"execution": {"screening": {
            "admitted": 61, "excluded": 2,
            "counts": {"PASS": 61, "REVIEW": 1, "HARD_FAIL": 1},
            "frames": [
                {"path": "source/src-1/NGC 7331_300.00s_L_cloudy.fits", "disposition": "HARD_FAIL", "admitted": false,
                 "summary": "clouds", "evidence": ["star count collapsed", "background rose"], "starCount": 12,
                 "reviewPreview": "details/runs/NGC7331/qc/review/0001-cloudy.png", "target": "NGC 7331"},
                {"path": "source/src-2/trail.fits", "disposition": "REVIEW", "admitted": true,
                 "summary": "trail", "evidence": [], "starCount": null,
                 "reviewPreview": "details/runs/NGC7331/qc/review/0002-not-a-png.png"},
                {"path": "source/src-3/missing.fits", "disposition": "REVIEW", "admitted": false,
                 "summary": "", "reviewPreview": "../escaped.png"}
            ]
        }}});
        let screening = screening_summary(&root, &receipt).unwrap().unwrap();
        assert_eq!((screening.admitted, screening.excluded), (61, 2));
        assert_eq!(screening.counts["REVIEW"], 1);
        assert_eq!(screening.frames.len(), 3);
        let cloudy = &screening.frames[0];
        assert_eq!(cloudy.name, "NGC 7331_300.00s_L_cloudy.fits");
        assert_eq!(cloudy.target.as_deref(), Some("NGC 7331"));
        assert_eq!(cloudy.star_count, Some(12));
        assert_eq!(cloudy.evidence.len(), 2);
        let preview = cloudy.preview_data_url.as_deref().unwrap();
        assert_eq!(
            preview,
            format!("data:image/png;base64,{}", base64_encode(&png))
        );
        // Not a PNG, and a path escaping the output: no preview, frame kept.
        assert!(screening.frames[1].preview_data_url.is_none() && screening.frames[1].admitted);
        assert!(screening.frames[2].preview_data_url.is_none());
        assert!(
            screening_summary(&root, &serde_json::json!({"execution": {}}))
                .unwrap()
                .is_none()
        );
        let bad = serde_json::json!({"execution": {"screening": {"admitted": 1, "excluded": 0, "counts": {},
            "frames": [{"path": "x", "disposition": "MAYBE"}]}}});
        assert!(screening_summary(&root, &bad).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "requires OAF_TEST_PROJECT_RECEIPT pointing to retained real project output"]
    fn retained_real_project_passes_native_final_gate_read_only() {
        let receipt_path = PathBuf::from(
            std::env::var("OAF_TEST_PROJECT_RECEIPT").expect("retained receipt path"),
        );
        let root = receipt_path.parent().unwrap();
        let result = serde_json::json!({"success":true, "state":"SOLVED", "outputDirectory":root, "receiptPath":receipt_path});
        let completion = validate_completion(root, &result)
            .expect("real project must pass native final artifact validation");
        let artifacts = completion.artifacts;
        assert_eq!(artifacts.len(), 12); // 11 products plus the outer receipt.
        let screening = completion
            .screening
            .expect("a current receipt records the run's screening");
        assert_eq!(
            screening.admitted + screening.excluded,
            screening.counts.values().sum::<u64>()
        );
        assert!(screening.frames.iter().all(|frame| frame
            .preview_data_url
            .as_deref()
            .is_some_and(|url| url.starts_with("data:image/png;base64,"))));
        println!(
            "Screening: {} admitted, {} excluded, {} frames with previews.",
            screening.admitted,
            screening.excluded,
            screening.frames.len()
        );
        assert_eq!(
            artifacts
                .iter()
                .filter(|item| item.kind == "SOLVED_MONO_FITS")
                .count(),
            4
        );
        assert_eq!(
            artifacts
                .iter()
                .filter(|item| item.kind == "LINEAR_RGB_FITS")
                .count(),
            1
        );
        assert!(artifacts
            .iter()
            .all(|item| checked_sha256(&item.receipt.sha256)));
        println!("Validated 11 retained real project products plus receipt; all GUI SHA-256 values normalized to bare hex.");
    }

    #[test]
    fn final_gate_accepts_known_hash_forms_and_rejects_malformed_or_changed_content() {
        let root = std::env::temp_dir().join(new_public_identifier("project-hash-test").unwrap());
        std::fs::create_dir_all(&root).unwrap();
        let artifact_path = root.join("product.fits");
        std::fs::write(&artifact_path, b"synthetic solved product").unwrap();
        let digest = sha256_file(&artifact_path).unwrap();
        let receipt_path = root.join("receipt.json");
        let result = serde_json::json!({"success":true,"state":"SOLVED","outputDirectory":root,"receiptPath":receipt_path});
        let mut receipt = serde_json::json!({"success":true,"state":"SOLVED","finalProducts":{
            "resultGate":{"status":"PASS","allMonoProductsSolved":true,"managedCatalogEvidenceRequired":true,"sourceIdentityVerifiedAtCommit":true,"mosaicCoverageOverlapSeamPassed":true},
            "guiArtifacts":[{"kind":"SOLVED_MONO_FITS","path":"product.fits","sha256":digest,"sizeBytes":artifact_path.metadata().unwrap().len(),"finalGate":{"status":"PASS"},"astrometry":{
                "referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":281.0,"centerDecDegrees":-6.0,
                "pixelScaleArcsec":1.4,"rotationDegrees":0.0,"rmsPixels":0.3,"rmsArcsec":0.42,"matchedStars":73,
                "parity":"POSITIVE","catalogIdentity":"2".repeat(64),"indexIdentities":["astrometry.net:index:4108:healpix:123:hpnside:4"],
                "correspondenceSha256":"3".repeat(64),"catalogManaged":true,"installedSetIdentity":"5".repeat(64),"catalogManifestSha256":"6".repeat(64),
                "indexArtifacts":[{"indexId":"4108","relativeName":"index-4108.fits","sizeBytes":94550400,"sha256":"7".repeat(64),"manifestSha256":"6".repeat(64),"installedSetIdentity":"5".repeat(64)}],
                "wcsSha256":"4".repeat(64),"imageShape":[4176,6248],"state":"SOLVED"
            }}]
        }});
        let validate = |receipt: &serde_json::Value| {
            std::fs::write(&receipt_path, serde_json::to_vec(receipt).unwrap()).unwrap();
            validate_completion(&root, &result).map(|completion| completion.artifacts)
        };
        for value in [digest.clone(), format!("sha256:{digest}")] {
            receipt["finalProducts"]["guiArtifacts"][0]["sha256"] = value.into();
            let artifacts = validate(&receipt).unwrap();
            assert_eq!(artifacts[0].receipt.sha256, digest);
        }
        for value in [
            format!("SHA256:{digest}"),
            format!("sha256:sha256:{digest}"),
            format!("sha256:{digest} "),
            format!("sha256:{}", "g".repeat(64)),
            format!("sha256:{}", "a".repeat(63)),
        ] {
            receipt["finalProducts"]["guiArtifacts"][0]["sha256"] = value.into();
            assert!(validate(&receipt)
                .unwrap_err()
                .contains("SHA-256 identity is malformed"));
        }
        receipt["finalProducts"]["guiArtifacts"][0]["sha256"] = format!("sha256:{digest}").into();
        std::fs::write(&artifact_path, b"Synthetic solved product").unwrap(); // Same size, different hash.
        assert!(validate(&receipt)
            .unwrap_err()
            .contains("content identity changed"));
        std::fs::write(&artifact_path, b"synthetic solved product").unwrap();
        receipt["finalProducts"]["guiArtifacts"][0]["sizeBytes"] = 1.into();
        assert!(validate(&receipt)
            .unwrap_err()
            .contains("content identity changed"));
        receipt["finalProducts"]["guiArtifacts"][0]["sizeBytes"] =
            artifact_path.metadata().unwrap().len().into();
        receipt["finalProducts"]["guiArtifacts"][0]["astrometry"]["catalogManaged"] = false.into();
        assert!(validate(&receipt).is_err());
        receipt["finalProducts"]["guiArtifacts"][0]["astrometry"]["catalogManaged"] = true.into();
        receipt["finalProducts"]["guiArtifacts"][0]["path"] = "../escaped.fits".into();
        assert!(validate(&receipt)
            .unwrap_err()
            .contains("unsafe relative path"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn project_progress_preserves_panel_context_and_reserves_completion() {
        let value = serde_json::json!({"stage":"complete", "status":"completed", "current":9, "total":2,
            "overallFraction":1.2, "scope":"panel", "panelId":"cartwheel__b", "panelTarget":"Cartwheel",
            "panelFilter":"B", "panelIndex":1, "panelCount":4});
        let event = normalize_progress("job", &value);
        assert_eq!(event.stage_id.as_deref(), Some("publish"));
        assert_eq!(event.fraction, 1.0);
        assert_eq!(event.overall_fraction, Some(0.99));
        assert_eq!(event.completed_units, Some(2));
        let serialized = serde_json::to_value(event).unwrap();
        assert_eq!(serialized["scope"], "panel");
        assert_eq!(serialized["panelFilter"], "B");
        assert_eq!(serialized["panelIndex"], 1);
        assert_eq!(serialized["panelCount"], 4);
    }

    #[test]
    fn progress_normalization_preserves_legacy_counts_and_failure_state() {
        let legacy = normalize_progress(
            "job",
            &serde_json::json!({"stage":"quality-control", "status":"running", "current":1, "total":2}),
        );
        assert_eq!(legacy.fraction, 0.5);
        assert_eq!(legacy.state, "running");
        assert!(legacy.overall_fraction.is_none());
        assert!(serde_json::to_value(legacy)
            .unwrap()
            .get("panelId")
            .is_none());
        let failed = normalize_progress(
            "job",
            &serde_json::json!({"stage":"failed", "status":"completed", "overallFraction":-1.0}),
        );
        assert_eq!(failed.state, "failed");
        assert_eq!(failed.overall_fraction, Some(0.0));
        assert_eq!(failed.fraction, 0.0);
        for stage in ["alignment", "color", "verify", "publish"] {
            assert_eq!(stage_id(stage).as_deref(), Some(stage));
        }
    }

    #[test]
    fn execution_failure_prefers_structured_reason_and_keeps_fallbacks() {
        let result = serde_json::json!({
            "code": "QC_INSUFFICIENT_LIGHTS",
            "message": "B: quality gate admitted 1 Light frame; at least 2 are required"
        });
        assert_eq!(
            execution_failure_detail(&result, "unrelated diagnostic"),
            result["message"].as_str().unwrap()
        );
        assert_eq!(
            execution_failure_detail(&serde_json::json!({"message": "  "}), " legacy detail "),
            "legacy detail"
        );
        assert_eq!(
            execution_failure_detail(&serde_json::json!({}), ""),
            "project execution failed closed"
        );
        assert_eq!(
            execution_failure_detail(
                &serde_json::json!({"message": "x".repeat(MAX_DIAGNOSTIC_BYTES + 1)}),
                ""
            )
            .len(),
            MAX_DIAGNOSTIC_BYTES
        );
    }

    #[cfg(unix)]
    #[test]
    fn malformed_response_keeps_exception_tail_and_never_echoes_stdout() {
        use std::os::unix::process::ExitStatusExt;
        let mut diagnostics = String::new();
        append_diagnostic(&mut diagnostics, &"earlier warning 隐私".repeat(2048));
        append_diagnostic(&mut diagnostics, "Traceback (most recent call last):");
        append_diagnostic(
            &mut diagnostics,
            "TypeError: final product header is invalid",
        );
        assert!(diagnostics.len() <= MAX_DIAGNOSTIC_BYTES);
        let (code, detail) = decode_project_response(
            b"PRIVATE_RAW_SOURCE_CONTENT and a partial response",
            Some(ExitStatus::from_raw(0)),
            &diagnostics,
        )
        .unwrap_err();
        assert_eq!(code, "PROJECT_RESPONSE_INVALID");
        assert!(detail.contains("invalid JSON result"));
        assert!(detail.ends_with("TypeError: final product header is invalid"));
        assert!(!detail.contains("PRIVATE_RAW_SOURCE_CONTENT"));
        assert!(detail.len() <= MAX_DIAGNOSTIC_BYTES);

        let (code, detail) =
            decode_project_response(b"", Some(ExitStatus::from_raw(1 << 8)), &diagnostics)
                .unwrap_err();
        assert_eq!(code, "PROJECT_EXECUTION_FAILED");
        assert!(detail.contains("no JSON result"));
        assert!(detail.contains("exit status: 1"));
        assert!(detail.ends_with("TypeError: final product header is invalid"));
        assert!(detail.len() <= MAX_DIAGNOSTIC_BYTES);
        assert!(decode_project_response(b"{}", None, "wait failed").is_err());
        assert!(decode_project_response(
            b"{\"success\":true}",
            Some(ExitStatus::from_raw(1 << 8)),
            "worker crashed"
        )
        .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn crashed_project_streams_stderr_to_gui_without_a_completion_event() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::mpsc;
        use tauri::Listener;

        let root = std::env::temp_dir().join(new_public_identifier("project-crash-test").unwrap());
        std::fs::create_dir_all(&root).unwrap();
        let light = root.join("fixture.fit");
        std::fs::write(&light, b"synthetic source fixture").unwrap();
        let script = root.join("fake-project-sidecar");
        std::fs::write(
            &script,
            r###"#!/usr/bin/env python3
import sys
print("earlier warnings " * 1024, file=sys.stderr)
print("Traceback (most recent call last):", file=sys.stderr)
print("TypeError: final product header is invalid", file=sys.stderr)
sys.exit(1)
"###,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let (error_sender, error_receiver) = mpsc::channel();
        let (complete_sender, complete_receiver) = mpsc::channel();
        handle.listen(ERROR_EVENT, move |event| {
            let _ = error_sender.send(event.payload().to_owned());
        });
        handle.listen(COMPLETE_EVENT, move |event| {
            let _ = complete_sender.send(event.payload().to_owned());
        });
        let registry = Arc::new(ProjectRegistry::default());
        let run = start_with(
            handle,
            registry.clone(),
            ProjectRunRequest {
                sources: vec![UiRunSource {
                    source_id: "light-1".to_owned(),
                    role: "LIGHT".to_owned(),
                    paths: vec![light.to_string_lossy().into_owned()],
                    recursive: false,
                }],
                project_name: "synthetic crash".to_owned(),
                run_label: String::new(),
                recipe: UiRecipeOptions {
                    balanced: true,
                    drizzle_enabled: false,
                    local_normalization_enabled: false,
                    solver_required: true,
                    calibration_workflow: default_calibration_workflow(),
                },
                master_metadata_overrides: vec![],
                raw_frame_metadata_overrides: vec![],
                review_selections: vec![],
                output_parent_directory: root.to_string_lossy().into_owned(),
            },
            crate::sidecar::EngineExecutable { path: script },
        )
        .unwrap();
        let error: serde_json::Value = serde_json::from_str(
            &error_receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(error["jobId"], run.job_id);
        assert_eq!(error["code"], "PROJECT_EXECUTION_FAILED");
        let message = error["message"].as_str().unwrap();
        assert!(message.contains("no JSON result"));
        assert!(message.ends_with("TypeError: final product header is invalid"));
        assert!(message.len() <= MAX_DIAGNOSTIC_BYTES);
        assert!(complete_receiver.try_recv().is_err());
        assert!(registry.jobs.lock().unwrap().is_empty());
        assert!(!Path::new(&run.output_directory).exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsafe_roles_and_ids_are_rejected_before_spawning() {
        assert!(checked_role("MASTER_DARK"));
        assert!(!checked_role("MASTER_LIGHT"));
        assert!(checked_id("light-0001"));
        assert!(!checked_id("--request-json"));
    }

    fn master_override_with_units(
        numeric_domain: Option<&str>,
        normalized_unit_scale: Option<f64>,
    ) -> UiMasterOverride {
        UiMasterOverride {
            source_sha256: format!("sha256:{}", "a".repeat(64)),
            camera: Some("QHY268M".to_owned()),
            gain: Some(0.0),
            offset: Some(30.0),
            binning: Some([1, 1]),
            filter: Some("NONE".to_owned()),
            cfa_pattern: Some("NONE".to_owned()),
            readout_mode: Some("HIGH GAIN 2CMS".to_owned()),
            temperature_celsius: Some(-10.0),
            exposure_seconds: Some(300.0),
            bias_included: Some(true),
            numeric_domain: numeric_domain.map(str::to_owned),
            normalized_unit_scale,
        }
    }

    #[test]
    fn standard_master_workflow_reaches_worker_without_fabricated_metadata() {
        let root = std::env::temp_dir().join(format!(
            "standard-master-wire-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let light = root.join("light.fits");
        let dark = root.join("masterDark.xisf");
        std::fs::write(&light, b"light input").unwrap();
        std::fs::write(&dark, b"master input").unwrap();
        let mut request = ProjectRunRequest {
            sources: vec![
                UiRunSource {
                    source_id: "light-1".into(),
                    role: "LIGHT".into(),
                    paths: vec![light.to_string_lossy().into_owned()],
                    recursive: false,
                },
                UiRunSource {
                    source_id: "dark-1".into(),
                    role: "MASTER_DARK".into(),
                    paths: vec![dark.to_string_lossy().into_owned()],
                    recursive: false,
                },
            ],
            project_name: "Standard masters".into(),
            run_label: String::new(),
            recipe: UiRecipeOptions {
                balanced: true,
                drizzle_enabled: false,
                local_normalization_enabled: false,
                solver_required: true,
                calibration_workflow: "mono-standard-v1".into(),
            },
            master_metadata_overrides: vec![],
            raw_frame_metadata_overrides: vec![],
            review_selections: vec![],
            output_parent_directory: root.to_string_lossy().into_owned(),
        };
        let output = root.join("new-output");
        let value = project_request_json(&request, &output).unwrap();
        assert_eq!(
            value["recipe"]["calibration"]["workflow"],
            "mono-standard-v1"
        );
        assert_eq!(value["recipe"]["calibration"]["bias"], "OPTIONAL");
        assert_eq!(
            value["recipe"]["calibration"]["masterMetadataOverrides"],
            serde_json::json!([])
        );
        assert_eq!(
            value["recipe"]["rawFrameMetadataOverrides"],
            serde_json::json!([])
        );
        assert!(!output.exists());
        request.recipe.calibration_workflow = "strict-v1".into();
        assert!(project_request_json(&request, &output)
            .unwrap_err()
            .contains("MasterDark requires"));
        request.recipe.calibration_workflow = "unknown".into();
        assert!(project_request_json(&request, &output)
            .unwrap_err()
            .contains("unsupported calibration workflow"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn master_overrides_preserve_unknowns_and_explicit_zero() {
        let override_: UiMasterOverride = serde_json::from_value(serde_json::json!({
            "sourceSha256": format!("sha256:{}", "a".repeat(64)),
            "biasIncluded": false, "offset": 0.0
        }))
        .unwrap();
        validate_master_override(&override_).unwrap();
        let encoded = serde_json::to_value(override_).unwrap();
        assert_eq!(encoded["biasIncluded"], false);
        assert_eq!(encoded["offset"], 0.0);
        assert!(encoded.get("temperatureCelsius").is_none());
        assert!(encoded.get("gain").is_none());
        assert!(encoded.get("camera").is_none());
        let unknown: UiMasterOverride = serde_json::from_value(serde_json::json!({
            "sourceSha256": format!("sha256:{}", "a".repeat(64)), "camera":"UNKNOWN"
        }))
        .unwrap();
        assert!(validate_master_override(&unknown).is_err());
    }

    #[test]
    fn additive_master_numeric_units_are_optional_but_atomic_and_bounded() {
        assert!(validate_master_override(&master_override_with_units(None, None)).is_ok());
        assert!(validate_master_override(&master_override_with_units(
            Some("NORMALIZED_UNIT"),
            Some(1.0),
        ))
        .is_ok());
        assert!(validate_master_override(&master_override_with_units(
            Some("SENSOR_CODE"),
            Some(65535.0),
        ))
        .is_ok());
        assert!(validate_master_override(&master_override_with_units(
            Some("NORMALIZED_UNIT"),
            None,
        ))
        .is_err());
        assert!(validate_master_override(&master_override_with_units(
            Some("ELECTRONS"),
            Some(1.0),
        ))
        .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn application_shutdown_terminates_every_project_process_tree() {
        let registry = ProjectRegistry::default();
        let mut command = std::process::Command::new("/bin/sleep");
        command.arg("30");
        platform::configure_child_process(&mut command);
        let child = Arc::new(Mutex::new(command.spawn().expect("project child")));
        registry.jobs.lock().unwrap().insert(
            "project-shutdown-test".to_owned(),
            ProjectJob {
                child: child.clone(),
                request_path: std::env::temp_dir().join(format!(
                    "{}.json",
                    new_public_identifier("project-shutdown-request").unwrap()
                )),
            },
        );

        let request_path = registry
            .jobs
            .lock()
            .unwrap()
            .get("project-shutdown-test")
            .unwrap()
            .request_path
            .clone();
        std::fs::write(&request_path, b"private request").expect("private request fixture");

        terminate_all(&registry).expect("terminate project children");
        let status = child.lock().unwrap().wait().expect("reap project child");
        assert!(!status.success());
        assert!(!request_path.exists());
        assert!(registry.shutting_down.load(Ordering::Acquire));
        terminate_all(&registry).expect("shutdown is idempotent");
    }

    #[test]
    fn managed_astrometry_schema_rejects_unbound_catalogs() {
        let value = serde_json::json!({
            "referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":1.0,"centerDecDegrees":2.0,
            "pixelScaleArcsec":1.0,"rotationDegrees":0.0,"rmsPixels":0.2,"rmsArcsec":0.3,"matchedStars":30,
            "parity":"POSITIVE","catalogIdentity":"1".repeat(64),"indexIdentities":["astrometry.net:index:4108:healpix:1:hpnside:1"],
            "correspondenceSha256":"2".repeat(64),"catalogManaged":false,"installedSetIdentity":"3".repeat(64),
            "catalogManifestSha256":"4".repeat(64),"indexArtifacts":[],"wcsSha256":"5".repeat(64)
        });
        let parsed: AstrometricSolutionReceipt = serde_json::from_value(value).unwrap();
        assert!(parsed.validate().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn fake_run_project_streams_progress_and_revalidates_the_outer_receipt() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::mpsc;
        use tauri::Listener;

        let root = std::env::temp_dir()
            .join(new_public_identifier("project-controller-test").expect("temporary identifier"));
        let input = root.join("Unicode 输入");
        let output_parent = root.join("Unicode 输出");
        std::fs::create_dir_all(&input).expect("input directory");
        std::fs::create_dir_all(&output_parent).expect("output directory");
        let light = input.join("盾牌座 light.fit");
        std::fs::write(&light, b"source-frame").expect("source fixture");
        let script = root.join("fake-project-sidecar");
        let source = r###"#!/usr/bin/env python3
import hashlib, json, pathlib, sys
assert sys.argv[1] == "run-project" and sys.argv[2] == "--request-json"
request_path = pathlib.Path(sys.argv[3])
request = json.loads(request_path.read_text())
assert request["sources"][0]["expectedRole"] == "LIGHT"
assert "盾牌座" in request["sources"][0]["hostPath"]
assert request["recipe"]["rawFrameMetadataOverrides"][0]["cfaPattern"] == "NONE"
print(json.dumps({"type":"progress","event":{"stage":"quality-control","status":"RUNNING","current":1,"total":2,"message":"checked one"}}), file=sys.stderr, flush=True)
output = pathlib.Path(request["outputDirectory"])
products = output / "products"
products.mkdir(parents=True)
artifact = products / "盾牌座_R_mosaic.fits"
artifact.write_bytes(b"verified-solved-product")
digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
astro = {
  "referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":281.0,"centerDecDegrees":-6.0,
  "pixelScaleArcsec":1.4,"rotationDegrees":0.0,"rmsPixels":0.3,"rmsArcsec":0.42,"matchedStars":73,
  "parity":"POSITIVE","catalogIdentity":"2"*64,"indexIdentities":["astrometry.net:index:4108:healpix:123:hpnside:4"],
  "correspondenceSha256":"3"*64,"catalogManaged":True,"installedSetIdentity":"5"*64,"catalogManifestSha256":"6"*64,
  "indexArtifacts":[{"indexId":"4108","relativeName":"index-4108.fits","sizeBytes":94550400,"sha256":"7"*64,"manifestSha256":"6"*64,"installedSetIdentity":"5"*64}],
  "wcsSha256":"4"*64,"imageShape":[4176,6248],"state":"SOLVED"
}
record = {"path":"products/盾牌座_R_mosaic.fits","relativePath":"products/盾牌座_R_mosaic.fits","kind":"SOLVED_MONO_FITS","sha256":digest,"sizeBytes":artifact.stat().st_size,"filter":"R","astrometry":astro,"finalGate":{"status":"PASS"}}
receipt = {"schemaVersion":1,"success":True,"state":"SOLVED","finalProducts":{"guiArtifacts":[record],"resultGate":{"status":"PASS","allMonoProductsSolved":True,"managedCatalogEvidenceRequired":True,"sourceIdentityVerifiedAtCommit":True,"mosaicCoverageOverlapSeamPassed":True,"rgbState":"MONO_ONLY_CHANNELS_MISSING"}}}
receipt_path = output / "receipt.json"
receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
print(json.dumps({"success":True,"code":"PROJECT_MONO_SUCCEEDED","state":"SOLVED","outputDirectory":str(output),"evidenceDirectory":None,"receiptPath":str(receipt_path),"productPaths":[str(artifact)],"previewPaths":[],"passedLightPaths":[],"excludedLightPaths":[],"monoFilters":["R"],"colorProductPath":None}), flush=True)
"###;
        std::fs::write(&script, source).expect("write fake sidecar");
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).unwrap();

        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let (progress_sender, progress_receiver) = mpsc::channel();
        let (complete_sender, complete_receiver) = mpsc::channel();
        handle.listen(PROGRESS_EVENT, move |event| {
            let _ = progress_sender.send(event.payload().to_owned());
        });
        handle.listen(COMPLETE_EVENT, move |event| {
            let _ = complete_sender.send(event.payload().to_owned());
        });
        let receipt = start_with(
            handle,
            Arc::new(ProjectRegistry::default()),
            ProjectRunRequest {
                sources: vec![UiRunSource {
                    source_id: "light-0001".to_owned(),
                    role: "LIGHT".to_owned(),
                    paths: vec![light.to_string_lossy().into_owned()],
                    recursive: false,
                }],
                project_name: "盾牌座 马赛克".to_owned(),
                run_label: "盾牌座 马赛克".to_owned(),
                recipe: UiRecipeOptions {
                    balanced: true,
                    drizzle_enabled: false,
                    local_normalization_enabled: true,
                    solver_required: true,
                    calibration_workflow: default_calibration_workflow(),
                },
                master_metadata_overrides: vec![],
                raw_frame_metadata_overrides: vec![UiRawFrameOverride {
                    source_sha256: format!("sha256:{}", sha256_file(&light).unwrap()),
                    cfa_pattern: "NONE".to_owned(),
                }],
                review_selections: vec![],
                output_parent_directory: output_parent.to_string_lossy().into_owned(),
            },
            crate::sidecar::EngineExecutable { path: script },
        )
        .expect("launch fake project");
        assert!(receipt.accepted);
        let progress: serde_json::Value = serde_json::from_str(
            &progress_receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("project progress"),
        )
        .expect("progress JSON");
        assert_eq!(progress["stageId"], "quality-control");
        assert_eq!(progress["fraction"], 0.5);
        let completion: serde_json::Value = serde_json::from_str(
            &complete_receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("project completion"),
        )
        .expect("completion JSON");
        assert_eq!(completion["gate"]["decision"], "ready");
        assert_eq!(completion["artifacts"][0]["kind"], "SOLVED_MONO_FITS");
        assert_eq!(
            completion["artifacts"][0]["receipt"]["astrometry"]["catalogManaged"],
            true
        );
        assert_eq!(completion["artifacts"][1]["kind"], "RECEIPT");
        let _ = std::fs::remove_dir_all(root);
    }
}
