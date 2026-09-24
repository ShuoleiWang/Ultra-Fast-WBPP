//! Product-level project execution through the strict local `run-project` CLI.
//!
//! The webview submits typed roles and recipe choices. This controller writes a
//! mode-0600 request file, launches the local sidecar without a shell, streams
//! bounded progress, removes the request file, and independently revalidates
//! the published receipt and every GUI artifact before emitting completion.

mod previews;
#[cfg(test)]
pub(crate) use previews::load_blink_preview_with;
use previews::*;
pub(crate) use previews::{
    blink_session_name_parts, image_data_url, load_blink_preview, resolve_blink_preview,
};
mod requests;
pub(crate) use requests::checked_flag_code;
use requests::*;
mod astrometry;
use astrometry::AstrometricSolutionReceipt;
mod completion;
use completion::*;
mod execution;
#[cfg(test)]
use execution::*;
pub(crate) use execution::{cancel, start, terminate_all};

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Runtime};

use crate::platform::{self, ManagedChild};
use crate::sidecar::{discover_engine, new_public_identifier, LossyLines};

const PROGRESS_EVENT: &str = "ufwbpp://pipeline-progress";
const COMPLETE_EVENT: &str = "ufwbpp://pipeline-complete";
const ERROR_EVENT: &str = "ufwbpp://pipeline-error";
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
    child: Arc<Mutex<ManagedChild>>,
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
    #[serde(default = "default_drizzle_scale")]
    drizzle_scale: u32,
    #[serde(default = "default_drizzle_drop_shrink")]
    drizzle_drop_shrink: f64,
    #[serde(default = "default_drizzle_kernel")]
    drizzle_kernel: String,
    solver_required: bool,
    #[serde(default = "default_calibration_workflow")]
    calibration_workflow: String,
    // Opt-in proper coaddition: absent unless the user checked it, so a
    // default run sends exactly the recipe it sent before it existed.  The
    // engine's robust IRLS combination is not offered here: it measured worse
    // than the default, and an unknown `integration` key is refused.
    #[serde(default)]
    proper_coaddition: Option<UiProperCoaddition>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiProperCoaddition {
    pub(crate) enabled: bool,
    #[serde(default = "default_outlier_handling")]
    pub(crate) outlier_handling: String,
    #[serde(default = "default_apodization_pixels")]
    pub(crate) apodization_pixels: u32,
}

const OUTLIER_HANDLING: [&str; 2] = ["reuse-rejection", "none"];

fn default_outlier_handling() -> String {
    "reuse-rejection".to_owned()
}

fn default_apodization_pixels() -> u32 {
    64
}

/// The engine refuses proper coaddition on a drizzled grid, and the block
/// names a closed set of values; refusing here keeps the run from starting.
pub(crate) fn validate_advanced_algorithms(recipe: &UiRecipeOptions) -> Result<(), String> {
    if let Some(coaddition) = &recipe.proper_coaddition {
        if coaddition.enabled {
            if recipe.drizzle_enabled {
                return Err("proper coaddition cannot run together with drizzle".to_owned());
            }
            if !OUTLIER_HANDLING.contains(&coaddition.outlier_handling.as_str()) {
                return Err(
                    "proper coaddition outlier handling must be reuse-rejection or none".to_owned(),
                );
            }
            if coaddition.apodization_pixels > 512 {
                return Err(
                    "proper coaddition apodization must be between 0 and 512 pixels".to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn default_calibration_workflow() -> String {
    "strict-v1".to_owned()
}

fn default_drizzle_scale() -> u32 {
    2
}

fn default_drizzle_drop_shrink() -> f64 {
    0.9
}

fn default_drizzle_kernel() -> String {
    "square".to_owned()
}

const DRIZZLE_KERNELS: [&str; 4] = ["square", "circular", "gaussian", "point"];

/// The engine recipe rejects the same values; checking here keeps the error
/// next to the control instead of a failed run.
fn validate_drizzle_options(recipe: &UiRecipeOptions) -> Result<(), String> {
    if !recipe.drizzle_enabled {
        return Ok(());
    }
    if !(1..=4).contains(&recipe.drizzle_scale) {
        return Err("drizzle scale must be 1, 2, 3 or 4".to_owned());
    }
    if !(recipe.drizzle_drop_shrink.is_finite()
        && (0.1..=1.0).contains(&recipe.drizzle_drop_shrink))
    {
        return Err("drizzle drop shrink must be between 0.1 and 1".to_owned());
    }
    if !DRIZZLE_KERNELS.contains(&recipe.drizzle_kernel.as_str()) {
        return Err("drizzle kernel must be square, circular, gaussian or point".to_owned());
    }
    Ok(())
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

/// Where a selection file came from; recorded by the engine, never enforced
/// (a hand-written selection has none).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiSelectionOrigin {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    blink_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flags_policy_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    created_at: Option<String>,
}

/// One Light's decision from the blink view, bound to the file's content.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiSelectionDecision {
    source_sha256: String,
    decision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_decision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

/// The `selection-v1` file the blink view produces: the user's KEEP/DROP per
/// Light, sent to `run-project` as the top-level `selection` and applied by
/// the engine under the `explicit-v1` policy.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiSelection {
    schema_version: u32,
    kind: String,
    policy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    origin: Option<UiSelectionOrigin>,
    undecided: String,
    decisions: Vec<UiSelectionDecision>,
}

const SELECTION_KIND: &str = "ultra-fast-wbpp-selection";
const SELECTION_POLICY: &str = "explicit-v1";
const MAX_SELECTION_DECISIONS: usize = 10_000;
const MAX_SELECTION_FLAGS: usize = 32;
const MAX_SELECTION_NOTE_CHARS: usize = 1000;
const MAX_SELECTION_ORIGIN_CHARS: usize = 256;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct BlinkReviewProof {
    session_directory: String,
    manifest_sha256: String,
    reviewed_source_sha256s: Vec<String>,
    confirmed_channel_ids: Vec<String>,
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
    /// The blink decisions; `None` keeps the legacy gate (with the optional
    /// REVIEW approvals in `review_selections`).
    #[serde(default)]
    selection: Option<UiSelection>,
    #[serde(default)]
    blink_review: Option<BlinkReviewProof>,
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
    /// The PNG previews as data URLs, so the result page shows them from any
    /// output location (another drive, a UNC share) without the webview's
    /// asset protocol having to reach that path.
    #[serde(skip_serializing_if = "Option::is_none")]
    preview_data_url: Option<String>,
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
    /// With an explicit selection: `USER_DROP` for a Light the user dropped,
    /// `USER_KEEP_OVERRIDE` for a kept Light that carried an EXCLUDE flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    /// The blink flag codes of the frame, when the run recorded them.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    flags: Vec<String>,
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
/// A mono preview is an 8-bit PNG with at most a 2048-pixel long edge (1.0 to
/// 1.5 MB for a 26 MP channel).  The RGB preview is the 16-bit PNG of the same
/// size, whose noisy low bits barely compress: 8.4 MB at 1541 pixels, up to
/// about 14 MB at 2048.  Both kinds are carried, mono first; a file over its
/// bound or over the total budget is left to the copy on disk.
const MAX_MONO_PREVIEW_BYTES: u64 = 2 * 1024 * 1024;
const MAX_RGB_PREVIEW_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARTIFACT_PREVIEW_TOTAL_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
/// SOI marker plus the first marker byte, common to every JPEG variant.
const JPEG_SIGNATURE: [u8; 3] = [0xff, 0xd8, 0xff];

/// Standard base64 with padding; the previews are small, so no crate.
#[cfg(test)]
mod tests;
