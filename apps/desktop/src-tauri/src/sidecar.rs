use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStderr, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use openastroflow_app_core::protocol::{
    ExecuteMessage, PeerRole, PlanMessage, ProgressState, ProtocolCursor, MAX_NDJSON_LINE_BYTES,
};
use openastroflow_app_core::{
    decode_ndjson_line, encode_ndjson_line, ArtifactReceipt, BackendCapabilities, BackendFeature,
    GateDecision, HandshakeMessage, HardwareProfile, RequiredResultGate, SafeFileName, StageKind,
    StageReceipt, Validate, WorkerEnvelope, WorkerMessage, WORKER_PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager, Runtime};

use crate::platform::{self, ManagedChild};

const PROGRESS_EVENT: &str = "openastroflow://pipeline-progress";
const ARTIFACT_EVENT: &str = "openastroflow://pipeline-artifact";
const COMPLETE_EVENT: &str = "openastroflow://pipeline-complete";
const ERROR_EVENT: &str = "openastroflow://pipeline-error";
const LOG_EVENT: &str = "openastroflow://pipeline-log";
const MAX_DIAGNOSTIC_BYTES: usize = 8 * 1024;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

static IDENTIFIER_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub(crate) struct EngineExecutable {
    pub(crate) path: PathBuf,
}

impl EngineExecutable {
    pub(crate) fn command(&self, subcommand: &str) -> Command {
        let mut command = Command::new(&self.path);
        command.arg(subcommand);
        // The worker's stdio carries JSON with user paths and captions.  A
        // Windows console code page (936 on a Chinese system) would replace
        // or garble every non-ANSI character; Python's UTF-8 mode makes the
        // pipes, the file-system encoding and the console UTF-8 everywhere.
        command
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8");
        platform::configure_child_process(&mut command);
        command
    }
}

const TEXT_BUSY_RETRIES: u32 = 20;
const TEXT_BUSY_RETRY_DELAY: Duration = Duration::from_millis(50);

/// Launches a sidecar command under the platform's process-tree control,
/// retrying for about a second while its executable is momentarily "text
/// busy".
///
/// On Unix, a fork elsewhere in this process (another sidecar launch, a
/// test thread) inherits every open descriptor until its own exec; if one of
/// them is a write handle on a just-installed executable, executing that
/// file meanwhile fails with `ETXTBSY` even though the writer already closed
/// it.  rustc, cargo and git retry the same way; any other launch error is
/// returned at once.
pub(crate) fn spawn_sidecar(command: &mut Command) -> std::io::Result<ManagedChild> {
    let mut attempt = 0;
    loop {
        match ManagedChild::spawn(command) {
            Err(error) if is_text_busy(&error) && attempt < TEXT_BUSY_RETRIES => {
                attempt += 1;
                std::thread::sleep(TEXT_BUSY_RETRY_DELAY);
            }
            result => return result,
        }
    }
}

/// `Command::output` with the same text-busy retry and process-tree control
/// as [`spawn_sidecar`].
pub(crate) fn sidecar_output(command: &mut Command) -> std::io::Result<std::process::Output> {
    let mut attempt = 0;
    loop {
        match ManagedChild::output(command) {
            Err(error) if is_text_busy(&error) && attempt < TEXT_BUSY_RETRIES => {
                attempt += 1;
                std::thread::sleep(TEXT_BUSY_RETRY_DELAY);
            }
            result => return result,
        }
    }
}

/// Lines of a worker's diagnostic stream, decoded leniently.
///
/// `BufRead::lines` stops at the first line that is not valid UTF-8, so one
/// stray byte from a solver's console output would end progress reporting
/// for the rest of the run.  Here an invalid sequence only becomes U+FFFD in
/// that line; reading stops at end of stream or on an I/O error.
pub(crate) struct LossyLines<R: BufRead> {
    reader: R,
    buffer: Vec<u8>,
}

impl<R: BufRead> LossyLines<R> {
    pub(crate) fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: Vec::new(),
        }
    }
}

impl<R: BufRead> Iterator for LossyLines<R> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        self.buffer.clear();
        match self.reader.read_until(b'\n', &mut self.buffer) {
            Ok(0) | Err(_) => None,
            Ok(_) => {
                let mut line = self.buffer.as_slice();
                if let Some(rest) = line.strip_suffix(b"\n") {
                    line = rest;
                }
                if let Some(rest) = line.strip_suffix(b"\r") {
                    line = rest;
                }
                Some(String::from_utf8_lossy(line).into_owned())
            }
        }
    }
}

#[cfg(unix)]
fn is_text_busy(error: &std::io::Error) -> bool {
    error.raw_os_error() == Some(libc::ETXTBSY)
}

#[cfg(not(unix))]
fn is_text_busy(_error: &std::io::Error) -> bool {
    false
}

#[derive(Default)]
pub(crate) struct PipelineRegistry {
    jobs: Mutex<HashMap<String, Arc<Mutex<ManagedChild>>>>,
    shutting_down: AtomicBool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RuntimeCapabilities {
    pub platform: &'static str,
    pub chip: String,
    pub cpu_backend: String,
    pub gpu_backend: String,
    pub optimization_tier: &'static str,
    pub available: bool,
    pub drizzle_available: bool,
    pub solver_available: bool,
    pub runtime_version: Option<String>,
    pub unavailable_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InspectRequest {
    pub paths: Vec<String>,
    pub role_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InspectedSource {
    pub role: String,
    pub paths: Vec<String>,
    pub file_count: usize,
    pub confidence: f32,
    pub needs_confirmation: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InspectResponse {
    pub sources: Vec<InspectedSource>,
    pub assets: Vec<InspectedAsset>,
    pub total_files: usize,
    pub project_name: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct InspectQualityRequest {
    pub paths: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct InspectCalibrationRequest {
    pub paths: Vec<String>,
    pub recipe: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CalibrationMatchCounts {
    raw_count: usize,
    master_count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CalibrationGroupInspection {
    group_id: String,
    target: String,
    filter: String,
    light_count: usize,
    observed_dates: Vec<String>,
    status: String,
    matches: BTreeMap<String, CalibrationMatchCounts>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CalibrationInspectionIssue {
    code: String,
    severity: String,
    message: String,
    paths: Vec<String>,
    light_groups: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CalibrationInspection {
    schema_version: u32,
    status: String,
    calibration_ready: bool,
    groups: Vec<CalibrationGroupInspection>,
    issues: Vec<CalibrationInspectionIssue>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InspectedQualityEvidence {
    pub code: String,
    pub family: String,
    pub severity: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InspectedLightQuality {
    pub path: String,
    pub source_sha256: Option<String>,
    pub disposition: String,
    pub decision: String,
    pub confidence: String,
    pub star_count: usize,
    pub summary: String,
    /// `false` when the quality pass found no transform for the frame: the
    /// run's registration would fail on it, so it cannot be approved.
    #[serde(default)]
    pub registrable: Option<bool>,
    pub preview_data_url: Option<String>,
    pub preview_sha256: Option<String>,
    pub evidence: Vec<InspectedQualityEvidence>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct QualityInspection {
    pub schema_version: u32,
    pub gate_policy_digest: String,
    pub workers: usize,
    pub counts: BTreeMap<String, usize>,
    pub frames: Vec<InspectedLightQuality>,
}

/// A master flat the blink measurement may use for the flat-corrected
/// gradient flag; optional, one per filter.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct BlinkMasterFlat {
    pub filter: String,
    pub path: String,
}

/// The webview's "Blink & select" request: the Lights to measure, and
/// optionally the master flats and a worker count for the engine.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct BlinkMeasureRequest {
    pub paths: Vec<String>,
    #[serde(default)]
    pub master_flats: Vec<BlinkMasterFlat>,
    #[serde(default)]
    pub workers: Option<usize>,
}

/// Frame counts of a blink manifest: `exclude` are the frames pre-marked DROP,
/// `attention` the kept frames with a flag, `clean` the rest.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkCounts {
    pub frames: usize,
    pub exclude: usize,
    pub attention: usize,
    pub clean: usize,
}

/// The frame every other frame of a channel is registered and normalised
/// to for blinking.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkChannelReference {
    pub index: usize,
    pub source_sha256: String,
    pub rule: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// One QC channel (target, filter, camera geometry, exposure bucket) of the
/// manifest.  Statistics, stretch, geometry and nights pass through `extra`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkChannel {
    pub channel_id: String,
    pub target: String,
    pub filter: String,
    pub frame_count: usize,
    pub reference: BlinkChannelReference,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// One flag the engine raised on a frame; `EXCLUDE` pre-marks the frame
/// DROP, `ATTENTION` only highlights it.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkFlag {
    pub code: String,
    pub severity: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// The rendered previews of a frame, relative to the session directory.
/// `filmstrip_data_url` is added by the desktop for the frames within the
/// transport budget; the rest are fetched with `load_blink_preview`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkPreviews {
    #[serde(default)]
    pub filmstrip: Option<String>,
    #[serde(default)]
    pub zoom: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filmstrip_data_url: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// One Light of the manifest.  The fields the desktop validates are typed;
/// metrics, score, gate, notes, transform and normalisation pass through
/// `extra` unchanged.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkFrame {
    pub index: usize,
    pub channel_id: String,
    pub path: String,
    pub source_sha256: String,
    pub reference: bool,
    pub default_decision: String,
    pub flags: Vec<BlinkFlag>,
    #[serde(default)]
    pub previews: BlinkPreviews,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// The `blink-manifest-v1` document `blink-measure` prints and writes as
/// `manifest.json` in the session directory.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkManifest {
    pub schema_version: u32,
    pub kind: String,
    pub session_id: String,
    pub session_directory: String,
    pub inventory_sha256: String,
    pub gate_policy_digest: String,
    pub flags_policy_digest: String,
    pub counts: BlinkCounts,
    pub channels: Vec<BlinkChannel>,
    pub frames: Vec<BlinkFrame>,
    /// `sha256:` digest of the session's `manifest.json`, added by the
    /// desktop: the `origin.blinkManifestSha256` of the selection file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_sha256: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct HashSourcesRequest {
    pub paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SourceHash {
    pub path: String,
    pub source_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HashSourcesResponse {
    pub entries: Vec<SourceHash>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunSource {
    pub role: String,
    pub paths: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunRequest {
    pub sources: Vec<RunSource>,
    pub recipe_id: String,
    pub output_parent_directory: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunReceipt {
    pub job_id: String,
    pub accepted: bool,
    pub execution_mode: &'static str,
    pub output_directory: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProgressEvent {
    job_id: String,
    stage_id: Option<String>,
    state: ProgressState,
    fraction: f64,
    completed_units: Option<u64>,
    total_units: Option<u64>,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactEvent {
    job_id: String,
    stage: StageReceipt,
    artifact: ArtifactReceipt,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PipelineErrorEvent {
    job_id: String,
    code: String,
    message: String,
    retryable: bool,
    details: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PipelineLogEvent {
    job_id: String,
    stream: &'static str,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompletedArtifact {
    kind: &'static str,
    name: String,
    path: String,
    detail: String,
    receipt: ArtifactReceipt,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompleteEvent {
    job_id: String,
    output_directory: String,
    artifacts: Vec<CompletedArtifact>,
    gate: openastroflow_app_core::ResultGateReport,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InventoryPayload {
    name: String,
    assets: Vec<InventoryAsset>,
    #[serde(default)]
    issues: Vec<InventoryIssue>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InventoryAsset {
    path: String,
    role: String,
    status: String,
    #[serde(default)]
    width: usize,
    #[serde(default)]
    height: usize,
    #[serde(default)]
    channels: usize,
    #[serde(default, rename = "filter")]
    filter_name: String,
    #[serde(default)]
    target: String,
    #[serde(default)]
    camera: String,
    #[serde(default)]
    exposure_seconds: Option<f64>,
    #[serde(default)]
    observed_at: Option<String>,
    #[serde(default)]
    temperature_celsius: Option<f64>,
    #[serde(default)]
    gain: Option<f64>,
    #[serde(default)]
    offset: Option<f64>,
    #[serde(default = "default_binning")]
    binning_x: usize,
    #[serde(default = "default_binning")]
    binning_y: usize,
    #[serde(default)]
    cfa_pattern: String,
    #[serde(default)]
    readout_mode: String,
    #[serde(default)]
    role_evidence: Vec<String>,
    error_code: Option<String>,
    error_message: Option<String>,
}

fn default_binning() -> usize {
    1
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InspectedAsset {
    path: String,
    role: String,
    width: usize,
    height: usize,
    channels: usize,
    filter: String,
    target: String,
    camera: String,
    exposure_seconds: Option<f64>,
    observed_at: Option<String>,
    temperature_celsius: Option<f64>,
    gain: Option<f64>,
    offset: Option<f64>,
    binning: [usize; 2],
    cfa_pattern: String,
    readout_mode: String,
    source_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InventoryIssue {
    code: String,
    severity: String,
    message: String,
}

#[derive(Debug)]
struct RuntimeProbe {
    executable: EngineExecutable,
    capabilities: BackendCapabilities,
    selected_profile: HardwareProfile,
    implementation_version: String,
}

fn now_ms() -> Result<u64, String> {
    let value = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis();
    u64::try_from(value).map_err(|_| "system clock is outside the supported range".to_owned())
}

fn checked_source_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn new_identifier(prefix: &str) -> Result<String, String> {
    let serial = IDENTIFIER_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(format!("{prefix}-{}-{serial}", now_ms()?))
}

pub(crate) fn new_public_identifier(prefix: &str) -> Result<String, String> {
    new_identifier(prefix)
}

#[cfg(any(debug_assertions, test))]
fn executable_name() -> &'static str {
    if cfg!(windows) {
        "openastroflow-worker.exe"
    } else {
        "openastroflow-worker"
    }
}

fn target_suffixed_name() -> &'static str {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "openastroflow-worker-aarch64-apple-darwin"
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "openastroflow-worker-x86_64-apple-darwin"
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "openastroflow-worker-aarch64-pc-windows-msvc.exe"
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "openastroflow-worker-x86_64-pc-windows-msvc.exe"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "openastroflow-worker-aarch64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "openastroflow-worker-x86_64-unknown-linux-gnu"
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64")
    )))]
    {
        "openastroflow-worker-unsupported-target"
    }
}

fn target_manifest_name() -> String {
    format!(
        "{}.manifest.json",
        target_suffixed_name().trim_end_matches(".exe")
    )
}

fn target_triple_name() -> &'static str {
    target_suffixed_name()
        .trim_start_matches("openastroflow-worker-")
        .trim_end_matches(".exe")
}

fn safe_runtime_relative_path(value: &str) -> Result<PathBuf, String> {
    if value.is_empty() || value.contains('\\') || value.contains('\0') {
        return Err("runtime manifest contains a non-portable path".to_owned());
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err("runtime manifest contains traversal or a non-normal path".to_owned());
    }
    Ok(path.to_path_buf())
}

fn object_keys(value: &serde_json::Value) -> Result<BTreeSet<&str>, String> {
    value
        .as_object()
        .map(|object| object.keys().map(String::as_str).collect())
        .ok_or_else(|| "runtime manifest record must be an object".to_owned())
}

fn collect_runtime_paths(
    root: &Path,
    relative: &Path,
    paths: &mut BTreeSet<String>,
) -> Result<(), String> {
    let directory = root.join(relative);
    let mut children = fs::read_dir(&directory)
        .map_err(|error| format!("cannot enumerate bundled runtime: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("cannot enumerate bundled runtime: {error}"))?;
    children.sort_by_key(std::fs::DirEntry::file_name);
    for child in children {
        let child_relative = relative.join(child.file_name());
        let portable = child_relative
            .to_str()
            .ok_or_else(|| "bundled runtime path is not UTF-8".to_owned())?
            .replace(std::path::MAIN_SEPARATOR, "/");
        paths.insert(portable);
        let metadata = fs::symlink_metadata(child.path())
            .map_err(|error| format!("cannot inspect bundled runtime: {error}"))?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            collect_runtime_paths(root, &child_relative, paths)?;
        }
    }
    Ok(())
}

fn verify_bundled_runtime(resource_root: &Path) -> Result<EngineExecutable, String> {
    let manifest_path = resource_root.join(target_manifest_name());
    let manifest_metadata = fs::symlink_metadata(&manifest_path)
        .map_err(|error| format!("bundled runtime manifest is missing: {error}"))?;
    if !manifest_metadata.is_file() || manifest_metadata.file_type().is_symlink() {
        return Err("bundled runtime manifest must be a regular non-symlink file".to_owned());
    }
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(&manifest_path)
            .map_err(|error| format!("cannot read bundled runtime manifest: {error}"))?,
    )
    .map_err(|error| format!("bundled runtime manifest is invalid JSON: {error}"))?;
    let expected_top = BTreeSet::from([
        "schemaVersion",
        "kind",
        "targetTriple",
        "runtime",
        "protocol",
        "versions",
        "collections",
    ]);
    if object_keys(&manifest)? != expected_top
        || manifest["schemaVersion"] != 2
        || manifest["kind"] != "openastroflow-worker-sidecar"
        || manifest["targetTriple"] != target_triple_name()
    {
        return Err("bundled runtime manifest has the wrong v2 contract".to_owned());
    }
    let runtime = &manifest["runtime"];
    let expected_runtime = BTreeSet::from([
        "directoryName",
        "entryPoint",
        "treeSha256",
        "sizeBytes",
        "fileCount",
        "entryCount",
        "entries",
    ]);
    if object_keys(runtime)? != expected_runtime
        || runtime["directoryName"] != target_suffixed_name().trim_end_matches(".exe")
        || runtime["entryPoint"] != target_suffixed_name()
    {
        return Err("bundled runtime identity does not match this target".to_owned());
    }
    let directory_name = runtime["directoryName"]
        .as_str()
        .ok_or_else(|| "runtime directoryName is invalid".to_owned())?;
    let runtime_root = resource_root.join(safe_runtime_relative_path(directory_name)?);
    let root_metadata = fs::symlink_metadata(&runtime_root)
        .map_err(|error| format!("bundled runtime tree is missing: {error}"))?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err("bundled runtime root must be a real directory".to_owned());
    }
    let entries = runtime["entries"]
        .as_array()
        .ok_or_else(|| "runtime entries must be an array".to_owned())?;
    let mut manifest_paths = BTreeSet::new();
    let mut previous: Option<&str> = None;
    let mut file_count = 0_u64;
    let mut size_bytes = 0_u64;
    for entry in entries {
        let path_value = entry["path"]
            .as_str()
            .ok_or_else(|| "runtime entry path is invalid".to_owned())?;
        if previous.is_some_and(|value| value >= path_value) {
            return Err("runtime entries are not strictly sorted".to_owned());
        }
        previous = Some(path_value);
        let relative = safe_runtime_relative_path(path_value)?;
        manifest_paths.insert(path_value.to_owned());
        let candidate = runtime_root.join(&relative);
        let metadata = fs::symlink_metadata(&candidate)
            .map_err(|error| format!("bundled runtime entry is missing: {path_value}: {error}"))?;
        match entry["type"].as_str() {
            Some("directory") => {
                if object_keys(entry)? != BTreeSet::from(["path", "type"])
                    || !metadata.is_dir()
                    || metadata.file_type().is_symlink()
                {
                    return Err(format!("runtime directory identity changed: {path_value}"));
                }
            }
            Some("symlink") => {
                if object_keys(entry)? != BTreeSet::from(["path", "target", "type"])
                    || !metadata.file_type().is_symlink()
                {
                    return Err(format!("runtime symlink identity changed: {path_value}"));
                }
                let expected = entry["target"]
                    .as_str()
                    .ok_or_else(|| "runtime symlink target is invalid".to_owned())?;
                let actual = fs::read_link(&candidate)
                    .map_err(|error| format!("cannot read runtime symlink: {error}"))?;
                if actual.to_str() != Some(expected) {
                    return Err(format!("runtime symlink target changed: {path_value}"));
                }
            }
            Some("file") => {
                if object_keys(entry)?
                    != BTreeSet::from(["executable", "path", "sha256", "sizeBytes", "type"])
                    || !metadata.is_file()
                    || metadata.file_type().is_symlink()
                {
                    return Err(format!("runtime file identity changed: {path_value}"));
                }
                let expected_size = entry["sizeBytes"]
                    .as_u64()
                    .ok_or_else(|| "runtime file size is invalid".to_owned())?;
                let expected_sha = entry["sha256"]
                    .as_str()
                    .ok_or_else(|| "runtime file digest is invalid".to_owned())?;
                if metadata.len() != expected_size || sha256_file(&candidate)? != expected_sha {
                    return Err(format!("runtime file bytes changed: {path_value}"));
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let expected_executable = entry["executable"]
                        .as_bool()
                        .ok_or_else(|| "runtime executable flag is invalid".to_owned())?;
                    if (metadata.permissions().mode() & 0o111 != 0) != expected_executable {
                        return Err(format!("runtime executable mode changed: {path_value}"));
                    }
                }
                file_count += 1;
                size_bytes = size_bytes
                    .checked_add(expected_size)
                    .ok_or_else(|| "runtime byte count overflowed".to_owned())?;
            }
            _ => return Err("runtime manifest has an unsupported entry type".to_owned()),
        }
    }
    let mut actual_paths = BTreeSet::new();
    collect_runtime_paths(&runtime_root, Path::new(""), &mut actual_paths)?;
    if actual_paths != manifest_paths
        || runtime["entryCount"].as_u64() != u64::try_from(entries.len()).ok()
        || runtime["fileCount"].as_u64() != Some(file_count)
        || runtime["sizeBytes"].as_u64() != Some(size_bytes)
    {
        return Err("bundled runtime tree cardinality changed".to_owned());
    }
    let canonical_entries = serde_json::to_vec(entries)
        .map_err(|error| format!("cannot canonicalize runtime manifest: {error}"))?;
    let declared_tree = runtime["treeSha256"]
        .as_str()
        .ok_or_else(|| "runtime tree digest is invalid".to_owned())?;
    let mut digest = Sha256::new();
    digest.update(canonical_entries);
    if format!("{:x}", digest.finalize()) != declared_tree {
        return Err("runtime tree manifest digest is inconsistent".to_owned());
    }
    let entry_point = runtime_root.join(safe_runtime_relative_path(target_suffixed_name())?);
    Ok(EngineExecutable { path: entry_point })
}

#[cfg(any(debug_assertions, test))]
fn development_candidate_paths(
    allow: bool,
    environment_override: Option<PathBuf>,
    current_executable: Option<PathBuf>,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if !allow {
        return candidates;
    }
    if let Some(value) = environment_override {
        candidates.push(value);
    }
    if let Some(current) = current_executable {
        if let Some(parent) = current.parent() {
            candidates.push(parent.join(executable_name()));
            candidates.push(parent.join(target_suffixed_name()));
            candidates.push(parent.join("binaries").join(target_suffixed_name()));
        }
    }
    candidates
}

pub(crate) fn discover_engine<R: Runtime>(app: &AppHandle<R>) -> Result<EngineExecutable, String> {
    let resources = app
        .path()
        .resource_dir()
        .map_err(|error| format!("cannot resolve signed application resources: {error}"))?;
    let bundled_root = resources.join("resources").join("openastroflow-worker");
    if bundled_root.is_dir() {
        return verify_bundled_runtime(&bundled_root);
    }

    #[cfg(not(debug_assertions))]
    return Err("signed Ultra-Fast WBPP runtime resource is missing".to_owned());

    #[cfg(debug_assertions)]
    let mut candidates = development_candidate_paths(
        true,
        std::env::var_os("OPENASTROFLOW_ENGINE_EXECUTABLE").map(PathBuf::from),
        std::env::current_exe().ok(),
    );
    #[cfg(debug_assertions)]
    candidates.push(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(".venv")
            .join(if cfg!(windows) { "Scripts" } else { "bin" })
            .join(if cfg!(windows) {
                "openastroflow-engine.exe"
            } else {
                "openastroflow-engine"
            }),
    );

    #[cfg(debug_assertions)]
    let mut seen = HashSet::new();
    #[cfg(debug_assertions)]
    for candidate in candidates {
        let Ok(path) = candidate.canonicalize() else {
            continue;
        };
        if !seen.insert(path.clone()) || !path.is_file() {
            continue;
        }
        return Ok(EngineExecutable { path });
    }
    #[cfg(debug_assertions)]
    Err("Ultra-Fast WBPP scientific sidecar was not found. Install a signed app bundle or set OPENASTROFLOW_ENGINE_EXECUTABLE to the development engine executable.".to_owned())
}

fn controller_handshake(session_id: &str) -> Result<WorkerEnvelope, String> {
    Ok(WorkerEnvelope {
        protocol_version: WORKER_PROTOCOL_VERSION,
        session_id: session_id.to_owned(),
        sequence: 0,
        sent_at_unix_ms: now_ms()?,
        message: WorkerMessage::Handshake(HandshakeMessage {
            role: PeerRole::Controller,
            implementation: "openastroflow-desktop".to_owned(),
            implementation_version: env!("CARGO_PKG_VERSION").to_owned(),
            supported_protocol_versions: vec![WORKER_PROTOCOL_VERSION],
            capabilities: None,
        }),
    })
}

fn read_protocol_line(
    reader: &mut BufReader<std::process::ChildStdout>,
) -> Result<WorkerEnvelope, String> {
    let mut line = Vec::new();
    let count = (&mut *reader)
        .take((MAX_NDJSON_LINE_BYTES + 1) as u64)
        .read_until(b'\n', &mut line)
        .map_err(|error| format!("cannot read worker protocol stream: {error}"))?;
    if count == 0 {
        return Err("worker exited before sending its handshake".to_owned());
    }
    if count > MAX_NDJSON_LINE_BYTES {
        return Err(format!(
            "worker protocol record exceeded {MAX_NDJSON_LINE_BYTES} bytes"
        ));
    }
    decode_ndjson_line(&line).map_err(|error| format!("worker protocol rejected: {error}"))
}

fn read_handshake_with_timeout(
    stdout: std::process::ChildStdout,
) -> Result<(WorkerEnvelope, BufReader<std::process::ChildStdout>), String> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let response = read_protocol_line(&mut reader);
        let _ = sender.send((response, reader));
    });
    let (response, reader) = receiver
        .recv_timeout(HANDSHAKE_TIMEOUT)
        .map_err(|_| "scientific sidecar handshake timed out after 30 seconds".to_owned())?;
    Ok((response?, reader))
}

fn select_profile(capabilities: &BackendCapabilities) -> Result<HardwareProfile, String> {
    let host = platform::detect();
    select_profile_for_host(capabilities, &host)
}

fn select_profile_for_host(
    capabilities: &BackendCapabilities,
    host: &platform::PlatformProfile,
) -> Result<HardwareProfile, String> {
    if host.platform == "windows"
        && capabilities
            .hardware_profiles
            .contains(&HardwareProfile::WindowsCpu)
    {
        return Ok(HardwareProfile::WindowsCpu);
    }
    if host.platform == "windows" {
        return Err("the Windows sidecar did not advertise windows-cpu".to_owned());
    }
    if capabilities
        .hardware_profiles
        .contains(&HardwareProfile::M3ProTuned)
        && host.platform == "macos"
        && host.architecture == "aarch64"
        && host.chip.to_ascii_lowercase().contains("m3 pro")
    {
        return Ok(HardwareProfile::M3ProTuned);
    }
    if host.platform == "macos"
        && host.architecture == "aarch64"
        && capabilities
            .hardware_profiles
            .contains(&HardwareProfile::GenericAppleMetal)
    {
        return Ok(HardwareProfile::GenericAppleMetal);
    }
    if host.architecture == "aarch64"
        && capabilities
            .hardware_profiles
            .contains(&HardwareProfile::GenericArm64Cpu)
    {
        return Ok(HardwareProfile::GenericArm64Cpu);
    }
    if matches!(host.architecture, "aarch64" | "x86_64")
        && capabilities
            .hardware_profiles
            .contains(&HardwareProfile::PortableCpu)
    {
        return Ok(HardwareProfile::PortableCpu);
    }
    Err("the sidecar did not advertise a hardware profile compatible with this host".to_owned())
}

/// Windows is release-validated on x86-64 only: the CPU kernels, the worker
/// runtime and the retained E2E evidence all target that architecture.  Any
/// other Windows architecture keeps the shell in its interface-only state.
fn platform_scientific_release_validated(host: &platform::PlatformProfile) -> bool {
    host.platform != "windows" || host.architecture == "x86_64"
}

fn platform_unavailable_reason(host: &platform::PlatformProfile) -> String {
    format!(
        "Windows on {} is not supported by this release; the validated Windows build is x86-64 only (ARM64 has no worker runtime or E2E acceptance)",
        host.architecture
    )
}

fn probe_runtime<R: Runtime>(app: &AppHandle<R>) -> Result<RuntimeProbe, String> {
    probe_runtime_with(discover_engine(app)?)
}

fn probe_runtime_with(executable: EngineExecutable) -> Result<RuntimeProbe, String> {
    let session_id = new_identifier("probe")?;
    let handshake = controller_handshake(&session_id)?;
    let mut command = executable.command("worker");
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = spawn_sidecar(&mut command)
        .map_err(|error| format!("cannot launch scientific sidecar: {error}"))?;
    let mut stdin = child.stdin.take().ok_or("worker stdin is unavailable")?;
    let stdout = child.stdout.take().ok_or("worker stdout is unavailable")?;
    stdin
        .write_all(&encode_ndjson_line(&handshake).map_err(|error| error.to_string())?)
        .map_err(|error| format!("cannot write controller handshake: {error}"))?;
    stdin.flush().map_err(|error| error.to_string())?;
    drop(stdin);
    let response = read_handshake_with_timeout(stdout).map(|(envelope, _)| envelope);
    let _ = platform::terminate_process_tree(&mut child);
    let response = response?;
    let mut cursor = ProtocolCursor::default();
    cursor
        .accept(&response)
        .map_err(|error| format!("worker handshake ordering rejected: {error}"))?;
    if response.session_id != session_id {
        return Err("worker handshake used the wrong session".to_owned());
    }
    let WorkerMessage::Handshake(message) = response.message else {
        return Err("worker did not answer with a handshake".to_owned());
    };
    if message.role != PeerRole::Worker {
        return Err("sidecar handshake did not identify a worker".to_owned());
    }
    let capabilities = message
        .capabilities
        .ok_or("worker handshake omitted backend capabilities")?;
    let selected_profile = select_profile(&capabilities)?;
    Ok(RuntimeProbe {
        executable,
        capabilities,
        selected_profile,
        implementation_version: message.implementation_version,
    })
}

pub(crate) fn get_capabilities<R: Runtime>(app: &AppHandle<R>) -> RuntimeCapabilities {
    let platform_profile = platform::detect();
    match probe_runtime(app) {
        Ok(probe) => {
            let required = [
                StageKind::QualityControl,
                StageKind::Calibration,
                StageKind::Registration,
                StageKind::Integration,
                StageKind::AstrometricSolve,
            ];
            let stages_ready = required
                .iter()
                .all(|stage| probe.capabilities.stages.contains(stage));
            let solver_available = probe
                .capabilities
                .features
                .contains(&BackendFeature::OfflineAstrometricSolver)
                && probe
                    .capabilities
                    .stages
                    .contains(&StageKind::AstrometricSolve);
            // Only architectures with retained scientific E2E evidence may
            // present themselves as a product-ready runtime.
            let release_validated = platform_scientific_release_validated(&platform_profile);
            let available = stages_ready && solver_available && release_validated;
            let unavailable_reason = (!available).then(|| {
                if release_validated {
                    "sidecar handshake passed, but one or more required E2E stages are unavailable"
                        .to_owned()
                } else {
                    platform_unavailable_reason(&platform_profile)
                }
            });
            RuntimeCapabilities {
                platform: platform_profile.platform,
                chip: platform_profile.chip,
                cpu_backend: if probe
                    .capabilities
                    .features
                    .contains(&BackendFeature::CpuExecution)
                {
                    "Native CPU execution".to_owned()
                } else {
                    "CPU backend unavailable".to_owned()
                },
                gpu_backend: if probe
                    .capabilities
                    .features
                    .contains(&BackendFeature::MetalExecution)
                {
                    "Metal execution".to_owned()
                } else if platform_profile.platform == "windows" {
                    platform_profile.gpu_backend.to_owned()
                } else {
                    "Portable CPU only".to_owned()
                },
                optimization_tier: match probe.selected_profile {
                    HardwareProfile::PortableCpu => "PORTABLE",
                    HardwareProfile::M3ProTuned => "M3_PRO_TUNED",
                    HardwareProfile::GenericAppleMetal => "APPLE_SILICON",
                    HardwareProfile::GenericArm64Cpu if platform_profile.platform == "macos" => {
                        "APPLE_SILICON"
                    }
                    HardwareProfile::GenericArm64Cpu => "PORTABLE",
                    HardwareProfile::WindowsCpu if release_validated => "WINDOWS_X64",
                    HardwareProfile::WindowsCpu => "PORTABLE",
                },
                available,
                drizzle_available: probe.capabilities.stages.contains(&StageKind::Drizzle)
                    && probe
                        .capabilities
                        .features
                        .contains(&BackendFeature::Drizzle),
                solver_available,
                runtime_version: Some(probe.implementation_version),
                unavailable_reason,
            }
        }
        Err(error) => RuntimeCapabilities {
            platform: platform_profile.platform,
            chip: platform_profile.chip,
            cpu_backend: platform_profile.cpu_backend.to_owned(),
            gpu_backend: platform_profile.gpu_backend.to_owned(),
            optimization_tier: platform_profile.optimization_tier,
            available: false,
            drizzle_available: false,
            solver_available: false,
            runtime_version: None,
            unavailable_reason: Some(error),
        },
    }
}

pub(crate) fn command_output(mut command: Command, operation: &str) -> Result<Vec<u8>, String> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let output = sidecar_output(&mut command)
        .map_err(|error| format!("cannot launch sidecar for {operation}: {error}"))?;
    if !output.status.success() {
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "sidecar {operation} failed ({}): {}",
            output.status,
            diagnostic
                .trim()
                .chars()
                .take(MAX_DIAGNOSTIC_BYTES)
                .collect::<String>()
        ));
    }
    Ok(output.stdout)
}

pub(crate) fn inspect_paths<R: Runtime>(
    app: &AppHandle<R>,
    request: InspectRequest,
) -> Result<InspectResponse, String> {
    if request.paths.is_empty() {
        return Err("at least one file or directory is required".to_owned());
    }
    inspect_paths_with(discover_engine(app)?, request)
}

fn inspect_paths_with(
    executable: EngineExecutable,
    request: InspectRequest,
) -> Result<InspectResponse, String> {
    let mut command = executable.command("inventory");
    command.args(&request.paths).arg("--compact");
    let stdout = command_output(command, "inventory")?;
    let inventory: InventoryPayload = serde_json::from_slice(&stdout)
        .map_err(|error| format!("sidecar inventory returned invalid JSON: {error}"))?;
    if inventory.assets.is_empty() {
        return Err("no supported astronomy frames were found".to_owned());
    }
    let blocking: Vec<_> = inventory
        .issues
        .iter()
        // A file-picker/drop batch may contain only calibration frames. The
        // complete project is validated separately before execution.
        .filter(|issue| issue.severity == "ERROR" && issue.code != "NO_LIGHTS")
        .collect();
    if !blocking.is_empty() {
        let summary = blocking
            .iter()
            .take(4)
            .map(|issue| format!("{}: {}", issue.code, issue.message))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!(
            "inventory contains blocking frame errors: {summary}"
        ));
    }
    let hint = request.role_hint.as_deref().map(str::to_ascii_uppercase);
    let mut groups: BTreeMap<String, (Vec<String>, f32, bool)> = BTreeMap::new();
    let mut inspected_assets = Vec::new();
    for asset in inventory.assets {
        if asset.status != "READY" {
            return Err(format!(
                "frame is not ready: {} ({})",
                asset.path,
                asset
                    .error_message
                    .or(asset.error_code)
                    .unwrap_or(asset.status)
            ));
        }
        if !matches!(
            asset.role.as_str(),
            "LIGHT" | "FLAT" | "DARK" | "BIAS" | "MASTER_FLAT" | "MASTER_DARK" | "MASTER_BIAS"
        ) {
            return Err(format!(
                "desktop import does not accept role {}: {}",
                asset.role, asset.path
            ));
        }
        let confidence = if asset.role_evidence.iter().any(|item| {
            let lower = item.to_ascii_lowercase();
            lower.contains("header") || lower.contains("imagetyp") || lower.contains("frametyp")
        }) {
            1.0
        } else {
            0.9
        };
        let needs_confirmation = hint.as_deref().is_some_and(|value| value != asset.role);
        let entry = groups
            .entry(asset.role.clone())
            .or_insert_with(|| (Vec::new(), confidence, needs_confirmation));
        entry.0.push(asset.path.clone());
        entry.1 = entry.1.min(confidence);
        entry.2 |= needs_confirmation;
        let source_sha256 = if asset.role.starts_with("MASTER_") {
            Some(format!("sha256:{}", sha256_file(Path::new(&asset.path))?))
        } else {
            None
        };
        inspected_assets.push(InspectedAsset {
            path: asset.path,
            role: asset.role,
            width: asset.width,
            height: asset.height,
            channels: asset.channels,
            filter: asset.filter_name,
            target: asset.target,
            camera: asset.camera,
            exposure_seconds: asset.exposure_seconds,
            observed_at: asset.observed_at,
            temperature_celsius: asset.temperature_celsius,
            gain: asset.gain,
            offset: asset.offset,
            binning: [asset.binning_x, asset.binning_y],
            cfa_pattern: asset.cfa_pattern,
            readout_mode: asset.readout_mode,
            source_sha256,
        });
    }
    let sources = groups
        .into_iter()
        .map(|(role, (mut paths, confidence, needs_confirmation))| {
            paths.sort();
            paths.dedup();
            InspectedSource {
                role,
                file_count: paths.len(),
                paths,
                confidence,
                needs_confirmation,
            }
        })
        .collect::<Vec<_>>();
    Ok(InspectResponse {
        total_files: sources.iter().map(|source| source.file_count).sum(),
        sources,
        assets: inspected_assets,
        project_name: inventory.name,
    })
}

pub(crate) fn hash_sources(request: HashSourcesRequest) -> Result<HashSourcesResponse, String> {
    if request.paths.is_empty() || request.paths.len() > 10_000 {
        return Err("hashSources requires between 1 and 10000 files".to_owned());
    }
    let mut seen = HashSet::new();
    let mut entries = Vec::with_capacity(request.paths.len());
    for value in request.paths {
        let path = Path::new(&value)
            .canonicalize()
            .map_err(|error| format!("cannot resolve source for hashing: {error}"))?;
        if !path.is_file() || !seen.insert(path.clone()) {
            return Err("hashSources accepts unique regular files only".to_owned());
        }
        entries.push(SourceHash {
            path: path.to_string_lossy().into_owned(),
            source_sha256: format!("sha256:{}", sha256_file(&path)?),
        });
    }
    Ok(HashSourcesResponse { entries })
}

fn create_private_quality_request(
    paths: &[String],
) -> Result<(PathBuf, BTreeSet<PathBuf>), String> {
    let canonical = canonical_light_paths(paths, "quality inspection")?;
    let request_path = std::env::temp_dir().join(format!(
        "{}.json",
        new_identifier("ultra-fast-wbpp-quality-request")?
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&request_path)
        .map_err(|error| format!("cannot create private quality request: {error}"))?;
    let payload = serde_json::json!({
        "schemaVersion": 1,
        "lightPaths": canonical.iter().map(|path| path.to_string_lossy()).collect::<Vec<_>>(),
    });
    let mut encoded = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    encoded.push(b'\n');
    file.write_all(&encoded)
        .map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    Ok((request_path, canonical))
}

pub(crate) fn inspect_calibration<R: Runtime>(
    app: &AppHandle<R>,
    request: InspectCalibrationRequest,
) -> Result<CalibrationInspection, String> {
    inspect_calibration_with(discover_engine(app)?, request)
}

fn inspect_calibration_with(
    executable: EngineExecutable,
    request: InspectCalibrationRequest,
) -> Result<CalibrationInspection, String> {
    if request.paths.is_empty() || request.paths.len() > 10_000 || !request.recipe.is_object() {
        return Err("calibration inspection requires 1–10000 files and a recipe object".to_owned());
    }
    let mut paths = BTreeSet::new();
    for value in &request.paths {
        let path = Path::new(value)
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if !path.is_file() || !paths.insert(path) {
            return Err("calibration inspection accepts unique regular files only".to_owned());
        }
    }
    let request_path = std::env::temp_dir().join(format!(
        "{}.json",
        new_identifier("ultra-fast-wbpp-calibration-request")?
    ));
    let payload = serde_json::json!({
        "schemaVersion": 1,
        "paths": paths.iter().map(|path| path.to_string_lossy()).collect::<Vec<_>>(),
        "recipe": request.recipe,
    });
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&request_path)
        .map_err(|error| error.to_string())?;
    let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    file.write_all(&bytes).map_err(|error| error.to_string())?;
    drop(file);
    let mut command = executable.command("calibration-check");
    command.arg("--request-json").arg(&request_path);
    let output = command_output(command, "calibration inspection");
    let _ = fs::remove_file(&request_path);
    let inspection: CalibrationInspection = serde_json::from_slice(&output?).map_err(|error| {
        format!("sidecar calibration inspection returned invalid JSON: {error}")
    })?;
    validate_calibration_inspection(&inspection)?;
    Ok(inspection)
}

fn validate_calibration_inspection(inspection: &CalibrationInspection) -> Result<(), String> {
    let statuses_valid = matches!(inspection.status.as_str(), "READY" | "BLOCKED")
        && inspection.groups.iter().all(|group| {
            matches!(group.status.as_str(), "READY" | "BLOCKED")
                && group.light_count > 0
                && ["FLAT", "DARK", "BIAS"]
                    .iter()
                    .all(|role| group.matches.contains_key(*role))
        })
        && inspection
            .issues
            .iter()
            .all(|issue| matches!(issue.severity.as_str(), "ERROR" | "WARNING" | "INFO"));
    let ready = !inspection.groups.is_empty()
        && inspection
            .groups
            .iter()
            .all(|group| group.status == "READY")
        && !inspection
            .issues
            .iter()
            .any(|issue| issue.severity == "ERROR");
    if inspection.schema_version != 1
        || !statuses_valid
        || inspection.calibration_ready != ready
        || (inspection.status == "READY") != ready
    {
        return Err(
            "sidecar calibration inspection returned an inconsistent readiness contract".to_owned(),
        );
    }
    Ok(())
}

pub(crate) fn inspect_quality<R: Runtime>(
    app: &AppHandle<R>,
    request: InspectQualityRequest,
) -> Result<QualityInspection, String> {
    inspect_quality_with(discover_engine(app)?, request)
}

fn inspect_quality_with(
    executable: EngineExecutable,
    request: InspectQualityRequest,
) -> Result<QualityInspection, String> {
    let (request_path, expected_paths) = create_private_quality_request(&request.paths)?;
    let mut command = executable.command("quality-check");
    command
        .arg("--request-json")
        .arg(&request_path)
        .arg("--compact");
    let output = command_output(command, "quality inspection");
    let _ = fs::remove_file(&request_path);
    let stdout = output?;
    let inspection: QualityInspection = serde_json::from_slice(&stdout)
        .map_err(|error| format!("sidecar quality inspection returned invalid JSON: {error}"))?;
    if inspection.schema_version != 1
        || !checked_source_digest(&inspection.gate_policy_digest)
        || inspection.workers == 0
        || inspection.frames.len() != expected_paths.len()
    {
        return Err("sidecar quality inspection returned an invalid contract".to_owned());
    }
    let mut observed_paths = BTreeSet::new();
    let mut observed_counts = BTreeMap::from([
        ("PASS".to_owned(), 0_usize),
        ("REVIEW".to_owned(), 0_usize),
        ("HARD_FAIL".to_owned(), 0_usize),
    ]);
    let mut preview_transport_bytes = 0_usize;
    for frame in &inspection.frames {
        let path = Path::new(&frame.path)
            .canonicalize()
            .map_err(|error| format!("cannot revalidate quality result path: {error}"))?;
        if !expected_paths.contains(&path)
            || !observed_paths.insert(path)
            || !matches!(frame.disposition.as_str(), "PASS" | "REVIEW" | "HARD_FAIL")
            || frame
                .source_sha256
                .as_deref()
                .is_some_and(|value| !checked_source_digest(value))
            || frame
                .preview_sha256
                .as_deref()
                .is_some_and(|value| !checked_source_digest(value))
        {
            return Err("sidecar quality inspection rebound an input or gate".to_owned());
        }
        if let Some(preview) = &frame.preview_data_url {
            preview_transport_bytes = preview_transport_bytes.saturating_add(preview.len());
            if !preview.starts_with("data:image/png;base64,")
                || preview.len() > 700_000
                || frame.preview_sha256.is_none()
                || !preview.is_ascii()
                || preview_transport_bytes > 32 * 1024 * 1024
            {
                return Err("sidecar quality preview exceeded its in-memory boundary".to_owned());
            }
        } else if frame.preview_sha256.is_some() {
            return Err("sidecar quality preview digest has no preview bytes".to_owned());
        }
        *observed_counts
            .get_mut(&frame.disposition)
            .expect("validated disposition") += 1;
    }
    if inspection.counts != observed_counts {
        return Err("sidecar quality inspection counts do not match its frames".to_owned());
    }
    Ok(inspection)
}

const BLINK_MANIFEST_KIND: &str = "blink-manifest-v1";
/// The filmstrip preview is the 1/8-scale grayscale JPEG (60–120 KB,
/// noise-limited): 100 frames are about 12 MB as data URLs.  A preview over
/// either bound is left to the on-demand `load_blink_preview` path.
const MAX_BLINK_FILMSTRIP_BYTES: u64 = 200 * 1024;
const MAX_BLINK_TRANSPORT_BYTES: usize = 32 * 1024 * 1024;
/// Sessions kept under the blink-sessions root, the new one included; the
/// previews of a 100-frame session take about 150 MB.
const MAX_BLINK_SESSIONS: usize = 3;
const MAX_BLINK_MASTER_FLATS: usize = 16;
const MAX_BLINK_WORKERS: usize = 64;
/// A 10 000-frame manifest with every metric is about 20 MB.
const MAX_BLINK_MANIFEST_BYTES: usize = 64 * 1024 * 1024;

/// `<user cache>/Ultra-Fast-WBPP/blink-sessions`: `~/Library/Caches` on
/// macOS, `%LOCALAPPDATA%` on Windows, the XDG cache on Linux — the same
/// product cache folder the engine keeps its QC analysis cache in, and
/// never inside a source folder.
pub(crate) fn blink_sessions_root<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let cache = app
        .path()
        .cache_dir()
        .map_err(|error| format!("cannot resolve the user cache directory: {error}"))?;
    Ok(cache.join("Ultra-Fast-WBPP").join("blink-sessions"))
}

/// Canonical, unique regular files for a request that names Lights.
fn canonical_light_paths(paths: &[String], operation: &str) -> Result<BTreeSet<PathBuf>, String> {
    if paths.is_empty() || paths.len() > 10_000 {
        return Err(format!(
            "{operation} requires between 1 and 10000 Light files"
        ));
    }
    let mut canonical = BTreeSet::new();
    for value in paths {
        let path = Path::new(value)
            .canonicalize()
            .map_err(|error| format!("cannot resolve Light for {operation}: {error}"))?;
        if !path.is_file() || !canonical.insert(path) {
            return Err(format!(
                "{operation} accepts unique regular Light files only"
            ));
        }
    }
    Ok(canonical)
}

/// Writes the private `blink-measure` request (mode 0600, create-only) and
/// returns its path.  `workers` and the master flats travel only when the
/// webview supplied them; the engine's hardware default applies otherwise.
fn create_private_blink_request(
    lights: &BTreeSet<PathBuf>,
    session_directory: &Path,
    master_flats: &[(String, PathBuf)],
    workers: Option<usize>,
) -> Result<PathBuf, String> {
    let request_path = std::env::temp_dir().join(format!(
        "{}.json",
        new_identifier("ultra-fast-wbpp-blink-request")?
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&request_path)
        .map_err(|error| format!("cannot create private blink request: {error}"))?;
    let mut payload = serde_json::json!({
        "schemaVersion": 1,
        "lightPaths": lights.iter().map(|path| path.to_string_lossy()).collect::<Vec<_>>(),
        "sessionDirectory": session_directory.to_string_lossy(),
        "previews": {
            "filmstripScale": 8, "zoomScale": 4, "filmstripFormat": "jpeg", "jpegQuality": 85,
        },
    });
    if let Some(workers) = workers {
        payload["workers"] = serde_json::json!(workers);
    }
    if !master_flats.is_empty() {
        payload["masterFlats"] = master_flats
            .iter()
            .map(|(filter, path)| {
                serde_json::json!({"filter": filter, "path": path.to_string_lossy()})
            })
            .collect();
    }
    let mut encoded = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    encoded.push(b'\n');
    file.write_all(&encoded)
        .map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    Ok(request_path)
}

/// Removes the oldest of the desktop's own blink sessions under `root` so
/// that at most `keep` remain.  Only directories the desktop named itself
/// are candidates; a failed removal is not an error (the next launch tries
/// again).
fn prune_blink_sessions(root: &Path, keep: usize) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let mut sessions = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_owned();
            let parts = crate::project::blink_session_name_parts(&name)?;
            let file_type = entry.file_type().ok()?;
            (file_type.is_dir() && !file_type.is_symlink()).then_some((parts, entry.path()))
        })
        .collect::<Vec<_>>();
    sessions.sort_by_key(|session| session.0);
    let excess = sessions.len().saturating_sub(keep);
    for (_, path) in sessions.into_iter().take(excess) {
        let _ = fs::remove_dir_all(path);
    }
}

/// A fresh session directory name under `root`: the path-list digest, the
/// local time and, when that name exists already, a counter.  The directory
/// itself is created by the engine (create-only).
fn new_blink_session_directory(root: &Path, lights: &BTreeSet<PathBuf>) -> Result<PathBuf, String> {
    let mut digest = Sha256::new();
    for path in lights {
        digest.update(path.to_string_lossy().as_bytes());
        digest.update(b"\n");
    }
    let stem = format!(
        "{:.16}-{}",
        format!("{:x}", digest.finalize()),
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    );
    // Counters never go backwards within a second: a name freed by the
    // pruning just before is not reused, so a caller holding the old path
    // cannot mistake the new session for it.
    let next_attempt = fs::read_dir(root)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| {
                    let name = entry.file_name().to_str()?.to_owned();
                    let counter = if name == stem {
                        1
                    } else {
                        name.strip_prefix(&format!("{stem}-"))?
                            .parse::<u32>()
                            .ok()?
                    };
                    Some(counter + 1)
                })
                .max()
                .unwrap_or(1)
        })
        .unwrap_or(1);
    for attempt in next_attempt..=99_u32 {
        let name = if attempt == 1 {
            stem.clone()
        } else {
            format!("{stem}-{attempt}")
        };
        let candidate = root.join(&name);
        if fs::symlink_metadata(&candidate).is_err() {
            return Ok(candidate);
        }
    }
    Err("cannot name a new blink session directory".to_owned())
}

pub(crate) fn blink_measure<R: Runtime>(
    app: &AppHandle<R>,
    request: BlinkMeasureRequest,
) -> Result<BlinkManifest, String> {
    let root = blink_sessions_root(app)?;
    blink_measure_with(discover_engine(app)?, request, &root)
}

/// Runs `blink-measure` on the requested Lights into a new session under
/// `sessions_root`, validates the manifest against the request and the
/// session directory, and attaches the filmstrip previews within the
/// transport budget.  A failed launch or an invalid manifest removes the
/// session it created.
fn blink_measure_with(
    executable: EngineExecutable,
    request: BlinkMeasureRequest,
    sessions_root: &Path,
) -> Result<BlinkManifest, String> {
    let lights = canonical_light_paths(&request.paths, "blink measurement")?;
    if request.master_flats.len() > MAX_BLINK_MASTER_FLATS {
        return Err("blink measurement accepts at most 16 master flats".to_owned());
    }
    let mut flats_by_filter: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for item in &request.master_flats {
        let filter = item.filter.trim();
        let path = Path::new(&item.path).canonicalize().map_err(|error| {
            format!("cannot resolve master flat for blink measurement: {error}")
        })?;
        if filter.is_empty() || filter.len() > 64 || !path.is_file() {
            return Err(
                "blink measurement master flats must be regular files with a filter".to_owned(),
            );
        }
        // The engine compares filters case-insensitively.
        flats_by_filter
            .entry(filter.to_ascii_uppercase())
            .or_default()
            .push(path);
    }
    // The flats only feed the optional gradient flag.  The webview sends every
    // imported master flat; a filter with more than one is ambiguous and is
    // left out rather than guessed, and never fails the measurement.
    let master_flats = flats_by_filter
        .into_iter()
        .filter_map(|(filter, mut paths)| (paths.len() == 1).then(|| (filter, paths.remove(0))))
        .collect::<Vec<_>>();
    if request
        .workers
        .is_some_and(|workers| workers == 0 || workers > MAX_BLINK_WORKERS)
    {
        return Err("blink measurement workers must be between 1 and 64".to_owned());
    }
    fs::create_dir_all(sessions_root)
        .map_err(|error| format!("cannot create the blink sessions directory: {error}"))?;
    let sessions_root = sessions_root
        .canonicalize()
        .map_err(|error| format!("cannot resolve the blink sessions directory: {error}"))?;
    prune_blink_sessions(&sessions_root, MAX_BLINK_SESSIONS - 1);
    let session_directory = new_blink_session_directory(&sessions_root, &lights)?;
    let request_path =
        create_private_blink_request(&lights, &session_directory, &master_flats, request.workers)?;
    let mut command = executable.command("blink-measure");
    command
        .arg("--request-json")
        .arg(&request_path)
        .arg("--compact");
    let output = command_output(command, "blink measurement");
    let _ = fs::remove_file(&request_path);
    let result = output.and_then(|stdout| {
        let mut manifest: BlinkManifest = serde_json::from_slice(&stdout)
            .map_err(|error| format!("sidecar blink measurement returned invalid JSON: {error}"))?;
        let session = session_directory.canonicalize().map_err(|error| {
            format!("sidecar blink measurement left no session directory: {error}")
        })?;
        validate_blink_manifest(&manifest, &lights, &session).map_err(|detail| {
            format!("sidecar blink measurement returned an invalid contract: {detail}")
        })?;
        manifest.manifest_sha256 = Some(blink_manifest_file_digest(&session, &manifest)?);
        attach_blink_previews(&mut manifest, &session, MAX_BLINK_TRANSPORT_BYTES);
        Ok(manifest)
    });
    if result.is_err() {
        // The directory was named for this launch and is unusable without a
        // valid manifest; the sessions root only ever holds our own output.
        let _ = fs::remove_dir_all(&session_directory);
    }
    result
}

/// The `blink-manifest-v1` contract as the desktop relies on it: identity
/// of the session and of every requested Light, well-formed digests and
/// enums, counts and channel references consistent with the frames, and
/// preview paths that stay inside the session directory.
fn validate_blink_manifest(
    manifest: &BlinkManifest,
    expected_paths: &BTreeSet<PathBuf>,
    session_directory: &Path,
) -> Result<(), String> {
    if manifest.schema_version != 1 || manifest.kind != BLINK_MANIFEST_KIND {
        return Err("unsupported manifest kind or schema version".to_owned());
    }
    if manifest.session_id.trim().is_empty() || manifest.session_id.len() > 128 {
        return Err("session id is missing".to_owned());
    }
    let reported = Path::new(&manifest.session_directory)
        .canonicalize()
        .map_err(|error| format!("session directory cannot be resolved: {error}"))?;
    if reported != session_directory {
        return Err("session directory differs from the requested one".to_owned());
    }
    if !checked_source_digest(&manifest.inventory_sha256)
        || !checked_source_digest(&manifest.gate_policy_digest)
        || !checked_source_digest(&manifest.flags_policy_digest)
    {
        return Err("inventory or policy digest is malformed".to_owned());
    }
    let frame_count = manifest.frames.len();
    if frame_count != expected_paths.len() || manifest.counts.frames != frame_count {
        return Err("frame count differs from the requested Lights".to_owned());
    }
    let channel_ids = manifest
        .channels
        .iter()
        .map(|channel| channel.channel_id.as_str())
        .collect::<BTreeSet<_>>();
    if channel_ids.len() != manifest.channels.len() || manifest.channels.is_empty() {
        return Err("channels are missing or not unique".to_owned());
    }
    let mut observed_paths = BTreeSet::new();
    let mut observed_indices = BTreeSet::new();
    let mut counts = BlinkCounts {
        frames: frame_count,
        exclude: 0,
        attention: 0,
        clean: 0,
    };
    let mut frames_per_channel: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for frame in &manifest.frames {
        let path = Path::new(&frame.path)
            .canonicalize()
            .map_err(|error| format!("frame path cannot be resolved: {error}"))?;
        if !expected_paths.contains(&path) || !observed_paths.insert(path) {
            return Err("frames do not name every requested Light exactly once".to_owned());
        }
        if frame.index >= frame_count || !observed_indices.insert(frame.index) {
            return Err("frame indices are not unique within the manifest".to_owned());
        }
        if !checked_source_digest(&frame.source_sha256) {
            return Err("frame content digest is malformed".to_owned());
        }
        if !channel_ids.contains(frame.channel_id.as_str()) {
            return Err("frame names an unknown channel".to_owned());
        }
        let mut excluded = false;
        for flag in &frame.flags {
            if !crate::project::checked_flag_code(&flag.code) {
                return Err("flag code is malformed".to_owned());
            }
            match flag.severity.as_str() {
                "EXCLUDE" => excluded = true,
                "ATTENTION" => {}
                _ => return Err("flag severity must be EXCLUDE or ATTENTION".to_owned()),
            }
        }
        match (frame.default_decision.as_str(), excluded) {
            ("DROP", true) => counts.exclude += 1,
            ("KEEP", false) if frame.flags.is_empty() => counts.clean += 1,
            ("KEEP", false) => counts.attention += 1,
            ("KEEP" | "DROP", _) => {
                return Err("default decision does not follow the EXCLUDE flags".to_owned())
            }
            _ => return Err("default decision must be KEEP or DROP".to_owned()),
        }
        for relative in [&frame.previews.filmstrip, &frame.previews.zoom]
            .into_iter()
            .flatten()
        {
            crate::project::resolve_blink_preview(session_directory, relative)?;
        }
        let entry = frames_per_channel
            .entry(frame.channel_id.as_str())
            .or_default();
        entry.0 += 1;
        entry.1 += usize::from(frame.reference);
    }
    if manifest.counts != counts {
        return Err("counts do not match the frames".to_owned());
    }
    for channel in &manifest.channels {
        let (frames, references) = frames_per_channel
            .get(channel.channel_id.as_str())
            .copied()
            .unwrap_or_default();
        if frames != channel.frame_count || references != 1 || channel.reference.rule.is_empty() {
            return Err("channel frame count or reference is inconsistent".to_owned());
        }
        let reference = manifest
            .frames
            .iter()
            .find(|frame| frame.index == channel.reference.index)
            .ok_or("channel reference names no frame")?;
        if !reference.reference
            || reference.channel_id != channel.channel_id
            || reference.source_sha256 != channel.reference.source_sha256
        {
            return Err("channel reference does not match its frame".to_owned());
        }
    }
    Ok(())
}

/// Digest of the `manifest.json` the engine wrote into the session, after
/// checking that it describes this session: a run's selection names it in
/// `origin`, and `qc/blink.json` is compared with it.
fn blink_manifest_file_digest(
    session_directory: &Path,
    manifest: &BlinkManifest,
) -> Result<String, String> {
    let path = session_directory.join("manifest.json");
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|error| format!("blink session has no manifest.json: {error}"))?
        .take((MAX_BLINK_MANIFEST_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > MAX_BLINK_MANIFEST_BYTES {
        return Err("blink session manifest.json is too large".to_owned());
    }
    let written: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("blink session manifest.json is not JSON: {error}"))?;
    if written.get("kind").and_then(serde_json::Value::as_str) != Some(BLINK_MANIFEST_KIND)
        || written.get("sessionId").and_then(serde_json::Value::as_str)
            != Some(manifest.session_id.as_str())
        || written
            .get("inventorySha256")
            .and_then(serde_json::Value::as_str)
            != Some(manifest.inventory_sha256.as_str())
    {
        return Err("blink session manifest.json describes another session".to_owned());
    }
    Ok(format!("sha256:{}", sha256_file(&path)?))
}

/// Loads the filmstrip previews as data URLs, in manifest order, until the
/// per-preview bound or the transport `budget` leaves a frame out.
fn attach_blink_previews(manifest: &mut BlinkManifest, session_directory: &Path, budget: usize) {
    let mut budget = budget;
    for frame in &mut manifest.frames {
        let Some(relative) = frame.previews.filmstrip.as_deref() else {
            continue;
        };
        let Ok((resolved, format)) =
            crate::project::resolve_blink_preview(session_directory, relative)
        else {
            continue;
        };
        frame.previews.filmstrip_data_url = crate::project::image_data_url(
            &resolved,
            format,
            MAX_BLINK_FILMSTRIP_BYTES,
            &mut budget,
        );
    }
}

fn recipe_cli_id(recipe_id: &str) -> Result<&'static str, String> {
    match recipe_id {
        "balanced" => Ok("balanced"),
        "drizzle-2x" => Ok("drizzle-2x"),
        other => Err(format!("unsupported recipe: {other}")),
    }
}

fn build_plan_envelope(
    probe: &RuntimeProbe,
    paths: &[String],
    recipe_id: &str,
    session_id: &str,
    request_id: &str,
    plan_id: &str,
) -> Result<WorkerEnvelope, String> {
    let mut command = probe.executable.command("controller-plan");
    command
        .args(paths)
        .args([
            "--mode",
            if recipe_cli_id(recipe_id)? == "drizzle-2x" {
                "drizzle"
            } else {
                "ordinary"
            },
        ])
        .args(["--session-id", session_id])
        .args(["--sequence", "1"])
        .args(["--request-id", request_id])
        .args(["--plan-id", plan_id])
        .args(["--hardware-profile", probe.selected_profile.wire_name()])
        .arg("--compact");
    let stdout = command_output(command, "controller-plan")?;
    let lines = stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .collect::<Vec<_>>();
    if lines.len() != 1 {
        return Err(format!(
            "controller-plan emitted {} records; expected exactly one",
            lines.len()
        ));
    }
    let envelope = decode_ndjson_line(lines[0])
        .map_err(|error| format!("controller-plan envelope rejected: {error}"))?;
    if envelope.session_id != session_id || envelope.sequence != 1 {
        return Err("controller-plan returned the wrong session or sequence".to_owned());
    }
    let WorkerMessage::Plan(plan) = &envelope.message else {
        return Err("controller-plan did not return a plan message".to_owned());
    };
    if plan.request_id != request_id || plan.plan_id != plan_id {
        return Err("controller-plan returned mismatched request identities".to_owned());
    }
    Ok(envelope)
}

fn unique_input_paths(sources: &[RunSource]) -> Result<Vec<String>, String> {
    if sources.is_empty() {
        return Err("no input sources were provided".to_owned());
    }
    let mut roles = HashSet::new();
    let mut paths = Vec::new();
    let mut seen = HashSet::new();
    for source in sources {
        if !matches!(source.role.as_str(), "LIGHT" | "FLAT" | "DARK" | "BIAS") {
            return Err(format!("unsupported source role: {}", source.role));
        }
        if !source.paths.is_empty() {
            roles.insert(source.role.as_str());
        }
        for path in &source.paths {
            if path.trim().is_empty() || path.contains('\0') {
                return Err("input path is blank or contains NUL".to_owned());
            }
            if seen.insert(path.clone()) {
                paths.push(path.clone());
            }
        }
    }
    for required in ["LIGHT", "FLAT", "BIAS"] {
        if !roles.contains(required) {
            return Err(format!("at least one {required} source is required"));
        }
    }
    Ok(paths)
}

pub(crate) fn start_pipeline<R: Runtime>(
    app: AppHandle<R>,
    registry: Arc<PipelineRegistry>,
    request: RunRequest,
) -> Result<RunReceipt, String> {
    if registry.shutting_down.load(Ordering::Acquire) {
        return Err("application shutdown is already in progress".to_owned());
    }
    let probe = probe_runtime(&app)?;
    start_pipeline_with_probe(app, registry, request, probe)
}

fn start_pipeline_with_probe<R: Runtime>(
    app: AppHandle<R>,
    registry: Arc<PipelineRegistry>,
    request: RunRequest,
    probe: RuntimeProbe,
) -> Result<RunReceipt, String> {
    if registry.shutting_down.load(Ordering::Acquire) {
        return Err("application shutdown is already in progress".to_owned());
    }
    let input_paths = unique_input_paths(&request.sources)?;
    let output_parent = PathBuf::from(&request.output_parent_directory)
        .canonicalize()
        .map_err(|error| format!("output parent does not exist: {error}"))?;
    if !output_parent.is_dir() {
        return Err("output parent must be an existing directory".to_owned());
    }
    if request.recipe_id == "drizzle-2x"
        && (!probe.capabilities.stages.contains(&StageKind::Drizzle)
            || !probe
                .capabilities
                .features
                .contains(&BackendFeature::Drizzle))
    {
        return Err(
            "the selected sidecar did not advertise validated Drizzle execution".to_owned(),
        );
    }
    let session_id = new_identifier("session")?;
    let request_id = new_identifier("request")?;
    let plan_id = new_identifier("plan")?;
    let run_id = new_identifier("run")?;
    let output_name = SafeFileName::new(format!(
        "openastroflow-{}-{}",
        now_ms()?,
        IDENTIFIER_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
    .map_err(|error| error.to_string())?;
    let output_directory = output_parent.join(output_name.as_str());
    if output_directory.exists() {
        return Err("generated output directory already exists; retry the run".to_owned());
    }
    let plan_envelope = build_plan_envelope(
        &probe,
        &input_paths,
        &request.recipe_id,
        &session_id,
        &request_id,
        &plan_id,
    )?;
    let WorkerMessage::Plan(PlanMessage { recipe, .. }) = &plan_envelope.message else {
        unreachable!("build_plan_envelope only returns plans")
    };
    let recipe = recipe.clone();
    let handshake = controller_handshake(&session_id)?;
    let execute = WorkerEnvelope {
        protocol_version: WORKER_PROTOCOL_VERSION,
        session_id: session_id.clone(),
        sequence: 2,
        sent_at_unix_ms: now_ms()?,
        message: WorkerMessage::Execute(ExecuteMessage {
            request_id: request_id.clone(),
            plan_id,
            run_id: run_id.clone(),
            output_parent_host_path: output_parent.to_string_lossy().into_owned(),
            output_directory_name: output_name,
            resume_from_run_id: None,
        }),
    };
    let mut outgoing = ProtocolCursor::default();
    for envelope in [&handshake, &plan_envelope, &execute] {
        outgoing
            .accept(envelope)
            .map_err(|error| format!("controller protocol rejected: {error}"))?;
    }

    let mut command = probe.executable.command("worker");
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = spawn_sidecar(&mut command)
        .map_err(|error| format!("cannot launch scientific sidecar: {error}"))?;
    let mut stdin = child.stdin.take().ok_or("worker stdin is unavailable")?;
    let stdout = child.stdout.take().ok_or("worker stdout is unavailable")?;
    let stderr = child.stderr.take().ok_or("worker stderr is unavailable")?;
    stdin
        .write_all(&encode_ndjson_line(&handshake).map_err(|error| error.to_string())?)
        .and_then(|()| stdin.flush())
        .map_err(|error| format!("cannot send controller handshake: {error}"))?;
    let (worker_handshake, stdout) = match read_handshake_with_timeout(stdout) {
        Ok(value) => value,
        Err(error) => {
            let _ = platform::terminate_process_tree(&mut child);
            return Err(error);
        }
    };
    let mut incoming = ProtocolCursor::default();
    incoming
        .accept(&worker_handshake)
        .map_err(|error| format!("worker handshake ordering rejected: {error}"))?;
    if worker_handshake.session_id != session_id {
        let _ = platform::terminate_process_tree(&mut child);
        return Err("worker answered with the wrong session".to_owned());
    }
    let WorkerMessage::Handshake(worker_hello) = &worker_handshake.message else {
        let _ = platform::terminate_process_tree(&mut child);
        return Err("worker did not begin with a handshake".to_owned());
    };
    if worker_hello.capabilities.as_ref() != Some(&probe.capabilities) {
        let _ = platform::terminate_process_tree(&mut child);
        return Err("worker capabilities changed between planning and execution".to_owned());
    }
    for envelope in [&plan_envelope, &execute] {
        stdin
            .write_all(&encode_ndjson_line(envelope).map_err(|error| error.to_string())?)
            .map_err(|error| format!("cannot send {}: {error}", envelope.message.kind()))?;
    }
    stdin.flush().map_err(|error| error.to_string())?;
    drop(stdin);

    let child = Arc::new(Mutex::new(child));
    let mut jobs = registry
        .jobs
        .lock()
        .map_err(|_| "pipeline registry lock was poisoned".to_owned())?;
    if registry.shutting_down.load(Ordering::Acquire) {
        drop(jobs);
        if let Ok(mut child) = child.lock() {
            let _ = platform::terminate_process_tree(&mut child);
        }
        return Err("application shutdown started before pipeline registration".to_owned());
    }
    jobs.insert(run_id.clone(), child.clone());
    drop(jobs);
    stream_stderr(app.clone(), run_id.clone(), stderr);
    let thread_job_id = run_id.clone();
    let thread_output = output_directory.clone();
    std::thread::spawn(move || {
        stream_worker(
            app,
            registry,
            child,
            stdout,
            incoming,
            thread_job_id,
            request_id,
            thread_output,
            recipe,
        );
    });
    Ok(RunReceipt {
        job_id: run_id,
        accepted: true,
        execution_mode: "native",
        output_directory: output_directory.to_string_lossy().into_owned(),
    })
}

fn stream_stderr<R: Runtime>(app: AppHandle<R>, job_id: String, stderr: ChildStderr) {
    std::thread::spawn(move || {
        let mut stderr = stderr;
        let mut buffer = [0_u8; 4096];
        while let Ok(count) = stderr.read(&mut buffer) {
            if count == 0 {
                break;
            }
            let message = String::from_utf8_lossy(&buffer[..count]).into_owned();
            if message.trim().is_empty() {
                continue;
            }
            let _ = app.emit(
                LOG_EVENT,
                PipelineLogEvent {
                    job_id: job_id.clone(),
                    stream: "stderr",
                    message,
                },
            );
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn stream_worker<R: Runtime>(
    app: AppHandle<R>,
    registry: Arc<PipelineRegistry>,
    child: Arc<Mutex<ManagedChild>>,
    mut stdout: BufReader<std::process::ChildStdout>,
    mut cursor: ProtocolCursor,
    job_id: String,
    request_id: String,
    output_directory: PathBuf,
    recipe: openastroflow_app_core::Recipe,
) {
    let mut stages: BTreeMap<String, StageReceipt> = BTreeMap::new();
    let mut artifacts: BTreeMap<String, ArtifactReceipt> = BTreeMap::new();
    let mut failed = false;
    loop {
        let mut line = Vec::new();
        match (&mut stdout)
            .take((MAX_NDJSON_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
        {
            Ok(0) => break,
            Ok(count) if count <= MAX_NDJSON_LINE_BYTES => {}
            Ok(_) => {
                emit_error(
                    &app,
                    &job_id,
                    "protocol-record-too-large",
                    "worker protocol record exceeded the maximum NDJSON line size",
                    false,
                );
                failed = true;
                break;
            }
            Err(error) => {
                emit_error(
                    &app,
                    &job_id,
                    "worker-read-failed",
                    &error.to_string(),
                    false,
                );
                failed = true;
                break;
            }
        }
        let envelope = match decode_ndjson_line(&line) {
            Ok(value) => value,
            Err(error) => {
                emit_error(&app, &job_id, "protocol-invalid", &error.to_string(), false);
                failed = true;
                break;
            }
        };
        if let Err(error) = cursor.accept(&envelope) {
            emit_error(
                &app,
                &job_id,
                "protocol-order-invalid",
                &error.to_string(),
                false,
            );
            failed = true;
            break;
        }
        match envelope.message {
            WorkerMessage::Progress(progress)
                if progress.request_id == request_id && progress.run_id == job_id =>
            {
                let _ = app.emit(
                    PROGRESS_EVENT,
                    ProgressEvent {
                        job_id: job_id.clone(),
                        stage_id: progress.stage_id,
                        state: progress.state,
                        fraction: progress.fraction,
                        completed_units: progress.completed_units,
                        total_units: progress.total_units,
                        message: progress.message,
                    },
                );
            }
            WorkerMessage::Artifact(message)
                if message.request_id == request_id && message.run_id == job_id =>
            {
                if stages
                    .get(&message.stage.stage_id)
                    .is_some_and(|existing| existing != &message.stage)
                    || artifacts
                        .get(&message.artifact.artifact_id)
                        .is_some_and(|existing| existing != &message.artifact)
                {
                    emit_error(
                        &app,
                        &job_id,
                        "receipt-identity-collision",
                        "worker rebound an existing stage or artifact identifier",
                        false,
                    );
                    failed = true;
                    break;
                }
                stages
                    .entry(message.stage.stage_id.clone())
                    .or_insert_with(|| message.stage.clone());
                artifacts
                    .entry(message.artifact.artifact_id.clone())
                    .or_insert_with(|| message.artifact.clone());
                let _ = app.emit(
                    ARTIFACT_EVENT,
                    ArtifactEvent {
                        job_id: job_id.clone(),
                        stage: message.stage,
                        artifact: message.artifact,
                    },
                );
            }
            WorkerMessage::Error(error) => {
                emit_error_with_details(
                    &app,
                    &job_id,
                    &error.code,
                    &error.message,
                    error.retryable,
                    error.details,
                );
                failed = true;
            }
            message => {
                emit_error(
                    &app,
                    &job_id,
                    "protocol-identity-mismatch",
                    &format!("unexpected {} message identity", message.kind()),
                    false,
                );
                failed = true;
                break;
            }
        }
    }
    let status = child
        .lock()
        .ok()
        .and_then(|mut process| process.wait().ok());
    if status.is_none_or(|value| !value.success()) && !failed {
        emit_error(
            &app,
            &job_id,
            "worker-exited",
            &format!("scientific sidecar exited with {status:?}"),
            false,
        );
        failed = true;
    }
    registry
        .jobs
        .lock()
        .ok()
        .map(|mut jobs| jobs.remove(&job_id));
    if failed {
        return;
    }
    let stage_values = stages.into_values().collect::<Vec<_>>();
    let artifact_values = artifacts.into_values().collect::<Vec<_>>();
    let gate = RequiredResultGate::evaluate(&recipe, &stage_values, &artifact_values);
    if gate.decision != GateDecision::Ready {
        let detail = gate
            .checks
            .iter()
            .filter(|check| check.required && !check.passed)
            .map(|check| check.code.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        emit_error(
            &app,
            &job_id,
            "result-gate-blocked",
            &format!("final result gate blocked publication: {detail}"),
            false,
        );
        return;
    }
    match verify_artifacts(&output_directory, &artifact_values) {
        Ok(completed) => {
            let _ = app.emit(
                COMPLETE_EVENT,
                CompleteEvent {
                    job_id,
                    output_directory: output_directory.to_string_lossy().into_owned(),
                    artifacts: completed,
                    gate,
                },
            );
        }
        Err(error) => emit_error(&app, &job_id, "artifact-verification-failed", &error, false),
    }
}

fn verify_artifacts(
    output_directory: &Path,
    artifacts: &[ArtifactReceipt],
) -> Result<Vec<CompletedArtifact>, String> {
    let root = output_directory
        .canonicalize()
        .map_err(|error| format!("cannot resolve output directory: {error}"))?;
    let mut completed = Vec::new();
    for artifact in artifacts.iter().filter(|artifact| {
        artifact.designation == openastroflow_app_core::ArtifactDesignation::FinalMaster
    }) {
        artifact.validate().map_err(|error| error.to_string())?;
        let candidate = root.join(artifact.relative_path.as_str());
        let resolved = candidate
            .canonicalize()
            .map_err(|error| format!("cannot resolve {}: {error}", candidate.display()))?;
        if !resolved.starts_with(&root) || !resolved.is_file() {
            return Err("final artifact escaped its output directory or is not a file".to_owned());
        }
        let metadata = resolved.metadata().map_err(|error| error.to_string())?;
        if metadata.len() != artifact.size_bytes {
            return Err(format!("artifact size changed: {}", resolved.display()));
        }
        let digest = sha256_file(&resolved)?;
        if digest != artifact.sha256 {
            return Err(format!("artifact digest changed: {}", resolved.display()));
        }
        let astrometry = artifact
            .astrometry
            .as_ref()
            .ok_or_else(|| "ready final artifact has no astrometry receipt".to_owned())?;
        let name = resolved
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("final-master.fits")
            .to_owned();
        completed.push(CompletedArtifact {
            kind: "MASTER",
            name,
            path: resolved.to_string_lossy().into_owned(),
            detail: format!(
                "WCS embedded · {:.3} arcsec RMS · {} matched stars",
                astrometry.rms_arcsec, astrometry.matched_stars
            ),
            receipt: artifact.clone(),
        });
    }
    if completed.is_empty() {
        return Err("result gate reported ready without a final master".to_owned());
    }
    Ok(completed)
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

fn emit_error<R: Runtime>(
    app: &AppHandle<R>,
    job_id: &str,
    code: &str,
    message: &str,
    retryable: bool,
) {
    emit_error_with_details(app, job_id, code, message, retryable, BTreeMap::new());
}

fn emit_error_with_details<R: Runtime>(
    app: &AppHandle<R>,
    job_id: &str,
    code: &str,
    message: &str,
    retryable: bool,
    details: BTreeMap<String, serde_json::Value>,
) {
    let _ = app.emit(
        ERROR_EVENT,
        PipelineErrorEvent {
            job_id: job_id.to_owned(),
            code: code.to_owned(),
            message: message.to_owned(),
            retryable,
            details,
        },
    );
}

pub(crate) fn cancel_pipeline(registry: &PipelineRegistry, job_id: &str) -> Result<(), String> {
    let child = registry
        .jobs
        .lock()
        .map_err(|_| "pipeline registry lock was poisoned".to_owned())?
        .get(job_id)
        .cloned()
        .ok_or_else(|| format!("pipeline is not running: {job_id}"))?;
    let mut child = child
        .lock()
        .map_err(|_| "worker process lock was poisoned".to_owned())?;
    platform::terminate_process_tree(&mut child)
}

pub(crate) fn terminate_all(registry: &PipelineRegistry) -> Result<(), String> {
    registry.shutting_down.store(true, Ordering::Release);
    let children = registry
        .jobs
        .lock()
        .map_err(|_| "pipeline registry lock was poisoned".to_owned())?
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    for child in children {
        match child.lock() {
            Ok(mut child) => {
                if let Err(error) = platform::terminate_process_tree(&mut child) {
                    failures.push(error);
                }
            }
            Err(_) => failures.push("worker process lock was poisoned".to_owned()),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "one or more pipeline process trees could not be terminated: {}",
            failures.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_discovery_contract_never_accepts_environment_or_adjacent_overrides() {
        let candidates = development_candidate_paths(
            false,
            Some(PathBuf::from("/tmp/injected-worker")),
            Some(PathBuf::from("/tmp/OpenAstroFlow")),
        );
        assert!(candidates.is_empty());
    }

    /// Linux refuses to execute a file that is open for writing (`ETXTBSY`);
    /// the launcher waits for the writer to finish instead of failing at once.
    #[cfg(target_os = "linux")]
    #[test]
    fn sidecar_launch_waits_for_a_text_busy_executable() {
        use std::os::unix::fs::OpenOptionsExt;

        let root = std::env::temp_dir()
            .join(new_public_identifier("text-busy-test").expect("temporary id"));
        std::fs::create_dir(&root).expect("create test directory");
        let script = root.join("busy-sidecar");
        let mut writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o700)
            .open(&script)
            .expect("create fake sidecar");
        writer
            .write_all(b"#!/bin/sh\nexit 0\n")
            .expect("write fake sidecar");
        writer.sync_all().expect("sync fake sidecar");
        // The writer closes only after the launcher has started retrying.
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(writer);
        });
        let mut command = Command::new(&script);
        let output = sidecar_output(&mut command).expect("launch after the writer closed");
        assert!(output.status.success());
        release.join().expect("writer thread");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn application_shutdown_terminates_every_pipeline_process_tree() {
        let registry = PipelineRegistry::default();
        let mut command = platform::test_support::sleeping_command();
        let child = Arc::new(Mutex::new(
            ManagedChild::spawn(&mut command).expect("pipeline child"),
        ));
        registry
            .jobs
            .lock()
            .unwrap()
            .insert("pipeline-shutdown-test".to_owned(), child.clone());

        terminate_all(&registry).expect("terminate pipeline children");
        let status = child.lock().unwrap().wait().expect("reap pipeline child");
        assert!(!status.success());
        assert!(registry.shutting_down.load(Ordering::Acquire));
        terminate_all(&registry).expect("shutdown is idempotent");
    }

    #[cfg(unix)]
    #[test]
    fn bundled_runtime_manifest_binds_every_file_and_executable_mode() {
        use std::os::unix::fs::PermissionsExt;

        let resource_root = std::env::temp_dir()
            .join(new_identifier("openastroflow-runtime-manifest").expect("temporary identifier"));
        let runtime_root = resource_root.join(target_suffixed_name().trim_end_matches(".exe"));
        let internal = runtime_root.join("_internal");
        fs::create_dir_all(&internal).expect("runtime fixture directories");
        let entry_point = runtime_root.join(target_suffixed_name());
        fs::write(&entry_point, b"#!/bin/sh\nexit 0\n").expect("runtime fixture executable");
        let mut permissions = fs::metadata(&entry_point).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&entry_point, permissions).unwrap();
        let data = internal.join("runtime.dat");
        fs::write(&data, b"python-runtime").expect("runtime fixture data");
        let entries = serde_json::json!([
            {"path":"_internal","type":"directory"},
            {"executable":false,"path":"_internal/runtime.dat","sha256":sha256_file(&data).unwrap(),"sizeBytes":14,"type":"file"},
            {"executable":true,"path":target_suffixed_name(),"sha256":sha256_file(&entry_point).unwrap(),"sizeBytes":17,"type":"file"}
        ]);
        let canonical = serde_json::to_vec(entries.as_array().unwrap()).unwrap();
        let mut tree = Sha256::new();
        tree.update(canonical);
        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "kind": "openastroflow-worker-sidecar",
            "targetTriple": target_triple_name(),
            "runtime": {
                "directoryName": target_suffixed_name().trim_end_matches(".exe"),
                "entryPoint": target_suffixed_name(),
                "treeSha256": format!("{:x}", tree.finalize()),
                "sizeBytes": 31,
                "fileCount": 2,
                "entryCount": 3,
                "entries": entries
            },
            "protocol": {},
            "versions": {},
            "collections": []
        });
        fs::write(
            resource_root.join(target_manifest_name()),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .expect("runtime fixture manifest");

        let verified = verify_bundled_runtime(&resource_root).expect("verified runtime tree");
        assert_eq!(verified.path, entry_point);
        fs::write(&data, b"tampered").expect("tamper runtime fixture");
        assert!(verify_bundled_runtime(&resource_root).is_err());
        let _ = fs::remove_dir_all(resource_root);
    }

    fn profile_capabilities(profiles: &[HardwareProfile]) -> BackendCapabilities {
        let hardware_profiles = profiles.iter().copied().collect();
        let mut features = std::collections::BTreeSet::from([BackendFeature::CpuExecution]);
        if profiles.iter().any(|profile| {
            matches!(
                profile,
                HardwareProfile::GenericAppleMetal | HardwareProfile::M3ProTuned
            )
        }) {
            features.insert(BackendFeature::MetalExecution);
        }
        if profiles.contains(&HardwareProfile::M3ProTuned) {
            features.insert(BackendFeature::M3ProTuning);
        }
        BackendCapabilities {
            schema_version: 1,
            backend_id: "profile-test".to_owned(),
            backend_version: "1".to_owned(),
            worker_build: "test".to_owned(),
            hardware_profiles,
            stages: std::collections::BTreeSet::from([StageKind::Integration]),
            features,
            maximum_parallel_stages: 1,
            input_extensions: std::collections::BTreeSet::new(),
            output_extensions: std::collections::BTreeSet::new(),
        }
    }

    fn simulated_host(
        platform: &'static str,
        architecture: &'static str,
        chip: &str,
    ) -> platform::PlatformProfile {
        platform::PlatformProfile {
            platform,
            architecture,
            chip: chip.to_owned(),
            cpu_backend: "test-cpu",
            gpu_backend: "test-gpu",
            optimization_tier: "PORTABLE",
        }
    }

    #[test]
    fn profile_selection_never_maps_linux_x86_to_arm64() {
        let capabilities = profile_capabilities(&[
            HardwareProfile::PortableCpu,
            HardwareProfile::GenericArm64Cpu,
        ]);
        let host = simulated_host("linux", "x86_64", "x86_64");
        assert_eq!(
            select_profile_for_host(&capabilities, &host),
            Ok(HardwareProfile::PortableCpu)
        );
    }

    #[test]
    fn profile_selection_keeps_windows_on_its_explicit_cpu_contract() {
        let host = simulated_host("windows", "x86_64", "x86_64");
        let capabilities =
            profile_capabilities(&[HardwareProfile::PortableCpu, HardwareProfile::WindowsCpu]);
        assert_eq!(
            select_profile_for_host(&capabilities, &host),
            Ok(HardwareProfile::WindowsCpu)
        );
        let portable_only = profile_capabilities(&[HardwareProfile::PortableCpu]);
        assert!(select_profile_for_host(&portable_only, &host).is_err());
    }

    #[test]
    fn profile_selection_uses_only_worker_advertised_metal() {
        let host = simulated_host("macos", "aarch64", "Apple M3 Pro");
        let cpu_only = profile_capabilities(&[
            HardwareProfile::PortableCpu,
            HardwareProfile::GenericArm64Cpu,
        ]);
        assert_eq!(
            select_profile_for_host(&cpu_only, &host),
            Ok(HardwareProfile::GenericArm64Cpu)
        );
        let probed = profile_capabilities(&[
            HardwareProfile::PortableCpu,
            HardwareProfile::GenericArm64Cpu,
            HardwareProfile::GenericAppleMetal,
            HardwareProfile::M3ProTuned,
        ]);
        assert_eq!(
            select_profile_for_host(&probed, &host),
            Ok(HardwareProfile::M3ProTuned)
        );
    }

    #[test]
    fn windows_release_validation_is_x86_64_only() {
        assert!(platform_scientific_release_validated(&simulated_host(
            "windows",
            "x86_64",
            "AMD Ryzen 7 5800H"
        )));
        assert!(platform_scientific_release_validated(&simulated_host(
            "macos",
            "aarch64",
            "Apple M3 Pro"
        )));
        let arm = simulated_host("windows", "aarch64", "Snapdragon X Elite");
        assert!(!platform_scientific_release_validated(&arm));
        let reason = platform_unavailable_reason(&arm);
        assert!(reason.contains("aarch64"));
        assert!(reason.contains("x86-64 only"));
    }

    #[test]
    fn worker_environment_forces_utf8_stdio() {
        let command = EngineExecutable {
            path: PathBuf::from("openastroflow-worker"),
        }
        .command("worker");
        let environment = command
            .get_envs()
            .filter_map(|(key, value)| Some((key.to_str()?, value?.to_str()?)))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(environment.get("PYTHONUTF8"), Some(&"1"));
        assert_eq!(environment.get("PYTHONIOENCODING"), Some(&"utf-8"));
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            vec![std::ffi::OsStr::new("worker")]
        );
    }

    #[test]
    fn lossy_lines_survive_invalid_utf8_and_keep_later_lines() {
        let stream: Vec<u8> =
            b"first\r\nbad \xff byte\n{\"type\":\"progress\"}\nno newline".to_vec();
        let lines = LossyLines::new(std::io::Cursor::new(stream)).collect::<Vec<_>>();
        assert_eq!(
            lines,
            vec![
                "first".to_owned(),
                "bad \u{fffd} byte".to_owned(),
                "{\"type\":\"progress\"}".to_owned(),
                "no newline".to_owned(),
            ]
        );
        assert!(LossyLines::new(std::io::Cursor::new(Vec::new()))
            .next()
            .is_none());
    }

    #[cfg(unix)]
    fn fake_sidecar() -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir()
            .join(new_identifier("openastroflow-fake-sidecar").expect("temporary identifier"));
        std::fs::create_dir(&root).expect("create fake sidecar directory");
        let script = root.join("fake-openastroflow-engine");
        let source = r###"#!/usr/bin/env python3
import argparse, base64, hashlib, json, os, pathlib, platform, re, sys

CPU_PROFILE = "generic-arm64-cpu" if platform.machine().lower() in {"arm64", "aarch64"} else "portable-cpu"

CAPS = {
  "schemaVersion": 1, "backendId": "fake-sidecar", "backendVersion": "0.1.0",
  "workerBuild": "test", "hardwareProfiles": [CPU_PROFILE],
  "stages": ["quality-control", "calibration", "registration", "integration", "drizzle", "astrometric-solve"],
  "features": ["cpu-execution", "deterministic-receipts", "offline-astrometric-solver", "drizzle", "fits"],
  "maximumParallelStages": 1, "inputExtensions": ["fits"], "outputExtensions": ["fits"]
}

def envelope(session, sequence, kind, payload):
  return {"protocolVersion": 1, "sessionId": session, "sequence": sequence,
          "sentAtUnixMs": 10 + sequence, "type": kind, "payload": payload}

def recipe(mode):
  stages = [
    {"stageId":"quality-control","kind":"quality-control","enabled":True,"dependsOn":[],"parameters":{}},
    {"stageId":"calibrate","kind":"calibration","enabled":True,"dependsOn":["quality-control"],"parameters":{}},
    {"stageId":"register","kind":"registration","enabled":True,"dependsOn":["calibrate"],"parameters":{}},
    {"stageId":"integrate","kind":"integration","enabled":True,"dependsOn":["register"],"parameters":{}},
  ]
  dependency = "integrate"
  if mode == "drizzle":
    stages.append({"stageId":"drizzle","kind":"drizzle","enabled":True,"dependsOn":["integrate"],"parameters":{}})
    dependency = "drizzle"
  stages.append({"stageId":"solve","kind":"astrometric-solve","enabled":True,"dependsOn":[dependency],"parameters":{}})
  return {"schemaVersion":1,"recipeId":"fake-e2e","displayName":"Fake E2E","stages":stages,
          "solver":{"result":"required","catalog":"astrometry-net-offline","projection":"TAN","minimumMatches":12,"maximumRmsArcsec":2.0},
          "drizzle":{"result":"required" if mode == "drizzle" else "disabled","scale":2.0,"dropShrink":0.9,"kernel":"square"},
          "parameters":{}}

def controller_plan(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+")
  parser.add_argument("--mode", default="ordinary")
  parser.add_argument("--session-id", required=True)
  parser.add_argument("--sequence", type=int, required=True)
  parser.add_argument("--request-id", required=True)
  parser.add_argument("--plan-id", required=True)
  parser.add_argument("--hardware-profile", required=True)
  parser.add_argument("--compact", action="store_true")
  args = parser.parse_args(argv)
  sources = []
  for index, path in enumerate(args.inputs):
    name = pathlib.Path(path).name.lower()
    role = "flat" if "flat" in name else "bias" if "bias" in name else "dark" if "dark" in name else "light"
    sources.append({"sourceId":f"source-{index}","role":role,"hostPath":path,"recursive":False})
  payload = {"requestId":args.request_id,"planId":args.plan_id,
    "project":{"schemaVersion":1,"projectId":"fake-project","displayName":"Unicode 盾牌座","createdAtUnixMs":1,"sources":sources,"labels":{}},
    "recipe":recipe(args.mode),"requestedHardwareProfile":args.hardware_profile,"inputManifestSha256":"0"*64}
  print(json.dumps(envelope(args.session_id,args.sequence,"plan",payload),ensure_ascii=False,separators=(",",":")),flush=True)

def inventory():
  print(json.dumps({"name":"Unicode 盾牌座","assets":[
    {"path":"/数据/盾牌座/亮场 01.fit","role":"LIGHT","status":"READY","cfaPattern":"NONE","roleEvidence":["header:IMAGETYP"]},
    {"path":"/数据/校准/平场 R.fit","role":"FLAT","status":"READY","cfaPattern":"NONE","roleEvidence":["header:IMAGETYP"]}],"issues":[]},ensure_ascii=False),flush=True)

def quality_check(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument("--request-json", required=True)
  parser.add_argument("--compact", action="store_true")
  args = parser.parse_args(argv)
  request = json.loads(pathlib.Path(args.request_json).read_text())
  frames = [{"path":path,"sourceSha256":"sha256:"+hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),
             "disposition":"REVIEW","decision":"REVIEW","confidence":"HIGH","starCount":42,
             "summary":"fake real gate","evidence":[{"code":"GATE_FAKE","family":"PROVENANCE","severity":"REVIEW","message":"review fixture"}]}
            for path in request["lightPaths"]]
  print(json.dumps({"schemaVersion":1,"gatePolicyDigest":"sha256:"+"7"*64,"workers":1,
                    "counts":{"PASS":0,"REVIEW":len(frames),"HARD_FAIL":0},"frames":frames},separators=(",",":")),flush=True)

FILMSTRIP_JPEG = base64.b64decode(
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/wAALCAAIAAwBAREA"
  "/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRol"
  "JicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi"
  "4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AMPzGdVym53+Z2ZGCgjpknGTkAeg5/FqXMsMakKhL5Y7sDGCR0yOwFf/2Q==")
ZOOM_PNG = base64.b64decode(
  "iVBORw0KGgoAAAANSUhEUgAAAAwAAAAICAAAAADoj0EtAAAAd0lEQVR4AQKMMVqB6R3nD/5/r39KvmBh+f7+l+Cr9wI/+Z8Ks/zj/8P0ipGX+x87/2cW3mc/uCQ+sD5nFeV7z8Im"
  "cP834/v//1j+32FnYbyj8PcZv+i7f3/4vzH9EWd+KsZ6W+jrf24ups/vGOWeMDC+kPz+/z0A4Qsw9hBiytcAAAAASUVORK5CYII=")

def blink_fail(code, message):
  print(json.dumps({"ok":False,"error":{"code":code,"message":message}}), file=sys.stderr, flush=True)
  raise SystemExit(2)

def blink_measure(argv):
  # A schema-exact blink-manifest-v1 fixture: two channels (L, R) with nights,
  # the combined EXCLUDE rule, an ATTENTION flag, one reference per channel and
  # real JPEG/PNG previews written create-only into the session directory.
  parser = argparse.ArgumentParser()
  parser.add_argument("--request-json", required=True)
  parser.add_argument("--compact", action="store_true")
  args = parser.parse_args(argv)
  request = json.loads(pathlib.Path(args.request_json).read_text(encoding="utf-8"))
  if request.get("schemaVersion") != 1 or set(request) - {"schemaVersion","lightPaths","sessionDirectory","workers","previews","masterFlats"}:
    blink_fail("BLINK_REQUEST_INVALID", "unsupported request fields")
  session = pathlib.Path(request["sessionDirectory"])
  try:
    session.mkdir(parents=False, exist_ok=False)
  except FileExistsError:
    blink_fail("BLINK_SESSION_EXISTS", "session directory exists")
  (session / "filmstrip").mkdir(); (session / "zoom").mkdir()
  frames = []; channels = {}
  ordered = sorted(request["lightPaths"], key=lambda p: (("_R_" in pathlib.Path(p).name), p))
  for index, path in enumerate(ordered):
    name = pathlib.Path(path).name
    filt = "R" if "_R_" in name else "L"
    night = re.search(r"(\d{4}-\d{2}-\d{2})", name); night = night.group(1) if night else "2026-08-17"
    flags = []
    if "moon" in name:
      flags = [{"code":"BLINK_SKY_BRIGHT","severity":"EXCLUDE","value":2.43,"threshold":1.6,"combined":True,"message":"Sky 2.43x the clean-sky level and 55 % of its stars"},
               {"code":"BLINK_SOURCES_LOW","severity":"ATTENTION","value":0.55,"threshold":0.6,"combined":True,"message":"55 % of the channel's best star count"}]
    elif "haze" in name:
      flags = [{"code":"BLINK_EXTINCTION","severity":"ATTENTION","value":0.67,"threshold":0.5,"combined":False,"message":"0.67 mag extra extinction"}]
    safe = re.sub(r"[^A-Za-z0-9]+", "-", pathlib.Path(name).stem)[:40]
    filmstrip = f"filmstrip/{index:04d}-{filt}-{safe}.jpg"; zoom = f"zoom/{index:04d}-{filt}-{safe}.png"
    (session / filmstrip).write_bytes(FILMSTRIP_JPEG + (b"\0" * 210 * 1024 if "oversize" in name else b""))
    (session / zoom).write_bytes(ZOOM_PNG)
    digest = "sha256:" + hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    channel = channels.setdefault(filt, {"channelId":f"group-{filt.lower()}0000000000","target":"NGC 6822","filter":filt,"frameCount":0,
      "reference":None,"statistics":{"skyClean":1090.0,"cleanCount":3,"sourcesBest":5680,"fwhmBest":4.07},
      "stretch":{"black":872.0,"white":1420.0,"softness":4.0,"skyReference":986.0,"sigmaReference":43.3},
      "previewGeometry":{"filmstrip":[12,8],"zoom":[12,8],"sourceShape":[4176,6252]},"nights":{}})
    channel["frameCount"] += 1
    excluded = any(f["severity"] == "EXCLUDE" for f in flags)
    reference = channel["reference"] is None and not flags
    if reference:
      channel["reference"] = {"index":index,"sourceSha256":digest,"rule":"psf-signal-weight-proxy-v1"}
    summary = channel["nights"].setdefault(night, {"night":night,"frameCount":0,"medianSky":2360 if excluded else 1000,"skyRatio":2.17 if excluded else 1.0,
      "medianSourceRatio":0.53 if excluded else 0.95,"medianExtinction":0.28,"exclude":0,"attention":0,"defaultDropNight":excluded})
    summary["frameCount"] += 1; summary["exclude"] += int(excluded); summary["attention"] += int(bool(flags) and not excluded)
    frames.append({"index":index,"channelId":channel["channelId"],"filter":filt,"target":"NGC 6822","night":night,"path":path,"name":name,
      "sourceSha256":digest,"observedAt":night+"T22:57:01","airmass":1.31,"reference":reference,"defaultDecision":"DROP" if excluded else "KEEP",
      "flags":flags,"notes":[] if flags else ["GATE_INSUFFICIENT_COHORT"],"gate":{"disposition":"PASS","codes":[]},
      "metrics":{"sky":2647.3 if excluded else 986.0,"skyRatio":2.43 if excluded else 1.0,"starCount":2947,"sourceRatio":0.55,"extinctionMag":0.23,
        "transparency":0.84,"fwhmNative":4.47,"fwhmRatio":1.1,"ellipticity":0.09,"eccentricity":0.38,"registrationRms":0.21,"matchedStars":1900,
        "overlap":1.0,"backgroundShape":0.13,"gradientRatio":None},
      "score":{"log10":-5.0,"z":-3.1,"rank":index+1},
      "previews":{"filmstrip":filmstrip,"zoom":zoom,"coverage":0.99},
      "transformToReference":[[1.0,0.0,1.27],[0.0,1.0,-0.81]],"normalization":{"skyOffset":2647.3,"fluxScale":1.19,"registered":True}})
  for channel in channels.values():
    if channel["reference"] is None:
      first = next(f for f in frames if f["channelId"] == channel["channelId"]); first["reference"] = True
      channel["reference"] = {"index":first["index"],"sourceSha256":first["sourceSha256"],"rule":"psf-signal-weight-proxy-v1"}
    channel["nights"] = list(channel["nights"].values())
  exclude = sum(f["defaultDecision"] == "DROP" for f in frames); attention = sum(f["defaultDecision"] == "KEEP" and bool(f["flags"]) for f in frames)
  manifest = {"schemaVersion":1,"kind":"blink-manifest-v1","sessionId":session.name,"sessionDirectory":str(session.resolve()),
    "createdAt":"2026-09-22T00:00:00","engineVersion":"fake","gatePolicyDigest":"sha256:"+"7"*64,"flagsPolicyDigest":"sha256:"+"8"*64,
    "flagsPolicy":{"version":"blink-flags-v1","skyBrightAttention":1.6},
    "inventorySha256":"sha256:"+hashlib.sha256("".join(f["sourceSha256"] for f in frames).encode()).hexdigest(),
    "timings":{"measurementSeconds":0.1,"analysisSeconds":0.1,"gateSeconds":0.1,"flagsSeconds":0.01,"previewSeconds":0.1},
    "counts":{"frames":len(frames),"exclude":exclude,"attention":attention,"clean":len(frames)-exclude-attention},
    "channels":list(channels.values()),"frames":frames,
    "requestEcho":{"workers":request.get("workers"),"masterFlats":request.get("masterFlats"),"previews":request["previews"]}}
  (session / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
  print(json.dumps(manifest, ensure_ascii=False, separators=(",",":") if args.compact else None), flush=True)

def artifact_payload(request_id, run_id, stage_id, stage_kind, artifact):
  stage={"schemaVersion":1,"stageId":stage_id,"kind":stage_kind,"status":"succeeded","startedAtUnixMs":1,
         "finishedAtUnixMs":2,"artifactIds":[artifact["artifactId"]],"metrics":{}}
  return {"requestId":request_id,"runId":run_id,"stage":stage,"artifact":artifact}

def worker():
  hello=json.loads(sys.stdin.readline())
  session=hello["sessionId"]
  print(json.dumps(envelope(session,0,"handshake",{"role":"worker","implementation":"fake-sidecar","implementationVersion":"0.1.0","supportedProtocolVersions":[1],"capabilities":CAPS}),separators=(",",":")),flush=True)
  plan_line=sys.stdin.readline()
  if not plan_line: return
  plan=json.loads(plan_line)
  execute=json.loads(sys.stdin.readline())
  request_id=execute["payload"]["requestId"]; run_id=execute["payload"]["runId"]
  sequence=1
  for stage_id,kind in [("quality-control","quality-control"),("calibrate","calibration"),("register","registration"),("integrate","integration")]:
    artifact={"schemaVersion":1,"artifactId":"evidence-"+stage_id,"producedByStageId":stage_id,"kind":"run-log","designation":"diagnostic",
              "relativePath":"evidence/"+stage_id+".json","mediaType":"application/json","sha256":"1"*64,"sizeBytes":1,"createdAtUnixMs":2,"attributes":{}}
    print(json.dumps(envelope(session,sequence,"artifact",artifact_payload(request_id,run_id,stage_id,kind,artifact)),separators=(",",":")),flush=True); sequence+=1
  output=pathlib.Path(execute["payload"]["outputParentHostPath"])/execute["payload"]["outputDirectoryName"]
  master=output/"master"/"Unicode 盾牌座_master.fits"; master.parent.mkdir(parents=True)
  master.write_bytes(b"FAKE-FITS-FOR-CONTROLLER-TEST")
  digest=hashlib.sha256(master.read_bytes()).hexdigest()
  final={"schemaVersion":1,"artifactId":"final-master","producedByStageId":"solve","kind":"final-master","designation":"final-master",
         "relativePath":"master/Unicode 盾牌座_master.fits","mediaType":"image/fits","sha256":digest,"sizeBytes":master.stat().st_size,"createdAtUnixMs":3,
         "astrometry":{"referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":281.0,"centerDecDegrees":-6.0,"pixelScaleArcsec":1.4,
           "rotationDegrees":0.0,"rmsPixels":0.3,"rmsArcsec":0.42,"matchedStars":73,"parity":"POSITIVE","catalogIdentity":"2"*64,
           "indexIdentities":["astrometry.net:index:4108:healpix:123:hpnside:4"],"correspondenceSha256":"3"*64,
           "catalogManaged":True,"installedSetIdentity":"5"*64,"catalogManifestSha256":"6"*64,
           "indexArtifacts":[{"indexId":"4108","relativeName":"index-4108.fits","sizeBytes":94550400,"sha256":"7"*64,"manifestSha256":"6"*64,"installedSetIdentity":"5"*64}],
           "wcsSha256":"4"*64},"attributes":{}}
  print(json.dumps(envelope(session,sequence,"artifact",artifact_payload(request_id,run_id,"solve","astrometric-solve",final)),ensure_ascii=False,separators=(",",":")),flush=True)
  print("fake diagnostic stays on stderr",file=sys.stderr,flush=True)

if sys.argv[1] == "controller-plan": controller_plan(sys.argv[2:])
elif sys.argv[1] == "inventory": inventory()
elif sys.argv[1] == "quality-check": quality_check(sys.argv[2:])
elif sys.argv[1] == "blink-measure": blink_measure(sys.argv[2:])
elif sys.argv[1] == "worker": worker()
else: raise SystemExit(2)
"###;
        std::fs::write(&script, source).expect("write fake sidecar");
        let mut permissions = std::fs::metadata(&script)
            .expect("fake sidecar metadata")
            .permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).expect("make fake sidecar executable");
        (root, script)
    }

    #[test]
    fn calibration_preflight_rejects_false_ready_reports() {
        let ready = serde_json::json!({
            "schemaVersion": 1, "status": "READY", "calibrationReady": true,
            "groups": [{"groupId": "group-1", "target": "M31", "filter": "B",
                "lightCount": 8, "observedDates": ["2026-09-04", "2026-09-05"],
                "status": "READY", "matches": {
                    "FLAT": {"rawCount": 20, "masterCount": 0},
                    "DARK": {"rawCount": 0, "masterCount": 1},
                    "BIAS": {"rawCount": 0, "masterCount": 1}}}],
            "issues": []
        });
        let good: CalibrationInspection = serde_json::from_value(ready.clone()).unwrap();
        validate_calibration_inspection(&good).expect("matching metadata should be ready");
        let mut contradictory = ready.clone();
        contradictory["issues"] = serde_json::json!([{
            "code": "CALIBRATION_MISSING", "severity": "ERROR", "message": "No matching flat",
            "paths": [], "lightGroups": ["group-1"]
        }]);
        let bad: CalibrationInspection = serde_json::from_value(contradictory).unwrap();
        assert!(validate_calibration_inspection(&bad).is_err());
        let mut empty = ready;
        empty["groups"] = serde_json::json!([]);
        let bad: CalibrationInspection = serde_json::from_value(empty).unwrap();
        assert!(validate_calibration_inspection(&bad).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn calibration_inspection_transports_recipe_and_returns_blockers() {
        use std::os::unix::fs::PermissionsExt;
        let root =
            std::env::temp_dir().join(new_identifier("wbpp-calibration-bridge-test").unwrap());
        fs::create_dir(&root).unwrap();
        let input = root.join("亮场.fit");
        fs::write(&input, b"read-only transport fixture").unwrap();
        let worker = root.join("worker.py");
        fs::write(
            &worker,
            r###"#!/usr/bin/env python3
import json, pathlib, sys
assert sys.argv[1:3] == ['calibration-check', '--request-json']
p = pathlib.Path(sys.argv[3]); data = json.loads(p.read_text())
assert data['schemaVersion'] == 1 and len(data['paths']) == 1
assert data['recipe']['calibration']['bias'] == 'REQUIRED'
(pathlib.Path(__file__).parent/'request-path.txt').write_text(str(p))
print(json.dumps({'schemaVersion':1, 'status':'BLOCKED', 'calibrationReady':False,
 'groups':[{'groupId':'b','target':'M31','filter':'B','lightCount':1,'observedDates':[],
 'status':'BLOCKED','matches':{k:{'rawCount':0,'masterCount':0} for k in ['FLAT','DARK','BIAS']}}],
 'issues':[{'code':'CALIBRATION_MISSING','severity':'ERROR','message':'No matching flat',
 'paths':data['paths'],'lightGroups':['b']}]}))
"###,
        )
        .unwrap();
        fs::set_permissions(&worker, fs::Permissions::from_mode(0o700)).unwrap();
        let result = inspect_calibration_with(
            EngineExecutable { path: worker },
            InspectCalibrationRequest {
                paths: vec![input.to_string_lossy().into_owned()],
                recipe: serde_json::json!({"calibration":{"bias":"REQUIRED"}}),
            },
        )
        .expect("a scientific blocker is a report, not a transport failure");
        assert!(!result.calibration_ready);
        assert_eq!(result.issues[0].code, "CALIBRATION_MISSING");
        let private_request = fs::read_to_string(root.join("request-path.txt")).unwrap();
        assert!(!Path::new(&private_request).exists());
        assert_eq!(fs::read(&input).unwrap(), b"read-only transport fixture");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_missing_calibration_roles_before_spawning() {
        let error = unique_input_paths(&[RunSource {
            role: "LIGHT".to_owned(),
            paths: vec!["/tmp/light.fit".to_owned()],
        }])
        .expect_err("missing flats and bias must fail");
        assert!(error.contains("FLAT"));
    }

    #[test]
    fn unicode_paths_are_preserved_and_deduplicated() {
        let path = "/tmp/盾牌座/亮场 01.fit".to_owned();
        let sources = [
            RunSource {
                role: "LIGHT".to_owned(),
                paths: vec![path.clone(), path.clone()],
            },
            RunSource {
                role: "FLAT".to_owned(),
                paths: vec!["/tmp/平场.fit".to_owned()],
            },
            RunSource {
                role: "BIAS".to_owned(),
                paths: vec!["/tmp/偏置.fit".to_owned()],
            },
        ];
        let result = unique_input_paths(&sources).expect("valid sources");
        assert_eq!(result.iter().filter(|item| *item == &path).count(), 1);
    }

    #[test]
    fn deferred_cfa_confirmation_hashes_unicode_files_without_mutating_them() {
        let root = std::env::temp_dir()
            .join(new_identifier("openastroflow-hash-source").expect("temporary identifier"));
        std::fs::create_dir(&root).expect("temporary directory");
        let path = root.join("盾牌座 亮场.fit");
        std::fs::write(&path, b"read-only-source-fixture").expect("source fixture");
        let before = std::fs::read(&path).unwrap();
        let result = hash_sources(HashSourcesRequest {
            paths: vec![path.to_string_lossy().into_owned()],
        })
        .expect("content hash");
        assert!(result.entries[0].source_sha256.starts_with("sha256:"));
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(hash_sources(HashSourcesRequest {
            paths: vec![
                path.to_string_lossy().into_owned(),
                path.to_string_lossy().into_owned(),
            ],
        })
        .is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn recipe_mapping_is_explicit() {
        assert_eq!(recipe_cli_id("balanced"), Ok("balanced"));
        assert_eq!(recipe_cli_id("drizzle-2x"), Ok("drizzle-2x"));
        assert!(recipe_cli_id("future-recipe").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn calibration_only_import_accepts_batch_without_lights_but_rejects_bad_frames() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(new_identifier("calibration-only-import").unwrap());
        let other_directory = root.join("other calibration directory");
        fs::create_dir_all(&other_directory).unwrap();
        let script = root.join("inventory-worker");
        fs::write(
            &script,
            r###"#!/usr/bin/env python3
import pathlib, sys
assert sys.argv[1] == 'inventory'
print((pathlib.Path(__file__).parent / 'inventory.json').read_text())
"###,
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        let executable = EngineExecutable { path: script };
        let no_lights = serde_json::json!({
            "code":"NO_LIGHTS", "severity":"ERROR",
            "message":"the selected inputs contain no unprocessed Light frames"
        });
        let mut last = serde_json::Value::Null;
        let mut input = String::new();
        for role in [
            "MASTER_FLAT",
            "MASTER_DARK",
            "MASTER_BIAS",
            "FLAT",
            "DARK",
            "BIAS",
        ] {
            let path = other_directory.join(format!("{role}.fits"));
            fs::write(&path, b"read-only calibration fixture").unwrap();
            input = path.to_string_lossy().into_owned();
            last = serde_json::json!({"name":"new batch", "assets":[{
                "path":input, "role":role, "status":"READY", "width":4,
                "height":4,"channels":1,"filter":"B","camera":"test camera",
                "roleEvidence":["FITS:IMAGETYP"]}], "issues":[no_lights.clone()]});
            fs::write(
                root.join("inventory.json"),
                serde_json::to_vec(&last).unwrap(),
            )
            .unwrap();
            let imported = inspect_paths_with(
                executable.clone(),
                InspectRequest {
                    paths: vec![input.clone()],
                    role_hint: None,
                },
            )
            .expect("a calibration-only addition must not require a Light in that batch");
            assert_eq!(imported.total_files, 1);
            assert_eq!(imported.sources[0].role, role);
            assert_eq!(imported.sources[0].paths, vec![input.clone()]);
            assert_eq!(
                imported.assets[0].source_sha256.is_some(),
                role.starts_with("MASTER_")
            );
            assert_eq!(fs::read(&path).unwrap(), b"read-only calibration fixture");
        }
        for code in ["ROLE_CONFLICT", "SOURCE_STAT_FAILED", "UNSUPPORTED_FORMAT"] {
            let mut invalid = last.clone();
            invalid["issues"] = serde_json::json!([no_lights.clone(), {
                "code":code,"severity":"ERROR","message":"invalid calibration source"
            }]);
            fs::write(
                root.join("inventory.json"),
                serde_json::to_vec(&invalid).unwrap(),
            )
            .unwrap();
            let error = inspect_paths_with(
                executable.clone(),
                InspectRequest {
                    paths: vec![input.clone()],
                    role_hint: None,
                },
            )
            .expect_err("real per-frame errors must still block import");
            assert!(error.contains(code));
        }
        last["assets"][0]["status"] = serde_json::json!("ERROR");
        fs::write(
            root.join("inventory.json"),
            serde_json::to_vec(&last).unwrap(),
        )
        .unwrap();
        assert!(inspect_paths_with(
            executable,
            InspectRequest {
                paths: vec![input],
                role_hint: None,
            }
        )
        .unwrap_err()
        .contains("frame is not ready"));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn fake_sidecar_handshake_and_inventory_preserve_unicode_roles() {
        let (root, script) = fake_sidecar();
        let executable = EngineExecutable { path: script };
        let probe = probe_runtime_with(executable.clone()).expect("canonical handshake");
        assert_eq!(probe.capabilities.backend_id, "fake-sidecar");
        let inventory = inspect_paths_with(
            executable,
            InspectRequest {
                paths: vec!["/选择的目录".to_owned()],
                role_hint: None,
            },
        )
        .expect("real fake-sidecar inventory");
        assert_eq!(inventory.total_files, 2);
        assert!(inventory.sources.iter().any(|source| source
            .paths
            .contains(&"/数据/盾牌座/亮场 01.fit".to_owned())));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn fake_sidecar_quality_inspection_is_content_bound_and_real() {
        let (root, script) = fake_sidecar();
        let light = root.join("盾牌座 真实亮场.fit");
        std::fs::write(&light, b"quality-light-bytes").expect("write quality fixture");
        let inspection = inspect_quality_with(
            EngineExecutable { path: script },
            InspectQualityRequest {
                paths: vec![light.to_string_lossy().into_owned()],
            },
        )
        .expect("quality inspection");
        assert_eq!(inspection.counts["REVIEW"], 1);
        assert_eq!(inspection.frames[0].disposition, "REVIEW");
        assert!(inspection.frames[0]
            .source_sha256
            .as_deref()
            .is_some_and(|value| value.starts_with("sha256:")));
        assert_eq!(std::fs::read(&light).unwrap(), b"quality-light-bytes");
        let _ = std::fs::remove_dir_all(root);
    }

    /// Four Lights for the fake `blink-measure`: an L reference night, a
    /// moonlit L frame (combined EXCLUDE rule), a hazy R frame (ATTENTION)
    /// and a clean R frame whose filmstrip exceeds the inline bound.
    #[cfg(unix)]
    fn blink_lights(root: &Path) -> Vec<String> {
        let lights = root.join("Lights 盾牌座");
        std::fs::create_dir_all(&lights).expect("lights directory");
        [
            "NGC 6822_300.00s_L_2026-08-17_22-57-01_+8.00°C.fits",
            "NGC 6822_300.00s_L_2026-08-20_21-15-27_moon.fits",
            "NGC 6822_300.00s_R_2026-09-06_haze.fits",
            "NGC 6822_300.00s_R_2026-09-08_oversize 盾牌座.fits",
        ]
        .iter()
        .map(|name| {
            let path = lights.join(name);
            std::fs::write(&path, format!("light bytes of {name}")).expect("write light");
            path.to_string_lossy().into_owned()
        })
        .collect()
    }

    #[cfg(unix)]
    #[test]
    fn fake_sidecar_blink_measure_is_validated_and_previews_are_bounded() {
        let (root, script) = fake_sidecar();
        let paths = blink_lights(&root);
        let flat = root.join("masterFlat_L.xisf");
        std::fs::write(&flat, b"flat bytes").expect("write flat");
        let second_r_flat = root.join("masterFlat_R_2.xisf");
        std::fs::write(&second_r_flat, b"flat bytes").expect("write flat");
        let sessions = root.join("blink-sessions");
        let master_flat = |filter: &str, path: &Path| BlinkMasterFlat {
            filter: filter.to_owned(),
            path: path.to_string_lossy().into_owned(),
        };
        let manifest = blink_measure_with(
            EngineExecutable { path: script },
            BlinkMeasureRequest {
                paths: paths.clone(),
                // Two flats for R: that filter is left out, L travels.
                master_flats: vec![
                    master_flat("L", &flat),
                    master_flat("R", &flat),
                    master_flat("R", &second_r_flat),
                ],
                workers: Some(3),
            },
            &sessions,
        )
        .expect("blink measurement");
        assert_eq!(manifest.kind, BLINK_MANIFEST_KIND);
        assert_eq!(
            manifest.counts,
            BlinkCounts {
                frames: 4,
                exclude: 1,
                attention: 1,
                clean: 2
            }
        );
        assert_eq!(manifest.channels.len(), 2);
        // The request transported the optional fields and the preview policy.
        assert_eq!(manifest.extra["requestEcho"]["workers"], 3);
        assert_eq!(
            manifest.extra["requestEcho"]["masterFlats"],
            serde_json::json!([{"filter": "L", "path": flat.canonicalize().unwrap()}])
        );
        assert_eq!(
            manifest.extra["requestEcho"]["previews"]["filmstripFormat"],
            "jpeg"
        );
        assert!(manifest.extra.contains_key("timings"));
        // The session lives under the sessions root with the desktop's name.
        let session = Path::new(&manifest.session_directory);
        assert_eq!(
            session.parent(),
            Some(sessions.canonicalize().unwrap().as_path())
        );
        assert!(crate::project::blink_session_name_parts(
            session.file_name().unwrap().to_str().unwrap()
        )
        .is_some());
        let manifest_file = session.join("manifest.json");
        assert!(manifest_file.is_file());
        assert_eq!(
            manifest.manifest_sha256.as_deref().unwrap(),
            format!("sha256:{}", sha256_file(&manifest_file).unwrap())
        );
        let frame = |needle: &str| {
            manifest
                .frames
                .iter()
                .find(|frame| frame.path.contains(needle))
                .expect("frame")
        };
        let reference = frame("22-57-01");
        assert!(reference.reference && reference.flags.is_empty());
        let l_channel = manifest
            .channels
            .iter()
            .find(|channel| channel.filter == "L")
            .unwrap();
        assert_eq!(l_channel.reference.index, reference.index);
        assert_eq!(l_channel.frame_count, 2);
        let moon = frame("moon");
        assert_eq!(moon.default_decision, "DROP");
        assert_eq!(moon.flags[0].code, "BLINK_SKY_BRIGHT");
        assert_eq!(moon.flags[0].extra["combined"], true);
        let haze = frame("haze");
        assert_eq!(haze.default_decision, "KEEP");
        assert_eq!(haze.flags[0].severity, "ATTENTION");
        for item in [reference, moon, haze] {
            let url = item.previews.filmstrip_data_url.as_deref().unwrap();
            assert!(url.starts_with("data:image/jpeg;base64,/9j/"));
            assert_eq!(item.extra["metrics"]["matchedStars"], 1900);
        }
        // The oversize filmstrip stays on disk for the on-demand path.
        let oversize = frame("oversize");
        assert!(oversize.previews.filmstrip_data_url.is_none());
        let zoom = oversize.previews.zoom.as_deref().unwrap();
        let loaded =
            crate::project::load_blink_preview_with(&sessions, &manifest.session_directory, zoom)
                .expect("zoom preview");
        assert!(loaded.starts_with("data:image/png;base64,iVBOR"));
        let filmstrip = oversize.previews.filmstrip.as_deref().unwrap();
        assert!(crate::project::load_blink_preview_with(
            &sessions,
            &manifest.session_directory,
            filmstrip
        )
        .is_ok());
        // Every source file is untouched and the serialised manifest keeps
        // the pass-through fields next to the typed ones.
        for path in &paths {
            assert!(std::fs::read_to_string(path)
                .unwrap()
                .starts_with("light bytes"));
        }
        let encoded = serde_json::to_value(&manifest).unwrap();
        assert_eq!(encoded["frames"][0]["score"]["rank"], 1);
        assert_eq!(encoded["channels"][0]["nights"][0]["night"], "2026-08-17");
        assert!(encoded["frames"][0]["previews"]["filmstripDataUrl"].is_string());
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn fake_sidecar_blink_sessions_are_pruned_to_the_newest_three() {
        let (root, script) = fake_sidecar();
        let paths = blink_lights(&root);
        let sessions = root.join("blink-sessions");
        std::fs::create_dir_all(sessions.join("not-ours")).unwrap();
        std::fs::write(sessions.join("not-ours/keep.txt"), b"foreign").unwrap();
        let mut directories = Vec::new();
        for _ in 0..4 {
            let manifest = blink_measure_with(
                EngineExecutable {
                    path: script.clone(),
                },
                BlinkMeasureRequest {
                    paths: paths.clone(),
                    master_flats: vec![],
                    workers: None,
                },
                &sessions,
            )
            .expect("blink measurement");
            directories.push(PathBuf::from(manifest.session_directory));
        }
        assert!(!directories[0].exists(), "oldest session removed");
        for directory in &directories[1..] {
            assert!(directory.join("manifest.json").is_file());
        }
        assert!(sessions.join("not-ours/keep.txt").is_file());
        // A refused request leaves no session behind: it names a Light twice,
        // which the request check rejects before spawning.
        let mut duplicated = paths.clone();
        duplicated.push(paths[0].clone());
        let error = blink_measure_with(
            EngineExecutable { path: script },
            BlinkMeasureRequest {
                paths: duplicated,
                master_flats: vec![],
                workers: Some(0),
            },
            &sessions,
        )
        .unwrap_err();
        assert!(error.contains("unique regular Light files"));
        assert_eq!(std::fs::read_dir(&sessions).unwrap().count(), 4);
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn fake_sidecar_blink_failure_removes_its_session() {
        use std::os::unix::fs::PermissionsExt;

        let (root, script) = fake_sidecar();
        let paths = blink_lights(&root);
        let sessions = root.join("blink-sessions");
        // A sidecar whose manifest rebinds an input: the fake script is
        // wrapped so that its stdout names a different Light.
        let wrapper = root.join("rebinding-sidecar");
        std::fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\n\"{}\" \"$@\" | sed 's/22-57-01/22-57-02/'\n",
                script.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&wrapper).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&wrapper, permissions).unwrap();
        let error = blink_measure_with(
            EngineExecutable { path: wrapper },
            BlinkMeasureRequest {
                paths,
                master_flats: vec![],
                workers: None,
            },
            &sessions,
        )
        .unwrap_err();
        assert!(error.contains("invalid contract"), "{error}");
        assert_eq!(std::fs::read_dir(&sessions).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(root);
    }

    /// The real engine on real Lights: `OAF_TEST_ENGINE` names an
    /// `ultra-fast-wbpp` executable, `OAF_TEST_BLINK_LIGHTS` a text file with
    /// one Light path per line.  Run with `--ignored --nocapture` to see the
    /// session summary.
    #[test]
    #[ignore = "requires OAF_TEST_ENGINE and OAF_TEST_BLINK_LIGHTS pointing to a real engine and Lights"]
    fn real_engine_blink_measure_session_is_accepted() {
        let engine = PathBuf::from(std::env::var("OAF_TEST_ENGINE").expect("engine path"));
        let paths = std::fs::read_to_string(
            std::env::var("OAF_TEST_BLINK_LIGHTS").expect("light list path"),
        )
        .expect("light list")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
        let sessions = std::env::temp_dir()
            .join(new_identifier("real-blink-sessions").expect("temporary identifier"));
        let started = std::time::Instant::now();
        let manifest = blink_measure_with(
            EngineExecutable { path: engine },
            BlinkMeasureRequest {
                paths: paths.clone(),
                master_flats: vec![],
                workers: None,
            },
            &sessions,
        )
        .expect("real blink-measure session");
        let elapsed = started.elapsed();
        assert_eq!(manifest.frames.len(), paths.len());
        let inline = manifest
            .frames
            .iter()
            .filter(|frame| frame.previews.filmstrip_data_url.is_some())
            .count();
        for frame in &manifest.frames {
            for relative in [&frame.previews.filmstrip, &frame.previews.zoom]
                .into_iter()
                .flatten()
            {
                crate::project::load_blink_preview_with(
                    &sessions,
                    &manifest.session_directory,
                    relative,
                )
                .expect("preview loads on demand");
            }
        }
        eprintln!(
            "blink-measure: {} frames in {:.1?}, {} channels, counts {:?}, {inline} inline filmstrip previews, manifest {}",
            manifest.frames.len(),
            elapsed,
            manifest.channels.len(),
            manifest.counts,
            manifest.manifest_sha256.as_deref().unwrap_or("-"),
        );
        for channel in &manifest.channels {
            let reference = manifest
                .frames
                .iter()
                .find(|frame| frame.index == channel.reference.index)
                .expect("validated reference");
            eprintln!(
                "  {} {}: {} frames, reference {}",
                channel.target,
                channel.filter,
                channel.frame_count,
                reference
                    .extra
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(&reference.path)
            );
        }
        for frame in &manifest.frames {
            eprintln!(
                "  {:>4} {:<5} {} {:?}",
                frame.index,
                frame.default_decision,
                frame
                    .extra
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(&frame.path),
                frame
                    .flags
                    .iter()
                    .map(|flag| format!("{}:{}", flag.code, flag.severity))
                    .collect::<Vec<_>>()
            );
        }
        let _ = std::fs::remove_dir_all(sessions);
    }

    /// A schema-exact manifest over `lights` with real preview files, for the
    /// validation cases below.
    fn sample_blink_manifest(session: &Path, lights: &[PathBuf]) -> serde_json::Value {
        std::fs::create_dir_all(session.join("filmstrip")).unwrap();
        std::fs::create_dir_all(session.join("zoom")).unwrap();
        let jpeg: Vec<u8> = [0xff, 0xd8, 0xff, 0xe0]
            .into_iter()
            .chain([7_u8; 64])
            .collect();
        let png: Vec<u8> = crate::project::PNG_SIGNATURE
            .iter()
            .copied()
            .chain([1_u8; 32])
            .collect();
        let frames = lights
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let filmstrip = format!("filmstrip/{index:04}-L.jpg");
                let zoom = format!("zoom/{index:04}-L.png");
                std::fs::write(session.join(&filmstrip), &jpeg).unwrap();
                std::fs::write(session.join(&zoom), &png).unwrap();
                let excluded = index == 1;
                serde_json::json!({
                    "index": index, "channelId": "group-l", "filter": "L", "night": "2026-08-17",
                    "path": path, "name": path.file_name().unwrap().to_str().unwrap(),
                    "sourceSha256": format!("sha256:{}", format!("{index}").repeat(64)),
                    "reference": index == 0, "defaultDecision": if excluded { "DROP" } else { "KEEP" },
                    "flags": if excluded {
                        serde_json::json!([{"code": "BLINK_SKY_BRIGHT", "severity": "EXCLUDE", "value": 2.4, "threshold": 1.6}])
                    } else if index == 2 {
                        serde_json::json!([{"code": "BLINK_EXTINCTION", "severity": "ATTENTION", "value": 0.6, "threshold": 0.5}])
                    } else {
                        serde_json::json!([])
                    },
                    "previews": {"filmstrip": filmstrip, "zoom": zoom, "coverage": 1.0},
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "schemaVersion": 1, "kind": "blink-manifest-v1", "sessionId": "session-1",
            "sessionDirectory": session, "inventorySha256": format!("sha256:{}", "a".repeat(64)),
            "gatePolicyDigest": format!("sha256:{}", "b".repeat(64)),
            "flagsPolicyDigest": format!("sha256:{}", "c".repeat(64)),
            "counts": {"frames": lights.len(), "exclude": 1, "attention": 1, "clean": lights.len() - 2},
            "channels": [{"channelId": "group-l", "target": "NGC 6822", "filter": "L", "frameCount": lights.len(),
                "reference": {"index": 0, "sourceSha256": format!("sha256:{}", "0".repeat(64)), "rule": "psf-signal-weight-proxy-v1"},
                "nights": []}],
            "frames": frames,
        })
    }

    #[test]
    fn blink_manifest_validation_rejects_rebound_inputs_and_inconsistent_records() {
        let root = std::env::temp_dir()
            .join(new_identifier("blink-manifest-validation").expect("temporary identifier"));
        let session = root.join("session");
        std::fs::create_dir_all(&session).unwrap();
        let session = session.canonicalize().unwrap();
        let lights = (0..3)
            .map(|index| {
                let path = root.join(format!("light {index} 盾牌座.fits"));
                std::fs::write(&path, format!("light {index}")).unwrap();
                path.canonicalize().unwrap()
            })
            .collect::<Vec<_>>();
        let expected = lights.iter().cloned().collect::<BTreeSet<_>>();
        let valid = sample_blink_manifest(&session, &lights);
        let check = |value: serde_json::Value| -> Result<(), String> {
            let manifest: BlinkManifest =
                serde_json::from_value(value).map_err(|e| e.to_string())?;
            validate_blink_manifest(&manifest, &expected, &session)
        };
        check(valid.clone()).expect("valid manifest");
        let mutated = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut value = valid.clone();
            edit(&mut value);
            check(value)
        };
        type Edit = fn(&mut serde_json::Value);
        let cases: &[(&str, Edit)] = &[
            ("kind", |v| v["kind"] = "quality-manifest".into()),
            ("schema", |v| v["schemaVersion"] = 2.into()),
            ("session", |v| {
                v["sessionDirectory"] = v["sessionDirectory"]
                    .as_str()
                    .unwrap()
                    .trim_end_matches("session")
                    .into()
            }),
            ("digest case", |v| {
                v["flagsPolicyDigest"] = format!("sha256:{}", "C".repeat(64)).into()
            }),
            ("rebound path", |v| {
                v["frames"][2]["path"] = v["frames"][0]["path"].clone()
            }),
            ("dropped frame", |v| {
                v["frames"].as_array_mut().unwrap().pop();
                v["counts"]["frames"] = 2.into();
                v["counts"]["clean"] = 0.into();
                v["channels"][0]["frameCount"] = 2.into();
            }),
            ("duplicate index", |v| v["frames"][2]["index"] = 0.into()),
            ("frame digest", |v| {
                v["frames"][1]["sourceSha256"] = "sha256:short".into()
            }),
            ("unknown channel", |v| {
                v["frames"][1]["channelId"] = "group-r".into()
            }),
            ("severity", |v| {
                v["frames"][1]["flags"][0]["severity"] = "WARN".into()
            }),
            ("flag code", |v| {
                v["frames"][1]["flags"][0]["code"] = "sky bright".into()
            }),
            ("drop without exclude", |v| {
                v["frames"][0]["defaultDecision"] = "DROP".into()
            }),
            ("keep with exclude", |v| {
                v["frames"][1]["defaultDecision"] = "KEEP".into()
            }),
            ("decision enum", |v| {
                v["frames"][0]["defaultDecision"] = "MAYBE".into()
            }),
            ("counts", |v| {
                v["counts"]["exclude"] = 2.into();
                v["counts"]["clean"] = 0.into();
            }),
            ("escaping preview", |v| {
                v["frames"][0]["previews"]["zoom"] = "../session/zoom/0000-L.png".into()
            }),
            ("absolute preview", |v| {
                v["frames"][0]["previews"]["filmstrip"] =
                    v["sessionDirectory"].as_str().unwrap().to_owned().into()
            }),
            ("missing preview", |v| {
                v["frames"][0]["previews"]["filmstrip"] = "filmstrip/9999-L.jpg".into()
            }),
            ("preview extension", |v| {
                v["frames"][0]["previews"]["filmstrip"] = "filmstrip/0000-L.txt".into()
            }),
            ("reference index", |v| {
                v["channels"][0]["reference"]["index"] = 2.into()
            }),
            ("reference digest", |v| {
                v["channels"][0]["reference"]["sourceSha256"] =
                    format!("sha256:{}", "9".repeat(64)).into()
            }),
            ("two references", |v| {
                v["frames"][2]["reference"] = true.into()
            }),
            ("channel count", |v| {
                v["channels"][0]["frameCount"] = 2.into()
            }),
            ("duplicate channel", |v| {
                let channel = v["channels"][0].clone();
                v["channels"].as_array_mut().unwrap().push(channel);
            }),
        ];
        for &(name, edit) in cases {
            assert!(mutated(&edit).is_err(), "{name} must be rejected");
        }
        // Null previews are allowed (a frame the renderer skipped is shown
        // without an image); unknown fields pass through.
        mutated(&|v| {
            v["frames"][0]["previews"]["filmstrip"] = serde_json::Value::Null;
            v["frames"][0]["previews"]["zoom"] = serde_json::Value::Null;
            v["frames"][0]["metrics"] = serde_json::json!({"sky": 986.0});
        })
        .expect("null previews and extra fields");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn blink_filmstrip_previews_respect_the_transport_budget() {
        let root = std::env::temp_dir()
            .join(new_identifier("blink-preview-budget").expect("temporary identifier"));
        let session = root.join("session");
        std::fs::create_dir_all(&session).unwrap();
        let session = session.canonicalize().unwrap();
        let lights = (0..3)
            .map(|index| {
                let path = root.join(format!("light{index}.fits"));
                std::fs::write(&path, b"light").unwrap();
                path.canonicalize().unwrap()
            })
            .collect::<Vec<_>>();
        let mut manifest: BlinkManifest =
            serde_json::from_value(sample_blink_manifest(&session, &lights)).unwrap();
        // The second filmstrip is a PNG under a .jpg name: not carried.
        std::fs::write(
            session.join("filmstrip/0001-L.jpg"),
            crate::project::PNG_SIGNATURE,
        )
        .unwrap();
        let one_preview = "data:image/jpeg;base64,".len() + 68_usize.div_ceil(3) * 4;
        attach_blink_previews(&mut manifest, &session, one_preview * 2 - 1);
        let urls = manifest
            .frames
            .iter()
            .map(|frame| frame.previews.filmstrip_data_url.is_some())
            .collect::<Vec<_>>();
        assert_eq!(urls, vec![true, false, false]);
        assert_eq!(
            manifest.frames[0]
                .previews
                .filmstrip_data_url
                .as_deref()
                .unwrap()
                .len(),
            one_preview
        );
        let mut generous: BlinkManifest =
            serde_json::from_value(sample_blink_manifest(&session, &lights)).unwrap();
        attach_blink_previews(&mut generous, &session, MAX_BLINK_TRANSPORT_BYTES);
        assert!(generous
            .frames
            .iter()
            .all(|frame| frame.previews.filmstrip_data_url.is_some()));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn fake_sidecar_error_is_reported_without_parsing_stderr_as_protocol() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir()
            .join(new_identifier("openastroflow-error-sidecar").expect("temporary identifier"));
        std::fs::create_dir(&root).expect("create fake sidecar directory");
        let script = root.join("error-sidecar");
        std::fs::write(
            &script,
            "#!/bin/sh\necho 'diagnostic only, not NDJSON' >&2\nexit 7\n",
        )
        .expect("write error sidecar");
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).unwrap();
        let error = inspect_paths_with(
            EngineExecutable { path: script },
            InspectRequest {
                paths: vec!["/tmp/input".to_owned()],
                role_hint: None,
            },
        )
        .expect_err("failing sidecar must fail inventory");
        assert!(error.contains("diagnostic only, not NDJSON"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn fake_sidecar_plan_execute_reaches_ready_gate_and_verified_artifact() {
        use std::sync::mpsc;
        use tauri::Listener;

        let (root, script) = fake_sidecar();
        let output_parent = root.join("Unicode 输出父目录");
        std::fs::create_dir(&output_parent).expect("output parent");
        let executable = EngineExecutable { path: script };
        let probe = probe_runtime_with(executable).expect("canonical handshake");
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let (sender, receiver) = mpsc::channel();
        handle.listen(COMPLETE_EVENT, move |event| {
            let _ = sender.send(event.payload().to_owned());
        });
        let registry = Arc::new(PipelineRegistry::default());
        let receipt = start_pipeline_with_probe(
            handle,
            registry,
            RunRequest {
                sources: vec![
                    RunSource {
                        role: "LIGHT".to_owned(),
                        paths: vec!["/输入/盾牌座_light.fit".to_owned()],
                    },
                    RunSource {
                        role: "FLAT".to_owned(),
                        paths: vec!["/输入/校准_flat.fit".to_owned()],
                    },
                    RunSource {
                        role: "BIAS".to_owned(),
                        paths: vec!["/输入/校准_bias.fit".to_owned()],
                    },
                ],
                recipe_id: "balanced".to_owned(),
                output_parent_directory: output_parent.to_string_lossy().into_owned(),
            },
            probe,
        )
        .expect("start fake canonical worker");
        assert_eq!(receipt.execution_mode, "native");
        let payload = receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("ready completion event");
        let value: serde_json::Value = serde_json::from_str(&payload).expect("completion JSON");
        assert_eq!(value["gate"]["decision"], "ready");
        assert_eq!(
            value["artifacts"][0]["receipt"]["astrometry"]["matchedStars"],
            73
        );
        assert!(Path::new(value["artifacts"][0]["path"].as_str().unwrap()).is_file());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cancellation_terminates_the_worker_process_tree() {
        let mut command = platform::test_support::sleeping_command();
        let child = ManagedChild::spawn(&mut command).expect("spawn cancellable process");
        let id = "cancel-test".to_owned();
        let registry = PipelineRegistry::default();
        registry
            .jobs
            .lock()
            .unwrap()
            .insert(id.clone(), Arc::new(Mutex::new(child)));
        cancel_pipeline(&registry, &id).expect("kill worker process group");
        let status = registry.jobs.lock().unwrap()[&id]
            .lock()
            .unwrap()
            .wait()
            .expect("wait cancelled process");
        assert!(!status.success());
    }
}
