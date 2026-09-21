use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
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

use crate::platform;

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
        platform::configure_child_process(&mut command);
        command
    }
}

const TEXT_BUSY_RETRIES: u32 = 20;
const TEXT_BUSY_RETRY_DELAY: Duration = Duration::from_millis(50);

/// Launches a sidecar command, retrying for about a second while its
/// executable is momentarily "text busy".
///
/// On Unix, a fork elsewhere in this process (another sidecar launch, a
/// test thread) inherits every open descriptor until its own exec; if one of
/// them is a write handle on a just-installed executable, executing that
/// file meanwhile fails with `ETXTBSY` even though the writer already closed
/// it.  rustc, cargo and git retry the same way; any other launch error is
/// returned at once.
pub(crate) fn spawn_sidecar(command: &mut Command) -> std::io::Result<Child> {
    let mut attempt = 0;
    loop {
        match command.spawn() {
            Err(error) if is_text_busy(&error) && attempt < TEXT_BUSY_RETRIES => {
                attempt += 1;
                std::thread::sleep(TEXT_BUSY_RETRY_DELAY);
            }
            result => return result,
        }
    }
}

/// `Command::output` with the same text-busy retry as [`spawn_sidecar`].
pub(crate) fn sidecar_output(command: &mut Command) -> std::io::Result<std::process::Output> {
    let mut attempt = 0;
    loop {
        match command.output() {
            Err(error) if is_text_busy(&error) && attempt < TEXT_BUSY_RETRIES => {
                attempt += 1;
                std::thread::sleep(TEXT_BUSY_RETRY_DELAY);
            }
            result => return result,
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
    jobs: Mutex<HashMap<String, Arc<Mutex<Child>>>>,
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

fn platform_scientific_release_validated(host: &platform::PlatformProfile) -> bool {
    host.platform != "windows"
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
            // Windows currently has compile/interface evidence only.  Do not
            // promote that seam to a product-ready scientific runtime until
            // the Windows release matrix has retained E2E evidence.
            let release_validated = platform_scientific_release_validated(&platform_profile);
            let available = stages_ready && solver_available && release_validated;
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
                unavailable_reason: (!available).then(|| {
                    if !release_validated {
                        "Windows adapter contracts compile and run in CI, but this release has no retained Windows scientific E2E acceptance"
                            .to_owned()
                    } else {
                        "sidecar handshake passed, but one or more required E2E stages are unavailable"
                            .to_owned()
                    }
                }),
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
    if paths.is_empty() || paths.len() > 10_000 {
        return Err("quality inspection requires between 1 and 10000 Light files".to_owned());
    }
    let mut canonical = BTreeSet::new();
    for value in paths {
        let path = Path::new(value)
            .canonicalize()
            .map_err(|error| format!("cannot resolve Light for quality inspection: {error}"))?;
        if !path.is_file() || !canonical.insert(path) {
            return Err("quality inspection accepts unique regular Light files only".to_owned());
        }
    }
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
    child: Arc<Mutex<Child>>,
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

    #[cfg(unix)]
    #[test]
    fn application_shutdown_terminates_every_pipeline_process_tree() {
        let registry = PipelineRegistry::default();
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        platform::configure_child_process(&mut command);
        let child = Arc::new(Mutex::new(command.spawn().expect("pipeline child")));
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
    fn windows_interface_evidence_is_not_scientific_release_acceptance() {
        assert!(!platform_scientific_release_validated(&simulated_host(
            "windows", "x86_64", "x86_64"
        )));
        assert!(platform_scientific_release_validated(&simulated_host(
            "macos",
            "aarch64",
            "Apple M3 Pro"
        )));
    }

    #[cfg(unix)]
    fn fake_sidecar() -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir()
            .join(new_identifier("openastroflow-fake-sidecar").expect("temporary identifier"));
        std::fs::create_dir(&root).expect("create fake sidecar directory");
        let script = root.join("fake-openastroflow-engine");
        let source = r###"#!/usr/bin/env python3
import argparse, hashlib, json, os, pathlib, platform, sys

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

    #[cfg(unix)]
    #[test]
    fn cancellation_terminates_the_worker_process_group() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "sleep 30"]);
        platform::configure_child_process(&mut command);
        let child = command.spawn().expect("spawn cancellable process");
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
