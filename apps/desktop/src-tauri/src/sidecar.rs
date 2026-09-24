use std::collections::{BTreeMap, BTreeSet, HashSet};

mod bundle;
pub(crate) use bundle::discover_engine;
#[cfg(test)]
use bundle::*;
mod inspection;
use inspection::*;
pub(crate) use inspection::{hash_sources, inspect_calibration, inspect_paths, inspect_quality};
mod blink;
use blink::*;
pub(crate) use blink::{blink_measure, blink_sessions_root};

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager, Runtime};

use crate::platform::{self, ManagedChild};

const MAX_DIAGNOSTIC_BYTES: usize = 8 * 1024;

static IDENTIFIER_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub(crate) struct EngineExecutable {
    pub(crate) path: PathBuf,
}

impl EngineExecutable {
    pub(crate) fn command(&self, subcommand: &str) -> Command {
        let mut command = Command::new(&self.path);
        command.arg(subcommand);
        // The engine's stdio carries JSON with user paths and captions.  A
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

/// Lines of an engine process's diagnostic stream, decoded leniently.
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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct BlinkMasterDark {
    pub path: String,
    #[serde(default)]
    pub exposure_seconds: Option<f64>,
}

/// The webview's "Blink & select" request: the Lights to measure, and
/// optionally preview calibration masters and a worker count for the engine.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct BlinkMeasureRequest {
    pub paths: Vec<String>,
    #[serde(default)]
    pub master_flats: Vec<BlinkMasterFlat>,
    #[serde(default)]
    pub master_darks: Vec<BlinkMasterDark>,
    #[serde(default)]
    pub master_bias: Option<String>,
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

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkDiagnosticPreviews {
    pub field: Option<String>,
    pub background: Option<String>,
    pub native_signal: Option<String>,
    pub native_shape: Option<String>,
}

/// The rendered previews of a frame, relative to the session directory.
/// `filmstrip_data_url` is added by the desktop for the frames within the
/// transport budget; the rest are fetched with `load_blink_preview`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlinkPreviews {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<BlinkDiagnosticPreviews>,
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

/// The part of the engine's `doctor --json` self-report the desktop needs.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DoctorReport {
    engine_version: String,
    tuning: DoctorTuning,
    status: DoctorStatus,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DoctorTuning {
    profile_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DoctorStatus {
    pixel_execution_ready: bool,
    solver_ready: bool,
    drizzle_ready: bool,
    metal_ready: bool,
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

/// Windows is release-validated on x86-64 only: the CPU kernels, the engine
/// runtime and the retained E2E evidence all target that architecture.  Any
/// other Windows architecture keeps the shell in its interface-only state.
fn platform_scientific_release_validated(host: &platform::PlatformProfile) -> bool {
    host.platform != "windows" || host.architecture == "x86_64"
}

fn platform_unavailable_reason(host: &platform::PlatformProfile) -> String {
    format!(
        "Windows on {} is not supported by this release; the validated Windows build is x86-64 only (ARM64 has no engine runtime or E2E acceptance)",
        host.architecture
    )
}

fn read_doctor_report(executable: &EngineExecutable) -> Result<DoctorReport, String> {
    let mut command = executable.command("doctor");
    command.arg("--json").stdin(Stdio::null());
    let stdout = command_output(command, "doctor")?;
    serde_json::from_slice(&stdout)
        .map_err(|error| format!("the engine's doctor report is invalid: {error}"))
}

fn optimization_tier(
    report: &DoctorReport,
    host: &platform::PlatformProfile,
    release_validated: bool,
) -> &'static str {
    match (host.platform, host.architecture) {
        ("windows", _) if release_validated => "WINDOWS_X64",
        ("macos", "aarch64")
            if report.status.metal_ready && report.tuning.profile_id == "apple-m3-pro-tuned-v1" =>
        {
            "M3_PRO_TUNED"
        }
        ("macos", "aarch64") => "APPLE_SILICON",
        _ => "PORTABLE",
    }
}

fn capabilities_from_report(
    report: &DoctorReport,
    host: &platform::PlatformProfile,
) -> RuntimeCapabilities {
    // Only architectures with retained scientific E2E evidence may present
    // themselves as a product-ready runtime.
    let release_validated = platform_scientific_release_validated(host);
    let status = &report.status;
    let available = status.pixel_execution_ready && status.solver_ready && release_validated;
    let unavailable_reason = (!available).then(|| {
        if !release_validated {
            platform_unavailable_reason(host)
        } else if !status.pixel_execution_ready {
            "the engine started, but its pixel executors failed their self-check".to_owned()
        } else {
            "the engine started, but no offline plate solver passed its self-check".to_owned()
        }
    });
    let metal = host.platform == "macos" && status.metal_ready;
    RuntimeCapabilities {
        platform: host.platform,
        chip: host.chip.clone(),
        cpu_backend: if status.pixel_execution_ready {
            "Native CPU execution".to_owned()
        } else {
            "CPU backend unavailable".to_owned()
        },
        gpu_backend: if metal {
            "Metal execution".to_owned()
        } else if host.platform == "windows" {
            host.gpu_backend.to_owned()
        } else {
            "Portable CPU only".to_owned()
        },
        optimization_tier: optimization_tier(report, host, release_validated),
        available,
        drizzle_available: status.drizzle_ready,
        solver_available: status.solver_ready,
        runtime_version: Some(report.engine_version.clone()),
        unavailable_reason,
    }
}

pub(crate) fn get_capabilities<R: Runtime>(app: &AppHandle<R>) -> RuntimeCapabilities {
    let host = platform::detect();
    match discover_engine(app).and_then(|executable| read_doctor_report(&executable)) {
        Ok(report) => capabilities_from_report(&report, &host),
        Err(error) => RuntimeCapabilities {
            platform: host.platform,
            chip: host.chip,
            cpu_backend: host.cpu_backend.to_owned(),
            gpu_backend: host.gpu_backend.to_owned(),
            optimization_tier: host.optimization_tier,
            available: false,
            drizzle_available: false,
            solver_available: false,
            runtime_version: None,
            unavailable_reason: Some(error),
        },
    }
}

/// Lowercase hexadecimal SHA-256 of a file's bytes.
pub(crate) fn sha256_file(path: &Path) -> Result<String, String> {
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

#[cfg(test)]
mod tests;
