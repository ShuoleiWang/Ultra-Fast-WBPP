//! Explicit, user-authorized offline solver catalog management.
//!
//! Listing and diagnosis are read-only. Downloads are started only after the
//! UI submits the exact checked provider acceptance identifier. The provider
//! CLI owns resumable partials and content verification; this host owns process
//! lifetime, progress delivery, and cancellation.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Runtime};

use crate::platform;
use crate::sidecar::{
    command_output, discover_engine, sidecar_output, spawn_sidecar, EngineExecutable,
};

const PROGRESS_EVENT: &str = "openastroflow://catalog-progress";
const COMPLETE_EVENT: &str = "openastroflow://catalog-complete";
const ERROR_EVENT: &str = "openastroflow://catalog-error";
const MAX_CATALOG_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_DIAGNOSTIC_BYTES: usize = 8 * 1024;

#[derive(Default)]
pub(crate) struct CatalogRegistry {
    jobs: Mutex<HashMap<String, Arc<Mutex<Child>>>>,
    shutting_down: AtomicBool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CatalogInstallRequest {
    pub catalog_id: String,
    pub accepted_terms_id: String,
    pub field_of_view_degrees: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogJobReceipt {
    pub job_id: String,
    pub accepted: bool,
    pub catalog_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogProgressEvent {
    job_id: String,
    catalog_id: String,
    artifact_id: String,
    downloaded_bytes: u64,
    size_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogCompleteEvent {
    job_id: String,
    catalog_id: String,
    install: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogErrorEvent {
    job_id: String,
    catalog_id: String,
    code: String,
    message: String,
}

fn checked_identifier(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 160
        || !value.as_bytes()[0].is_ascii_alphanumeric()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("{label} is not a checked identifier"));
    }
    Ok(())
}

fn parse_json(bytes: &[u8], operation: &str) -> Result<serde_json::Value, String> {
    if bytes.len() > MAX_CATALOG_JSON_BYTES {
        return Err(format!("sidecar {operation} response is too large"));
    }
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("sidecar {operation} returned invalid JSON: {error}"))?;
    if value
        .get("schemaVersion")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
    {
        return Err(format!(
            "sidecar {operation} returned an unsupported schema"
        ));
    }
    Ok(value)
}

fn catalog_command_output(
    executable: &EngineExecutable,
    command_name: &str,
    configure: bool,
) -> Result<serde_json::Value, String> {
    let mut command = executable.command("catalog");
    command.arg(command_name);
    if configure {
        command.arg("--configure");
    }
    command.arg("--json");
    if command_name == "doctor" {
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let output = sidecar_output(&mut command)
            .map_err(|error| format!("cannot launch sidecar for catalog doctor: {error}"))?;
        return parse_catalog_doctor_output(output);
    }
    let output = command_output(command, &format!("catalog {command_name}"))?;
    parse_json(&output, &format!("catalog {command_name}"))
}

fn parse_catalog_doctor_output(output: std::process::Output) -> Result<serde_json::Value, String> {
    // The CLI uses exit 3 for a completed diagnosis whose catalogs are not
    // ready. Its JSON is still the authoritative result for the setup UI.
    let not_ready = output.status.code() == Some(3);
    if !output.status.success() && !not_ready {
        return Err(format!(
            "sidecar catalog doctor failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
                .trim()
                .chars()
                .take(MAX_DIAGNOSTIC_BYTES)
                .collect::<String>()
        ));
    }
    let diagnosis = parse_json(&output.stdout, "catalog doctor")?;
    if not_ready && diagnosis.get("ok").and_then(serde_json::Value::as_bool) != Some(false) {
        return Err(
            "sidecar catalog doctor returned exit 3 without an ok=false diagnosis".to_owned(),
        );
    }
    Ok(diagnosis)
}

pub(crate) fn list<R: Runtime>(app: &AppHandle<R>) -> Result<serde_json::Value, String> {
    catalog_command_output(&discover_engine(app)?, "list", false)
}

pub(crate) fn doctor<R: Runtime>(app: &AppHandle<R>) -> Result<serde_json::Value, String> {
    catalog_command_output(&discover_engine(app)?, "doctor", false)
}

pub(crate) fn solver_doctor<R: Runtime>(app: &AppHandle<R>) -> Result<serde_json::Value, String> {
    let executable = discover_engine(app)?;
    let mut command = executable.command("doctor");
    command.arg("--json");
    let output = command_output(command, "solver doctor")?;
    parse_json(&output, "solver doctor")
}

pub(crate) fn verify<R: Runtime>(
    app: &AppHandle<R>,
    catalog_id: &str,
    configure: bool,
) -> Result<serde_json::Value, String> {
    checked_identifier(catalog_id, "catalogId")?;
    let executable = discover_engine(app)?;
    let mut command = executable.command("catalog");
    command.args(["verify", catalog_id]);
    if configure {
        command.arg("--configure");
    }
    let output = command_output(command, "catalog verify")?;
    parse_json(&output, "catalog verify")
}

fn validate_acceptance(
    listing: &serde_json::Value,
    request: &CatalogInstallRequest,
) -> Result<(), String> {
    let catalogs = listing
        .get("catalogs")
        .and_then(serde_json::Value::as_array)
        .ok_or("catalog list has no catalogs array")?;
    let catalog = catalogs
        .iter()
        .find(|item| {
            item.get("catalogId").and_then(serde_json::Value::as_str) == Some(&request.catalog_id)
        })
        .ok_or_else(|| {
            format!(
                "catalog is not present in the checked manifest set: {}",
                request.catalog_id
            )
        })?;
    let terms = catalog
        .get("providerTerms")
        .ok_or("checked catalog has no provider terms")?;
    let expected = terms
        .get("acceptanceId")
        .and_then(serde_json::Value::as_str)
        .ok_or("checked catalog has no acceptance identifier")?;
    if expected != request.accepted_terms_id {
        return Err("provider terms acceptance does not match the checked manifest".to_owned());
    }
    if terms
        .get("requiresExplicitAcceptance")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return Err("managed download requires an explicitly accepted checked manifest".to_owned());
    }
    if catalog
        .get("allowedDownloadOrigins")
        .and_then(serde_json::Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        return Err("this catalog is external-only and cannot be downloaded by the app".to_owned());
    }
    Ok(())
}

fn emit_error<R: Runtime>(
    app: &AppHandle<R>,
    job_id: &str,
    catalog_id: &str,
    code: &str,
    message: &str,
) {
    let _ = app.emit(
        ERROR_EVENT,
        CatalogErrorEvent {
            job_id: job_id.to_owned(),
            catalog_id: catalog_id.to_owned(),
            code: code.to_owned(),
            message: message.to_owned(),
        },
    );
}

fn stream_progress<R: Runtime>(
    app: AppHandle<R>,
    job_id: String,
    catalog_id: String,
    stderr: std::process::ChildStderr,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut diagnostics = String::new();
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if line.len() <= 256 * 1024 {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    if value.get("event").and_then(serde_json::Value::as_str)
                        == Some("catalog-download")
                    {
                        let event = CatalogProgressEvent {
                            job_id: job_id.clone(),
                            catalog_id: catalog_id.clone(),
                            artifact_id: value
                                .get("artifactId")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("catalog-artifact")
                                .to_owned(),
                            downloaded_bytes: value
                                .get("downloadedBytes")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0),
                            size_bytes: value
                                .get("sizeBytes")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0),
                        };
                        let _ = app.emit(PROGRESS_EVENT, event);
                        continue;
                    }
                }
            }
            if diagnostics.len() < MAX_DIAGNOSTIC_BYTES {
                diagnostics.push_str(&line);
                diagnostics.push('\n');
            }
        }
        diagnostics
    })
}

pub(crate) fn start_install<R: Runtime>(
    app: AppHandle<R>,
    registry: Arc<CatalogRegistry>,
    request: CatalogInstallRequest,
) -> Result<CatalogJobReceipt, String> {
    if registry.shutting_down.load(Ordering::Acquire) {
        return Err("application shutdown is already in progress".to_owned());
    }
    let executable = discover_engine(&app)?;
    start_install_with(app, registry, request, executable)
}

fn start_install_with<R: Runtime>(
    app: AppHandle<R>,
    registry: Arc<CatalogRegistry>,
    request: CatalogInstallRequest,
    executable: EngineExecutable,
) -> Result<CatalogJobReceipt, String> {
    if registry.shutting_down.load(Ordering::Acquire) {
        return Err("application shutdown is already in progress".to_owned());
    }
    checked_identifier(&request.catalog_id, "catalogId")?;
    checked_identifier(&request.accepted_terms_id, "acceptedTermsId")?;
    if request
        .field_of_view_degrees
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("fieldOfViewDegrees must be finite and positive".to_owned());
    }
    let listing = catalog_command_output(&executable, "list", false)?;
    validate_acceptance(&listing, &request)?;
    let serial = crate::sidecar::new_public_identifier("catalog")?;
    let mut command = executable.command("catalog");
    command
        .args(["install", &request.catalog_id])
        .args(["--accept-provider-terms", &request.accepted_terms_id])
        .arg("--progress-json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(value) = request.field_of_view_degrees {
        command.args(["--field-of-view", &value.to_string()]);
    }
    let mut child = spawn_sidecar(&mut command)
        .map_err(|error| format!("cannot launch catalog installer: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or("catalog installer stdout is unavailable")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("catalog installer stderr is unavailable")?;
    let shared = Arc::new(Mutex::new(child));
    let mut jobs = registry
        .jobs
        .lock()
        .map_err(|_| "catalog registry lock was poisoned".to_owned())?;
    if registry.shutting_down.load(Ordering::Acquire) {
        drop(jobs);
        if let Ok(mut child) = shared.lock() {
            let _ = platform::terminate_process_tree(&mut child);
        }
        return Err("application shutdown started before catalog registration".to_owned());
    }
    jobs.insert(serial.clone(), shared.clone());
    drop(jobs);

    let app_for_thread = app.clone();
    let job_for_thread = serial.clone();
    let catalog_for_thread = request.catalog_id.clone();
    let registry_for_thread = registry.clone();
    std::thread::spawn(move || {
        let diagnostic_thread = stream_progress(
            app_for_thread.clone(),
            job_for_thread.clone(),
            catalog_for_thread.clone(),
            stderr,
        );
        let mut bytes = Vec::new();
        let read_result = stdout
            .take((MAX_CATALOG_JSON_BYTES + 1) as u64)
            .read_to_end(&mut bytes);
        let status = shared.lock().ok().and_then(|mut child| child.wait().ok());
        let diagnostics = diagnostic_thread.join().unwrap_or_default();
        registry_for_thread
            .jobs
            .lock()
            .ok()
            .map(|mut jobs| jobs.remove(&job_for_thread));
        if read_result.is_err() || bytes.len() > MAX_CATALOG_JSON_BYTES {
            emit_error(
                &app_for_thread,
                &job_for_thread,
                &catalog_for_thread,
                "CATALOG_RESPONSE_INVALID",
                "catalog installer response could not be read safely",
            );
            return;
        }
        if status.is_none_or(|status| !status.success()) {
            emit_error(
                &app_for_thread,
                &job_for_thread,
                &catalog_for_thread,
                "CATALOG_INSTALL_FAILED",
                diagnostics
                    .trim()
                    .chars()
                    .take(MAX_DIAGNOSTIC_BYTES)
                    .collect::<String>()
                    .as_str(),
            );
            return;
        }
        match parse_json(&bytes, "catalog install") {
            Ok(install) => {
                let _ = app_for_thread.emit(
                    COMPLETE_EVENT,
                    CatalogCompleteEvent {
                        job_id: job_for_thread,
                        catalog_id: catalog_for_thread,
                        install,
                    },
                );
            }
            Err(error) => emit_error(
                &app_for_thread,
                &job_for_thread,
                &catalog_for_thread,
                "CATALOG_RESPONSE_INVALID",
                &error,
            ),
        }
    });
    Ok(CatalogJobReceipt {
        job_id: serial,
        accepted: true,
        catalog_id: request.catalog_id,
    })
}

pub(crate) fn cancel_install(registry: &CatalogRegistry, job_id: &str) -> Result<(), String> {
    let child = registry
        .jobs
        .lock()
        .map_err(|_| "catalog registry lock was poisoned".to_owned())?
        .get(job_id)
        .cloned()
        .ok_or_else(|| format!("catalog install is not running: {job_id}"))?;
    let mut child = child
        .lock()
        .map_err(|_| "catalog child lock was poisoned".to_owned())?;
    platform::terminate_process_tree(&mut child)
}

pub(crate) fn terminate_all(registry: &CatalogRegistry) -> Result<(), String> {
    registry.shutting_down.store(true, Ordering::Release);
    let children = registry
        .jobs
        .lock()
        .map_err(|_| "catalog registry lock was poisoned".to_owned())?
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
            Err(_) => failures.push("catalog child lock was poisoned".to_owned()),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "one or more catalog process trees could not be terminated: {}",
            failures.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn doctor_output(code: i32, stdout: &[u8], stderr: &[u8]) -> std::process::Output {
        use std::os::unix::process::ExitStatusExt;
        std::process::Output {
            status: std::process::ExitStatus::from_raw(code << 8),
            stdout: stdout.to_vec(),
            stderr: stderr.to_vec(),
        }
    }

    #[cfg(unix)]
    #[test]
    fn catalog_doctor_preserves_completed_not_ready_diagnosis() {
        let diagnosis = serde_json::json!({
            "schemaVersion": 1,
            "ok": false,
            "catalogRoot": "/missing/catalogs",
            "config": {"present": false, "valid": false},
            "installedSetBindingReady": false,
            "message": "No catalog directory exists"
        });
        let output = doctor_output(3, &serde_json::to_vec(&diagnosis).unwrap(), b"");
        assert_eq!(parse_catalog_doctor_output(output).unwrap(), diagnosis);

        let ready = doctor_output(0, br#"{"schemaVersion":1,"ok":true}"#, b"");
        assert_eq!(parse_catalog_doctor_output(ready).unwrap()["ok"], true);
    }

    #[cfg(unix)]
    #[test]
    fn catalog_doctor_keeps_failures_and_invalid_diagnoses_as_errors() {
        let failed = doctor_output(2, br#"{"schemaVersion":1,"ok":false}"#, b"probe failed");
        assert!(parse_catalog_doctor_output(failed)
            .unwrap_err()
            .contains("probe failed"));

        for invalid in [
            &b"not JSON"[..],
            br#"{"schemaVersion":2,"ok":false}"#,
            br#"{"schemaVersion":1,"ok":true}"#,
            br#"{"schemaVersion":1}"#,
        ] {
            assert!(parse_catalog_doctor_output(doctor_output(3, invalid, b"")).is_err());
        }
    }

    #[test]
    fn identifiers_cannot_become_cli_options() {
        assert!(checked_identifier("astrometry-net-4107-4112", "catalogId").is_ok());
        assert!(checked_identifier("--manifest-dir", "catalogId").is_err());
        assert!(checked_identifier("catalog/../../escape", "catalogId").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn application_shutdown_terminates_every_catalog_process_tree() {
        let registry = CatalogRegistry::default();
        let mut command = std::process::Command::new("/bin/sleep");
        command.arg("30");
        platform::configure_child_process(&mut command);
        let child = Arc::new(Mutex::new(command.spawn().expect("catalog child")));
        registry
            .jobs
            .lock()
            .unwrap()
            .insert("catalog-shutdown-test".to_owned(), child.clone());

        terminate_all(&registry).expect("terminate catalog children");
        let status = child.lock().unwrap().wait().expect("reap catalog child");
        assert!(!status.success());
        assert!(registry.shutting_down.load(Ordering::Acquire));
        terminate_all(&registry).expect("shutdown is idempotent");
    }

    #[test]
    fn exact_acceptance_is_required() {
        let listing = serde_json::json!({
            "schemaVersion": 1,
            "catalogs": [{
                "catalogId": "checked-set",
                "allowedDownloadOrigins": ["https://provider.example"],
                "providerTerms": {
                    "acceptanceId": "provider-terms-v1",
                    "requiresExplicitAcceptance": true
                }
            }]
        });
        let mut request = CatalogInstallRequest {
            catalog_id: "checked-set".to_owned(),
            accepted_terms_id: "wrong-version".to_owned(),
            field_of_view_degrees: None,
        };
        assert!(validate_acceptance(&listing, &request).is_err());
        request.accepted_terms_id = "provider-terms-v1".to_owned();
        assert!(validate_acceptance(&listing, &request).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn fake_sidecar_progress_is_streamed_before_completion() {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        use std::sync::mpsc;
        use tauri::Listener;

        let root = std::env::temp_dir().join(
            crate::sidecar::new_public_identifier("catalog-progress-test").expect("temporary id"),
        );
        std::fs::create_dir(&root).expect("create test directory");
        let script = root.join("fake-catalog-sidecar");
        let source = r###"#!/usr/bin/env python3
import json, sys
if sys.argv[1:3] == ["catalog", "list"]:
    print(json.dumps({"schemaVersion":1,"catalogRoot":"/checked/catalog","catalogs":[{"catalogId":"checked-set","allowedDownloadOrigins":["https://provider.example"],"providerTerms":{"acceptanceId":"provider-terms-v1","requiresExplicitAcceptance":True}}]}), flush=True)
elif sys.argv[1:3] == ["catalog", "install"]:
    print(json.dumps({"event":"catalog-download","artifactId":"index-4108.fits","downloadedBytes":25,"sizeBytes":100}), file=sys.stderr, flush=True)
    print(json.dumps({"event":"catalog-download","artifactId":"index-4108.fits","downloadedBytes":100,"sizeBytes":100}), file=sys.stderr, flush=True)
    print(json.dumps({"schemaVersion":1,"catalogId":"checked-set","status":"INSTALLED"}), flush=True)
else:
    raise SystemExit(2)
"###;
        let staged_script = root.join(".fake-catalog-sidecar.partial");
        let mut staged = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o700)
            .open(&staged_script)
            .expect("create staged fake sidecar");
        staged
            .write_all(source.as_bytes())
            .expect("write staged fake sidecar");
        staged.sync_all().expect("sync staged fake sidecar");
        drop(staged);
        std::fs::rename(staged_script, &script).expect("publish fake sidecar");

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
        let receipt = start_install_with(
            handle,
            Arc::new(CatalogRegistry::default()),
            CatalogInstallRequest {
                catalog_id: "checked-set".to_owned(),
                accepted_terms_id: "provider-terms-v1".to_owned(),
                field_of_view_degrees: None,
            },
            EngineExecutable { path: script },
        )
        .expect("start fake catalog install");
        assert!(receipt.accepted);
        let first: serde_json::Value = serde_json::from_str(
            &progress_receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("progress event"),
        )
        .expect("progress JSON");
        assert_eq!(first["artifactId"], "index-4108.fits");
        assert_eq!(first["downloadedBytes"], 25);
        let completion: serde_json::Value = serde_json::from_str(
            &complete_receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("completion event"),
        )
        .expect("completion JSON");
        assert_eq!(completion["catalogId"], "checked-set");
        let _ = std::fs::remove_dir_all(root);
    }
}
