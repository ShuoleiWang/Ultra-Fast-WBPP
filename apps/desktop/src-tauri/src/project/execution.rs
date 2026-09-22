//! execution for the desktop controller.

use super::*;

pub(super) fn stage_id(value: &str) -> Option<String> {
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

pub(super) fn normalize_progress(job_id: &str, event: &serde_json::Value) -> ProgressEvent {
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

/// Forwards the worker's `--progress-json` records from its diagnostic stream
/// and keeps the rest as the failure diagnostics.  Lines are decoded leniently
/// so a solver's stray console byte cannot end progress for the rest of the run.
pub(super) fn stream_progress<R: Runtime, S: Read + Send + 'static>(
    app: AppHandle<R>,
    job_id: String,
    stderr: S,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut diagnostics = String::new();
        for line in LossyLines::new(BufReader::new(stderr)) {
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

pub(super) fn emit_error<R: Runtime>(app: &AppHandle<R>, job_id: &str, code: &str, message: &str) {
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
    validate_blink_review(&request, &crate::sidecar::blink_sessions_root(&app)?)?;
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
pub(super) fn unique_output_directory(
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

pub(super) fn start_with<R: Runtime>(
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
