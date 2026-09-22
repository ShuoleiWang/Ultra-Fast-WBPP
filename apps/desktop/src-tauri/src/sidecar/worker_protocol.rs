//! worker protocol for the desktop controller.

use super::*;

pub(super) fn recipe_cli_id(recipe_id: &str) -> Result<&'static str, String> {
    match recipe_id {
        "balanced" => Ok("balanced"),
        "drizzle-2x" => Ok("drizzle-2x"),
        other => Err(format!("unsupported recipe: {other}")),
    }
}

pub(super) fn build_plan_envelope(
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

pub(super) fn unique_input_paths(sources: &[RunSource]) -> Result<Vec<String>, String> {
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

pub(super) fn start_pipeline_with_probe<R: Runtime>(
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

pub(super) fn stream_stderr<R: Runtime>(app: AppHandle<R>, job_id: String, stderr: ChildStderr) {
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
pub(super) fn stream_worker<R: Runtime>(
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

pub(super) fn verify_artifacts(
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

pub(super) fn sha256_file(path: &Path) -> Result<String, String> {
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

pub(super) fn emit_error<R: Runtime>(
    app: &AppHandle<R>,
    job_id: &str,
    code: &str,
    message: &str,
    retryable: bool,
) {
    emit_error_with_details(app, job_id, code, message, retryable, BTreeMap::new());
}

pub(super) fn emit_error_with_details<R: Runtime>(
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
