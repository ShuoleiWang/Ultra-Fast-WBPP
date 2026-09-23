//! blink for the desktop controller.

use super::*;

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
pub(super) fn canonical_light_paths(
    paths: &[String],
    operation: &str,
) -> Result<BTreeSet<PathBuf>, String> {
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
/// returns its path. `workers` and calibration masters travel only when the
/// webview supplied them; the engine's hardware default applies otherwise.
pub(super) fn create_private_blink_request(
    lights: &BTreeSet<PathBuf>,
    session_directory: &Path,
    master_flats: &[(String, PathBuf)],
    master_darks: &[BlinkMasterDark],
    master_bias: Option<&str>,
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
            "displayAlgorithm": "blink-complementary-display-v2",
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
    if !master_darks.is_empty() {
        payload["masterDarks"] =
            serde_json::to_value(master_darks).map_err(|error| error.to_string())?;
    }
    if let Some(path) = master_bias {
        payload["masterBias"] = serde_json::json!(path);
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
pub(super) fn prune_blink_sessions(root: &Path, keep: usize) {
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
pub(super) fn new_blink_session_directory(
    root: &Path,
    lights: &BTreeSet<PathBuf>,
) -> Result<PathBuf, String> {
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
pub(super) fn blink_measure_with(
    executable: EngineExecutable,
    request: BlinkMeasureRequest,
    sessions_root: &Path,
) -> Result<BlinkManifest, String> {
    let lights = canonical_light_paths(&request.paths, "blink measurement")?;
    if request.master_darks.len() > 64 {
        return Err("blink measurement accepts at most 64 master darks".to_owned());
    }
    for dark in &request.master_darks {
        if !Path::new(&dark.path).is_file()
            || dark
                .exposure_seconds
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err("blink master dark needs a regular file and positive exposure".to_owned());
        }
    }
    if request
        .master_bias
        .as_ref()
        .is_some_and(|path| !Path::new(path).is_file())
    {
        return Err("blink master bias must be a regular file".to_owned());
    }
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
    let request_path = create_private_blink_request(
        &lights,
        &session_directory,
        &master_flats,
        &request.master_darks,
        request.master_bias.as_deref(),
        request.workers,
    )?;
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
pub(super) fn validate_blink_manifest(
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
        if let Some(diagnostic) = &frame.previews.diagnostic {
            for relative in [
                &diagnostic.field,
                &diagnostic.background,
                &diagnostic.native_signal,
                &diagnostic.native_shape,
            ]
            .into_iter()
            .flatten()
            {
                crate::project::resolve_blink_preview(session_directory, relative)?;
            }
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
pub(super) fn blink_manifest_file_digest(
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
pub(super) fn attach_blink_previews(
    manifest: &mut BlinkManifest,
    session_directory: &Path,
    budget: usize,
) {
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
