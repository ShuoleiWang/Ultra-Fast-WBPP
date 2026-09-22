//! inspection for the desktop controller.

use super::*;

pub(crate) fn inspect_paths<R: Runtime>(
    app: &AppHandle<R>,
    request: InspectRequest,
) -> Result<InspectResponse, String> {
    if request.paths.is_empty() {
        return Err("at least one file or directory is required".to_owned());
    }
    inspect_paths_with(discover_engine(app)?, request)
}

pub(super) fn inspect_paths_with(
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

pub(super) fn create_private_quality_request(
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

pub(super) fn inspect_calibration_with(
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

pub(super) fn validate_calibration_inspection(
    inspection: &CalibrationInspection,
) -> Result<(), String> {
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

pub(super) fn inspect_quality_with(
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

pub(super) const BLINK_MANIFEST_KIND: &str = "blink-manifest-v1";
/// The filmstrip preview is the 1/8-scale grayscale JPEG (60–120 KB,
/// noise-limited): 100 frames are about 12 MB as data URLs.  A preview over
/// either bound is left to the on-demand `load_blink_preview` path.
pub(super) const MAX_BLINK_FILMSTRIP_BYTES: u64 = 200 * 1024;
pub(super) const MAX_BLINK_TRANSPORT_BYTES: usize = 32 * 1024 * 1024;
/// Sessions kept under the blink-sessions root, the new one included; the
/// previews of a 100-frame session take about 150 MB.
pub(super) const MAX_BLINK_SESSIONS: usize = 3;
pub(super) const MAX_BLINK_MASTER_FLATS: usize = 16;
pub(super) const MAX_BLINK_WORKERS: usize = 64;
/// A 10 000-frame manifest with every metric is about 20 MB.
pub(super) const MAX_BLINK_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
