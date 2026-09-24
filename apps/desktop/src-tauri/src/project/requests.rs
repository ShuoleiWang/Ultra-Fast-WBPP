//! requests for the desktop controller.

use super::*;

pub(super) fn checked_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub(super) fn artifact_sha256_hex(value: &str) -> Result<&str, String> {
    // Python artifact identities are tagged; GUI typed receipts use bare hex.
    // Accept exactly these two established forms, without relaxing the digest.
    let hex = value.strip_prefix("sha256:").unwrap_or(value);
    if checked_sha256(hex) {
        Ok(hex)
    } else {
        Err("GUI artifact SHA-256 identity is malformed".to_owned())
    }
}

pub(super) fn checked_source_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(checked_sha256)
}

pub(super) fn checked_role(value: &str) -> bool {
    matches!(
        value,
        "LIGHT" | "FLAT" | "DARK" | "BIAS" | "MASTER_FLAT" | "MASTER_DARK" | "MASTER_BIAS"
    )
}

pub(super) fn checked_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

pub(super) fn canonical_output_parent(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value)
        .canonicalize()
        .map_err(|error| format!("cannot resolve output parent: {error}"))?;
    if !path.is_dir() {
        return Err("output parent must be an existing directory".to_owned());
    }
    Ok(path)
}

pub(super) fn validate_master_override(value: &UiMasterOverride) -> Result<(), String> {
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

/// A flag code as the engine writes it: `BLINK_SKY_BRIGHT`, `GATE_…`.
pub(crate) fn checked_flag_code(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.as_bytes()[0].is_ascii_uppercase()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
}

/// The `selection-v1` rules (§B.4.3): schema and policy names, the enums,
/// unique lowercase content digests, bounded size.  The engine repeats the
/// check; this keeps a malformed selection an error next to the Start
/// button instead of a failed run.
pub(super) fn validate_selection(selection: &UiSelection) -> Result<(), String> {
    let invalid = |detail: &str| Err(format!("SELECTION_INVALID: {detail}"));
    if selection.schema_version != 1 || selection.kind != SELECTION_KIND {
        return invalid("selection must be schemaVersion 1 of kind ultra-fast-wbpp-selection");
    }
    if selection.policy != SELECTION_POLICY {
        return invalid("selection policy must be explicit-v1");
    }
    if !matches!(selection.undecided.as_str(), "ERROR" | "DROP" | "KEEP") {
        return invalid("selection undecided must be ERROR, DROP or KEEP");
    }
    if selection.decisions.is_empty() || selection.decisions.len() > MAX_SELECTION_DECISIONS {
        return invalid("selection must contain between 1 and 10000 decisions");
    }
    let mut digests = HashSet::new();
    for item in &selection.decisions {
        if !checked_source_sha256(&item.source_sha256)
            || !digests.insert(item.source_sha256.as_str())
        {
            return invalid("selection decisions must carry unique lowercase sha256: digests");
        }
        if !matches!(item.decision.as_str(), "KEEP" | "DROP")
            || item
                .default_decision
                .as_deref()
                .is_some_and(|value| !matches!(value, "KEEP" | "DROP"))
        {
            return invalid("selection decisions must be KEEP or DROP");
        }
        if item.flags.as_ref().is_some_and(|flags| {
            flags.len() > MAX_SELECTION_FLAGS || flags.iter().any(|code| !checked_flag_code(code))
        }) {
            return invalid("selection decision flags must be engine flag codes");
        }
        if item
            .note
            .as_deref()
            .is_some_and(|note| note.chars().count() > MAX_SELECTION_NOTE_CHARS)
        {
            return invalid("selection decision note is too long");
        }
    }
    if let Some(origin) = &selection.origin {
        if origin
            .blink_manifest_sha256
            .as_deref()
            .is_some_and(|value| !checked_source_sha256(value))
            || origin
                .flags_policy_digest
                .as_deref()
                .is_some_and(|value| !checked_source_sha256(value))
        {
            return invalid("selection origin digests must be lowercase sha256: digests");
        }
        if [origin.session_id.as_deref(), origin.created_at.as_deref()]
            .into_iter()
            .flatten()
            .any(|value| value.is_empty() || value.chars().count() > MAX_SELECTION_ORIGIN_CHARS)
        {
            return invalid("selection origin fields must be short non-empty strings");
        }
    }
    Ok(())
}

pub(super) fn project_request_json(
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
    validate_drizzle_options(&request.recipe)?;
    for item in &request.master_metadata_overrides {
        validate_master_override(item)?;
    }
    let mut raw_override_digests = HashSet::new();
    for item in &request.raw_frame_metadata_overrides {
        // A confirmation names the sensor: mono (NONE) or one of the Bayer
        // patterns the engine processes as one-shot colour.
        let pattern = item.cfa_pattern.trim().to_ascii_uppercase();
        if !checked_source_sha256(&item.source_sha256)
            || !matches!(pattern.as_str(), "NONE" | "RGGB" | "BGGR" | "GRBG" | "GBRG")
            || !raw_override_digests.insert(item.source_sha256.as_str())
        {
            return Err(
                "rawFrameMetadataOverrides must contain unique SHA-bound NONE or Bayer pattern declarations"
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
    if let Some(selection) = &request.selection {
        validate_selection(selection)?;
        // The explicit selection replaces the legacy gate's admission; REVIEW
        // approvals belong to that gate and cannot be combined with it.
        if !request.review_selections.is_empty() {
            return Err(
                "SELECTION_POLICY_CONFLICT: a blink selection and legacy REVIEW approvals cannot be sent together"
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
    let mut payload = serde_json::json!({
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
            // `auto` lets the engine take whichever installed backend is
            // science-ready on this platform (solve-field, or ASTAP verified
            // against the managed indexes); the gates below stay unchanged.
            "solver": { "policy": "REQUIRED", "backend": "auto", "searchRadiusDegrees": 15.0 },
            "drizzle": {
                "enabled": request.recipe.drizzle_enabled, "backend": "auto",
                "scale": request.recipe.drizzle_scale, "dropShrink": request.recipe.drizzle_drop_shrink,
                "kernel": request.recipe.drizzle_kernel, "cfaDrizzle": false,
            },
            "outputFormat": "FITS", "overwrite": false, "reviewApprovals": [],
            "rawFrameMetadataOverrides": request.raw_frame_metadata_overrides,
        },
        "solverHints": {},
        "execution": {},
        "reviewSelections": request.review_selections,
    });
    // Top-level, and only when the blink view produced one: the engine's
    // request loader treats the key itself as the switch to `explicit-v1`.
    if let Some(selection) = &request.selection {
        payload["selection"] =
            serde_json::to_value(selection).map_err(|error| error.to_string())?;
    }
    Ok(payload)
}

pub(super) fn create_private_request(value: &serde_json::Value) -> Result<PathBuf, String> {
    let path = std::env::temp_dir().join(format!(
        "{}.json",
        new_public_identifier("ultra-fast-wbpp-project-request")?
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

pub(super) use crate::sidecar::sha256_file;

pub(super) fn relative_artifact(root: &Path, value: &str) -> Result<(String, PathBuf), String> {
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

pub(super) fn value_string<'a>(
    value: &'a serde_json::Value,
    field: &str,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("published artifact has no {field}"))
}

/// Desktop processing requires a completed review of the exact measured set.
/// CLI selection files remain independent of this GUI lifecycle contract.
pub(super) fn validate_blink_review(
    request: &ProjectRunRequest,
    sessions_root: &Path,
) -> Result<(), String> {
    let proof = request
        .blink_review
        .as_ref()
        .ok_or("BLINK_REVIEW_REQUIRED: review every channel before processing")?;
    let selection = request
        .selection
        .as_ref()
        .ok_or("BLINK_REVIEW_REQUIRED: a manual selection is required")?;
    validate_selection(selection)?;
    if !request.review_selections.is_empty() || !checked_source_sha256(&proof.manifest_sha256) {
        return Err("BLINK_REVIEW_INVALID: invalid review identity".to_owned());
    }
    let session = checked_blink_session_directory(sessions_root, &proof.session_directory)?;
    let path = session.join("manifest.json");
    let metadata = std::fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
    const LIMIT: u64 = 64 * 1024 * 1024;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > LIMIT {
        return Err("BLINK_REVIEW_INVALID: invalid session manifest".to_owned());
    }
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|error| error.to_string())?
        .take(LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > LIMIT
        || format!("sha256:{:x}", Sha256::digest(&bytes)) != proof.manifest_sha256
    {
        return Err("BLINK_REVIEW_STALE: session changed after review".to_owned());
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if manifest["kind"] != "blink-manifest-v1" {
        return Err("BLINK_REVIEW_INVALID: wrong manifest kind".to_owned());
    }
    let origin = selection
        .origin
        .as_ref()
        .ok_or("BLINK_REVIEW_INVALID: missing selection origin")?;
    if origin.blink_manifest_sha256.as_deref() != Some(proof.manifest_sha256.as_str())
        || origin.session_id.as_deref() != manifest["sessionId"].as_str()
        || origin.flags_policy_digest.as_deref() != manifest["flagsPolicyDigest"].as_str()
    {
        return Err("BLINK_REVIEW_INVALID: selection belongs to another session".to_owned());
    }
    let frames = manifest["frames"]
        .as_array()
        .ok_or("BLINK_REVIEW_INVALID: missing frames")?;
    let channels = manifest["channels"]
        .as_array()
        .ok_or("BLINK_REVIEW_INVALID: missing channels")?;
    if frames.is_empty() || frames.len() > MAX_SELECTION_DECISIONS || channels.is_empty() {
        return Err("BLINK_REVIEW_INVALID: invalid review set".to_owned());
    }
    let expected: BTreeSet<&str> = frames
        .iter()
        .map(|frame| value_string(frame, "sourceSha256"))
        .collect::<Result<_, _>>()?;
    let expected_channels: BTreeSet<&str> = channels
        .iter()
        .map(|channel| value_string(channel, "channelId"))
        .collect::<Result<_, _>>()?;
    let viewed: BTreeSet<&str> = proof
        .reviewed_source_sha256s
        .iter()
        .map(String::as_str)
        .collect();
    let confirmed: BTreeSet<&str> = proof
        .confirmed_channel_ids
        .iter()
        .map(String::as_str)
        .collect();
    let selected: BTreeSet<&str> = selection
        .decisions
        .iter()
        .map(|decision| decision.source_sha256.as_str())
        .collect();
    if expected.len() != frames.len()
        || viewed.len() != proof.reviewed_source_sha256s.len()
        || confirmed.len() != proof.confirmed_channel_ids.len()
        || viewed != expected
        || selected != expected
        || confirmed != expected_channels
    {
        return Err(
            "BLINK_REVIEW_INCOMPLETE: view all frames and confirm every channel".to_owned(),
        );
    }
    let measured_paths: BTreeSet<PathBuf> = frames
        .iter()
        .map(|frame| {
            Path::new(value_string(frame, "path")?)
                .canonicalize()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let requested_paths: BTreeSet<PathBuf> = request
        .sources
        .iter()
        .filter(|source| source.role == "LIGHT")
        .flat_map(|source| source.paths.iter())
        .map(|path| {
            Path::new(path)
                .canonicalize()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    if requested_paths != measured_paths {
        return Err("BLINK_REVIEW_STALE: imported Lights changed".to_owned());
    }
    for frame in frames {
        if frame["previews"]["error"].as_str().is_some()
            && selection.decisions.iter().any(|decision| {
                Some(decision.source_sha256.as_str()) == frame["sourceSha256"].as_str()
                    && decision.decision == "KEEP"
            })
        {
            return Err(
                "BLINK_REVIEW_UNAVAILABLE: explicitly drop frames without a preview".to_owned(),
            );
        }
    }
    Ok(())
}
