//! completion for the desktop controller.

use super::*;

pub(super) fn validate_completion(
    output: &Path,
    result: &serde_json::Value,
) -> Result<Completion, String> {
    if result.get("success").and_then(serde_json::Value::as_bool) != Some(true)
        || result.get("state").and_then(serde_json::Value::as_str) != Some("SOLVED")
    {
        return Err(format!(
            "{}: project did not publish a solved result",
            result
                .get("code")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("PROJECT_FAILED")
        ));
    }
    let reported_output = Path::new(
        result
            .get("outputDirectory")
            .and_then(serde_json::Value::as_str)
            .ok_or("project result has no outputDirectory")?,
    )
    .canonicalize()
    .map_err(|error| error.to_string())?;
    let root = output.canonicalize().map_err(|error| error.to_string())?;
    if reported_output != root {
        return Err("project result names a different output directory".to_owned());
    }
    let receipt_path = Path::new(
        result
            .get("receiptPath")
            .and_then(serde_json::Value::as_str)
            .ok_or("project result has no receiptPath")?,
    )
    .canonicalize()
    .map_err(|error| error.to_string())?;
    if !receipt_path.starts_with(&root) || !receipt_path.is_file() {
        return Err("project receipt escaped its output directory".to_owned());
    }
    let mut receipt_bytes = Vec::new();
    File::open(&receipt_path)
        .map_err(|error| error.to_string())?
        .take((MAX_RESULT_BYTES + 1) as u64)
        .read_to_end(&mut receipt_bytes)
        .map_err(|error| error.to_string())?;
    if receipt_bytes.len() > MAX_RESULT_BYTES {
        return Err("project receipt is too large".to_owned());
    }
    let receipt: serde_json::Value =
        serde_json::from_slice(&receipt_bytes).map_err(|error| error.to_string())?;
    if receipt.get("success").and_then(serde_json::Value::as_bool) != Some(true)
        || receipt.get("state").and_then(serde_json::Value::as_str) != Some("SOLVED")
    {
        return Err("project receipt is not a solved success receipt".to_owned());
    }
    let final_products = receipt
        .get("finalProducts")
        .ok_or("project receipt has no finalProducts")?;
    let gate = final_products
        .get("resultGate")
        .ok_or("project receipt has no resultGate")?;
    for field in [
        "allMonoProductsSolved",
        "managedCatalogEvidenceRequired",
        "sourceIdentityVerifiedAtCommit",
        "mosaicCoverageOverlapSeamPassed",
    ] {
        if gate.get(field).and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(format!("project result gate did not pass {field}"));
        }
    }
    if gate.get("status").and_then(serde_json::Value::as_str) != Some("PASS") {
        return Err("project result gate is not PASS".to_owned());
    }
    let records = final_products
        .get("guiArtifacts")
        .and_then(serde_json::Value::as_array)
        .ok_or("project receipt has no guiArtifacts")?;
    let mut artifacts = Vec::new();
    let mut solved_count = 0_u32;
    for (index, record) in records.iter().enumerate() {
        let kind = value_string(record, "kind")?;
        if !matches!(
            kind,
            "SOLVED_MONO_FITS"
                | "LINEAR_RGB_FITS"
                | "RGB_PREVIEW_TIFF_16"
                | "RGB_PREVIEW_PNG_16"
                | "MONO_PREVIEW_PNG"
        ) {
            return Err(format!("unsupported GUI artifact kind: {kind}"));
        }
        let relative_value = record
            .get("relativePath")
            .or_else(|| record.get("path"))
            .and_then(serde_json::Value::as_str)
            .ok_or("GUI artifact has no relativePath")?;
        let (relative, resolved) = relative_artifact(&root, relative_value)?;
        let expected_size = record
            .get("sizeBytes")
            .and_then(serde_json::Value::as_u64)
            .ok_or("GUI artifact has no sizeBytes")?;
        let expected_sha = artifact_sha256_hex(value_string(record, "sha256")?)?;
        if resolved
            .metadata()
            .map_err(|error| error.to_string())?
            .len()
            != expected_size
            || sha256_file(&resolved)? != expected_sha
        {
            return Err("GUI artifact content identity changed before publication".to_owned());
        }
        let astrometry = if matches!(kind, "SOLVED_MONO_FITS" | "LINEAR_RGB_FITS") {
            let mut value = record
                .get("astrometry")
                .and_then(serde_json::Value::as_object)
                .cloned()
                .ok_or("final FITS artifact has no astrometry evidence")?;
            value.remove("imageShape");
            value.remove("state");
            let parsed: AstrometricSolutionReceipt = serde_json::from_value(value.into())
                .map_err(|error| format!("astrometry receipt is invalid: {error}"))?;
            parsed.validate().map_err(|error| error.to_string())?;
            if record
                .get("finalGate")
                .and_then(|item| item.get("status"))
                .and_then(serde_json::Value::as_str)
                != Some("PASS")
            {
                return Err("final FITS artifact gate is not PASS".to_owned());
            }
            solved_count += u32::from(kind == "SOLVED_MONO_FITS");
            Some(parsed)
        } else {
            None
        };
        let detail = astrometry.as_ref().map_or_else(
            || match kind {
                "RGB_PREVIEW_TIFF_16" => "16-bit display preview".to_owned(),
                "RGB_PREVIEW_PNG_16" => "16-bit display preview".to_owned(),
                "MONO_PREVIEW_PNG" => "auto-stretched mono preview".to_owned(),
                _ => "verified product".to_owned(),
            },
            |value| {
                format!(
                    "WCS embedded · {:.3} arcsec RMS · {} matched stars",
                    value.rms_arcsec, value.matched_stars
                )
            },
        );
        let name = resolved
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("artifact")
            .to_owned();
        artifacts.push(UiArtifact {
            kind: kind.to_owned(),
            name,
            path: resolved.to_string_lossy().into_owned(),
            detail,
            filter: record
                .get("filter")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            target: record
                .get("target")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            preview_data_url: None,
            receipt: UiArtifactReceipt {
                artifact_id: format!("project-artifact-{}", index + 1),
                relative_path: relative,
                sha256: expected_sha.to_owned(),
                size_bytes: expected_size,
                astrometry,
            },
        });
    }
    if solved_count == 0 {
        return Err("project receipt has no solved mono product".to_owned());
    }
    attach_artifact_previews(&mut artifacts);
    let receipt_relative = receipt_path
        .strip_prefix(&root)
        .map_err(|_| "receipt path escaped output")?
        .to_string_lossy()
        .trim_start_matches(std::path::MAIN_SEPARATOR)
        .replace('\\', "/");
    artifacts.push(UiArtifact {
        kind: "RECEIPT".to_owned(),
        name: receipt_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("receipt.json")
            .to_owned(),
        path: receipt_path.to_string_lossy().into_owned(),
        detail: "outer project receipt · content verified".to_owned(),
        filter: None,
        target: None,
        preview_data_url: None,
        receipt: UiArtifactReceipt {
            artifact_id: "project-receipt".to_owned(),
            relative_path: receipt_relative,
            sha256: sha256_file(&receipt_path)?,
            size_bytes: receipt_path
                .metadata()
                .map_err(|error| error.to_string())?
                .len(),
            astrometry: None,
        },
    });
    let screening = screening_summary(&root, &receipt)?;
    Ok(Completion {
        artifacts,
        screening,
    })
}
