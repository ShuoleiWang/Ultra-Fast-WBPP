//! previews for the desktop controller.

use super::*;

pub(super) fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let value = chunk.iter().enumerate().fold(0_u32, |acc, (index, byte)| {
            acc | (u32::from(*byte) << (16 - 8 * index))
        });
        for position in 0..4 {
            if position <= chunk.len() {
                let index = (value >> (18 - 6 * position)) & 0x3f;
                encoded.push(ALPHABET[index as usize] as char);
            } else {
                encoded.push('=');
            }
        }
    }
    encoded
}

/// The two image encodings the worker publishes for the webview: the PNG
/// product and review previews, and the JPEG blink filmstrip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreviewFormat {
    Png,
    Jpeg,
}

impl PreviewFormat {
    /// The format a preview's file extension declares (`jpg`, `jpeg`, `png`,
    /// case-insensitive); anything else is not a preview.
    pub(super) fn from_extension(path: &Path) -> Option<Self> {
        match path
            .extension()
            .and_then(|value| value.to_str())?
            .to_ascii_lowercase()
            .as_str()
        {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            _ => None,
        }
    }

    fn media_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
        }
    }

    fn signature(self) -> &'static [u8] {
        match self {
            Self::Png => &PNG_SIGNATURE,
            Self::Jpeg => &JPEG_SIGNATURE,
        }
    }
}

/// Loads one image the worker published at `resolved` (already checked to lie
/// under its root) as a data URL, or `None` when its bytes do not start with
/// the `format` signature, it is larger than `max_bytes`, or it would exceed
/// the remaining transport `budget`.
pub(crate) fn image_data_url(
    resolved: &Path,
    format: PreviewFormat,
    max_bytes: u64,
    budget: &mut usize,
) -> Option<String> {
    let size = resolved.metadata().ok()?.len();
    if size == 0 || size > max_bytes {
        return None;
    }
    let mut bytes = Vec::with_capacity(size as usize);
    File::open(resolved)
        .ok()?
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 != size || !bytes.starts_with(format.signature()) {
        return None;
    }
    let encoded = format!(
        "data:{};base64,{}",
        format.media_type(),
        base64_encode(&bytes)
    );
    if *budget < encoded.len() {
        return None;
    }
    *budget -= encoded.len();
    Some(encoded)
}

/// Loads one PNG the worker published at `resolved` (already checked to lie
/// under the output root) as a data URL, or `None` when it is not a PNG, is
/// larger than `max_bytes`, or would exceed the remaining transport `budget`.
pub(super) fn png_data_url(resolved: &Path, max_bytes: u64, budget: &mut usize) -> Option<String> {
    image_data_url(resolved, PreviewFormat::Png, max_bytes, budget)
}

/// A blink zoom preview is the 1/4-scale 8-bit PNG (1.0 to 1.5 MB for a
/// 26 MP channel); it is loaded on demand, one at a time.
pub(super) const MAX_BLINK_PREVIEW_BYTES: u64 = 2 * 1024 * 1024;

/// Resolves a preview path from a blink manifest against its session
/// directory (already canonical): relative, plain components only, no
/// symlink anywhere below the session directory, a regular `.jpg`/`.jpeg`/
/// `.png` file.  Returns the resolved path and the format its extension
/// declares.
pub(crate) fn resolve_blink_preview(
    session_directory: &Path,
    relative: &str,
) -> Result<(PathBuf, PreviewFormat), String> {
    let relative = Path::new(relative);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|item| !matches!(item, Component::Normal(_)))
    {
        return Err("blink preview has an unsafe relative path".to_owned());
    }
    let format = PreviewFormat::from_extension(relative)
        .ok_or("blink preview must be a .jpg, .jpeg or .png file")?;
    let mut current = session_directory.to_path_buf();
    for component in relative.components() {
        current.push(component);
        let metadata = std::fs::symlink_metadata(&current)
            .map_err(|error| format!("cannot resolve blink preview: {error}"))?;
        if metadata.file_type().is_symlink() {
            return Err("blink preview path crosses a symbolic link".to_owned());
        }
    }
    let resolved = current
        .canonicalize()
        .map_err(|error| format!("cannot resolve blink preview: {error}"))?;
    if !resolved.starts_with(session_directory) || !resolved.is_file() {
        return Err("blink preview escaped its session directory".to_owned());
    }
    Ok((resolved, format))
}

/// A blink session directory is a direct child of the desktop's sessions
/// root that the desktop named itself (`<16 hex>-<yyyymmdd>-<hhmmss>`, with an
/// optional `-N` counter); anything else is not ours to read or remove.
pub(crate) fn blink_session_name_parts(name: &str) -> Option<(u64, u64)> {
    let mut parts = name.split('-');
    let digest = parts.next()?;
    let date = parts.next()?;
    let time = parts.next()?;
    let counter = match parts.next() {
        None => 1,
        Some(value) => {
            if value.is_empty()
                || value.len() > 6
                || !value.bytes().all(|byte| byte.is_ascii_digit())
                || parts.next().is_some()
            {
                return None;
            }
            value.parse::<u64>().ok()?
        }
    };
    let hex = digest.len() == 16
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
    let digits =
        |value: &str, len: usize| value.len() == len && value.bytes().all(|b| b.is_ascii_digit());
    if !hex || !digits(date, 8) || !digits(time, 6) {
        return None;
    }
    let stamp = format!("{date}{time}").parse::<u64>().ok()?;
    Some((stamp, counter))
}

/// Canonicalises `session_directory` and checks that it is one of the
/// desktop's own blink sessions under `root`.
pub(crate) fn checked_blink_session_directory(
    root: &Path,
    session_directory: &str,
) -> Result<PathBuf, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("blink sessions root is unavailable: {error}"))?;
    let session = Path::new(session_directory)
        .canonicalize()
        .map_err(|error| format!("cannot resolve blink session directory: {error}"))?;
    let owned = session.parent() == Some(root.as_path())
        && session
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| blink_session_name_parts(name).is_some());
    if !owned || !session.is_dir() {
        return Err("blink session directory is not one of this application's sessions".to_owned());
    }
    Ok(session)
}

/// One blink preview (`filmstrip/…` or `zoom/…`) of a session as a data URL,
/// for the previews the manifest transport left out and for the zoom images.
pub(crate) fn load_blink_preview<R: Runtime>(
    app: &AppHandle<R>,
    session_directory: &str,
    relative_path: &str,
) -> Result<String, String> {
    let root = crate::sidecar::blink_sessions_root(app)?;
    load_blink_preview_with(&root, session_directory, relative_path)
}

pub(crate) fn load_blink_preview_with(
    root: &Path,
    session_directory: &str,
    relative_path: &str,
) -> Result<String, String> {
    let session = checked_blink_session_directory(root, session_directory)?;
    let (resolved, format) = resolve_blink_preview(&session, relative_path)?;
    let mut budget = usize::MAX;
    image_data_url(&resolved, format, MAX_BLINK_PREVIEW_BYTES, &mut budget).ok_or_else(|| {
        "blink preview is empty, over 2 MB or not the image its name declares".to_owned()
    })
}

/// Load one review preview the worker published under `root` as a data URL.
pub(super) fn screening_preview(root: &Path, relative: &str, budget: &mut usize) -> Option<String> {
    let (_, resolved) = relative_artifact(root, relative).ok()?;
    png_data_url(&resolved, MAX_SCREENING_PREVIEW_BYTES, budget)
}

/// Attaches the product previews to the verified artifacts: the mono previews
/// first (one per channel card), then the RGB preview, so the per-channel
/// cards keep theirs when the budget runs short.
pub(super) fn attach_artifact_previews(artifacts: &mut [UiArtifact]) {
    let mut budget = MAX_ARTIFACT_PREVIEW_TOTAL_BYTES;
    for (kind, max_bytes) in [
        ("MONO_PREVIEW_PNG", MAX_MONO_PREVIEW_BYTES),
        ("RGB_PREVIEW_PNG_16", MAX_RGB_PREVIEW_BYTES),
    ] {
        for artifact in artifacts.iter_mut().filter(|item| item.kind == kind) {
            artifact.preview_data_url =
                png_data_url(Path::new(&artifact.path), max_bytes, &mut budget);
        }
    }
}

/// The receipt's `execution.screening`, with previews loaded; `None` when a
/// receipt predates screening records.  A malformed record is an error: the
/// result page must not show a partial screening as if it were complete.
pub(super) fn screening_summary(
    root: &Path,
    receipt: &serde_json::Value,
) -> Result<Option<UiScreening>, String> {
    let Some(record) = receipt
        .get("execution")
        .and_then(|item| item.get("screening"))
    else {
        return Ok(None);
    };
    let count = |field: &str| -> Result<u64, String> {
        record
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("screening record has no {field}"))
    };
    let counts = record
        .get("counts")
        .and_then(serde_json::Value::as_object)
        .ok_or("screening record has no counts")?
        .iter()
        .map(|(key, value)| {
            value
                .as_u64()
                .map(|count| (key.clone(), count))
                .ok_or_else(|| format!("screening count {key} is not a number"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let records = record
        .get("frames")
        .and_then(serde_json::Value::as_array)
        .ok_or("screening record has no frames")?;
    if records.len() > MAX_SCREENING_FRAMES {
        return Err("screening record lists too many frames".to_owned());
    }
    let mut budget = MAX_SCREENING_PREVIEW_TOTAL_BYTES;
    let mut previews = 0_usize;
    let mut frames = Vec::with_capacity(records.len());
    for item in records {
        let path = value_string(item, "path")?;
        let disposition = value_string(item, "disposition")?;
        if !matches!(disposition, "PASS" | "REVIEW" | "HARD_FAIL") {
            return Err(format!(
                "screening frame has an unknown disposition {disposition}"
            ));
        }
        let evidence = item
            .get("evidence")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .take(8)
                    .map(|text| text.chars().take(500).collect::<String>())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let reason = match item.get("reason") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(value))
                if matches!(value.as_str(), "USER_DROP" | "USER_KEEP_OVERRIDE") =>
            {
                Some(value.clone())
            }
            Some(_) => return Err("screening frame has an unknown reason".to_owned()),
        };
        let flags = item
            .get("flags")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .filter(|code| checked_flag_code(code))
                    .take(16)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let preview_data_url = item
            .get("reviewPreview")
            .and_then(serde_json::Value::as_str)
            .filter(|_| previews < MAX_SCREENING_PREVIEWS)
            .and_then(|relative| screening_preview(root, relative, &mut budget));
        previews += usize::from(preview_data_url.is_some());
        frames.push(UiScreeningFrame {
            name: path
                .rsplit(['/', '\\'])
                .next()
                .filter(|name| !name.is_empty())
                .unwrap_or(path)
                .to_owned(),
            target: item
                .get("target")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            disposition: disposition.to_owned(),
            admitted: item
                .get("admitted")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            summary: item
                .get("summary")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .chars()
                .take(2000)
                .collect(),
            evidence,
            star_count: item.get("starCount").and_then(serde_json::Value::as_u64),
            preview_data_url,
            reason,
            flags,
        });
    }
    Ok(Some(UiScreening {
        admitted: count("admitted")?,
        excluded: count("excluded")?,
        counts,
        frames,
    }))
}
