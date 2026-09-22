//! bundle for the desktop controller.

use super::*;

#[cfg(any(debug_assertions, test))]
pub(super) fn executable_name() -> &'static str {
    if cfg!(windows) {
        "openastroflow-worker.exe"
    } else {
        "openastroflow-worker"
    }
}

pub(super) fn target_suffixed_name() -> &'static str {
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

pub(super) fn target_manifest_name() -> String {
    format!(
        "{}.manifest.json",
        target_suffixed_name().trim_end_matches(".exe")
    )
}

pub(super) fn target_triple_name() -> &'static str {
    target_suffixed_name()
        .trim_start_matches("openastroflow-worker-")
        .trim_end_matches(".exe")
}

pub(super) fn safe_runtime_relative_path(value: &str) -> Result<PathBuf, String> {
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

pub(super) fn object_keys(value: &serde_json::Value) -> Result<BTreeSet<&str>, String> {
    value
        .as_object()
        .map(|object| object.keys().map(String::as_str).collect())
        .ok_or_else(|| "runtime manifest record must be an object".to_owned())
}

pub(super) fn collect_runtime_paths(
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

pub(super) fn verify_bundled_runtime(resource_root: &Path) -> Result<EngineExecutable, String> {
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
pub(super) fn development_candidate_paths(
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
