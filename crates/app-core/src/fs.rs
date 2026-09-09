use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::gate::RequiredResultGate;
use crate::model::{ArtifactReceipt, Recipe, StageReceipt};
use crate::validation::{Validate, ValidationError, validate_identifier, validate_sha256};

const PUBLICATION_MANIFEST: &str = ".openastroflow-publication.json";
const COMPLETION_RECEIPT: &str = ".openastroflow-complete.json";

/// A normalized, UTF-8, platform-neutral relative path.
///
/// Backslashes and Windows-reserved components are rejected even on Unix so a
/// project receipt can be moved safely between macOS and Windows.
#[derive(Clone, Debug, Eq, Hash, JsonSchema, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct SafeRelativePath(#[schemars(length(min = 1, max = 4096))] String);

impl SafeRelativePath {
    /// Validate and construct a portable relative path.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] for unsafe, absolute, non-portable, or
    /// overlong paths.
    pub fn new(value: impl AsRef<str>) -> Result<Self, ValidationError> {
        let value = value.as_ref();
        validate_portable_relative_path("relativePath", value, false)?;
        Ok(Self(value.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn components(&self) -> impl Iterator<Item = &str> {
        self.0.split('/')
    }

    #[must_use]
    pub fn join_under(&self, root: &Path) -> PathBuf {
        self.components()
            .fold(root.to_path_buf(), |path, component| path.join(component))
    }
}

impl Display for SafeRelativePath {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SafeRelativePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Hash, JsonSchema, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct SafeFileName(#[schemars(length(min = 1, max = 255))] String);

impl SafeFileName {
    /// Validate and construct one portable filename component.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] for unsafe or non-portable names.
    pub fn new(value: impl AsRef<str>) -> Result<Self, ValidationError> {
        let value = value.as_ref();
        validate_portable_relative_path("fileName", value, true)?;
        Ok(Self(value.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for SafeFileName {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SafeFileName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

fn validate_portable_relative_path(
    path: &str,
    value: &str,
    single_component: bool,
) -> Result<(), ValidationError> {
    if value.is_empty() || value.len() > 4096 {
        return Err(ValidationError::new(
            path,
            "must contain between 1 and 4096 UTF-8 bytes",
        ));
    }
    if value.starts_with('/') || value.contains('\\') || value.contains('\0') {
        return Err(ValidationError::new(
            path,
            "must be relative, use forward slashes, and contain no NUL",
        ));
    }
    let components: Vec<_> = value.split('/').collect();
    if single_component && components.len() != 1 {
        return Err(ValidationError::new(
            path,
            "must be a single path component",
        ));
    }
    for component in components {
        if component.is_empty() || matches!(component, "." | "..") {
            return Err(ValidationError::new(
                path,
                "empty, current-directory, and parent-directory components are forbidden",
            ));
        }
        if component.len() > 255 {
            return Err(ValidationError::new(
                path,
                "a component exceeds 255 UTF-8 bytes",
            ));
        }
        if component.ends_with('.') || component.ends_with(' ') {
            return Err(ValidationError::new(
                path,
                "components may not end in dot or space",
            ));
        }
        if component.chars().any(|character| {
            character.is_control() || matches!(character, ':' | '*' | '?' | '"' | '<' | '>' | '|')
        }) {
            return Err(ValidationError::new(
                path,
                "contains a character forbidden by the portable path contract",
            ));
        }
        let basename = component
            .split_once('.')
            .map_or(component, |(basename, _)| basename)
            .to_ascii_uppercase();
        let reserved = matches!(basename.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || (basename.len() == 4
                && (basename.starts_with("COM") || basename.starts_with("LPT"))
                && basename.as_bytes()[3].is_ascii_digit()
                && basename.as_bytes()[3] != b'0');
        if reserved {
            return Err(ValidationError::new(
                path,
                "contains a Windows-reserved device name",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FsMetadata {
    pub is_file: bool,
    pub is_directory: bool,
    pub is_symlink: bool,
    pub length: u64,
}

/// OS adapter used by the control plane. Methods with `new` semantics must fail
/// with `AlreadyExists`; they must never truncate or replace an entry.
pub trait PlatformFs: Send + Sync {
    /// Resolve an existing path to its canonical form.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the path cannot be resolved.
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf>;
    /// Inspect a path without following its final symlink.
    ///
    /// # Errors
    ///
    /// Returns an I/O error other than absence.
    fn metadata_no_follow(&self, path: &Path) -> io::Result<Option<FsMetadata>>;
    /// Open an existing file for reading.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the file cannot be opened.
    fn open_read(&self, path: &Path) -> io::Result<Box<dyn Read + Send>>;
    /// Create exactly one new directory and fail if it already exists.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the directory cannot be created new.
    fn create_directory_new(&self, path: &Path) -> io::Result<()>;
    /// Copy a file into a newly created destination without replacement.
    ///
    /// # Errors
    ///
    /// Returns an I/O error on read, create-new, write, or sync failure.
    fn copy_file_new(&self, source: &Path, destination: &Path) -> io::Result<u64>;
    /// Write and sync a newly created file without replacement.
    ///
    /// # Errors
    ///
    /// Returns an I/O error on create-new, write, or sync failure.
    fn write_file_new(&self, destination: &Path, contents: &[u8]) -> io::Result<()>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StdPlatformFs;

impl PlatformFs for StdPlatformFs {
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        fs::canonicalize(path)
    }

    fn metadata_no_follow(&self, path: &Path) -> io::Result<Option<FsMetadata>> {
        match fs::symlink_metadata(path) {
            Ok(metadata) => {
                let file_type = metadata.file_type();
                Ok(Some(FsMetadata {
                    is_file: file_type.is_file(),
                    is_directory: file_type.is_dir(),
                    is_symlink: file_type.is_symlink(),
                    length: metadata.len(),
                }))
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn open_read(&self, path: &Path) -> io::Result<Box<dyn Read + Send>> {
        Ok(Box::new(File::open(path)?))
    }

    fn create_directory_new(&self, path: &Path) -> io::Result<()> {
        fs::create_dir(path)
    }

    fn copy_file_new(&self, source: &Path, destination: &Path) -> io::Result<u64> {
        let mut input = File::open(source)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?;
        let copied = io::copy(&mut input, &mut output)?;
        output.sync_all()?;
        Ok(copied)
    }

    fn write_file_new(&self, destination: &Path, contents: &[u8]) -> io::Result<()> {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?;
        output.write_all(contents)?;
        output.sync_all()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicationFile {
    pub source_relative_path: SafeRelativePath,
    pub destination_relative_path: SafeRelativePath,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NewDirectoryPublication {
    pub schema_version: u16,
    pub publication_id: String,
    pub destination_name: SafeFileName,
    pub files: Vec<PublicationFile>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicationReceipt {
    pub schema_version: u16,
    pub publication_id: String,
    pub destination_name: SafeFileName,
    pub manifest_sha256: String,
    pub file_count: u64,
    pub total_bytes: u64,
    pub completed_at_unix_ms: u64,
}

/// Move-only structural authorization binding a ready scientific result gate
/// and artifact receipts to one exact publication plan.
#[derive(Debug)]
pub struct PublicationAuthorization {
    publication_id: String,
    plan_sha256: String,
}

impl Validate for NewDirectoryPublication {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != 1 {
            return Err(ValidationError::new(
                "publication.schemaVersion",
                "unsupported version; expected 1",
            ));
        }
        validate_identifier("publication.publicationId", &self.publication_id)?;
        if self.files.is_empty() {
            return Err(ValidationError::new(
                "publication.files",
                "must contain at least one file",
            ));
        }
        let mut destinations = BTreeSet::new();
        let mut total_bytes = 0_u64;
        for (index, file) in self.files.iter().enumerate() {
            validate_sha256(&format!("publication.files[{index}].sha256"), &file.sha256)?;
            if matches!(
                file.destination_relative_path.as_str(),
                PUBLICATION_MANIFEST | COMPLETION_RECEIPT
            ) {
                return Err(ValidationError::new(
                    format!("publication.files[{index}].destinationRelativePath"),
                    "reserved publication marker name",
                ));
            }
            if !destinations.insert(file.destination_relative_path.as_str()) {
                return Err(ValidationError::new(
                    format!("publication.files[{index}].destinationRelativePath"),
                    "duplicate destination",
                ));
            }
            total_bytes = total_bytes.checked_add(file.size_bytes).ok_or_else(|| {
                ValidationError::new("publication.files", "total byte count overflows u64")
            })?;
        }
        for first in &destinations {
            for second in &destinations {
                if first != second
                    && second
                        .strip_prefix(first)
                        .is_some_and(|suffix| suffix.starts_with('/'))
                {
                    return Err(ValidationError::new(
                        "publication.files.destinationRelativePath",
                        format!("file/directory prefix collision between {first} and {second}"),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl NewDirectoryPublication {
    /// Bind a structurally ready result gate and every declared artifact to
    /// this exact publication. Actual FITS/WCS verification is performed by
    /// the trusted controller before it creates these receipts.
    ///
    /// # Errors
    ///
    /// Returns [`PublicationError`] when the result gate is blocked, a receipt
    /// is invalid, or a publication member differs from its artifact receipt.
    #[allow(clippy::too_many_lines)] // One fail-closed transaction is easier to audit in sequence.
    pub fn authorize(
        &self,
        recipe: &Recipe,
        stages: &[StageReceipt],
        artifacts: &[ArtifactReceipt],
    ) -> Result<PublicationAuthorization, PublicationError> {
        self.validate().map_err(PublicationError::InvalidPlan)?;
        let mut stage_by_id = BTreeMap::new();
        for stage in stages {
            stage
                .validate()
                .map_err(PublicationError::InvalidScientificReceipt)?;
            if stage_by_id.insert(stage.stage_id.as_str(), stage).is_some() {
                return Err(PublicationError::ArtifactBinding(
                    "duplicate stage receipt identifier".to_owned(),
                ));
            }
            if !recipe.stages.iter().any(|configured| {
                configured.enabled
                    && configured.stage_id == stage.stage_id
                    && configured.kind == stage.kind
            }) {
                return Err(PublicationError::ArtifactBinding(format!(
                    "stage {} is absent or differs from the recipe",
                    stage.stage_id
                )));
            }
        }
        for configured in recipe.stages.iter().filter(|stage| stage.enabled) {
            if !stage_by_id
                .get(configured.stage_id.as_str())
                .is_some_and(|receipt| stage_allows_publication(recipe, configured.kind, receipt))
            {
                return Err(PublicationError::ArtifactBinding(format!(
                    "enabled stage {} has no successful receipt",
                    configured.stage_id
                )));
            }
        }
        let mut by_path = BTreeMap::new();
        let mut artifact_ids = BTreeSet::new();
        for artifact in artifacts {
            artifact
                .validate()
                .map_err(PublicationError::InvalidScientificReceipt)?;
            if by_path
                .insert(artifact.relative_path.as_str(), artifact)
                .is_some()
            {
                return Err(PublicationError::ArtifactBinding(
                    "duplicate artifact relative path".to_owned(),
                ));
            }
            if !artifact_ids.insert(artifact.artifact_id.as_str()) {
                return Err(PublicationError::ArtifactBinding(
                    "duplicate artifact receipt identifier".to_owned(),
                ));
            }
            let Some(stage) = stage_by_id.get(artifact.produced_by_stage_id.as_str()) else {
                return Err(PublicationError::ArtifactBinding(format!(
                    "artifact {} names an unknown producing stage",
                    artifact.artifact_id
                )));
            };
            if !stage.artifact_ids.contains(&artifact.artifact_id) {
                return Err(PublicationError::ArtifactBinding(format!(
                    "producing stage does not link artifact {}",
                    artifact.artifact_id
                )));
            }
        }
        for stage in stages {
            let mut stage_artifacts = BTreeSet::new();
            for artifact_id in &stage.artifact_ids {
                if !stage_artifacts.insert(artifact_id.as_str()) {
                    return Err(PublicationError::ArtifactBinding(format!(
                        "stage {} repeats artifact {}",
                        stage.stage_id, artifact_id
                    )));
                }
                if !artifact_ids.contains(artifact_id.as_str()) {
                    return Err(PublicationError::ArtifactBinding(format!(
                        "stage {} links unknown artifact {}",
                        stage.stage_id, artifact_id
                    )));
                }
            }
        }
        let gate = RequiredResultGate::evaluate(recipe, stages, artifacts);
        if !gate.is_ready() {
            let failed = gate
                .checks
                .iter()
                .filter(|check| check.required && !check.passed)
                .map(|check| check.code.as_str())
                .collect::<Vec<_>>()
                .join(",");
            return Err(PublicationError::ResultGateBlocked(failed));
        }
        if self.files.len() != artifacts.len() {
            return Err(PublicationError::ArtifactBinding(
                "publication and artifact receipt counts differ".to_owned(),
            ));
        }
        for file in &self.files {
            let Some(artifact) = by_path.get(file.destination_relative_path.as_str()) else {
                return Err(PublicationError::ArtifactBinding(format!(
                    "no artifact receipt for {}",
                    file.destination_relative_path
                )));
            };
            if artifact.sha256 != file.sha256 || artifact.size_bytes != file.size_bytes {
                return Err(PublicationError::ArtifactBinding(format!(
                    "identity differs for {}",
                    file.destination_relative_path
                )));
            }
        }
        let plan = serde_json::to_vec(self).map_err(PublicationError::Serialize)?;
        Ok(PublicationAuthorization {
            publication_id: self.publication_id.clone(),
            plan_sha256: sha256_bytes(&plan),
        })
    }

    /// Publish into a destination that must not already exist.
    ///
    /// `create_directory_new` is the exclusive claim. No existing path is ever
    /// replaced. The directory is consumable only after the completion receipt
    /// appears; a failure deliberately preserves the incomplete directory for
    /// audit/recovery. Staging is required to be immutable while this runs.
    ///
    /// # Errors
    ///
    /// Returns [`PublicationError`] before claiming the destination for an
    /// invalid plan/source, or an `Incomplete` error after a claimed publication
    /// cannot be completed. An incomplete directory is intentionally preserved.
    pub fn publish<F: PlatformFs>(
        &self,
        authorization: &PublicationAuthorization,
        fs: &F,
        staging_root: &Path,
        destination_parent: &Path,
        completed_at_unix_ms: u64,
    ) -> Result<PublicationReceipt, PublicationError> {
        self.validate().map_err(PublicationError::InvalidPlan)?;
        let plan_bytes = serde_json::to_vec(self).map_err(PublicationError::Serialize)?;
        if authorization.publication_id != self.publication_id
            || authorization.plan_sha256 != sha256_bytes(&plan_bytes)
        {
            return Err(PublicationError::AuthorizationMismatch);
        }
        let staging_root = canonical_directory(fs, staging_root, "staging root")?;
        let destination_parent = canonical_directory(fs, destination_parent, "destination parent")?;
        let destination = destination_parent.join(self.destination_name.as_str());
        if fs
            .metadata_no_follow(&destination)
            .map_err(|source| PublicationError::Io {
                operation: "inspect destination".to_owned(),
                path: destination.clone(),
                source,
            })?
            .is_some()
        {
            return Err(PublicationError::DestinationExists(destination));
        }

        let mut sources = Vec::with_capacity(self.files.len());
        for file in &self.files {
            assert_no_symlink_components(fs, &staging_root, &file.source_relative_path)?;
            let source_candidate = file.source_relative_path.join_under(&staging_root);
            let source =
                fs.canonicalize(&source_candidate)
                    .map_err(|source| PublicationError::Io {
                        operation: "canonicalize source".to_owned(),
                        path: source_candidate.clone(),
                        source,
                    })?;
            if !source.starts_with(&staging_root) || source == staging_root {
                return Err(PublicationError::SourceEscapesStaging(source));
            }
            let metadata = fs
                .metadata_no_follow(&source)
                .map_err(|source_error| PublicationError::Io {
                    operation: "inspect source".to_owned(),
                    path: source.clone(),
                    source: source_error,
                })?
                .ok_or_else(|| PublicationError::SourceNotRegular(source.clone()))?;
            if !metadata.is_file || metadata.is_symlink || metadata.length != file.size_bytes {
                return Err(PublicationError::SourceNotRegular(source));
            }
            let digest = sha256_file(fs, &source).map_err(|source_error| PublicationError::Io {
                operation: "hash source".to_owned(),
                path: source.clone(),
                source: source_error,
            })?;
            if digest != file.sha256 {
                return Err(PublicationError::SourceIdentityMismatch {
                    path: source,
                    expected: file.sha256.clone(),
                    actual: digest,
                });
            }
            sources.push(source);
        }

        let manifest_bytes = serde_json::to_vec(self).map_err(PublicationError::Serialize)?;
        let manifest_sha256 = sha256_bytes(&manifest_bytes);
        fs.create_directory_new(&destination)
            .map_err(|source| PublicationError::Io {
                operation: "claim new destination directory".to_owned(),
                path: destination.clone(),
                source,
            })?;

        self.publish_after_claim(
            fs,
            &destination,
            &sources,
            &manifest_bytes,
            manifest_sha256,
            completed_at_unix_ms,
        )
        .map_err(|error| PublicationError::Incomplete {
            destination,
            cause: Box::new(error),
        })
    }

    fn publish_after_claim<F: PlatformFs>(
        &self,
        fs: &F,
        destination: &Path,
        sources: &[PathBuf],
        manifest_bytes: &[u8],
        manifest_sha256: String,
        completed_at_unix_ms: u64,
    ) -> Result<PublicationReceipt, PublicationError> {
        write_new(fs, &destination.join(PUBLICATION_MANIFEST), manifest_bytes)?;
        for (file, source) in self.files.iter().zip(sources) {
            let target = file.destination_relative_path.join_under(destination);
            ensure_new_parent_directories(fs, destination, &file.destination_relative_path)?;
            let copied =
                fs.copy_file_new(source, &target)
                    .map_err(|source_error| PublicationError::Io {
                        operation: "copy file without replacement".to_owned(),
                        path: target.clone(),
                        source: source_error,
                    })?;
            if copied != file.size_bytes {
                return Err(PublicationError::CopiedIdentityMismatch {
                    path: target,
                    expected: file.sha256.clone(),
                    actual: format!("size:{copied}"),
                });
            }
            let digest = sha256_file(fs, &target).map_err(|source_error| PublicationError::Io {
                operation: "verify copied file".to_owned(),
                path: target.clone(),
                source: source_error,
            })?;
            if digest != file.sha256 {
                return Err(PublicationError::CopiedIdentityMismatch {
                    path: target,
                    expected: file.sha256.clone(),
                    actual: digest,
                });
            }
        }

        let receipt = PublicationReceipt {
            schema_version: 1,
            publication_id: self.publication_id.clone(),
            destination_name: self.destination_name.clone(),
            manifest_sha256,
            file_count: self.files.len() as u64,
            total_bytes: self.files.iter().map(|file| file.size_bytes).sum::<u64>(),
            completed_at_unix_ms,
        };
        let receipt_bytes = serde_json::to_vec(&receipt).map_err(PublicationError::Serialize)?;
        write_new(fs, &destination.join(COMPLETION_RECEIPT), &receipt_bytes)?;
        Ok(receipt)
    }
}

fn stage_allows_publication(
    recipe: &Recipe,
    configured_kind: crate::model::StageKind,
    receipt: &StageReceipt,
) -> bool {
    use crate::model::{ResultRequirement, StageKind, StageStatus};

    if receipt.kind != configured_kind {
        return false;
    }
    if receipt.status == StageStatus::Succeeded {
        return true;
    }
    let best_effort = match configured_kind {
        StageKind::AstrometricSolve => recipe.solver.result == ResultRequirement::BestEffort,
        StageKind::Drizzle => recipe.drizzle.result == ResultRequirement::BestEffort,
        _ => false,
    };
    best_effort && matches!(receipt.status, StageStatus::Failed | StageStatus::Skipped)
}

fn canonical_directory<F: PlatformFs>(
    fs: &F,
    path: &Path,
    description: &str,
) -> Result<PathBuf, PublicationError> {
    let canonical = fs
        .canonicalize(path)
        .map_err(|source| PublicationError::Io {
            operation: format!("canonicalize {description}"),
            path: path.to_path_buf(),
            source,
        })?;
    let metadata = fs
        .metadata_no_follow(&canonical)
        .map_err(|source| PublicationError::Io {
            operation: format!("inspect {description}"),
            path: canonical.clone(),
            source,
        })?
        .ok_or_else(|| PublicationError::NotDirectory(canonical.clone()))?;
    if !metadata.is_directory || metadata.is_symlink {
        return Err(PublicationError::NotDirectory(canonical));
    }
    Ok(canonical)
}

fn assert_no_symlink_components<F: PlatformFs>(
    fs: &F,
    root: &Path,
    relative: &SafeRelativePath,
) -> Result<(), PublicationError> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        current.push(component);
        let metadata = fs
            .metadata_no_follow(&current)
            .map_err(|source| PublicationError::Io {
                operation: "inspect source path component".to_owned(),
                path: current.clone(),
                source,
            })?
            .ok_or_else(|| PublicationError::SourceNotRegular(current.clone()))?;
        if metadata.is_symlink {
            return Err(PublicationError::SourceSymlink(current));
        }
    }
    Ok(())
}

fn ensure_new_parent_directories<F: PlatformFs>(
    fs: &F,
    destination_root: &Path,
    relative_file: &SafeRelativePath,
) -> Result<(), PublicationError> {
    let components: Vec<_> = relative_file.components().collect();
    let mut directory = destination_root.to_path_buf();
    for component in components.iter().take(components.len().saturating_sub(1)) {
        directory.push(component);
        match fs
            .metadata_no_follow(&directory)
            .map_err(|source| PublicationError::Io {
                operation: "inspect destination directory".to_owned(),
                path: directory.clone(),
                source,
            })? {
            Some(metadata) if metadata.is_directory && !metadata.is_symlink => {}
            Some(_) => return Err(PublicationError::UnsafeDestinationEntry(directory)),
            None => fs
                .create_directory_new(&directory)
                .map_err(|source| PublicationError::Io {
                    operation: "create destination directory".to_owned(),
                    path: directory.clone(),
                    source,
                })?,
        }
    }
    Ok(())
}

fn write_new<F: PlatformFs>(
    fs: &F,
    destination: &Path,
    contents: &[u8],
) -> Result<(), PublicationError> {
    fs.write_file_new(destination, contents)
        .map_err(|source| PublicationError::Io {
            operation: "write publication marker without replacement".to_owned(),
            path: destination.to_path_buf(),
            source,
        })
}

fn sha256_file<F: PlatformFs>(fs: &F, path: &Path) -> io::Result<String> {
    let mut reader = fs.open_read(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 128 * 1024].into_boxed_slice();
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Error)]
pub enum PublicationError {
    #[error("invalid publication plan: {0}")]
    InvalidPlan(ValidationError),
    #[error("invalid scientific receipt: {0}")]
    InvalidScientificReceipt(ValidationError),
    #[error("required scientific result gate is blocked: {0}")]
    ResultGateBlocked(String),
    #[error("publication/artifact binding failed: {0}")]
    ArtifactBinding(String),
    #[error("publication authorization does not match this plan")]
    AuthorizationMismatch,
    #[error("path is not a real directory: {0}")]
    NotDirectory(PathBuf),
    #[error("destination already exists; refusing to replace it: {0}")]
    DestinationExists(PathBuf),
    #[error("source escapes the canonical staging directory: {0}")]
    SourceEscapesStaging(PathBuf),
    #[error("source is not a regular file with the declared size: {0}")]
    SourceNotRegular(PathBuf),
    #[error("source path contains a symlink: {0}")]
    SourceSymlink(PathBuf),
    #[error("unsafe existing entry in the newly claimed destination: {0}")]
    UnsafeDestinationEntry(PathBuf),
    #[error("source identity mismatch at {path}: expected {expected}, got {actual}")]
    SourceIdentityMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("copied identity mismatch at {path}: expected {expected}, got {actual}")]
    CopiedIdentityMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("failed to serialize publication receipt: {0}")]
    Serialize(serde_json::Error),
    #[error("{operation} failed for {path}: {source}")]
    Io {
        operation: String,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "publication failed after the destination was claimed; incomplete directory is preserved at {destination}: {cause}"
    )]
    Incomplete {
        destination: PathBuf,
        #[source]
        cause: Box<PublicationError>,
    },
}
