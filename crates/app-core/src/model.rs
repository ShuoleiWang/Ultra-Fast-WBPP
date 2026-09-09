use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::backend::HardwareProfile;
use crate::fs::SafeRelativePath;
use crate::validation::{Validate, ValidationError, validate_identifier, validate_sha256};

pub const PROJECT_SCHEMA_VERSION: u16 = 1;
pub const RECIPE_SCHEMA_VERSION: u16 = 1;
pub const RECEIPT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Project {
    pub schema_version: u16,
    pub project_id: String,
    pub display_name: String,
    pub created_at_unix_ms: u64,
    pub sources: Vec<ProjectSource>,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProjectSource {
    pub source_id: String,
    pub role: InputRole,
    /// Native path interpreted only by the worker running on the same host.
    pub host_path: String,
    #[serde(default)]
    pub recursive: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
}

#[derive(
    Clone, Copy, Debug, Deserialize, Eq, JsonSchema, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum InputRole {
    Light,
    Flat,
    Dark,
    Bias,
    MasterFlat,
    MasterDark,
    MasterBias,
}

impl Validate for Project {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != PROJECT_SCHEMA_VERSION {
            return Err(ValidationError::new(
                "project.schemaVersion",
                format!(
                    "unsupported version {}; expected {PROJECT_SCHEMA_VERSION}",
                    self.schema_version
                ),
            ));
        }
        validate_identifier("project.projectId", &self.project_id)?;
        if self.display_name.trim().is_empty() {
            return Err(ValidationError::new(
                "project.displayName",
                "must not be blank",
            ));
        }
        if self.sources.is_empty() {
            return Err(ValidationError::new(
                "project.sources",
                "at least one source is required",
            ));
        }
        let mut source_ids = BTreeSet::new();
        for (index, source) in self.sources.iter().enumerate() {
            let path = format!("project.sources[{index}]");
            validate_identifier(&format!("{path}.sourceId"), &source.source_id)?;
            if !source_ids.insert(&source.source_id) {
                return Err(ValidationError::new(
                    format!("{path}.sourceId"),
                    "duplicate source identifier",
                ));
            }
            if source.host_path.trim().is_empty() || source.host_path.contains('\0') {
                return Err(ValidationError::new(
                    format!("{path}.hostPath"),
                    "must be a non-empty native path without NUL characters",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Recipe {
    pub schema_version: u16,
    pub recipe_id: String,
    pub display_name: String,
    pub stages: Vec<StageSpec>,
    pub solver: SolverSettings,
    pub drizzle: DrizzleSettings,
    #[serde(default)]
    pub parameters: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StageSpec {
    pub stage_id: String,
    pub kind: StageKind,
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub parameters: BTreeMap<String, serde_json::Value>,
}

const fn enabled_by_default() -> bool {
    true
}

#[derive(
    Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum StageKind {
    Ingest,
    QualityControl,
    Calibration,
    CosmeticCorrection,
    Debayer,
    Registration,
    LocalNormalization,
    Integration,
    Drizzle,
    AstrometricSolve,
    Mosaic,
    Export,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResultRequirement {
    Disabled,
    BestEffort,
    Required,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SolverSettings {
    pub result: ResultRequirement,
    #[serde(default = "default_solver_catalog")]
    pub catalog: String,
    #[serde(default = "default_solver_projection")]
    pub projection: String,
    #[serde(default = "default_solver_minimum_matches")]
    pub minimum_matches: u32,
    #[serde(default = "default_solver_maximum_rms")]
    pub maximum_rms_arcsec: f64,
}

fn default_solver_catalog() -> String {
    "gaia-dr3-offline".to_owned()
}

fn default_solver_projection() -> String {
    "TAN".to_owned()
}

const fn default_solver_minimum_matches() -> u32 {
    12
}

const fn default_solver_maximum_rms() -> f64 {
    2.0
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            result: ResultRequirement::Required,
            catalog: default_solver_catalog(),
            projection: default_solver_projection(),
            minimum_matches: default_solver_minimum_matches(),
            maximum_rms_arcsec: default_solver_maximum_rms(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DrizzleSettings {
    pub result: ResultRequirement,
    #[serde(default = "default_drizzle_scale")]
    pub scale: f32,
    #[serde(default = "default_drop_shrink")]
    pub drop_shrink: f32,
    #[serde(default = "default_drizzle_kernel")]
    pub kernel: String,
}

const fn default_drizzle_scale() -> f32 {
    2.0
}

const fn default_drop_shrink() -> f32 {
    0.9
}

fn default_drizzle_kernel() -> String {
    "square".to_owned()
}

impl Default for DrizzleSettings {
    fn default() -> Self {
        Self {
            result: ResultRequirement::Disabled,
            scale: default_drizzle_scale(),
            drop_shrink: default_drop_shrink(),
            kernel: default_drizzle_kernel(),
        }
    }
}

impl Validate for Recipe {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != RECIPE_SCHEMA_VERSION {
            return Err(ValidationError::new(
                "recipe.schemaVersion",
                format!(
                    "unsupported version {}; expected {RECIPE_SCHEMA_VERSION}",
                    self.schema_version
                ),
            ));
        }
        validate_identifier("recipe.recipeId", &self.recipe_id)?;
        if self.display_name.trim().is_empty() {
            return Err(ValidationError::new(
                "recipe.displayName",
                "must not be blank",
            ));
        }
        if self.stages.is_empty() {
            return Err(ValidationError::new("recipe.stages", "must not be empty"));
        }

        let mut by_id = BTreeMap::new();
        for (index, stage) in self.stages.iter().enumerate() {
            validate_identifier(&format!("recipe.stages[{index}].stageId"), &stage.stage_id)?;
            if by_id.insert(stage.stage_id.as_str(), stage).is_some() {
                return Err(ValidationError::new(
                    format!("recipe.stages[{index}].stageId"),
                    "duplicate stage identifier",
                ));
            }
        }
        for (index, stage) in self.stages.iter().enumerate() {
            let mut unique_dependencies = BTreeSet::new();
            for dependency in &stage.depends_on {
                if !unique_dependencies.insert(dependency) {
                    return Err(ValidationError::new(
                        format!("recipe.stages[{index}].dependsOn"),
                        "duplicate dependency",
                    ));
                }
                let Some(dependency_stage) = by_id.get(dependency.as_str()) else {
                    return Err(ValidationError::new(
                        format!("recipe.stages[{index}].dependsOn"),
                        format!("unknown stage {dependency}"),
                    ));
                };
                if stage.enabled && !dependency_stage.enabled {
                    return Err(ValidationError::new(
                        format!("recipe.stages[{index}].dependsOn"),
                        format!("enabled stage depends on disabled stage {dependency}"),
                    ));
                }
            }
        }
        ensure_acyclic(&self.stages, &by_id)?;

        validate_solver_settings(&self.solver)?;
        validate_drizzle_settings(&self.drizzle)?;
        require_stage_for_result(
            self,
            StageKind::AstrometricSolve,
            self.solver.result,
            "solver.result",
        )?;
        require_stage_for_result(
            self,
            StageKind::Drizzle,
            self.drizzle.result,
            "drizzle.result",
        )?;
        Ok(())
    }
}

fn validate_solver_settings(settings: &SolverSettings) -> Result<(), ValidationError> {
    if settings.result != ResultRequirement::Disabled {
        if settings.catalog.trim().is_empty() {
            return Err(ValidationError::new(
                "recipe.solver.catalog",
                "must not be blank",
            ));
        }
        if settings.projection.trim().is_empty() {
            return Err(ValidationError::new(
                "recipe.solver.projection",
                "must not be blank",
            ));
        }
        if settings.minimum_matches < 3 {
            return Err(ValidationError::new(
                "recipe.solver.minimumMatches",
                "must be at least 3",
            ));
        }
        if !settings.maximum_rms_arcsec.is_finite() || settings.maximum_rms_arcsec <= 0.0 {
            return Err(ValidationError::new(
                "recipe.solver.maximumRmsArcsec",
                "must be finite and positive",
            ));
        }
    }
    Ok(())
}

fn validate_drizzle_settings(settings: &DrizzleSettings) -> Result<(), ValidationError> {
    if settings.result != ResultRequirement::Disabled {
        if !settings.scale.is_finite() || settings.scale < 1.0 || settings.scale > 4.0 {
            return Err(ValidationError::new(
                "recipe.drizzle.scale",
                "must be finite and in [1, 4]",
            ));
        }
        if !settings.drop_shrink.is_finite()
            || settings.drop_shrink <= 0.0
            || settings.drop_shrink > 1.0
        {
            return Err(ValidationError::new(
                "recipe.drizzle.dropShrink",
                "must be finite and in (0, 1]",
            ));
        }
        if settings.kernel.trim().is_empty() {
            return Err(ValidationError::new(
                "recipe.drizzle.kernel",
                "must not be blank",
            ));
        }
    }
    Ok(())
}

fn require_stage_for_result(
    recipe: &Recipe,
    kind: StageKind,
    requirement: ResultRequirement,
    path: &str,
) -> Result<(), ValidationError> {
    if requirement != ResultRequirement::Disabled
        && !recipe
            .stages
            .iter()
            .any(|stage| stage.enabled && stage.kind == kind)
    {
        return Err(ValidationError::new(
            format!("recipe.{path}"),
            format!("requires an enabled {kind:?} stage"),
        ));
    }
    Ok(())
}

fn ensure_acyclic<'a>(
    stages: &'a [StageSpec],
    by_id: &BTreeMap<&'a str, &'a StageSpec>,
) -> Result<(), ValidationError> {
    fn visit<'a>(
        stage: &'a StageSpec,
        by_id: &BTreeMap<&'a str, &'a StageSpec>,
        visiting: &mut BTreeSet<&'a str>,
        visited: &mut BTreeSet<&'a str>,
    ) -> Result<(), ValidationError> {
        if visited.contains(stage.stage_id.as_str()) {
            return Ok(());
        }
        if !visiting.insert(stage.stage_id.as_str()) {
            return Err(ValidationError::new(
                "recipe.stages",
                format!("dependency cycle includes {}", stage.stage_id),
            ));
        }
        for dependency in &stage.depends_on {
            visit(by_id[dependency.as_str()], by_id, visiting, visited)?;
        }
        visiting.remove(stage.stage_id.as_str());
        visited.insert(stage.stage_id.as_str());
        Ok(())
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for stage in stages {
        visit(stage, by_id, &mut visiting, &mut visited)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Planned,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// Immutable identity binding for the project snapshot consumed by a run.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProjectReceipt {
    pub schema_version: u16,
    pub project_id: String,
    pub project_sha256: String,
    pub input_manifest_sha256: String,
    pub resolved_source_count: u64,
    pub captured_at_unix_ms: u64,
}

/// Immutable identity binding for the recipe snapshot consumed by a run.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RecipeReceipt {
    pub schema_version: u16,
    pub recipe_id: String,
    pub recipe_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StageStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Skipped,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StageReceipt {
    pub schema_version: u16,
    pub stage_id: String,
    pub kind: StageKind,
    pub status: StageStatus,
    pub started_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at_unix_ms: Option<u64>,
    #[serde(default)]
    pub artifact_ids: Vec<String>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ReceiptError>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReceiptError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(
    Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactKind {
    FrameManifest,
    QualityReport,
    MasterBias,
    MasterDark,
    MasterFlat,
    CalibratedLight,
    RegisteredLight,
    LocalNormalizationModel,
    DrizzleData,
    IntegrationMaster,
    DrizzledMaster,
    SolvedMaster,
    MosaicMaster,
    FinalMaster,
    RunLog,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactDesignation {
    Intermediate,
    Diagnostic,
    FinalMaster,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactReceipt {
    pub schema_version: u16,
    pub artifact_id: String,
    pub produced_by_stage_id: String,
    pub kind: ArtifactKind,
    pub designation: ArtifactDesignation,
    pub relative_path: SafeRelativePath,
    pub media_type: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub created_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub astrometry: Option<AstrometricSolutionReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drizzle: Option<DrizzleReceipt>,
    #[serde(default)]
    pub attributes: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AstrometricSolutionReceipt {
    #[schemars(length(min = 1))]
    pub reference_frame: String,
    #[schemars(length(min = 1))]
    pub projection: String,
    pub center_ra_degrees: f64,
    pub center_dec_degrees: f64,
    #[schemars(range(min = 0.0))]
    pub pixel_scale_arcsec: f64,
    pub rotation_degrees: f64,
    #[schemars(range(min = 0.0))]
    pub rms_pixels: f64,
    #[schemars(range(min = 0.0))]
    pub rms_arcsec: f64,
    #[schemars(range(min = 3))]
    pub matched_stars: u32,
    pub parity: AstrometricParity,
    /// SHA-256 identity of the exact catalog rows/release used for the solve.
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
    pub catalog_identity: String,
    /// Backend-native index identifiers (for example INDEXID/healpix tuples).
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub index_identities: Vec<String>,
    /// Digest of the correspondence table from which match/RMS evidence was recomputed.
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
    pub correspondence_sha256: String,
    /// True only when the solver INDEXID is bound to app-managed index bytes.
    pub catalog_managed: bool,
    /// Identity of the immutable installed-set receipt used for this solution.
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
    pub installed_set_identity: String,
    /// Digest of the checked catalog manifest that authorized those bytes.
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
    pub catalog_manifest_sha256: String,
    /// Exact managed index artifacts selected by the backend's match table.
    #[schemars(length(min = 1))]
    pub index_artifacts: Vec<SolverIndexArtifactReceipt>,
    /// Digest of the canonical WCS card set embedded in the artifact.
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
    pub wcs_sha256: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SolverIndexArtifactReceipt {
    #[schemars(regex(pattern = "^[0-9]+$"))]
    pub index_id: String,
    #[schemars(length(min = 1))]
    pub relative_name: String,
    #[schemars(range(min = 1))]
    pub size_bytes: u64,
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
    pub sha256: String,
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
    pub manifest_sha256: String,
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
    pub installed_set_identity: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AstrometricParity {
    Positive,
    Negative,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DrizzleReceipt {
    pub scale: f32,
    pub drop_shrink: f32,
    pub kernel: String,
    pub input_frames: u32,
    pub output_width: u32,
    pub output_height: u32,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RunReceipt {
    pub schema_version: u16,
    pub run_id: String,
    pub plan_id: String,
    pub project: ProjectReceipt,
    pub recipe: RecipeReceipt,
    pub backend_id: String,
    pub backend_version: String,
    pub hardware_profile: HardwareProfile,
    pub status: RunStatus,
    pub started_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at_unix_ms: Option<u64>,
    pub stages: Vec<StageReceipt>,
    pub artifacts: Vec<ArtifactReceipt>,
}

impl Validate for ProjectReceipt {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(ValidationError::new(
                "projectReceipt.schemaVersion",
                format!(
                    "unsupported version {}; expected {RECEIPT_SCHEMA_VERSION}",
                    self.schema_version
                ),
            ));
        }
        validate_identifier("projectReceipt.projectId", &self.project_id)?;
        validate_sha256("projectReceipt.projectSha256", &self.project_sha256)?;
        validate_sha256(
            "projectReceipt.inputManifestSha256",
            &self.input_manifest_sha256,
        )?;
        if self.resolved_source_count == 0 {
            return Err(ValidationError::new(
                "projectReceipt.resolvedSourceCount",
                "must be positive",
            ));
        }
        Ok(())
    }
}

impl Validate for RecipeReceipt {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(ValidationError::new(
                "recipeReceipt.schemaVersion",
                format!(
                    "unsupported version {}; expected {RECEIPT_SCHEMA_VERSION}",
                    self.schema_version
                ),
            ));
        }
        validate_identifier("recipeReceipt.recipeId", &self.recipe_id)?;
        validate_sha256("recipeReceipt.recipeSha256", &self.recipe_sha256)
    }
}

impl Validate for StageReceipt {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(ValidationError::new(
                "stageReceipt.schemaVersion",
                format!(
                    "unsupported version {}; expected {RECEIPT_SCHEMA_VERSION}",
                    self.schema_version
                ),
            ));
        }
        validate_identifier("stageReceipt.stageId", &self.stage_id)?;
        if let Some(finished) = self.finished_at_unix_ms {
            if finished < self.started_at_unix_ms {
                return Err(ValidationError::new(
                    "stageReceipt.finishedAtUnixMs",
                    "must not precede start time",
                ));
            }
        }
        let is_terminal = matches!(
            self.status,
            StageStatus::Succeeded
                | StageStatus::Failed
                | StageStatus::Skipped
                | StageStatus::Cancelled
        );
        if is_terminal != self.finished_at_unix_ms.is_some() {
            return Err(ValidationError::new(
                "stageReceipt.finishedAtUnixMs",
                "must be present exactly when the stage is terminal",
            ));
        }
        if self.status == StageStatus::Failed && self.error.is_none() {
            return Err(ValidationError::new(
                "stageReceipt.error",
                "failed stage must include an error",
            ));
        }
        if self.status != StageStatus::Failed && self.error.is_some() {
            return Err(ValidationError::new(
                "stageReceipt.error",
                "only a failed stage may include an error",
            ));
        }
        if self.metrics.values().any(|value| !value.is_finite()) {
            return Err(ValidationError::new(
                "stageReceipt.metrics",
                "metric values must be finite",
            ));
        }
        Ok(())
    }
}

impl Validate for ArtifactReceipt {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(ValidationError::new(
                "artifactReceipt.schemaVersion",
                format!(
                    "unsupported version {}; expected {RECEIPT_SCHEMA_VERSION}",
                    self.schema_version
                ),
            ));
        }
        validate_identifier("artifactReceipt.artifactId", &self.artifact_id)?;
        validate_identifier(
            "artifactReceipt.producedByStageId",
            &self.produced_by_stage_id,
        )?;
        validate_sha256("artifactReceipt.sha256", &self.sha256)?;
        if self.size_bytes == 0 {
            return Err(ValidationError::new(
                "artifactReceipt.sizeBytes",
                "must be positive",
            ));
        }
        if self.media_type.trim().is_empty() || !self.media_type.contains('/') {
            return Err(ValidationError::new(
                "artifactReceipt.mediaType",
                "must be a non-empty media type",
            ));
        }
        if self.designation == ArtifactDesignation::FinalMaster {
            if self.kind != ArtifactKind::FinalMaster {
                return Err(ValidationError::new(
                    "artifactReceipt.kind",
                    "a final-master designation requires kind=final-master",
                ));
            }
            if !matches!(
                self.media_type.to_ascii_lowercase().as_str(),
                "image/fits" | "application/fits"
            ) {
                return Err(ValidationError::new(
                    "artifactReceipt.mediaType",
                    "a final master must be a FITS image",
                ));
            }
            let path = self.relative_path.as_str().to_ascii_lowercase();
            let extension = path.rsplit_once('.').map_or("", |(_, value)| value);
            if !matches!(extension, "fit" | "fits" | "fts") {
                return Err(ValidationError::new(
                    "artifactReceipt.relativePath",
                    "a final master must use a FITS filename extension",
                ));
            }
        } else if self.kind == ArtifactKind::FinalMaster {
            return Err(ValidationError::new(
                "artifactReceipt.designation",
                "kind=final-master requires designation=final-master",
            ));
        }
        if let Some(astrometry) = &self.astrometry {
            astrometry.validate()?;
        }
        if let Some(provenance) = self
            .attributes
            .get("astrometryProvenanceType")
            .and_then(serde_json::Value::as_str)
        {
            let fresh = self
                .attributes
                .get("freshSolveOnThisArtifactGrid")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| {
                    ValidationError::new(
                        "artifactReceipt.attributes.freshSolveOnThisArtifactGrid",
                        "astrometry provenance requires an explicit boolean fresh-solve declaration",
                    )
                })?;
            match provenance {
                "FRESH_SOLVE_UNCHANGED_GRID" if fresh => {}
                "PROPAGATED_VERIFIED" if !fresh => {}
                "FRESH_SOLVE_UNCHANGED_GRID" | "PROPAGATED_VERIFIED" => {
                    return Err(ValidationError::new(
                        "artifactReceipt.attributes.astrometryProvenanceType",
                        "fresh-solve declaration disagrees with astrometry provenance",
                    ));
                }
                _ => {
                    return Err(ValidationError::new(
                        "artifactReceipt.attributes.astrometryProvenanceType",
                        "unsupported astrometry provenance type",
                    ));
                }
            }
        }
        if let Some(drizzle) = &self.drizzle {
            drizzle.validate()?;
        }
        Ok(())
    }
}

impl Validate for AstrometricSolutionReceipt {
    // Keep the publish gate linear: every catalog, correspondence, and WCS
    // invariant is checked in the same fail-closed order as the serialized
    // receipt. Splitting this into partially reusable helpers would make it
    // easier to call an incomplete subset at another publication boundary.
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<(), ValidationError> {
        if self.reference_frame.trim().is_empty() || self.projection.trim().is_empty() {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry",
                "reference frame and projection must not be blank",
            ));
        }
        if !self.center_ra_degrees.is_finite()
            || !(0.0..360.0).contains(&self.center_ra_degrees)
            || !self.center_dec_degrees.is_finite()
            || !(-90.0..=90.0).contains(&self.center_dec_degrees)
        {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.center",
                "RA must be in [0, 360) and declination in [-90, 90]",
            ));
        }
        if !self.pixel_scale_arcsec.is_finite() || self.pixel_scale_arcsec <= 0.0 {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.pixelScaleArcsec",
                "must be finite and positive",
            ));
        }
        if !self.rotation_degrees.is_finite() {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.rotationDegrees",
                "must be finite",
            ));
        }
        if !self.rms_pixels.is_finite()
            || self.rms_pixels < 0.0
            || !self.rms_arcsec.is_finite()
            || self.rms_arcsec < 0.0
        {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.rms",
                "pixel and angular RMS must be finite and non-negative",
            ));
        }
        let expected_arcsec = self.rms_pixels * self.pixel_scale_arcsec;
        let rms_consistent = if expected_arcsec == 0.0 {
            self.rms_arcsec <= 1.0e-9
        } else {
            let ratio = self.rms_arcsec / expected_arcsec;
            (0.5..=2.0).contains(&ratio)
        };
        if !rms_consistent {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.rms",
                "rmsPixels and rmsArcsec must agree with pixelScaleArcsec",
            ));
        }
        if self.matched_stars < 3 {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.matchedStars",
                "must be at least 3",
            ));
        }
        validate_sha256(
            "artifactReceipt.astrometry.catalogIdentity",
            &self.catalog_identity,
        )?;
        if self.index_identities.is_empty()
            || self
                .index_identities
                .iter()
                .any(|identity| identity.trim().is_empty())
            || self.index_identities.iter().collect::<BTreeSet<_>>().len()
                != self.index_identities.len()
        {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.indexIdentities",
                "must contain one or more unique, non-blank index identities",
            ));
        }
        validate_sha256(
            "artifactReceipt.astrometry.correspondenceSha256",
            &self.correspondence_sha256,
        )?;
        if !self.catalog_managed {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.catalogManaged",
                "a publishable solution must bind an app-managed catalog",
            ));
        }
        validate_sha256(
            "artifactReceipt.astrometry.installedSetIdentity",
            &self.installed_set_identity,
        )?;
        validate_sha256(
            "artifactReceipt.astrometry.catalogManifestSha256",
            &self.catalog_manifest_sha256,
        )?;
        if self.index_artifacts.is_empty() {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.indexArtifacts",
                "must contain at least one managed index artifact",
            ));
        }
        let mut artifact_index_ids = BTreeSet::new();
        for artifact in &self.index_artifacts {
            if artifact.index_id.is_empty()
                || !artifact.index_id.bytes().all(|byte| byte.is_ascii_digit())
                || !artifact_index_ids.insert(artifact.index_id.as_str())
                || artifact.relative_name != format!("index-{}.fits", artifact.index_id)
                || artifact.size_bytes == 0
            {
                return Err(ValidationError::new(
                    "artifactReceipt.astrometry.indexArtifacts",
                    "index IDs must be unique decimals with their exact managed filename and nonzero size",
                ));
            }
            validate_sha256(
                "artifactReceipt.astrometry.indexArtifacts.sha256",
                &artifact.sha256,
            )?;
            if artifact.manifest_sha256 != self.catalog_manifest_sha256
                || artifact.installed_set_identity != self.installed_set_identity
            {
                return Err(ValidationError::new(
                    "artifactReceipt.astrometry.indexArtifacts",
                    "artifact manifest or installed-set identity disagrees with its parent receipt",
                ));
            }
        }
        let logical_index_ids = self
            .index_identities
            .iter()
            .filter_map(|identity| {
                let parts = identity.split(':').collect::<Vec<_>>();
                if parts.len() == 7
                    && parts[0] == "astrometry.net"
                    && parts[1] == "index"
                    && parts[2].bytes().all(|byte| byte.is_ascii_digit())
                    && parts[3] == "healpix"
                    && parts[5] == "hpnside"
                {
                    Some(parts[2])
                } else {
                    None
                }
            })
            .collect::<BTreeSet<_>>();
        if logical_index_ids.len() != self.index_identities.len()
            || logical_index_ids != artifact_index_ids
        {
            return Err(ValidationError::new(
                "artifactReceipt.astrometry.indexArtifacts",
                "managed artifact INDEXIDs must exactly match indexIdentities",
            ));
        }
        validate_sha256("artifactReceipt.astrometry.wcsSha256", &self.wcs_sha256)
    }
}

impl Validate for DrizzleReceipt {
    fn validate(&self) -> Result<(), ValidationError> {
        if !self.scale.is_finite() || self.scale < 1.0 || self.scale > 4.0 {
            return Err(ValidationError::new(
                "artifactReceipt.drizzle.scale",
                "must be finite and in [1, 4]",
            ));
        }
        if !self.drop_shrink.is_finite() || self.drop_shrink <= 0.0 || self.drop_shrink > 1.0 {
            return Err(ValidationError::new(
                "artifactReceipt.drizzle.dropShrink",
                "must be finite and in (0, 1]",
            ));
        }
        if self.kernel.trim().is_empty()
            || self.input_frames == 0
            || self.output_width == 0
            || self.output_height == 0
        {
            return Err(ValidationError::new(
                "artifactReceipt.drizzle",
                "kernel, input frame count, and output dimensions are required",
            ));
        }
        Ok(())
    }
}

impl Validate for RunReceipt {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(ValidationError::new(
                "runReceipt.schemaVersion",
                format!(
                    "unsupported version {}; expected {RECEIPT_SCHEMA_VERSION}",
                    self.schema_version
                ),
            ));
        }
        for (path, value) in [
            ("runReceipt.runId", &self.run_id),
            ("runReceipt.planId", &self.plan_id),
            ("runReceipt.backendId", &self.backend_id),
        ] {
            validate_identifier(path, value)?;
        }
        self.project.validate()?;
        self.recipe.validate()?;
        if self.backend_version.trim().is_empty() {
            return Err(ValidationError::new(
                "runReceipt.backend",
                "backend version must not be blank",
            ));
        }
        validate_run_timing(self)?;
        validate_run_stage_artifact_links(self)
    }
}

fn validate_run_timing(receipt: &RunReceipt) -> Result<(), ValidationError> {
    if let Some(finished) = receipt.finished_at_unix_ms {
        if finished < receipt.started_at_unix_ms {
            return Err(ValidationError::new(
                "runReceipt.finishedAtUnixMs",
                "must not precede start time",
            ));
        }
    }
    let terminal = matches!(
        receipt.status,
        RunStatus::Succeeded | RunStatus::Failed | RunStatus::Cancelled
    );
    if terminal != receipt.finished_at_unix_ms.is_some() {
        return Err(ValidationError::new(
            "runReceipt.finishedAtUnixMs",
            "must be present exactly when the run is terminal",
        ));
    }
    Ok(())
}

fn validate_run_stage_artifact_links(receipt: &RunReceipt) -> Result<(), ValidationError> {
    let mut stage_ids = BTreeSet::new();
    for stage in &receipt.stages {
        stage.validate()?;
        if !stage_ids.insert(stage.stage_id.as_str()) {
            return Err(ValidationError::new(
                "runReceipt.stages",
                "duplicate stage receipt",
            ));
        }
        if receipt.status == RunStatus::Succeeded
            && !matches!(stage.status, StageStatus::Succeeded | StageStatus::Skipped)
        {
            return Err(ValidationError::new(
                "runReceipt.stages.status",
                "a succeeded run may contain only succeeded or skipped stages",
            ));
        }
    }
    let mut artifacts_by_id = BTreeMap::new();
    for artifact in &receipt.artifacts {
        artifact.validate()?;
        if !stage_ids.contains(artifact.produced_by_stage_id.as_str()) {
            return Err(ValidationError::new(
                "runReceipt.artifacts.producedByStageId",
                format!("unknown stage {}", artifact.produced_by_stage_id),
            ));
        }
        if artifacts_by_id
            .insert(artifact.artifact_id.as_str(), artifact)
            .is_some()
        {
            return Err(ValidationError::new(
                "runReceipt.artifacts",
                "duplicate artifact receipt",
            ));
        }
    }
    for stage in &receipt.stages {
        for artifact_id in &stage.artifact_ids {
            let Some(artifact) = artifacts_by_id.get(artifact_id.as_str()) else {
                return Err(ValidationError::new(
                    "runReceipt.stages.artifactIds",
                    format!("unknown artifact {artifact_id}"),
                ));
            };
            if artifact.produced_by_stage_id != stage.stage_id {
                return Err(ValidationError::new(
                    "runReceipt.stages.artifactIds",
                    format!("artifact {artifact_id} belongs to another stage"),
                ));
            }
        }
    }
    for artifact in &receipt.artifacts {
        let producing_stage = receipt
            .stages
            .iter()
            .find(|stage| stage.stage_id == artifact.produced_by_stage_id)
            .expect("producing stage existence checked above");
        if !producing_stage.artifact_ids.contains(&artifact.artifact_id) {
            return Err(ValidationError::new(
                "runReceipt.artifacts",
                format!(
                    "artifact {} is not listed by its producing stage",
                    artifact.artifact_id
                ),
            ));
        }
    }
    Ok(())
}
