use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::{Recipe, ResultRequirement, StageKind};
use crate::validation::{Validate, ValidationError, validate_identifier};

pub const CAPABILITIES_SCHEMA_VERSION: u16 = 1;

/// Stable execution profiles. Profiles describe dispatch contracts, not chip names.
#[derive(
    Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum HardwareProfile {
    /// Portable non-Windows CPU implementation, including Linux x86-64 CI.
    PortableCpu,
    /// Portable `AArch64` CPU implementation for every Apple M-series generation
    /// and other compatible Arm64 hosts.
    GenericArm64Cpu,
    /// Portable Metal kernels for every Apple Silicon M-series generation.
    GenericAppleMetal,
    /// Aggressively tuned Metal/CPU scheduling for Apple M3 Pro only.
    M3ProTuned,
    /// Portable Windows CPU implementation (x86-64 or Arm64).
    WindowsCpu,
}

impl HardwareProfile {
    #[must_use]
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::PortableCpu => "portable-cpu",
            Self::GenericArm64Cpu => "generic-arm64-cpu",
            Self::GenericAppleMetal => "generic-apple-metal",
            Self::M3ProTuned => "m3-pro-tuned",
            Self::WindowsCpu => "windows-cpu",
        }
    }

    /// Select the fastest compatible profile advertised by a backend.
    ///
    /// An unknown Apple chip never selects the M3 Pro path. It safely falls
    /// back to generic Metal, then generic Arm64 CPU.
    #[must_use]
    pub fn select(host: &HostPlatform, supported: &BTreeSet<Self>) -> Option<Self> {
        if host.is_apple_silicon() {
            if host.is_m3_pro() && host.metal_available && supported.contains(&Self::M3ProTuned) {
                return Some(Self::M3ProTuned);
            }
            if host.metal_available && supported.contains(&Self::GenericAppleMetal) {
                return Some(Self::GenericAppleMetal);
            }
            if supported.contains(&Self::GenericArm64Cpu) {
                return Some(Self::GenericArm64Cpu);
            }
        }
        if host.os == OperatingSystem::Windows {
            return supported
                .contains(&Self::WindowsCpu)
                .then_some(Self::WindowsCpu);
        }
        if host.architecture == Architecture::Aarch64 && supported.contains(&Self::GenericArm64Cpu)
        {
            return Some(Self::GenericArm64Cpu);
        }
        if supported.contains(&Self::PortableCpu)
            && !matches!(host.architecture, Architecture::Other)
        {
            return Some(Self::PortableCpu);
        }
        None
    }

    #[must_use]
    pub fn is_compatible_with(self, host: &HostPlatform) -> bool {
        match self {
            Self::PortableCpu => {
                host.os != OperatingSystem::Windows
                    && !matches!(host.architecture, Architecture::Other)
            }
            Self::GenericArm64Cpu => host.architecture == Architecture::Aarch64,
            Self::GenericAppleMetal => host.is_apple_silicon() && host.metal_available,
            Self::M3ProTuned => host.is_m3_pro() && host.metal_available,
            Self::WindowsCpu => host.os == OperatingSystem::Windows,
        }
    }
}

impl std::fmt::Display for HardwareProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.wire_name())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum OperatingSystem {
    Macos,
    Windows,
    Linux,
    Other,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Architecture {
    Aarch64,
    X86_64,
    Other,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HostPlatform {
    pub os: OperatingSystem,
    pub architecture: Architecture,
    pub metal_available: bool,
    /// Human-readable chip label obtained by the platform adapter, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apple_chip: Option<String>,
}

impl HostPlatform {
    #[must_use]
    pub fn current_compile_target() -> Self {
        let os = if cfg!(target_os = "macos") {
            OperatingSystem::Macos
        } else if cfg!(target_os = "windows") {
            OperatingSystem::Windows
        } else if cfg!(target_os = "linux") {
            OperatingSystem::Linux
        } else {
            OperatingSystem::Other
        };
        let architecture = if cfg!(target_arch = "aarch64") {
            Architecture::Aarch64
        } else if cfg!(target_arch = "x86_64") {
            Architecture::X86_64
        } else {
            Architecture::Other
        };
        Self {
            os,
            architecture,
            // Availability and chip identity are runtime facts. The generic
            // control plane does not guess them from the compile target.
            metal_available: false,
            apple_chip: None,
        }
    }

    #[must_use]
    pub const fn is_apple_silicon(&self) -> bool {
        matches!(self.os, OperatingSystem::Macos)
            && matches!(self.architecture, Architecture::Aarch64)
    }

    #[must_use]
    pub fn is_m3_pro(&self) -> bool {
        self.is_apple_silicon()
            && self.apple_chip.as_ref().is_some_and(|chip| {
                let normalized: String = chip
                    .chars()
                    .filter(|character| !character.is_ascii_whitespace() && *character != '-')
                    .flat_map(char::to_lowercase)
                    .collect();
                normalized.contains("applem3pro") || normalized == "m3pro"
            })
    }
}

#[derive(
    Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum BackendFeature {
    CpuExecution,
    MetalExecution,
    M3ProTuning,
    CheckpointResume,
    DeterministicReceipts,
    OfflineAstrometricSolver,
    Drizzle,
    Mosaic,
    Fits,
    Xisf,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BackendCapabilities {
    pub schema_version: u16,
    pub backend_id: String,
    pub backend_version: String,
    pub worker_build: String,
    pub hardware_profiles: BTreeSet<HardwareProfile>,
    pub stages: BTreeSet<StageKind>,
    pub features: BTreeSet<BackendFeature>,
    pub maximum_parallel_stages: u16,
    #[serde(default)]
    pub input_extensions: BTreeSet<String>,
    #[serde(default)]
    pub output_extensions: BTreeSet<String>,
}

impl Validate for BackendCapabilities {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != CAPABILITIES_SCHEMA_VERSION {
            return Err(ValidationError::new(
                "backendCapabilities.schemaVersion",
                format!(
                    "unsupported version {}; expected {CAPABILITIES_SCHEMA_VERSION}",
                    self.schema_version
                ),
            ));
        }
        validate_identifier("backendCapabilities.backendId", &self.backend_id)?;
        if self.backend_version.trim().is_empty() || self.worker_build.trim().is_empty() {
            return Err(ValidationError::new(
                "backendCapabilities",
                "backend version and worker build must not be blank",
            ));
        }
        if self.hardware_profiles.is_empty() || self.stages.is_empty() {
            return Err(ValidationError::new(
                "backendCapabilities",
                "at least one hardware profile and stage are required",
            ));
        }
        if self.maximum_parallel_stages == 0 {
            return Err(ValidationError::new(
                "backendCapabilities.maximumParallelStages",
                "must be positive",
            ));
        }
        for extension in self
            .input_extensions
            .iter()
            .chain(self.output_extensions.iter())
        {
            if extension.is_empty()
                || extension.starts_with('.')
                || !extension
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            {
                return Err(ValidationError::new(
                    "backendCapabilities.extensions",
                    "extensions must be lowercase alphanumeric without a leading dot",
                ));
            }
        }
        for profile in &self.hardware_profiles {
            let required = match profile {
                HardwareProfile::PortableCpu
                | HardwareProfile::GenericArm64Cpu
                | HardwareProfile::WindowsCpu => BackendFeature::CpuExecution,
                HardwareProfile::GenericAppleMetal => BackendFeature::MetalExecution,
                HardwareProfile::M3ProTuned => {
                    if !self.features.contains(&BackendFeature::MetalExecution) {
                        return Err(ValidationError::new(
                            "backendCapabilities.hardwareProfiles",
                            "m3-pro-tuned also requires metal-execution",
                        ));
                    }
                    BackendFeature::M3ProTuning
                }
            };
            if !self.features.contains(&required) {
                return Err(ValidationError::new(
                    "backendCapabilities.hardwareProfiles",
                    format!("{profile} requires feature {required:?}"),
                ));
            }
        }
        Ok(())
    }
}

impl BackendCapabilities {
    /// Verify that this backend can execute every enabled stage and required
    /// result in a recipe with the selected hardware profile.
    ///
    /// # Errors
    ///
    /// Returns [`CapabilityError`] when the capability document or recipe is
    /// invalid, or when a profile, stage, or required feature is unsupported.
    pub fn check_recipe(
        &self,
        recipe: &Recipe,
        profile: HardwareProfile,
    ) -> Result<(), CapabilityError> {
        self.validate()
            .map_err(CapabilityError::InvalidCapabilities)?;
        recipe.validate().map_err(CapabilityError::InvalidRecipe)?;
        if !self.hardware_profiles.contains(&profile) {
            return Err(CapabilityError::UnsupportedProfile(profile));
        }
        for stage in recipe.stages.iter().filter(|stage| stage.enabled) {
            if !self.stages.contains(&stage.kind) {
                return Err(CapabilityError::UnsupportedStage(stage.kind));
            }
        }
        if recipe.solver.result != ResultRequirement::Disabled
            && !self
                .features
                .contains(&BackendFeature::OfflineAstrometricSolver)
        {
            return Err(CapabilityError::MissingFeature(
                BackendFeature::OfflineAstrometricSolver,
            ));
        }
        if recipe.drizzle.result != ResultRequirement::Disabled
            && !self.features.contains(&BackendFeature::Drizzle)
        {
            return Err(CapabilityError::MissingFeature(BackendFeature::Drizzle));
        }
        Ok(())
    }

    /// Verify both backend support and compatibility with the detected host.
    ///
    /// # Errors
    ///
    /// Returns [`CapabilityError`] for every error from [`Self::check_recipe`]
    /// and when the requested profile is incompatible with the actual host.
    pub fn check_dispatch(
        &self,
        recipe: &Recipe,
        profile: HardwareProfile,
        host: &HostPlatform,
    ) -> Result<(), CapabilityError> {
        self.check_recipe(recipe, profile)?;
        if !profile.is_compatible_with(host) {
            return Err(CapabilityError::IncompatibleHost(profile));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CapabilityError {
    #[error("invalid backend capabilities: {0}")]
    InvalidCapabilities(ValidationError),
    #[error("invalid recipe: {0}")]
    InvalidRecipe(ValidationError),
    #[error("hardware profile is not supported: {0}")]
    UnsupportedProfile(HardwareProfile),
    #[error("hardware profile is incompatible with the detected host: {0}")]
    IncompatibleHost(HardwareProfile),
    #[error("stage is not supported: {0:?}")]
    UnsupportedStage(StageKind),
    #[error("required backend feature is missing: {0:?}")]
    MissingFeature(BackendFeature),
}
