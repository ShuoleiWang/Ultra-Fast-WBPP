//! The astrometric evidence of a solved product, as the engine's run receipt
//! records it.  The desktop re-validates it before reporting success.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// The first invalid field of a receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ValidationError {
    path: String,
    message: String,
}

impl ValidationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.path, self.message)
    }
}

fn validate_sha256(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ValidationError::new(
            path,
            "must be 64 lowercase hexadecimal SHA-256 characters",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AstrometricSolutionReceipt {
    pub(crate) reference_frame: String,
    pub(crate) projection: String,
    pub(crate) center_ra_degrees: f64,
    pub(crate) center_dec_degrees: f64,
    pub(crate) pixel_scale_arcsec: f64,
    pub(crate) rotation_degrees: f64,
    pub(crate) rms_pixels: f64,
    pub(crate) rms_arcsec: f64,
    pub(crate) matched_stars: u32,
    pub(crate) parity: AstrometricParity,
    /// SHA-256 identity of the exact catalog rows/release used for the solve.
    pub(crate) catalog_identity: String,
    /// Backend-native index identifiers (for example INDEXID/healpix tuples).
    pub(crate) index_identities: Vec<String>,
    /// Digest of the correspondence table from which match/RMS evidence was recomputed.
    pub(crate) correspondence_sha256: String,
    /// True only when the solver INDEXID is bound to app-managed index bytes.
    pub(crate) catalog_managed: bool,
    /// Identity of the immutable installed-set receipt used for this solution.
    pub(crate) installed_set_identity: String,
    /// Digest of the checked catalog manifest that authorized those bytes.
    pub(crate) catalog_manifest_sha256: String,
    /// Exact managed index artifacts selected by the backend's match table.
    pub(crate) index_artifacts: Vec<SolverIndexArtifactReceipt>,
    /// Digest of the canonical WCS card set embedded in the artifact.
    pub(crate) wcs_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SolverIndexArtifactReceipt {
    pub(crate) index_id: String,
    pub(crate) relative_name: String,
    pub(crate) size_bytes: u64,
    pub(crate) sha256: String,
    pub(crate) manifest_sha256: String,
    pub(crate) installed_set_identity: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum AstrometricParity {
    Positive,
    Negative,
}

impl AstrometricSolutionReceipt {
    // Keep the publish gate linear: every catalog, correspondence, and WCS
    // invariant is checked in the same fail-closed order as the serialized
    // receipt. Splitting this into partially reusable helpers would make it
    // easier to call an incomplete subset at another publication boundary.
    #[allow(clippy::too_many_lines)]
    /// Checks the invariants the engine's receipt must satisfy before the
    /// desktop reports a solved product.
    pub(crate) fn validate(&self) -> Result<(), ValidationError> {
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
