#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use openastroflow_app_core::{
    ArtifactDesignation, ArtifactKind, ArtifactReceipt, AstrometricParity,
    AstrometricSolutionReceipt, BackendCapabilities, BackendFeature, DrizzleReceipt,
    DrizzleSettings, HardwareProfile, Recipe, ResultRequirement, SafeRelativePath,
    SolverIndexArtifactReceipt, SolverSettings, StageKind, StageReceipt, StageSpec, StageStatus,
};

pub fn recipe(solver: ResultRequirement, drizzle: ResultRequirement) -> Recipe {
    let mut stages = vec![StageSpec {
        stage_id: "integrate".to_owned(),
        kind: StageKind::Integration,
        enabled: true,
        depends_on: Vec::new(),
        parameters: BTreeMap::new(),
    }];
    let mut last = "integrate".to_owned();
    if drizzle != ResultRequirement::Disabled {
        stages.push(StageSpec {
            stage_id: "drizzle".to_owned(),
            kind: StageKind::Drizzle,
            enabled: true,
            depends_on: vec![last.clone()],
            parameters: BTreeMap::new(),
        });
        "drizzle".clone_into(&mut last);
    }
    if solver != ResultRequirement::Disabled {
        stages.push(StageSpec {
            stage_id: "solve".to_owned(),
            kind: StageKind::AstrometricSolve,
            enabled: true,
            depends_on: vec![last],
            parameters: BTreeMap::new(),
        });
    }
    Recipe {
        schema_version: 1,
        recipe_id: "e2e-default".to_owned(),
        display_name: "End to end".to_owned(),
        stages,
        solver: SolverSettings {
            result: solver,
            ..SolverSettings::default()
        },
        drizzle: DrizzleSettings {
            result: drizzle,
            ..DrizzleSettings::default()
        },
        parameters: BTreeMap::new(),
    }
}

pub fn stage(stage_id: &str, kind: StageKind) -> StageReceipt {
    StageReceipt {
        schema_version: 1,
        stage_id: stage_id.to_owned(),
        kind,
        status: StageStatus::Succeeded,
        started_at_unix_ms: 10,
        finished_at_unix_ms: Some(20),
        artifact_ids: vec!["master-rgb".to_owned()],
        metrics: BTreeMap::new(),
        error: None,
    }
}

pub fn final_master() -> ArtifactReceipt {
    ArtifactReceipt {
        schema_version: 1,
        artifact_id: "master-rgb".to_owned(),
        produced_by_stage_id: "solve".to_owned(),
        kind: ArtifactKind::FinalMaster,
        designation: ArtifactDesignation::FinalMaster,
        relative_path: SafeRelativePath::new("master/final-rgb.fits").expect("safe fixture"),
        media_type: "image/fits".to_owned(),
        sha256: "a".repeat(64),
        size_bytes: 1234,
        created_at_unix_ms: 20,
        astrometry: Some(AstrometricSolutionReceipt {
            reference_frame: "ICRS".to_owned(),
            projection: "TAN".to_owned(),
            center_ra_degrees: 281.0,
            center_dec_degrees: -6.0,
            pixel_scale_arcsec: 1.21,
            rotation_degrees: 3.0,
            rms_pixels: 0.35,
            rms_arcsec: 0.42,
            matched_stars: 73,
            parity: AstrometricParity::Negative,
            catalog_identity: "c".repeat(64),
            index_identities: vec!["astrometry.net:index:4206:healpix:17:hpnside:32".to_owned()],
            correspondence_sha256: "d".repeat(64),
            catalog_managed: true,
            installed_set_identity: "e".repeat(64),
            catalog_manifest_sha256: "f".repeat(64),
            index_artifacts: vec![SolverIndexArtifactReceipt {
                index_id: "4206".to_owned(),
                relative_name: "index-4206.fits".to_owned(),
                size_bytes: 94_550_400,
                sha256: "1".repeat(64),
                manifest_sha256: "f".repeat(64),
                installed_set_identity: "e".repeat(64),
            }],
            wcs_sha256: "b".repeat(64),
        }),
        drizzle: Some(DrizzleReceipt {
            scale: 2.0,
            drop_shrink: 0.9,
            kernel: "square".to_owned(),
            input_frames: 42,
            output_width: 4000,
            output_height: 3000,
        }),
        attributes: BTreeMap::new(),
    }
}

pub fn capabilities() -> BackendCapabilities {
    BackendCapabilities {
        schema_version: 1,
        backend_id: "native-worker".to_owned(),
        backend_version: "0.1.0".to_owned(),
        worker_build: "test-build".to_owned(),
        hardware_profiles: BTreeSet::from([
            HardwareProfile::PortableCpu,
            HardwareProfile::GenericArm64Cpu,
            HardwareProfile::GenericAppleMetal,
            HardwareProfile::M3ProTuned,
            HardwareProfile::WindowsCpu,
        ]),
        stages: BTreeSet::from([
            StageKind::Integration,
            StageKind::Drizzle,
            StageKind::AstrometricSolve,
        ]),
        features: BTreeSet::from([
            BackendFeature::CpuExecution,
            BackendFeature::MetalExecution,
            BackendFeature::M3ProTuning,
            BackendFeature::OfflineAstrometricSolver,
            BackendFeature::Drizzle,
        ]),
        maximum_parallel_stages: 1,
        input_extensions: BTreeSet::from(["fits".to_owned(), "xisf".to_owned()]),
        output_extensions: BTreeSet::from(["fits".to_owned()]),
    }
}
