mod common;

use std::collections::BTreeMap;

use openastroflow_app_core::{
    HardwareProfile, InputRole, Project, ProjectReceipt, ProjectSource, RECEIPT_SCHEMA_VERSION,
    Recipe, RecipeReceipt, ResultRequirement, RunReceipt, RunStatus, StageKind, StageSpec,
    Validate,
};

#[test]
fn project_rejects_duplicate_sources() {
    let source = ProjectSource {
        source_id: "lights".to_owned(),
        role: InputRole::Light,
        host_path: "/data/lights".to_owned(),
        recursive: true,
        filter: None,
    };
    let project = Project {
        schema_version: 1,
        project_id: "shield-mosaic".to_owned(),
        display_name: "Shield Mosaic".to_owned(),
        created_at_unix_ms: 1,
        sources: vec![source.clone(), source],
        labels: BTreeMap::new(),
    };
    assert!(project.validate().is_err());
}

#[test]
fn recipe_rejects_dependency_cycle() {
    let mut recipe = common::recipe(ResultRequirement::Disabled, ResultRequirement::Disabled);
    recipe.stages = vec![
        StageSpec {
            stage_id: "first".to_owned(),
            kind: StageKind::Calibration,
            enabled: true,
            depends_on: vec!["second".to_owned()],
            parameters: BTreeMap::new(),
        },
        StageSpec {
            stage_id: "second".to_owned(),
            kind: StageKind::Integration,
            enabled: true,
            depends_on: vec!["first".to_owned()],
            parameters: BTreeMap::new(),
        },
    ];
    assert!(recipe.validate().is_err());
}

#[test]
fn required_solver_needs_enabled_solver_stage() {
    let mut recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    recipe
        .stages
        .retain(|stage| stage.kind != StageKind::AstrometricSolve);
    assert!(recipe.validate().is_err());
}

#[test]
fn enabled_stage_cannot_depend_on_disabled_stage() {
    let mut recipe = common::recipe(ResultRequirement::Disabled, ResultRequirement::Disabled);
    recipe.stages.insert(
        0,
        StageSpec {
            stage_id: "disabled".to_owned(),
            kind: StageKind::Calibration,
            enabled: false,
            depends_on: vec![],
            parameters: BTreeMap::new(),
        },
    );
    recipe.stages[1].depends_on = vec!["disabled".to_owned()];
    assert!(recipe.validate().is_err());
}

#[test]
fn run_receipt_requires_known_stage_for_artifacts() {
    let artifact = common::final_master();
    let receipt = RunReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        run_id: "run-1".to_owned(),
        plan_id: "plan-1".to_owned(),
        project: ProjectReceipt {
            schema_version: 1,
            project_id: "project-1".to_owned(),
            project_sha256: "b".repeat(64),
            input_manifest_sha256: "c".repeat(64),
            resolved_source_count: 1,
            captured_at_unix_ms: 1,
        },
        recipe: RecipeReceipt {
            schema_version: 1,
            recipe_id: "recipe-1".to_owned(),
            recipe_sha256: "d".repeat(64),
        },
        backend_id: "worker-1".to_owned(),
        backend_version: "0.1.0".to_owned(),
        hardware_profile: HardwareProfile::GenericArm64Cpu,
        status: RunStatus::Succeeded,
        started_at_unix_ms: 1,
        finished_at_unix_ms: Some(2),
        stages: vec![common::stage("integrate", StageKind::Integration)],
        artifacts: vec![artifact],
    };
    assert!(receipt.validate().is_err());
}

#[test]
fn recipe_round_trips_without_losing_stage_parameters() {
    let mut recipe: Recipe =
        common::recipe(ResultRequirement::Required, ResultRequirement::Required);
    recipe.stages[0]
        .parameters
        .insert("rejection".to_owned(), serde_json::json!("linear-fit"));
    let encoded = serde_json::to_vec(&recipe).expect("serialize");
    let decoded: Recipe = serde_json::from_slice(&encoded).expect("deserialize");
    assert_eq!(decoded, recipe);
    decoded.validate().expect("valid recipe");
}

#[test]
fn astrometry_provenance_cannot_masquerade_as_a_fresh_solve() {
    let mut propagated = common::final_master();
    propagated.attributes.insert(
        "astrometryProvenanceType".to_owned(),
        serde_json::json!("PROPAGATED_VERIFIED"),
    );
    propagated.attributes.insert(
        "freshSolveOnThisArtifactGrid".to_owned(),
        serde_json::json!(false),
    );
    propagated
        .validate()
        .expect("explicit propagated WCS is valid");

    propagated.attributes.insert(
        "freshSolveOnThisArtifactGrid".to_owned(),
        serde_json::json!(true),
    );
    assert!(propagated.validate().is_err());

    let mut fresh = common::final_master();
    fresh.attributes.insert(
        "astrometryProvenanceType".to_owned(),
        serde_json::json!("FRESH_SOLVE_UNCHANGED_GRID"),
    );
    fresh.attributes.insert(
        "freshSolveOnThisArtifactGrid".to_owned(),
        serde_json::json!(true),
    );
    fresh.validate().expect("fresh solve provenance agrees");
}
