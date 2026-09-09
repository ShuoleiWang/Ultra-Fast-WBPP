mod common;

use std::collections::BTreeMap;

use openastroflow_app_core::{
    ArtifactKind, GateDecision, RequiredResultGate, ResultRequirement, StageKind, StageSpec,
};

#[test]
fn required_solver_and_drizzle_pass_only_with_final_provenance() {
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Required);
    let stages = vec![
        common::stage("drizzle", StageKind::Drizzle),
        common::stage("solve", StageKind::AstrometricSolve),
    ];
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[common::final_master()]);
    assert_eq!(report.decision, GateDecision::Ready);
    assert!(report.checks.iter().all(|check| check.passed));
}

#[test]
fn process_success_without_embedded_wcs_is_blocked() {
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let stages = vec![common::stage("solve", StageKind::AstrometricSolve)];
    let mut master = common::final_master();
    master.astrometry = None;
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[master]);
    assert_eq!(report.decision, GateDecision::Blocked);
    assert!(report.checks.iter().any(|check| {
        check.code == "final-master-astrometry-validated" && check.required && !check.passed
    }));
}

#[test]
fn solver_receipt_must_meet_recipe_quality_limits() {
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let stages = vec![common::stage("solve", StageKind::AstrometricSolve)];
    let mut master = common::final_master();
    master.astrometry.as_mut().expect("fixture").rms_arcsec = 4.0;
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[master]);
    assert_eq!(report.decision, GateDecision::Blocked);

    let mut master = common::final_master();
    master.astrometry.as_mut().expect("fixture").matched_stars = 11;
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[master]);
    assert_eq!(report.decision, GateDecision::Blocked);
}

#[test]
fn solver_receipt_requires_complete_catalog_correspondence_provenance() {
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let stages = vec![common::stage("solve", StageKind::AstrometricSolve)];
    let mut invalid = Vec::new();

    let mut master = common::final_master();
    master.astrometry.as_mut().expect("fixture").rms_pixels = f64::NAN;
    invalid.push(master);

    let mut master = common::final_master();
    master.astrometry.as_mut().expect("fixture").catalog_managed = false;
    invalid.push(master);

    let mut master = common::final_master();
    master
        .astrometry
        .as_mut()
        .expect("fixture")
        .installed_set_identity = "not-a-digest".to_owned();
    invalid.push(master);

    let mut master = common::final_master();
    master.astrometry.as_mut().expect("fixture").index_artifacts[0].sha256 =
        "not-a-digest".to_owned();
    invalid.push(master);

    let mut master = common::final_master();
    master.astrometry.as_mut().expect("fixture").rms_pixels = 20.0;
    invalid.push(master);

    let mut master = common::final_master();
    master
        .astrometry
        .as_mut()
        .expect("fixture")
        .catalog_identity = String::new();
    invalid.push(master);

    let mut master = common::final_master();
    master
        .astrometry
        .as_mut()
        .expect("fixture")
        .index_identities
        .clear();
    invalid.push(master);

    let mut master = common::final_master();
    master
        .astrometry
        .as_mut()
        .expect("fixture")
        .correspondence_sha256 = "not-a-digest".to_owned();
    invalid.push(master);

    for master in invalid {
        let report = RequiredResultGate::evaluate(&recipe, &stages, &[master]);
        assert_eq!(report.decision, GateDecision::Blocked);
    }
}

#[test]
fn required_drizzle_must_match_recipe_settings() {
    let recipe = common::recipe(ResultRequirement::Disabled, ResultRequirement::Required);
    let stages = vec![common::stage("drizzle", StageKind::Drizzle)];
    let mut master = common::final_master();
    master.produced_by_stage_id = "drizzle".to_owned();
    master.drizzle.as_mut().expect("fixture").scale = 3.0;
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[master]);
    assert_eq!(report.decision, GateDecision::Blocked);
}

#[test]
fn best_effort_solver_failure_does_not_block_otherwise_valid_master() {
    let recipe = common::recipe(ResultRequirement::BestEffort, ResultRequirement::Disabled);
    let mut master = common::final_master();
    master.astrometry = None;
    master.produced_by_stage_id = "integrate".to_owned();
    let stages = vec![common::stage("integrate", StageKind::Integration)];
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[master]);
    assert_eq!(report.decision, GateDecision::Ready);
    assert!(
        report
            .checks
            .iter()
            .any(|check| !check.required && !check.passed)
    );
}

#[test]
fn every_final_master_must_satisfy_required_solver_result() {
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let stages = vec![common::stage("solve", StageKind::AstrometricSolve)];
    let good = common::final_master();
    let mut bad = common::final_master();
    bad.artifact_id = "master-luminance".to_owned();
    bad.astrometry = None;
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[good, bad]);
    assert_eq!(report.decision, GateDecision::Blocked);
    let check = report
        .checks
        .iter()
        .find(|check| check.code == "final-master-astrometry-validated")
        .expect("solver check");
    assert_eq!(check.artifact_ids, vec!["master-luminance"]);
}

#[test]
fn no_final_master_is_always_blocked() {
    let recipe = common::recipe(ResultRequirement::Disabled, ResultRequirement::Disabled);
    let report = RequiredResultGate::evaluate(&recipe, &[], &[]);
    assert_eq!(report.decision, GateDecision::Blocked);
}

#[test]
fn stage_receipt_must_match_the_configured_recipe_stage() {
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let stages = vec![common::stage(
        "unconfigured-solver",
        StageKind::AstrometricSolve,
    )];
    let mut master = common::final_master();
    master.produced_by_stage_id = "unconfigured-solver".to_owned();
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[master]);
    assert_eq!(report.decision, GateDecision::Blocked);
    assert!(
        report
            .checks
            .iter()
            .any(|check| check.code == "solver-stage-succeeded" && !check.passed)
    );
}

#[test]
fn diagnostic_artifact_cannot_impersonate_a_final_master() {
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let stages = vec![common::stage("solve", StageKind::AstrometricSolve)];
    let mut artifact = common::final_master();
    artifact.kind = ArtifactKind::QualityReport;
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[artifact]);
    assert_eq!(report.decision, GateDecision::Blocked);
    assert!(
        report
            .checks
            .iter()
            .any(|check| { check.code == "final-master-receipts-valid" && !check.passed })
    );
}

#[test]
fn required_solver_master_cannot_be_produced_by_quality_control() {
    let mut recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let solve = recipe
        .stages
        .iter_mut()
        .find(|stage| stage.kind == StageKind::AstrometricSolve)
        .expect("solver stage");
    solve.kind = StageKind::QualityControl;
    let stages = vec![common::stage("solve", StageKind::QualityControl)];
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[common::final_master()]);
    assert_eq!(report.decision, GateDecision::Blocked);
}

#[test]
fn export_must_descend_from_the_required_solver_stage() {
    let mut recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    recipe.stages.push(StageSpec {
        stage_id: "export".to_owned(),
        kind: StageKind::Export,
        enabled: true,
        depends_on: Vec::new(),
        parameters: BTreeMap::new(),
    });
    let stages = vec![
        common::stage("solve", StageKind::AstrometricSolve),
        common::stage("export", StageKind::Export),
    ];
    let mut artifact = common::final_master();
    artifact.produced_by_stage_id = "export".to_owned();
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[artifact]);
    assert_eq!(report.decision, GateDecision::Blocked);
    assert!(
        report.checks.iter().any(|check| {
            check.code == "final-master-producing-stage-succeeded" && !check.passed
        })
    );

    recipe
        .stages
        .iter_mut()
        .find(|stage| stage.stage_id == "export")
        .expect("export stage")
        .depends_on = vec!["solve".to_owned()];
    let mut artifact = common::final_master();
    artifact.produced_by_stage_id = "export".to_owned();
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[artifact]);
    assert_eq!(report.decision, GateDecision::Ready);
}

#[test]
fn final_solver_stage_must_descend_from_required_drizzle() {
    let mut recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Required);
    recipe
        .stages
        .iter_mut()
        .find(|stage| stage.kind == StageKind::AstrometricSolve)
        .expect("solver stage")
        .depends_on
        .clear();
    let stages = vec![
        common::stage("drizzle", StageKind::Drizzle),
        common::stage("solve", StageKind::AstrometricSolve),
    ];
    let report = RequiredResultGate::evaluate(&recipe, &stages, &[common::final_master()]);
    assert_eq!(report.decision, GateDecision::Blocked);
    assert!(
        report.checks.iter().any(|check| {
            check.code == "final-master-producing-stage-succeeded" && !check.passed
        })
    );
}
