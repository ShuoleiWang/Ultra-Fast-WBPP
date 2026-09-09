use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::model::{
    ArtifactDesignation, ArtifactReceipt, Recipe, ResultRequirement, StageKind, StageReceipt,
    StageStatus,
};
use crate::validation::Validate;

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GateDecision {
    Ready,
    Blocked,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GateCheck {
    pub code: String,
    pub required: bool,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifact_ids: Vec<String>,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResultGateReport {
    pub decision: GateDecision,
    pub checks: Vec<GateCheck>,
}

impl ResultGateReport {
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.decision == GateDecision::Ready
    }
}

/// Final-product gate. A successful worker process is not sufficient: all
/// recipe-required scientific results must be present and validated on every
/// artifact designated as a final master.
pub struct RequiredResultGate;

impl RequiredResultGate {
    #[must_use]
    pub fn evaluate(
        recipe: &Recipe,
        stages: &[StageReceipt],
        artifacts: &[ArtifactReceipt],
    ) -> ResultGateReport {
        let mut checks = Vec::new();
        if let Err(error) = recipe.validate() {
            checks.push(GateCheck {
                code: "recipe-invalid".to_owned(),
                required: true,
                passed: false,
                artifact_ids: Vec::new(),
                message: error.to_string(),
            });
            return finish(checks);
        }

        let final_masters: Vec<_> = artifacts
            .iter()
            .filter(|artifact| artifact.designation == ArtifactDesignation::FinalMaster)
            .collect();
        checks.push(GateCheck {
            code: "final-master-present".to_owned(),
            required: true,
            passed: !final_masters.is_empty(),
            artifact_ids: final_masters
                .iter()
                .map(|artifact| artifact.artifact_id.clone())
                .collect(),
            message: if final_masters.is_empty() {
                "no artifact is designated as the final master".to_owned()
            } else {
                format!("{} final master artifact(s) present", final_masters.len())
            },
        });

        let orphaned_final_masters: Vec<_> = final_masters
            .iter()
            .filter(|artifact| {
                !stages.iter().any(|stage| {
                    stage.stage_id == artifact.produced_by_stage_id
                        && stage.status == StageStatus::Succeeded
                        && final_stage_allowed(recipe, &stage.stage_id, stage.kind)
                        && stage.artifact_ids.contains(&artifact.artifact_id)
                        && recipe.stages.iter().any(|configured| {
                            configured.enabled
                                && configured.stage_id == stage.stage_id
                                && configured.kind == stage.kind
                        })
                })
            })
            .map(|artifact| artifact.artifact_id.clone())
            .collect();
        checks.push(GateCheck {
            code: "final-master-producing-stage-succeeded".to_owned(),
            required: true,
            passed: !final_masters.is_empty() && orphaned_final_masters.is_empty(),
            artifact_ids: orphaned_final_masters.clone(),
            message: if final_masters.is_empty() {
                "cannot bind a producing stage without a final master".to_owned()
            } else if orphaned_final_masters.is_empty() {
                "every final master is bound to its successful producing stage".to_owned()
            } else {
                "a final master is not bound to a successful producing stage".to_owned()
            },
        });

        let invalid_artifacts: Vec<_> = final_masters
            .iter()
            .filter(|artifact| artifact.validate().is_err())
            .map(|artifact| artifact.artifact_id.clone())
            .collect();
        checks.push(GateCheck {
            code: "final-master-receipts-valid".to_owned(),
            required: true,
            passed: invalid_artifacts.is_empty(),
            artifact_ids: invalid_artifacts.clone(),
            message: if invalid_artifacts.is_empty() {
                "final master receipts are structurally valid".to_owned()
            } else {
                "one or more final master receipts are invalid".to_owned()
            },
        });

        append_solver_checks(recipe, stages, &final_masters, &mut checks);
        append_drizzle_checks(recipe, stages, &final_masters, &mut checks);
        finish(checks)
    }
}

fn final_stage_allowed(recipe: &Recipe, stage_id: &str, kind: StageKind) -> bool {
    let scientifically_publishable = matches!(
        kind,
        StageKind::Integration
            | StageKind::Drizzle
            | StageKind::AstrometricSolve
            | StageKind::Mosaic
            | StageKind::Export
    );
    let solver_ok = recipe.solver.result != ResultRequirement::Required
        || kind == StageKind::AstrometricSolve
        || (kind == StageKind::Export
            && stage_depends_on_kind(recipe, stage_id, StageKind::AstrometricSolve));
    let drizzle_ok = recipe.drizzle.result != ResultRequirement::Required
        || kind == StageKind::Drizzle
        || stage_depends_on_kind(recipe, stage_id, StageKind::Drizzle);
    scientifically_publishable && solver_ok && drizzle_ok
}

fn stage_depends_on_kind(recipe: &Recipe, stage_id: &str, required: StageKind) -> bool {
    let mut pending = vec![stage_id];
    let mut visited = std::collections::BTreeSet::new();
    while let Some(candidate) = pending.pop() {
        if !visited.insert(candidate) {
            continue;
        }
        let Some(stage) = recipe
            .stages
            .iter()
            .find(|stage| stage.enabled && stage.stage_id == candidate)
        else {
            continue;
        };
        for dependency in &stage.depends_on {
            let Some(upstream) = recipe
                .stages
                .iter()
                .find(|item| item.enabled && item.stage_id == *dependency)
            else {
                continue;
            };
            if upstream.kind == required {
                return true;
            }
            pending.push(&upstream.stage_id);
        }
    }
    false
}

fn stage_succeeded(recipe: &Recipe, stages: &[StageReceipt], kind: StageKind) -> bool {
    recipe.stages.iter().any(|configured| {
        configured.enabled
            && configured.kind == kind
            && stages.iter().any(|receipt| {
                receipt.stage_id == configured.stage_id
                    && receipt.kind == configured.kind
                    && receipt.status == StageStatus::Succeeded
            })
    })
}

fn append_solver_checks(
    recipe: &Recipe,
    stages: &[StageReceipt],
    final_masters: &[&ArtifactReceipt],
    checks: &mut Vec<GateCheck>,
) {
    if recipe.solver.result == ResultRequirement::Disabled {
        return;
    }
    let required = recipe.solver.result == ResultRequirement::Required;
    let stage_passed = stage_succeeded(recipe, stages, StageKind::AstrometricSolve);
    checks.push(GateCheck {
        code: "solver-stage-succeeded".to_owned(),
        required,
        passed: stage_passed,
        artifact_ids: Vec::new(),
        message: if stage_passed {
            "astrometric solver stage succeeded".to_owned()
        } else {
            "astrometric solver stage did not succeed".to_owned()
        },
    });

    let failed: Vec<_> = final_masters
        .iter()
        .filter(|artifact| {
            let Some(solution) = &artifact.astrometry else {
                return true;
            };
            solution.validate().is_err()
                || solution.matched_stars < recipe.solver.minimum_matches
                || solution.rms_arcsec > recipe.solver.maximum_rms_arcsec
                || !solution
                    .projection
                    .eq_ignore_ascii_case(&recipe.solver.projection)
        })
        .map(|artifact| artifact.artifact_id.clone())
        .collect();
    checks.push(GateCheck {
        code: "final-master-astrometry-validated".to_owned(),
        required,
        passed: !final_masters.is_empty() && failed.is_empty(),
        artifact_ids: failed.clone(),
        message: if final_masters.is_empty() {
            "cannot validate astrometry without a final master".to_owned()
        } else if failed.is_empty() {
            "every final master embeds a validated astrometric solution".to_owned()
        } else {
            "a final master is missing a valid in-file astrometric solution or exceeds recipe limits"
                .to_owned()
        },
    });
}

fn append_drizzle_checks(
    recipe: &Recipe,
    stages: &[StageReceipt],
    final_masters: &[&ArtifactReceipt],
    checks: &mut Vec<GateCheck>,
) {
    if recipe.drizzle.result == ResultRequirement::Disabled {
        return;
    }
    let required = recipe.drizzle.result == ResultRequirement::Required;
    let stage_passed = stage_succeeded(recipe, stages, StageKind::Drizzle);
    checks.push(GateCheck {
        code: "drizzle-stage-succeeded".to_owned(),
        required,
        passed: stage_passed,
        artifact_ids: Vec::new(),
        message: if stage_passed {
            "drizzle stage succeeded".to_owned()
        } else {
            "drizzle stage did not succeed".to_owned()
        },
    });

    let failed: Vec<_> = final_masters
        .iter()
        .filter(|artifact| {
            let Some(drizzle) = &artifact.drizzle else {
                return true;
            };
            drizzle.validate().is_err()
                || (drizzle.scale - recipe.drizzle.scale).abs() > f32::EPSILON * 8.0
                || (drizzle.drop_shrink - recipe.drizzle.drop_shrink).abs() > f32::EPSILON * 8.0
                || !drizzle.kernel.eq_ignore_ascii_case(&recipe.drizzle.kernel)
        })
        .map(|artifact| artifact.artifact_id.clone())
        .collect();
    checks.push(GateCheck {
        code: "final-master-drizzle-validated".to_owned(),
        required,
        passed: !final_masters.is_empty() && failed.is_empty(),
        artifact_ids: failed.clone(),
        message: if final_masters.is_empty() {
            "cannot validate drizzle without a final master".to_owned()
        } else if failed.is_empty() {
            "every final master carries a matching drizzle receipt".to_owned()
        } else {
            "a final master is missing drizzle provenance or does not match the recipe".to_owned()
        },
    });
}

fn finish(checks: Vec<GateCheck>) -> ResultGateReport {
    let blocked = checks.iter().any(|check| check.required && !check.passed);
    ResultGateReport {
        decision: if blocked {
            GateDecision::Blocked
        } else {
            GateDecision::Ready
        },
        checks,
    }
}
