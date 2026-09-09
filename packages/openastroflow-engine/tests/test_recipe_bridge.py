from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from openastroflow_engine.recipe import SolverPolicy
from openastroflow_engine.recipe_bridge import (
    BridgeError,
    ResultRequirement,
    bridge_plan,
    bridge_project,
    bridge_recipe,
)


REPOSITORY = Path(__file__).resolve().parents[3]
PROTOCOL = REPOSITORY / "protocol"


def _controller_plan() -> dict[str, object]:
    lines = (
        PROTOCOL / "examples" / "controller-to-worker-v1.ndjson"
    ).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[1])


def test_schema_canonical_recipe_maps_without_losing_contract_fields() -> None:
    schema = json.loads(
        (PROTOCOL / "schema" / "recipe-v1.schema.json").read_text(encoding="utf-8")
    )
    assert set(schema["properties"]) >= {"stages", "solver", "drizzle"}
    assert set(schema["definitions"]["SolverSettings"]["properties"]) >= {
        "result",
        "catalog",
        "projection",
        "minimumMatches",
        "maximumRmsArcsec",
    }
    assert set(schema["definitions"]["DrizzleSettings"]["properties"]) >= {
        "result",
        "scale",
        "dropShrink",
        "kernel",
    }

    canonical = _controller_plan()["payload"]["recipe"]
    bridged = bridge_recipe(canonical)
    assert bridged.canonical() == canonical
    assert bridged.solver.result == ResultRequirement.REQUIRED
    assert bridged.solver.catalog == "astrometry-net-offline"
    assert bridged.solver.projection == "TAN"
    assert bridged.solver.minimum_matches == 12
    assert bridged.solver.maximum_rms_arcsec == 2.0
    assert bridged.drizzle.result == ResultRequirement.REQUIRED
    assert bridged.drizzle.scale == 2.0
    assert bridged.drizzle.drop_shrink == 0.9
    assert bridged.drizzle.kernel == "square"
    assert bridged.python_recipe.solver.policy == SolverPolicy.REQUIRED
    assert bridged.python_recipe.drizzle.enabled is True


def test_required_solver_and_drizzle_receipts_are_gated_not_downgraded() -> None:
    bridged = bridge_recipe(_controller_plan()["payload"]["recipe"])
    worker_artifact = json.loads(
        (PROTOCOL / "examples" / "worker-to-controller-v1.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()[2]
    )["payload"]["artifact"]
    bridged.validate_final_artifact(worker_artifact)

    bad = deepcopy(worker_artifact)
    bad["astrometry"]["matchedStars"] = 11
    with pytest.raises(BridgeError) as error:
        bridged.validate_final_artifact(bad)
    assert error.value.code == "SOLVER_MATCHES_BELOW_MINIMUM"

    bad = deepcopy(worker_artifact)
    bad["astrometry"] = None
    with pytest.raises(BridgeError) as error:
        bridged.validate_final_artifact(bad)
    assert error.value.code == "SOLVER_RESULT_MISSING"

    bad = deepcopy(worker_artifact)
    bad["drizzle"]["kernel"] = "gaussian"
    with pytest.raises(BridgeError) as error:
        bridged.validate_final_artifact(bad)
    assert error.value.code == "DRIZZLE_KERNEL_MISMATCH"


@pytest.mark.parametrize(
    ("section", "field", "value", "code"),
    [
        ("astrometry", "projection", "SIN", "SOLVER_PROJECTION_MISMATCH"),
        ("astrometry", "rmsArcsec", 2.01, "SOLVER_RMS_ABOVE_MAXIMUM"),
        ("astrometry", "rmsPixels", 20.0, "SOLVER_RMS_INCONSISTENT"),
        ("astrometry", "parity", "FLIPPED", "SOLVER_PARITY_INVALID"),
        ("astrometry", "catalogIdentity", "bad", "SOLVER_DIGEST_INVALID"),
        (
            "astrometry",
            "correspondenceSha256",
            "bad",
            "SOLVER_DIGEST_INVALID",
        ),
        (
            "astrometry",
            "indexIdentities",
            [],
            "SOLVER_INDEX_IDENTITY_INVALID",
        ),
        ("drizzle", "scale", 3.0, "DRIZZLE_SCALE_MISMATCH"),
        ("drizzle", "dropShrink", 0.8, "DRIZZLE_DROP_SHRINK_MISMATCH"),
    ],
)
def test_every_canonical_result_threshold_is_enforced(
    section: str, field: str, value: object, code: str
) -> None:
    bridged = bridge_recipe(_controller_plan()["payload"]["recipe"])
    artifact = json.loads(
        (PROTOCOL / "examples" / "worker-to-controller-v1.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()[2]
    )["payload"]["artifact"]
    artifact[section][field] = value
    with pytest.raises(BridgeError) as error:
        bridged.validate_final_artifact(artifact)
    assert error.value.code == code


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda recipe: recipe.update({"parameters": {"magic": True}}),
            "RECIPE_PARAMETERS_UNMAPPABLE",
        ),
        (
            lambda recipe: recipe["stages"][0].update(
                {"parameters": {"rejection": "linear-fit"}}
            ),
            "STAGE_PARAMETERS_UNMAPPABLE",
        ),
        (
            lambda recipe: recipe["solver"].update({"result": "best-effort"}),
            "SOLVER_BEST_EFFORT_UNMAPPABLE",
        ),
        (
            lambda recipe: recipe["drizzle"].update({"result": "disabled"}),
            "DRIZZLE_STAGE_RESULT_UNMAPPABLE",
        ),
        (
            lambda recipe: recipe["drizzle"].update({"scale": 1.5}),
            "DRIZZLE_SCALE_UNSUPPORTED",
        ),
        (
            lambda recipe: recipe["drizzle"].update({"kernel": "gaussian"}),
            "DRIZZLE_KERNEL_UNSUPPORTED",
        ),
        (
            lambda recipe: recipe["solver"].update({"catalog": "unknown-catalog"}),
            "SOLVER_CATALOG_UNSUPPORTED",
        ),
        (
            lambda recipe: recipe["solver"].update({"projection": "SIN"}),
            "SOLVER_PROJECTION_UNSUPPORTED",
        ),
        (
            lambda recipe: recipe["stages"][0].update({"kind": "mosaic"}),
            "STAGE_UNSUPPORTED",
        ),
    ],
)
def test_unmappable_recipe_semantics_fail_closed(mutate, code: str) -> None:
    recipe = deepcopy(_controller_plan()["payload"]["recipe"])
    mutate(recipe)
    with pytest.raises(BridgeError) as error:
        bridge_recipe(recipe)
    assert error.value.code == code


def test_plan_bridge_preserves_canonical_snapshots_without_touching_host_paths() -> None:
    envelope = _controller_plan()
    envelope["payload"]["project"]["sources"][0]["hostPath"] = (
        r"C:\NINA\Shield\LIGHT"
    )
    bridged = bridge_plan(envelope, build_inventory=False)
    assert bridged.request_id == "request-1"
    assert bridged.project.sources[0].display_path == r"C:\NINA\Shield\LIGHT"
    assert bridged.project.canonical() == envelope["payload"]["project"]
    assert bridged.recipe.canonical() == envelope["payload"]["recipe"]


def test_project_bridge_builds_role_bound_inventory(nina_project: Path) -> None:
    project_schema = json.loads(
        (PROTOCOL / "schema" / "project-v1.schema.json").read_text(encoding="utf-8")
    )
    assert project_schema["x-openastroflow-schema-version"] == 1
    project = {
        "schemaVersion": 1,
        "projectId": "m16-project",
        "displayName": "M16",
        "createdAtUnixMs": 1,
        "sources": [
            {
                "sourceId": "lights",
                "role": "light",
                "hostPath": str(nina_project / "LIGHT"),
                "recursive": True,
                "filter": "R",
            },
            {
                "sourceId": "flats",
                "role": "flat",
                "hostPath": str(nina_project / "FLAT"),
                "recursive": True,
                "filter": "R",
            },
            {
                "sourceId": "darks",
                "role": "dark",
                "hostPath": str(nina_project / "DARK"),
                "recursive": True,
            },
            {
                "sourceId": "bias",
                "role": "bias",
                "hostPath": str(nina_project / "BIAS"),
                "recursive": True,
            },
        ],
        "labels": {"target": "M16"},
    }
    bridged = bridge_project(project)
    assert bridged.inventory is not None
    assert bridged.inventory.project_id == "m16-project"
    assert bridged.inventory.counts["LIGHT"] == 1
    assert bridged.inventory.counts["FLAT"] == 1
    assert bridged.inventory.counts["DARK"] == 1
    assert bridged.inventory.counts["BIAS"] == 1


def test_foreign_windows_source_can_be_displayed_but_not_inventoried_on_posix() -> None:
    project = deepcopy(_controller_plan()["payload"]["project"])
    project["sources"] = [
        {
            "sourceId": "lights",
            "role": "light",
            "hostPath": r"C:\NINA\LIGHT",
            "recursive": True,
        }
    ]
    displayed = bridge_project(project, build_inventory=False)
    assert displayed.sources[0].display_path == r"C:\NINA\LIGHT"
    if Path("/").anchor == "/":
        with pytest.raises(BridgeError) as error:
            bridge_project(project)
        assert error.value.code == "HOST_PATH_PLATFORM_MISMATCH"
