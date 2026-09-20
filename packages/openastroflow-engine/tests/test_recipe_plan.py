from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest
from astropy.io import fits

from openastroflow_engine.drizzle import DrizzleBackend, drizzle_backends
from openastroflow_engine.inventory import inventory_project
from openastroflow_engine.planning import PlanIssueCategory, StageState, build_plan
from openastroflow_engine.recipe import Recipe, RecipeError
from openastroflow_engine.runtime import (
    RuntimeConfigurationError,
    build_e2e_request,
)
from conftest import write_frame


def test_default_recipe_reports_actual_runtime_readiness_without_contract_placeholders(
    nina_project: Path,
) -> None:
    inventory = inventory_project([nina_project])
    plan = build_plan(
        inventory,
        Recipe.from_dict({"calibration": {"allowMasters": False}}),
    )

    assert plan.contract_valid is True
    core = {
        stage.stage.value: stage.state
        for stage in plan.stages
        if stage.stage.value
        in {"QUALITY_GATE", "CALIBRATION", "REGISTRATION", "INTEGRATION"}
    }
    assert set(core.values()) == {StageState.READY}
    assert all(stage.state is not StageState.NOT_IMPLEMENTED for stage in plan.stages)
    solver = next(stage for stage in plan.stages if stage.stage.value == "SOLVER")
    assert plan.execution_ready is (solver.state is StageState.READY)
    assert not any(
        issue.category == PlanIssueCategory.IMPLEMENTATION for issue in plan.issues
    )
    assert plan.serializable()["claims"]["pixelExecutionImplemented"] is True


def test_raw_and_master_flat_for_same_profile_is_ambiguous(
    nina_project: Path,
) -> None:
    inventory = inventory_project([nina_project])
    plan = build_plan(inventory, Recipe())

    assert plan.contract_valid is False
    assert any(issue.code == "FLAT_MATCH_AMBIGUOUS" for issue in plan.issues)


def test_runtime_routes_supplied_calibration_masters_without_raw_duplicates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "master-only"
    light = write_frame(root / "LIGHT" / "light.fits", "Light")
    bias = write_frame(
        root / "MASTERS" / "master_bias.fits", "Master Bias", exposure=0.001
    )
    dark = write_frame(
        root / "MASTERS" / "master_dark.fits", "Master Dark", exposure=120.0
    )
    flat = write_frame(
        root / "MASTERS" / "master_flat_R.fits", "Master Flat", exposure=2.0
    )
    for path in (light, bias, dark, flat):
        fits.setval(path, "CCD-TEMP", value=-10.0)
    inventory = inventory_project([root])

    with pytest.raises(RuntimeConfigurationError) as semantics_error:
        build_e2e_request(inventory, Recipe(), tmp_path / "missing-semantics")
    assert semantics_error.value.code == "MASTER_DARK_BIAS_SEMANTICS_REQUIRED"
    digest = "sha256:" + hashlib.sha256(dark.read_bytes()).hexdigest()
    recipe = Recipe.from_dict(
        {
            "calibration": {
                "masterMetadataOverrides": [
                    {
                        "sourceSha256": digest,
                        "camera": "ASI2600MM PRO",
                        "gain": 100,
                        "offset": 50,
                        "binning": [1, 1],
                        "filter": "R",
                        "cfaPattern": "NONE",
                        "readoutMode": "MODE 1",
                        "temperatureCelsius": -10.0,
                        "exposureSeconds": 120.0,
                        "biasIncluded": True,
                    }
                ]
            }
        }
    )
    request = build_e2e_request(inventory, recipe, tmp_path / "output")

    assert request.bias_files == ()
    assert request.dark_files == ()
    assert request.flat_files == ()
    assert request.master_bias_files == (str(bias.resolve()),)
    assert request.master_dark_files == (str(dark.resolve()),)
    assert request.master_flat_files == (str(flat.resolve()),)
    with pytest.raises(RuntimeConfigurationError) as error:
        build_e2e_request(
            inventory,
            Recipe.from_dict({"calibration": {"allowMasters": False}}),
            tmp_path / "disabled",
        )
    assert error.value.code == "MASTER_CALIBRATION_INPUT_DISABLED"


@pytest.mark.parametrize(
    ("readout", "compatible"),
    [("High Gain 2CMS", True), ("High Gain 1CMS", False), ("unknown", False)],
)
def test_public_plan_matches_master_override_text_like_pixel_reader(
    tmp_path: Path, readout: str, compatible: bool,
) -> None:
    root = tmp_path / "mixed-header-case"
    light = write_frame(root / "LIGHT" / "light.fits", "Light")
    flat = write_frame(root / "FLAT" / "flat.fits", "Flat", exposure=2.0)
    bias = write_frame(root / "MASTER_BIAS" / "bias.fits", "Master Bias", exposure=0.001)
    for path in (light, flat):
        fits.setval(path, "INSTRUME", value="Asi2600Mm Pro")
        fits.setval(path, "READOUTM", value=readout)
    recipe = Recipe.from_dict({
        "calibration": {
            "masterMetadataOverrides": [{
                "sourceSha256": "sha256:" + hashlib.sha256(bias.read_bytes()).hexdigest(),
                "camera": "ASI2600MM PRO", "gain": 100, "offset": 50,
                "binning": [1, 1], "filter": "R", "cfaPattern": "NONE",
                "readoutMode": "HIGH GAIN 2CMS", "temperatureCelsius": -10,
                "exposureSeconds": 0.001,
            }],
        },
    })
    plan = build_plan(inventory_project([root]), recipe)
    assert plan.contract_valid is compatible
    bias_missing = any(issue.code == "BIAS_MATCH_MISSING" for issue in plan.issues)
    assert bias_missing is not compatible


def test_unknown_cfa_requires_hash_bound_confirmation_and_detects_drift(
    nina_project: Path, tmp_path: Path
) -> None:
    light = next((nina_project / "LIGHT").glob("*.fits"))
    with fits.open(light, mode="update") as hdul:
        del hdul[0].header["BAYERPAT"]
        hdul.flush()
    inventory = inventory_project(
        [nina_project / role for role in ("LIGHT", "FLAT", "DARK", "BIAS")]
    )
    light_asset = next(item for item in inventory.assets if item.role.value == "LIGHT")
    assert light_asset.cfa_explicit is False

    with pytest.raises(RuntimeConfigurationError) as missing:
        build_e2e_request(inventory, Recipe(), tmp_path / "missing-cfa")
    assert missing.value.code == "CFA_CONFIRMATION_REQUIRED"

    digest = "sha256:" + hashlib.sha256(light.read_bytes()).hexdigest()
    recipe = Recipe.from_dict(
        {
            "rawFrameMetadataOverrides": [
                {"sourceSha256": digest, "cfaPattern": "NONE"}
            ]
        }
    )
    request = build_e2e_request(inventory, recipe, tmp_path / "confirmed-mono")
    assert request.pipeline_parameters.raw_frame_metadata_overrides[0].source_sha256 == digest
    assert request.pipeline_parameters.raw_frame_metadata_overrides[0].cfa_pattern == "NONE"

    fits.setval(light, "COMMENT", value="content drift after confirmation")
    with pytest.raises(RuntimeConfigurationError) as drift:
        build_e2e_request(inventory, recipe, tmp_path / "drifted")
    assert drift.value.code == "RAW_FRAME_METADATA_OVERRIDE_SOURCE_AMBIGUOUS"


def test_explicit_cfa_light_cannot_be_overridden_to_mono(
    nina_project: Path, tmp_path: Path
) -> None:
    light = next((nina_project / "LIGHT").glob("*.fits"))
    fits.setval(light, "BAYERPAT", value="RGGB")
    inventory = inventory_project(
        [nina_project / role for role in ("LIGHT", "FLAT", "DARK", "BIAS")]
    )
    digest = "sha256:" + hashlib.sha256(light.read_bytes()).hexdigest()
    recipe = Recipe.from_dict(
        {
            "rawFrameMetadataOverrides": [
                {"sourceSha256": digest, "cfaPattern": "NONE"}
            ]
        }
    )
    with pytest.raises(RuntimeConfigurationError) as blocked:
        build_e2e_request(inventory, recipe, tmp_path / "explicit-cfa")
    assert blocked.value.code == "RAW_CFA_OVERRIDE_CONFLICT"


def test_unsupported_cfa_pattern_is_rejected_before_execution(
    nina_project: Path, tmp_path: Path
) -> None:
    light = next((nina_project / "LIGHT").glob("*.fits"))
    fits.setval(light, "BAYERPAT", value="CYGM")
    inventory = inventory_project(
        [nina_project / role for role in ("LIGHT", "FLAT", "DARK", "BIAS")]
    )
    with pytest.raises(RuntimeConfigurationError) as blocked:
        build_e2e_request(inventory, Recipe.from_dict({}), tmp_path / "cygm")
    assert blocked.value.code == "CFA_PATTERN_UNSUPPORTED"


def test_drizzle_capability_mismatch_blocks_contract(nina_project: Path) -> None:
    inventory = inventory_project([nina_project])
    recipe = Recipe.from_dict(
        {"drizzle": {"enabled": True, "scale": 2, "dropShrink": 0.05, "backend": "native-drizzle"}}
    )
    plan = build_plan(inventory, recipe)

    assert plan.contract_valid is False
    assert any(issue.code == "DRIZZLE_CAPABILITY_MISMATCH" for issue in plan.issues)
    drizzle = next(stage for stage in plan.stages if stage.stage.value == "DRIZZLE")
    assert drizzle.state == StageState.BLOCKED


def test_required_missing_dark_is_a_recipe_input_error(nina_project: Path) -> None:
    inventory = inventory_project([nina_project / "LIGHT", nina_project / "FLAT"])
    recipe = Recipe.from_dict({"calibration": {"dark": "REQUIRED"}})
    plan = build_plan(inventory, recipe)

    assert plan.contract_valid is False
    assert any(issue.code == "DARK_MATCH_MISSING" for issue in plan.issues)


def test_unknown_recipe_keys_fail_closed() -> None:
    with pytest.raises(RecipeError, match="unknown keys"):
        Recipe.from_dict({"magicAutoProcess": True})


def test_recipe_parses_content_bound_master_override_and_local_normalization() -> None:
    recipe = Recipe.from_dict(
        {
            "calibration": {
                "masterMetadataOverrides": [
                    {
                        "sourceSha256": "sha256:" + "a" * 64,
                        "camera": "QHY268M",
                        "gain": 100,
                        "offset": 50,
                        "binning": [1, 1],
                        "filter": "R",
                        "cfaPattern": "NONE",
                        "readoutMode": "Mode 1",
                        "temperatureCelsius": -10,
                        "exposureSeconds": 1,
                        "numericDomain": "NORMALIZED_UNIT",
                        "normalizedUnitScale": 1,
                    }
                ]
            },
            "localNormalization": {"enabled": True, "tileSizePixels": 384},
        }
    )
    assert recipe.calibration.master_metadata_overrides[0].gain == 100
    assert (
        recipe.calibration.master_metadata_overrides[0].numeric_domain
        == "NORMALIZED_UNIT"
    )
    assert recipe.calibration.master_metadata_overrides[0].normalized_unit_scale == 1
    assert recipe.local_normalization.enabled is True
    assert recipe.local_normalization.tile_size_pixels == 384


def test_review_approval_recipe_requires_three_canonical_digests() -> None:
    digest = "sha256:" + "a" * 64
    recipe = Recipe.from_dict(
        {
            "reviewApprovals": [
                {
                    "sourceSha256": digest,
                    "gatePolicyDigest": digest,
                    "requestDigest": digest,
                }
            ]
        }
    )
    assert recipe.serializable()["reviewApprovals"][0]["sourceSha256"] == digest
    with pytest.raises(RecipeError, match="lowercase sha256"):
        Recipe.from_dict(
            {
                "reviewApprovals": [
                    {
                        "sourceSha256": "A" * 64,
                        "gatePolicyDigest": digest,
                        "requestDigest": digest,
                    }
                ]
            }
        )


def test_solver_required_by_default() -> None:
    assert Recipe().solver.policy.value == "REQUIRED"


def test_published_recipe_examples_parse() -> None:
    package_root = Path(__file__).parents[1]
    for path in sorted((package_root / "examples").glob("*.json")):
        recipe = Recipe.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert recipe.schema_version == 1


def test_native_drizzle_backend_reports_scales_kernels_and_cfa() -> None:
    from openastroflow_engine.native_kernels import load_native_kernels

    backend = drizzle_backends()[0]
    assert isinstance(backend, DrizzleBackend)
    assert backend.descriptor.execution_ready is (load_native_kernels() is not None)
    assert backend.drizzle_capabilities.scales == (1, 2, 3, 4)
    assert backend.drizzle_capabilities.supports_cfa_drizzle
    assert backend.validate_options({"scale": 5}) != ()
    assert backend.validate_options({"kernel": "lanczos3"}) != ()
    assert backend.validate_options({"scale": 3, "kernel": "circular", "dropShrink": 0.7}) == ()
