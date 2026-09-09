from __future__ import annotations

from pathlib import Path

from openastroflow_engine.inventory import inventory_project
from openastroflow_engine.models import AssetRole, AssetStatus

from conftest import write_frame


def test_inventory_recursively_classifies_nina_roles(nina_project: Path) -> None:
    inventory = inventory_project([nina_project])

    assert inventory.counts["LIGHT"] == 1
    assert inventory.counts["FLAT"] == 1
    assert inventory.counts["DARK"] == 1
    assert inventory.counts["BIAS"] == 1
    assert inventory.counts["MASTER_FLAT"] == 1
    assert inventory.has_errors is False
    light = next(asset for asset in inventory.assets if asset.role == AssetRole.LIGHT)
    assert light.status == AssetStatus.READY
    assert light.filter_name == "R"
    assert light.width == light.height == 16
    assert light.source_stat is not None
    assert light.asset_id.startswith("sha256:")
    assert light.group_id.startswith("sha256:")


def test_explicit_unknown_role_is_not_promoted_by_light_directory(tmp_path: Path) -> None:
    root = tmp_path / "project"
    write_frame(root / "LIGHT" / "focus.fits", "Focus")

    inventory = inventory_project([root])

    assert inventory.assets[0].role == AssetRole.UNKNOWN
    assert {issue.code for issue in inventory.issues} == {"ROLE_UNKNOWN", "NO_LIGHTS"}


def test_inventory_is_deterministic(nina_project: Path) -> None:
    first = inventory_project([nina_project])
    second = inventory_project([nina_project])

    assert first.project_id == second.project_id
    assert first.serializable() == second.serializable()
