from __future__ import annotations

from pathlib import Path

from ufwbpp.inventory import inventory_project
from ufwbpp.models import AssetRole, AssetStatus

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


def test_pixinsight_processed_lights_are_refused_with_one_clear_issue(tmp_path: Path) -> None:
    import numpy as np
    from lightframeqc.xisf import XISF

    raw = write_frame(tmp_path / "raw" / "light.fits", "Light")
    write_frame(tmp_path / "wbpp" / "calibrated" / "Light_R" / "light_c.fits", "Light", exposure=121)
    write_frame(tmp_path / "wbpp" / "registered" / "Light_R" / "light_c_r.fits", "Light", exposure=122)
    XISF.write(
        tmp_path / "wbpp" / "elsewhere.xisf",
        np.zeros((16, 16, 1), dtype=np.float32),
        image_metadata={
            "FITSKeywords": {"IMAGETYP": [{"value": "LIGHT", "comment": ""}], "EXPTIME": [{"value": "123", "comment": ""}]},
            "XISFProperties": {"PCL:Calibration:CosmeticCorrection:HighCounts": {"type": "Int32", "value": 2}},
        },
    )

    inventory = inventory_project([tmp_path])

    by_name = {Path(asset.path).name: asset for asset in inventory.assets}
    assert by_name[raw.name].status == AssetStatus.READY
    for name in ("light_c.fits", "light_c_r.fits", "elsewhere.xisf"):
        assert by_name[name].role == AssetRole.LIGHT
        assert by_name[name].status == AssetStatus.CONFLICT
        assert "by PixInsight" in by_name[name].role_conflicts[-1]
    (issue,) = [issue for issue in inventory.issues if issue.code == "PROCESSED_LIGHT"]
    assert issue.severity.value == "ERROR"
    assert (issue.details["count"], issue.details["calibrated"], issue.details["registered"]) == (3, 3, 1)
    assert "NO_LIGHTS" not in {issue.code for issue in inventory.issues}


def test_processed_lights_alone_leave_no_usable_light(tmp_path: Path) -> None:
    write_frame(tmp_path / "calibrated" / "Light_R" / "light_c.fits", "Light")

    codes = {issue.code for issue in inventory_project([tmp_path]).issues}

    assert codes == {"PROCESSED_LIGHT", "NO_LIGHTS"}


def test_byte_identical_light_copies_are_refused(tmp_path: Path) -> None:
    import shutil

    original = write_frame(tmp_path / "night" / "light.fits", "Light")
    copy = tmp_path / "copy" / "light.fits"
    copy.parent.mkdir()
    shutil.copyfile(original, copy)
    different = write_frame(tmp_path / "night" / "other.fits", "Light", exposure=121)

    inventory = inventory_project([tmp_path])

    status = {asset.path: asset.status for asset in inventory.assets}
    kept, dropped = sorted((str(original), str(copy)))
    assert status[kept] == AssetStatus.READY
    assert status[dropped] == AssetStatus.CONFLICT
    assert status[str(different)] == AssetStatus.READY
    (issue,) = [issue for issue in inventory.issues if issue.code == "DUPLICATE_LIGHT"]
    assert issue.details["copies"] == {dropped: kept}


def test_master_lights_are_ignored_with_a_warning(tmp_path: Path) -> None:
    write_frame(tmp_path / "raw" / "light.fits", "Light")
    write_frame(tmp_path / "master" / "masterLight_R.fits", "Master Light")

    inventory = inventory_project([tmp_path])

    (issue,) = [issue for issue in inventory.issues if issue.code == "MASTER_LIGHT_IGNORED"]
    assert issue.severity.value == "WARNING"
    assert inventory.has_errors is False
