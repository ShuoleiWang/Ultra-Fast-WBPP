from __future__ import annotations

from pathlib import Path

from ufwbpp.inventory import inventory_project
from ufwbpp.models import AssetRole, AssetStatus

from astropy.io import fits
import pytest

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


@pytest.mark.parametrize(
    ("image_type", "role"),
    [
        ("Light Frame", "LIGHT"),
        ("Flat Field", "FLAT"),
        ("Flat Frame", "FLAT"),
        ("Dark Frame", "DARK"),
        ("Bias Frame", "BIAS"),
        ("DARKFLAT", "DARK"),
        ("Master Dark", "MASTER_DARK"),
        ("Science", "LIGHT"),
        ("Lights", "LIGHT"),
        ("zero", "BIAS"),
        ("SNAPSHOT", "UNKNOWN"),
    ],
)
def test_inventory_and_pixel_reader_read_one_role_table(tmp_path: Path, image_type: str, role: str) -> None:
    from ufwbpp.stacking.integration import read_frame_info

    frame = write_frame(tmp_path / "frames" / "frame.fits", "Light")
    fits.setval(frame, "IMAGETYP", value=image_type)

    (asset,) = inventory_project([tmp_path]).assets

    assert asset.role.value == role
    assert read_frame_info(frame).role == role


def test_inventory_and_pixel_reader_name_a_filter_alike(tmp_path: Path) -> None:
    from ufwbpp.calibration.inputs import with_path_metadata
    from ufwbpp.stacking.integration import read_frame_info

    for index, (spelling, expected) in enumerate(
        (("Red", "R"), ("Lum", "L"), ("H-alpha", "HA"), ("O3", "OIII"), ("L-Pro", "L-PRO"), (None, "NOFILTER"), ("None", "NOFILTER"))
    ):
        frame = write_frame(tmp_path / str(index) / "frame.fits", "Light")
        if spelling is None:
            fits.delval(frame, "FILTER")
        else:
            fits.setval(frame, "FILTER", value=spelling)
        (asset,) = inventory_project([frame]).assets
        assert asset.filter_name == expected
        assert with_path_metadata(read_frame_info(frame), frame).filter_name == expected


def test_housekeeping_files_beside_the_frames_are_not_inventoried(tmp_path: Path) -> None:
    light = write_frame(tmp_path / "LIGHT" / "L_0001.fits", "Light")
    (light.parent / "._L_0001.fits").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 60)
    (light.parent / ".DS_Store").write_bytes(b"store")
    write_frame(tmp_path / ".out.a1b2c3d4.pstage" / "masters" / "master_light_R.fits", "Light")

    inventory = inventory_project([tmp_path])

    assert [asset.path for asset in inventory.assets] == [str(light.resolve())]
    assert inventory.has_errors is False


def test_lights_without_object_share_a_target_and_never_get_one_each(tmp_path: Path) -> None:
    written: list[Path] = []

    def light(path: Path, target: str | None = None) -> Path:
        frame = write_frame(path, "Light", target=target or "x")
        if target is None:
            fits.delval(frame, "OBJECT")
        # Distinct bytes: identical copies of a Light are refused.
        fits.setval(frame, "DATE-OBS", value=f"2026-03-22T12:{len(written):02d}:00Z")
        written.append(frame)
        return frame

    # N.I.N.A.'s default file pattern: a date and time before the exposure.
    for index in range(3):
        light(tmp_path / "2026-03-22" / "LIGHT" / f"2026-03-22_21-1{index}-11_R_-10.00_120.00s_000{index}.fits")
    # Two nights of one target named by its folder, one of them with a Chinese name.
    for night in ("2026-03-22", "2026-03-23"):
        light(tmp_path / "M31" / night / "LIGHT" / "img_0001.fits")
        light(tmp_path / f"玫瑰星云_{night}" / "LIGHT" / "img_0001.fits")
    # A header OBJECT that astropy could not decode is no name either.
    light(tmp_path / "NGC 7000" / "LIGHT" / "img_0001.fits", target="????")
    # A file name that states the target of every Light of its folder is kept.
    for index in range(2):
        light(tmp_path / "named" / f"M42_120s_000{index}.fits")
    named = light(tmp_path / "header" / "img_0001.fits", target="M 51")

    inventory = inventory_project([tmp_path])

    targets: dict[str, int] = {}
    for asset in inventory.assets:
        targets[asset.target] = targets.get(asset.target, 0) + 1
    assert targets == {"UNNAMED": 3, "M31": 2, "玫瑰星云": 2, "NGC 7000": 1, "M42": 2, "M 51": 1}
    assert {issue.severity.value for issue in inventory.issues} == {"WARNING"}
    codes = sorted((issue.code, issue.details["target"], issue.details["count"]) for issue in inventory.issues)
    assert codes == [
        ("TARGET_NAMED_BY_FOLDER", "M31", 2),
        ("TARGET_NAMED_BY_FOLDER", "NGC 7000", 1),
        ("TARGET_NAMED_BY_FOLDER", "玫瑰星云", 2),
        ("TARGET_UNNAMED", "UNNAMED", 3),
    ]
    assert inventory.has_errors is False
    # Lights of one target and set-up are one group whatever named them.
    by_target: dict[str, set[str]] = {}
    for asset in inventory.assets:
        by_target.setdefault(asset.target, set()).add(asset.group_id)
    assert all(len(groups) == 1 for groups in by_target.values())
    assert by_target["M 51"] == {next(asset.group_id for asset in inventory.assets if asset.path == str(named.resolve()))}


def test_an_odd_bayer_offset_is_reported_because_it_is_not_applied(tmp_path: Path) -> None:
    frame = write_frame(tmp_path / "LIGHT" / "osc.fits", "Light")
    fits.setval(frame, "BAYERPAT", value="RGGB")
    fits.setval(frame, "YBAYROFF", value=1)
    even = write_frame(tmp_path / "LIGHT" / "osc_even.fits", "Light", exposure=60.0)
    fits.setval(even, "BAYERPAT", value="RGGB")
    fits.setval(even, "XBAYROFF", value=0)

    inventory = inventory_project([tmp_path])

    assert [(issue.code, Path(issue.path).name) for issue in inventory.issues] == [("BAYER_OFFSET_NOT_APPLIED", "osc.fits")]
