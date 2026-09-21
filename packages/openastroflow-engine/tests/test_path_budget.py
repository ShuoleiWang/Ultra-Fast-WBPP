"""Projected path lengths and the fail-closed ``OUTPUT_PATH_TOO_LONG`` check.

Every case injects the path limit so the Windows behaviour (259 characters
without the long-path policy, 32767 with it) is exercised on every host.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import pytest

from openastroflow_engine import e2e, pixel_pipeline, project_e2e
from openastroflow_engine import platform as platform_services
from openastroflow_engine.hardware import detect_hardware
from openastroflow_engine.inventory import inventory_project
from openastroflow_engine.path_budget import (
    PIXEL_PIPELINE_STAGING_STEM,
    PROJECT_STAGING_SUFFIX,
    RANDOM_NAME_LENGTH,
    STAGING_SUFFIX,
    check_output_path_budget,
    frame_index_width,
    light_stem,
    name_token,
    projected_maximum_path,
    staging_directory_name,
    target_key,
)
from openastroflow_engine.platform import PathLimit
from openastroflow_engine.platform.windows import LONG_PATHS_REGISTRY_KEY, path_limit
from openastroflow_engine.recipe import Recipe
from openastroflow_engine.runtime import (
    RuntimeConfigurationError,
    check_run_path_budget,
    prepare_execution,
    prepare_project_execution,
)


LONG_STEM = "2026-09-17_22-29-49_NGC_7331_R_-10.00_300.00s_0001"


def test_layout_tokens_match_the_pipeline_rules() -> None:
    # The directory tokens the pipeline modules derive are the ones the
    # projection uses; a drift here would make the budget wrong.
    for value in ("Ha 3nm", "r", "NGC 7331", "L-Pro", "SII_6.5nm"):
        assert e2e._safe_token(value) == name_token(value)
        assert pixel_pipeline._safe_token(value) == name_token(value)
        assert project_e2e._normalized_token(value) == target_key(value)
    assert name_token("Ha 3nm") == "HA_3NM"
    assert target_key("NGC 7331") == "ngc7331"
    assert light_stem(Path("/data/M 31 (R) #1.fits")) == "M_31_R_1"
    assert light_stem(Path("/data/___.fits")) == "light"
    assert frame_index_width(63) == 5 and frame_index_width(123456) == 6
    assert staging_directory_name("out") == f".out.{'x' * RANDOM_NAME_LENGTH}{STAGING_SUFFIX}"
    assert staging_directory_name("out", suffix=PROJECT_STAGING_SUFFIX).endswith(".pstage")


def test_projection_walks_the_nested_staging_layout(tmp_path: Path) -> None:
    output = tmp_path / "out"
    lights = [tmp_path / f"{LONG_STEM}.fits", tmp_path / "short.fits", tmp_path / "frame.xisf"]
    length, deepest = projected_maximum_path(
        output, targets=["NGC 7331", "M 31"], filters=["R", "Ha 3nm"], light_count=63, light_paths=lights
    )
    assert length == len(deepest)
    relative = Path(deepest).relative_to(output.parent.resolve()).as_posix()
    parts = relative.split("/")
    random = "x" * RANDOM_NAME_LENGTH
    # project staging / details / runs / run staging (longest target) /
    # work / pixel staging / <deepest temporary carrying the longest stem>.
    assert parts[0] == f".out.{random}{PROJECT_STAGING_SUFFIX}"
    assert parts[1:3] == ["details", "runs"]
    assert parts[3] == f".NGC7331.{random}{STAGING_SUFFIX}"
    assert parts[4] == "work"
    assert parts[5] == f".{PIXEL_PIPELINE_STAGING_STEM}.{random}{STAGING_SUFFIX}"
    assert LONG_STEM in parts[-1]
    assert parts[-1].endswith(".partial")
    # Shallower layouts project shorter paths; a bare pipeline shortest.
    run_length, run_deepest = projected_maximum_path(
        output, filters=["R"], light_count=63, light_paths=lights, layout="run"
    )
    pixel_length, pixel_deepest = projected_maximum_path(
        output, filters=["R"], light_count=63, light_paths=lights, layout="pixels"
    )
    assert length > run_length > pixel_length
    assert f".out.{random}{STAGING_SUFFIX}/work/" in run_deepest.replace("\\", "/")
    assert f".out.{random}{STAGING_SUFFIX}/" in pixel_deepest.replace("\\", "/")
    # A long filter token changes which candidate is deepest, never the layout.
    long_filter_length, long_filter_deepest = projected_maximum_path(
        output, filters=["Hydrogen alpha ultra narrow band"], light_count=3, light_paths=[tmp_path / "a.fits"]
    )
    assert "HYDROGEN_ALPHA_ULTRA_NARROW_BAND" in long_filter_deepest
    assert long_filter_length == len(long_filter_deepest)
    with pytest.raises(ValueError):
        projected_maximum_path(output, layout="nested")  # type: ignore[arg-type]


def test_real_staging_names_follow_the_projected_pattern(tmp_path: Path) -> None:
    import tempfile

    random = re.escape("x" * RANDOM_NAME_LENGTH).replace("x" * RANDOM_NAME_LENGTH, "[a-z0-9_]{8}")
    for stem, suffix in (("out", STAGING_SUFFIX), ("out", PROJECT_STAGING_SUFFIX), (PIXEL_PIPELINE_STAGING_STEM, STAGING_SUFFIX)):
        created = Path(tempfile.mkdtemp(prefix=f".{stem}.", suffix=suffix, dir=tmp_path))
        pattern = re.escape(staging_directory_name(stem, suffix=suffix)).replace("x" * RANDOM_NAME_LENGTH, random)
        assert re.fullmatch(pattern, created.name), created.name
        assert len(created.name) == len(staging_directory_name(stem, suffix=suffix))


def test_check_refuses_with_the_projection_the_limit_and_both_remedies(tmp_path: Path) -> None:
    output = tmp_path / "out"
    lights = [tmp_path / f"{LONG_STEM}.fits"]
    projected, deepest = projected_maximum_path(output, filters=["R"], light_count=1, light_paths=lights)
    disabled = path_limit(False)
    assert disabled.max_characters == 259 and disabled.long_paths_enabled is False
    tight = replace(disabled, max_characters=projected - 1)
    with pytest.raises(RuntimeConfigurationError) as refused:
        check_output_path_budget(output, filters=["R"], light_count=1, light_paths=lights, limit=tight)
    assert refused.value.code == "OUTPUT_PATH_TOO_LONG"
    message = str(refused.value)
    assert f"up to {projected} characters" in message
    assert f"at most {projected - 1}" in message
    assert deepest in message
    assert "shorter output directory" in message
    budget = (projected - 1) - (projected - len(str(output.resolve())))
    assert f"at most {budget} characters" in message
    assert LONG_PATHS_REGISTRY_KEY in message and "enable long paths" in message
    # Exactly at the limit fits; long paths enabled and POSIX never refuse.
    exact = replace(disabled, max_characters=projected)
    assert check_output_path_budget(output, filters=["R"], light_paths=lights, limit=exact) == (projected, deepest)
    assert check_output_path_budget(output, filters=["R"], light_paths=lights, limit=path_limit(True))[0] == projected
    assert check_output_path_budget(output, filters=["R"], light_paths=lights, limit=PathLimit(None, None, "unlimited"))[0] == projected
    # An unreadable policy keeps the short limit and says so in the message.
    default_projection, _ = projected_maximum_path(output)
    unknown = PathLimit(default_projection - 1, None, "MAX_PATH (registry unavailable)")
    with pytest.raises(RuntimeConfigurationError) as unreadable:
        check_output_path_budget(output, limit=unknown)
    assert "could not be read" in str(unreadable.value)


def test_prepare_execution_refuses_a_long_output_before_planning(nina_project: Path, tmp_path: Path) -> None:
    inventory = inventory_project([nina_project])
    recipe = Recipe.from_dict({"calibration": {"allowMasters": False}})
    windows = detect_hardware(system="Windows", machine="AMD64", cpu_brand="AMD Ryzen 7 5800H")
    # A 200-character output directory (the laptop acceptance case): fine
    # with long paths, over MAX_PATH once the nested staging names and the
    # Light stem are added.
    target_length = max(200, len(str(tmp_path)) + 40)
    output = tmp_path / ("o" * (target_length - len(str(tmp_path)) - 1))
    assert len(str(output)) == target_length
    short_limit = replace(windows, path_limit=path_limit(False))
    with pytest.raises(RuntimeConfigurationError) as refused:
        prepare_execution(inventory, recipe, output, hardware=short_limit)
    assert refused.value.code == "OUTPUT_PATH_TOO_LONG"
    with pytest.raises(RuntimeConfigurationError) as project_refused:
        prepare_project_execution(inventory, recipe, output, hardware=short_limit)
    assert project_refused.value.code == "OUTPUT_PATH_TOO_LONG"
    # The project layout is the deeper one: its projection exceeds the run's.
    project_length = int(re.search(r"up to (\d+) characters", str(project_refused.value)).group(1))
    run_length = int(re.search(r"up to (\d+) characters", str(refused.value)).group(1))
    assert project_length > run_length
    long_paths = replace(windows, path_limit=path_limit(True))
    check_run_path_budget(inventory, output, long_paths, layout="project")
    # An injected profile without a limit defers to the running platform: a
    # Windows host without long paths refuses this output like the injected
    # short limit did, every other host accepts it.
    host_limit = platform_services.current().path_limit().max_characters
    if host_limit is not None and host_limit < project_length:
        with pytest.raises(RuntimeConfigurationError) as host_refused:
            check_run_path_budget(inventory, output, windows, layout="project")
        assert host_refused.value.code == "OUTPUT_PATH_TOO_LONG"
    else:
        check_run_path_budget(inventory, output, windows, layout="project")


def test_prepare_execution_refuses_windows_outside_x86_64(nina_project: Path, tmp_path: Path) -> None:
    inventory = inventory_project([nina_project])
    recipe = Recipe.from_dict({"calibration": {"allowMasters": False}})
    arm = detect_hardware(system="Windows", machine="ARM64", cpu_brand="Snapdragon X Elite")
    with pytest.raises(RuntimeConfigurationError) as refused:
        prepare_execution(inventory, recipe, tmp_path / "out", hardware=arm)
    assert refused.value.code == "PLATFORM_UNSUPPORTED"
    assert "arm64" in str(refused.value) and "x86-64" in str(refused.value)
    with pytest.raises(RuntimeConfigurationError) as project_refused:
        prepare_project_execution(inventory, recipe, tmp_path / "out", hardware=arm)
    assert project_refused.value.code == "PLATFORM_UNSUPPORTED"
    # x86-64 Windows and every other platform pass this gate (the plan's own
    # readiness checks follow).
    from openastroflow_engine.runtime import refuse_unsupported_platform

    refuse_unsupported_platform(detect_hardware(system="Windows", machine="AMD64", cpu_brand="x"))
    refuse_unsupported_platform(detect_hardware(system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro"))
    refuse_unsupported_platform(detect_hardware(system="Linux", machine="aarch64", cpu_brand="x"))
