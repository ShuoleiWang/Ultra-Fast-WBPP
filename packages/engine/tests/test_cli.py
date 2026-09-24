from __future__ import annotations

import json
from pathlib import Path

import pytest

from ufwbpp.cli import main
from ufwbpp.hardware import detect_hardware
from ufwbpp.metal_integration import metal_executor_available


def test_cli_doctor_distinguishes_executors_from_catalog_coverage(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"]["inventoryReady"] is True
    assert payload["status"]["pixelExecutionReady"] is True
    assert payload["status"]["catalogCoverageVerified"] is False
    assert payload["status"]["endToEndReady"] is False
    assert isinstance(payload["status"]["metalReady"], bool)
    assert "sirilReady" not in payload["status"]
    assert {backend["stage"] for backend in payload["backends"]} >= {
        "CALIBRATION",
        "DRIZZLE",
        "SOLVER",
    }
    # Platform facts: measured memory and cores, the tuning row, and whether
    # the native kernel library loaded (with its digest) or why not.
    assert payload["hardware"]["memoryBytes"] > 0
    assert payload["hardware"]["memorySource"] not in {"fallback", "unavailable"}
    assert payload["hardware"]["platformId"] in {"darwin", "windows", "linux"}
    assert payload["tuning"]["profileId"] == payload["hardware"]["optimizationProfile"] or (
        payload["tuning"]["profileId"] == "apple-silicon-generic-v1"
    )
    assert payload["tuning"]["memoryBytes"] == payload["hardware"]["memoryBytes"]
    native = payload["nativeKernels"]
    assert isinstance(native["loaded"], bool)
    if native["loaded"]:
        assert native["sha256"].startswith("sha256:")
        assert native["cpuArchitecture"] in {"x86-64", "arm64"}
        assert native["kernels"]
    else:
        assert native["reason"]


def test_cli_inventory_and_plan(
    nina_project: Path, capsys
) -> None:
    inputs = [
        str(nina_project / role) for role in ("LIGHT", "FLAT", "DARK", "BIAS")
    ]
    assert main(["inventory", *inputs, "--compact"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["counts"]["LIGHT"] == 1
    assert len(inventory["inputManifestSha256"]) == 64

    assert main(["plan", *inputs, "--compact"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["contractValid"] is True
    assert plan["claims"]["pixelExecutionImplemented"] is True



def test_metal_readiness_needs_apple_silicon_and_a_real_executor(monkeypatch) -> None:
    import ufwbpp.metal_integration as metal

    linux = detect_hardware(system="Linux", machine="x86_64", cpu_brand="x86_64")
    assert metal_executor_available(linux) is False

    apple = detect_hardware(system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro")

    class BrokenExecutor:
        def __enter__(self):
            raise OSError("native Metal library is absent")

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(metal, "NativeMetalExecutor", BrokenExecutor)
    assert metal_executor_available(apple) is False


@pytest.mark.parametrize("console_encoding", ["cp936", "ascii"])
def test_cli_streams_are_utf8_regardless_of_the_console_code_page(
    tmp_path: Path, console_encoding: str
) -> None:
    """A CJK path and a degree sign survive a Windows OEM console code page.

    ``PYTHONIOENCODING`` stands in for the console: Python would otherwise
    encode stdout with it, turning the JSON into cp936 bytes (mojibake for
    the UTF-8 reader on the other end) or failing with ``UnicodeEncodeError``
    under a code page that lacks the character.
    """

    import os
    import subprocess
    import sys

    from conftest import write_frame

    project = tmp_path / "目录 (12°)"
    write_frame(project / "LIGHT" / "M16_120s_R_001.fits", "Light")
    environment = {**os.environ, "PYTHONIOENCODING": console_encoding}
    environment.pop("PYTHONUTF8", None)
    completed = subprocess.run(
        [sys.executable, "-m", "ufwbpp", "inventory", str(project), "--compact"],
        capture_output=True,
        env=environment,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    payload = json.loads(completed.stdout.decode("utf-8"))
    rendered = json.dumps(payload, ensure_ascii=False)
    assert "目录 (12°)" in rendered
    # The bytes on the pipe are UTF-8, not the console code page's encoding.
    assert "目录 (12°)".encode("utf-8") in completed.stdout
