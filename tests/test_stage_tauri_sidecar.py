from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.stage_tauri_sidecar import SidecarStageError, stage_sidecar
from scripts.build_engine_sidecar import (
    build_manifest,
    detect_host_target_triple,
    runtime_directory_name,
    sidecar_filename,
)


TEST_TARGET = detect_host_target_triple()
MISMATCH_TARGET = (
    "aarch64-apple-darwin"
    if TEST_TARGET != "aarch64-apple-darwin"
    else "x86_64-pc-windows-msvc"
)


VERSIONS = {
    "python": "3.12.3",
    "pyinstaller": "6.22.2",
    "packages": {
        "astropy": "8.0.1",
        "light-frame-qc": "0.3.0",
        "ufwbpp": "0.1.0",
        "ufwbpp-registration": "0.1.0a1",
        "reproject": "0.21.0",
        "shapely": "2.1.0",
    },
}
def _pair(root: Path, payload: bytes = b"frozen-worker") -> Path:
    name = runtime_directory_name(TEST_TARGET)
    runtime = root / name
    runtime.mkdir()
    artifact = runtime / sidecar_filename(TEST_TARGET)
    artifact.write_bytes(payload)
    if os.name != "nt":
        artifact.chmod(0o755)
    (runtime / "_internal").mkdir()
    (runtime / "_internal" / "python.dat").write_bytes(b"python")
    manifest = root / f"{name}.manifest.json"
    manifest.write_text(
        json.dumps(
            build_manifest(
                runtime,
                TEST_TARGET,
                versions=VERSIONS,
                engine_version="0.1.0",
            )
        ),
        encoding="utf-8",
    )
    return manifest


def test_stages_only_the_identity_bound_target_create_only(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    manifest = _pair(source)
    artifact, copied_manifest = stage_sidecar(
        manifest,
        tmp_path / "tauri" / "binaries",
        expected_target=TEST_TARGET,
    )
    assert (artifact / sidecar_filename(TEST_TARGET)).read_bytes() == b"frozen-worker"
    assert copied_manifest.is_file()
    if os.name != "nt":
        assert (artifact / sidecar_filename(TEST_TARGET)).stat().st_mode & 0o111
    with pytest.raises(FileExistsError):
        stage_sidecar(manifest, artifact.parent)


@pytest.mark.parametrize("drift", ["bytes", "target", "symlink"])
def test_rejects_identity_target_and_file_type_drift(tmp_path: Path, drift: str) -> None:
    source = tmp_path / "input"
    source.mkdir()
    manifest = _pair(source)
    if drift == "bytes":
        runtime = source / runtime_directory_name(TEST_TARGET)
        (runtime / sidecar_filename(TEST_TARGET)).write_bytes(b"changed")
    elif drift == "target":
        with pytest.raises(SidecarStageError, match="not requested target"):
            stage_sidecar(
                manifest,
                tmp_path / "out",
                expected_target=MISMATCH_TARGET,
            )
        return
    else:
        runtime = source / runtime_directory_name(TEST_TARGET)
        artifact = runtime / sidecar_filename(TEST_TARGET)
        artifact.unlink()
        artifact.symlink_to("../" + manifest.name)
    with pytest.raises(SidecarStageError):
        stage_sidecar(manifest, tmp_path / "out")
