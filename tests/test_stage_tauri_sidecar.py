from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.stage_tauri_sidecar import SidecarStageError, stage_sidecar
from scripts.build_worker_sidecar import (
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
        "drizzle": "2.2.0",
        "light-frame-qc": "0.3.0",
        "openastroflow-engine": "0.1.0",
        "openastroflow-registration": "0.1.0a1",
        "reproject": "0.21.0",
        "shapely": "2.1.0",
        "xisf": "0.9.6",
    },
}
HANDSHAKE = {
    "protocolVersion": 1,
    "sessionId": "packaging-smoke",
    "sequence": 0,
    "sentAtUnixMs": 1,
    "type": "handshake",
    "payload": {
        "role": "worker",
        "implementation": "openastroflow-python-worker",
        "implementationVersion": "0.1.0",
        "supportedProtocolVersions": [1],
        "capabilities": {
            "schemaVersion": 1,
            "backendId": "openastroflow-python-worker",
            "backendVersion": "0.1.0",
            "workerBuild": "test",
            "hardwareProfiles": ["generic-arm64-cpu"],
            "stages": ["quality-control", "calibration", "registration", "integration", "astrometric-solve"],
            "features": ["cpu-execution", "deterministic-receipts", "offline-astrometric-solver", "fits"],
            "maximumParallelStages": 1,
            "inputExtensions": ["fit", "fits", "fts"],
            "outputExtensions": ["fits"],
        },
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
                handshake=HANDSHAKE,
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
