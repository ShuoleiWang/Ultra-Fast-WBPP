from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_worker_sidecar.py"
SPEC = importlib.util.spec_from_file_location("build_worker_sidecar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
worker_packaging = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker_packaging
SPEC.loader.exec_module(worker_packaging)

resource_policy = sys.modules["_openastroflow_worker_resource_policy"]
HOST_TARGET = worker_packaging.detect_host_target_triple()


VERSIONS = {
    "python": "3.12.3",
    "pyinstaller": "6.15.0",
    "packages": {
        "astropy": "8.0.1",
        "drizzle": "2.2.0",
        "light-frame-qc": "0.3.0",
        "openastroflow-engine": "0.1.0",
        "openastroflow-registration": "0.1.0a1",
        "reproject": "0.21.0",
        "shapely": "2.1.0",
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
            "stages": [
                "quality-control",
                "calibration",
                "registration",
                "integration",
                "astrometric-solve",
            ],
            "features": [
                "cpu-execution",
                "deterministic-receipts",
                "offline-astrometric-solver",
                "fits",
            ],
            "maximumParallelStages": 1,
            "inputExtensions": ["fit", "fits", "fts"],
            "outputExtensions": ["fits"],
        },
    },
}


def _runtime(root: Path, target: str = HOST_TARGET) -> Path:
    runtime = root / worker_packaging.runtime_directory_name(target)
    internal = runtime / "_internal"
    internal.mkdir(parents=True)
    entry = runtime / worker_packaging.sidecar_filename(target)
    entry.write_bytes(b"synthetic onedir launcher\n")
    if os.name != "nt":
        entry.chmod(0o755)
    (internal / "python-runtime.dat").write_bytes(b"runtime")
    return runtime


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Darwin", "arm64", "aarch64-apple-darwin"),
        ("Darwin", "x86_64", "x86_64-apple-darwin"),
        ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
        ("Windows", "ARM64", "aarch64-pc-windows-msvc"),
        ("Linux", "aarch64", "aarch64-unknown-linux-gnu"),
    ],
)
def test_tauri_target_detection_and_names(
    system: str, machine: str, expected: str
) -> None:
    assert (
        worker_packaging.detect_host_target_triple(system=system, machine=machine)
        == expected
    )
    suffix = ".exe" if "windows-msvc" in expected else ""
    assert worker_packaging.sidecar_filename(expected) == (
        f"openastroflow-worker-{expected}{suffix}"
    )
    assert worker_packaging.manifest_filename(expected) == (
        f"openastroflow-worker-{expected}.manifest.json"
    )


def test_unknown_or_noncanonical_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Tauri target triple"):
        worker_packaging.normalize_target_triple("arm64-apple-darwin")
    with pytest.raises(ValueError, match="unsupported Tauri target triple"):
        worker_packaging.normalize_target_triple("AARCH64-APPLE-DARWIN")


def test_manifest_attests_exact_runtime_tree_without_build_paths(tmp_path: Path) -> None:
    artifact = _runtime(tmp_path)
    payload = worker_packaging.build_manifest(
        artifact,
        HOST_TARGET,
        versions=VERSIONS,
        handshake=HANDSHAKE,
    )

    assert payload["schemaVersion"] == 2
    assert payload["runtime"]["directoryName"] == artifact.name
    assert payload["runtime"]["entryPoint"] == worker_packaging.sidecar_filename(
        HOST_TARGET
    )
    assert payload["runtime"]["fileCount"] == 2
    assert payload["runtime"]["sizeBytes"] == sum(
        entry["sizeBytes"]
        for entry in payload["runtime"]["entries"]
        if entry["type"] == "file"
    )
    assert len(payload["runtime"]["treeSha256"]) == 64
    assert payload["versions"] == VERSIONS
    assert payload["collections"] == list(resource_policy.REQUIRED_COLLECTIONS)
    assert payload["protocol"]["capabilities"]["stages"] == [
        "quality-control",
        "calibration",
        "registration",
        "integration",
        "astrometric-solve",
    ]
    rendered = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in rendered
    worker_packaging.validate_manifest(payload)

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / "packaging" / "worker" / "sidecar-manifest-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(payload)


def test_manifest_and_target_publication_are_create_only(tmp_path: Path) -> None:
    target = HOST_TARGET
    artifact = tmp_path / worker_packaging.runtime_directory_name(target)
    artifact.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        worker_packaging.ensure_outputs_absent(tmp_path, target)
    assert artifact.is_dir()

    artifact.rmdir()
    staged_root = tmp_path / "stage"
    staged_root.mkdir()
    staged = _runtime(staged_root, target)
    payload = worker_packaging.build_manifest(
        staged, target, versions=VERSIONS, handshake=HANDSHAKE
    )
    manifest = tmp_path / worker_packaging.manifest_filename(target)
    worker_packaging.write_manifest_create_only(manifest, payload)
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        worker_packaging.write_manifest_create_only(manifest, payload)
    assert manifest.read_bytes() == original

    staged_copy_root = tmp_path / "new-stage"
    staged_copy_root.mkdir()
    staged_copy = _runtime(staged_copy_root, target)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        worker_packaging._publish_runtime_create_only(staged_copy, staged)
    worker_packaging.verify_runtime_tree(staged, payload)


def test_resource_policy_excludes_tests_catalogs_raw_frames_and_local_paths() -> None:
    assert resource_policy.validate_resource_members(
        [
            "astropy/config/data/astropy.cfg",
            "drizzle/cdrizzle.cpython-312-darwin.so",
            "openastroflow_engine/py.typed",
            "_internal/resources/catalogs/astrometry-net-4107-4112-v1.json",
        ]
    )
    rejected = [
        "astropy/tests/data/sample." + "fits",
        "solver/catalogs/index-4200.dat",
        "_internal/resources/catalogs/unreviewed-manifest.json",
        "../outside/module.py",
        "/" + "Users/private/build/module.py",
        "C:" + "\\Users\\private\\build\\module.py",
    ]
    for member in rejected:
        with pytest.raises(resource_policy.ResourceBoundaryError):
            resource_policy.validate_resource_members([member])

    source = "/" + "Users/private/site-packages/astropy/config/data/astropy.cfg"
    entries = [
        (source, "astropy/config/data"),
        (source, "astropy/tests/data"),
    ]
    assert resource_policy.filter_pyinstaller_entries(entries) == [entries[0]]
    assert resource_policy.module_is_runtime("astropy.io.fits") is True
    assert resource_policy.module_is_runtime("astropy.visualization") is False
    assert resource_policy.module_is_runtime("astropy.visualization.wcsaxes") is False


def test_manifest_rejects_sensitive_version_and_injected_fields(tmp_path: Path) -> None:
    artifact = _runtime(tmp_path)
    payload = worker_packaging.build_manifest(
        artifact,
        HOST_TARGET,
        versions=VERSIONS,
        handshake=HANDSHAKE,
    )
    sensitive = deepcopy(payload)
    sensitive["versions"]["python"] = "/" + "Users/private/.venv/bin/python"
    with pytest.raises(resource_policy.ResourceBoundaryError, match="absolute path"):
        worker_packaging.validate_manifest(sensitive)

    injected = deepcopy(payload)
    injected["sourceRoot"] = str(tmp_path)
    with pytest.raises(resource_policy.ResourceBoundaryError, match="unexpected"):
        worker_packaging.validate_manifest(injected)


def test_build_runs_source_handshake_and_public_audit_before_pyinstaller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        worker_packaging,
        "detect_host_target_triple",
        lambda: HOST_TARGET,
    )

    def handshake() -> dict[str, object]:
        events.append("handshake")
        return dict(HANDSHAKE)

    def public_tree() -> None:
        events.append("public-tree")

    def versions() -> dict[str, object]:
        events.append("versions")
        raise worker_packaging.SidecarBuildError("STOP", "test stop")

    monkeypatch.setattr(worker_packaging, "run_source_handshake", handshake)
    monkeypatch.setattr(worker_packaging, "run_public_tree_check", public_tree)
    monkeypatch.setattr(worker_packaging, "collect_versions", versions)
    with pytest.raises(worker_packaging.SidecarBuildError, match="test stop"):
        worker_packaging.build_sidecar(HOST_TARGET, tmp_path)
    assert events == ["handshake", "public-tree", "versions"]
    assert list(tmp_path.iterdir()) == []


def test_production_launcher_rejects_the_legacy_unbound_handshake() -> None:
    completed = worker_packaging.subprocess.run(
        [sys.executable, str(ROOT / "packaging" / "worker" / "launcher.py")],
        input='{"id":"packaging-smoke","type":"handshake","protocolVersion":1}\n',
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=worker_packaging.source_environment(),
        timeout=30,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""


def test_production_launcher_dispatches_one_shot_cli_commands() -> None:
    completed = worker_packaging.subprocess.run(
        [
            sys.executable,
            str(ROOT / "packaging" / "worker" / "launcher.py"),
            "--version",
        ],
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=worker_packaging.source_environment(),
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.startswith("ultra-fast-wbpp ")


def test_packaging_spec_declares_every_required_collection() -> None:
    spec_text = (
        ROOT / "packaging" / "worker" / "openastroflow_worker.spec"
    ).read_text(encoding="utf-8")
    policy_text = (
        ROOT / "packaging" / "worker" / "resource_policy.py"
    ).read_text(encoding="utf-8")
    assert "collect_data_files(" in spec_text
    assert "collect_all(" not in spec_text
    assert "collect_submodules(" not in spec_text
    assert "copy_metadata(distribution_name)" in spec_text
    assert "exclude_binaries=True" in spec_text
    assert "COLLECT(" in spec_text
    assert 'hookspath=[str(spec_dir / "hooks")]' in spec_text
    astropy_hook = (
        ROOT / "packaging" / "worker" / "hooks" / "hook-astropy.py"
    ).read_text(encoding="utf-8")
    assert "collect_submodules" not in astropy_hook
    assert '"astropy.io.fits"' in astropy_hook
    assert '"astropy.wcs"' in astropy_hook
    assert '"**/tests/**"' in astropy_hook
    for package_name in (
        "openastroflow_engine",
        "lightframeqc",
        "openastroflow_registration",
        "astropy",
        "drizzle",
    ):
        assert package_name in policy_text


def test_macos14_runtime_library_policy_pins_oci_and_exact_abi_set() -> None:
    policy = worker_packaging.load_macos14_runtime_policy()

    assert policy["targetTriple"] == "aarch64-apple-darwin"
    assert policy["maximumDeploymentTarget"] == "14.0.0"
    assert policy["ociSource"] == {
        "registry": "ghcr.io",
        "apiBaseUrl": "https://ghcr.io/v2",
        "tokenUrl": "https://ghcr.io/token",
        "tokenService": "ghcr.io",
        "maximumRedirects": 3,
        "allowedRedirectHosts": ["pkg-containers.githubusercontent.com"],
        "allowedRedirectHostSuffixes": [".blob.core.windows.net"],
    }
    assert policy["runtimeSmoke"] == {
        "opensslPrefix": "OpenSSL 3.6.4 ",
        "libmpdecVersion": "4.0.1",
    }
    libraries = {
        library["destinationName"]: library
        for package in policy["packages"]
        for library in package["libraries"]
    }
    assert set(libraries) == {
        "libcrypto.3.dylib",
        "libssl.3.dylib",
        "libmpdec.4.dylib",
    }
    assert all(len(library["sourceSha256"]) == 64 for library in libraries.values())
    assert {
        package["bottleTag"] for package in policy["packages"]
    } == {"arm64_sonoma"}
    assert {package["oci"]["repository"] for package in policy["packages"]} == {
        "homebrew/core/openssl/3",
        "homebrew/core/mpdecimal",
    }
    assert all(
        package["oci"]["manifestDigest"]
        == f"sha256:{package['oci']['manifestSha256']}"
        for package in policy["packages"]
    )
    assert "/private/" not in json.dumps(policy, sort_keys=True)
    assert "/Users/" not in json.dumps(policy, sort_keys=True)


def test_source_launcher_exposes_runtime_library_abi_smoke() -> None:
    payload = worker_packaging.run_runtime_library_smoke(
        [sys.executable, str(ROOT / "packaging" / "worker" / "launcher.py")]
    )

    assert payload["schemaVersion"] == 1
    assert payload["openssl"].startswith("OpenSSL ")
    assert payload["libmpdec"]
    assert payload["python"]
