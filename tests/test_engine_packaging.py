from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_engine_sidecar.py"
SPEC = importlib.util.spec_from_file_location("build_engine_sidecar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
engine_packaging = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = engine_packaging
SPEC.loader.exec_module(engine_packaging)

resource_policy = sys.modules["_ufwbpp_engine_resource_policy"]
HOST_TARGET = engine_packaging.detect_host_target_triple()


VERSIONS = {
    "python": "3.12.3",
    "pyinstaller": "6.15.0",
    "packages": {
        "astropy": "8.0.1",
        "light-frame-qc": "0.3.0",
        "ufwbpp": "0.1.0",
        "ufwbpp-registration": "0.1.0a1",
        "reproject": "0.21.0",
        "shapely": "2.1.0",
    },
}
DOCTOR = {
    "schemaVersion": 1,
    "engineVersion": "0.1.0",
    "status": {"pixelExecutionReady": True, "solverReady": True},
    "nativeKernels": {"loaded": True, "sha256": "sha256:" + "0" * 64},
}

def _runtime(root: Path, target: str = HOST_TARGET) -> Path:
    runtime = root / engine_packaging.runtime_directory_name(target)
    internal = runtime / "_internal"
    internal.mkdir(parents=True)
    entry = runtime / engine_packaging.sidecar_filename(target)
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
        engine_packaging.detect_host_target_triple(system=system, machine=machine)
        == expected
    )
    suffix = ".exe" if "windows-msvc" in expected else ""
    assert engine_packaging.sidecar_filename(expected) == (
        f"ufwbpp-engine-{expected}{suffix}"
    )
    assert engine_packaging.manifest_filename(expected) == (
        f"ufwbpp-engine-{expected}.manifest.json"
    )


def test_unknown_or_noncanonical_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Tauri target triple"):
        engine_packaging.normalize_target_triple("arm64-apple-darwin")
    with pytest.raises(ValueError, match="unsupported Tauri target triple"):
        engine_packaging.normalize_target_triple("AARCH64-APPLE-DARWIN")


def test_manifest_attests_exact_runtime_tree_without_build_paths(tmp_path: Path) -> None:
    artifact = _runtime(tmp_path)
    payload = engine_packaging.build_manifest(
        artifact,
        HOST_TARGET,
        versions=VERSIONS,
        engine_version="0.1.0",
    )

    assert payload["schemaVersion"] == 3
    assert payload["runtime"]["directoryName"] == artifact.name
    assert payload["runtime"]["entryPoint"] == engine_packaging.sidecar_filename(
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
    assert payload["engine"] == {"version": "0.1.0"}
    rendered = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in rendered
    engine_packaging.validate_manifest(payload)

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / "packaging" / "engine" / "sidecar-manifest-v3.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(payload)


def test_manifest_and_target_publication_are_create_only(tmp_path: Path) -> None:
    target = HOST_TARGET
    artifact = tmp_path / engine_packaging.runtime_directory_name(target)
    artifact.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        engine_packaging.ensure_outputs_absent(tmp_path, target)
    assert artifact.is_dir()

    artifact.rmdir()
    staged_root = tmp_path / "stage"
    staged_root.mkdir()
    staged = _runtime(staged_root, target)
    payload = engine_packaging.build_manifest(
        staged, target, versions=VERSIONS, engine_version="0.1.0"
    )
    manifest = tmp_path / engine_packaging.manifest_filename(target)
    engine_packaging.write_manifest_create_only(manifest, payload)
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        engine_packaging.write_manifest_create_only(manifest, payload)
    assert manifest.read_bytes() == original

    staged_copy_root = tmp_path / "new-stage"
    staged_copy_root.mkdir()
    staged_copy = _runtime(staged_copy_root, target)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        engine_packaging._publish_runtime_create_only(staged_copy, staged)
    engine_packaging.verify_runtime_tree(staged, payload)


def test_resource_policy_excludes_tests_catalogs_raw_frames_and_local_paths() -> None:
    assert resource_policy.validate_resource_members(
        [
            "astropy/config/data/astropy.cfg",
            "reproject/mosaicking/__init__.py",
            "ufwbpp/py.typed",
            "_internal/ufwbpp/solvers/catalog_manifests/astrometry-net-4107-4112-v1.json",
        ]
    )
    rejected = [
        "astropy/tests/data/sample." + "fits",
        "solver/catalogs/index-4200.dat",
        "_internal/ufwbpp/solvers/catalog_manifests/unreviewed-manifest.json",
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
    payload = engine_packaging.build_manifest(
        artifact,
        HOST_TARGET,
        versions=VERSIONS,
        engine_version="0.1.0",
    )
    sensitive = deepcopy(payload)
    sensitive["versions"]["python"] = "/" + "Users/private/.venv/bin/python"
    with pytest.raises(resource_policy.ResourceBoundaryError, match="absolute path"):
        engine_packaging.validate_manifest(sensitive)

    injected = deepcopy(payload)
    injected["sourceRoot"] = str(tmp_path)
    with pytest.raises(resource_policy.ResourceBoundaryError, match="unexpected"):
        engine_packaging.validate_manifest(injected)


def test_build_runs_source_doctor_and_public_audit_before_pyinstaller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        engine_packaging,
        "detect_host_target_triple",
        lambda: HOST_TARGET,
    )

    def doctor() -> dict[str, object]:
        events.append("doctor")
        return dict(DOCTOR)

    def public_tree() -> None:
        events.append("public-tree")

    def versions() -> dict[str, object]:
        events.append("versions")
        raise engine_packaging.SidecarBuildError("STOP", "test stop")

    monkeypatch.setattr(engine_packaging, "run_source_doctor", doctor)
    monkeypatch.setattr(engine_packaging, "run_public_tree_check", public_tree)
    monkeypatch.setattr(engine_packaging, "collect_versions", versions)
    with pytest.raises(engine_packaging.SidecarBuildError, match="test stop"):
        engine_packaging.build_sidecar(HOST_TARGET, tmp_path)
    assert events == ["doctor", "public-tree", "versions"]
    assert list(tmp_path.iterdir()) == []


def test_engine_doctor_is_validated_and_native_kernels_are_required(tmp_path: Path) -> None:
    script = tmp_path / "fake_engine.py"
    script.write_text(
        "import json, sys\n"
        "assert sys.argv[1:] == ['doctor', '--json'], sys.argv\n"
        "print(open(sys.argv[0] + '.report').read())\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "fake_engine.py.report"
    command = [sys.executable, str(script)]

    report_path.write_text(json.dumps(DOCTOR), encoding="utf-8")
    report = engine_packaging.run_engine_doctor(command)
    assert report["engineVersion"] == "0.1.0"
    assert engine_packaging.loaded_native_kernel_facts(report)["sha256"].startswith("sha256:")

    missing = dict(DOCTOR, nativeKernels={"loaded": False, "reason": "native library not found"})
    report_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(engine_packaging.SidecarBuildError) as error:
        engine_packaging.loaded_native_kernel_facts(engine_packaging.run_engine_doctor(command))
    assert error.value.code == "NATIVE_KERNELS_MISSING"
    assert "native library not found" in str(error.value)

    not_ready = dict(DOCTOR, status={"pixelExecutionReady": False})
    report_path.write_text(json.dumps(not_ready), encoding="utf-8")
    with pytest.raises(engine_packaging.SidecarBuildError) as error:
        engine_packaging.run_engine_doctor(command)
    assert error.value.code == "DOCTOR_INVALID"

    report_path.write_text("not json", encoding="utf-8")
    with pytest.raises(engine_packaging.SidecarBuildError) as error:
        engine_packaging.run_engine_doctor(command)
    assert error.value.code == "DOCTOR_INVALID"
    with pytest.raises(ValueError):
        engine_packaging.run_engine_doctor([])


def test_production_launcher_without_a_command_prints_usage() -> None:
    completed = engine_packaging.subprocess.run(
        [sys.executable, str(ROOT / "packaging" / "engine" / "launcher.py")],
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=engine_packaging.source_environment(),
        timeout=30,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage: ultra-fast-wbpp" in completed.stderr


def test_production_launcher_dispatches_one_shot_cli_commands() -> None:
    completed = engine_packaging.subprocess.run(
        [
            sys.executable,
            str(ROOT / "packaging" / "engine" / "launcher.py"),
            "--version",
        ],
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=engine_packaging.source_environment(),
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.startswith("ultra-fast-wbpp ")


def test_packaging_spec_declares_every_required_collection() -> None:
    spec_text = (
        ROOT / "packaging" / "engine" / "ufwbpp_engine.spec"
    ).read_text(encoding="utf-8")
    policy_text = (
        ROOT / "packaging" / "engine" / "resource_policy.py"
    ).read_text(encoding="utf-8")
    assert "collect_data_files(" in spec_text
    assert "collect_all(" not in spec_text
    assert "collect_submodules(" not in spec_text
    assert "copy_metadata(distribution_name)" in spec_text
    assert "exclude_binaries=True" in spec_text
    assert "COLLECT(" in spec_text
    assert 'hookspath=[str(spec_dir / "hooks")]' in spec_text
    astropy_hook = (
        ROOT / "packaging" / "engine" / "hooks" / "hook-astropy.py"
    ).read_text(encoding="utf-8")
    assert "collect_submodules" not in astropy_hook
    assert '"astropy.io.fits"' in astropy_hook
    assert '"astropy.wcs"' in astropy_hook
    assert '"**/tests/**"' in astropy_hook
    for package_name in (
        "ufwbpp",
        "lightframeqc",
        "ufwbpp_registration",
        "astropy",
        "reproject",
    ):
        assert package_name in policy_text


def test_macos14_runtime_library_policy_pins_oci_and_exact_abi_set() -> None:
    policy = engine_packaging.load_macos14_runtime_policy()

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
    payload = engine_packaging.run_runtime_library_smoke(
        [sys.executable, str(ROOT / "packaging" / "engine" / "launcher.py")]
    )

    assert payload["schemaVersion"] == 1
    assert payload["openssl"].startswith("OpenSSL ")
    assert payload["libmpdec"]
    assert payload["python"]


def test_frozen_launcher_keeps_spawned_worker_pools_possible() -> None:
    """The QC/registration pools spawn children from the frozen executable.

    A child re-enters ``main`` with ``--multiprocessing-fork``; without
    ``freeze_support()`` before any argument dispatch it would run the command
    line instead of its task loop, and ``FrameRunner`` would fall back to
    threads (recording ``fallbackReason``).  The spec keeps a console
    subsystem so the children inherit the parent's standard streams.
    """

    import ast

    launcher_path = ROOT / "packaging" / "engine" / "launcher.py"
    tree = ast.parse(launcher_path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    statements = main.body

    def calls(node: ast.AST, qualified: str) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == qualified
        )

    freeze_index = next(
        index for index, node in enumerate(statements) if calls(node, "multiprocessing.freeze_support")
    )
    first_argv_use = next(
        index for index, node in enumerate(statements) if "sys.argv" in ast.unparse(node)
    )
    assert freeze_index < first_argv_use, "freeze_support() must run before any argument dispatch"

    spec_text = (ROOT / "packaging" / "engine" / "ufwbpp_engine.spec").read_text(encoding="utf-8")
    assert "console=True" in spec_text and "console=False" not in spec_text

    from lightframeqc import parallel

    source = (Path(parallel.__file__)).read_text(encoding="utf-8")
    assert "fallbackReason" in source and "freeze_support()" in source
    runner = parallel.FrameRunner(2, 20)
    assert runner.stats["fallbackReason"] is None


def test_source_environment_names_existing_package_roots() -> None:
    roots = engine_packaging.source_environment({})["PYTHONPATH"].split(os.pathsep)
    assert len(roots) == 3
    assert all(Path(root).is_dir() for root in roots)
