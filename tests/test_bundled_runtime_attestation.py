from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import plistlib
import stat
import struct

import pytest

from scripts.attest_bundled_runtime import (
    BundleAttestationError,
    _attest_runtime_libraries,
    attest_legal_resources,
    attest_macos_deployment,
    attest_runtime,
    inspect_macos_bundle,
    write_create_only,
)
from scripts.build_worker_sidecar import (
    build_manifest,
    detect_host_target_triple,
    manifest_filename,
    runtime_directory_name,
    sidecar_filename,
)


TARGET = detect_host_target_triple()
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
WORKER_HANDSHAKE = {
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


def _packed_version(version: tuple[int, int, int]) -> int:
    return version[0] << 16 | version[1] << 8 | version[2]


def _string_command(command_id: int, header_size: int, value: str) -> bytes:
    encoded = value.encode("utf-8") + b"\0"
    command_size = (header_size + len(encoded) + 3) & ~3
    if header_size == 24:
        header = struct.pack("<IIIIII", command_id, command_size, header_size, 0, 0, 0)
    elif header_size == 12:
        header = struct.pack("<III", command_id, command_size, header_size)
    else:  # pragma: no cover - test helper misuse
        raise AssertionError("unsupported synthetic load-command header")
    return header + encoded + b"\0" * (command_size - header_size - len(encoded))


def _thin_macho_bytes(
    *,
    minimum: tuple[int, int, int] | None = (14, 0, 0),
    file_type: int = 6,
    dependencies: tuple[str, ...] = (),
    rpaths: tuple[str, ...] = (),
    cpu_type: int = 0x0100000C,
) -> bytes:
    commands: list[bytes] = []
    if minimum is not None:
        commands.append(
            struct.pack(
                "<IIIIII",
                0x32,
                24,
                1,
                _packed_version(minimum),
                _packed_version((14, 4, 0)),
                0,
            )
        )
    commands.extend(_string_command(0x0C, 24, item) for item in dependencies)
    commands.extend(_string_command(0x8000001C, 12, item) for item in rpaths)
    command_blob = b"".join(commands)
    header = struct.pack(
        "<IiiIIIII",
        0xFEEDFACF,
        cpu_type,
        0,
        file_type,
        len(commands),
        len(command_blob),
        0,
        0,
    )
    return header + command_blob


def _fat_macho_bytes(slices: tuple[tuple[int, bytes], ...]) -> bytes:
    table_size = 8 + 20 * len(slices)
    offset = table_size
    entries: list[bytes] = []
    payloads: list[bytes] = []
    for cpu_type, payload in slices:
        entries.append(struct.pack(">iiIII", cpu_type, 0, offset, len(payload), 0))
        payloads.append(payload)
        offset += len(payload)
    return struct.pack(">II", 0xCAFEBABE, len(slices)) + b"".join(entries + payloads)


def _test_app(root: Path) -> Path:
    app = root / "Ultra-Fast WBPP.app"
    macos = app / "Contents" / "MacOS"
    frameworks = app / "Contents" / "Frameworks"
    macos.mkdir(parents=True)
    frameworks.mkdir(parents=True)
    with (app / "Contents" / "Info.plist").open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleExecutable": "Ultra-Fast WBPP",
                "LSMinimumSystemVersion": "14.0",
            },
            stream,
        )
    (macos / "Ultra-Fast WBPP").write_bytes(
        _thin_macho_bytes(
            file_type=2,
            dependencies=(
                "@rpath/libworker.dylib",
                "/usr/lib/libSystem.B.dylib",
            ),
            rpaths=("@executable_path/../Frameworks",),
        )
    )
    (frameworks / "libworker.dylib").write_bytes(_thin_macho_bytes())
    return app


def _resource_root(root: Path) -> Path:
    resource = root / "openastroflow-worker"
    runtime = resource / runtime_directory_name(TARGET)
    runtime.mkdir(parents=True)
    entry = runtime / sidecar_filename(TARGET)
    handshake = json.dumps(WORKER_HANDSHAKE, sort_keys=True, separators=(",", ":"))
    catalogs = json.dumps(
        {
            "schemaVersion": 1,
            "catalogs": [
                {"catalogId": "astap-external"},
                {"catalogId": "astrometry-net-4107-4112"},
                {"catalogId": "astrometry-net-4108"},
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    entry.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "ultra-fast-wbpp 0.1.0"; exit 0; fi\n'
        f"if [ \"$1\" = \"catalog\" ]; then printf '%s\\n' '{catalogs}'; exit 0; fi\n"
        'if [ "$1" = "run-project" ]; then echo "usage: ultra-fast-wbpp run-project"; exit 0; fi\n'
        "read request\n"
        f"printf '%s\\n' '{handshake}'\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        entry.chmod(entry.stat().st_mode | stat.S_IXUSR)
    internal = runtime / "_internal"
    internal.mkdir()
    (internal / "python.dat").write_bytes(b"python")
    manifest = build_manifest(
        runtime, TARGET, versions=VERSIONS, handshake=WORKER_HANDSHAKE
    )
    (resource / manifest_filename(TARGET)).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return resource


@pytest.mark.skipif(
    os.name == "nt",
    reason="uses a POSIX shell launcher and executable-bit semantics",
)
def test_post_sign_attestation_binds_actual_tree_and_launch_budget(tmp_path: Path) -> None:
    resource = _resource_root(tmp_path)
    runtime = resource / runtime_directory_name(TARGET)
    entry = runtime / sidecar_filename(TARGET)
    # Simulate signing changing entry-point bytes after the pre-sign manifest.
    entry.write_text(entry.read_text(encoding="utf-8") + "# signed bytes\n", encoding="utf-8")
    payload = attest_runtime(
        resource,
        TARGET,
        max_start_seconds=5.0,
        bundle_name="Ultra-Fast WBPP.app",
        signature={"verified": True, "mode": "ad-hoc", "hardenedRuntime": True},
    )
    assert payload["preSignTreeSha256"] != payload["runtime"]["treeSha256"]
    assert payload["launch"]["version"] == "ultra-fast-wbpp 0.1.0"
    assert payload["launch"]["versionSeconds"] < 5.0
    assert payload["launch"]["handshakeSeconds"] < 5.0
    assert payload["launch"]["catalogListSeconds"] < 5.0
    assert payload["launch"]["runProjectHelpSeconds"] < 5.0

    output = tmp_path / "attestation.json"
    write_create_only(output, payload)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_create_only(output, payload)
    assert output.read_bytes() == original


def test_post_sign_attestation_rejects_logical_tree_drift(tmp_path: Path) -> None:
    resource = _resource_root(tmp_path)
    runtime = resource / runtime_directory_name(TARGET)
    (runtime / "injected.txt").write_text("not in pre-sign tree", encoding="utf-8")
    with pytest.raises(BundleAttestationError, match="logical layout"):
        attest_runtime(
            resource,
            TARGET,
            max_start_seconds=5.0,
            bundle_name="Ultra-Fast WBPP.app",
            signature={"verified": True, "mode": "ad-hoc", "hardenedRuntime": True},
        )


def test_legal_resource_attestation_requires_exact_root_bytes(tmp_path: Path) -> None:
    canonical = tmp_path / "source"
    resources = tmp_path / "Ultra-Fast WBPP.app" / "Contents" / "Resources"
    legal = resources / "legal"
    canonical.mkdir()
    legal.mkdir(parents=True)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "docs/licensing.md",
                 "LICENSES/GPL-3.0.txt", "LICENSES/astroalign-MIT.txt"):
        content = f"canonical {name}\n".encode("utf-8")
        (canonical / name).parent.mkdir(parents=True, exist_ok=True)
        (legal / name).parent.mkdir(parents=True, exist_ok=True)
        (canonical / name).write_bytes(content)
        (legal / name).write_bytes(content)

    payload = attest_legal_resources(resources, canonical)
    assert payload["verified"] is True
    assert [entry["bundlePath"] for entry in payload["files"]] == [
        "legal/LICENSE",
        "legal/NOTICE",
        "legal/THIRD_PARTY_NOTICES.md",
        "legal/docs/licensing.md",
        "legal/LICENSES/GPL-3.0.txt",
        "legal/LICENSES/astroalign-MIT.txt",
    ]

    (legal / "NOTICE").write_text("drifted\n", encoding="utf-8")
    with pytest.raises(BundleAttestationError, match="differs from the canonical"):
        attest_legal_resources(resources, canonical)

    (legal / "NOTICE").write_bytes((canonical / "NOTICE").read_bytes())
    (legal / "LICENSES/GPL-3.0.txt").unlink()
    with pytest.raises(BundleAttestationError, match="bundled legal resource is missing"):
        attest_legal_resources(resources, canonical)


def test_macos_deployment_attestation_scans_every_macho_and_resolves_loads(
    tmp_path: Path,
) -> None:
    app = _test_app(tmp_path)

    payload = attest_macos_deployment(app)

    assert payload["verified"] is True
    assert payload["policy"]["maximumMachOMinimumVersion"] == "14.0.0"
    assert payload["declaredMinimumSystemVersion"] == "14.0.0"
    assert payload["maximumObservedMachOMinimumVersion"] == "14.0.0"
    assert payload["machOFileCount"] == 2
    assert payload["sliceCount"] == 2
    assert payload["dependencyClosure"] == {
        "bundledEdgeCount": 1,
        "systemEdgeCount": 1,
        "unresolvedEdgeCount": 0,
    }
    executable = next(
        item for item in payload["files"] if item["bundlePath"].startswith("Contents/MacOS/")
    )
    dependency = next(
        item
        for item in executable["slices"][0]["dependencies"]
        if item["installName"] == "@rpath/libworker.dylib"
    )
    assert dependency["resolution"] == "bundled"
    assert dependency["bundleTargets"] == ["Contents/Frameworks/libworker.dylib"]
    assert all(not str(value).startswith(str(tmp_path)) for value in executable.values())


def test_macos_deployment_attestation_rejects_any_slice_newer_than_14(
    tmp_path: Path,
) -> None:
    app = _test_app(tmp_path)
    offender = app / "Contents" / "Frameworks" / "libtoo-new.dylib"
    offender.write_bytes(_thin_macho_bytes(minimum=(26, 0, 0)))

    audit = inspect_macos_bundle(app)
    assert audit["maximumObservedMachOMinimumVersion"] == "26.0.0"
    assert audit["violations"] == [
        {
            "code": "DEPLOYMENT_TARGET_TOO_NEW",
            "bundlePath": "Contents/Frameworks/libtoo-new.dylib",
            "architecture": "arm64",
            "minimumVersion": "26.0.0",
        }
    ]
    with pytest.raises(BundleAttestationError, match="DEPLOYMENT_TARGET_TOO_NEW"):
        attest_macos_deployment(app)


def test_macos_deployment_attestation_checks_every_universal_binary_slice(
    tmp_path: Path,
) -> None:
    app = _test_app(tmp_path)
    universal = app / "Contents" / "Frameworks" / "libuniversal.dylib"
    universal.write_bytes(
        _fat_macho_bytes(
            (
                (0x0100000C, _thin_macho_bytes(minimum=(14, 0, 0))),
                (
                    0x01000007,
                    _thin_macho_bytes(
                        minimum=(15, 0, 0), cpu_type=0x01000007
                    ),
                ),
            )
        )
    )

    audit = inspect_macos_bundle(app)
    assert audit["machOFileCount"] == 3
    assert audit["sliceCount"] == 4
    assert any(
        item["code"] == "DEPLOYMENT_TARGET_TOO_NEW"
        and item["architecture"] == "x86_64"
        for item in audit["violations"]
    )


def test_macos_deployment_attestation_rejects_missing_slice_metadata(
    tmp_path: Path,
) -> None:
    app = _test_app(tmp_path)
    (app / "Contents" / "Frameworks" / "libunknown.dylib").write_bytes(
        _thin_macho_bytes(minimum=None)
    )

    with pytest.raises(BundleAttestationError, match="exactly one macOS deployment command"):
        attest_macos_deployment(app)


def test_macos_deployment_attestation_rejects_external_or_unresolved_loads(
    tmp_path: Path,
) -> None:
    app = _test_app(tmp_path)
    plugin = app / "Contents" / "Frameworks" / "plugin.so"
    plugin.write_bytes(
        _thin_macho_bytes(
            file_type=8,
            dependencies=("/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib",),
        )
    )

    audit = inspect_macos_bundle(app)
    assert audit["dependencyClosure"]["unresolvedEdgeCount"] == 1
    assert audit["violations"][0]["resolution"] == "external-absolute"
    with pytest.raises(BundleAttestationError, match="UNRESOLVED_DYLIB_DEPENDENCY"):
        attest_macos_deployment(app)


def test_macos_deployment_attestation_rejects_misleading_info_plist(
    tmp_path: Path,
) -> None:
    app = _test_app(tmp_path)
    info_path = app / "Contents" / "Info.plist"
    with info_path.open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleExecutable": "Ultra-Fast WBPP",
                "LSMinimumSystemVersion": "11.0",
            },
            stream,
        )

    with pytest.raises(BundleAttestationError, match="DECLARED_MINIMUM_MISMATCH"):
        attest_macos_deployment(app)


def test_runtime_library_attestation_rejects_tampered_package_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "packaging"
        / "worker"
        / "macos14-runtime-libraries-v1.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    runtime = tmp_path / "runtime"
    internal = runtime / "_internal"
    internal.mkdir(parents=True)
    policy_libraries = {
        library["destinationName"]: (package, library)
        for package in policy["packages"]
        for library in package["libraries"]
    }
    library_receipts = []
    for destination_name, (package, library) in policy_libraries.items():
        payload = f"synthetic-{destination_name}".encode()
        (internal / destination_name).write_bytes(payload)
        library_receipts.append(
            {
                "destinationName": destination_name,
                "formula": package["formula"],
                "version": package["version"],
                "license": package["license"],
                "sourceSha256": library["sourceSha256"],
                "relocatedSha256": hashlib.sha256(payload).hexdigest(),
                "minimumDeploymentTarget": "14.0.0",
            }
        )
    package_receipts = [
        {
            "formula": package["formula"],
            "version": package["version"],
            "license": package["license"],
            "bottleTag": package["bottleTag"],
            "sourceRevision": package["sourceRevision"],
            "ociManifestSha256": package["oci"]["manifestSha256"],
            "ociBlobSha256": package["oci"]["blobSha256"],
        }
        for package in policy["packages"]
    ]
    provenance = {
        "schemaVersion": 1,
        "kind": "ultra-fast-wbpp-macos-runtime-library-provenance",
        "targetTriple": "aarch64-apple-darwin",
        "policySha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "packages": list(reversed(package_receipts)),
        "libraries": list(reversed(library_receipts)),
    }
    (internal / "macos-runtime-library-provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    entry_point = runtime / "worker"
    entry_point.write_bytes(b"synthetic worker entry point")
    smoke_calls = []

    def runtime_smoke(command, *, timeout_seconds):
        # This test checks synthetic macOS provenance on every host. A POSIX
        # shell fixture is not a Windows executable and is not an ABI test.
        assert command == [str(entry_point)]
        assert timeout_seconds == 5.0
        smoke_calls.append(command)
        return {"libmpdec": "4.0.1", "openssl": "OpenSSL 3.6.4 test",
                "python": "3.12.3", "schemaVersion": 1}

    monkeypatch.setattr(
        "scripts.attest_bundled_runtime.run_runtime_library_smoke", runtime_smoke
    )

    attested, smoke, elapsed = _attest_runtime_libraries(
        runtime, entry_point, 5.0
    )
    assert attested["packages"] == list(reversed(package_receipts))
    assert smoke["openssl"] == "OpenSSL 3.6.4 test"
    assert elapsed < 5.0
    assert len(smoke_calls) == 1

    provenance["packages"][0]["sourceRevision"] = "0" * 40
    (internal / "macos-runtime-library-provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )

    with pytest.raises(BundleAttestationError, match="differs from pinned policy"):
        _attest_runtime_libraries(runtime, entry_point, 5.0)
    assert len(smoke_calls) == 1  # Reject altered provenance before launching.
