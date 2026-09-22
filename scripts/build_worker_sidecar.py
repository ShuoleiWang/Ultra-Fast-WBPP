#!/usr/bin/env python3
"""Build and attest the target-suffixed Ultra-Fast WBPP Python worker sidecar."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_PACKAGING = REPO_ROOT / "packaging" / "worker"
_POLICY_MODULE_NAME = "_openastroflow_worker_resource_policy"
_policy_spec = importlib.util.spec_from_file_location(
    _POLICY_MODULE_NAME, WORKER_PACKAGING / "resource_policy.py"
)
if _policy_spec is None or _policy_spec.loader is None:
    raise RuntimeError("cannot load the worker resource policy")
_policy = importlib.util.module_from_spec(_policy_spec)
sys.modules[_POLICY_MODULE_NAME] = _policy
_policy_spec.loader.exec_module(_policy)
DISTRIBUTION_NAMES = _policy.DISTRIBUTION_NAMES
CHECKED_CATALOG_RESOURCES = _policy.CHECKED_CATALOG_RESOURCES
REQUIRED_COLLECTIONS = _policy.REQUIRED_COLLECTIONS
REQUIRED_METADATA_DISTRIBUTIONS = _policy.REQUIRED_METADATA_DISTRIBUTIONS
ResourceBoundaryError = _policy.ResourceBoundaryError
validate_resource_members = _policy.validate_resource_members


SCHEMA_VERSION = 2
PROTOCOL_VERSION = 1
SIDECAR_PREFIX = "openastroflow-worker"
SUPPORTED_TARGETS = frozenset(
    {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "aarch64-pc-windows-msvc",
        "x86_64-pc-windows-msvc",
        "aarch64-unknown-linux-gnu",
        "x86_64-unknown-linux-gnu",
    }
)
WINDOWS_TARGETS = frozenset(
    {"aarch64-pc-windows-msvc", "x86_64-pc-windows-msvc"}
)
MACOS14_RUNTIME_POLICY = WORKER_PACKAGING / "macos14-runtime-libraries-v1.json"
HANDSHAKE_REQUEST = {
    "protocolVersion": PROTOCOL_VERSION,
    "sessionId": "packaging-smoke",
    "sequence": 0,
    "sentAtUnixMs": 0,
    "type": "handshake",
    "payload": {
        "role": "controller",
        "implementation": "packaging-preflight",
        "implementationVersion": "1",
        "supportedProtocolVersions": [PROTOCOL_VERSION],
    },
}


class SidecarBuildError(RuntimeError):
    """A release sidecar could not pass a fail-closed build gate."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def normalize_target_triple(value: str) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_TARGETS:
        supported = ", ".join(sorted(SUPPORTED_TARGETS))
        raise ValueError(f"unsupported Tauri target triple {value!r}; choose one of: {supported}")
    return value


def detect_host_target_triple(
    *, system: str | None = None, machine: str | None = None
) -> str:
    """Map the running Python interpreter to its native Tauri target triple."""

    system_name = (system or platform.system()).casefold()
    machine_name = (machine or platform.machine()).casefold()
    architecture = {
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
    }.get(machine_name)
    if architecture is None:
        raise SidecarBuildError(
            "HOST_UNSUPPORTED", f"unsupported Python architecture: {machine_name or 'unknown'}"
        )
    if system_name == "darwin":
        return f"{architecture}-apple-darwin"
    if system_name == "windows":
        return f"{architecture}-pc-windows-msvc"
    if system_name == "linux":
        return f"{architecture}-unknown-linux-gnu"
    raise SidecarBuildError(
        "HOST_UNSUPPORTED", f"unsupported build operating system: {system_name or 'unknown'}"
    )


def sidecar_stem(target_triple: str) -> str:
    return f"{SIDECAR_PREFIX}-{normalize_target_triple(target_triple)}"


def sidecar_filename(target_triple: str) -> str:
    stem = sidecar_stem(target_triple)
    return f"{stem}.exe" if target_triple in WINDOWS_TARGETS else stem


def runtime_directory_name(target_triple: str) -> str:
    """Return the immutable directory name containing one frozen runtime."""

    return sidecar_stem(target_triple)


def manifest_filename(target_triple: str) -> str:
    return f"{sidecar_stem(target_triple)}.manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SidecarBuildError("RUNTIME_LIBRARY_POLICY_INVALID", f"{field} is not SHA-256")
    return value


def _validated_positive_size(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", f"{field} is not a positive byte size"
        )
    return value


def _validated_relative_policy_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", f"{field} is not a safe relative path"
        )
    portable = PurePosixPath(value)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", f"{field} is not a safe relative path"
        )
    return portable.as_posix()


def _source_file(root: Path, relative_value: object, field: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", f"{field} is not a relative path"
        )
    portable = PurePosixPath(relative_value)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", f"{field} is not a safe relative path"
        )
    candidate = root.joinpath(*portable.parts)
    try:
        root_resolved = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
        metadata_value = candidate.lstat()
    except (OSError, ValueError) as error:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SOURCE_INVALID", f"{field} is missing or outside its source root"
        ) from error
    if stat.S_ISLNK(metadata_value.st_mode) or not stat.S_ISREG(metadata_value.st_mode):
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SOURCE_INVALID", f"{field} is not a regular non-symlink file"
        )
    return resolved


def load_macos14_runtime_policy(
    policy_path: Path = MACOS14_RUNTIME_POLICY,
) -> dict[str, Any]:
    if policy_path.is_symlink() or not policy_path.is_file():
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", "macOS runtime-library policy is missing or unsafe"
        )
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", f"cannot read macOS runtime-library policy: {error}"
        ) from error
    expected_top = {
        "schemaVersion",
        "kind",
        "targetTriple",
        "maximumDeploymentTarget",
        "ociSource",
        "runtimeSmoke",
        "packages",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", "macOS runtime-library policy fields changed"
        )
    if (
        payload["schemaVersion"] != 1
        or payload["kind"] != "ultra-fast-wbpp-macos-runtime-library-overlay"
        or payload["targetTriple"] != "aarch64-apple-darwin"
        or payload["maximumDeploymentTarget"] != "14.0.0"
    ):
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", "macOS runtime-library policy identity changed"
        )
    source = payload["ociSource"]
    expected_source = {
        "registry": "ghcr.io",
        "apiBaseUrl": "https://ghcr.io/v2",
        "tokenUrl": "https://ghcr.io/token",
        "tokenService": "ghcr.io",
        "maximumRedirects": 3,
        "allowedRedirectHosts": ["pkg-containers.githubusercontent.com"],
        "allowedRedirectHostSuffixes": [".blob.core.windows.net"],
    }
    if source != expected_source:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID",
            "OCI source or redirect boundary differs from the reviewed GHCR policy",
        )
    smoke = payload["runtimeSmoke"]
    if not isinstance(smoke, dict) or set(smoke) != {"opensslPrefix", "libmpdecVersion"}:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", "runtime smoke policy is malformed"
        )
    for field in ("opensslPrefix", "libmpdecVersion"):
        if not isinstance(smoke[field], str) or not smoke[field]:
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_POLICY_INVALID", f"runtimeSmoke.{field} is malformed"
            )
    packages = payload["packages"]
    if not isinstance(packages, list) or not packages:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", "runtime-library packages are missing"
        )
    destination_names: set[str] = set()
    package_formulas: set[str] = set()
    expected_repositories = {
        "openssl@3": "homebrew/core/openssl/3",
        "mpdecimal": "homebrew/core/mpdecimal",
    }
    expected_licenses = {"openssl@3": "Apache-2.0", "mpdecimal": "BSD-2-Clause"}
    expected_layout_directories = {"openssl@3": "openssl", "mpdecimal": "mpdecimal"}
    for package_index, package in enumerate(packages):
        package_fields = {
            "formula",
            "version",
            "license",
            "bottleTag",
            "sourceRevision",
            "oci",
            "libraries",
        }
        if not isinstance(package, dict) or set(package) != package_fields:
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_POLICY_INVALID",
                f"packages[{package_index}] fields changed",
            )
        for field in ("formula", "version", "license", "bottleTag", "sourceRevision"):
            value = package[field]
            if not isinstance(value, str) or not value or len(value) > 128:
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_POLICY_INVALID",
                    f"packages[{package_index}].{field} is malformed",
                )
        formula = package["formula"]
        if formula in package_formulas or formula not in expected_repositories:
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_POLICY_INVALID", "runtime-library formula set changed"
            )
        package_formulas.add(formula)
        if (
            package["bottleTag"] != "arm64_sonoma"
            or package["license"] != expected_licenses[formula]
            or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", package["version"])
            or not re.fullmatch(r"[0-9a-f]{40}", package["sourceRevision"])
        ):
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_POLICY_INVALID", "runtime-library package identity is malformed"
            )
        oci = package["oci"]
        oci_fields = {
            "repository",
            "manifestDigest",
            "manifestMediaType",
            "configMediaType",
            "layerMediaType",
            "manifestRelativePath",
            "manifestSha256",
            "manifestSizeBytes",
            "configRelativePath",
            "configSha256",
            "configSizeBytes",
            "blobRelativePath",
            "blobSha256",
            "blobSizeBytes",
        }
        if not isinstance(oci, dict) or set(oci) != oci_fields:
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_POLICY_INVALID",
                f"packages[{package_index}].oci fields changed",
            )
        if (
            oci["repository"] != expected_repositories[formula]
            or oci["manifestMediaType"]
            != "application/vnd.oci.image.manifest.v1+json"
            or oci["configMediaType"] != "application/vnd.oci.image.config.v1+json"
            or oci["layerMediaType"]
            != "application/vnd.oci.image.layer.v1.tar+gzip"
        ):
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_POLICY_INVALID", "OCI repository or media types changed"
            )
        manifest_sha = _validated_digest(
            oci["manifestSha256"],
            f"packages[{package_index}].oci.manifestSha256",
        )
        if oci["manifestDigest"] != f"sha256:{manifest_sha}":
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_POLICY_INVALID", "OCI manifest digest is inconsistent"
            )
        layout_directory = expected_layout_directories[formula]
        for prefix in ("manifest", "config", "blob"):
            relative = oci[f"{prefix}RelativePath"]
            relative = _validated_relative_policy_path(
                relative, f"packages[{package_index}].oci.{prefix}RelativePath"
            )
            _validated_digest(
                oci[f"{prefix}Sha256"],
                f"packages[{package_index}].oci.{prefix}Sha256",
            )
            _validated_positive_size(
                oci[f"{prefix}SizeBytes"],
                f"packages[{package_index}].oci.{prefix}SizeBytes",
            )
            digest = oci[f"{prefix}Sha256"]
            expected_relative = (
                f"{layout_directory}/manifest.json"
                if prefix == "manifest"
                else f"{layout_directory}/{digest}"
            )
            if relative != expected_relative:
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_POLICY_INVALID", "OCI local layout changed"
                )
        libraries = package["libraries"]
        if not isinstance(libraries, list) or not libraries:
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_POLICY_INVALID",
                f"packages[{package_index}].libraries are missing",
            )
        for library_index, library in enumerate(libraries):
            library_fields = {
                "archiveMemberPath",
                "sourceRelativePath",
                "sourceSha256",
                "sourceSizeBytes",
                "destinationName",
                "sourceInstallName",
                "relocatedInstallName",
                "changes",
                "rpaths",
            }
            if not isinstance(library, dict) or set(library) != library_fields:
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_POLICY_INVALID",
                    f"packages[{package_index}].libraries[{library_index}] fields changed",
                )
            _validated_digest(
                library["sourceSha256"],
                f"packages[{package_index}].libraries[{library_index}].sourceSha256",
            )
            _validated_positive_size(
                library["sourceSizeBytes"],
                f"packages[{package_index}].libraries[{library_index}].sourceSizeBytes",
            )
            destination = library["destinationName"]
            if (
                not isinstance(destination, str)
                or not destination
                or "/" in destination
                or "\\" in destination
                or destination in destination_names
            ):
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_POLICY_INVALID", "runtime-library destinations are unsafe"
                )
            destination_names.add(destination)
            for field in (
                "archiveMemberPath",
                "sourceRelativePath",
                "sourceInstallName",
                "relocatedInstallName",
            ):
                if not isinstance(library[field], str) or not library[field]:
                    raise SidecarBuildError(
                        "RUNTIME_LIBRARY_POLICY_INVALID",
                        "packages"
                        f"[{package_index}].libraries[{library_index}].{field} is malformed",
                    )
            archive_member = _validated_relative_policy_path(
                library["archiveMemberPath"],
                f"packages[{package_index}].libraries[{library_index}].archiveMemberPath",
            )
            source_relative = _validated_relative_policy_path(
                library["sourceRelativePath"],
                f"packages[{package_index}].libraries[{library_index}].sourceRelativePath",
            )
            if source_relative != f"extracted/{archive_member}":
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_POLICY_INVALID",
                    "runtime-library archive and extracted paths differ",
                )
            if not isinstance(library["changes"], list) or not all(
                isinstance(item, dict)
                and set(item) == {"from", "to"}
                and all(isinstance(value, str) and value for value in item.values())
                for item in library["changes"]
            ):
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_POLICY_INVALID", "runtime-library changes are malformed"
                )
            if not isinstance(library["rpaths"], list) or not all(
                isinstance(item, str) and item.startswith("@")
                for item in library["rpaths"]
            ):
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_POLICY_INVALID", "runtime-library rpaths are malformed"
                )
    if package_formulas != set(expected_repositories) or destination_names != {
        "libcrypto.3.dylib",
        "libssl.3.dylib",
        "libmpdec.4.dylib",
    }:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_POLICY_INVALID", "runtime-library policy must bind the exact ABI set"
        )
    return payload


def _run_packaging_tool(command: Sequence[str], code: str) -> str:
    completed = subprocess.run(
        list(command), text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()[-2000:] or completed.stdout.strip()[-2000:]
        raise SidecarBuildError(code, diagnostic or "packaging tool failed without output")
    return completed.stdout


def _verify_pinned_file(
    root: Path,
    relative_path: object,
    expected_sha256: object,
    expected_size: object,
    field: str,
) -> Path:
    path = _source_file(root, relative_path, field)
    digest = _validated_digest(expected_sha256, f"{field}.sha256")
    size = _validated_positive_size(expected_size, f"{field}.sizeBytes")
    if path.stat().st_size != size or _sha256(path) != digest:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SOURCE_INVALID", f"{field} bytes do not match pinned provenance"
        )
    return path


def _verify_oci_package(root: Path, package: Mapping[str, Any]) -> None:
    oci = package["oci"]
    manifest_path = _verify_pinned_file(
        root,
        oci["manifestRelativePath"],
        oci["manifestSha256"],
        oci["manifestSizeBytes"],
        f"{package['formula']}.manifest",
    )
    _verify_pinned_file(
        root,
        oci["configRelativePath"],
        oci["configSha256"],
        oci["configSizeBytes"],
        f"{package['formula']}.config",
    )
    _verify_pinned_file(
        root,
        oci["blobRelativePath"],
        oci["blobSha256"],
        oci["blobSizeBytes"],
        f"{package['formula']}.blob",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        annotations = manifest["annotations"]
        layers = manifest["layers"]
        config = manifest["config"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SOURCE_INVALID", "pinned OCI bottle manifest is malformed"
        ) from error
    if (
        manifest.get("schemaVersion") != 2
        or not isinstance(layers, list)
        or len(layers) != 1
        or layers[0].get("digest") != f"sha256:{oci['blobSha256']}"
        or layers[0].get("size") != oci["blobSizeBytes"]
        or config.get("digest") != f"sha256:{oci['configSha256']}"
        or config.get("size") != oci["configSizeBytes"]
        or annotations.get("org.opencontainers.image.version") != package["version"]
        or annotations.get("org.opencontainers.image.ref.name")
        != f"{package['version']}.{package['bottleTag']}"
        or annotations.get("org.opencontainers.image.revision") != package["sourceRevision"]
        or annotations.get("org.opencontainers.image.licenses") != package["license"]
        or annotations.get("sh.brew.bottle.digest") != oci["blobSha256"]
    ):
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SOURCE_INVALID", "OCI bottle identity differs from pinned policy"
        )


def _macho_minimum_versions(path: Path) -> tuple[str, ...]:
    output = _run_packaging_tool(
        ["xcrun", "vtool", "-show-build", str(path)],
        "RUNTIME_LIBRARY_MACHO_INVALID",
    )
    values = tuple(
        line.split()[-1]
        for line in output.splitlines()
        if line.strip().startswith("minos ")
    )
    if not values:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_MACHO_INVALID", "runtime library has no deployment target"
        )
    return values


def apply_macos14_runtime_libraries(
    runtime: Path,
    bottle_root: Path,
    *,
    policy_path: Path = MACOS14_RUNTIME_POLICY,
) -> dict[str, Any]:
    """Replace three ABI-identical Tahoe dylibs with pinned Sonoma bottles.

    The source OCI manifests, compressed blobs, extracted bytes, versions, and
    licenses are pinned before any mutation. Only ordinary install-name/rpath
    relocation and ad-hoc signing are performed; deployment load commands are
    never patched.
    """

    if platform.system() != "Darwin":
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_OVERLAY_UNSUPPORTED", "macOS runtime overlay is Darwin-only"
        )
    if bottle_root.is_symlink() or not bottle_root.is_dir():
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SOURCE_INVALID", "bottle root must be a real directory"
        )
    policy = load_macos14_runtime_policy(policy_path)
    internal = runtime / "_internal"
    if internal.is_symlink() or not internal.is_dir():
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_DESTINATION_INVALID", "PyInstaller _internal directory is missing"
        )
    source_records: list[tuple[Mapping[str, Any], Mapping[str, Any], Path]] = []
    for package in policy["packages"]:
        _verify_oci_package(bottle_root, package)
        for library in package["libraries"]:
            source = _verify_pinned_file(
                bottle_root,
                library["sourceRelativePath"],
                library["sourceSha256"],
                library["sourceSizeBytes"],
                f"{package['formula']}.{library['destinationName']}",
            )
            architectures = _run_packaging_tool(
                ["lipo", "-archs", str(source)], "RUNTIME_LIBRARY_MACHO_INVALID"
            ).split()
            if architectures != ["arm64"] or _macho_minimum_versions(source) != (
                "14.0",
            ):
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_MACHO_INVALID",
                    f"{library['destinationName']} is not an arm64 macOS 14 bottle",
                )
            install_names = [
                line.strip()
                for line in _run_packaging_tool(
                    ["otool", "-D", str(source)], "RUNTIME_LIBRARY_MACHO_INVALID"
                ).splitlines()[1:]
                if line.strip()
            ]
            if install_names != [library["sourceInstallName"]]:
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_MACHO_INVALID",
                    f"{library['destinationName']} install name differs from policy",
                )
            source_records.append((package, library, source))

    library_receipts: list[dict[str, Any]] = []
    for package, library, source in source_records:
        destination = internal / library["destinationName"]
        if destination.is_symlink() or not destination.is_file():
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_DESTINATION_INVALID",
                f"PyInstaller did not collect {library['destinationName']}",
            )
        shutil.copy2(source, destination)
        destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
        _run_packaging_tool(
            [
                "install_name_tool",
                "-id",
                library["relocatedInstallName"],
                str(destination),
            ],
            "RUNTIME_LIBRARY_RELOCATION_FAILED",
        )
        for change in library["changes"]:
            _run_packaging_tool(
                [
                    "install_name_tool",
                    "-change",
                    change["from"],
                    change["to"],
                    str(destination),
                ],
                "RUNTIME_LIBRARY_RELOCATION_FAILED",
            )
        for rpath in library["rpaths"]:
            _run_packaging_tool(
                ["install_name_tool", "-add_rpath", rpath, str(destination)],
                "RUNTIME_LIBRARY_RELOCATION_FAILED",
            )
        _run_packaging_tool(
            ["codesign", "--force", "--sign", "-", "--timestamp=none", str(destination)],
            "RUNTIME_LIBRARY_SIGNING_FAILED",
        )
        _run_packaging_tool(
            ["codesign", "--verify", "--strict", str(destination)],
            "RUNTIME_LIBRARY_SIGNING_FAILED",
        )
        if _macho_minimum_versions(destination) != ("14.0",):
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_MACHO_INVALID",
                "normal relocation unexpectedly changed the deployment target",
            )
        linkage = _run_packaging_tool(
            ["otool", "-L", str(destination)], "RUNTIME_LIBRARY_MACHO_INVALID"
        )
        if "@@HOMEBREW" in linkage or "/opt/homebrew" in linkage:
            raise SidecarBuildError(
                "RUNTIME_LIBRARY_RELOCATION_FAILED",
                f"{library['destinationName']} retains a build-machine load path",
            )
        library_receipts.append(
            {
                "destinationName": library["destinationName"],
                "formula": package["formula"],
                "version": package["version"],
                "license": package["license"],
                "sourceSha256": library["sourceSha256"],
                "relocatedSha256": _sha256(destination),
                "minimumDeploymentTarget": "14.0.0",
            }
        )

    provenance = {
        "schemaVersion": 1,
        "kind": "ultra-fast-wbpp-macos-runtime-library-provenance",
        "targetTriple": policy["targetTriple"],
        "policySha256": _sha256(policy_path),
        "packages": [
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
        ],
        "libraries": sorted(
            library_receipts, key=lambda item: item["destinationName"]
        ),
    }
    provenance_path = internal / "macos-runtime-library-provenance.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(provenance_path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(provenance, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return provenance


def run_runtime_library_smoke(
    command: Sequence[str], *, timeout_seconds: float = 30.0
) -> dict[str, Any]:
    if not command or timeout_seconds <= 0:
        raise ValueError("runtime library smoke requires a command and positive timeout")
    try:
        completed = subprocess.run(
            [*command, "__packaging-runtime-smoke-v1"],
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SidecarBuildError("RUNTIME_LIBRARY_SMOKE_FAILED", str(error)) from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()[-1000:] or "no stderr"
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SMOKE_FAILED",
            f"frozen runtime library smoke exited with {completed.returncode}: {diagnostic}",
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SMOKE_FAILED", "runtime library smoke did not emit JSON"
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schemaVersion", "python", "openssl", "libmpdec"}
        or payload["schemaVersion"] != 1
        or not all(
            isinstance(payload[field], str) and payload[field]
            for field in ("python", "openssl", "libmpdec")
        )
    ):
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_SMOKE_FAILED", "runtime library smoke payload is malformed"
        )
    return payload


def _tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    rendered = json.dumps(
        list(entries),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _normalized_link_target(link_path: str, target: str) -> str:
    if not isinstance(target, str) or not target or "\x00" in target:
        raise ResourceBoundaryError("runtime symlink target is malformed")
    portable = target.replace("\\", "/")
    if PurePosixPath(portable).is_absolute() or PureWindowsPath(target).is_absolute():
        raise ResourceBoundaryError("absolute runtime symlink target is forbidden")
    stack = list(PurePosixPath(link_path).parent.parts)
    for part in PurePosixPath(portable).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not stack:
                raise ResourceBoundaryError("runtime symlink escapes its tree")
            stack.pop()
        else:
            stack.append(part)
    if not stack:
        raise ResourceBoundaryError("runtime symlink cannot target the tree root")
    return PurePosixPath(*stack).as_posix()


def runtime_tree_entries(root: Path) -> tuple[dict[str, Any], ...]:
    """Inventory every file, directory, and safe relative link in a runtime.

    Paths are logical POSIX paths. No local build path is serialized. Symlinks
    are permitted only when their lexical and resolved targets remain inside
    the runtime, which preserves Python framework layouts without allowing a
    staged tree to reach outside the signed application bundle.
    """

    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise ResourceBoundaryError(f"cannot inspect runtime root: {error}") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ResourceBoundaryError("runtime root must be a real directory")
    resolved_root = root.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names.sort()
        file_names.sort()
        for name in list(directory_names):
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            validate_resource_members([relative])
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                directory_names.remove(name)
                target = os.readlink(candidate)
                _normalized_link_target(relative, target)
                resolved = candidate.resolve(strict=True)
                try:
                    resolved.relative_to(resolved_root)
                except ValueError as error:
                    raise ResourceBoundaryError("runtime symlink resolves outside its tree") from error
                entries.append({"path": relative, "type": "symlink", "target": target})
            elif stat.S_ISDIR(metadata.st_mode):
                entries.append({"path": relative, "type": "directory"})
            else:
                raise ResourceBoundaryError("runtime contains a non-directory tree entry")
        for name in file_names:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            validate_resource_members([relative])
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(candidate)
                _normalized_link_target(relative, target)
                resolved = candidate.resolve(strict=True)
                try:
                    resolved.relative_to(resolved_root)
                except ValueError as error:
                    raise ResourceBoundaryError("runtime symlink resolves outside its tree") from error
                entries.append({"path": relative, "type": "symlink", "target": target})
            elif stat.S_ISREG(metadata.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "sha256": _sha256(candidate),
                        "sizeBytes": metadata.st_size,
                        "executable": bool(metadata.st_mode & 0o111),
                    }
                )
            else:
                raise ResourceBoundaryError("runtime contains a special filesystem entry")
    entries.sort(key=lambda entry: entry["path"])
    if not entries:
        raise ResourceBoundaryError("runtime tree is empty")
    return tuple(entries)


def _safe_scalar(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ResourceBoundaryError(f"{field} must be a short non-empty string")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ResourceBoundaryError(f"{field} contains a control character")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ResourceBoundaryError(f"{field} must not contain an absolute path")
    if "/" in value or "\\" in value:
        raise ResourceBoundaryError(f"{field} must not contain path separators")
    portable = value.replace("\\", "/")
    if ".." in PurePosixPath(portable).parts:
        raise ResourceBoundaryError(f"{field} must not contain parent traversal")
    return value


def collect_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for import_name in REQUIRED_COLLECTIONS:
        distribution_name = DISTRIBUTION_NAMES[import_name]
        try:
            packages[distribution_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError as error:
            raise SidecarBuildError(
                "DEPENDENCY_MISSING",
                f"required worker distribution is not installed: {distribution_name}",
            ) from error
    for distribution_name in REQUIRED_METADATA_DISTRIBUTIONS:
        try:
            packages[distribution_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError as error:
            raise SidecarBuildError(
                "DEPENDENCY_MISSING",
                f"required worker distribution is not installed: {distribution_name}",
            ) from error
    try:
        pyinstaller_version = metadata.version("pyinstaller")
    except metadata.PackageNotFoundError as error:
        raise SidecarBuildError(
            "PYINSTALLER_MISSING",
            "PyInstaller is not installed; install packaging/worker/requirements-build.txt "
            "inside the project build environment",
        ) from error
    return {
        "python": platform.python_version(),
        "pyinstaller": pyinstaller_version,
        "packages": dict(sorted(packages.items())),
    }


def validate_handshake(response: object) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker handshake was not a JSON object")
    if set(response) != {
        "protocolVersion",
        "sessionId",
        "sequence",
        "sentAtUnixMs",
        "type",
        "payload",
    }:
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker returned a non-canonical envelope")
    if response.get("protocolVersion") != PROTOCOL_VERSION:
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker returned the wrong protocol version")
    if response.get("sessionId") != HANDSHAKE_REQUEST["sessionId"]:
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker returned the wrong session id")
    if response.get("sequence") != 0 or response.get("type") != "handshake":
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker stream did not start with handshake sequence 0")
    sent_at = response.get("sentAtUnixMs")
    if isinstance(sent_at, bool) or not isinstance(sent_at, int) or sent_at < 0:
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker sentAtUnixMs is invalid")
    payload = response.get("payload")
    if not isinstance(payload, dict) or payload.get("role") != "worker":
        raise SidecarBuildError("HANDSHAKE_INVALID", "handshake payload is not a worker advertisement")
    if payload.get("supportedProtocolVersions") != [PROTOCOL_VERSION]:
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker did not advertise exact protocol v1 support")
    _safe_scalar(payload.get("implementation"), "handshake.implementation")
    _safe_scalar(payload.get("implementationVersion"), "handshake.implementationVersion")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker capabilities are missing")
    for field in ("backendId", "backendVersion", "workerBuild"):
        _safe_scalar(capabilities.get(field), f"handshake.capabilities.{field}")
    for field in ("hardwareProfiles", "stages", "features"):
        values = capabilities.get(field)
        if not isinstance(values, list) or not values or not all(
            isinstance(item, str) and item for item in values
        ):
            raise SidecarBuildError(
                "HANDSHAKE_INVALID", f"worker capability {field} is malformed"
            )
    if capabilities.get("schemaVersion") != 1:
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker capability schema is not v1")
    return response


def run_worker_handshake(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Run an actual worker process and require exactly one protocol response."""

    if not command:
        raise ValueError("worker command must not be empty")
    try:
        completed = subprocess.run(
            list(command),
            input=json.dumps(HANDSHAKE_REQUEST, separators=(",", ":")) + "\n",
            text=True,
            capture_output=True,
            cwd=cwd,
            env=None if env is None else dict(env),
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SidecarBuildError("HANDSHAKE_PROCESS_FAILED", str(error)) from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()[-1000:] or "no stderr"
        raise SidecarBuildError(
            "HANDSHAKE_PROCESS_FAILED",
            f"worker exited with {completed.returncode}: {diagnostic}",
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SidecarBuildError(
            "HANDSHAKE_INVALID",
            f"worker emitted {len(lines)} non-empty stdout lines; expected exactly one",
        )
    try:
        response = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise SidecarBuildError("HANDSHAKE_INVALID", "worker stdout was not JSON") from error
    return validate_handshake(response)


def expected_catalog_ids() -> tuple[str, ...]:
    catalog_root = REPO_ROOT / "resources" / "catalogs"
    identifiers: list[str] = []
    for name in CHECKED_CATALOG_RESOURCES:
        if name.endswith(".schema.json"):
            continue
        try:
            payload = json.loads((catalog_root / name).read_text(encoding="utf-8"))
            catalog_id = payload["catalogId"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise SidecarBuildError(
                "CATALOG_RESOURCE_INVALID", f"checked catalog resource {name} is invalid"
            ) from error
        _safe_scalar(catalog_id, f"catalogResource[{name}].catalogId")
        identifiers.append(catalog_id)
    if len(identifiers) != len(set(identifiers)) or not identifiers:
        raise SidecarBuildError(
            "CATALOG_RESOURCE_INVALID", "checked catalog IDs are empty or duplicated"
        )
    return tuple(sorted(identifiers))


def run_catalog_list_smoke(
    command: Sequence[str], *, timeout_seconds: float = 180.0
) -> dict[str, Any]:
    if not command:
        raise ValueError("worker command must not be empty")
    with tempfile.TemporaryDirectory(prefix="openastroflow-empty-catalog-") as catalog_root:
        try:
            completed = subprocess.run(
                [
                    *command,
                    "catalog",
                    "list",
                    "--json",
                    "--catalog-dir",
                    catalog_root,
                ],
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SidecarBuildError("CATALOG_SMOKE_FAILED", str(error)) from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()[-1000:] or "no stderr"
        raise SidecarBuildError(
            "CATALOG_SMOKE_FAILED",
            f"catalog list exited with {completed.returncode}: {diagnostic}",
        )
    try:
        payload = json.loads(completed.stdout)
        catalogs = payload["catalogs"]
        identifiers = tuple(sorted(item["catalogId"] for item in catalogs))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SidecarBuildError(
            "CATALOG_SMOKE_FAILED", "catalog list emitted invalid JSON"
        ) from error
    if payload.get("schemaVersion") != 1 or identifiers != expected_catalog_ids():
        raise SidecarBuildError(
            "CATALOG_SMOKE_FAILED", "frozen catalog list omitted or changed checked manifests"
        )
    return payload


def source_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    roots = [
        REPO_ROOT / "packages" / "openastroflow-engine" / "src",
        REPO_ROOT / "packages" / "light-frame-qc" / "src",
        REPO_ROOT / "packages" / "openastroflow-registration" / "src",
    ]
    existing = environment.get("PYTHONPATH")
    values = [str(path) for path in roots]
    if existing:
        values.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(values)
    return environment


def run_native_kernel_smoke(
    command: Sequence[str], *, timeout_seconds: float = 60.0
) -> dict[str, Any]:
    """Prove the frozen worker loads the bundled native kernel library.

    Runs ``<worker> doctor --json`` and returns its ``nativeKernels`` facts
    (library path, SHA-256, ISA features).  A worker that would silently run
    the NumPy fallback is a packaging error, not a runtime condition.
    """

    if not command or timeout_seconds <= 0:
        raise ValueError("native kernel smoke requires a command and positive timeout")
    try:
        completed = subprocess.run(
            [*command, "doctor", "--json"],
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SidecarBuildError("NATIVE_KERNEL_SMOKE_FAILED", str(error)) from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()[-1000:] or "no stderr"
        raise SidecarBuildError(
            "NATIVE_KERNEL_SMOKE_FAILED",
            f"frozen doctor exited with {completed.returncode}: {diagnostic}",
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SidecarBuildError(
            "NATIVE_KERNEL_SMOKE_FAILED", "frozen doctor did not emit JSON"
        ) from error
    facts = payload.get("nativeKernels") if isinstance(payload, dict) else None
    if not isinstance(facts, dict) or "loaded" not in facts:
        raise SidecarBuildError(
            "NATIVE_KERNEL_SMOKE_FAILED", "frozen doctor omitted the nativeKernels facts"
        )
    if facts["loaded"] is not True:
        raise SidecarBuildError(
            "NATIVE_KERNELS_MISSING",
            "the frozen worker did not load the native kernel library: "
            + str(facts.get("reason", "unknown reason")),
        )
    return facts


def run_source_handshake() -> dict[str, Any]:
    return run_worker_handshake(
        [sys.executable, str(WORKER_PACKAGING / "launcher.py")],
        env=source_environment(),
    )


def run_public_tree_check() -> None:
    checker = REPO_ROOT / "scripts" / "check_public_tree.py"
    try:
        completed = subprocess.run(
            [sys.executable, str(checker), str(REPO_ROOT)],
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SidecarBuildError("PUBLIC_TREE_CHECK_FAILED", str(error)) from error
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SidecarBuildError(
            "PUBLIC_TREE_CHECK_FAILED", "public-tree checker did not emit JSON"
        ) from error
    if completed.returncode != 0 or payload.get("ok") is not True:
        count = payload.get("findingCount", "unknown")
        raise SidecarBuildError(
            "PUBLIC_TREE_CHECK_FAILED", f"public-tree audit reported {count} finding(s)"
        )


def ensure_outputs_absent(output_dir: Path, target_triple: str) -> tuple[Path, Path]:
    artifact = output_dir / runtime_directory_name(target_triple)
    manifest = output_dir / manifest_filename(target_triple)
    collisions = [path.name for path in (artifact, manifest) if path.exists() or path.is_symlink()]
    if collisions:
        raise FileExistsError(
            "refusing to replace existing sidecar target: " + ", ".join(collisions)
        )
    return artifact, manifest


def build_manifest(
    artifact: Path,
    target_triple: str,
    *,
    versions: Mapping[str, Any],
    handshake: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a path-free deterministic attestation for one runtime tree."""

    target = normalize_target_triple(target_triple)
    runtime = build_runtime_record(artifact, target)
    validated_handshake = validate_handshake(dict(handshake))
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "openastroflow-worker-sidecar",
        "targetTriple": target,
        "runtime": runtime,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "engineVersion": validated_handshake["payload"]["implementationVersion"],
            "capabilities": {
                key: validated_handshake["payload"]["capabilities"][key]
                for key in ("backendId", "hardwareProfiles", "stages", "features")
            },
        },
        "versions": {
            "python": versions.get("python"),
            "pyinstaller": versions.get("pyinstaller"),
            "packages": dict(sorted(dict(versions.get("packages", {})).items())),
        },
        "collections": list(REQUIRED_COLLECTIONS),
    }
    validate_manifest(payload)
    return payload


def build_runtime_record(artifact: Path, target_triple: str) -> dict[str, Any]:
    """Build the identity record shared by pre-sign and bundled attestations."""

    target = normalize_target_triple(target_triple)
    if artifact.name != runtime_directory_name(target):
        raise ResourceBoundaryError("runtime directory name does not match its target triple")
    entries = runtime_tree_entries(artifact)
    entry_point = sidecar_filename(target)
    entry_record = next(
        (entry for entry in entries if entry["path"] == entry_point), None
    )
    if entry_record is None or entry_record["type"] != "file":
        raise ResourceBoundaryError("runtime entry point is missing")
    if target not in WINDOWS_TARGETS and entry_record["executable"] is not True:
        raise ResourceBoundaryError("runtime entry point is not executable")
    files = [entry for entry in entries if entry["type"] == "file"]
    return {
        "directoryName": artifact.name,
        "entryPoint": entry_point,
        "treeSha256": _tree_digest(entries),
        "sizeBytes": sum(entry["sizeBytes"] for entry in files),
        "fileCount": len(files),
        "entryCount": len(entries),
        "entries": list(entries),
    }


def validate_manifest(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ResourceBoundaryError("sidecar manifest must be an object")
    expected_top = {
        "schemaVersion",
        "kind",
        "targetTriple",
        "runtime",
        "protocol",
        "versions",
        "collections",
    }
    if set(payload) != expected_top:
        raise ResourceBoundaryError("sidecar manifest has unexpected or missing fields")
    if payload["schemaVersion"] != SCHEMA_VERSION:
        raise ResourceBoundaryError("unsupported sidecar manifest schema")
    if payload["kind"] != "openastroflow-worker-sidecar":
        raise ResourceBoundaryError("wrong sidecar manifest kind")
    target = normalize_target_triple(payload["targetTriple"])

    runtime = payload["runtime"]
    expected_runtime = {
        "directoryName",
        "entryPoint",
        "treeSha256",
        "sizeBytes",
        "fileCount",
        "entryCount",
        "entries",
    }
    if not isinstance(runtime, dict) or set(runtime) != expected_runtime:
        raise ResourceBoundaryError("invalid runtime manifest fields")
    if runtime["directoryName"] != runtime_directory_name(target):
        raise ResourceBoundaryError("manifest runtime directory does not match target")
    if runtime["entryPoint"] != sidecar_filename(target):
        raise ResourceBoundaryError("manifest runtime entry point does not match target")
    validate_resource_members([runtime["directoryName"], runtime["entryPoint"]])
    entries = runtime["entries"]
    if not isinstance(entries, list) or not entries:
        raise ResourceBoundaryError("runtime entries must be a non-empty array")
    if entries != sorted(entries, key=lambda entry: entry.get("path", "") if isinstance(entry, dict) else ""):
        raise ResourceBoundaryError("runtime entries must be sorted")
    seen_paths: set[str] = set()
    file_count = 0
    total_size = 0
    entry_point_found = False
    for entry in entries:
        if not isinstance(entry, dict):
            raise ResourceBoundaryError("runtime entry must be an object")
        entry_type = entry.get("type")
        path = entry.get("path")
        if not isinstance(path, str):
            raise ResourceBoundaryError("runtime entry path is malformed")
        validate_resource_members([path])
        if path in seen_paths:
            raise ResourceBoundaryError("runtime entry paths must be unique")
        seen_paths.add(path)
        if entry_type == "directory":
            if set(entry) != {"path", "type"}:
                raise ResourceBoundaryError("runtime directory entry is malformed")
        elif entry_type == "symlink":
            if set(entry) != {"path", "type", "target"}:
                raise ResourceBoundaryError("runtime symlink entry is malformed")
            target_value = entry["target"]
            if not isinstance(target_value, str):
                raise ResourceBoundaryError("runtime symlink target is malformed")
            _normalized_link_target(path, target_value)
        elif entry_type == "file":
            if set(entry) != {"path", "type", "sha256", "sizeBytes", "executable"}:
                raise ResourceBoundaryError("runtime file entry is malformed")
            digest = entry["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ResourceBoundaryError("runtime file SHA-256 is malformed")
            size = entry["sizeBytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ResourceBoundaryError("runtime file size is malformed")
            if not isinstance(entry["executable"], bool):
                raise ResourceBoundaryError("runtime executable flag is malformed")
            file_count += 1
            total_size += size
            entry_point_found |= path == runtime["entryPoint"]
        else:
            raise ResourceBoundaryError("runtime entry type is unsupported")
    if not entry_point_found:
        raise ResourceBoundaryError("runtime entry point is not a file")
    for field, expected in (
        ("entryCount", len(entries)),
        ("fileCount", file_count),
        ("sizeBytes", total_size),
    ):
        value = runtime[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ResourceBoundaryError(f"runtime {field} is inconsistent")
    tree_digest = runtime["treeSha256"]
    if (
        not isinstance(tree_digest, str)
        or len(tree_digest) != 64
        or any(character not in "0123456789abcdef" for character in tree_digest)
        or tree_digest != _tree_digest(entries)
    ):
        raise ResourceBoundaryError("runtime tree SHA-256 is inconsistent")

    protocol = payload["protocol"]
    if not isinstance(protocol, dict) or set(protocol) != {
        "version",
        "engineVersion",
        "capabilities",
    }:
        raise ResourceBoundaryError("invalid protocol manifest fields")
    if protocol["version"] != PROTOCOL_VERSION:
        raise ResourceBoundaryError("manifest protocol version mismatch")
    _safe_scalar(protocol["engineVersion"], "protocol.engineVersion")
    capabilities = protocol["capabilities"]
    if not isinstance(capabilities, dict) or set(capabilities) != {
        "backendId",
        "hardwareProfiles",
        "stages",
        "features",
    }:
        raise ResourceBoundaryError("invalid protocol capability fields")
    _safe_scalar(capabilities["backendId"], "protocol.capabilities.backendId")
    for field in ("hardwareProfiles", "stages", "features"):
        values = capabilities[field]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ResourceBoundaryError(f"protocol capability {field} is malformed")
        for index, value in enumerate(values):
            _safe_scalar(value, f"protocol.capabilities.{field}[{index}]")

    versions = payload["versions"]
    if not isinstance(versions, dict) or set(versions) != {
        "python",
        "pyinstaller",
        "packages",
    }:
        raise ResourceBoundaryError("invalid versions manifest fields")
    _safe_scalar(versions["python"], "versions.python")
    _safe_scalar(versions["pyinstaller"], "versions.pyinstaller")
    packages = versions["packages"]
    expected_distributions = {
        *(DISTRIBUTION_NAMES[name] for name in REQUIRED_COLLECTIONS),
        *REQUIRED_METADATA_DISTRIBUTIONS,
    }
    if not isinstance(packages, dict) or set(packages) != expected_distributions:
        raise ResourceBoundaryError("manifest package versions are incomplete")
    if list(packages) != sorted(packages):
        raise ResourceBoundaryError("manifest package versions must be sorted")
    for name, version in packages.items():
        _safe_scalar(name, "versions.packages.name")
        _safe_scalar(version, f"versions.packages[{name}]")

    collections = payload["collections"]
    if collections != list(REQUIRED_COLLECTIONS):
        raise ResourceBoundaryError("manifest resource collections do not match policy")
    for index, collection in enumerate(collections):
        _safe_scalar(collection, f"collections[{index}]")
    return payload


def write_manifest_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    validate_manifest(dict(payload))
    rendered = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def verify_runtime_tree(root: Path, manifest: Mapping[str, Any]) -> None:
    """Require the on-disk runtime to equal every manifest tree record."""

    validated = validate_manifest(dict(manifest))
    entries = runtime_tree_entries(root)
    if list(entries) != validated["runtime"]["entries"]:
        raise SidecarBuildError(
            "RUNTIME_IDENTITY_MISMATCH", "runtime tree differs from its manifest"
        )


def _publish_runtime_create_only(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ResourceBoundaryError("staged runtime must be a real directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace existing runtime: {destination.name}"
        ) from error


def _run_pyinstaller(stage: Path, target_triple: str) -> Path:
    if importlib.util.find_spec("PyInstaller") is None:
        raise SidecarBuildError(
            "PYINSTALLER_MISSING",
            "PyInstaller is not installed; install packaging/worker/requirements-build.txt "
            "inside the project build environment",
        )
    dist = stage / "dist"
    work = stage / "work"
    dist.mkdir()
    work.mkdir()
    environment = source_environment()
    environment["OAF_WORKER_BASENAME"] = sidecar_stem(target_triple)
    environment["PYTHONHASHSEED"] = "0"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--log-level",
        "WARN",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        str(WORKER_PACKAGING / "openastroflow_worker.spec"),
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()[-4000:] or completed.stdout.strip()[-4000:]
        raise SidecarBuildError(
            "PYINSTALLER_FAILED",
            f"PyInstaller exited with {completed.returncode}: {diagnostic}",
        )
    expected = dist / runtime_directory_name(target_triple)
    if expected.is_symlink() or not expected.is_dir():
        raise SidecarBuildError(
            "ARTIFACT_MISSING", f"PyInstaller did not produce {expected.name}"
        )
    unexpected = [path.name for path in dist.iterdir() if path != expected]
    if unexpected:
        raise SidecarBuildError(
            "ARTIFACT_LAYOUT_INVALID",
            "one-directory build emitted unexpected top-level entries: "
            + ", ".join(sorted(unexpected)),
        )
    entry_point = expected / sidecar_filename(target_triple)
    if entry_point.is_symlink() or not entry_point.is_file():
        raise SidecarBuildError("ARTIFACT_INVALID", "PyInstaller omitted the runtime entry point")
    if target_triple not in WINDOWS_TARGETS and not os.access(entry_point, os.X_OK):
        raise SidecarBuildError("ARTIFACT_INVALID", "sidecar is not executable")
    runtime_tree_entries(expected)
    return expected


def build_sidecar(
    target_triple: str,
    output_dir: Path,
    *,
    macos14_bottle_root: Path | None = None,
    require_native_kernels: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    target = normalize_target_triple(target_triple)
    _require_native_target(target)
    if macos14_bottle_root is not None and target != "aarch64-apple-darwin":
        raise SidecarBuildError(
            "RUNTIME_LIBRARY_OVERLAY_UNSUPPORTED",
            "the pinned Sonoma runtime-library overlay is Apple Silicon-only",
        )

    output = output_dir.expanduser()
    artifact_destination, manifest_destination = ensure_outputs_absent(output, target)

    # These gates intentionally precede dependency collection and PyInstaller.
    source_handshake = run_source_handshake()
    run_public_tree_check()
    versions = collect_versions()
    source_version = source_handshake["payload"]["implementationVersion"]
    if source_version != versions["packages"]["openastroflow-engine"]:
        raise SidecarBuildError(
            "VERSION_MISMATCH", "worker handshake and installed engine distribution disagree"
        )

    output.mkdir(parents=True, exist_ok=True)
    artifact_destination, manifest_destination = ensure_outputs_absent(output, target)
    with tempfile.TemporaryDirectory(prefix=".worker-build-", dir=output) as temporary_name:
        staged = _run_pyinstaller(Path(temporary_name), target)
        overlay = None
        if macos14_bottle_root is not None:
            overlay = apply_macos14_runtime_libraries(staged, macos14_bottle_root)
        frozen_command = [str(staged / sidecar_filename(target))]
        runtime_library_smoke = run_runtime_library_smoke(frozen_command)
        if overlay is not None:
            policy = load_macos14_runtime_policy()
            if (
                not runtime_library_smoke["openssl"].startswith(
                    policy["runtimeSmoke"]["opensslPrefix"]
                )
                or runtime_library_smoke["libmpdec"]
                != policy["runtimeSmoke"]["libmpdecVersion"]
            ):
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_ABI_MISMATCH",
                    "frozen Python did not load the pinned OpenSSL/mpdecimal ABI",
                )
        frozen_handshake = run_worker_handshake(frozen_command)
        if frozen_handshake["payload"]["implementationVersion"] != source_version:
            raise SidecarBuildError(
                "VERSION_MISMATCH", "source and frozen worker handshakes disagree"
            )
        run_catalog_list_smoke(frozen_command)
        if require_native_kernels:
            native_facts = run_native_kernel_smoke(frozen_command)
            print(
                "frozen worker loaded native kernels "
                + str(native_facts.get("sha256", "")),
                file=sys.stderr,
            )
        manifest = build_manifest(
            staged,
            target,
            versions=versions,
            handshake=frozen_handshake,
        )
        artifact_published = False
        try:
            _publish_runtime_create_only(staged, artifact_destination)
            artifact_published = True
            verify_runtime_tree(artifact_destination, manifest)
            write_manifest_create_only(manifest_destination, manifest)
        except Exception:
            if artifact_published:
                try:
                    if artifact_destination.is_dir() and not artifact_destination.is_symlink():
                        verify_runtime_tree(artifact_destination, manifest)
                        shutil.rmtree(artifact_destination)
                except OSError:
                    pass
            raise
    return artifact_destination, manifest_destination, manifest


def _require_native_target(target: str) -> None:
    host = detect_host_target_triple()
    if target != host:
        raise SidecarBuildError(
            "CROSS_COMPILE_UNSUPPORTED",
            f"PyInstaller must run natively: requested {target}, running interpreter is {host}",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=None,
        help="Tauri target triple; defaults to the running Python interpreter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "build" / "sidecars",
        help="create-only destination for the sidecar and manifest",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run source handshake and public-tree gates without requiring PyInstaller",
    )
    parser.add_argument(
        "--allow-missing-native-kernels",
        action="store_true",
        help=(
            "do not fail when the frozen worker cannot load the native kernel "
            "library (development builds only; releases must bundle it)"
        ),
    )
    parser.add_argument(
        "--macos14-bottle-root",
        type=Path,
        help=(
            "verified OCI/extracted Homebrew arm64_sonoma bottle root for the pinned "
            "OpenSSL/mpdecimal compatibility overlay"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        target = normalize_target_triple(arguments.target or detect_host_target_triple())
        if arguments.preflight_only:
            if arguments.macos14_bottle_root is not None:
                raise SidecarBuildError(
                    "RUNTIME_LIBRARY_OVERLAY_UNUSED",
                    "--macos14-bottle-root cannot be used with --preflight-only",
                )
            _require_native_target(target)
            handshake = run_source_handshake()
            run_public_tree_check()
            result: dict[str, Any] = {
                "ok": True,
                "mode": "preflight",
                "targetTriple": target,
                "protocolVersion": handshake["protocolVersion"],
                "engineVersion": handshake["payload"]["implementationVersion"],
            }
        else:
            artifact, manifest_path, manifest = build_sidecar(
                target,
                arguments.output_dir,
                macos14_bottle_root=arguments.macos14_bottle_root,
                require_native_kernels=not arguments.allow_missing_native_kernels,
            )
            result = {
                "ok": True,
                "mode": "build",
                "targetTriple": target,
                "runtime": artifact.name,
                "manifest": manifest_path.name,
                "treeSha256": manifest["runtime"]["treeSha256"],
                "sizeBytes": manifest["runtime"]["sizeBytes"],
                "fileCount": manifest["runtime"]["fileCount"],
            }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (FileExistsError, ResourceBoundaryError, SidecarBuildError, ValueError, OSError) as error:
        if isinstance(error, SidecarBuildError):
            code = error.code
        elif isinstance(error, FileExistsError):
            code = "OUTPUT_EXISTS"
        elif isinstance(error, ResourceBoundaryError):
            code = "RESOURCE_BOUNDARY"
        elif isinstance(error, ValueError):
            code = "TARGET_INVALID"
        else:
            code = "IO_ERROR"
        print(
            json.dumps(
                {"ok": False, "error": {"code": code, "message": str(error)}},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HANDSHAKE_REQUEST",
    "MACOS14_RUNTIME_POLICY",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "SIDECAR_PREFIX",
    "SUPPORTED_TARGETS",
    "SidecarBuildError",
    "apply_macos14_runtime_libraries",
    "build_manifest",
    "build_runtime_record",
    "build_sidecar",
    "collect_versions",
    "detect_host_target_triple",
    "ensure_outputs_absent",
    "expected_catalog_ids",
    "main",
    "load_macos14_runtime_policy",
    "manifest_filename",
    "normalize_target_triple",
    "run_public_tree_check",
    "run_runtime_library_smoke",
    "run_catalog_list_smoke",
    "run_source_handshake",
    "run_worker_handshake",
    "runtime_directory_name",
    "runtime_tree_entries",
    "sidecar_filename",
    "sidecar_stem",
    "validate_handshake",
    "validate_manifest",
    "verify_runtime_tree",
    "write_manifest_create_only",
]
