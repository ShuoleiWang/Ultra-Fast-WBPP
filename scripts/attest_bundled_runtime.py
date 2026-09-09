#!/usr/bin/env python3
"""Attest and time the actual runtime tree embedded in a release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

try:
    from scripts.build_worker_sidecar import (
        ResourceBoundaryError,
        SidecarBuildError,
        build_runtime_record,
        expected_catalog_ids,
        load_macos14_runtime_policy,
        manifest_filename,
        normalize_target_triple,
        run_runtime_library_smoke,
        run_worker_handshake,
        runtime_directory_name,
        validate_manifest,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from build_worker_sidecar import (
        ResourceBoundaryError,
        SidecarBuildError,
        build_runtime_record,
        expected_catalog_ids,
        load_macos14_runtime_policy,
        manifest_filename,
        normalize_target_triple,
        run_runtime_library_smoke,
        run_worker_handshake,
        runtime_directory_name,
        validate_manifest,
    )


class BundleAttestationError(RuntimeError):
    """The installed runtime failed identity, signing, or launch gates."""


REPOSITORY = Path(__file__).resolve().parents[1]
LEGAL_RESOURCE_NAMES = (
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "docs/licensing.md",
    "LICENSES/GPL-3.0.txt", "LICENSES/astroalign-MIT.txt",
)
MACOS_MAXIMUM_DEPLOYMENT_TARGET = (14, 0, 0)

_THIN_MACHO_MAGICS = {
    b"\xce\xfa\xed\xfe": ("<", False),
    b"\xcf\xfa\xed\xfe": ("<", True),
    b"\xfe\xed\xfa\xce": (">", False),
    b"\xfe\xed\xfa\xcf": (">", True),
}
_FAT_MACHO_MAGICS = {
    b"\xca\xfe\xba\xbe": (">", False),
    b"\xca\xfe\xba\xbf": (">", True),
    b"\xbe\xba\xfe\xca": ("<", False),
    b"\xbf\xba\xfe\xca": ("<", True),
}
_MACHO_MAGICS = frozenset((*_THIN_MACHO_MAGICS, *_FAT_MACHO_MAGICS))

_LC_VERSION_MIN_MACOSX = 0x24
_LC_BUILD_VERSION = 0x32
_LC_RPATH = 0x8000001C
_LC_ID_DYLIB = 0x0D
_DYLIB_LOAD_COMMANDS = {
    0x0C: "load",
    0x18 | 0x80000000: "weak",
    0x1F | 0x80000000: "reexport",
    0x20: "lazy",
    0x23 | 0x80000000: "upward",
}
_CPU_NAMES = {
    7: "x86",
    12: "arm",
    0x01000007: "x86_64",
    0x0100000C: "arm64",
}
_FILE_TYPE_NAMES = {
    0x1: "object",
    0x2: "executable",
    0x3: "fixed-vm-library",
    0x4: "core",
    0x5: "preload",
    0x6: "dylib",
    0x7: "dylinker",
    0x8: "bundle",
    0x9: "dylib-stub",
    0xA: "dsym",
    0xB: "kext-bundle",
    0xC: "fileset",
}


def _version_tuple(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 0xFFFF, (value >> 8) & 0xFF, value & 0xFF)


def _version_text(value: tuple[int, int, int]) -> str:
    return ".".join(str(component) for component in value)


def _parse_version_text(value: object, field: str) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise BundleAttestationError(f"{field} must be a version string")
    components = value.split(".")
    if not 1 <= len(components) <= 3 or not all(item.isdecimal() for item in components):
        raise BundleAttestationError(f"{field} is not a canonical numeric version")
    numbers = tuple(int(item) for item in components)
    if numbers[0] > 0xFFFF or any(item > 0xFF for item in numbers[1:]):
        raise BundleAttestationError(f"{field} is outside the Mach-O version range")
    return (numbers + (0, 0, 0))[:3]


def _load_command_string(command: bytes, offset: int, field: str) -> str:
    if offset < 8 or offset >= len(command):
        raise BundleAttestationError(f"Mach-O {field} offset is outside its load command")
    terminator = command.find(b"\0", offset)
    if terminator < 0:
        raise BundleAttestationError(f"Mach-O {field} is not NUL terminated")
    try:
        value = command[offset:terminator].decode("utf-8")
    except UnicodeDecodeError as error:
        raise BundleAttestationError(f"Mach-O {field} is not UTF-8") from error
    if not value or any(character in value for character in ("\n", "\r")):
        raise BundleAttestationError(f"Mach-O {field} is empty or contains controls")
    return value


def _parse_thin_macho(
    data: bytes, start: int, size: int, *, declared_cpu_type: int | None = None
) -> dict[str, Any]:
    if start < 0 or size < 32 or start + size > len(data):
        raise BundleAttestationError("Mach-O slice extends outside its file")
    magic = data[start : start + 4]
    try:
        endian, is_64_bit = _THIN_MACHO_MAGICS[magic]
    except KeyError as error:
        raise BundleAttestationError(
            "fat Mach-O entry does not contain a thin Mach-O slice"
        ) from error
    header_size = 32 if is_64_bit else 28
    if size < header_size:
        raise BundleAttestationError("Mach-O slice header is truncated")
    cpu_type = struct.unpack_from(f"{endian}i", data, start + 4)[0]
    cpu_subtype = struct.unpack_from(f"{endian}i", data, start + 8)[0]
    file_type, command_count, command_bytes = struct.unpack_from(
        f"{endian}III", data, start + 12
    )
    if declared_cpu_type is not None and cpu_type != declared_cpu_type:
        raise BundleAttestationError("fat Mach-O architecture table disagrees with its slice")
    if command_count > 100_000:
        raise BundleAttestationError("Mach-O load-command count is unreasonable")
    commands_start = start + header_size
    commands_end = commands_start + command_bytes
    if commands_end > start + size:
        raise BundleAttestationError("Mach-O load-command area is truncated")

    deployment_commands: list[dict[str, Any]] = []
    dependencies: list[dict[str, str]] = []
    rpaths: list[str] = []
    install_name: str | None = None
    cursor = commands_start
    for _ in range(command_count):
        if cursor + 8 > commands_end:
            raise BundleAttestationError("Mach-O load-command header is truncated")
        command_id, command_size = struct.unpack_from(f"{endian}II", data, cursor)
        if command_size < 8 or command_size % 4 or cursor + command_size > commands_end:
            raise BundleAttestationError("Mach-O load-command size is invalid")
        command = data[cursor : cursor + command_size]
        if command_id == _LC_BUILD_VERSION:
            if command_size < 24:
                raise BundleAttestationError("LC_BUILD_VERSION is truncated")
            platform_id, minimum, sdk = struct.unpack_from(f"{endian}III", command, 8)
            deployment_commands.append(
                {
                    "command": "LC_BUILD_VERSION",
                    "platformId": platform_id,
                    "minimumVersion": _version_tuple(minimum),
                    "sdkVersion": _version_tuple(sdk),
                }
            )
        elif command_id == _LC_VERSION_MIN_MACOSX:
            if command_size < 16:
                raise BundleAttestationError("LC_VERSION_MIN_MACOSX is truncated")
            minimum, sdk = struct.unpack_from(f"{endian}II", command, 8)
            deployment_commands.append(
                {
                    "command": "LC_VERSION_MIN_MACOSX",
                    "platformId": 1,
                    "minimumVersion": _version_tuple(minimum),
                    "sdkVersion": _version_tuple(sdk),
                }
            )
        elif command_id in _DYLIB_LOAD_COMMANDS or command_id == _LC_ID_DYLIB:
            if command_size < 24:
                raise BundleAttestationError("Mach-O dylib command is truncated")
            name_offset = struct.unpack_from(f"{endian}I", command, 8)[0]
            name = _load_command_string(command, name_offset, "dylib name")
            if command_id == _LC_ID_DYLIB:
                if install_name is not None:
                    raise BundleAttestationError("Mach-O contains more than one LC_ID_DYLIB")
                install_name = name
            else:
                dependencies.append(
                    {"kind": _DYLIB_LOAD_COMMANDS[command_id], "installName": name}
                )
        elif command_id == _LC_RPATH:
            if command_size < 12:
                raise BundleAttestationError("LC_RPATH is truncated")
            path_offset = struct.unpack_from(f"{endian}I", command, 8)[0]
            rpaths.append(_load_command_string(command, path_offset, "rpath"))
        cursor += command_size
    if cursor != commands_end:
        raise BundleAttestationError("Mach-O load-command sizes do not match sizeofcmds")
    if len(deployment_commands) != 1:
        raise BundleAttestationError(
            "every Mach-O slice must contain exactly one macOS deployment command"
        )
    deployment = deployment_commands[0]
    if deployment["platformId"] != 1:
        raise BundleAttestationError(
            "bundled Mach-O slice targets a non-macOS Apple platform"
        )
    minimum_version = deployment["minimumVersion"]
    sdk_version = deployment["sdkVersion"]
    return {
        "architecture": _CPU_NAMES.get(cpu_type, f"cpu-{cpu_type & 0xFFFFFFFF}"),
        "cpuType": cpu_type,
        "cpuSubtype": cpu_subtype,
        "bits": 64 if is_64_bit else 32,
        "fileType": _FILE_TYPE_NAMES.get(file_type, f"type-{file_type}"),
        "deploymentCommand": deployment["command"],
        "minimumVersion": _version_text(minimum_version),
        "minimumVersionTuple": minimum_version,
        "sdkVersion": _version_text(sdk_version),
        "installName": install_name,
        "rpaths": sorted(set(rpaths)),
        "dependencies": sorted(
            dependencies, key=lambda item: (item["installName"], item["kind"])
        ),
    }


def _parse_macho(path: Path) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise BundleAttestationError(f"cannot read bundled Mach-O: {error}") from error
    magic = data[:4]
    if magic in _THIN_MACHO_MAGICS:
        return [_parse_thin_macho(data, 0, len(data))]
    try:
        endian, is_64_bit = _FAT_MACHO_MAGICS[magic]
    except KeyError as error:
        raise BundleAttestationError("file selected as Mach-O has an unknown magic") from error
    if len(data) < 8:
        raise BundleAttestationError("fat Mach-O header is truncated")
    architecture_count = struct.unpack_from(f"{endian}I", data, 4)[0]
    if not 1 <= architecture_count <= 64:
        raise BundleAttestationError("fat Mach-O architecture count is invalid")
    entry_size = 32 if is_64_bit else 20
    table_end = 8 + architecture_count * entry_size
    if table_end > len(data):
        raise BundleAttestationError("fat Mach-O architecture table is truncated")
    slices: list[dict[str, Any]] = []
    ranges: list[tuple[int, int]] = []
    for index in range(architecture_count):
        entry = 8 + index * entry_size
        cpu_type = struct.unpack_from(f"{endian}i", data, entry)[0]
        if is_64_bit:
            offset, size = struct.unpack_from(f"{endian}QQ", data, entry + 8)
        else:
            offset, size = struct.unpack_from(f"{endian}II", data, entry + 8)
        if offset < table_end or size < 28 or offset + size > len(data):
            raise BundleAttestationError("fat Mach-O slice range is invalid")
        current_range = (offset, offset + size)
        if any(current_range[0] < end and start < current_range[1] for start, end in ranges):
            raise BundleAttestationError("fat Mach-O slices overlap")
        ranges.append(current_range)
        slices.append(
            _parse_thin_macho(
                data, offset, size, declared_cpu_type=cpu_type
            )
        )
    if len({item["cpuType"] for item in slices}) != len(slices):
        raise BundleAttestationError("fat Mach-O repeats an architecture slice")
    return sorted(slices, key=lambda item: (item["architecture"], item["cpuSubtype"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _macho_candidates(root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = sorted(
            name for name in directory_names if not (directory_path / name).is_symlink()
        )
        for name in sorted(file_names):
            path = directory_path / name
            try:
                metadata = path.lstat()
                if path.is_symlink() or not path.is_file() or metadata.st_size < 4:
                    continue
                with path.open("rb") as stream:
                    magic = stream.read(4)
            except OSError as error:
                raise BundleAttestationError(
                    f"cannot inspect application bundle member: {error}"
                ) from error
            if magic in _MACHO_MAGICS:
                candidates.append(path)
    return tuple(sorted(candidates))


def _system_install_name(value: str) -> bool:
    return value.startswith("/System/Library/") or value.startswith("/usr/lib/")


def _inside_root(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        return resolved
    except (OSError, ValueError):
        return None


def _expand_loader_token(
    value: str, *, loader_directory: Path, executable_directories: Sequence[Path]
) -> tuple[Path, ...]:
    if value == "@loader_path":
        return (loader_directory,)
    if value.startswith("@loader_path/"):
        return (loader_directory / value[len("@loader_path/") :],)
    if value == "@executable_path":
        return tuple(executable_directories)
    if value.startswith("@executable_path/"):
        suffix = value[len("@executable_path/") :]
        return tuple(directory / suffix for directory in executable_directories)
    if value.startswith("/"):
        return (Path(value),)
    return ()


def _resolve_dependency(
    install_name: str,
    *,
    loader: Path,
    rpaths: Sequence[str],
    executable_directories: Sequence[Path],
    resolved_root: Path,
    macho_by_resolved_path: Mapping[Path, Mapping[str, Any]],
    macho_by_basename: Mapping[str, Sequence[Mapping[str, Any]]],
    cpu_type: int,
) -> tuple[str, list[str]]:
    if _system_install_name(install_name):
        return "system", []

    candidates: list[Path] = []
    if install_name.startswith("@rpath/"):
        suffix = install_name[len("@rpath/") :]
        for rpath in rpaths:
            for expanded in _expand_loader_token(
                rpath,
                loader_directory=loader.parent,
                executable_directories=executable_directories,
            ):
                candidates.append(expanded / suffix)
    else:
        candidates.extend(
            _expand_loader_token(
                install_name,
                loader_directory=loader.parent,
                executable_directories=executable_directories,
            )
        )

    matched: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        resolved = _inside_root(candidate, resolved_root)
        if resolved is None:
            continue
        target = macho_by_resolved_path.get(resolved)
        if target is None:
            continue
        if not any(item["cpuType"] == cpu_type for item in target["slices"]):
            continue
        matched[target["bundlePath"]] = target
    if matched:
        return "bundled", sorted(matched)
    if install_name.startswith("@rpath/"):
        basename = Path(install_name[len("@rpath/") :]).name
        fallback = [
            target
            for target in macho_by_basename.get(basename, ())
            if any(item["cpuType"] == cpu_type for item in target["slices"])
        ]
        # A caller's LC_RPATH stack is assembled dynamically. When the loader
        # itself has no LC_RPATH, bind the install name to byte-identical
        # in-bundle candidates. Different candidate bytes are ambiguous and
        # therefore cannot satisfy a release closure.
        if fallback and len({target["sha256"] for target in fallback}) == 1:
            return "bundled-rpath-stack", sorted(
                target["bundlePath"] for target in fallback
            )
        if fallback:
            return "ambiguous-rpath-candidates", []
    if install_name.startswith("/"):
        return "external-absolute", []
    if install_name.startswith("@"):
        return "unresolved-token", []
    return "unresolved-relative", []


def inspect_macos_bundle(
    app: Path,
    *,
    maximum_deployment_target: tuple[int, int, int] = MACOS_MAXIMUM_DEPLOYMENT_TARGET,
) -> dict[str, Any]:
    """Inspect every Mach-O slice and its static dylib closure inside a .app.

    This parser reads Mach-O headers directly. It does not trust filename
    extensions, Info.plist, the host SDK, or the output of a best-effort launch.
    All paths in the returned record are relative to the application bundle.
    """

    if app.is_symlink() or not app.is_dir() or app.suffix != ".app":
        raise BundleAttestationError("macOS deployment audit requires a real .app directory")
    if len(maximum_deployment_target) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in maximum_deployment_target
    ):
        raise ValueError("maximum_deployment_target must be a three-part version")
    resolved_root = app.resolve(strict=True)
    info_path = app / "Contents" / "Info.plist"
    if info_path.is_symlink() or not info_path.is_file():
        raise BundleAttestationError("application Info.plist is missing or unsafe")
    try:
        with info_path.open("rb") as stream:
            info = plistlib.load(stream)
    except (OSError, plistlib.InvalidFileException) as error:
        raise BundleAttestationError(f"application Info.plist is invalid: {error}") from error
    if not isinstance(info, dict):
        raise BundleAttestationError("application Info.plist is not a dictionary")
    declared_minimum = _parse_version_text(
        info.get("LSMinimumSystemVersion"), "LSMinimumSystemVersion"
    )
    executable_name = info.get("CFBundleExecutable")
    if (
        not isinstance(executable_name, str)
        or not executable_name
        or "/" in executable_name
        or "\\" in executable_name
    ):
        raise BundleAttestationError("CFBundleExecutable is missing or unsafe")
    main_executable = app / "Contents" / "MacOS" / executable_name
    if main_executable.is_symlink() or not main_executable.is_file():
        raise BundleAttestationError("declared application executable is missing or unsafe")

    records: list[dict[str, Any]] = []
    for path in _macho_candidates(app):
        relative = path.relative_to(app).as_posix()
        try:
            slices = _parse_macho(path)
        except BundleAttestationError as error:
            raise BundleAttestationError(f"invalid bundled Mach-O {relative}: {error}") from error
        records.append(
            {
                "bundlePath": relative,
                "sha256": _sha256(path),
                "slices": slices,
                "_path": path,
            }
        )
    if not records:
        raise BundleAttestationError("application bundle contains no Mach-O files")
    main_relative = main_executable.relative_to(app).as_posix()
    main_record = next(
        (record for record in records if record["bundlePath"] == main_relative), None
    )
    if main_record is None or not any(
        item["fileType"] == "executable" for item in main_record["slices"]
    ):
        raise BundleAttestationError("CFBundleExecutable is not a Mach-O executable")

    executable_directories = tuple(
        sorted(
            {
                record["_path"].parent.resolve(strict=True)
                for record in records
                if any(item["fileType"] == "executable" for item in record["slices"])
            }
        )
    )
    macho_by_resolved_path = {
        record["_path"].resolve(strict=True): record for record in records
    }
    by_basename: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        names = {record["_path"].name}
        names.update(
            Path(slice_record["installName"]).name
            for slice_record in record["slices"]
            if slice_record["installName"]
        )
        for name in names:
            by_basename.setdefault(name, []).append(record)
    violations: list[dict[str, Any]] = []
    maximum_observed = (0, 0, 0)
    bundled_edge_count = 0
    system_edge_count = 0
    for record in records:
        path = record["_path"]
        for slice_record in record["slices"]:
            minimum = _parse_version_text(
                slice_record["minimumVersion"], "Mach-O minimumVersion"
            )
            maximum_observed = max(maximum_observed, minimum)
            if minimum > maximum_deployment_target:
                violations.append(
                    {
                        "code": "DEPLOYMENT_TARGET_TOO_NEW",
                        "bundlePath": record["bundlePath"],
                        "architecture": slice_record["architecture"],
                        "minimumVersion": slice_record["minimumVersion"],
                    }
                )
            resolved_dependencies: list[dict[str, Any]] = []
            for dependency in slice_record.pop("dependencies"):
                resolution, targets = _resolve_dependency(
                    dependency["installName"],
                    loader=path,
                    rpaths=slice_record["rpaths"],
                    executable_directories=executable_directories,
                    resolved_root=resolved_root,
                    macho_by_resolved_path=macho_by_resolved_path,
                    macho_by_basename=by_basename,
                    cpu_type=slice_record["cpuType"],
                )
                resolved_dependency: dict[str, Any] = {
                    **dependency,
                    "resolution": resolution,
                }
                if targets:
                    resolved_dependency["bundleTargets"] = targets
                resolved_dependencies.append(resolved_dependency)
                if resolution in {"bundled", "bundled-rpath-stack"}:
                    bundled_edge_count += 1
                elif resolution == "system":
                    system_edge_count += 1
                else:
                    violations.append(
                        {
                            "code": "UNRESOLVED_DYLIB_DEPENDENCY",
                            "bundlePath": record["bundlePath"],
                            "architecture": slice_record["architecture"],
                            "installName": dependency["installName"],
                            "resolution": resolution,
                        }
                    )
            slice_record["dependencies"] = resolved_dependencies
            slice_record.pop("minimumVersionTuple")
        record.pop("_path")
    if declared_minimum != maximum_deployment_target:
        violations.append(
            {
                "code": "DECLARED_MINIMUM_MISMATCH",
                "declaredMinimumVersion": _version_text(declared_minimum),
                "requiredMinimumVersion": _version_text(maximum_deployment_target),
            }
        )
    slice_count = sum(len(record["slices"]) for record in records)
    return {
        "policy": {
            "platform": "macOS",
            "requiredMinimumSystemVersion": _version_text(maximum_deployment_target),
            "maximumMachOMinimumVersion": _version_text(maximum_deployment_target),
        },
        "declaredMinimumSystemVersion": _version_text(declared_minimum),
        "maximumObservedMachOMinimumVersion": _version_text(maximum_observed),
        "machOFileCount": len(records),
        "sliceCount": slice_count,
        "dependencyClosure": {
            "bundledEdgeCount": bundled_edge_count,
            "systemEdgeCount": system_edge_count,
            "unresolvedEdgeCount": sum(
                violation["code"] == "UNRESOLVED_DYLIB_DEPENDENCY"
                for violation in violations
            ),
        },
        "files": records,
        "violations": violations,
    }


def attest_macos_deployment(app: Path) -> dict[str, Any]:
    """Fail closed unless the complete signed bundle can launch on macOS 14."""

    audit = inspect_macos_bundle(app)
    violations = audit.pop("violations")
    if violations:
        examples = "; ".join(
            f"{item['code']}:{item.get('bundlePath', 'Info.plist')}"
            for item in violations[:8]
        )
        raise BundleAttestationError(
            f"macOS 14 deployment audit found {len(violations)} violation(s): {examples}"
        )
    audit["verified"] = True
    return audit


def attest_legal_resources(
    application_resources: Path, canonical_root: Path = REPOSITORY
) -> dict[str, Any]:
    """Prove that required bundle notices are byte-identical to repository roots."""

    legal_root = application_resources / "legal"
    if legal_root.is_symlink() or not legal_root.is_dir():
        raise BundleAttestationError("bundle legal resource directory is missing or unsafe")
    entries: list[dict[str, Any]] = []
    for name in LEGAL_RESOURCE_NAMES:
        source = canonical_root / name
        bundled = legal_root / name
        if source.is_symlink() or not source.is_file():
            raise BundleAttestationError(f"canonical legal resource is missing or unsafe: {name}")
        if bundled.is_symlink() or not bundled.is_file():
            raise BundleAttestationError(f"bundled legal resource is missing or unsafe: {name}")
        source_digest = _sha256(source)
        bundled_digest = _sha256(bundled)
        if source_digest != bundled_digest or source.stat().st_size != bundled.stat().st_size:
            raise BundleAttestationError(
                f"bundled legal resource differs from the canonical root file: {name}"
            )
        entries.append(
            {
                "bundlePath": f"legal/{name}",
                "sha256": bundled_digest,
                "sizeBytes": bundled.stat().st_size,
            }
        )
    return {"verified": True, "files": entries}


def _load_pre_sign_manifest(root: Path, target: str) -> Mapping[str, Any]:
    path = root / manifest_filename(target)
    if path.is_symlink() or not path.is_file():
        raise BundleAttestationError("pre-sign runtime manifest is missing from bundle resources")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return validate_manifest(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BundleAttestationError(f"pre-sign runtime manifest is invalid: {error}") from error


def _logical_entry(entry: Mapping[str, Any]) -> tuple[object, ...]:
    entry_type = entry.get("type")
    common: tuple[object, ...] = (entry.get("path"), entry_type)
    if entry_type == "file":
        return common + (entry.get("executable"),)
    if entry_type == "symlink":
        return common + (entry.get("target"),)
    return common


def _require_same_logical_layout(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    left = [_logical_entry(entry) for entry in before["entries"]]
    right = [_logical_entry(entry) for entry in after["entries"]]
    if left != right:
        raise BundleAttestationError(
            "bundle signing or resource staging changed the runtime logical layout"
        )


def _run_version(entry_point: Path, timeout_seconds: float) -> tuple[str, float]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [str(entry_point), "--version"],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BundleAttestationError(f"bundled --version launch failed: {error}") from error
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        diagnostic = result.stderr.strip()[-1000:] or "no stderr"
        raise BundleAttestationError(
            f"bundled --version exited with {result.returncode}: {diagnostic}"
        )
    version = result.stdout.strip()
    if not version.startswith("ultra-fast-wbpp ") or "\n" in version:
        raise BundleAttestationError("bundled --version output is non-canonical")
    return version, elapsed


def _run_handshake(entry_point: Path, timeout_seconds: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    try:
        handshake = run_worker_handshake(
            [str(entry_point)], timeout_seconds=timeout_seconds
        )
    except SidecarBuildError as error:
        raise BundleAttestationError(f"bundled handshake failed: {error}") from error
    return handshake, time.perf_counter() - started


def _run_catalog_list(entry_point: Path, timeout_seconds: float) -> tuple[tuple[str, ...], float]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="openastroflow-attest-catalog-") as catalog_root:
        try:
            result = subprocess.run(
                [
                    str(entry_point),
                    "catalog",
                    "list",
                    "--json",
                    "--catalog-dir",
                    catalog_root,
                ],
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BundleAttestationError(f"bundled catalog-list launch failed: {error}") from error
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        diagnostic = result.stderr.strip()[-1000:] or "no stderr"
        raise BundleAttestationError(
            f"bundled catalog list exited with {result.returncode}: {diagnostic}"
        )
    try:
        payload = json.loads(result.stdout)
        identifiers = tuple(sorted(item["catalogId"] for item in payload["catalogs"]))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise BundleAttestationError("bundled catalog list emitted invalid JSON") from error
    if payload.get("schemaVersion") != 1 or identifiers != expected_catalog_ids():
        raise BundleAttestationError("bundled catalog list omitted checked manifests")
    return identifiers, elapsed


def _run_project_help(entry_point: Path, timeout_seconds: float) -> float:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [str(entry_point), "run-project", "--help"],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BundleAttestationError(f"bundled run-project help failed: {error}") from error
    elapsed = time.perf_counter() - started
    if result.returncode != 0 or "usage: ultra-fast-wbpp run-project" not in result.stdout:
        diagnostic = result.stderr.strip()[-1000:] or "non-canonical help output"
        raise BundleAttestationError(f"bundled run-project help failed: {diagnostic}")
    return elapsed


def _attest_runtime_libraries(
    runtime_path: Path, entry_point: Path, timeout_seconds: float
) -> tuple[dict[str, Any], dict[str, Any], float]:
    provenance_path = runtime_path / "_internal" / "macos-runtime-library-provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise BundleAttestationError(
            "pinned macOS runtime-library provenance is missing or unsafe"
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleAttestationError(
            f"pinned macOS runtime-library provenance is invalid: {error}"
        ) from error
    expected_fields = {
        "schemaVersion",
        "kind",
        "targetTriple",
        "policySha256",
        "packages",
        "libraries",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_fields:
        raise BundleAttestationError("macOS runtime-library provenance fields changed")
    policy = load_macos14_runtime_policy()
    if (
        provenance["schemaVersion"] != 1
        or provenance["kind"]
        != "ultra-fast-wbpp-macos-runtime-library-provenance"
        or provenance["targetTriple"] != "aarch64-apple-darwin"
        or provenance["policySha256"]
        != _sha256(REPOSITORY / "packaging" / "worker" / "macos14-runtime-libraries-v1.json")
    ):
        raise BundleAttestationError("macOS runtime-library provenance identity changed")
    expected_packages = [
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
    packages = provenance["packages"]
    if (
        not isinstance(packages, list)
        or not all(isinstance(item, dict) for item in packages)
        or sorted(packages, key=lambda item: item.get("formula", ""))
        != sorted(expected_packages, key=lambda item: item["formula"])
    ):
        raise BundleAttestationError(
            "macOS runtime-library package provenance differs from pinned policy"
        )
    libraries = provenance["libraries"]
    if not isinstance(libraries, list) or {
        item.get("destinationName") for item in libraries if isinstance(item, dict)
    } != {"libcrypto.3.dylib", "libssl.3.dylib", "libmpdec.4.dylib"}:
        raise BundleAttestationError("macOS runtime-library provenance ABI set changed")
    policy_libraries = {
        library["destinationName"]: (package, library)
        for package in policy["packages"]
        for library in package["libraries"]
    }
    for item in libraries:
        if not isinstance(item, dict) or set(item) != {
            "destinationName",
            "formula",
            "version",
            "license",
            "sourceSha256",
            "relocatedSha256",
            "minimumDeploymentTarget",
        }:
            raise BundleAttestationError("macOS runtime-library receipt is malformed")
        package, library = policy_libraries[item["destinationName"]]
        destination = runtime_path / "_internal" / item["destinationName"]
        if destination.is_symlink() or not destination.is_file():
            raise BundleAttestationError("attested macOS runtime library is missing or unsafe")
        if (
            item["formula"] != package["formula"]
            or item["version"] != package["version"]
            or item["license"] != package["license"]
            or item["sourceSha256"] != library["sourceSha256"]
            or item["relocatedSha256"] != _sha256(destination)
            or item["minimumDeploymentTarget"] != "14.0.0"
        ):
            raise BundleAttestationError("macOS runtime-library receipt does not match its bytes")
    started = time.perf_counter()
    try:
        smoke = run_runtime_library_smoke(
            [str(entry_point)], timeout_seconds=timeout_seconds
        )
    except SidecarBuildError as error:
        raise BundleAttestationError(
            f"bundled runtime-library ABI smoke failed: {error}"
        ) from error
    elapsed = time.perf_counter() - started
    if (
        not smoke["openssl"].startswith(policy["runtimeSmoke"]["opensslPrefix"])
        or smoke["libmpdec"] != policy["runtimeSmoke"]["libmpdecVersion"]
    ):
        raise BundleAttestationError(
            "bundled Python loaded a different OpenSSL/mpdecimal ABI"
        )
    return provenance, smoke, elapsed


def _mac_signature(app: Path) -> dict[str, Any]:
    if app.is_symlink() or not app.is_dir() or app.suffix != ".app":
        raise BundleAttestationError("macOS bundle path must be a real .app directory")
    verified = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
        text=True,
        capture_output=True,
        check=False,
    )
    if verified.returncode != 0:
        raise BundleAttestationError(
            "codesign verification failed: " + verified.stderr.strip()[-1000:]
        )
    detail = subprocess.run(
        ["codesign", "-dv", "--verbose=4", str(app)],
        text=True,
        capture_output=True,
        check=False,
    )
    output = detail.stderr
    mode = "ad-hoc" if "Signature=adhoc" in output else "developer-id"
    hardened = "flags=0x10000(runtime)" in output or "runtime" in output
    if not hardened:
        raise BundleAttestationError("macOS application signature is not hardened-runtime")
    return {"verified": True, "mode": mode, "hardenedRuntime": True}


def attest_runtime(
    resource_root: Path,
    target_triple: str,
    *,
    max_start_seconds: float,
    bundle_name: str,
    signature: Mapping[str, Any],
    require_macos_runtime_provenance: bool = False,
) -> dict[str, Any]:
    target = normalize_target_triple(target_triple)
    if max_start_seconds <= 0:
        raise ValueError("max_start_seconds must be positive")
    pre_sign = _load_pre_sign_manifest(resource_root, target)
    runtime_path = resource_root / runtime_directory_name(target)
    post_sign_runtime = build_runtime_record(runtime_path, target)
    _require_same_logical_layout(pre_sign["runtime"], post_sign_runtime)
    entry_point = runtime_path / post_sign_runtime["entryPoint"]

    version, version_seconds = _run_version(entry_point, max_start_seconds)
    handshake, handshake_seconds = _run_handshake(entry_point, max_start_seconds)
    catalog_ids, catalog_seconds = _run_catalog_list(entry_point, max_start_seconds)
    project_help_seconds = _run_project_help(entry_point, max_start_seconds)
    runtime_library_provenance = None
    runtime_library_smoke = None
    runtime_library_smoke_seconds = 0.0
    if require_macos_runtime_provenance:
        if target != "aarch64-apple-darwin":
            raise BundleAttestationError(
                "macOS runtime-library provenance was requested for a non-arm64 target"
            )
        (
            runtime_library_provenance,
            runtime_library_smoke,
            runtime_library_smoke_seconds,
        ) = _attest_runtime_libraries(runtime_path, entry_point, max_start_seconds)
    if max(
        version_seconds,
        handshake_seconds,
        catalog_seconds,
        project_help_seconds,
        runtime_library_smoke_seconds,
    ) > max_start_seconds:
        raise BundleAttestationError("bundled runtime exceeded the startup budget")
    expected_version = pre_sign["protocol"]["engineVersion"]
    if handshake["payload"]["implementationVersion"] != expected_version:
        raise BundleAttestationError("bundled worker version differs from pre-sign manifest")

    payload = {
        "schemaVersion": 1,
        "kind": "openastroflow-bundled-runtime-attestation",
        "targetTriple": target,
        "bundleName": bundle_name,
        "preSignTreeSha256": pre_sign["runtime"]["treeSha256"],
        "runtime": post_sign_runtime,
        "signature": dict(signature),
        "launch": {
            "maximumSeconds": max_start_seconds,
            "version": version,
            "versionSeconds": round(version_seconds, 6),
            "handshakeSeconds": round(handshake_seconds, 6),
            "catalogListSeconds": round(catalog_seconds, 6),
            "catalogIds": list(catalog_ids),
            "runProjectHelpSeconds": round(project_help_seconds, 6),
            "protocolVersion": handshake["protocolVersion"],
            "implementationVersion": handshake["payload"]["implementationVersion"],
        },
    }
    if runtime_library_provenance is not None and runtime_library_smoke is not None:
        payload["runtimeLibraries"] = {
            "provenance": runtime_library_provenance,
            "smoke": runtime_library_smoke,
            "smokeSeconds": round(runtime_library_smoke_seconds, 6),
        }
    return payload


def write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--app", type=Path, help="signed macOS .app bundle")
    location.add_argument(
        "--resource-root", type=Path, help="installed openastroflow-worker resource root"
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-start-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        legal_resources = None
        macos_deployment = None
        if arguments.app is not None:
            if platform.system() != "Darwin":
                raise BundleAttestationError("--app signing verification is macOS-only")
            application_resources = arguments.app / "Contents" / "Resources"
            resource_root = (
                application_resources
                / "resources"
                / "openastroflow-worker"
            )
            legal_resources = attest_legal_resources(application_resources)
            signature = _mac_signature(arguments.app)
            macos_deployment = attest_macos_deployment(arguments.app)
            bundle_name = arguments.app.name
        else:
            resource_root = arguments.resource_root
            signature = {
                "verified": False,
                "mode": "unsigned-prerelease",
                "hardenedRuntime": False,
            }
            bundle_name = "installed-runtime"
        payload = attest_runtime(
            resource_root,
            arguments.target,
            max_start_seconds=arguments.max_start_seconds,
            bundle_name=bundle_name,
            signature=signature,
            require_macos_runtime_provenance=arguments.app is not None,
        )
        if legal_resources is not None:
            payload["legalResources"] = legal_resources
        if macos_deployment is not None:
            payload["macosDeployment"] = macos_deployment
        write_create_only(arguments.output, payload)
        print(json.dumps({"ok": True, "output": arguments.output.name, "launch": payload["launch"]}, sort_keys=True))
        return 0
    except (
        BundleAttestationError,
        FileExistsError,
        OSError,
        ResourceBoundaryError,
        ValueError,
    ) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BundleAttestationError",
    "MACOS_MAXIMUM_DEPLOYMENT_TARGET",
    "attest_legal_resources",
    "attest_macos_deployment",
    "attest_runtime",
    "inspect_macos_bundle",
    "main",
    "write_create_only",
]
