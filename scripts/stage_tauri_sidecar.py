#!/usr/bin/env python3
"""Verify and create-only stage a frozen worker tree for a Tauri bundle.

The PyInstaller builder emits an attested onedir tree. Tauri embeds that tree
as an application resource, so Python and native libraries are memory-mapped
in place and are never unpacked on GUI launches. This release gate validates
every tree entry immediately before bundling and never replaces a destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePath
import shutil
import stat
from typing import Any, Mapping, Sequence

try:
    from scripts.build_worker_sidecar import (
        build_runtime_record,
        validate_manifest,
        verify_runtime_tree,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from build_worker_sidecar import build_runtime_record, validate_manifest, verify_runtime_tree


class SidecarStageError(RuntimeError):
    """The sidecar/manifest pair is not safe to place in the app bundle."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_tree_digest(manifest_path: Path) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(payload["runtime"]["treeSha256"])


def _regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SidecarStageError("INPUT_MISSING", f"cannot inspect {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SidecarStageError("INPUT_NOT_REGULAR", f"{label} must be a regular non-symlink file")


def _artifact_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SidecarStageError("MANIFEST_INVALID", "runtime.directoryName is missing")
    if PurePath(value).name != value or value in {".", ".."} or "\x00" in value:
        raise SidecarStageError("MANIFEST_INVALID", "runtime.directoryName must be one basename")
    if not value.startswith("openastroflow-worker-"):
        raise SidecarStageError("MANIFEST_INVALID", "runtime.directoryName has the wrong prefix")
    return value


def load_verified_pair(
    manifest_path: Path, *, expected_target: str | None = None
) -> tuple[Path, Mapping[str, Any]]:
    """Return the identity-bound runtime tree described by one manifest."""

    _regular_file(manifest_path, "sidecar manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SidecarStageError("MANIFEST_INVALID", str(error)) from error
    if not isinstance(payload, dict):
        raise SidecarStageError("MANIFEST_INVALID", "manifest must contain one JSON object")
    try:
        validate_manifest(payload)
    except (ValueError, TypeError) as error:
        raise SidecarStageError("MANIFEST_INVALID", str(error)) from error
    target = payload.get("targetTriple")
    if not isinstance(target, str) or not target:
        raise SidecarStageError("MANIFEST_INVALID", "targetTriple is missing")
    if expected_target is not None and target != expected_target:
        raise SidecarStageError(
            "TARGET_MISMATCH", f"manifest target {target!r} is not requested target {expected_target!r}"
        )
    runtime_record = payload.get("runtime")
    if not isinstance(runtime_record, dict):
        raise SidecarStageError("MANIFEST_INVALID", "runtime record is missing")
    name = _artifact_name(runtime_record.get("directoryName"))
    runtime = manifest_path.parent / name
    try:
        verify_runtime_tree(runtime, payload)
    except (ValueError, OSError, RuntimeError) as error:
        raise SidecarStageError("RUNTIME_IDENTITY_MISMATCH", str(error)) from error
    return runtime, payload




def _write_payload_create_only(destination: Path, payload: Mapping[str, Any]) -> None:
    rendered = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def stage_sidecar(
    manifest_path: Path,
    destination_directory: Path,
    *,
    expected_target: str | None = None,
) -> tuple[Path, Path]:
    runtime, payload = load_verified_pair(manifest_path, expected_target=expected_target)
    if destination_directory.exists() or destination_directory.is_symlink():
        raise FileExistsError(f"refusing to replace staged runtime root: {destination_directory}")
    destination_directory.parent.mkdir(parents=True, exist_ok=True)
    destination_directory.mkdir(exist_ok=False)
    runtime_destination = destination_directory / runtime.name
    manifest_destination = destination_directory / manifest_path.name
    try:
        # Tauri's resource walker does not accept POSIX symlinks in a resource
        # directory. The source tree was already verified to contain only safe
        # in-tree links, so materialize them and bind the resulting exact tree
        # with a new v2 manifest before bundling.
        shutil.copytree(runtime, runtime_destination, symlinks=False, copy_function=shutil.copy2)
        staged_payload = json.loads(json.dumps(payload))
        staged_payload["runtime"] = build_runtime_record(
            runtime_destination, str(payload["targetTriple"])
        )
        validate_manifest(staged_payload)
        _write_payload_create_only(manifest_destination, staged_payload)
        verify_runtime_tree(runtime_destination, staged_payload)
    except Exception:
        shutil.rmtree(destination_directory, ignore_errors=True)
        raise
    return runtime_destination, manifest_destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("apps/desktop/src-tauri/resources/openastroflow-worker"),
    )
    parser.add_argument("--target", help="require this exact target triple")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        artifact, manifest = stage_sidecar(
            arguments.manifest,
            arguments.destination,
            expected_target=arguments.target,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "runtime": artifact.name,
                    "manifest": manifest.name,
                    "treeSha256": payload_tree_digest(manifest),
                },
                sort_keys=True,
            )
        )
        return 0
    except (FileExistsError, OSError, SidecarStageError) as error:
        code = error.code if isinstance(error, SidecarStageError) else "OUTPUT_EXISTS" if isinstance(error, FileExistsError) else "IO_ERROR"
        print(json.dumps({"ok": False, "error": {"code": code, "message": str(error)}}, sort_keys=True), file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SidecarStageError", "load_verified_pair", "main", "stage_sidecar"]
