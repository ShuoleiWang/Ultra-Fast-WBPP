"""Small helpers shared by the single-target and project workflows: progress events, create-only JSON and directory commits, artifact records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import platform as platform_services
from ..integrity import canonical_json_document, sha256_digest
from ..path_budget import name_token
from ..platform import NoReplaceError
from .contracts import ProgressStage, ProgressEvent, E2EError, ProgressCallback


def _emit(
    callback: ProgressCallback | None,
    stage: ProgressStage,
    status: str,
    message: str,
    *,
    current: int = 0,
    total: int = 0,
) -> None:
    if callback is not None:
        callback(ProgressEvent(stage, status, current, total, message))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_document(payload)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise E2EError("OUTPUT_EXISTS", "refusing to replace JSON artifact", path=str(path)) from error


def _safe_token(value: str) -> str:
    token = name_token(value)
    if not token:
        raise E2EError("FILTER_INVALID", "filter name cannot be encoded safely")
    return token


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Create-only directory publication through the platform service layer.

    ``OUTPUT_EXISTS`` only when the destination exists; any other failure of
    the rename is ``PUBLICATION_FAILED``.  Either way ``source`` is untouched.
    """

    try:
        platform_services.current().rename_directory_no_replace(source, destination)
    except NoReplaceError as error:
        if error.code == "OUTPUT_EXISTS":
            message = (
                "refusing to replace output directory"
                if error.precheck
                else "output appeared during publication"
            )
            raise E2EError("OUTPUT_EXISTS", message, path=str(destination)) from error
        raise E2EError(error.code, error.message) from error
    except OSError as error:
        if os.path.lexists(destination):
            raise E2EError("OUTPUT_EXISTS", "output appeared during publication", path=str(destination)) from error
        reason = os.strerror(error.errno) if error.errno else str(error)
        raise E2EError(
            "PUBLICATION_FAILED", f"the directory could not be renamed: {reason}", path=str(destination)
        ) from error


def _unpublished_reason(error: E2EError, output: Path) -> str:
    """Why a publication failed, naming the directories beside the output by
    name: this text also goes into the failure receipt, which holds no host path."""

    return str(error).removeprefix(f"{error.code}: ").replace(str(output.parent) + os.sep, "")


def _fsync_directory(path: Path) -> None:
    platform_services.current().fsync_directory(path)


def _artifact_records(staging: Path, roots: Sequence[Path], *, workers: int = 4) -> list[dict[str, Any]]:
    """Digest every artifact under ``roots`` in a stable order; the files are
    independent and hashing releases the GIL, so they are digested concurrently."""

    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend([root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file()))
    count = max(1, min(int(workers), len(paths)))
    if count <= 1:
        digests = [sha256_digest(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="ufwbpp-artifacts") as pool:
            digests = list(pool.map(sha256_digest, paths))
    return [
        {
            "path": path.relative_to(staging).as_posix(),
            "sha256": digest,
            "sizeBytes": path.stat().st_size,
        }
        for path, digest in zip(paths, digests, strict=True)
    ]


def _relativize_solver_attempts(
    attempts: list[dict[str, Any]], staging: Path
) -> list[dict[str, Any]]:
    """Relativize outer paths while preserving signed backend evidence bytes.

    Adapter evidence contains its own receipt digest.  Recursively rewriting
    paths inside that evidence would silently invalidate the backend receipt,
    so only the E2E-owned wrapper fields are changed here.
    """

    for attempt in attempts:
        result = attempt.get("result")
        if isinstance(result, dict):
            output_path = result.get("outputPath")
            if isinstance(output_path, str):
                try:
                    result["outputPath"] = Path(output_path).relative_to(staging).as_posix()
                except ValueError:
                    pass
        artifact = attempt.get("artifact")
        if isinstance(artifact, dict):
            artifact_path = artifact.get("path")
            if isinstance(artifact_path, str):
                try:
                    artifact["path"] = Path(artifact_path).relative_to(staging).as_posix()
                except ValueError:
                    pass
        if attempt.get("executionVerified") is True:
            attempt["verificationBoundary"] = (
                "verified before atomic outer-directory publication; final content "
                "identity is rebound by the E2E receipt"
            )
    return attempts
