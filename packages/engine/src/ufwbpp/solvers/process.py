"""Bounded external-process execution, source identity, and solved-file evidence.

Shared by the ASTAP and Astrometry.net adapters; no solver discovery policy lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

from astropy.io import fits
from astropy.wcs import WCS

from .. import platform as platform_services
from ..solver import SolutionKind, SolveRequest, SolverResult, SolverStatus, canonical_wcs_sha256, validate_solver_result, validate_wcs_header


_MAX_CONTROL_FILE_BYTES = 16 * 1024 * 1024
_LOG_TAIL_BYTES = 64 * 1024
_COPY_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExecutableProbe:
    path: str | None
    available: bool
    execution_ready: bool
    version: str
    capabilities: tuple[str, ...] = ()
    error_code: str | None = None
    message: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        value = {
            "available": self.available,
            "executionReady": self.execution_ready,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "errorCode": self.error_code,
            "message": self.message,
            "evidence": self.evidence,
        }
        sanitized = share_safe_evidence(value)
        assert isinstance(sanitized, dict)
        return sanitized


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int
    sha256: str

    def serializable(self, *, expose_path: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "sizeBytes": self.size_bytes,
            "mtimeNs": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
            "sha256": f"sha256:{self.sha256}",
        }
        if expose_path:
            value["path"] = self.path
        return value


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    duration_ms: int
    stdout_sha256: str
    stderr_sha256: str
    stdout_tail: str
    stderr_tail: str
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    spawn_error: str | None = None

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.spawn_error is None and self.exit_code == 0

    def serializable(self) -> dict[str, Any]:
        arguments_sha256 = hashlib.sha256(
            json.dumps(self.argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "argumentCount": len(self.argv),
            "argumentsSha256": f"sha256:{arguments_sha256}",
            "shell": False,
            "exitCode": self.exit_code,
            "timedOut": self.timed_out,
            "durationMs": self.duration_ms,
            "stdoutSha256": f"sha256:{self.stdout_sha256}",
            "stderrSha256": f"sha256:{self.stderr_sha256}",
            "stdoutBytes": self.stdout_bytes,
            "stderrBytes": self.stderr_bytes,
            "spawnErrorClass": (
                self.spawn_error.split(":", 1)[0] if self.spawn_error is not None else None
            ),
        }


class SolverExecutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def is_executable_file(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve(strict=True)
        return resolved.is_file() and (os.name == "nt" or os.access(resolved, os.X_OK))
    except (OSError, RuntimeError):
        return False


def candidate_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not is_executable_file(path):
        return None
    # Preserve the invocation path.  In particular, a Python virtual-environment
    # launcher is a symlink whose argv[0] location selects that environment.
    # The resolved target is still hashed separately before every execution.
    return str(path.absolute())


def sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(_COPY_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def regular_identity(path: Path, *, max_bytes: int | None = None) -> FileIdentity:
    try:
        before = path.lstat()
    except OSError as error:
        raise SolverExecutionError("ARTIFACT_MISSING", f"missing artifact: {path}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SolverExecutionError(
            "ARTIFACT_NOT_REGULAR", f"artifact is not a regular, non-symlink file: {path}"
        )
    if max_bytes is not None and before.st_size > max_bytes:
        raise SolverExecutionError(
            "ARTIFACT_TOO_LARGE", f"artifact exceeds the {max_bytes}-byte safety limit: {path}"
        )
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise SolverExecutionError("ARTIFACT_DRIFT", f"artifact changed while opening: {path}")
            digest = sha256_stream(handle)
            after_open = os.fstat(handle.fileno())
        after = path.lstat()
    except SolverExecutionError:
        raise
    except OSError as error:
        raise SolverExecutionError("ARTIFACT_READ_FAILED", f"cannot read artifact {path}: {error}") from error
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after_open, key) for key in identity_fields) or any(
        getattr(before, key) != getattr(after, key) for key in identity_fields
    ):
        raise SolverExecutionError("ARTIFACT_DRIFT", f"artifact changed while reading: {path}")
    return FileIdentity(
        path=str(path.resolve(strict=True)),
        size_bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,
        device=before.st_dev,
        inode=before.st_ino,
        sha256=digest,
    )


def copy_source_to_stage(source: Path, destination: Path) -> tuple[FileIdentity, FileIdentity]:
    resolved = source.expanduser().resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise SolverExecutionError("INPUT_NOT_REGULAR", f"input is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as reader, destination.open("xb") as writer:
            opened = os.fstat(reader.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise SolverExecutionError("INPUT_DRIFT", "input changed while opening")
            while True:
                chunk = reader.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
            after_open = os.fstat(reader.fileno())
        after = resolved.stat()
    except SolverExecutionError:
        raise
    except OSError as error:
        raise SolverExecutionError("INPUT_COPY_FAILED", f"cannot stage input: {error}") from error
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after_open, key) for key in keys) or any(
        getattr(before, key) != getattr(after, key) for key in keys
    ):
        raise SolverExecutionError("INPUT_DRIFT", "input changed while it was staged")
    digest_value = digest.hexdigest()
    staged_stat = destination.stat()
    source_identity = FileIdentity(
        str(resolved), before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino, digest_value
    )
    staged_identity = FileIdentity(
        str(destination.resolve()),
        staged_stat.st_size,
        staged_stat.st_mtime_ns,
        staged_stat.st_dev,
        staged_stat.st_ino,
        digest_value,
    )
    return source_identity, staged_identity


def same_stat(path: Path, identity: FileIdentity) -> bool:
    try:
        value = path.expanduser().resolve(strict=True).stat()
    except OSError:
        return False
    return (
        value.st_size == identity.size_bytes
        and value.st_mtime_ns == identity.mtime_ns
        and value.st_dev == identity.device
        and value.st_ino == identity.inode
    )


def require_unchanged_identity(
    path: Path,
    identity: FileIdentity,
    *,
    max_bytes: int | None = None,
) -> None:
    after = regular_identity(path, max_bytes=max_bytes)
    if after != identity:
        raise SolverExecutionError("ARTIFACT_DRIFT", f"artifact changed after validation: {path}")


class BoundedLogCapture:
    """Drain a child pipe while retaining only a fixed-size tail in memory."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._tail = bytearray()
        self.byte_count = 0
        self.error: str | None = None

    def consume(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                self._digest.update(chunk)
                self.byte_count += len(chunk)
                if len(chunk) >= _LOG_TAIL_BYTES:
                    self._tail[:] = chunk[-_LOG_TAIL_BYTES:]
                else:
                    self._tail.extend(chunk)
                    overflow = len(self._tail) - _LOG_TAIL_BYTES
                    if overflow > 0:
                        del self._tail[:overflow]
        except (OSError, ValueError) as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            try:
                stream.close()
            except OSError:
                pass

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def tail(self) -> str:
        return bytes(self._tail).decode("utf-8", errors="replace")


def kill_process_tree(process: subprocess.Popen[Any]) -> None:
    platform_services.current().kill_process_tree(process)


def kill_lingering_posix_group(process: subprocess.Popen[Any]) -> None:
    """Close the private process group after the direct child exits.

    A solver is not allowed to leave helpers running that can mutate staged
    evidence after the parent has reported success.
    """

    if os.name != "posix":
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def solver_search_path(executable: str, current: str | None) -> str:
    """``PATH`` for the solver process: the solver's own directories first.

    ``solve-field`` runs its helpers (``image2pnm``, netpbm's ``pnmfile``,
    ``astrometry-engine``) through ``/bin/sh`` and looks them up on ``PATH``.
    A GUI launched from the Finder inherits launchd's minimal ``PATH``
    without ``/opt/homebrew/bin``, so a solver discovered at a well-known
    location would still fail with ``pnmfile: command not found`` (exit 255
    within a second).  Putting the invocation directory and the resolved
    (symlink target) directory of the executable in front of the inherited
    ``PATH`` makes the solver process self-sufficient without widening the
    environment to anything the solver did not come with.
    """

    directories: list[str] = []
    path = Path(executable)
    for candidate in (path.parent, path.resolve().parent):
        text = str(candidate)
        if text not in directories:
            directories.append(text)
    for entry in (current or "").split(os.pathsep):
        if entry and entry not in directories:
            directories.append(entry)
    return os.pathsep.join(directories)


def restricted_environment(overrides: Mapping[str, str] | None, *, executable: str | None = None) -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "ProgramFiles",
        "ProgramFiles(x86)",
        "SystemRoot",
        "WINDIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
    }
    platform_id = platform_services.current().platform_id
    if platform_id == "windows":
        # CreateProcess and the tools' own helpers resolve the system drive,
        # extensions and the command interpreter from these.
        allowed |= {"SystemDrive", "ProgramData", "ComSpec", "PATHEXT"}
    allowed_names = {name.casefold() for name in allowed}
    # ``os.environ`` upper-cases its keys on Windows; match by name, keep the
    # spelling the process had.
    environment = {
        key: value for key, value in os.environ.items() if key.casefold() in allowed_names
    }
    environment = platform_services.merged_environment(
        environment, {"LANG": "C", "LC_ALL": "C"}, platform_id=platform_id
    )
    if overrides:
        for key, value in overrides.items():
            if not isinstance(key, str) or not isinstance(value, str) or "\x00" in key + value:
                raise ValueError("solver environment keys and values must be NUL-free strings")
        environment = platform_services.merged_environment(environment, overrides, platform_id=platform_id)
    if executable is not None:
        view = platform_services.environment_view(environment, platform_id=platform_id)
        search_path = solver_search_path(executable, view.get("PATH"))
        environment = platform_services.merged_environment(environment, {"PATH": search_path}, platform_id=platform_id)
    return environment


class SolverProcessRuntime:
    """Run one external solver command in a new process group with bounded logs."""

    def __init__(
        self,
        executable: str,
        *,
        executable_args: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        diagnostic_log_root: str | os.PathLike[str] | None = None,
    ) -> None:
        resolved = candidate_path(executable)
        if resolved is None:
            raise SolverExecutionError("EXECUTABLE_UNAVAILABLE", f"not an executable file: {executable}")
        if any(not isinstance(item, str) or "\x00" in item for item in executable_args):
            raise ValueError("executable_args must be NUL-free strings")
        self.executable = resolved
        self.executable_args = tuple(executable_args)
        self.environment = restricted_environment(environment, executable=resolved)
        configured_log_root = diagnostic_log_root or os.environ.get(
            "UFWBPP_SOLVER_DIAGNOSTIC_DIR"
        )
        self.diagnostic_log_root = (
            Path(configured_log_root).expanduser().absolute()
            if configured_log_root is not None
            else None
        )

    @property
    def command_prefix(self) -> tuple[str, ...]:
        return (self.executable, *self.executable_args)

    def executable_identities(self) -> tuple[FileIdentity, ...]:
        identities = [regular_identity(Path(self.executable).resolve(strict=True))]
        for argument in self.executable_args:
            path = Path(argument)
            if path.is_file():
                identities.append(regular_identity(path))
        return tuple(identities)

    def run(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        log_stem: str,
    ) -> tuple[ProcessOutcome, tuple[FileIdentity, ...]]:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        argv = (*self.command_prefix, *tuple(str(item) for item in arguments))
        before = self.executable_identities()
        # perf_counter keeps sub-millisecond resolution on Windows, where
        # monotonic() ticks every 15.6 ms and would round short solves away.
        started = time.perf_counter()
        exit_code: int | None = None
        timed_out = False
        spawn_error: str | None = None
        process: subprocess.Popen[Any] | None = None
        popen_options: dict[str, Any] = {
            "args": list(argv),
            "cwd": str(cwd),
            "env": self.environment,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
            "close_fds": True,
            "bufsize": 0,
        }
        popen_options.update(platform_services.current().child_process_options().popen_kwargs())
        stdout_capture = BoundedLogCapture()
        stderr_capture = BoundedLogCapture()
        capture_threads: list[threading.Thread] = []
        try:
            process = subprocess.Popen(**popen_options)
            assert process.stdout is not None and process.stderr is not None
            capture_threads = [
                threading.Thread(target=stdout_capture.consume, args=(process.stdout,), daemon=True),
                threading.Thread(target=stderr_capture.consume, args=(process.stderr,), daemon=True),
            ]
            for thread in capture_threads:
                thread.start()
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_process_tree(process)
                try:
                    exit_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError) as error:
            spawn_error = f"{type(error).__name__}: {error}"
        finally:
            if process is not None:
                kill_lingering_posix_group(process)
                for thread in capture_threads:
                    thread.join(timeout=2)
                for stream, thread in zip((process.stdout, process.stderr), capture_threads, strict=False):
                    if thread.is_alive() and stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
                for thread in capture_threads:
                    thread.join(timeout=2)
        duration_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        capture_errors = [value for value in (stdout_capture.error, stderr_capture.error) if value]
        if capture_errors and spawn_error is None:
            spawn_error = "log capture failed: " + "; ".join(capture_errors)
        after = self.executable_identities()
        if before != after:
            raise SolverExecutionError("EXECUTABLE_DRIFT", "solver executable changed during execution")
        outcome = ProcessOutcome(
                argv=argv,
                exit_code=exit_code,
                timed_out=timed_out,
                duration_ms=duration_ms,
                stdout_sha256=stdout_capture.sha256,
                stderr_sha256=stderr_capture.sha256,
                stdout_tail=stdout_capture.tail,
                stderr_tail=stderr_capture.tail,
                stdout_bytes=stdout_capture.byte_count,
                stderr_bytes=stderr_capture.byte_count,
                spawn_error=spawn_error,
            )
        self._write_local_diagnostic(outcome, cwd=cwd, log_stem=log_stem)
        return (outcome, before)

    def _write_local_diagnostic(
        self,
        outcome: ProcessOutcome,
        *,
        cwd: Path,
        log_stem: str,
    ) -> None:
        root = self.diagnostic_log_root
        if root is None:
            return
        root.mkdir(parents=True, exist_ok=True)
        metadata = root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SolverExecutionError(
                "DIAGNOSTIC_LOG_DIRECTORY_UNSAFE",
                "solver diagnostic log root is not a regular directory",
            )
        secrets = {
            str(cwd),
            self.executable,
            str(Path.home()),
            *(value for value in self.environment.values() if value),
            *(item for item in self.executable_args if item),
        }
        payload = {
            "schemaVersion": 1,
            "process": outcome.serializable(),
            "argv": [redact_diagnostic_text(item, secrets) for item in outcome.argv],
            "stdoutTail": redact_diagnostic_text(outcome.stdout_tail, secrets),
            "stderrTail": redact_diagnostic_text(outcome.stderr_tail, secrets),
        }
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", log_stem).strip(".-") or "solver"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{safe_stem}-",
            suffix=".json",
            dir=root,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")


def redact_diagnostic_text(value: str, secrets: set[str]) -> str:
    redacted = value
    for secret in sorted((item for item in secrets if len(item) >= 3), key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    # Catch previously unknown absolute paths emitted by a tool.  URLs are not
    # matched because the leading slash must be at the beginning or whitespace.
    redacted = re.sub(
        r"(?:(?<=\s)|^)/(?:[^\s\x00]+)",
        "<redacted-local-path>",
        redacted,
    )
    redacted = re.sub(
        r"(?:(?<=\s)|^)[A-Za-z]:[\\/][^\s\x00]+",
        "<redacted-local-path>",
        redacted,
    )
    return redacted


def build_execution_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    value = share_safe_evidence(payload)
    if not isinstance(value, dict):
        raise TypeError("receipt payload must remain an object")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    value["receiptSha256"] = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    return value


_PRIVATE_RECEIPT_KEYS = {
    "argv",
    "path",
    "configPath",
    "inputPath",
    "device",
    "inode",
    "mtimeNs",
    "outputPath",
    "receiptPath",
    "stdoutTail",
    "stderrTail",
    "spawnError",
}


def share_safe_evidence(value: Any) -> Any:
    """Remove machine-local process details from a publishable receipt.

    External-solver stdout/stderr and exact argv remain runtime diagnostics.
    A receipt keeps only their hashes, sizes, exit status, duration, and the
    adapter's separately structured scientific parameters.
    """

    if isinstance(value, Mapping):
        return {
            str(key): share_safe_evidence(item)
            for key, item in value.items()
            if str(key) not in _PRIVATE_RECEIPT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [share_safe_evidence(item) for item in value]
    if isinstance(value, str):
        # Defense in depth for an exception or tool message that consists of a
        # local absolute path.  Normal scientific strings and URLs are retained.
        if os.path.isabs(value) or re.fullmatch(r"[A-Za-z]:[\\/].*", value):
            return "<redacted-local-path>"
    return value


def verify_execution_receipt(evidence: Mapping[str, Any]) -> bool:
    """Verify the self-digest of an adapter receipt (not a digital signature)."""

    try:
        claimed = evidence["receiptSha256"]
        if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
            return False
        payload = dict(evidence)
        payload.pop("receiptSha256")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return hmac.compare_digest(claimed, f"sha256:{actual}")
    except (KeyError, TypeError, ValueError):
        return False


def wcs_header_sha256(header: Mapping[str, Any] | fits.Header) -> str:
    return canonical_wcs_sha256(header)


def verify_solver_execution_result(result: SolverResult) -> bool:
    """Verify receipt, returned WCS, and currently published file as one unit."""

    try:
        if not verify_execution_receipt(result.evidence):
            return False
        if result.evidence.get("receiptVersion") != 2 or result.evidence.get("outcome") != "SOLVED":
            return False
        if not validate_solver_result(result).valid:
            return False
        if result.evidence.get("backendId") != result.backend_id:
            return False
        claimed_header = result.evidence.get("solutionWcsSha256")
        if claimed_header != f"sha256:{wcs_header_sha256(result.header)}":
            return False
        receipt_quality = result.evidence.get("astrometricQuality")
        if result.astrometric_quality is None:
            if not isinstance(receipt_quality, Mapping) or receipt_quality.get("status") != "UNAVAILABLE":
                return False
        else:
            serialized_quality = result.astrometric_quality.serializable()
            if receipt_quality != serialized_quality:
                return False
            outputs = result.evidence.get("outputs")
            if not isinstance(outputs, Mapping):
                return False
            correspondence = outputs.get("corr")
            match_artifact = outputs.get("match")
            match_diagnostics = result.evidence.get("matchDiagnostics")
            if (
                not isinstance(correspondence, Mapping)
                or correspondence.get("sha256")
                != f"sha256:{serialized_quality['correspondenceSha256']}"
                or not isinstance(match_artifact, Mapping)
                or not isinstance(match_artifact.get("sha256"), str)
                or not isinstance(match_diagnostics, Mapping)
                or match_diagnostics.get("indexIdentities") != serialized_quality["indexIdentities"]
            ):
                return False
        if result.output_path is None:
            return False
        published = result.evidence["outputs"]["published"]
        current = regular_identity(Path(result.output_path))
        return (
            published.get("sha256") == f"sha256:{current.sha256}"
            and published.get("sizeBytes") == current.size_bytes
        )
    except (KeyError, OSError, TypeError, ValueError, SolverExecutionError):
        return False


def failure(
    backend_id: str,
    code: str,
    message: str,
    *,
    status: SolverStatus = SolverStatus.FAILED,
    evidence: Mapping[str, Any] | None = None,
) -> SolverResult:
    receipt = {
        "receiptVersion": 1,
        "backendId": backend_id,
        "outcome": "FAILURE",
        "failureCode": code,
    }
    if evidence:
        receipt.update(evidence)
    return SolverResult(
        backend_id=backend_id,
        status=status,
        solution_kind=SolutionKind.NONE,
        backend_confirmed=False,
        evidence=build_execution_receipt(receipt),
        error=f"{code}: {message}",
    )


def validate_request(request: SolveRequest) -> None:
    if not isinstance(request, SolveRequest):
        raise SolverExecutionError("REQUEST_INVALID", "request must be a SolveRequest")
    pairs = (request.ra_hint_degrees, request.dec_hint_degrees)
    if (pairs[0] is None) != (pairs[1] is None):
        raise SolverExecutionError("SEED_INCOMPLETE", "RA and Dec hints must be provided together")
    numeric = {
        "ra_hint_degrees": (request.ra_hint_degrees, 0.0, 360.0, False),
        "dec_hint_degrees": (request.dec_hint_degrees, -90.0, 90.0, True),
        "field_of_view_degrees": (request.field_of_view_degrees, 0.0, 180.0, False),
        "search_radius_degrees": (request.search_radius_degrees, 0.0, 180.0, False),
    }
    for name, (value, lower, upper, inclusive_upper) in numeric.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SolverExecutionError("SEED_INVALID", f"{name} must be finite")
        allowed = lower <= float(value) <= upper if inclusive_upper else lower < float(value) <= upper
        if name == "ra_hint_degrees":
            allowed = 0.0 <= float(value) < 360.0
        if not allowed:
            raise SolverExecutionError("SEED_INVALID", f"{name} is outside its supported range")
    input_path = Path(request.input_path).expanduser()
    output_path = Path(request.output_path).expanduser()
    if output_path.suffix.lower() not in {".fits", ".fit", ".fts"}:
        raise SolverExecutionError(
            "OUTPUT_FORMAT_UNSUPPORTED",
            "output_path must use a FITS extension (.fits, .fit, or .fts)",
        )
    try:
        if input_path.resolve(strict=True) == output_path.resolve(strict=False):
            raise SolverExecutionError("IN_PLACE_OUTPUT_FORBIDDEN", "output_path must not be the input path")
    except OSError as error:
        raise SolverExecutionError("INPUT_UNAVAILABLE", str(error)) from error
    if output_path.exists() or output_path.is_symlink():
        raise SolverExecutionError("OUTPUT_EXISTS", f"refusing to replace existing output: {output_path}")


def input_shape(path: Path) -> tuple[int, int]:
    try:
        header = fits.getheader(path, ext=0, memmap=False)
        width = int(header["NAXIS1"])
        height = int(header["NAXIS2"])
    except Exception as error:
        raise SolverExecutionError("INPUT_GEOMETRY_INVALID", f"cannot read 2-D FITS geometry: {error}") from error
    if int(header.get("NAXIS", 0)) < 2 or width < 2 or height < 2:
        raise SolverExecutionError(
            "INPUT_GEOMETRY_INVALID",
            "solver input must expose at least two primary FITS image axes",
        )
    return height, width


def read_control_text(path: Path) -> tuple[str, FileIdentity]:
    identity = regular_identity(path, max_bytes=_MAX_CONTROL_FILE_BYTES)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise SolverExecutionError("ARTIFACT_READ_FAILED", str(error)) from error
    require_unchanged_identity(path, identity, max_bytes=_MAX_CONTROL_FILE_BYTES)
    return text, identity


def read_wcs_header(path: Path) -> tuple[fits.Header, FileIdentity]:
    identity = regular_identity(path, max_bytes=_MAX_CONTROL_FILE_BYTES)
    errors: list[str] = []
    for loader in (
        lambda: fits.Header.fromfile(path, sep="", endcard=True, padding=True),
        lambda: fits.Header.fromstring(path.read_bytes().decode("ascii"), sep=""),
        lambda: fits.Header.fromtextfile(path),
        lambda: fits.getheader(path, ext=0, memmap=False),
    ):
        try:
            header = loader()
            require_unchanged_identity(path, identity, max_bytes=_MAX_CONTROL_FILE_BYTES)
            return header, identity
        except SolverExecutionError:
            raise
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    raise SolverExecutionError("WCS_ARTIFACT_INVALID", "; ".join(errors[-2:]))


def wcs_only_header(header: fits.Header) -> fits.Header:
    try:
        return WCS(header, relax=True).celestial.to_header(relax=True)
    except Exception as error:
        raise SolverExecutionError("WCS_ARTIFACT_INVALID", str(error)) from error


def is_wcs_keyword(keyword: str) -> bool:
    upper = keyword.upper()
    if upper in {"WCSAXES", "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX", "MJDREF"}:
        return True
    return bool(
        re.fullmatch(r"(?:CTYPE|CUNIT|CRPIX|CRVAL|CDELT|CROTA)\d+[A-Z]?", upper)
        or re.fullmatch(r"(?:CD|PC|PV|PS)\d+_\d+[A-Z]?", upper)
        or re.fullmatch(r"(?:A|B|AP|BP)_ORDER", upper)
        or re.fullmatch(r"(?:A|B|AP|BP)_\d+_\d+", upper)
    )


def publish_solved_copy(input_path: Path, output_path: Path, wcs_header: fits.Header) -> FileIdentity:
    output_parent = output_path.expanduser().resolve(strict=False).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise SolverExecutionError("OUTPUT_EXISTS", f"refusing to replace existing output: {output_path}")
    wcs_only = wcs_only_header(wcs_header)
    temporary_path: Path | None = None
    try:
        with fits.open(input_path, mode="readonly", memmap=False) as hdul:
            for key in list(hdul[0].header):
                upper = key.upper()
                if upper in wcs_only or is_wcs_keyword(upper):
                    del hdul[0].header[key]
            hdul[0].header.update(wcs_only)
            hdul[0].header.add_history("Ultra-Fast WBPP: fresh external plate-solver WCS")
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=".ultra-fast-wbpp-solved-",
                suffix=".fits",
                dir=output_parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                hdul.writeto(handle, output_verify="exception")
                handle.flush()
                os.fsync(handle.fileno())
        assert temporary_path is not None
        temporary_shape = input_shape(temporary_path)
        temporary_header = fits.getheader(temporary_path, ext=0, memmap=False)
        temporary_validation = validate_wcs_header(temporary_header, image_shape=temporary_shape)
        if not temporary_validation.valid:
            raise SolverExecutionError(
                "PUBLISHED_WCS_INVALID",
                f"{temporary_validation.code}: {temporary_validation.message}",
            )
        temporary_identity = regular_identity(temporary_path)
        created_output = False
        committed = False
        try:
            os.link(temporary_path, output_path)
            created_output = True
        except FileExistsError as error:
            raise SolverExecutionError("OUTPUT_EXISTS", f"output appeared during publication: {output_path}") from error
        except OSError as error:
            raise SolverExecutionError(
                "ATOMIC_PUBLICATION_UNAVAILABLE",
                f"cannot publish with atomic no-replace semantics: {error}",
            ) from error
        try:
            published = regular_identity(output_path)
            if (published.device, published.inode, published.sha256) != (
                temporary_identity.device,
                temporary_identity.inode,
                temporary_identity.sha256,
            ):
                raise SolverExecutionError(
                    "OUTPUT_DRIFT", "published output does not match the staged artifact"
                )
            fsync_directory(output_parent)
            committed = True
            return published
        finally:
            if created_output and not committed:
                try:
                    current = output_path.lstat()
                    if (current.st_dev, current.st_ino) == (
                        temporary_identity.device,
                        temporary_identity.inode,
                    ):
                        output_path.unlink(missing_ok=True)
                except OSError:
                    pass
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def fsync_directory(path: Path) -> None:
    try:
        platform_services.current().fsync_directory(path)
    except OSError as error:
        raise SolverExecutionError("DIRECTORY_SYNC_FAILED", str(error)) from error

