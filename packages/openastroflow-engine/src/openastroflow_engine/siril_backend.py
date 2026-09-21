"""Audited Siril 1.4.4 external workflow backend.

This module intentionally supports one documented Siril command surface only:
Siril 1.4.4's ``siril-cli -d WORKDIR -s SCRIPT`` runner and its scriptable
``convert``, ``calibrate``, ``register``, ``stack``, ``load``, ``platesolve``
and ``save`` commands.  A different version is discoverable but not execution
ready.  This conservative boundary keeps a future Siril syntax change from
silently changing an Ultra-Fast WBPP recipe.

Official command references used by the adapter:

* https://siril.org/docs/man/
* https://siril.readthedocs.io/en/stable/Commands.html

Caller-owned FITS files are never passed to Siril.  They are identity-checked,
copied under fixed names into a fresh private working directory, made
read-only, and checked again after the process exits.  Siril is invoked with an
argv vector (never a shell).  A timeout kills the private process group through
the shared external-process runtime and the temporary directory is then
removed.  Outputs and receipts use no-replace hard-link publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from astropy.io import fits

from . import platform as platform_services
from .astap_backend import (
    ExecutableProbe,
    SolverExecutionError,
    SolverProcessRuntime,
    _FileIdentity,
    _copy_source_to_stage,
    _regular_identity,
    _same_stat,
)
from .backends import BackendDescriptor, DeviceKind, StageKind
from .solver import validate_wcs_header


ADAPTER_VERSION = "siril-cli-workflow-v1"
CERTIFIED_SIRIL_VERSION = (1, 4, 4)
CERTIFIED_SIRIL_VERSION_TEXT = "1.4.4"
_COPY_CHUNK_BYTES = 4 * 1024 * 1024
_SUCCESS_MARKER = re.compile(r"(?i)script execution finished successfully\s*\.")
_FAILURE_MARKER = re.compile(
    r"(?i)(?:script execution failed|error in line\s+\d+|exiting batch processing)"
)
_SUPPORTED_SUFFIXES = {".fit", ".fits", ".fts"}
_DRIZZLE_KERNELS = {"point", "turbo", "square", "gaussian", "lanczos2", "lanczos3"}

_BASE_CAPABILITIES = (
    "calibration-light-flat-dark-bias",
    "registration-global-star",
    "integration-mean-rejection",
    "isolated-staging-copy",
    "process-timeout",
    "no-shell",
    "identity-bound-receipt-v1",
)
_DOCUMENTED_OPTIONAL_CAPABILITIES = (
    "drizzle-hst-registration",
    "plate-solve-local-astrometry-net",
    "fail-closed-wcs-validation",
)


class SirilWorkflowError(RuntimeError):
    """Stable fail-closed error used at the Siril adapter boundary."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SirilWorkflowRequest:
    """One mono or already channel-grouped calibration-to-master workflow.

    Calibration collections contain raw frames, not prebuilt masters.  Empty
    calibration collections are allowed.  ``drizzle_scale=None`` requests the
    ordinary registration/integration path; 1, 2, or 3 enables Siril's
    documented HST-drizzle registration export.

    Plate solving uses Siril's documented local astrometry.net mode so the
    adapter never silently depends on a remote catalogue.  The local
    ``solve-field`` executable and suitable indexes remain runtime
    prerequisites and a missing prerequisite fails the script closed.
    """

    light_files: tuple[str, ...]
    output_path: str
    flat_files: tuple[str, ...] = ()
    dark_files: tuple[str, ...] = ()
    bias_files: tuple[str, ...] = ()
    receipt_path: str | None = None
    workers: int | None = None
    drizzle_scale: int | None = None
    drizzle_pixfrac: float = 1.0
    drizzle_kernel: str = "square"
    plate_solve: bool = False
    ra_hint_degrees: float | None = None
    dec_hint_degrees: float | None = None
    focal_length_mm: float | None = None
    pixel_size_microns: float | None = None
    search_radius_degrees: float | None = None


@dataclass(frozen=True, slots=True)
class SirilWorkflowResult:
    success: bool
    code: str
    output_path: str | None = None
    receipt_path: str | None = None
    receipt: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "code": self.code,
            "outputPath": self.output_path,
            "receiptPath": self.receipt_path,
            "receipt": self.receipt,
            "error": self.error,
        }


def _candidate_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or (os.name != "nt" and not os.access(resolved, os.X_OK)):
            return None
    except (OSError, RuntimeError):
        return None
    # Preserve a virtual-environment launcher path; the shared runtime hashes
    # its resolved executable before and after every invocation.
    return str(path.absolute())


def _sys_platform() -> str:
    import sys

    return sys.platform


def discover_siril_cli(
    executable: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Find ``siril-cli`` on macOS, Windows, or Linux without invoking it."""

    if executable is not None:
        return _candidate_path(executable)
    env = platform_services.environment_view(
        os.environ if environment is None else environment,
        platform_id=platform_services.current().platform_id,
    )
    for key in ("OPENASTROFLOW_SIRIL_CLI", "SIRIL_CLI_PATH"):
        candidate = _candidate_path(env.get(key))
        if candidate:
            return candidate
    for name in ("siril-cli", "siril-cli.exe"):
        candidate = _candidate_path(shutil.which(name, path=env.get("PATH")))
        if candidate:
            return candidate

    candidates = platform_services.current().well_known_executables("siril-cli", environment=env)
    for path in candidates:
        candidate = _candidate_path(path)
        if candidate:
            return candidate
    return None


def _parse_version(text: str) -> tuple[str, tuple[int, int, int] | None]:
    match = re.search(
        r"(?i)\bsiril(?:-cli)?(?:\s+version)?[^0-9]{0,12}"
        r"([0-9]+)\.([0-9]+)\.([0-9]+)(?:[-+._a-z0-9]*)?",
        text,
    )
    if match is None:
        match = re.search(r"\b([0-9]+)\.([0-9]+)\.([0-9]+)\b", text)
    if match is None:
        return "unknown", None
    parsed = tuple(int(match.group(index)) for index in range(1, 4))
    return ".".join(str(item) for item in parsed), parsed  # type: ignore[return-value]


def _has_cli_option(help_text: str, short: str, long: str) -> bool:
    return bool(
        re.search(rf"(?:^|[\s,]){re.escape(short)}(?:[\s,]|$)", help_text)
        or re.search(rf"(?:^|[\s,]){re.escape(long)}(?:[=\s,]|$)", help_text)
    )


def _probe_runtime(runtime: SolverProcessRuntime, timeout_seconds: float) -> ExecutableProbe:
    try:
        with tempfile.TemporaryDirectory(prefix="openastroflow-siril-probe-") as raw:
            cwd = Path(raw)
            version_outcome, version_executables = runtime.run(
                ("--version",), cwd=cwd, timeout_seconds=timeout_seconds, log_stem="version"
            )
            help_outcome, help_executables = runtime.run(
                ("--help",), cwd=cwd, timeout_seconds=timeout_seconds, log_stem="help"
            )
        version_text = f"{version_outcome.stdout_tail}\n{version_outcome.stderr_tail}"
        help_text = f"{help_outcome.stdout_tail}\n{help_outcome.stderr_tail}"
        version, parsed = _parse_version(version_text)
        missing = tuple(
            name
            for name, short, long in (
                ("directory", "-d", "--directory"),
                ("script", "-s", "--script"),
            )
            if not _has_cli_option(help_text, short, long)
        )
        process_ok = all(
            outcome.spawn_error is None and not outcome.timed_out and outcome.exit_code == 0
            for outcome in (version_outcome, help_outcome)
        )
        certified = parsed == CERTIFIED_SIRIL_VERSION
        ready = process_ok and not missing and certified
        if ready:
            error_code = None
            message = None
        elif version_outcome.timed_out or help_outcome.timed_out:
            error_code = "PROBE_TIMEOUT"
            message = "siril-cli probe exceeded its timeout"
        elif not process_ok:
            error_code = "CAPABILITY_PROBE_FAILED"
            message = "siril-cli --version/--help probe failed"
        elif missing:
            error_code = "CAPABILITY_PROBE_FAILED"
            message = "siril-cli help is missing: " + ", ".join(missing)
        elif parsed is None:
            error_code = "VERSION_UNKNOWN"
            message = "could not parse the Siril version"
        else:
            error_code = "VERSION_UNSUPPORTED"
            message = (
                f"adapter is certified for Siril {CERTIFIED_SIRIL_VERSION_TEXT}; "
                f"found {version}"
            )
        executable_map = {
            (item.path, item.sha256): item
            for item in (*version_executables, *help_executables)
        }
        capabilities = (*_BASE_CAPABILITIES, *_DOCUMENTED_OPTIONAL_CAPABILITIES) if ready else ()
        return ExecutableProbe(
            path=runtime.executable,
            available=True,
            execution_ready=ready,
            version=version,
            capabilities=capabilities,
            error_code=error_code,
            message=message,
            evidence={
                "certifiedVersion": CERTIFIED_SIRIL_VERSION_TEXT,
                "versionProcess": version_outcome.serializable(),
                "helpProcess": help_outcome.serializable(),
                "executables": [
                    item.serializable() for item in executable_map.values()
                ],
                "requiredCliOptions": ["-d/--directory", "-s/--script"],
                "missingCliOptions": list(missing),
                "documentedScriptCommands": [
                    "requires",
                    "convert",
                    "calibrate",
                    "register",
                    "stack",
                    "load",
                    "platesolve",
                    "save",
                ],
            },
        )
    except (OSError, ValueError, SolverExecutionError) as error:
        code = error.code if isinstance(error, SolverExecutionError) else "CAPABILITY_PROBE_FAILED"
        return ExecutableProbe(
            runtime.executable,
            True,
            False,
            "unknown",
            error_code=code,
            message=str(error),
        )


def probe_siril_cli(
    executable: str | os.PathLike[str] | None = None,
    *,
    executable_args: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 5.0,
) -> ExecutableProbe:
    discovery_environment = platform_services.merged_environment(
        os.environ, environment, platform_id=platform_services.current().platform_id
    )
    path = discover_siril_cli(executable, environment=discovery_environment)
    if path is None:
        return ExecutableProbe(
            None,
            False,
            False,
            "unavailable",
            error_code="EXECUTABLE_UNAVAILABLE",
            message="siril-cli was not found",
        )
    try:
        runtime = SolverProcessRuntime(
            path, executable_args=executable_args, environment=environment
        )
    except (ValueError, SolverExecutionError) as error:
        return ExecutableProbe(
            path,
            False,
            False,
            "unavailable",
            error_code="EXECUTABLE_UNAVAILABLE",
            message=str(error),
        )
    return _probe_runtime(runtime, timeout_seconds)


def _finite_number(
    name: str,
    value: float | None,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = False,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SirilWorkflowError("OPTION_INVALID", f"{name} must be finite")
    numeric = float(value)
    lower_ok = numeric >= minimum if minimum_inclusive else numeric > minimum
    if not lower_ok or numeric > maximum:
        raise SirilWorkflowError("OPTION_INVALID", f"{name} is outside its supported range")


def _receipt_path(request: SirilWorkflowRequest, output: Path) -> Path:
    value = request.receipt_path
    if value is None:
        return output.with_name(output.name + ".receipt.json")
    return Path(value).expanduser().resolve(strict=False)


def _validate_request(request: SirilWorkflowRequest) -> tuple[Path, Path]:
    if not isinstance(request, SirilWorkflowRequest):
        raise SirilWorkflowError("REQUEST_INVALID", "request must be a SirilWorkflowRequest")
    if not request.light_files:
        raise SirilWorkflowError("LIGHT_INPUT_MISSING", "at least one Light frame is required")
    for role, values in (
        ("LIGHT", request.light_files),
        ("FLAT", request.flat_files),
        ("DARK", request.dark_files),
        ("BIAS", request.bias_files),
    ):
        if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
            raise SirilWorkflowError("INPUT_INVALID", f"{role} inputs must be non-empty path strings")
    if request.workers is not None and (
        isinstance(request.workers, bool)
        or not isinstance(request.workers, int)
        or not 1 <= request.workers <= 1024
    ):
        raise SirilWorkflowError("OPTION_INVALID", "workers must be an integer in [1, 1024]")
    if request.drizzle_scale is not None and (
        isinstance(request.drizzle_scale, bool)
        or not isinstance(request.drizzle_scale, int)
        or request.drizzle_scale not in {1, 2, 3}
    ):
        raise SirilWorkflowError("DRIZZLE_OPTION_INVALID", "drizzle_scale must be 1, 2, or 3")
    _finite_number(
        "drizzle_pixfrac", request.drizzle_pixfrac, minimum=0.0, maximum=1.0
    )
    if request.drizzle_kernel not in _DRIZZLE_KERNELS:
        raise SirilWorkflowError("DRIZZLE_OPTION_INVALID", "unsupported Siril drizzle kernel")
    pair = (request.ra_hint_degrees, request.dec_hint_degrees)
    if (pair[0] is None) != (pair[1] is None):
        raise SirilWorkflowError("PLATE_SOLVE_OPTION_INVALID", "RA and Dec hints are a pair")
    _finite_number(
        "ra_hint_degrees", request.ra_hint_degrees, minimum=0.0, maximum=360.0, minimum_inclusive=True
    )
    if request.ra_hint_degrees is not None and float(request.ra_hint_degrees) >= 360.0:
        raise SirilWorkflowError("PLATE_SOLVE_OPTION_INVALID", "RA hint must be below 360 degrees")
    _finite_number(
        "dec_hint_degrees", request.dec_hint_degrees, minimum=-90.0, maximum=90.0, minimum_inclusive=True
    )
    _finite_number("focal_length_mm", request.focal_length_mm, minimum=0.0, maximum=100_000.0)
    _finite_number("pixel_size_microns", request.pixel_size_microns, minimum=0.0, maximum=1_000.0)
    _finite_number(
        "search_radius_degrees",
        request.search_radius_degrees,
        minimum=0.0,
        maximum=180.0,
        minimum_inclusive=True,
    )

    output = Path(request.output_path).expanduser().resolve(strict=False)
    if output.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise SirilWorkflowError("OUTPUT_FORMAT_UNSUPPORTED", "output must be a FITS file")
    receipt = _receipt_path(request, output)
    if output == receipt:
        raise SirilWorkflowError("OUTPUT_CONFLICT", "output and receipt paths must differ")
    for path, label in ((output, "output"), (receipt, "receipt")):
        if path.exists() or path.is_symlink():
            raise SirilWorkflowError("OUTPUT_EXISTS", f"refusing to replace existing {label}: {path}")
    return output, receipt


def _source_paths(request: SirilWorkflowRequest) -> tuple[tuple[str, int, Path], ...]:
    values: list[tuple[str, int, Path]] = []
    seen: dict[Path, str] = {}
    for role, paths in (
        ("LIGHT", request.light_files),
        ("FLAT", request.flat_files),
        ("DARK", request.dark_files),
        ("BIAS", request.bias_files),
    ):
        for index, raw in enumerate(paths, start=1):
            path = Path(raw).expanduser()
            try:
                before = path.lstat()
            except OSError as error:
                raise SirilWorkflowError("INPUT_MISSING", f"missing {role} input {raw}: {error}") from error
            if stat.S_ISLNK(before.st_mode):
                raise SirilWorkflowError("INPUT_SYMLINK_FORBIDDEN", f"{role} input is a symlink: {raw}")
            if not stat.S_ISREG(before.st_mode):
                raise SirilWorkflowError("INPUT_NOT_REGULAR", f"{role} input is not a regular file: {raw}")
            if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                raise SirilWorkflowError("INPUT_FORMAT_UNSUPPORTED", f"{role} input is not FITS: {raw}")
            resolved = path.resolve(strict=True)
            previous = seen.get(resolved)
            if previous is not None:
                raise SirilWorkflowError(
                    "INPUT_DUPLICATE", f"input appears more than once ({previous}, {role}): {resolved}"
                )
            seen[resolved] = role
            values.append((role, index, resolved))
    return tuple(values)


def _fits_shape(path: Path) -> tuple[int, int]:
    try:
        header = fits.getheader(path, ext=0, memmap=False)
        if int(header.get("NAXIS", 0)) < 2:
            raise ValueError("primary HDU has fewer than two axes")
        width = int(header["NAXIS1"])
        height = int(header["NAXIS2"])
    except Exception as error:
        raise SirilWorkflowError("FITS_INVALID", f"cannot read 2-D FITS geometry from {path}: {error}") from error
    if width < 2 or height < 2:
        raise SirilWorkflowError("FITS_INVALID", f"invalid FITS geometry in {path}")
    return height, width


def _format_number(value: float) -> str:
    # Inputs have already been bounded and made finite.  This produces only a
    # sign, ASCII digits, a decimal point, and possibly an exponent.
    return f"{float(value):.12g}"


def build_siril_script(request: SirilWorkflowRequest) -> str:
    """Build a path-independent, reviewable Siril 1.4.4 script.

    All filenames in the script are adapter-controlled constants.  Caller path
    text can therefore never become Siril syntax.
    """

    _validate_request(request)
    lines = [
        "# Ultra-Fast WBPP audited Siril workflow",
        f"# Adapter: {ADAPTER_VERSION}",
        "requires 1.4.4 1.4.5",
        "setext fit",
        "set32bits",
        "setcompress 0",
    ]
    if request.workers is not None:
        lines.append(f"setcpu {request.workers}")

    if request.bias_files:
        lines.extend(
            (
                "cd bias",
                "convert bias -out=../process",
                "cd ../process",
                "stack bias rej w 3 3 -nonorm -out=../masters/master_bias -32b",
                "cd ..",
            )
        )
    if request.flat_files:
        lines.extend(("cd flat", "convert flat -out=../process", "cd ../process"))
        flat_sequence = "flat"
        if request.bias_files:
            lines.append("calibrate flat -bias=../masters/master_bias")
            flat_sequence = "pp_flat"
        lines.extend(
            (
                f"stack {flat_sequence} rej w 3 3 -norm=mul -out=../masters/master_flat -32b",
                "cd ..",
            )
        )
    if request.dark_files:
        lines.extend(
            (
                "cd dark",
                "convert dark -out=../process",
                "cd ../process",
                "stack dark rej w 3 3 -nonorm -out=../masters/master_dark -32b",
                "cd ..",
            )
        )

    lines.extend(("cd light", "convert light -out=../process", "cd ../process"))
    calibration_options: list[str] = []
    if request.dark_files:
        calibration_options.append("-dark=../masters/master_dark")
    elif request.bias_files:
        # A raw dark master already contains the bias pedestal.  Passing bias
        # alongside that dark would subtract it twice.
        calibration_options.append("-bias=../masters/master_bias")
    if request.flat_files:
        calibration_options.append("-flat=../masters/master_flat")
    light_sequence = "light"
    if calibration_options:
        lines.append("calibrate light " + " ".join(calibration_options))
        light_sequence = "pp_light"

    registration = f"register {light_sequence}"
    if request.drizzle_scale is not None:
        registration += (
            f" -drizzle -scale={request.drizzle_scale}"
            f" -pixfrac={_format_number(request.drizzle_pixfrac)}"
            f" -kernel={request.drizzle_kernel}"
        )
    lines.append(registration)
    lines.append(
        f"stack r_{light_sequence} rej w 3 3 -norm=addscale "
        "-output_norm -out=../output/master -32b"
    )

    if request.plate_solve:
        lines.append("load ../output/master.fit")
        plate = ["platesolve", "-force", "-noflip", "-localasnet"]
        if request.ra_hint_degrees is None:
            plate.append("-blindpos")
        else:
            plate.append(
                f"{_format_number(request.ra_hint_degrees)},{_format_number(request.dec_hint_degrees or 0.0)}"
            )
        if request.focal_length_mm is None or request.pixel_size_microns is None:
            plate.append("-blindres")
        else:
            plate.extend(
                (
                    f"-focal={_format_number(request.focal_length_mm)}",
                    f"-pixelsize={_format_number(request.pixel_size_microns)}",
                )
            )
        if request.search_radius_degrees is not None:
            plate.append(f"-radius={_format_number(request.search_radius_degrees)}")
        lines.append(" ".join(plate))
        lines.append("save ../output/solved -chksum")
    return "\n".join(lines) + "\n"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _signed_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    # This is an integrity checksum, not an authenticity signature.
    value = dict(payload)
    canonical = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    value["receiptSha256"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return value


def _identity_matches_serialized(path: Path, value: Mapping[str, Any]) -> bool:
    identity = _regular_identity(path)
    return (
        value.get("path") == identity.path
        and value.get("sizeBytes") == identity.size_bytes
        and value.get("mtimeNs") == identity.mtime_ns
        and value.get("device") == identity.device
        and value.get("inode") == identity.inode
        and value.get("sha256") == f"sha256:{identity.sha256}"
    )


def verify_siril_receipt(receipt: Mapping[str, Any], *, verify_files: bool = True) -> bool:
    """Verify receipt self-digest, script identity, and bound file identities."""

    try:
        claimed = receipt.get("receiptSha256")
        if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
            return False
        payload = dict(receipt)
        payload.pop("receiptSha256", None)
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        actual = "sha256:" + hashlib.sha256(canonical).hexdigest()
        if not hmac.compare_digest(claimed, actual):
            return False
        script = receipt["script"]
        content = script["content"]
        if not isinstance(content, str):
            return False
        encoded = content.encode("utf-8")
        if script.get("sizeBytes") != len(encoded):
            return False
        if script.get("sha256") != "sha256:" + hashlib.sha256(encoded).hexdigest():
            return False
        if not verify_files:
            return True
        for source in receipt["inputs"]:
            if not _identity_matches_serialized(Path(source["path"]), source):
                return False
        artifact = receipt.get("artifact")
        if receipt.get("status") == "succeeded":
            if not isinstance(artifact, Mapping):
                return False
            if not _identity_matches_serialized(Path(str(artifact["path"])), artifact):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, SolverExecutionError):
        return False


def _copy_to_temporary(source: Path, parent: Path, suffix: str) -> tuple[Path, _FileIdentity]:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=".openastroflow-siril-publish-", suffix=suffix, dir=parent, delete=False
        ) as writer:
            temporary = Path(writer.name)
            with source.open("rb") as reader:
                while chunk := reader.read(_COPY_CHUNK_BYTES):
                    writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        identity = _regular_identity(temporary)
        return temporary, identity
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _publish_no_replace(temporary: Path, temporary_identity: _FileIdentity, destination: Path) -> _FileIdentity:
    created = False
    committed = False
    try:
        try:
            os.link(temporary, destination)
            created = True
        except FileExistsError as error:
            raise SirilWorkflowError("OUTPUT_EXISTS", f"output appeared during publication: {destination}") from error
        except OSError as error:
            raise SirilWorkflowError(
                "ATOMIC_PUBLICATION_UNAVAILABLE",
                f"cannot publish with atomic no-replace semantics: {error}",
            ) from error
        published = _regular_identity(destination)
        if (
            published.device,
            published.inode,
            published.size_bytes,
            published.sha256,
        ) != (
            temporary_identity.device,
            temporary_identity.inode,
            temporary_identity.size_bytes,
            temporary_identity.sha256,
        ):
            raise SirilWorkflowError("OUTPUT_DRIFT", "published file differs from validated temporary")
        _fsync_directory(destination.parent)
        committed = True
        return published
    finally:
        if created and not committed:
            try:
                current = destination.lstat()
                if (current.st_dev, current.st_ino) == (
                    temporary_identity.device,
                    temporary_identity.inode,
                ):
                    destination.unlink(missing_ok=True)
            except OSError:
                pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if os.name == "nt" or error.errno in {errno.EACCES, errno.EINVAL, errno.ENOTSUP}:
            return
        raise SirilWorkflowError("DIRECTORY_SYNC_FAILED", str(error)) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        if not (os.name == "nt" or error.errno in {errno.EINVAL, errno.ENOTSUP}):
            raise SirilWorkflowError("DIRECTORY_SYNC_FAILED", str(error)) from error
    finally:
        os.close(descriptor)


def _failure(
    code: str,
    message: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> SirilWorkflowResult:
    payload: dict[str, Any] = {
        "receiptVersion": 1,
        "adapterVersion": ADAPTER_VERSION,
        "backendId": "siril-cli",
        "status": "failed",
        "code": code,
        "error": message,
    }
    if evidence:
        payload["evidence"] = dict(evidence)
    return SirilWorkflowResult(False, code, receipt=_signed_receipt(payload), error=message)


class SirilBackend:
    """Optional cross-platform Siril 1.4.4 calibration/stacking worker."""

    backend_id = "siril-cli"

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        *,
        executable_args: Sequence[str] = (),
        timeout_seconds: float = 3600.0,
        probe_timeout_seconds: float = 5.0,
        staging_root: str | os.PathLike[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.staging_root = Path(staging_root).expanduser() if staging_root is not None else None
        self.environment = dict(environment or {})
        discovery_environment = platform_services.merged_environment(
            os.environ, self.environment, platform_id=platform_services.current().platform_id
        )
        path = discover_siril_cli(executable, environment=discovery_environment)
        self.runtime: SolverProcessRuntime | None = None
        if path is None:
            self.probe = ExecutableProbe(
                None,
                False,
                False,
                "unavailable",
                error_code="EXECUTABLE_UNAVAILABLE",
                message="siril-cli was not found",
            )
        else:
            try:
                self.runtime = SolverProcessRuntime(
                    path, executable_args=executable_args, environment=self.environment
                )
                self.probe = _probe_runtime(self.runtime, probe_timeout_seconds)
            except (ValueError, SolverExecutionError) as error:
                self.probe = ExecutableProbe(
                    path,
                    False,
                    False,
                    "unavailable",
                    error_code="EXECUTABLE_UNAVAILABLE",
                    message=str(error),
                )

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.backend_id,
            stage=StageKind.INTEGRATION,
            display_name="Siril CLI end-to-end worker",
            version=self.probe.version,
            available=self.probe.available,
            execution_ready=self.probe.execution_ready,
            devices=(DeviceKind.CPU,),
            capabilities=self.probe.capabilities,
            reason=self.probe.message,
            metadata={
                "adapterVersion": ADAPTER_VERSION,
                "certifiedSirilVersion": CERTIFIED_SIRIL_VERSION_TEXT,
                "probe": self.probe.serializable(),
                "platforms": ["macOS", "Windows", "Linux"],
                "runtimePrerequisites": {
                    "plateSolve": "local solve-field plus matching astrometry.net indexes",
                },
                "officialDocumentation": [
                    "https://siril.org/docs/man/",
                    "https://siril.readthedocs.io/en/stable/Commands.html",
                ],
            },
        )

    def validate_options(self, options: dict[str, Any]) -> tuple[str, ...]:
        allowed = {
            "workers",
            "drizzleScale",
            "drizzlePixfrac",
            "drizzleKernel",
            "plateSolve",
            "raHintDegrees",
            "decHintDegrees",
            "focalLengthMm",
            "pixelSizeMicrons",
            "searchRadiusDegrees",
        }
        return tuple(f"unknown Siril option: {key}" for key in sorted(set(options) - allowed))

    def run(self, request: SirilWorkflowRequest) -> SirilWorkflowResult:
        if self.runtime is None or not self.probe.execution_ready:
            return _failure(
                self.probe.error_code or "EXECUTABLE_UNAVAILABLE",
                self.probe.message or "siril-cli is not execution-ready",
                evidence={"probe": self.probe.serializable()},
            )
        try:
            if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
                raise SirilWorkflowError("TIMEOUT_INVALID", "timeout must be finite and positive")
            output_path, receipt_path = _validate_request(request)
            sources = _source_paths(request)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() or output_path.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
                raise SirilWorkflowError("OUTPUT_EXISTS", "output or receipt appeared before execution")

            stage_parent = self.staging_root
            if stage_parent is not None:
                stage_parent.mkdir(parents=True, exist_ok=True)
                stage_parent = stage_parent.resolve(strict=True)
            else:
                stage_parent = output_path.parent
            with tempfile.TemporaryDirectory(
                prefix=".openastroflow-siril-", dir=stage_parent
            ) as raw_stage:
                stage = Path(raw_stage)
                try:
                    stage.chmod(0o700)
                except OSError:
                    pass
                for name in ("light", "flat", "dark", "bias", "process", "masters", "output"):
                    (stage / name).mkdir(mode=0o700)

                source_records: list[dict[str, Any]] = []
                staged_records: list[tuple[Path, _FileIdentity]] = []
                expected_shape: tuple[int, int] | None = None
                for role, index, source in sources:
                    shape = _fits_shape(source)
                    if expected_shape is None:
                        expected_shape = shape
                    elif shape != expected_shape:
                        raise SirilWorkflowError(
                            "INPUT_GEOMETRY_MISMATCH",
                            f"{role} geometry {shape} does not match {expected_shape}: {source}",
                        )
                    role_dir = role.lower()
                    staged = stage / role_dir / f"frame_{index:06d}.fits"
                    source_identity, staged_identity = _copy_source_to_stage(source, staged)
                    try:
                        staged.chmod(stat.S_IRUSR)
                    except OSError:
                        pass
                    record = source_identity.serializable()
                    record.update({"role": role, "index": index})
                    source_records.append(record)
                    staged_records.append((staged, staged_identity))

                script = build_siril_script(request)
                script_bytes = script.encode("utf-8")
                script_path = stage / "workflow.ssf"
                with script_path.open("xb") as stream:
                    stream.write(script_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    script_path.chmod(stat.S_IRUSR)
                except OSError:
                    pass
                script_identity = _regular_identity(script_path)
                outcome, executable_identities = self.runtime.run(
                    ("-d", str(stage), "-s", str(script_path)),
                    cwd=stage,
                    timeout_seconds=self.timeout_seconds,
                    log_stem="workflow",
                )
                base_evidence = {
                    "adapterVersion": ADAPTER_VERSION,
                    "probe": self.probe.serializable(),
                    "inputs": source_records,
                    "script": {
                        "content": script,
                        "sizeBytes": len(script_bytes),
                        "sha256": f"sha256:{script_identity.sha256}",
                    },
                    "process": outcome.serializable(),
                    "executables": [item.serializable() for item in executable_identities],
                    "workDirectory": {"private": True, "retained": False},
                }
                if outcome.timed_out:
                    return _failure("SIRIL_TIMEOUT", "siril-cli exceeded its execution timeout", evidence=base_evidence)
                if outcome.spawn_error:
                    return _failure("SIRIL_SPAWN_FAILED", outcome.spawn_error, evidence=base_evidence)
                if outcome.exit_code != 0:
                    return _failure(
                        "SIRIL_EXIT_NONZERO",
                        f"siril-cli exited with code {outcome.exit_code}",
                        evidence=base_evidence,
                    )
                log_tail = f"{outcome.stdout_tail}\n{outcome.stderr_tail}"
                if _FAILURE_MARKER.search(log_tail):
                    return _failure("SIRIL_LOG_FAILURE", "Siril reported script failure", evidence=base_evidence)
                if _SUCCESS_MARKER.search(log_tail) is None:
                    return _failure(
                        "SIRIL_CONFIRMATION_MISSING",
                        "Siril did not emit its documented successful-script completion marker",
                        evidence=base_evidence,
                    )

                for record in source_records:
                    identity = _FileIdentity(
                        path=record["path"],
                        size_bytes=record["sizeBytes"],
                        mtime_ns=record["mtimeNs"],
                        device=record["device"],
                        inode=record["inode"],
                        sha256=str(record["sha256"]).removeprefix("sha256:"),
                    )
                    if not _same_stat(Path(record["path"]), identity):
                        return _failure("SOURCE_DRIFT", "a source changed while Siril ran", evidence=base_evidence)
                    current = _regular_identity(Path(record["path"]))
                    if current != identity:
                        return _failure("SOURCE_DRIFT", "a source changed while Siril ran", evidence=base_evidence)
                for staged, identity in staged_records:
                    current = _regular_identity(staged)
                    if current.sha256 != identity.sha256 or current.size_bytes != identity.size_bytes:
                        return _failure(
                            "STAGED_INPUT_DRIFT",
                            "Siril modified a read-only staged input",
                            evidence=base_evidence,
                        )
                current_script = _regular_identity(script_path)
                if current_script.sha256 != script_identity.sha256:
                    return _failure("SCRIPT_DRIFT", "Siril modified its audited script", evidence=base_evidence)

                stage_output = stage / "output" / ("solved.fit" if request.plate_solve else "master.fit")
                try:
                    stage_identity = _regular_identity(stage_output)
                    shape = _fits_shape(stage_output)
                    header = fits.getheader(stage_output, ext=0, memmap=False)
                except (SirilWorkflowError, SolverExecutionError, OSError) as error:
                    code = error.code if hasattr(error, "code") else "SIRIL_OUTPUT_INVALID"
                    return _failure(str(code), str(error), evidence=base_evidence)

                wcs_evidence: dict[str, Any] | None = None
                if request.plate_solve:
                    validation = validate_wcs_header(header, image_shape=shape)
                    wcs_evidence = validation.serializable()
                    if not validation.valid:
                        return _failure(
                            "WCS_VALIDATION_FAILED",
                            f"{validation.code}: {validation.message}",
                            evidence={**base_evidence, "wcsValidation": wcs_evidence},
                        )

                publish_temporary, publish_identity = _copy_to_temporary(
                    stage_output, output_path.parent, output_path.suffix
                )
                published_identity: _FileIdentity | None = None
                published_receipt_identity: _FileIdentity | None = None
                receipt_temporary: Path | None = None
                try:
                    published_identity = _publish_no_replace(
                        publish_temporary, publish_identity, output_path
                    )
                    payload: dict[str, Any] = {
                        "receiptVersion": 1,
                        "adapterVersion": ADAPTER_VERSION,
                        "backendId": self.backend_id,
                        "backendVersion": self.probe.version,
                        "status": "succeeded",
                        "code": "SIRIL_WORKFLOW_SUCCEEDED",
                        "inputs": source_records,
                        "script": base_evidence["script"],
                        "process": base_evidence["process"],
                        "executables": base_evidence["executables"],
                        "workDirectory": base_evidence["workDirectory"],
                        "request": {
                            "lightCount": len(request.light_files),
                            "flatCount": len(request.flat_files),
                            "darkCount": len(request.dark_files),
                            "biasCount": len(request.bias_files),
                            "workers": request.workers,
                            "drizzle": request.drizzle_scale is not None,
                            "drizzleScale": request.drizzle_scale,
                            "drizzlePixfrac": request.drizzle_pixfrac,
                            "drizzleKernel": request.drizzle_kernel,
                            "plateSolve": request.plate_solve,
                            "raHintDegrees": request.ra_hint_degrees,
                            "decHintDegrees": request.dec_hint_degrees,
                            "focalLengthMm": request.focal_length_mm,
                            "pixelSizeMicrons": request.pixel_size_microns,
                            "searchRadiusDegrees": request.search_radius_degrees,
                        },
                        "stageArtifact": stage_identity.serializable(expose_path=False),
                        "artifact": published_identity.serializable(),
                        "imageShape": list(shape),
                        "wcsValidation": wcs_evidence,
                        "receiptPath": str(receipt_path),
                    }
                    receipt = _signed_receipt(payload)
                    receipt_bytes = _canonical_json(receipt)
                    with tempfile.NamedTemporaryFile(
                        mode="w+b",
                        prefix=".openastroflow-siril-receipt-",
                        suffix=".json",
                        dir=receipt_path.parent,
                        delete=False,
                    ) as stream:
                        receipt_temporary = Path(stream.name)
                        stream.write(receipt_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    receipt_temp_identity = _regular_identity(receipt_temporary)
                    published_receipt_identity = _publish_no_replace(
                        receipt_temporary, receipt_temp_identity, receipt_path
                    )
                    if not verify_siril_receipt(receipt):
                        raise SirilWorkflowError(
                            "RECEIPT_VERIFICATION_FAILED", "new Siril receipt did not verify"
                        )
                    return SirilWorkflowResult(
                        True,
                        "SIRIL_WORKFLOW_SUCCEEDED",
                        output_path=str(output_path),
                        receipt_path=str(receipt_path),
                        receipt=receipt,
                    )
                except Exception:
                    if published_receipt_identity is not None:
                        try:
                            current_receipt = receipt_path.lstat()
                            if (current_receipt.st_dev, current_receipt.st_ino) == (
                                published_receipt_identity.device,
                                published_receipt_identity.inode,
                            ):
                                receipt_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                    if published_identity is not None:
                        try:
                            current = output_path.lstat()
                            if (current.st_dev, current.st_ino) == (
                                published_identity.device,
                                published_identity.inode,
                            ):
                                output_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                    raise
                finally:
                    publish_temporary.unlink(missing_ok=True)
                    if receipt_temporary is not None:
                        receipt_temporary.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError, SirilWorkflowError, SolverExecutionError) as error:
            code = error.code if hasattr(error, "code") else "SIRIL_WORKFLOW_FAILED"
            return _failure(str(code), str(error))


def execute_siril_workflow(
    request: SirilWorkflowRequest,
    *,
    executable: str | os.PathLike[str] | None = None,
    executable_args: Sequence[str] = (),
    timeout_seconds: float = 3600.0,
    probe_timeout_seconds: float = 5.0,
    staging_root: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> SirilWorkflowResult:
    """Convenience entry point for callers that do not retain a backend."""

    backend = SirilBackend(
        executable,
        executable_args=executable_args,
        timeout_seconds=timeout_seconds,
        probe_timeout_seconds=probe_timeout_seconds,
        staging_root=staging_root,
        environment=environment,
    )
    return backend.run(request)


__all__ = [
    "ADAPTER_VERSION",
    "CERTIFIED_SIRIL_VERSION_TEXT",
    "SirilBackend",
    "SirilWorkflowError",
    "SirilWorkflowRequest",
    "SirilWorkflowResult",
    "build_siril_script",
    "discover_siril_cli",
    "execute_siril_workflow",
    "probe_siril_cli",
    "verify_siril_receipt",
]
