"""Fail-closed ASTAP process adapter and shared external-solver runtime.

The adapter deliberately gives ASTAP a private copy of the input and never uses
``-update``.  A successful process exit is only one item of evidence: ASTAP's
``PLTSOLVD=T`` marker, a stable ``.wcs`` artifact, the numerical validator in
``solver.py`` and an engine-side catalog-correspondence check against the
app-managed Astrometry.net index stars must all agree before a solved FITS
file is published.  The last step gives ASTAP solutions the same recomputed
match/RMS/index evidence and managed-catalog binding that ``solve-field``
solutions carry, so the E2E gate accepts both routes on equal terms.
"""

from __future__ import annotations
from .solvers.process import (
    ExecutableProbe,
    FileIdentity as _FileIdentity,
    ProcessOutcome as _ProcessOutcome,
    SolverExecutionError,
    is_executable_file as _is_executable_file,
    candidate_path as _candidate_path,
    sha256_stream as _sha256_stream,
    regular_identity as _regular_identity,
    copy_source_to_stage as _copy_source_to_stage,
    same_stat as _same_stat,
    require_unchanged_identity as _require_unchanged_identity,
    BoundedLogCapture as _BoundedLogCapture,
    kill_process_tree as _kill_process_tree,
    kill_lingering_posix_group as _kill_lingering_posix_group,
    solver_search_path as _solver_search_path,
    restricted_environment as _restricted_environment,
    SolverProcessRuntime,
    redact_diagnostic_text as _redact_diagnostic_text,
    build_execution_receipt as _receipt,
    share_safe_evidence as _share_safe_evidence,
    verify_execution_receipt,
    wcs_header_sha256 as _wcs_header_sha256,
    verify_solver_execution_result,
    failure as _failure,
    validate_request as _validate_request,
    input_shape as _input_shape,
    read_control_text as _read_control_text,
    read_wcs_header as _read_wcs_header,
    wcs_only_header as _wcs_only_header,
    is_wcs_keyword as _is_wcs_keyword,
    publish_solved_copy as _publish_solved_copy,
    fsync_directory as _fsync_directory,
)


import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from astropy.io import fits
import numpy as np

from . import platform as platform_services
from .backends import BackendDescriptor, DeviceKind, StageKind
from .catalog_correspondence import (
    CORRESPONDENCE_ARTIFACT_NAME,
    MATCH_ARTIFACT_NAME,
    VERIFICATION_METHOD,
    CatalogCorrespondenceError,
    CorrespondenceParameters,
    bind_installed_set,
    verify_solution,
)
from .catalogs import CatalogError, installed_set_snapshot_for_solver_config
from .solver import (
    AstrometricQuality,
    SolutionKind,
    SolveRequest,
    SolverResult,
    SolverStatus,
    canonical_wcs_sha256,
    validate_solver_result,
    validate_wcs_header,
)


ADAPTER_VERSION = "astap-process-v4"
_CAPABILITIES = (
    "seed-hints",
    "celestial-wcs",
    "fail-closed-wcs-validation",
    "isolated-staging-copy",
    "process-timeout",
    "provenance-receipt-v1",
    "catalog-correspondence-quality-v1",
    "managed-catalog-byte-binding-v1",
    "share-safe-process-evidence-v1",
)
_MAX_CONTROL_FILE_BYTES = 16 * 1024 * 1024
_MAX_CONFIG_BYTES = 1024 * 1024
# ASTAP star databases (D05/D20/D50/D80, G05/G17, H17/H18, V17, W08) are sets
# of files named ``<family><mag>_<area>.<1476|290>``, for example
# ``d20_0101.1476``; ASTAP looks for them beside its executable.
_STAR_DATABASE_FILE = re.compile(r"(?i)^[a-z][0-9]{2}_[0-9]{4}\.(?:1476|290)$")
_STAR_DATABASE_WELL_KNOWN = {
    "darwin": ("/usr/local/opt/astap", "/Applications/ASTAP.app/Contents/MacOS"),
    "linux": ("/opt/astap", "/usr/share/astap/data", "/usr/local/share/astap/data"),
}
_LOG_TAIL_BYTES = 64 * 1024
_COPY_CHUNK_BYTES = 4 * 1024 * 1024


def discover_astap(
    executable: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Find ASTAP without invoking it.

    Explicit paths win, followed by environment overrides, PATH, and conservative
    platform install locations.  Returned paths are canonical regular files.
    """

    if executable is not None:
        return _candidate_path(executable)
    env = platform_services.environment_view(
        os.environ if environment is None else environment,
        platform_id=platform_services.current().platform_id,
    )
    for key in ("UFWBPP_ASTAP", "ASTAP_PATH"):
        candidate = _candidate_path(env.get(key))
        if candidate:
            return candidate
    search_path = env.get("PATH")
    for name in ("astap_cli", "astap", "ASTAP"):
        found = shutil.which(name, path=search_path)
        candidate = _candidate_path(found)
        if candidate:
            return candidate

    candidates = platform_services.current().well_known_executables("astap", environment=env)
    for path in candidates:
        candidate = _candidate_path(path)
        if candidate:
            return candidate
    return None


def _star_database_families(directory: Path) -> dict[str, int]:
    families: dict[str, int] = {}
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if _STAR_DATABASE_FILE.fullmatch(entry.name) and entry.is_file(follow_symlinks=False):
                    family = entry.name[:3].lower()
                    families[family] = families.get(family, 0) + 1
    except OSError:
        return {}
    return families


def discover_astap_star_database(
    executable: str | os.PathLike[str],
    *,
    configured: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Locate ASTAP's star database without invoking ASTAP.

    A configured directory is authoritative and is later passed to ASTAP with
    ``-d``.  Otherwise the directories ASTAP itself searches are scanned: the
    executable's directory (invocation path and symlink target) and the
    platform's conventional install locations.  Paths are runtime state and
    stay out of shareable receipts; only the families and counts are reported.
    """

    candidates: list[tuple[Path, str]] = []
    if configured is not None:
        candidates.append((Path(configured).expanduser(), "configured"))
    else:
        path = Path(executable).expanduser()
        for directory in (path.parent, path.resolve().parent):
            if all(directory != seen for seen, _ in candidates):
                candidates.append((directory, "executable-directory"))
        platform_id = platform_services.current().platform_id
        for value in _STAR_DATABASE_WELL_KNOWN.get(platform_id, ()):
            directory = Path(value)
            if all(directory != seen for seen, _ in candidates):
                candidates.append((directory, "well-known-location"))
    for directory, source in candidates:
        families = _star_database_families(directory)
        if families:
            return {
                "found": True,
                "source": source,
                "path": str(directory.absolute()),
                "families": sorted(families),
                "fileCount": int(sum(families.values())),
            }
    return {"found": False, "source": None, "path": None, "families": [], "fileCount": 0}


def sys_platform() -> str:
    # Kept as a tiny seam so platform discovery can be tested without mutating
    # global interpreter state.
    import sys

    return sys.platform


def _probe_astap_runtime(
    runtime: SolverProcessRuntime,
    timeout_seconds: float,
    *,
    star_database_dir: str | os.PathLike[str] | None = None,
) -> ExecutableProbe:
    try:
        with tempfile.TemporaryDirectory(prefix="ultra-fast-wbpp-astap-probe-") as raw:
            outcome, identities = runtime.run(
                ("-help",), cwd=Path(raw), timeout_seconds=timeout_seconds, log_stem="probe"
            )
        output = f"{outcome.stdout_tail}\n{outcome.stderr_tail}"
        required = ("-f", "-o", "-wcs")
        missing = tuple(option for option in required if option not in output)
        # ``-sip`` adds distortion terms to the solution.  Without them a
        # linear TAN fit of a wide, fast optical system leaves residuals that
        # approach the E2E RMS limit, so the option is used whenever this
        # build advertises it.
        optional = ("-sip",)
        supported_optional = [option for option in optional if option in output]
        match = re.search(r"(?i)ASTAP(?:\s+version)?[^0-9]{0,16}([0-9]{4}[.\-][0-9]{1,2}[.\-][0-9]{1,2}|[0-9]+(?:\.[0-9]+)+)", output)
        version = match.group(1) if match else "unknown"
        # Several CLI builds use a non-zero status for their help screen.  The
        # capability evidence is the bounded help output itself; timeout/spawn
        # failure or missing required switches still disables execution.
        process_ready = not outcome.timed_out and outcome.spawn_error is None and not missing
        # Without a star database ASTAP cannot solve anything (it exits with
        # code 32); saying so here gives the doctor and the GUI a setup
        # instruction instead of a failed run.
        star_database = discover_astap_star_database(runtime.executable, configured=star_database_dir)
        ready = process_ready and bool(star_database["found"])
        if ready:
            error_code = None
            message = None
        elif outcome.timed_out:
            error_code, message = "PROBE_TIMEOUT", "ASTAP probe failed"
        elif outcome.spawn_error is not None:
            error_code, message = "CAPABILITY_PROBE_FAILED", "ASTAP probe failed"
        elif missing:
            error_code, message = "CAPABILITY_PROBE_FAILED", f"ASTAP help is missing: {', '.join(missing)}"
        else:
            error_code = "STAR_DATABASE_MISSING"
            message = (
                "ASTAP star database (for example the D20, D50 or G05 files) was not found "
                "beside the ASTAP executable"
            )
        return ExecutableProbe(
            path=runtime.executable,
            available=True,
            execution_ready=ready,
            version=version,
            capabilities=_CAPABILITIES if ready else (),
            error_code=error_code,
            message=message,
            evidence={
                "process": outcome.serializable(),
                "executables": [item.serializable(expose_path=False) for item in identities],
                "requiredOptions": list(required),
                "missingOptions": list(missing),
                "optionalOptions": list(optional),
                "supportedOptionalOptions": supported_optional,
                "starDatabase": star_database,
            },
        )
    except (OSError, ValueError, SolverExecutionError) as error:
        code = error.code if isinstance(error, SolverExecutionError) else "CAPABILITY_PROBE_FAILED"
        return ExecutableProbe(runtime.executable, True, False, "unknown", error_code=code, message=str(error))


def probe_astap(
    executable: str | os.PathLike[str] | None = None,
    *,
    executable_args: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 5.0,
    star_database_dir: str | os.PathLike[str] | None = None,
) -> ExecutableProbe:
    discovery_environment = platform_services.merged_environment(
        os.environ, environment, platform_id=platform_services.current().platform_id
    )
    path = discover_astap(executable, environment=discovery_environment)
    if path is None:
        return ExecutableProbe(None, False, False, "unavailable", error_code="EXECUTABLE_UNAVAILABLE", message="ASTAP was not found")
    try:
        runtime = SolverProcessRuntime(path, executable_args=executable_args, environment=environment)
    except (ValueError, SolverExecutionError) as error:
        return ExecutableProbe(path, False, False, "unavailable", error_code="EXECUTABLE_UNAVAILABLE", message=str(error))
    return _probe_astap_runtime(runtime, timeout_seconds, star_database_dir=star_database_dir)


class AstapSolverBackend:
    backend_id = "astap"

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        *,
        executable_args: Sequence[str] = (),
        config_path: str | os.PathLike[str] | None = None,
        catalog_manifest_dir: str | os.PathLike[str] | None = None,
        require_managed_catalog: bool = True,
        star_database_dir: str | os.PathLike[str] | None = None,
        sip_polynomial: bool = True,
        correspondence_parameters: CorrespondenceParameters | None = None,
        timeout_seconds: float = 120.0,
        probe_timeout_seconds: float = 5.0,
        staging_root: str | os.PathLike[str] | None = None,
        diagnostic_log_root: str | os.PathLike[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        # Imported here because the astrometry.net adapter imports this module
        # for the shared process runtime; the managed-config discovery it
        # hosts is catalog logic, not solve-field logic.
        from .astrometry_net_backend import discover_astrometry_config

        self.timeout_seconds = float(timeout_seconds)
        self.staging_root = Path(staging_root).expanduser() if staging_root is not None else None
        self.environment = dict(environment or {})
        if not isinstance(require_managed_catalog, bool):
            raise ValueError("require_managed_catalog must be a boolean")
        self.require_managed_catalog = require_managed_catalog
        self.catalog_manifest_dir = catalog_manifest_dir
        self.correspondence_parameters = correspondence_parameters or CorrespondenceParameters()
        self.correspondence_parameters.validate()
        self.star_database_dir = (
            Path(star_database_dir).expanduser() if star_database_dir is not None else None
        )
        self.sip_polynomial = bool(sip_polynomial)
        merged_environment = platform_services.merged_environment(
            os.environ, self.environment, platform_id=platform_services.current().platform_id
        )
        discovered_config = discover_astrometry_config(config_path, environment=merged_environment)
        self.config_path = Path(discovered_config) if discovered_config is not None else None
        path = discover_astap(executable, environment=merged_environment)
        self.runtime: SolverProcessRuntime | None = None
        if path is None:
            self.probe = ExecutableProbe(None, False, False, "unavailable", error_code="EXECUTABLE_UNAVAILABLE", message="ASTAP was not found")
        else:
            try:
                self.runtime = SolverProcessRuntime(
                    path,
                    executable_args=executable_args,
                    environment=self.environment,
                    diagnostic_log_root=diagnostic_log_root,
                )
                self.probe = _probe_astap_runtime(
                    self.runtime, probe_timeout_seconds, star_database_dir=self.star_database_dir
                )
            except (ValueError, SolverExecutionError) as error:
                self.probe = ExecutableProbe(path, False, False, "unavailable", error_code="EXECUTABLE_UNAVAILABLE", message=str(error))

    @property
    def sip_requested(self) -> bool:
        """Whether solves ask ASTAP for SIP terms (policy on, CLI supports it)."""

        return self.sip_polynomial and "-sip" in self.probe.evidence.get("supportedOptionalOptions", ())

    @property
    def descriptor(self) -> BackendDescriptor:
        star_database = self.probe.evidence.get("starDatabase", {})
        return BackendDescriptor(
            backend_id=self.backend_id,
            stage=StageKind.SOLVER,
            display_name="ASTAP",
            version=self.probe.version,
            available=self.probe.available,
            execution_ready=self.probe.execution_ready,
            devices=(DeviceKind.CPU,),
            capabilities=self.probe.capabilities,
            reason=self.probe.message,
            metadata={
                "adapterVersion": ADAPTER_VERSION,
                "probe": self.probe.serializable(),
                "runtimePrerequisites": {
                    "starDatabase": "required; exact sky/scale coverage is validated by the solve result",
                    "starDatabaseDiscovered": bool(star_database.get("found", False)),
                    "starDatabaseFamilies": list(star_database.get("families", [])),
                    "indexFiles": (
                        "required; managed Astrometry.net index stars verify every solution "
                        "and exact coverage is validated by the solve result"
                    ),
                    "configDiscovered": self.config_path is not None,
                    "configIdentityValidatedAtSolve": self.config_path is not None,
                    "managedCatalogRequired": self.require_managed_catalog,
                },
                "scientificQualityEvidence": {
                    "status": "recomputed",
                    "strictE2EBehavior": "final-gate",
                    "method": VERIFICATION_METHOD,
                    "correspondenceArtifact": CORRESPONDENCE_ARTIFACT_NAME,
                    "indexIdentityArtifact": MATCH_ARTIFACT_NAME,
                    "detection": self.correspondence_parameters.serializable(),
                },
                "solveOptions": {
                    "sipPolynomial": self.sip_polynomial,
                    "sipRequested": self.sip_requested,
                    "starDatabaseDirectoryConfigured": self.star_database_dir is not None,
                },
            },
        )

    def validate_options(self, options: dict[str, Any]) -> tuple[str, ...]:
        allowed = {"raHintDegrees", "decHintDegrees", "fieldOfViewDegrees", "searchRadiusDegrees"}
        return tuple(f"unknown solver option: {key}" for key in sorted(set(options) - allowed))

    def _merged_environment(self) -> dict[str, str]:
        return platform_services.merged_environment(
            os.environ, self.environment, platform_id=platform_services.current().platform_id
        )

    def _catalog_preflight(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Snapshot the managed installed set before ASTAP runs.

        The same contract as the astrometry.net adapter: strict mode refuses
        to solve without the catalog-manager config, and the snapshot taken
        here is what the post-solve binding compares against.
        """

        if self.config_path is None:
            if self.require_managed_catalog:
                raise SolverExecutionError(
                    "CATALOG_CONFIG_UNMANAGED",
                    "strict ASTAP solving requires an Ultra-Fast WBPP managed catalog config",
                )
            return None, {"managed": False, "reason": "NO_MANAGED_CONFIG"}
        try:
            snapshot = installed_set_snapshot_for_solver_config(
                self.config_path,
                manifest_dir=self.catalog_manifest_dir,
                environment=self._merged_environment(),
            )
        except CatalogError as error:
            if self.require_managed_catalog:
                raise SolverExecutionError(error.code, str(error)) from error
            return None, {"managed": False, "reason": error.code}
        preflight = {
            "managed": True,
            "config": {
                "relativeName": "astrometry.cfg",
                "sizeBytes": snapshot["config"]["sizeBytes"],
                "sha256": snapshot["config"]["sha256"],
            },
            "installedSetIdentities": sorted(item["installedSetIdentity"] for item in snapshot["receipts"]),
            "verifiedIndexCount": len(snapshot["artifacts"]),
        }
        return snapshot, preflight

    def _verify_against_catalog(
        self,
        staged_input: Path,
        wcs_header: fits.Header,
        image_shape: tuple[int, int],
        catalog_snapshot: Mapping[str, Any],
        stage: Path,
    ) -> tuple[AstrometricQuality, dict[str, Any], Any]:
        """Recompute match/RMS evidence from the managed index stars and bind it."""

        try:
            with fits.open(staged_input, mode="readonly", memmap=False) as hdul:
                image = np.asarray(hdul[0].data)
        except Exception as error:
            raise SolverExecutionError(
                "CORRESPONDENCE_INPUT_UNREADABLE", f"cannot read the staged image pixels: {error}"
            ) from error
        if image.ndim > 2:
            # A degenerate leading axis (NAXIS3 = 1) is still a mono image;
            # genuine colour cubes are refused by the detector below.
            image = np.squeeze(image)
        if image.ndim != 2 or tuple(image.shape) != tuple(image_shape):
            raise SolverExecutionError(
                "CORRESPONDENCE_INPUT_UNSUPPORTED",
                "catalog verification needs a two-dimensional image matching the FITS geometry",
            )
        try:
            verification = verify_solution(
                image=image,
                wcs_header=wcs_header,
                image_shape=image_shape,
                catalog_root=Path(str(catalog_snapshot["catalogRoot"])),
                index_artifacts=catalog_snapshot["artifacts"],
                artifact_dir=stage,
                parameters=self.correspondence_parameters,
            )
        except CatalogCorrespondenceError as error:
            raise SolverExecutionError(error.code, str(error)) from error
        try:
            quality, catalog_binding = bind_installed_set(
                verification.quality,
                catalog_snapshot,
                manifest_dir=self.catalog_manifest_dir,
                environment=self._merged_environment(),
            )
        except CatalogError as error:
            if self.require_managed_catalog:
                raise SolverExecutionError(error.code, str(error)) from error
            quality = verification.quality
            catalog_binding = {"catalogManaged": False, "reason": error.code}
        return quality, catalog_binding, verification

    def solve(self, request: SolveRequest) -> SolverResult:
        if self.runtime is None or not self.probe.execution_ready:
            return _failure(
                self.backend_id,
                self.probe.error_code or "EXECUTABLE_UNAVAILABLE",
                self.probe.message or "ASTAP is not execution-ready",
                status=SolverStatus.UNAVAILABLE,
                evidence={"adapterVersion": ADAPTER_VERSION, "probe": self.probe.serializable()},
            )
        try:
            _validate_request(request)
            if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
                raise SolverExecutionError("TIMEOUT_INVALID", "timeout must be finite and positive")
            input_path = Path(request.input_path).expanduser().resolve(strict=True)
            output_path = Path(request.output_path).expanduser().resolve(strict=False)
            catalog_snapshot, catalog_preflight = self._catalog_preflight()
            with tempfile.TemporaryDirectory(
                prefix="ultra-fast-wbpp-astap-", dir=self.staging_root
            ) as raw_stage:
                stage = Path(raw_stage)
                try:
                    stage.chmod(0o700)
                except OSError:
                    pass
                staged_input = stage / f"input{input_path.suffix or '.fits'}"
                source_identity, staged_identity = _copy_source_to_stage(input_path, staged_input)
                image_shape = _input_shape(staged_input)
                config_identity = (
                    _regular_identity(self.config_path, max_bytes=_MAX_CONFIG_BYTES)
                    if self.config_path is not None
                    else None
                )
                output_base = stage / "solution"
                arguments: list[str] = ["-f", str(staged_input), "-o", str(output_base), "-wcs"]
                if self.sip_requested:
                    arguments.append("-sip")
                if self.star_database_dir is not None:
                    arguments.extend(("-d", str(self.star_database_dir)))
                if request.ra_hint_degrees is not None and request.dec_hint_degrees is not None:
                    arguments.extend(("-ra", f"{request.ra_hint_degrees / 15.0:.12g}", "-spd", f"{request.dec_hint_degrees + 90.0:.12g}"))
                if request.field_of_view_degrees is not None:
                    # The engine's hint is the field width; ASTAP's -fov is the
                    # field height, and a hint off by the aspect ratio (1.5 for
                    # a 3:2 sensor) makes it give up with "no solution".
                    height_pixels, width_pixels = image_shape[0], image_shape[1]
                    field_height = request.field_of_view_degrees * max(height_pixels - 1, 1) / max(width_pixels - 1, 1)
                    arguments.extend(("-fov", f"{field_height:.12g}"))
                if request.search_radius_degrees is not None:
                    arguments.extend(("-r", f"{request.search_radius_degrees:.12g}"))
                outcome, executable_identities = self.runtime.run(
                    arguments,
                    cwd=stage,
                    timeout_seconds=self.timeout_seconds,
                    log_stem="solve",
                )
                base_evidence = {
                    "adapterVersion": ADAPTER_VERSION,
                    "probe": self.probe.serializable(),
                    "source": source_identity.serializable(expose_path=False),
                    "stagedInput": staged_identity.serializable(expose_path=False),
                    "process": outcome.serializable(),
                    "executables": [item.serializable(expose_path=False) for item in executable_identities],
                    "environmentKeys": sorted(self.runtime.environment),
                    "environmentSha256": f"sha256:{hashlib.sha256(json.dumps(self.runtime.environment, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}",
                    "config": (
                        config_identity.serializable(expose_path=False)
                        if config_identity is not None
                        else None
                    ),
                    "catalogPreflight": catalog_preflight,
                    "solveOptions": {
                        "sipRequested": self.sip_requested,
                        "starDatabaseDirectoryConfigured": self.star_database_dir is not None,
                    },
                }
                if outcome.timed_out:
                    return _failure(self.backend_id, "SOLVER_TIMEOUT", "ASTAP exceeded its execution timeout", evidence=base_evidence)
                if outcome.spawn_error:
                    return _failure(self.backend_id, "SOLVER_SPAWN_FAILED", outcome.spawn_error, evidence=base_evidence)
                if outcome.exit_code != 0:
                    return _failure(self.backend_id, "SOLVER_EXIT_NONZERO", f"ASTAP exited with code {outcome.exit_code}", evidence=base_evidence)
                if not _same_stat(input_path, source_identity):
                    return _failure(self.backend_id, "SOURCE_DRIFT", "source input changed during solving", evidence=base_evidence)
                staged_after = _regular_identity(staged_input)
                if staged_after.sha256 != staged_identity.sha256 or staged_after.size_bytes != staged_identity.size_bytes:
                    return _failure(self.backend_id, "STAGED_INPUT_DRIFT", "ASTAP modified its staged input without authorization", evidence=base_evidence)
                if config_identity is not None:
                    _require_unchanged_identity(self.config_path, config_identity, max_bytes=_MAX_CONFIG_BYTES)

                try:
                    ini_text, ini_identity = _read_control_text(output_base.with_suffix(".ini"))
                except SolverExecutionError as error:
                    return _failure(self.backend_id, error.code, str(error), evidence=base_evidence)
                outputs: dict[str, Any] = {"ini": ini_identity.serializable(expose_path=False)}
                marker = re.search(r"(?im)^\s*PLTSOLVD\s*=\s*([^\s/;]+)", ini_text)
                if marker is None or marker.group(1).strip().upper() not in {"T", "TRUE", "1", "Y", "YES"}:
                    return _failure(
                        self.backend_id,
                        "BACKEND_CONFIRMATION_MISSING",
                        "ASTAP did not emit PLTSOLVD=T; seed coordinates are not a solution",
                        evidence={**base_evidence, "outputs": outputs},
                    )
                try:
                    wcs_header, wcs_identity = _read_wcs_header(output_base.with_suffix(".wcs"))
                except SolverExecutionError as error:
                    return _failure(self.backend_id, error.code, str(error), evidence={**base_evidence, "outputs": outputs})
                outputs["wcs"] = wcs_identity.serializable(expose_path=False)
                validation = validate_wcs_header(wcs_header, image_shape=image_shape)
                if not validation.valid:
                    return _failure(
                        self.backend_id,
                        "WCS_VALIDATION_FAILED",
                        f"{validation.code}: {validation.message}",
                        evidence={**base_evidence, "outputs": outputs, "wcsValidation": validation.serializable()},
                    )
                astrometric_quality: AstrometricQuality | None = None
                quality_evidence: dict[str, Any] = {
                    "status": "UNAVAILABLE",
                    "reason": (
                        "no managed Astrometry.net index set is bound to this solve, so the "
                        "engine could not recompute catalog correspondences; diagnostic only"
                    ),
                }
                correspondence_diagnostics: dict[str, Any] | None = None
                match_diagnostics: dict[str, Any] | None = None
                catalog_binding: dict[str, Any] | None = None
                if catalog_snapshot is not None:
                    # A valid WCS is not proof of the right sky position; the
                    # catalog check runs before anything is published so a
                    # failure leaves no output behind.
                    try:
                        astrometric_quality, catalog_binding, verification = self._verify_against_catalog(
                            staged_input, wcs_header, image_shape, catalog_snapshot, stage
                        )
                    except SolverExecutionError as error:
                        return _failure(
                            self.backend_id,
                            error.code,
                            str(error),
                            evidence={**base_evidence, "outputs": outputs, "wcsValidation": validation.serializable()},
                        )
                    outputs["corr"] = _regular_identity(verification.correspondence_path).serializable(expose_path=False)
                    outputs["match"] = _regular_identity(verification.match_path).serializable(expose_path=False)
                    quality_evidence = astrometric_quality.serializable()
                    correspondence_diagnostics = verification.correspondence_diagnostics
                    match_diagnostics = verification.match_diagnostics
                published = _publish_solved_copy(staged_input, output_path, wcs_header)
                evidence = _receipt(
                    {
                        "receiptVersion": 2,
                        "backendId": self.backend_id,
                        "adapterVersion": ADAPTER_VERSION,
                        "outcome": "SOLVED",
                        "probe": self.probe.serializable(),
                        "source": source_identity.serializable(expose_path=False),
                        "stagedInput": staged_identity.serializable(expose_path=False),
                        "process": outcome.serializable(),
                        "executables": [item.serializable(expose_path=False) for item in executable_identities],
                        "environmentKeys": sorted(self.runtime.environment),
                        "environmentSha256": f"sha256:{hashlib.sha256(json.dumps(self.runtime.environment, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}",
                        "config": base_evidence["config"],
                        "catalogPreflight": catalog_preflight,
                        "solveOptions": base_evidence["solveOptions"],
                        "outputs": {**outputs, "published": published.serializable(expose_path=False)},
                        "wcsValidation": validation.serializable(),
                        "solutionWcsSha256": f"sha256:{_wcs_header_sha256(wcs_header)}",
                        "backendConfirmation": "PLTSOLVD=T",
                        "astrometricQuality": quality_evidence,
                        "correspondenceDiagnostics": correspondence_diagnostics,
                        "matchDiagnostics": match_diagnostics,
                        "catalogBinding": catalog_binding,
                        "verification": {
                            "method": VERIFICATION_METHOD if catalog_snapshot is not None else None,
                            "correspondenceArtifact": CORRESPONDENCE_ARTIFACT_NAME if catalog_snapshot is not None else None,
                            "indexIdentityArtifact": MATCH_ARTIFACT_NAME if catalog_snapshot is not None else None,
                        },
                    }
                )
                return SolverResult(
                    backend_id=self.backend_id,
                    status=SolverStatus.SOLVED,
                    solution_kind=SolutionKind.SOLVED,
                    backend_confirmed=True,
                    header=wcs_header,
                    image_shape=image_shape,
                    output_path=str(output_path),
                    astrometric_quality=astrometric_quality,
                    evidence=evidence,
                )
        except SolverExecutionError as error:
            return _failure(self.backend_id, error.code, str(error), evidence={"adapterVersion": ADAPTER_VERSION, "probe": self.probe.serializable()})
        except Exception as error:
            return _failure(self.backend_id, "SOLVER_EXECUTION_FAILED", f"{type(error).__name__}: {error}", evidence={"adapterVersion": ADAPTER_VERSION, "probe": self.probe.serializable()})


ASTAPSolverBackend = AstapSolverBackend
ASTAPBackend = AstapSolverBackend


__all__ = [
    "ADAPTER_VERSION",
    "ASTAPSolverBackend",
    "ASTAPBackend",
    "AstapSolverBackend",
    "ExecutableProbe",
    "SolverExecutionError",
    "SolverProcessRuntime",
    "discover_astap",
    "discover_astap_star_database",
    "probe_astap",
    "verify_execution_receipt",
    "verify_solver_execution_result",
]
