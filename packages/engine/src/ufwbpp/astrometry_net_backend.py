"""Fail-closed local astrometry.net ``solve-field`` process adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from . import platform as platform_services
from .solvers.process import (
    ExecutableProbe,
    SolverExecutionError,
    SolverProcessRuntime,
    copy_source_to_stage as _copy_source_to_stage,
    failure as _failure,
    input_shape as _input_shape,
    is_executable_file as _is_executable_file,
    publish_solved_copy as _publish_solved_copy,
    require_unchanged_identity as _require_unchanged_identity,
    build_execution_receipt as _receipt,
    regular_identity as _regular_identity,
    same_stat as _same_stat,
    validate_request as _validate_request,
    wcs_header_sha256 as _wcs_header_sha256,
)
from .backends import BackendDescriptor, DeviceKind, StageKind
from .catalogs import (
    CatalogError,
    bundled_astrometry_root,
    installed_set_identity_for_solver_indexes,
    installed_set_snapshot_for_solver_config,
    verify_installed_set_snapshot,
)
from .solver import (
    AstrometricQuality,
    SolutionKind,
    SolveRequest,
    SolverResult,
    SolverStatus,
    SolverIndexArtifact,
    validate_wcs_header,
    wcs_parity,
)


ADAPTER_VERSION = "astrometry-net-process-v4"
_CAPABILITIES = (
    "blind-solve",
    "seed-hints",
    "celestial-wcs",
    "fail-closed-wcs-validation",
    "isolated-staging-copy",
    "process-timeout",
    "provenance-receipt-v1",
    "catalog-correspondence-quality-v1",
    "managed-catalog-byte-binding-v1",
    "share-safe-process-evidence-v1",
    "adaptive-source-extraction-v1",
    "bounded-unconstrained-hint-fallback-v1",
    "no-existing-wcs-verification",
)
_MAX_WCS_BYTES = 16 * 1024 * 1024
_MAX_CORRESPONDENCE_BYTES = 256 * 1024 * 1024
_MAX_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AstrometryNetSolveProfile:
    """Bounded source-extraction and retry policy for ``solve-field``.

    The defaults were selected for modern, well-sampled astronomy cameras: a
    large frame is first reduced before source extraction, while an independent
    second attempt uses gentler reduction and removes pointing/scale constraints.
    This keeps a stale N.I.N.A. hint from consuming the complete solver budget.
    These options are deliberately applied only to solve invocations; capability
    probes remain plain ``--version``/``--help`` calls.
    """

    downsample: int | None = None
    source_limit: int = 500
    depth_min: int = 10
    depth_max: int = 500
    pixel_error: float = 2.0
    scale_tolerance_fraction: float = 0.20
    hinted_attempt_fraction: float = 0.25
    hinted_attempt_cap_seconds: float = 30.0
    minimum_fallback_seconds: float = 3.0
    unconstrained_fallback: bool = True
    fallback_downsample_divisor: int = 2
    fallback_source_limit: int = 1000
    fallback_depth_min: int = 1
    fallback_depth_max: int = 1000

    def validate(self) -> None:
        integers = {
            "downsample": self.downsample,
            "source_limit": self.source_limit,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "fallback_downsample_divisor": self.fallback_downsample_divisor,
            "fallback_source_limit": self.fallback_source_limit,
            "fallback_depth_min": self.fallback_depth_min,
            "fallback_depth_max": self.fallback_depth_max,
        }
        for name, value in integers.items():
            if value is None and name == "downsample":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.depth_min > self.depth_max:
            raise ValueError("depth_min must be no greater than depth_max")
        if self.fallback_depth_min > self.fallback_depth_max:
            raise ValueError("fallback_depth_min must be no greater than fallback_depth_max")
        finite_positive = {
            "pixel_error": self.pixel_error,
            "hinted_attempt_cap_seconds": self.hinted_attempt_cap_seconds,
            "minimum_fallback_seconds": self.minimum_fallback_seconds,
        }
        for name, value in finite_positive.items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.scale_tolerance_fraction, bool)
            or not math.isfinite(self.scale_tolerance_fraction)
            or not 0.05 <= self.scale_tolerance_fraction <= 1.0
        ):
            raise ValueError("scale_tolerance_fraction must be in [0.05, 1.0]")
        if (
            isinstance(self.hinted_attempt_fraction, bool)
            or not math.isfinite(self.hinted_attempt_fraction)
            or not 0.05 <= self.hinted_attempt_fraction <= 0.95
        ):
            raise ValueError("hinted_attempt_fraction must be in [0.05, 0.95]")

    def adaptive_downsample(self, image_shape: tuple[int, int]) -> int:
        if self.downsample is not None:
            return self.downsample
        largest_axis = max(image_shape)
        if largest_axis >= 5000:
            return 4
        if largest_axis >= 2500:
            return 2
        return 1

    def serializable(self, image_shape: tuple[int, int]) -> dict[str, Any]:
        return {
            "profileVersion": 1,
            "configuredDownsample": self.downsample,
            "adaptiveDownsample": self.adaptive_downsample(image_shape),
            "sourceLimit": self.source_limit,
            "depth": [self.depth_min, self.depth_max],
            "pixelError": self.pixel_error,
            "scaleToleranceFraction": self.scale_tolerance_fraction,
            "hintedAttemptFraction": self.hinted_attempt_fraction,
            "hintedAttemptCapSeconds": self.hinted_attempt_cap_seconds,
            "minimumFallbackSeconds": self.minimum_fallback_seconds,
            "unconstrainedFallback": self.unconstrained_fallback,
            "fallbackDownsampleDivisor": self.fallback_downsample_divisor,
            "fallbackSourceLimit": self.fallback_source_limit,
            "fallbackDepth": [self.fallback_depth_min, self.fallback_depth_max],
        }


@dataclass(frozen=True, slots=True)
class _SolveAttempt:
    name: str
    timeout_seconds: float
    downsample: int
    source_limit: int
    depth_min: int
    depth_max: int
    include_hints: bool

    def serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timeoutSeconds": self.timeout_seconds,
            "downsample": self.downsample,
            "sourceLimit": self.source_limit,
            "depth": [self.depth_min, self.depth_max],
            "includeHints": self.include_hints,
        }


def _candidate_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not _is_executable_file(path):
        return None
    return str(path.absolute())


def discover_astrometry_net(
    executable: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Find the local ``solve-field`` frontend without invoking it."""

    if executable is not None:
        return _candidate_path(executable)
    import shutil

    env = platform_services.environment_view(
        os.environ if environment is None else environment,
        platform_id=platform_services.current().platform_id,
    )
    for key in ("UFWBPP_SOLVE_FIELD", "ASTROMETRY_NET_SOLVE_FIELD"):
        candidate = _candidate_path(env.get(key))
        if candidate:
            return candidate
    # A self-contained build's pinned solver wins over whatever the host has.
    bundled = bundled_astrometry_root(environment=env)
    if bundled is not None:
        candidate = _candidate_path(bundled / "bin" / "solve-field")
        if candidate:
            return candidate
    for name in ("solve-field", "solve-field.exe"):
        candidate = _candidate_path(shutil.which(name, path=env.get("PATH")))
        if candidate:
            return candidate
    candidates = platform_services.current().well_known_executables("solve-field", environment=env)
    for path in candidates:
        candidate = _candidate_path(path)
        if candidate:
            return candidate
    return None


def discover_astrometry_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Find an app/user-managed Astrometry.net config without invoking tools."""

    if config_path is not None:
        return str(Path(config_path).expanduser().absolute())
    env = platform_services.environment_view(
        os.environ if environment is None else environment,
        platform_id=platform_services.current().platform_id,
    )
    candidates: list[Path] = []
    for key in ("UFWBPP_ASTROMETRY_CONFIG", "ASTROMETRY_NET_CONFIG"):
        value = env.get(key)
        if value:
            candidates.append(Path(value).expanduser())
    # The managed catalog lives under the data root (see ``platform``); only
    # locations named by this environment are consulted, never the caller's.
    roots: list[Path] = []
    configured_root = env.get("UFWBPP_DATA_DIR")
    if configured_root:
        roots.append(Path(configured_root).expanduser())
    local_app_data = env.get("LOCALAPPDATA")
    if local_app_data:
        roots.append(
            platform_services.resolve_data_root(
                Path(local_app_data) / "Ultra-Fast-WBPP", Path(local_app_data) / "OpenAstroFlow"
            )
        )
    home = env.get("HOME")
    if home:
        roots.append(
            platform_services.resolve_data_root(Path(home) / ".ultra-fast-wbpp", Path(home) / ".openastroflow")
        )
    candidates.extend(root / "catalogs" / "astrometry-net" / "astrometry.cfg" for root in roots)
    for candidate in candidates:
        try:
            metadata = candidate.lstat()
            if candidate.is_file() and not candidate.is_symlink() and metadata.st_size > 0:
                return str(candidate.absolute())
        except OSError:
            continue
    return None


def _probe_astrometry_runtime(runtime: SolverProcessRuntime, timeout_seconds: float) -> ExecutableProbe:
    try:
        with tempfile.TemporaryDirectory(prefix="ultra-fast-wbpp-astrometry-probe-") as raw:
            stage = Path(raw)
            version_outcome, version_executables = runtime.run(
                ("--version",), cwd=stage, timeout_seconds=timeout_seconds, log_stem="version"
            )
            help_outcome, help_executables = runtime.run(
                ("--help",), cwd=stage, timeout_seconds=timeout_seconds, log_stem="help"
            )
        version_output = f"{version_outcome.stdout_tail}\n{version_outcome.stderr_tail}"
        help_output = f"{help_outcome.stdout_tail}\n{help_outcome.stderr_tail}"
        required = (
            "--dir",
            "--out",
            "--wcs",
            "--new-fits",
            "--solved",
            "--corr",
            "--match",
            "--no-verify",
            "--no-plots",
            "--config",
            "--downsample",
            "--objs",
            "--depth",
            "--pixel-error",
        )
        missing = tuple(option for option in required if option not in help_output)
        combined_version_output = f"{version_output}\n{help_output}"
        match = re.search(r"(?i)(?:astrometry(?:\.net)?|revision|version)[^0-9]{0,20}([0-9]+(?:\.[0-9]+)+(?:[-+._a-z0-9]*)?)", combined_version_output)
        if match is None:
            match = re.search(r"\b([0-9]+(?:\.[0-9]+){1,3})\b", combined_version_output)
        version = match.group(1) if match else "unknown"
        # Upstream solve-field historically lacked --version.  A failed version
        # query is therefore recorded but does not override a successful,
        # capability-complete help probe.
        ready = not help_outcome.timed_out and help_outcome.spawn_error is None and not missing
        timed_out = version_outcome.timed_out or help_outcome.timed_out
        return ExecutableProbe(
            path=runtime.executable,
            available=True,
            execution_ready=ready,
            version=version,
            capabilities=_CAPABILITIES if ready else (),
            error_code=None if ready else ("PROBE_TIMEOUT" if timed_out else "CAPABILITY_PROBE_FAILED"),
            message=None if ready else (f"solve-field help is missing: {', '.join(missing)}" if missing else "solve-field probe failed"),
            evidence={
                "versionProcess": version_outcome.serializable(),
                "helpProcess": help_outcome.serializable(),
                "executables": [item.serializable(expose_path=False) for item in (*version_executables, *help_executables)],
                "requiredOptions": list(required),
                "missingOptions": list(missing),
            },
        )
    except (OSError, ValueError, SolverExecutionError) as error:
        code = error.code if isinstance(error, SolverExecutionError) else "CAPABILITY_PROBE_FAILED"
        return ExecutableProbe(runtime.executable, True, False, "unknown", error_code=code, message=str(error))


def probe_astrometry_net(
    executable: str | os.PathLike[str] | None = None,
    *,
    executable_args: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 5.0,
) -> ExecutableProbe:
    discovery_environment = platform_services.merged_environment(
        os.environ, environment, platform_id=platform_services.current().platform_id
    )
    path = discover_astrometry_net(executable, environment=discovery_environment)
    if path is None:
        return ExecutableProbe(None, False, False, "unavailable", error_code="EXECUTABLE_UNAVAILABLE", message="solve-field was not found")
    try:
        runtime = SolverProcessRuntime(path, executable_args=executable_args, environment=environment)
    except (ValueError, SolverExecutionError) as error:
        return ExecutableProbe(path, False, False, "unavailable", error_code="EXECUTABLE_UNAVAILABLE", message=str(error))
    return _probe_astrometry_runtime(runtime, timeout_seconds)


def _solve_attempts(
    profile: AstrometryNetSolveProfile,
    request: SolveRequest,
    image_shape: tuple[int, int],
    total_timeout_seconds: float,
) -> tuple[_SolveAttempt, ...]:
    """Allocate a hard total timeout across hinted and blind attempts."""

    downsample = profile.adaptive_downsample(image_shape)
    has_hints = (
        request.field_of_view_degrees is not None
        or (request.ra_hint_degrees is not None and request.dec_hint_degrees is not None)
    )
    if not has_hints or not profile.unconstrained_fallback:
        return (
            _SolveAttempt(
                "adaptive-unconstrained" if not has_hints else "adaptive-hinted",
                total_timeout_seconds,
                downsample,
                profile.source_limit,
                profile.depth_min,
                profile.depth_max,
                has_hints,
            ),
        )

    hinted_budget = min(
        profile.hinted_attempt_cap_seconds,
        total_timeout_seconds * profile.hinted_attempt_fraction,
    )
    # A very small caller budget cannot safely support two process starts.  In
    # that case preserve fail-closed timeout behavior and run one constrained
    # attempt instead of pretending that a fallback was attempted.
    if total_timeout_seconds - hinted_budget < profile.minimum_fallback_seconds:
        return (
            _SolveAttempt(
                "adaptive-hinted",
                total_timeout_seconds,
                downsample,
                profile.source_limit,
                profile.depth_min,
                profile.depth_max,
                True,
            ),
        )
    fallback_downsample = max(1, downsample // profile.fallback_downsample_divisor)
    return (
        _SolveAttempt(
            "adaptive-hinted",
            hinted_budget,
            downsample,
            profile.source_limit,
            profile.depth_min,
            profile.depth_max,
            True,
        ),
        _SolveAttempt(
            "conservative-unconstrained-fallback",
            total_timeout_seconds - hinted_budget,
            fallback_downsample,
            profile.fallback_source_limit,
            profile.fallback_depth_min,
            profile.fallback_depth_max,
            False,
        ),
    )


def _read_marker(path: Path) -> tuple[bytes, Any]:
    identity = _regular_identity(path, max_bytes=16)
    try:
        value = path.read_bytes()
    except OSError as error:
        raise SolverExecutionError("SOLVED_MARKER_INVALID", str(error)) from error
    _require_unchanged_identity(path, identity, max_bytes=16)
    return value, identity


def _read_header_artifact(path: Path) -> tuple[fits.Header, Any]:
    identity = _regular_identity(path, max_bytes=_MAX_WCS_BYTES)
    errors: list[str] = []
    for loader in (
        lambda: fits.Header.fromfile(path, sep="", endcard=True, padding=True),
        lambda: fits.Header.fromstring(path.read_bytes().decode("ascii"), sep=""),
        lambda: fits.Header.fromtextfile(path),
        lambda: fits.getheader(path, ext=0, memmap=False),
    ):
        try:
            header = loader()
            _require_unchanged_identity(path, identity, max_bytes=_MAX_WCS_BYTES)
            return header, identity
        except SolverExecutionError:
            raise
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    raise SolverExecutionError("WCS_ARTIFACT_INVALID", "; ".join(errors[-2:]))


def _read_new_fits(path: Path) -> tuple[fits.Header, tuple[int, int], Any]:
    identity = _regular_identity(path)
    try:
        header = fits.getheader(path, ext=0, memmap=False)
        shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    except Exception as error:
        raise SolverExecutionError("NEW_FITS_INVALID", str(error)) from error
    _require_unchanged_identity(path, identity)
    return header, shape, identity


def _column(data: Any, name: str) -> np.ndarray | None:
    names = getattr(getattr(data, "columns", None), "names", None) or ()
    actual = next((item for item in names if str(item).lower() == name.lower()), None)
    if actual is None:
        return None
    return np.asarray(data[actual])


def _stable_scalar(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").strip()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SolverExecutionError("CORRESPONDENCE_INVALID", "non-finite correspondence identifier")
        return format(value, ".17g")
    return str(value).strip()


def _angular_separation_arcsec(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_ra = np.deg2rad(left[:, 0])
    left_dec = np.deg2rad(left[:, 1])
    right_ra = np.deg2rad(right[:, 0])
    right_dec = np.deg2rad(right[:, 1])
    left_vectors = np.column_stack(
        (np.cos(left_dec) * np.cos(left_ra), np.cos(left_dec) * np.sin(left_ra), np.sin(left_dec))
    )
    right_vectors = np.column_stack(
        (np.cos(right_dec) * np.cos(right_ra), np.cos(right_dec) * np.sin(right_ra), np.sin(right_dec))
    )
    chord = np.linalg.norm(left_vectors - right_vectors, axis=1)
    return np.rad2deg(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))) * 3600.0


def _read_correspondence_quality(
    path: Path,
    wcs_header: fits.Header,
    image_shape: tuple[int, int],
    index_identities: tuple[str, ...],
    expected_correspondence_count: int,
) -> tuple[AstrometricQuality, Any, dict[str, Any]]:
    """Measure unique source/catalog residuals from solve-field's .corr table."""

    identity = _regular_identity(path, max_bytes=_MAX_CORRESPONDENCE_BYTES)
    try:
        with fits.open(path, mode="readonly", memmap=False, checksum=True) as hdul:
            table = next(
                (
                    hdu
                    for hdu in hdul
                    if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
                    and hdu.data is not None
                    and len(hdu.data) > 0
                ),
                None,
            )
            if table is None or table.data is None:
                raise SolverExecutionError(
                    "CORRESPONDENCE_TABLE_MISSING",
                    "solve-field did not emit a non-empty FITS correspondence table",
                )
            data = table.data
            field_x = _column(data, "field_x")
            field_y = _column(data, "field_y")
            index_ra = _column(data, "index_ra")
            index_dec = _column(data, "index_dec")
            field_id = _column(data, "field_id")
            index_id = _column(data, "index_id")
            if any(item is None for item in (field_x, field_y, index_ra, index_dec, index_id)):
                raise SolverExecutionError(
                    "CORRESPONDENCE_COLUMNS_MISSING",
                    "the .corr table must contain field_x, field_y, index_ra, index_dec, and index_id",
                )
            arrays = [np.ravel(np.asarray(item)) for item in (field_x, field_y, index_ra, index_dec, index_id)]
            if field_id is not None:
                arrays.append(np.ravel(np.asarray(field_id)))
            lengths = {len(item) for item in arrays}
            if len(lengths) != 1:
                raise SolverExecutionError("CORRESPONDENCE_INVALID", "correspondence columns have different lengths")
            field_x_values = np.asarray(arrays[0], dtype=np.float64)
            field_y_values = np.asarray(arrays[1], dtype=np.float64)
            index_ra_values = np.asarray(arrays[2], dtype=np.float64)
            index_dec_values = np.asarray(arrays[3], dtype=np.float64)
            index_id_values = arrays[4]
            field_id_values = arrays[5] if field_id is not None else None
    except SolverExecutionError:
        raise
    except Exception as error:
        raise SolverExecutionError("CORRESPONDENCE_INVALID", str(error)) from error
    _require_unchanged_identity(path, identity, max_bytes=_MAX_CORRESPONDENCE_BYTES)

    finite = (
        np.isfinite(field_x_values)
        & np.isfinite(field_y_values)
        & np.isfinite(index_ra_values)
        & np.isfinite(index_dec_values)
        & (index_dec_values >= -90.0)
        & (index_dec_values <= 90.0)
    )
    height, width = image_shape
    # Astrometry.net's FIELD_X/FIELD_Y coordinates use the FITS one-based
    # convention.  Keep a one-pixel tolerance for sources on the boundary.
    finite &= (
        (field_x_values >= 0.0)
        & (field_x_values <= width + 1.0)
        & (field_y_values >= 0.0)
        & (field_y_values <= height + 1.0)
    )
    selected: list[int] = []
    seen_fields: set[str] = set()
    seen_catalog: set[str] = set()
    for index in np.flatnonzero(finite):
        try:
            catalog_key = _stable_scalar(index_id_values[index])
            field_key = (
                _stable_scalar(field_id_values[index])
                if field_id_values is not None
                else f"{field_x_values[index]:.8f},{field_y_values[index]:.8f}"
            )
        except SolverExecutionError:
            continue
        if not catalog_key or not field_key or catalog_key in seen_catalog or field_key in seen_fields:
            continue
        seen_catalog.add(catalog_key)
        seen_fields.add(field_key)
        selected.append(int(index))
    if not selected:
        raise SolverExecutionError(
            "CORRESPONDENCE_UNIQUE_MATCHES_MISSING",
            "the .corr table contains no finite one-to-one image/catalog matches",
        )
    if len(selected) != expected_correspondence_count:
        raise SolverExecutionError(
            "CORRESPONDENCE_MATCH_COUNT_MISMATCH",
            f".corr contains {len(selected)} unique one-to-one matches but .match implies {expected_correspondence_count}",
        )

    chosen = np.asarray(selected, dtype=np.int64)
    field_pixels = np.column_stack((field_x_values[chosen], field_y_values[chosen]))
    catalog_world = np.column_stack((index_ra_values[chosen] % 360.0, index_dec_values[chosen]))
    try:
        celestial = WCS(wcs_header, relax=False).celestial
        predicted_pixels = celestial.all_world2pix(catalog_world, 1)
        field_world = celestial.all_pix2world(field_pixels, 1)
    except Exception as error:
        raise SolverExecutionError("CORRESPONDENCE_WCS_FAILED", str(error)) from error
    if not np.all(np.isfinite(predicted_pixels)) or not np.all(np.isfinite(field_world)):
        raise SolverExecutionError("CORRESPONDENCE_WCS_FAILED", "correspondence transforms are non-finite")
    pixel_residuals = np.linalg.norm(field_pixels - predicted_pixels, axis=1)
    angular_residuals = _angular_separation_arcsec(field_world, catalog_world)
    if not np.all(np.isfinite(pixel_residuals)) or not np.all(np.isfinite(angular_residuals)):
        raise SolverExecutionError("CORRESPONDENCE_RMS_INVALID", "correspondence residuals are non-finite")
    rms_pixels = float(math.sqrt(float(np.mean(np.square(pixel_residuals)))))
    rms_arcsec = float(math.sqrt(float(np.mean(np.square(angular_residuals)))))

    catalog_rows = sorted(
        (
            _stable_scalar(index_id_values[index]),
            format(float(index_ra_values[index] % 360.0), ".17g"),
            format(float(index_dec_values[index]), ".17g"),
        )
        for index in selected
    )
    catalog_payload = json.dumps(
        {"indexes": list(index_identities), "matchedCatalogRows": catalog_rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    catalog_identity = hashlib.sha256(catalog_payload.encode("utf-8")).hexdigest()
    quality = AstrometricQuality(
        matched_stars=len(selected),
        rms_pixels=rms_pixels,
        rms_arcsec=rms_arcsec,
        parity=wcs_parity(wcs_header),
        catalog_identity=catalog_identity,
        index_identities=index_identities,
        correspondence_sha256=identity.sha256,
    )
    return quality, identity, {
        "tableRows": int(len(field_x_values)),
        "finiteRows": int(np.count_nonzero(finite)),
        "uniqueOneToOneMatches": len(selected),
        "coordinateOrigin": 1,
        "uniqueCatalogSourceIds": len(seen_catalog),
        "residualMethod": "WCS(index_ra,index_dec)->pixel vs field_x,field_y; spherical sky RMS",
    }


def _wcs_agrees(left: fits.Header, right: fits.Header, image_shape: tuple[int, int]) -> tuple[bool, float | None]:
    height, width = image_shape
    pixels = np.asarray(
        (
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (0.0, float(height - 1)),
            (float(width - 1), float(height - 1)),
            ((width - 1) / 2.0, (height - 1) / 2.0),
        ),
        dtype=np.float64,
    )
    try:
        left_world = WCS(left, relax=False).celestial.all_pix2world(pixels, 0)
        right_world = WCS(right, relax=False).celestial.all_pix2world(pixels, 0)
        if not np.all(np.isfinite(left_world)) or not np.all(np.isfinite(right_world)):
            return False, None
        left_ra = np.deg2rad(left_world[:, 0])
        left_dec = np.deg2rad(left_world[:, 1])
        right_ra = np.deg2rad(right_world[:, 0])
        right_dec = np.deg2rad(right_world[:, 1])
        left_vectors = np.column_stack(
            (np.cos(left_dec) * np.cos(left_ra), np.cos(left_dec) * np.sin(left_ra), np.sin(left_dec))
        )
        right_vectors = np.column_stack(
            (np.cos(right_dec) * np.cos(right_ra), np.cos(right_dec) * np.sin(right_ra), np.sin(right_dec))
        )
        dots = np.clip(np.sum(left_vectors * right_vectors, axis=1), -1.0, 1.0)
        maximum_arcsec = float(np.rad2deg(np.max(np.arccos(dots))) * 3600.0)
        return maximum_arcsec <= 0.01, maximum_arcsec
    except Exception:
        return False, None


def _read_match_index_identity(
    path: Path,
    wcs_header: fits.Header,
    image_shape: tuple[int, int],
) -> tuple[tuple[str, ...], int, Any, dict[str, Any]]:
    """Bind the accepted WCS to astrometry.net's index ID/healpix tuple."""

    identity = _regular_identity(path, max_bytes=_MAX_CORRESPONDENCE_BYTES)
    linear_header = wcs_header.copy()
    for key in list(linear_header):
        upper = key.upper()
        if re.fullmatch(r"(?:A|B|AP|BP)_(?:ORDER|\d+_\d+)", upper):
            del linear_header[key]
    for key in ("CTYPE1", "CTYPE2"):
        if key in linear_header:
            linear_header[key] = str(linear_header[key]).replace("-SIP", "")
    try:
        with fits.open(path, mode="readonly", memmap=False, checksum=True) as hdul:
            table = next(
                (
                    hdu
                    for hdu in hdul
                    if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
                    and hdu.data is not None
                    and len(hdu.data) > 0
                ),
                None,
            )
            if table is None or table.data is None:
                raise SolverExecutionError("MATCH_TABLE_MISSING", "solve-field emitted no non-empty .match table")
            data = table.data
            required_names = (
                "crval",
                "crpix",
                "cd",
                "wcs_valid",
                "indexid",
                "healpix",
                "hpnside",
                "parity",
                "nmatch",
                "dimquads",
            )
            columns = {name: _column(data, name) for name in required_names}
            if any(value is None for value in columns.values()):
                raise SolverExecutionError(
                    "MATCH_COLUMNS_MISSING",
                    "the .match table lacks CRVAL/CRPIX/CD/WCS_VALID/INDEXID/HEALPIX/HPNSIDE/PARITY/NMATCH/DIMQUADS",
                )
            row_count = len(data)
            logodds = _column(data, "logodds")
            candidates: list[tuple[float, int, float | None]] = []
            for row in range(row_count):
                if not bool(np.ravel(columns["wcs_valid"])[row]):
                    continue
                crval = np.asarray(columns["crval"][row], dtype=np.float64).reshape(-1)
                crpix = np.asarray(columns["crpix"][row], dtype=np.float64).reshape(-1)
                cd = np.asarray(columns["cd"][row], dtype=np.float64).reshape(-1)
                if crval.size != 2 or crpix.size != 2 or cd.size != 4:
                    continue
                candidate_header = fits.Header()
                candidate_header["CTYPE1"] = "RA---TAN"
                candidate_header["CTYPE2"] = "DEC--TAN"
                candidate_header["CUNIT1"] = "deg"
                candidate_header["CUNIT2"] = "deg"
                candidate_header["CRVAL1"] = float(crval[0])
                candidate_header["CRVAL2"] = float(crval[1])
                candidate_header["CRPIX1"] = float(crpix[0])
                candidate_header["CRPIX2"] = float(crpix[1])
                candidate_header["CD1_1"] = float(cd[0])
                candidate_header["CD1_2"] = float(cd[1])
                candidate_header["CD2_1"] = float(cd[2])
                candidate_header["CD2_2"] = float(cd[3])
                agrees, maximum_arcsec = _wcs_agrees(candidate_header, linear_header, image_shape)
                if agrees:
                    score = float(np.ravel(logodds)[row]) if logodds is not None else 0.0
                    candidates.append((score, row, maximum_arcsec))
            if not candidates:
                raise SolverExecutionError(
                    "MATCH_WCS_BINDING_FAILED",
                    "no WCS_VALID .match row agrees with the emitted .wcs solution",
                )
            _, selected, maximum_arcsec = max(candidates, key=lambda item: item[0])
            index_id = _stable_scalar(np.ravel(columns["indexid"])[selected])
            healpix = _stable_scalar(np.ravel(columns["healpix"])[selected])
            hpnside = _stable_scalar(np.ravel(columns["hpnside"])[selected])
            if not index_id or not healpix or not hpnside:
                raise SolverExecutionError("INDEX_IDENTITY_MISSING", "the accepted .match row has no index identity")
            index_identities = (
                f"astrometry.net:index:{index_id}:healpix:{healpix}:hpnside:{hpnside}",
            )
            raw_parity = bool(np.ravel(columns["parity"])[selected])
            backend_match_count = int(np.ravel(columns["nmatch"])[selected])
            matched_quad_stars = int(np.ravel(columns["dimquads"])[selected])
            if backend_match_count < 0 or matched_quad_stars < 3:
                raise SolverExecutionError("MATCH_COUNT_INVALID", ".match reports invalid match/quad counts")
            # In astrometry.net 0.97, NMATCH is already the total number of
            # accepted field/index correspondences written to the .corr table.
            # DIMQUADS describes the stars in the seed quad; those rows are a
            # subset of NMATCH, not extra correspondences.  Adding DIMQUADS
            # here rejected genuine solutions (for example NMATCH=56,
            # DIMQUADS=4 with 56 .corr rows) after solve-field had solved them.
            backend_correspondence_count = backend_match_count
            validated_parity = wcs_parity(wcs_header)
            # Astrometry.net's MatchObj convention is TRUE="neg" and
            # FALSE="pos" (solver.c).  Its image-quad parity name is opposite
            # the sign of the celestial CD determinant, as verified against
            # upstream 0.97 artifacts; bind the two conventions explicitly.
            if raw_parity != (validated_parity.value == "POSITIVE"):
                raise SolverExecutionError(
                    "MATCH_PARITY_MISMATCH",
                    "the .match parity marker disagrees with the emitted WCS determinant",
                )
    except SolverExecutionError:
        raise
    except Exception as error:
        raise SolverExecutionError("MATCH_TABLE_INVALID", str(error)) from error
    _require_unchanged_identity(path, identity, max_bytes=_MAX_CORRESPONDENCE_BYTES)
    return index_identities, backend_correspondence_count, identity, {
        "selectedRow": int(selected),
        "candidateRowsAgreeingWithWcs": len(candidates),
        "wcsAgreementMaxArcsec": maximum_arcsec,
        "indexIdentities": list(index_identities),
        "backendNmatch": backend_match_count,
        "matchedQuadStars": matched_quad_stars,
        "expectedCorrespondenceRows": backend_correspondence_count,
        "backendParity": "neg" if raw_parity else "pos",
        "validatedParity": validated_parity.value,
    }


class AstrometryNetSolverBackend:
    backend_id = "astrometry-net"

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        *,
        executable_args: Sequence[str] = (),
        config_path: str | os.PathLike[str] | None = None,
        catalog_manifest_dir: str | os.PathLike[str] | None = None,
        require_managed_catalog: bool = True,
        solve_profile: AstrometryNetSolveProfile | None = None,
        timeout_seconds: float = 180.0,
        probe_timeout_seconds: float = 5.0,
        staging_root: str | os.PathLike[str] | None = None,
        diagnostic_log_root: str | os.PathLike[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.staging_root = Path(staging_root).expanduser() if staging_root is not None else None
        self.environment = dict(environment or {})
        self.solve_profile = solve_profile or AstrometryNetSolveProfile()
        self.solve_profile.validate()
        if not isinstance(require_managed_catalog, bool):
            raise ValueError("require_managed_catalog must be a boolean")
        self.require_managed_catalog = require_managed_catalog
        self.catalog_manifest_dir = catalog_manifest_dir
        discovered_config = discover_astrometry_config(
            config_path,
            environment=platform_services.merged_environment(os.environ, self.environment, platform_id=platform_services.current().platform_id),
        )
        self.config_path = Path(discovered_config) if discovered_config is not None else None
        path = discover_astrometry_net(executable, environment=platform_services.merged_environment(os.environ, self.environment, platform_id=platform_services.current().platform_id))
        # A bundled solver ships without upstream's Python ``image2pnm`` and
        # netpbm's ``pnmfile``, which only sniff the file type and write a
        # PNM that ``--no-plots`` never uses; image2xy reads the same FITS
        # either way, so the solution is identical.
        bundled = bundled_astrometry_root(
            environment=platform_services.merged_environment(os.environ, self.environment, platform_id=platform_services.current().platform_id)
        )
        self.assume_fits_image = (
            path is not None and bundled is not None and Path(path).resolve().is_relative_to(bundled.resolve())
        )
        self.runtime: SolverProcessRuntime | None = None
        if path is None:
            self.probe = ExecutableProbe(None, False, False, "unavailable", error_code="EXECUTABLE_UNAVAILABLE", message="solve-field was not found")
        else:
            try:
                self.runtime = SolverProcessRuntime(
                    path,
                    executable_args=executable_args,
                    environment=self.environment,
                    diagnostic_log_root=diagnostic_log_root,
                )
                self.probe = _probe_astrometry_runtime(self.runtime, probe_timeout_seconds)
            except (ValueError, SolverExecutionError) as error:
                self.probe = ExecutableProbe(path, False, False, "unavailable", error_code="EXECUTABLE_UNAVAILABLE", message=str(error))

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.backend_id,
            stage=StageKind.SOLVER,
            display_name="astrometry.net solve-field",
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
                    "indexFiles": "required; exact scale coverage is validated by the solve result",
                    "configDiscovered": self.config_path is not None,
                    "configIdentityValidatedAtSolve": self.config_path is not None,
                    "managedCatalogRequired": self.require_managed_catalog,
                },
                "scientificQualityEvidence": {
                    "status": "recomputed",
                    "correspondenceArtifact": ".corr",
                    "indexIdentityArtifact": ".match",
                },
                "solveProfile": {
                    "profileVersion": 1,
                    "adaptive": self.solve_profile.downsample is None,
                    "unconstrainedFallback": self.solve_profile.unconstrained_fallback,
                    "probeIsolation": "solve-only options are not appended to probe argv",
                },
            },
        )

    def validate_options(self, options: dict[str, Any]) -> tuple[str, ...]:
        allowed = {"raHintDegrees", "decHintDegrees", "fieldOfViewDegrees", "searchRadiusDegrees"}
        return tuple(f"unknown solver option: {key}" for key in sorted(set(options) - allowed))

    def solve(self, request: SolveRequest) -> SolverResult:
        if self.runtime is None or not self.probe.execution_ready:
            return _failure(
                self.backend_id,
                self.probe.error_code or "EXECUTABLE_UNAVAILABLE",
                self.probe.message or "solve-field is not execution-ready",
                status=SolverStatus.UNAVAILABLE,
                evidence={"adapterVersion": ADAPTER_VERSION, "probe": self.probe.serializable()},
            )
        try:
            _validate_request(request)
            if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
                raise SolverExecutionError("TIMEOUT_INVALID", "timeout must be finite and positive")
            input_path = Path(request.input_path).expanduser().resolve(strict=True)
            output_path = Path(request.output_path).expanduser().resolve(strict=False)
            catalog_snapshot: dict[str, Any] | None = None
            catalog_preflight: dict[str, Any]
            if self.config_path is None:
                if self.require_managed_catalog:
                    raise SolverExecutionError(
                        "CATALOG_CONFIG_UNMANAGED",
                        "strict astrometry.net solving requires an Ultra-Fast WBPP managed catalog config",
                    )
                catalog_preflight = {"managed": False, "reason": "NO_MANAGED_CONFIG"}
            else:
                try:
                    catalog_snapshot = installed_set_snapshot_for_solver_config(
                        self.config_path,
                        manifest_dir=self.catalog_manifest_dir,
                        environment=platform_services.merged_environment(os.environ, self.environment, platform_id=platform_services.current().platform_id),
                    )
                    catalog_preflight = {
                        "managed": True,
                        "config": {
                            "relativeName": "astrometry.cfg",
                            "sizeBytes": catalog_snapshot["config"]["sizeBytes"],
                            "sha256": catalog_snapshot["config"]["sha256"],
                        },
                        "installedSetIdentities": sorted(
                            item["installedSetIdentity"] for item in catalog_snapshot["receipts"]
                        ),
                        "verifiedIndexCount": len(catalog_snapshot["artifacts"]),
                    }
                except CatalogError as error:
                    if self.require_managed_catalog:
                        raise SolverExecutionError(error.code, str(error)) from error
                    catalog_preflight = {"managed": False, "reason": error.code}
            with tempfile.TemporaryDirectory(
                prefix="ultra-fast-wbpp-astrometry-", dir=self.staging_root
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
                plan = _solve_attempts(
                    self.solve_profile,
                    request,
                    image_shape,
                    self.timeout_seconds,
                )
                attempt_evidence: list[dict[str, Any]] = []
                selected: tuple[
                    Any,
                    Any,
                    Path,
                    Path,
                    Path,
                    Path,
                    Path,
                    Any,
                ] | None = None
                for attempt_index, attempt in enumerate(plan, start=1):
                    attempt_stage = stage / f"attempt-{attempt_index}"
                    attempt_stage.mkdir(mode=0o700)
                    output_base = attempt_stage / "solution"
                    wcs_path = attempt_stage / "solution.wcs"
                    new_path = attempt_stage / "solution.new"
                    solved_path = attempt_stage / "solution.solved"
                    corr_path = attempt_stage / "solution.corr"
                    match_path = attempt_stage / "solution.match"
                    arguments: list[str] = []
                    if self.config_path is not None:
                        arguments.extend(("--config", str(self.config_path)))
                    arguments.extend(
                        (
                            "--dir",
                            str(attempt_stage),
                            "--out",
                            output_base.name,
                            "--wcs",
                            str(wcs_path),
                            "--new-fits",
                            str(new_path),
                            "--solved",
                            str(solved_path),
                            "--corr",
                            str(corr_path),
                            "--match",
                            str(match_path),
                            "--no-verify",
                            "--no-plots",
                            "--downsample",
                            str(attempt.downsample),
                            "--objs",
                            str(attempt.source_limit),
                            "--depth",
                            f"{attempt.depth_min}-{attempt.depth_max}",
                            "--pixel-error",
                            f"{self.solve_profile.pixel_error:.12g}",
                        )
                    )
                    if self.assume_fits_image:
                        arguments.append("--fits-image")
                    if (
                        attempt.include_hints
                        and request.ra_hint_degrees is not None
                        and request.dec_hint_degrees is not None
                    ):
                        arguments.extend(
                            (
                                "--ra",
                                f"{request.ra_hint_degrees:.12g}",
                                "--dec",
                                f"{request.dec_hint_degrees:.12g}",
                            )
                        )
                        radius = (
                            request.search_radius_degrees
                            if request.search_radius_degrees is not None
                            else 15.0
                        )
                        arguments.extend(("--radius", f"{radius:.12g}"))
                    if attempt.include_hints and request.field_of_view_degrees is not None:
                        tolerance = self.solve_profile.scale_tolerance_fraction
                        low = max(1e-9, request.field_of_view_degrees * (1.0 - tolerance))
                        high = min(180.0, request.field_of_view_degrees * (1.0 + tolerance))
                        arguments.extend(
                            (
                                "--scale-units",
                                "degwidth",
                                "--scale-low",
                                f"{low:.12g}",
                                "--scale-high",
                                f"{high:.12g}",
                            )
                        )
                    arguments.append(str(staged_input))
                    outcome, executable_identities = self.runtime.run(
                        arguments,
                        cwd=attempt_stage,
                        timeout_seconds=attempt.timeout_seconds,
                        log_stem=f"solve-{attempt_index}",
                    )
                    attempt_record: dict[str, Any] = {
                        "plan": attempt.serializable(),
                        "process": outcome.serializable(),
                        "backendConfirmed": False,
                    }
                    attempt_evidence.append(attempt_record)
                    if not _same_stat(input_path, source_identity):
                        return _failure(
                            self.backend_id,
                            "SOURCE_DRIFT",
                            "source input changed during solving",
                            evidence={"adapterVersion": ADAPTER_VERSION, "attempts": attempt_evidence},
                        )
                    staged_after = _regular_identity(staged_input)
                    if (
                        staged_after.sha256 != staged_identity.sha256
                        or staged_after.size_bytes != staged_identity.size_bytes
                    ):
                        return _failure(
                            self.backend_id,
                            "STAGED_INPUT_DRIFT",
                            "solve-field modified its staged input without authorization",
                            evidence={"adapterVersion": ADAPTER_VERSION, "attempts": attempt_evidence},
                        )
                    if config_identity is not None:
                        _require_unchanged_identity(
                            self.config_path,
                            config_identity,
                            max_bytes=_MAX_CONFIG_BYTES,
                        )
                    if outcome.spawn_error:
                        attempt_record["failureCode"] = "SOLVER_SPAWN_FAILED"
                        break
                    if outcome.timed_out:
                        attempt_record["failureCode"] = "SOLVER_TIMEOUT"
                        continue
                    if outcome.exit_code != 0:
                        attempt_record["failureCode"] = "SOLVER_EXIT_NONZERO"
                        continue
                    try:
                        marker, marker_identity = _read_marker(solved_path)
                    except SolverExecutionError as error:
                        attempt_record["failureCode"] = error.code
                        continue
                    attempt_record["solvedMarker"] = marker_identity.serializable(
                        expose_path=False
                    )
                    if marker != b"\x01":
                        attempt_record["failureCode"] = "BACKEND_CONFIRMATION_MISSING"
                        continue
                    attempt_record["backendConfirmed"] = True
                    selected = (
                        outcome,
                        executable_identities,
                        wcs_path,
                        new_path,
                        solved_path,
                        corr_path,
                        match_path,
                        marker_identity,
                    )
                    break

                selected_process = attempt_evidence[-1]["process"] if attempt_evidence else None
                base_evidence = {
                    "adapterVersion": ADAPTER_VERSION,
                    "probe": self.probe.serializable(),
                    "source": source_identity.serializable(expose_path=False),
                    "stagedInput": staged_identity.serializable(expose_path=False),
                    "process": selected_process,
                    "attempts": attempt_evidence,
                    "solveProfile": self.solve_profile.serializable(image_shape),
                    "timeoutBudgetSeconds": self.timeout_seconds,
                    "hintFallbackPolicy": (
                        "remove-coordinate-and-scale-constraints"
                        if len(plan) > 1
                        else "single-attempt"
                    ),
                    "config": (
                        config_identity.serializable(expose_path=False)
                        if config_identity is not None
                        else None
                    ),
                    "catalogPreflight": catalog_preflight,
                    "environmentKeys": sorted(self.runtime.environment),
                    "environmentSha256": f"sha256:{hashlib.sha256(json.dumps(self.runtime.environment, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}",
                }
                if selected is None:
                    failure_code = str(attempt_evidence[-1].get("failureCode", "BACKEND_CONFIRMATION_MISSING"))
                    messages = {
                        "SOLVER_TIMEOUT": "solve-field exhausted its bounded execution attempts",
                        "SOLVER_SPAWN_FAILED": "solve-field could not be started",
                        "SOLVER_EXIT_NONZERO": "solve-field attempts exited without a solution",
                        "BACKEND_CONFIRMATION_MISSING": "solve-field emitted no exact binary solved marker",
                    }
                    return _failure(
                        self.backend_id,
                        failure_code,
                        messages.get(failure_code, "solve-field did not produce a verified solution"),
                        evidence=base_evidence,
                    )
                (
                    outcome,
                    executable_identities,
                    wcs_path,
                    new_path,
                    solved_path,
                    corr_path,
                    match_path,
                    marker_identity,
                ) = selected
                base_evidence["process"] = outcome.serializable()
                base_evidence["executables"] = [
                    item.serializable(expose_path=False) for item in executable_identities
                ]
                try:
                    wcs_header, wcs_identity = _read_header_artifact(wcs_path)
                    new_header, new_shape, new_identity = _read_new_fits(new_path)
                except SolverExecutionError as error:
                    return _failure(
                        self.backend_id,
                        error.code,
                        str(error),
                        evidence={
                            **base_evidence,
                            "outputs": {"solved": marker_identity.serializable(expose_path=False)},
                        },
                    )
                outputs = {
                    "solved": marker_identity.serializable(expose_path=False),
                    "wcs": wcs_identity.serializable(expose_path=False),
                    "new": new_identity.serializable(expose_path=False),
                }
                if new_shape != image_shape:
                    return _failure(
                        self.backend_id,
                        "OUTPUT_GEOMETRY_DRIFT",
                        f".new geometry {new_shape} differs from staged input {image_shape}",
                        evidence={**base_evidence, "outputs": outputs},
                    )
                wcs_validation = validate_wcs_header(wcs_header, image_shape=image_shape)
                new_validation = validate_wcs_header(new_header, image_shape=image_shape)
                if not wcs_validation.valid or not new_validation.valid:
                    failed = wcs_validation if not wcs_validation.valid else new_validation
                    return _failure(
                        self.backend_id,
                        "WCS_VALIDATION_FAILED",
                        f"{failed.code}: {failed.message}",
                        evidence={
                            **base_evidence,
                            "outputs": outputs,
                            "wcsValidation": wcs_validation.serializable(),
                            "newFitsWcsValidation": new_validation.serializable(),
                        },
                    )
                agrees, maximum_arcsec = _wcs_agrees(wcs_header, new_header, image_shape)
                if not agrees:
                    return _failure(
                        self.backend_id,
                        "SOLVER_OUTPUT_DRIFT",
                        ".wcs and .new contain different astrometric solutions",
                        evidence={
                            **base_evidence,
                            "outputs": outputs,
                            "wcsAgreementMaxArcsec": maximum_arcsec,
                        },
                    )
                try:
                    (
                        index_identities,
                        expected_correspondence_count,
                        match_identity,
                        match_diagnostics,
                    ) = _read_match_index_identity(
                        match_path,
                        wcs_header,
                        image_shape,
                    )
                    astrometric_quality, corr_identity, corr_diagnostics = _read_correspondence_quality(
                        corr_path,
                        wcs_header,
                        image_shape,
                        index_identities,
                        expected_correspondence_count,
                    )
                    catalog_binding: dict[str, Any] | None = None
                    if catalog_snapshot is not None:
                        try:
                            binding = installed_set_identity_for_solver_indexes(
                                index_identities,
                                catalog_root=Path(catalog_snapshot["catalogRoot"]),
                                manifest_dir=self.catalog_manifest_dir,
                                environment=platform_services.merged_environment(os.environ, self.environment, platform_id=platform_services.current().platform_id),
                            )
                            pre_receipts = {
                                item["installedSetIdentity"]: item
                                for item in catalog_snapshot["receipts"]
                            }
                            if binding["installedSetIdentity"] not in pre_receipts:
                                raise CatalogError(
                                    "CATALOG_INSTALLED_SET_DRIFT",
                                    "the matched index was not present in the pre-solve installed set",
                                )
                            pre_artifacts = {
                                item["artifactId"]: item
                                for item in catalog_snapshot["artifacts"]
                            }
                            for artifact in binding["indexArtifacts"]:
                                before = pre_artifacts.get(artifact["relativeName"])
                                if before is None or any(
                                    before[field] != artifact[field]
                                    for field in ("sizeBytes", "sha256")
                                ):
                                    raise CatalogError(
                                        "CATALOG_INSTALLED_SET_DRIFT",
                                        "the matched index differs from its pre-solve byte identity",
                                    )
                            verify_installed_set_snapshot(
                                catalog_snapshot,
                                manifest_dir=self.catalog_manifest_dir,
                                environment=platform_services.merged_environment(os.environ, self.environment, platform_id=platform_services.current().platform_id),
                            )
                            index_artifacts = tuple(
                                SolverIndexArtifact(
                                    index_id=item["indexId"],
                                    relative_name=item["relativeName"],
                                    size_bytes=item["sizeBytes"],
                                    sha256=item["sha256"],
                                    manifest_sha256=item["manifestSha256"],
                                    installed_set_identity=item["installedSetIdentity"],
                                )
                                for item in binding["indexArtifacts"]
                            )
                            astrometric_quality = replace(
                                astrometric_quality,
                                catalog_managed=True,
                                installed_set_identity=binding["installedSetIdentity"],
                                catalog_manifest_sha256=binding["manifestSha256"],
                                index_artifacts=index_artifacts,
                            )
                            catalog_binding = {
                                "catalogManaged": True,
                                "catalogId": binding["catalogId"],
                                "installedSetIdentity": binding["installedSetIdentity"],
                                "manifestSha256": binding["manifestSha256"],
                                "config": {
                                    "relativeName": "astrometry.cfg",
                                    "sizeBytes": binding["config"]["sizeBytes"],
                                    "sha256": binding["config"]["sha256"],
                                },
                                "indexArtifacts": [item.serializable() for item in index_artifacts],
                            }
                        except CatalogError as error:
                            if self.require_managed_catalog:
                                raise SolverExecutionError(error.code, str(error)) from error
                            catalog_binding = {
                                "catalogManaged": False,
                                "reason": error.code,
                            }
                    elif self.require_managed_catalog:
                        raise SolverExecutionError(
                            "CATALOG_INSTALLED_SET_UNBOUND",
                            "strict solve produced an INDEXID without a managed installed-set snapshot",
                        )
                except SolverExecutionError as error:
                    return _failure(
                        self.backend_id,
                        error.code,
                        str(error),
                        evidence={**base_evidence, "outputs": outputs},
                    )
                outputs["corr"] = corr_identity.serializable(expose_path=False)
                outputs["match"] = match_identity.serializable(expose_path=False)
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
                        "attempts": attempt_evidence,
                        "solveProfile": self.solve_profile.serializable(image_shape),
                        "timeoutBudgetSeconds": self.timeout_seconds,
                        "hintFallbackPolicy": base_evidence["hintFallbackPolicy"],
                        "config": base_evidence["config"],
                        "executables": [item.serializable(expose_path=False) for item in executable_identities],
                        "environmentKeys": sorted(self.runtime.environment),
                        "environmentSha256": f"sha256:{hashlib.sha256(json.dumps(self.runtime.environment, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}",
                        "outputs": {**outputs, "published": published.serializable(expose_path=False)},
                        "wcsValidation": wcs_validation.serializable(),
                        "newFitsWcsValidation": new_validation.serializable(),
                        "wcsAgreementMaxArcsec": maximum_arcsec,
                        "astrometricQuality": astrometric_quality.serializable(),
                        "correspondenceDiagnostics": corr_diagnostics,
                        "matchDiagnostics": match_diagnostics,
                        "catalogBinding": catalog_binding,
                        "solutionWcsSha256": f"sha256:{_wcs_header_sha256(wcs_header)}",
                        "backendConfirmation": "binary-solved-marker=1",
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


AstrometryNetBackend = AstrometryNetSolverBackend


__all__ = [
    "ADAPTER_VERSION",
    "AstrometryNetBackend",
    "AstrometryNetSolveProfile",
    "AstrometryNetSolverBackend",
    "discover_astrometry_config",
    "discover_astrometry_net",
    "probe_astrometry_net",
]
