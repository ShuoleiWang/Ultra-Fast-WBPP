"""Solved-panel mosaicking with an explicit final-solve boundary.

This backend uses the optional :mod:`reproject` package when available.  Input
panels must carry fresh Ultra-Fast WBPP SOLVED markers and numerically valid
celestial WCS metadata.  The optimal output WCS is a reprojection grid, not a
new astrometric solution: every mosaic is therefore marked
``NEEDS_FINAL_SOLVE`` and must pass through a solver before publication as a
finished scientific product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray

from .integrity import canonical_json_document, sha256_digest
from .platform import remove_tree
from .publication import (
    ColorProductError,
    DirectoryPublisher,
    _best_effort_fsync_directory,
    _fsync_directory,
    _publish_directory_no_replace,
)
from .solver import validate_wcs_header


MOSAIC_VERSION = "ultra-fast-wbpp-solved-panel-mosaic-v2"
MOSAIC_STATE = "NEEDS_FINAL_SOLVE"
_MINIMUM_PHOTOMETRIC_GAIN = 0.25
_MAXIMUM_PHOTOMETRIC_GAIN = 4.0


class MosaicError(RuntimeError):
    """Stable, user-actionable failure at the mosaic execution boundary."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        self.code = code
        self.path = path
        detail = f"{path}: {message}" if path else message
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class MosaicCapability:
    backend_id: str
    available: bool
    execution_ready: bool
    version: str | None
    reason: str | None
    auto_rotate_available: bool = False
    auto_rotate_reason: str | None = None
    device: str = "CPU"
    capabilities: tuple[str, ...] = (
        "optimal-celestial-grid",
        "wcs-reprojection",
        "science-coadd",
        "footprint",
        "coverage",
        "identity-bound-receipt",
        "needs-final-solve",
    )

    @property
    def capable(self) -> bool:
        return self.available and self.execution_ready

    def serializable(self) -> dict[str, Any]:
        return {
            "backendId": self.backend_id,
            "capability": self.capable,
            "available": self.available,
            "executionReady": self.execution_ready,
            "version": self.version,
            "reason": self.reason,
            "autoRotateAvailable": self.auto_rotate_available,
            "autoRotateReason": self.auto_rotate_reason,
            "device": self.device,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True, slots=True)
class ReprojectProvider:
    backend_id: str
    version: str
    find_optimal_celestial_wcs: Callable[..., Any] = field(repr=False, compare=False)
    reproject_and_coadd: Callable[..., Any] = field(repr=False, compare=False)
    reproject_function: Callable[..., Any] | None = field(default=None, repr=False, compare=False)
    auto_rotate_available: bool = True
    auto_rotate_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MosaicRequest:
    panel_paths: tuple[str, ...]
    output_directory: str
    combine_function: str = "mean"
    match_background: bool = False
    auto_rotate: bool = False
    max_output_pixels: int = 256 * 1024 * 1024
    minimum_covered_fraction: float = 0.70
    minimum_pair_overlap_pixels: int = 16
    minimum_pair_overlap_fraction: float = 0.005
    maximum_seam_normalized_mad: float = 0.25


@dataclass(frozen=True, slots=True)
class MosaicResult:
    output_directory: str
    mosaic_path: str
    receipt_path: str
    state: str
    receipt: Mapping[str, Any]

    def serializable(self) -> dict[str, Any]:
        return {
            "outputDirectory": self.output_directory,
            "mosaicPath": self.mosaic_path,
            "receiptPath": self.receipt_path,
            "state": self.state,
            "receipt": dict(self.receipt),
        }


@dataclass(frozen=True, slots=True)
class _Panel:
    path: Path
    data: NDArray[np.float32]
    header: fits.Header
    wcs: WCS
    shape: tuple[int, int]
    identity: Mapping[str, Any]
    wcs_validation: Mapping[str, Any]


def _supports_keyword(callable_object: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


def _load_reproject_provider(
    importer: Callable[[str], Any] = importlib.import_module,
) -> tuple[MosaicCapability, ReprojectProvider | None]:
    backend_id = "reproject-cpu"
    try:
        package = importer("reproject")
        mosaicking = importer("reproject.mosaicking")
    except (ImportError, ModuleNotFoundError) as error:
        return (
            MosaicCapability(
                backend_id=backend_id,
                available=False,
                execution_ready=False,
                version=None,
                reason=f"optional reproject dependency is unavailable: {error}",
            ),
            None,
        )
    except Exception as error:
        return (
            MosaicCapability(
                backend_id=backend_id,
                available=False,
                execution_ready=False,
                version=None,
                reason=f"reproject discovery failed: {error}",
            ),
            None,
        )

    find_optimal = getattr(mosaicking, "find_optimal_celestial_wcs", None)
    coadd = getattr(mosaicking, "reproject_and_coadd", None)
    reproject_function = getattr(package, "reproject_interp", None)
    version = getattr(package, "__version__", None)
    if not isinstance(version, str) or not version:
        try:
            version = importlib.metadata.version("reproject")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    reason: str | None = None
    if not callable(find_optimal):
        reason = "reproject.mosaicking.find_optimal_celestial_wcs is missing"
    elif not callable(coadd):
        reason = "reproject.mosaicking.reproject_and_coadd is missing"
    elif not callable(reproject_function):
        reason = "reproject.reproject_interp is missing"
    if reason is not None:
        return (
            MosaicCapability(
                backend_id=backend_id,
                available=True,
                execution_ready=False,
                version=version,
                reason=reason,
            ),
            None,
        )
    try:
        importer("shapely")
        auto_rotate_available = True
        auto_rotate_reason = None
    except Exception as error:
        auto_rotate_available = False
        auto_rotate_reason = f"auto_rotate requires optional shapely: {error}"
    return (
        MosaicCapability(
            backend_id=backend_id,
            available=True,
            execution_ready=True,
            version=version,
            reason=None,
            auto_rotate_available=auto_rotate_available,
            auto_rotate_reason=auto_rotate_reason,
        ),
        ReprojectProvider(
            backend_id,
            version,
            find_optimal,
            coadd,
            reproject_function,
            auto_rotate_available,
            auto_rotate_reason,
        ),
    )


def reproject_capability(
    importer: Callable[[str], Any] = importlib.import_module,
) -> MosaicCapability:
    """Discover the optional backend without making package import depend on it."""

    capability, _ = _load_reproject_provider(importer)
    return capability


def _source_identity(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as error:
        raise MosaicError("PANEL_UNREADABLE", str(error), path=str(path)) from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise MosaicError("PANEL_NOT_REGULAR_FILE", "panel must be a non-symlink regular file", path=str(path))
    return {
        "path": str(path),
        "sha256": sha256_digest(path),
        "sizeBytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _verify_source_stat(panel: _Panel) -> None:
    try:
        info = panel.path.lstat()
    except OSError as error:
        raise MosaicError("SOURCE_CHANGED", "panel disappeared during execution", path=str(panel.path)) from error
    current = (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)
    expected = (
        panel.identity["sizeBytes"],
        panel.identity["mtimeNs"],
        panel.identity["device"],
        panel.identity["inode"],
    )
    if panel.path.is_symlink() or current != expected:
        raise MosaicError("SOURCE_CHANGED", "panel identity changed during execution", path=str(panel.path))


def _read_panel(value: str) -> _Panel:
    if not isinstance(value, str) or not value.strip():
        raise MosaicError("PANEL_PATH_INVALID", "every panel path must be non-empty")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise MosaicError("PANEL_MISSING", str(error), path=value) from error
    identity = _source_identity(path)
    try:
        with fits.open(path, mode="readonly", memmap=False, uint=True, checksum=False) as hdul:
            image_hdu = next(
                (hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2),
                None,
            )
            if image_hdu is None:
                raise MosaicError("PANEL_IMAGE_INVALID", "a two-dimensional image HDU is required", path=str(path))
            data = np.asarray(image_hdu.data, dtype=np.float32).copy()
            header = hdul[0].header.copy()
            if image_hdu is not hdul[0]:
                header.extend(image_hdu.header, update=True, strip=True)
    except MosaicError:
        raise
    except Exception as error:
        raise MosaicError("PANEL_READ_FAILED", str(error), path=str(path)) from error
    if header.get("OAFSTATE") != "SOLVED" or header.get("OAFWCS") != "SOLVED":
        raise MosaicError(
            "PANEL_NOT_SOLVED",
            "OAFSTATE=SOLVED and OAFWCS=SOLVED are required before mosaicking",
            path=str(path),
        )
    shape = tuple(int(item) for item in data.shape)
    validation = validate_wcs_header(header, image_shape=shape)
    if not validation.valid:
        raise MosaicError("PANEL_WCS_INVALID", f"{validation.code}: {validation.message}", path=str(path))
    try:
        wcs = WCS(header, relax=False).celestial
    except Exception as error:
        raise MosaicError("PANEL_WCS_INVALID", str(error), path=str(path)) from error
    panel = _Panel(path, data, header, wcs, shape, identity, validation.serializable())
    _verify_source_stat(panel)
    return panel


def _validate_request(request: MosaicRequest) -> Path:
    if (
        isinstance(request.panel_paths, (str, bytes))
        or not isinstance(request.panel_paths, Sequence)
        or len(request.panel_paths) < 2
    ):
        raise MosaicError("PANELS_INSUFFICIENT", "at least two solved panel paths are required")
    if request.combine_function not in {"mean", "sum"}:
        raise MosaicError(
            "COMBINE_FUNCTION_INVALID",
            "only mean and sum preserve cumulative footprint semantics required for coverage evidence",
        )
    if not isinstance(request.match_background, bool) or not isinstance(request.auto_rotate, bool):
        raise MosaicError("MOSAIC_OPTION_INVALID", "match_background and auto_rotate must be booleans")
    if request.match_background:
        raise MosaicError(
            "MOSAIC_BACKGROUND_MATCH_CONFLICT",
            "backend background matching cannot be combined with the built-in content-bound gain/offset calibration",
        )
    limit = request.max_output_pixels
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise MosaicError("OUTPUT_LIMIT_INVALID", "max_output_pixels must be a positive integer")
    if not (
        isinstance(request.minimum_covered_fraction, (int, float))
        and not isinstance(request.minimum_covered_fraction, bool)
        and math.isfinite(float(request.minimum_covered_fraction))
        and 0.0 < float(request.minimum_covered_fraction) <= 1.0
    ):
        raise MosaicError(
            "MOSAIC_GATE_INVALID",
            "minimum_covered_fraction must be finite and in (0, 1]",
        )
    if (
        isinstance(request.minimum_pair_overlap_pixels, bool)
        or not isinstance(request.minimum_pair_overlap_pixels, int)
        or request.minimum_pair_overlap_pixels < 1
    ):
        raise MosaicError(
            "MOSAIC_GATE_INVALID",
            "minimum_pair_overlap_pixels must be a positive integer",
        )
    for name in ("minimum_pair_overlap_fraction", "maximum_seam_normalized_mad"):
        value = getattr(request, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise MosaicError("MOSAIC_GATE_INVALID", f"{name} must be finite and positive")
    if request.minimum_pair_overlap_fraction > 1.0:
        raise MosaicError(
            "MOSAIC_GATE_INVALID",
            "minimum_pair_overlap_fraction cannot exceed 1",
        )
    if not isinstance(request.output_directory, str) or not request.output_directory.strip():
        raise MosaicError("OUTPUT_PATH_INVALID", "output_directory is required")
    output = Path(request.output_directory).expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise MosaicError("OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(output))
    return output


def _normalized_filter(value: Any) -> str | None:
    token = "".join(character for character in str(value).casefold() if character.isalnum())
    if token in {"", "unknown", "none", "unspecified"}:
        return None
    aliases = {
        "red": "r",
        "green": "g",
        "blue": "b",
        "lum": "l",
        "luminance": "l",
        "halpha": "ha",
        "hydrogenalpha": "ha",
        "oxygeniii": "oiii",
        "sulfurii": "sii",
        "sulphurii": "sii",
    }
    return aliases.get(token, token)


def _call_optimal_wcs(provider: ReprojectProvider, inputs: Sequence[tuple[np.ndarray, WCS]], auto_rotate: bool) -> tuple[WCS, tuple[int, int]]:
    kwargs: dict[str, Any] = {}
    if _supports_keyword(provider.find_optimal_celestial_wcs, "auto_rotate"):
        kwargs["auto_rotate"] = auto_rotate
    try:
        value = provider.find_optimal_celestial_wcs(inputs, **kwargs)
    except Exception as error:
        raise MosaicError("OPTIMAL_WCS_FAILED", str(error)) from error
    if not isinstance(value, tuple) or len(value) != 2:
        raise MosaicError("OPTIMAL_WCS_INVALID", "backend must return (WCS, shape)")
    output_wcs, raw_shape = value
    if not isinstance(output_wcs, WCS):
        raise MosaicError("OPTIMAL_WCS_INVALID", "backend returned a non-WCS projection")
    try:
        shape = tuple(int(item) for item in raw_shape)
    except (TypeError, ValueError) as error:
        raise MosaicError("OPTIMAL_WCS_INVALID", "backend returned an invalid output shape") from error
    if len(shape) != 2 or any(item <= 0 for item in shape):
        raise MosaicError("OPTIMAL_WCS_INVALID", "output shape must be positive (height, width)")
    return output_wcs.celestial, shape


def _call_coadd(
    provider: ReprojectProvider,
    inputs: Sequence[tuple[np.ndarray, WCS]],
    output_wcs: WCS,
    shape: tuple[int, int],
    request: MosaicRequest,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    kwargs: dict[str, Any] = {}
    candidates = {
        "shape_out": shape,
        "reproject_function": provider.reproject_function,
        "combine_function": request.combine_function,
        "match_background": request.match_background,
    }
    for name, value in candidates.items():
        if _supports_keyword(provider.reproject_and_coadd, name):
            kwargs[name] = value
    try:
        result = provider.reproject_and_coadd(inputs, output_wcs, **kwargs)
    except Exception as error:
        raise MosaicError("REPROJECT_COADD_FAILED", str(error)) from error
    if not isinstance(result, tuple) or len(result) != 2:
        raise MosaicError("REPROJECT_RESULT_INVALID", "backend must return (science, footprint)")
    try:
        science = np.asarray(result[0], dtype=np.float32)
        coverage = np.asarray(result[1], dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise MosaicError("REPROJECT_RESULT_INVALID", str(error)) from error
    if science.shape != shape or coverage.shape != shape:
        raise MosaicError(
            "REPROJECT_SHAPE_MISMATCH",
            f"backend returned science {science.shape} and coverage {coverage.shape}; expected {shape}",
        )
    if not np.all(np.isfinite(coverage)) or np.any(coverage < 0.0):
        raise MosaicError("REPROJECT_COVERAGE_INVALID", "coverage must be finite and non-negative")
    covered = coverage > 0.0
    if not np.any(covered):
        raise MosaicError("REPROJECT_COVERAGE_EMPTY", "mosaic has no covered output pixels")
    if not np.all(np.isfinite(science[covered])):
        raise MosaicError("REPROJECT_SCIENCE_INVALID", "covered science pixels must be finite")
    science = science.copy()
    science[~covered] = np.nan
    return science, coverage


def _call_reproject_one(
    provider: ReprojectProvider,
    panel: _Panel,
    output_wcs: WCS,
    shape: tuple[int, int],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Reproject one panel for exact overlap and seam evidence.

    The aggregate coadd footprint cannot say which two panels overlap, and it
    cannot expose a discontinuity hidden by averaging.  Production mosaics
    therefore require the backend's per-panel reprojection surface as an
    independent gate input.
    """

    function = provider.reproject_function
    if not callable(function):
        raise MosaicError(
            "MOSAIC_SEAM_EVIDENCE_UNAVAILABLE",
            "the reproject backend has no per-panel reprojection function",
        )
    kwargs: dict[str, Any] = {}
    if _supports_keyword(function, "shape_out"):
        kwargs["shape_out"] = shape
    if _supports_keyword(function, "return_footprint"):
        kwargs["return_footprint"] = True
    try:
        raw = function((panel.data, panel.wcs), output_wcs, **kwargs)
    except Exception as error:
        raise MosaicError(
            "MOSAIC_SEAM_EVIDENCE_FAILED", str(error), path=str(panel.path)
        ) from error
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise MosaicError(
            "MOSAIC_SEAM_EVIDENCE_INVALID",
            "per-panel reprojection must return (science, footprint)",
            path=str(panel.path),
        )
    science = np.asarray(raw[0], dtype=np.float32)
    footprint = np.asarray(raw[1], dtype=np.float32)
    if science.shape != shape or footprint.shape != shape:
        raise MosaicError(
            "MOSAIC_SEAM_EVIDENCE_INVALID",
            f"per-panel reprojection returned {science.shape}/{footprint.shape}; expected {shape}",
            path=str(panel.path),
        )
    if not np.all(np.isfinite(footprint)) or np.any(footprint < 0):
        raise MosaicError(
            "MOSAIC_SEAM_EVIDENCE_INVALID",
            "per-panel footprint must be finite and non-negative",
            path=str(panel.path),
        )
    covered = footprint > 0
    if not np.any(covered) or not np.all(np.isfinite(science[covered])):
        raise MosaicError(
            "MOSAIC_SEAM_EVIDENCE_INVALID",
            "per-panel science has no finite covered pixels",
            path=str(panel.path),
        )
    return science, footprint


def _seam_normalized_mad(left: NDArray[np.float32], right: NDArray[np.float32]) -> dict[str, float]:
    """Return a robust, gain/offset-insensitive overlap discontinuity metric."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x_median = float(np.median(x))
    y_median = float(np.median(y))
    x_scale = float(np.percentile(x, 75) - np.percentile(x, 25))
    y_scale = float(np.percentile(y, 75) - np.percentile(y, 25))
    floor = max(abs(x_median), abs(y_median), 1.0) * 1e-9
    if x_scale > floor and y_scale > floor:
        gain = y_scale / x_scale
        offset = y_median - gain * x_median
    elif abs(x_median) > floor:
        gain = y_median / x_median
        offset = 0.0
    else:
        gain = 1.0
        offset = y_median - x_median
    residual = y - (gain * x + offset)
    residual_median = float(np.median(residual))
    residual_mad = float(1.4826 * np.median(np.abs(residual - residual_median)))
    signal_scale = max(
        float(1.4826 * np.median(np.abs(y - y_median))),
        abs(y_median) * 1e-6,
        floor,
    )
    return {
        "fittedGain": gain,
        "fittedOffset": offset,
        "residualMad": residual_mad,
        "normalizationScale": signal_scale,
        "normalizedMad": residual_mad / signal_scale,
    }


def _direct_seam_normalized_error(
    left: NDArray[np.float32], right: NDArray[np.float32]
) -> dict[str, float]:
    """Measure an already-normalized overlap without fitting away a seam."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    residual = y - x
    residual_median = float(np.median(residual))
    residual_mad = float(
        1.4826 * np.median(np.abs(residual - residual_median))
    )
    y_median = float(np.median(y))
    signal_scale = max(
        float(1.4826 * np.median(np.abs(y - y_median))),
        abs(y_median) * 1e-6,
        1e-9,
    )
    absolute_error = max(abs(residual_median), residual_mad)
    return {
        "fittedGain": 1.0,
        "fittedOffset": 0.0,
        "residualMedian": residual_median,
        "residualMad": residual_mad,
        "normalizationScale": signal_scale,
        "normalizedMad": absolute_error / signal_scale,
    }


def _exact_overlap_and_seam_evidence(
    panels: Sequence[_Panel],
    projected: Sequence[tuple[NDArray[np.float32], NDArray[np.float32]]],
    request: MosaicRequest,
    *,
    fit_photometry: bool = True,
) -> dict[str, Any]:
    pairwise: list[dict[str, Any]] = []
    edges: list[tuple[int, int]] = []
    for left_index in range(len(panels)):
        left_science, left_footprint = projected[left_index]
        left_mask = left_footprint > 0
        for right_index in range(left_index + 1, len(panels)):
            right_science, right_footprint = projected[right_index]
            right_mask = right_footprint > 0
            overlap = left_mask & right_mask
            count = int(np.count_nonzero(overlap))
            denominator = min(
                int(np.count_nonzero(left_mask)), int(np.count_nonzero(right_mask))
            )
            fraction = float(count / denominator) if denominator else 0.0
            overlap_pass = (
                count >= request.minimum_pair_overlap_pixels
                and fraction >= request.minimum_pair_overlap_fraction
            )
            seam = None
            if count:
                seam = (
                    _seam_normalized_mad(
                        left_science[overlap], right_science[overlap]
                    )
                    if fit_photometry
                    else _direct_seam_normalized_error(
                        left_science[overlap], right_science[overlap]
                    )
                )
            gain_pass = bool(
                not fit_photometry
                or (
                    seam is not None
                    and math.isfinite(seam["fittedGain"])
                    and _MINIMUM_PHOTOMETRIC_GAIN
                    <= seam["fittedGain"]
                    <= _MAXIMUM_PHOTOMETRIC_GAIN
                )
            )
            seam_pass = bool(
                seam is not None
                and seam["normalizedMad"] <= request.maximum_seam_normalized_mad
                and (gain_pass or not fit_photometry)
            )
            if overlap_pass and seam_pass:
                edges.append((left_index, right_index))
            pairwise.append(
                {
                    "leftPath": str(panels[left_index].path),
                    "rightPath": str(panels[right_index].path),
                    "overlapPixels": count,
                    "overlapFractionOfSmallerFootprint": fraction,
                    "overlapGatePassed": overlap_pass,
                    "seam": seam,
                    "photometricGainGatePassed": gain_pass,
                    "seamGatePassed": seam_pass,
                    "edgeAccepted": overlap_pass and seam_pass,
                }
            )

    reached = {0}
    changed = True
    while changed:
        changed = False
        for left, right in edges:
            if left in reached and right not in reached:
                reached.add(right)
                changed = True
            elif right in reached and left not in reached:
                reached.add(left)
                changed = True
    connected = len(reached) == len(panels)
    return {
        "pairwise": pairwise,
        "acceptedEdges": [[left, right] for left, right in edges],
        "connected": connected,
        "reachablePanelCount": len(reached),
        "metric": (
            "affine-fit-residual" if fit_photometry else "direct-corrected-residual"
        ),
    }


def _photometrically_calibrated_inputs(
    panels: Sequence[_Panel],
    projected: Sequence[tuple[NDArray[np.float32], NDArray[np.float32]]],
    request: MosaicRequest,
) -> tuple[
    tuple[tuple[NDArray[np.float32], WCS], ...],
    tuple[tuple[NDArray[np.float32], NDArray[np.float32]], ...],
    dict[str, Any],
]:
    """Place every connected panel on panel zero's affine flux scale.

    Pairwise relations are measured only in exact WCS overlap.  A spanning-tree
    propagation makes the applied gain/offset deterministic and auditable; all
    corrected overlap edges are then checked again without refitting anything.
    """

    fitted = _exact_overlap_and_seam_evidence(
        panels, projected, request, fit_photometry=True
    )
    if not fitted["connected"]:
        raise MosaicError(
            "MOSAIC_PHOTOMETRIC_GRAPH_FAILED",
            "panels do not form one connected graph of overlap, bounded gain, and seam-passing edges",
        )

    relations: dict[tuple[int, int], tuple[float, float]] = {}
    pair_index = 0
    for left in range(len(panels)):
        for right in range(left + 1, len(panels)):
            record = fitted["pairwise"][pair_index]
            pair_index += 1
            if record["edgeAccepted"]:
                seam = record["seam"]
                assert isinstance(seam, dict)
                relations[(left, right)] = (
                    float(seam["fittedGain"]),
                    float(seam["fittedOffset"]),
                )

    corrections: list[tuple[float, float] | None] = [None] * len(panels)
    parent_edges: list[tuple[int, int] | None] = [None] * len(panels)
    corrections[0] = (1.0, 0.0)
    pending = [0]
    while pending:
        known = pending.pop(0)
        known_gain, known_offset = corrections[known] or (math.nan, math.nan)
        for (left, right), (pair_gain, pair_offset) in sorted(relations.items()):
            if known == left and corrections[right] is None:
                gain = known_gain / pair_gain
                offset = known_offset - known_gain * pair_offset / pair_gain
                destination = right
            elif known == right and corrections[left] is None:
                gain = known_gain * pair_gain
                offset = known_gain * pair_offset + known_offset
                destination = left
            else:
                continue
            if not math.isfinite(gain) or not math.isfinite(offset) or gain <= 0:
                raise MosaicError(
                    "MOSAIC_PHOTOMETRIC_SOLUTION_INVALID",
                    "the overlap graph produced a non-finite or non-positive panel correction",
                )
            if not _MINIMUM_PHOTOMETRIC_GAIN <= gain <= _MAXIMUM_PHOTOMETRIC_GAIN:
                raise MosaicError(
                    "MOSAIC_PHOTOMETRIC_SOLUTION_EXTREME",
                    f"panel {destination} requires gain {gain:.6g}, outside the "
                    f"safe [{_MINIMUM_PHOTOMETRIC_GAIN:.6g}, "
                    f"{_MAXIMUM_PHOTOMETRIC_GAIN:.6g}] range",
                )
            corrections[destination] = (gain, offset)
            parent_edges[destination] = (left, right)
            pending.append(destination)

    if any(value is None for value in corrections):
        raise MosaicError(
            "MOSAIC_PHOTOMETRIC_GRAPH_FAILED",
            "the accepted overlap graph did not calibrate every panel",
        )

    calibrated_inputs: list[tuple[NDArray[np.float32], WCS]] = []
    corrected_projected: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
    records: list[dict[str, Any]] = []
    for index, (panel, projection, correction) in enumerate(
        zip(panels, projected, corrections, strict=True)
    ):
        assert correction is not None
        gain, offset = correction
        calibrated = (
            panel.data.astype(np.float64) * gain + offset
        ).astype(np.float32)
        projected_science, projected_footprint = projection
        corrected_science = (
            projected_science.astype(np.float64) * gain + offset
        ).astype(np.float32)
        calibrated_inputs.append((calibrated, panel.wcs))
        corrected_projected.append((corrected_science, projected_footprint))
        records.append(
            {
                "panelIndex": index,
                "path": str(panel.path),
                "referencePanel": index == 0,
                "appliedGain": gain,
                "appliedOffset": offset,
                "parentEdge": (
                    list(parent_edges[index])
                    if parent_edges[index] is not None
                    else None
                ),
            }
        )

    verified = _exact_overlap_and_seam_evidence(
        panels,
        tuple(corrected_projected),
        request,
        fit_photometry=False,
    )
    if not verified["connected"]:
        raise MosaicError(
            "MOSAIC_CORRECTED_SEAM_GATE_FAILED",
            "applied photometric corrections did not pass direct overlap seam validation",
        )
    failed_corrected_pairs = [
        item
        for item in verified["pairwise"]
        if item["overlapGatePassed"] and not item["seamGatePassed"]
    ]
    if failed_corrected_pairs:
        raise MosaicError(
            "MOSAIC_CORRECTED_SEAM_GATE_FAILED",
            f"{len(failed_corrected_pairs)} materially overlapping panel pair(s) "
            "retain a direct seam after calibration",
        )
    return (
        tuple(calibrated_inputs),
        tuple(corrected_projected),
        {
            "algorithm": "reference-panel-affine-overlap-spanning-tree-v1",
            "referencePanelIndex": 0,
            "pairGainBounds": [
                _MINIMUM_PHOTOMETRIC_GAIN,
                _MAXIMUM_PHOTOMETRIC_GAIN,
            ],
            "fittedOverlapGraph": fitted,
            "panelCorrections": records,
            "correctedOverlapVerification": verified,
            "backendBackgroundMatching": False,
        },
    )


def _panel_projection_evidence(panel: _Panel, output_wcs: WCS) -> dict[str, Any]:
    height, width = panel.shape
    pixels = np.asarray(
        [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0), ((width - 1.0) / 2.0, (height - 1.0) / 2.0)],
        dtype=np.float64,
    )
    try:
        sky = np.asarray(panel.wcs.all_pix2world(pixels, 0), dtype=np.float64)
        output_pixels = np.asarray(output_wcs.all_world2pix(sky, 0), dtype=np.float64)
    except Exception as error:
        raise MosaicError("PANEL_PROJECTION_EVIDENCE_FAILED", str(error), path=str(panel.path)) from error
    if not np.all(np.isfinite(output_pixels)):
        raise MosaicError("PANEL_PROJECTION_EVIDENCE_FAILED", "panel footprint is non-finite", path=str(panel.path))
    corners = output_pixels[:4]
    bbox = [
        float(np.min(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(np.max(corners[:, 0])),
        float(np.max(corners[:, 1])),
    ]
    return {
        "identity": panel.identity,
        "shape": list(panel.shape),
        "filter": str(panel.header.get("FILTER", "UNKNOWN")),
        "stateCards": {
            "OAFSTATE": str(panel.header.get("OAFSTATE")),
            "OAFWCS": str(panel.header.get("OAFWCS")),
            "OAFSOLVR": (
                str(panel.header.get("OAFSOLVR"))
                if panel.header.get("OAFSOLVR") is not None
                else None
            ),
        },
        "wcsValidation": panel.wcs_validation,
        "samplePixels": pixels.tolist(),
        "sampleSkyDegrees": sky.tolist(),
        "outputPixels": output_pixels.tolist(),
        "outputBoundingBox": bbox,
    }


def _pairwise_bbox_overlap(panel_evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for left_index in range(len(panel_evidence)):
        for right_index in range(left_index + 1, len(panel_evidence)):
            left = panel_evidence[left_index]
            right = panel_evidence[right_index]
            left_box = left["outputBoundingBox"]
            right_box = right["outputBoundingBox"]
            width = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
            height = max(0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]))
            evidence.append(
                {
                    "leftPath": left["identity"]["path"],
                    "rightPath": right["identity"]["path"],
                    "boundingBoxIntersectionPixels": width * height,
                    "hasBoundingBoxOverlap": width > 0.0 and height > 0.0,
                    "note": "bounding-box estimate; aggregate coverage is the exact backend evidence",
                }
            )
    return evidence


def _write_mosaic_fits(
    path: Path,
    science: NDArray[np.float32],
    coverage: NDArray[np.float32],
    output_wcs: WCS,
    metadata_header: fits.Header,
    *,
    panel_count: int,
    combine_function: str,
) -> None:
    base = output_wcs.to_header(relax=True)
    for key in ("OBJECT", "FILTER", "INSTRUME"):
        if key in metadata_header:
            base[key] = metadata_header[key]
    base["OAFSTATE"] = (MOSAIC_STATE, "Final mosaic astrometric solve is required")
    base["OAFWCS"] = ("PROPAGATED", "Reprojection grid; not a final plate solution")
    base["OAFPROD"] = ("PANEL_MOSAIC", "Ultra-Fast WBPP product kind")
    base["OAFVERS"] = (MOSAIC_VERSION, "Ultra-Fast WBPP mosaic version")
    base["OAFNPAN"] = (panel_count, "Number of input panels")
    base["OAFCOADD"] = (combine_function.upper(), "Panel coadd function")
    base.add_history("Ultra-Fast WBPP: propagated panel WCS defines only the reprojection grid")
    base.add_history("Ultra-Fast WBPP: run a fresh plate solve before marking this product SOLVED")
    footprint = (coverage > 0.0).astype(np.uint8)
    primary = fits.PrimaryHDU(data=np.asarray(science, dtype=np.float32), header=base)
    footprint_header = output_wcs.to_header(relax=True)
    footprint_header["OAFSTATE"] = MOSAIC_STATE
    coverage_header = footprint_header.copy()
    hdul = fits.HDUList(
        [
            primary,
            fits.ImageHDU(data=footprint, header=footprint_header, name="FOOTPRINT"),
            fits.ImageHDU(data=np.asarray(coverage, dtype=np.float32), header=coverage_header, name="COVERAGE"),
        ]
    )
    hdul.writeto(path, overwrite=False, checksum=True, output_verify="exception")
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def build_solved_panel_mosaic(
    request: MosaicRequest,
    *,
    provider: ReprojectProvider | None = None,
    publisher: DirectoryPublisher = _publish_directory_no_replace,
) -> MosaicResult:
    """Reproject solved panels and publish a working mosaic that needs a solve."""

    output = _validate_request(request)
    if provider is None:
        capability, provider = _load_reproject_provider()
        if provider is None:
            raise MosaicError("MOSAIC_BACKEND_UNAVAILABLE", capability.reason or "reproject is unavailable")
    if not all(
        callable(value)
        for value in (provider.find_optimal_celestial_wcs, provider.reproject_and_coadd)
    ):
        raise MosaicError("MOSAIC_BACKEND_INVALID", "provider does not implement the required callables")
    if request.auto_rotate and not provider.auto_rotate_available:
        raise MosaicError(
            "MOSAIC_AUTO_ROTATE_UNAVAILABLE",
            provider.auto_rotate_reason or "auto_rotate requires optional shapely",
        )
    panels = tuple(_read_panel(path) for path in request.panel_paths)
    if len({str(panel.path) for panel in panels}) != len(panels):
        raise MosaicError("DUPLICATE_PANEL", "panel paths must be unique")
    observed_filters = [str(panel.header.get("FILTER", "")).strip() for panel in panels]
    normalized_filter_values = [_normalized_filter(value) for value in observed_filters]
    if any(value is None for value in normalized_filter_values):
        raise MosaicError("PANEL_FILTER_MISSING", "every panel requires explicit FILTER metadata")
    normalized_filters = set(normalized_filter_values)
    if len(normalized_filters) != 1:
        raise MosaicError("PANEL_FILTER_MISMATCH", f"panels have different filters: {observed_filters}")
    inputs = tuple((panel.data, panel.wcs) for panel in panels)
    output_wcs, output_shape = _call_optimal_wcs(provider, inputs, request.auto_rotate)
    output_pixels = math.prod(output_shape)
    if output_pixels > request.max_output_pixels:
        raise MosaicError(
            "MOSAIC_OUTPUT_TOO_LARGE",
            f"optimal grid has {output_pixels} pixels; limit is {request.max_output_pixels}",
        )
    validation = validate_wcs_header(output_wcs.to_header(relax=True), image_shape=output_shape)
    if not validation.valid:
        raise MosaicError("OPTIMAL_WCS_INVALID", f"{validation.code}: {validation.message}")
    projected = tuple(
        _call_reproject_one(provider, panel, output_wcs, output_shape)
        for panel in panels
    )
    calibrated_inputs, corrected_projected, photometric = (
        _photometrically_calibrated_inputs(panels, projected, request)
    )
    science, coverage = _call_coadd(
        provider, calibrated_inputs, output_wcs, output_shape, request
    )
    covered = coverage > 0.0
    multiple = coverage > (1.0 + 1e-6)
    covered_fraction = float(np.count_nonzero(covered) / coverage.size)
    if covered_fraction < request.minimum_covered_fraction:
        raise MosaicError(
            "MOSAIC_COVERAGE_GATE_FAILED",
            f"covered fraction {covered_fraction:.6g} is below {request.minimum_covered_fraction:.6g}",
        )
    panel_evidence = [_panel_projection_evidence(panel, output_wcs) for panel in panels]
    for panel in panels:
        _verify_source_stat(panel)

    exact_evidence = _exact_overlap_and_seam_evidence(
        panels, corrected_projected, request, fit_photometry=False
    )
    coverage_evidence = {
        "coveredPixels": int(np.count_nonzero(covered)),
        "uncoveredPixels": int(coverage.size - np.count_nonzero(covered)),
        "coveredFraction": covered_fraction,
        "multipleContributorPixels": int(np.count_nonzero(multiple)),
        "minimumPositiveCoverage": float(np.min(coverage[covered])),
        "maximumCoverage": float(np.max(coverage)),
        "pairwiseBoundingBoxes": _pairwise_bbox_overlap(panel_evidence),
        "exact": exact_evidence,
    }
    if not exact_evidence["connected"]:
        raise MosaicError(
            "MOSAIC_OVERLAP_SEAM_GATE_FAILED",
            "panels do not form one connected graph of overlap and seam-passing edges",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise MosaicError("OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(output))
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        mosaic_path = staging / "mosaic-working.fits"
        receipt_path = staging / "receipt.json"
        _write_mosaic_fits(
            mosaic_path,
            science,
            coverage,
            output_wcs,
            panels[0].header,
            panel_count=len(panels),
            combine_function=request.combine_function,
        )
        info = mosaic_path.stat()
        artifact = {
            "path": mosaic_path.name,
            "kind": "WORKING_PANEL_MOSAIC_FITS",
            "sha256": sha256_digest(mosaic_path),
            "sizeBytes": info.st_size,
            "extensions": ["PRIMARY:SCIENCE", "FOOTPRINT", "COVERAGE"],
            "state": MOSAIC_STATE,
        }
        receipt_core: dict[str, Any] = {
            "schemaVersion": 1,
            "mosaicVersion": MOSAIC_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "backend": {"backendId": provider.backend_id, "version": provider.version},
            "state": MOSAIC_STATE,
            "requiresFinalSolve": True,
            "propagatedWcsIsFinalSolution": False,
            "panels": panel_evidence,
            "outputGrid": {
                "shape": list(output_shape),
                "wcsValidation": validation.serializable(),
                "wcsRole": "REPROJECTION_GRID_ONLY",
            },
            "coadd": {
                "combineFunction": request.combine_function,
                "matchBackground": False,
                "autoRotate": request.auto_rotate,
                "photometricNormalization": photometric,
            },
            "overlap": coverage_evidence,
            "qualityGate": {
                "status": "PASS",
                "minimumCoveredFraction": request.minimum_covered_fraction,
                "minimumPairOverlapPixels": request.minimum_pair_overlap_pixels,
                "minimumPairOverlapFraction": request.minimum_pair_overlap_fraction,
                "maximumSeamNormalizedMad": request.maximum_seam_normalized_mad,
                "requiresConnectedOverlapGraph": True,
            },
            "artifact": artifact,
            "publication": {
                "atomic": True,
                "noReplace": True,
                "receiptIsCommitMarker": True,
                "parentDirectoryFsync": "BEST_EFFORT_AFTER_COMMIT",
            },
        }
        receipt_id = "sha256:" + hashlib.sha256(canonical_json_document(receipt_core)).hexdigest()
        receipt = {"receiptId": receipt_id, **receipt_core}
        with receipt_path.open("xb") as stream:
            stream.write(canonical_json_document(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        try:
            publisher(staging, output)
        except ColorProductError as error:
            raise MosaicError(error.code, str(error), path=error.path) from error
        except MosaicError:
            raise
        except Exception as error:
            raise MosaicError("ATOMIC_PUBLICATION_FAILED", str(error), path=str(output)) from error
        _best_effort_fsync_directory(output.parent)
    finally:
        remove_tree(staging)

    return MosaicResult(
        output_directory=str(output),
        mosaic_path=str(output / "mosaic-working.fits"),
        receipt_path=str(output / "receipt.json"),
        state=MOSAIC_STATE,
        receipt=receipt,
    )


create_solved_panel_mosaic = build_solved_panel_mosaic
mosaic_capability = reproject_capability


__all__ = [
    "MOSAIC_STATE",
    "MOSAIC_VERSION",
    "MosaicCapability",
    "MosaicError",
    "MosaicRequest",
    "MosaicResult",
    "ReprojectProvider",
    "build_solved_panel_mosaic",
    "create_solved_panel_mosaic",
    "mosaic_capability",
    "reproject_capability",
]
