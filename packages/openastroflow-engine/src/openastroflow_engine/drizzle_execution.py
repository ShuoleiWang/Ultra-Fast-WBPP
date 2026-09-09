"""Portable, fail-closed CPU execution for the Drizzle stage.

This module intentionally has no dependency on PixInsight ``.xdrz`` files.  It
adapts Ultra-Fast WBPP's output-to-input registration convention to the
input-to-output pixel maps expected by :mod:`drizzle.resample`.

The optional STScI dependency is discovered at runtime.  Importing this module
therefore remains safe on installations that only need inventory/planning.  A
successful run publishes one multi-extension FITS file (``SCI``, ``WHT``, and
``COVERAGE``) and then a JSON receipt.  Both are create-only; the receipt is the
commit marker and is never present for a failed or partial run.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import errno
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from astropy.io import fits
import numpy as np


_SUPPORTED_SCALES = (1, 2, 3)
_SUPPORTED_KERNELS = (
    "square",
    "point",
    "turbo",
    "gaussian",
    "lanczos2",
    "lanczos3",
)
_MIN_PIXFRAC = 0.1
_MAX_PIXFRAC = 1.0
_MAX_TILE_ROWS = 4096
_DEFAULT_MAX_OUTPUT_PIXELS = 128 * 1024 * 1024
_DEFAULT_MAX_WORKING_SET_BYTES = 4 * 1024**3
_DEFAULT_MAX_TILE_BYTES = 256 * 1024**2
_MAPPING_RESIDUAL_TOLERANCE_PIXELS = 0.05
_DEFAULT_MINIMUM_COVERAGE_FRACTION = 0.90
_DEFAULT_MAXIMUM_NULL_FRACTION = 0.10
_DEFAULT_MINIMUM_DITHER_PHASES = 3
_DEFAULT_MINIMUM_DITHER_SEPARATION_PIXELS = 0.15
_DEFAULT_MINIMUM_DITHER_SPAN_PIXELS = 0.35
_DEFAULT_MAXIMUM_FWHM_FOR_UPSAMPLING_PIXELS = 3.0
_DITHER_PHASE_EVIDENCE_TOLERANCE_PIXELS = 0.05


class DrizzleExecutionError(RuntimeError):
    """A stable, user-actionable failure at the execution boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DrizzleExecutionCapability:
    backend_id: str
    available: bool
    execution_ready: bool
    version: str | None
    reason: str | None
    scales: tuple[int, ...] = _SUPPORTED_SCALES
    kernels: tuple[str, ...] = _SUPPORTED_KERNELS
    mapping_convention: str = "OUTPUT_TO_INPUT"
    device: str = "CPU"

    def serializable(self) -> dict[str, Any]:
        return {
            "backendId": self.backend_id,
            "available": self.available,
            "executionReady": self.execution_ready,
            "version": self.version,
            "reason": self.reason,
            "scales": list(self.scales),
            "kernels": list(self.kernels),
            "mappingConvention": self.mapping_convention,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class DrizzleProvider:
    """Injectable adapter seam around ``drizzle.resample.Drizzle``."""

    backend_id: str
    version: str
    factory: Callable[..., Any] = field(repr=False, compare=False)


HduSelector = int | str | tuple[str, int]
ProjectiveMatrix = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True, slots=True)
class DrizzleFrameInput:
    """One calibrated frame and its output-to-input registration evidence.

    Exactly one of ``output_to_input_pixmap_path`` and
    ``output_to_input_projective`` is required.  A positive value in the
    rejection mask means rejected.  Pixel weights must be finite and
    non-negative; non-finite science pixels receive zero effective weight.
    """

    calibrated_path: str
    output_to_input_pixmap_path: str | None = None
    output_to_input_projective: ProjectiveMatrix | None = None
    weight_path: str | None = None
    rejection_mask_path: str | None = None
    dither_phase: tuple[float, float] | None = None
    exposure_seconds: float = 1.0
    weight_scale: float = 1.0
    science_hdu: HduSelector = 0
    pixmap_hdu: HduSelector = 0
    weight_hdu: HduSelector = 0
    rejection_mask_hdu: HduSelector = 0


@dataclass(frozen=True, slots=True)
class DrizzleExecutionRequest:
    frames: tuple[DrizzleFrameInput, ...]
    output_path: str
    receipt_path: str
    output_shape: tuple[int, int]
    scale: int = 2
    pixfrac: float = 0.5
    kernel: str = "square"
    tile_rows: int = 256
    max_tile_bytes: int = _DEFAULT_MAX_TILE_BYTES
    max_output_pixels: int = _DEFAULT_MAX_OUTPUT_PIXELS
    max_working_set_bytes: int = _DEFAULT_MAX_WORKING_SET_BYTES
    minimum_coverage_fraction: float = _DEFAULT_MINIMUM_COVERAGE_FRACTION
    maximum_null_fraction: float = _DEFAULT_MAXIMUM_NULL_FRACTION
    minimum_distinct_dither_phases: int = _DEFAULT_MINIMUM_DITHER_PHASES
    minimum_dither_phase_separation_pixels: float = (
        _DEFAULT_MINIMUM_DITHER_SEPARATION_PIXELS
    )
    minimum_dither_span_pixels: float = _DEFAULT_MINIMUM_DITHER_SPAN_PIXELS
    median_fwhm_native_pixels: float | None = None
    maximum_fwhm_for_upsampling_pixels: float = (
        _DEFAULT_MAXIMUM_FWHM_FOR_UPSAMPLING_PIXELS
    )
    pixel_scale_arcsec: float | None = None


@dataclass(frozen=True, slots=True)
class DrizzleExecutionResult:
    completed: bool
    code: str
    message: str
    backend_id: str | None = None
    output_path: str | None = None
    receipt_path: str | None = None
    output_sha256: str | None = None
    receipt_sha256: str | None = None
    receipt: Mapping[str, Any] | None = None


@runtime_checkable
class _Accumulator(Protocol):
    out_img: np.ndarray
    out_wht: np.ndarray

    def add_image(
        self,
        data: np.ndarray,
        exptime: float,
        pixmap: np.ndarray,
        **kwargs: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int

    def serializable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "mtimeNs": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


def _selector_payload(selector: HduSelector) -> int | str | list[Any]:
    return list(selector) if isinstance(selector, tuple) else selector


def _supports_keyword(callable_object: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in parameters
    )


def _load_stsci_provider(
    importer: Callable[[str], Any] = importlib.import_module,
) -> tuple[DrizzleExecutionCapability, DrizzleProvider | None]:
    backend_id = "stsci-drizzle-cpu"
    try:
        package = importer("drizzle")
        resample = importer("drizzle.resample")
    except (ImportError, ModuleNotFoundError) as error:
        return (
            DrizzleExecutionCapability(
                backend_id=backend_id,
                available=False,
                execution_ready=False,
                version=None,
                reason=f"optional STScI drizzle dependency is unavailable: {error}",
            ),
            None,
        )
    except Exception as error:
        return (
            DrizzleExecutionCapability(
                backend_id=backend_id,
                available=False,
                execution_ready=False,
                version=None,
                reason=f"STScI drizzle discovery failed: {error}",
            ),
            None,
        )

    drizzle_class = getattr(resample, "Drizzle", None)
    version = getattr(package, "__version__", None)
    if not isinstance(version, str) or not version:
        try:
            version = importlib.metadata.version("drizzle")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    if drizzle_class is None or not callable(drizzle_class):
        reason = "drizzle.resample.Drizzle is missing"
    elif not all(
        _supports_keyword(drizzle_class, name)
        for name in ("out_shape", "kernel", "fillval")
    ):
        reason = "Drizzle constructor does not implement the required array API"
    else:
        add_image = getattr(drizzle_class, "add_image", None)
        required = (
            "data",
            "exptime",
            "pixmap",
            "weight_map",
            "wht_scale",
            "pixfrac",
            "pixel_scale_ratio",
            "in_units",
        )
        reason = (
            None
            if callable(add_image)
            and all(_supports_keyword(add_image, name) for name in required)
            else "Drizzle.add_image does not implement the required array API"
        )

    if reason is not None:
        return (
            DrizzleExecutionCapability(
                backend_id=backend_id,
                available=True,
                execution_ready=False,
                version=version,
                reason=reason,
            ),
            None,
        )
    return (
        DrizzleExecutionCapability(
            backend_id=backend_id,
            available=True,
            execution_ready=True,
            version=version,
            reason=None,
        ),
        DrizzleProvider(backend_id, version, drizzle_class),
    )


def stsci_drizzle_capability(
    importer: Callable[[str], Any] = importlib.import_module,
) -> DrizzleExecutionCapability:
    """Report availability without making planning imports depend on drizzle."""

    capability, _ = _load_stsci_provider(importer)
    return capability


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DrizzleExecutionError("INVALID_REQUEST", f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise DrizzleExecutionError("INVALID_REQUEST", f"{name} must be {qualifier}")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DrizzleExecutionError(
            "INVALID_REQUEST", f"{name} must be a positive integer"
        )
    return value


def _closed_unit_fraction(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if not 0.0 <= result <= 1.0:
        raise DrizzleExecutionError(
            "INVALID_REQUEST", f"{name} must be in the closed interval [0, 1]"
        )
    return result


def _phase(value: tuple[float, float] | None, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise DrizzleExecutionError(
            "DITHER_PHASE_INVALID", f"{name} must be a two-element tuple"
        )
    x = _finite_number(value[0], f"{name}[0]")
    y = _finite_number(value[1], f"{name}[1]")
    if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
        raise DrizzleExecutionError(
            "DITHER_PHASE_INVALID", f"{name} coordinates must be in [0, 1)"
        )
    return x, y


def _matrix(value: ProjectiveMatrix | None) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DrizzleExecutionError(
            "TRANSFORM_INVALID", f"projective transform is not numeric: {error}"
        ) from error
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise DrizzleExecutionError(
            "TRANSFORM_INVALID", "projective transform must be a finite 3x3 matrix"
        )
    norm = float(np.linalg.norm(matrix, ord=np.inf))
    determinant = float(np.linalg.det(matrix))
    if (
        norm == 0.0
        or not math.isfinite(determinant)
        or abs(determinant) <= 1e-12 * norm**3
    ):
        raise DrizzleExecutionError(
            "TRANSFORM_SINGULAR", "projective transform is singular or ill-conditioned"
        )
    return matrix


def _output_path(value: str, name: str, suffix: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DrizzleExecutionError("INVALID_REQUEST", f"{name} must be a path")
    path = Path(value).expanduser().resolve(strict=False)
    if path.suffix.lower() != suffix:
        raise DrizzleExecutionError("INVALID_REQUEST", f"{name} must end in {suffix}")
    if path.exists() or path.is_symlink():
        raise DrizzleExecutionError(
            "OUTPUT_EXISTS", f"refusing to replace existing output: {path}"
        )
    return path


def validate_drizzle_request(request: DrizzleExecutionRequest) -> None:
    """Validate all non-I/O constraints before allocating or publishing."""

    if not isinstance(request, DrizzleExecutionRequest):
        raise DrizzleExecutionError("INVALID_REQUEST", "request has the wrong type")
    if not isinstance(request.frames, tuple) or not request.frames:
        raise DrizzleExecutionError(
            "INVALID_REQUEST", "frames must be a non-empty tuple"
        )
    if len(request.frames) > np.iinfo(np.int32).max:
        raise DrizzleExecutionError("INVALID_REQUEST", "too many input frames")
    if (
        isinstance(request.scale, bool)
        or not isinstance(request.scale, int)
        or request.scale not in _SUPPORTED_SCALES
    ):
        raise DrizzleExecutionError(
            "SCALE_UNSUPPORTED",
            f"scale must be one of {_SUPPORTED_SCALES}, got {request.scale!r}",
        )
    pixfrac = _finite_number(request.pixfrac, "pixfrac")
    if not _MIN_PIXFRAC <= pixfrac <= _MAX_PIXFRAC:
        raise DrizzleExecutionError(
            "PIXFRAC_INVALID",
            f"pixfrac must be in [{_MIN_PIXFRAC}, {_MAX_PIXFRAC}]",
        )
    if not isinstance(request.kernel, str) or request.kernel not in _SUPPORTED_KERNELS:
        raise DrizzleExecutionError(
            "KERNEL_UNSUPPORTED",
            f"kernel must be one of {_SUPPORTED_KERNELS}",
        )
    if request.kernel.startswith("lanczos") and (
        request.scale != 1 or not math.isclose(pixfrac, 1.0)
    ):
        raise DrizzleExecutionError(
            "KERNEL_GEOMETRY_UNSAFE",
            "Lanczos drizzle is accepted only with scale=1 and pixfrac=1",
        )

    if not isinstance(request.output_shape, tuple) or len(request.output_shape) != 2:
        raise DrizzleExecutionError(
            "GEOMETRY_INVALID", "output_shape must be a (height, width) tuple"
        )
    height, width = request.output_shape
    _positive_integer(height, "output height")
    _positive_integer(width, "output width")
    if height < 2 or width < 2:
        raise DrizzleExecutionError(
            "GEOMETRY_INVALID", "output dimensions must each be at least 2 pixels"
        )
    max_pixels = _positive_integer(request.max_output_pixels, "max_output_pixels")
    pixels = height * width
    if pixels > max_pixels:
        raise DrizzleExecutionError(
            "GEOMETRY_TOO_LARGE",
            f"output has {pixels} pixels, exceeding limit {max_pixels}",
        )
    working_limit = _positive_integer(
        request.max_working_set_bytes, "max_working_set_bytes"
    )
    # SCI + WHT + backend scratch/context + per-frame WHT snapshot + COVERAGE.
    estimated_output_bytes = pixels * 24
    if estimated_output_bytes > working_limit:
        raise DrizzleExecutionError(
            "WORKING_SET_EXCEEDED",
            f"estimated output working set {estimated_output_bytes} exceeds "
            f"limit {working_limit}",
        )
    tile_rows = _positive_integer(request.tile_rows, "tile_rows")
    if tile_rows < 2 or tile_rows > _MAX_TILE_ROWS:
        raise DrizzleExecutionError(
            "TILE_LIMIT_INVALID",
            f"tile_rows must be between 2 and {_MAX_TILE_ROWS}",
        )
    _positive_integer(request.max_tile_bytes, "max_tile_bytes")
    minimum_coverage = _closed_unit_fraction(
        request.minimum_coverage_fraction, "minimum_coverage_fraction"
    )
    maximum_null = _closed_unit_fraction(
        request.maximum_null_fraction, "maximum_null_fraction"
    )
    if minimum_coverage < _DEFAULT_MINIMUM_COVERAGE_FRACTION:
        raise DrizzleExecutionError(
            "COVERAGE_POLICY_UNSAFE",
            f"minimum_coverage_fraction cannot be below "
            f"{_DEFAULT_MINIMUM_COVERAGE_FRACTION:.2f}",
        )
    if maximum_null > _DEFAULT_MAXIMUM_NULL_FRACTION:
        raise DrizzleExecutionError(
            "COVERAGE_POLICY_UNSAFE",
            f"maximum_null_fraction cannot exceed "
            f"{_DEFAULT_MAXIMUM_NULL_FRACTION:.2f}",
        )
    minimum_phases = _positive_integer(
        request.minimum_distinct_dither_phases,
        "minimum_distinct_dither_phases",
    )
    if minimum_phases > len(request.frames):
        raise DrizzleExecutionError(
            "DITHER_PHASES_INSUFFICIENT",
            f"at least {minimum_phases} frames are required to establish "
            "the requested number of subpixel phases",
        )
    phase_separation = _finite_number(
        request.minimum_dither_phase_separation_pixels,
        "minimum_dither_phase_separation_pixels",
    )
    if not 0.0 < phase_separation <= math.sqrt(0.5):
        raise DrizzleExecutionError(
            "INVALID_REQUEST",
            "minimum_dither_phase_separation_pixels must be in (0, sqrt(0.5)]",
        )
    phase_span = _finite_number(
        request.minimum_dither_span_pixels,
        "minimum_dither_span_pixels",
    )
    if not 0.0 <= phase_span <= 0.5:
        raise DrizzleExecutionError(
            "INVALID_REQUEST", "minimum_dither_span_pixels must be in [0, 0.5]"
        )
    maximum_fwhm = _finite_number(
        request.maximum_fwhm_for_upsampling_pixels,
        "maximum_fwhm_for_upsampling_pixels",
        positive=True,
    )
    if maximum_fwhm > _DEFAULT_MAXIMUM_FWHM_FOR_UPSAMPLING_PIXELS:
        raise DrizzleExecutionError(
            "SAMPLING_POLICY_UNSAFE",
            "maximum_fwhm_for_upsampling_pixels cannot exceed the production "
            f"ceiling {_DEFAULT_MAXIMUM_FWHM_FOR_UPSAMPLING_PIXELS:.1f}",
        )
    if request.median_fwhm_native_pixels is not None:
        _finite_number(
            request.median_fwhm_native_pixels,
            "median_fwhm_native_pixels",
            positive=True,
        )
    if request.pixel_scale_arcsec is not None:
        _finite_number(request.pixel_scale_arcsec, "pixel_scale_arcsec", positive=True)
    if request.scale > 1:
        if request.median_fwhm_native_pixels is None:
            raise DrizzleExecutionError(
                "DRIZZLE_SAMPLING_REVIEW_REQUIRED",
                "native PSF sampling is unknown; 2x/3x drizzle requires explicit "
                "QC FWHM evidence instead of assuming undersampling",
            )
        if float(request.median_fwhm_native_pixels) >= maximum_fwhm:
            raise DrizzleExecutionError(
                "DRIZZLE_UPSCALE_NOT_RECOMMENDED",
                f"median native FWHM {request.median_fwhm_native_pixels:.3f} px "
                f"is already adequately sampled (block threshold {maximum_fwhm:.3f} px)",
            )

    output_path = _output_path(request.output_path, "output_path", ".fits")
    receipt_path = _output_path(request.receipt_path, "receipt_path", ".json")
    if output_path == receipt_path:
        raise DrizzleExecutionError(
            "INVALID_REQUEST", "output_path and receipt_path must be different"
        )
    if output_path.parent != receipt_path.parent:
        raise DrizzleExecutionError(
            "ATOMIC_PUBLICATION_UNSUPPORTED",
            "output and receipt must share one parent directory",
        )

    for index, frame in enumerate(request.frames):
        if not isinstance(frame, DrizzleFrameInput):
            raise DrizzleExecutionError(
                "INVALID_REQUEST", f"frames[{index}] has the wrong type"
            )
        has_pixmap = frame.output_to_input_pixmap_path is not None
        has_projective = frame.output_to_input_projective is not None
        if has_pixmap == has_projective:
            raise DrizzleExecutionError(
                "MAPPING_REQUIRED",
                f"frames[{index}] must provide exactly one output-to-input mapping",
            )
        _finite_number(
            frame.exposure_seconds,
            f"frames[{index}].exposure_seconds",
            positive=True,
        )
        _finite_number(
            frame.weight_scale,
            f"frames[{index}].weight_scale",
            positive=True,
        )
        if has_projective:
            _matrix(frame.output_to_input_projective)
        _phase(frame.dither_phase, f"frames[{index}].dither_phase")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path_value: str, label: str) -> _SourceIdentity:
    if not isinstance(path_value, str) or not path_value.strip():
        raise DrizzleExecutionError("SOURCE_INVALID", f"{label} must be a path")
    try:
        path = Path(path_value).expanduser().resolve(strict=True)
        before = path.stat()
    except OSError as error:
        raise DrizzleExecutionError(
            "SOURCE_UNREADABLE", f"cannot stat {label} {path_value!r}: {error}"
        ) from error
    if not stat.S_ISREG(before.st_mode):
        raise DrizzleExecutionError(
            "SOURCE_INVALID", f"{label} is not a regular file: {path}"
        )
    try:
        digest = _sha256(path)
        after = path.stat()
    except OSError as error:
        raise DrizzleExecutionError(
            "SOURCE_UNREADABLE", f"cannot read {label} {path}: {error}"
        ) from error
    signature_before = (
        before.st_size,
        before.st_mtime_ns,
        before.st_dev,
        before.st_ino,
    )
    signature_after = (
        after.st_size,
        after.st_mtime_ns,
        after.st_dev,
        after.st_ino,
    )
    if signature_before != signature_after:
        raise DrizzleExecutionError(
            "SOURCE_CHANGED", f"{label} changed while its identity was computed: {path}"
        )
    return _SourceIdentity(
        str(path),
        digest,
        after.st_size,
        after.st_mtime_ns,
        after.st_dev,
        after.st_ino,
    )


def _verify_source_unchanged(identity: _SourceIdentity) -> None:
    try:
        current = Path(identity.path).stat()
    except OSError as error:
        raise DrizzleExecutionError(
            "SOURCE_CHANGED",
            f"source disappeared during execution: {identity.path}: {error}",
        ) from error
    if (
        current.st_size,
        current.st_mtime_ns,
        current.st_dev,
        current.st_ino,
    ) != (
        identity.size_bytes,
        identity.mtime_ns,
        identity.device,
        identity.inode,
    ):
        raise DrizzleExecutionError(
            "SOURCE_CHANGED", f"source changed during execution: {identity.path}"
        )


def _hdu_data(hdul: fits.HDUList, selector: HduSelector, label: str) -> np.ndarray:
    try:
        data = hdul[selector].data
    except (IndexError, KeyError, TypeError) as error:
        raise DrizzleExecutionError(
            "FITS_HDU_INVALID", f"cannot select {label} HDU {selector!r}: {error}"
        ) from error
    if data is None:
        raise DrizzleExecutionError(
            "FITS_DATA_MISSING", f"{label} HDU has no image data"
        )
    array = np.asanyarray(data)
    if array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise DrizzleExecutionError(
            "FITS_DATA_INVALID", f"{label} must contain a real numeric image"
        )
    return array


@contextmanager
def _open_fits(path: str, selector: HduSelector, label: str) -> Iterator[np.ndarray]:
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            lazy_load_hdus=False,
        ) as hdul:
            yield _hdu_data(hdul, selector, label)
    except DrizzleExecutionError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise DrizzleExecutionError(
            "FITS_READ_FAILED", f"cannot read {label} {path}: {error}"
        ) from error


def _projective_jacobian(matrix: np.ndarray, x: float, y: float) -> np.ndarray:
    numerator_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    numerator_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        raise DrizzleExecutionError(
            "TRANSFORM_INVALID", "projective transform is undefined at output center"
        )
    denominator2 = denominator * denominator
    return np.asarray(
        [
            [
                (matrix[0, 0] * denominator - numerator_x * matrix[2, 0])
                / denominator2,
                (matrix[0, 1] * denominator - numerator_x * matrix[2, 1])
                / denominator2,
            ],
            [
                (matrix[1, 0] * denominator - numerator_y * matrix[2, 0])
                / denominator2,
                (matrix[1, 1] * denominator - numerator_y * matrix[2, 1])
                / denominator2,
            ],
        ],
        dtype=np.float64,
    )


def _validate_local_scale(actual: float, scale: int, label: str) -> None:
    expected = 1.0 / float(scale)
    ratio = actual / expected
    if not math.isfinite(ratio) or not 0.65 <= ratio <= 1.35:
        raise DrizzleExecutionError(
            "MAPPING_SCALE_MISMATCH",
            f"{label} local output-to-input scale is {actual:.6g}; "
            f"expected approximately {expected:.6g} for {scale}x drizzle",
        )


def _validate_projective_geometry(
    matrix: np.ndarray,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    scale: int,
) -> None:
    out_h, out_w = output_shape
    jacobian = _projective_jacobian(matrix, (out_w - 1) / 2, (out_h - 1) / 2)
    determinant = float(np.linalg.det(jacobian))
    if not np.all(np.isfinite(jacobian)) or abs(determinant) < 1e-12:
        raise DrizzleExecutionError(
            "TRANSFORM_SINGULAR", "projective transform has a singular local Jacobian"
        )
    _validate_local_scale(math.sqrt(abs(determinant)), scale, "projective mapping")

    input_h, input_w = input_shape
    inverse = np.linalg.inv(matrix)
    input_x = np.asarray([0.0, input_w - 1.0, 0.0, input_w - 1.0])
    input_y = np.asarray([0.0, 0.0, input_h - 1.0, input_h - 1.0])
    projected = inverse @ np.vstack([input_x, input_y, np.ones(4)])
    finite = np.abs(projected[2]) > 1e-12
    if not np.any(finite):
        raise DrizzleExecutionError(
            "GEOMETRY_NO_OVERLAP", "projective input footprint is undefined"
        )
    projected_x = projected[0, finite] / projected[2, finite]
    projected_y = projected[1, finite] / projected[2, finite]
    # A bounding-box test is intentionally permissive for rotated/perspective
    # footprints: the numerical backend remains the authority on exact pixel
    # overlap, while clearly disjoint geometry fails before allocation.
    if (
        float(np.max(projected_x)) < -0.5
        or float(np.min(projected_x)) > out_w - 0.5
        or float(np.max(projected_y)) < -0.5
        or float(np.min(projected_y)) > out_h - 0.5
    ):
        raise DrizzleExecutionError(
            "GEOMETRY_NO_OVERLAP", "projective input footprint misses the output grid"
        )


def projective_input_to_output_pixmap(
    output_to_input: Sequence[Sequence[float]],
    input_shape: tuple[int, int],
    *,
    row_start: int = 0,
    row_stop: int | None = None,
) -> np.ndarray:
    """Invert an output-to-input homography for a bounded input row tile."""

    matrix = _matrix(output_to_input)  # type: ignore[arg-type]
    height, width = input_shape
    _positive_integer(height, "input height")
    _positive_integer(width, "input width")
    stop = height if row_stop is None else row_stop
    if (
        isinstance(row_start, bool)
        or isinstance(stop, bool)
        or not isinstance(row_start, int)
        or not isinstance(stop, int)
        or not 0 <= row_start < stop <= height
    ):
        raise DrizzleExecutionError(
            "GEOMETRY_INVALID", "projective tile rows are outside the input image"
        )
    inverse = np.linalg.inv(matrix)
    y, x = np.indices((stop - row_start, width), dtype=np.float64)
    y += row_start
    denominator = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    out_x = np.full_like(x, np.nan)
    out_y = np.full_like(y, np.nan)
    out_x[valid] = (
        inverse[0, 0] * x[valid] + inverse[0, 1] * y[valid] + inverse[0, 2]
    ) / denominator[valid]
    out_y[valid] = (
        inverse[1, 0] * x[valid] + inverse[1, 1] * y[valid] + inverse[1, 2]
    ) / denominator[valid]
    return np.dstack([out_x, out_y])


def _sample_dense_mapping(
    mapping: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width, _ = mapping.shape
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= width - 1.0)
        & (y >= 0.0)
        & (y <= height - 1.0)
    )
    safe_x = np.clip(np.where(inside, x, 0.0), 0.0, width - 1.0)
    safe_y = np.clip(np.where(inside, y, 0.0), 0.0, height - 1.0)
    x0 = np.minimum(np.floor(safe_x).astype(np.int64), width - 2)
    y0 = np.minimum(np.floor(safe_y).astype(np.int64), height - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    tx = safe_x - x0
    ty = safe_y - y0
    p00 = np.asarray(mapping[y0, x0], dtype=np.float64)
    p10 = np.asarray(mapping[y0, x1], dtype=np.float64)
    p01 = np.asarray(mapping[y1, x0], dtype=np.float64)
    p11 = np.asarray(mapping[y1, x1], dtype=np.float64)
    finite = (
        inside
        & np.all(np.isfinite(p00), axis=-1)
        & np.all(np.isfinite(p10), axis=-1)
        & np.all(np.isfinite(p01), axis=-1)
        & np.all(np.isfinite(p11), axis=-1)
    )
    value = (
        p00 * ((1.0 - tx) * (1.0 - ty))[..., None]
        + p10 * (tx * (1.0 - ty))[..., None]
        + p01 * ((1.0 - tx) * ty)[..., None]
        + p11 * (tx * ty)[..., None]
    )
    derivative_x = (p10 - p00) * (1.0 - ty)[..., None] + (p11 - p01) * ty[..., None]
    derivative_y = (p01 - p00) * (1.0 - tx)[..., None] + (p11 - p10) * tx[..., None]
    return value, derivative_x, derivative_y, finite


def _dense_affine_seed(mapping: np.ndarray) -> np.ndarray:
    height, width, _ = mapping.shape
    ys = np.unique(np.linspace(0, height - 1, min(height, 7)).round().astype(int))
    xs = np.unique(np.linspace(0, width - 1, min(width, 7)).round().astype(int))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    values = np.asarray(mapping[yy, xx], dtype=np.float64).reshape(-1, 2)
    design = np.column_stack(
        [
            xx.reshape(-1).astype(np.float64),
            yy.reshape(-1).astype(np.float64),
            np.ones(xx.size),
        ]
    )
    finite = np.all(np.isfinite(values), axis=1)
    if np.count_nonzero(finite) < 3:
        raise DrizzleExecutionError(
            "PIXMAP_INVALID", "dense pixmap has too few finite samples"
        )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design[finite], values[finite], rcond=None
    )
    if rank < 3:
        raise DrizzleExecutionError(
            "PIXMAP_SINGULAR", "dense pixmap cannot provide an invertible affine seed"
        )
    affine = np.asarray(
        [
            [coefficients[0, 0], coefficients[1, 0], coefficients[2, 0]],
            [coefficients[0, 1], coefficients[1, 1], coefficients[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(affine[:2, :2]))) < 1e-12:
        raise DrizzleExecutionError(
            "PIXMAP_SINGULAR", "dense pixmap has a singular affine seed"
        )
    return affine


def _validate_dense_geometry(
    mapping: np.ndarray,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    scale: int,
) -> np.ndarray:
    if mapping.ndim != 3 or mapping.shape != (*output_shape, 2):
        raise DrizzleExecutionError(
            "PIXMAP_GEOMETRY_MISMATCH",
            f"output-to-input pixmap shape {mapping.shape} does not match "
            f"{(*output_shape, 2)}",
        )
    if mapping.shape[0] < 2 or mapping.shape[1] < 2:
        raise DrizzleExecutionError(
            "PIXMAP_GEOMETRY_MISMATCH",
            "dense pixmap dimensions must each be at least 2",
        )
    affine = _dense_affine_seed(mapping)
    local_scale = math.sqrt(abs(float(np.linalg.det(affine[:2, :2]))))
    _validate_local_scale(local_scale, scale, "dense pixmap")
    input_h, input_w = input_shape
    has_overlap = False
    # Do not materialize an output-sized float64 copy merely to validate the
    # map.  Bounded row chunks also work for memory-mapped FITS arrays.
    validation_rows = max(1, min(1024, mapping.shape[0]))
    for row_start in range(0, mapping.shape[0], validation_rows):
        values = np.asarray(
            mapping[row_start : row_start + validation_rows], dtype=np.float64
        )
        overlap = (
            np.isfinite(values[..., 0])
            & np.isfinite(values[..., 1])
            & (values[..., 0] >= -0.5)
            & (values[..., 0] <= input_w - 0.5)
            & (values[..., 1] >= -0.5)
            & (values[..., 1] <= input_h - 0.5)
        )
        if np.any(overlap):
            has_overlap = True
            break
    if not has_overlap:
        raise DrizzleExecutionError(
            "GEOMETRY_NO_OVERLAP", "dense pixmap has no overlap with the input image"
        )
    return affine


def _circular_axis_distance(first: float, second: float) -> float:
    difference = abs(first - second) % 1.0
    return min(difference, 1.0 - difference)


def _phase_distance(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    return math.hypot(
        _circular_axis_distance(first[0], second[0]),
        _circular_axis_distance(first[1], second[1]),
    )


def _derived_dither_phase(
    output_to_input: np.ndarray,
    input_shape: tuple[int, int],
    scale: int,
) -> tuple[float, float]:
    """Measure the subpixel phase at the input-frame centre.

    The registration contract is output-to-input at the requested high
    resolution.  Inverting it and dividing by ``scale`` recovers the native
    reference coordinate.  Measuring the displacement at the frame centre
    remains meaningful for both affine and gently projective registrations.
    """

    height, width = input_shape
    center = np.asarray([(width - 1.0) / 2.0, (height - 1.0) / 2.0, 1.0])
    input_to_output = np.linalg.inv(output_to_input)
    projected = input_to_output @ center
    if not math.isfinite(float(projected[2])) or abs(float(projected[2])) < 1e-12:
        raise DrizzleExecutionError(
            "DITHER_PHASE_UNKNOWN",
            "registration is undefined at the frame centre",
        )
    reference = projected[:2] / (projected[2] * float(scale))
    displacement = reference - center[:2]
    if not np.all(np.isfinite(displacement)):
        raise DrizzleExecutionError(
            "DITHER_PHASE_UNKNOWN", "derived subpixel displacement is not finite"
        )
    return float(displacement[0] % 1.0), float(displacement[1] % 1.0)


def _dither_statistics(
    phases: Sequence[tuple[float, float]],
    minimum_separation: float,
) -> dict[str, Any]:
    ordered = sorted(phases)
    distinct: list[tuple[float, float]] = []
    for candidate in ordered:
        if all(
            _phase_distance(candidate, accepted) >= minimum_separation
            for accepted in distinct
        ):
            distinct.append(candidate)
    span_x = max(
        (_circular_axis_distance(a[0], b[0]) for a in phases for b in phases),
        default=0.0,
    )
    span_y = max(
        (_circular_axis_distance(a[1], b[1]) for a in phases for b in phases),
        default=0.0,
    )
    return {
        "phases": [[float(x), float(y)] for x, y in phases],
        "distinctPhaseCount": len(distinct),
        "minimumSeparationPixels": float(minimum_separation),
        "spanXPixels": float(span_x),
        "spanYPixels": float(span_y),
    }


def _validate_dither_evidence(
    request: DrizzleExecutionRequest,
    phases: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    statistics = _dither_statistics(
        phases, float(request.minimum_dither_phase_separation_pixels)
    )
    distinct = int(statistics["distinctPhaseCount"])
    if distinct < request.minimum_distinct_dither_phases:
        raise DrizzleExecutionError(
            "DITHER_PHASES_INSUFFICIENT",
            f"found {distinct} distinct subpixel phases; at least "
            f"{request.minimum_distinct_dither_phases} are required",
        )
    minimum_span = float(request.minimum_dither_span_pixels)
    if (
        float(statistics["spanXPixels"]) < minimum_span
        or float(statistics["spanYPixels"]) < minimum_span
    ):
        raise DrizzleExecutionError(
            "DITHER_SPAN_INSUFFICIENT",
            "subpixel dither phases do not span both detector axes sufficiently: "
            f"x={statistics['spanXPixels']:.3f}, y={statistics['spanYPixels']:.3f}, "
            f"required={minimum_span:.3f}",
        )
    statistics["requiredDistinctPhaseCount"] = (
        request.minimum_distinct_dither_phases
    )
    statistics["minimumSpanPixels"] = minimum_span
    statistics["status"] = "PASS"
    return statistics


def _invert_dense_pixmap_tile(
    mapping: np.ndarray,
    affine_output_to_input: np.ndarray,
    input_shape: tuple[int, int],
    row_start: int,
    row_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = input_shape
    y_target, x_target = np.indices((row_stop - row_start, width), dtype=np.float64)
    y_target += row_start
    inverse_seed = np.linalg.inv(affine_output_to_input)
    x = (
        inverse_seed[0, 0] * x_target
        + inverse_seed[0, 1] * y_target
        + inverse_seed[0, 2]
    )
    y = (
        inverse_seed[1, 0] * x_target
        + inverse_seed[1, 1] * y_target
        + inverse_seed[1, 2]
    )
    # Least-squares round-off can put exact edge pixels a few ulps outside the
    # sampled map.  Clamp Newton iterates to the domain; a genuinely
    # non-invertible/out-of-domain target still fails the final residual gate.
    map_height, map_width, _ = mapping.shape
    x = np.clip(x, 0.0, map_width - 1.0)
    y = np.clip(y, 0.0, map_height - 1.0)
    active = np.ones_like(x, dtype=bool)
    for _ in range(12):
        value, derivative_x, derivative_y, sampled = _sample_dense_mapping(
            mapping, x, y
        )
        residual_x = value[..., 0] - x_target
        residual_y = value[..., 1] - y_target
        determinant = (
            derivative_x[..., 0] * derivative_y[..., 1]
            - derivative_y[..., 0] * derivative_x[..., 1]
        )
        solvable = sampled & np.isfinite(determinant) & (np.abs(determinant) > 1e-10)
        active &= solvable
        delta_x = np.zeros_like(x)
        delta_y = np.zeros_like(y)
        delta_x[solvable] = (
            derivative_y[..., 1][solvable] * residual_x[solvable]
            - derivative_y[..., 0][solvable] * residual_y[solvable]
        ) / determinant[solvable]
        delta_y[solvable] = (
            -derivative_x[..., 1][solvable] * residual_x[solvable]
            + derivative_x[..., 0][solvable] * residual_y[solvable]
        ) / determinant[solvable]
        x[solvable] -= delta_x[solvable]
        y[solvable] -= delta_y[solvable]
        x[solvable] = np.clip(x[solvable], 0.0, map_width - 1.0)
        y[solvable] = np.clip(y[solvable], 0.0, map_height - 1.0)
        if (
            np.any(solvable)
            and float(
                np.nanmax(
                    np.maximum(np.abs(delta_x[solvable]), np.abs(delta_y[solvable]))
                )
            )
            < 1e-5
        ):
            break

    value, _, _, sampled = _sample_dense_mapping(mapping, x, y)
    residual = np.hypot(value[..., 0] - x_target, value[..., 1] - y_target)
    valid = (
        active
        & sampled
        & np.isfinite(residual)
        & (residual <= _MAPPING_RESIDUAL_TOLERANCE_PIXELS)
    )
    pixmap = np.dstack([x, y])
    pixmap[~valid] = 0.0
    return pixmap, valid


def _actual_tile_rows(request: DrizzleExecutionRequest, width: int) -> int:
    # data, weights, rejection, target coordinates, pixmap, Jacobian scratch.
    conservative_bytes_per_row = width * 80
    if conservative_bytes_per_row > request.max_tile_bytes:
        raise DrizzleExecutionError(
            "TILE_MEMORY_EXCEEDED",
            f"one input row needs approximately {conservative_bytes_per_row} bytes, "
            f"exceeding max_tile_bytes={request.max_tile_bytes}",
        )
    rows = min(
        request.tile_rows,
        max(1, request.max_tile_bytes // conservative_bytes_per_row),
    )
    if rows < 2:
        raise DrizzleExecutionError(
            "TILE_MEMORY_EXCEEDED",
            "the STScI mapping API needs at least two input rows per tile; "
            "increase max_tile_bytes",
        )
    return rows


def _row_tiles(height: int, maximum_rows: int) -> tuple[tuple[int, int], ...]:
    """Partition rows without a one-row tile rejected by the STScI API."""

    if height <= maximum_rows:
        return ((0, height),)
    tile_count = math.ceil(height / maximum_rows)
    base, remainder = divmod(height, tile_count)
    if base < 2:
        raise DrizzleExecutionError(
            "TILE_GEOMETRY_UNSUPPORTED",
            f"cannot partition {height} rows into tiles of 2..{maximum_rows} rows",
        )
    sizes = [base + 1] * remainder + [base] * (tile_count - remainder)
    tiles: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        tiles.append((start, start + size))
        start += size
    return tuple(tiles)


def _make_accumulator(
    provider: DrizzleProvider, request: DrizzleExecutionRequest
) -> _Accumulator:
    kwargs: dict[str, Any] = {
        "out_shape": request.output_shape,
        "kernel": request.kernel,
        "fillval": "NaN",
    }
    if _supports_keyword(provider.factory, "disable_ctx"):
        kwargs["disable_ctx"] = True
    try:
        accumulator = provider.factory(**kwargs)
    except Exception as error:
        raise DrizzleExecutionError(
            "BACKEND_INITIALIZATION_FAILED",
            f"cannot initialize {provider.backend_id}: {error}",
        ) from error
    add_image = getattr(accumulator, "add_image", None)
    if (
        not callable(add_image)
        or not hasattr(accumulator, "out_img")
        or not hasattr(accumulator, "out_wht")
    ):
        raise DrizzleExecutionError(
            "BACKEND_PROTOCOL_ERROR",
            "drizzle backend does not expose required arrays/method",
        )
    return accumulator


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DrizzleExecutionError(
            "RECEIPT_INVALID", f"receipt is not strict JSON: {error}"
        ) from error


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if os.name == "nt" or error.errno in {
            errno.EACCES,
            errno.EINVAL,
            errno.ENOTSUP,
        }:
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError as error:
        if not (os.name == "nt" or error.errno in {errno.EINVAL, errno.ENOTSUP}):
            raise
    finally:
        os.close(descriptor)


def _publish_new_pair(
    staged_output: Path,
    output_path: Path,
    staged_receipt: Path,
    receipt_path: Path,
) -> None:
    created_output = False
    created_receipt = False
    committed = False
    try:
        os.link(staged_output, output_path)
        created_output = True
        # Make the complete artifact durable before publishing the receipt.
        # The receipt link is deliberately the final fallible commit action: a
        # crash can lose that directory entry (safe false-negative), but cannot
        # leave a durable success marker pointing at a missing/partial FITS.
        _fsync_directory(output_path.parent)
        os.link(staged_receipt, receipt_path)
        created_receipt = True
        committed = True
    except FileExistsError as error:
        raise DrizzleExecutionError(
            "OUTPUT_EXISTS", "an output appeared while drizzle was running"
        ) from error
    except OSError as error:
        raise DrizzleExecutionError(
            "ATOMIC_PUBLICATION_FAILED",
            f"cannot publish drizzle result atomically: {error}",
        ) from error
    finally:
        if not committed:
            # Unlink only entries created by this call.  In particular, a
            # pre-existing receipt that made the second link fail is user data.
            if created_receipt:
                receipt_path.unlink(missing_ok=True)
            if created_output:
                output_path.unlink(missing_ok=True)


def _percentiles(array: np.ndarray) -> dict[str, float]:
    values = np.percentile(array.astype(np.float64, copy=False), [0, 10, 50, 90, 100])
    return {
        "p0": float(values[0]),
        "p10": float(values[1]),
        "p50": float(values[2]),
        "p90": float(values[3]),
        "p100": float(values[4]),
    }


def _execute(
    request: DrizzleExecutionRequest,
    provider: DrizzleProvider,
) -> DrizzleExecutionResult:
    validate_drizzle_request(request)
    output_path = Path(request.output_path).expanduser().resolve(strict=False)
    receipt_path = Path(request.receipt_path).expanduser().resolve(strict=False)

    identity_cache: dict[str, _SourceIdentity] = {}

    def identity(path: str, label: str) -> _SourceIdentity:
        canonical = str(Path(path).expanduser().resolve(strict=True))
        existing = identity_cache.get(canonical)
        if existing is not None:
            return existing
        value = _source_identity(path, label)
        identity_cache[canonical] = value
        return value

    frame_sources: list[dict[str, Any]] = []
    runtime_sources: list[dict[str, _SourceIdentity]] = []
    for index, frame in enumerate(request.frames):
        calibrated_identity = identity(
            frame.calibrated_path, f"frame {index} calibrated"
        )
        runtime_source = {"calibrated": calibrated_identity}
        source: dict[str, Any] = {
            "index": index,
            "calibrated": calibrated_identity.serializable(),
            "mappingKind": (
                "DENSE_OUTPUT_TO_INPUT"
                if frame.output_to_input_pixmap_path is not None
                else "PROJECTIVE_OUTPUT_TO_INPUT"
            ),
            "exposureSeconds": float(frame.exposure_seconds),
            "weightScale": float(frame.weight_scale),
            "scienceHdu": _selector_payload(frame.science_hdu),
        }
        if frame.output_to_input_pixmap_path is not None:
            mapping_identity = identity(
                frame.output_to_input_pixmap_path, f"frame {index} pixmap"
            )
            runtime_source["mapping"] = mapping_identity
            source["mapping"] = mapping_identity.serializable()
            source["mappingHdu"] = _selector_payload(frame.pixmap_hdu)
        else:
            source["projective"] = _matrix(frame.output_to_input_projective).tolist()
        if frame.weight_path is not None:
            weight_identity = identity(frame.weight_path, f"frame {index} weight")
            runtime_source["weight"] = weight_identity
            source["weight"] = weight_identity.serializable()
            source["weightHdu"] = _selector_payload(frame.weight_hdu)
        if frame.rejection_mask_path is not None:
            rejection_identity = identity(
                frame.rejection_mask_path, f"frame {index} rejection mask"
            )
            runtime_source["rejectionMask"] = rejection_identity
            source["rejectionMask"] = rejection_identity.serializable()
            source["rejectionMaskHdu"] = _selector_payload(
                frame.rejection_mask_hdu
            )
        frame_sources.append(source)
        runtime_sources.append(runtime_source)

    source_paths = {value.path for value in identity_cache.values()}
    if str(output_path) in source_paths or str(receipt_path) in source_paths:
        raise DrizzleExecutionError(
            "OUTPUT_ALIASES_SOURCE", "an output path aliases an input source"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or receipt_path.exists():
        raise DrizzleExecutionError(
            "OUTPUT_EXISTS", "refusing to replace an output created during validation"
        )

    derived_phases: list[tuple[float, float]] = []
    preflight_input_shapes: list[tuple[int, int]] = []
    for index, frame in enumerate(request.frames):
        runtime_source = runtime_sources[index]
        with ExitStack() as stack:
            science = stack.enter_context(
                _open_fits(
                    runtime_source["calibrated"].path,
                    frame.science_hdu,
                    f"frame {index} science",
                )
            )
            if science.ndim != 2 or science.shape[0] < 2 or science.shape[1] < 2:
                raise DrizzleExecutionError(
                    "FRAME_GEOMETRY_INVALID",
                    f"frame {index} science must be a two-dimensional image "
                    "with both dimensions at least 2 pixels",
                )
            input_shape = (int(science.shape[0]), int(science.shape[1]))
            preflight_input_shapes.append(input_shape)
            if frame.output_to_input_pixmap_path is not None:
                mapping = stack.enter_context(
                    _open_fits(
                        runtime_source["mapping"].path,
                        frame.pixmap_hdu,
                        f"frame {index} output-to-input pixmap",
                    )
                )
                output_to_input = _validate_dense_geometry(
                    mapping, input_shape, request.output_shape, request.scale
                )
            else:
                output_to_input = _matrix(frame.output_to_input_projective)
                _validate_projective_geometry(
                    output_to_input,
                    input_shape,
                    request.output_shape,
                    request.scale,
                )
            derived = _derived_dither_phase(
                output_to_input, input_shape, request.scale
            )
            claimed = _phase(frame.dither_phase, f"frames[{index}].dither_phase")
            if (
                claimed is not None
                and _phase_distance(claimed, derived)
                > _DITHER_PHASE_EVIDENCE_TOLERANCE_PIXELS
            ):
                raise DrizzleExecutionError(
                    "DITHER_PHASE_MISMATCH",
                    f"frame {index} claimed phase {claimed!r} disagrees with "
                    f"registration-derived phase {derived!r}",
                )
            derived_phases.append(derived)
            frame_sources[index]["ditherPhase"] = [derived[0], derived[1]]

    dither_statistics = _validate_dither_evidence(request, derived_phases)
    sampling_statistics: dict[str, Any] = {
        "status": "NOT_APPLICABLE" if request.scale == 1 else "PASS",
        "medianNativeFwhmPixels": request.median_fwhm_native_pixels,
        "maximumFwhmForUpsamplingPixels": float(
            request.maximum_fwhm_for_upsampling_pixels
        ),
        "pixelScaleArcsec": request.pixel_scale_arcsec,
    }

    accumulator = _make_accumulator(provider, request)
    try:
        initial_weight = np.asarray(accumulator.out_wht)
    except Exception as error:
        raise DrizzleExecutionError(
            "BACKEND_PROTOCOL_ERROR", f"backend weight array is unreadable: {error}"
        ) from error
    if initial_weight.shape != request.output_shape:
        raise DrizzleExecutionError(
            "BACKEND_PROTOCOL_ERROR",
            f"backend output shape {initial_weight.shape} does not match {request.output_shape}",
        )

    coverage = np.zeros(request.output_shape, dtype=np.int32)
    input_shapes: list[list[int]] = []
    accepted_input_pixels = 0
    excluded_input_pixels = 0
    rejection_mask_pixels = 0
    rejection_mask_counts = [0 for _ in request.frames]
    tiles_processed = 0

    for index, frame in enumerate(request.frames):
        runtime_source = runtime_sources[index]
        with ExitStack() as stack:
            science = stack.enter_context(
                _open_fits(
                    runtime_source["calibrated"].path,
                    frame.science_hdu,
                    f"frame {index} science",
                )
            )
            if science.ndim != 2:
                raise DrizzleExecutionError(
                    "FRAME_GEOMETRY_INVALID",
                    f"frame {index} science must be two-dimensional, got {science.shape}",
                )
            input_shape = (int(science.shape[0]), int(science.shape[1]))
            if input_shape != preflight_input_shapes[index]:
                raise DrizzleExecutionError(
                    "SOURCE_CHANGED",
                    f"frame {index} geometry changed after scientific preflight",
                )
            if input_shape[0] < 2 or input_shape[1] < 2:
                raise DrizzleExecutionError(
                    "FRAME_GEOMETRY_INVALID", f"frame {index} is smaller than 2x2"
                )
            input_shapes.append([input_shape[0], input_shape[1]])
            tile_rows = _actual_tile_rows(request, input_shape[1])

            weight = (
                stack.enter_context(
                    _open_fits(
                        runtime_source["weight"].path,
                        frame.weight_hdu,
                        f"frame {index} weight",
                    )
                )
                if "weight" in runtime_source
                else None
            )
            rejection = (
                stack.enter_context(
                    _open_fits(
                        runtime_source["rejectionMask"].path,
                        frame.rejection_mask_hdu,
                        f"frame {index} rejection mask",
                    )
                )
                if "rejectionMask" in runtime_source
                else None
            )
            for label, array in (("weight", weight), ("rejection mask", rejection)):
                if array is not None and array.shape != input_shape:
                    raise DrizzleExecutionError(
                        "AUXILIARY_GEOMETRY_MISMATCH",
                        f"frame {index} {label} shape {array.shape} does not match {input_shape}",
                    )

            dense_mapping = None
            affine_seed = None
            projective = None
            if frame.output_to_input_pixmap_path is not None:
                dense_mapping = stack.enter_context(
                    _open_fits(
                        runtime_source["mapping"].path,
                        frame.pixmap_hdu,
                        f"frame {index} output-to-input pixmap",
                    )
                )
                affine_seed = _validate_dense_geometry(
                    dense_mapping,
                    input_shape,
                    request.output_shape,
                    request.scale,
                )
            else:
                projective = _matrix(frame.output_to_input_projective)
                _validate_projective_geometry(
                    projective,
                    input_shape,
                    request.output_shape,
                    request.scale,
                )

            before_weight = np.asarray(accumulator.out_wht, dtype=np.float32).copy()
            for row_start, row_stop in _row_tiles(input_shape[0], tile_rows):
                data_tile = np.asarray(science[row_start:row_stop], dtype=np.float32)
                if weight is None:
                    weight_tile = np.ones(data_tile.shape, dtype=np.float32)
                else:
                    weight_tile = np.asarray(
                        weight[row_start:row_stop], dtype=np.float32
                    )
                    if np.any(~np.isfinite(weight_tile)) or np.any(weight_tile < 0.0):
                        raise DrizzleExecutionError(
                            "WEIGHT_INVALID",
                            f"frame {index} weight contains a non-finite or negative value",
                        )
                    weight_tile = weight_tile.copy()
                valid = np.isfinite(data_tile)
                if rejection is not None:
                    rejection_tile = np.asarray(rejection[row_start:row_stop])
                    if np.any(~np.isfinite(rejection_tile)):
                        raise DrizzleExecutionError(
                            "REJECTION_MASK_INVALID",
                            f"frame {index} rejection mask contains a non-finite value",
                        )
                    rejected_by_mask = rejection_tile != 0
                    mask_count = int(np.count_nonzero(rejected_by_mask))
                    rejection_mask_counts[index] += mask_count
                    rejection_mask_pixels += mask_count
                    valid &= ~rejected_by_mask

                if dense_mapping is not None:
                    assert affine_seed is not None
                    pixmap, mapping_valid = _invert_dense_pixmap_tile(
                        dense_mapping,
                        affine_seed,
                        input_shape,
                        row_start,
                        row_stop,
                    )
                    valid &= mapping_valid
                else:
                    assert projective is not None
                    pixmap = projective_input_to_output_pixmap(
                        projective,
                        input_shape,
                        row_start=row_start,
                        row_stop=row_stop,
                    )
                    valid &= np.all(np.isfinite(pixmap), axis=-1)

                weight_tile *= valid.astype(np.float32)
                positive = weight_tile > 0.0
                accepted_input_pixels += int(np.count_nonzero(positive))
                excluded_input_pixels += int(
                    positive.size - np.count_nonzero(positive)
                )
                if not np.any(positive):
                    continue
                clean_data = np.where(valid, data_tile, 0.0).astype(
                    np.float32, copy=False
                )
                clean_pixmap = np.where(
                    np.all(np.isfinite(pixmap), axis=-1)[..., None], pixmap, 0.0
                ).astype(np.float64, copy=False)
                try:
                    accumulator.add_image(
                        data=clean_data,
                        exptime=float(frame.exposure_seconds),
                        pixmap=clean_pixmap,
                        weight_map=weight_tile,
                        wht_scale=float(frame.weight_scale),
                        pixfrac=float(request.pixfrac),
                        pixel_scale_ratio=1.0 / float(request.scale),
                        in_units="cps",
                    )
                except Exception as error:
                    raise DrizzleExecutionError(
                        "BACKEND_EXECUTION_FAILED",
                        f"{provider.backend_id} failed on frame {index}, rows "
                        f"[{row_start}, {row_stop}): {error}",
                    ) from error
                tiles_processed += 1

            current_weight = np.asarray(accumulator.out_wht, dtype=np.float32)
            if current_weight.shape != request.output_shape:
                raise DrizzleExecutionError(
                    "BACKEND_PROTOCOL_ERROR", "backend changed output weight geometry"
                )
            increased = current_weight > np.nextafter(before_weight, np.float32(np.inf))
            coverage += increased.astype(np.int32)
        frame_sources[index]["rejection"] = {
            "maskApplied": frame.rejection_mask_path is not None,
            "maskRejectedPixels": rejection_mask_counts[index],
        }

    for source in identity_cache.values():
        _verify_source_unchanged(source)

    try:
        science_output = np.asarray(accumulator.out_img, dtype=np.float32).copy()
        weight_output = np.asarray(accumulator.out_wht, dtype=np.float32).copy()
    except Exception as error:
        raise DrizzleExecutionError(
            "BACKEND_PROTOCOL_ERROR", f"backend output arrays are unreadable: {error}"
        ) from error
    if (
        science_output.shape != request.output_shape
        or weight_output.shape != request.output_shape
    ):
        raise DrizzleExecutionError(
            "BACKEND_PROTOCOL_ERROR", "backend returned incorrect output geometry"
        )
    if np.any(~np.isfinite(weight_output)) or np.any(weight_output < 0.0):
        raise DrizzleExecutionError(
            "BACKEND_OUTPUT_INVALID", "backend returned invalid output weights"
        )
    covered = weight_output > 0.0
    if not np.any(covered):
        raise DrizzleExecutionError(
            "NO_OUTPUT_COVERAGE", "no accepted input pixel contributed to the output"
        )
    if np.any(~np.isfinite(science_output[covered])):
        raise DrizzleExecutionError(
            "BACKEND_OUTPUT_INVALID",
            "covered science output contains non-finite values",
        )
    science_output[~covered] = np.nan
    coverage[~covered] = 0
    null_pixels = int(np.count_nonzero(~covered))
    output_pixels = int(covered.size)
    covered_weights = weight_output[covered]
    coverage_fraction = float((output_pixels - null_pixels) / output_pixels)
    null_fraction = float(null_pixels / output_pixels)
    if coverage_fraction + 1e-12 < float(request.minimum_coverage_fraction):
        raise DrizzleExecutionError(
            "COVERAGE_BELOW_MINIMUM",
            f"drizzle output covers {coverage_fraction:.3%}; production minimum is "
            f"{request.minimum_coverage_fraction:.3%}",
        )
    if null_fraction > float(request.maximum_null_fraction) + 1e-12:
        raise DrizzleExecutionError(
            "NULL_FRACTION_EXCEEDED",
            f"drizzle output has {null_fraction:.3%} null pixels; production maximum is "
            f"{request.maximum_null_fraction:.3%}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".openastroflow-drizzle-", dir=output_path.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        staged_output = temporary / output_path.name
        staged_receipt = temporary / receipt_path.name
        primary = fits.PrimaryHDU(science_output)
        primary.header["EXTNAME"] = "SCI"
        primary.header["OAFPROD"] = ("DRIZZLE", "Ultra-Fast WBPP product role")
        primary.header["DRZSCALE"] = (request.scale, "Output linear scale factor")
        primary.header["PIXFRAC"] = (float(request.pixfrac), "Drizzle pixel fraction")
        primary.header["DRIZKERN"] = (request.kernel, "Drizzle kernel")
        primary.header["NINPUT"] = (len(request.frames), "Input calibrated frames")
        hdul = fits.HDUList(
            [
                primary,
                fits.ImageHDU(weight_output, name="WHT"),
                fits.ImageHDU(coverage.astype(np.int32, copy=False), name="COVERAGE"),
            ]
        )
        try:
            hdul.writeto(staged_output, overwrite=False, checksum=True)
            with staged_output.open("r+b") as stream:
                os.fsync(stream.fileno())
        except (OSError, ValueError) as error:
            raise DrizzleExecutionError(
                "OUTPUT_WRITE_FAILED", f"cannot write staged drizzle FITS: {error}"
            ) from error
        finally:
            hdul.close()
        output_digest = _sha256(staged_output)
        output_size = staged_output.stat().st_size
        receipt_core: dict[str, Any] = {
            "schemaVersion": 2,
            "stage": "drizzle",
            "status": "succeeded",
            "backend": {
                "id": provider.backend_id,
                "version": provider.version,
                "device": "CPU",
                "referenceApi": "drizzle.resample.Drizzle.add_image",
            },
            "recipe": {
                "scale": request.scale,
                "pixfrac": float(request.pixfrac),
                "kernel": request.kernel,
                "inputUnits": "cps",
            },
            "scienceGate": {
                "sampling": sampling_statistics,
                "dither": dither_statistics,
                "coverage": {
                    "status": "PASS",
                    "minimumCoverageFraction": float(
                        request.minimum_coverage_fraction
                    ),
                    "maximumNullFraction": float(request.maximum_null_fraction),
                    "observedCoverageFraction": coverage_fraction,
                    "observedNullFraction": null_fraction,
                },
            },
            "geometry": {
                "outputHeight": request.output_shape[0],
                "outputWidth": request.output_shape[1],
                "inputShapes": input_shapes,
            },
            "inputs": frame_sources,
            "execution": {
                "tileRowsMaximum": request.tile_rows,
                "maxTileBytes": request.max_tile_bytes,
                "tilesProcessed": tiles_processed,
                "sourceMutation": False,
                "publication": "atomic-create-only-receipt-last",
            },
            "statistics": {
                "inputFrames": len(request.frames),
                "acceptedInputPixels": accepted_input_pixels,
                "excludedInputPixels": excluded_input_pixels,
                "rejectionMasksProvided": sum(
                    frame.rejection_mask_path is not None for frame in request.frames
                ),
                "allFramesHaveRejectionMasks": all(
                    frame.rejection_mask_path is not None for frame in request.frames
                ),
                "rejectionMaskPixels": rejection_mask_pixels,
                "outputPixels": output_pixels,
                "coveredPixels": output_pixels - null_pixels,
                "coverageFraction": coverage_fraction,
                "nullPixels": null_pixels,
                "nullPixelFraction": null_fraction,
                "coveragePercentiles": _percentiles(coverage),
                "coveredWeightPercentiles": _percentiles(covered_weights),
            },
            "artifact": {
                "path": str(output_path),
                "mediaType": "image/fits",
                "sha256": output_digest,
                "sizeBytes": output_size,
                "extensions": ["SCI", "WHT", "COVERAGE"],
            },
        }
        receipt_id = (
            "sha256:" + hashlib.sha256(_canonical_json(receipt_core)).hexdigest()
        )
        receipt: dict[str, Any] = {"receiptId": receipt_id, **receipt_core}
        receipt_bytes = _canonical_json(receipt)
        try:
            with staged_receipt.open("xb") as stream:
                stream.write(receipt_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise DrizzleExecutionError(
                "OUTPUT_WRITE_FAILED", f"cannot write staged drizzle receipt: {error}"
            ) from error
        receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
        _publish_new_pair(staged_output, output_path, staged_receipt, receipt_path)

    return DrizzleExecutionResult(
        completed=True,
        code="DRIZZLE_SUCCEEDED",
        message="drizzle output and receipt were published",
        backend_id=provider.backend_id,
        output_path=str(output_path),
        receipt_path=str(receipt_path),
        output_sha256=output_digest,
        receipt_sha256=receipt_digest,
        receipt=receipt,
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DrizzleExecutionError(
                "RECEIPT_INVALID", f"duplicate JSON key in drizzle receipt: {key}"
            )
        value[key] = item
    return value


def _strict_json_constant(value: str) -> None:
    raise DrizzleExecutionError(
        "RECEIPT_INVALID", f"non-finite JSON constant in drizzle receipt: {value}"
    )


def _receipt_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DrizzleExecutionError("RECEIPT_INVALID", f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DrizzleExecutionError("RECEIPT_INVALID", f"{name} must be finite")
    return result


def verify_drizzle_result(
    request: DrizzleExecutionRequest,
    result: DrizzleExecutionResult,
) -> Mapping[str, Any]:
    """Independently verify a published drizzle pair before E2E promotion.

    This deliberately re-hashes the output and recomputes coverage from the
    FITS arrays.  The check prevents a changed receipt, changed coverage map,
    or a false rejection-mask claim from crossing the final publication gate.
    """

    if not result.completed:
        raise DrizzleExecutionError(
            "DRIZZLE_RESULT_INCOMPLETE", "cannot verify an incomplete drizzle result"
        )
    if not all(
        isinstance(value, str) and value
        for value in (
            result.output_path,
            result.receipt_path,
            result.output_sha256,
            result.receipt_sha256,
        )
    ):
        raise DrizzleExecutionError(
            "DRIZZLE_RESULT_INVALID", "completed result lacks artifact identities"
        )
    output_path = Path(result.output_path).resolve(strict=True)
    receipt_path = Path(result.receipt_path).resolve(strict=True)
    if output_path != Path(request.output_path).expanduser().resolve(strict=True):
        raise DrizzleExecutionError(
            "DRIZZLE_RESULT_INVALID", "result output path does not match the request"
        )
    if receipt_path != Path(request.receipt_path).expanduser().resolve(strict=True):
        raise DrizzleExecutionError(
            "DRIZZLE_RESULT_INVALID", "result receipt path does not match the request"
        )

    receipt_bytes = receipt_path.read_bytes()
    actual_receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if actual_receipt_sha != result.receipt_sha256:
        raise DrizzleExecutionError(
            "RECEIPT_IDENTITY_MISMATCH", "drizzle receipt changed after execution"
        )
    try:
        receipt = json.loads(
            receipt_bytes,
            object_pairs_hook=_strict_json_object,
            parse_constant=_strict_json_constant,
        )
    except DrizzleExecutionError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DrizzleExecutionError(
            "RECEIPT_INVALID", f"cannot decode drizzle receipt: {error}"
        ) from error
    if not isinstance(receipt, dict) or result.receipt is None:
        raise DrizzleExecutionError("RECEIPT_INVALID", "receipt must be a JSON object")
    if receipt != dict(result.receipt):
        raise DrizzleExecutionError(
            "RECEIPT_IDENTITY_MISMATCH",
            "on-disk receipt differs from the executor's in-memory receipt",
        )
    receipt_id = receipt.get("receiptId")
    core = {key: value for key, value in receipt.items() if key != "receiptId"}
    expected_receipt_id = "sha256:" + hashlib.sha256(_canonical_json(core)).hexdigest()
    if receipt_id != expected_receipt_id:
        raise DrizzleExecutionError(
            "RECEIPT_IDENTITY_MISMATCH", "drizzle receiptId is not canonical"
        )
    if receipt.get("status") != "succeeded" or receipt.get("stage") != "drizzle":
        raise DrizzleExecutionError(
            "RECEIPT_INVALID", "receipt does not describe a successful drizzle stage"
        )
    recipe = receipt.get("recipe")
    if not isinstance(recipe, dict) or (
        recipe.get("scale") != request.scale
        or recipe.get("kernel") != request.kernel
        or not math.isclose(
            _receipt_number(recipe.get("pixfrac"), "recipe.pixfrac"),
            float(request.pixfrac),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise DrizzleExecutionError(
            "RECEIPT_INVALID", "receipt recipe does not match the execution request"
        )

    artifact = receipt.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("path") != str(output_path):
        raise DrizzleExecutionError(
            "RECEIPT_INVALID", "receipt artifact path does not match the output"
        )
    actual_output_sha = _sha256(output_path)
    if (
        actual_output_sha != result.output_sha256
        or artifact.get("sha256") != actual_output_sha
        or artifact.get("sizeBytes") != output_path.stat().st_size
    ):
        raise DrizzleExecutionError(
            "ARTIFACT_IDENTITY_MISMATCH", "drizzle FITS identity changed or is unbound"
        )

    try:
        with fits.open(output_path, mode="readonly", memmap=True) as hdul:
            science = np.asarray(hdul["SCI"].data)
            weights = np.asarray(hdul["WHT"].data)
            coverage = np.asarray(hdul["COVERAGE"].data)
            if (
                science.shape != request.output_shape
                or weights.shape != request.output_shape
                or coverage.shape != request.output_shape
            ):
                raise DrizzleExecutionError(
                    "COVERAGE_RECEIPT_MISMATCH", "drizzle extensions have wrong geometry"
                )
            if (
                np.any(~np.isfinite(weights))
                or np.any(weights < 0)
                or np.any(~np.isfinite(coverage))
                or np.any(coverage < 0)
                or np.any(coverage > len(request.frames))
            ):
                raise DrizzleExecutionError(
                    "COVERAGE_RECEIPT_MISMATCH", "coverage or weight data are invalid"
                )
            covered = weights > 0
            if np.any(coverage[~covered] != 0) or np.any(~np.isfinite(science[covered])):
                raise DrizzleExecutionError(
                    "COVERAGE_RECEIPT_MISMATCH",
                    "SCI/WHT/COVERAGE extensions disagree about valid output pixels",
                )
            output_pixels = int(covered.size)
            covered_pixels = int(np.count_nonzero(covered))
            null_pixels = output_pixels - covered_pixels
            coverage_fraction = covered_pixels / output_pixels
            null_fraction = null_pixels / output_pixels
            coverage_percentiles = _percentiles(coverage)
    except DrizzleExecutionError:
        raise
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise DrizzleExecutionError(
            "ARTIFACT_INVALID", f"cannot verify drizzle FITS: {error}"
        ) from error

    statistics = receipt.get("statistics")
    if not isinstance(statistics, dict):
        raise DrizzleExecutionError("RECEIPT_INVALID", "statistics are missing")
    exact_statistics = {
        "outputPixels": output_pixels,
        "coveredPixels": covered_pixels,
        "nullPixels": null_pixels,
    }
    for name, expected in exact_statistics.items():
        if statistics.get(name) != expected:
            raise DrizzleExecutionError(
                "COVERAGE_RECEIPT_MISMATCH", f"receipt {name} is incorrect"
            )
    for name, expected in (
        ("coverageFraction", coverage_fraction),
        ("nullPixelFraction", null_fraction),
    ):
        if not math.isclose(
            _receipt_number(statistics.get(name), f"statistics.{name}"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DrizzleExecutionError(
                "COVERAGE_RECEIPT_MISMATCH", f"receipt {name} is incorrect"
            )
    if statistics.get("coveragePercentiles") != coverage_percentiles:
        raise DrizzleExecutionError(
            "COVERAGE_RECEIPT_MISMATCH", "coverage percentiles do not match the FITS map"
        )
    if (
        coverage_fraction + 1e-12 < request.minimum_coverage_fraction
        or null_fraction > request.maximum_null_fraction + 1e-12
    ):
        raise DrizzleExecutionError(
            "COVERAGE_GATE_FAILED", "verified output does not meet production coverage gates"
        )
    science_gate = receipt.get("scienceGate")
    if not isinstance(science_gate, dict):
        raise DrizzleExecutionError("RECEIPT_INVALID", "scienceGate is missing")
    coverage_gate = science_gate.get("coverage")
    sampling_gate = science_gate.get("sampling")
    dither_gate = science_gate.get("dither")
    if not all(
        isinstance(value, dict)
        for value in (coverage_gate, sampling_gate, dither_gate)
    ):
        raise DrizzleExecutionError(
            "RECEIPT_INVALID", "scienceGate evidence is incomplete"
        )
    assert isinstance(coverage_gate, dict)
    assert isinstance(sampling_gate, dict)
    assert isinstance(dither_gate, dict)
    expected_coverage_gate = (
        ("minimumCoverageFraction", request.minimum_coverage_fraction),
        ("maximumNullFraction", request.maximum_null_fraction),
        ("observedCoverageFraction", coverage_fraction),
        ("observedNullFraction", null_fraction),
    )
    if coverage_gate.get("status") != "PASS" or any(
        not math.isclose(
            _receipt_number(coverage_gate.get(name), f"scienceGate.coverage.{name}"),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name, expected in expected_coverage_gate
    ):
        raise DrizzleExecutionError(
            "COVERAGE_RECEIPT_MISMATCH", "coverage gate evidence is inconsistent"
        )
    expected_sampling_status = "NOT_APPLICABLE" if request.scale == 1 else "PASS"
    if (
        sampling_gate.get("status") != expected_sampling_status
        or sampling_gate.get("medianNativeFwhmPixels")
        != request.median_fwhm_native_pixels
        or sampling_gate.get("pixelScaleArcsec") != request.pixel_scale_arcsec
        or not math.isclose(
            _receipt_number(
                sampling_gate.get("maximumFwhmForUpsamplingPixels"),
                "scienceGate.sampling.maximumFwhmForUpsamplingPixels",
            ),
            float(request.maximum_fwhm_for_upsampling_pixels),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise DrizzleExecutionError(
            "SAMPLING_RECEIPT_MISMATCH", "sampling gate evidence is inconsistent"
        )
    if (
        dither_gate.get("status") != "PASS"
        or dither_gate.get("requiredDistinctPhaseCount")
        != request.minimum_distinct_dither_phases
        or len(dither_gate.get("phases", [])) != len(request.frames)
        or _receipt_number(
            dither_gate.get("minimumSeparationPixels"),
            "scienceGate.dither.minimumSeparationPixels",
        )
        != float(request.minimum_dither_phase_separation_pixels)
        or _receipt_number(
            dither_gate.get("minimumSpanPixels"),
            "scienceGate.dither.minimumSpanPixels",
        )
        != float(request.minimum_dither_span_pixels)
    ):
        raise DrizzleExecutionError(
            "DITHER_RECEIPT_MISMATCH", "dither gate evidence is inconsistent"
        )

    inputs = receipt.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(request.frames):
        raise DrizzleExecutionError("RECEIPT_INVALID", "input bindings are incomplete")
    total_mask_pixels = 0
    masks_provided = 0
    for index, (frame, source) in enumerate(zip(request.frames, inputs, strict=True)):
        if not isinstance(source, dict):
            raise DrizzleExecutionError("RECEIPT_INVALID", "input binding is malformed")
        if (
            source.get("index") != index
            or not math.isclose(
                _receipt_number(
                    source.get("exposureSeconds"),
                    f"inputs[{index}].exposureSeconds",
                ),
                float(frame.exposure_seconds),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _receipt_number(
                    source.get("weightScale"), f"inputs[{index}].weightScale"
                ),
                float(frame.weight_scale),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise DrizzleExecutionError(
                "RECEIPT_INVALID", f"frame {index} scalar inputs are unbound"
            )
        calibrated = source.get("calibrated")
        if (
            not isinstance(calibrated, dict)
            or calibrated.get("path")
            != str(Path(frame.calibrated_path).expanduser().resolve(strict=True))
            or source.get("scienceHdu") != _selector_payload(frame.science_hdu)
        ):
            raise DrizzleExecutionError(
                "RECEIPT_INVALID", f"frame {index} calibrated identity is unbound"
            )
        if frame.output_to_input_pixmap_path is None:
            expected_projective = _matrix(
                frame.output_to_input_projective
            ).tolist()
            if (
                source.get("mappingKind") != "PROJECTIVE_OUTPUT_TO_INPUT"
                or source.get("projective") != expected_projective
                or "mapping" in source
            ):
                raise DrizzleExecutionError(
                    "RECEIPT_INVALID", f"frame {index} projective mapping is unbound"
                )
        else:
            expected_mapping = str(
                Path(frame.output_to_input_pixmap_path).expanduser().resolve(strict=True)
            )
            mapping_identity = source.get("mapping")
            if (
                source.get("mappingKind") != "DENSE_OUTPUT_TO_INPUT"
                or not isinstance(mapping_identity, dict)
                or mapping_identity.get("path") != expected_mapping
                or source.get("mappingHdu") != _selector_payload(frame.pixmap_hdu)
                or "projective" in source
            ):
                raise DrizzleExecutionError(
                    "RECEIPT_INVALID", f"frame {index} dense mapping is unbound"
                )
        if frame.weight_path is None:
            if "weight" in source or "weightHdu" in source:
                raise DrizzleExecutionError(
                    "RECEIPT_INVALID", f"frame {index} falsely claims a weight map"
                )
        else:
            weight_identity = source.get("weight")
            if (
                not isinstance(weight_identity, dict)
                or weight_identity.get("path")
                != str(Path(frame.weight_path).expanduser().resolve(strict=True))
                or source.get("weightHdu") != _selector_payload(frame.weight_hdu)
            ):
                raise DrizzleExecutionError(
                    "RECEIPT_INVALID", f"frame {index} weight identity is unbound"
                )
        rejection = source.get("rejection")
        if not isinstance(rejection, dict):
            raise DrizzleExecutionError(
                "RECEIPT_INVALID", f"frame {index} rejection evidence is missing"
            )
        if frame.rejection_mask_path is None:
            if (
                "rejectionMask" in source
                or "rejectionMaskHdu" in source
                or rejection.get("maskApplied") is not False
                or rejection.get("maskRejectedPixels") != 0
            ):
                raise DrizzleExecutionError(
                    "REJECTION_RECEIPT_MISMATCH",
                    f"frame {index} falsely claims rejection-mask evidence",
                )
            continue
        masks_provided += 1
        mask_path = Path(frame.rejection_mask_path).expanduser().resolve(strict=True)
        mask_identity = source.get("rejectionMask")
        if (
            not isinstance(mask_identity, dict)
            or mask_identity.get("path") != str(mask_path)
            or source.get("rejectionMaskHdu")
            != _selector_payload(frame.rejection_mask_hdu)
            or mask_identity.get("sha256") != _sha256(mask_path)
            or mask_identity.get("sizeBytes") != mask_path.stat().st_size
            or rejection.get("maskApplied") is not True
        ):
            raise DrizzleExecutionError(
                "REJECTION_RECEIPT_MISMATCH",
                f"frame {index} rejection mask identity is invalid",
            )
        with _open_fits(
            str(mask_path), frame.rejection_mask_hdu, f"frame {index} rejection mask"
        ) as mask:
            actual_count = int(np.count_nonzero(mask != 0))
        if rejection.get("maskRejectedPixels") != actual_count:
            raise DrizzleExecutionError(
                "REJECTION_RECEIPT_MISMATCH",
                f"frame {index} rejection count does not match its mask",
            )
        total_mask_pixels += actual_count
    if (
        statistics.get("rejectionMasksProvided") != masks_provided
        or statistics.get("allFramesHaveRejectionMasks")
        is not (masks_provided == len(request.frames))
        or statistics.get("rejectionMaskPixels") != total_mask_pixels
    ):
        raise DrizzleExecutionError(
            "REJECTION_RECEIPT_MISMATCH", "aggregate rejection evidence is inconsistent"
        )
    return receipt


def execute_drizzle(
    request: DrizzleExecutionRequest,
    *,
    provider: DrizzleProvider | None = None,
) -> DrizzleExecutionResult:
    """Execute a portable CPU drizzle run without ever overwriting outputs.

    Errors are returned as ``completed=False`` results so a worker boundary can
    serialize them.  No failure result includes an output or receipt path, and
    the function never converts file existence or a backend return value into a
    success claim.
    """

    chosen = provider
    if chosen is None:
        capability, chosen = _load_stsci_provider()
        if chosen is None:
            return DrizzleExecutionResult(
                completed=False,
                code="BACKEND_UNAVAILABLE",
                message=capability.reason or "STScI drizzle backend is unavailable",
                backend_id=capability.backend_id,
            )
    try:
        return _execute(request, chosen)
    except DrizzleExecutionError as error:
        return DrizzleExecutionResult(
            completed=False,
            code=error.code,
            message=str(error),
            backend_id=chosen.backend_id,
        )
    except Exception:
        # Keep unexpected implementation details out of a machine protocol and
        # never imply that a result exists.  Callers may log the traceback at a
        # separate trusted diagnostic boundary.
        return DrizzleExecutionResult(
            completed=False,
            code="INTERNAL_ERROR",
            message="unexpected drizzle execution failure",
            backend_id=chosen.backend_id,
        )


__all__ = [
    "DrizzleExecutionCapability",
    "DrizzleExecutionError",
    "DrizzleExecutionRequest",
    "DrizzleExecutionResult",
    "DrizzleFrameInput",
    "DrizzleProvider",
    "execute_drizzle",
    "projective_input_to_output_pixmap",
    "stsci_drizzle_capability",
    "validate_drizzle_request",
    "verify_drizzle_result",
]
