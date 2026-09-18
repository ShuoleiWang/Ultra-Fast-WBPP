"""Bounded-memory FITS calibration and robust vertical-slice integration.

This low-level kernel is intentionally FITS-only. The public pixel pipeline
converts supported XISF images through the bounded, content-bound private
staging bridge before entering this module. All sources are opened read-only
and every destination is a new, atomically published file.
"""

from __future__ import annotations

from .calibration_policy import bias_from_header

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray

from .native_kernels import (
    MAD_KERNEL_ID,
    MEAN_KERNEL_ID,
    default_kernel_threads,
    load_native_kernels,
)
from .transient_rejection import TransientRejectionModel, detect_transient_trails
from .residual_background import fit_residual_background
from .robust_statistics import nanmedian_frames


FITS_BLOCK_BYTES = 2880
DEFAULT_MEMORY_BUDGET = 256 * 1024 * 1024
REJECTION_FLOOR_ALGORITHM = "fixed-grid-group-mad-plus-float32-ulp-v1"
REJECTION_FLOOR_MAX_SAMPLES = 65_536
# Rows evaluated per read while binning frames for spatial transient
# detection; a performance knob only, the block statistics do not depend on it.
TRANSIENT_BAND_ROWS = 128
REJECTION_FLOOR_GROUP_FRACTION = 0.05
REJECTION_FLOOR_ABSOLUTE = 1.0e-7
REJECTION_FLOOR_EPSILON_FACTOR = 16.0
NUMPY_MAD_KERNEL_ID = "numpy-nanmedian-pooled-mad-v2"
NUMPY_MEAN_KERNEL_ID = "numpy-float64-weighted-mean-v1"
# Rejection scale model: the per-pixel MAD of a stack of N frames has a
# relative error of ~1.2/sqrt(N) (30% at N = 11), and the pixels where it
# comes out low clip 1% of their good samples, which costs 2-3% of the master's
# SNR at N = 11-13.  The noise part of every pixel's scale is therefore taken
# from the median of the per-pixel MADs over a window of 2*12+1 pixels of the
# same row (25 N samples); a frame whose per-pixel noise exceeds the group's
# sampled mixture scale (what a per-pixel MAD over all frames measures) is
# judged against its own noise, never against a tighter one; and excess
# per-pixel variance beyond the pooled noise (star cores under variable
# seeing, registration residuals) is kept per pixel.  The v2 threshold is
# therefore never below the v1 threshold: v2 rejects a subset of v1's samples.
REJECTION_SCALE_ALGORITHM = "row-pooled-mad-frame-studentized-v2"
REJECTION_POOL_HALF_WIDTH = 12
REJECTION_POOL_MAX_HALF_WIDTH = 64
REJECTION_FRAME_SCALE_DIGITS = 6
# Noise weights: 1/sigma^2 of the noise of 4x4 block means.  Per-pixel noise
# on a registered frame depends on the sub-pixel phase of its Lanczos-3
# resampling (the kernel's sum of squared weights is 0.62-1.0), so per-pixel
# sigma would weight frames by their dither phase; block means recover the
# low-frequency noise that the integration actually averages (within 2-3% of
# the unresampled value at every phase).
NOISE_WEIGHT_ALGORITHM = "block-mean-effective-noise-v2"
NOISE_WEIGHT_BLOCK_SIZE = 4
NOISE_WEIGHT_MINIMUM_DIFFERENCES = 64
# Lanczos-3 tap constants for offsets k = -2..3, evaluated through exact
# trigonometric identities: sin(pi(f-k)) = (-1)^k sin(pi f) and
# sin(pi(f-k)/3) = sin(pi f/3) cos(k pi/3) - cos(pi f/3) sin(k pi/3).
# The same literals appear in the native kernel; both paths must agree.
LANCZOS3_TAP_OFFSETS = (-2, -1, 0, 1, 2, 3)
_SQRT3_HALF = 0.8660254037844386
LANCZOS3_TAP_SIGNS = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
LANCZOS3_TAP_COSINES = (-0.5, 0.5, 1.0, 0.5, -0.5, -1.0)
LANCZOS3_TAP_SINES = (-_SQRT3_HALF, -_SQRT3_HALF, 0.0, _SQRT3_HALF, _SQRT3_HALF, 0.0)


class CalibrationError(RuntimeError):
    """Stable fail-closed error raised by the portable pixel path."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        self.code = code
        self.path = path
        detail = f"{path}: {message}" if path else message
        super().__init__(f"{code}: {detail}")


def _plain_header_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _number(header: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = header.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _text(header: Mapping[str, Any], *keys: str, default: str = "UNKNOWN") -> str:
    for key in keys:
        value = header.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().strip("'\"").strip().upper()
    return default


def normalize_role(value: Any) -> str:
    compact = re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())
    return {
        "BIAS": "BIAS",
        "BIASFRAME": "BIAS",
        "ZERO": "BIAS",
        "DARK": "DARK",
        "DARKFRAME": "DARK",
        "FLAT": "FLAT",
        "FLATFRAME": "FLAT",
        "LIGHT": "LIGHT",
        "LIGHTFRAME": "LIGHT",
        "MASTERBIAS": "MASTER_BIAS",
        "MASTERBIASFRAME": "MASTER_BIAS",
        "MASTERDARK": "MASTER_DARK",
        "MASTERDARKFRAME": "MASTER_DARK",
        "MASTERFLAT": "MASTER_FLAT",
        "MASTERFLATFRAME": "MASTER_FLAT",
    }.get(compact, "UNKNOWN")


def _numeric_domain_from_header(
    header: Mapping[str, Any],
) -> tuple[str, float | None, str]:
    declared = str(header.get("OAFNDOM", "")).strip().upper()
    declared_scale = _number(header, "OAFNSCL")
    if declared:
        if declared_scale is None or declared_scale <= 0:
            return "UNDECLARED", None, "UNRESOLVED"
        return declared, declared_scale, "SELF_DECLARED_HEADER"
    bitpix_value = _number(header, "BITPIX")
    bitpix = int(bitpix_value) if bitpix_value is not None else 0
    raw_bunit = header.get("BUNIT")
    if raw_bunit is not None and str(raw_bunit).strip():
        bunit = re.sub(r"[^A-Z0-9]+", "", str(raw_bunit).upper())
        if bunit not in {"ADU", "DN", "COUNT", "COUNTS", "CODE", "CODES"}:
            return "UNDECLARED", None, "FITS_BUNIT_UNSUPPORTED"
    bzero = _number(header, "BZERO") or 0.0
    declared_bscale = _number(header, "BSCALE")
    bscale = declared_bscale if declared_bscale is not None else 1.0
    if bitpix in {8, 16, 32, 64} and bscale > 0:
        if bitpix == 8:
            stored_low, stored_high = 0.0, 255.0
        else:
            stored_low = -float(1 << (bitpix - 1))
            stored_high = float((1 << (bitpix - 1)) - 1)
        physical_low = bscale * stored_low + bzero
        physical_high = bscale * stored_high + bzero
        if (
            math.isfinite(physical_low)
            and math.isfinite(physical_high)
            and math.isclose(physical_low, 0.0, rel_tol=0.0, abs_tol=1e-9)
            and physical_high > physical_low
        ):
            return (
                f"INTEGER_{bitpix}_PHYSICAL_0_BASED",
                physical_high,
                "FITS_STORAGE_ENDPOINTS",
            )
    return "UNDECLARED", None, "UNRESOLVED"


def _numeric_domain_evidence_from_header(
    header: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    bitpix_value = _number(header, "BITPIX")
    bitpix = int(bitpix_value) if bitpix_value is not None else None
    bscale = _number(header, "BSCALE")
    if bscale is None:
        bscale = 1.0
    bzero = _number(header, "BZERO")
    if bzero is None:
        bzero = 0.0
    evidence: dict[str, Any] = {
        "container": "FITS",
        "BITPIX": bitpix,
        "BSCALE": bscale,
        "BZERO": bzero,
        "BUNIT": (
            str(header.get("BUNIT")).strip()
            if header.get("BUNIT") is not None
            else None
        ),
    }
    if bitpix in {8, 16, 32, 64} and math.isfinite(bscale) and math.isfinite(bzero):
        if bitpix == 8:
            stored_low, stored_high = 0.0, 255.0
        else:
            stored_low = -float(1 << (bitpix - 1))
            stored_high = float((1 << (bitpix - 1)) - 1)
        evidence["derivedPhysicalLow"] = bscale * stored_low + bzero
        evidence["derivedPhysicalHigh"] = bscale * stored_high + bzero
    sample_format = header.get("OAFXSFMT")
    bounds = header.get("OAFXBD")
    if sample_format is not None:
        evidence["sourceSampleFormat"] = str(sample_format).strip()
    if bounds is not None:
        evidence["sourceBounds"] = str(bounds).strip()
    return tuple(evidence.items())


@dataclass(frozen=True, slots=True)
class FrameInfo:
    path: str
    role: str
    shape: tuple[int, int]
    filter_name: str
    exposure_seconds: float | None
    temperature_celsius: float | None
    camera: str
    gain: float | None
    offset: float | None
    binning_x: int | None
    binning_y: int | None
    cfa_pattern: str
    readout_mode: str
    target: str
    numeric_domain: str = "UNDECLARED"
    normalized_unit_scale: float | None = None
    numeric_domain_authority: str = "UNRESOLVED"
    numeric_domain_evidence: tuple[tuple[str, Any], ...] = ()
    bias_included: bool | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "shape": list(self.shape),
            "filter": self.filter_name,
            "exposureSeconds": self.exposure_seconds,
            "temperatureCelsius": self.temperature_celsius,
            "camera": self.camera,
            "gain": self.gain,
            "offset": self.offset,
            "binning": [self.binning_x, self.binning_y],
            "cfaPattern": self.cfa_pattern,
            "readoutMode": self.readout_mode,
            "target": self.target,
            "numericDomain": self.numeric_domain,
            "normalizedUnitScale": self.normalized_unit_scale,
            "numericDomainAuthority": self.numeric_domain_authority,
            "numericDomainEvidence": dict(self.numeric_domain_evidence),
            "biasIncluded": self.bias_included,
        }


class FitsFrame:
    """Read-only, manually scaled view of one uncompressed 2-D FITS image."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        name = self.path.name.casefold()
        if not name.endswith((".fit", ".fits", ".fts")):
            raise CalibrationError(
                "PIXEL_FORMAT_UNSUPPORTED",
                "portable pixels currently require uncompressed .fit/.fits/.fts",
                path=str(self.path),
            )
        self._hdul: fits.HDUList | None = None
        self._data: NDArray[Any] | None = None
        self.header: fits.Header | None = None
        self.shape: tuple[int, int] = (0, 0)
        self._bscale = 1.0
        self._bzero = 0.0
        self._blank: int | None = None

    def __enter__(self) -> FitsFrame:
        try:
            self._hdul = fits.open(
                self.path,
                mode="readonly",
                memmap=True,
                lazy_load_hdus=True,
                do_not_scale_image_data=True,
                uint=False,
                checksum=False,
            )
            image_hdu: fits.ImageHDU | fits.PrimaryHDU | None = None
            for hdu in self._hdul:
                if isinstance(hdu, fits.CompImageHDU):
                    if int(hdu.header.get("ZNAXIS", 0)) > 0:
                        raise CalibrationError(
                            "COMPRESSED_FITS_UNBOUNDED",
                            "compressed FITS decoding is disabled in the bounded-memory path",
                            path=str(self.path),
                        )
                    continue
                if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                    naxis = int(hdu.header.get("NAXIS", 0))
                    if naxis > 0:
                        image_hdu = hdu
                        break
            if image_hdu is None:
                raise CalibrationError(
                    "FITS_IMAGE_MISSING", "no image HDU found", path=str(self.path)
                )
            if int(image_hdu.header.get("NAXIS", 0)) != 2:
                raise CalibrationError(
                    "FITS_GEOMETRY_UNSUPPORTED",
                    "portable pixels require a two-dimensional mono/CFA image",
                    path=str(self.path),
                )
            height = int(image_hdu.header.get("NAXIS2", 0))
            width = int(image_hdu.header.get("NAXIS1", 0))
            if width < 1 or height < 1:
                raise CalibrationError(
                    "FITS_GEOMETRY_INVALID", "invalid image dimensions", path=str(self.path)
                )
            self._data = image_hdu.data
            if self._data is None or self._data.shape != (height, width):
                raise CalibrationError(
                    "FITS_DATA_INVALID",
                    "image data does not match its header",
                    path=str(self.path),
                )
            self.header = image_hdu.header.copy()
            self.shape = (height, width)
            self._bscale = float(image_hdu.header.get("BSCALE", 1.0))
            self._bzero = float(image_hdu.header.get("BZERO", 0.0))
            blank = image_hdu.header.get("BLANK")
            self._blank = int(blank) if blank is not None else None
            return self
        except CalibrationError:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise CalibrationError(
                "FITS_OPEN_FAILED", str(error), path=str(self.path)
            ) from error

    def close(self) -> None:
        self._data = None
        if self._hdul is not None:
            self._hdul.close()
            self._hdul = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def info(self) -> FrameInfo:
        if self.header is None:
            raise RuntimeError("FitsFrame is not open")
        header = self.header
        (
            numeric_domain,
            normalized_unit_scale,
            numeric_domain_authority,
        ) = _numeric_domain_from_header(header)
        shared_bin = _number(header, "BINNING")
        bin_x = _number(header, "XBINNING", "CCDBINX") or shared_bin
        bin_y = _number(header, "YBINNING", "CCDBINY") or shared_bin
        return FrameInfo(
            path=str(self.path),
            role=normalize_role(header.get("IMAGETYP", header.get("IMAGETYPE"))),
            shape=self.shape,
            filter_name=_text(header, "FILTER", "INSFLNAM", "FILTERID"),
            exposure_seconds=_number(header, "EXPTIME", "EXPOSURE", "EXPOSURETIME"),
            temperature_celsius=_number(
                header, "CCD-TEMP", "CCD_TEMP", "SENSORT", "SENSOR-T", "CAMTEMP"
            ),
            camera=_text(header, "INSTRUME", "CAMERA", "DETECTOR"),
            gain=_number(header, "GAIN", "EGAIN", "CAMGAIN"),
            offset=_number(header, "OFFSET", "CAMOFFSET"),
            binning_x=int(bin_x) if bin_x is not None else None,
            binning_y=int(bin_y) if bin_y is not None else None,
            cfa_pattern=_text(
                header, "BAYERPAT", "BAYERPATN", "CFAPAT", "CFAPATTERN"
            ),
            readout_mode=_text(
                header, "READOUTM", "READOUT", "READMODE", "READOUTMODE"
            ),
            target=_text(header, "OBJECT", "OBJNAME", "TARGET"),
            numeric_domain=numeric_domain,
            normalized_unit_scale=normalized_unit_scale,
            numeric_domain_authority=numeric_domain_authority,
            numeric_domain_evidence=_numeric_domain_evidence_from_header(header),
            bias_included=bias_from_header(header),
        )

    def read_rows(self, y0: int, y1: int) -> NDArray[np.float32]:
        if self._data is None:
            raise RuntimeError("FitsFrame is not open")
        if y0 < 0 or y1 > self.shape[0] or y0 >= y1:
            raise ValueError("invalid row interval")
        raw = self._data[y0:y1]
        return self._physical_values(raw)

    def read_sampled_rows(self, y_indices: NDArray[np.int64]) -> NDArray[np.float32]:
        """Decode the listed rows in one gather; values equal per-row reads."""

        if self._data is None:
            raise RuntimeError("FitsFrame is not open")
        rows = np.asarray(y_indices, dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0 or np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise ValueError("invalid sampled rows")
        return self._physical_values(self._data[rows])

    def _physical_values(self, raw: NDArray[Any]) -> NDArray[np.float32]:
        # Apply the FITS storage transform before rounding to the output type.
        # In particular, signed storage for uint32 uses BZERO=2**31: casting
        # first erases low ADU values through catastrophic cancellation.
        if (
            raw.dtype.kind == "i"
            and raw.dtype.itemsize == 8
            and self._bscale == 1.0
            and self._bzero == float(1 << 63)
        ):
            # Even float64 cannot retain low ADUs next to 2**63. Undo this
            # standard unsigned convention exactly in integer arithmetic.
            unsigned = raw.astype(np.uint64)
            unsigned ^= np.uint64(1 << 63)
            values = unsigned.astype(np.float32)
        elif self._bscale == 1.0 and (
            self._bzero == 0.0
            or (
                raw.dtype.kind == "i"
                and raw.dtype.itemsize == 2
                and self._bzero == 32768.0
            )
        ):
            # Preserve the common unscaled float/uint16 fast path; all uint16
            # storage values and its offset are exactly representable here.
            values = np.array(raw, dtype=np.float32, copy=True)
            if self._bzero != 0.0:
                values += np.float32(self._bzero)
        else:
            physical = np.array(raw, dtype=np.float64, copy=True)
            physical *= self._bscale
            physical += self._bzero
            values = physical.astype(np.float32)
        if self._blank is not None:
            values[np.asarray(raw) == self._blank] = np.nan
        return values

    def _physical_index(
        self, y: NDArray[np.int64], x: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        if self._data is None:
            raise RuntimeError("FitsFrame is not open")
        raw = self._data[y, x]
        return self._physical_values(raw)

    def full_values(self) -> NDArray[np.float32]:
        """Decode the complete image to native Float32 physical values."""

        return self.read_rows(0, self.shape[0])

    def sample_bilinear(
        self,
        x_coordinates: NDArray[Any],
        y_coordinates: NDArray[Any],
    ) -> NDArray[np.float32]:
        """Sample a bounded coordinate tile without decoding the full image."""

        return _sample_bilinear_from(self, x_coordinates, y_coordinates)

    def sample_lanczos3_clamped(
        self,
        x_coordinates: NDArray[Any],
        y_coordinates: NDArray[Any],
    ) -> NDArray[np.float32]:
        """Sample with a normalized, bounded six-tap Lanczos-3 kernel.

        See :func:`_sample_lanczos3_clamped_from` for the contract; the same
        implementation serves file-backed and in-memory frames.
        """

        return _sample_lanczos3_clamped_from(self, x_coordinates, y_coordinates)


class _MemoryFrame:
    """In-memory Float32 physical image with the FitsFrame sampling interface.

    Fused calibration keeps a calibrated Light in memory and registers it
    directly, so the resamplers below accept either a FitsFrame or this
    object.  ``read_rows`` returns copies because expression evaluation
    mutates the rows it receives.
    """

    def __init__(
        self,
        values: NDArray[Any],
        info: FrameInfo,
        path: str | os.PathLike[str] | None = None,
    ) -> None:
        array = np.ascontiguousarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError("memory frames require a nonempty two-dimensional image")
        self._values = array
        self.shape: tuple[int, int] = (int(array.shape[0]), int(array.shape[1]))
        self.info = info
        self.path = Path(path) if path is not None else Path(info.path)

    def __enter__(self) -> _MemoryFrame:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def values(self) -> NDArray[np.float32]:
        return self._values

    def read_rows(self, y0: int, y1: int) -> NDArray[np.float32]:
        if y0 < 0 or y1 > self.shape[0] or y0 >= y1:
            raise ValueError("invalid row interval")
        return np.array(self._values[y0:y1], dtype=np.float32, copy=True)

    def read_sampled_rows(self, y_indices: NDArray[np.int64]) -> NDArray[np.float32]:
        rows = np.asarray(y_indices, dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0 or np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise ValueError("invalid sampled rows")
        return np.array(self._values[rows], dtype=np.float32, copy=True)

    def full_values(self) -> NDArray[np.float32]:
        return self._values

    def _physical_index(
        self, y: NDArray[np.int64], x: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        return self._values[y, x]

    def sample_bilinear(
        self,
        x_coordinates: NDArray[Any],
        y_coordinates: NDArray[Any],
    ) -> NDArray[np.float32]:
        return _sample_bilinear_from(self, x_coordinates, y_coordinates)

    def sample_lanczos3_clamped(
        self,
        x_coordinates: NDArray[Any],
        y_coordinates: NDArray[Any],
    ) -> NDArray[np.float32]:
        return _sample_lanczos3_clamped_from(self, x_coordinates, y_coordinates)


def _sample_bilinear_from(
    frame: Any,
    x_coordinates: NDArray[Any],
    y_coordinates: NDArray[Any],
) -> NDArray[np.float32]:
    """Sample a bounded coordinate tile without decoding the full image."""

    x = np.asarray(x_coordinates, dtype=np.float64)
    y = np.asarray(y_coordinates, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("coordinate tiles must be same-shaped two-dimensional arrays")
    height, width = frame.shape
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= width - 1)
        & (y >= 0.0)
        & (y <= height - 1)
    )
    result = np.full(x.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return result
    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(np.int64)
    y0 = np.floor(yv).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (xv - x0).astype(np.float32)
    dy = (yv - y0).astype(np.float32)
    v00 = frame._physical_index(y0, x0)
    v10 = frame._physical_index(y0, x1)
    v01 = frame._physical_index(y1, x0)
    v11 = frame._physical_index(y1, x1)
    weights = (
        (1.0 - dx) * (1.0 - dy),
        dx * (1.0 - dy),
        (1.0 - dx) * dy,
        dx * dy,
    )
    samples = np.zeros(v00.shape, dtype=np.float32)
    sample_valid = np.ones(v00.shape, dtype=bool)
    for neighbor, weight in zip((v00, v10, v01, v11), weights, strict=True):
        active = weight > 0
        finite_neighbor = np.isfinite(neighbor)
        sample_valid &= ~active | finite_neighbor
        samples += np.where(active & finite_neighbor, neighbor * weight, 0.0)
    selected = result[valid]
    selected[sample_valid] = samples[sample_valid]
    result[valid] = selected
    return result


def _sample_lanczos3_clamped_from(
    frame: Any,
    x_coordinates: NDArray[Any],
    y_coordinates: NDArray[Any],
) -> NDArray[np.float32]:
    """Sample with a normalized, bounded six-tap Lanczos-3 kernel.

    The separable kernel is evaluated from first principles and normalized
    independently on both axes so a constant field remains constant.  A
    sample is valid only when the complete nonzero support is available;
    any nonfinite pixel carrying nonzero weight invalidates the result.

    Lanczos' negative lobes retain substantially more stellar detail than
    bilinear interpolation, but can create new out-of-domain extrema around
    saturated or defective pixels.  The final bound is the union of the
    declared physical domain and the finite 6x6 source support.  This clips
    interpolation-created excursions without clipping negative or
    over-range values that were already present in the calibrated input,
    and without applying any post-registration sharpening.

    The native ``warp_lanczos3`` kernel reproduces this arithmetic value for
    value; this NumPy implementation remains the portable reference.
    """

    x = np.asarray(x_coordinates, dtype=np.float64)
    y = np.asarray(y_coordinates, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("coordinate tiles must be same-shaped two-dimensional arrays")
    height, width = frame.shape
    domain_scale = frame.info.normalized_unit_scale
    if (
        domain_scale is None
        or not math.isfinite(domain_scale)
        or domain_scale <= 0.0
    ):
        raise ValueError(
            "Lanczos-3 registration requires a declared finite numeric domain"
        )
    # For a fractional coordinate, Lanczos-3 touches floor(q)-2 through
    # floor(q)+3.  At the upper integer endpoint the final coefficient is
    # exactly zero, so clipping that unused index is safe and keeps the
    # symmetric two-pixel interpolation margin used by registration.
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 2.0)
        & (x <= width - 3.0)
        & (y >= 2.0)
        & (y <= height - 3.0)
    )
    result = np.full(x.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return result

    xv = x[valid]
    yv = y[valid]
    x_floor = np.floor(xv).astype(np.int64)
    y_floor = np.floor(yv).astype(np.int64)
    x_fraction = xv - x_floor
    y_fraction = yv - y_floor
    offsets = LANCZOS3_TAP_OFFSETS

    def weights(fraction: NDArray[np.float64]) -> tuple[NDArray[np.float32], ...]:
        values: list[NDArray[np.float32]] = []
        total = np.zeros(fraction.shape, dtype=np.float64)
        # Three transcendental evaluations serve all six taps exactly.
        primary_sine = np.sin(np.pi * fraction)
        reduced = np.pi * fraction / 3.0
        reduced_sine = np.sin(reduced)
        reduced_cosine = np.cos(reduced)
        for offset, sign, tap_cosine, tap_sine in zip(
            offsets, LANCZOS3_TAP_SIGNS, LANCZOS3_TAP_COSINES, LANCZOS3_TAP_SINES,
            strict=True,
        ):
            distance = fraction - float(offset)
            absolute = np.abs(distance)
            weight = np.zeros(distance.shape, dtype=np.float64)
            at_origin = absolute <= 1.0e-14
            inside = (absolute < 3.0) & ~at_origin
            weight[at_origin] = 1.0
            if np.any(inside):
                phase = np.pi * distance[inside]
                primary = (sign * primary_sine[inside]) / phase
                secondary = (
                    reduced_sine[inside] * tap_cosine
                    - reduced_cosine[inside] * tap_sine
                ) / (phase / 3.0)
                weight[inside] = primary * secondary
            # Exact integer offsets other than zero are mathematical
            # zeros.  Making them exact prevents a distant NaN from
            # invalidating an integer-coordinate sample through roundoff.
            integer_zero = (
                (absolute > 1.0e-14)
                & (np.abs(distance - np.rint(distance)) <= 1.0e-14)
            )
            weight[integer_zero] = 0.0
            total += weight
            values.append(np.asarray(weight, dtype=np.float32))
        if not np.all(np.isfinite(total)) or np.any(np.abs(total) < 1.0e-12):
            raise RuntimeError("Lanczos-3 weight normalization is singular")
        for weight in values:
            np.divide(weight, total, out=weight, casting="unsafe")
        return tuple(values)

    x_weights = weights(x_fraction)
    y_weights = weights(y_fraction)
    samples = np.zeros(xv.shape, dtype=np.float32)
    sample_valid = np.ones(xv.shape, dtype=bool)
    support_minimum = np.full(xv.shape, np.inf, dtype=np.float32)
    support_maximum = np.full(xv.shape, -np.inf, dtype=np.float32)
    x_index = np.empty_like(x_floor)
    y_index = np.empty_like(y_floor)
    combined_weight = np.empty_like(samples)

    for y_offset, y_weight in zip(offsets, y_weights, strict=True):
        np.add(y_floor, y_offset, out=y_index)
        if y_offset == 3:
            np.minimum(y_index, height - 1, out=y_index)
        for x_offset, x_weight in zip(offsets, x_weights, strict=True):
            np.multiply(y_weight, x_weight, out=combined_weight)
            active = combined_weight != 0.0
            all_active = bool(np.all(active))
            if not all_active and not np.any(active):
                continue
            # Valid coordinates guarantee every offset except +3 is in
            # bounds. Only the zero-weight upper integer endpoint needs
            # clipping; keep the original tap order and pixel values.
            np.add(x_floor, x_offset, out=x_index)
            if x_offset == 3:
                np.minimum(x_index, width - 1, out=x_index)
            neighbor = frame._physical_index(y_index, x_index)
            finite_neighbor = np.isfinite(neighbor)
            if all_active:
                # Fractional affine coordinates normally activate every
                # tap. A nonfinite neighbor permanently invalidates its
                # sample, so its intermediate sum/bounds are never used.
                # Unmasked ufuncs avoid several full-tile boolean/select
                # passes while retaining the exact finite-pixel ordering.
                sample_valid &= finite_neighbor
                np.minimum(support_minimum, neighbor, out=support_minimum)
                np.maximum(support_maximum, neighbor, out=support_maximum)
                np.multiply(neighbor, combined_weight, out=neighbor)
                np.add(samples, neighbor, out=samples, where=sample_valid)
                continue
            accepted = active & finite_neighbor
            sample_valid &= ~active | finite_neighbor
            np.minimum(
                support_minimum, neighbor, out=support_minimum, where=accepted
            )
            np.maximum(
                support_maximum, neighbor, out=support_maximum, where=accepted
            )
            # Neighbor is a private physical-value copy. Reuse it only
            # after recording the support bounds, preserving tap order
            # and Float32 multiply/add rounding without a product tile.
            np.multiply(neighbor, combined_weight, out=neighbor)
            neighbor[~accepted] = 0.0
            samples += neighbor

    lower_bound = np.minimum(support_minimum, np.float32(0.0))
    upper_bound = np.maximum(support_maximum, np.float32(domain_scale))
    samples = np.minimum(np.maximum(samples, lower_bound), upper_bound)
    selected = result[valid]
    selected[sample_valid] = samples[sample_valid]
    result[valid] = selected
    return result


def read_frame_info(path: str | os.PathLike[str]) -> FrameInfo:
    with FitsFrame(path) as frame:
        return frame.info


def _fits_header(
    shape: tuple[int, int], metadata: Mapping[str, Any] | None
) -> tuple[fits.Header, bytes]:
    height, width = shape
    header = fits.Header()
    header.append(("SIMPLE", True, "conforms to FITS standard"))
    header.append(("BITPIX", -32, "32-bit floating point"))
    header.append(("NAXIS", 2))
    header.append(("NAXIS1", width))
    header.append(("NAXIS2", height))
    header.append(("EXTEND", True))
    for key, value in (metadata or {}).items():
        key = str(key).strip().upper()
        if not key or key in {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2"}:
            continue
        if value is not None:
            header[key] = _plain_header_value(value)
    encoded = header.tostring(endcard=True, padding=True).encode("ascii")
    return header, encoded


class FitsFloatWriter:
    """Incremental big-endian Float32 FITS writer.

    Rows are written through one buffered file handle (sequential rows
    append; out-of-order rows seek), so the page cache absorbs the data
    without the page-fault and synchronous writeback cost of a memory map.
    ``durable`` controls the closing ``fsync``: published products keep it,
    transient intermediates that are deleted before the run ends skip it.
    When rows arrive once each in ascending order the writer also folds the
    exact file bytes (header, big-endian samples, block padding) into a
    SHA-256 digest, so publication receipts never reread large intermediates.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        shape: tuple[int, int],
        metadata: Mapping[str, Any] | None = None,
        *,
        durable: bool = True,
    ) -> None:
        self.path = Path(path)
        self.shape = shape
        self.metadata = metadata
        self.durable = bool(durable)
        self._stream: Any = None
        self._data_offset = 0
        self._padding = 0
        self._position_row = 0
        self._digest: Any = hashlib.sha256()
        self._next_row = 0
        self._sequential = True
        self.sha256: str | None = None

    def __enter__(self) -> FitsFloatWriter:
        height, width = self.shape
        if height < 1 or width < 1:
            raise ValueError("output shape must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _, encoded = _fits_header(self.shape, self.metadata)
        data_bytes = height * width * np.dtype(">f4").itemsize
        self._padding = (-data_bytes) % FITS_BLOCK_BYTES
        self._data_offset = len(encoded)
        try:
            stream = self.path.open("xb", buffering=8 * 1024 * 1024)
        except FileExistsError as error:
            raise CalibrationError(
                "OUTPUT_EXISTS", "refusing to overwrite output", path=str(self.path)
            ) from error
        try:
            stream.write(encoded)
            # Pre-size the file so unwritten rows and the block padding are
            # zero and seeks beyond the current position are valid.
            stream.truncate(self._data_offset + data_bytes + self._padding)
        except Exception:
            stream.close()
            raise
        self._stream = stream
        self._digest.update(encoded)
        self._position_row = 0
        return self

    def write_rows(self, y0: int, values: NDArray[Any]) -> None:
        if self._stream is None:
            raise RuntimeError("FitsFloatWriter is not open")
        rows = np.asarray(values, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.shape[1]:
            raise ValueError("output tile has the wrong shape")
        y1 = y0 + rows.shape[0]
        if y0 < 0 or y1 > self.shape[0]:
            raise ValueError("output tile is outside image bounds")
        # One big-endian conversion feeds both the file and the digest.
        encoded = np.ascontiguousarray(rows, dtype=">f4")
        if y0 != self._position_row:
            self._stream.seek(self._data_offset + y0 * self.shape[1] * 4)
        self._stream.write(memoryview(encoded).cast("B"))
        self._position_row = y1
        if self._sequential and y0 == self._next_row:
            self._digest.update(memoryview(encoded).cast("B"))
            self._next_row = y1
        else:
            self._sequential = False

    def close(self) -> None:
        if self._stream is not None:
            stream = self._stream
            self._stream = None
            try:
                stream.flush()
                if self.durable:
                    os.fsync(stream.fileno())
            finally:
                stream.close()
            if self._sequential and self._next_row == self.shape[0]:
                self._digest.update(bytes(self._padding))
                self.sha256 = "sha256:" + self._digest.hexdigest()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _atomic_publish_file(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output", path=str(destination)
        ) from error
    temporary.unlink()


def _temporary_output(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    Path(name).unlink()
    return Path(name)


@dataclass(frozen=True, slots=True)
class FrameExpression:
    source_path: str
    subtract_path: str | None = None
    subtract_paths: tuple[str, ...] = ()
    divide_path: str | None = None
    scale: float = 1.0
    offset: float = 0.0
    offset_grid: tuple[tuple[float, ...], ...] = ()
    offset_grid_x: tuple[float, ...] = ()
    offset_grid_y: tuple[float, ...] = ()
    subtract_scale: float = 1.0
    subtract_scales: tuple[float, ...] = ()

    def serializable(self) -> dict[str, Any]:
        return {
            "source": self.source_path,
            "subtract": self.subtract_path,
            "subtractScale": self.subtract_scale,
            "subtractMany": list(self.subtract_paths),
            "subtractManyScales": list(self.subtract_scales),
            "divide": self.divide_path,
            "scale": self.scale,
            "offset": self.offset,
            "offsetGrid": [list(row) for row in self.offset_grid],
            "offsetGridX": list(self.offset_grid_x),
            "offsetGridY": list(self.offset_grid_y),
        }


@lru_cache(maxsize=64)
def _offset_grid_x_plan(
    x_nodes_value: tuple[float, ...], width: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Return one immutable horizontal interpolation plan shared by a group.

    Global-normalization frames in one integration group use equal node
    coordinates.  Caching only coordinate lookup keeps memory O(width), even
    for the 512-frame policy ceiling, while avoiding thousands of repeated
    arange/clip/searchsorted allocations in statistics and integration passes.
    """

    x_nodes = np.asarray(x_nodes_value, dtype=np.float64)
    x_clipped = np.clip(
        np.arange(width, dtype=np.float64), x_nodes[0], x_nodes[-1]
    )
    x_hi = np.asarray(
        np.clip(
            np.searchsorted(x_nodes, x_clipped, side="right"),
            1,
            len(x_nodes) - 1,
        ),
        dtype=np.int64,
    )
    x_lo = np.asarray(x_hi - 1, dtype=np.int64)
    wx = np.asarray(
        (x_clipped - x_nodes[x_lo]) / (x_nodes[x_hi] - x_nodes[x_lo]),
        dtype=np.float64,
    )
    for value in (x_lo, x_hi, wx):
        value.setflags(write=False)
    return x_lo, x_hi, wx


def _add_offset_grid_rows(
    result: NDArray[np.float32],
    grid_value: tuple[tuple[float, ...], ...],
    x_nodes_value: tuple[float, ...],
    y_nodes_value: tuple[float, ...],
    y0: int,
    y1: int,
    width: int,
    absolute_rows: Sequence[int] | None = None,
) -> None:
    """Add the bilinear offset grid to ``result``; rows are ``range(y0, y1)`` or
    the explicit ``absolute_rows`` (one per result row), evaluated identically."""

    grid = np.asarray(grid_value, dtype=np.float64)
    y_nodes = np.asarray(y_nodes_value, dtype=np.float64)
    x_lo, x_hi, wx = _offset_grid_x_plan(x_nodes_value, width)
    horizontal_rows: dict[int, NDArray[np.float64]] = {}

    def horizontal(node_index: int) -> NDArray[np.float64]:
        cached = horizontal_rows.get(node_index)
        if cached is None:
            node_row = grid[node_index]
            cached = np.asarray(
                node_row[x_lo] * (1.0 - wx) + node_row[x_hi] * wx,
                dtype=np.float64,
            )
            horizontal_rows[node_index] = cached
        return cached

    rows = range(y0, y1) if absolute_rows is None else absolute_rows
    for local_row, absolute_y in enumerate(rows):
        clipped_y = min(max(float(absolute_y), y_nodes[0]), y_nodes[-1])
        y_hi = min(
            max(int(np.searchsorted(y_nodes, clipped_y, side="right")), 1),
            len(y_nodes) - 1,
        )
        y_lo = y_hi - 1
        wy = (clipped_y - y_nodes[y_lo]) / (y_nodes[y_hi] - y_nodes[y_lo])
        top = horizontal(y_lo)
        bottom = horizontal(y_hi)
        result[local_row] += np.asarray(
            top * (1.0 - wy) + bottom * wy, dtype=np.float32
        )


def _expression_rows(
    expression: FrameExpression,
    sources: Mapping[str, FitsFrame],
    y0: int,
    y1: int,
    *,
    division_floor: float,
) -> NDArray[np.float32]:
    result = sources[expression.source_path].read_rows(y0, y1)
    if expression.subtract_path is not None:
        subtract = sources[expression.subtract_path].read_rows(y0, y1)
        if expression.subtract_scale != 1.0:
            subtract *= np.float32(expression.subtract_scale)
        result -= subtract
    subtract_scales = expression.subtract_scales or (1.0,) * len(
        expression.subtract_paths
    )
    for subtract_path, subtract_scale in zip(
        expression.subtract_paths, subtract_scales, strict=True
    ):
        subtract = sources[subtract_path].read_rows(y0, y1)
        if subtract_scale != 1.0:
            subtract *= np.float32(subtract_scale)
        result -= subtract
    if expression.divide_path is not None:
        divisor = sources[expression.divide_path].read_rows(y0, y1)
        # A flat is a sensitivity response. Zero and negative responses are
        # invalid pixels, even though division by a negative value is finite.
        valid = np.isfinite(divisor) & (divisor > division_floor)
        np.divide(result, divisor, out=result, where=valid)
        result[~valid] = np.nan
    if expression.scale != 1.0:
        result *= np.float32(expression.scale)
    if expression.offset_grid:
        _add_offset_grid_rows(
            result,
            expression.offset_grid,
            expression.offset_grid_x,
            expression.offset_grid_y,
            y0,
            y1,
            result.shape[1],
        )
    if expression.offset != 0.0:
        result += np.float32(expression.offset)
    return result


def _expression_sampled_rows(
    expression: FrameExpression,
    sources: Mapping[str, Any],
    y_indices: NDArray[np.int64],
    *,
    division_floor: float,
) -> NDArray[np.float32]:
    """Evaluate ``expression`` on the listed rows with one gather per source.

    Every operation is elementwise, so each returned row equals the row that
    ``_expression_rows`` produces for the same absolute row.
    """

    rows = np.asarray(y_indices, dtype=np.int64)
    result = sources[expression.source_path].read_sampled_rows(rows)
    if expression.subtract_path is not None:
        subtract = sources[expression.subtract_path].read_sampled_rows(rows)
        if expression.subtract_scale != 1.0:
            subtract *= np.float32(expression.subtract_scale)
        result -= subtract
    subtract_scales = expression.subtract_scales or (1.0,) * len(
        expression.subtract_paths
    )
    for subtract_path, subtract_scale in zip(
        expression.subtract_paths, subtract_scales, strict=True
    ):
        subtract = sources[subtract_path].read_sampled_rows(rows)
        if subtract_scale != 1.0:
            subtract *= np.float32(subtract_scale)
        result -= subtract
    if expression.divide_path is not None:
        divisor = sources[expression.divide_path].read_sampled_rows(rows)
        valid = np.isfinite(divisor) & (divisor > division_floor)
        np.divide(result, divisor, out=result, where=valid)
        result[~valid] = np.nan
    if expression.scale != 1.0:
        result *= np.float32(expression.scale)
    if expression.offset_grid:
        _add_offset_grid_rows(
            result,
            expression.offset_grid,
            expression.offset_grid_x,
            expression.offset_grid_y,
            0,
            int(rows.size),
            result.shape[1],
            absolute_rows=[int(value) for value in rows],
        )
    if expression.offset != 0.0:
        result += np.float32(expression.offset)
    return result


def _open_expression_sources(
    stack: ExitStack, expressions: Iterable[FrameExpression]
) -> dict[str, FitsFrame]:
    paths: dict[str, str] = {}
    for expression in expressions:
        for path in (
            expression.source_path,
            expression.subtract_path,
            *expression.subtract_paths,
            expression.divide_path,
        ):
            if path is not None:
                resolved = str(Path(path).expanduser().resolve(strict=True))
                paths[resolved] = resolved
    return {path: stack.enter_context(FitsFrame(path)) for path in sorted(paths)}


def _canonical_expression(expression: FrameExpression) -> FrameExpression:
    result = FrameExpression(
        source_path=str(Path(expression.source_path).expanduser().resolve(strict=True)),
        subtract_path=(
            str(Path(expression.subtract_path).expanduser().resolve(strict=True))
            if expression.subtract_path
            else None
        ),
        subtract_scale=float(expression.subtract_scale),
        subtract_paths=tuple(
            str(Path(path).expanduser().resolve(strict=True))
            for path in expression.subtract_paths
        ),
        subtract_scales=tuple(float(value) for value in expression.subtract_scales),
        divide_path=(
            str(Path(expression.divide_path).expanduser().resolve(strict=True))
            if expression.divide_path
            else None
        ),
        scale=float(expression.scale),
        offset=float(expression.offset),
        offset_grid=tuple(
            tuple(float(item) for item in row) for row in expression.offset_grid
        ),
        offset_grid_x=tuple(float(item) for item in expression.offset_grid_x),
        offset_grid_y=tuple(float(item) for item in expression.offset_grid_y),
    )
    if (
        not math.isfinite(result.scale)
        or not math.isfinite(result.offset)
        or not math.isfinite(result.subtract_scale)
        or result.subtract_scale <= 0
        or any(not math.isfinite(value) or value <= 0 for value in result.subtract_scales)
    ):
        raise CalibrationError(
            "FRAME_EXPRESSION_INVALID",
            "expression scale and offset must be finite",
            path=result.source_path,
        )
    if result.subtract_path is None and result.subtract_scale != 1.0:
        raise CalibrationError(
            "FRAME_EXPRESSION_INVALID",
            "subtract_scale requires subtract_path",
            path=result.source_path,
        )
    if result.subtract_scales and len(result.subtract_scales) != len(
        result.subtract_paths
    ):
        raise CalibrationError(
            "FRAME_EXPRESSION_INVALID",
            "subtract_scales must match subtract_paths cardinality",
            path=result.source_path,
        )
    if result.offset_grid or result.offset_grid_x or result.offset_grid_y:
        grid = np.asarray(result.offset_grid, dtype=np.float64)
        x_nodes = np.asarray(result.offset_grid_x, dtype=np.float64)
        y_nodes = np.asarray(result.offset_grid_y, dtype=np.float64)
        if (
            grid.ndim != 2
            or x_nodes.ndim != 1
            or y_nodes.ndim != 1
            or len(x_nodes) < 2
            or len(y_nodes) < 2
            or grid.shape != (len(y_nodes), len(x_nodes))
            or not np.all(np.isfinite(grid))
            or not np.all(np.isfinite(x_nodes))
            or not np.all(np.isfinite(y_nodes))
            or np.any(np.diff(x_nodes) <= 0)
            or np.any(np.diff(y_nodes) <= 0)
        ):
            raise CalibrationError(
                "FRAME_EXPRESSION_OFFSET_GRID_INVALID",
                "offset grid must be finite, strictly ordered, and match its nodes",
                path=result.source_path,
            )
    return result


def _validate_expression_shapes(
    expressions: tuple[FrameExpression, ...], sources: Mapping[str, FitsFrame]
) -> tuple[int, int]:
    expected: tuple[int, int] | None = None
    for expression in expressions:
        shape = sources[expression.source_path].shape
        if expected is None:
            expected = shape
        if shape != expected:
            raise CalibrationError("GEOMETRY_MISMATCH", "source image shapes differ")
        for path in (
            expression.subtract_path,
            *expression.subtract_paths,
            expression.divide_path,
        ):
            if path is not None and sources[path].shape != expected:
                raise CalibrationError(
                    "CALIBRATION_GEOMETRY_MISMATCH",
                    "calibration image shape does not match source",
                    path=path,
                )
        if expression.offset_grid:
            height, width = shape
            if (
                expression.offset_grid_x[0] < 0
                or expression.offset_grid_x[-1] > width - 1
                or expression.offset_grid_y[0] < 0
                or expression.offset_grid_y[-1] > height - 1
            ):
                raise CalibrationError(
                    "FRAME_EXPRESSION_OFFSET_GRID_GEOMETRY_MISMATCH",
                    "offset grid nodes fall outside the source image geometry",
                    path=expression.source_path,
                )
    if expected is None:
        raise CalibrationError("NO_INPUTS", "at least one frame is required")
    return expected


def _uniform_integer_indices(length: int, count: int) -> NDArray[np.int64]:
    if count <= 1:
        return np.zeros(1, dtype=np.int64)
    denominator = count - 1
    values: list[int] = []
    for index in range(count):
        quotient, remainder = divmod(index * (length - 1), denominator)
        twice_remainder = 2 * remainder
        # Match round-to-nearest-even without relying on platform floating point.
        if twice_remainder > denominator or (
            twice_remainder == denominator and quotient % 2 == 1
        ):
            quotient += 1
        values.append(quotient)
    return np.asarray(values, dtype=np.int64)


def _sample_coordinates(
    shape: tuple[int, int], max_samples: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return one deterministic, spatially uniform integer sampling lattice.

    The lattice depends only on image geometry and the science parameter that
    bounds statistics samples. It deliberately does not depend on tile size,
    memory budget, worker count, or hardware profile.
    """

    height, width = shape
    target_rows = min(
        height,
        max(1, math.isqrt(max_samples * height // max(1, width))),
    )
    target_columns = min(width, max(1, max_samples // target_rows))
    y_indices = _uniform_integer_indices(height, target_rows)
    x_indices = _uniform_integer_indices(width, target_columns)
    return y_indices, x_indices


def _sample_expression(
    expression: FrameExpression,
    sources: Mapping[str, FitsFrame],
    shape: tuple[int, int],
    *,
    max_samples: int,
    division_floor: float,
) -> NDArray[np.float32]:
    y_indices, x_indices = _sample_coordinates(shape, max_samples)
    sampled_rows = _expression_sampled_rows(
        expression, sources, y_indices, division_floor=division_floor
    )[:, x_indices]
    chunks: list[NDArray[np.float32]] = []
    for sampled in sampled_rows:
        finite = sampled[np.isfinite(sampled)]
        if finite.size:
            chunks.append(finite.astype(np.float32, copy=False))
    if not chunks:
        raise CalibrationError(
            "NO_FINITE_PIXELS", "frame expression contains no finite sample pixels"
        )
    result = np.concatenate(chunks)
    if result.size > max_samples:
        result = result[:max_samples]
    return result


def robust_location(
    expression: FrameExpression,
    *,
    max_samples: int = 200_000,
    division_floor: float = 1e-12,
    max_memory_bytes: int = DEFAULT_MEMORY_BUDGET,
) -> float:
    expression = _canonical_expression(expression)
    with ExitStack() as stack:
        sources = _open_expression_sources(stack, (expression,))
        shape = _validate_expression_shapes((expression,), sources)
        if shape[1] * 20 > max_memory_bytes:
            raise CalibrationError(
                "MEMORY_BUDGET_TOO_SMALL",
                "one robust-location row exceeds max_memory_bytes",
            )
        sample = _sample_expression(
            expression,
            sources,
            shape,
            max_samples=max_samples,
            division_floor=division_floor,
        )
    location = float(np.median(sample))
    if not math.isfinite(location):
        raise CalibrationError("LOCATION_INVALID", "robust location is non-finite")
    return location


@dataclass(frozen=True, slots=True)
class PixelStatistics:
    finite_pixels: int
    invalid_pixels: int
    minimum: float | None
    maximum: float | None
    mean: float | None

    def serializable(self) -> dict[str, Any]:
        return {
            "finitePixels": self.finite_pixels,
            "invalidPixels": self.invalid_pixels,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
        }


class _StatsAccumulator:
    def __init__(self) -> None:
        self.finite = 0
        self.invalid = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.total = 0.0

    def update(self, values: NDArray[Any]) -> None:
        array = np.asarray(values)
        finite = np.isfinite(array)
        count = int(np.count_nonzero(finite))
        self.finite += count
        self.invalid += int(array.size - count)
        if count:
            # Masked reductions avoid materializing a compacted copy of every
            # finite sample for each written frame.
            self.minimum = min(
                self.minimum, float(np.min(array, initial=np.inf, where=finite))
            )
            self.maximum = max(
                self.maximum, float(np.max(array, initial=-np.inf, where=finite))
            )
            self.total += float(np.sum(array, dtype=np.float64, where=finite))

    def result(self) -> PixelStatistics:
        return PixelStatistics(
            finite_pixels=self.finite,
            invalid_pixels=self.invalid,
            minimum=self.minimum if self.finite else None,
            maximum=self.maximum if self.finite else None,
            mean=self.total / self.finite if self.finite else None,
        )


@dataclass(frozen=True, slots=True)
class IntegrationParameters:
    sigma_clip: float = 4.0
    minimum_rejection_frames: int = 3
    max_memory_bytes: int = DEFAULT_MEMORY_BUDGET
    max_statistics_samples: int = 200_000
    division_floor: float = 1e-12
    transient_rejection: bool = True
    # Half width of the same-row window whose per-pixel MADs are pooled into
    # the noise part of the rejection scale; 0 restores the plain per-pixel
    # MAD.  Frame noise scaling widens the threshold of frames noisier than
    # the group's mixture scale to their own noise.
    rejection_pool_half_width: int = REJECTION_POOL_HALF_WIDTH
    rejection_frame_noise_scaling: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.sigma_clip) or self.sigma_clip <= 0:
            raise ValueError("sigma_clip must be positive and finite")
        if self.minimum_rejection_frames < 3:
            raise ValueError("minimum_rejection_frames must be at least 3")
        if self.max_memory_bytes < 1024:
            raise ValueError("max_memory_bytes is too small")
        if self.max_statistics_samples < 100:
            raise ValueError("max_statistics_samples must be at least 100")
        if not math.isfinite(self.division_floor) or self.division_floor <= 0:
            raise ValueError("division_floor must be positive and finite")
        if not isinstance(self.transient_rejection, bool):
            raise ValueError("transient_rejection must be a boolean")
        if (
            isinstance(self.rejection_pool_half_width, bool)
            or not isinstance(self.rejection_pool_half_width, int)
            or not 0 <= self.rejection_pool_half_width <= REJECTION_POOL_MAX_HALF_WIDTH
        ):
            raise ValueError(
                "rejection_pool_half_width must be an integer in "
                f"[0, {REJECTION_POOL_MAX_HALF_WIDTH}]"
            )
        if not isinstance(self.rejection_frame_noise_scaling, bool):
            raise ValueError("rejection_frame_noise_scaling must be a boolean")

    def serializable(self) -> dict[str, Any]:
        return {
            "sigmaClip": self.sigma_clip,
            "minimumRejectionFrames": self.minimum_rejection_frames,
            "maxMemoryBytes": self.max_memory_bytes,
            "maxStatisticsSamples": self.max_statistics_samples,
            "divisionFloor": self.division_floor,
            "transientRejection": self.transient_rejection,
            "rejectionPoolHalfWidth": self.rejection_pool_half_width,
            "rejectionFrameNoiseScaling": self.rejection_frame_noise_scaling,
        }


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    output_path: str
    shape: tuple[int, int]
    frame_count: int
    tile_rows: int
    weights: tuple[float, ...]
    rejected_samples: int
    accepted_samples: int
    statistics: PixelStatistics
    noise_weights: tuple[float, ...] = ()
    quality_weights: tuple[float, ...] = ()
    map_paths: Mapping[str, str] = field(default_factory=dict)
    execution: Mapping[str, Any] = field(default_factory=dict)
    # Digests computed by the streaming writer over the exact published bytes;
    # None when a writer could not stream (never rereads a file to fill them).
    output_sha256: str | None = None
    map_sha256: Mapping[str, str] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "outputPath": self.output_path,
            "outputSha256": self.output_sha256,
            "shape": list(self.shape),
            "frameCount": self.frame_count,
            "tileRows": self.tile_rows,
            "weights": list(self.weights),
            "rejectedSamples": self.rejected_samples,
            "acceptedSamples": self.accepted_samples,
            "statistics": self.statistics.serializable(),
            "weightComponents": {
                "noise": list(self.noise_weights),
                "registrationQuality": list(self.quality_weights),
                "combined": list(self.weights),
            },
            "maps": dict(self.map_paths),
            "execution": dict(self.execution),
        }


@dataclass(frozen=True, slots=True)
class IntegrationMapPaths:
    """Create-only destinations for ordinary-integration evidence maps."""

    accepted_count: str | os.PathLike[str]
    coverage: str | os.PathLike[str]
    rejection_count: str | os.PathLike[str]

    def resolved(self, output_path: str | os.PathLike[str]) -> dict[str, Path]:
        destinations = {
            "acceptedSampleCount": Path(self.accepted_count).expanduser().resolve(
                strict=False
            ),
            "coverageFraction": Path(self.coverage).expanduser().resolve(strict=False),
            "rejectionCount": Path(self.rejection_count).expanduser().resolve(
                strict=False
            ),
        }
        output = Path(output_path).expanduser().resolve(strict=False)
        values = [output, *destinations.values()]
        if len({os.path.normcase(str(path)) for path in values}) != len(values):
            raise CalibrationError(
                "INTEGRATION_MAP_PATH_CONFLICT",
                "master and evidence-map destinations must be distinct",
            )
        for path in destinations.values():
            if path.exists() or os.path.lexists(path):
                raise CalibrationError(
                    "OUTPUT_EXISTS",
                    "refusing to overwrite integration evidence map",
                    path=str(path),
                )
        return destinations


@dataclass(frozen=True, slots=True)
class _RejectionSigmaFloor:
    """Group-wide rejection scale evidence: the sigma floor and, since v2,
    the per-frame noise scale factors and the row pooling window that the
    per-pixel decision applies.  Empty ``frame_scales`` means every frame is
    compared against the mixture scale (factor 1) and ``pool_half_width`` 0
    means the plain per-pixel MAD; that combination reproduces the v1
    decisions exactly."""

    applicable: bool
    group_sigma_floor: float
    sampled_sigma_median: float | None
    requested_max_samples: int
    algorithm_max_samples: int
    y_coordinate_count: int
    x_coordinate_count: int
    coordinate_count: int
    usable_sigma_count: int
    minimum_finite_frames_per_coordinate: int
    coordinate_sha256: str
    frame_scales: tuple[float, ...] = ()
    pool_half_width: int = 0
    frame_sigma_pixel: tuple[float, ...] = ()
    mixture_sigma: float | None = None

    @property
    def pooled(self) -> bool:
        return self.pool_half_width > 0 or any(
            value != 1.0 for value in self.frame_scales
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "algorithm": REJECTION_FLOOR_ALGORITHM,
            "status": "APPLIED" if self.applicable else "NOT_APPLICABLE",
            "coordinateGeneration": "uniform-rational-round-even-grid-v1",
            "requestedMaxSamples": self.requested_max_samples,
            "algorithmMaxSamples": self.algorithm_max_samples,
            "yCoordinateCount": self.y_coordinate_count,
            "xCoordinateCount": self.x_coordinate_count,
            "coordinateCount": self.coordinate_count,
            "usableSigmaCount": self.usable_sigma_count,
            "minimumFiniteFramesPerCoordinate": self.minimum_finite_frames_per_coordinate,
            "coordinateSha256": self.coordinate_sha256,
            "sampledSigmaMedian": self.sampled_sigma_median,
            "groupFloorFraction": REJECTION_FLOOR_GROUP_FRACTION,
            "groupSigmaFloor": self.group_sigma_floor,
            "absoluteFloor": REJECTION_FLOOR_ABSOLUTE,
            "float32EpsilonFactor": REJECTION_FLOOR_EPSILON_FACTOR,
            "tileInvariant": True,
            # Group-level (frame-order invariant) part of the scale model;
            # the per-frame factors are reported by ``frame_evidence``.
            "scaleModel": {
                "algorithm": REJECTION_SCALE_ALGORITHM,
                "status": "APPLIED" if self.pooled else "PER_PIXEL_MAD",
                "poolHalfWidth": self.pool_half_width,
                "poolWindowPixels": 2 * self.pool_half_width + 1,
                "poolAxis": "row",
                "frameNoiseScaling": any(value != 1.0 for value in self.frame_scales),
                "mixtureSigma": _json_number(self.mixture_sigma)
                if self.mixture_sigma is not None
                else None,
                "excessVariancePerPixel": True,
            },
        }

    def frame_evidence(self) -> dict[str, Any]:
        """Per-frame factors in input order (deliberately outside ``serializable``)."""

        return {
            "frameScales": list(self.frame_scales),
            "frameSigmaPixel": [_json_number(value) for value in self.frame_sigma_pixel],
        }


def _rejection_frame_scales(
    frame_sigma_pixel: Sequence[float],
    mixture_sigma: float | None,
) -> tuple[tuple[float, ...], float | None]:
    """Per-frame factors that turn the pooled stack scale into a frame's own noise.

    ``mixture_sigma`` is the group-wide median of the per-pixel robust sigma
    (the scale a per-pixel MAD over all frames measures, including its
    small-sample bias); a frame whose per-pixel noise exceeds it gets the
    factor ``sigma_j / mixture_sigma`` so its samples are judged against
    their own noise.  Factors never drop below 1, so no frame is ever held
    to a tighter threshold than the pooled mixture scale: the v2 decisions
    are then a subset of the v1 rejections at every pixel.  Frames without a
    usable estimate keep factor 1.  Factors are rounded to
    ``REJECTION_FRAME_SCALE_DIGITS`` significant Float32 digits.
    """

    sigmas = np.asarray(frame_sigma_pixel, dtype=np.float64)
    if (
        mixture_sigma is None
        or not math.isfinite(mixture_sigma)
        or mixture_sigma <= 0.0
    ):
        return tuple(1.0 for _ in sigmas), None
    scales: list[float] = []
    for sigma in sigmas:
        if not math.isfinite(sigma) or sigma <= 0.0:
            scales.append(1.0)
            continue
        ratio = max(1.0, float(sigma) / float(mixture_sigma))
        ratio = float(f"{ratio:.{REJECTION_FRAME_SCALE_DIGITS}g}")
        scales.append(max(1.0, float(np.float32(ratio))))
    return tuple(scales), float(mixture_sigma)


def _coordinate_digest(
    shape: tuple[int, int],
    y_indices: NDArray[np.int64],
    x_indices: NDArray[np.int64],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"openastroflow-integration-sample-grid-v1\0")
    digest.update(np.asarray(shape, dtype="<i8").tobytes())
    digest.update(np.asarray(y_indices, dtype="<i8").tobytes())
    digest.update(np.asarray(x_indices, dtype="<i8").tobytes())
    return "sha256:" + digest.hexdigest()


def _estimate_rejection_sigma_floor(
    expressions: tuple[FrameExpression, ...],
    sources: Mapping[str, FitsFrame],
    shape: tuple[int, int],
    parameters: IntegrationParameters,
    *,
    frame_noise: _FrameNoiseEstimates | None = None,
) -> _RejectionSigmaFloor:
    """Estimate one group-wide floor without a full-image statistics pass.

    Each sampled coordinate is evaluated across the complete frame group. Rows
    are processed one at a time, so temporary memory is bounded by one sampled
    row times the frame count. Only the final scalar floor is consumed by tiled
    integration, making rejection decisions independent of tile partitioning.

    ``frame_noise`` (the per-frame per-pixel noise measured for the weights)
    supplies the frame scale factors of the v2 rejection scale model; the
    pooling window comes from the parameters.  Both are group-wide scalars, so
    the decisions stay independent of the tile partition.
    """

    sample_limit = min(
        int(parameters.max_statistics_samples), REJECTION_FLOOR_MAX_SAMPLES
    )
    y_indices, x_indices = _sample_coordinates(shape, sample_limit)
    coordinate_count = int(y_indices.size * x_indices.size)
    coordinate_sha256 = _coordinate_digest(shape, y_indices, x_indices)
    applicable = len(expressions) >= parameters.minimum_rejection_frames
    frame_scales: tuple[float, ...] = ()
    frame_sigma_pixel: tuple[float, ...] = ()
    mixture_sigma: float | None = None
    pool_half_width = 0
    if not applicable:
        return _RejectionSigmaFloor(
            False,
            REJECTION_FLOOR_ABSOLUTE,
            None,
            parameters.max_statistics_samples,
            REJECTION_FLOOR_MAX_SAMPLES,
            int(y_indices.size),
            int(x_indices.size),
            coordinate_count,
            0,
            parameters.minimum_rejection_frames,
            coordinate_sha256,
        )

    # Every sampled coordinate is evaluated across the complete frame group in
    # one pass: one gather per frame, then per-coordinate statistics in
    # row-major coordinate order, exactly as a row-by-row loop would produce.
    coordinate_values = np.empty(
        (len(expressions), int(y_indices.size) * int(x_indices.size)), dtype=np.float32
    )
    for frame_index, expression in enumerate(expressions):
        coordinate_values[frame_index] = _expression_sampled_rows(
            expression, sources, y_indices, division_floor=parameters.division_floor
        )[:, x_indices].reshape(-1)
    finite_count = np.count_nonzero(np.isfinite(coordinate_values), axis=0)
    eligible = finite_count >= parameters.minimum_rejection_frames
    sampled_sigma = np.empty(0, dtype=np.float32)
    if np.any(eligible):
        selected = coordinate_values[:, eligible]
        selected[~np.isfinite(selected)] = np.nan
        center = nanmedian_frames(selected)
        mad = nanmedian_frames(np.abs(selected - center[None, :]))
        robust_sigma = np.asarray(np.float32(1.4826) * mad, dtype=np.float32)
        usable = np.isfinite(robust_sigma) & (robust_sigma > 0)
        sampled_sigma = robust_sigma[usable]

    if sampled_sigma.size:
        sampled_sigma_median = float(np.median(sampled_sigma))
        usable_sigma_count = int(sampled_sigma.size)
        group_sigma_floor = max(
            REJECTION_FLOOR_ABSOLUTE,
            sampled_sigma_median * REJECTION_FLOOR_GROUP_FRACTION,
        )
    else:
        sampled_sigma_median = None
        usable_sigma_count = 0
        group_sigma_floor = REJECTION_FLOOR_ABSOLUTE
    pool_half_width = int(parameters.rejection_pool_half_width)
    if parameters.rejection_frame_noise_scaling and frame_noise is not None:
        frame_sigma_pixel = tuple(frame_noise.sigma_pixel)
        frame_scales, mixture_sigma = _rejection_frame_scales(
            frame_sigma_pixel, sampled_sigma_median
        )
    return _RejectionSigmaFloor(
        True,
        group_sigma_floor,
        sampled_sigma_median,
        parameters.max_statistics_samples,
        REJECTION_FLOOR_MAX_SAMPLES,
        int(y_indices.size),
        int(x_indices.size),
        coordinate_count,
        usable_sigma_count,
        parameters.minimum_rejection_frames,
        coordinate_sha256,
        frame_scales,
        pool_half_width,
        frame_sigma_pixel,
        mixture_sigma,
    )


def _rejection_kernel_id() -> str:
    return MAD_KERNEL_ID if load_native_kernels() is not None else NUMPY_MAD_KERNEL_ID


def _rejection_method_id(sigma_floor: _RejectionSigmaFloor) -> str:
    return "median-pooled-mad-sigma-v2" if sigma_floor.pooled else "median-mad-sigma"


def _reduction_kernel_id() -> str:
    return MEAN_KERNEL_ID if load_native_kernels() is not None else NUMPY_MEAN_KERNEL_ID


def _ordinary_mad_rejection_decision(
    values: NDArray[np.float32],
    parameters: IntegrationParameters,
    sigma_floor: _RejectionSigmaFloor,
    *,
    transient_model: TransientRejectionModel | None = None,
    first_row: int = 0,
    native_threads: int | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.float32], NDArray[np.bool_]]:
    """Return finite samples, per-pixel centre, and accepted samples.

    This is the single rejection-decision implementation shared by portable CPU
    integration and the CPU-produced mask consumed by Metal.  The native
    multithreaded kernel and the NumPy reference below make value-identical
    decisions; the kernel merely runs them on every core.
    """

    samples = np.asarray(values, dtype=np.float32)
    if samples.ndim != 3:
        raise ValueError("ordinary integration values must be frame-major 3-D")
    frame_count = samples.shape[0]
    frame_scales = tuple(sigma_floor.frame_scales)
    if frame_scales and len(frame_scales) != frame_count:
        raise ValueError("rejection frame scale count differs from the frame count")
    pool_half_width = int(sigma_floor.pool_half_width)
    kernels = load_native_kernels()
    if kernels is not None:
        finite = np.isfinite(samples)
        accepted, center = kernels.mad_rejection(
            samples,
            sigma_clip=parameters.sigma_clip,
            minimum_rejection_frames=parameters.minimum_rejection_frames,
            group_sigma_floor=sigma_floor.group_sigma_floor,
            absolute_floor=REJECTION_FLOOR_ABSOLUTE,
            epsilon_floor=float(
                np.float32(REJECTION_FLOOR_EPSILON_FACTOR * np.finfo(np.float32).eps)
            ),
            frame_scales=frame_scales or None,
            pool_half_width=pool_half_width,
            threads=native_threads,
        )
        if transient_model is not None and frame_count >= parameters.minimum_rejection_frames:
            enough_samples = (
                np.count_nonzero(finite, axis=0) >= parameters.minimum_rejection_frames
            )
            transient_model.reject_rows(accepted, first_row, enough_samples)
        return finite, center, accepted
    finite = np.isfinite(samples)
    valid_pixels = np.any(finite, axis=0)
    center = np.full(samples.shape[1:], np.nan, dtype=np.float32)
    if np.any(valid_pixels):
        selected = samples[:, valid_pixels]
        selected[~np.isfinite(selected)] = np.nan
        center[valid_pixels] = np.asarray(
            np.nanmedian(selected, axis=0), dtype=np.float32
        )
    if frame_count < parameters.minimum_rejection_frames:
        return finite, center, finite.copy()
    mad = np.full(samples.shape[1:], np.nan, dtype=np.float32)
    if np.any(valid_pixels):
        selected = samples[:, valid_pixels]
        selected[~np.isfinite(selected)] = np.nan
        selected_center = center[valid_pixels]
        mad[valid_pixels] = np.asarray(
            np.nanmedian(np.abs(selected - selected_center[None, :]), axis=0),
            dtype=np.float32,
        )
    # Dither boundaries and masked detector defects may have fewer usable
    # samples than the group size. Do not infer outliers from too few samples.
    enough_samples = (
        np.count_nonzero(finite, axis=0) >= parameters.minimum_rejection_frames
    )
    robust_sigma = np.asarray(np.float32(1.4826) * mad, dtype=np.float32)
    numerical_floor = np.maximum(
        np.float32(REJECTION_FLOOR_ABSOLUTE),
        np.float32(REJECTION_FLOOR_EPSILON_FACTOR * np.finfo(np.float32).eps)
        * np.maximum(np.float32(1.0), np.abs(center)),
    )
    group_floor = np.float32(sigma_floor.group_sigma_floor)
    if pool_half_width > 0 or any(scale != 1.0 for scale in frame_scales):
        # v2 scale model: the noise part of the scale is the pooled MAD of
        # the row window, scaled to each frame's own noise; per-pixel excess
        # variance beyond the pooled noise is kept.  Pixels without enough
        # samples contribute no MAD to their neighbours.
        pooled = _pooled_row_mad(
            np.where(enough_samples, mad, np.float32(np.nan)).astype(np.float32),
            pool_half_width,
        )
        sigma_pool = np.asarray(np.float32(1.4826) * pooled, dtype=np.float32)
        excess = np.maximum(
            robust_sigma * robust_sigma - sigma_pool * sigma_pool, np.float32(0.0)
        )
        scales = np.asarray(
            frame_scales if frame_scales else (1.0,) * frame_count, dtype=np.float32
        ).reshape(frame_count, 1, 1)
        scaled = scales * sigma_pool[None, :, :]
        sigma_frame = np.sqrt(scaled * scaled + excess[None, :, :])
        effective_sigma = np.maximum(
            np.maximum(sigma_frame, group_floor), numerical_floor[None, :, :]
        )
    else:
        effective_sigma = np.maximum(
            np.maximum(robust_sigma, group_floor), numerical_floor
        )
    threshold = np.float32(parameters.sigma_clip) * effective_sigma
    accepted = finite & (np.abs(samples - center[None, :, :]) <= threshold)
    accepted |= finite & ~enough_samples[None, :, :]
    if transient_model is not None:
        transient_model.reject_rows(accepted, first_row, enough_samples)
    return finite, center, accepted


def _pooled_row_mad(
    mad: NDArray[np.float32], half_width: int
) -> NDArray[np.float32]:
    """``np.nanmedian`` of each pixel's MAD over ``[x-h, x+h]`` of its own row.

    The window is clipped to the row (no padding values take part), NaN
    entries are ignored, and a pixel whose window holds no finite MAD stays
    NaN.  Rows are evaluated one at a time so the temporary window stack is
    ``(2h+1) x width`` regardless of the tile height.  The native kernel
    reproduces this arithmetic value for value.
    """

    rows, width = mad.shape
    if half_width <= 0:
        return np.array(mad, dtype=np.float32, copy=True)
    window = 2 * half_width + 1
    padded = np.full((rows, width + 2 * half_width), np.nan, dtype=np.float32)
    padded[:, half_width : half_width + width] = mad
    result = np.empty_like(mad, dtype=np.float32)
    for row in range(rows):
        windows = np.lib.stride_tricks.sliding_window_view(padded[row], window)
        result[row] = nanmedian_frames(np.ascontiguousarray(windows.T))
    return result


def _ordinary_mad_rejection_mask(
    values: NDArray[np.float32],
    parameters: IntegrationParameters,
    sigma_floor: _RejectionSigmaFloor,
    *,
    transient_model: TransientRejectionModel | None = None,
    first_row: int = 0,
) -> NDArray[np.uint8]:
    finite, _, accepted = _ordinary_mad_rejection_decision(
        values, parameters, sigma_floor,
        transient_model=transient_model, first_row=first_row,
    )
    # Nonfinite samples are rejected independently by both reducers. The mask
    # records only robust finite-sample decisions.
    return np.asarray(finite & ~accepted, dtype=np.uint8)


def _prepare_transient_rejection(
    expressions: tuple[FrameExpression, ...],
    sources: Mapping[str, FitsFrame],
    shape: tuple[int, int],
    parameters: IntegrationParameters,
    weights: NDArray[np.float64],
    *,
    workers: int | None = None,
) -> TransientRejectionModel:
    """Read block means once and fit one tile-independent spatial mask model.

    Each frame's block means are independent, so frames are read concurrently
    by ``workers`` threads; the fitted model is identical for every worker
    count.
    """
    height, width = shape
    if not parameters.transient_rejection:
        return TransientRejectionModel(1, status="DISABLED")
    if len(expressions) < 5 or min(shape) < 512:
        return TransientRejectionModel(1, status="NOT_APPLICABLE")
    # Fixed algorithm limits, independent of the reduction tile/memory budget.
    # No subsampling: every finite pixel contributes to its bin, so a narrow
    # trail cannot disappear between sampled coordinate columns.
    factor = max(4, math.ceil(max(shape) / 1600),
                 math.ceil(math.sqrt(len(expressions) * height * width / 4_000_000)))
    by, bx = math.ceil(height / factor), math.ceil(width / factor)
    estimated_bytes = (32 * len(expressions) + 192) * by * bx + 20 * factor * width
    if estimated_bytes > parameters.max_memory_bytes:
        raise CalibrationError(
            "TRANSIENT_REJECTION_MEMORY_BUDGET_TOO_SMALL",
            f"spatial rejection needs {estimated_bytes} bytes; increase max_memory_bytes",
        )
    preview = np.full((len(expressions), by, bx), np.nan, dtype=np.float32)

    # Several block rows are evaluated per read; each block row is then
    # reduced from its own row slice, so the statistics equal a
    # one-block-row-at-a-time evaluation.
    block_rows_per_band = max(1, TRANSIENT_BAND_ROWS // factor)
    offsets = np.arange(0, width, factor)
    covered_width = np.minimum(factor, width - offsets)

    def block_means(index: int) -> None:
        expression = expressions[index]
        for first_block in range(0, by, block_rows_per_band):
            last_block = min(by, first_block + block_rows_per_band)
            band_y0 = first_block * factor
            band_y1 = min(height, last_block * factor)
            band = _expression_rows(expression, sources, band_y0, band_y1,
                                    division_floor=parameters.division_floor)
            for row in range(first_block, last_block):
                y0, y1 = row * factor, min(height, (row + 1) * factor)
                values = band[y0 - band_y0 : y1 - band_y0]
                finite = np.isfinite(values)
                sums = np.sum(np.where(finite, values, 0), axis=0, dtype=np.float64)
                counts = np.sum(finite, axis=0)
                sums = np.add.reduceat(sums, offsets)
                counts = np.add.reduceat(counts, offsets)
                # Exclude partly covered blocks from spatial detection. Ordinary
                # per-pixel rejection still handles all valid edge samples.
                expected = (y1 - y0) * covered_width
                np.divide(sums, counts, out=preview[index, row],
                          where=counts == expected, casting="unsafe")

    reader_count = max(1, min(int(workers or 1), len(expressions)))
    if reader_count == 1:
        for index in range(len(expressions)):
            block_means(index)
    else:
        with ThreadPoolExecutor(
            max_workers=reader_count, thread_name_prefix="oaf-transient"
        ) as pool:
            list(pool.map(block_means, range(len(expressions))))
    try:
        background = fit_residual_background(preview, factor, weights)
    except ValueError as error:
        raise CalibrationError("RESIDUAL_BACKGROUND_UNDERCONSTRAINED", str(error)) from error
    if background is not None:
        background.apply_coordinates(preview,
            np.arange(bx,dtype=np.float64)*factor+(factor-1)/2,
            np.arange(by,dtype=np.float64)*factor+(factor-1)/2)
    model = detect_transient_trails(preview, factor, workers=reader_count)
    return TransientRejectionModel(factor, model.trails, model.status, background)


def _ordinary_integration_tile(
    values: NDArray[np.float32], parameters: IntegrationParameters,
    sigma_floor: _RejectionSigmaFloor, transient_model: TransientRejectionModel,
    first_row: int, native_threads: int | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.float32], NDArray[np.bool_]]:
    """Original MAD decisions plus spatial rejection, with its sky reference kept.

    Only pixels with an additional spatial rejection have their temporary sample
    values normalized. This routine is shared by CPU and Metal preparation.
    """
    finite, center, accepted = _ordinary_mad_rejection_decision(
        values, parameters, sigma_floor, native_threads=native_threads
    )
    if transient_model.trails:
        original = accepted.copy()
        enough = np.sum(finite, axis=0) >= parameters.minimum_rejection_frames
        transient_model.reject_rows(accepted, first_row, enough)
        transient_model.normalize_rejected_rows(values, first_row, original, accepted)
    return finite, center, accepted


@dataclass(frozen=True, slots=True)
class _FrameNoiseEstimates:
    """Per-frame noise measured on one sampling pass over the registered frames.

    ``sigma_pixel`` is the per-pixel noise (dispersion of neighbouring lattice
    samples of a row); it is the scale the stack's per-pixel MAD sees and
    feeds the rejection frame scales.  ``sigma_block`` is the noise of
    ``block_size`` x ``block_size`` block means (the low-frequency noise the
    integration averages, insensitive to the resampling phase) and feeds the
    weights.  Frames without a usable estimate carry NaN.
    """

    sigma_pixel: tuple[float, ...]
    sigma_block: tuple[float, ...]
    block_size: int
    band_count: int
    lattice_column_count: int
    block_column_count: int
    pixel_difference_counts: tuple[int, ...]
    block_difference_counts: tuple[int, ...]
    # "lattice-differences", or "sample-mad" when a frame offered fewer than
    # NOISE_WEIGHT_MINIMUM_DIFFERENCES finite differences (tiny images).
    pixel_sigma_methods: tuple[str, ...] = ()

    def serializable(self) -> dict[str, Any]:
        return {
            "algorithm": NOISE_WEIGHT_ALGORITHM,
            "blockSize": self.block_size,
            "bandCount": self.band_count,
            "latticeColumnCount": self.lattice_column_count,
            "blockColumnCount": self.block_column_count,
            "frameSigmaPixel": [_json_number(value) for value in self.sigma_pixel],
            "frameSigmaBlock": [_json_number(value) for value in self.sigma_block],
            "pixelDifferenceCounts": list(self.pixel_difference_counts),
            "blockDifferenceCounts": list(self.block_difference_counts),
            "pixelSigmaMethods": list(self.pixel_sigma_methods),
            "minimumDifferences": NOISE_WEIGHT_MINIMUM_DIFFERENCES,
        }


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _difference_sigma(differences: NDArray[np.float64]) -> float:
    """Noise of one sample from the robust dispersion of pairwise differences.

    Adjacent lattice samples (or adjacent block means) share the same sky to
    well below the noise, so a sky gradient or a moonlit night does not enter
    the estimate, and their difference is sqrt(2) times the noise of one.
    """

    finite = differences[np.isfinite(differences)]
    if finite.size < NOISE_WEIGHT_MINIMUM_DIFFERENCES:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(finite - np.median(finite)))) / math.sqrt(2.0)


def _sample_sigma(samples: NDArray[np.float64]) -> float:
    """Plain robust dispersion of the samples (the tiny-image fallback)."""

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    return 1.4826 * float(np.median(np.abs(finite - median)))


def _frame_noise_estimates(
    expressions: tuple[FrameExpression, ...],
    sources: Mapping[str, FitsFrame],
    shape: tuple[int, int],
    parameters: IntegrationParameters,
) -> _FrameNoiseEstimates:
    """Measure per-pixel and block-mean noise of every frame on one lattice.

    Every fourth lattice row starts a band of ``NOISE_WEIGHT_BLOCK_SIZE``
    consecutive rows, so the rows read per frame equal the plain lattice.
    Per-pixel differences are taken between the lattice columns within each
    band row; block differences between the means of the ``b x b`` blocks
    that start at consecutive lattice columns.  The lattice depends only on
    the image geometry and ``max_statistics_samples``.
    """

    height, width = shape
    block = NOISE_WEIGHT_BLOCK_SIZE
    y_indices, x_indices = _sample_coordinates(shape, parameters.max_statistics_samples)
    band_starts = y_indices[::block]
    band_starts = band_starts[band_starts + block <= height]
    rows_per_band = block
    if band_starts.size == 0:
        # Images shorter than one block: single lattice rows, no block noise.
        band_starts = y_indices
        rows_per_band = 1
    rows = (band_starts[:, None] + np.arange(rows_per_band)[None, :]).reshape(-1)
    block_columns = _non_overlapping_block_columns(x_indices, block, width)
    column_offsets = np.arange(block)
    sigma_pixel: list[float] = []
    sigma_block: list[float] = []
    pixel_counts: list[int] = []
    block_counts: list[int] = []
    methods: list[str] = []
    for expression in expressions:
        sampled = _expression_sampled_rows(
            expression, sources, rows, division_floor=parameters.division_floor
        )
        bands = sampled.reshape(band_starts.size, rows_per_band, width)
        lattice = bands[:, :, x_indices].astype(np.float64, copy=False)
        with np.errstate(invalid="ignore"):
            pixel_differences = np.diff(lattice, axis=2).reshape(-1)
        pixel_counts.append(int(np.count_nonzero(np.isfinite(pixel_differences))))
        sigma = _difference_sigma(pixel_differences)
        method = "lattice-differences"
        if not math.isfinite(sigma):
            sigma = _sample_sigma(lattice.reshape(-1))
            method = "sample-mad" if math.isfinite(sigma) else "unavailable"
        sigma_pixel.append(sigma)
        methods.append(method)
        if rows_per_band == block and block_columns.size >= 2:
            gathered = bands[:, :, block_columns[:, None] + column_offsets[None, :]]
            with np.errstate(invalid="ignore"):
                means = np.mean(gathered.astype(np.float64, copy=False), axis=(1, 3))
                block_differences = np.diff(means, axis=1).reshape(-1)
            block_counts.append(int(np.count_nonzero(np.isfinite(block_differences))))
            sigma_block.append(_difference_sigma(block_differences))
        else:
            block_counts.append(0)
            sigma_block.append(float("nan"))
    return _FrameNoiseEstimates(
        tuple(sigma_pixel),
        tuple(sigma_block),
        block,
        int(band_starts.size),
        int(x_indices.size),
        int(block_columns.size),
        tuple(pixel_counts),
        tuple(block_counts),
        tuple(methods),
    )


def _non_overlapping_block_columns(
    x_indices: NDArray[np.int64], block: int, width: int
) -> NDArray[np.int64]:
    """Lattice columns whose ``block``-wide blocks fit the row and do not overlap.

    A dense lattice (small images) would otherwise difference overlapping
    block means, which measures the noise of the strip between them, not the
    block noise.  Greedy in lattice order, so the selection depends only on
    the lattice.
    """

    selected: list[int] = []
    last = -block
    for column in x_indices:
        value = int(column)
        if value + block > width:
            continue
        if value - last >= block:
            selected.append(value)
            last = value
    return np.asarray(selected, dtype=np.int64)


def _noise_weights_from_estimates(
    estimates: _FrameNoiseEstimates,
) -> tuple[NDArray[np.float64], tuple[float, ...]]:
    """Inverse-variance weights from the block noise (per-pixel noise as fallback)."""

    sigmas: list[float] = []
    for block_sigma, pixel_sigma in zip(
        estimates.sigma_block, estimates.sigma_pixel, strict=True
    ):
        sigma = block_sigma
        if not math.isfinite(sigma) or sigma <= 1e-12:
            sigma = pixel_sigma
        if not math.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        sigmas.append(sigma)
    raw = 1.0 / np.square(np.asarray(sigmas, dtype=np.float64))
    # Prevent one almost-noiseless frame from completely dominating a group.
    positive = raw[np.isfinite(raw) & (raw > 0)]
    if not positive.size:
        raw = np.ones(len(sigmas), dtype=np.float64)
    else:
        median_weight = float(np.median(positive))
        raw = np.clip(raw, median_weight / 16.0, median_weight * 16.0)
    normalized = raw / np.sum(raw)
    return normalized, tuple(float(value) for value in normalized)


def _normalized_noise_weights(
    expressions: tuple[FrameExpression, ...],
    sources: Mapping[str, FitsFrame],
    shape: tuple[int, int],
    parameters: IntegrationParameters,
) -> tuple[NDArray[np.float64], tuple[float, ...]]:
    return _noise_weights_from_estimates(
        _frame_noise_estimates(expressions, sources, shape, parameters)
    )


def _combined_integration_weights(
    expressions: tuple[FrameExpression, ...],
    sources: Mapping[str, FitsFrame],
    shape: tuple[int, int],
    parameters: IntegrationParameters,
    quality_weights: Sequence[float] | None,
) -> tuple[
    NDArray[np.float64],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    _FrameNoiseEstimates,
]:
    frame_noise = _frame_noise_estimates(expressions, sources, shape, parameters)
    noise, serialized_noise = _noise_weights_from_estimates(frame_noise)
    if quality_weights is None:
        return noise, serialized_noise, (), serialized_noise, frame_noise
    else:
        if len(quality_weights) != len(expressions):
            raise CalibrationError(
                "QUALITY_WEIGHT_COUNT_MISMATCH",
                "registration quality weight count differs from integration inputs",
            )
        try:
            quality = np.asarray(quality_weights, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                "QUALITY_WEIGHT_INVALID",
                "registration quality weights must be finite positive numbers",
            ) from error
        if quality.ndim != 1 or not np.all(np.isfinite(quality)) or np.any(quality <= 0):
            raise CalibrationError(
                "QUALITY_WEIGHT_INVALID",
                "registration quality weights must be finite positive numbers",
            )
        quality = quality / np.sum(quality, dtype=np.float64)
    combined = noise * quality
    total = float(np.sum(combined, dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise CalibrationError(
            "QUALITY_WEIGHT_INVALID", "combined integration weights are not positive"
        )
    combined /= total
    return (
        combined,
        serialized_noise,
        tuple(float(value) for value in quality),
        tuple(float(value) for value in combined),
        frame_noise,
    )


def integrate_expressions(
    expressions: Iterable[FrameExpression],
    output_path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    parameters: IntegrationParameters | None = None,
    quality_weights: Sequence[float] | None = None,
    map_paths: IntegrationMapPaths | None = None,
    native_threads: int | None = None,
    durable: bool = True,
) -> IntegrationResult:
    """Robustly integrate expressions without materializing full input images.

    ``native_threads`` bounds the threads used by the native rejection and
    reduction kernels (and the transient-preparation readers); the result does
    not depend on it.  ``durable`` selects whether outputs are fsynced.
    """

    parameters = parameters or IntegrationParameters()
    parameters.validate()
    native_threads = (
        default_kernel_threads() if native_threads is None else int(native_threads)
    )
    canonical = tuple(_canonical_expression(item) for item in expressions)
    if not canonical:
        raise CalibrationError("NO_INPUTS", "at least one integration input is required")
    destination = Path(output_path)
    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output", path=str(destination)
        )
    temporary = _temporary_output(destination)
    map_destinations = map_paths.resolved(destination) if map_paths is not None else {}
    map_temporaries = {
        name: _temporary_output(path) for name, path in map_destinations.items()
    }
    try:
        with ExitStack() as stack:
            sources = _open_expression_sources(stack, canonical)
            shape = _validate_expression_shapes(canonical, sources)
            height, width = shape
            spatial_applicable = (parameters.transient_rejection
                                  and len(canonical) >= 5 and min(shape) >= 512)
            sample_bytes = 16 if spatial_applicable else 12
            bytes_per_row = width * (sample_bytes * len(canonical) + 64)
            if bytes_per_row > parameters.max_memory_bytes:
                raise CalibrationError(
                    "MEMORY_BUDGET_TOO_SMALL",
                    "one robust-integration row exceeds max_memory_bytes",
                )
            tile_rows = max(
                1, min(height, parameters.max_memory_bytes // max(1, bytes_per_row))
            )
            (
                weights,
                serialized_noise_weights,
                serialized_quality_weights,
                serialized_weights,
                frame_noise,
            ) = _combined_integration_weights(
                canonical, sources, shape, parameters, quality_weights
            )
            rejection_sigma_floor = _estimate_rejection_sigma_floor(
                canonical, sources, shape, parameters, frame_noise=frame_noise
            )
            transient_model = _prepare_transient_rejection(
                canonical, sources, shape, parameters, weights,
                workers=native_threads,
            )
            kernels = load_native_kernels()
            stats = _StatsAccumulator()
            accepted_total = 0
            rejected_total = 0
            output_metadata = dict(metadata or {})
            output_metadata.setdefault("OAFSTATE", "UNSOLVED_WORKING")
            output_metadata.setdefault("OAFNFRM", len(canonical))
            output_metadata.setdefault("OAFREJ", parameters.sigma_clip)
            writer = stack.enter_context(
                FitsFloatWriter(temporary, shape, output_metadata, durable=durable)
            )
            map_writers: dict[str, FitsFloatWriter] = {}
            map_metadata = {
                "acceptedSampleCount": {
                    "IMAGETYP": "Integration accepted-sample count",
                    "OAFMAP": "ACCEPTED_COUNT",
                    "OAFNFRM": len(canonical),
                },
                "coverageFraction": {
                    "IMAGETYP": "Integration coverage fraction",
                    "OAFMAP": "COVERAGE",
                    "OAFNFRM": len(canonical),
                },
                "rejectionCount": {
                    "IMAGETYP": "Integration rejection count",
                    "OAFMAP": "REJECTION_COUNT",
                    "OAFNFRM": len(canonical),
                    "OAFREJ": parameters.sigma_clip,
                },
            }
            for name, temporary_map in map_temporaries.items():
                map_writers[name] = stack.enter_context(
                    FitsFloatWriter(
                        temporary_map, shape, map_metadata[name], durable=durable
                    )
                )
            for y0 in range(0, height, tile_rows):
                y1 = min(height, y0 + tile_rows)
                stack_values = np.empty(
                    (len(canonical), y1 - y0, width), dtype=np.float32
                )
                for index, expression in enumerate(canonical):
                    stack_values[index] = _expression_rows(
                        expression,
                        sources,
                        y0,
                        y1,
                        division_floor=parameters.division_floor,
                    )
                finite, center, accepted = _ordinary_integration_tile(
                    stack_values, parameters, rejection_sigma_floor, transient_model, y0,
                    native_threads,
                )
                if kernels is not None:
                    result, accepted_map, rejected_map = kernels.masked_weighted_mean(
                        stack_values, accepted, weights, threads=native_threads
                    )
                    accepted_per_pixel = accepted_map.astype(np.float32)
                    rejected_per_pixel = rejected_map.astype(np.float32)
                else:
                    weighted = accepted * weights[:, None, None]
                    denominator = np.sum(weighted, axis=0, dtype=np.float64)
                    numerator = np.sum(
                        np.where(accepted, stack_values, 0.0)
                        * weights[:, None, None],
                        axis=0,
                        dtype=np.float64,
                    )
                    result = np.full(center.shape, np.nan, dtype=np.float32)
                    np.divide(
                        numerator,
                        denominator,
                        out=result,
                        where=denominator > 0,
                        casting="unsafe",
                    )
                    accepted_per_pixel = np.sum(
                        accepted, axis=0, dtype=np.uint16
                    ).astype(np.float32)
                    rejected_per_pixel = np.sum(
                        finite & ~accepted, axis=0, dtype=np.uint16
                    ).astype(np.float32)
                accepted_count = int(np.count_nonzero(accepted))
                finite_count = int(np.count_nonzero(finite))
                accepted_total += accepted_count
                rejected_total += finite_count - accepted_count
                stats.update(result)
                writer.write_rows(y0, result)
                if map_writers:
                    map_writers["acceptedSampleCount"].write_rows(
                        y0, accepted_per_pixel
                    )
                    map_writers["coverageFraction"].write_rows(
                        y0, accepted_per_pixel / np.float32(len(canonical))
                    )
                    map_writers["rejectionCount"].write_rows(
                        y0, rejected_per_pixel
                    )
        final_statistics = stats.result()
        if final_statistics.finite_pixels == 0:
            raise CalibrationError(
                "NO_FINITE_OUTPUT", "integration produced no finite pixels"
            )
        _atomic_publish_file(temporary, destination)
        for name, map_destination in map_destinations.items():
            _atomic_publish_file(map_temporaries[name], map_destination)
        return IntegrationResult(
            output_path=str(destination),
            shape=shape,
            frame_count=len(canonical),
            tile_rows=tile_rows,
            weights=serialized_weights,
            rejected_samples=rejected_total,
            accepted_samples=accepted_total,
            statistics=final_statistics,
            noise_weights=serialized_noise_weights,
            quality_weights=serialized_quality_weights,
            map_paths={name: str(path) for name, path in map_destinations.items()},
            output_sha256=writer.sha256,
            map_sha256={
                name: digest
                for name, digest in (
                    (name, map_writer.sha256) for name, map_writer in map_writers.items()
                )
                if digest is not None
            },
            execution={
                "requestedBackend": "portable-cpu",
                "selectedBackend": "portable-cpu",
                "acceleratorUsed": False,
                "rejectionMask": {
                    "producer": "portable-cpu",
                    "method": _rejection_method_id(rejection_sigma_floor),
                    "kernel": _rejection_kernel_id(),
                    "sigma": parameters.sigma_clip,
                    "scope": "all-frames-per-pixel",
                    "partialMeanBatching": False,
                    "sigmaFloor": rejection_sigma_floor.serializable(),
                    "frameScaleModel": rejection_sigma_floor.frame_evidence(),
                    "spatialTransients": transient_model.serializable(),
                },
                "noiseWeights": frame_noise.serializable(),
                "reducer": _reduction_kernel_id(),
                "nativeThreads": native_threads,
                "fastMath": False,
            },
        )
    finally:
        if temporary.exists():
            temporary.unlink()
        for map_temporary in map_temporaries.values():
            if map_temporary.exists():
                map_temporary.unlink()


def write_expression(
    expression: FrameExpression,
    output_path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    max_memory_bytes: int = DEFAULT_MEMORY_BUDGET,
    division_floor: float = 1e-12,
) -> PixelStatistics:
    """Write one calibrated expression to a new Float32 FITS file by rows."""

    canonical = _canonical_expression(expression)
    destination = Path(output_path)
    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output", path=str(destination)
        )
    temporary = _temporary_output(destination)
    try:
        with ExitStack() as stack:
            sources = _open_expression_sources(stack, (canonical,))
            shape = _validate_expression_shapes((canonical,), sources)
            height, width = shape
            bytes_per_row = width * 20
            if bytes_per_row > max_memory_bytes:
                raise CalibrationError(
                    "MEMORY_BUDGET_TOO_SMALL", "one calibration row exceeds memory budget"
                )
            tile_rows = max(1, min(height, max_memory_bytes // bytes_per_row))
            output_metadata = dict(metadata or {})
            output_metadata.setdefault("OAFSTATE", "UNSOLVED_WORKING")
            stats = _StatsAccumulator()
            with FitsFloatWriter(temporary, shape, output_metadata) as writer:
                for y0 in range(0, height, tile_rows):
                    y1 = min(height, y0 + tile_rows)
                    values = _expression_rows(
                        canonical,
                        sources,
                        y0,
                        y1,
                        division_floor=division_floor,
                    )
                    stats.update(values)
                    writer.write_rows(y0, values)
        final_statistics = stats.result()
        if final_statistics.finite_pixels == 0:
            raise CalibrationError(
                "NO_FINITE_OUTPUT", "calibration produced no finite pixels"
            )
        _atomic_publish_file(temporary, destination)
        return final_statistics
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "CalibrationError",
    "DEFAULT_MEMORY_BUDGET",
    "FitsFloatWriter",
    "FitsFrame",
    "_MemoryFrame",
    "FrameExpression",
    "FrameInfo",
    "IntegrationMapPaths",
    "IntegrationParameters",
    "IntegrationResult",
    "PixelStatistics",
    "integrate_expressions",
    "normalize_role",
    "read_frame_info",
    "robust_location",
    "write_expression",
]
