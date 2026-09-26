"""Bounded FITS readers, numeric-domain metadata, writers and pixel samplers.

Sources stay read-only; callers stage XISF separately and publish new files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray

from lightframeqc.cfa import CFA_PATTERNS, normalize_pattern as normalize_cfa_pattern
from lightframeqc.fits_bands import FitsBandReader, close_image_data, open_fits_image_data

from ..calibration.policy import bias_from_header
from ..lanczos_table import TAP_OFFSETS, tap_weights
from ..platform import remove_file

FITS_BLOCK_BYTES = 2880
LANCZOS3_TAP_OFFSETS = TAP_OFFSETS


class CalibrationError(RuntimeError):
    """Stable fail-closed error raised by the portable pixel path."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        self.code = code
        self.path = path
        detail = f"{path}: {message}" if path else message
        super().__init__(f"{code}: {detail}")


def plain_header_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def header_number(header: Mapping[str, Any], *keys: str) -> float | None:
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


def header_text(header: Mapping[str, Any], *keys: str, default: str = "UNKNOWN") -> str:
    for key in keys:
        value = header.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().strip("'\"").strip().upper()
    return default


def normalize_role(value: Any) -> str:
    compact = re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())
    # The image-type words WBPP recognises; a Flat-Dark is a Dark.
    return {
        "BIAS": "BIAS",
        "BIASFRAME": "BIAS",
        "ZERO": "BIAS",
        "DARK": "DARK",
        "DARKFRAME": "DARK",
        "DARKFLAT": "DARK",
        "FLATDARK": "DARK",
        "FLAT": "FLAT",
        "FLATFRAME": "FLAT",
        "FLATFIELD": "FLAT",
        "LIGHT": "LIGHT",
        "LIGHTFRAME": "LIGHT",
        "SCIENCE": "LIGHT",
        "SCIENCEFRAME": "LIGHT",
        "MASTERBIAS": "MASTER_BIAS",
        "MASTERBIASFRAME": "MASTER_BIAS",
        "MASTERDARK": "MASTER_DARK",
        "MASTERDARKFRAME": "MASTER_DARK",
        "MASTERDARKFLAT": "MASTER_DARK",
        "MASTERFLATDARK": "MASTER_DARK",
        "MASTERFLAT": "MASTER_FLAT",
        "MASTERFLATFRAME": "MASTER_FLAT",
        "MASTERFLATFIELD": "MASTER_FLAT",
    }.get(compact, "UNKNOWN")


def numeric_domain_from_header(
    header: Mapping[str, Any],
) -> tuple[str, float | None, str]:
    declared = str(header.get("OAFNDOM", "")).strip().upper()
    declared_scale = header_number(header, "OAFNSCL")
    if declared:
        if declared_scale is None or declared_scale <= 0:
            return "UNDECLARED", None, "UNRESOLVED"
        return declared, declared_scale, "SELF_DECLARED_HEADER"
    bitpix_value = header_number(header, "BITPIX")
    bitpix = int(bitpix_value) if bitpix_value is not None else 0
    raw_bunit = header.get("BUNIT")
    if raw_bunit is not None and str(raw_bunit).strip():
        bunit = re.sub(r"[^A-Z0-9]+", "", str(raw_bunit).upper())
        if bunit not in {"ADU", "DN", "COUNT", "COUNTS", "CODE", "CODES"}:
            return "UNDECLARED", None, "FITS_BUNIT_UNSUPPORTED"
    bzero = header_number(header, "BZERO") or 0.0
    declared_bscale = header_number(header, "BSCALE")
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


def numeric_domain_evidence_from_header(
    header: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    bitpix_value = header_number(header, "BITPIX")
    bitpix = int(bitpix_value) if bitpix_value is not None else None
    bscale = header_number(header, "BSCALE")
    if bscale is None:
        bscale = 1.0
    bzero = header_number(header, "BZERO")
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
    # WBPP grouping keywords (NIGHT, SESSION, PANEL, ...) as sorted pairs.
    grouping_keywords: tuple[tuple[str, str], ...] = ()

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
            **({"groupingKeywords": dict(self.grouping_keywords)} if self.grouping_keywords else {}),
        }


def cfa_metadata(info: Any) -> dict[str, str]:
    """``BAYERPAT`` for a master built from Bayer frames, so the master's
    previews and stamps are read as a mosaic like the Lights it calibrates."""

    pattern = normalize_cfa_pattern(getattr(info, "cfa_pattern", None))
    return {"BAYERPAT": pattern} if pattern in CFA_PATTERNS else {}


class FitsFrame:
    """Read-only, manually scaled view of one uncompressed 2-D FITS image.

    The stored samples are reached through astropy's memory map on POSIX and
    through ``lightframeqc.fits_bands.FitsBandReader`` (``seek`` +
    ``readinto`` bands) on Windows, where page faults make a map several
    times slower; both deliver the same bytes with the same dtype, so the
    decoding below never sees a difference.
    """

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
            try:
                self._data = open_fits_image_data(image_hdu, self.path)
            except ValueError as error:
                raise CalibrationError(
                    "FITS_DATA_INVALID", str(error), path=str(self.path)
                ) from error
            if self._data is None or tuple(self._data.shape) != (height, width):
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
            if isinstance(self._data, FitsBandReader):
                # The reader has its own handle; astropy's is not needed and,
                # on Windows, would be one more open handle per frame.
                self._hdul.close()
                self._hdul = None
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
        close_image_data(self._data)
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
        ) = numeric_domain_from_header(header)
        shared_bin = header_number(header, "BINNING")
        bin_x = header_number(header, "XBINNING", "CCDBINX") or shared_bin
        bin_y = header_number(header, "YBINNING", "CCDBINY") or shared_bin
        return FrameInfo(
            path=str(self.path),
            role=normalize_role(header.get("IMAGETYP", header.get("IMAGETYPE"))),
            shape=self.shape,
            filter_name=header_text(header, "FILTER", "INSFLNAM", "FILTERID"),
            exposure_seconds=header_number(header, "EXPTIME", "EXPOSURE", "EXPOSURETIME"),
            temperature_celsius=header_number(
                header, "CCD-TEMP", "CCD_TEMP", "SENSORT", "SENSOR-T", "CAMTEMP"
            ),
            camera=header_text(header, "INSTRUME", "CAMERA", "DETECTOR"),
            gain=header_number(header, "GAIN", "EGAIN", "CAMGAIN"),
            offset=header_number(header, "OFFSET", "CAMOFFSET"),
            binning_x=int(bin_x) if bin_x is not None else None,
            binning_y=int(bin_y) if bin_y is not None else None,
            cfa_pattern=header_text(
                header, "BAYERPAT", "BAYERPATN", "CFAPAT", "CFAPATTERN"
            ),
            readout_mode=header_text(
                header, "READOUTM", "READOUT", "READMODE", "READOUTMODE"
            ),
            target=header_text(header, "OBJECT", "OBJNAME", "TARGET"),
            numeric_domain=numeric_domain,
            normalized_unit_scale=normalized_unit_scale,
            numeric_domain_authority=numeric_domain_authority,
            numeric_domain_evidence=numeric_domain_evidence_from_header(header),
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

        return sample_bilinear_from(self, x_coordinates, y_coordinates)

    def sample_lanczos3_clamped(
        self,
        x_coordinates: NDArray[Any],
        y_coordinates: NDArray[Any],
    ) -> NDArray[np.float32]:
        """Sample with a normalized, bounded six-tap Lanczos-3 kernel.

        See :func:`sample_lanczos3_clamped_from` for the contract; the same
        implementation serves file-backed and in-memory frames.
        """

        return sample_lanczos3_clamped_from(self, x_coordinates, y_coordinates)


class MemoryFrame:
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

    def __enter__(self) -> MemoryFrame:
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
        return sample_bilinear_from(self, x_coordinates, y_coordinates)

    def sample_lanczos3_clamped(
        self,
        x_coordinates: NDArray[Any],
        y_coordinates: NDArray[Any],
    ) -> NDArray[np.float32]:
        return sample_lanczos3_clamped_from(self, x_coordinates, y_coordinates)


def sample_bilinear_from(
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


def sample_lanczos3_clamped_from(
    frame: Any,
    x_coordinates: NDArray[Any],
    y_coordinates: NDArray[Any],
) -> NDArray[np.float32]:
    """Sample with a normalized, bounded six-tap Lanczos-3 kernel.

    The separable kernel's six tap weights come from the deterministic
    Lanczos-3 table (``lanczos_table.py``), normalized independently on both
    axes so a constant field remains constant.  A
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
        # Six normalized taps from the deterministic Lanczos-3 table; the
        # native kernel reads the same table with the same arithmetic.
        return tap_weights(fraction)

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


def fits_header(
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
            header[key] = plain_header_value(value)
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
        _, encoded = fits_header(self.shape, self.metadata)
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


def atomic_publish_file(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output", path=str(destination)
        ) from error
    # The destination link exists; dropping the temporary name waits out a
    # scanner that may still hold the freshly written file (Windows).
    remove_file(temporary, missing_ok=False)


def temporary_output(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    remove_file(name, missing_ok=False)
    return Path(name)

