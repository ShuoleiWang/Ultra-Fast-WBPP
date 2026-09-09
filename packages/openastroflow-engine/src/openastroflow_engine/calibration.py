"""Bounded-memory FITS calibration and robust vertical-slice integration.

This low-level kernel is intentionally FITS-only. The public pixel pipeline
converts supported XISF images through the bounded, content-bound private
staging bridge before entering this module. All sources are opened read-only
and every destination is a new, atomically published file.
"""

from __future__ import annotations

from .calibration_policy import bias_from_header

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

from .transient_rejection import TransientRejectionModel, detect_transient_trails
from .residual_background import fit_residual_background


FITS_BLOCK_BYTES = 2880
DEFAULT_MEMORY_BUDGET = 256 * 1024 * 1024
REJECTION_FLOOR_ALGORITHM = "fixed-grid-group-mad-plus-float32-ulp-v1"
REJECTION_FLOOR_MAX_SAMPLES = 65_536
REJECTION_FLOOR_GROUP_FRACTION = 0.05
REJECTION_FLOOR_ABSOLUTE = 1.0e-7
REJECTION_FLOOR_EPSILON_FACTOR = 16.0


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

    def sample_bilinear(
        self,
        x_coordinates: NDArray[Any],
        y_coordinates: NDArray[Any],
    ) -> NDArray[np.float32]:
        """Sample a bounded coordinate tile without decoding the full image."""

        x = np.asarray(x_coordinates, dtype=np.float64)
        y = np.asarray(y_coordinates, dtype=np.float64)
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError("coordinate tiles must be same-shaped two-dimensional arrays")
        height, width = self.shape
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
        v00 = self._physical_index(y0, x0)
        v10 = self._physical_index(y0, x1)
        v01 = self._physical_index(y1, x0)
        v11 = self._physical_index(y1, x1)
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

    def sample_lanczos3_clamped(
        self,
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
        """

        x = np.asarray(x_coordinates, dtype=np.float64)
        y = np.asarray(y_coordinates, dtype=np.float64)
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError("coordinate tiles must be same-shaped two-dimensional arrays")
        height, width = self.shape
        domain_scale = self.info.normalized_unit_scale
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
        offsets = (-2, -1, 0, 1, 2, 3)

        def weights(fraction: NDArray[np.float64]) -> tuple[NDArray[np.float32], ...]:
            values: list[NDArray[np.float32]] = []
            total = np.zeros(fraction.shape, dtype=np.float64)
            for offset in offsets:
                distance = fraction - float(offset)
                absolute = np.abs(distance)
                weight = np.zeros(distance.shape, dtype=np.float64)
                at_origin = absolute <= 1.0e-14
                inside = (absolute < 3.0) & ~at_origin
                weight[at_origin] = 1.0
                if np.any(inside):
                    phase = np.pi * distance[inside]
                    weight[inside] = (
                        (np.sin(phase) / phase)
                        * (np.sin(phase / 3.0) / (phase / 3.0))
                    )
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
                neighbor = self._physical_index(y_index, x_index)
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
    """Incremental big-endian Float32 FITS writer."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        shape: tuple[int, int],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.shape = shape
        self.metadata = metadata
        self._mapping: np.memmap[Any] | None = None

    def __enter__(self) -> FitsFloatWriter:
        height, width = self.shape
        if height < 1 or width < 1:
            raise ValueError("output shape must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _, encoded = _fits_header(self.shape, self.metadata)
        data_bytes = height * width * np.dtype(">f4").itemsize
        padding = (-data_bytes) % FITS_BLOCK_BYTES
        try:
            with self.path.open("xb") as stream:
                stream.write(encoded)
                stream.truncate(len(encoded) + data_bytes + padding)
                stream.flush()
                os.fsync(stream.fileno())
            self._mapping = np.memmap(
                self.path,
                dtype=">f4",
                mode="r+",
                offset=len(encoded),
                shape=self.shape,
                order="C",
            )
        except FileExistsError as error:
            raise CalibrationError(
                "OUTPUT_EXISTS", "refusing to overwrite output", path=str(self.path)
            ) from error
        return self

    def write_rows(self, y0: int, values: NDArray[Any]) -> None:
        if self._mapping is None:
            raise RuntimeError("FitsFloatWriter is not open")
        rows = np.asarray(values, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.shape[1]:
            raise ValueError("output tile has the wrong shape")
        y1 = y0 + rows.shape[0]
        if y0 < 0 or y1 > self.shape[0]:
            raise ValueError("output tile is outside image bounds")
        self._mapping[y0:y1] = rows

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.flush()
            self._mapping = None
            with self.path.open("r+b") as stream:
                os.fsync(stream.fileno())

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
) -> None:
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

    for local_row, absolute_y in enumerate(range(y0, y1)):
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
    chunks: list[NDArray[np.float32]] = []
    for y in y_indices:
        row = _expression_rows(
            expression,
            sources,
            int(y),
            int(y) + 1,
            division_floor=division_floor,
        )
        sampled = row[0, x_indices]
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
            selected = array[finite]
            self.minimum = min(self.minimum, float(np.min(selected)))
            self.maximum = max(self.maximum, float(np.max(selected)))
            self.total += float(np.sum(selected, dtype=np.float64))

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

    def serializable(self) -> dict[str, Any]:
        return {
            "sigmaClip": self.sigma_clip,
            "minimumRejectionFrames": self.minimum_rejection_frames,
            "maxMemoryBytes": self.max_memory_bytes,
            "maxStatisticsSamples": self.max_statistics_samples,
            "divisionFloor": self.division_floor,
            "transientRejection": self.transient_rejection,
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

    def serializable(self) -> dict[str, Any]:
        return {
            "outputPath": self.output_path,
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
        }


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
) -> _RejectionSigmaFloor:
    """Estimate one group-wide floor without a full-image statistics pass.

    Each sampled coordinate is evaluated across the complete frame group. Rows
    are processed one at a time, so temporary memory is bounded by one sampled
    row times the frame count. Only the final scalar floor is consumed by tiled
    integration, making rejection decisions independent of tile partitioning.
    """

    sample_limit = min(
        int(parameters.max_statistics_samples), REJECTION_FLOOR_MAX_SAMPLES
    )
    y_indices, x_indices = _sample_coordinates(shape, sample_limit)
    coordinate_count = int(y_indices.size * x_indices.size)
    coordinate_sha256 = _coordinate_digest(shape, y_indices, x_indices)
    applicable = len(expressions) >= parameters.minimum_rejection_frames
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

    sigma_chunks: list[NDArray[np.float32]] = []
    for y in y_indices:
        row_values = np.empty((len(expressions), x_indices.size), dtype=np.float32)
        for frame_index, expression in enumerate(expressions):
            row = _expression_rows(
                expression,
                sources,
                int(y),
                int(y) + 1,
                division_floor=parameters.division_floor,
            )
            row_values[frame_index] = row[0, x_indices]
        finite_count = np.count_nonzero(np.isfinite(row_values), axis=0)
        eligible = finite_count >= parameters.minimum_rejection_frames
        if not np.any(eligible):
            continue
        selected = row_values[:, eligible]
        selected[~np.isfinite(selected)] = np.nan
        center = np.nanmedian(selected, axis=0)
        mad = np.nanmedian(np.abs(selected - center[None, :]), axis=0)
        robust_sigma = np.asarray(np.float32(1.4826) * mad, dtype=np.float32)
        usable = np.isfinite(robust_sigma) & (robust_sigma > 0)
        if np.any(usable):
            sigma_chunks.append(robust_sigma[usable])

    if sigma_chunks:
        sampled_sigma = np.concatenate(sigma_chunks)
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
    )


def _ordinary_mad_rejection_decision(
    values: NDArray[np.float32],
    parameters: IntegrationParameters,
    sigma_floor: _RejectionSigmaFloor,
    *,
    transient_model: TransientRejectionModel | None = None,
    first_row: int = 0,
) -> tuple[NDArray[np.bool_], NDArray[np.float32], NDArray[np.bool_]]:
    """Return finite samples, per-pixel centre, and accepted samples.

    This is the single rejection-decision implementation shared by portable CPU
    integration and the CPU-produced mask consumed by Metal.
    """

    samples = np.asarray(values, dtype=np.float32)
    if samples.ndim != 3:
        raise ValueError("ordinary integration values must be frame-major 3-D")
    finite = np.isfinite(samples)
    valid_pixels = np.any(finite, axis=0)
    center = np.full(samples.shape[1:], np.nan, dtype=np.float32)
    if np.any(valid_pixels):
        selected = samples[:, valid_pixels]
        selected[~np.isfinite(selected)] = np.nan
        center[valid_pixels] = np.asarray(
            np.nanmedian(selected, axis=0), dtype=np.float32
        )
    if samples.shape[0] < parameters.minimum_rejection_frames:
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
    robust_sigma = np.asarray(np.float32(1.4826) * mad, dtype=np.float32)
    numerical_floor = np.maximum(
        np.float32(REJECTION_FLOOR_ABSOLUTE),
        np.float32(REJECTION_FLOOR_EPSILON_FACTOR * np.finfo(np.float32).eps)
        * np.maximum(np.float32(1.0), np.abs(center)),
    )
    effective_sigma = np.maximum(
        np.maximum(robust_sigma, np.float32(sigma_floor.group_sigma_floor)),
        numerical_floor,
    )
    threshold = np.float32(parameters.sigma_clip) * effective_sigma
    accepted = finite & (np.abs(samples - center[None, :, :]) <= threshold)
    # Dither boundaries and masked detector defects may have fewer usable
    # samples than the group size. Do not infer outliers from too few samples.
    enough_samples = (
        np.count_nonzero(finite, axis=0) >= parameters.minimum_rejection_frames
    )
    accepted |= finite & ~enough_samples[None, :, :]
    if transient_model is not None:
        transient_model.reject_rows(accepted, first_row, enough_samples)
    return finite, center, accepted


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
) -> TransientRejectionModel:
    """Read block means once and fit one tile-independent spatial mask model."""
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
    for index, expression in enumerate(expressions):
        for row in range(by):
            y0, y1 = row * factor, min(height, (row + 1) * factor)
            values = _expression_rows(expression, sources, y0, y1,
                                      division_floor=parameters.division_floor)
            finite = np.isfinite(values)
            sums = np.sum(np.where(finite, values, 0), axis=0, dtype=np.float64)
            counts = np.sum(finite, axis=0)
            offsets = np.arange(0, width, factor)
            sums = np.add.reduceat(sums, offsets)
            counts = np.add.reduceat(counts, offsets)
            # Exclude partly covered blocks from spatial detection. Ordinary
            # per-pixel rejection still handles all valid edge samples.
            expected = (y1 - y0) * np.minimum(factor, width - offsets)
            np.divide(sums, counts, out=preview[index, row],
                      where=counts == expected, casting="unsafe")
    try:
        background = fit_residual_background(preview, factor, weights)
    except ValueError as error:
        raise CalibrationError("RESIDUAL_BACKGROUND_UNDERCONSTRAINED", str(error)) from error
    if background is not None:
        background.apply_coordinates(preview,
            np.arange(bx,dtype=np.float64)*factor+(factor-1)/2,
            np.arange(by,dtype=np.float64)*factor+(factor-1)/2)
    model = detect_transient_trails(preview, factor)
    return TransientRejectionModel(factor, model.trails, model.status, background)


def _ordinary_integration_tile(
    values: NDArray[np.float32], parameters: IntegrationParameters,
    sigma_floor: _RejectionSigmaFloor, transient_model: TransientRejectionModel,
    first_row: int,
) -> tuple[NDArray[np.bool_], NDArray[np.float32], NDArray[np.bool_]]:
    """Original MAD decisions plus spatial rejection, with its sky reference kept.

    Only pixels with an additional spatial rejection have their temporary sample
    values normalized. This routine is shared by CPU and Metal preparation.
    """
    finite, center, accepted = _ordinary_mad_rejection_decision(values, parameters, sigma_floor)
    if transient_model.trails:
        original = accepted.copy()
        enough = np.sum(finite, axis=0) >= parameters.minimum_rejection_frames
        transient_model.reject_rows(accepted, first_row, enough)
        transient_model.normalize_rejected_rows(values, first_row, original, accepted)
    return finite, center, accepted


def _normalized_noise_weights(
    expressions: tuple[FrameExpression, ...],
    sources: Mapping[str, FitsFrame],
    shape: tuple[int, int],
    parameters: IntegrationParameters,
) -> tuple[NDArray[np.float64], tuple[float, ...]]:
    estimates: list[float] = []
    for expression in expressions:
        sample = _sample_expression(
            expression,
            sources,
            shape,
            max_samples=parameters.max_statistics_samples,
            division_floor=parameters.division_floor,
        ).astype(np.float64, copy=False)
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        sigma = 1.4826 * mad
        if not math.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        estimates.append(sigma)
    raw = 1.0 / np.square(np.asarray(estimates, dtype=np.float64))
    # Prevent one almost-noiseless frame from completely dominating a group.
    positive = raw[np.isfinite(raw) & (raw > 0)]
    if not positive.size:
        raw = np.ones(len(expressions), dtype=np.float64)
    else:
        median_weight = float(np.median(positive))
        raw = np.clip(raw, median_weight / 16.0, median_weight * 16.0)
    normalized = raw / np.sum(raw)
    return normalized, tuple(float(value) for value in normalized)


def _combined_integration_weights(
    expressions: tuple[FrameExpression, ...],
    sources: Mapping[str, FitsFrame],
    shape: tuple[int, int],
    parameters: IntegrationParameters,
    quality_weights: Sequence[float] | None,
) -> tuple[
    NDArray[np.float64], tuple[float, ...], tuple[float, ...], tuple[float, ...]
]:
    noise, serialized_noise = _normalized_noise_weights(
        expressions, sources, shape, parameters
    )
    if quality_weights is None:
        return noise, serialized_noise, (), serialized_noise
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
    )


def integrate_expressions(
    expressions: Iterable[FrameExpression],
    output_path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    parameters: IntegrationParameters | None = None,
    quality_weights: Sequence[float] | None = None,
    map_paths: IntegrationMapPaths | None = None,
) -> IntegrationResult:
    """Robustly integrate expressions without materializing full input images."""

    parameters = parameters or IntegrationParameters()
    parameters.validate()
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
            ) = _combined_integration_weights(
                canonical, sources, shape, parameters, quality_weights
            )
            rejection_sigma_floor = _estimate_rejection_sigma_floor(
                canonical, sources, shape, parameters
            )
            transient_model = _prepare_transient_rejection(canonical, sources, shape, parameters, weights)
            stats = _StatsAccumulator()
            accepted_total = 0
            rejected_total = 0
            output_metadata = dict(metadata or {})
            output_metadata.setdefault("OAFSTATE", "UNSOLVED_WORKING")
            output_metadata.setdefault("OAFNFRM", len(canonical))
            output_metadata.setdefault("OAFREJ", parameters.sigma_clip)
            writer = stack.enter_context(
                FitsFloatWriter(temporary, shape, output_metadata)
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
                    FitsFloatWriter(temporary_map, shape, map_metadata[name])
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
                )
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
                accepted_count = int(np.count_nonzero(accepted))
                finite_count = int(np.count_nonzero(finite))
                accepted_total += accepted_count
                rejected_total += finite_count - accepted_count
                stats.update(result)
                writer.write_rows(y0, result)
                if map_writers:
                    accepted_per_pixel = np.sum(
                        accepted, axis=0, dtype=np.uint16
                    ).astype(np.float32)
                    rejected_per_pixel = np.sum(
                        finite & ~accepted, axis=0, dtype=np.uint16
                    ).astype(np.float32)
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
            execution={
                "requestedBackend": "portable-cpu",
                "selectedBackend": "portable-cpu",
                "acceleratorUsed": False,
                "rejectionMask": {
                    "producer": "portable-cpu",
                    "method": "median-mad-sigma",
                    "sigma": parameters.sigma_clip,
                    "scope": "all-frames-per-pixel",
                    "partialMeanBatching": False,
                    "sigmaFloor": rejection_sigma_floor.serializable(),
                    "spatialTransients": transient_model.serializable(),
                },
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
