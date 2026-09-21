"""Native-resolution PSF measurement of the brightest catalogue stars.

The QC catalogue is measured on a block-mean preview (typically one third of
the native resolution), which inflates and desensitises FWHM.  This module
cuts small native-resolution stamps around the brightest unflagged preview
detections and measures a robust half-flux radius and a wing fraction, the
two quantities the unattended selection needs to weight soft frames and to
recognise dew or halo growth.  It is deliberately simple: no PSF fitting, no
external dependencies beyond NumPy and astropy.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray

from .models import Star
from .xisf import XISF
from .cfa import is_cfa_pattern, luminance, normalize_pattern, shifted_pattern
from .fits_bands import close_image_data, open_fits_image_data


@dataclass(frozen=True)
class NativePsfSummary:
    star_count: int
    r50_pixels: float | None
    fwhm_pixels: float | None
    wing_fraction: float | None
    evidence: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "starCount": self.star_count,
            "r50Pixels": self.r50_pixels,
            "fwhmPixels": self.fwhm_pixels,
            "wingFraction": self.wing_fraction,
            **self.evidence,
        }


class NativeImage:
    """Read-only 2-D view of a native frame that scales stamps on access.

    ``data`` is the raw (memory-mapped or decoded) array in one of the
    layouts ``mono`` (H, W), ``channels_first`` (C, H, W) or
    ``channels_last`` (H, W, C); integer FITS data keeps its stored values and
    ``bscale``/``bzero`` are applied to each stamp, so a 16-bit Light with
    BZERO = 32768 is never decoded as a whole.  Multi-channel data returns the
    finite channel mean, as the preview reader does.
    """

    def __init__(
        self,
        data: Any,
        layout: str = "mono",
        *,
        bscale: float = 1.0,
        bzero: float = 0.0,
        blank: int | float | None = None,
        cfa_pattern: str | None = None,
    ) -> None:
        if layout == "mono":
            height, width = int(data.shape[0]), int(data.shape[1])
        elif layout == "channels_first":
            height, width = int(data.shape[1]), int(data.shape[2])
        elif layout == "channels_last":
            height, width = int(data.shape[0]), int(data.shape[1])
        else:
            raise ValueError(f"unsupported native image layout {layout!r}")
        self._data = data
        self._layout = layout
        self.shape = (height, width)
        self.bscale = float(bscale)
        self.bzero = float(bzero)
        self.blank = blank
        # A Bayer mosaic returns the bilinear-debayered luminance of each
        # stamp, so star profiles are measured on a smooth image instead of
        # the colour checkerboard.
        self.cfa_pattern = cfa_pattern if layout == "mono" and is_cfa_pattern(cfa_pattern) else None

    def __getitem__(self, key: tuple[slice, slice]) -> NDArray[np.float64]:
        rows, columns = key
        if self._layout == "mono" and self.cfa_pattern is not None:
            height, width = self.shape
            y0, y1, _ = rows.indices(height)
            x0, x1, _ = columns.indices(width)
            # One pixel of halo so the edge pixels of the stamp interpolate
            # from real neighbours; frame edges are replicated by the debayer.
            hy0, hy1 = max(0, y0 - 1), min(height, y1 + 1)
            hx0, hx1 = max(0, x0 - 1), min(width, x1 + 1)
            raw = np.asanyarray(self._data[hy0:hy1, hx0:hx1])
            values = np.asarray(raw, dtype=np.float64)
            if self.blank is not None and np.issubdtype(raw.dtype, np.integer):
                values[np.asarray(raw) == self.blank] = np.nan
            if self.bscale != 1.0 or self.bzero != 0.0:
                values = values * self.bscale + self.bzero
            lum = luminance(values, shifted_pattern(self.cfa_pattern, hy0, hx0))
            return np.asarray(lum[y0 - hy0 : y1 - hy0, x0 - hx0 : x1 - hx0], dtype=np.float64)
        if self._layout == "mono":
            raw = np.asanyarray(self._data[rows, columns])
            values = np.asarray(raw, dtype=np.float64)
            if self.blank is not None and np.issubdtype(raw.dtype, np.integer):
                values[np.asarray(raw) == self.blank] = np.nan
        else:
            axis = 0 if self._layout == "channels_first" else 2
            raw = (
                np.asanyarray(self._data[:, rows, columns])
                if axis == 0
                else np.asanyarray(self._data[rows, columns, :])
            )
            values = np.asarray(raw, dtype=np.float64)
            if self.blank is not None and np.issubdtype(raw.dtype, np.integer):
                values[np.asarray(raw) == self.blank] = np.nan
            finite = np.isfinite(values)
            total = np.where(finite, values, 0.0).sum(axis=axis)
            count = finite.sum(axis=axis)
            values = np.full(total.shape, np.nan, dtype=np.float64)
            np.divide(total, count, out=values, where=count > 0)
        if self.bscale != 1.0 or self.bzero != 0.0:
            values = values * self.bscale + self.bzero
        return values


def _fits_layout(shape: tuple[int, ...]) -> str | None:
    if len(shape) == 2:
        return "mono"
    if len(shape) == 3:
        if shape[0] in (1, 3, 4) and shape[0] <= shape[1] and shape[0] <= shape[2]:
            return "channels_first"
        if shape[2] in (1, 3, 4):
            return "channels_last"
    return None


@contextmanager
def open_native_image(path: str) -> Iterator[NativeImage | None]:
    """Open a FITS or XISF Light for stamp reads without decoding it whole.

    FITS data is memory mapped with ``do_not_scale_image_data`` so scaled
    integer files (BZERO/BSCALE) open; the scaling is applied per stamp.
    Uncompressed XISF attachments are memory mapped at their offset; other
    XISF images are decoded through the XISF reader.  Yields None when the
    file holds no usable image.
    """

    suffix = Path(path).suffix.lower()
    if suffix == ".xisf":
        document = XISF(str(path))
        images = document.get_images_metadata()
        if not images:
            yield None
            return
        image_metadata = images[0]
        geometry = tuple(int(value) for value in image_metadata["geometry"])
        if len(geometry) != 3:
            yield None
            return
        width, height, channels = geometry
        location = tuple(image_metadata["location"])
        dtype = _xisf_dtype(image_metadata)
        if (
            dtype is not None
            and location
            and location[0] == "attachment"
            and "compression" not in image_metadata
        ):
            _, offset, _stored = location
            mapped = np.memmap(
                str(path),
                dtype=dtype,
                mode="r",
                offset=int(offset),
                shape=(channels, height, width),
            )
            pattern = _xisf_cfa_pattern(image_metadata) if channels == 1 else None
            try:
                if channels == 1:
                    yield NativeImage(mapped[0], "mono", cfa_pattern=pattern)
                else:
                    yield NativeImage(mapped, "channels_first")
            finally:
                del mapped
            return
        decoded = np.asarray(document.read_image(0, data_format="channels_last"))
        if channels == 1:
            yield NativeImage(decoded[:, :, 0], "mono", cfa_pattern=_xisf_cfa_pattern(image_metadata))
        else:
            yield NativeImage(decoded, "channels_last")
        return
    with fits.open(
        path,
        mode="readonly",
        memmap=True,
        lazy_load_hdus=True,
        do_not_scale_image_data=True,
        uint=False,
        checksum=False,
    ) as hdul:
        for hdu in hdul:
            if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
                continue
            # The memory map, or on Windows a band reader over the same bytes
            # (``fits_bands``); stamps are small row bands either way.
            data = open_fits_image_data(hdu, path)
            if data is None:
                continue
            layout = _fits_layout(tuple(data.shape))
            if layout is None:
                close_image_data(data)
                continue
            try:
                yield NativeImage(
                    data,
                    layout,
                    bscale=float(hdu.header.get("BSCALE", 1.0) or 1.0),
                    bzero=float(hdu.header.get("BZERO", 0.0) or 0.0),
                    blank=hdu.header.get("BLANK"),
                    cfa_pattern=_fits_cfa_pattern(hdu.header),
                )
            finally:
                close_image_data(data)
            return
        yield None


def _fits_cfa_pattern(header: Any) -> str | None:
    for key in ("BAYERPAT", "BAYERPATN", "CFAPAT", "CFAPATTERN"):
        value = header.get(key)
        if value is not None and str(value).strip():
            return normalize_pattern(value)
    return None


def _xisf_cfa_pattern(image_metadata: Any) -> str | None:
    keywords = image_metadata.get("FITSKeywords", {}) if isinstance(image_metadata, dict) else {}
    for key in ("BAYERPAT", "BAYERPATN", "CFAPAT", "CFAPATTERN"):
        entries = keywords.get(key) if isinstance(keywords, dict) else None
        if entries:
            value = entries[0].get("value") if isinstance(entries[0], dict) else entries[0]
            if value is not None and str(value).strip():
                return normalize_pattern(value)
    properties = image_metadata.get("XISFProperties", {}) if isinstance(image_metadata, dict) else {}
    if isinstance(properties, dict):
        entry = properties.get("PCL:CFASourcePattern")
        value = entry.get("value") if isinstance(entry, dict) else entry
        if value is not None and str(value).strip():
            return normalize_pattern(value)
    return None


def _xisf_dtype(image_metadata: Any) -> np.dtype[Any] | None:
    """The stored sample dtype of an XISF image (None when it is not plain)."""

    from .readers import _xisf_file_dtype

    try:
        return _xisf_file_dtype(image_metadata)
    except (KeyError, TypeError, ValueError):
        return None


def measure_native_psf_from_array(
    image: np.ndarray,
    stars: Sequence[Star],
    scale_x: float,
    scale_y: float,
    *,
    max_stars: int = 200,
    radius: int = 10,
    minimum_peak_sigma: float = 25.0,
    minimum_stars: int = 12,
) -> NativePsfSummary:
    """Measure r50 and wing fraction of the brightest stars on a native image."""

    height, width = image.shape
    half = radius + 4
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    ring = np.hypot(yy, xx) > radius
    r50_values: list[float] = []
    wing_values: list[float] = []
    rejected = {"edge": 0, "nonfinite": 0, "faint": 0, "flat": 0, "drift": 0, "shape": 0}
    ordered = sorted(
        (star for star in stars if star.flags == 0 and star.flux > 0),
        key=lambda star: -star.flux,
    )
    for star in ordered:
        if len(r50_values) >= max_stars:
            break
        cx = (star.x + 0.5) * scale_x - 0.5
        cy = (star.y + 0.5) * scale_y - 0.5
        ix, iy = int(round(cx)), int(round(cy))
        if ix - half < 0 or iy - half < 0 or ix + half >= width or iy + half >= height:
            rejected["edge"] += 1
            continue
        stamp = np.asarray(image[iy - half : iy + half + 1, ix - half : ix + half + 1], dtype=np.float64)
        if not np.all(np.isfinite(stamp)):
            rejected["nonfinite"] += 1
            continue
        background = float(np.median(stamp[ring]))
        noise = float(1.4826 * np.median(np.abs(stamp[ring] - background)))
        signal = stamp - background
        peak = float(signal.max())
        if peak < minimum_peak_sigma * max(noise, 1e-9):
            rejected["faint"] += 1
            continue
        if int(np.count_nonzero(signal >= 0.98 * peak)) >= 3:
            rejected["flat"] += 1
            continue
        # Flux-weighted centroid within 5 px, two iterations.
        ox, oy = float(half), float(half)
        for _ in range(2):
            distance = np.hypot(yy - (oy - half), xx - (ox - half))
            core = (distance <= 5.0) & (signal > 0)
            weight = np.where(core, signal, 0.0)
            total = float(weight.sum())
            if total <= 0:
                break
            ox = float((weight * (xx + half)).sum() / total)
            oy = float((weight * (yy + half)).sum() / total)
        if np.hypot(ox - half, oy - half) > 3.0:
            rejected["drift"] += 1
            continue
        distance = np.hypot(yy - (oy - half), xx - (ox - half)).ravel()
        values = signal.ravel()
        inside = distance <= radius
        order = np.argsort(distance[inside], kind="stable")
        radii = distance[inside][order]
        cumulative = np.cumsum(values[inside][order])
        total = float(cumulative[-1])
        if total <= 0:
            rejected["shape"] += 1
            continue
        target = 0.5 * total
        index = int(np.searchsorted(cumulative, target))
        if index >= radii.size:
            rejected["shape"] += 1
            continue
        if index == 0:
            r50 = float(radii[0])
        else:
            lower, upper = cumulative[index - 1], cumulative[index]
            fraction = (target - lower) / max(upper - lower, 1e-12)
            r50 = float(radii[index - 1] + fraction * (radii[index] - radii[index - 1]))
        if not 0.3 <= r50 <= radius / 2.0:
            rejected["shape"] += 1
            continue
        # Wing fraction: flux beyond one FWHM (2 r50) relative to the flux
        # within ``radius``; about 0.06 for a Gaussian, rising with halos.
        core_radius = min(2.0 * r50, float(radius))
        core_flux = float(cumulative[np.searchsorted(radii, core_radius, side="right") - 1])
        r50_values.append(r50)
        wing_values.append(max(0.0, 1.0 - core_flux / total))
    evidence = {
        "candidates": len(ordered),
        "rejected": rejected,
        "radiusPixels": radius,
        "algorithm": "native-stamp-half-flux-radius-v1",
    }
    if len(r50_values) < minimum_stars:
        return NativePsfSummary(len(r50_values), None, None, None, evidence)
    r50 = float(np.median(r50_values))
    return NativePsfSummary(
        star_count=len(r50_values),
        r50_pixels=r50,
        fwhm_pixels=2.0 * r50,
        wing_fraction=float(np.median(wing_values)),
        evidence=evidence,
    )


def measure_native_psf(
    path: str,
    stars: Sequence[Star],
    scale_x: float,
    scale_y: float,
    **options: Any,
) -> NativePsfSummary:
    """Open ``path`` read-only (memory mapped where possible) and measure the native PSF."""

    with open_native_image(str(path)) as image:
        if image is None:
            return NativePsfSummary(0, None, None, None, {"error": "no 2-D image"})
        return measure_native_psf_from_array(image, stars, scale_x, scale_y, **options)
