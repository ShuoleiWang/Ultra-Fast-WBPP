"""Corresponding original-pixel Blink crops; display-only and input-read-only."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from astropy.io import fits
import numpy as np
from scipy import ndimage

from lightframeqc.readers import read_frame_preview
from .blink_diagnostics import DisplayReference, detail_transfer, local_noise

@lru_cache(maxsize=2)
def native_master(path: str) -> np.ndarray:
    return read_frame_preview(path, max_long_edge=100000).data


def star_positions(reference: DisplayReference) -> list[tuple[float, float]]:
    detail = reference.image - reference.background
    h, w = detail.shape
    peaks = detail == ndimage.maximum_filter(detail, size=7)
    snr = detail / reference.noise
    peaks &= (snr > 20) & (snr < 300)
    points = []
    for gy in range(3):
        for gx in range(3):
            y0, y1 = int(h*(gy+0.15)/3), int(h*(gy+0.85)/3)
            x0, x1 = int(w*(gx+0.15)/3), int(w*(gx+0.85)/3)
            local = snr[y0:y1, x0:x1]
            yy, xx = np.indices(local.shape)
            distance = ((yy-local.shape[0]/2)/local.shape[0])**2 + ((xx-local.shape[1]/2)/local.shape[1])**2
            score = np.where(peaks[y0:y1, x0:x1], -np.abs(np.log(np.maximum(local, 1)/120))-distance, -np.inf)
            if not np.any(np.isfinite(score)):
                points.append((float((x0+x1)/2), float((y0+y1)/2)))
                continue
            y, x = np.unravel_index(np.argmax(score), score.shape)
            points.append((float(x+x0), float(y+y0)))
    return points


@dataclass(frozen=True)
class NativeAtlas:
    signal: np.ndarray
    shape: np.ndarray
    shape_regions: int


def shape_crop(signal: np.ndarray) -> np.ndarray | None:
    half = signal.shape[0]//2
    core = signal[half-8:half+9, half-8:half+9]
    peak = float(np.percentile(core, 99))
    try:
        noise = local_noise(signal)
    except ValueError:
        return None
    if not np.isfinite(peak) or peak < 10 * noise:
        return None
    # Only this explicitly labelled morphology panel matches star amplitude.
    # It must not be mistaken for retained flux or comparable image noise.
    return np.sqrt(np.clip(signal/peak, 0, 1)).astype(np.float32)


def native_atlas(frame: dict, transform: np.ndarray | None, points: list[tuple[float,float]], reference: DisplayReference, flat_path: str | None, dark_path: str | None, scale: float) -> NativeAtlas | None:
    if transform is None or Path(frame["path"]).suffix.lower() not in {".fit", ".fits", ".fts"}:
        return None
    half=40
    atlas=np.zeros((3*(2*half+4),3*(2*half+4)),np.float32)
    shape_atlas=np.zeros_like(atlas); shape_regions=0
    flat = native_master(flat_path) if flat_path else None
    dark = native_master(dark_path) if dark_path else None
    flat_scale = float(np.nanmedian(flat)) if flat is not None else 1.0
    pedestal_scale = 65535 if dark is not None and np.nanmax(dark) <= 1.001 else 1
    # Read exactly the same sky positions, but display original source pixels:
    # no registration interpolation, sharpening or denoising in the crop.
    inverse=np.linalg.inv(transform)
    with fits.open(frame["path"],memmap=False) as hdus:
        hdu=next(hdu for hdu in hdus if hdu.header.get("NAXIS")==2)
        height, width = hdu.header["NAXIS2"], hdu.header["NAXIS1"]
        if any(master is not None and master.shape != (height, width) for master in (flat, dark)):
            raise ValueError("native master geometry mismatch")
        for index,(x,y) in enumerate(points):
            xx,yy,_=inverse@np.array([x,y,1])
            xx,yy=round((xx+0.5)*scale-0.5),round((yy+0.5)*scale-0.5)
            if yy-half<0 or xx-half<0 or yy+half>height or xx+half>width:
                continue
            sl=np.s_[yy-half:yy+half,xx-half:xx+half]
            data=np.asarray(hdu.section[sl],np.float32)
            if dark is not None:
                data = data - dark[sl] * pedestal_scale
            if flat is not None:
                data = data / np.where(flat[sl] / flat_scale > 0.05, flat[sl] / flat_scale, np.nan)
            if not np.all(np.isfinite(data)):
                continue
            # A 180-degree meridian flip is a pixel-preserving rotation.
            if transform[0,0]<0 and transform[1,1]<0:
                data=np.rot90(data,2)
            signal=data-np.median(data)
            pixels=detail_transfer(signal,reference.noise*scale)
            shape=shape_crop(signal)
            row,col=divmod(index,3)
            atlas[row*(2*half+4):row*(2*half+4)+2*half,col*(2*half+4):col*(2*half+4)+2*half]=pixels
            if shape is not None:
                shape_atlas[row*(2*half+4):row*(2*half+4)+2*half,col*(2*half+4):col*(2*half+4)+2*half]=shape
                shape_regions+=1
    return NativeAtlas(atlas,shape_atlas,shape_regions)
