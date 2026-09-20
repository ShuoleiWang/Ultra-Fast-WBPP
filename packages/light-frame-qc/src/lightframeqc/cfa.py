"""Bayer colour filter array helpers shared by QC, registration and the engine.

A CFA (one-shot-colour) frame stores one colour sample per pixel in a 2x2
tile.  The pattern names the tile row-major from the frame's origin:
``RGGB`` means pixel (0, 0) is red, (0, 1) and (1, 0) are green and (1, 1)
is blue.  Everything here is a pure function of the mosaic and the pattern.

``bilinear_debayer`` reconstructs the three colour planes at full resolution
(each missing sample is the mean of its nearest same-colour neighbours; the
frame edges are replicated), which is the same-position, unbiased estimate a
star centroid or a stamp measurement needs.  ``luminance`` averages the
planes with equal weight, so a mosaic of a grey scene with per-channel gains
becomes a smooth image whose star shapes are those of the sensor.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Channel index 0 red, 1 green, 2 blue at tile position (y & 1, x & 1).
CFA_PATTERNS: dict[str, tuple[int, int, int, int]] = {
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}
CHANNEL_NAMES: tuple[str, str, str] = ("R", "G", "B")
MONO_PATTERNS: frozenset[str] = frozenset({"NONE", "NOCFA", "MONO", "FALSE", "0"})
UNKNOWN_PATTERNS: frozenset[str] = frozenset({"", "UNKNOWN", "AUTO", "UNSPECIFIED"})


def normalize_pattern(value: Any | None) -> str:
    """Canonical pattern name: a Bayer pattern, ``NONE`` for mono, else ``UNKNOWN``."""

    if value is None:
        return "UNKNOWN"
    compact = re.sub(r"[^A-Z0-9]+", "", str(value).upper())
    if compact in UNKNOWN_PATTERNS:
        return "UNKNOWN"
    if compact in MONO_PATTERNS:
        return "NONE"
    return compact


def is_cfa_pattern(value: Any | None) -> bool:
    """True when ``value`` names one of the four supported Bayer patterns."""

    return normalize_pattern(value) in CFA_PATTERNS


def pattern_layout(pattern: str) -> tuple[int, int, int, int]:
    """Channel index of the tile positions (0,0), (0,1), (1,0), (1,1)."""

    name = normalize_pattern(pattern)
    try:
        return CFA_PATTERNS[name]
    except KeyError:
        raise ValueError(f"unsupported CFA pattern {pattern!r}") from None


def shifted_pattern(pattern: str, dy: int, dx: int) -> str:
    """Pattern of the sub-mosaic whose origin sits at ``(dy, dx)`` of the frame."""

    layout = pattern_layout(pattern)
    shifted = tuple(layout[(((index >> 1) + dy) & 1) << 1 | (((index & 1) + dx) & 1)] for index in range(4))
    for name, candidate in CFA_PATTERNS.items():
        if candidate == shifted:
            return name
    raise ValueError(f"unsupported CFA pattern {pattern!r}")  # pragma: no cover


def channel_offsets(pattern: str, channel: int) -> tuple[tuple[int, int], ...]:
    """Tile positions ``(dy, dx)`` that carry ``channel`` (two for green)."""

    layout = pattern_layout(pattern)
    return tuple(
        (index >> 1, index & 1) for index, value in enumerate(layout) if value == channel
    )


def channel_mask(shape: tuple[int, int], pattern: str, channel: int) -> NDArray[np.bool_]:
    """Boolean mask of the mosaic pixels that sample ``channel``."""

    mask = np.zeros(shape, dtype=bool)
    for dy, dx in channel_offsets(pattern, channel):
        mask[dy::2, dx::2] = True
    return mask


def channel_medians(mosaic: NDArray[Any], pattern: str) -> tuple[float, float, float]:
    """Median of the finite samples of each colour channel of the mosaic."""

    values = np.asarray(mosaic)
    result = []
    for channel in range(3):
        samples = np.concatenate(
            [values[dy::2, dx::2].reshape(-1) for dy, dx in channel_offsets(pattern, channel)]
        )
        finite = samples[np.isfinite(samples)]
        result.append(float(np.median(finite)) if finite.size else float("nan"))
    return result[0], result[1], result[2]


def pattern_scale_map(
    shape: tuple[int, int], pattern: str, channel_scales: tuple[float, float, float]
) -> NDArray[np.float32]:
    """Per-pixel multiplier that applies ``channel_scales`` by Bayer position."""

    layout = pattern_layout(pattern)
    scales = np.empty(shape, dtype=np.float32)
    for index, channel in enumerate(layout):
        scales[index >> 1 :: 2, index & 1 :: 2] = np.float32(channel_scales[channel])
    return scales


def _fill_plane(
    values: NDArray[np.float64],
    known: NDArray[np.bool_],
    first: int,
    last: int,
) -> NDArray[np.float32]:
    """Fill rows ``[first, last)`` of one colour plane band from its known samples.

    ``values`` holds the known samples (zeros elsewhere) of a band with its
    one-row halo where the frame provides one.  An unknown pixel becomes the
    mean of its known 4-neighbours (checkerboard green, or the two same-row /
    same-column lattice neighbours of red and blue), else the mean of its
    known diagonal neighbours (the lattice channels' odd/odd positions), else
    NaN.  Frame edges reuse the nearest sample.  Sums are formed in Float64 in
    a fixed order — ``(left + right) + (up + down)`` — so a native kernel can
    reproduce every value exactly.
    """

    rows = last - first
    pad_top = 1 if first == 0 else 0
    pad_bottom = 1 if last == values.shape[0] else 0
    padded = np.pad(values, ((pad_top, pad_bottom), (1, 1)), mode="edge")
    padded_known = np.pad(known, ((pad_top, pad_bottom), (1, 1)), mode="edge")
    # With the pads, band row ``first`` sits at padded row 1 when the frame
    # has no halo above it, and at padded row ``first`` (>= 1) otherwise.
    base = first + pad_top
    centre = padded[base : base + rows, 1:-1]
    centre_known = padded_known[base : base + rows, 1:-1]
    left, right = padded[base : base + rows, :-2], padded[base : base + rows, 2:]
    up, down = padded[base - 1 : base - 1 + rows, 1:-1], padded[base + 1 : base + 1 + rows, 1:-1]
    left_known, right_known = padded_known[base : base + rows, :-2], padded_known[base : base + rows, 2:]
    up_known = padded_known[base - 1 : base - 1 + rows, 1:-1]
    down_known = padded_known[base + 1 : base + 1 + rows, 1:-1]
    horizontal = np.where(left_known, left, 0.0) + np.where(right_known, right, 0.0)
    vertical = np.where(up_known, up, 0.0) + np.where(down_known, down, 0.0)
    count = (
        left_known.astype(np.int8) + right_known.astype(np.int8)
        + up_known.astype(np.int8) + down_known.astype(np.int8)
    )
    total = horizontal + vertical
    result = np.array(centre, dtype=np.float64)
    unknown = ~centre_known
    direct = unknown & (count > 0)
    result[direct] = total[direct] / count[direct]
    remaining = unknown & (count == 0)
    if np.any(remaining):
        corners = (
            (padded[base - 1 : base - 1 + rows, :-2], padded_known[base - 1 : base - 1 + rows, :-2]),
            (padded[base - 1 : base - 1 + rows, 2:], padded_known[base - 1 : base - 1 + rows, 2:]),
            (padded[base + 1 : base + 1 + rows, :-2], padded_known[base + 1 : base + 1 + rows, :-2]),
            (padded[base + 1 : base + 1 + rows, 2:], padded_known[base + 1 : base + 1 + rows, 2:]),
        )
        corner_total = (
            (np.where(corners[0][1], corners[0][0], 0.0) + np.where(corners[1][1], corners[1][0], 0.0))
            + (np.where(corners[2][1], corners[2][0], 0.0) + np.where(corners[3][1], corners[3][0], 0.0))
        )
        corner_count = sum(flag.astype(np.int8) for _value, flag in corners)
        fillable = remaining & (corner_count > 0)
        result[fillable] = corner_total[fillable] / corner_count[fillable]
        result[remaining & (corner_count == 0)] = np.nan
    return result.astype(np.float32)


DEBAYER_BAND_ROWS = 256


def bilinear_debayer(mosaic: NDArray[Any], pattern: str) -> NDArray[np.float32]:
    """Reconstruct the ``(3, H, W)`` R/G/B planes of a Bayer mosaic.

    Non-finite mosaic samples are missing: their neighbours interpolate
    without them and the plane pixel at their own position is NaN (as is any
    pixel with no same-colour sample around it).  The work proceeds in row
    bands, so the only frame-sized allocation is the result.
    """

    values = np.asarray(mosaic)
    if values.ndim != 2:
        raise ValueError("a Bayer mosaic must be a 2-D array")
    layout = pattern_layout(pattern)
    height, width = values.shape
    planes = np.empty((3, height, width), dtype=np.float32)
    for row0 in range(0, height, DEBAYER_BAND_ROWS):
        row1 = min(height, row0 + DEBAYER_BAND_ROWS)
        top = max(0, row0 - 1)
        bottom = min(height, row1 + 1)
        band = np.asarray(values[top:bottom], dtype=np.float64)
        finite = np.isfinite(band)
        for channel in range(3):
            known = np.zeros(band.shape, dtype=bool)
            for index, value in enumerate(layout):
                if value == channel:
                    known[((index >> 1) - top) % 2 :: 2, index & 1 :: 2] = True
            known &= finite
            plane = np.where(known, band, 0.0)
            planes[channel, row0:row1] = _fill_plane(plane, known, row0 - top, row1 - top)
    return planes


def luminance(mosaic: NDArray[Any], pattern: str) -> NDArray[np.float32]:
    """Equal-weight mean of the debayered planes at full resolution."""

    planes = bilinear_debayer(mosaic, pattern)
    return np.mean(planes, axis=0, dtype=np.float64).astype(np.float32)


def superpixel_luminance(mosaic: NDArray[Any], pattern: str) -> NDArray[np.float32]:
    """Half-resolution luminance: the mean of each 2x2 tile (one of each colour
    plus a second green).  Coordinates scale by two relative to the mosaic."""

    values = np.asarray(mosaic, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("a Bayer mosaic must be a 2-D array")
    pattern_layout(pattern)
    height = values.shape[0] // 2 * 2
    width = values.shape[1] // 2 * 2
    tiles = values[:height, :width].reshape(height // 2, 2, width // 2, 2)
    finite = np.isfinite(tiles)
    total = np.where(finite, tiles, 0.0).sum(axis=(1, 3))
    count = finite.sum(axis=(1, 3))
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result.astype(np.float32)


def even_block_size(block_size: int) -> int:
    """Round a preview block size up to an even number so that every block
    averages the same number of red, green and blue samples."""

    return block_size if block_size % 2 == 0 else block_size + 1


__all__ = [
    "CFA_PATTERNS",
    "CHANNEL_NAMES",
    "bilinear_debayer",
    "channel_mask",
    "channel_medians",
    "channel_offsets",
    "even_block_size",
    "is_cfa_pattern",
    "luminance",
    "normalize_pattern",
    "pattern_layout",
    "pattern_scale_map",
    "shifted_pattern",
    "superpixel_luminance",
]
