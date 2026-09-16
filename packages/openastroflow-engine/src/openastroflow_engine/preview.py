"""Bounded auto-stretch previews for linear FITS masters."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .calibration import CalibrationError, FitsFrame


@dataclass(frozen=True, slots=True)
class PreviewResult:
    output_path: str
    source_shape: tuple[int, int]
    preview_shape: tuple[int, int]
    block_size: int
    black_point: float
    white_point: float
    median: float
    mad: float
    flipped_vertically: bool

    def serializable(self) -> dict[str, Any]:
        return {
            "outputPath": self.output_path,
            "sourceShape": list(self.source_shape),
            "previewShape": list(self.preview_shape),
            "blockSize": self.block_size,
            "blackPoint": self.black_point,
            "whitePoint": self.white_point,
            "median": self.median,
            "mad": self.mad,
            "flippedVertically": self.flipped_vertically,
        }


def _block_mean_preview(
    frame: FitsFrame, max_long_edge: int, max_memory_bytes: int
) -> tuple[NDArray[np.float32], int]:
    height, width = frame.shape
    factor = max(1, math.ceil(max(height, width) / max_long_edge))
    output_height = math.ceil(height / factor)
    output_width = math.ceil(width / factor)
    estimated_bytes = output_height * output_width * 4 + width * 24
    if estimated_bytes > max_memory_bytes:
        raise CalibrationError(
            "MEMORY_BUDGET_TOO_SMALL",
            "preview buffer and one source row exceed max_memory_bytes",
        )
    result = np.full((output_height, output_width), np.nan, dtype=np.float32)
    x_starts = np.arange(0, width, factor, dtype=np.int64)
    for output_y in range(output_height):
        y0 = output_y * factor
        y1 = min(height, y0 + factor)
        column_sum = np.zeros(width, dtype=np.float64)
        column_count = np.zeros(width, dtype=np.int64)
        for source_y in range(y0, y1):
            source_row = frame.read_rows(source_y, source_y + 1)[0]
            finite = np.isfinite(source_row)
            column_sum += np.where(finite, source_row, 0.0)
            column_count += finite
        block_sum = np.add.reduceat(column_sum, x_starts)
        block_count = np.add.reduceat(column_count, x_starts)
        row = np.full(output_width, np.nan, dtype=np.float64)
        np.divide(block_sum, block_count, out=row, where=block_count > 0)
        result[output_y] = row.astype(np.float32, copy=False)
    return result, factor


def _stretch(preview: NDArray[np.float32]) -> tuple[NDArray[np.uint8], tuple[float, ...]]:
    finite = preview[np.isfinite(preview)].astype(np.float64, copy=False)
    if finite.size < 16:
        raise CalibrationError(
            "PREVIEW_FINITE_PIXELS_INSUFFICIENT",
            "at least 16 finite preview pixels are required",
        )
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = max(1.4826 * mad, np.finfo(np.float32).eps)
    percentile_low, percentile_high = (
        float(value) for value in np.percentile(finite, (0.1, 99.8))
    )
    black = max(percentile_low, median - 2.8 * sigma)
    white = max(percentile_high, median + 5.0 * sigma)
    if not math.isfinite(black) or not math.isfinite(white) or white <= black:
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        black, white = minimum, maximum
    if white <= black:
        raise CalibrationError(
            "PREVIEW_DYNAMIC_RANGE_EMPTY", "linear master has no previewable range"
        )
    normalized = np.zeros(preview.shape, dtype=np.float32)
    valid = np.isfinite(preview)
    normalized[valid] = np.clip(
        (preview[valid] - black) / (white - black), 0.0, 1.0
    )
    # A gentle deterministic asinh transfer makes faint linear signal visible
    # while preserving bright-star headroom. It never touches the FITS master.
    softness = 12.0
    normalized = np.arcsinh(normalized * softness) / math.asinh(softness)
    pixels = np.rint(normalized * 255.0).astype(np.uint8)
    return pixels, (black, white, median, mad)


def render_auto_stretch_preview(
    input_fits: str | os.PathLike[str],
    output_png: str | os.PathLike[str],
    *,
    max_long_edge: int = 1600,
    flip_vertical: bool = True,
    max_memory_bytes: int = 64 * 1024 * 1024,
) -> PreviewResult:
    """Render a new PNG preview without modifying or stretching the master."""

    if isinstance(max_long_edge, bool) or max_long_edge < 16:
        raise ValueError("max_long_edge must be an integer of at least 16")
    if max_memory_bytes < 1024:
        raise ValueError("max_memory_bytes is too small")
    destination = Path(output_png)
    if destination.suffix.casefold() != ".png":
        raise CalibrationError(
            "PREVIEW_FORMAT_UNSUPPORTED", "preview destination must end in .png"
        )
    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite preview", path=str(destination)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with FitsFrame(input_fits) as source:
            preview, factor = _block_mean_preview(
                source, max_long_edge, max_memory_bytes
            )
            source_shape = source.shape
        pixels, stretch = _stretch(preview)
        if flip_vertical:
            pixels = np.flipud(pixels)
        # Lossless PNG at a fast deflate level: the pixels are identical to the
        # optimized encoding, only the byte stream (and hence the file digest)
        # differs, and the exhaustive strategy search cost seconds per preview.
        Image.fromarray(pixels, mode="L").save(
            temporary, format="PNG", optimize=False, compress_level=1
        )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise CalibrationError(
                "OUTPUT_EXISTS", "refusing to overwrite preview", path=str(destination)
            ) from error
        temporary.unlink()
        black, white, median, mad = stretch
        return PreviewResult(
            output_path=str(destination),
            source_shape=source_shape,
            preview_shape=tuple(int(value) for value in pixels.shape),
            block_size=factor,
            black_point=black,
            white_point=white,
            median=median,
            mad=mad,
            flipped_vertically=flip_vertical,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = ["PreviewResult", "render_auto_stretch_preview"]
