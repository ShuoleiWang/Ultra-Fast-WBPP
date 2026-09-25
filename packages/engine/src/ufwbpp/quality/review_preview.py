"""Small grayscale previews of screened Light frames for the desktop.

The quality gate's thumbnails are full previews (about a megabyte each); a
review preview is the same picture shrunk to a bounded PNG that can travel
through the desktop's event channel as a data URL.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

from PIL import Image

MAX_REVIEW_PREVIEWS = 128
MAX_PREVIEW_BYTES = 512 * 1024
MAX_TOTAL_PREVIEW_BYTES = 24 * 1024 * 1024
REVIEW_DIRECTORY = "review"


def bounded_review_preview(thumbnail_path: str | None) -> bytes | None:
    """PNG bytes of ``thumbnail_path`` at the largest edge that fits the cap."""

    if thumbnail_path is None:
        return None
    try:
        with Image.open(thumbnail_path) as source:
            image = source.convert("L")
            for edge in (640, 512, 384, 256):
                candidate = image.copy()
                candidate.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                stream = BytesIO()
                candidate.save(stream, format="PNG", optimize=False, compress_level=3)
                value = stream.getvalue()
                if len(value) <= MAX_PREVIEW_BYTES:
                    return value
    except (OSError, ValueError):
        return None
    return None


def review_preview_name(index: int, source_path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(source_path).stem).strip("-._") or "frame"
    return f"{index:04d}-{stem[:64]}.png"


__all__ = [
    "MAX_PREVIEW_BYTES",
    "MAX_REVIEW_PREVIEWS",
    "MAX_TOTAL_PREVIEW_BYTES",
    "REVIEW_DIRECTORY",
    "bounded_review_preview",
    "review_preview_name",
]
