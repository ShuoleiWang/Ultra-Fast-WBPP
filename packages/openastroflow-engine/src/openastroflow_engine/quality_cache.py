"""Shared, disposable QC analysis cache for desktop review and production runs."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def quality_cache_directory() -> Path | None:
    """Choose a per-user cache; the analysis layer treats all I/O as optional.

    Pixel measurements and source hashing still run on every inspection. Only
    analysis of identical measured cohorts can be reused; gates and approvals
    are evaluated again. The environment override also isolates benchmarks.
    """
    override = os.environ.get("OPENASTROFLOW_QC_CACHE_DIR")
    if override is not None:
        if not override.strip() or override.strip().lower() in {"off", "none", "disabled"}:
            return None
        return Path(override).expanduser().absolute()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return root / "Ultra-Fast-WBPP" / "quality-analysis-v1"
