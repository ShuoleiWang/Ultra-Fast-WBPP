"""Shared, disposable QC analysis cache for desktop review and production runs."""

from __future__ import annotations

from . import platform as platform_services
import os
from pathlib import Path


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
    root = platform_services.current().cache_root()
    return root / "Ultra-Fast-WBPP" / "quality-analysis-v1"
