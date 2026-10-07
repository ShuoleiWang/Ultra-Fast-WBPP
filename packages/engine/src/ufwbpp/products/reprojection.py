"""The optional ``reproject`` backend that channel alignment can resample with.

The channels of a project share one pixel grid (the filters of a target are
registered onto one reference, and a mosaic's filters onto one canvas), so
channel alignment publishes them as they are; a channel on another grid is
resampled with ``reproject`` only as a last resort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.metadata
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ReprojectProvider:
    backend_id: str
    version: str
    reproject_function: Callable[..., Any] = field(repr=False, compare=False)


def load_reproject_provider(
    importer: Callable[[str], Any] = importlib.import_module,
) -> tuple[ReprojectProvider | None, str | None]:
    """The backend, or ``None`` and the reason it is unavailable."""

    try:
        package = importer("reproject")
    except ImportError as error:
        return None, f"optional reproject dependency is unavailable: {error}"
    except Exception as error:
        return None, f"reproject discovery failed: {error}"
    function = getattr(package, "reproject_interp", None)
    if not callable(function):
        return None, "reproject.reproject_interp is missing"
    version = getattr(package, "__version__", None)
    if not isinstance(version, str) or not version:
        try:
            version = importlib.metadata.version("reproject")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return ReprojectProvider("reproject-cpu", version, function), None


__all__ = ["ReprojectProvider", "load_reproject_provider"]
