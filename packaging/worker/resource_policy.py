"""Fail-closed resource policy for the distributable Python worker.

This module has no third-party imports so it can be evaluated directly by the
PyInstaller spec as well as by the release builder and its unit tests.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Iterable, Sequence, TypeVar


REQUIRED_COLLECTIONS = (
    "openastroflow_engine",
    "lightframeqc",
    "openastroflow_registration",
    "astropy",
    "reproject",
    "shapely",
)

DISTRIBUTION_NAMES = {
    "openastroflow_engine": "openastroflow-engine",
    "lightframeqc": "light-frame-qc",
    "openastroflow_registration": "openastroflow-registration",
    "astropy": "astropy",
    "reproject": "reproject",
    "shapely": "shapely",
}
# Distributions whose metadata must ship because a module reads its own
# version at import time; none of the current dependencies do.
REQUIRED_METADATA_DISTRIBUTIONS: tuple[str, ...] = ()
CHECKED_CATALOG_RESOURCES = (
    "astap-external-v1.json",
    "astrometry-net-4107-4112-v1.json",
    "astrometry-net-4108-v1.json",
    "catalog-manifest-v1.schema.json",
    "installed-set-v1.schema.json",
)

RAW_ASTRONOMY_SUFFIXES = frozenset(
    {".fit", ".fits", ".fts", ".fz", ".xisf", ".xdrz", ".xnml"}
)
FORBIDDEN_DIRECTORY_PARTS = frozenset(
    {
        "catalog",
        "catalogs",
        "downloads",
        "raw-frames",
        "raw_frames",
        "user-data",
        "user_data",
    }
)
NON_RUNTIME_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "benchmarks",
        "docs",
        "examples",
        "test",
        "tests",
    }
)
NON_RUNTIME_MODULE_PREFIXES = (
    # The worker renders previews with Pillow and never imports Astropy's
    # Matplotlib-based visualization/UI package. Importing every optional
    # submodule during PyInstaller analysis otherwise turns an absent
    # Matplotlib into a build-time pytest Skip exception.
    "astropy.visualization",
)


class ResourceBoundaryError(ValueError):
    """A candidate bundle member crossed the public worker boundary."""


def normalize_member_path(value: str) -> PurePosixPath:
    """Return a portable relative member path or reject it.

    Bundle member names are logical archive paths. They must never depend on
    the machine that produced the release.
    """

    if not isinstance(value, str) or not value.strip():
        raise ResourceBoundaryError("bundle member path must be a non-empty string")
    if "\x00" in value:
        raise ResourceBoundaryError("bundle member path contains NUL")
    portable = value.replace("\\", "/")
    windows_path = PureWindowsPath(value)
    if (
        PurePosixPath(portable).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ResourceBoundaryError(f"absolute bundle member path is forbidden: {value!r}")
    member = PurePosixPath(portable)
    if any(part in {"", ".", ".."} for part in member.parts):
        raise ResourceBoundaryError(f"non-canonical bundle member path: {value!r}")
    return member


def resource_exclusion_reason(value: str) -> str | None:
    """Explain why a package resource must not enter a public sidecar."""

    member = normalize_member_path(value)
    lowered = tuple(part.casefold() for part in member.parts)
    if len(lowered) >= 2 and lowered[-2:] == ("resources", "catalogs"):
        return None
    if (
        len(lowered) >= 3
        and lowered[-3:-1] == ("resources", "catalogs")
        and member.name in CHECKED_CATALOG_RESOURCES
    ):
        return None
    if any(part in NON_RUNTIME_PARTS for part in lowered):
        return "non-runtime-resource"
    if any(part in FORBIDDEN_DIRECTORY_PARTS for part in lowered):
        return "catalog-or-user-data"
    if member.suffix.casefold() in RAW_ASTRONOMY_SUFFIXES:
        return "raw-astronomy-data"
    return None


def validate_resource_members(values: Iterable[str]) -> tuple[str, ...]:
    """Validate final logical member names and return normalized paths."""

    normalized: list[str] = []
    for value in values:
        member = normalize_member_path(value)
        reason = resource_exclusion_reason(member.as_posix())
        if reason is not None:
            raise ResourceBoundaryError(f"{reason}: {member.as_posix()}")
        normalized.append(member.as_posix())
    return tuple(normalized)


def module_is_runtime(module_name: str) -> bool:
    """Exclude test/documentation modules discovered by ``collect_all``."""

    if not isinstance(module_name, str) or not module_name:
        return False
    if any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in NON_RUNTIME_MODULE_PREFIXES
    ):
        return False
    return not any(
        part.casefold() in NON_RUNTIME_PARTS for part in module_name.split(".")
    )


_Pair = TypeVar("_Pair", bound=Sequence[str])


def filter_pyinstaller_entries(entries: Iterable[_Pair]) -> list[_Pair]:
    """Remove non-runtime or private data from PyInstaller TOC-style pairs.

    PyInstaller data/binary pairs contain an absolute source path and a logical
    destination directory. Only the latter is eligible for a public manifest;
    the source path is deliberately never serialized.
    """

    accepted: list[_Pair] = []
    for entry in entries:
        if len(entry) < 2:
            raise ResourceBoundaryError("PyInstaller entry must have source and destination")
        source, destination = str(entry[0]), str(entry[1])
        source_name = PurePosixPath(source.replace("\\", "/")).name
        member = PurePosixPath(destination.replace("\\", "/")) / source_name
        try:
            reason = resource_exclusion_reason(member.as_posix())
        except ResourceBoundaryError:
            # Destination paths are archive paths. An absolute one is always a
            # spec error, not something to silently normalize.
            raise
        if reason is None:
            accepted.append(entry)
    return accepted


__all__ = [
    "DISTRIBUTION_NAMES",
    "CHECKED_CATALOG_RESOURCES",
    "FORBIDDEN_DIRECTORY_PARTS",
    "NON_RUNTIME_PARTS",
    "NON_RUNTIME_MODULE_PREFIXES",
    "RAW_ASTRONOMY_SUFFIXES",
    "REQUIRED_COLLECTIONS",
    "REQUIRED_METADATA_DISTRIBUTIONS",
    "ResourceBoundaryError",
    "filter_pyinstaller_entries",
    "module_is_runtime",
    "normalize_member_path",
    "resource_exclusion_reason",
    "validate_resource_members",
]
