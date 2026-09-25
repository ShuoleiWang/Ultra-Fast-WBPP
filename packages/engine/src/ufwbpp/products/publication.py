"""Shared create-only product publication, hashing and canonical serialization.

Color and mosaic workflows own their science gates; these primitives own the
filesystem commit and durability boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .. import platform as platform_services
from ..platform import NoReplaceError


class ColorProductError(RuntimeError):
    """Stable, user-actionable failure at the color-product boundary."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        self.code = code
        self.path = path
        detail = f"{path}: {message}" if path else message
        super().__init__(f"{code}: {detail}")




def _fsync_directory(path: Path) -> None:
    platform_services.current().fsync_directory(path)


def _best_effort_fsync_directory(path: Path) -> None:
    """Sync a committed directory when supported, without creating ambiguity."""

    try:
        _fsync_directory(path)
    except OSError:
        # The atomic rename has already committed.  Reporting failure here
        # would leave a complete create-only product while telling the caller
        # to retry, which can only produce OUTPUT_EXISTS.
        return


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without replacing an existing path."""

    try:
        platform_services.current().rename_directory_no_replace(source, destination)
    except NoReplaceError as error:
        if error.code == "OUTPUT_EXISTS":
            message = (
                "refusing to overwrite output directory"
                if error.precheck
                else "output appeared during publication"
            )
            raise ColorProductError("OUTPUT_EXISTS", message, path=str(destination)) from error
        raise ColorProductError(
            "ATOMIC_DIRECTORY_PUBLISH_UNSUPPORTED",
            "platform has no create-only atomic directory publication primitive",
        ) from error
    except OSError as error:
        if os.path.lexists(destination):
            raise ColorProductError(
                "OUTPUT_EXISTS", "output appeared during publication", path=str(destination)
            ) from error
        raise ColorProductError(
            "ATOMIC_PUBLICATION_FAILED", os.strerror(error.errno) if error.errno else str(error), path=str(destination)
        ) from error


DirectoryPublisher = Callable[[Path, Path], None]
