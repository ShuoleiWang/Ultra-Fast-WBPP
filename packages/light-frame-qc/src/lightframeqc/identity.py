"""Read-only, race-aware file identity helpers.

The hash is accepted only when the regular file has the same device, inode,
size and timestamps before, during and after the read.  Callers can therefore
bind measurements and later preparation decisions to exact input bytes rather
than to a mutable pathname.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from .content_hash import record_sha256, stat_identity
from .models import FileIdentity


class FileIdentityError(RuntimeError):
    """A file could not be identified without observing a mutation."""

    def __init__(
        self, code: str, path: str | os.PathLike[str], detail: str
    ) -> None:
        self.code = code
        self.path = str(path)
        self.detail = detail
        super().__init__(f"{code}: {self.path}: {detail}")


def _signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _identity_signature(value: FileIdentity) -> tuple[int, int, int, int]:
    return (value.device, value.inode, value.size_bytes, value.mtime_ns)


def _content_stat_signature(value: os.stat_result) -> tuple[int, int]:
    """Fields Windows reports consistently for pathname and CRT descriptors."""

    return (int(value.st_size), int(value.st_mtime_ns))


def _cached_digest(value: os.stat_result) -> str | None:
    """A digest already computed for exactly this stat identity."""

    from .content_hash import _lookup

    return _lookup(stat_identity(value))


def compute_file_identity(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = 4 * 1024 * 1024,
) -> FileIdentity:
    """Hash a stable regular file after stat-before/hash/stat-after checks."""

    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    source = Path(path).expanduser().resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise FileIdentityError("INPUT_NOT_FILE", source, "path is not a regular file")
    known = _cached_digest(before)
    if known is not None:
        return FileIdentity(
            sha256=known,
            size_bytes=int(before.st_size),
            mtime_ns=int(before.st_mtime_ns),
            device=int(before.st_dev),
            inode=int(before.st_ino),
        )

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        descriptor_before = os.fstat(stream.fileno())
        # CPython's Windows pathname stat and CRT fstat can synthesize
        # different st_dev/st_ino coordinates for the same handle. Compare
        # those coordinates only within the same API below, while still
        # requiring size/mtime agreement across the pathname/descriptor
        # boundary. POSIX keeps the stronger direct comparison.
        cross_boundary_matches = (
            _content_stat_signature(descriptor_before)
            == _content_stat_signature(before)
            if os.name == "nt"
            else _signature(descriptor_before) == _signature(before)
        )
        if not cross_boundary_matches:
            raise FileIdentityError(
                "FILE_CHANGED_DURING_HASH",
                source,
                "pathname and opened descriptor identities differ",
            )
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
        descriptor_after = os.fstat(stream.fileno())

    after = source.stat()
    if os.name == "nt":
        stable = (
            _signature(before) == _signature(after)
            and _signature(descriptor_before) == _signature(descriptor_after)
            and _content_stat_signature(after)
            == _content_stat_signature(descriptor_after)
        )
    else:
        stable = (
            _signature(before)
            == _signature(descriptor_before)
            == _signature(descriptor_after)
            == _signature(after)
        )
    if not stable:
        raise FileIdentityError(
            "FILE_CHANGED_DURING_HASH",
            source,
            "device, inode, size, mtime, or ctime changed while hashing",
        )

    value = digest.hexdigest()
    record_sha256(stat_identity(after), value)
    return FileIdentity(
        sha256=value,
        size_bytes=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev),
        inode=int(after.st_ino),
    )


def verify_file_identity_stat(
    path: str | os.PathLike[str], identity: FileIdentity
) -> None:
    """Fail when a pathname no longer has an identity's stat coordinates."""

    source = Path(path).expanduser().resolve(strict=True)
    current = source.stat()
    current_signature = (
        int(current.st_dev),
        int(current.st_ino),
        int(current.st_size),
        int(current.st_mtime_ns),
    )
    if current_signature != _identity_signature(identity):
        raise FileIdentityError(
            "FILE_CHANGED_SINCE_IDENTITY",
            source,
            "device, inode, size, or mtime no longer matches the bound identity",
        )


__all__ = [
    "FileIdentityError",
    "compute_file_identity",
    "verify_file_identity_stat",
]
