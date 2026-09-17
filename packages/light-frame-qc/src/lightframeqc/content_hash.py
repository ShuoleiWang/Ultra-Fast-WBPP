"""One SHA-256 per file per process.

Every stage of the pipeline binds its receipts to the bytes of the source
frames, and several stages used to hash the same 100 MB files again. The
digest is a function of the bytes, so it is cached here under the file's
stat identity (device, inode, size, mtime and ctime in nanoseconds): a file
that is rewritten in place changes at least its ctime and is hashed again,
and a hit costs one ``stat``. Callers that need the race-aware
before/during/after checks keep them (``lightframeqc.identity``); they only
skip the read when the identity is already known.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import threading

__all__ = ["file_sha256", "cache_size", "clear_cache", "record_sha256", "stat_identity"]

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024
MAXIMUM_ENTRIES = 8192

_LOCK = threading.Lock()
_DIGESTS: OrderedDict[tuple[int, int, int, int, int], str] = OrderedDict()


def stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _lookup(identity: tuple[int, int, int, int, int]) -> str | None:
    with _LOCK:
        digest = _DIGESTS.get(identity)
        if digest is not None:
            _DIGESTS.move_to_end(identity)
        return digest


def record_sha256(identity: tuple[int, int, int, int, int], digest: str) -> None:
    """Remember a digest computed elsewhere for a file with this identity."""

    with _LOCK:
        _DIGESTS[identity] = digest
        _DIGESTS.move_to_end(identity)
        while len(_DIGESTS) > MAXIMUM_ENTRIES:
            _DIGESTS.popitem(last=False)


def file_sha256(path: str | os.PathLike[str], *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Hex SHA-256 of a regular file, read once per stat identity.

    The file is hashed from an open descriptor and the digest is kept only
    when the descriptor's identity matches the pathname's identity before and
    after the read, so a file replaced or rewritten while it is being hashed
    is neither cached nor silently mis-attributed.
    """

    source = Path(path)
    before = source.stat()
    identity = stat_identity(before)
    cached = _lookup(identity)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        opened = stat_identity(os.fstat(stream.fileno()))
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
        closing = stat_identity(os.fstat(stream.fileno()))
    after = stat_identity(source.stat())
    value = digest.hexdigest()
    # Windows reports device/inode differently for pathname and descriptor
    # stats; size and timestamps are comparable across both.
    comparable = (lambda item: item[2:]) if os.name == "nt" else (lambda item: item)
    if comparable(opened) == comparable(closing) == comparable(identity) == comparable(after):
        record_sha256(after, value)
    return value


def cache_size() -> int:
    with _LOCK:
        return len(_DIGESTS)


def clear_cache() -> None:
    with _LOCK:
        _DIGESTS.clear()
