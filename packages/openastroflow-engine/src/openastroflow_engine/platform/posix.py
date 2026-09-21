"""POSIX facts and primitives shared by macOS and Linux."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import os
import posixpath
from pathlib import Path
import signal
import subprocess
from typing import Any, Callable, Iterator, Mapping

from .base import (
    ChildProcessOptions,
    MemoryStatus,
    NoReplaceError,
    PathLimit,
    UNLIMITED_PATH_LIMIT,
    VolumeCapabilities,
    fallback_memory,
)


_HARDLINK_FILESYSTEMS = {
    "apfs", "hfs", "hfs+", "ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "ntfs",
    "f2fs", "jfs", "reiserfs", "tmpfs", "ufs",
}
_NO_HARDLINK_FILESYSTEMS = {
    "exfat", "msdos", "fat", "fat32", "vfat", "smbfs", "cifs", "nfs", "fuse", "iso9660",
}


def sysconf_memory(
    sysconf: Callable[[str], int] | None = None, *, available_bytes: int | None = None
) -> MemoryStatus:
    """Physical memory from ``sysconf``; ``fallback`` when it is unavailable.

    ``os.sysconf`` is looked up lazily because this module is imported on
    Windows too (every platform's parsers are tested on every host).
    """

    if sysconf is None:
        sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return fallback_memory()
    try:
        pages = int(sysconf("SC_PHYS_PAGES"))
        page_size = int(sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return fallback_memory()
    if pages <= 0 or page_size <= 0:
        return fallback_memory()
    return MemoryStatus(pages * page_size, available_bytes, "sysconf")


def hardlink_support(filesystem: str) -> bool | None:
    name = filesystem.casefold()
    if name in _HARDLINK_FILESYSTEMS:
        return True
    if name in _NO_HARDLINK_FILESYSTEMS:
        return False
    return None


def volume_from_mount_table(
    path: Path, mounts: list[tuple[str, str]], *, source: str
) -> VolumeCapabilities:
    """Longest-prefix match of ``path`` against ``(mount_point, fstype)`` rows.

    Mount tables are POSIX data, so the match uses POSIX path semantics on
    every host (the parser is exercised on Windows CI too).
    """

    resolved = posixpath.normpath(Path(path).as_posix())
    if not resolved.startswith("/"):
        resolved = "/" + resolved
    best: tuple[str, str] | None = None
    for mount_point, fstype in mounts:
        point = mount_point.rstrip("/") or "/"
        if resolved == point or resolved.startswith(point.rstrip("/") + "/") or point == "/":
            if best is None or len(point) > len(best[0]):
                best = (point, fstype)
    if best is None:
        return VolumeCapabilities("unknown", None, None, True, source)
    fstype = best[1]
    return VolumeCapabilities(fstype, hardlink_support(fstype), None, True, source)


def path_limit() -> PathLimit:
    """POSIX ``PATH_MAX`` (1024 on macOS, 4096 on Linux) is far beyond any
    path the engine composes, so no budget check applies."""

    return UNLIMITED_PATH_LIMIT


def fsync_directory(path: Path) -> bool:
    """Flush a directory's entries; ``False`` when the platform cannot.

    Opening the directory may legitimately fail (``EACCES``, ``EINVAL``,
    ``ENOTSUP``); ``fsync`` on a directory may be unsupported (``EINVAL``,
    ``ENOTSUP``).  Those are not errors.  Anything else propagates.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EINVAL, errno.ENOTSUP}:
            return False
        raise
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno in {errno.EINVAL, errno.ENOTSUP}:
            return False
        raise
    finally:
        os.close(descriptor)
    return True


def rename_directory_no_replace_with(
    source: Path,
    destination: Path,
    rename: Callable[[bytes, bytes], int] | None,
) -> None:
    """Create-only directory rename through a ``rename(src, dst) -> status``.

    ``rename`` wraps ``renamex_np(RENAME_EXCL)`` on macOS or
    ``renameat2(RENAME_NOREPLACE)`` on Linux and returns the C status; ``None``
    means the platform has no such primitive.
    """

    if os.path.lexists(destination):
        raise NoReplaceError(
            "OUTPUT_EXISTS", "refusing to replace output directory", str(destination), precheck=True
        )
    if rename is None:
        raise NoReplaceError(
            "ATOMIC_DIRECTORY_PUBLISH_UNSUPPORTED",
            "platform has no no-replace directory rename primitive",
            str(destination),
        )
    result = rename(os.fsencode(source), os.fsencode(destination))
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise NoReplaceError(
                "OUTPUT_EXISTS", "output appeared during publication", str(destination)
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))


def libc_no_replace_rename(platform_id: str) -> Callable[[bytes, bytes], int] | None:
    library = ctypes.CDLL(None, use_errno=True)
    if platform_id == "darwin" and hasattr(library, "renamex_np"):
        renamex_np = library.renamex_np
        return lambda source, destination: int(renamex_np(source, destination, 0x00000004))
    if platform_id == "linux" and hasattr(library, "renameat2"):
        renameat2 = library.renameat2
        return lambda source, destination: int(renameat2(-100, source, -100, destination, 0x00000001))
    return None


def publish_file_no_replace(temporary: Path, destination: Path) -> str:
    """Hard-link ``temporary`` to ``destination`` (create-only) and drop it.

    Returns the mode used (``hardlink``).  ``FileExistsError`` propagates so the
    caller keeps its ``OUTPUT_EXISTS`` mapping.
    """

    os.link(temporary, destination)
    temporary.unlink()
    return "hardlink"


def child_process_options() -> ChildProcessOptions:
    return ChildProcessOptions(start_new_session=True)


def kill_process_tree(process: subprocess.Popen[Any]) -> None:
    """Kill the child's whole process group (it was started in its own session)."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def kill_lingering_process_group(process: subprocess.Popen[Any]) -> None:
    """Close the private process group after the direct child exited."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@contextmanager
def no_keep_awake() -> Iterator[None]:
    yield


def executables_from_table(
    table: Mapping[str, tuple[str, ...]], tool: str
) -> tuple[Path, ...]:
    return tuple(Path(entry) for entry in table.get(tool, ()))


__all__ = [
    "child_process_options",
    "executables_from_table",
    "fsync_directory",
    "hardlink_support",
    "kill_lingering_process_group",
    "kill_process_tree",
    "libc_no_replace_rename",
    "no_keep_awake",
    "path_limit",
    "publish_file_no_replace",
    "rename_directory_no_replace_with",
    "sysconf_memory",
    "volume_from_mount_table",
]
