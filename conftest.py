"""Repository-wide pytest configuration: Windows' file rules on macOS and Linux.

Windows refuses to rename, replace, truncate or delete a file that the process
still has open or memory-mapped, to rename or remove a directory that holds
one, and to ``os.rename`` onto a path that exists.  macOS and Linux allow all
of it, so a test or an engine function that keeps a handle, an ``HDUList`` or
a ``numpy.memmap`` across such an operation passes on the development machine
and fails only in the Windows CI job, ten minutes after the push.

On macOS and Linux this file makes those operations fail in the local run,
with the exception Windows raises.  A test about the host's own file semantics
asks for the ``host_file_rules`` fixture; ``UFWBPP_TEST_WINDOWS_FILE_RULES=0``
switches the check off for a whole run.
"""

from __future__ import annotations

import builtins
import errno
import io
import mmap
import os
import shutil
import stat
import sys
import traceback
import weakref
from typing import Any, Iterator

import pytest


WINDOWS_FILE_RULES = os.name == "posix" and os.environ.get("UFWBPP_TEST_WINDOWS_FILE_RULES", "1") != "0"

_DESCRIPTOR_DIRECTORY = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"
_HERE = os.path.dirname(os.path.abspath(__file__))
# id(mapping) -> (weak reference, mapped file, where it was mapped).
_MAPPINGS: dict[int, tuple[weakref.ref[mmap.mmap], str, str]] = {}
_lifted = False


def _descriptor_path(descriptor: int) -> str | None:
    """The regular file an open descriptor refers to, else ``None``."""

    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        if sys.platform == "darwin":
            import fcntl

            raw = fcntl.fcntl(descriptor, fcntl.F_GETPATH, bytes(1024))
            return os.fsdecode(raw.split(b"\0", 1)[0]) or None
        return os.readlink(f"{_DESCRIPTOR_DIRECTORY}/{descriptor}")
    except OSError:
        return None


def _caller() -> str:
    """The innermost frame of this repository on the stack, for the message."""

    installed = f"{os.sep}site-packages{os.sep}"
    for frame in reversed(traceback.extract_stack(limit=24)[:-2]):
        name = frame.filename
        if name.startswith(_HERE) and installed not in name and name != os.path.abspath(__file__):
            return f"{os.path.relpath(name, _HERE)}:{frame.lineno}"
    return "outside the repository"


class _TrackedMmap(mmap.mmap):
    """``mmap.mmap`` that remembers which file it maps while it is alive."""

    def __new__(cls, fileno: int, length: int, *args: Any, **kwargs: Any) -> "_TrackedMmap":
        mapping = super().__new__(cls, fileno, length, *args, **kwargs)
        path = _descriptor_path(fileno) if fileno >= 0 else None
        if path is not None:
            key = id(mapping)
            _MAPPINGS[key] = (weakref.ref(mapping, lambda _reference: _MAPPINGS.pop(key, None)), path, _caller())
        return mapping


def _held_files(*, mapped_only: bool = False) -> Iterator[tuple[str, str]]:
    """``(path, how it is held)`` for every file this process has mapped or open."""

    for reference, path, where in list(_MAPPINGS.values()):
        mapping = reference()
        if mapping is not None and not mapping.closed:
            yield path, f"memory-mapped at {where}"
    if mapped_only:
        return
    try:
        names = os.listdir(_DESCRIPTOR_DIRECTORY)
    except OSError:
        return
    for name in names:
        if name.isdigit() and int(name) > 2:
            path = _descriptor_path(int(name))
            if path is not None:
                yield path, "open"


def _refuse_if_held(operation: str, path: Any, *, mapped_only: bool = False, file_only: bool = False) -> None:
    """Raise when ``path`` is held, or (unless ``file_only``) is a directory holding a held file."""

    if _lifted:
        return
    try:
        target = os.path.realpath(os.fspath(path))
    except TypeError:
        return
    inside = target + os.sep
    for held, how in _held_files(mapped_only=mapped_only):
        real = os.path.realpath(held)
        if real == target or (not file_only and real.startswith(inside)):
            raise PermissionError(
                errno.EACCES,
                f"Windows refuses to {operation} this path while {held} is still {how} "
                "(close the file, or read it into memory, first)",
                os.fspath(path),
            )


def _install_windows_file_rules(patch: pytest.MonkeyPatch) -> None:
    rename, replace, unlink, rmtree, open_file = os.rename, os.replace, os.unlink, shutil.rmtree, builtins.open

    def guarded_rename(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        if not args and not kwargs:
            if not _lifted and os.path.lexists(dst) and not _same_file(src, dst):
                raise FileExistsError(errno.EEXIST, "Windows refuses os.rename onto an existing path", os.fspath(dst))
            _refuse_if_held("rename", src)
        return rename(src, dst, *args, **kwargs)

    def guarded_replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        if not args and not kwargs:
            _refuse_if_held("rename", src)
            _refuse_if_held("replace", dst)
        return replace(src, dst, *args, **kwargs)

    def guarded_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        # The standard library's unnamed temporary file is unlinked while open
        # on POSIX and created another way on Windows: not this rule's business.
        if not args and not kwargs and sys._getframe(1).f_globals.get("__name__") != "tempfile":
            _refuse_if_held("delete", path)
        return unlink(path, *args, **kwargs)

    def guarded_rmtree(path: Any, ignore_errors: bool = False, *args: Any, **kwargs: Any) -> None:
        if not ignore_errors and "dir_fd" not in kwargs:
            _refuse_if_held("remove", path)
        return rmtree(path, ignore_errors, *args, **kwargs)

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "w" in mode and not isinstance(file, int):
            # Only the file itself: opening a directory for writing is how
            # Linux creates an unnamed temporary file in it (O_TMPFILE).
            _refuse_if_held("truncate", file, mapped_only=True, file_only=True)
        return open_file(file, mode, *args, **kwargs)

    patch.setattr(mmap, "mmap", _TrackedMmap)
    patch.setattr(os, "rename", guarded_rename)
    patch.setattr(os, "replace", guarded_replace)
    patch.setattr(os, "unlink", guarded_unlink)
    patch.setattr(os, "remove", guarded_unlink)
    patch.setattr(shutil, "rmtree", guarded_rmtree)
    patch.setattr(builtins, "open", guarded_open)
    patch.setattr(io, "open", guarded_open)

    # The engine publishes a result directory with the platform's no-replace
    # rename, which does not go through ``os.rename``.
    try:
        from ufwbpp.platform import base, darwin, linux
    except ImportError:
        return
    # Windows waits out a handle that is released a moment later (a view that
    # is still being collected, another thread closing its file); so does the
    # engine's bounded retry, which only exists for Windows.
    retry = base.retry_file_operation

    def retry_as_on_windows(operation: Any, *, platform_id: Any = None, **kwargs: Any) -> Any:
        return retry(operation, platform_id=platform_id or "windows", **kwargs)

    patch.setattr(base, "retry_file_operation", retry_as_on_windows)
    for module in (darwin, linux):
        publish = module.rename_directory_no_replace_with

        def guarded_publish(source: Any, destination: Any, primitive: Any, _publish: Any = publish) -> None:
            _refuse_if_held("rename", source)
            return _publish(source, destination, primitive)

        patch.setattr(module, "rename_directory_no_replace_with", guarded_publish)


def _same_file(first: Any, second: Any) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def windows_file_rules() -> Iterator[None]:
    if not WINDOWS_FILE_RULES:
        yield
        return
    patch = pytest.MonkeyPatch()
    _install_windows_file_rules(patch)
    try:
        yield
    finally:
        patch.undo()


@pytest.fixture
def host_file_rules() -> Iterator[None]:
    """Lift the Windows rules for one test that is about the host's own file semantics."""

    global _lifted
    previous, _lifted = _lifted, True
    try:
        yield
    finally:
        _lifted = previous


@pytest.fixture
def symlink() -> Any:
    """``symlink(link, target)``; skips the test where the account may not create symlinks.

    An ordinary Windows account needs Developer Mode for them; the hosted
    Windows runner has the privilege, a user's own machine usually does not.
    """

    def create(link: Any, target: Any) -> None:
        try:
            os.symlink(target, link)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("this Windows account may not create symbolic links")
            raise

    return create
