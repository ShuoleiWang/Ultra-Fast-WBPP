"""Platform service layer: the one place that knows operating-system differences.

Every module of the engine that needs an OS fact (memory, CPU topology, the
native library's file name, ...) asks ``openastroflow_engine.platform.current()``
instead of branching on ``sys.platform``.  The services are plain objects whose
probes take their raw inputs as parameters, so every platform can be exercised
on every other platform from tests with synthetic inputs.

Nothing in this package changes a scientific value: it only reports facts that
tuning and receipts consume.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Literal, Mapping, Protocol


PlatformId = Literal["darwin", "windows", "linux"]

MEMORY_FALLBACK_BYTES = 8 * 1024**3


def platform_id_for(sys_platform: str = sys.platform) -> PlatformId:
    if sys_platform == "darwin":
        return "darwin"
    if sys_platform in {"win32", "cygwin", "msys"}:
        return "windows"
    return "linux"


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    """Physical memory as reported by the operating system.

    ``source`` names the API that produced the numbers so a receipt can show
    whether the value was measured or assumed (``fallback``).
    """

    total_bytes: int
    available_bytes: int | None
    source: str

    def serializable(self) -> dict[str, object]:
        return {
            "totalBytes": self.total_bytes,
            "availableBytes": self.available_bytes,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class CpuTopology:
    """Core layout of the running machine.

    ``performance_cores``/``efficiency_cores`` are zero when the platform does
    not distinguish core classes; ``physical_cores`` is zero when unknown.
    ``isa_features`` lists instruction-set extensions reported by the native
    library's cpuid probe (empty when that probe is unavailable).
    """

    brand: str
    logical_cores: int
    physical_cores: int = 0
    performance_cores: int = 0
    efficiency_cores: int = 0
    smt: bool = False
    isa_features: tuple[str, ...] = ()
    source: str = "unavailable"

    def serializable(self) -> dict[str, object]:
        return {
            "brand": self.brand,
            "logicalCores": self.logical_cores,
            "physicalCores": self.physical_cores,
            "performanceCores": self.performance_cores,
            "efficiencyCores": self.efficiency_cores,
            "smt": self.smt,
            "isaFeatures": list(self.isa_features),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class GpuAdapter:
    """A display/compute adapter as enumerated by the platform (report only)."""

    vendor: str
    name: str
    dedicated_memory_bytes: int | None = None
    shared_memory_bytes: int | None = None
    driver_version: str | None = None
    apis: tuple[str, ...] = ()

    def serializable(self) -> dict[str, object]:
        return {
            "vendor": self.vendor,
            "name": self.name,
            "dedicatedMemoryBytes": self.dedicated_memory_bytes,
            "sharedMemoryBytes": self.shared_memory_bytes,
            "driverVersion": self.driver_version,
            "apis": list(self.apis),
        }


@dataclass(frozen=True, slots=True)
class VolumeCapabilities:
    """What the volume holding a path can do (report and publish-mode input).

    ``hardlinks``/``case_sensitive`` are ``None`` when the platform could not
    tell; ``durable_directory_sync`` says whether ``fsync_directory`` flushes
    directory entries on this platform.
    """

    filesystem: str
    hardlinks: bool | None
    case_sensitive: bool | None
    durable_directory_sync: bool
    source: str

    def serializable(self) -> dict[str, object]:
        return {
            "filesystem": self.filesystem,
            "hardlinks": self.hardlinks,
            "caseSensitive": self.case_sensitive,
            "durableDirectorySync": self.durable_directory_sync,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class PathLimit:
    """The longest path the platform's file APIs accept, in characters.

    ``max_characters`` is ``None`` when the platform imposes no limit the
    engine could reach (POSIX ``PATH_MAX`` is 1024 or more).  Windows stops
    at ``MAX_PATH`` (259 characters plus the terminator) unless the
    ``LongPathsEnabled`` policy is on, which raises it to 32767; the run's
    staging directories nest three levels deep, so a long output directory
    can overflow the short limit (``path_budget`` projects it before any
    work starts).  ``long_paths_enabled`` is ``None`` where the policy does
    not exist and ``source`` names where the number came from.
    """

    max_characters: int | None
    long_paths_enabled: bool | None
    source: str

    def serializable(self) -> dict[str, object]:
        return {
            "maxCharacters": self.max_characters,
            "longPathsEnabled": self.long_paths_enabled,
            "source": self.source,
        }


UNLIMITED_PATH_LIMIT = PathLimit(None, None, "unlimited")


@dataclass(frozen=True, slots=True)
class ChildProcessOptions:
    """``subprocess.Popen`` keywords that isolate a child for tree termination."""

    start_new_session: bool = False
    creationflags: int = 0

    def popen_kwargs(self) -> dict[str, object]:
        options: dict[str, object] = {}
        if self.start_new_session:
            options["start_new_session"] = True
        if self.creationflags:
            options["creationflags"] = self.creationflags
        return options


class NoReplaceError(Exception):
    """A create-only publication could not be performed.

    ``code`` is ``OUTPUT_EXISTS`` (``precheck`` tells whether the destination
    existed before the attempt or appeared during it) or
    ``ATOMIC_DIRECTORY_PUBLISH_UNSUPPORTED``.  Any other operating-system
    failure propagates as ``OSError`` so callers keep their own mapping.
    """

    def __init__(self, code: str, message: str, path: str, *, precheck: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.precheck = precheck


class PlatformServices(Protocol):
    """Facts and primitives the engine may ask of the operating system."""

    platform_id: PlatformId
    # Whether the scientific pipeline has retained execution evidence on this
    # platform; the GUI keeps execution locked while this is False.
    scientific_execution_validated: bool

    # Hardware
    def memory_status(self) -> MemoryStatus: ...

    def cpu_topology(self) -> CpuTopology: ...

    def gpu_adapters(self) -> tuple[GpuAdapter, ...]: ...

    # Native library
    def native_library_filename(self) -> str: ...

    # File system
    def volume_capabilities(self, path: Path) -> VolumeCapabilities: ...

    def path_limit(self) -> PathLimit: ...

    def rename_directory_no_replace(self, source: Path, destination: Path) -> None: ...

    def publish_file_no_replace(self, temporary: Path, destination: Path) -> str: ...

    def fsync_directory(self, path: Path) -> bool: ...

    def cache_root(self) -> Path: ...

    def data_root(self, *, environment: Mapping[str, str] | None = None, home: str | os.PathLike[str] | None = None) -> Path: ...

    # Processes
    def child_process_options(self) -> ChildProcessOptions: ...

    def kill_process_tree(self, process: subprocess.Popen[Any]) -> None: ...

    def keep_awake(self) -> AbstractContextManager[None]: ...

    def well_known_executables(self, tool: str, *, environment: Mapping[str, str] | None = None) -> tuple[Path, ...]: ...


class EnvironmentView(Mapping[str, str]):
    """A read-only view of an environment mapping with Windows semantics.

    Windows resolves variable names without regard to case: ``os.environ``
    does so too, but the moment it is merged into a plain ``dict`` its keys
    come back upper-cased and a lookup of ``ProgramFiles`` returns nothing.
    The view keeps the original keys for iteration and, when
    ``case_insensitive`` is set, resolves ``get``/``[]``/``in`` by casefolded
    name; on POSIX the view is transparent.
    """

    __slots__ = ("_mapping", "_case_insensitive", "_index")

    def __init__(self, mapping: Mapping[str, str], *, case_insensitive: bool) -> None:
        self._mapping = mapping
        self._case_insensitive = case_insensitive
        self._index: dict[str, str] | None = None

    def _key(self, key: str) -> str | None:
        if not self._case_insensitive:
            return key if key in self._mapping else None
        if self._index is None:
            # Later duplicates (different spellings of one name) lose, which
            # matches how Windows itself resolves a variable.
            index: dict[str, str] = {}
            for name in self._mapping:
                index.setdefault(name.casefold(), name)
            self._index = index
        return self._index.get(key.casefold())

    def __getitem__(self, key: str) -> str:
        resolved = self._key(key)
        if resolved is None:
            raise KeyError(key)
        return self._mapping[resolved]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._key(key) is not None

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    @property
    def case_insensitive(self) -> bool:
        return self._case_insensitive


def environment_view(mapping: Mapping[str, str], *, platform_id: PlatformId) -> Mapping[str, str]:
    """``mapping`` as the platform resolves variable names."""

    if isinstance(mapping, EnvironmentView):
        return mapping
    return EnvironmentView(mapping, case_insensitive=platform_id == "windows")


def merged_environment(
    base: Mapping[str, str],
    overrides: Mapping[str, str] | None,
    *,
    platform_id: PlatformId,
) -> dict[str, str]:
    """``base`` with ``overrides`` applied the way the platform would.

    On Windows an override replaces the existing spelling of the same name
    instead of adding a second key, so a solver started with ``PATH=...``
    never sees both ``Path`` and ``PATH``.
    """

    merged = dict(base)
    if not overrides:
        return merged
    if platform_id != "windows":
        merged.update(overrides)
        return merged
    spelling = {name.casefold(): name for name in merged}
    for key, value in overrides.items():
        existing = spelling.get(key.casefold())
        if existing is not None and existing != key:
            del merged[existing]
        merged[key] = value
        spelling[key.casefold()] = key
    return merged


# Windows keeps a deleted file's name until its last handle closes, and a
# file the engine just wrote may still be open in an antivirus scanner or the
# search indexer for a moment.  Either surfaces as ERROR_ACCESS_DENIED (5),
# ERROR_SHARING_VIOLATION (32), ERROR_LOCK_VIOLATION (33) or, for a directory
# whose entries are still delete-pending, ERROR_DIR_NOT_EMPTY (145).  Those
# are the only errors worth waiting for: they clear within milliseconds once
# the other handle closes, and nothing else ever resolves by itself.
RETRIED_WINERRORS = frozenset({5, 32, 33, 145})
RETRY_INITIAL_SECONDS = 0.02
RETRY_MAXIMUM_SECONDS = 0.25
RETRY_BUDGET_SECONDS = 2.0


def _transient_sharing_error(error: OSError) -> bool:
    if isinstance(error, PermissionError):
        return True
    return getattr(error, "winerror", None) in RETRIED_WINERRORS


def retry_file_operation(
    operation: Callable[[], Any],
    *,
    platform_id: PlatformId | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Run ``operation``, waiting out Windows sharing violations.

    The retry is bounded (about ``RETRY_BUDGET_SECONDS`` with a 20 ms to
    250 ms backoff) and only exists on Windows; POSIX calls run exactly once
    because a POSIX unlink or rename never fails for a reason that waiting
    would fix.  ``platform_id``, ``sleep`` and ``clock`` are injectable so
    the Windows behaviour is exercised by the tests of every host.
    """

    if (platform_id or platform_id_for()) != "windows":
        return operation()
    deadline = clock() + RETRY_BUDGET_SECONDS
    delay = RETRY_INITIAL_SECONDS
    while True:
        try:
            return operation()
        except OSError as error:
            if not _transient_sharing_error(error) or clock() >= deadline:
                raise
        sleep(delay)
        delay = min(delay * 2.0, RETRY_MAXIMUM_SECONDS)


def remove_file(
    path: str | os.PathLike[str],
    *,
    missing_ok: bool = True,
    platform_id: PlatformId | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Delete one file; returns whether a file was removed.

    Idempotent by default (``missing_ok``): cleanup code that used to guard
    ``unlink`` with ``exists()`` raced with the deletion itself and, on
    Windows, with delete-pending names that ``exists()`` still reports.
    """

    target = Path(path)

    def unlink() -> bool:
        try:
            target.unlink()
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        return True

    return bool(retry_file_operation(unlink, platform_id=platform_id, sleep=sleep, clock=clock))


def remove_tree(
    path: str | os.PathLike[str],
    *,
    missing_ok: bool = True,
    ignore_errors: bool = False,
    platform_id: PlatformId | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Delete a directory tree; returns whether a tree was removed.

    ``shutil.rmtree`` gives up at the first file another process still holds
    open; on Windows the whole removal is retried within the bounded budget
    because a delete-pending file also keeps its parent from disappearing
    (``ERROR_DIR_NOT_EMPTY``).  ``ignore_errors`` keeps a best-effort cleanup
    from turning a completed run into a failure.
    """

    target = Path(path)

    def rmtree() -> bool:
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        return True

    try:
        return bool(retry_file_operation(rmtree, platform_id=platform_id, sleep=sleep, clock=clock))
    except OSError:
        if ignore_errors:
            return False
        raise


def rename_with_retry(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    replace: bool = False,
    platform_id: PlatformId | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """``os.rename`` (or ``os.replace`` with ``replace``) with the same bounded
    retry, for files the engine rewrites in place after another process may
    have opened the previous version.  Create-only publication keeps its own
    primitives (``publish_file_no_replace``, ``rename_directory_no_replace``):
    a retry there could turn a lost race into a silent overwrite.
    """

    move = os.replace if replace else os.rename
    retry_file_operation(
        lambda: move(source, destination), platform_id=platform_id, sleep=sleep, clock=clock
    )


def reconfigure_utf8_stdio(
    streams: tuple[Any, ...] | None = None,
) -> tuple[str, ...]:
    """Make the standard streams speak UTF-8 whatever the console code page.

    The worker protocol, receipts and progress messages are UTF-8 in both
    directions; a Windows console defaults to the OEM code page (936 on a
    Chinese system, 437 elsewhere), where a degree sign or a CJK path either
    raises ``UnicodeEncodeError`` or reaches the GUI as mojibake.
    ``backslashreplace`` keeps a stray undecodable byte from stopping a
    stream.  Returns the names of the streams that were reconfigured;
    replaced or closed streams (tests, ``pythonw``) are skipped.
    """

    if streams is None:
        streams = (sys.stdin, sys.stdout, sys.stderr)
    reconfigured: list[str] = []
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (LookupError, OSError, ValueError):
            continue
        reconfigured.append(getattr(stream, "name", repr(stream)))
    return tuple(reconfigured)


def fallback_topology(brand: str = "") -> CpuTopology:
    """Topology when no platform probe is available: logical cores only."""

    return CpuTopology(
        brand=brand,
        logical_cores=max(1, int(os.cpu_count() or 1)),
        source="os.cpu_count",
    )


def fallback_memory() -> MemoryStatus:
    """The conservative assumption used only when every probe failed."""

    return MemoryStatus(MEMORY_FALLBACK_BYTES, None, "fallback")


__all__ = [
    "ChildProcessOptions",
    "CpuTopology",
    "GpuAdapter",
    "MEMORY_FALLBACK_BYTES",
    "MemoryStatus",
    "NoReplaceError",
    "PathLimit",
    "PlatformId",
    "PlatformServices",
    "RETRIED_WINERRORS",
    "UNLIMITED_PATH_LIMIT",
    "VolumeCapabilities",
    "fallback_memory",
    "fallback_topology",
    "platform_id_for",
    "reconfigure_utf8_stdio",
    "remove_file",
    "remove_tree",
    "rename_with_retry",
    "retry_file_operation",
]
