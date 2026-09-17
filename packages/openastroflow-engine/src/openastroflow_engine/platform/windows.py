"""Windows platform services (x86-64, Windows 10/11).

Facts come from Win32 APIs through ``ctypes`` (no ``psutil``, no WMI):

* ``GlobalMemoryStatusEx`` for physical/available memory;
* ``GetLogicalProcessorInformationEx(RelationProcessorCore)`` for physical
  cores, SMT and the hybrid ``EfficiencyClass`` (P/E cores);
* the registry ``ProcessorNameString`` for the CPU brand.

The raw-buffer parsers are pure functions so they are tested on every host.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping

from .base import (
    ChildProcessOptions,
    CpuTopology,
    GpuAdapter,
    MemoryStatus,
    NoReplaceError,
    VolumeCapabilities,
    fallback_memory,
    fallback_topology,
)


# GetVolumeInformationW file-system flags.
_FILE_CASE_SENSITIVE_SEARCH = 0x00000001
_FILE_SUPPORTS_HARD_LINKS = 0x00400000
# SetThreadExecutionState flags.
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_CONTINUOUS = 0x80000000
# CreateFileW arguments for a directory handle.
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_ALL = 0x00000007
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

WELL_KNOWN_EXECUTABLES: dict[str, tuple[str, ...]] = {
    # Relative to each of ProgramFiles, ProgramFiles(x86) and LOCALAPPDATA.
    "astap": (r"ASTAP\astap.exe", r"ASTAP\astap_cli.exe"),
    "solve-field": (r"Astrometry.net\bin\solve-field.exe", r"astrometry\bin\solve-field.exe"),
    "siril-cli": (r"Siril\bin\siril-cli.exe", r"Siril\siril-cli.exe", r"Programs\Siril\bin\siril-cli.exe"),
}
_EXECUTABLE_ROOT_KEYS = ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")


_RELATION_PROCESSOR_CORE = 0
_LTP_PC_SMT = 0x1
_ERROR_INSUFFICIENT_BUFFER = 122


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def global_memory_status() -> MemoryStatus:
    """``GlobalMemoryStatusEx``; ``fallback`` when the call is unavailable."""

    if sys.platform != "win32":
        return fallback_memory()
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return fallback_memory()
    except (AttributeError, OSError):
        return fallback_memory()
    total = int(status.ullTotalPhys)
    if total <= 0:
        return fallback_memory()
    return MemoryStatus(total, int(status.ullAvailPhys), "GlobalMemoryStatusEx")


def parse_processor_core_records(buffer: bytes) -> tuple[int, int, bool, dict[int, int]]:
    """Parse ``SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`` core records.

    Returns ``(physical_cores, logical_cores, smt, cores_per_efficiency_class)``.
    Layout (x86-64): ``Relationship`` (u32), ``Size`` (u32), then
    ``PROCESSOR_RELATIONSHIP``: ``Flags`` (u8), ``EfficiencyClass`` (u8),
    ``Reserved[20]``, ``GroupCount`` (u16) and ``GroupCount`` ``GROUP_AFFINITY``
    entries of a 64-bit ``Mask`` plus 8 bytes of group/reserved fields.
    """

    physical = logical = 0
    smt = False
    classes: dict[int, int] = {}
    offset = 0
    view = memoryview(buffer)
    while offset + 8 <= len(view):
        relationship, size = struct.unpack_from("<II", view, offset)
        if size < 8 or offset + size > len(view):
            break
        if relationship == _RELATION_PROCESSOR_CORE and size >= 32:
            flags, efficiency_class = struct.unpack_from("<BB", view, offset + 8)
            (group_count,) = struct.unpack_from("<H", view, offset + 30)
            mask_offset = offset + 32
            core_logical = 0
            for _ in range(group_count):
                if mask_offset + 16 > offset + size:
                    break
                (mask,) = struct.unpack_from("<Q", view, mask_offset)
                core_logical += bin(mask).count("1")
                mask_offset += 16
            physical += 1
            logical += core_logical
            if flags & _LTP_PC_SMT or core_logical > 1:
                smt = True
            classes[efficiency_class] = classes.get(efficiency_class, 0) + 1
        offset += size
    return physical, logical, smt, classes


def topology_from_core_records(buffer: bytes, brand: str) -> CpuTopology:
    physical, logical, smt, classes = parse_processor_core_records(buffer)
    if physical == 0:
        return fallback_topology(brand)
    performance = efficiency = 0
    if len(classes) > 1:
        # Higher EfficiencyClass values are the faster cores on hybrid CPUs.
        top = max(classes)
        performance = classes[top]
        efficiency = sum(count for value, count in classes.items() if value != top)
    return CpuTopology(
        brand=brand,
        logical_cores=logical or max(1, int(os.cpu_count() or 1)),
        physical_cores=physical,
        performance_cores=performance,
        efficiency_cores=efficiency,
        smt=smt,
        source="GetLogicalProcessorInformationEx",
    )


def logical_processor_information() -> bytes:
    """Raw ``GetLogicalProcessorInformationEx(RelationProcessorCore)`` buffer."""

    if sys.platform != "win32":
        return b""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        length = ctypes.c_uint32(0)
        kernel32.GetLogicalProcessorInformationEx(
            _RELATION_PROCESSOR_CORE, None, ctypes.byref(length)
        )
        if ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER or length.value == 0:
            return b""
        raw = ctypes.create_string_buffer(length.value)
        if not kernel32.GetLogicalProcessorInformationEx(
            _RELATION_PROCESSOR_CORE, raw, ctypes.byref(length)
        ):
            return b""
        return raw.raw[: length.value]
    except (AttributeError, OSError):
        return b""


def registry_processor_brand() -> str:
    if sys.platform != "win32":
        return ""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
    except (ImportError, OSError):
        return ""
    return str(value).strip()


def volume_capabilities_from(
    filesystem: str, flags: int, *, source: str = "GetVolumeInformationW"
) -> VolumeCapabilities:
    """Interpret ``GetVolumeInformationW`` results (pure, tested everywhere)."""

    return VolumeCapabilities(
        filesystem=filesystem or "unknown",
        hardlinks=bool(flags & _FILE_SUPPORTS_HARD_LINKS),
        case_sensitive=bool(flags & _FILE_CASE_SENSITIVE_SEARCH),
        durable_directory_sync=True,
        source=source,
    )


def volume_information(path: Path) -> VolumeCapabilities:
    """``GetVolumePathNameW`` + ``GetVolumeInformationW`` for the volume of ``path``."""

    if sys.platform != "win32":
        return VolumeCapabilities("unknown", None, None, True, "unavailable")
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        root = ctypes.create_unicode_buffer(1024)
        target = str(Path(path).resolve()) if Path(path).exists() else str(Path(path).absolute())
        if not kernel32.GetVolumePathNameW(target, root, 1024):
            return VolumeCapabilities("unknown", None, None, True, "GetVolumePathNameW-failed")
        name = ctypes.create_unicode_buffer(256)
        serial = ctypes.c_uint32(0)
        maximum_component = ctypes.c_uint32(0)
        flags = ctypes.c_uint32(0)
        filesystem = ctypes.create_unicode_buffer(256)
        if not kernel32.GetVolumeInformationW(
            root.value,
            name,
            256,
            ctypes.byref(serial),
            ctypes.byref(maximum_component),
            ctypes.byref(flags),
            filesystem,
            256,
        ):
            return VolumeCapabilities("unknown", None, None, True, "GetVolumeInformationW-failed")
    except (AttributeError, OSError):
        return VolumeCapabilities("unknown", None, None, True, "unavailable")
    return volume_capabilities_from(filesystem.value, int(flags.value))


def rename_directory_no_replace_with(
    source: Path, destination: Path, rename: Callable[[Path, Path], None] = os.rename
) -> None:
    """Create-only directory rename: Windows ``MoveFileExW`` without
    ``MOVEFILE_REPLACE_EXISTING`` fails with ``FileExistsError`` for an
    existing target, which is the no-replace guarantee."""

    if os.path.lexists(destination):
        raise NoReplaceError(
            "OUTPUT_EXISTS", "refusing to replace output directory", str(destination), precheck=True
        )
    try:
        rename(source, destination)
    except FileExistsError as error:
        raise NoReplaceError(
            "OUTPUT_EXISTS", "output appeared during publication", str(destination)
        ) from error


def publish_file_no_replace_with(
    temporary: Path,
    destination: Path,
    *,
    link: Callable[[Path, Path], None] = os.link,
    rename: Callable[[Path, Path], None] = os.rename,
) -> str:
    """Create-only file publication.

    NTFS supports hard links, so the POSIX link-then-unlink form is used first;
    exFAT/FAT32/SMB volumes reject ``CreateHardLink`` and fall back to
    ``MoveFileExW`` without replace, which is equally create-only.  Returns the
    mode used; ``FileExistsError`` propagates for the caller's mapping.
    """

    try:
        link(temporary, destination)
    except FileExistsError:
        raise
    except OSError:
        rename(temporary, destination)
        return "rename"
    temporary.unlink()
    return "hardlink"


def flush_directory(path: Path) -> bool:
    """``FlushFileBuffers`` on a directory handle; failures are not errors."""

    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.CreateFileW.restype = ctypes.c_void_p
        handle = kernel32.CreateFileW(
            str(path),
            _GENERIC_WRITE,
            _FILE_SHARE_ALL,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS,
            None,
        )
        if handle is None or handle == _INVALID_HANDLE_VALUE:
            return False
        try:
            return bool(kernel32.FlushFileBuffers(ctypes.c_void_p(handle)))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    except (AttributeError, OSError):
        return False


@contextmanager
def execution_state_keep_awake() -> Iterator[None]:
    """``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` for the block.

    Keeps the system (not the display) awake while processing; restored on exit.
    """

    kernel32 = None
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
            kernel32.SetThreadExecutionState(ctypes.c_uint32(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))
        except (AttributeError, OSError):
            kernel32 = None
    try:
        yield
    finally:
        if kernel32 is not None:
            try:
                kernel32.SetThreadExecutionState(ctypes.c_uint32(_ES_CONTINUOUS))
            except (AttributeError, OSError):
                pass


def child_process_options() -> ChildProcessOptions:
    return ChildProcessOptions(
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )


def kill_process_tree(process: subprocess.Popen[Any]) -> None:
    """``taskkill /T /F`` on the child (a Job Object is a later phase), then kill."""

    if process.poll() is not None:
        return
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    taskkill = Path(system_root) / "System32" / "taskkill.exe"
    if taskkill.is_file():
        try:
            subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    if process.poll() is None:
        process.kill()


def well_known_executables(
    tool: str, environment: Mapping[str, str], table: Mapping[str, tuple[str, ...]] = WELL_KNOWN_EXECUTABLES
) -> tuple[Path, ...]:
    """Install locations under ProgramFiles, ProgramFiles(x86) and LOCALAPPDATA."""

    candidates: list[Path] = []
    for root_key in _EXECUTABLE_ROOT_KEYS:
        root = environment.get(root_key)
        if root:
            candidates.extend(Path(root) / relative for relative in table.get(tool, ()))
    return tuple(candidates)


class WindowsPlatform:
    platform_id = "windows"
    # Windows execution evidence is being established (see docs/windows.md).
    scientific_execution_validated = False

    def memory_status(self) -> MemoryStatus:
        return global_memory_status()

    @lru_cache(maxsize=1)
    def cpu_topology(self) -> CpuTopology:
        brand = registry_processor_brand()
        buffer = logical_processor_information()
        if not buffer:
            return fallback_topology(brand)
        return topology_from_core_records(buffer, brand)

    def gpu_adapters(self) -> tuple[GpuAdapter, ...]:
        # DXGI enumeration is a later phase; never load a compute API here.
        return ()

    def native_library_filename(self) -> str:
        return "openastroflow_native.dll"

    def volume_capabilities(self, path: Path) -> VolumeCapabilities:
        return volume_information(path)

    def rename_directory_no_replace(self, source: Path, destination: Path) -> None:
        rename_directory_no_replace_with(source, destination)

    def publish_file_no_replace(self, temporary: Path, destination: Path) -> str:
        return publish_file_no_replace_with(temporary, destination)

    def fsync_directory(self, path: Path) -> bool:
        return flush_directory(path)

    def cache_root(self) -> Path:
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))

    def data_root(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        home: str | os.PathLike[str] | None = None,
    ) -> Path:
        env = os.environ if environment is None else environment
        override = env.get("OPENASTROFLOW_DATA_DIR")
        if override:
            return Path(override).expanduser()
        local_app_data = env.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "OpenAstroFlow"
        return (Path(home).expanduser() if home is not None else Path.home()) / ".openastroflow"

    def child_process_options(self) -> ChildProcessOptions:
        return child_process_options()

    def kill_process_tree(self, process: subprocess.Popen[Any]) -> None:
        kill_process_tree(process)

    def keep_awake(self):
        return execution_state_keep_awake()

    def well_known_executables(
        self, tool: str, *, environment: Mapping[str, str] | None = None
    ) -> tuple[Path, ...]:
        return well_known_executables(tool, os.environ if environment is None else environment)


__all__ = [
    "WELL_KNOWN_EXECUTABLES",
    "WindowsPlatform",
    "execution_state_keep_awake",
    "flush_directory",
    "global_memory_status",
    "parse_processor_core_records",
    "publish_file_no_replace_with",
    "registry_processor_brand",
    "rename_directory_no_replace_with",
    "topology_from_core_records",
    "volume_capabilities_from",
    "volume_information",
    "well_known_executables",
]
