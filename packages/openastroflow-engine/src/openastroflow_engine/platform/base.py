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
import subprocess
from typing import Any, Literal, Mapping, Protocol


PlatformId = Literal["darwin", "windows", "linux"]

MEMORY_FALLBACK_BYTES = 8 * 1024**3


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
    "PlatformId",
    "PlatformServices",
    "VolumeCapabilities",
    "fallback_memory",
    "fallback_topology",
]
