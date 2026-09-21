"""Linux platform services: interface complete, execution not validated."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .base import (
    ChildProcessOptions,
    CpuTopology,
    GpuAdapter,
    MemoryStatus,
    PathLimit,
    VolumeCapabilities,
    fallback_topology,
)
from .posix import (
    child_process_options,
    executables_from_table,
    fsync_directory,
    kill_process_tree,
    libc_no_replace_rename,
    path_limit,
    no_keep_awake,
    publish_file_no_replace,
    rename_directory_no_replace_with,
    sysconf_memory,
    volume_from_mount_table,
)


WELL_KNOWN_EXECUTABLES: dict[str, tuple[str, ...]] = {
    "astap": ("/usr/bin/astap", "/usr/local/bin/astap"),
    "solve-field": ("/opt/homebrew/bin/solve-field", "/usr/local/bin/solve-field", "/usr/bin/solve-field"),
    "siril-cli": ("/usr/bin/siril-cli", "/usr/local/bin/siril-cli"),
}


def parse_proc_cpuinfo(text: str) -> CpuTopology:
    """Topology from ``/proc/cpuinfo``: brand, logical count, physical cores."""

    brand = ""
    logical = 0
    cores: set[tuple[str, str]] = set()
    physical_id = core_id = None
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if key == "processor":
            logical += 1
            physical_id = core_id = None
        elif key == "model name" and not brand:
            brand = value
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            core_id = value
        if physical_id is not None and core_id is not None:
            cores.add((physical_id, core_id))
    if logical == 0:
        return fallback_topology(brand)
    physical = len(cores)
    return CpuTopology(
        brand=brand,
        logical_cores=logical,
        physical_cores=physical,
        smt=physical > 0 and logical > physical,
        source="/proc/cpuinfo",
    )


def parse_proc_meminfo_available(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def parse_proc_mounts(text: str) -> list[tuple[str, str]]:
    """``/proc/mounts`` rows ``device point fstype options ...`` -> ``[(point, fstype)]``."""

    mounts: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            point = parts[1].replace("\\040", " ")
            mounts.append((point, parts[2]))
    return mounts


class LinuxPlatform:
    platform_id = "linux"
    scientific_execution_validated = False

    def memory_status(self) -> MemoryStatus:
        available = None
        try:
            with open("/proc/meminfo", encoding="ascii", errors="replace") as stream:
                available = parse_proc_meminfo_available(stream.read())
        except OSError:
            pass
        return sysconf_memory(available_bytes=available)

    @lru_cache(maxsize=1)
    def cpu_topology(self) -> CpuTopology:
        try:
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as stream:
                return parse_proc_cpuinfo(stream.read())
        except OSError:
            return fallback_topology(os.uname().machine if hasattr(os, "uname") else "")

    def gpu_adapters(self) -> tuple[GpuAdapter, ...]:
        return ()

    def native_library_filename(self) -> str:
        return "libopenastroflow_native.so"

    def volume_capabilities(self, path: Path) -> VolumeCapabilities:
        try:
            with open("/proc/mounts", encoding="utf-8", errors="replace") as stream:
                mounts = parse_proc_mounts(stream.read())
        except OSError:
            mounts = []
        return volume_from_mount_table(path, mounts, source="/proc/mounts")

    def path_limit(self) -> PathLimit:
        return path_limit()

    def rename_directory_no_replace(self, source: Path, destination: Path) -> None:
        rename_directory_no_replace_with(source, destination, libc_no_replace_rename("linux"))

    def publish_file_no_replace(self, temporary: Path, destination: Path) -> str:
        return publish_file_no_replace(temporary, destination)

    def fsync_directory(self, path: Path) -> bool:
        return fsync_directory(path)

    def cache_root(self) -> Path:
        return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))

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
        return (Path(home).expanduser() if home is not None else Path.home()) / ".openastroflow"

    def child_process_options(self) -> ChildProcessOptions:
        return child_process_options()

    def kill_process_tree(self, process: subprocess.Popen[Any]) -> None:
        kill_process_tree(process)

    def keep_awake(self):
        return no_keep_awake()

    def well_known_executables(
        self, tool: str, *, environment: Mapping[str, str] | None = None
    ) -> tuple[Path, ...]:
        return executables_from_table(WELL_KNOWN_EXECUTABLES, tool)


__all__ = [
    "LinuxPlatform",
    "WELL_KNOWN_EXECUTABLES",
    "parse_proc_cpuinfo",
    "parse_proc_meminfo_available",
    "parse_proc_mounts",
]
