"""macOS platform services."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterator, Mapping

from .base import (
    ChildProcessOptions,
    CpuTopology,
    GpuAdapter,
    MemoryStatus,
    VolumeCapabilities,
    fallback_topology,
)
from .posix import (
    child_process_options,
    executables_from_table,
    fsync_directory,
    kill_process_tree,
    libc_no_replace_rename,
    publish_file_no_replace,
    rename_directory_no_replace_with,
    sysconf_memory,
    volume_from_mount_table,
)


_SYSCTL_KEYS = (
    "machdep.cpu.brand_string",
    "hw.physicalcpu",
    "hw.logicalcpu",
    "hw.perflevel0.physicalcpu",
    "hw.perflevel1.physicalcpu",
)
_MOUNT_LINE = re.compile(r"^(?P<device>\S+) on (?P<point>.+?) \((?P<options>[^)]*)\)$")
WELL_KNOWN_EXECUTABLES: dict[str, tuple[str, ...]] = {
    "astap": (
        "/Applications/ASTAP.app/Contents/MacOS/astap",
        "/Applications/ASTAP.app/Contents/MacOS/ASTAP",
        "/opt/homebrew/bin/astap",
        "/usr/local/bin/astap",
    ),
    "solve-field": (
        "/opt/homebrew/bin/solve-field",
        "/usr/local/bin/solve-field",
        "/usr/bin/solve-field",
    ),
    "siril-cli": (
        "/Applications/Siril.app/Contents/MacOS/siril-cli",
        "/opt/homebrew/bin/siril-cli",
        "/usr/local/bin/siril-cli",
    ),
}


def run_sysctl(keys: tuple[str, ...] = _SYSCTL_KEYS) -> str:
    """Return ``sysctl key ...`` output; unknown keys only add stderr noise."""

    try:
        result = subprocess.run(
            ["sysctl", *keys],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


def _system_profiler_chip() -> str:
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    match = re.search(r"^\s*Chip:\s*(.+?)\s*$", result.stdout, re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_sysctl_topology(
    output: str, *, brand_fallback: Callable[[], str] = _system_profiler_chip
) -> CpuTopology:
    """Build the topology from ``sysctl`` ``key: value`` lines.

    Apple silicon reports ``hw.perflevel0`` (performance) and ``hw.perflevel1``
    (efficiency) core counts; Intel Macs lack those keys and report SMT through
    ``hw.logicalcpu > hw.physicalcpu``.
    """

    values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            values[key.strip()] = value.strip()

    def integer(key: str) -> int:
        try:
            return max(0, int(values.get(key, "0")))
        except ValueError:
            return 0

    brand = values.get("machdep.cpu.brand_string", "").strip()
    if not brand:
        brand = brand_fallback() or ""
    logical = integer("hw.logicalcpu") or max(1, int(os.cpu_count() or 1))
    physical = integer("hw.physicalcpu")
    performance = integer("hw.perflevel0.physicalcpu")
    efficiency = integer("hw.perflevel1.physicalcpu")
    if performance + efficiency != physical:
        performance = efficiency = 0
    return CpuTopology(
        brand=brand,
        logical_cores=logical,
        physical_cores=physical,
        performance_cores=performance,
        efficiency_cores=efficiency,
        smt=physical > 0 and logical > physical,
        source="sysctl" if values else "os.cpu_count",
    )


def parse_mount_output(output: str) -> list[tuple[str, str]]:
    """``mount`` lines ``dev on /point (fstype, options)`` -> ``[(point, fstype)]``."""

    mounts: list[tuple[str, str]] = []
    for line in output.splitlines():
        match = _MOUNT_LINE.match(line.strip())
        if not match:
            continue
        options = [item.strip() for item in match.group("options").split(",")]
        mounts.append((match.group("point"), options[0] if options else "unknown"))
    return mounts


def _mount_output() -> str:
    try:
        result = subprocess.run(["mount"], check=False, capture_output=True, text=True, timeout=4)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


@contextmanager
def caffeinate_keep_awake(command: tuple[str, ...] = ("caffeinate", "-i", "-w")) -> Iterator[None]:
    """Prevent idle sleep for the lifetime of the block (``caffeinate -i -w <pid>``).

    The display may still sleep.  A missing ``caffeinate`` is not an error.
    """

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [*command, str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        process = None
    try:
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


class DarwinPlatform:
    platform_id = "darwin"
    scientific_execution_validated = True

    def memory_status(self) -> MemoryStatus:
        return sysconf_memory()

    @lru_cache(maxsize=1)
    def cpu_topology(self) -> CpuTopology:
        output = run_sysctl()
        if not output.strip():
            return fallback_topology(_system_profiler_chip())
        return parse_sysctl_topology(output)

    def gpu_adapters(self) -> tuple[GpuAdapter, ...]:
        # Metal device names are reported by the executor itself; the platform
        # layer does not enumerate adapters on macOS.
        return ()

    def native_library_filename(self) -> str:
        return "libopenastroflow_native.dylib"

    def volume_capabilities(self, path: Path) -> VolumeCapabilities:
        return volume_from_mount_table(path, parse_mount_output(_mount_output()), source="mount")

    def rename_directory_no_replace(self, source: Path, destination: Path) -> None:
        rename_directory_no_replace_with(source, destination, libc_no_replace_rename("darwin"))

    def publish_file_no_replace(self, temporary: Path, destination: Path) -> str:
        return publish_file_no_replace(temporary, destination)

    def fsync_directory(self, path: Path) -> bool:
        return fsync_directory(path)

    def cache_root(self) -> Path:
        return Path.home() / "Library" / "Caches"

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
        return caffeinate_keep_awake()

    def well_known_executables(
        self, tool: str, *, environment: Mapping[str, str] | None = None
    ) -> tuple[Path, ...]:
        return executables_from_table(WELL_KNOWN_EXECUTABLES, tool)


__all__ = [
    "DarwinPlatform",
    "WELL_KNOWN_EXECUTABLES",
    "caffeinate_keep_awake",
    "parse_mount_output",
    "parse_sysctl_topology",
    "run_sysctl",
]
