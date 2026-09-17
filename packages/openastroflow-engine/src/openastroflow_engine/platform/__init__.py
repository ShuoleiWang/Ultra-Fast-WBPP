"""Platform service layer selection (see ``base.PlatformServices``)."""

from __future__ import annotations

from functools import lru_cache
import sys

from .base import (
    ChildProcessOptions,
    CpuTopology,
    GpuAdapter,
    MemoryStatus,
    NoReplaceError,
    PlatformId,
    PlatformServices,
    VolumeCapabilities,
    fallback_memory,
    fallback_topology,
)


def platform_id_for(sys_platform: str = sys.platform) -> PlatformId:
    if sys_platform == "darwin":
        return "darwin"
    if sys_platform in {"win32", "cygwin", "msys"}:
        return "windows"
    return "linux"


def services_for(identifier: PlatformId) -> PlatformServices:
    """Construct the services of one platform (importable on every host)."""

    if identifier == "darwin":
        from .darwin import DarwinPlatform

        return DarwinPlatform()
    if identifier == "windows":
        from .windows import WindowsPlatform

        return WindowsPlatform()
    from .linux import LinuxPlatform

    return LinuxPlatform()


@lru_cache(maxsize=1)
def current() -> PlatformServices:
    """The services of the running platform, constructed once per process."""

    return services_for(platform_id_for())


__all__ = [
    "ChildProcessOptions",
    "CpuTopology",
    "GpuAdapter",
    "MemoryStatus",
    "NoReplaceError",
    "PlatformId",
    "PlatformServices",
    "VolumeCapabilities",
    "current",
    "fallback_memory",
    "fallback_topology",
    "platform_id_for",
    "services_for",
]
