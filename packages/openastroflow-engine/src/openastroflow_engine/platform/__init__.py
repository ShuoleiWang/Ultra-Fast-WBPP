"""Platform service layer selection (see ``base.PlatformServices``)."""

from __future__ import annotations

from functools import lru_cache

from .base import (
    ChildProcessOptions,
    CpuTopology,
    EnvironmentView,
    GpuAdapter,
    MemoryStatus,
    NoReplaceError,
    PathLimit,
    PlatformId,
    PlatformServices,
    VolumeCapabilities,
    environment_view,
    fallback_memory,
    fallback_topology,
    merged_environment,
    platform_id_for,
    reconfigure_utf8_stdio,
    remove_file,
    remove_tree,
    rename_with_retry,
)


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
    "EnvironmentView",
    "GpuAdapter",
    "MemoryStatus",
    "NoReplaceError",
    "PathLimit",
    "PlatformId",
    "PlatformServices",
    "VolumeCapabilities",
    "current",
    "environment_view",
    "fallback_memory",
    "fallback_topology",
    "merged_environment",
    "platform_id_for",
    "reconfigure_utf8_stdio",
    "remove_file",
    "remove_tree",
    "rename_with_retry",
    "services_for",
]
