"""Hardware detection: what the machine is, kept separate from execution readiness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import os
import platform
import re
from typing import Callable

from .backends import DeviceKind
from .platform import (
    CpuTopology,
    GpuAdapter,
    MemoryStatus,
    PathLimit,
    PlatformId,
    fallback_topology,
    platform_id_for,
    services_for,
)


class CpuFamily(StrEnum):
    APPLE_M = "APPLE_M"
    X86_64 = "X86_64"
    GENERIC = "GENERIC"


_X86_64_MACHINES = {"x86_64", "amd64", "x64"}
_ARM64_MACHINES = {"arm64", "aarch64"}


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    operating_system: str
    architecture: str
    cpu_brand: str
    cpu_family: CpuFamily
    devices: tuple[DeviceKind, ...]
    cpu_backend: str
    accelerator_backend: str | None
    optimization_profile: str
    warnings: tuple[str, ...] = ()
    platform_id: PlatformId = "linux"
    logical_cores: int = 1
    physical_cores: int = 0
    performance_cores: int = 0
    efficiency_cores: int = 0
    smt: bool = False
    isa_features: tuple[str, ...] = ()
    memory_bytes: int = 0
    memory_source: str = "unavailable"
    topology_source: str = "unavailable"
    gpus: tuple[GpuAdapter, ...] = field(default_factory=tuple)
    # Memory the OS reported as free at detection time (``None`` where the
    # platform does not say, e.g. macOS); pool sizing bounds itself by it.
    available_memory_bytes: int | None = None
    # ``None`` for an injected host: the fact belongs to the running machine
    # and the path budget then asks the platform layer itself.
    path_limit: PathLimit | None = None

    @property
    def apple_silicon(self) -> bool:
        return self.cpu_family == CpuFamily.APPLE_M

    @property
    def m3_pro_tuned(self) -> bool:
        return self.optimization_profile == "apple-m3-pro-tuned-v1"

    def serializable(self) -> dict[str, object]:
        return {
            "operatingSystem": self.operating_system,
            "architecture": self.architecture,
            "cpuBrand": self.cpu_brand,
            "cpuFamily": self.cpu_family.value,
            "devices": [device.value for device in self.devices],
            "cpuBackend": self.cpu_backend,
            "acceleratorBackend": self.accelerator_backend,
            "optimizationProfile": self.optimization_profile,
            "appleSilicon": self.apple_silicon,
            "m3ProTuned": self.m3_pro_tuned,
            "warnings": list(self.warnings),
            "platformId": self.platform_id,
            "logicalCores": self.logical_cores,
            "physicalCores": self.physical_cores,
            "performanceCores": self.performance_cores,
            "efficiencyCores": self.efficiency_cores,
            "smt": self.smt,
            "isaFeatures": list(self.isa_features),
            "memoryBytes": self.memory_bytes,
            "memorySource": self.memory_source,
            "availableMemoryBytes": self.available_memory_bytes,
            "topologySource": self.topology_source,
            "gpus": [adapter.serializable() for adapter in self.gpus],
            "pathLimit": self.path_limit.serializable() if self.path_limit is not None else None,
        }


def _native_isa_features() -> tuple[str, ...]:
    """ISA extensions from the native library's cpuid probe, if it is loaded."""

    try:
        from .native_kernels import load_native_kernels

        kernels = load_native_kernels()
    except Exception:  # pragma: no cover - defensive: a probe never blocks detection
        return ()
    if kernels is None:
        return ()
    try:
        return kernels.cpu_features()
    except Exception:  # pragma: no cover - defensive
        return ()


def _platform_id_for_system(system: str) -> PlatformId:
    lowered = system.casefold()
    if lowered == "darwin":
        return "darwin"
    if lowered == "windows":
        return "windows"
    return "linux"


def detect_hardware(
    *,
    system: str | None = None,
    machine: str | None = None,
    cpu_brand: str | None = None,
    cpu_brand_probe: Callable[[], str] | None = None,
    topology: CpuTopology | None = None,
    memory: MemoryStatus | None = None,
    path_limit: PathLimit | None = None,
    isa_probe: Callable[[], tuple[str, ...]] = _native_isa_features,
) -> HardwareProfile:
    """Detect compatibility, keeping execution readiness a separate concern.

    Facts come from the platform service layer of the *running* host; every
    parameter can be injected so any platform's profile can be built in tests
    on any other platform (an injected ``system`` never probes the real host's
    topology or memory).  All Apple-silicon generations intentionally share a
    generic CPU + Metal path; only an exact M3 Pro brand match selects the
    tuned profile.  x86-64 hosts (Windows, Linux) share one CPU family and are
    tuned by core count and memory, never by product name.
    """

    injected_host = system is not None or machine is not None
    os_name = (system or platform.system() or "Unknown").strip()
    architecture = (machine or platform.machine() or "unknown").strip().lower()
    platform_id = _platform_id_for_system(os_name)
    services = None if injected_host else services_for(platform_id_for())

    if topology is None:
        if services is not None:
            topology = services.cpu_topology()
        else:
            topology = fallback_topology(cpu_brand or "")
    if memory is None:
        # An injected host never claims a memory size it did not measure; the
        # tuning layer then asks the running platform (or its fallback) itself.
        memory = (
            services.memory_status()
            if services is not None
            else MemoryStatus(0, None, "unavailable")
        )

    if path_limit is None and services is not None:
        path_limit = services.path_limit()

    brand = cpu_brand
    if brand is None and cpu_brand_probe is not None:
        brand = cpu_brand_probe()
    if brand is None:
        brand = topology.brand
    brand = (brand or platform.processor() or architecture).strip()

    isa_features = topology.isa_features
    if not isa_features and services is not None:
        isa_features = tuple(isa_probe())

    common = {
        "architecture": architecture,
        "cpu_brand": brand,
        "platform_id": platform_id,
        "logical_cores": max(1, int(topology.logical_cores or os.cpu_count() or 1)),
        "physical_cores": int(topology.physical_cores),
        "performance_cores": int(topology.performance_cores),
        "efficiency_cores": int(topology.efficiency_cores),
        "smt": bool(topology.smt),
        "isa_features": tuple(isa_features),
        "memory_bytes": int(memory.total_bytes),
        "memory_source": memory.source,
        "available_memory_bytes": (
            int(memory.available_bytes) if memory.available_bytes is not None else None
        ),
        "topology_source": topology.source,
        "path_limit": path_limit,
    }

    if platform_id == "darwin" and architecture in _ARM64_MACHINES:
        optimization = (
            "apple-m3-pro-tuned-v1"
            if re.search(r"\bapple\s+m3\s+pro\b", brand, re.IGNORECASE)
            else "apple-silicon-generic-v1"
        )
        warnings: tuple[str, ...] = ()
        if not brand.casefold().startswith("apple m"):
            warnings = (
                "Apple-silicon architecture detected but the exact M-series model was unavailable; using the generic safe profile.",
            )
        return HardwareProfile(
            operating_system="macOS",
            cpu_family=CpuFamily.APPLE_M,
            devices=(DeviceKind.CPU, DeviceKind.METAL),
            cpu_backend="apple-silicon-cpu-v1",
            accelerator_backend="metal-generic-v1",
            optimization_profile=optimization,
            warnings=warnings,
            **common,
        )

    if platform_id == "windows":
        x86 = architecture in _X86_64_MACHINES
        warnings = (
            (
                "Windows accelerator execution is an extension seam; this release advertises CPU compatibility only.",
            )
            if x86
            else (
                f"Windows on {architecture} is outside the supported x86-64 boundary; the portable CPU path runs without tuning evidence.",
            )
        )
        return HardwareProfile(
            operating_system="Windows",
            cpu_family=CpuFamily.X86_64 if x86 else CpuFamily.GENERIC,
            devices=(DeviceKind.CPU,),
            cpu_backend="windows-cpu-v1",
            accelerator_backend=None,
            optimization_profile="windows-x86-64-cpu-v1" if x86 else "portable-cpu-generic-v1",
            warnings=warnings,
            **common,
        )

    x86 = architecture in _X86_64_MACHINES
    return HardwareProfile(
        operating_system=os_name,
        cpu_family=CpuFamily.X86_64 if x86 else CpuFamily.GENERIC,
        devices=(DeviceKind.CPU,),
        cpu_backend="portable-cpu-v1",
        accelerator_backend=None,
        optimization_profile=(
            "linux-x86-64-cpu-v1" if x86 and platform_id == "linux" else "portable-cpu-generic-v1"
        ),
        **common,
    )


__all__ = ["CpuFamily", "HardwareProfile", "detect_hardware"]
