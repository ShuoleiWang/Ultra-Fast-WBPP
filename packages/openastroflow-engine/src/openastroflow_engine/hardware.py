from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import platform
import re
import subprocess
from typing import Callable

from .backends import DeviceKind


class CpuFamily(StrEnum):
    APPLE_M = "APPLE_M"
    WINDOWS = "WINDOWS"
    GENERIC = "GENERIC"


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
        }


def _mac_cpu_brand() -> str:
    for command, pattern in (
        (["sysctl", "-n", "machdep.cpu.brand_string"], None),
        (
            ["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
            re.compile(r"^\s*Chip:\s*(.+?)\s*$", re.MULTILINE),
        ),
    ):
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=4,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        output = result.stdout.strip()
        if pattern is None and output:
            return output
        if pattern is not None:
            match = pattern.search(output)
            if match:
                return match.group(1).strip()
    return ""


def detect_hardware(
    *,
    system: str | None = None,
    machine: str | None = None,
    cpu_brand: str | None = None,
    cpu_brand_probe: Callable[[], str] = _mac_cpu_brand,
) -> HardwareProfile:
    """Detect compatibility, keeping execution readiness a separate concern.

    All Apple-silicon generations intentionally share a generic CPU + Metal
    path. Only an exact M3 Pro brand match selects the tuned profile. Unknown
    future M chips therefore remain supported without inheriting unsafe tuning.
    """

    os_name = (system or platform.system() or "Unknown").strip()
    architecture = (machine or platform.machine() or "unknown").strip().lower()
    brand = cpu_brand
    if brand is None:
        if os_name.casefold() == "darwin":
            brand = cpu_brand_probe()
        brand = brand or platform.processor() or architecture
    brand = brand.strip()

    if os_name.casefold() == "darwin" and architecture in {"arm64", "aarch64"}:
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
            architecture=architecture,
            cpu_brand=brand,
            cpu_family=CpuFamily.APPLE_M,
            devices=(DeviceKind.CPU, DeviceKind.METAL),
            cpu_backend="apple-silicon-cpu-v1",
            accelerator_backend="metal-generic-v1",
            optimization_profile=optimization,
            warnings=warnings,
        )

    if os_name.casefold() == "windows":
        return HardwareProfile(
            operating_system="Windows",
            architecture=architecture,
            cpu_brand=brand,
            cpu_family=CpuFamily.WINDOWS,
            devices=(DeviceKind.CPU,),
            cpu_backend="windows-cpu-v1",
            accelerator_backend=None,
            optimization_profile="windows-cpu-generic-v1",
            warnings=(
                "Windows accelerator execution is an extension seam; this release advertises CPU compatibility only.",
            ),
        )

    return HardwareProfile(
        operating_system=os_name,
        architecture=architecture,
        cpu_brand=brand,
        cpu_family=CpuFamily.GENERIC,
        devices=(DeviceKind.CPU,),
        cpu_backend="portable-cpu-v1",
        accelerator_backend=None,
        optimization_profile="portable-cpu-generic-v1",
    )


__all__ = ["CpuFamily", "HardwareProfile", "detect_hardware"]
