from __future__ import annotations

from openastroflow_engine.backends import DeviceKind
from openastroflow_engine.hardware import CpuFamily, HardwareProfile, detect_hardware
from openastroflow_engine.performance_profile import GIB, select_execution_tuning


def test_m3_pro_36gb_selects_measured_profile() -> None:
    hardware = detect_hardware(
        system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro"
    )
    tuning = select_execution_tuning(
        hardware, logical_cores=12, physical_memory_bytes=36 * GIB
    )
    assert tuning.profile_id == "apple-m3-pro-tuned-v1"
    assert tuning.cpu_workers == 8
    assert tuning.qc_workers == 8
    assert tuning.kernel_threads == 8
    assert tuning.gpu_inflight_buffers == 2
    assert tuning.integration_tile_rows == 64
    assert tuning.evidence_class == "performance-validated-m3-pro"
    assert not tuning.fast_math


def test_m3_pro_low_memory_falls_back_to_generic_profile() -> None:
    hardware = detect_hardware(
        system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro"
    )
    tuning = select_execution_tuning(
        hardware, logical_cores=8, physical_memory_bytes=8 * GIB
    )
    assert tuning.profile_id == "apple-silicon-generic-v1"
    assert tuning.cpu_workers == 2
    assert tuning.qc_workers == 2
    assert tuning.gpu_inflight_buffers == 1


def test_every_other_m_series_uses_generic_capability_profile() -> None:
    for brand in ("Apple M1", "Apple M2 Max", "Apple M3", "Apple M4 Ultra", "Apple M8"):
        hardware = detect_hardware(system="Darwin", machine="arm64", cpu_brand=brand)
        tuning = select_execution_tuning(
            hardware, logical_cores=16, physical_memory_bytes=32 * GIB
        )
        assert hardware.devices == (DeviceKind.CPU, DeviceKind.METAL)
        assert tuning.profile_id == "apple-silicon-generic-v1"
        assert tuning.cpu_workers == 8
        assert tuning.qc_workers == 8
        assert not tuning.fast_math


def test_generic_apple_profile_scales_with_cores_and_memory() -> None:
    hardware = detect_hardware(system="Darwin", machine="arm64", cpu_brand="Apple M2")
    small = select_execution_tuning(hardware, logical_cores=8, physical_memory_bytes=16 * GIB)
    assert small.cpu_workers == 4
    assert small.qc_workers == 4
    assert small.registration_memory_bytes >= 1 * GIB
    tiny = select_execution_tuning(hardware, logical_cores=4, physical_memory_bytes=8 * GIB)
    assert tiny.cpu_workers == 2
    large = select_execution_tuning(hardware, logical_cores=10, physical_memory_bytes=64 * GIB)
    assert large.cpu_workers == 8
    assert large.registration_memory_bytes == 4 * GIB


def test_windows_cpu_profile_never_claims_a_gpu() -> None:
    hardware = HardwareProfile(
        operating_system="Windows",
        architecture="amd64",
        cpu_brand="Example CPU",
        cpu_family=CpuFamily.X86_64,
        devices=(DeviceKind.CPU,),
        cpu_backend="windows-cpu-v1",
        accelerator_backend=None,
        optimization_profile="windows-x86-64-cpu-v1",
        platform_id="windows",
    )
    tuning = select_execution_tuning(
        hardware, logical_cores=16, physical_memory_bytes=32 * GIB
    )
    assert tuning.profile_id == "windows-x86-64-cpu-v1"
    assert tuning.gpu_inflight_buffers == 0
    assert tuning.cpu_workers == 8
    assert tuning.evidence_class == "compatible-generic"
    assert not tuning.fast_math


def test_x86_64_table_scales_with_memory_and_cores() -> None:
    windows = detect_hardware(system="Windows", machine="AMD64", cpu_brand="AMD Ryzen 7 5800H")
    laptop = select_execution_tuning(windows, logical_cores=16, physical_memory_bytes=16 * GIB)
    assert laptop.profile_id == "windows-x86-64-cpu-v1"
    assert (laptop.cpu_workers, laptop.qc_workers, laptop.integration_tile_rows) == (4, 8, 48)
    # Workers are memory-bound; the kernels still get every logical core.
    assert laptop.kernel_threads == 16
    assert laptop.serializable()["kernelThreads"] == 16
    assert select_execution_tuning(windows, logical_cores=128, physical_memory_bytes=64 * GIB).kernel_threads == 64
    assert laptop.memory_bytes == 16 * GIB
    assert laptop.logical_cores == 16
    small = select_execution_tuning(windows, logical_cores=4, physical_memory_bytes=8 * GIB)
    assert (small.cpu_workers, small.integration_tile_rows, small.local_normalization_sample_stride) == (2, 32, 3)
    assert (small.qc_workers, small.kernel_threads) == (2, 4)
    quad = select_execution_tuning(windows, logical_cores=2, physical_memory_bytes=64 * GIB)
    assert quad.cpu_workers == 2
    linux = detect_hardware(system="Linux", machine="x86_64", cpu_brand="Intel")
    assert select_execution_tuning(linux, logical_cores=8, physical_memory_bytes=32 * GIB).profile_id == "linux-x86-64-cpu-v1"
    other = detect_hardware(system="FreeBSD", machine="riscv64", cpu_brand="riscv")
    portable = select_execution_tuning(other, logical_cores=8, physical_memory_bytes=32 * GIB)
    assert portable.profile_id == "portable-cpu-generic-v1"
    assert (portable.cpu_workers, portable.qc_workers) == (4, 2)


def test_tuning_takes_memory_and_cores_from_the_hardware_profile() -> None:
    from openastroflow_engine.platform import CpuTopology, MemoryStatus

    hardware = detect_hardware(
        system="Windows",
        machine="AMD64",
        topology=CpuTopology(brand="x", logical_cores=12, physical_cores=6, smt=True, source="test"),
        memory=MemoryStatus(20 * GIB, 10 * GIB, "GlobalMemoryStatusEx"),
    )
    tuning = select_execution_tuning(hardware)
    assert tuning.memory_bytes == 20 * GIB
    assert tuning.memory_source == "GlobalMemoryStatusEx"
    assert tuning.logical_cores == 12
    assert tuning.cpu_workers == 4
    assert tuning.kernel_threads == 12
    serialized = tuning.serializable()
    assert serialized["memorySource"] == "GlobalMemoryStatusEx"
    assert serialized["memoryBytes"] == 20 * GIB


def test_host_tuning_never_uses_the_fallback_memory_guess() -> None:
    tuning = select_execution_tuning(detect_hardware())
    assert tuning.memory_source not in {"fallback", "unavailable"}
    assert tuning.memory_bytes >= 2 * GIB
