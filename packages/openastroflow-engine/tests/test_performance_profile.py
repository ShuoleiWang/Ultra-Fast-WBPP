from __future__ import annotations

import pytest

from openastroflow_engine.backends import DeviceKind
from openastroflow_engine.hardware import CpuFamily, HardwareProfile, detect_hardware
from openastroflow_engine.performance_profile import (
    GIB,
    QC_WORKER_COMMIT_BYTES,
    QC_WORKER_RESERVED_BYTES,
    memory_bound_qc_workers,
    select_execution_tuning,
)
from openastroflow_engine.platform import CpuTopology, MemoryStatus


def test_m3_pro_36gb_selects_measured_profile() -> None:
    hardware = detect_hardware(
        system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro"
    )
    tuning = select_execution_tuning(
        hardware, logical_cores=12, physical_memory_bytes=36 * GIB
    )
    assert tuning.profile_id == "apple-m3-pro-tuned-v1"
    assert tuning.cpu_workers == 12
    assert tuning.qc_workers == 8
    assert tuning.kernel_threads == 12
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


def _m3_pro_row() -> dict[str, object]:
    hardware = detect_hardware(system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro")
    return select_execution_tuning(hardware, logical_cores=12, physical_memory_bytes=36 * GIB).serializable()


# The measured M3 Pro row, pinned value by value: memory-aware pool sizing
# and every other Windows change must leave the Mac tuning untouched.
M3_PRO_ROW = {
    "profileId": "apple-m3-pro-tuned-v1",
    "cpuWorkers": 12,
    "qcWorkers": 8,
    "registrationMemoryBytes": min(max(128 * 1024**2, int(36 * GIB * 0.62) // 6), 4 * GIB),
    "integrationMemoryBytes": min(max(256 * 1024**2, int(36 * GIB * 0.62) // 4), 4 * GIB),
    "integrationTileRows": 64,
    "gpuInflightBuffers": 2,
    "localNormalizationSampleStride": 2,
    "fastMath": False,
    "evidenceClass": "performance-validated-m3-pro",
    "memoryBytes": 36 * GIB,
    "memorySource": "unavailable",
    "logicalCores": 12,
    "kernelThreads": 12,
    "availableMemoryBytes": None,
}


def test_m3_pro_row_is_unchanged_by_memory_aware_pool_sizing() -> None:
    assert _m3_pro_row() == M3_PRO_ROW
    # Even a reported free memory never touches the Apple rows: the bound
    # belongs to the x86-64 and portable tables.
    hardware = detect_hardware(system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro")
    with_available = select_execution_tuning(
        hardware, logical_cores=12, physical_memory_bytes=36 * GIB, available_memory_bytes=3 * GIB
    )
    assert with_available.qc_workers == 8 and with_available.cpu_workers == 12
    assert with_available.available_memory_bytes == 3 * GIB


def test_memory_bound_qc_workers_rule() -> None:
    assert memory_bound_qc_workers(8, None) == 8
    assert memory_bound_qc_workers(8, 6 * GIB) == 3
    assert memory_bound_qc_workers(8, 12 * GIB) == 8
    assert memory_bound_qc_workers(8, 2 * GIB) == 1
    assert memory_bound_qc_workers(8, 0) == 1
    assert memory_bound_qc_workers(2, 20 * GIB) == 2
    assert QC_WORKER_COMMIT_BYTES == 1.25 * GIB and QC_WORKER_RESERVED_BYTES == 2 * GIB


@pytest.mark.parametrize(
    ("label", "topology", "memory", "expected"),
    [
        (
            "Ryzen 7 5800H laptop, 16 GiB with 6 GiB free",
            CpuTopology(brand="AMD Ryzen 7 5800H", logical_cores=16, physical_cores=8, smt=True, source="GetLogicalProcessorInformationEx"),
            MemoryStatus(16 * GIB, 6 * GIB, "GlobalMemoryStatusEx"),
            {"cpuWorkers": 4, "qcWorkers": 3, "kernelThreads": 16, "integrationTileRows": 48, "localNormalizationSampleStride": 2},
        ),
        (
            "Intel hybrid 6P+8E/20T, 32 GiB with 20 GiB free",
            CpuTopology(brand="12th Gen Intel(R) Core(TM) i7-12700H", logical_cores=20, physical_cores=14, performance_cores=6, efficiency_cores=8, smt=True, source="GetLogicalProcessorInformationEx"),
            MemoryStatus(32 * GIB, 20 * GIB, "GlobalMemoryStatusEx"),
            {"cpuWorkers": 8, "qcWorkers": 8, "kernelThreads": 20, "integrationTileRows": 48, "localNormalizationSampleStride": 2},
        ),
        (
            "4C/8T desktop, 8 GiB with 4.5 GiB free",
            CpuTopology(brand="Intel(R) Core(TM) i3-10100", logical_cores=8, physical_cores=4, smt=True, source="GetLogicalProcessorInformationEx"),
            MemoryStatus(8 * GIB, int(4.5 * GIB), "GlobalMemoryStatusEx"),
            {"cpuWorkers": 2, "qcWorkers": 2, "kernelThreads": 8, "integrationTileRows": 32, "localNormalizationSampleStride": 3},
        ),
        (
            "32C/64T workstation, 128 GiB with 100 GiB free",
            CpuTopology(brand="AMD Ryzen Threadripper PRO 5975WX", logical_cores=64, physical_cores=32, smt=True, source="GetLogicalProcessorInformationEx"),
            MemoryStatus(128 * GIB, 100 * GIB, "GlobalMemoryStatusEx"),
            {"cpuWorkers": 8, "qcWorkers": 8, "kernelThreads": 64, "integrationTileRows": 48, "localNormalizationSampleStride": 2},
        ),
    ],
)
def test_simulated_windows_machines_select_bounded_rows(label: str, topology: CpuTopology, memory: MemoryStatus, expected: dict[str, int]) -> None:
    hardware = detect_hardware(system="Windows", machine="AMD64", topology=topology, memory=memory)
    tuning = select_execution_tuning(hardware)
    row = tuning.serializable()
    assert row["profileId"] == "windows-x86-64-cpu-v1", label
    for key, value in expected.items():
        assert row[key] == value, (label, key, row[key], value)
    assert row["memoryBytes"] == memory.total_bytes and row["memorySource"] == "GlobalMemoryStatusEx"
    assert row["availableMemoryBytes"] == memory.available_bytes
    assert row["gpuInflightBuffers"] == 0 and row["fastMath"] is False
    # The same machine with its free memory unknown keeps the table value.
    unknown = detect_hardware(system="Windows", machine="AMD64", topology=topology, memory=MemoryStatus(memory.total_bytes, None, "GlobalMemoryStatusEx"))
    table_row = select_execution_tuning(unknown)
    assert table_row.qc_workers == memory_bound_qc_workers(table_row.qc_workers, None)
    assert table_row.qc_workers >= tuning.qc_workers
    assert table_row.available_memory_bytes is None


def test_low_free_memory_bounds_the_pool_on_the_laptop_row() -> None:
    windows = detect_hardware(system="Windows", machine="AMD64", cpu_brand="AMD Ryzen 7 5800H")
    laptop = select_execution_tuning(windows, logical_cores=16, physical_memory_bytes=16 * GIB, available_memory_bytes=6 * GIB)
    assert laptop.qc_workers == 3 and laptop.cpu_workers == 4
    plenty = select_execution_tuning(windows, logical_cores=16, physical_memory_bytes=16 * GIB, available_memory_bytes=None)
    assert plenty.qc_workers == 8
    starved = select_execution_tuning(windows, logical_cores=16, physical_memory_bytes=16 * GIB, available_memory_bytes=1 * GIB)
    assert starved.qc_workers == 1
    assert laptop.serializable()["availableMemoryBytes"] == 6 * GIB
    # Linux reports MemAvailable and is bounded the same way; the portable
    # row too.
    linux = detect_hardware(system="Linux", machine="x86_64", cpu_brand="Intel")
    assert select_execution_tuning(linux, logical_cores=8, physical_memory_bytes=32 * GIB, available_memory_bytes=4 * GIB).qc_workers == 1
    other = detect_hardware(system="FreeBSD", machine="riscv64", cpu_brand="riscv")
    assert select_execution_tuning(other, logical_cores=8, physical_memory_bytes=32 * GIB, available_memory_bytes=3 * GIB).qc_workers == 1
    assert select_execution_tuning(other, logical_cores=8, physical_memory_bytes=32 * GIB).qc_workers == 2
