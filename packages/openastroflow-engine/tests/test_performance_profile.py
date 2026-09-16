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
        cpu_family=CpuFamily.WINDOWS,
        devices=(DeviceKind.CPU,),
        cpu_backend="windows-cpu-v1",
        accelerator_backend=None,
        optimization_profile="windows-cpu-generic-v1",
    )
    tuning = select_execution_tuning(
        hardware, logical_cores=16, physical_memory_bytes=32 * GIB
    )
    assert tuning.profile_id == "windows-cpu-generic-v1"
    assert tuning.gpu_inflight_buffers == 0
    assert tuning.cpu_workers == 4
