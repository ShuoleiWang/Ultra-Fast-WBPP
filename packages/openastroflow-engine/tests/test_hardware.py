from openastroflow_engine.backends import DeviceKind
from openastroflow_engine.controller import default_hardware_profile
from openastroflow_engine.hardware import CpuFamily, detect_hardware


def test_all_apple_m_series_receive_generic_cpu_and_metal() -> None:
    for brand in ("Apple M1", "Apple M2 Ultra", "Apple M3 Max", "Apple M4 Pro"):
        profile = detect_hardware(system="Darwin", machine="arm64", cpu_brand=brand)
        assert profile.cpu_family == CpuFamily.APPLE_M
        assert profile.devices == (DeviceKind.CPU, DeviceKind.METAL)
        assert profile.optimization_profile == "apple-silicon-generic-v1"


def test_m3_pro_has_exact_tuned_profile() -> None:
    profile = detect_hardware(
        system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro"
    )
    assert profile.m3_pro_tuned is True
    assert profile.optimization_profile == "apple-m3-pro-tuned-v1"


def test_unknown_future_apple_silicon_remains_supported_safely() -> None:
    profile = detect_hardware(system="Darwin", machine="arm64", cpu_brand="")
    assert profile.apple_silicon is True
    assert DeviceKind.METAL in profile.devices
    assert profile.optimization_profile == "apple-silicon-generic-v1"
    assert profile.warnings


def test_windows_cpu_seam_does_not_claim_an_accelerator() -> None:
    profile = detect_hardware(
        system="Windows", machine="AMD64", cpu_brand="AMD Ryzen"
    )
    assert profile.cpu_family == CpuFamily.X86_64
    assert profile.platform_id == "windows"
    assert profile.devices == (DeviceKind.CPU,)
    assert profile.cpu_backend == "windows-cpu-v1"
    assert profile.accelerator_backend is None
    assert profile.optimization_profile == "windows-x86-64-cpu-v1"
    assert default_hardware_profile(profile) == "windows-cpu"
    # An injected host never probes the real machine: memory is unknown and
    # the topology carries no physical-core claim.
    assert profile.memory_source == "unavailable"
    assert profile.memory_bytes == 0
    assert profile.physical_cores == 0
    assert profile.isa_features == ()


def test_windows_arm64_stays_outside_the_x86_64_boundary() -> None:
    profile = detect_hardware(system="Windows", machine="ARM64", cpu_brand="Snapdragon")
    assert profile.cpu_family == CpuFamily.GENERIC
    assert profile.optimization_profile == "portable-cpu-generic-v1"
    assert any("x86-64" in warning for warning in profile.warnings)


def test_injected_topology_and_memory_reach_the_profile() -> None:
    from openastroflow_engine.platform import CpuTopology, MemoryStatus

    topology = CpuTopology(
        brand="AMD Ryzen 7 5800H with Radeon Graphics",
        logical_cores=16,
        physical_cores=8,
        smt=True,
        isa_features=("sse4.2", "avx2", "fma"),
        source="GetLogicalProcessorInformationEx",
    )
    memory = MemoryStatus(17024741376, 9000000000, "GlobalMemoryStatusEx")
    profile = detect_hardware(
        system="Windows", machine="AMD64", topology=topology, memory=memory
    )
    assert profile.cpu_brand == "AMD Ryzen 7 5800H with Radeon Graphics"
    assert (profile.logical_cores, profile.physical_cores, profile.smt) == (16, 8, True)
    assert profile.isa_features == ("sse4.2", "avx2", "fma")
    assert profile.memory_bytes == 17024741376
    assert profile.memory_source == "GlobalMemoryStatusEx"
    serialized = profile.serializable()
    assert serialized["platformId"] == "windows"
    assert serialized["cpuFamily"] == "X86_64"
    assert serialized["topologySource"] == "GetLogicalProcessorInformationEx"


def test_host_detection_reports_platform_facts() -> None:
    profile = detect_hardware()
    assert profile.logical_cores >= 1
    assert profile.memory_bytes > 0
    assert profile.memory_source != "unavailable"
    assert profile.platform_id in {"darwin", "windows", "linux"}


def test_controller_profiles_match_linux_and_apple_wire_contracts() -> None:
    linux = detect_hardware(system="Linux", machine="x86_64", cpu_brand="x86_64")
    apple = detect_hardware(
        system="Darwin", machine="arm64", cpu_brand="Apple M4 Pro"
    )
    assert default_hardware_profile(linux) == "portable-cpu"
    assert default_hardware_profile(apple) == "generic-arm64-cpu"
    assert linux.cpu_family == CpuFamily.X86_64
    assert linux.optimization_profile == "linux-x86-64-cpu-v1"
