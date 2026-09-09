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
    assert profile.cpu_family == CpuFamily.WINDOWS
    assert profile.devices == (DeviceKind.CPU,)
    assert profile.cpu_backend == "windows-cpu-v1"
    assert profile.accelerator_backend is None
    assert default_hardware_profile(profile) == "windows-cpu"


def test_controller_profiles_match_linux_and_apple_wire_contracts() -> None:
    linux = detect_hardware(system="Linux", machine="x86_64", cpu_brand="x86_64")
    apple = detect_hardware(
        system="Darwin", machine="arm64", cpu_brand="Apple M4 Pro"
    )
    assert default_hardware_profile(linux) == "portable-cpu"
    assert default_hardware_profile(apple) == "generic-arm64-cpu"
