mod common;

use std::collections::BTreeSet;

use openastroflow_app_core::{
    Architecture, BackendFeature, CapabilityError, HardwareProfile, HostPlatform, OperatingSystem,
    ResultRequirement, Validate,
};

#[test]
fn m3_pro_selects_tuned_profile() {
    let host = HostPlatform {
        os: OperatingSystem::Macos,
        architecture: Architecture::Aarch64,
        metal_available: true,
        apple_chip: Some("Apple M3 Pro".to_owned()),
    };
    let supported = common::capabilities().hardware_profiles;
    assert_eq!(
        HardwareProfile::select(&host, &supported),
        Some(HardwareProfile::M3ProTuned)
    );
}

#[test]
fn every_other_m_series_falls_back_to_generic_metal() {
    for chip in ["Apple M1", "Apple M2 Max", "Apple M3 Max", "Apple M4 Ultra"] {
        let host = HostPlatform {
            os: OperatingSystem::Macos,
            architecture: Architecture::Aarch64,
            metal_available: true,
            apple_chip: Some(chip.to_owned()),
        };
        assert_eq!(
            HardwareProfile::select(&host, &common::capabilities().hardware_profiles),
            Some(HardwareProfile::GenericAppleMetal),
            "chip={chip}"
        );
    }
}

#[test]
fn m3_pro_tuned_cannot_be_forced_on_another_apple_chip() {
    let host = HostPlatform {
        os: OperatingSystem::Macos,
        architecture: Architecture::Aarch64,
        metal_available: true,
        apple_chip: Some("Apple M4 Pro".to_owned()),
    };
    let error = common::capabilities()
        .check_dispatch(
            &common::recipe(ResultRequirement::Required, ResultRequirement::Disabled),
            HardwareProfile::M3ProTuned,
            &host,
        )
        .expect_err("M3 Pro tuning must be positively gated");
    assert!(matches!(
        error,
        CapabilityError::IncompatibleHost(HardwareProfile::M3ProTuned)
    ));
}

#[test]
fn apple_silicon_without_metal_falls_back_to_arm_cpu() {
    let host = HostPlatform {
        os: OperatingSystem::Macos,
        architecture: Architecture::Aarch64,
        metal_available: false,
        apple_chip: None,
    };
    assert_eq!(
        HardwareProfile::select(&host, &common::capabilities().hardware_profiles),
        Some(HardwareProfile::GenericArm64Cpu)
    );
}

#[test]
fn windows_uses_the_windows_cpu_contract() {
    let host = HostPlatform {
        os: OperatingSystem::Windows,
        architecture: Architecture::X86_64,
        metal_available: false,
        apple_chip: None,
    };
    assert_eq!(
        HardwareProfile::select(&host, &common::capabilities().hardware_profiles),
        Some(HardwareProfile::WindowsCpu)
    );
    assert_eq!(
        HardwareProfile::select(&host, &BTreeSet::from([HardwareProfile::PortableCpu])),
        None
    );
    assert!(!HardwareProfile::PortableCpu.is_compatible_with(&host));
}

#[test]
fn capability_check_fails_closed_when_drizzle_is_missing() {
    let mut capabilities = common::capabilities();
    capabilities.features.remove(&BackendFeature::Drizzle);
    let error = capabilities
        .check_recipe(
            &common::recipe(ResultRequirement::Required, ResultRequirement::Required),
            HardwareProfile::GenericArm64Cpu,
        )
        .expect_err("drizzle must be advertised");
    assert!(matches!(
        error,
        CapabilityError::MissingFeature(BackendFeature::Drizzle)
    ));
}

#[test]
fn invalid_profile_feature_pair_is_rejected() {
    let mut capabilities = common::capabilities();
    capabilities
        .features
        .remove(&BackendFeature::MetalExecution);
    assert!(capabilities.validate().is_err());
}

#[test]
fn unsupported_host_has_no_implicit_profile() {
    let host = HostPlatform {
        os: OperatingSystem::Linux,
        architecture: Architecture::X86_64,
        metal_available: false,
        apple_chip: None,
    };
    assert_eq!(
        HardwareProfile::select(&host, &BTreeSet::from([HardwareProfile::GenericArm64Cpu])),
        None
    );
}

#[test]
fn linux_x86_64_selects_the_portable_cpu_contract() {
    let host = HostPlatform {
        os: OperatingSystem::Linux,
        architecture: Architecture::X86_64,
        metal_available: false,
        apple_chip: None,
    };
    assert_eq!(
        HardwareProfile::select(&host, &BTreeSet::from([HardwareProfile::PortableCpu])),
        Some(HardwareProfile::PortableCpu)
    );
    assert!(HardwareProfile::PortableCpu.is_compatible_with(&host));
}
