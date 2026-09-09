use super::PlatformProfile;

pub(crate) fn detect() -> PlatformProfile {
    PlatformProfile {
        platform: "linux",
        architecture: std::env::consts::ARCH,
        chip: std::env::consts::ARCH.to_string(),
        cpu_backend: "Portable CPU adapter seam",
        gpu_backend: "Portable GPU adapter seam",
        optimization_tier: "PORTABLE",
    }
}
