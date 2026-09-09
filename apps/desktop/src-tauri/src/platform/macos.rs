use super::PlatformProfile;

fn chip_name() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "Apple Silicon".to_string())
}

pub(crate) fn detect() -> PlatformProfile {
    let chip = chip_name();
    let apple_silicon = std::env::consts::ARCH == "aarch64";
    let tuned = apple_silicon && chip.to_lowercase().contains("m3 pro");
    PlatformProfile {
        platform: "macos",
        architecture: std::env::consts::ARCH,
        chip,
        cpu_backend: if apple_silicon {
            "Apple Silicon CPU"
        } else {
            "Portable CPU adapter seam"
        },
        gpu_backend: if apple_silicon {
            "Metal adapter seam"
        } else {
            "GPU adapter unavailable"
        },
        optimization_tier: if tuned {
            "M3_PRO_TUNED"
        } else if apple_silicon {
            "APPLE_SILICON"
        } else {
            "PORTABLE"
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_target_reports_a_matching_tier() {
        let profile = detect();
        if profile.architecture == "aarch64" {
            assert!(matches!(
                profile.optimization_tier,
                "M3_PRO_TUNED" | "APPLE_SILICON"
            ));
        } else {
            assert_eq!(profile.optimization_tier, "PORTABLE");
        }
    }
}
