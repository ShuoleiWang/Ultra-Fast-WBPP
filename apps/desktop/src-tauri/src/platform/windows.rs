use super::PlatformProfile;
use std::os::windows::process::CommandExt;
use std::process::{Child, Command, Stdio};

const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;

/// Windows adapter placeholder. A future native engine can select DirectML,
/// CUDA, or CPU after probing drivers; the UI command contract stays unchanged.
pub(crate) fn detect() -> PlatformProfile {
    PlatformProfile {
        platform: "windows",
        architecture: std::env::consts::ARCH,
        chip: std::env::consts::ARCH.to_string(),
        cpu_backend: "Portable CPU adapter seam",
        gpu_backend: "DirectML / CUDA adapter seam",
        optimization_tier: "PORTABLE",
    }
}

pub(crate) fn configure_child_process(command: &mut Command) {
    command.creation_flags(CREATE_NEW_PROCESS_GROUP);
}

pub(crate) fn terminate_process_tree(child: &mut Child) -> Result<(), String> {
    if child
        .try_wait()
        .map_err(|error| error.to_string())?
        .is_some()
    {
        return Ok(());
    }
    let status = Command::new("taskkill")
        .args(["/PID", &child.id().to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    match status {
        Ok(value) if value.success() => Ok(()),
        _ => {
            let _ = child.kill();
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_profile_is_cpu_only_and_architecture_bound() {
        let profile = detect();
        assert_eq!(profile.platform, "windows");
        assert_eq!(profile.architecture, std::env::consts::ARCH);
        assert_eq!(profile.optimization_tier, "PORTABLE");
        assert!(profile.gpu_backend.contains("adapter seam"));
    }

    #[test]
    fn taskkill_adapter_terminates_a_configured_child_tree() {
        let mut command = Command::new("cmd.exe");
        command.args(["/D", "/S", "/C", "ping -n 30 127.0.0.1 >NUL"]);
        configure_child_process(&mut command);
        let mut child = command.spawn().expect("spawn Windows child tree");
        terminate_process_tree(&mut child).expect("terminate Windows child tree");
        let status = child.wait().expect("reap terminated Windows process");
        assert!(!status.success());
    }
}
