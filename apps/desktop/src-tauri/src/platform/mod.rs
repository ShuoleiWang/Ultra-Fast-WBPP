//! Platform capability boundary for the desktop shell.
//!
//! The profiles describe adapter seams only. `engine_available` must remain false
//! until a platform adapter has passed the project's scientific validation suite.

#[derive(Debug, Clone)]
pub(crate) struct PlatformProfile {
    pub platform: &'static str,
    pub architecture: &'static str,
    pub chip: String,
    pub cpu_backend: &'static str,
    pub gpu_backend: &'static str,
    pub optimization_tier: &'static str,
}

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub(crate) use macos::detect;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub(crate) use unix::{configure_child_process, terminate_process_tree};

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub(crate) use windows::{configure_child_process, detect, terminate_process_tree};

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod portable;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub(crate) use portable::detect;
