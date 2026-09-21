//! Windows adapter: every sidecar runs inside its own Job Object so that
//! cancellation and application exit take the worker pools and `astap_cli.exe`
//! down with the worker, and the hardware profile names the real processor.

use std::ffi::c_void;
use std::io;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};

use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Threading::{
    OpenThread, ResumeThread, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CREATE_SUSPENDED,
    THREAD_SUSPEND_RESUME,
};

use super::{ManagedChild, PlatformProfile};

/// Exit code of processes the job terminates; the same value `taskkill /F`
/// reports, so a cancelled tree looks alike in logs whichever path stopped it.
const TERMINATED_EXIT_CODE: u32 = 1;

/// The processor brand string (for example `AMD Ryzen 7 5800H with Radeon
/// Graphics`), from cpuid leaves 0x80000002-0x80000004, or the OS's
/// `PROCESSOR_IDENTIFIER` when the processor does not expose them.
fn chip_name() -> String {
    cpuid_brand_string()
        .or_else(|| {
            std::env::var("PROCESSOR_IDENTIFIER")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_string())
}

// The cpuid intrinsics are `unsafe fn` on the pinned toolchain and safe on
// later ones; the blocks stay so both compile under `-D warnings`.
#[allow(unused_unsafe)]
#[cfg(target_arch = "x86_64")]
fn cpuid_brand_string() -> Option<String> {
    use std::arch::x86_64::__cpuid_count;

    // Leaf 0x80000000 reports the highest extended leaf; the brand leaves
    // exist only when it reaches 0x80000004.
    // SAFETY: every x86-64 processor implements cpuid; the intrinsic reads
    // registers only and takes no pointers.
    let highest = unsafe { __cpuid_count(0x8000_0000, 0) }.eax;
    if highest < 0x8000_0004 {
        return None;
    }
    let mut bytes = Vec::with_capacity(48);
    for leaf in 0x8000_0002..=0x8000_0004 {
        // SAFETY: as above; the leaf was reported as supported.
        let registers = unsafe { __cpuid_count(leaf, 0) };
        for value in [registers.eax, registers.ebx, registers.ecx, registers.edx] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    // Vendors pad the 48 bytes with spaces; collapse them.
    let name = String::from_utf8_lossy(&bytes[..end])
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!name.is_empty()).then_some(name)
}

#[cfg(not(target_arch = "x86_64"))]
fn cpuid_brand_string() -> Option<String> {
    None
}

/// Windows x86-64 is the validated CPU path of this release; any other
/// Windows architecture only reports itself so the shell can refuse it.
pub(crate) fn detect() -> PlatformProfile {
    let x86_64 = std::env::consts::ARCH == "x86_64";
    PlatformProfile {
        platform: "windows",
        architecture: std::env::consts::ARCH,
        chip: chip_name(),
        cpu_backend: "Native CPU execution",
        gpu_backend: "GPU acceleration not used",
        optimization_tier: if x86_64 { "WINDOWS_X64" } else { "PORTABLE" },
    }
}

/// A GUI process starting a console-subsystem worker would otherwise flash a
/// console window for every sidecar call; the new process group keeps console
/// control events apart from the application's.
pub(crate) fn configure_child_process(command: &mut Command) {
    command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
}

fn win32_error() -> io::Error {
    io::Error::last_os_error()
}

/// Owned handle of a Job Object that terminates its processes when the last
/// handle to it closes.
#[derive(Debug)]
pub(crate) struct JobObject(OwnedHandle);

impl JobObject {
    fn kill_on_close() -> io::Result<Self> {
        // SAFETY: no security attributes and no name are passed; a null return
        // is checked before the handle is adopted, and `OwnedHandle` closes it
        // exactly once.
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() {
            return Err(win32_error());
        }
        // SAFETY: the handle is open and owned by nothing else.
        let job = Self(unsafe { OwnedHandle::from_raw_handle(handle) });
        // SAFETY: the record is plain data, so all-zero is a valid value.
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: the pointer and length describe `limits`, which outlives the call.
        let ok = unsafe {
            SetInformationJobObject(
                job.handle(),
                JobObjectExtendedLimitInformation,
                std::ptr::from_ref(&limits).cast::<c_void>(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            return Err(win32_error());
        }
        Ok(job)
    }

    pub(crate) fn handle(&self) -> HANDLE {
        self.0.as_raw_handle()
    }

    fn assign(&self, process: HANDLE) -> io::Result<()> {
        // SAFETY: both handles are open for the duration of the call.
        if unsafe { AssignProcessToJobObject(self.handle(), process) } == 0 {
            return Err(win32_error());
        }
        Ok(())
    }

    /// Terminates every process still in the job.
    pub(crate) fn terminate(&self) -> io::Result<()> {
        // SAFETY: the job handle is open; the exit code is a plain value.
        if unsafe { TerminateJobObject(self.handle(), TERMINATED_EXIT_CODE) } == 0 {
            return Err(win32_error());
        }
        Ok(())
    }
}

/// `CreateProcess` hands `std` the primary thread's handle, which `std` closes
/// at once, so the suspended thread is found again through a thread snapshot.
fn resume_threads(process_id: u32) -> io::Result<()> {
    // SAFETY: a thread snapshot takes no process id; the invalid-handle return
    // is checked before the handle is adopted.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot == INVALID_HANDLE_VALUE || snapshot.is_null() {
        return Err(win32_error());
    }
    // SAFETY: the snapshot handle is open and owned by nothing else.
    let snapshot = unsafe { OwnedHandle::from_raw_handle(snapshot) };
    // SAFETY: the record is plain data, so all-zero is a valid value.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `entry` is a valid, correctly sized record for the call.
    let mut more = unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) } != 0;
    let mut resumed = 0_u32;
    while more {
        if entry.th32OwnerProcessID == process_id {
            // SAFETY: the thread id came from the snapshot; a null return is checked.
            let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) };
            if thread.is_null() {
                return Err(win32_error());
            }
            // SAFETY: the thread handle is open and owned by nothing else.
            let thread = unsafe { OwnedHandle::from_raw_handle(thread) };
            // SAFETY: the handle is open; `ResumeThread` returns the previous
            // suspend count, or `u32::MAX` on failure.
            if unsafe { ResumeThread(thread.as_raw_handle()) } == u32::MAX {
                return Err(win32_error());
            }
            resumed += 1;
        }
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        // SAFETY: as for `Thread32First`.
        more = unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) } != 0;
    }
    if resumed == 0 {
        return Err(io::Error::other(
            "the suspended sidecar process has no thread to resume",
        ));
    }
    Ok(())
}

/// Spawns the process suspended, places it in a fresh kill-on-close job and
/// only then lets it run: no worker pool or solver can start before the tree
/// is accounted for.  A host that refuses the assignment (a job hierarchy that
/// forbids nesting) still gets a running child, controlled through `taskkill`.
pub(crate) fn spawn_in_job(command: &mut Command) -> io::Result<ManagedChild> {
    let job = JobObject::kill_on_close()?;
    command.creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    let mut child = command.spawn()?;
    let process: HANDLE = child.as_raw_handle();
    let job = job.assign(process).ok().map(|()| job);
    if let Err(error) = resume_threads(child.id()) {
        // A suspended process never exits on its own.
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    Ok(ManagedChild { child, job })
}

fn taskkill_tree(process_id: u32) -> bool {
    Command::new("taskkill")
        .args(["/PID", &process_id.to_string(), "/T", "/F"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub(crate) fn terminate_process_tree(child: &mut ManagedChild) -> Result<(), String> {
    if child
        .try_wait()
        .map_err(|error| error.to_string())?
        .is_some()
    {
        return Ok(());
    }
    if child
        .job
        .as_ref()
        .is_some_and(|job| job.terminate().is_ok())
    {
        return Ok(());
    }
    // Without a job (or when the kernel refused to terminate it) walk the
    // parent chain from user space, then stop the direct child at least.
    if !taskkill_tree(child.id()) {
        let _ = child.kill();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::JobObjects::{
        IsProcessInJob, JobObjectBasicAccountingInformation, QueryInformationJobObject,
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    use super::*;

    /// Live processes (those still owning a thread) whose parent is `parent`.
    fn live_children_of(parent: u32) -> Vec<u32> {
        // SAFETY: a process snapshot takes no process id; the handle is adopted
        // only after the invalid-handle check.
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        assert!(
            snapshot != INVALID_HANDLE_VALUE && !snapshot.is_null(),
            "process snapshot"
        );
        // SAFETY: the snapshot handle is open and owned by nothing else.
        let snapshot = unsafe { OwnedHandle::from_raw_handle(snapshot) };
        let mut entry = PROCESSENTRY32W {
            dwSize: size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let mut children = Vec::new();
        // SAFETY: `entry` is a valid, correctly sized record for the call.
        let mut more = unsafe { Process32FirstW(snapshot.as_raw_handle(), &mut entry) } != 0;
        while more {
            if entry.th32ParentProcessID == parent && entry.cntThreads > 0 {
                children.push(entry.th32ProcessID);
            }
            entry.dwSize = size_of::<PROCESSENTRY32W>() as u32;
            // SAFETY: as for `Process32FirstW`.
            more = unsafe { Process32NextW(snapshot.as_raw_handle(), &mut entry) } != 0;
        }
        children
    }

    fn process_is_in_job(process_id: u32, job: &JobObject) -> bool {
        // SAFETY: a limited-information handle on a process id from the
        // snapshot; a null return is checked before adoption.
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id) };
        assert!(!process.is_null(), "open descendant process");
        // SAFETY: the process handle is open and owned by nothing else.
        let process = unsafe { OwnedHandle::from_raw_handle(process) };
        let mut result = 0;
        // SAFETY: both handles are open; `result` receives one BOOL.
        assert_ne!(
            unsafe { IsProcessInJob(process.as_raw_handle(), job.handle(), &mut result) },
            0,
            "IsProcessInJob"
        );
        result != 0
    }

    fn active_processes(job: &JobObject) -> u32 {
        // SAFETY: the record is plain data, so all-zero is a valid value.
        let mut accounting: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: the pointer and length describe `accounting` for the call.
        let ok = unsafe {
            QueryInformationJobObject(
                job.handle(),
                JobObjectBasicAccountingInformation,
                std::ptr::from_mut(&mut accounting).cast::<c_void>(),
                size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                std::ptr::null_mut(),
            )
        };
        assert_ne!(ok, 0, "QueryInformationJobObject");
        accounting.ActiveProcesses
    }

    fn wait_until(deadline: Duration, mut condition: impl FnMut() -> bool) -> bool {
        let started = Instant::now();
        loop {
            if condition() {
                return true;
            }
            if started.elapsed() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn windows_profile_reports_the_validated_x64_cpu_path() {
        let profile = detect();
        assert_eq!(profile.platform, "windows");
        assert_eq!(profile.architecture, std::env::consts::ARCH);
        assert_eq!(profile.cpu_backend, "Native CPU execution");
        assert_eq!(profile.gpu_backend, "GPU acceleration not used");
        assert!(!profile.chip.trim().is_empty());
        if std::env::consts::ARCH == "x86_64" {
            assert_eq!(profile.optimization_tier, "WINDOWS_X64");
            // A brand string, not the architecture token.
            assert!(profile.chip.chars().any(|c| c.is_ascii_alphabetic()));
            assert_ne!(profile.chip, "x86_64");
        } else {
            assert_eq!(profile.optimization_tier, "PORTABLE");
        }
    }

    #[test]
    fn job_object_terminates_the_whole_child_tree() {
        // cmd.exe runs the first ping as its own child: a two-level tree in
        // which the leaf was created by the worker, not by this process.
        let mut command = Command::new("cmd.exe");
        command.args([
            "/D",
            "/S",
            "/C",
            "ping -n 30 127.0.0.1 >NUL & ping -n 30 127.0.0.1 >NUL",
        ]);
        configure_child_process(&mut command);
        let mut child = ManagedChild::spawn(&mut command).expect("spawn Windows child tree");
        let job = child.job.as_ref().expect("the child runs inside a job");
        let root = child.id();
        assert!(process_is_in_job(root, job));

        let mut descendants = Vec::new();
        assert!(
            wait_until(Duration::from_secs(10), || {
                descendants = live_children_of(root);
                !descendants.is_empty()
            }),
            "cmd.exe started no ping child"
        );
        // Membership is inherited: the grandchild joined the job without any
        // action by this process.
        assert!(descendants
            .iter()
            .all(|process_id| process_is_in_job(*process_id, job)));
        assert!(active_processes(job) >= 2);

        terminate_process_tree(&mut child).expect("terminate Windows child tree");
        let status = child.wait().expect("reap terminated Windows process");
        assert!(!status.success());
        let job = child.job.as_ref().expect("job survives termination");
        assert!(
            wait_until(Duration::from_secs(5), || active_processes(job) == 0),
            "job still reports active processes"
        );
        let descendants_gone = || {
            let live = live_children_of(root);
            descendants
                .iter()
                .all(|process_id| !live.contains(process_id))
        };
        assert!(wait_until(Duration::from_secs(5), descendants_gone));
        // Terminating an already finished tree is a no-op.
        terminate_process_tree(&mut child).expect("idempotent termination");
    }

    #[test]
    fn spawn_failure_reports_the_missing_executable() {
        let mut command = Command::new("openastroflow-missing-sidecar.exe");
        configure_child_process(&mut command);
        let error = ManagedChild::spawn(&mut command).expect_err("missing executable");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }
}
