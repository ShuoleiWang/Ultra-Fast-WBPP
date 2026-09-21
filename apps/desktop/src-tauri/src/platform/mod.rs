//! Platform capability boundary for the desktop shell.
//!
//! The profiles describe adapter seams only. `engine_available` must remain false
//! until a platform adapter has passed the project's scientific validation suite.
//!
//! Every sidecar child is spawned through [`ManagedChild`], which pairs the
//! `std::process::Child` with the platform's handle on the whole process tree:
//! a private process group on POSIX, a Job Object on Windows.  The registries
//! keep that value alive for as long as the job runs so that cancellation and
//! application shutdown can stop every descendant, not only the direct child.

use std::io::{self, Read};
use std::ops::{Deref, DerefMut};
use std::process::{Child, Command, Output, Stdio};

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
pub(crate) use unix::configure_child_process;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub(crate) use windows::{configure_child_process, detect};

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod portable;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub(crate) use portable::detect;

/// A sidecar child together with the platform's control over its process tree.
///
/// Dereferences to the wrapped [`Child`] so the stdio handles, `id`, `wait`
/// and `try_wait` are used exactly as before.  On Windows the value also owns
/// the Job Object the child and all of its descendants belong to; closing that
/// handle (dropping this value) is itself a kill of whatever the tree left
/// behind, because the job carries `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.  The
/// job is absent only when the host refused the assignment, in which case
/// termination falls back to `taskkill`.
#[derive(Debug)]
pub(crate) struct ManagedChild {
    child: Child,
    #[cfg(target_os = "windows")]
    job: Option<windows::JobObject>,
}

impl ManagedChild {
    /// Spawns `command` under the platform's process-tree control.
    ///
    /// The caller has already applied [`configure_child_process`] through
    /// `EngineExecutable::command`; on Windows the spawn additionally starts
    /// the process suspended, assigns it to a fresh Job Object and only then
    /// lets it run, so no descendant can ever be created outside the job.
    pub(crate) fn spawn(command: &mut Command) -> io::Result<Self> {
        #[cfg(target_os = "windows")]
        {
            windows::spawn_in_job(command)
        }
        #[cfg(not(target_os = "windows"))]
        {
            Ok(Self {
                child: command.spawn()?,
            })
        }
    }

    /// Runs `command` to completion with captured stdout and stderr, exactly
    /// as `Command::output` does (stdin closed, both streams piped), but
    /// through the same process-tree control as a long-running spawn.  Short
    /// sidecar commands (inventory, quality checks) start their own worker
    /// pools; on Windows this keeps those pools inside a job that dies with
    /// the application even when nobody waits for them.
    pub(crate) fn output(command: &mut Command) -> io::Result<Output> {
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Self::spawn(command)?.wait_with_output()
    }

    /// Drains stdout and stderr concurrently (a child that fills one pipe
    /// while the other is being read must not deadlock), then reaps the child.
    pub(crate) fn wait_with_output(mut self) -> io::Result<Output> {
        drop(self.child.stdin.take());
        let stdout = self.child.stdout.take();
        let stderr = self.child.stderr.take();
        let stderr_reader = stderr.map(|mut stream| {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                stream.read_to_end(&mut bytes).map(|_| bytes)
            })
        });
        let mut stdout_bytes = Vec::new();
        let stdout_result = match stdout {
            Some(mut stream) => stream.read_to_end(&mut stdout_bytes).map(|_| ()),
            None => Ok(()),
        };
        let stderr_result = match stderr_reader {
            Some(handle) => handle
                .join()
                .unwrap_or_else(|_| Err(io::Error::other("stderr reader thread panicked"))),
            None => Ok(Vec::new()),
        };
        let status = self.child.wait()?;
        stdout_result?;
        Ok(Output {
            status,
            stdout: stdout_bytes,
            stderr: stderr_result?,
        })
    }
}

impl Deref for ManagedChild {
    type Target = Child;

    fn deref(&self) -> &Child {
        &self.child
    }
}

impl DerefMut for ManagedChild {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.child
    }
}

/// Stops the child and every process it started.  Returns `Ok` when the tree
/// is gone or was already gone; the caller still reaps the child with `wait`.
pub(crate) fn terminate_process_tree(child: &mut ManagedChild) -> Result<(), String> {
    #[cfg(unix)]
    {
        unix::terminate_process_tree(&mut child.child)
    }
    #[cfg(target_os = "windows")]
    {
        windows::terminate_process_tree(child)
    }
}

/// Test fixtures shared by the registries' cancellation and shutdown tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::process::Command;

    /// A configured sidecar stand-in: a shell (or `cmd.exe`) whose child
    /// sleeps for about thirty seconds, so termination has a tree to stop.
    pub(crate) fn sleeping_command() -> Command {
        #[cfg(unix)]
        let mut command = {
            let mut command = Command::new("/bin/sh");
            command.args(["-c", "sleep 30"]);
            command
        };
        #[cfg(target_os = "windows")]
        let mut command = {
            let mut command = Command::new("cmd.exe");
            command.args(["/D", "/S", "/C", "ping -n 30 127.0.0.1 >NUL"]);
            command
        };
        super::configure_child_process(&mut command);
        command
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registries share a child between the request thread, the stream
    /// thread and shutdown; the platform handle must not break that.
    #[test]
    fn managed_children_can_be_shared_between_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ManagedChild>();
    }

    #[test]
    fn output_captures_both_streams_of_a_managed_child() {
        #[cfg(unix)]
        let mut command = {
            let mut command = Command::new("/bin/sh");
            command.args(["-c", "echo captured-out; echo captured-err >&2"]);
            command
        };
        #[cfg(target_os = "windows")]
        let mut command = {
            let mut command = Command::new("cmd.exe");
            command.args([
                "/D",
                "/S",
                "/C",
                "echo captured-out & echo captured-err 1>&2",
            ]);
            command
        };
        configure_child_process(&mut command);
        let output = ManagedChild::output(&mut command).expect("run the child to completion");
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("captured-out"));
        assert!(String::from_utf8_lossy(&output.stderr).contains("captured-err"));
    }
}
