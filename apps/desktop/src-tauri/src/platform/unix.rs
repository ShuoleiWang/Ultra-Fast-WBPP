use std::io;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command};

pub(crate) fn configure_child_process(command: &mut Command) {
    command.process_group(0);
}

fn isolated_process_group(child: &Child) -> Result<libc::pid_t, String> {
    let pid = libc::pid_t::try_from(child.id()).map_err(|_| "child PID is out of range")?;
    // SAFETY: getpgid and getpgrp do not dereference pointers. The child PID
    // came from std::process::Child and remains owned by the caller.
    let group = unsafe { libc::getpgid(pid) };
    if group < 0 {
        return Err(format!(
            "cannot inspect child process group: {}",
            io::Error::last_os_error()
        ));
    }
    // SAFETY: getpgrp has no arguments and returns the caller's process group.
    let current_group = unsafe { libc::getpgrp() };
    if group != pid || group == current_group {
        return Err(format!(
            "refusing process-group signal: child pid={pid}, pgid={group}, current pgid={current_group}"
        ));
    }
    Ok(group)
}

fn signal_group(group: libc::pid_t, signal: libc::c_int) -> Result<bool, String> {
    // SAFETY: a negative, previously verified PGID targets that process group.
    // No pointer crosses the FFI boundary.
    if unsafe { libc::kill(-group, signal) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(false);
    }
    Err(format!(
        "cannot signal child process group {group}: {error}"
    ))
}

pub(crate) fn terminate_process_tree(child: &mut Child) -> Result<(), String> {
    if child
        .try_wait()
        .map_err(|error| error.to_string())?
        .is_some()
    {
        return Ok(());
    }

    let group = match isolated_process_group(child) {
        Ok(group) => group,
        Err(error) => {
            let _ = child.kill();
            return Err(error);
        }
    };
    // Cancellation has no resumable/checkpoint contract. Force the verified
    // private group in one step so a parent cannot exit after SIGTERM while a
    // helper remains alive, and so the PGID cannot be reused during a grace
    // interval before escalation.
    if !signal_group(group, libc::SIGKILL)? {
        let _ = child.kill();
        return Ok(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unisolated_child_is_killed_directly_without_signalling_the_callers_group() {
        let mut child = Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn unisolated child");
        let error = terminate_process_tree(&mut child)
            .expect_err("an inherited process group must be rejected");
        assert!(error.contains("refusing process-group signal"));
        let status = child.wait().expect("reap directly killed child");
        assert!(!status.success());
    }
}
