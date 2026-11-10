use std::io;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

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
    // Two steps. SIGTERM lets the engine kill the solver processes it started:
    // each leads its own session, where no signal to this group reaches it.
    // SIGKILL then stops whatever the group still holds, so a parent cannot
    // exit after SIGTERM while a helper remains alive, and an engine that
    // ignores SIGTERM is killed all the same. The leader is not reaped in
    // between (the caller owns it and reaps it afterwards), so its PID, and
    // with it the PGID, cannot be reused during the grace interval.
    if matches!(signal_group(group, libc::SIGTERM), Ok(true)) {
        // The group ID is the leader's PID.
        let deadline = Instant::now() + TERMINATION_GRACE;
        while !exited_unreaped(group) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    match signal_group(group, libc::SIGKILL) {
        Ok(true) => Ok(()),
        Ok(false) => {
            let _ = child.kill();
            Ok(())
        }
        // macOS refuses the signal (EPERM) when the group holds nothing but
        // exited processes: the leader is gone and no member is left to kill.
        Err(_) if exited_unreaped(group) => Ok(()),
        Err(error) => Err(error),
    }
}

/// How long the engine has to stop its solver processes after SIGTERM.
const TERMINATION_GRACE: Duration = Duration::from_millis(1500);

/// Whether the child `pid` has exited.  The child stays unreaped: its exit
/// status is left for the owner of the `Child` to collect.
fn exited_unreaped(pid: libc::pid_t) -> bool {
    // SAFETY: the record is plain data, so all-zero is a valid value.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `info` outlives the call and is the only pointer passed.
    let status = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    // Without an exited child the record stays zeroed.
    status == 0 && info.si_signo != 0
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

    fn process_exists(pid: libc::pid_t) -> bool {
        // SAFETY: signal 0 only checks that the process exists.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    fn first_line(child: &mut Child) -> String {
        use std::io::BufRead;
        let mut line = String::new();
        std::io::BufReader::new(child.stdout.take().expect("piped stdout"))
            .read_line(&mut line)
            .expect("read the child's first line");
        line.trim().to_owned()
    }

    /// A stand-in engine with a solver: the solver leads its own session, and
    /// SIGTERM makes the engine kill the solver's group before dying of the
    /// signal, as the engine's own handler does.
    const ENGINE_WITH_SOLVER: &str = r#"
import os, signal, subprocess, time
solver = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
def stop(signum, frame):
    os.killpg(solver.pid, signal.SIGKILL)
    solver.wait()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
signal.signal(signal.SIGTERM, stop)
print(solver.pid, flush=True)
time.sleep(60)
"#;

    #[test]
    fn termination_stops_a_solver_that_runs_in_its_own_session() {
        use std::os::unix::process::ExitStatusExt;

        let mut command = Command::new("python3");
        command
            .args(["-c", ENGINE_WITH_SOLVER])
            .stdout(std::process::Stdio::piped());
        configure_child_process(&mut command);
        let mut child = command.spawn().expect("spawn the stand-in engine");
        let solver: libc::pid_t = first_line(&mut child).parse().expect("solver PID");
        assert!(process_exists(solver));
        // SAFETY: getpgid takes no pointer.
        let solver_group = unsafe { libc::getpgid(solver) };
        assert_eq!(solver_group, solver, "the solver leads its own group");

        let started = Instant::now();
        terminate_process_tree(&mut child).expect("terminate the engine tree");
        // The engine's exit ends the grace interval; it is not waited out.
        assert!(started.elapsed() < TERMINATION_GRACE);
        let status = child.wait().expect("reap the engine");
        assert_eq!(status.signal(), Some(libc::SIGTERM));
        let deadline = Instant::now() + Duration::from_secs(5);
        while process_exists(solver) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!process_exists(solver), "the solver outlived the engine");
    }

    #[test]
    fn an_engine_that_ignores_sigterm_is_killed_after_the_grace_interval() {
        use std::os::unix::process::ExitStatusExt;

        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "trap '' TERM; echo ready; sleep 60"])
            .stdout(std::process::Stdio::piped());
        configure_child_process(&mut command);
        let mut child = command.spawn().expect("spawn a child that ignores SIGTERM");
        assert_eq!(first_line(&mut child), "ready");

        let started = Instant::now();
        terminate_process_tree(&mut child).expect("terminate the stubborn tree");
        assert!(started.elapsed() >= TERMINATION_GRACE);
        let status = child.wait().expect("reap the killed child");
        assert_eq!(status.signal(), Some(libc::SIGKILL));
    }
}
