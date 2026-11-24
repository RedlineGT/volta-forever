//! `VF_SUMMARY=FILE`: the report that the process prints when it exits (the kernels with emulated
//! instructions given to it, the second translations, the counts of `VF_STATS`) is also written to
//! `FILE.PID`, and rewritten while the process runs, every `VF_SUMMARY_INTERVAL` seconds (30 by
//! default) when it changed: a process that is killed without exiting (some servers end their
//! workers with SIGKILL) leaves the report as it was at most that long before. The file is replaced
//! whole (written beside it, then renamed), and its first line says whether the process had exited.
//! Without the variable, nothing is started.
//!
//! As the reports at exit, it is the report of the process that loaded the library: a child made by
//! `fork()` has no thread to write it, and does not write it when it exits (`fini`). The lock that
//! the thread holds while it reads the report and writes the file is taken around every fork
//! (`pthread_atfork`): a child never starts with it held, nor with a lock of what the report reads
//! (the announcer's, the counts') held by a thread it does not have.

use std::cell::UnsafeCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::{Duration, Instant};

use crate::log;

struct Settings {
    /// `FILE`; the report goes to `FILE.PID`.
    path: String,
    interval: Duration,
    loaded: Instant,
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();
/// The report at exit was written: the thread writes no more.
static FINISHED: AtomicBool = AtomicBool::new(false);
/// A failure to write the file was said.
static FAILED: AtomicBool = AtomicBool::new(false);

/// The lock of the file and of reading the report for it: a mutex of the C library, which the
/// handlers of `pthread_atfork` take and give back around a fork.
struct Lock(UnsafeCell<libc::pthread_mutex_t>);
// SAFETY: a pthread mutex is made to be shared between threads; it is only used through its address.
unsafe impl Sync for Lock {}
static LOCK: Lock = Lock(UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER));

extern "C" fn lock() {
    // SAFETY: an initialized mutex that is never moved.
    unsafe { libc::pthread_mutex_lock(LOCK.0.get()) };
}

extern "C" fn unlock() {
    // SAFETY: as for `lock`; called by the thread that locked it (in a child, the one that forked).
    unsafe { libc::pthread_mutex_unlock(LOCK.0.get()) };
}

/// Start writing the report, if `VF_SUMMARY` asks for it (when the library is loaded).
pub fn start() {
    let Some(path) = std::env::var("VF_SUMMARY").ok().filter(|p| !p.is_empty() && p != "0") else {
        return;
    };
    let seconds = std::env::var("VF_SUMMARY_INTERVAL")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30)
        .max(1);
    let settings = Settings {
        path,
        interval: Duration::from_secs(seconds),
        loaded: Instant::now(),
    };
    if SETTINGS.set(settings).is_err() {
        return;
    }
    // SAFETY: the handlers only lock and unlock `LOCK`.
    unsafe { libc::pthread_atfork(Some(lock), Some(unlock), Some(unlock)) };
    if let Err(e) = std::thread::Builder::new().name("vf-summary".into()).spawn(rewrite) {
        log::warn(format_args!(
            "VF_SUMMARY: cannot start the thread that writes the report while the process runs ({e}); it is written at exit"
        ));
    }
}

/// The thread: the report, written again whenever it changed.
fn rewrite() {
    let Some(settings) = SETTINGS.get() else {
        return;
    };
    let mut last = None;
    loop {
        std::thread::sleep(settings.interval);
        lock();
        if FINISHED.load(Relaxed) {
            unlock();
            return;
        }
        let report = crate::report();
        if last.as_ref() != Some(&report) {
            write(settings, &report, false);
            last = Some(report);
        }
        unlock();
    }
}

/// The process exits: its report, written for the last time.
pub fn finish(report: &[String]) {
    let Some(settings) = SETTINGS.get() else {
        return;
    };
    lock();
    FINISHED.store(true, Relaxed);
    write(settings, report, true);
    unlock();
}

fn write(settings: &Settings, report: &[String], exited: bool) {
    let pid = std::process::id();
    let path = format!("{}.{pid}", settings.path);
    let mut text = if exited {
        format!("[volta-forever] report of process {pid}, at its exit\n")
    } else {
        format!(
            "[volta-forever] report of process {pid}, running for {} s (written every {} s while it changes, and at exit)\n",
            settings.loaded.elapsed().as_secs(),
            settings.interval.as_secs()
        )
    };
    for line in report {
        text += &format!("[volta-forever] {line}\n");
    }
    let temporary = format!("{path}.tmp");
    let written = std::fs::write(&temporary, text).and_then(|()| std::fs::rename(&temporary, &path));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&temporary);
        if !FAILED.swap(true, Relaxed) {
            log::warn(format_args!("VF_SUMMARY: cannot write {path}: {e}"));
        }
    }
}
