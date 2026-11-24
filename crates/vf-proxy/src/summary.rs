//! `VF_SUMMARY=FILE`: the report that the process prints when it exits (the kernels with emulated
//! instructions given to it, the second translations, the counts of `VF_STATS`) is also written to
//! `FILE.PID`, and rewritten while the process runs, every `VF_SUMMARY_INTERVAL` seconds (30 by
//! default) when it changed: a process that is killed without exiting (some servers end their
//! workers with SIGKILL) leaves the report as it was at most that long before. The file is replaced
//! whole (`replace`: written to a new file of this process's own beside it, which no link can lead
//! elsewhere, then renamed), readable by its owner only, and its first line says whether the process
//! had exited. Without the variable, nothing is started.
//!
//! As the reports at exit, it is the report of the process that loaded the library: a child made by
//! `fork()` has no thread to write it, and does not write it when it exits (`fini`). The lock that
//! the thread holds while it reads the report and writes the file is taken around every fork
//! (`pthread_atfork`): a child never starts with it held, nor with a lock of what the report reads
//! (the announcer's, the counts') held by a thread it does not have.

use std::cell::UnsafeCell;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
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
    if let Err(e) = replace(Path::new(&path), text.as_bytes())
        && !FAILED.swap(true, Relaxed)
    {
        log::warn(format_args!("VF_SUMMARY: cannot write {path}: {e}"));
    }
}

/// Names tried for the temporary file of one write, if the ones before are taken.
const TEMPORARY_NAMES: usize = 16;

/// Replace the file `path` by one that holds `text`: a new file is made beside it, under a name with
/// a random part (`PATH.RANDOM.tmp`), with `O_CREAT | O_EXCL | O_NOFOLLOW` and mode 0600, so that
/// nothing put there before (a file, or a link to another one) is written through; then it is renamed
/// over `path`, which replaces a link at `path` itself and not what the link leads to. The temporary
/// file is removed if anything fails. (Also the profile's, `VF_PROFILE`.)
pub fn replace(path: &Path, text: &[u8]) -> std::io::Result<()> {
    let mut taken = std::io::Error::from(std::io::ErrorKind::AlreadyExists);
    for _ in 0..TEMPORARY_NAMES {
        let mut temporary = path.as_os_str().to_owned();
        temporary.push(format!(".{:016x}.tmp", random()));
        let temporary = Path::new(&temporary);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(temporary);
        let mut file = match file {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                taken = e;
                continue;
            }
            Err(e) => return Err(e),
        };
        let written = file.write_all(text).and_then(|()| {
            drop(file);
            std::fs::rename(temporary, path)
        });
        if written.is_err() {
            let _ = std::fs::remove_file(temporary);
        }
        return written;
    }
    Err(taken)
}

/// 64 random bits, from the kernel; if it gives none, the process, a count and the time mixed.
fn random() -> u64 {
    let mut bits = 0u64;
    // SAFETY: eight writable bytes.
    let got = unsafe { libc::getrandom((&raw mut bits).cast(), 8, libc::GRND_NONBLOCK) };
    if got == 8 {
        return bits;
    }
    use std::hash::Hasher;
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut hasher = twox_hash::XxHash64::with_seed(0x7375_6d6d_6172_7900);
    hasher.write_u32(std::process::id());
    hasher.write_u64(COUNT.fetch_add(1, Relaxed));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    hasher.write_u128(now.as_nanos());
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_link_where_the_file_goes_is_replaced_not_followed() {
        let dir = std::env::temp_dir().join(format!("vf-summary-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let victim = dir.join("victim");
        std::fs::write(&victim, b"victim").unwrap();
        let path = dir.join("report");
        std::os::unix::fs::symlink(&victim, &path).unwrap();
        replace(&path, b"report").unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"victim");
        let meta = std::fs::symlink_metadata(&path).unwrap();
        assert!(meta.is_file());
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o777,
            0o600
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"report");
        // A write that fails (here: a directory where the file goes) leaves no temporary file.
        let blocked = dir.join("blocked");
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(blocked.join("inside"), b"x").unwrap();
        assert!(replace(&blocked, b"report").is_err());
        let left: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(left.len(), 3, "{left:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
