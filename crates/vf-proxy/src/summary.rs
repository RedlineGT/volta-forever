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
//!
//! Another library can put lines of its own in the file, after the layer's (`docs/architecture.md`,
//! "The summary's lines of other libraries"): `vf_summary_add` gives the layer a function of the
//! library's, `fill`, that writes them into a buffer, and `vf_summary_remove` takes it back. These two
//! are the layer's own functions, not NVIDIA's. Every write of the file calls each `fill`, one after the
//! other, with a lock of theirs held (`FILLS`), not the file's (`LOCK`, which the handlers of
//! `pthread_atfork` take): a fork waits for no library's code. A `fill`'s text, cleaned (`text_lines`),
//! counts in whether the report changed, and each of its lines is written with the library's name in
//! front (`[NAME] `), which no line of the layer's (`[volta-forever] `) has. A removal takes `FILLS`, so
//! it waits for the `fill`s being called: a library that removes itself in its destructor is not called
//! after it. While the file is still to be written, the removal calls `fill` once more, in the process
//! that loaded the layer, and keeps what it gives for the writes that follow, marked as the library's
//! last report (`[NAME, last report] `): at exit, the layer's last write (`fini`) and the destructor of
//! a library that removes itself both run in `_dl_fini`, in the order of the libraries' loading, and
//! the library's lines are in that write whichever comes first. The last write waits for `FILLS` 2 s
//! at most (`LAST_WAIT`): a `fill` that does not return would hold the exit for ever. Once it is made,
//! both functions return at once and do nothing. In a child made by `fork()`, both functions do
//! nothing (the child writes no file), and no `fill` is called. A `fill` is called by the thread that
//! writes the file, by a removal, and at exit by the thread that exits, as the rest of the report: the
//! layer has no handler of a signal, and calls one from a handler only if the program calls `exit()`
//! from it, which POSIX does not allow.
//!
//! What a `fill` may do is little (the contract, `docs/architecture.md`): read its library's counters,
//! write text, and return at once. Every other library's `fill` and every removal waits for it; it must
//! not block, wait for another thread (which may be waiting for `FILLS`: `EDEADLK` covers the calling
//! thread only), fork, exit, call `pthread_exit` or `longjmp`, or throw (an unwinding through the
//! layer's frames aborts the process). The two functions are not async-signal-safe.

use std::cell::UnsafeCell;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
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

/// A mutex of the C library.
struct Lock(UnsafeCell<libc::pthread_mutex_t>);
// SAFETY: a pthread mutex is made to be shared between threads; it is only used through its address.
unsafe impl Sync for Lock {}

impl Lock {
    const fn new() -> Self {
        Lock(UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER))
    }

    /// The lock, held until what this gives is dropped (also when what holds it panics).
    fn take(&'static self) -> Held {
        // SAFETY: an initialized mutex that is never moved.
        unsafe { libc::pthread_mutex_lock(self.0.get()) };
        Held(self)
    }

    /// The lock if it can be taken within `wait`, else `None`.
    fn take_within(&'static self, wait: Duration) -> Option<Held> {
        let mut deadline = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: a timespec to write.
        unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut deadline) };
        let nanos = deadline.tv_nsec as u64 + u64::from(wait.subsec_nanos());
        deadline.tv_sec += (wait.as_secs() + nanos / 1_000_000_000) as libc::time_t;
        deadline.tv_nsec = (nanos % 1_000_000_000) as _;
        // SAFETY: as for `take`.
        (unsafe { libc::pthread_mutex_timedlock(self.0.get(), &deadline) } == 0).then_some(Held(self))
    }
}

struct Held(&'static Lock);

impl Drop for Held {
    fn drop(&mut self) {
        // SAFETY: the mutex that this thread took.
        unsafe { libc::pthread_mutex_unlock(self.0.0.get()) };
    }
}

/// The lock of the file and of reading the report for it, which the handlers of `pthread_atfork` take
/// and give back around a fork. Held around the layer's own code only: no library's `fill` runs with it.
static LOCK: Lock = Lock::new();

/// The lock of the registrations and of the calls of their `fill`s: a removal takes it, and so waits
/// for the `fill`s under way. Not taken around a fork, so that a fork waits for no library's code (and
/// a `fill` that forks does not wait for itself): a child made by `fork()` may have it held, by a thread
/// it does not have, and never takes it (`refused_here`).
static FILLS: Lock = Lock::new();

/// How long the last write waits for `FILLS`: a `fill` that does not return would hold the exit for
/// ever. Past it, the file is written without the other libraries' lines, and says so.
const LAST_WAIT: Duration = Duration::from_secs(2);

extern "C" fn lock() {
    // SAFETY: an initialized mutex that is never moved.
    unsafe { libc::pthread_mutex_lock(LOCK.0.get()) };
}

extern "C" fn unlock() {
    // SAFETY: as for `lock`; called by the thread that locked it (in a child, the one that forked).
    unsafe { libc::pthread_mutex_unlock(LOCK.0.get()) };
}

// ---- the lines of other libraries (`vf_summary_add`, `vf_summary_remove`) --------------------------

/// The function by which a library gives its lines: it writes at most `cap` bytes of text into `buf`,
/// reading its own counters only (no lock, no allocation, no call of the driver), and returns how many
/// it wrote. `ctx` is what the library registered with it.
pub type Fill = unsafe extern "C" fn(buf: *mut c_char, cap: usize, ctx: *mut c_void) -> usize;

/// The number of the interface of `vf_summary_add`, its first argument: a library built for another
/// one is refused, not misread.
const ABI: c_uint = 1;
/// How many libraries can be registered at once: a name keeps its slot after its library is removed
/// (its last report is written there), and the same name registered again takes it back.
const SLOTS: usize = 4;
/// The bytes that `fill` is given; a return above it is taken as it.
const CAP: usize = 4096;
/// The longest name of a library.
const NAME_MAX: usize = 64;

/// A library's registration.
struct Slot {
    name: String,
    /// Its `fill` and the `ctx` given back to it; `None` once it was removed.
    fill: Option<(Fill, usize)>,
    /// Its last report: what its `fill` gave when it was removed, as the file holds it.
    kept: Vec<String>,
}

/// The registrations, read and changed with `FILLS` held only.
struct Registry(UnsafeCell<[Option<Slot>; SLOTS]>);
// SAFETY: only used with `FILLS` held.
unsafe impl Sync for Registry {}
static REGISTRY: Registry = Registry(UnsafeCell::new([const { None }; SLOTS]));

/// The thread (its `pthread_self`) that is in a library's `fill`, with `FILLS` held, else 0: a call of
/// ours from that `fill` would wait for the lock that its thread holds. (Not a thread-local: the first
/// use of one in a thread can take a lock of the loader's, with `FILLS` held.)
static FILLING: AtomicUsize = AtomicUsize::new(0);

fn this_thread() -> usize {
    // SAFETY: no argument, no failure.
    (unsafe { libc::pthread_self() }) as usize
}

/// No thread is in a `fill` any more, if this one was (after a panic).
fn not_filling() {
    let _ = FILLING.compare_exchange(this_thread(), 0, Relaxed, Relaxed);
}

/// Adds the library `name` to those whose lines `VF_SUMMARY`'s file holds: each write of the file calls
/// `fill(buf, 4096, ctx)` (in the thread that writes it while the process runs, in the one that exits
/// for the last write), with the lock of the `fill`s held, and writes each line of what it gave with
/// `[name] ` in front, its bytes that are not UTF-8 dropped, and its control characters but the
/// newline, the line and paragraph separators and the controls of bidirectional text (`text_lines`);
/// without `VF_SUMMARY`, nothing is called. `fill` must return at once (the module's contract). `name`
/// is copied. After the last write (at exit), nothing is done and 0 returned. Returns 0, or:
/// - `ENOTSUP`: `abi` is not 1, the number of this interface;
/// - `EINVAL`: `name` or `fill` is null, or `name` is not 1 to 64 ASCII letters, digits, `.`, `_`,
///   `+` and `-`, or is `volta-forever` (the layer's own, in any case);
/// - `EEXIST`: `fill` with this `ctx` is registered under another name, or `name` with another `fill`
///   or `ctx` (a name registered again takes its slot back once its library was removed);
/// - `ENOSPC`: 4 other names are registered (a removed library's name keeps its slot);
/// - `ECHILD`: the process is a child made by `fork()` of the one that loaded the layer, which writes
///   no file: nothing is registered;
/// - `EDEADLK`: called from a `fill`;
/// - `EIO`: an internal error of the layer.
///
/// # Safety
/// `name` is null or a string that ends with a zero; `fill`, until it is removed, a function as
/// `Fill` says, that `ctx` is valid for.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vf_summary_add(
    abi: c_uint,
    name: *const c_char,
    fill: Option<Fill>,
    ctx: *mut c_void,
) -> c_int {
    guarded("vf_summary_add", || {
        if let Some(refused) = refused_here() {
            return refused;
        }
        if abi != ABI {
            return libc::ENOTSUP;
        }
        // SAFETY: as the caller promises.
        let (Some(fill), Some(name)) = (fill, unsafe { name_of(name) }) else {
            return libc::EINVAL;
        };
        if FINISHED.load(Relaxed) {
            return 0; // after the last write: nothing more is written
        }
        let _held = FILLS.take();
        if FINISHED.load(Relaxed) {
            return 0;
        }
        // SAFETY: with `FILLS` held.
        register(unsafe { &mut *REGISTRY.0.get() }, &name, fill, ctx as usize)
    })
}

/// Takes back the `fill` registered with this `ctx`: the layer does not call it after this returns, and
/// waits for the calls of `fill`s under way (every library's: the layer calls them one after the other,
/// under one lock). While `VF_SUMMARY`'s file is still to be written, in the process that loaded the
/// layer, `fill` is called once more here, and what it gives is written in its place from then on,
/// marked as the library's last report (`[name, last report] `); once the last write (at exit) is made,
/// nothing is called, nothing waited for, and 0 returned. Returns 0, or `ENOENT` (not registered),
/// `EINVAL` (`fill` is null), `ECHILD` (in a child made by `fork()`: nothing done), `EDEADLK` (called
/// from a `fill`), `EIO` (an internal error of the layer).
///
/// # Safety
/// As for `vf_summary_add`: `fill` and `ctx` still valid when this is called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vf_summary_remove(fill: Option<Fill>, ctx: *mut c_void) -> c_int {
    guarded("vf_summary_remove", || {
        if let Some(refused) = refused_here() {
            return refused;
        }
        let Some(fill) = fill else {
            return libc::EINVAL;
        };
        if FINISHED.load(Relaxed) {
            return 0; // after the last write: nothing is called, nothing more is written
        }
        let _held = FILLS.take();
        // SAFETY: with `FILLS` held.
        let slots = unsafe { &mut *REGISTRY.0.get() };
        let Some(slot) = slots.iter_mut().find(|slot| {
            slot.as_ref()
                .is_some_and(|s| s.fill.is_some_and(|f| same(f, (fill, ctx as usize))))
        }) else {
            return libc::ENOENT;
        };
        if SETTINGS.get().is_some() && !FINISHED.load(Relaxed) {
            // SAFETY: as the caller promises.
            let last = unsafe { filled(fill, ctx as usize) };
            if let Some(s) = slot.as_mut() {
                s.fill = None;
                s.kept = last;
            }
        } else {
            *slot = None;
        }
        0
    })
}

/// `f` under a guard: these functions are called by other libraries, where a panic would end the
/// process.
fn guarded(who: &str, f: impl FnOnce() -> c_int) -> c_int {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        not_filling();
        log::error(format_args!("{who}: internal error; nothing done"));
        libc::EIO
    })
}

/// Why `vf_summary_add` and `vf_summary_remove` do nothing here: in a child made by `fork()`, which has
/// its parent's registrations in its copy of the memory and writes no file (it never takes the lock,
/// which may have been held when it was made, by a thread it does not have); from a `fill`, whose
/// caller holds the lock.
fn refused_here() -> Option<c_int> {
    if !crate::loaded_here() {
        return Some(libc::ECHILD);
    }
    if FILLING.load(Relaxed) == this_thread() {
        return Some(libc::EDEADLK);
    }
    None
}

/// Two registrations of one function and one `ctx`. (Compared as addresses.)
fn same(a: (Fill, usize), b: (Fill, usize)) -> bool {
    a.0 as usize == b.0 as usize && a.1 == b.1
}

/// The name given to `vf_summary_add`, copied, when it is one (`valid_name`). No more than its
/// first 65 bytes are read.
///
/// # Safety
/// `name` is null or a string that ends with a zero.
unsafe fn name_of(name: *const c_char) -> Option<String> {
    if name.is_null() {
        return None;
    }
    let mut bytes = Vec::with_capacity(NAME_MAX + 1);
    for i in 0..=NAME_MAX {
        // SAFETY: the string's bytes up to its zero, or up to the 65th.
        let byte = unsafe { *name.add(i) } as u8;
        if byte == 0 {
            break;
        }
        bytes.push(byte);
    }
    valid_name(&bytes).then(|| String::from_utf8(bytes).ok()).flatten()
}

/// 1 to 64 ASCII letters, digits, `.`, `_`, `+` and `-`, and not the layer's own name in any case:
/// what goes between the brackets in front of a line of the file, where a reader tells whose it is.
fn valid_name(name: &[u8]) -> bool {
    (1..=NAME_MAX).contains(&name.len())
        && name.iter().all(|b| b.is_ascii_alphanumeric() || b"._+-".contains(b))
        && !name.eq_ignore_ascii_case(b"volta-forever")
}

/// `name` registered in `slots` with `fill` and `ctx`: in the slot of that name when there is one (its
/// library removed, or the same registration again), else in a free one. A name that another `fill`
/// or `ctx` holds is refused: it would take another library's lines out of the file.
fn register(slots: &mut [Option<Slot>; SLOTS], name: &str, fill: Fill, ctx: usize) -> c_int {
    let taken = |s: &Slot| s.fill.is_some_and(|f| (s.name == name) != same(f, (fill, ctx)));
    if slots.iter().flatten().any(taken) {
        return libc::EEXIST;
    }
    if let Some(slot) = slots.iter_mut().flatten().find(|s| s.name == name) {
        slot.fill = Some((fill, ctx));
        slot.kept.clear();
        return 0;
    }
    match slots.iter_mut().find(|s| s.is_none()) {
        Some(free) => {
            *free = Some(Slot {
                name: name.to_string(),
                fill: Some((fill, ctx)),
                kept: Vec::new(),
            });
            0
        }
        None => libc::ENOSPC,
    }
}

/// What `fill` gives now, line by line, as the file holds it (`text_lines`): it is given `CAP` bytes,
/// and no more of them are read, whatever it returns. With `FILLS` held, so that a removal waits.
///
/// # Safety
/// `fill` and `ctx` as registered, the library not removed.
unsafe fn filled(fill: Fill, ctx: usize) -> Vec<String> {
    let mut buffer = vec![0u8; CAP];
    FILLING.store(this_thread(), Relaxed);
    // SAFETY: a buffer of `CAP` bytes; the rest as the caller promises.
    let written = unsafe { fill(buffer.as_mut_ptr().cast(), CAP, ctx as *mut c_void) };
    FILLING.store(0, Relaxed);
    text_lines(&buffer[..written.min(CAP)])
}

/// A library's text as lines of the file: its bytes that are not UTF-8 dropped, and the characters by
/// which a reader could take a piece of a line for a line of its own or see it in another order (the
/// control characters but the newline; the line and paragraph separators, U+2028 and U+2029, at which
/// Python's `splitlines` cuts too; the marks and controls of bidirectional text), then cut at each
/// newline, empty lines left out.
fn text_lines(text: &[u8]) -> Vec<String> {
    let shown = |c: char| {
        c == '\n'
            || !(c.is_control()
                || matches!(c, '\u{2028}' | '\u{2029}' | '\u{061c}' | '\u{200e}' | '\u{200f}')
                || ('\u{202a}'..='\u{202e}').contains(&c)
                || ('\u{2066}'..='\u{2069}').contains(&c))
    };
    let mut kept = String::with_capacity(text.len());
    for chunk in text.utf8_chunks() {
        kept.extend(chunk.valid().chars().filter(|&c| shown(c)));
    }
    kept.split('\n')
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

/// The lines of the registered libraries, in the order of their slots, for a write of the file: what
/// each `fill` gives now, or a removed library's last report. With `FILLS` held. For the thread's
/// writes (`finished`), no `fill` is called once the last write is made (an exit under way). A panic
/// leaves them out of this write.
fn libraries_lines(finished: &AtomicBool) -> Vec<String> {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: with `FILLS` held.
        let slots = unsafe { &*REGISTRY.0.get() };
        let mut lines = Vec::new();
        for slot in slots.iter().flatten() {
            let (prefix, text) = match slot.fill {
                _ if finished.load(Relaxed) => break,
                // SAFETY: registered and not removed: the library is there (it removes itself before it
                // goes, and the removal waits for the lock).
                Some((fill, ctx)) => (format!("[{}] ", slot.name), unsafe { filled(fill, ctx) }),
                None => (format!("[{}, last report] ", slot.name), slot.kept.clone()),
            };
            lines.extend(text.into_iter().map(|line| format!("{prefix}{line}")));
        }
        lines
    }))
    .unwrap_or_else(|_| {
        not_filling();
        log::error(format_args!(
            "VF_SUMMARY: internal error while reading the lines of other libraries; left out"
        ));
        Vec::new()
    })
}

// ---- the file ---------------------------------------------------------------------------------------

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

/// The thread: the report, written again whenever it changed (the lines of other libraries included).
/// The other libraries' lines are read first, with `FILLS` held, then the layer's report, with `LOCK`.
fn rewrite() {
    let Some(settings) = SETTINGS.get() else {
        return;
    };
    let mut last = None;
    loop {
        std::thread::sleep(settings.interval);
        if FINISHED.load(Relaxed) {
            return;
        }
        let others = {
            let _held = FILLS.take();
            libraries_lines(&FINISHED)
        };
        let _locked = LOCK.take();
        if FINISHED.load(Relaxed) {
            return;
        }
        let mut lines = layer_lines(&crate::report());
        lines.extend(others);
        if last.as_ref() != Some(&lines) {
            write(settings, &lines, false);
            last = Some(lines);
        }
    }
}

/// The process exits: its report, written for the last time, with the other libraries' lines when
/// `FILLS` can be had within `LAST_WAIT`, and not when this thread is in a `fill` (it called `exit()`).
pub fn finish(report: &[String]) {
    let Some(settings) = SETTINGS.get() else {
        return;
    };
    let others = if FILLING.load(Relaxed) == this_thread() {
        Err("the process exited from one of their functions")
    } else if let Some(_held) = FILLS.take_within(LAST_WAIT) {
        let lines = libraries_lines(&AtomicBool::new(false));
        FINISHED.store(true, Relaxed);
        Ok(lines)
    } else {
        Err("one of their functions had not returned 2 s after the exit began")
    };
    FINISHED.store(true, Relaxed);
    let _locked = LOCK.take();
    let mut lines = layer_lines(report);
    match others {
        Ok(others) => lines.extend(others),
        Err(why) => lines.push(format!(
            "[volta-forever] the lines of other libraries are not in this last write: {why}"
        )),
    }
    write(settings, &lines, true);
}

/// The layer's lines of the file, after its first.
fn layer_lines(report: &[String]) -> Vec<String> {
    report.iter().map(|line| format!("[volta-forever] {line}")).collect()
}

fn write(settings: &Settings, lines: &[String], exited: bool) {
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
    for line in lines {
        text += line;
        text.push('\n');
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

    unsafe extern "C" fn ignored(_: *mut c_char, _: usize, _: *mut c_void) -> usize {
        0
    }

    unsafe extern "C" fn other(_: *mut c_char, _: usize, _: *mut c_void) -> usize {
        0
    }

    #[test]
    fn a_name_is_what_can_go_between_the_brackets_of_a_line() {
        for name in ["libvf_cublas_bf16.so", "a", "A-b_c.d+e", &"x".repeat(64)] {
            assert!(valid_name(name.as_bytes()), "{name}");
        }
        for name in [
            "",
            "a b",
            "a]b",
            "a\nb",
            "é",
            "volta-forever",
            "Volta-Forever",
            &"x".repeat(65),
        ] {
            assert!(!valid_name(name.as_bytes()), "{name:?}");
        }
        // No more than 65 bytes are read: a name without its zero within them is refused.
        let long = [b'x'; 65];
        assert_eq!(unsafe { name_of(long.as_ptr().cast()) }, None);
        assert_eq!(unsafe { name_of(c"lib.so".as_ptr()) }.as_deref(), Some("lib.so"));
        assert_eq!(unsafe { name_of(std::ptr::null()) }, None);
    }

    #[test]
    fn what_is_refused_before_the_registry_is_looked_at() {
        let add = |abi, name: &std::ffi::CStr, fill| unsafe {
            vf_summary_add(abi, name.as_ptr(), fill, std::ptr::null_mut())
        };
        assert_eq!(add(2, c"lib", Some(ignored as Fill)), libc::ENOTSUP);
        assert_eq!(add(0, c"lib", Some(ignored as Fill)), libc::ENOTSUP);
        assert_eq!(add(ABI, c"lib", None), libc::EINVAL);
        assert_eq!(add(ABI, c"volta-forever", Some(ignored as Fill)), libc::EINVAL);
        assert_eq!(add(ABI, c"a b", Some(ignored as Fill)), libc::EINVAL);
        assert_eq!(
            unsafe { vf_summary_add(ABI, std::ptr::null(), Some(ignored as Fill), std::ptr::null_mut()) },
            libc::EINVAL
        );
        assert_eq!(unsafe { vf_summary_remove(None, std::ptr::null_mut()) }, libc::EINVAL);
    }

    #[test]
    fn a_name_takes_its_slot_back_and_the_slots_are_four() {
        let mut slots: [Option<Slot>; SLOTS] = [const { None }; SLOTS];
        assert_eq!(register(&mut slots, "a", ignored, 1), 0);
        // The same function with the same ctx under another name: refused; with another ctx: a second.
        assert_eq!(register(&mut slots, "b", ignored, 1), libc::EEXIST);
        assert_eq!(register(&mut slots, "b", ignored, 2), 0);
        // A removed library keeps its slot, with its last report, until its name comes back.
        let a = slots[0].as_mut().unwrap();
        a.fill = None;
        a.kept = vec!["last".into()];
        assert_eq!(register(&mut slots, "c", other, 1), 0);
        assert_eq!(register(&mut slots, "d", other, 2), 0);
        assert_eq!(register(&mut slots, "e", other, 3), libc::ENOSPC);
        assert_eq!(register(&mut slots, "a", other, 4), 0);
        let a = slots[0].as_ref().unwrap();
        assert!(a.name == "a" && a.fill.is_some_and(|f| same(f, (other as Fill, 4))) && a.kept.is_empty());
        // A name that a library holds: refused to another function or ctx, the same one again taken.
        assert_eq!(register(&mut slots, "b", other, 5), libc::EEXIST);
        assert_eq!(register(&mut slots, "b", ignored, 3), libc::EEXIST);
        assert_eq!(register(&mut slots, "b", ignored, 2), 0);
        assert!(
            slots[1]
                .as_ref()
                .unwrap()
                .fill
                .is_some_and(|f| same(f, (ignored as Fill, 2)))
        );
        assert_eq!(slots.iter().flatten().count(), 4);
    }

    #[test]
    fn a_library_s_text_cannot_break_a_line_of_the_file() {
        let text = b"one\n\ntwo\tthree\r\x1b[31mred\x00\x7f\xc2\x85\xff\xc3 caf\xc3\xa9 \xe6\xbc\xa2\xed\xa0\x80\xc0\x80 end\nlast";
        assert_eq!(
            text_lines(text),
            ["one", "twothree[31mred caf\u{e9} \u{6f22} end", "last"]
        );
        assert!(text_lines(b"\n\n\x01\n").is_empty());
        // Neither a line separator nor a reversal of the text makes a line that could pass for another.
        let forged = "x\u{2028}[volta-forever] 9 kernels\u{2029}y \u{202e}desrever\u{202c} \u{2067}z\u{2069}\u{200f}";
        assert_eq!(
            text_lines(forged.as_bytes()),
            ["x[volta-forever] 9 kernelsy desrever z"]
        );
    }

    #[test]
    fn a_lock_held_elsewhere_is_given_up_after_the_wait() {
        static HELD: Lock = Lock::new();
        let held = HELD.take();
        let t0 = Instant::now();
        let wait = || HELD.take_within(Duration::from_millis(200)).is_some();
        assert!(!std::thread::spawn(wait).join().unwrap());
        assert!(t0.elapsed() >= Duration::from_millis(190), "{:?}", t0.elapsed());
        drop(held);
        assert!(std::thread::spawn(wait).join().unwrap());
    }

    static CALLED_BACK: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

    /// Writes its whole buffer, `x` then `ok` at the end, and says it wrote more; calls the layer back.
    unsafe extern "C" fn greedy(buf: *mut c_char, cap: usize, _: *mut c_void) -> usize {
        let buf = unsafe { std::slice::from_raw_parts_mut(buf.cast::<u8>(), cap) };
        buf.fill(b'x');
        buf[cap - 2..].copy_from_slice(b"ok");
        CALLED_BACK.store(
            unsafe { vf_summary_remove(Some(greedy), std::ptr::null_mut()) },
            Relaxed,
        );
        usize::MAX
    }

    #[test]
    fn a_fill_is_given_4096_bytes_and_no_more_is_read() {
        let lines = unsafe { filled(greedy, 0) };
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].len(), CAP);
        assert!(lines[0].ends_with("xok"));
        // Called from a fill, whose caller holds the lock: refused, not waited for.
        assert_eq!(CALLED_BACK.load(Relaxed), libc::EDEADLK);
        assert_eq!(FILLING.load(Relaxed), 0);
    }
}
