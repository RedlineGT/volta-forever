//! Log lines, prefixed `[volta-forever]`, to standard error or to the file named by `VF_LOG`.

use std::fmt::Arguments;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

static FILE: Mutex<Option<File>> = Mutex::new(None);
static LEVEL: AtomicU8 = AtomicU8::new(1);

#[cfg(test)]
thread_local! {
    /// For a unit test: the level in this thread, whatever `VF_LOG_LEVEL` says.
    static TEST_LEVEL: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}

/// For a unit test: log at `level` in this thread (`None`: as the process does).
#[cfg(test)]
pub fn set_test_level(level: Option<u8>) {
    TEST_LEVEL.set(level);
}

fn level() -> u8 {
    #[cfg(test)]
    if let Some(level) = TEST_LEVEL.get() {
        return level;
    }
    LEVEL.load(Relaxed)
}

pub fn open() {
    if let Some(level) = std::env::var("VF_LOG_LEVEL").ok().and_then(|v| v.parse().ok()) {
        LEVEL.store(level, Relaxed);
    }
    if let Ok(path) = std::env::var("VF_LOG") {
        match File::options().create(true).append(true).open(&path) {
            Ok(file) => *FILE.lock().unwrap() = Some(file),
            Err(e) => eprintln!("[volta-forever] cannot open log file {path}: {e}"),
        }
    }
}

fn write(level: u8, args: Arguments<'_>) {
    if self::level() < level {
        return;
    }
    let line = format!("[volta-forever] {args}\n");
    let mut file = FILE.lock().unwrap_or_else(|e| e.into_inner());
    // A failed log write must not disturb the program.
    let _ = match file.as_mut() {
        Some(file) => file.write_all(line.as_bytes()),
        None => std::io::stderr().write_all(line.as_bytes()),
    };
}

/// A line at `level` (0 to 3, as below).
pub fn line(level: u8, args: Arguments<'_>) {
    write(level, args);
}

/// The library cannot work.
pub fn error(args: Arguments<'_>) {
    write(0, args);
}

/// An image that could not be adapted: the program is about to fail on it.
pub fn warn(args: Arguments<'_>) {
    write(1, args);
}

/// What the library does: where it forwards, what it adapted.
pub fn info(args: Arguments<'_>) {
    write(2, args);
}

/// Every image and every answered lookup.
pub fn trace(args: Arguments<'_>) {
    write(3, args);
}

pub fn tracing() -> bool {
    level() >= 3
}

/// Whether the current log level is at least `level`.
pub fn at_level(level: u8) -> bool {
    self::level() >= level
}

/// Per-stage timing, printed at level 0 (always visible) when `VF_TIMING` is set.
pub fn timing(args: Arguments<'_>) {
    if crate::config().timing {
        write(0, args);
    }
}
