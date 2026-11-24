//! Bounded scoped worker pool for CPU-bound PTX passes with automatic calling-thread fallback
//! when `RLIMIT_NPROC` or a process limit prevents thread creation.

use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

/// Default cap on worker threads (including the calling thread) when `VF_THREADS` is unset.
pub(crate) const DEFAULT_MAX_THREADS: usize = 8;

static CONFIGURED_THREADS: OnceLock<usize> = OnceLock::new();

fn allowed_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from).max(1)
}

fn parse_threads(raw: Option<&str>, allowed: usize) -> usize {
    let default = allowed.clamp(1, DEFAULT_MAX_THREADS);
    match raw.map(str::trim) {
        Some("0" | "1") => 1,
        Some(v) if !v.is_empty() => v
            .parse::<usize>()
            .ok()
            .filter(|&n| n >= 1)
            .map_or(default, |n| n.min(allowed)),
        _ => default,
    }
}

/// Thread count configured by `VF_THREADS` (computed once per process, capped at allowed CPUs).
pub(crate) fn configured_threads() -> usize {
    *CONFIGURED_THREADS.get_or_init(|| {
        let raw = std::env::var("VF_THREADS").ok();
        parse_threads(raw.as_deref(), allowed_cpus())
    })
}

/// Apply `f` to every item in `items` in index order, using up to `max_threads` threads
/// (including the calling thread). If spawning a worker fails, stop spawning further workers and
/// execute all remaining items on the calling thread.
pub(crate) fn par_map<T: Sync, R: Send + 'static, F>(items: &[T], max_threads: usize, f: &F) -> Vec<R>
where
    F: Fn(&T) -> R + Sync,
{
    par_map_with(items, max_threads, f, |i, scope, job| {
        std::thread::Builder::new()
            .name(format!("vf-ptx-{}", i + 1))
            .spawn_scoped(scope, job)
    })
}

type ScopedChunkJob<'scope, R> = Box<dyn FnOnce() -> Vec<(usize, R)> + Send + 'scope>;

fn par_map_with<T: Sync, R: Send + 'static, F, S>(items: &[T], max_threads: usize, f: &F, mut spawn_worker: S) -> Vec<R>
where
    F: Fn(&T) -> R + Sync,
    S: for<'scope, 'env> FnMut(
        usize,
        &'scope std::thread::Scope<'scope, 'env>,
        ScopedChunkJob<'scope, R>,
    ) -> std::io::Result<std::thread::ScopedJoinHandle<'scope, Vec<(usize, R)>>>,
{
    if max_threads <= 1 || items.len() <= 1 {
        return items.iter().map(f).collect();
    }

    let next = AtomicUsize::new(0);
    let workers = max_threads.min(items.len()) - 1;
    let work_loop = || {
        let mut mine = Vec::new();
        loop {
            let idx = next.fetch_add(1, Ordering::Relaxed);
            if idx >= items.len() {
                break;
            }
            mine.push((idx, f(&items[idx])));
        }
        mine
    };

    let mut slots: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    let chunks = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let job: ScopedChunkJob<'_, R> = Box::new(work_loop);
            match spawn_worker(i, scope, job) {
                Ok(handle) => handles.push(handle),
                Err(_) => break,
            }
        }
        let mut all = Vec::with_capacity(handles.len() + 1);
        all.push(work_loop());
        for handle in handles {
            if let Ok(chunk) = handle.join() {
                all.push(chunk);
            }
        }
        all
    });
    for chunk in chunks {
        for (idx, item) in chunk {
            slots[idx] = Some(item);
        }
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(idx, slot)| slot.unwrap_or_else(|| f(&items[idx])))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_threads_handles_zero_one_explicit_and_invalid() {
        assert_eq!(parse_threads(None, 8), 8);
        assert_eq!(parse_threads(None, 16), 8);
        assert_eq!(parse_threads(None, 4), 4);
        assert_eq!(parse_threads(Some(""), 6), 6);
        assert_eq!(parse_threads(Some("garbage"), 16), 8);
        assert_eq!(parse_threads(Some("garbage"), 4), 4);
        assert_eq!(parse_threads(Some("0"), 6), 1);
        assert_eq!(parse_threads(Some("1"), 6), 1);
        assert_eq!(parse_threads(Some("4"), 16), 4);
        assert_eq!(parse_threads(Some("16"), 8), 8);
        assert_eq!(parse_threads(Some("100000"), 16), 16);
    }

    #[test]
    fn spawn_refusal_falls_back_to_calling_thread_completely_or_partially() {
        let inputs: Vec<usize> = (0..64).collect();
        let square = |x: &usize| x * x + 1;
        let expected: Vec<usize> = inputs.iter().map(&square).collect();

        // Refusal on the very first worker spawn: zero workers spawned, calling thread runs all.
        let none_spawned = par_map_with(&inputs, 8, &square, |_i, _scope, _job| {
            Err(std::io::Error::from_raw_os_error(11))
        });
        assert_eq!(none_spawned, expected);

        // Partial spawn failure after 2 workers: existing workers + calling thread finish all items.
        let mut count = 0usize;
        let partial = par_map_with(&inputs, 8, &square, |i, scope, job| {
            if count < 2 {
                count += 1;
                std::thread::Builder::new()
                    .name(format!("vf-ptx-{}", i + 1))
                    .spawn_scoped(scope, job)
            } else {
                Err(std::io::Error::from_raw_os_error(11))
            }
        });
        assert_eq!(partial, expected);
    }

    #[test]
    fn worker_threads_have_vf_ptx_names() {
        let arrived = AtomicUsize::new(0);
        let start = std::time::Instant::now();
        let inputs = [0usize, 1];
        let names = par_map(&inputs, 2, &|_| {
            arrived.fetch_add(1, Ordering::AcqRel);
            while arrived.load(Ordering::Acquire) < 2 && start.elapsed() < std::time::Duration::from_secs(2) {
                std::thread::yield_now();
            }
            std::thread::current().name().unwrap_or("").to_string()
        });
        assert!(names.iter().any(|n| n == "vf-ptx-1"), "got names: {names:?}");
    }
}
