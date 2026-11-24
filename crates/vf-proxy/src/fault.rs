//! Which kernel faulted.
//!
//! A kernel that faults (a write through a bad address, a `trap`, a misaligned access) ends its CUDA
//! context: from then on nearly every call of the driver that takes the context returns the same
//! error, and the program stops with "an illegal memory access" or "unspecified launch failure", at
//! a call made later than the launch of the kernel that faulted (a synchronisation, a copy, another
//! launch), since launches are asynchronous. Measured on a V100 with driver 580.178.04
//! (`tests/kernels/test_fault.c`, modes `probe-*`): after a write through a null pointer (700), a
//! `trap` (719) and a misaligned read (716), the launch made 7 µs after the faulting one still
//! returned 0, and half a second later every query, synchronisation, launch, copy, module load and
//! unload, `cuFuncGetName`, `cuFuncGetModule`, `cuFuncGetAttribute` and `cuStreamIsCapturing`
//! returned that error; `cuKernelGetName`, `cuKernelGetLibrary`, `cuCtxGetCurrent`,
//! `cuCtxPushCurrent`, `cuCtxPopCurrent`, `cuLibraryUnload`, `cuCtxDestroy` and the release of the
//! primary context returned 0. Through the layer (2026-11-24, modes `fault-graph` and `fault-copy`),
//! `cuStreamGetCtx` returned the error too (the context of a stream's call is then the current one,
//! `context_of_stream`), and no other context could be made in the process.
//!
//! What the layer can say then is which kernels were launched in that context and not seen to end,
//! and what it did to each:
//!
//! - every launch that the driver accepted through the layer's launch entries is written, with its
//!   function, stream, grid and block, into a ring of the last `KEPT` launches of the thread that
//!   made it (`launched`);
//! - every synchronisation that succeeded (`cuCtxSynchronize`, `cuStreamSynchronize`, a
//!   `cuStreamQuery` that finds a stream idle) is kept by what it covers (`synchronized`): the
//!   launches kept before it began, in its context or on its stream (`Covers`);
//! - when a call that the layer routes returns an error of a fault (`is_fault`) for the first time in
//!   a context, the launches of that context that no synchronisation covers are the candidates: they
//!   are named in one report at level 0, the last first, with the image of each and what the layer
//!   made of it (`report`), and again in the report at exit (`summary_lines`). The calls looked at
//!   are the launches, the synchronisations and queries of streams, events and contexts, the loads
//!   and unloads of modules, and, before a context is destroyed or at exit, one query of the
//!   context's stream (`probe`): a fault that a copy met first (not routed: copies go straight to
//!   NVIDIA's driver) is found there.
//!
//! The cost of a launch. What a launch keeps is written with plain stores into a ring that its own
//! thread alone writes (a sequence lock that this thread alone takes, as the guard's slots are
//! read), with the value of a counter that only the synchronisations move (`EPOCH`): a launch kept
//! with a value that a synchronisation's beginning had not moved yet was accepted before that
//! synchronisation began, which therefore waited for it. No instruction that locks memory runs
//! after the driver's call, the driver is not asked which context is current, no clock is read,
//! and the record is written in the launch entry itself (inlined: a call to a function elsewhere in
//! the library cost more than the record). On a V100, in a loop of empty kernels launched from one
//! CPU (2026-11-24, `docs/performance.md`): about 25 ns a launch of 1.85 µs, where one ring shared
//! by the threads (an atomic increment and a compare-and-swap after the driver's call) took about
//! 65, reading the processor's time stamp counter too about 15 more, and finding the thread's ring
//! by its thread pointer instead of its thread-local data no less. A launch's context is found
//! when a fault is reported, from its kernel: the context of the module the program took it from
//! (`identify`); a library's kernel launched by its own handle runs in whatever context is current,
//! and is a candidate of every context.
//!
//! The candidates are said the last first: in the order of their launches in each thread, and
//! between threads by the synchronisations begun between them (`EPOCH`); launches of two threads
//! between the same two synchronisations are not ordered between threads.
//!
//! The guard (`guard`) keeps only that a handle was found allowed, not what it is, so there was
//! nothing to reuse; and since the driver names no function after a fault, the name of a kernel is
//! the one the program asked for at its lookup (`taken`). When its module goes (an unload, the end
//! of its context), what the program took from it is kept apart with that module's context and the
//! epoch then (`Forgotten`): a launch kept before is still named, and still of that context, though
//! the driver gives the handle to another kernel.
//!
//! `VF_FAULT_TRACE=1` (exact mode) synchronises the stream of every launch right after it
//! (`settle`): the launch whose synchronisation meets the fault is the kernel that faulted, unless
//! another thread was launching in the context at the same time (its launch, not yet synchronised,
//! is then listed with it). The program waits for every kernel: it runs much slower, and kernels
//! that wait for one another (a collective across GPUs, a kernel that waits for a flag that a later
//! one sets) never end. A launch into a stream being captured is not synchronised (nothing runs),
//! nor any launch while a capture is under way in the process (a synchronisation could end it).
//!
//! Locks. The locks of this file are taken in this order, and none of them is held while another
//! part of the layer is asked anything:
//!
//! - `contexts`' lock of the modules (`MODULES`) comes before all of them: `contexts::destroyed`
//!   holds it while it has the names of each module that goes forgotten (`forget_in`), which takes
//!   `TAKEN`, then `FORGOTTEN`, and under that `RINGS_HELD` (`oldest_kept`). So nothing here asks
//!   `contexts` (`context_of`, `known`) while it holds a lock of its own: the record is copied out
//!   and the lock let go first (`identify`, `forget_in`).
//! - `TAKEN` → `FORGOTTEN` → `RINGS_HELD`: never the other way.
//! - `COVERS`, `GRAPHS`, `EXECS`, `REPORTED` and `SUMMARIES` are each taken alone: no other lock is
//!   taken under them, and they are not taken under another (a value is copied out of one before
//!   the next is taken).
//! - No lock is held across a call of the driver, nor across a line written to the log; the report
//!   (`seen`) runs with none held, and the images, deferred libraries and kernels it asks of
//!   (`images::origin_text`, `lazy`, `kernels`) take their own locks alone.
//!
//! CUDA graphs: a graph's kernels run without passing through the layer. A launch made while its
//! stream is captured is recorded as a node of the graph being captured, not as a launch, and so is
//! a kernel node added by hand (`cuGraphAddKernelNode`, `cuGraphAddNode`); an instantiated graph
//! keeps what its graph had then, and a launch of it (`cuGraphLaunch`) is a candidate that names
//! those kernels. Under exact mode the graph that faulted is named, as a whole. What a graph gets
//! otherwise (a child graph, a clone, `cuGraphExecUpdate`, a node's new parameters) is not known.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, fence};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use crate::{REALS, config, idx, log};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;

/// Whether `result` is an error that a fault in a kernel leaves in its context (every later call
/// that takes the context returns it). 700, 716 and 719 were seen on a V100 (`tests/kernels/
/// test_fault.c`); the others are the errors that NVIDIA's documentation says the same of.
#[inline]
pub fn is_fault(result: CUresult) -> bool {
    matches!(result, 214 | 220 | 226 | 700 | 702 | 710 | 714..=719 | 721 | 911)
}

/// The name of an error of a fault.
fn error_name(result: CUresult) -> &'static str {
    match result {
        214 => "CUDA_ERROR_ECC_UNCORRECTABLE",
        220 => "CUDA_ERROR_NVLINK_UNCORRECTABLE",
        226 => "CUDA_ERROR_CONTAINED",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        702 => "CUDA_ERROR_LAUNCH_TIMEOUT",
        710 => "CUDA_ERROR_ASSERT",
        714 => "CUDA_ERROR_HARDWARE_STACK_ERROR",
        715 => "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        716 => "CUDA_ERROR_MISALIGNED_ADDRESS",
        717 => "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        718 => "CUDA_ERROR_INVALID_PC",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        721 => "CUDA_ERROR_TENSOR_MEMORY_LEAK",
        911 => "CUDA_ERROR_EXTERNAL_DEVICE",
        _ => "?",
    }
}

/// The launch functions of the driver, for the lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry {
    LaunchKernel,
    LaunchKernelEx,
    LaunchKernelPtsz,
    LaunchKernelExPtsz,
    Cooperative,
    CooperativePtsz,
    CooperativeMultiDevice,
    Launch,
    LaunchGrid,
    LaunchGridAsync,
    GraphLaunch,
    GraphLaunchPtsz,
}

const ENTRIES: [Entry; 12] = [
    Entry::LaunchKernel,
    Entry::LaunchKernelEx,
    Entry::LaunchKernelPtsz,
    Entry::LaunchKernelExPtsz,
    Entry::Cooperative,
    Entry::CooperativePtsz,
    Entry::CooperativeMultiDevice,
    Entry::Launch,
    Entry::LaunchGrid,
    Entry::LaunchGridAsync,
    Entry::GraphLaunch,
    Entry::GraphLaunchPtsz,
];

impl Entry {
    pub fn name(self) -> &'static str {
        match self {
            Entry::LaunchKernel => "cuLaunchKernel",
            Entry::LaunchKernelEx => "cuLaunchKernelEx",
            Entry::LaunchKernelPtsz => "cuLaunchKernel_ptsz",
            Entry::LaunchKernelExPtsz => "cuLaunchKernelEx_ptsz",
            Entry::Cooperative => "cuLaunchCooperativeKernel",
            Entry::CooperativePtsz => "cuLaunchCooperativeKernel_ptsz",
            Entry::CooperativeMultiDevice => "cuLaunchCooperativeKernelMultiDevice",
            Entry::Launch => "cuLaunch",
            Entry::LaunchGrid => "cuLaunchGrid",
            Entry::LaunchGridAsync => "cuLaunchGridAsync",
            Entry::GraphLaunch => "cuGraphLaunch",
            Entry::GraphLaunchPtsz => "cuGraphLaunch_ptsz",
        }
    }

    /// Whether the null stream is the thread's own default stream for this function.
    pub fn per_thread(self) -> bool {
        matches!(
            self,
            Entry::LaunchKernelPtsz | Entry::LaunchKernelExPtsz | Entry::CooperativePtsz | Entry::GraphLaunchPtsz
        )
    }

    fn code(self) -> u64 {
        ENTRIES.iter().position(|&e| e == self).unwrap_or(0) as u64
    }

    fn of(code: u64) -> Entry {
        ENTRIES.get(code as usize).copied().unwrap_or(Entry::LaunchKernel)
    }
}

/// The stream of a launch: the null stream as the legacy stream of its context, or as the thread's
/// own; any other, its handle.
const LEGACY: u64 = 1;
const PER_THREAD: u64 = 2;

fn stream_id(stream: *mut c_void, per_thread: bool) -> u64 {
    match stream as u64 {
        0 if per_thread => PER_THREAD,
        0 | 1 => LEGACY,
        2 => PER_THREAD,
        handle => handle,
    }
}

// ---- the launches of each thread -----------------------------------------------------------------

/// The launches kept of each thread: its last `KEPT`.
const KEPT: usize = 256;
/// The threads whose launches are kept at a time. A ring whose thread ended is given to the next
/// thread that launches (its launches stay until they are written over); a thread that finds every
/// ring held keeps nothing, and the report says how many did (`UNKEPT`).
const RINGS: usize = 128;

/// What a launch is.
const LAUNCH: u64 = 1;
const GRAPH: u64 = 2;

/// One launch, as it is read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Launch {
    /// The ring it is in (`Ring::index`), and its number there.
    ring: usize,
    number: u64,
    /// The thread that made it, as the lines number threads.
    thread: u64,
    /// `LAUNCH`, or `GRAPH` (`what` is then the instantiated graph).
    kind: u64,
    entry: Entry,
    /// Launched from its second translation (`fit`).
    fitted: bool,
    what: u64,
    stream: u64,
    grid: [u32; 3],
    block: [u32; 3],
    /// The value of `EPOCH` when it was kept.
    epoch: u64,
}

impl Launch {
    /// The grid in one word: x in 32 bits, y and z in 16 (65,535 at most on this GPU).
    fn grid_word(&self) -> u64 {
        u64::from(self.grid[0]) | (u64::from(self.grid[1] & 0xffff) << 32) | (u64::from(self.grid[2] & 0xffff) << 48)
    }

    /// The block in 48 bits (1,024 at most), then the kind, the entry and whether it was fitted.
    fn block_word(&self) -> u64 {
        u64::from(self.block[0] & 0xffff)
            | (u64::from(self.block[1] & 0xffff) << 16)
            | (u64::from(self.block[2] & 0xffff) << 32)
            | ((self.kind & 0xf) << 48)
            | ((self.entry.code() & 0xf) << 52)
            | (u64::from(self.fitted) << 56)
    }
}

/// An entry of a ring: written by the thread that holds the ring, read without a lock. `version`
/// is `2n + 1` while launch `n` is written, `2n + 2` once it is.
#[repr(align(64))]
struct Slot {
    version: AtomicU64,
    what: AtomicU64,
    stream: AtomicU64,
    grid: AtomicU64,
    block: AtomicU64,
    epoch: AtomicU64,
    /// The thread that made it, as the lines number threads: the whole number, as the
    /// synchronisations of the thread's own default stream are kept by it (`Covers::own`).
    thread: AtomicU64,
}

impl Slot {
    const fn empty() -> Slot {
        Slot {
            version: AtomicU64::new(0),
            what: AtomicU64::new(0),
            stream: AtomicU64::new(0),
            grid: AtomicU64::new(0),
            block: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            thread: AtomicU64::new(0),
        }
    }

    /// Write `launch` as launch `number`. Only the thread that holds the ring writes.
    #[inline(always)]
    fn write(&self, number: u64, launch: &Launch) {
        self.version.store(2 * number + 1, Relaxed);
        fence(Release);
        self.what.store(launch.what, Relaxed);
        self.stream.store(launch.stream, Relaxed);
        self.grid.store(launch.grid_word(), Relaxed);
        self.block.store(launch.block_word(), Relaxed);
        self.epoch.store(launch.epoch, Relaxed);
        self.thread.store(launch.thread, Relaxed);
        self.version.store(2 * number + 2, Release);
    }

    /// The launch in the slot, if one was written and none is being written.
    fn read(&self, ring: usize) -> Option<Launch> {
        let version = self.version.load(Acquire);
        if version & 1 != 0 || version == 0 {
            return None;
        }
        let (what, stream, grid, block, epoch, thread) = (
            self.what.load(Relaxed),
            self.stream.load(Relaxed),
            self.grid.load(Relaxed),
            self.block.load(Relaxed),
            self.epoch.load(Relaxed),
            self.thread.load(Relaxed),
        );
        fence(Acquire);
        if self.version.load(Relaxed) != version {
            return None;
        }
        Some(Launch {
            ring,
            number: version / 2 - 1,
            thread,
            kind: (block >> 48) & 0xf,
            entry: Entry::of((block >> 52) & 0xf),
            fitted: (block >> 56) & 1 != 0,
            what,
            stream,
            grid: [grid as u32, ((grid >> 32) & 0xffff) as u32, (grid >> 48) as u32],
            block: [
                (block & 0xffff) as u32,
                ((block >> 16) & 0xffff) as u32,
                ((block >> 32) & 0xffff) as u32,
            ],
            epoch,
        })
    }
}

/// The last `KEPT` launches of a thread.
struct Ring {
    slots: [Slot; KEPT],
    /// Its place in `RINGS_HELD`.
    index: usize,
    /// The number of the next launch. Written by the thread that holds the ring alone.
    next: AtomicU64,
    /// Exact mode: the launches numbered below this were waited for and ended without a fault.
    done: AtomicU64,
    /// The thread that holds it, as the lines number threads (from 1).
    thread: AtomicU64,
    /// Whether a thread holds it.
    held: AtomicBool,
}

impl Ring {
    fn new(index: usize) -> Ring {
        Ring {
            slots: [const { Slot::empty() }; KEPT],
            index,
            next: AtomicU64::new(0),
            done: AtomicU64::new(0),
            thread: AtomicU64::new(0),
            held: AtomicBool::new(false),
        }
    }

    /// Keep `launch`, by the thread that holds the ring; its number.
    #[inline(always)]
    fn push(&self, launch: &mut Launch) -> u64 {
        let number = self.next.load(Relaxed);
        launch.thread = self.thread.load(Relaxed);
        self.slots[number as usize % KEPT].write(number, launch);
        self.next.store(number + 1, Release);
        number
    }

    /// The launches kept, in no order.
    fn launches(&self) -> Vec<Launch> {
        self.slots.iter().filter_map(|slot| slot.read(self.index)).collect()
    }

    /// Whether launches were kept that the ring no longer holds.
    fn wrapped(&self) -> bool {
        self.next.load(Acquire) > KEPT as u64
    }
}

/// The rings given to threads so far (never freed: a ring whose thread ended is given again).
static RINGS_HELD: Mutex<Vec<&'static Ring>> = Mutex::new(Vec::new());
/// The threads numbered so far.
static THREADS: AtomicU64 = AtomicU64::new(0);
/// The threads that launched kernels and found no ring.
static UNKEPT: AtomicU64 = AtomicU64::new(0);
/// Whether any launch was kept (`probe` asks nothing of a process that launched nothing).
static LAUNCHED: AtomicBool = AtomicBool::new(false);
/// The value of this thread's ring pointer once the thread has none to keep its launches in.
const NO_RING: usize = 1;

thread_local! {
    /// This thread's ring: null before its first launch, `NO_RING` when it has none.
    static MINE: Cell<*const Ring> = const { Cell::new(std::ptr::null()) };
    /// Gives this thread's ring back when the thread ends.
    static HOLDER: Holder = const { Holder };
}

struct Holder;

impl Drop for Holder {
    fn drop(&mut self) {
        let ring = MINE.try_with(|mine| mine.replace(NO_RING as *const Ring));
        if let Ok(ring) = ring
            && !ring.is_null()
            && ring as usize != NO_RING
        {
            // SAFETY: a ring is never freed.
            unsafe { &*ring }.held.store(false, Release);
        }
    }
}

/// This thread's ring, given at its first launch.
#[inline(always)]
fn mine() -> Option<&'static Ring> {
    let ring = MINE.try_with(Cell::get).unwrap_or(NO_RING as *const Ring);
    if ring.is_null() {
        return take_ring();
    }
    if ring as usize == NO_RING {
        return None;
    }
    // SAFETY: a ring is never freed, and this thread holds it.
    Some(unsafe { &*ring })
}

/// The ring of this thread's first launch: one whose thread ended, else a new one.
#[cold]
#[inline(never)]
fn take_ring() -> Option<&'static Ring> {
    // (Its holder first: a thread that is ending, whose holder cannot be made, takes no ring.)
    if HOLDER.try_with(|_| ()).is_err() {
        let _ = MINE.try_with(|mine| mine.set(NO_RING as *const Ring));
        return None;
    }
    let ring = {
        let mut rings = RINGS_HELD.lock().unwrap_or_else(|e| e.into_inner());
        match rings.iter().find(|ring| !ring.held.load(Acquire)) {
            Some(&ring) => Some(ring),
            None if rings.len() < RINGS => {
                let ring: &'static Ring = Box::leak(Box::new(Ring::new(rings.len())));
                rings.push(ring);
                Some(ring)
            }
            None => None,
        }
        .inspect(|ring| {
            ring.thread.store(THREADS.fetch_add(1, Relaxed) + 1, Relaxed);
            ring.held.store(true, Release);
        })
    };
    let Some(ring) = ring else {
        UNKEPT.fetch_add(1, Relaxed);
        let _ = MINE.try_with(|mine| mine.set(NO_RING as *const Ring));
        return None;
    };
    let _ = MINE.try_with(|mine| mine.set(ring));
    LAUNCHED.store(true, Relaxed);
    // (Registered now, after the driver's own handlers: those of `cuInit` and of the contexts made.)
    static EXIT_HANDLER: std::sync::Once = std::sync::Once::new();
    EXIT_HANDLER.call_once(|| unsafe {
        libc::atexit(crate::at_exit);
    });
    Some(ring)
}

/// What `launched` did with a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kept {
    /// Kept in this thread's ring as its launch of this number.
    At(u64),
    /// Not kept: this thread has no ring.
    Not,
    /// The stream is being captured: the kernel is a node of the graph, and nothing runs.
    Captured,
}

/// Keep a launch of `kind` in this thread's ring.
#[inline(always)]
fn keep(kind: u64, what: u64, grid: [u32; 3], block: [u32; 3], stream: u64, entry: Entry, fitted: bool) -> Kept {
    let Some(ring) = mine() else {
        return Kept::Not;
    };
    let mut launch = Launch {
        ring: ring.index,
        number: 0,
        thread: 0,
        kind,
        entry,
        fitted,
        what,
        stream,
        grid,
        block,
        epoch: EPOCH.load(Acquire),
    };
    Kept::At(ring.push(&mut launch))
}

/// The driver accepted the launch of `function` (with `fitted`: of its second translation) on
/// `stream` through `entry`: it is kept, unless the stream is being captured (the kernel is then a
/// node of the graph being captured, and nothing runs).
#[inline(always)]
pub fn launched(
    function: *mut c_void,
    grid: [u32; 3],
    block: [u32; 3],
    stream: *mut c_void,
    entry: Entry,
    fitted: bool,
) -> Kept {
    if CAPTURING.load(Relaxed) != 0
        && let Some(graph) = unsafe { capturing_into(stream, entry.per_thread()) }
    {
        node_added(graph, function as u64, grid, block);
        return Kept::Captured;
    }
    keep(
        LAUNCH,
        function as u64,
        grid,
        block,
        stream_id(stream, entry.per_thread()),
        entry,
        fitted,
    )
}

// ---- what the synchronisations cover -------------------------------------------------------------

/// The synchronisations begun so far. A launch keeps the value it finds (`Launch::epoch`); a
/// synchronisation moves it when it begins (`begin`), and once it succeeded covers the launches
/// that found the value it moved: those were accepted by the driver before it began.
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// A synchronisation (or the destruction of a stream or a context) begins: the launches it covers,
/// those whose epoch is at most this.
#[inline]
pub fn begin() -> u64 {
    EPOCH.fetch_add(1, SeqCst)
}

/// The most records of each kind kept: beyond, the records of that kind are dropped, which makes
/// more candidates, never fewer.
const COVERS_KEPT: usize = 4096;
/// The spans of epochs kept for a stream's handle (`StreamCover`).
const SPANS: usize = 8;

/// What the synchronisations that succeeded cover.
#[derive(Clone, Default, Debug)]
struct Covers {
    /// By context: a synchronisation of the whole context, or its destruction (the launches of a
    /// context at that address before are not of a context made there since): its launches whose
    /// epoch is at most this.
    contexts: HashMap<u64, u64>,
    /// By context: its legacy stream.
    legacy: HashMap<u64, u64>,
    /// By the handle of a stream.
    streams: HashMap<u64, StreamCover>,
    /// The first epoch of a stream whose handle has no record (records were dropped: what was
    /// launched before may have been on another stream of that handle).
    streams_from: u64,
    /// By thread (as the lines number threads) and context: the thread's own default stream. (Not
    /// by ring: a ring whose thread ended is given to another, whose own stream is another.)
    own: HashMap<(u64, u64), u64>,
    /// The contexts destroyed, and not used since.
    gone: HashSet<u64>,
}

/// What the synchronisations of a stream's handle cover: spans of epochs, from the first to the
/// last, each from a stream at that handle. A stream destroyed and another made at its handle
/// begins a span of its own: what a synchronisation of the new one covers is not the old one's.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct StreamCover {
    /// The first epoch of the stream at the handle now.
    from: u64,
    /// Spans covered, inclusive.
    spans: Vec<(u64, u64)>,
}

impl StreamCover {
    fn covers(&self, epoch: u64) -> bool {
        self.spans.iter().any(|&(from, to)| from <= epoch && epoch <= to)
    }

    fn synchronized(&mut self, covers: u64) {
        if covers < self.from {
            return;
        }
        match self.spans.iter_mut().find(|(from, _)| *from == self.from) {
            Some(span) => span.1 = span.1.max(covers),
            None => {
                if self.spans.len() >= SPANS {
                    self.spans.remove(0);
                }
                self.spans.push((self.from, covers));
            }
        }
    }
}

static COVERS: LazyLock<Mutex<Covers>> = LazyLock::new(|| Mutex::new(Covers::default()));

/// The record of what the synchronisations that succeeded cover.
fn record() -> MutexGuard<'static, Covers> {
    COVERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Keep `value` for `key` in `map`, the larger of the two; the map is emptied when it is full.
fn cover<K: std::hash::Hash + Eq>(map: &mut HashMap<K, u64>, key: K, value: u64) {
    if map.len() >= COVERS_KEPT && !map.contains_key(&key) {
        map.clear();
    }
    let kept = map.entry(key).or_insert(value);
    *kept = (*kept).max(value);
}

impl Covers {
    fn context_synchronized(&mut self, context: u64, covers: u64) {
        self.gone.remove(&context);
        cover(&mut self.contexts, context, covers);
    }

    fn context_gone(&mut self, context: u64, covers: u64) {
        cover(&mut self.contexts, context, covers);
        if self.gone.len() < COVERS_KEPT {
            self.gone.insert(context);
        }
    }

    fn legacy_synchronized(&mut self, context: u64, covers: u64) {
        self.gone.remove(&context);
        cover(&mut self.legacy, context, covers);
    }

    fn own_synchronized(&mut self, thread: u64, context: u64, covers: u64) {
        cover(&mut self.own, (thread, context), covers);
    }

    fn stream(&mut self, stream: u64) -> &mut StreamCover {
        if self.streams.len() >= COVERS_KEPT && !self.streams.contains_key(&stream) {
            self.streams.clear();
            self.streams_from = begin() + 1;
        }
        let from = self.streams_from;
        self.streams.entry(stream).or_insert_with(|| StreamCover {
            from,
            spans: Vec::new(),
        })
    }

    fn stream_synchronized(&mut self, stream: u64, covers: u64) {
        self.stream(stream).synchronized(covers);
    }

    /// The stream at `stream` is destroyed when the epoch moved from `covers`: a stream made at its
    /// handle later begins after.
    fn stream_destroyed(&mut self, stream: u64, covers: u64) {
        let record = self.stream(stream);
        record.from = record.from.max(covers + 1);
    }

    /// Whether `launch`, taken to be of `context`, is covered.
    fn covers(&self, launch: &Launch, context: u64) -> bool {
        if self.contexts.get(&context).is_some_and(|&c| launch.epoch <= c) {
            return true;
        }
        match launch.stream {
            LEGACY => self.legacy.get(&context).is_some_and(|&c| launch.epoch <= c),
            PER_THREAD => self
                .own
                .get(&(launch.thread, context))
                .is_some_and(|&c| launch.epoch <= c),
            stream => {
                launch.epoch >= self.streams_from
                    && self
                        .streams
                        .get(&stream)
                        .is_some_and(|record| record.covers(launch.epoch))
            }
        }
    }
}

/// The context current in this thread (0: none).
fn current_context() -> u64 {
    let real = REALS[idx::cuCtxGetCurrent].load(Relaxed);
    if real == 0 {
        return 0;
    }
    // SAFETY: NVIDIA's cuCtxGetCurrent, or a stub that ignores its argument.
    let get: unsafe extern "C" fn(*mut *mut c_void) -> CUresult = unsafe { std::mem::transmute(real) };
    let mut context = std::ptr::null_mut();
    match unsafe { get(&mut context) } {
        CUDA_SUCCESS => context as u64,
        _ => 0,
    }
}

/// The context of `stream` (of the thread's current context for the null stream), as the driver
/// says, else the current one.
fn context_of_stream(stream: *mut c_void, per_thread: bool) -> u64 {
    if matches!(stream_id(stream, per_thread), LEGACY | PER_THREAD) {
        return current_context();
    }
    let real = REALS[idx::cuStreamGetCtx].load(Relaxed);
    if real != 0 {
        // SAFETY: NVIDIA's cuStreamGetCtx, or a stub that ignores its arguments.
        let get: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> CUresult = unsafe { std::mem::transmute(real) };
        let mut context = std::ptr::null_mut();
        if unsafe { get(stream, &mut context) } == CUDA_SUCCESS && !context.is_null() {
            return context as u64;
        }
    }
    current_context()
}

/// What the program's call `call` returned, when it is an error of a fault, is reported for the
/// current context (once in each context).
#[inline]
pub fn check(result: CUresult, call: &str) {
    if is_fault(result) {
        seen(result, call, None);
    }
}

/// What a synchronisation is of.
pub enum Scope {
    /// The current context (`cuCtxSynchronize`).
    Context,
    /// A context named (`cuCtxSynchronize_v2`; null: the current one).
    Named(*mut c_void),
    /// A stream, and whether its null is the thread's own (`_ptsz`).
    Stream(*mut c_void, bool),
}

/// A synchronisation of `scope` that began with `begin` (which gave `covers`) returned `result`.
#[inline]
pub fn synchronized(result: CUresult, covers: u64, scope: Scope, call: &str) {
    if result != CUDA_SUCCESS {
        if is_fault(result) {
            let context = match scope {
                Scope::Named(context) if !context.is_null() => context as u64,
                Scope::Stream(stream, per_thread) => context_of_stream(stream, per_thread),
                _ => current_context(),
            };
            seen(result, call, Some(context));
        }
        return;
    }
    match scope {
        Scope::Named(context) if !context.is_null() => record().context_synchronized(context as u64, covers),
        Scope::Context | Scope::Named(_) => {
            let context = current_context();
            record().context_synchronized(context, covers);
        }
        Scope::Stream(stream, per_thread) => match stream_id(stream, per_thread) {
            LEGACY => {
                let context = current_context();
                record().legacy_synchronized(context, covers);
            }
            PER_THREAD => {
                // (A thread that launched nothing has nothing of its own to cover.)
                let Some(ring) = MINE
                    .try_with(Cell::get)
                    .ok()
                    .filter(|r| !r.is_null() && *r as usize != NO_RING)
                else {
                    return;
                };
                // SAFETY: a ring is never freed.
                let thread = unsafe { &*ring }.thread.load(Relaxed);
                let context = current_context();
                record().own_synchronized(thread, context, covers);
            }
            stream => record().stream_synchronized(stream, covers),
        },
    }
}

/// The program destroys `stream`: a stream made at its handle later is another (`StreamCover`).
pub fn stream_destroyed(stream: *mut c_void) {
    let stream = stream_id(stream, false);
    if !matches!(stream, LEGACY | PER_THREAD) {
        let epoch = begin();
        record().stream_destroyed(stream, epoch);
    }
}

/// Exact mode (`VF_FAULT_TRACE=1`): the launch that `kept` says, on `stream`, through `call`, is
/// waited for; what the program gets in place of the launch's success: the error of a fault, if
/// the synchronisation met one, else success. Nothing is waited for when nothing ran (the stream is
/// being captured), nor while a graph is being captured anywhere in the process: a synchronisation
/// then may be refused (900) and end that capture.
///
/// # Safety
/// `stream` is the stream the program launched on.
pub unsafe fn settle(kept: Kept, stream: *mut c_void, per_thread: bool, call: &str) -> CUresult {
    if kept == Kept::Captured || CAPTURING.load(Relaxed) != 0 {
        return CUDA_SUCCESS;
    }
    let index = if per_thread {
        idx::cuStreamSynchronize_ptsz
    } else {
        idx::cuStreamSynchronize
    };
    // SAFETY: NVIDIA's cuStreamSynchronize (or its form for the thread's stream), or a stub.
    let sync: unsafe extern "C" fn(*mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[index].load(Relaxed)) };
    let result = unsafe { sync(stream) };
    if result == CUDA_SUCCESS {
        // (This thread waited for each of its launches in turn: every one before ended too.)
        if let (Kept::At(number), Some(ring)) = (kept, mine()) {
            ring.done.store(number + 1, Release);
        }
    } else if is_fault(result) {
        seen(
            result,
            &format!("the synchronisation of the stream after {call} (VF_FAULT_TRACE=1)"),
            Some(context_of_stream(stream, per_thread)),
        );
        return result;
    }
    // (Another error is the synchronisation's, not the launch's: the program meets it later.)
    CUDA_SUCCESS
}

// ---- the names of kernels ------------------------------------------------------------------------

/// What the program took a kernel or function as: the module or library it asked (or the deferred
/// library that a kernel compiled alone belongs to), and the name.
#[derive(Clone)]
struct Taken {
    owner: usize,
    name: Arc<str>,
    /// The handle that the lookup gave (for a function of a library's kernel, the kernel's).
    handle: usize,
    /// For a function of a library's kernel (`cuKernelGetFunction`): the module that the driver
    /// made of the library in the context where it gave the function.
    module: Option<usize>,
}

static TAKEN: LazyLock<Mutex<HashMap<usize, Taken>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn taken_map() -> MutexGuard<'static, HashMap<usize, Taken>> {
    TAKEN.lock().unwrap_or_else(|e| e.into_inner())
}

/// The program took `handle` as the kernel `name` of `owner` (a lookup or an enumeration).
pub fn taken(handle: usize, owner: usize, name: &str) {
    if handle == 0 {
        return;
    }
    let mut map = taken_map();
    if map
        .get(&handle)
        .is_some_and(|known| known.owner == owner && &*known.name == name)
    {
        return;
    }
    map.insert(
        handle,
        Taken {
            owner,
            name: Arc::from(name),
            handle,
            module: None,
        },
    );
}

/// The program took `function` as the function of the kernel `kernel` (`cuKernelGetFunction`) in
/// the context where the driver made `module` of its library: the same name.
pub fn alias(function: usize, kernel: usize, module: Option<usize>) {
    let mut map = taken_map();
    let Some(known) = map.get(&kernel).cloned() else {
        return;
    };
    if map
        .get(&function)
        .is_some_and(|f| f.owner == known.owner && Arc::ptr_eq(&f.name, &known.name) && f.module == module)
    {
        return;
    }
    map.insert(function, Taken { module, ..known });
}

/// What the program took by a handle whose module or library went since (`forget`), with the
/// context of that module and the epoch when it went: a launch kept before is of that kernel, in
/// that context, whatever the driver gave the handle to since.
#[derive(Clone)]
struct Forgotten {
    taken: Taken,
    context: Option<u64>,
    at: u64,
}

static FORGOTTEN: LazyLock<Mutex<HashMap<usize, Forgotten>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
/// Past this many handles forgotten, those that no launch kept can be of are dropped (`oldest_kept`);
/// past `FORGOTTEN_KEPT`, all (their launches are then named by their address, of no known context).
const FORGOTTEN_PRUNED: usize = 256;
const FORGOTTEN_KEPT: usize = 2048;

fn forgotten() -> MutexGuard<'static, HashMap<usize, Forgotten>> {
    FORGOTTEN.lock().unwrap_or_else(|e| e.into_inner())
}

/// What `forget` moved: the names taken from `owner`, and the records of the handles that they
/// replaced among those kept apart, for `unforget` when the driver does not unload the module.
pub struct Forgot {
    gone: Vec<(usize, Taken)>,
    /// When they were kept apart (`None`: they were not, as no launch was kept).
    at: Option<u64>,
    replaced: Vec<(usize, Option<Forgotten>)>,
}

/// The module or library `owner` is to be unloaded, or went with its context: the driver may give
/// the handles of its kernels to others. What `unforget` takes to undo it, if the unload fails.
pub fn forget(owner: usize) -> Forgot {
    forget_in(owner, None)
}

/// The driver did not unload the module or library of which `forget` gave `forgot`: its kernels are
/// named as before, by the handles the program took (unless another took a handle since), and the
/// launches kept before are not of a module gone.
pub fn unforget(forgot: Forgot) {
    let Forgot { gone, at, replaced } = forgot;
    {
        let mut map = taken_map();
        for (handle, taken) in gone {
            map.entry(handle).or_insert(taken);
        }
    }
    let Some(at) = at else {
        return;
    };
    let mut forgotten = forgotten();
    for (handle, before) in replaced {
        if forgotten.get(&handle).is_some_and(|f| f.at == at) {
            match before {
                Some(before) => forgotten.insert(handle, before),
                None => forgotten.remove(&handle),
            };
        }
    }
}

/// As `forget`, for a module of `context` whose record `contexts` has dropped already (under its
/// lock: `contexts` is not asked).
pub fn forget_in(owner: usize, context: Option<usize>) -> Forgot {
    let gone: Vec<(usize, Taken)> = {
        let mut map = taken_map();
        let handles: Vec<usize> = map
            .iter()
            .filter(|(_, known)| known.owner == owner || known.module == Some(owner))
            .map(|(&handle, _)| handle)
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| map.remove(&handle).map(|known| (handle, known)))
            .collect()
    };
    // (Before any launch was kept, no launch can be of what goes.)
    if gone.is_empty() || !LAUNCHED.load(Relaxed) {
        return Forgot {
            gone,
            at: None,
            replaced: Vec::new(),
        };
    }
    // (The launches kept before found an epoch at most this; those of a kernel given the handle
    // since, a later one.)
    let at = begin();
    let kept: Vec<(usize, Forgotten)> = gone
        .iter()
        .map(|(handle, taken)| {
            let context = match context {
                Some(context) => Some(context),
                None => taken
                    .module
                    .and_then(crate::contexts::context_of)
                    .or_else(|| crate::contexts::context_of(taken.owner)),
            };
            let context = context.map(|c| c as u64);
            (
                *handle,
                Forgotten {
                    taken: taken.clone(),
                    context,
                    at,
                },
            )
        })
        .collect();
    let mut forgotten = forgotten();
    let replaced = kept
        .into_iter()
        .map(|(handle, record)| (handle, forgotten.insert(handle, record)))
        .collect();
    if forgotten.len() > FORGOTTEN_PRUNED {
        match oldest_kept() {
            Some(oldest) => forgotten.retain(|_, f| f.at >= oldest),
            None => forgotten.clear(),
        }
        if forgotten.len() > FORGOTTEN_KEPT {
            forgotten.clear();
        }
    }
    Forgot {
        gone,
        at: Some(at),
        replaced,
    }
}

/// The epoch of the earliest launch that a ring still holds (`None`: none holds any). What went
/// before it is of no launch kept.
fn oldest_kept() -> Option<u64> {
    let rings: Vec<&'static Ring> = RINGS_HELD.lock().unwrap_or_else(|e| e.into_inner()).clone();
    oldest_of(&rings)
}

/// The epoch of the earliest launch that `rings` hold: the least of all the launches they hold now.
/// (Not the launch of the slot that the count of a ring says is the earliest: its thread may have
/// written its next launch there already, whose epoch is later, before it counts it. A slot being
/// written is not read: the launch it held is being dropped, and the one written is the latest.)
/// Asked only when many names were kept apart (`forget_in`): it reads every slot.
fn oldest_of(rings: &[&Ring]) -> Option<u64> {
    rings
        .iter()
        .flat_map(|ring| ring.slots.iter().filter_map(|slot| slot.read(ring.index)))
        .map(|launch| launch.epoch)
        .min()
}

/// What the layer knows of what the program launched as a handle (`identify`).
#[derive(Debug, Default, PartialEq, Eq)]
struct Identity {
    /// The module or library it was taken from.
    owner: Option<usize>,
    name: Option<String>,
    /// The handle that the lookup gave (for a function of a library's kernel, the kernel's).
    handle: usize,
    /// Its context, when the layer knows it (a module's function, or the function of a library's
    /// kernel in one context).
    context: Option<u64>,
    /// Its module or library went since the launch (`Forgotten`).
    gone: bool,
}

/// What the program launched as `handle` in `epoch` (`None`: now): as the program took it (by the
/// handle's record when it was launched, if its module went since), else as the driver names a
/// library's kernel (which it still does after a fault).
fn identify(handle: usize, epoch: Option<u64>) -> Identity {
    let then = epoch.and_then(|epoch| forgotten().get(&handle).filter(|f| epoch <= f.at).cloned());
    if let Some(Forgotten { taken, context, .. }) = then {
        return Identity {
            owner: Some(taken.owner),
            name: Some(taken.name.to_string()),
            handle: taken.handle,
            context,
            gone: true,
        };
    }
    // (The record copied out and the lock let go before `contexts` is asked: see "Locks" above.)
    let known = taken_map().get(&handle).cloned();
    if let Some(known) = known {
        let context = known
            .module
            .and_then(crate::contexts::context_of)
            .or_else(|| crate::contexts::context_of(known.owner));
        return Identity {
            owner: Some(known.owner),
            name: Some(known.name.to_string()),
            handle: known.handle,
            context: context.map(|c| c as u64),
            gone: false,
        };
    }
    let unknown = Identity {
        handle,
        ..Identity::default()
    };
    type GetHandle = unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult;
    type GetName = unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> CUresult;
    // SAFETY: NVIDIA's functions of these C types, or stubs; the handle is one the program launched.
    let get_library: GetHandle = unsafe { std::mem::transmute(REALS[idx::cuKernelGetLibrary].load(Relaxed)) };
    let mut library = std::ptr::null_mut();
    if unsafe { get_library(&mut library, handle as *mut c_void) } != CUDA_SUCCESS || library.is_null() {
        return unknown;
    }
    let owner = Some(crate::lazy::kernel_library(handle).unwrap_or(library as usize));
    let get_name: GetName = unsafe { std::mem::transmute(REALS[idx::cuKernelGetName].load(Relaxed)) };
    let mut name: *const c_char = std::ptr::null();
    if unsafe { get_name(&mut name, handle as *mut c_void) } != CUDA_SUCCESS || name.is_null() {
        return Identity { owner, ..unknown };
    }
    Identity {
        owner,
        name: Some(unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()),
        ..unknown
    }
}

// ---- graphs --------------------------------------------------------------------------------------

/// The number of captures under way (begun and not ended, as far as the layer saw): while there is
/// none, a launch does not ask whether its stream is captured.
static CAPTURING: AtomicUsize = AtomicUsize::new(0);

/// A kernel recorded in a graph.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    function: u64,
    grid: [u32; 3],
    block: [u32; 3],
}

/// The kernel nodes recorded in a graph: the first `GRAPH_KERNELS` distinct kernels, and how many
/// nodes in all.
#[derive(Clone, Default, Debug)]
struct GraphKernels {
    kernels: Vec<Node>,
    nodes: usize,
}

const GRAPH_KERNELS: usize = 64;
/// The most graphs and instantiated graphs kept (a program that never destroys them does not make
/// the layer grow without end).
const GRAPHS_KEPT: usize = 4096;

static GRAPHS: LazyLock<Mutex<HashMap<usize, GraphKernels>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static EXECS: LazyLock<Mutex<HashMap<usize, Arc<GraphKernels>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn graphs() -> MutexGuard<'static, HashMap<usize, GraphKernels>> {
    GRAPHS.lock().unwrap_or_else(|e| e.into_inner())
}

fn execs() -> MutexGuard<'static, HashMap<usize, Arc<GraphKernels>>> {
    EXECS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The graph that `stream` is being captured into, if it is.
#[cold]
#[inline(never)]
unsafe fn capturing_into(stream: *mut c_void, per_thread: bool) -> Option<usize> {
    // (CU_STREAM_CAPTURE_STATUS_ACTIVE.)
    match unsafe { capture_status(stream, per_thread) } {
        (1, graph) if graph != 0 => Some(graph),
        _ => None,
    }
}

/// What the driver says of the capture of `stream`: its status (0: none, 1: active, 2:
/// invalidated) and the graph being captured into.
unsafe fn capture_status(stream: *mut c_void, per_thread: bool) -> (c_int, usize) {
    let index = if per_thread {
        idx::cuStreamGetCaptureInfo_v2_ptsz
    } else {
        idx::cuStreamGetCaptureInfo_v2
    };
    type Info = unsafe extern "C" fn(
        *mut c_void,
        *mut c_int,
        *mut u64,
        *mut *mut c_void,
        *mut *const c_void,
        *mut usize,
    ) -> CUresult;
    // SAFETY: NVIDIA's cuStreamGetCaptureInfo_v2 (or its form for the thread's stream), or a stub.
    let info: Info = unsafe { std::mem::transmute(REALS[index].load(Relaxed)) };
    let (mut status, mut id, mut graph) = (0, 0u64, std::ptr::null_mut());
    let (mut dependencies, mut count) = (std::ptr::null(), 0usize);
    match unsafe { info(stream, &mut status, &mut id, &mut graph, &mut dependencies, &mut count) } {
        CUDA_SUCCESS => (status, graph as usize),
        _ => (0, 0),
    }
}

/// A capture began (`result` of the driver's call).
pub fn capture_began(result: CUresult) {
    if result == CUDA_SUCCESS {
        CAPTURING.fetch_add(1, Relaxed);
    }
}

/// Whether `stream` is being captured, asked before the call that ends its capture.
///
/// # Safety
/// `stream` is the stream the program names.
pub unsafe fn capture_ending(stream: *mut c_void, per_thread: bool) -> bool {
    // (Active or invalidated: either way, the call ends it.)
    CAPTURING.load(Relaxed) != 0 && unsafe { capture_status(stream, per_thread) }.0 != 0
}

/// A capture that was under way ended (`was`: `capture_ending` said so).
pub fn capture_ended(was: bool) {
    if was {
        let mut count = CAPTURING.load(Relaxed);
        while count > 0 {
            match CAPTURING.compare_exchange_weak(count, count - 1, Relaxed, Relaxed) {
                Ok(_) => break,
                Err(now) => count = now,
            }
        }
    }
}

/// A kernel node of `function` was added to `graph`.
pub fn node_added(graph: usize, function: u64, grid: [u32; 3], block: [u32; 3]) {
    if graph == 0 || function == 0 {
        return;
    }
    let mut graphs = graphs();
    if !graphs.contains_key(&graph) && graphs.len() >= GRAPHS_KEPT {
        return;
    }
    let kernels = graphs.entry(graph).or_default();
    kernels.nodes += 1;
    if kernels.kernels.len() < GRAPH_KERNELS && !kernels.kernels.iter().any(|k| k.function == function) {
        kernels.kernels.push(Node { function, grid, block });
    }
}

/// `exec` was instantiated from `graph`: it has what the graph has now.
pub fn instantiated(result: CUresult, exec: *mut *mut c_void, graph: *mut c_void) {
    if result != CUDA_SUCCESS || exec.is_null() {
        return;
    }
    let exec = unsafe { *exec } as usize;
    let kernels = graphs().get(&(graph as usize)).cloned();
    let mut execs = execs();
    match kernels {
        Some(kernels) if execs.len() < GRAPHS_KEPT || execs.contains_key(&exec) => {
            execs.insert(exec, Arc::new(kernels));
        }
        _ => {
            execs.remove(&exec);
        }
    }
}

/// `graph` is destroyed (`cuGraphDestroy`).
pub fn graph_destroyed(graph: *mut c_void) {
    graphs().remove(&(graph as usize));
}

/// `exec` is destroyed (`cuGraphExecDestroy`).
pub fn exec_destroyed(exec: *mut c_void) {
    execs().remove(&(exec as usize));
}

/// The driver answered `result` to the launch of the instantiated graph `exec` on `stream`: what
/// the program gets (in exact mode, what the synchronisation that follows answers when it faults).
///
/// # Safety
/// `stream` is the stream the program launched on.
pub unsafe fn graph_launched(result: CUresult, exec: *mut c_void, stream: *mut c_void, entry: Entry) -> CUresult {
    if result != CUDA_SUCCESS {
        check(result, entry.name());
        return result;
    }
    // (A graph launched into a capture runs nothing now.)
    if CAPTURING.load(Relaxed) != 0 && unsafe { capturing_into(stream, entry.per_thread()) }.is_some() {
        return result;
    }
    let kept = keep(
        GRAPH,
        exec as u64,
        [0; 3],
        [0; 3],
        stream_id(stream, entry.per_thread()),
        entry,
        false,
    );
    if config().fault_trace {
        return unsafe { settle(kept, stream, entry.per_thread(), entry.name()) };
    }
    result
}

// ---- the candidates ------------------------------------------------------------------------------

/// What the layer found of a fault in a context.
#[derive(Debug, Default, PartialEq, Eq)]
struct Found {
    /// The launches that may have faulted, the last first.
    launches: Vec<Launch>,
    /// Of them, those whose context the layer does not know (a library's kernel launched by its
    /// handle, a kernel the program did not take through the layer), in a process where it knows
    /// several contexts: they may be of another.
    elsewhere: usize,
    /// Launches no longer kept may be candidates too: a ring wrapped since the last synchronisation
    /// of the whole context.
    earlier: bool,
    /// Threads that launched kernels and kept none.
    unkept: u64,
}

/// The candidates among `launches` (of every ring, in any order) for a fault in `context`: those
/// that no synchronisation covers and that are not known to be of another context (`context_of`)
/// nor waited for in exact mode (`done`: by ring, the launches numbered below were). `wrapped`: the
/// rings that no longer hold their earliest launches.
fn candidates(
    launches: &[Launch],
    covers: &Covers,
    done: &[u64],
    wrapped: &[usize],
    context: u64,
    context_of: impl Fn(&Launch) -> Option<u64>,
    contexts: usize,
) -> Found {
    let mut found = Found::default();
    for launch in launches {
        if done.get(launch.ring).is_some_and(|&done| launch.number < done) {
            continue;
        }
        let of = context_of(launch);
        if of.is_some_and(|of| of != context) || covers.covers(launch, context) {
            continue;
        }
        if of.is_none() && contexts > 1 {
            found.elsewhere += 1;
        }
        found.launches.push(*launch);
    }
    // The last first: by the synchronisations begun before each, then in each thread's order.
    found.launches.sort_by(|a, b| {
        (b.epoch.cmp(&a.epoch))
            .then(a.ring.cmp(&b.ring))
            .then(b.number.cmp(&a.number))
    });
    // The launches that a wrapped ring lost are older than its earliest kept: covered when that one
    // is, by a synchronisation of the whole context.
    let context_covers = covers.contexts.get(&context).copied();
    found.earlier = wrapped.iter().any(|&ring| {
        let earliest = launches
            .iter()
            .filter(|launch| launch.ring == ring)
            .map(|launch| launch.epoch)
            .min();
        earliest.is_none_or(|epoch| context_covers.is_none_or(|covers| epoch > covers))
    });
    found
}

// ---- the report ----------------------------------------------------------------------------------

/// The contexts whose fault was reported.
static REPORTED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// The lines of the report at exit: one for each fault reported (at most `SUMMARIES_KEPT`).
static SUMMARIES: Mutex<Vec<String>> = Mutex::new(Vec::new());
const SUMMARIES_KEPT: usize = 8;
/// The most candidates named one by one.
const NAMED: usize = 8;

fn reported() -> MutexGuard<'static, Vec<u64>> {
    REPORTED.lock().unwrap_or_else(|e| e.into_inner())
}

/// A call returned `result`, an error of a fault, in `context` (`None`: the current one).
#[cold]
#[inline(never)]
fn seen(result: CUresult, call: &str, context: Option<u64>) {
    let context = context.unwrap_or_else(current_context);
    {
        let mut reported = reported();
        if reported.contains(&context) {
            return;
        }
        reported.push(context);
    }
    // Whatever goes wrong in making the report, the program gets what the driver answered.
    let made = catch_unwind(AssertUnwindSafe(|| report(result, call, context)));
    let Ok((lines, summary)) = made else {
        log::error(format_args!(
            "a kernel faulted ({call} returned {result}); the layer could not say which (internal error)"
        ));
        return;
    };
    for line in &lines {
        log::error(format_args!("{line}"));
    }
    let mut summaries = SUMMARIES.lock().unwrap_or_else(|e| e.into_inner());
    if summaries.len() < SUMMARIES_KEPT {
        summaries.push(summary);
    }
}

/// The lines of the report at exit (`report_lines`): the faults seen.
pub fn summary_lines() -> Vec<String> {
    SUMMARIES.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn dimensions(d: [u32; 3]) -> String {
    format!("{}x{}x{}", d[0], d[1], d[2])
}

fn stream_text(stream: u64) -> String {
    match stream {
        LEGACY => "the null stream".to_string(),
        PER_THREAD => "the thread's default stream".to_string(),
        handle => format!("stream {handle:#x}"),
    }
}

/// What the layer knows of the kernel `function` launched in `epoch` (by its owner): its name, and
/// what was made of it.
fn kernel_text(function: u64, fitted: bool, epoch: u64) -> (String, String) {
    let Identity {
        owner,
        name,
        handle,
        gone,
        ..
    } = identify(function as usize, Some(epoch));
    let mut how = match (owner, &name) {
        _ if gone => "of a module unloaded since, or gone with its context: what the layer made of it is no longer \
                      known"
            .to_string(),
        (Some(owner), Some(name)) => crate::images::origin_text(owner, handle, name),
        (Some(owner), None) => crate::images::origin_text(owner, handle, ""),
        (None, _) => "of an image the layer knows nothing of".to_string(),
    };
    let name = name.unwrap_or_else(|| {
        format!("a kernel at {function:#x} that the program did not take by name through the layer")
    });
    if fitted {
        how += "; launched from its second translation, fitted within its own registers (VF_FIT=0 turns that off)";
    }
    (name, how)
}

/// The context of `launch` when the layer knows it: its kernel's (`identify`); of a graph, that of
/// the first of its kernels whose context is known.
fn launch_context(launch: &Launch) -> Option<u64> {
    if launch.kind == GRAPH {
        let kernels = execs().get(&(launch.what as usize)).cloned()?;
        return kernels
            .kernels
            .iter()
            .find_map(|node| identify(node.function as usize, Some(launch.epoch)).context);
    }
    identify(launch.what as usize, Some(launch.epoch)).context
}

/// The report of a fault: its lines, and its line at exit.
fn report(result: CUresult, call: &str, context: u64) -> (Vec<String>, String) {
    let rings: Vec<&'static Ring> = RINGS_HELD.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let launches: Vec<Launch> = rings.iter().flat_map(|ring| ring.launches()).collect();
    let done: Vec<u64> = rings.iter().map(|ring| ring.done.load(Acquire)).collect();
    let wrapped: Vec<usize> = rings
        .iter()
        .filter(|ring| ring.wrapped())
        .map(|ring| ring.index)
        .collect();
    let covers = record().clone();
    let contexts = crate::contexts::known().len();
    let mut found = candidates(&launches, &covers, &done, &wrapped, context, launch_context, contexts);
    found.unkept = UNKEPT.load(Relaxed);
    compose(&found, config().fault_trace, contexts, result, call)
}

/// The report of a fault that `call` met, from what was `found`, in exact mode or not, in a process
/// that the layer knows `contexts` contexts of.
fn compose(found: &Found, exact_mode: bool, contexts: usize, result: CUresult, call: &str) -> (Vec<String>, String) {
    let exact = exact_mode && found.launches.len() == 1;
    let mut lines = Vec::new();
    let error = format!("{result} ({})", error_name(result));
    let mut names = Vec::new();
    let mut details = Vec::new();
    // The threads that made the launches (named when there are several).
    let mut threads: Vec<u64> = found.launches.iter().map(|launch| launch.thread).collect();
    threads.sort_unstable();
    threads.dedup();
    for launch in found.launches.iter().take(NAMED) {
        let thread = if threads.len() > 1 {
            format!(", thread {}", launch.thread)
        } else {
            String::new()
        };
        if launch.kind == GRAPH {
            let kernels = execs().get(&(launch.what as usize)).cloned();
            let Some(kernels) = kernels else {
                names.push("a CUDA graph".to_string());
                details.push(format!(
                    "  a CUDA graph ({}, {}{thread}) that the layer did not see made: its kernels ran without passing \
                     through the layer",
                    launch.entry.name(),
                    stream_text(launch.stream)
                ));
                continue;
            };
            names.push(format!("a CUDA graph of {} kernel nodes", kernels.nodes));
            details.push(format!(
                "  a CUDA graph ({}, {}{thread}): its kernels ran without passing through the layer; it was made of \
                 {} kernel nodes, of {} kernels{}:",
                launch.entry.name(),
                stream_text(launch.stream),
                kernels.nodes,
                kernels.kernels.len(),
                if kernels.kernels.len() >= GRAPH_KERNELS {
                    " at least"
                } else {
                    ""
                }
            ));
            for node in kernels.kernels.iter().take(NAMED) {
                let (name, how) = kernel_text(node.function, false, launch.epoch);
                details.push(format!(
                    "    {name} (grid {}, block {}): {how}",
                    dimensions(node.grid),
                    dimensions(node.block)
                ));
            }
            if kernels.kernels.len() > NAMED {
                details.push(format!("    and {} kernels more", kernels.kernels.len() - NAMED));
            }
            continue;
        }
        let (name, how) = kernel_text(launch.what, launch.fitted, launch.epoch);
        details.push(format!(
            "  {name} ({}, grid {}, block {}, {}{thread}): {how}",
            launch.entry.name(),
            dimensions(launch.grid),
            dimensions(launch.block),
            stream_text(launch.stream)
        ));
        names.push(name);
    }
    let more = found.launches.len().saturating_sub(NAMED);
    if more > 0 {
        details.push(format!("  and {more} launches before them"));
    }
    let mut also = String::new();
    if found.elsewhere > 0 {
        also += &format!(
            " ({} of them of a context that the layer does not know: they may be of another of the {contexts} \
             contexts it knows)",
            found.elsewhere
        );
    }
    if found.earlier {
        also += &format!(
            " (and maybe launches made before these, which the layer no longer keeps: it keeps the last {KEPT} \
             launches of each thread)"
        );
    }
    if found.unkept > 0 {
        also += &format!(
            " (and the launches of {} threads, which the layer could not keep: it keeps those of {RINGS} threads at \
             a time)",
            found.unkept
        );
    }
    match found.launches.len() {
        0 => lines.push(format!(
            "a kernel faulted on the GPU: {call} returned {error}, and this CUDA context can run nothing more. The \
             layer knows no launch in this context since its last synchronisation that succeeded{also}: the kernel \
             was launched where the layer does not see it (a CUDA graph it did not see made, a kernel launched by \
             a kernel, launches left to the driver by VF_ADAPT=0 with VF_FIT=0)"
        )),
        1 if exact && found.launches[0].kind == GRAPH => lines.push(format!(
            "a kernel of a CUDA graph faulted on the GPU: {call} returned {error}, and this CUDA context can run \
             nothing more (VF_FAULT_TRACE=1: every launch is synchronised right after it, and this graph's is the \
             one whose synchronisation met the fault; its kernels are named only as a whole, run without graphs to \
             name one):"
        )),
        1 if exact => lines.push(format!(
            "kernel {} faulted on the GPU: {call} returned {error}, and this CUDA context can run nothing more \
             (VF_FAULT_TRACE=1: every launch is synchronised right after it, and this is the one whose \
             synchronisation met the fault):",
            names[0]
        )),
        count => lines.push(format!(
            "a kernel faulted on the GPU: {call} returned {error}, and this CUDA context can run nothing more. \
             Faults are reported at a later call than the launch: the kernel is {} launched in this context since \
             its last synchronisation that succeeded{also}, the last first:",
            if count == 1 {
                "the one".to_string()
            } else {
                format!("one of the {count}")
            }
        )),
    }
    lines.extend(details);
    if !exact {
        lines.push(
            "to name the kernel alone: VF_FAULT_TRACE=1 synchronises every launch right after it (each launch \
             then waits for its kernel; a CUDA graph is named as a whole); VF_LOG_LEVEL=2 names each image when \
             it is loaded (its size and XXH64)"
                .to_string(),
        );
    } else {
        lines.push("VF_LOG_LEVEL=2 names each image when it is loaded (its size and XXH64)".to_string());
    }
    let summary = format!(
        "a kernel faulted: {call} returned {error}; {}: {}",
        match (exact, found.launches.first()) {
            (true, Some(launch)) if launch.kind == GRAPH => "the graph",
            (true, _) => "the kernel",
            _ => "the candidates, the last launched first",
        },
        if names.is_empty() {
            "none known".to_string()
        } else if more > 0 {
            format!("{}, and {more} more", names.join(", "))
        } else {
            names.join(", ")
        }
    );
    (lines, summary)
}

// ---- before a context goes, and at exit ----------------------------------------------------------

/// Ask the driver whether `context` faulted, without waiting: a query of its null stream, which
/// returns the error of a fault when there was one (`cuStreamQuery` returned 700 after a fault on a
/// V100); reported if so. When the layer saw launches and the fault of the context, if it had one,
/// was not reported: a fault that only an unrouted call (a copy) met is found here.
fn probe(context: u64, when: &str) {
    if context == 0 || !LAUNCHED.load(Relaxed) || reported().contains(&context) {
        return;
    }
    type Push = unsafe extern "C" fn(*mut c_void) -> CUresult;
    type Pop = unsafe extern "C" fn(*mut *mut c_void) -> CUresult;
    type Query = unsafe extern "C" fn(*mut c_void) -> CUresult;
    // SAFETY: NVIDIA's functions of these C types, or stubs; the context is one the program used and
    // has not destroyed.
    let push: Push = unsafe { std::mem::transmute(REALS[idx::cuCtxPushCurrent_v2].load(Relaxed)) };
    let pop: Pop = unsafe { std::mem::transmute(REALS[idx::cuCtxPopCurrent_v2].load(Relaxed)) };
    let query: Query = unsafe { std::mem::transmute(REALS[idx::cuStreamQuery].load(Relaxed)) };
    let pushed = unsafe { push(context as *mut c_void) };
    if pushed != CUDA_SUCCESS {
        log::trace(format_args!(
            "fault: context {context:#x} not asked {when}: cuCtxPushCurrent returned {pushed}"
        ));
        return;
    }
    let result = unsafe { query(std::ptr::null_mut()) };
    let mut popped = std::ptr::null_mut();
    unsafe { pop(&mut popped) };
    log::trace(format_args!(
        "fault: context {context:#x} asked {when}: cuStreamQuery returned {result}"
    ));
    if is_fault(result) {
        seen(
            result,
            &format!("cuStreamQuery, asked by the layer {when}"),
            Some(context),
        );
    }
}

/// `context` is to be destroyed (or the primary context released): asked whether it faulted.
pub fn context_ending(context: usize) {
    probe(context as u64, "before the context was destroyed");
}

/// `context` was destroyed: a context given its handle later is another, whose fault is its own,
/// and whose launches are those made after.
pub fn context_gone(context: usize) {
    let context = context as u64;
    reported().retain(|&c| c != context);
    let epoch = begin();
    record().context_gone(context, epoch);
}

/// The `known` contexts (`contexts::known`: of the modules loaded, and the primary contexts held),
/// and of those that synchronisations were made in, the ones not destroyed since.
fn live_contexts(covers: &Covers, known: &[u64]) -> Vec<u64> {
    let synchronized = covers
        .contexts
        .keys()
        .chain(covers.legacy.keys())
        .chain(covers.own.keys().map(|(_, context)| context))
        .filter(|context| !covers.gone.contains(context));
    let mut live: Vec<u64> = known
        .iter()
        .chain(synchronized)
        .copied()
        .filter(|&context| context != 0)
        .collect();
    live.sort_unstable();
    live.dedup();
    live
}

/// At exit: the contexts that the layer knows, not destroyed, whose fault was not reported, are
/// asked whether they faulted (when the layer saw launches).
pub fn at_exit() {
    if !LAUNCHED.load(Relaxed) {
        return;
    }
    let known: Vec<u64> = crate::contexts::known().into_iter().map(|c| c as u64).collect();
    let live = live_contexts(&record().clone(), &known);
    for context in live {
        probe(context, "at exit");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch(ring: usize, number: u64, stream: u64, what: u64, epoch: u64) -> Launch {
        Launch {
            ring,
            number,
            thread: ring as u64 + 1,
            kind: LAUNCH,
            entry: Entry::LaunchKernel,
            fitted: false,
            what,
            stream,
            grid: [70_000, 3, 65_535],
            block: [1024, 1, 64],
            epoch,
        }
    }

    /// The launches `whats` on `stream` in ring 0, all in `epoch`, numbered from `first`.
    fn launches(first: u64, stream: u64, whats: &[u64], epoch: u64) -> Vec<Launch> {
        whats
            .iter()
            .enumerate()
            .map(|(i, &what)| launch(0, first + i as u64, stream, what, epoch))
            .collect()
    }

    fn whats(found: &Found) -> Vec<u64> {
        found.launches.iter().map(|l| l.what).collect()
    }

    fn found_in(launches: &[Launch], covers: &Covers, context: u64) -> Found {
        candidates(launches, covers, &[], &[], context, |_| None, 1)
    }

    #[test]
    fn a_launch_is_read_back_as_it_was_written() {
        let ring = Box::new(Ring::new(3));
        ring.thread.store(5, Relaxed);
        let mut written = launch(3, 0, 0x7f00_1234, 0xdead_beef, 0xab_cdef);
        written.entry = Entry::LaunchKernelExPtsz;
        written.fitted = true;
        written.kind = GRAPH;
        let number = ring.push(&mut written);
        assert_eq!(number, 0);
        let read = ring.launches();
        assert_eq!(read, vec![Launch { thread: 5, ..written }]);
        // A slot being written is not read.
        let slot = &ring.slots[0];
        slot.version.fetch_add(1, Relaxed);
        assert!(ring.launches().is_empty());
        slot.version.fetch_sub(1, Relaxed);
        assert_eq!(ring.launches().len(), 1);
    }

    #[test]
    fn a_ring_keeps_the_last_launches_of_its_thread() {
        let ring = Box::new(Ring::new(0));
        for n in 0..(KEPT as u64 + 10) {
            ring.push(&mut launch(0, 0, LEGACY, n, 0));
        }
        let mut read = ring.launches();
        read.sort_by_key(|l| l.number);
        assert_eq!(read.len(), KEPT);
        assert_eq!(read[0].what, 10);
        assert_eq!(read[0].number, 10);
        assert_eq!(read[KEPT - 1].what, KEPT as u64 + 9);
        assert!(ring.wrapped());
    }

    #[test]
    fn the_oldest_launch_kept_is_not_one_written_over_it() {
        let ring = Box::new(Ring::new(0));
        for n in 0..KEPT as u64 {
            ring.push(&mut launch(0, 0, LEGACY, n, n));
        }
        assert_eq!(oldest_of(&[&ring]), Some(0));
        // The next launch written into the earliest's slot, its number not yet counted (as while
        // its thread writes it, read from another thread): the earliest kept is now launch 1.
        ring.slots[0].write(KEPT as u64, &launch(0, 0, LEGACY, 999, 1000));
        assert_eq!(oldest_of(&[&ring]), Some(1));
    }

    #[test]
    fn a_thread_takes_a_ring_and_gives_it_back_when_it_ends() {
        let first = std::thread::spawn(|| mine().map(|ring| ring.index)).join().unwrap();
        let second = std::thread::spawn(|| mine().map(|ring| ring.index)).join().unwrap();
        // (Other tests take rings too: the second thread takes a ring that one of them gave back.)
        assert!(first.is_some() && second.is_some());
        let rings = RINGS_HELD.lock().unwrap().len();
        assert!(rings <= RINGS);
        // A thread holds its ring until it ends.
        let held = std::thread::spawn(|| {
            let ring = mine().unwrap();
            ring.held.load(Relaxed) && std::ptr::eq(mine().unwrap(), ring)
        })
        .join()
        .unwrap();
        assert!(held);
    }

    #[test]
    fn a_synchronisation_covers_the_launches_kept_before_it_began() {
        let mut covers = Covers::default();
        // Epoch 0: two launches; a synchronisation of context 7 begins (covers 0), and another
        // thread launches meanwhile (it finds epoch 1).
        let mut all = launches(0, LEGACY, &[10, 11], 0);
        all.push(launch(1, 0, LEGACY, 12, 1));
        covers.context_synchronized(7, 0);
        all.extend(launches(2, LEGACY, &[13], 1));
        // A synchronisation of another context covers nothing of this one.
        covers.context_synchronized(8, 1);
        all.extend(launches(3, LEGACY, &[99], 2));
        let found = found_in(&all, &covers, 7);
        assert_eq!(whats(&found), vec![99, 13, 12]);
        let found = found_in(&all, &covers, 8);
        assert_eq!(whats(&found), vec![99]);
        // A context that was not synchronised: every launch.
        assert_eq!(found_in(&all, &covers, 9).launches.len(), 5);
    }

    #[test]
    fn a_launch_of_another_context_is_not_a_candidate() {
        let covers = Covers::default();
        let all = launches(0, 0x5000, &[1, 2, 3], 0);
        // Launch 2 is known to be of context 8, launch 3 of no known context.
        let context_of = |l: &Launch| match l.what {
            1 => Some(7),
            2 => Some(8),
            _ => None,
        };
        let found = candidates(&all, &covers, &[], &[], 7, context_of, 2);
        assert_eq!(whats(&found), vec![3, 1]);
        assert_eq!(found.elsewhere, 1);
        // In a process of one context, a launch of no known context is not said to be elsewhere.
        let found = candidates(&all, &covers, &[], &[], 7, context_of, 1);
        assert_eq!(found.elsewhere, 0);
    }

    #[test]
    fn a_stream_synchronisation_covers_its_stream_only() {
        let mut covers = Covers::default();
        let mut all = launches(0, 0xa000, &[1], 0);
        all.extend(launches(1, 0xb000, &[2], 0));
        all.extend(launches(2, LEGACY, &[3], 0));
        all.extend(launches(3, PER_THREAD, &[4], 0));
        covers.stream_synchronized(0xa000, 0);
        covers.legacy_synchronized(7, 0);
        let found = found_in(&all, &covers, 7);
        assert_eq!(whats(&found), vec![4, 2]);
        // Context 7's legacy stream is not context 8's.
        assert_eq!(whats(&found_in(&all, &covers, 8)), vec![4, 3, 2]);
        // The thread's own stream, synchronised in context 7, by this thread (thread 1, of ring 0).
        covers.own_synchronized(1, 7, 0);
        assert_eq!(whats(&found_in(&all, &covers, 7)), vec![2]);
        covers.own_synchronized(2, 8, 0);
        assert_eq!(whats(&found_in(&all, &covers, 8)), vec![4, 3, 2]);
        // Another thread given the same ring later: its own stream is not this thread's.
        let mut later = launch(0, 1, PER_THREAD, 5, 0);
        later.thread = 3;
        assert_eq!(whats(&found_in(&[later], &covers, 7)), vec![5]);
    }

    #[test]
    fn a_stream_made_at_the_handle_of_one_destroyed_is_another() {
        let mut covers = Covers::default();
        // Launches 1 and 2 on the stream at 0xa000, 1 before its synchronisation (covers 0).
        let mut all = launches(0, 0xa000, &[1], 0);
        covers.stream_synchronized(0xa000, 0);
        all.extend(launches(1, 0xa000, &[2], 1));
        // The stream destroyed when the epoch was 1; another made at its handle, launch 3,
        // synchronised (covers 2): launch 2 of the stream before stays a candidate.
        covers.stream_destroyed(0xa000, 1);
        all.extend(launches(2, 0xa000, &[3], 2));
        covers.stream_synchronized(0xa000, 2);
        assert_eq!(whats(&found_in(&all, &covers, 7)), vec![2]);
        // Records dropped when there are too many: what was launched before is no stream's.
        for stream in 0..COVERS_KEPT as u64 {
            covers.stream_synchronized(0x10_0000 + stream, 2);
        }
        assert!(covers.streams.len() <= COVERS_KEPT);
        covers.stream_synchronized(0xa000, 5);
        assert!(covers.streams_from > 0);
        assert!(whats(&found_in(&all, &covers, 7)).contains(&1));
    }

    #[test]
    fn a_context_destroyed_takes_its_launches_with_it() {
        let mut covers = Covers::default();
        let mut all = launches(0, LEGACY, &[1], 0);
        covers.context_gone(7, 0);
        // A new context given the same handle.
        assert!(found_in(&all, &covers, 7).launches.is_empty());
        assert_eq!(live_contexts(&covers, &[8]), vec![8]);
        // (One that the layer knows a module of is alive, as `contexts` says.)
        assert_eq!(live_contexts(&covers, &[7, 8]), vec![7, 8]);
        all.extend(launches(1, LEGACY, &[3], 1));
        assert_eq!(whats(&found_in(&all, &covers, 7)), vec![3]);
        // Synchronised since: alive.
        covers.legacy_synchronized(7, 1);
        assert_eq!(live_contexts(&covers, &[8]), vec![7, 8]);
    }

    #[test]
    fn exact_mode_leaves_the_launch_not_done() {
        let covers = Covers::default();
        let all = launches(0, PER_THREAD, &[1, 2, 3], 0);
        let found = candidates(&all, &covers, &[2], &[], 1, |_| None, 1);
        assert_eq!(whats(&found), vec![3]);
    }

    #[test]
    fn the_candidates_are_in_the_order_of_their_threads_and_synchronisations() {
        let covers = Covers::default();
        // Thread of ring 1: launch 2 in epoch 0, launch 4 in epoch 1; ring 0: 1 and 3 in epoch 0.
        let all = [
            launch(0, 0, LEGACY, 1, 0),
            launch(1, 0, 0x5000, 2, 0),
            launch(0, 1, LEGACY, 3, 0),
            launch(1, 1, 0x5000, 4, 1),
        ];
        let found = found_in(&all, &covers, 7);
        assert_eq!(whats(&found), vec![4, 3, 1, 2]);
    }

    #[test]
    fn a_wrapped_ring_may_have_lost_candidates() {
        let mut covers = Covers::default();
        let all = launches(300, LEGACY, &[1, 2], 4);
        assert!(candidates(&all, &covers, &[], &[0], 7, |_| None, 1).earlier);
        // What it lost is older than what a synchronisation of the whole context covered.
        covers.context_synchronized(7, 3);
        assert!(candidates(&all, &covers, &[], &[0], 7, |_| None, 1).earlier);
        covers.context_synchronized(7, 4);
        let found = candidates(&all, &covers, &[], &[0], 7, |_| None, 1);
        assert!(!found.earlier && found.launches.is_empty());
        assert!(!candidates(&all, &Covers::default(), &[], &[], 7, |_| None, 1).earlier);
    }

    #[test]
    fn the_errors_of_a_fault() {
        for code in [214, 220, 226, 700, 702, 710, 714, 715, 716, 717, 718, 719, 721, 911] {
            assert!(is_fault(code), "{code}");
            assert_ne!(error_name(code), "?");
        }
        // Not faults: success, out of resources (a launch refused), a refused kernel, not ready, an
        // invalid handle, a cooperative launch too large.
        for code in [0, 701, 218, 600, 400, 720, 1, 201] {
            assert!(!is_fault(code), "{code}");
        }
    }

    #[test]
    fn streams_are_told_apart() {
        let null = std::ptr::null_mut();
        assert_eq!(stream_id(null, false), LEGACY);
        assert_eq!(stream_id(null, true), PER_THREAD);
        assert_eq!(stream_id(std::ptr::without_provenance_mut(1), true), LEGACY);
        assert_eq!(stream_id(std::ptr::without_provenance_mut(2), false), PER_THREAD);
        assert_eq!(stream_id(std::ptr::without_provenance_mut(0x7f00), true), 0x7f00);
    }

    #[test]
    fn a_graph_keeps_its_first_distinct_kernels() {
        let graph = 0x5150_0000;
        for n in 0..(GRAPH_KERNELS as u64 + 20) {
            node_added(graph, 0x1000 + n % (GRAPH_KERNELS as u64 + 10), [1, 1, 1], [32, 1, 1]);
        }
        let kernels = graphs().get(&graph).cloned().unwrap();
        assert_eq!(kernels.nodes, GRAPH_KERNELS + 20);
        assert_eq!(kernels.kernels.len(), GRAPH_KERNELS);
        let mut exec = std::ptr::without_provenance_mut::<c_void>(0x6160_0000);
        instantiated(CUDA_SUCCESS, &mut exec, graph as *mut c_void);
        graph_destroyed(graph as *mut c_void);
        assert_eq!(execs().get(&0x6160_0000).unwrap().nodes, GRAPH_KERNELS + 20);
        exec_destroyed(exec);
        assert!(execs().get(&0x6160_0000).is_none());
    }

    #[test]
    fn the_report_names_the_candidates_the_last_first() {
        taken(0xa100, 0xa000, "add_one");
        taken(0xa200, 0xa000, "write_null");
        let mut covers = Covers::default();
        let mut all = launches(0, LEGACY, &[0xa100], 0);
        covers.context_synchronized(7, 0);
        all.extend(launches(1, LEGACY, &[0xa100, 0xa200, 0xa100], 1));
        let found = found_in(&all, &covers, 7);
        let (lines, summary) = compose(&found, false, 1, 700, "cuCtxSynchronize");
        assert!(
            lines[0]
                .starts_with("a kernel faulted on the GPU: cuCtxSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS)"),
            "{lines:?}"
        );
        assert!(lines[0].contains("one of the 3 launched in this context since its last synchronisation"));
        assert!(
            lines[1].starts_with("  add_one (cuLaunchKernel, grid 70000x3x65535, block 1024x1x64, the null stream): ")
        );
        assert!(lines[2].starts_with("  write_null ("), "{lines:?}");
        assert!(lines[3].starts_with("  add_one ("));
        assert!(lines[4].starts_with("to name the kernel alone: VF_FAULT_TRACE=1"));
        assert_eq!(
            summary,
            "a kernel faulted: cuCtxSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS); the candidates, the last \
             launched first: add_one, write_null, add_one"
        );
        // Exact mode: the launch not done is named alone.
        let all = launches(0, LEGACY, &[0xa100, 0xa200], 0);
        let found = candidates(&all, &Covers::default(), &[1], &[], 7, |_| None, 1);
        let (lines, summary) = compose(&found, true, 1, 719, "a synchronisation");
        assert!(lines[0].starts_with(
            "kernel write_null faulted on the GPU: a synchronisation returned 719 (CUDA_ERROR_LAUNCH_FAILED)"
        ));
        assert!(summary.ends_with("the kernel: write_null"), "{summary}");
        // Nothing known.
        let (lines, _) = compose(&Found::default(), false, 1, 716, "cuStreamQuery");
        assert!(
            lines[0].contains("knows no launch in this context since its last synchronisation"),
            "{lines:?}"
        );
        // A graph: its kernels as recorded.
        let graph = 0xb000_0000;
        node_added(graph, 0xa100, [1, 1, 1], [32, 1, 1]);
        node_added(graph, 0xa200, [2, 1, 1], [64, 1, 1]);
        let mut exec = std::ptr::without_provenance_mut::<c_void>(0xb100_0000);
        instantiated(CUDA_SUCCESS, &mut exec, std::ptr::without_provenance_mut(graph));
        let mut event = launch(0, 0, 0x5000, 0xb100_0000, 0);
        event.kind = GRAPH;
        event.entry = Entry::GraphLaunch;
        let found = found_in(&[event], &Covers::default(), 7);
        let (lines, summary) = compose(&found, false, 1, 700, "cuStreamSynchronize");
        assert!(
            lines[1].starts_with("  a CUDA graph (cuGraphLaunch, stream 0x5000): its kernels ran without passing"),
            "{lines:?}"
        );
        assert!(lines[1].contains("made of 2 kernel nodes, of 2 kernels:"));
        assert!(lines[2].starts_with("    add_one (grid 1x1x1, block 32x1x1): "));
        assert!(lines[3].starts_with("    write_null (grid 2x1x1, block 64x1x1): "));
        assert!(summary.ends_with("a CUDA graph of 2 kernel nodes"), "{summary}");
        // Launches of two threads, in a process with two contexts: said.
        let first = launch(0, 0, LEGACY, 0xa100, 0);
        let second = launch(1, 0, 0x9000, 0xa200, 0);
        let found = candidates(&[first, second], &Covers::default(), &[], &[], 7, |_| None, 2);
        let (lines, _) = compose(&found, false, 2, 700, "cuCtxSynchronize");
        assert!(
            lines[0].contains("(2 of them of a context that the layer does not know: they may be of another of the 2"),
            "{lines:?}"
        );
        // (Between two synchronisations, the launches of each thread in turn.)
        assert!(lines[1].starts_with(
            "  add_one (cuLaunchKernel, grid 70000x3x65535, block 1024x1x64, the null stream, thread 1): "
        ));
        assert!(lines[2].starts_with(
            "  write_null (cuLaunchKernel, grid 70000x3x65535, block 1024x1x64, stream 0x9000, thread 2): "
        ));
        // More than are named.
        let all = launches(0, LEGACY, &[0xa100; NAMED + 3], 0);
        let (lines, summary) = compose(
            &found_in(&all, &Covers::default(), 7),
            false,
            1,
            700,
            "cuCtxSynchronize",
        );
        assert!(lines.contains(&"  and 3 launches before them".to_string()), "{lines:?}");
        assert!(summary.ends_with(", and 3 more"), "{summary}");
        exec_destroyed(exec);
        graph_destroyed(std::ptr::without_provenance_mut(graph));
    }

    #[test]
    fn a_name_is_kept_until_its_owner_goes() {
        taken(0x9100, 0x9000, "k");
        alias(0x9200, 0x9100, Some(0x9500));
        taken(0x9300, 0x9001, "other");
        let known = identify(0x9200, None);
        assert_eq!(
            (known.owner, known.name.as_deref(), known.handle, known.gone),
            (Some(0x9000), Some("k"), 0x9100, false)
        );
        assert_eq!(taken_map().get(&0x9200).unwrap().module, Some(0x9500));
        // The module of the library in a context goes with that context: the function too (kept
        // apart, once a launch was kept).
        LAUNCHED.store(true, Relaxed);
        let before = EPOCH.load(SeqCst);
        forget_in(0x9500, Some(0x77));
        assert!(taken_map().get(&0x9200).is_none() && taken_map().get(&0x9100).is_some());
        forget(0x9000);
        assert!(taken_map().get(&0x9100).is_none());
        assert_eq!(taken_map().get(&0x9300).map(|t| t.owner), Some(0x9001));
        // A launch kept before is named as it was, in its context; the handle given to another
        // kernel since names that one for a launch made after.
        let then = identify(0x9200, Some(before));
        assert_eq!(
            (then.name.as_deref(), then.context, then.gone),
            (Some("k"), Some(0x77), true)
        );
        taken(0x9200, 0x9002, "new");
        let after = EPOCH.load(SeqCst);
        assert_eq!(identify(0x9200, Some(after)).name.as_deref(), Some("new"));
        assert_eq!(identify(0x9200, Some(before)).name.as_deref(), Some("k"));
    }
}

#[cfg(test)]
mod unload {
    use super::*;

    #[test]
    fn a_failed_unload_leaves_the_names_as_they_were() {
        LAUNCHED.store(true, Relaxed);
        // The handle 0xc200 was a kernel of a module gone before (kept apart), then given to "k".
        taken(0xc200, 0xc000, "old");
        let earlier = EPOCH.load(SeqCst);
        let _ = forget(0xc000);
        taken(0xc200, 0xc100, "k");
        let before = EPOCH.load(SeqCst);
        let forgot = forget(0xc100);
        assert!(identify(0xc200, Some(before)).gone);
        // The driver did not unload it: "k" is named as it is, and the launch before the first
        // module went is still of that one.
        unforget(forgot);
        let now = identify(0xc200, Some(before));
        assert_eq!((now.name.as_deref(), now.gone), (Some("k"), false));
        let then = identify(0xc200, Some(earlier));
        assert_eq!((then.name.as_deref(), then.gone), (Some("old"), true));
    }
}

#[cfg(test)]
mod cost {
    use super::*;

    /// What keeping a launch costs, alone (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn cost_of_a_launch_kept() {
        let n = 20_000_000u64;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            for i in 0..n {
                std::hint::black_box(keep(
                    LAUNCH,
                    i,
                    [1, 1, 1],
                    [32, 1, 1],
                    LEGACY,
                    Entry::LaunchKernel,
                    false,
                ));
            }
            let t = start.elapsed().as_secs_f64() * 1e9 / n as f64;
            let start = std::time::Instant::now();
            for _ in 0..n {
                std::hint::black_box(mine());
            }
            let m = start.elapsed().as_secs_f64() * 1e9 / n as f64;
            eprintln!("keep {t:.2} ns, mine {m:.2} ns");
        }
    }
}
