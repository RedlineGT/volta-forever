//! The last defence against a refused kernel: a call that would launch it, or use it, is refused.
//!
//! A kernel that cannot run on this GPU is refused when the program asks for it by name, and left
//! out of the enumerations (`kernels`): the program gets no handle to it from the layer. Its
//! function is still in the module that the driver loaded, made to trap, and a program can have a
//! handle to it all the same: from a private interface of the driver (cuFFT enumerates the
//! functions of each module it loads through an entry of a table of `cuGetExportTable`, without a
//! lookup), or from an enumeration made before the driver's compiler refused something of a
//! deferred library. Launched,
//! it traps, and the GPU ends the context. So every call that launches a function, or records one
//! in a graph, or asks about it, is refused (`REFUSED`, the error of the lookup) when the handle is
//! that of a refused kernel, and the driver is not asked.
//!
//! What the handle is, the driver says (`kernels::verdict`): its module or library and its
//! name, which lead to the layer's records. That takes calls of the driver and a lock: it is done
//! once for each handle, and what was found allowed is kept in a table without a lock (`ALLOWED`)
//! for the generation it was found in (`kernels::generation`). A generation ends whenever what is
//! known of refusals changes and whenever a module, a library or a context goes, after which the
//! driver may give a handle to another function. While no module or library known to the layer
//! has a refused kernel, nothing is asked: a launch then costs one more load of a counter.
//!
//! A kernel without code (its code in the image is a trap and nothing else, `vf_adapt::Stubs`) is
//! given at a lookup, as NVIDIA's driver gives it, and may be asked about (`allows`): what the
//! driver answers of it is true. A call that would run it (`allows_run`: a launch, a kernel node of
//! a graph) is refused, with a line that names it and its image: launched, its trap would end the
//! context, on this GPU as on any other.

use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering::Acquire, Ordering::Relaxed, Ordering::Release, fence};

use crate::kernels;

/// What a call with a handle to a refused kernel returns: what the lookup of that kernel by name
/// returns (`CUDA_ERROR_INVALID_PTX`), for the same cause; not `CUDA_ERROR_INVALID_HANDLE` (400),
/// NVIDIA's answer to a handle that is no function, which would tell the program that its handle
/// is wrong.
pub const REFUSED: c_int = crate::images::CUDA_ERROR_INVALID_PTX;

/// A handle found allowed and the generation it was found in, written by one thread at a time and
/// read without a lock: `version` is odd while a thread writes, and changes with each write, so that
/// a reader that sees it the same, and even, before and after reading the two words read both of one
/// write. The whole generation is kept: an entry is never taken for one of another generation, however
/// many generations go by.
struct Slot {
    version: AtomicU64,
    handle: AtomicU64,
    generation: AtomicU64,
}

impl Slot {
    const fn empty() -> Slot {
        Slot {
            version: AtomicU64::new(0),
            handle: AtomicU64::new(0),
            generation: AtomicU64::new(0),
        }
    }

    /// Whether the slot holds `handle` found allowed in `generation`.
    #[inline]
    fn holds(&self, handle: u64, generation: u64) -> bool {
        let version = self.version.load(Acquire);
        if version & 1 != 0 || self.handle.load(Relaxed) != handle {
            return false;
        }
        let found = self.generation.load(Relaxed);
        fence(Acquire);
        found == generation && self.version.load(Relaxed) == version
    }

    /// Keep `handle` found allowed in `generation`, unless another thread writes the slot now.
    fn keep(&self, handle: u64, generation: u64) {
        let version = self.version.load(Relaxed);
        if version & 1 != 0
            || self
                .version
                .compare_exchange(version, version + 1, Relaxed, Relaxed)
                .is_err()
        {
            return;
        }
        fence(Release);
        self.handle.store(handle, Relaxed);
        self.generation.store(generation, Relaxed);
        self.version.store(version + 2, Release);
    }
}

/// Four slots for the handles of one hash, in two cache lines, and which of the four a fifth
/// replaces.
#[repr(align(128))]
struct Set {
    ways: [Slot; WAYS],
    next: AtomicU64,
}

const WAYS: usize = 4;

impl Set {
    const fn empty() -> Set {
        Set {
            ways: [const { Slot::empty() }; WAYS],
            next: AtomicU64::new(0),
        }
    }

    #[inline]
    fn holds(&self, handle: u64, generation: u64) -> bool {
        self.ways.iter().any(|way| way.holds(handle, generation))
    }

    /// Keep `handle` found allowed in `generation`: in the slot that has the handle, else in one of
    /// another generation, else in each slot in turn.
    fn keep(&self, handle: u64, generation: u64) {
        let way = (0..WAYS)
            .find(|&way| self.ways[way].handle.load(Relaxed) == handle)
            .or_else(|| (0..WAYS).find(|&way| self.ways[way].generation.load(Relaxed) != generation))
            .unwrap_or_else(|| self.next.fetch_add(1, Relaxed) as usize % WAYS);
        self.ways[way].keep(handle, generation);
    }
}

/// Handles found allowed, by a hash of the handle: 4,096 sets of four (512 KB, of which only the pages
/// used are touched). A handle whose set holds four others of this generation is asked about
/// again. (With 4,096 slots of one, 743 of 2,000 kernels launched in turn were asked about again at
/// every turn on a V100.)
const SETS: usize = 4096;
static ALLOWED: [Set; SETS] = [const { Set::empty() }; SETS];

fn set(handle: usize) -> &'static Set {
    // (Fibonacci hashing: the high bits of the product depend on every bit of the handle.)
    &ALLOWED[((handle as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - SETS.trailing_zeros())) as usize]
}

/// What the refusal did, for `VF_STATS`: the times it asked the driver what a handle is, and the
/// calls it refused.
static QUESTIONS: AtomicU64 = AtomicU64::new(0);
static REFUSALS: AtomicU64 = AtomicU64::new(0);

/// The line of `VF_STATS` about the refusal at a use.
pub fn summary() -> String {
    format!(
        "guard: the driver was asked what a handle is {} times, {} calls were refused, {} generations",
        QUESTIONS.load(Relaxed),
        REFUSALS.load(Relaxed),
        kernels::generation()
    )
}

/// Whether `call` (the name of the driver's function, for the log) may be made with `handle`, a
/// function or a kernel, in the current context, when the call asks about it or sets something of
/// it: not when it is one of a refused kernel, which is then said once.
#[inline]
pub fn allows(handle: *mut c_void, call: &str) -> bool {
    allows_where(handle, call, None, false)
}

/// `allows`, for a call about a library's kernel on `device` (`cuKernelGetAttribute` and the like).
#[inline]
pub fn allows_on(handle: *mut c_void, call: &str, device: c_int) -> bool {
    allows_where(handle, call, Some(device), false)
}

/// `allows`, for a call that would run the kernel (a launch, a kernel node of a graph): not either
/// when it is a kernel without code, which is then said once.
#[inline]
pub fn allows_run(handle: *mut c_void, call: &str) -> bool {
    allows_where(handle, call, None, true)
}

/// `allows_run`, for a kernel node of a graph that names a library's kernel and the context of
/// `device`.
#[inline]
pub fn allows_run_on(handle: *mut c_void, call: &str, device: c_int) -> bool {
    allows_where(handle, call, Some(device), true)
}

#[inline]
fn allows_where(handle: *mut c_void, call: &str, device: Option<c_int>, run: bool) -> bool {
    if !kernels::refusing() {
        return true;
    }
    let handle = handle as usize;
    let generation = kernels::generation();
    if handle == 0 || set(handle).holds(handle as u64, generation) {
        return true;
    }
    decide(handle, generation, call, device, run)
}

/// `allows`, for a handle not found allowed in this generation: what the driver says it is.
#[cold]
#[inline(never)]
fn decide(handle: usize, generation: u64, call: &str, device: Option<c_int>, run: bool) -> bool {
    QUESTIONS.fetch_add(1, Relaxed);
    // SAFETY: the program passes the handle to a call of the driver, which accepts it or refuses
    // it; the layer asks the driver about it the same way.
    match unsafe { kernels::verdict(handle, device) } {
        kernels::Verdict::Refused(refusal) => {
            REFUSALS.fetch_add(1, Relaxed);
            refusal.explain(call);
            false
        }
        // (Not kept as allowed: a launch of the same handle must be refused.)
        kernels::Verdict::NoCode(_) if !run => true,
        kernels::Verdict::NoCode(no_code) => {
            REFUSALS.fetch_add(1, Relaxed);
            no_code.explain(call);
            false
        }
        kernels::Verdict::Allowed => {
            // (Of the generation read before the records were: if it has ended since, the entry
            // is never found. Not kept while a module, a library or a context is going: the
            // handle may already be another function's, `kernels::Ending`.)
            if !kernels::ending_now() {
                set(handle).keep(handle as u64, generation);
            }
            true
        }
        // (Not kept: the same handle may be used on this GPU next.)
        kernels::Verdict::AllowedHere => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_holds_a_handle_for_its_generation_only() {
        let slot = Slot::empty();
        assert!(!slot.holds(0x7f12_3456_7890, 0));
        slot.keep(0x7f12_3456_7890, 3);
        assert!(slot.holds(0x7f12_3456_7890, 3));
        assert!(!slot.holds(0x7f12_3456_7890, 4));
        // However many generations go by (no bits of it are dropped).
        assert!(!slot.holds(0x7f12_3456_7890, 3 + (1 << 16)));
        assert!(!slot.holds(0x7f12_3456_7891, 3));
        slot.keep(0x5555_0000_1230, 9);
        assert!(slot.holds(0x5555_0000_1230, 9) && !slot.holds(0x7f12_3456_7890, 3));
    }

    #[test]
    fn a_slot_being_written_holds_nothing() {
        let slot = Slot::empty();
        slot.keep(0x1230, 1);
        slot.version.fetch_add(1, Relaxed);
        assert!(!slot.holds(0x1230, 1));
        // (Nor is it written by another thread meanwhile.)
        slot.keep(0x4560, 2);
        assert!(slot.handle.load(Relaxed) == 0x1230);
    }
}
