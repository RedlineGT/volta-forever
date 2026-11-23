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

/// Handles found allowed, by a hash of the handle. A handle that collides with another is asked
/// about again.
const SLOTS: usize = 4096;
static ALLOWED: [Slot; SLOTS] = [const { Slot::empty() }; SLOTS];

fn slot(handle: usize) -> &'static Slot {
    &ALLOWED[((handle >> 4) ^ (handle >> 16)) % SLOTS]
}

/// Whether `call` (the name of the driver's function, for the log) may be made with `handle`, a
/// function or a kernel, in the current context: not when it is one of a refused kernel, which is
/// then said once.
#[inline]
pub fn allows(handle: *mut c_void, call: &str) -> bool {
    allows_where(handle, call, None)
}

/// `allows`, for a call about a library's kernel on `device` (`cuKernelGetAttribute` and the like).
#[inline]
pub fn allows_on(handle: *mut c_void, call: &str, device: c_int) -> bool {
    allows_where(handle, call, Some(device))
}

#[inline]
fn allows_where(handle: *mut c_void, call: &str, device: Option<c_int>) -> bool {
    if !kernels::refusing() {
        return true;
    }
    let handle = handle as usize;
    let generation = kernels::generation();
    if handle == 0 || slot(handle).holds(handle as u64, generation) {
        return true;
    }
    decide(handle, generation, call, device)
}

/// `allows`, for a handle not found allowed in this generation: what the driver says it is.
#[cold]
#[inline(never)]
fn decide(handle: usize, generation: u64, call: &str, device: Option<c_int>) -> bool {
    // SAFETY: the program passes the handle to a call of the driver, which accepts it or refuses
    // it; the layer asks the driver about it the same way.
    match unsafe { kernels::verdict(handle, device) } {
        kernels::Verdict::Refused(refusal) => {
            refusal.explain(call);
            false
        }
        kernels::Verdict::Allowed => {
            // (Of the generation read before the records were: if it has ended since, the entry
            // is never found. Not kept while a module, a library or a context is going: the
            // handle may already be another function's, `kernels::Ending`.)
            if !kernels::ending_now() {
                slot(handle).keep(handle as u64, generation);
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
