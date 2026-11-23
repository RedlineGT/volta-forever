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
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};

use crate::kernels;

/// What a call with a handle to a refused kernel returns: what the lookup of that kernel by name
/// returns (`CUDA_ERROR_INVALID_PTX`), for the same cause; not `CUDA_ERROR_INVALID_HANDLE` (400),
/// NVIDIA's answer to a handle that is no function, which would tell the program that its handle
/// is wrong.
pub const REFUSED: c_int = crate::images::CUDA_ERROR_INVALID_PTX;

/// Handles found allowed, each with the generation it was found in, by a hash of the handle. A
/// handle that collides with another is asked about again.
const SLOTS: usize = 4096;
static ALLOWED: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
/// The bits of an entry that hold the handle; the bits above hold the low bits of the generation.
const HANDLE_BITS: u32 = 48;

fn slot(handle: usize) -> usize {
    ((handle >> 4) ^ (handle >> 16)) % SLOTS
}

/// The entry of `handle` found allowed in `generation`: none for a handle that does not fit.
fn entry(handle: usize, generation: u64) -> Option<u64> {
    (handle != 0 && (handle as u64) >> HANDLE_BITS == 0).then_some(handle as u64 | generation << HANDLE_BITS)
}

/// A new generation began (`kernels::new_generation`): when the bits of it that an entry holds
/// come back to 0, every entry is dropped, so that one of a generation that long gone is not taken
/// for one of this one.
pub fn generation_began(generation: u64) {
    if generation & ((1 << (64 - HANDLE_BITS)) - 1) == 0 {
        for allowed in &ALLOWED {
            allowed.store(0, SeqCst);
        }
    }
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
    let entry = entry(handle, generation);
    if handle == 0 || entry.is_some_and(|entry| ALLOWED[slot(handle)].load(SeqCst) == entry) {
        return true;
    }
    decide(handle, entry, call, device)
}

/// `allows`, for a handle not found allowed in this generation: what the driver says it is.
#[cold]
#[inline(never)]
fn decide(handle: usize, entry: Option<u64>, call: &str, device: Option<c_int>) -> bool {
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
            if let Some(entry) = entry
                && !kernels::ending_now()
            {
                ALLOWED[slot(handle)].store(entry, SeqCst);
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
    fn an_entry_holds_the_handle_and_the_generation() {
        let handle = 0x7f12_3456_7890;
        assert_eq!(entry(handle, 3), Some(handle as u64 | 3 << HANDLE_BITS));
        // Generations that differ in the bits an entry holds give other entries.
        assert_ne!(entry(handle, 3), entry(handle, 4));
        // No entry for a null handle, nor for one with bits where the generation goes.
        assert_eq!(entry(0, 3), None);
        assert_eq!(entry(1 << HANDLE_BITS, 3), None);
    }

    #[test]
    fn entries_go_when_the_generation_comes_back() {
        let handle = 0x5555_0000_1230;
        ALLOWED[slot(handle)].store(entry(handle, 0x10000).unwrap(), SeqCst);
        generation_began(0x1ffff);
        assert_ne!(ALLOWED[slot(handle)].load(SeqCst), 0);
        // 0x20000 holds the same low bits as 0x10000: the entry would be found again.
        generation_began(0x20000);
        assert_eq!(ALLOWED[slot(handle)].load(SeqCst), 0);
    }
}
