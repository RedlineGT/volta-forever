//! The modules of each context. A module is loaded in the context that is current, and goes with
//! it: when a context is destroyed (`cuCtxDestroy`, the last release or a reset of the primary
//! context of a device), the driver unloads its modules without a call of the program, and what
//! the layer kept for them is dropped here, as at an unload. A library is in every context and
//! stays; the module that the driver gives for it in a context goes with that context.
//!
//! The context of a module is the one current when the driver gave the module. A context that
//! goes without one of these calls (a green context) leaves what the layer kept for its modules
//! until the driver gives their handles again (`images::keep`).

use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_void};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::{REALS, idx};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;

/// The context of each module that the program holds, by the module's handle.
static MODULES: LazyLock<Mutex<HashMap<usize, usize>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
/// The primary context of each device, as the driver gave it to the program.
static PRIMARY: LazyLock<Mutex<HashMap<c_int, usize>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn modules() -> MutexGuard<'static, HashMap<usize, usize>> {
    MODULES.lock().unwrap_or_else(|e| e.into_inner())
}

fn primary() -> MutexGuard<'static, HashMap<c_int, usize>> {
    PRIMARY.lock().unwrap_or_else(|e| e.into_inner())
}

/// The context current in this thread, if there is one.
fn current() -> Option<usize> {
    let real = REALS[idx::cuCtxGetCurrent].load(Relaxed);
    if real == 0 {
        return None;
    }
    // SAFETY: NVIDIA's cuCtxGetCurrent, or a stub that ignores its argument.
    let get: unsafe extern "C" fn(*mut *mut c_void) -> CUresult = unsafe { std::mem::transmute(real) };
    let mut context = std::ptr::null_mut();
    (unsafe { get(&mut context) } == CUDA_SUCCESS && !context.is_null()).then_some(context as usize)
}

/// The driver gave `module` to the program, or to the layer, in the current context: a module
/// just loaded, or the module of a library in this context.
pub fn loaded(module: usize) {
    if let Some(context) = current() {
        modules().insert(module, context);
    }
}

/// `handle` is unloaded, or the driver gave it again: it is no module of a context known here.
pub fn forget(handle: usize) {
    modules().remove(&handle);
}

/// The driver destroyed `context`, and its modules with it.
pub fn destroyed(context: usize) {
    let mut gone = Vec::new();
    modules().retain(|&module, &mut of| {
        let keep = of != context;
        if !keep {
            gone.push(module);
        }
        keep
    });
    // As at an unload, without the driver's. (The second translation of a module, for `fit`, is a
    // module of the same context: it went too, and is one of these.)
    for module in gone {
        crate::fit::forget(module);
        crate::images::release(module);
    }
}

/// The driver gave `context` as the primary context of `device`.
pub fn primary_retained(device: c_int, context: usize) {
    primary().insert(device, context);
}

/// The program released the primary context of `device`: it is destroyed if that was its last
/// retain, which the driver tells by its state.
pub fn primary_released(device: c_int) {
    let real = REALS[idx::cuDevicePrimaryCtxGetState].load(Relaxed);
    if real == 0 {
        return;
    }
    // SAFETY: NVIDIA's cuDevicePrimaryCtxGetState, or a stub that ignores its arguments.
    let state: unsafe extern "C" fn(c_int, *mut c_uint, *mut c_int) -> CUresult = unsafe { std::mem::transmute(real) };
    let (mut flags, mut active) = (0, 1);
    if unsafe { state(device, &mut flags, &mut active) } != CUDA_SUCCESS || active != 0 {
        return;
    }
    let context = primary().remove(&device);
    if let Some(context) = context {
        destroyed(context);
    }
}

/// The program reset the primary context of `device`: destroyed, however many retains it had (it
/// is made again when it is used).
pub fn primary_reset(device: c_int) {
    let context = primary().get(&device).copied();
    if let Some(context) = context {
        destroyed(context);
    }
}
