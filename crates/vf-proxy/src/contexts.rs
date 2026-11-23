//! The modules of each context. A module is loaded in the context that is current, and goes with
//! it: when a context is destroyed (`cuCtxDestroy`, the last release or a reset of the primary
//! context of a device), the driver unloads its modules without a call of the program, and what
//! the layer kept for them is dropped here, as at an unload. A library is in every context and
//! stays; the module that the driver gives for it in a context goes with that context.
//!
//! The context of a module is the one current when the driver gave the module. A context that
//! goes without one of these calls (a green context) leaves what the layer kept for its modules
//! until the driver gives their handles again (`images::keep`).
//!
//! What is known of the modules of a context is taken before the driver is asked to destroy it
//! (`of`), and dropped after (`destroyed`): once the driver destroyed it, another thread can be
//! given a context at the same address and load modules in it, or be given the handle of one of
//! its modules for a module of its own, and what the layer keeps of those is theirs. A module
//! whose handle the driver gave again meanwhile is left: a load forgets the handle here first
//! (`images::keep`), under the lock that `destroyed` holds while it drops what was kept.

use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_void};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::{REALS, idx};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;

/// What is known of a module: its context, and the number of the record (a module given again,
/// even in the same context, has a record of another number).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Record {
    context: usize,
    number: u64,
}

/// The number of the next record.
static NUMBERS: AtomicU64 = AtomicU64::new(0);
/// The record of each module that the program holds, by the module's handle.
static MODULES: LazyLock<Mutex<HashMap<usize, Record>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
/// The primary context of each device, as the driver gave it to the program.
static PRIMARY: LazyLock<Mutex<HashMap<c_int, usize>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn modules() -> MutexGuard<'static, HashMap<usize, Record>> {
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
/// just loaded, or the module of a library in this context. (Called before anything else is
/// kept for a module that the driver gives for a library: see `destroyed`.)
pub fn loaded(module: usize) {
    let context = current();
    let mut modules = modules();
    match context {
        Some(context) => modules.insert(
            module,
            Record {
                context,
                number: NUMBERS.fetch_add(1, Relaxed),
            },
        ),
        None => modules.remove(&module),
    };
}

/// `handle` is unloaded, or the driver gave it again: it is no module of a context known here.
/// (Called before anything else is kept for a handle that the driver gives again: see
/// `destroyed`.)
pub fn forget(handle: usize) {
    modules().remove(&handle);
}

/// The record of a module that `take` took, for `restore`.
pub struct Taken(Option<(usize, Record)>);

/// The module `handle` is to be unloaded: its record is taken away (as `forget`), and given back
/// for `restore` if the driver does not unload it.
pub fn take(handle: usize) -> Taken {
    Taken(modules().remove(&handle).map(|record| (handle, record)))
}

/// The driver did not unload the module of which `take` took `taken`: its record is kept again,
/// unless another was made for its handle since.
pub fn restore(taken: Taken) {
    if let Taken(Some((handle, record))) = taken {
        modules().entry(handle).or_insert(record);
    }
}

/// What is known of the modules of a context, taken before the driver is asked to destroy it.
pub struct Modules {
    context: Option<usize>,
    records: Vec<(usize, Record)>,
}

/// The modules of `context` as they are known now, for `destroyed` once the driver destroyed it.
pub fn of(context: usize) -> Modules {
    let records = modules()
        .iter()
        .filter(|(_, record)| record.context == context)
        .map(|(&module, &record)| (module, record))
        .collect();
    Modules {
        context: Some(context),
        records,
    }
}

/// The driver destroyed the context of `modules`, and those modules with it: what the layer kept
/// for each is dropped, as at an unload, without the driver's. A module whose record is not the
/// one taken before (its handle given again meanwhile: forgotten, or recorded again) is left, and
/// so is a module loaded since in a context at the same address. (The second translation of a
/// module, for `fit`, is a module of the same context: it went too, and is one of these.)
pub fn destroyed(modules: Modules) {
    let mut known = self::modules();
    for (module, record) in modules.records {
        if known.get(&module) != Some(&record) {
            continue;
        }
        known.remove(&module);
        // Under the lock: a load that the driver gives this handle to now waits for this to be
        // done before it keeps anything for it (`forget`, `loaded`).
        crate::fit::forget(module);
        crate::images::dropped(module);
    }
}

/// The driver gave `context` as the primary context of `device`.
pub fn primary_retained(device: c_int, context: usize) {
    primary().insert(device, context);
}

/// The modules of the primary context of `device`, as `of`, before the driver is asked to release
/// or reset it.
pub fn of_primary(device: c_int) -> Modules {
    let context = primary().get(&device).copied();
    match context {
        Some(context) => of(context),
        None => Modules {
            context: None,
            records: Vec::new(),
        },
    }
}

/// The program released the primary context of `device`, whose modules were `modules` before: it
/// is destroyed if that was its last retain, which the driver tells by its state.
pub fn primary_released(device: c_int, modules: Modules) {
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
    {
        let mut primary = primary();
        if modules.context.is_some() && primary.get(&device).copied() == modules.context {
            primary.remove(&device);
        }
    }
    destroyed(modules);
}

/// The program reset the primary context of `device`, whose modules were `modules` before:
/// destroyed, however many retains it had. (It stays the device's primary context: a reset does
/// not release it, and it is made again when it is used.)
pub fn primary_reset(modules: Modules) {
    destroyed(modules);
}
