//! A second translation at launch, for a kernel that the driver refuses for its registers.
//!
//! The usual translation of a kernel's machine code gives the translation registers of its own
//! above the kernel's: it is the fastest code. The count may then pass what a large block can
//! have (64 registers a thread for 1,024 threads), which the driver says at the launch only
//! (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, 701), and only for the blocks that are too large. When
//! it does, for a kernel of a module whose machine code the layer translated, the layer
//! translates the module's image again within each kernel's own count (`borrow = 5` of the
//! translator: slower code, in no more registers than the kernel declared, rounded up to eight),
//! loads that as a module of its own, and launches the kernel from there. Whenever that cannot be
//! done, or is refused too, the program gets the driver's first answer, as without any of this.
//!
//! Every launch still goes to the first translation first: a smaller block may fit it, and it is
//! the faster code. What the refused call costs is in
//! `docs/performance.md`, "Where the time goes".
//!
//! **Data of the module.** The second module has a copy of the module's data, which nothing
//! keeps in step with the first's. So a kernel that can reach such data
//! (`vf_adapt::kernel_names_global_data`) is never launched from it, nor any kernel of a module
//! from which the program asked for a variable or a texture or surface reference by name
//! (through the module, the library, or the module that the driver gives for a library: one
//! image under two handles).
//!
//! **Attributes.** What the program sets on its kernel (`cuFuncSetAttribute`,
//! `cuKernelSetAttribute`, the two forms of the cache configuration) is set on the kernel of the
//! second module too, when that one is taken and whenever the program sets one after. A
//! library's kernel and the function of the same name in the module that the driver gives for
//! the library are one kernel to the driver: what is set on the library's kernel for a device
//! holds for the function in a context of that device, unless the same attribute was set on
//! the function itself, whatever the order of the calls. The second translation of such a
//! function is a function of a module: it is given, with `cuFuncSetAttribute`, the settings of
//! the library's kernel for the device of the current context, then the function's own; a
//! setting of the library's kernel that comes later is given at the next refused launch, where
//! the function's context is current. (The other way round, a setting made on the function is
//! not carried to the second translation of the library's kernel: a launch through the
//! library's kernel that needs it is refused there too, and the program gets the first answer.)
//!
//! **A launch and the exclusion for data.** No launch into a second translation starts after
//! the program was given data of its module by name: a refused launch is made from the second
//! translation under a lock that the exclusion takes alone (`COMMIT`), after looking again
//! whether the kernel may still be launched from there. A launch whose second translation was
//! being made when the program asked gets the driver's first answer.
//!
//! **A setting and its copy.** A setting of a kernel is one step (`ATTR_ORDER`): the driver's
//! own call on the kernel, what is remembered of it, and the same setting on the second kernel
//! when there is one. Two threads that set the same thing at once leave the kernel and its
//! second translation with the same value, whichever of the two the driver took last.
//!
//! **An unload.** The unload of a module takes both locks over what it forgets and the unload of
//! its second translation: no setting is being given to the second translation then, and no
//! launch made from it, whose handle the driver may give at once to a function of another module.
//!
//! **Contexts.** A library's kernel is valid in every context, and the second translation of a
//! library is loaded as a library. A module belongs to the context it was loaded in: its second
//! translation is loaded in the context that is current at the refused launch, which is that one
//! (the driver looks at the function before it counts registers).
//!
//! `VF_FIT=0` turns all of it off.

use std::collections::{HashMap, HashSet};
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, RwLock};

use crate::{REALS, idx, log};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
/// `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.
pub const OUT_OF_RESOURCES: CUresult = 701;

/// Whether the mechanism is on (`VF_FIT`, on unless it is 0).
pub fn enabled() -> bool {
    crate::config().fit
}

/// Something the program set on a kernel, in the form it used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attribute {
    /// `cuFuncSetAttribute(function, attribute, value)`.
    Function(c_int, c_int),
    /// `cuKernelSetAttribute(attribute, value, kernel, device)`.
    Kernel(c_int, c_int, c_int),
    /// `cuFuncSetCacheConfig(function, config)`.
    FunctionCache(c_int),
    /// `cuKernelSetCacheConfig(kernel, config, device)`.
    KernelCache(c_int, c_int),
}

/// A kernel that the program took from a module or a library.
struct Kernel {
    /// The handle of that module or library.
    module: usize,
    name: String,
    /// From a library (`cuLibraryGetKernel`), not from a module (`cuModuleGetFunction`).
    library: bool,
    /// The same kernel in the module's second translation, once taken from there.
    second: Option<usize>,
    /// The second translation cannot be had for this kernel: it is not asked for again.
    unavailable: bool,
    /// What the program set on the kernel, in order.
    attributes: Vec<Attribute>,
    /// A launch from the second translation succeeded (said once in the log).
    launched: bool,
    /// The effective attributes need to be re-applied to the second image (a kernel-level attribute
    /// was set on the library's CUkernel after the second image was created).
    attrs_stale: bool,
}

/// A module or library whose machine code the layer translated with a kernel over its budget.
struct Module {
    /// The cubin that the program gave and the layer translated, as `Policy::kept_for_second` gave
    /// it to keep (the entry of a fat binary stays compressed until it is needed).
    original: Arc<Vec<u8>>,
    /// The program asked for a variable or a texture reference of it by name.
    asked_for_data: bool,
    /// The second translation, loaded: its handle, and whether it is a library.
    second: Option<(usize, bool)>,
    /// The kernels that the second translation does not have (their code there is a stub that
    /// stops the program).
    refused: Vec<String>,
    /// A thread is making the second translation.
    making: bool,
    /// The second translation could not be made or loaded.
    failed: bool,
    /// The library that this is the driver's module of (`cuLibraryGetModule`): the same image
    /// under another handle, held once.
    library: Option<usize>,
    /// Which record this is: the driver can give the handle to another module once this one is
    /// gone, while a second translation is being made for this one (`second_module`).
    id: u64,
}

/// The number of the next record of a module (`Module::id`).
static MODULE_IDS: AtomicU64 = AtomicU64::new(0);

impl Module {
    /// The entry is gone: what it held of the counts. (The module of a library holds the
    /// library's image, which is counted once.)
    fn released(&self) {
        if self.library.is_none() {
            HELD.fetch_sub(1, Relaxed);
            HELD_BYTES.fetch_sub(self.original.len(), Relaxed);
        }
    }
}

#[derive(Default)]
struct State {
    kernels: HashMap<usize, Kernel>,
    modules: HashMap<usize, Module>,
    /// The modules whose second translation was being made when they went without an unload by
    /// the program (with their context, or their handle given again: `forget`): the second
    /// translation went with them, and its handle is not to be unloaded.
    abandoned: HashSet<u64>,
}

impl Attribute {
    /// The attribute code, for deduplication by kind.
    fn code(&self) -> c_int {
        match self {
            Attribute::Function(a, _) | Attribute::Kernel(a, _, _) => *a,
            Attribute::FunctionCache(_) | Attribute::KernelCache(_, _) => -1, // cache config is its own kind
        }
    }

    /// Whether this is a function-level setting (cuFuncSetAttribute / cuFuncSetCacheConfig).
    fn is_function_level(&self) -> bool {
        matches!(self, Attribute::Function(..) | Attribute::FunctionCache(..))
    }

    /// Convert a kernel-level attribute to its function-level equivalent, for applying to a
    /// second image loaded as a module (the handle is a CUfunction, not a CUkernel).
    fn as_function_level(&self) -> Attribute {
        match *self {
            Attribute::Kernel(a, v, _) => Attribute::Function(a, v),
            Attribute::KernelCache(c, _) => Attribute::FunctionCache(c),
            other => other,
        }
    }
}

impl State {
    /// What the second translation of the function `handle` (a function of a module) is to be
    /// given, as settings of a function, in order: those of the library's kernel of the same
    /// name for `device`, when the function is of the module that the driver gives for a
    /// library, without the attributes that the function has a setting of its own for (which
    /// wins whatever the order: the driver's rule); then the function's own. None when the
    /// library's kernel has settings and the device is not known: they cannot be given
    /// faithfully.
    fn effective_attributes_for_module(&self, handle: usize, device: Option<c_int>) -> Option<Vec<Attribute>> {
        let kernel = self.kernels.get(&handle)?;
        let own = |code: c_int| {
            kernel
                .attributes
                .iter()
                .any(|a| a.is_function_level() && a.code() == code)
        };
        let mut effective: Vec<Attribute> = Vec::new();
        if let Some(library) = self.modules.get(&kernel.module).and_then(|module| module.library) {
            let of_library = self
                .kernels
                .values()
                .filter(|other| other.library && other.module == library && other.name == kernel.name);
            for attribute in of_library.flat_map(|other| &other.attributes) {
                let on = match *attribute {
                    Attribute::Kernel(_, _, on) | Attribute::KernelCache(_, on) => on,
                    Attribute::Function(..) | Attribute::FunctionCache(..) => continue,
                };
                if own(attribute.code()) {
                    continue;
                }
                match device {
                    None => return None,
                    Some(current) if current == on => effective.push(attribute.as_function_level()),
                    Some(_) => {}
                }
            }
        }
        effective.extend(kernel.attributes.iter().map(Attribute::as_function_level));
        Some(effective)
    }

    /// Take out what is known under `handle`, and under the handle of the module that the driver
    /// gave for it if it is a library, with their kernels.
    fn remove(&mut self, handle: usize) -> Vec<(usize, Module)> {
        let handles: Vec<usize> = self
            .modules
            .iter()
            .filter(|(known, module)| **known == handle || module.library == Some(handle))
            .map(|(known, _)| *known)
            .collect();
        self.kernels
            .retain(|_, kernel| kernel.module != handle && !handles.contains(&kernel.module));
        handles
            .into_iter()
            .filter_map(|known| self.modules.remove(&known).map(|module| (known, module)))
            .collect()
    }
}

static STATE: LazyLock<(Mutex<State>, Condvar)> = LazyLock::new(|| (Mutex::new(State::default()), Condvar::new()));

/// Commit point for second-image launches vs. data exclusion and unloads. `retry` takes this shared
/// around the [settle the attributes + re-check + launch] step; `asked_for_data` takes it exclusive
/// around its update, `unload` around its forgetting and the unload of the second image. The slow
/// part (second_module: translation + load) stays outside all. Lock order: COMMIT, then
/// ATTR_ORDER, then STATE's mutex.
static COMMIT: LazyLock<RwLock<()>> = LazyLock::new(|| RwLock::new(()));

/// The order of the settings. Held in `set_attribute` over [the driver's call on the kernel +
/// record + apply to the second kernel] and in `retry` over [take the list + apply all]: the
/// settings reach a second kernel in the order in which the driver took them on the first, and
/// one that arrives while a list is being applied is neither applied before it nor lost. (Until
/// 2026-11-21 the driver's call was made before the lock was taken: of two calls at once, the
/// one that the driver took first could be recorded last, and the second kernel was left with
/// a value that the first no longer had.) One lock for all kernels: a setting is a short call
/// that a program makes a few times, and the driver's own lock is the context's. Held in `unload`
/// too, over [forget + unload the second image]: no setting is applied to a second image that is
/// being unloaded.
static ATTR_ORDER: LazyLock<Mutex<()>> = LazyLock::new(Mutex::default);

/// For the summary: original images held now (count, bytes), the most bytes held at once, second
/// translations made, and launches from them.
static HELD: AtomicUsize = AtomicUsize::new(0);
static HELD_BYTES: AtomicUsize = AtomicUsize::new(0);
static MOST_BYTES: AtomicUsize = AtomicUsize::new(0);
static MADE: AtomicUsize = AtomicUsize::new(0);
static LAUNCHES: AtomicUsize = AtomicUsize::new(0);

fn state() -> MutexGuard<'static, State> {
    STATE.0.lock().unwrap_or_else(|e| e.into_inner())
}

// ---- what the hooks tell ----------------------------------------------------------------------------

/// The driver loaded, as `handle`, the translation of a cubin in which a kernel takes more
/// registers than its budget; `original` is what `Policy::kept_for_second` gave to keep of it.
pub fn register_module(handle: usize, original: Vec<u8>) {
    if !enabled() {
        return;
    }
    forget(handle);
    HELD.fetch_add(1, Relaxed);
    let bytes = HELD_BYTES.fetch_add(original.len(), Relaxed) + original.len();
    MOST_BYTES.fetch_max(bytes, Relaxed);
    state().modules.insert(
        handle,
        Module {
            original: Arc::new(original),
            asked_for_data: false,
            second: None,
            refused: Vec::new(),
            making: false,
            failed: false,
            library: None,
            id: MODULE_IDS.fetch_add(1, Relaxed),
        },
    );
}

/// The driver gave `handle` to a module or library just loaded: whatever was known under that
/// handle is of a module that is gone (unloaded with its context, without a word to the layer).
/// Its second translation, if it had one, went the same way: nothing is unloaded here.
pub fn forget(handle: usize) {
    if !enabled() {
        return;
    }
    let mut state = state();
    for (_, module) in state.remove(handle) {
        module.released();
        if module.making {
            state.abandoned.insert(module.id);
        }
    }
}

/// The program took the kernel `name` of `module`, and got `kernel`.
pub fn register_kernel(kernel: usize, module: usize, name: &str, library: bool) {
    if !enabled() {
        return;
    }
    let mut state = state();
    // (A program may ask twice: the same kernel, and what is known of it is kept.)
    if state
        .kernels
        .get(&kernel)
        .is_some_and(|known| known.module == module && known.name == name)
    {
        return;
    }
    state.kernels.insert(
        kernel,
        Kernel {
            module,
            name: name.to_string(),
            library,
            second: None,
            unavailable: false,
            attributes: Vec::new(),
            launched: false,
            attrs_stale: false,
        },
    );
}

/// `module` is the module that the driver gives for `library` (`cuLibraryGetModule`): the same
/// image under another handle. A kernel that the program takes from it (`cuModuleGetFunction`)
/// is launched from a second translation as any other, and what excludes the library's kernels
/// excludes these, and the other way round (`asked_for_data`). The second translation itself is
/// not shared: the library's is loaded as a library, whose handle `cuModuleGetFunction` refuses
/// (error 400); the module's, when a launch asks for one, is loaded as a module.
pub fn share_module(module: usize, library: usize) {
    if !enabled() {
        return;
    }
    let mut state = state();
    let Some(known) = state.modules.get(&library) else {
        return;
    };
    // (Asked for again: the same module, and what is known of it is kept.)
    if state.modules.get(&module).is_some_and(|m| m.library == Some(library)) {
        return;
    }
    let shared = Module {
        original: known.original.clone(),
        asked_for_data: known.asked_for_data,
        second: None,
        refused: Vec::new(),
        making: false,
        failed: false,
        library: Some(library),
        id: MODULE_IDS.fetch_add(1, Relaxed),
    };
    // (Whatever else was known under that handle is of a module that is gone: `forget`.)
    for (_, gone) in state.remove(module) {
        gone.released();
        if gone.making {
            state.abandoned.insert(gone.id);
        }
    }
    state.modules.insert(module, shared);
}

/// The program asked for a variable or a texture or surface reference of `module` by name: from
/// now on none of its kernels is launched from a second translation, those that were included.
pub fn asked_for_data(module: usize) {
    if !enabled() {
        return;
    }
    // Exclusive: no launch call into a second image starts after this returns.
    let _commit = COMMIT.write().unwrap_or_else(|e| e.into_inner());
    let mut state = state();
    let Some(known) = state.modules.get(&module) else {
        return;
    };
    // A library and the module the driver gives for it are one image: the data that the program
    // reaches through one handle is the data of the kernels taken through the other.
    let library = known.library.unwrap_or(module);
    let same: Vec<usize> = state
        .modules
        .iter()
        .filter(|(handle, known)| **handle == library || known.library == Some(library))
        .map(|(handle, _)| *handle)
        .collect();
    for handle in &same {
        if let Some(known) = state.modules.get_mut(handle) {
            known.asked_for_data = true;
        }
    }
    for kernel in state
        .kernels
        .values_mut()
        .filter(|kernel| same.contains(&kernel.module))
    {
        kernel.second = None;
        kernel.unavailable = true;
    }
}

/// The program sets `attribute` on `kernel`: `set` is the driver's own call, made here so that
/// it, what is remembered of it and its copy on the second kernel are one step (`ATTR_ORDER`).
pub fn set_attribute(kernel: usize, mut attribute: Attribute, set: impl FnOnce() -> CUresult) -> CUresult {
    let fit = enabled();
    let fit_shared = crate::config().fit_shared;
    let _order = (fit || fit_shared).then(|| ATTR_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
    let mut result = set();
    if result == CUDA_ERROR_INVALID_VALUE
        && fit_shared
        && let Some(reduced) = fitted_shared_attribute(kernel, attribute)
        && set_on_driver(kernel, reduced) == CUDA_SUCCESS
    {
        log::info(format_args!(
            "fit: driver accepted {reduced:?} in place of {attribute:?} on V100"
        ));
        attribute = reduced;
        result = CUDA_SUCCESS;
    }
    if fit && result == CUDA_SUCCESS {
        record_attribute(kernel, attribute);
    }
    result
}

/// The driver's opt-in limit includes both the kernel's static shared allocation and the dynamic
/// bytes supplied at launch. All queries use the physical driver, including the device identity.
fn fitted_shared_attribute(kernel: usize, attribute: Attribute) -> Option<Attribute> {
    const MAX_DYNAMIC: c_int = 8;
    const STATIC_SHARED: c_int = 1;
    const OPT_IN_LIMIT: c_int = 97;
    let (requested, device) = match attribute {
        Attribute::Function(MAX_DYNAMIC, value) if value > 0 => (value, current_device()?),
        Attribute::Kernel(MAX_DYNAMIC, value, device) if value > 0 => (value, device),
        _ => return None,
    };
    // The real entry points have these CUDA driver ABI signatures.
    let capability: unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuDeviceComputeCapability].load(Relaxed)) };
    let (mut major, mut minor) = (0, 0);
    if unsafe { capability(&mut major, &mut minor, device) } != CUDA_SUCCESS || (major, minor) != (7, 0) {
        return None;
    }
    let name: unsafe extern "C" fn(*mut std::ffi::c_char, c_int, c_int) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuDeviceGetName].load(Relaxed)) };
    let mut bytes = [0u8; 256];
    if unsafe { name(bytes.as_mut_ptr().cast(), bytes.len() as c_int, device) } != CUDA_SUCCESS {
        return None;
    }
    let name = std::ffi::CStr::from_bytes_until_nul(&bytes).ok()?.to_bytes();
    if !name
        .split(|b| !b.is_ascii_alphanumeric())
        .any(|part| part == b"V100" || part == b"V100S")
    {
        return None;
    }
    let device_attr: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuDeviceGetAttribute].load(Relaxed)) };
    let mut limit = 0;
    if unsafe { device_attr(&mut limit, OPT_IN_LIMIT, device) } != CUDA_SUCCESS || limit <= 0 {
        return None;
    }
    let mut static_bytes = 0;
    let handle = kernel as *mut c_void;
    let query = match attribute {
        Attribute::Function(_, _) => {
            let get: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void) -> CUresult =
                unsafe { std::mem::transmute(REALS[idx::cuFuncGetAttribute].load(Relaxed)) };
            unsafe { get(&mut static_bytes, STATIC_SHARED, handle) }
        }
        _ => {
            let get: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void, c_int) -> CUresult =
                unsafe { std::mem::transmute(REALS[idx::cuKernelGetAttribute].load(Relaxed)) };
            unsafe { get(&mut static_bytes, STATIC_SHARED, handle, device) }
        }
    };
    if query != CUDA_SUCCESS || !(0..=limit).contains(&static_bytes) {
        return None;
    }
    let available = limit - static_bytes;
    if requested <= available {
        return None;
    }
    Some(match attribute {
        Attribute::Function(_, _) => Attribute::Function(MAX_DYNAMIC, available),
        _ => Attribute::Kernel(MAX_DYNAMIC, available, device),
    })
}

/// The driver accepted `attribute` on `kernel` (under `ATTR_ORDER`).
fn record_attribute(kernel: usize, attribute: Attribute) {
    let second = {
        let mut state = state();
        let Some(known) = state.kernels.get_mut(&kernel) else {
            return;
        };
        known.attributes.push(attribute);
        let second = known.second;
        // When a kernel-level attribute is set on a library's CUkernel, mark every CUfunction of
        // the library's module with the same name as needing re-application of the effective
        // attributes at the next retry (if it already has a second image). The re-application
        // happens at retry time, in the right context.
        if known.library && !attribute.is_function_level() {
            let name = known.name.clone();
            let lib = known.module;
            // Find the library's driver module and all functions of it with the same name.
            let module_handles: Vec<usize> = state
                .modules
                .iter()
                .filter(|(_, m)| m.library == Some(lib))
                .map(|(h, _)| *h)
                .collect();
            for other in state.kernels.values_mut() {
                if other.name == name
                    && !other.library
                    && module_handles.contains(&other.module)
                    && other.second.is_some()
                {
                    other.attrs_stale = true;
                }
            }
        }
        second
    };
    if let Some(second) = second {
        apply(second, attribute);
    }
}

/// The program unloaded the module or library `handle`.
pub fn unload(handle: usize) {
    if !enabled() {
        return;
    }
    // Alone: no launch from its second translation and no setting of it is under way while the
    // second translation is unloaded, and none starts after (its handle is then gone from here).
    let _commit = COMMIT.write().unwrap_or_else(|e| e.into_inner());
    let _order = ATTR_ORDER.lock().unwrap_or_else(|e| e.into_inner());
    let gone = state().remove(handle);
    STATE.1.notify_all();
    // (With a library goes the module that the driver gave for it, and its second translation.)
    for (_, module) in gone {
        module.released();
        if let Some((second, library)) = module.second {
            unload_second(second, library);
        }
    }
}

fn unload_second(second: usize, library: bool) {
    // (Forgotten before the driver's unload, which may give the handle to another load at once.)
    let _kept = crate::images::release(second);
    let unload: unsafe extern "C" fn(*mut c_void) -> CUresult = unsafe {
        std::mem::transmute(
            REALS[if library {
                idx::cuLibraryUnload
            } else {
                idx::cuModuleUnload
            }]
            .load(Relaxed),
        )
    };
    let result = unsafe { unload(second as *mut c_void) };
    // (4: the driver is shutting down.)
    if result != CUDA_SUCCESS && result != 4 {
        log::warn(format_args!(
            "fit: the second translation could not be unloaded (error {result})"
        ));
    }
}

/// When the process ends, at the level of the log's information: what was held and done.
pub fn summary() {
    let (made, launches) = (MADE.load(Relaxed), LAUNCHES.load(Relaxed));
    if !enabled() || (made == 0 && MOST_BYTES.load(Relaxed) == 0) {
        return;
    }
    log::info(format_args!(
        "fit: {made} second translations made, {launches} launches from them; original images held for it: {} now, {} bytes at most",
        HELD.load(Relaxed),
        MOST_BYTES.load(Relaxed)
    ));
}

// ---- the refused launch -----------------------------------------------------------------------------

/// The driver refused to launch `function` with 701. `launch` launches the kernel it is given
/// with the program's own arguments. `Some(CUDA_SUCCESS)` when the kernel was launched from the
/// second translation of its module; `None` otherwise: the program is to get the first answer.
///
/// # Safety
/// `launch` must be safe to call with a kernel of the same signature as `function`.
pub unsafe fn retry(
    function: *mut c_void,
    block_threads: c_uint,
    launch: impl Fn(*mut c_void) -> CUresult,
) -> Option<CUresult> {
    if !enabled() {
        return None;
    }
    let handle = function as usize;
    let (module, name, library, known, id) = {
        let state = state();
        let kernel = state.kernels.get(&handle)?;
        if kernel.unavailable {
            return None;
        }
        let id = state.modules.get(&kernel.module).map(|module| module.id);
        (kernel.module, kernel.name.clone(), kernel.library, kernel.second, id)
    };
    // (Outside every lock: a tenth of a second a megabyte, and the driver's time to load.)
    let made = known
        .is_none()
        .then(|| second_module(module, id?, &name, library))
        .flatten();
    // The commit point: `asked_for_data` and `unload` take this lock alone, so that neither
    // returns while a second translation is being looked in, given settings or launched from, and
    // none of that starts after them: the second translation is not unloaded meanwhile.
    let _commit = COMMIT.read().unwrap_or_else(|e| e.into_inner());
    let second = match known {
        Some(second) => second,
        None => {
            // Still the second translation of the same load of the module (not unloaded since,
            // nor the module's handle given to another).
            let current = made.filter(|&second| {
                state()
                    .modules
                    .get(&module)
                    .is_some_and(|known| Some(known.id) == id && known.second == Some((second, library)))
            });
            let taken = current.and_then(|second| kernel_of(second, &name, library));
            let mut state = state();
            if state.modules.get(&module).map(|known| known.id) != id {
                // The module went meanwhile: what is known under the kernel's handle now, if
                // anything, is of another load, and is left as it is.
                return None;
            }
            let kernel = state.kernels.get_mut(&handle)?;
            let Some(second) = taken else {
                kernel.unavailable = true;
                return None;
            };
            if kernel.unavailable {
                // (The program asked for data of the module meanwhile: `asked_for_data`.)
                return None;
            }
            kernel.second = Some(second);
            // (Nothing of what the program set is on it yet.)
            kernel.attrs_stale = true;
            second
        }
    };
    if !settle_attributes(handle, second, library, &name) {
        return None;
    }
    {
        let state = state();
        let kernel = state.kernels.get(&handle)?;
        if kernel.unavailable || kernel.second != Some(second) {
            return None;
        }
        if state
            .modules
            .get(&kernel.module)
            .is_none_or(|module| module.asked_for_data)
        {
            return None;
        }
    }
    let result = launch(second as *mut c_void);
    drop(_commit);
    if result != CUDA_SUCCESS {
        // (Not remembered against the kernel: the refusal may be for what this launch asks, shared
        // memory that the program has not allowed yet for one.)
        log::info(format_args!(
            "fit: {name}: blocks of {block_threads} threads refused of the second translation too (error {result})"
        ));
        return None;
    }
    LAUNCHES.fetch_add(1, Relaxed);
    let first = state()
        .kernels
        .get_mut(&handle)
        .is_some_and(|kernel| !std::mem::replace(&mut kernel.launched, true));
    if first {
        log::info(format_args!(
            "fit: {name}: blocks of {block_threads} threads take more registers than there are with the usual translation; launched from a second one, within the kernel's own count"
        ));
    }
    Some(CUDA_SUCCESS)
}

/// Give the second kernel of `handle` what the program set and it does not have yet (all of it
/// when it was just taken; again, for a function of a library's module, after a setting of the
/// library's kernel). False when that cannot be done faithfully, with a line in the log, or when
/// `second` is not the kernel's second any more: the kernel is then not launched from there this
/// time. Called under `COMMIT`, shared, which keeps `second` from being unloaded.
fn settle_attributes(handle: usize, second: usize, library: bool, name: &str) -> bool {
    let _order = ATTR_ORDER.lock().unwrap_or_else(|e| e.into_inner());
    // (Asked of the driver before the state is locked: no call to the driver under that lock.)
    let device = if library { None } else { current_device() };
    let attributes = {
        let mut state = state();
        // (Unloaded, or excluded, since the caller read it.)
        let Some(kernel) = state
            .kernels
            .get(&handle)
            .filter(|kernel| kernel.second == Some(second))
        else {
            return false;
        };
        if !kernel.attrs_stale {
            return true;
        }
        let attributes = if library {
            Some(kernel.attributes.clone())
        } else {
            state.effective_attributes_for_module(handle, device)
        };
        let Some(attributes) = attributes else {
            log::info(format_args!(
                "fit: {name} is not launched from a second translation: its kernel in the library has settings, and the device of the current context is not known"
            ));
            return false;
        };
        if let Some(kernel) = state.kernels.get_mut(&handle) {
            kernel.attrs_stale = false;
        }
        attributes
    };
    for attribute in attributes {
        apply(second, attribute);
    }
    true
}

/// The device of the current context: the one a function of a module is launched on.
fn current_device() -> Option<c_int> {
    let real = REALS[idx::cuCtxGetDevice].load(Relaxed);
    if real == 0 {
        return None;
    }
    let get: unsafe extern "C" fn(*mut c_int) -> CUresult = unsafe { std::mem::transmute(real) };
    let mut device: c_int = -1;
    (unsafe { get(&mut device) } == CUDA_SUCCESS).then_some(device)
}

/// The second translation of `module`, of the load whose record is `id` (`Module::id`), for its
/// kernel `name`, which is made and loaded if it was not yet (by one thread: others wait for
/// it). None when the kernel may not be launched from there or is refused there, with a line in
/// the log, or when that load is gone (unloaded, or its handle given to another load).
fn second_module(module: usize, id: u64, name: &str, library: bool) -> Option<usize> {
    let (mutex, made) = &*STATE;
    // The kernel itself first: whatever the module has, this one may be excluded.
    let original = {
        let state = mutex.lock().unwrap_or_else(|e| e.into_inner());
        let known = state.modules.get(&module).filter(|known| known.id == id)?;
        if known.asked_for_data {
            log::info(format_args!(
                "fit: {name} is not launched from a second translation: the program asked for data of its module by name"
            ));
            return None;
        }
        known.original.clone()
    };
    let Some(cubin) = vf_adapt::kept_cubin(&original) else {
        log::warn(format_args!("fit: the cubin kept for {name} cannot be read again"));
        return None;
    };
    if vf_adapt::kernel_names_global_data(&cubin, name) {
        log::info(format_args!(
            "fit: {name} is not launched from a second translation: it can reach data of its module"
        ));
        return None;
    }
    // The module's second translation: there, being made, or to make. (The same load all along:
    // the cubin read above is its.)
    let mut state = mutex.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        let known = state.modules.get_mut(&module).filter(|known| known.id == id)?;
        if known.failed {
            return None;
        }
        if let Some((second, _)) = known.second {
            if known.refused.iter().any(|refused| refused == name) {
                log::info(format_args!(
                    "fit: {name} is not launched from a second translation: it is refused there"
                ));
                return None;
            }
            return Some(second);
        }
        if !known.making {
            // (Under the lock that saw it false: no other thread makes it too.)
            known.making = true;
            break;
        }
        state = made.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    drop(state);
    // Outside the lock: a tenth of a second a megabyte, and the driver's time to load.
    let (second, refused) = match translate_again(&cubin) {
        Some((image, refused)) => (load(image, library), refused),
        None => (None, Vec::new()),
    };
    let is_refused = refused.iter().any(|refused| refused == name);
    let mut state = mutex.lock().unwrap_or_else(|e| e.into_inner());
    match state.modules.get_mut(&module).filter(|known| known.id == id) {
        Some(known) => {
            known.making = false;
            known.second = second.map(|second| (second, library));
            known.refused = refused;
            known.failed = second.is_none();
            if second.is_some() {
                MADE.fetch_add(1, Relaxed);
            }
        }
        // The module went meanwhile (and its handle may be another's now). Unloaded by the
        // program: its second translation is unloaded here. Gone with its context, or without a
        // word (`forget`): its second translation went with it, and its handle may be another's.
        None => {
            let abandoned = state.abandoned.remove(&id);
            drop(state);
            if let Some(second) = second.filter(|_| !abandoned) {
                unload_second(second, library);
            }
            made.notify_all();
            return None;
        }
    }
    drop(state);
    made.notify_all();
    if is_refused {
        log::info(format_args!(
            "fit: {name} is not launched from a second translation: it is refused there"
        ));
        return None;
    }
    second
}

/// The cubin translated within each kernel's budget, and the kernels refused that way. (Not kept
/// in the cache of translations: it is made for the few modules that have a refused launch.)
fn translate_again(original: &[u8]) -> Option<(Vec<u8>, Vec<String>)> {
    let policy = vf_adapt::Policy {
        borrow: 5,
        ..crate::config().policy
    };
    let adapted = policy.adapt_cubin(original);
    let vf_adapt::Outcome::Translated { refused, .. } = adapted.outcome else {
        log::warn(format_args!(
            "fit: the cubin was not translated a second time ({:?})",
            adapted.outcome
        ));
        return None;
    };
    Some((
        adapted.image?,
        refused.into_iter().map(|refused| refused.kernel).collect(),
    ))
}

/// Load `image` as a library or as a module of the current context; it is kept until the unload
/// (the driver may read it again).
fn load(image: Vec<u8>, library: bool) -> Option<usize> {
    let mut handle: *mut c_void = std::ptr::null_mut();
    let result = if library {
        let load: unsafe extern "C" fn(
            *mut *mut c_void,
            *const c_void,
            *mut c_int,
            *mut *mut c_void,
            c_uint,
            *mut c_int,
            *mut *mut c_void,
            c_uint,
        ) -> CUresult = unsafe { std::mem::transmute(REALS[idx::cuLibraryLoadData].load(Relaxed)) };
        let (no_options, no_values) = (std::ptr::null_mut::<c_int>(), std::ptr::null_mut::<*mut c_void>());
        unsafe {
            load(
                &mut handle,
                image.as_ptr().cast(),
                no_options,
                no_values,
                0,
                no_options,
                no_values,
                0,
            )
        }
    } else {
        let load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CUresult =
            unsafe { std::mem::transmute(REALS[idx::cuModuleLoadData].load(Relaxed)) };
        unsafe { load(&mut handle, image.as_ptr().cast()) }
    };
    if result != CUDA_SUCCESS || handle.is_null() {
        log::warn(format_args!(
            "fit: the driver did not load the second translation (error {result})"
        ));
        return None;
    }
    // (The handle is new: nothing of the layer's own may be left under it.)
    forget(handle as usize);
    crate::images::keep(handle as usize, crate::images::Replacement::from_vec(image));
    if !library {
        crate::contexts::loaded(handle as usize);
    }
    Some(handle as usize)
}

/// The kernel `name` of the second translation `second`. None when it is not there (the
/// translation within the budget refused it), with a line in the log.
fn kernel_of(second: usize, name: &str, library: bool) -> Option<usize> {
    let get: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult = unsafe {
        std::mem::transmute(
            REALS[if library {
                idx::cuLibraryGetKernel
            } else {
                idx::cuModuleGetFunction
            }]
            .load(Relaxed),
        )
    };
    let c_name = CString::new(name).ok()?;
    let mut kernel: *mut c_void = std::ptr::null_mut();
    let result = unsafe { get(&mut kernel, second as *mut c_void, c_name.as_ptr()) };
    if result != CUDA_SUCCESS || kernel.is_null() {
        log::info(format_args!(
            "fit: {name} is not in the second translation of its module (error {result})"
        ));
        return None;
    }
    Some(kernel as usize)
}

/// Set the attribute through the original CUDA entry point, without re-entering these hooks.
fn set_on_driver(handle: usize, attribute: Attribute) -> CUresult {
    let kernel = handle as *mut c_void;
    let real = |index: usize| REALS[index].load(Relaxed);
    match attribute {
        Attribute::Function(attribute, value) => {
            let set: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> CUresult =
                unsafe { std::mem::transmute(real(idx::cuFuncSetAttribute)) };
            unsafe { set(kernel, attribute, value) }
        }
        Attribute::Kernel(attribute, value, device) => {
            let set: unsafe extern "C" fn(c_int, c_int, *mut c_void, c_int) -> CUresult =
                unsafe { std::mem::transmute(real(idx::cuKernelSetAttribute)) };
            unsafe { set(attribute, value, kernel, device) }
        }
        Attribute::FunctionCache(config) => {
            let set: unsafe extern "C" fn(*mut c_void, c_int) -> CUresult =
                unsafe { std::mem::transmute(real(idx::cuFuncSetCacheConfig)) };
            unsafe { set(kernel, config) }
        }
        Attribute::KernelCache(config, device) => {
            let set: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> CUresult =
                unsafe { std::mem::transmute(real(idx::cuKernelSetCacheConfig)) };
            unsafe { set(kernel, config, device) }
        }
    }
}

/// Replay a setting on the second translation.
fn apply(second: usize, attribute: Attribute) {
    let result = set_on_driver(second, attribute);
    if result != CUDA_SUCCESS {
        log::info(format_args!(
            "fit: {attribute:?} was not taken by the second translation's kernel (error {result})"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Many threads at once want the second translation of one module: one makes it.
    #[test]
    fn made_once() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;
        // The same steps as `second_module`, on a state of the test's own (the real one needs a driver).
        let state = Arc::new((Mutex::new((false, None::<usize>)), Condvar::new())); // (making, second)
        let makers = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(std::sync::Barrier::new(16));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let (state, makers, start) = (state.clone(), makers.clone(), start.clone());
                std::thread::spawn(move || {
                    start.wait();
                    let (mutex, made) = &*state;
                    let mut known = mutex.lock().unwrap();
                    loop {
                        if let Some(second) = known.1 {
                            return second;
                        }
                        if !known.0 {
                            known.0 = true;
                            break;
                        }
                        known = made.wait(known).unwrap();
                    }
                    drop(known);
                    makers.fetch_add(1, Relaxed);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    let mut known = mutex.lock().unwrap();
                    *known = (false, Some(42));
                    drop(known);
                    made.notify_all();
                    42
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), 42);
        }
        assert_eq!(makers.load(Relaxed), 1);
    }
}
