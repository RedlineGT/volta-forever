//! What this library knows about the kernels of the images it adapted, by the handle of their module
//! or library.
//!
//! - Kernels that must not be given to the program: they are, or call, a function that had to be made
//!   to trap (an instruction without a sequence for this GPU, or something the driver's compiler
//!   refused). Launched, such a kernel ends the CUDA context of the program; refused at the lookup, it
//!   is an error the program can report. A program can have a handle to one without a lookup (a
//!   private interface of the driver, an enumeration made before a refusal was known): what the
//!   driver says of a handle leads back to these records (`verdict`), for `guard`.
//! - Kernels that run emulated instructions: announced on the console when the program takes them.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{LazyLock, Mutex};

use vf_adapt::emulation::Announcer;
use vf_adapt::{Emulated, EmulatedCounts, Refused};

use crate::{REALS, config, idx, log, profile};

/// How the code of a module reached the device, as far as the profiler needs to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The image already had sm_70 code.
    Native,
    /// PTX for a later GPU, rewritten by the layer.
    PtxRewritten,
    /// Machine code translated to sm_70.
    MachineCodeTranslated,
    /// Loaded from the proxy cache.
    CacheHit,
}

#[derive(Default)]
struct Kernels {
    /// By kernel: the function it needs that traps, and why.
    refused: HashMap<String, (String, String)>,
    emulated: HashMap<String, EmulatedCounts>,
    /// For the module of a library: that library, with which it goes away.
    library: Option<usize>,
    /// The handle is a library's (`cuLibraryLoadData`), not a module's: the driver makes a module
    /// of it in each context where it is used, whose functions the program may hold.
    is_library: bool,
    /// How the code of this module reached the device.
    origin: Option<Origin>,
}

static KNOWN: LazyLock<Mutex<HashMap<usize, Kernels>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
/// The number of modules and libraries known here with a refused kernel: while there is none, no
/// handle that the program holds can be one of a refused kernel, and `guard` asks nothing.
static REFUSING: AtomicUsize = AtomicUsize::new(0);
/// Changed whenever what is known here changes, and whenever a module, a library or a context
/// goes (after which the driver may give its handles to others): what `guard` concluded of a
/// handle holds for one generation.
static GENERATION: AtomicU64 = AtomicU64::new(0);
static ANNOUNCER: Announcer = Announcer::new();
/// Refused kernels the program already asked for: the refusal is explained once.
static EXPLAINED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));
/// Whether the remedy for NCCL's kernels was given (`nccl_remedy`).
static NCCL_REMEDY_GIVEN: AtomicBool = AtomicBool::new(false);

/// Whether `name` is the name of one of NCCL's device kernels (`ncclDevKernel_...`,
/// `ncclSymkDevKernel_...`; `ncclKernel_...` before NCCL 2.19), as the compiler gives it: plain, or
/// mangled as a function at global scope (`_Z`, the length of the name, the name, its parameters).
fn nccl_kernel(name: &str) -> bool {
    let identifier = match name.strip_prefix("_Z") {
        Some(rest) => {
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            match rest[..digits].parse::<usize>() {
                Ok(length) => rest.get(digits..digits + length).unwrap_or(""),
                Err(_) => "",
            }
        }
        None => name,
    };
    ["ncclDevKernel_", "ncclSymkDevKernel_", "ncclKernel_"]
        .iter()
        .any(|prefix| identifier.starts_with(prefix))
}

/// For the first refused kernel of NCCL: what the program needs instead. The CUDA 13 build of NCCL
/// reaches instructions that Volta does not have; its CUDA 12 build has code for Volta, and the
/// launcher gives it to the program when it is installed.
fn nccl_remedy(name: &str) -> Option<String> {
    if !nccl_kernel(name) || NCCL_REMEDY_GIVEN.swap(true, Relaxed) {
        return None;
    }
    Some(format!(
        "{name} is a kernel of NCCL, whose CUDA 13 build cannot run its collectives on this GPU: a program \
         that uses several GPUs needs NCCL's CUDA 12 build, which has code for it. `vf fetch --only libnccl` \
         downloads it, and a program started through `vf` is given it in place of its own (`vf doctor` says \
         whether it is found)"
    ))
}

fn known() -> std::sync::MutexGuard<'static, HashMap<usize, Kernels>> {
    KNOWN.lock().unwrap_or_else(|e| e.into_inner())
}

/// What is known here changed (`known` is the map, under its lock): `REFUSING` is counted again, and
/// a new generation begins.
fn changed(known: &HashMap<usize, Kernels>) {
    REFUSING.store(known.values().filter(|k| !k.refused.is_empty()).count(), SeqCst);
    new_generation();
}

/// A new generation begins (`GENERATION`): a module, a library or a context went, or what is known
/// here changed. (After what it changed.)
pub fn new_generation() {
    let generation = GENERATION.fetch_add(1, SeqCst).wrapping_add(1);
    crate::guard::generation_began(generation);
}

/// Whether a module or library known here has a refused kernel.
#[inline]
pub fn refusing() -> bool {
    REFUSING.load(SeqCst) != 0
}

/// The current generation (`GENERATION`).
#[inline]
pub fn generation() -> u64 {
    GENERATION.load(SeqCst)
}

/// Module aliases consult their library's current metadata, including compiler
/// refusals learned after a lazy submodule first exposed a function.
fn metadata(known: &HashMap<usize, Kernels>, handle: usize) -> Option<&Kernels> {
    let entry = known.get(&handle)?;
    match entry.library {
        Some(library) => known.get(&library),
        None => Some(entry),
    }
}

/// Record what the adaptation found about the kernels of the image loaded as `handle`, a library's
/// when `library`.
pub fn describe(handle: usize, refused: Vec<Refused>, emulated: Vec<Emulated>, library: bool) {
    if refused.is_empty() && emulated.is_empty() {
        return;
    }
    let mut known = known();
    let kernels = known.entry(handle).or_default();
    kernels.is_library |= library;
    for kernel in refused {
        kernels
            .refused
            .entry(kernel.kernel)
            .or_insert((kernel.function, kernel.reason));
    }
    for kernel in emulated {
        kernels.emulated.insert(kernel.kernel, kernel.counts);
    }
    changed(&known);
}

/// Record the code origin for the image loaded as `handle`.
pub fn set_origin(handle: usize, origin: Origin) {
    known().entry(handle).or_default().origin = Some(origin);
}

/// The code origin for the module or library `handle`.
#[allow(dead_code)]
pub fn origin(handle: usize) -> Option<Origin> {
    metadata(&known(), handle).and_then(|k| k.origin)
}

/// `module` is the module of library `library`: the same kernels. (A runtime may say it at every
/// launch, through `cuKernelGetFunction`: when it is known already, nothing changes, and no new
/// generation begins.)
pub fn share(module: usize, library: usize) {
    let mut known = known();
    if module == library {
        return;
    }
    if known.contains_key(&library) {
        if known
            .get(&module)
            .is_some_and(|kernels| kernels.library == Some(library))
        {
            return;
        }
        known.insert(
            module,
            Kernels {
                library: Some(library),
                ..Default::default()
            },
        );
    } else if known.remove(&module).is_none() {
        // (A context can return the former module address for a different library, including a
        // native one with no recorded refusals: what was known of it goes. Here nothing was.)
        return;
    }
    changed(&known);
}

/// The module or library is unloaded.
pub fn forget(handle: usize) {
    drop(take(handle));
}

/// What `take` took of a module or library, for `restore`.
pub struct Taken(Vec<(usize, Kernels)>);

/// The module or library `handle` is to be unloaded: what is known of its kernels, and of the
/// modules of a library, is taken away, and given back for `restore` if the driver does not
/// unload it.
pub fn take(handle: usize) -> Taken {
    let profiling = config().profile;
    let mut taken = Vec::new();
    let mut known = known();
    let gone: Vec<usize> = known
        .iter()
        .filter(|&(&known, kernels)| known == handle || kernels.library == Some(handle))
        .map(|(&known, _)| known)
        .collect();
    for at in &gone {
        if let Some(kernels) = known.remove(at) {
            taken.push((*at, kernels));
        }
    }
    // (Also when nothing was known of it: the handles of its functions may be given to others.)
    changed(&known);
    drop(known);
    if profiling {
        // The handle itself may have had no entry (a module the proxy knew nothing about).
        let mut gone = gone;
        if !gone.contains(&handle) {
            gone.push(handle);
        }
        profile::forget(&gone);
    }
    Taken(taken)
}

/// The driver did not unload the module or library of which `take` took `taken`: what was known
/// of its kernels is known again, where nothing was recorded under those handles since.
pub fn restore(taken: Taken) {
    let mut known = known();
    for (handle, kernels) in taken.0 {
        known.entry(handle).or_insert(kernels);
    }
    changed(&known);
}

/// Whether enumerations need to filter this module or library.
pub fn has_refused(handle: usize) -> bool {
    metadata(&known(), handle).is_some_and(|k| !k.refused.is_empty())
}

/// How many kernels of `handle` must not be given to the program, the first of them by name and
/// why; `None` when there is none.
pub fn first_refused(handle: usize) -> Option<(usize, String, String)> {
    let known = known();
    let refused = &metadata(&known, handle)?.refused;
    let (kernel, (_, reason)) = refused.iter().min_by(|a, b| a.0.cmp(b.0))?;
    Some((refused.len(), kernel.clone(), reason.clone()))
}

/// Kernel `name` of `handle` must not be given to the program. The first time the program asks for
/// it, the reason is printed.
pub fn refused(handle: usize, name: &str) -> bool {
    let known = known();
    let Some((function, reason)) = metadata(&known, handle).and_then(|kernels| kernels.refused.get(name)) else {
        return false;
    };
    if EXPLAINED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(name.to_string())
    {
        let through = if function == name {
            String::new()
        } else {
            format!(" (in {function}, which it calls)")
        };
        log::warn(format_args!(
            "kernel {name} cannot run on this GPU and is not given to the program: {reason}{through}"
        ));
        if let Some(remedy) = nccl_remedy(name) {
            log::warn(format_args!("{remedy}"));
        }
    }
    true
}

/// A refused kernel of which the program holds a handle: its name, the function it needs that
/// traps, and why.
pub struct Refusal {
    name: String,
    function: String,
    reason: String,
}

/// Refused kernels of which the program used a handle: the refusal is explained once.
static GUARDED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

impl Refusal {
    /// Say, the first time for this kernel, that `call` with a handle to it is refused.
    pub fn explain(&self, call: &str) {
        if !GUARDED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(self.name.clone())
        {
            return;
        }
        let name = &self.name;
        let through = if self.function == self.name {
            String::new()
        } else {
            format!(" (in {}, which it calls)", self.function)
        };
        log::warn(format_args!(
            "kernel {name} cannot run on this GPU: {}{through}. The program holds a handle to it, which a lookup by name \
             refuses, and {call} with it is refused (error {})",
            self.reason,
            crate::guard::REFUSED
        ));
        if let Some(remedy) = nccl_remedy(name) {
            log::warn(format_args!("{remedy}"));
        }
    }
}

type GetHandle = unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> c_int;
type GetName = unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> c_int;
type FindKernel = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;

/// NVIDIA's function `index`, of the C type `T` (not counted: it is the layer that asks).
///
/// # Safety
/// `T` must be the C type of that function.
unsafe fn driver<T: Copy>(index: usize) -> T {
    let address = REALS[index].load(Relaxed);
    // SAFETY: REALS holds NVIDIA's function, or a stub that ignores its arguments.
    unsafe { std::mem::transmute_copy::<usize, T>(&address) }
}

/// The name that the driver's `getter` (`cuFuncGetName`, `cuKernelGetName`) gives for `handle`.
unsafe fn name_by(getter: usize, handle: usize) -> Option<String> {
    let get: GetName = unsafe { driver(getter) };
    let mut name: *const c_char = std::ptr::null();
    if unsafe { get(&mut name, handle as *mut c_void) } != 0 || name.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned())
}

/// What the driver says `handle` is: a kernel of a library (`cuKernelGetLibrary`; a kernel of a
/// deferred library is the library the program holds), or a function of a module
/// (`cuFuncGetModule`), with its name. `None` when it says neither. (The library first: NVIDIA's
/// driver answers 400 to `cuKernelGetLibrary` for a function, measured on a V100 with driver
/// 580.178.04; `cuFuncGetName` takes a function only.)
unsafe fn identify(handle: usize) -> Option<(usize, String)> {
    let get_library: GetHandle = unsafe { driver(idx::cuKernelGetLibrary) };
    let mut library = std::ptr::null_mut();
    if unsafe { get_library(&mut library, handle as *mut c_void) } == 0 && !library.is_null() {
        let owner = crate::lazy::kernel_library(handle).unwrap_or(library as usize);
        return unsafe { name_by(idx::cuKernelGetName, handle) }.map(|name| (owner, name));
    }
    let get_module: GetHandle = unsafe { driver(idx::cuFuncGetModule) };
    let mut module = std::ptr::null_mut();
    if unsafe { get_module(&mut module, handle as *mut c_void) } != 0 || module.is_null() {
        return None;
    }
    unsafe { name_by(idx::cuFuncGetName, handle) }.map(|name| (module as usize, name))
}

/// Whether `handle` is the kernel `name` of `library` (a library's, in the driver), or its function
/// in the current context. A deferred library has it only once it was loaded whole.
unsafe fn of_library(handle: usize, library: usize, name: &str) -> bool {
    let whole = if crate::lazy::is_lazy(library) {
        match crate::lazy::materialized(library) {
            Some(whole) => Some(whole),
            None => return false,
        }
    } else {
        None
    };
    let target = whole.as_ref().map_or(library, |whole| whole.handle);
    let Ok(name) = std::ffi::CString::new(name) else {
        return false;
    };
    let find: FindKernel = unsafe { driver(idx::cuLibraryGetKernel) };
    let mut kernel = std::ptr::null_mut();
    if unsafe { find(&mut kernel, target as *mut c_void, name.as_ptr()) } != 0 || kernel.is_null() {
        return false;
    }
    if kernel as usize == handle {
        return true;
    }
    let function_of: GetHandle = unsafe { driver(idx::cuKernelGetFunction) };
    let mut function = std::ptr::null_mut();
    let found = unsafe { function_of(&mut function, kernel) };
    found == 0 && function as usize == handle
}

/// What `verdict` says of a handle.
pub enum Verdict {
    /// The handle is that of a refused kernel.
    Refused(Refusal),
    /// It is not, wherever it is used.
    Allowed,
    /// It is a refused kernel of a library, or its function, used where the GPU is not the one
    /// whose code the layer adapts: the driver runs the library's own code there (a library keeps
    /// the entries of the other GPUs), as the module of the library in such a context gives the
    /// kernel by name. Used on this GPU, it is refused. (Not for a deferred library, whose code is
    /// the PTX adapted for this GPU on every GPU.)
    AllowedHere,
}

/// What `handle` is: a refused kernel, if it is a function or a kernel whose module or library is
/// known here with that kernel refused, or a function of a module unknown here that is the module
/// the driver made in this context for a library with that kernel refused (which the program did
/// not ask for with `cuLibraryGetModule` or `cuKernelGetFunction`). The refusals of a library that
/// is not deferred hold where the handle is used on this GPU: on `device`, or in the current
/// context when `None`.
///
/// # Safety
/// `handle` must be one that the driver accepts for a launch, or that it refuses.
pub unsafe fn verdict(handle: usize, device: Option<c_int>) -> Verdict {
    let Some((owner, name)) = (unsafe { identify(handle) }) else {
        return Verdict::Allowed;
    };
    let refusal = |kernels: &Kernels| {
        kernels.refused.get(&name).map(|(function, reason)| Refusal {
            name: name.clone(),
            function: function.clone(),
            reason: reason.clone(),
        })
    };
    let on_this_gpu = || match device {
        Some(device) => crate::hooks::device_needs_adapting(device),
        None => crate::hooks::context_needs_adapting(),
    };
    // A refusal of `library`: for the GPU where the handle is used. (The module that the driver
    // gives for a library in a context is known here only for a context on this GPU.)
    let of_a_library = |library: usize, refusal: Option<Refusal>| match refusal {
        Some(_) if !crate::lazy::is_lazy(library) && !on_this_gpu() => Verdict::AllowedHere,
        Some(refusal) => Verdict::Refused(refusal),
        None => Verdict::Allowed,
    };
    let libraries: Vec<usize> = {
        let known = known();
        if let Some(kernels) = metadata(&known, owner) {
            let found = refusal(kernels);
            let library_kernel = known.get(&owner).is_some_and(|entry| entry.is_library);
            drop(known);
            return match found {
                Some(found) if library_kernel => of_a_library(owner, Some(found)),
                Some(found) => Verdict::Refused(found),
                None => Verdict::Allowed,
            };
        }
        known
            .iter()
            .filter(|(_, kernels)| kernels.is_library && kernels.refused.contains_key(&name))
            .map(|(&library, _)| library)
            .collect()
    };
    for library in libraries {
        if unsafe { of_library(handle, library, &name) } {
            let found = known().get(&library).and_then(refusal);
            return of_a_library(library, found);
        }
    }
    Verdict::Allowed
}

/// Kernel `name` of `handle` is given to the program: announce it if it runs emulated instructions.
/// When the profiler is on, also record the code origin and emulated counts for the kernel.
pub fn given(handle: usize, name: &str) {
    let (counts, origin_val) = {
        let k = known();
        let kernels = metadata(&k, handle);
        let counts = kernels.and_then(|k| k.emulated.get(name)).cloned();
        let origin_val = kernels.and_then(|k| k.origin);
        (counts, origin_val)
    };
    if let Some(ref counts) = counts {
        for line in ANNOUNCER.given(name, counts, config().emulation_warnings, log::at_level(2)) {
            log::warn(format_args!("{line}"));
        }
    }
    // Feed the profiler with the kernel's origin and emulated counts.
    if config().profile {
        let prof_origin = match origin_val {
            Some(Origin::Native) => profile::CodeOrigin::Native,
            Some(Origin::PtxRewritten) => profile::CodeOrigin::PtxRewritten,
            Some(Origin::MachineCodeTranslated) => profile::CodeOrigin::MachineCodeTranslated,
            Some(Origin::CacheHit) => profile::CodeOrigin::CacheHit,
            None => {
                // No origin recorded for this handle: check if the handle is known at all.
                // If known (has emulated or refused info), it was adapted. If not, it is native
                // or an untracked module.
                if is_adapted(handle) {
                    profile::CodeOrigin::PtxRewritten
                } else {
                    profile::CodeOrigin::Unknown
                }
            }
        };
        profile::record_kernel_info_by_name(name, prof_origin, counts);
    }
}

/// The emulated instruction counts for kernel `name` in module or library `handle`, if known.
#[allow(dead_code)]
pub fn emulated_counts(handle: usize, name: &str) -> Option<EmulatedCounts> {
    metadata(&known(), handle)
        .and_then(|kernels| kernels.emulated.get(name))
        .cloned()
}

/// Whether the handle has any known kernels at all (the adaptation found something about it).
pub fn is_adapted(handle: usize) -> bool {
    known().contains_key(&handle)
}

/// When the process ends: the totals of what was announced.
pub fn summary() {
    if let Some(line) = ANNOUNCER.summary(config().emulation_warnings) {
        log::warn(format_args!("{line}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nccl_kernels_are_known_by_their_names() {
        // Names of NCCL 2.30 for CUDA 13, as the layer refused them on a V100.
        for name in [
            "_Z39ncclDevKernel_AllReduce_Sum_f32_RING_LL24ncclDevKernelArgsStorageILm4096EE",
            "_Z22ncclDevKernel_SendRecv24ncclDevKernelArgsStorageILm4096EE",
            "_Z32ncclSymkDevKernel_AllGather_STMC21ncclSymkDevWorkArgs4K",
            "ncclDevKernel_Generic",
            "_Z38ncclKernel_AllReduce_RING_LL_Sum_float14ncclDevKernelArgs",
        ] {
            assert!(nccl_kernel(name), "{name}");
        }
        for name in [
            "sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_execute_kernel__5x_cudnn",
            // The length says where the name ends: here it ends before the prefix is complete.
            "_Z4ncclDevKernel_AllReduce",
            "_Z999ncclDevKernel_",
            "_ZN4nccl13ncclDevKernel_E",
            "_Z12ncclDevFunc_v",
            "my_ncclDevKernel_copy",
            "_Z",
            "",
        ] {
            assert!(!nccl_kernel(name), "{name}");
        }
    }

    #[test]
    fn the_remedy_for_nccl_is_given_once() {
        assert_eq!(nccl_remedy("refused_by_layer"), None);
        let first = nccl_remedy("_Z22ncclDevKernel_SendRecv24ncclDevKernelArgsStorageILm4096EE").unwrap();
        assert!(first.contains("vf fetch --only libnccl"), "{first}");
        assert_eq!(
            nccl_remedy("_Z39ncclDevKernel_AllReduce_Sum_f32_RING_LL24ncclDevKernelArgsStorageILm4096EE"),
            None
        );
    }
}
