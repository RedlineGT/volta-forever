//! What this library knows about the kernels of the images it adapted, by the handle of their module
//! or library.
//!
//! - Kernels that must not be given to the program: they are, or call, a function that had to be made
//!   to trap (an instruction without a sequence for this GPU, or something the driver's compiler
//!   refused). Launched, such a kernel ends the CUDA context of the program; refused at the lookup, it
//!   is an error the program can report.
//! - Kernels that run emulated instructions: announced on the console when the program takes them.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{LazyLock, Mutex};

use vf_adapt::emulation::Announcer;
use vf_adapt::{Emulated, EmulatedCounts, Refused};

use crate::{config, log, profile};

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
    /// How the code of this module reached the device.
    origin: Option<Origin>,
}

static KNOWN: LazyLock<Mutex<HashMap<usize, Kernels>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
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

/// Module aliases consult their library's current metadata, including compiler
/// refusals learned after a lazy submodule first exposed a function.
fn metadata(known: &HashMap<usize, Kernels>, handle: usize) -> Option<&Kernels> {
    let entry = known.get(&handle)?;
    match entry.library {
        Some(library) => known.get(&library),
        None => Some(entry),
    }
}

/// Record what the adaptation found about the kernels of the image loaded as `handle`.
pub fn describe(handle: usize, refused: Vec<Refused>, emulated: Vec<Emulated>) {
    if refused.is_empty() && emulated.is_empty() {
        return;
    }
    let mut known = known();
    let kernels = known.entry(handle).or_default();
    for kernel in refused {
        kernels
            .refused
            .entry(kernel.kernel)
            .or_insert((kernel.function, kernel.reason));
    }
    for kernel in emulated {
        kernels.emulated.insert(kernel.kernel, kernel.counts);
    }
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

/// `module` is the module of library `library`: the same kernels.
pub fn share(module: usize, library: usize) {
    let mut known = known();
    if module == library {
        return;
    }
    if known.contains_key(&library) {
        known.insert(
            module,
            Kernels {
                library: Some(library),
                ..Default::default()
            },
        );
    } else {
        // A context can return the former module address for a different
        // library, including a native one with no recorded refusals.
        known.remove(&module);
    }
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
}

/// Whether enumerations need to filter this module or library.
pub fn has_refused(handle: usize) -> bool {
    metadata(&known(), handle).is_some_and(|k| !k.refused.is_empty())
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
