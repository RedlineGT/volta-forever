//! Kernels that only the machine code of an image for the capability told has.
//!
//! The layer gives an image that has machine code for sm_75 and for sm_80, and no PTX, the
//! translation of its sm_75 code, whatever the program is told (`vf_adapt::Policy`). A program told
//! 8.0 asks for its kernels by the names they have in the sm_80 code, and one of them can be in
//! that code only: compiled for the later architectures alone, or under another name in the code
//! for sm_75 (`kernels::Elsewhere`). The driver has no kernel of that name in the image it was
//! given, and the lookup fails (500).
//!
//! At that answer the kernel is given from the code that has it: the cubin of the image for the
//! capability told is loaded beside the program's module or library, as a module or a library of
//! its own, through the layer's own entry points, so that it is to the layer what an image of
//! sm_80 machine code alone is to it when a program loads one (translated by the same translator,
//! its kernels refused, guarded, launched, cached and reported the same way), and the kernel is
//! looked up there. It goes with the program's module or library.
//!
//! What makes it sound, and is refused otherwise with the line of `kernels::not_given`:
//!
//! - **The kernel can reach no data of its image** (`vf_adapt::told_kernel_names_global_data`). A
//!   module of its own has its own copy of the image's variables, which nothing keeps in step
//!   with the copy that the program reads and writes through the module it holds. A kernel whose
//!   code names none of them (no relocation, no constant bank that holds their addresses) shares
//!   nothing with the other kernels but the memory that the program gives it as arguments.
//! - **Its code is translated.** A kernel that the translator refuses is not given; nor one of an
//!   image that the driver refuses.
//! - **The image is one that `Elsewhere` is known of**: a fat binary of executable cubins, not one
//!   linked from relocatable code.
//!
//! What it does not change: the kernels that the code given has stay there (a GPU of the capability
//! told would run its own code for every one of them); the kernels that the program is given when
//! it lists those of a library are those of the code given.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use crate::{config, hooks, images, kernels, log};

type CUresult = i32;
const CUDA_SUCCESS: CUresult = 0;

/// The entry point through which the program asked: for a kernel of a library, or for a function
/// of a module (its own, or the one the driver gives for a library in a context).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Through {
    Library,
    Module,
}

/// A cubin of the code for the capability told, loaded for the kernels that only it has.
#[derive(Clone)]
struct Second {
    /// The handle of the module or library that the layer loaded it as.
    handle: usize,
    library: bool,
    /// Its bytes in the program's image, and their digest.
    cubin: Arc<Vec<u8>>,
    digest: u64,
}

#[derive(Default)]
struct State {
    /// By the program's module or library, and the number of its load: what was loaded for it.
    seconds: HashMap<usize, (u64, Vec<Second>)>,
    /// By the handle of a kernel or function given from such code: the program's module or library.
    given: HashMap<usize, usize>,
    /// The kernels whose line was said.
    said: HashSet<String>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));
/// One at a time loads the code for a module or library.
static LOADING: Mutex<()> = Mutex::new(());

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The lookup of kernel `name` in the module or library `handle` failed as the driver answers for a
/// name that it does not have. When the image's machine code for the capability told has the
/// kernel, it is given from there if it can be (the module doc), with a line the first time, and
/// the answer is success; else the line of `kernels::not_given` says why not, and the answer is
/// `None`: the lookup fails as it did.
///
/// # Safety
/// `c_name` is the C string of `name`; `out` is null or writable. The current context is the one
/// the program made the call in.
pub unsafe fn lookup(
    handle: usize,
    name: &str,
    c_name: *const c_char,
    out: *mut *mut c_void,
    through: Through,
) -> Option<CUresult> {
    let told = kernels::told_code(handle, name)?;
    if !config().told_code {
        kernels::not_given(name, &told.elsewhere, None);
        return None;
    }
    match unsafe { give(&told, name, c_name, out, through) } {
        Ok(second) => {
            let kernel = if out.is_null() { 0 } else { (unsafe { *out }) as usize };
            let first = {
                let mut state = state();
                if kernel != 0 {
                    state.given.insert(kernel, told.owner);
                }
                state.said.len() < vf_adapt::emulation::KEPT_KERNELS && state.said.insert(name.to_string())
            };
            if first {
                log::warn(format_args!("{}", given_line(name, &told.elsewhere, second.library)));
            }
            Some(CUDA_SUCCESS)
        }
        Err(why) => {
            if !out.is_null() {
                unsafe { *out = std::ptr::null_mut() };
            }
            kernels::not_given(name, &told.elsewhere, (!why.is_empty()).then_some(why.as_str()));
            None
        }
    }
}

/// The line that says where a kernel given from the code for the capability told comes from.
fn given_line(name: &str, elsewhere: &kernels::Elsewhere, library: bool) -> String {
    let kernels::Elsewhere {
        given_sm,
        told_sm,
        loads_sm,
        image,
        ..
    } = elsewhere;
    format!(
        "kernel {name} is given from the machine code for sm_{loads_sm} of its image ({image}), translated, in a {} \
         of its own: the code that the layer gives for the image, its machine code for sm_{given_sm}, has no kernel \
         of that name, and a GPU of the capability told ({}.{}) loads the code for sm_{loads_sm}. Its translation \
         is not independently validated (LIMITS.md, \"The capability told with VF_CAPABILITY\")",
        if library { "library" } else { "module" },
        told_sm / 10,
        told_sm % 10
    )
}

/// `lookup`: the kernel from the code for the capability told, or why it is not given (nothing:
/// the line of `kernels::not_given` says it already).
unsafe fn give(
    told: &kernels::Told,
    name: &str,
    c_name: *const c_char,
    out: *mut *mut c_void,
    through: Through,
) -> Result<Second, String> {
    let elsewhere = &told.elsewhere;
    if !config().policy.translates(elsewhere.loads_sm) {
        return Err(String::new());
    }
    let load = images::load_of(told.owner);
    let image = images::retained(told.owner).ok_or("its image is not kept")?;
    let cubin = vf_fatbin::parse(image.data_bytes())
        .ok()
        .and_then(|fat| vf_adapt::told_cubin(&fat, elsewhere.loads_sm, name))
        .ok_or("its code is not read")?;
    drop(image);
    let digest = images::xxh64(&cubin);
    let second = {
        let _loading = LOADING.lock().unwrap_or_else(|e| e.into_inner());
        let known = state()
            .seconds
            .get(&told.owner)
            .filter(|(of, _)| *of == load)
            .and_then(|(_, seconds)| seconds.iter().find(|s| s.digest == digest).cloned());
        match known {
            Some(second) => second,
            None => {
                let handle = unsafe { hooks::load_told(cubin.as_ptr().cast(), elsewhere.told_sm, told.library) }
                    .map_err(|result| format!("its code was not loaded (error {result})"))?;
                let second = Second {
                    handle,
                    library: told.library,
                    cubin: Arc::new(cubin),
                    digest,
                };
                let mut state = state();
                let entry = state.seconds.entry(told.owner).or_insert_with(|| (load, Vec::new()));
                if entry.0 != load {
                    // (The handle was given to another load since: what was loaded for the one
                    // before went with it, `take`.)
                    *entry = (load, Vec::new());
                }
                entry.1.push(second.clone());
                second
            }
        }
    };
    if let Some(reason) = kernels::refusal(second.handle, name) {
        return Err(reason);
    }
    if vf_adapt::told_kernel_names_global_data(&second.cubin, name) {
        return Err(format!(
            "it can reach data of its image, of which a {} of its own would have another copy than the one that \
             the program reads and writes",
            if told.library { "library" } else { "module" }
        ));
    }
    let result = unsafe { hooks::lookup_told(second.handle, second.library, c_name, out, through) };
    if result != CUDA_SUCCESS {
        return Err(format!("the lookup in its translation failed (error {result})"));
    }
    Ok(second)
}

/// Whether `kernel` (a kernel or a function) was given from the code for the capability told.
pub fn gave(kernel: usize) -> bool {
    kernel != 0 && state().given.contains_key(&kernel)
}

/// For a kernel given from the code for the capability told: the program's library, which is the
/// one it asked (`cuKernelGetLibrary`).
pub fn library_of(kernel: usize) -> Option<usize> {
    state().given.get(&kernel).copied()
}

/// The program's module or library `owner` is unloaded, or went with its context: what was loaded
/// for it is forgotten here, and returned (handle, whether a library) for the caller to unload
/// where the driver has not done so itself.
pub fn take(owner: usize) -> Vec<(usize, bool)> {
    let mut state = state();
    state.given.retain(|_, of| *of != owner);
    state
        .seconds
        .remove(&owner)
        .map(|(_, seconds)| seconds.iter().map(|s| (s.handle, s.library)).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_the_code_and_the_image() {
        let elsewhere = kernels::Elsewhere {
            given_sm: 75,
            told_sm: 80,
            loads_sm: 80,
            why: String::new(),
            image: "lib.so: 9184 bytes, XXH64 0123456789abcdef".to_string(),
        };
        assert_eq!(
            given_line("k_from_80", &elsewhere, true),
            "kernel k_from_80 is given from the machine code for sm_80 of its image (lib.so: 9184 bytes, XXH64 \
             0123456789abcdef), translated, in a library of its own: the code that the layer gives for the image, \
             its machine code for sm_75, has no kernel of that name, and a GPU of the capability told (8.0) loads \
             the code for sm_80. Its translation is not independently validated (LIMITS.md, \"The capability told \
             with VF_CAPABILITY\")"
        );
    }

    #[test]
    fn what_was_loaded_for_a_module_goes_with_it() {
        let (owner, other) = (0x7e57_3000, 0x7e57_3010);
        {
            let mut state = state();
            for (of, handle) in [(owner, 0x7e57_3100), (other, 0x7e57_3110)] {
                state.seconds.insert(
                    of,
                    (
                        1,
                        vec![Second {
                            handle,
                            library: false,
                            cubin: Arc::new(vec![0]),
                            digest: 1,
                        }],
                    ),
                );
            }
            state.given.insert(0x7e57_3200, owner);
            state.given.insert(0x7e57_3210, other);
        }
        assert!(gave(0x7e57_3200));
        assert_eq!(library_of(0x7e57_3200), Some(owner));
        assert_eq!(take(owner), vec![(0x7e57_3100, false)]);
        assert!(!gave(0x7e57_3200));
        assert!(gave(0x7e57_3210));
        assert_eq!(take(owner), vec![]);
        assert_eq!(take(other), vec![(0x7e57_3110, false)]);
    }
}
