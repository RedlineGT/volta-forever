//! `libcuda.so.1` for Volta GPUs.
//!
//! This library takes the place of NVIDIA's user-mode driver library in the library search path and
//! forwards every function to it. Programs built with CUDA 13 carry no code for Volta (sm_70); the few
//! functions through which code reaches the driver are intercepted (`hooks`) to hand the driver an
//! image it accepts (`vf_adapt`).
//!
//! Two facts about the CUDA runtime shape the design (measured with runtimes 12.9 and 13.5 on driver
//! 580):
//! - the runtime resolves driver functions with `cuGetProcAddress`, not `dlsym`, so lookups of hooked
//!   functions are answered with this library's entry points. Lookups of every other function return
//!   NVIDIA's own pointer: copies and the rest of the hot path do not pass through this library at
//!   all. (A kernel launch does, unless `VF_FIT=0`: its entry here calls NVIDIA's and looks at the
//!   answer only when it is a refusal for registers, `fit`.);
//! - the private tables behind `cuGetExportTable` are integrity-checked by the runtime, so they are
//!   forwarded untouched. Code is loaded through public functions (`cuLibraryLoadData`).
//!
//! Environment:
//! - `VF_REAL_LIBCUDA`: path of NVIDIA's library (default: the `libcuda.so.<version>` of the loaded
//!   kernel module, in the system library directories);
//! - `VF_LOG`: log file (default: standard error); `VF_LOG_LEVEL`: 0 fatal problems, 1 images that
//!   could not be adapted (default), 2 what was adapted, 3 every image and lookup;
//! - `VF_ADAPT=0`: forward images unchanged;
//! - `VF_LAZY=0`: compile whole modules when they are loaded, not each kernel when it is asked for;
//! - `VF_ALIGNED_ROWS=0`: do not take it that programs keep the rows `ldmatrix` reads at multiples
//!   of 16, as the PTX manual requires and as the layer assumes: test the addresses when the code
//!   runs. Slower emulated tensor-core kernels; for a program that stops with a misaligned-address
//!   fault (error 716) without it;
//! - `VF_TRANSLATE=0`: disable machine-code translation (an image with nothing else is then
//!   loaded as it is, and the driver refuses it); `VF_TRANSLATE=first`: translate it also where
//!   the image has PTX that would otherwise be rewritten;
//! - `VF_EMULATION_WARNINGS=N`: list the first N kernels given to the program with emulated
//!   instructions (default 20; 0: none);
//! - `VF_CAPABILITY=8.0`: compute capability reported to the callers that have no code for this GPU
//!   (default: the architecture their code starts at when it can be adapted, else the real one; see
//!   `callers`). The PTX taken from a program is then the PTX it has for that capability; a caller
//!   whose code runs from its sm_75 machine code is still told 7.5, unless `VF_CAPABILITY_ALL=1`;
//! - `VF_STATS=1`: route every function through this library and print, when the process exits,
//!   the number of calls per function and the kernels launched most often (adds an atomic increment
//!   to each driver call, and a name lookup to each launch);
//! - `VF_TRACE_LAUNCH=1`: route kernel launches through this library and log the ones the driver
//!   refuses (kernel name, grid, block, shared memory, launch attributes).

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[allow(non_snake_case)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}
mod cache;
mod callers;
pub mod fit;
mod hooks;
mod images;
mod kernels;
mod lazy;
mod log;
pub mod marker;
mod profile;

pub use generated::{HOOKED, NAMES, NSYM, idx};

/// Address of NVIDIA's implementation of each function (read by the generated jumps).
pub static REALS: [AtomicUsize; NSYM] = [const { AtomicUsize::new(0) }; NSYM];
/// Number of calls of each function.
pub static COUNTS: [AtomicUsize; NSYM] = [const { AtomicUsize::new(0) }; NSYM];

const CUDA_ERROR_NOT_INITIALIZED: i32 = 3;
const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;

/// Settings read from the environment when the library is loaded.
pub struct Config {
    pub adapt: bool,
    /// Lazy kernel compilation: compile only the kernels the program actually uses.
    pub lazy: bool,
    /// Compute capability to report to callers without code for this GPU, instead of what `callers`
    /// works out.
    pub capability: Option<(i32, i32)>,
    /// `VF_CAPABILITY_ALL=1`: `capability` is reported also to the callers whose code runs from
    /// machine code of an earlier generation, which are otherwise told that one (`callers`).
    pub capability_all: bool,
    pub stats: bool,
    pub trace_launch: bool,
    pub policy: vf_adapt::Policy,
    /// Whether to retry a launch that fails with 701 using a second translation within the
    /// register budget (`VF_FIT`, default on).
    pub fit: bool,
    /// Retry oversized dynamic shared-memory opt-ins at the physical V100 limit.
    pub fit_shared: bool,
    /// How many kernels with emulated instructions are listed on the console before the following
    /// are only counted (0: none, and no summary).
    pub emulation_warnings: usize,
    /// Print per-module and per-kernel timing breakdown (`VF_TIMING=1`).
    pub timing: bool,
    /// `VF_PROFILE=FILE`: write per-kernel timing profile at exit. Each launch is synchronised
    /// on its stream after it returns, so the GPU time is known. One file per process (`.PID`
    /// appended). Off when unset or empty.
    pub profile: bool,
}

static CONFIG: OnceLock<Config> = OnceLock::new();
/// NVIDIA's address of a function -> the entry point of this library that lookups return instead:
/// the hooked functions, or all functions when statistics are on.
static OWN_ENTRY: OnceLock<HashMap<usize, usize>> = OnceLock::new();

pub fn config() -> &'static Config {
    CONFIG.get_or_init(|| {
        let var = |name: &str| std::env::var(name).ok();
        let policy = vf_adapt::Policy::default().with_environment();
        let capability = policy.told_sm.map(|sm| ((sm / 10) as i32, (sm % 10) as i32));
        Config {
            adapt: var("VF_ADAPT").is_none_or(|v| v != "0"),
            lazy: var("VF_LAZY").is_none_or(|v| v != "0"),
            capability,
            capability_all: var("VF_CAPABILITY_ALL").as_deref() == Some("1"),
            stats: var("VF_STATS").is_some_and(|v| v != "0"),
            trace_launch: var("VF_TRACE_LAUNCH").is_some_and(|v| v != "0"),
            fit: var("VF_FIT").is_none_or(|v| v != "0"),
            fit_shared: var("VF_FIT_SHARED").as_deref() == Some("1"),
            policy,
            emulation_warnings: var("VF_EMULATION_WARNINGS").and_then(|v| v.parse().ok()).unwrap_or(20),
            timing: var("VF_TIMING").is_some_and(|v| v != "0"),
            profile: var("VF_PROFILE").is_some_and(|v| !v.is_empty() && v != "0"),
        }
    })
}

/// The entry point of this library to return from a lookup that found NVIDIA's function at `real`.
pub fn own_entry(real: usize) -> Option<usize> {
    OWN_ENTRY.get()?.get(&real).copied()
}

/// Hooks that only observe: lookups lead to them only when `VF_TRACE_LAUNCH` or the fit
/// mechanism asks for it (the launch hooks are always needed when fit is on).
const TRACE_ONLY: [usize; 4] = [
    idx::cuLaunchKernel,
    idx::cuLaunchKernelEx,
    idx::cuLaunchKernel_ptsz,
    idx::cuLaunchKernelEx_ptsz,
];

/// Stands in for every function when NVIDIA's library cannot be loaded.
extern "C" fn not_initialized() -> i32 {
    CUDA_ERROR_NOT_INITIALIZED
}

/// Stands in for a function that NVIDIA's library does not export.
extern "C" fn not_supported() -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

/// Path of the file this library was loaded from.
fn own_path() -> Option<std::path::PathBuf> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    if unsafe { libc::dladdr(&REALS as *const _ as *const c_void, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(info.dli_fname) }.to_str().ok()?;
    std::fs::canonicalize(name).ok()
}

/// NVIDIA's `libcuda.so`: `VF_REAL_LIBCUDA`, else the library of the loaded kernel module's version,
/// else any `libcuda.so.1` in the system directories that is not this file.
fn real_library() -> Option<CString> {
    if let Ok(path) = std::env::var("VF_REAL_LIBCUDA") {
        return CString::new(path).ok();
    }
    const DIRS: [&str; 4] = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/nvidia/lib64",
    ];
    let own = own_path();
    let usable = |path: &str| {
        std::fs::canonicalize(path)
            .ok()
            .filter(|p| Some(p) != own.as_ref())
            .is_some()
    };
    // "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Tue Jul ..."
    let version = std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .and_then(|text| {
            let rest = text.split("Kernel Module").nth(1)?;
            Some(rest.split_whitespace().next()?.to_string())
        });
    let names = version
        .iter()
        .map(|v| format!("libcuda.so.{v}"))
        .chain(["libcuda.so.1".to_string()]);
    for name in names {
        for dir in DIRS {
            let path = format!("{dir}/{name}");
            if usable(&path) {
                return CString::new(path).ok();
            }
        }
    }
    None
}

extern "C" fn init() {
    let config = config();
    log::open();
    if let Err(problem) = vf_adapt::requested_capability() {
        log::warn(format_args!("{problem}"));
    }
    profile::open();
    let path = real_library();
    let handle = match &path {
        Some(path) => unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) },
        None => std::ptr::null_mut(),
    };
    if handle.is_null() {
        let reason = match &path {
            Some(path) => format!("cannot load {}", path.to_string_lossy()),
            None => "NVIDIA's libcuda.so not found (set VF_REAL_LIBCUDA)".to_string(),
        };
        log::error(format_args!("{reason}: every CUDA call will fail"));
        for real in &REALS {
            real.store(not_initialized as *const () as usize, Relaxed);
        }
        return;
    }
    let mut missing = 0;
    for (name, real) in NAMES.iter().zip(&REALS) {
        let address = unsafe { libc::dlsym(handle, name.as_ptr()) } as usize;
        missing += usize::from(address == 0);
        real.store(
            if address == 0 {
                not_supported as *const () as usize
            } else {
                address
            },
            Relaxed,
        );
    }
    // Hooked functions last: when NVIDIA exports one implementation under two names, lookups must
    // lead to the hook.
    let entries = if config.stats {
        generated::counted_entries()
    } else {
        generated::own_entries()
    };
    let mut map = HashMap::new();
    for hooked in [false, true] {
        let wanted = |i: usize| {
            if TRACE_ONLY.contains(&i) {
                config.trace_launch || config.stats || config.profile || config.fit
            } else {
                hooked || config.stats
            }
        };
        for i in (0..NSYM).filter(|&i| HOOKED[i] == hooked && wanted(i)) {
            let real = REALS[i].load(Relaxed);
            if real != not_supported as *const () as usize {
                map.insert(real, entries[i]);
            }
        }
    }
    let _ = OWN_ENTRY.set(map);
    log::info(format_args!(
        "forwarding to {} ({missing} of {NSYM} functions absent), adapt {}, lazy {}, capability {}",
        path.unwrap().to_string_lossy(),
        if config.adapt { "on" } else { "off" },
        if config.lazy { "on" } else { "off" },
        config
            .capability
            .map_or("by caller".to_string(), |(major, minor)| format!("{major}.{minor}")),
    ));
}

extern "C" fn fini() {
    profile::write();
    kernels::summary();
    fit::summary();
    if !config().stats {
        return;
    }
    let mut calls: Vec<(usize, &CStr)> = COUNTS
        .iter()
        .zip(NAMES)
        .map(|(count, name)| (count.load(Relaxed), name))
        .filter(|(n, _)| *n > 0)
        .collect();
    calls.sort_by(|a, b| b.cmp(a));
    let line: Vec<String> = calls
        .iter()
        .map(|(n, name)| format!("{}={n}", name.to_string_lossy()))
        .collect();
    log::error(format_args!("calls: {}", line.join(" "))); // asked for: printed at every level
    if let Some(launches) = hooks::launches() {
        let total: u64 = launches.iter().map(|(_, n)| n).sum();
        let line: Vec<String> = launches
            .iter()
            .take(20)
            .map(|(name, n)| format!("{name}={n}"))
            .collect();
        log::error(format_args!(
            "launches: {total} of {} kernels, the most launched: {}",
            launches.len(),
            line.join(" ")
        ));
    }
}

#[used]
#[unsafe(link_section = ".init_array")]
static INIT: extern "C" fn() = init;

#[used]
#[unsafe(link_section = ".fini_array")]
static FINI: extern "C" fn() = fini;
