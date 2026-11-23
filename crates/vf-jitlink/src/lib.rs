//! libnvJitLink.so.13 for Volta GPUs: presents the CUDA 13.5 nvJitLink API to callers, serves it
//! with nvJitLink 12.9.2 (which can target sm_70), and adapts the inputs so the result is sm_70 code.
//!
//! - `-arch=sm_NN` (NN > 70) becomes `-arch=sm_70`.
//! - PTX inputs are retargeted: `.version` lowered to 8.8 (the 12.9.2 ptxas limit), `.target` to
//!   sm_70, and Turing-only instructions rewritten by `vf_adapt::Policy`.
//! - Machine code inputs (cubin, object, library) for sm_75+ are rejected.
//! - Options only in 13.5: `-Ofc`, `-nvvm-latest`, `-no-cache`, `-device-stack-protector` dropped
//!   silently (optimisation hints); `-r` (relocatable link) returns NVJITLINK_ERROR_UNRECOGNIZED_OPTION.
//! - `nvJitLinkVersion` reports 13.5 (the version the caller compiled against).
//! - `nvJitLinkGetLinkedLTOIR[Size]` returns NVJITLINK_ERROR_INTERNAL (12.9.2 has no LTOIR output).
//!
//! The real nvJitLink 12.9.2 is opened at first use from `VF_JITLINK_REAL` (env var) or from
//! `toolkits/cuda-12.9.2/lib/libnvJitLink.so.12` next to this tree.

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::io::Read as _;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::{LazyLock, Mutex, OnceLock};

// --- nvJitLink error codes (matching the CUDA 13.5 header) ---

const NVJITLINK_SUCCESS: c_int = 0;
#[allow(dead_code)]
const NVJITLINK_ERROR_UNRECOGNIZED_OPTION: c_int = 1;
#[allow(dead_code)]
const NVJITLINK_ERROR_MISSING_ARCH: c_int = 2;
const NVJITLINK_ERROR_INVALID_INPUT: c_int = 3;
#[allow(dead_code)]
const NVJITLINK_ERROR_PTX_COMPILE: c_int = 4;
#[allow(dead_code)]
const NVJITLINK_ERROR_NVVM_COMPILE: c_int = 5;
const NVJITLINK_ERROR_INTERNAL: c_int = 6;
const NVJITLINK_ERROR_UNRECOGNIZED_INPUT: c_int = 8;

// Input types
const NVJITLINK_INPUT_CUBIN: c_int = 1;
const NVJITLINK_INPUT_PTX: c_int = 2;
const NVJITLINK_INPUT_LTOIR: c_int = 3;
const NVJITLINK_INPUT_FATBIN: c_int = 4;
const NVJITLINK_INPUT_OBJECT: c_int = 5;
const NVJITLINK_INPUT_LIBRARY: c_int = 6;
#[allow(dead_code)]
const NVJITLINK_INPUT_INDEX: c_int = 7;
const NVJITLINK_INPUT_ANY: c_int = 10;

// --- Logging (same conventions as the proxy: VF_LOG, VF_LOG_LEVEL) ---

mod log {
    use std::fmt::Arguments;
    use std::fs::File;
    use std::io::Write;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

    static FILE: Mutex<Option<File>> = Mutex::new(None);
    static LEVEL: AtomicU8 = AtomicU8::new(1);

    pub fn open() {
        if let Some(level) = std::env::var("VF_LOG_LEVEL").ok().and_then(|v| v.parse().ok()) {
            LEVEL.store(level, Relaxed);
        }
        if let Ok(path) = std::env::var("VF_LOG") {
            match File::options().create(true).append(true).open(&path) {
                Ok(file) => *FILE.lock().unwrap() = Some(file),
                Err(e) => eprintln!("[vf-jitlink] cannot open log file {path}: {e}"),
            }
        }
    }

    fn write(level: u8, args: Arguments<'_>) {
        if LEVEL.load(Relaxed) < level {
            return;
        }
        let line = format!("[vf-jitlink] {args}\n");
        let mut file = FILE.lock().unwrap_or_else(|e| e.into_inner());
        let _ = match file.as_mut() {
            Some(file) => file.write_all(line.as_bytes()),
            None => std::io::stderr().write_all(line.as_bytes()),
        };
    }

    pub fn error(args: Arguments<'_>) {
        write(0, args);
    }
    pub fn warn(args: Arguments<'_>) {
        write(1, args);
    }
    pub fn info(args: Arguments<'_>) {
        write(2, args);
    }
    /// The log level is at least `level`.
    pub fn at_level(level: u8) -> bool {
        LEVEL.load(Relaxed) >= level
    }
}

// --- Real library function pointers ---

type HandlePtr = *mut c_void;

type FnCreate = unsafe extern "C" fn(*mut HandlePtr, c_uint, *const *const c_char) -> c_int;
type FnDestroy = unsafe extern "C" fn(*mut HandlePtr) -> c_int;
type FnAddData = unsafe extern "C" fn(HandlePtr, c_int, *const c_void, usize, *const c_char) -> c_int;
type FnAddFile = unsafe extern "C" fn(HandlePtr, c_int, *const c_char) -> c_int;
type FnComplete = unsafe extern "C" fn(HandlePtr) -> c_int;
type FnGetSize = unsafe extern "C" fn(HandlePtr, *mut usize) -> c_int;
type FnGetBuf = unsafe extern "C" fn(HandlePtr, *mut c_void) -> c_int;
type FnGetCharBuf = unsafe extern "C" fn(HandlePtr, *mut c_char) -> c_int;

struct RealLib {
    _handle: *mut c_void,
    create: FnCreate,
    destroy: FnDestroy,
    add_data: FnAddData,
    add_file: FnAddFile,
    complete: FnComplete,
    get_linked_cubin_size: FnGetSize,
    get_linked_cubin: FnGetBuf,
    get_linked_ptx_size: FnGetSize,
    get_linked_ptx: FnGetCharBuf,
    get_error_log_size: FnGetSize,
    get_error_log: FnGetCharBuf,
    get_info_log_size: FnGetSize,
    get_info_log: FnGetCharBuf,
}

// Safety: the library handle and function pointers are process-global and safe to share.
unsafe impl Send for RealLib {}
unsafe impl Sync for RealLib {}

/// Where the nvJitLink that does the work is, when `VF_JITLINK_REAL` does not name the file:
/// in the directory `VF_JITLINK_REAL_DIR`, or wherever the loader finds `libnvJitLink.so.12` (its
/// search path, or an entry of the audit library's map).
fn default_real_path() -> String {
    match std::env::var("VF_JITLINK_REAL_DIR") {
        Ok(dir) => format!("{dir}/libnvJitLink.so.12"),
        Err(_) => "libnvJitLink.so.12".to_string(),
    }
}

fn load_real() -> Result<RealLib, String> {
    let path = std::env::var("VF_JITLINK_REAL").unwrap_or_else(|_| default_real_path());
    let cpath = CString::new(path.as_bytes()).map_err(|e| format!("bad path: {e}"))?;
    // Safety: calling dlopen/dlsym with valid C strings.
    let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        let err = unsafe { CStr::from_ptr(libc::dlerror()) };
        return Err(format!("dlopen({path}): {}", err.to_string_lossy()));
    }

    macro_rules! sym {
        ($name:expr, $ty:ty) => {{
            let cname = CString::new($name).unwrap();
            let p = unsafe { libc::dlsym(handle, cname.as_ptr()) };
            if p.is_null() {
                return Err(format!("dlsym({path}, {}): not found", $name));
            }
            unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
        }};
    }

    log::info(format_args!("opened real nvJitLink 12.9.2: {path}"));

    Ok(RealLib {
        _handle: handle,
        create: sym!("nvJitLinkCreate", FnCreate),
        destroy: sym!("nvJitLinkDestroy", FnDestroy),
        add_data: sym!("nvJitLinkAddData", FnAddData),
        add_file: sym!("nvJitLinkAddFile", FnAddFile),
        complete: sym!("nvJitLinkComplete", FnComplete),
        get_linked_cubin_size: sym!("nvJitLinkGetLinkedCubinSize", FnGetSize),
        get_linked_cubin: sym!("nvJitLinkGetLinkedCubin", FnGetBuf),
        get_linked_ptx_size: sym!("nvJitLinkGetLinkedPtxSize", FnGetSize),
        get_linked_ptx: sym!("nvJitLinkGetLinkedPtx", FnGetCharBuf),
        get_error_log_size: sym!("nvJitLinkGetErrorLogSize", FnGetSize),
        get_error_log: sym!("nvJitLinkGetErrorLog", FnGetCharBuf),
        get_info_log_size: sym!("nvJitLinkGetInfoLogSize", FnGetSize),
        get_info_log: sym!("nvJitLinkGetInfoLog", FnGetCharBuf),
    })
}

static REAL: OnceLock<Result<RealLib, String>> = OnceLock::new();

fn real() -> Result<&'static RealLib, c_int> {
    match REAL.get_or_init(|| {
        log::open();
        load_real()
    }) {
        Ok(lib) => Ok(lib),
        Err(e) => {
            log::error(format_args!("cannot load nvJitLink 12.9.2: {e}"));
            Err(NVJITLINK_ERROR_INTERNAL)
        }
    }
}

// --- Per-handle state ---

/// The PTX adaptation policy for nvJitLink: ptx_max is (8,8) because ptxas/nvJitLink 12.9.2 only
/// accepts up to PTX ISA 8.8 (the driver JIT accepts 9.0, but we are not using the driver JIT here).
fn jitlink_policy() -> vf_adapt::Policy {
    vf_adapt::Policy::cuda_12_9().with_environment()
}

// --- Option rewriting ---

// Options that exist only in 13.5 (not in 12.9.2 header/docs).
//
// Evidence from the headers:
//   13.5 header documents: -Ofc=<level>, -nvvm-latest, -r, -no-cache, -device-stack-protector
//   12.9.2 header documents: none of these (compare the "Supported Link Options" sections)
//
// -Ofc=<level>: optimisation fast-compile hint, prefix match (always has =<value>)
// -nvvm-latest: NVVM preview flag, exact match
// -no-cache: caching control, exact match
// -device-stack-protector: stack canaries, exact match
// -r: relocatable link -- returns a different kind of output, must error (not silently drop)

/// Prefix-matched options to drop (optimisation hints harmless to ignore).
const OPTIONS_13_ONLY_PREFIX: &[&str] = &["-Ofc="];
/// Exact-matched options to drop (harmless flags 12.9.2 does not know).
const OPTIONS_13_ONLY_DROP: &[&str] = &["-nvvm-latest", "-no-cache", "-device-stack-protector"];
/// Exact-matched options that change semantics and must error.
const OPTIONS_13_ONLY_ERROR: &[&str] = &["-r"];

/// Rewrite the options array: change `-arch=sm_NN` (NN > 70) to `-arch=sm_70`, drop or reject
/// 13.5-only options. Returns Ok((new options, original arch)) or Err(error code).
///
/// A null array is NVJITLINK_ERROR_INVALID_INPUT, even of no options, as nvJitLink 12.9.2 has it;
/// so is a null option (where 12.9.2 reads through it).
fn rewrite_options(
    num_options: c_uint,
    options: *const *const c_char,
) -> Result<(Vec<CString>, Option<String>), c_int> {
    if options.is_null() {
        return Err(NVJITLINK_ERROR_INVALID_INPUT);
    }
    // (Not reserved by the number the caller gives.)
    let mut result = Vec::new();
    let mut original_arch = None;

    for i in 0..num_options as usize {
        // Safety: options array has num_options pointers, each null or to a C string.
        let opt = unsafe { *options.add(i) };
        if opt.is_null() {
            return Err(NVJITLINK_ERROR_INVALID_INPUT);
        }
        let opt = unsafe { CStr::from_ptr(opt) };
        let opt_str = opt.to_string_lossy();

        // Error on options that change semantics.
        if OPTIONS_13_ONLY_ERROR.iter().any(|e| *opt_str == **e) {
            log::error(format_args!(
                "unsupported 13.5-only option: {opt_str} (would change link semantics)"
            ));
            return Err(NVJITLINK_ERROR_UNRECOGNIZED_OPTION);
        }

        // Drop harmless 13.5-only options.
        if OPTIONS_13_ONLY_PREFIX.iter().any(|prefix| opt_str.starts_with(prefix)) {
            log::info(format_args!("dropped 13.5-only option: {opt_str}"));
            continue;
        }
        if OPTIONS_13_ONLY_DROP.iter().any(|e| *opt_str == **e) {
            log::info(format_args!("dropped 13.5-only option: {opt_str}"));
            continue;
        }

        // Rewrite -arch=sm_NN or -arch=compute_NN where NN > 70.
        if let Some(arch_val) = opt_str.strip_prefix("-arch=") {
            let (prefix, num_str) = if let Some(s) = arch_val.strip_prefix("sm_") {
                ("sm_", s)
            } else if let Some(s) = arch_val.strip_prefix("compute_") {
                ("compute_", s)
            } else {
                result.push(opt.to_owned());
                continue;
            };
            // sm_90a, sm_100f: the number is what counts, the variant has no meaning for Volta.
            let digits = num_str.trim_end_matches(|c: char| c.is_ascii_alphabetic());
            if let Ok(sm) = digits.parse::<u32>()
                && sm > 70
            {
                original_arch = Some(format!("{prefix}{sm}"));
                let new_opt = format!("-arch={prefix}70");
                log::info(format_args!("arch: {prefix}{sm} -> {prefix}70"));
                result.push(CString::new(new_opt).unwrap());
                continue;
            }
        }

        result.push(opt.to_owned());
    }

    Ok((result, original_arch))
}

/// Ensure PTX data is null-terminated: nvJitLink 12.9.2 reads past the stated size looking for
/// a null byte (empirically established: AddData rejects PTX that is not null-terminated).
fn ensure_null_terminated(data: Vec<u8>) -> Vec<u8> {
    if data.last() == Some(&0) {
        data
    } else {
        let mut v = data;
        v.push(0);
        v
    }
}

/// With `VF_JITLINK_DUMP=DIR`, every PTX input is written to `DIR/<pid>-<number>.ptx` as it was
/// received: a way to see what a program's own compiler (XLA, Triton, ...) emits for the device it
/// was told about.
fn dump_input(data: &[u8]) {
    static NUMBER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if let Ok(dir) = std::env::var("VF_JITLINK_DUMP") {
        let number = NUMBER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = format!("{dir}/{}-{number:04}.ptx", std::process::id());
        if let Err(error) = std::fs::write(&path, data) {
            log::warn(format_args!("{path}: {error}"));
        }
    }
}

/// The kernels compiled so far with emulated instructions.
static ANNOUNCER: vf_adapt::emulation::Announcer = vf_adapt::emulation::Announcer::new();

/// The process in which the library was loaded.
static LOADED_IN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

extern "C" fn loaded() {
    LOADED_IN.store(std::process::id(), std::sync::atomic::Ordering::Relaxed);
}

#[used]
#[unsafe(link_section = ".init_array")]
static LOADED: extern "C" fn() = loaded;

/// When the process ends: the totals of what was announced.
extern "C" fn summary() {
    // A child made by fork() comes here too when it ends with exit(). What it would report is its
    // parent's, counted before the fork; and it has its parent's locks as they were then: one that
    // another thread of the parent held (the announcer's, or the log's, which every line takes)
    // stays held for ever in the child, and the summary would wait for it. The summary is that of
    // the process that loaded the library.
    if std::process::id() != LOADED_IN.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    if let Some(line) = ANNOUNCER.summary(emulation_warnings()) {
        log::warn(format_args!("{line}"));
    }
}

#[used]
#[unsafe(link_section = ".fini_array")]
static SUMMARY: extern "C" fn() = summary;

/// How many of them are listed on the console (`VF_EMULATION_WARNINGS`, as for the proxy).
fn emulation_warnings() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("VF_EMULATION_WARNINGS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20)
    })
}

/// The kernels (`.entry`) of PTX text.
fn ptx_kernels(text: &[u8]) -> Vec<String> {
    vf_ptx::functions(text)
        .into_iter()
        .filter(|function| function.entry)
        .map(|function| String::from_utf8_lossy(&text[function.name]).into_owned())
        .collect()
}

/// The kernels that trap in the inputs of each link, by its handle: the cubin that the link makes
/// names them (`marked_cubin`).
static MARKS: LazyLock<Mutex<HashMap<usize, Vec<vf_adapt::Refused>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn marks() -> std::sync::MutexGuard<'static, HashMap<usize, Vec<vf_adapt::Refused>>> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// An input of the link `handle`, in which the kernels `refused` trap, was added.
fn mark(handle: HandlePtr, refused: Vec<vf_adapt::Refused>) {
    if !refused.is_empty() {
        marks().entry(handle as usize).or_default().extend(refused);
    }
}

/// The cubin that the link `handle` made, with a note that names the kernels that trap in its
/// inputs (`vf_fatbin::cubin_with_refused`), when there are some: the layer refuses them when the
/// program asks for them, wherever the cubin is loaded. `Ok(None)`: there are none.
fn marked_cubin(lib: &RealLib, handle: HandlePtr) -> Result<Option<Vec<u8>>, c_int> {
    let refused: Vec<vf_fatbin::RefusedKernel> = match marks().get(&(handle as usize)) {
        Some(refused) => refused
            .iter()
            .map(|r| vf_fatbin::RefusedKernel {
                kernel: r.kernel.clone(),
                function: r.function.clone(),
                reason: r.reason.clone(),
            })
            .collect(),
        None => return Ok(None),
    };
    let mut size = 0usize;
    let result = unsafe { (lib.get_linked_cubin_size)(handle, &mut size) };
    if result != NVJITLINK_SUCCESS {
        return Err(result);
    }
    let mut cubin = vec![0u8; size];
    let result = unsafe { (lib.get_linked_cubin)(handle, cubin.as_mut_ptr().cast()) };
    if result != NVJITLINK_SUCCESS {
        return Err(result);
    }
    match vf_fatbin::cubin_with_refused(&cubin, &refused) {
        Some(marked) => Ok(Some(marked)),
        None => {
            // Refuse rather than give machine code with kernels that trap, unmarked.
            log::error(format_args!(
                "the linked cubin cannot be marked with the {} kernels that cannot run on this GPU (the first is {}); \
                 it is not given",
                refused.len(),
                refused[0].kernel
            ));
            Err(NVJITLINK_ERROR_INTERNAL)
        }
    }
}

/// What is compiled here comes back to the program as machine code, whose kernels the driver
/// hands out: a kernel that cannot run can only be refused by refusing the compilation, when it
/// would lose nothing else, or by the layer, at the lookup of the kernel in the cubin that the link
/// makes, which names it (`marked_cubin`). `kernels` are the kernels of what is compiled (the
/// PTX given, or the entries of a fat binary that the linker takes), `refused` those of them that
/// trap, `what` what they are of. True: the compilation is refused (every kernel traps). Otherwise
/// each refused kernel is said, and the kernels that run emulated instructions are announced.
fn refuses_compilation(
    kernels: &[String],
    refused: &[vf_adapt::Refused],
    emulated: &[vf_adapt::Emulated],
    what: &str,
) -> bool {
    if let Some(first) = refused.first() {
        if kernels.iter().all(|kernel| refused.iter().any(|r| r.kernel == *kernel)) {
            log::error(format_args!(
                "compilation refused: {} cannot run on this GPU ({}: {})",
                if kernels.len() == 1 {
                    format!("kernel {}", first.kernel)
                } else {
                    format!(
                        "none of the {} kernels of this {what}, of which {}",
                        kernels.len(),
                        first.kernel
                    )
                },
                first.function,
                first.reason
            ));
            return true;
        }
        for kernel in refused {
            log::warn(format_args!(
                "kernel {} cannot run on this GPU ({}: {}); it is compiled as a trap with the others of its \
                 {what}, and named in the cubin that the link makes: loaded through the layer, it is refused when \
                 the program asks for it; loaded otherwise, launching it ends the CUDA context of the program",
                kernel.kernel, kernel.function, kernel.reason
            ));
        }
    }
    for kernel in emulated {
        for line in ANNOUNCER.given(&kernel.kernel, &kernel.counts, emulation_warnings(), log::at_level(2)) {
            log::warn(format_args!("{line}"));
        }
    }
    false
}

/// Adapt PTX data using the jitlink policy (ptx_max 8.8, sm 70).
/// Returns Ok(adapted_bytes, the kernels that trap in them) on success, Err(error_code) if the PTX
/// cannot be used on sm_70.
fn adapt_ptx_data(data: &[u8]) -> Result<(Vec<u8>, Vec<vf_adapt::Refused>), c_int> {
    dump_input(data);
    let policy = jitlink_policy();
    let adapted = policy.adapt_ptx(data);
    match &adapted.outcome {
        vf_adapt::Outcome::Retargeted {
            from_sm,
            tensor_rewritten,
            tanh_expanded,
            trapped,
            refused,
            emulated,
            ..
        } => {
            log::info(format_args!(
                "PTX adapted: sm_{from_sm} -> sm_70 (tensor: {tensor_rewritten}, tanh: {tanh_expanded}, trapped: {})",
                trapped.len()
            ));
            if refuses_compilation(&ptx_kernels(data), refused, emulated, "PTX") {
                return Err(NVJITLINK_ERROR_PTX_COMPILE);
            }
        }
        vf_adapt::Outcome::Native => {
            log::info(format_args!("PTX already native for sm_70"));
            // Native PTX still needs null-termination for 12.9.2.
            return Ok((ensure_null_terminated(data.to_vec()), Vec::new()));
        }
        vf_adapt::Outcome::NeedsTranslation { lowest_sm } => {
            log::error(format_args!(
                "PTX targets sm_{lowest_sm} (different generation), cannot retarget to sm_70"
            ));
            return Err(NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
        }
        // Not of text.
        vf_adapt::Outcome::Translated { .. } | vf_adapt::Outcome::NotTranslated { .. } => {}
        vf_adapt::Outcome::Unknown => {
            log::error(format_args!("input has no recognisable PTX header"));
            return Err(NVJITLINK_ERROR_INVALID_INPUT);
        }
    }
    let refused = match adapted.outcome {
        vf_adapt::Outcome::Retargeted { refused, .. } => refused,
        _ => Vec::new(),
    };
    match adapted.image {
        Some(img) => Ok((ensure_null_terminated(img), refused)),
        None => Ok((ensure_null_terminated(data.to_vec()), refused)),
    }
}

/// The kernels of the entries of `image`, a fat binary the policy made, that are of `kind` for `sm`:
/// those that the linker takes for sm_70.
fn fatbin_kernels(image: &[u8], kind: u16, sm: u32) -> Vec<String> {
    let mut kernels = Vec::new();
    let Ok(fat) = vf_fatbin::parse(image) else {
        return kernels;
    };
    for entry in fat
        .entries
        .iter()
        .filter(|entry| entry.kind() == kind && entry.sm() == sm)
    {
        let Ok(payload) = entry.payload() else {
            continue;
        };
        if kind == vf_fatbin::KIND_PTX {
            kernels.extend(ptx_kernels(&payload));
        } else if let Some(names) = vf_fatbin::cubin_entries(&payload) {
            kernels.extend(names.into_iter().map(str::to_string));
        }
    }
    kernels
}

/// Try to adapt a fat binary for sm_70: the image to link in its place, if there is one, and the
/// kernels that trap in what the linker takes of it.
fn check_fatbin_data(data: &[u8]) -> Result<(Option<Vec<u8>>, Vec<vf_adapt::Refused>), c_int> {
    let policy = jitlink_policy();
    if let Ok(fat) = vf_fatbin::parse(data) {
        match policy.adapt_fatbin(&fat) {
            Ok(adapted) => {
                let image = adapted.image.as_deref().unwrap_or_default();
                match &adapted.outcome {
                    vf_adapt::Outcome::Retargeted {
                        from_sm,
                        tensor_rewritten,
                        refused,
                        emulated,
                        ..
                    } => {
                        log::info(format_args!(
                            "fatbin adapted: from sm_{from_sm}, {tensor_rewritten} tensor rewrites"
                        ));
                        // The same decision as for PTX given as text: the rewritten entries are
                        // the PTX for sm_70 (for a lower target, that target) of the image.
                        let kernels = fatbin_kernels(image, vf_fatbin::KIND_PTX, policy.device_sm.min(*from_sm));
                        if refuses_compilation(&kernels, refused, emulated, "fat binary's PTX") {
                            return Err(NVJITLINK_ERROR_PTX_COMPILE);
                        }
                    }
                    vf_adapt::Outcome::Translated {
                        from_sm,
                        functions,
                        refused,
                        emulated,
                        ..
                    } => {
                        log::info(format_args!(
                            "fatbin adapted: machine code for sm_{from_sm} translated ({functions} functions, {} not)",
                            refused.len()
                        ));
                        // The translated entries are the machine code for sm_70 of the image.
                        let kernels = fatbin_kernels(image, vf_fatbin::KIND_ELF, policy.device_sm);
                        if refuses_compilation(&kernels, refused, emulated, "fat binary's machine code") {
                            return Err(NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
                        }
                    }
                    vf_adapt::Outcome::NotTranslated { from_sm, reason } => {
                        log::error(format_args!(
                            "fatbin has only machine code for sm_{from_sm}, which could not be read ({reason})"
                        ));
                        return Err(NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
                    }
                    vf_adapt::Outcome::NeedsTranslation { lowest_sm } => {
                        log::error(format_args!(
                            "fatbin has only machine code for sm_{lowest_sm}+, cannot produce sm_70"
                        ));
                        return Err(NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
                    }
                    vf_adapt::Outcome::Native => {
                        log::info(format_args!("fatbin already has sm_70 code"));
                        // Machine code for sm_70 that a link through this library made names what
                        // traps in it.
                        let refused = fat
                            .entries
                            .iter()
                            .filter(|entry| entry.kind() == vf_fatbin::KIND_ELF && policy.usable(entry))
                            .filter_map(|entry| entry.payload().ok())
                            .flat_map(|cubin| marked_kernels(&cubin))
                            .collect();
                        return Ok((None, refused));
                    }
                    vf_adapt::Outcome::Unknown => {}
                }
                let refused = match adapted.outcome {
                    vf_adapt::Outcome::Retargeted { refused, .. } | vf_adapt::Outcome::Translated { refused, .. } => {
                        refused
                    }
                    _ => Vec::new(),
                };
                return Ok((adapted.image, refused));
            }
            Err(e) => {
                log::warn(format_args!("fatbin parse error during adapt: {e:?}"));
            }
        }
    }
    Ok((None, Vec::new()))
}

/// The kernels that the note of a cubin made by a link through this library names
/// (`marked_cubin`).
fn marked_kernels(cubin: &[u8]) -> Vec<vf_adapt::Refused> {
    vf_fatbin::cubin_refused(cubin)
        .into_iter()
        .map(|r| vf_adapt::Refused {
            kernel: r.kernel,
            function: r.function,
            reason: r.reason,
        })
        .collect()
}

/// Check a raw CUDA cubin. Host objects and archives are read by the native linker.
fn check_machine_code(
    lib: &RealLib,
    handle: HandlePtr,
    input_type: c_int,
    data: *const c_void,
    size: usize,
    name: *const c_char,
    raw: &[u8],
) -> c_int {
    let policy = jitlink_policy();
    let adapted = policy.adapt_cubin(raw);
    match adapted.outcome {
        vf_adapt::Outcome::Native => {
            let result = unsafe { (lib.add_data)(handle, input_type, data, size, name) };
            if result == NVJITLINK_SUCCESS {
                mark(handle, marked_kernels(raw));
            }
            result
        }
        vf_adapt::Outcome::NeedsTranslation { lowest_sm } => {
            log::error(format_args!(
                "rejected machine code input (type {input_type}): sm_{lowest_sm} cannot run on sm_70"
            ));
            NVJITLINK_ERROR_UNRECOGNIZED_INPUT
        }
        _ => {
            // Unknown architecture in machine code: reject. Passing it through could yield
            // sm_75 code labelled sm_70.
            log::error(format_args!(
                "rejected machine code input (type {input_type}): architecture not recognised"
            ));
            NVJITLINK_ERROR_UNRECOGNIZED_INPUT
        }
    }
}

// --- Core implementation functions ---

fn do_create(handle: *mut HandlePtr, num_options: c_uint, options: *const *const c_char) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    if handle.is_null() {
        return NVJITLINK_ERROR_INVALID_INPUT;
    }

    let (new_opts, _original_arch) = match rewrite_options(num_options, options) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let opt_ptrs: Vec<*const c_char> = new_opts.iter().map(|s| s.as_ptr()).collect();
    // Safety: passing valid handle pointer and option arrays to the real library.
    let result = unsafe { (lib.create)(handle, opt_ptrs.len() as c_uint, opt_ptrs.as_ptr()) };

    if result != NVJITLINK_SUCCESS {
        log::warn(format_args!("nvJitLinkCreate failed: {result}"));
    } else {
        // (Of a link destroyed before that had this handle, nothing is left.)
        marks().remove(&(unsafe { *handle } as usize));
    }
    result
}

fn do_destroy(handle: *mut HandlePtr) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    if handle.is_null() {
        return NVJITLINK_ERROR_INVALID_INPUT;
    }
    // Before 12.9.2 frees the handle, which another thread's link can then be given; put back if
    // the link was not destroyed.
    let link = unsafe { *handle } as usize;
    let kept = marks().remove(&link);
    let result = unsafe { (lib.destroy)(handle) };
    if result != NVJITLINK_SUCCESS
        && let Some(kept) = kept
    {
        marks().entry(link).or_insert(kept);
    }
    result
}

fn do_add_data(handle: HandlePtr, input_type: c_int, data: *const c_void, size: usize, name: *const c_char) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    // No data, or none at all: NVJITLINK_ERROR_INVALID_INPUT for every kind of input, as nvJitLink
    // 12.9.2 has it (and before a slice is made of a null pointer).
    if handle.is_null() || data.is_null() || size == 0 {
        return NVJITLINK_ERROR_INVALID_INPUT;
    }

    // Safety: data points to `size` bytes.
    let raw = unsafe { std::slice::from_raw_parts(data as *const u8, size) };

    // Add what was made of the input, and note its kernels that trap for the cubin of the link.
    let added = |input_type: c_int, image: Option<&[u8]>, refused: Vec<vf_adapt::Refused>| {
        let (data, size) = image.map_or((data, size), |image| (image.as_ptr().cast(), image.len()));
        let result = unsafe { (lib.add_data)(handle, input_type, data, size, name) };
        if result == NVJITLINK_SUCCESS {
            mark(handle, refused);
        }
        result
    };
    match input_type {
        NVJITLINK_INPUT_PTX => match adapt_ptx_data(raw) {
            Ok((adapted, refused)) => added(input_type, Some(&adapted), refused),
            Err(e) => e,
        },
        NVJITLINK_INPUT_FATBIN => match check_fatbin_data(raw) {
            Ok((adapted, refused)) => added(input_type, adapted.as_deref(), refused),
            Err(e) => e,
        },
        NVJITLINK_INPUT_OBJECT | NVJITLINK_INPUT_LIBRARY => {
            // The ELF header belongs to the host, and an archive has no ELF header at all.
            // nvJitLink selects the embedded device code for the architecture set at Create.
            unsafe { (lib.add_data)(handle, input_type, data, size, name) }
        }
        NVJITLINK_INPUT_CUBIN => check_machine_code(lib, handle, input_type, data, size, name, raw),
        NVJITLINK_INPUT_LTOIR => {
            // LTO-IR: 12.9.2 supports LTO. The architecture is embedded in the IR;
            // 12.9.2 will reject an architecture it cannot handle.
            log::info(format_args!("LTO-IR input: passing to 12.9.2"));
            unsafe { (lib.add_data)(handle, input_type, data, size, name) }
        }
        NVJITLINK_INPUT_ANY => {
            // Auto-detect. Try PTX first (text with a header).
            if vf_ptx::header(raw).is_some() {
                return match adapt_ptx_data(raw) {
                    Ok((adapted, refused)) => added(NVJITLINK_INPUT_PTX, Some(&adapted), refused),
                    Err(e) => e,
                };
            }
            // Fatbin magic.
            if size >= 4 {
                let magic = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                if magic == vf_fatbin::MAGIC {
                    return match check_fatbin_data(raw) {
                        Ok((adapted, refused)) => added(NVJITLINK_INPUT_FATBIN, adapted.as_deref(), refused),
                        Err(e) => e,
                    };
                }
            }
            if raw.starts_with(b"!<arch>\n") || raw.starts_with(b"!<thin>\n") {
                return unsafe { (lib.add_data)(handle, NVJITLINK_INPUT_LIBRARY, data, size, name) };
            }
            // EM_CUDA (190) identifies a cubin. Other ELF files may contain device code.
            if vf_fatbin::is_elf(raw) {
                if raw.get(18..20) != Some(&190u16.to_le_bytes()) {
                    return unsafe { (lib.add_data)(handle, NVJITLINK_INPUT_OBJECT, data, size, name) };
                }
                return check_machine_code(lib, handle, NVJITLINK_INPUT_CUBIN, data, size, name, raw);
            }
            // Unrecognised: reject rather than pass through unexamined data.
            log::error(format_args!("ANY input: unrecognised format (not PTX, fatbin, or ELF)"));
            NVJITLINK_ERROR_UNRECOGNIZED_INPUT
        }
        _ => {
            log::error(format_args!("unknown input type {input_type}"));
            NVJITLINK_ERROR_INVALID_INPUT
        }
    }
}

/// The bytes of the file `name`, a name of the file system: its bytes, whatever they are (made into
/// text, a name that is not UTF-8 would be another file's).
fn read_file(name: &CStr) -> std::io::Result<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt as _;
    let mut buf = Vec::new();
    std::fs::File::open(std::ffi::OsStr::from_bytes(name.to_bytes()))?.read_to_end(&mut buf)?;
    Ok(buf)
}

fn do_add_file(handle: HandlePtr, input_type: c_int, file_name: *const c_char) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    if handle.is_null() || file_name.is_null() {
        return NVJITLINK_ERROR_INVALID_INPUT;
    }

    // For types that need adaptation, read the file and use do_add_data.
    let needs_adaptation = matches!(
        input_type,
        NVJITLINK_INPUT_PTX | NVJITLINK_INPUT_FATBIN | NVJITLINK_INPUT_CUBIN | NVJITLINK_INPUT_ANY
    );

    if needs_adaptation {
        // Safety: file_name is a valid C string.
        let name = unsafe { CStr::from_ptr(file_name) };
        return match read_file(name) {
            Ok(buf) => do_add_data(handle, input_type, buf.as_ptr() as *const c_void, buf.len(), file_name),
            Err(e) => {
                log::error(format_args!("cannot read file {}: {e}", name.to_string_lossy()));
                NVJITLINK_ERROR_INVALID_INPUT
            }
        };
    }

    unsafe { (lib.add_file)(handle, input_type, file_name) }
}

fn do_complete(handle: HandlePtr) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    unsafe { (lib.complete)(handle) }
}

fn do_get_linked_cubin_size(handle: HandlePtr, size: *mut usize) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    if !handle.is_null() && !size.is_null() {
        match marked_cubin(lib, handle) {
            Ok(Some(marked)) => {
                unsafe { *size = marked.len() };
                return NVJITLINK_SUCCESS;
            }
            Ok(None) => {}
            Err(e) => return e,
        }
    }
    unsafe { (lib.get_linked_cubin_size)(handle, size) }
}

fn do_get_linked_cubin(handle: HandlePtr, cubin: *mut c_void) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    if !handle.is_null() && !cubin.is_null() {
        match marked_cubin(lib, handle) {
            Ok(Some(marked)) => {
                unsafe { std::ptr::copy_nonoverlapping(marked.as_ptr(), cubin.cast::<u8>(), marked.len()) };
                return NVJITLINK_SUCCESS;
            }
            Ok(None) => {}
            Err(e) => return e,
        }
    }
    unsafe { (lib.get_linked_cubin)(handle, cubin) }
}

fn do_get_linked_ptx_size(handle: HandlePtr, size: *mut usize) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    unsafe { (lib.get_linked_ptx_size)(handle, size) }
}

fn do_get_linked_ptx(handle: HandlePtr, ptx: *mut c_char) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    unsafe { (lib.get_linked_ptx)(handle, ptx) }
}

fn do_get_error_log_size(handle: HandlePtr, size: *mut usize) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    unsafe { (lib.get_error_log_size)(handle, size) }
}

fn do_get_error_log(handle: HandlePtr, log_buf: *mut c_char) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    unsafe { (lib.get_error_log)(handle, log_buf) }
}

fn do_get_info_log_size(handle: HandlePtr, size: *mut usize) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    unsafe { (lib.get_info_log_size)(handle, size) }
}

fn do_get_info_log(handle: HandlePtr, log_buf: *mut c_char) -> c_int {
    let lib = match real() {
        Ok(l) => l,
        Err(e) => return e,
    };
    unsafe { (lib.get_info_log)(handle, log_buf) }
}

fn do_get_linked_ltoir_size(_handle: HandlePtr, size: *mut usize) -> c_int {
    log::warn(format_args!(
        "nvJitLinkGetLinkedLTOIRSize: not supported by 12.9.2 backend"
    ));
    if !size.is_null() {
        // Safety: size is not null.
        unsafe { *size = 0 };
    }
    NVJITLINK_ERROR_INTERNAL
}

fn do_get_linked_ltoir(_handle: HandlePtr, _ltoir: *mut c_void) -> c_int {
    log::warn(format_args!("nvJitLinkGetLinkedLTOIR: not supported by 12.9.2 backend"));
    NVJITLINK_ERROR_INTERNAL
}

fn do_version(major: *mut c_uint, minor: *mut c_uint) -> c_int {
    if major.is_null() || minor.is_null() {
        return NVJITLINK_ERROR_INVALID_INPUT;
    }
    // Safety: pointers are not null.
    unsafe {
        *major = 13;
        *minor = 5;
    }
    NVJITLINK_SUCCESS
}

// --- Exported functions ---
// Every function uses #[unsafe(no_mangle)] as required by Rust edition 2024.
// Versioned aliases (_13_0 through _13_5) all call the same implementation.

// Macros for generating versioned aliases.

macro_rules! versioned_create {
    ($name:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(handle: *mut HandlePtr, num_options: c_uint, options: *const *const c_char) -> c_int {
            do_create(handle, num_options, options)
        }
    };
}

macro_rules! versioned_destroy {
    ($name:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(handle: *mut HandlePtr) -> c_int {
            do_destroy(handle)
        }
    };
}

macro_rules! versioned_add_data {
    ($name:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(
            handle: HandlePtr,
            input_type: c_int,
            data: *const c_void,
            size: usize,
            name: *const c_char,
        ) -> c_int {
            do_add_data(handle, input_type, data, size, name)
        }
    };
}

macro_rules! versioned_add_file {
    ($name:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(handle: HandlePtr, input_type: c_int, file_name: *const c_char) -> c_int {
            do_add_file(handle, input_type, file_name)
        }
    };
}

macro_rules! versioned_simple {
    ($name:ident, $impl_fn:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(handle: HandlePtr) -> c_int {
            $impl_fn(handle)
        }
    };
}

macro_rules! versioned_get_size {
    ($name:ident, $impl_fn:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(handle: HandlePtr, size: *mut usize) -> c_int {
            $impl_fn(handle, size)
        }
    };
}

macro_rules! versioned_get_buf {
    ($name:ident, $impl_fn:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(handle: HandlePtr, buf: *mut c_void) -> c_int {
            $impl_fn(handle, buf)
        }
    };
}

macro_rules! versioned_get_char_buf {
    ($name:ident, $impl_fn:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(handle: HandlePtr, buf: *mut c_char) -> c_int {
            $impl_fn(handle, buf)
        }
    };
}

// --- Create ---

versioned_create!(nvJitLinkCreate);
versioned_create!(__nvJitLinkCreate_13_0);
versioned_create!(__nvJitLinkCreate_13_1);
versioned_create!(__nvJitLinkCreate_13_2);
versioned_create!(__nvJitLinkCreate_13_3);
versioned_create!(__nvJitLinkCreate_13_4);
versioned_create!(__nvJitLinkCreate_13_5);

// --- Destroy ---

versioned_destroy!(nvJitLinkDestroy);
versioned_destroy!(__nvJitLinkDestroy_13_0);
versioned_destroy!(__nvJitLinkDestroy_13_1);
versioned_destroy!(__nvJitLinkDestroy_13_2);
versioned_destroy!(__nvJitLinkDestroy_13_3);
versioned_destroy!(__nvJitLinkDestroy_13_4);
versioned_destroy!(__nvJitLinkDestroy_13_5);

// --- AddData ---

versioned_add_data!(nvJitLinkAddData);
versioned_add_data!(__nvJitLinkAddData_13_0);
versioned_add_data!(__nvJitLinkAddData_13_1);
versioned_add_data!(__nvJitLinkAddData_13_2);
versioned_add_data!(__nvJitLinkAddData_13_3);
versioned_add_data!(__nvJitLinkAddData_13_4);
versioned_add_data!(__nvJitLinkAddData_13_5);

// --- AddFile ---

versioned_add_file!(nvJitLinkAddFile);
versioned_add_file!(__nvJitLinkAddFile_13_0);
versioned_add_file!(__nvJitLinkAddFile_13_1);
versioned_add_file!(__nvJitLinkAddFile_13_2);
versioned_add_file!(__nvJitLinkAddFile_13_3);
versioned_add_file!(__nvJitLinkAddFile_13_4);
versioned_add_file!(__nvJitLinkAddFile_13_5);

// --- Complete ---

versioned_simple!(nvJitLinkComplete, do_complete);
versioned_simple!(__nvJitLinkComplete_13_0, do_complete);
versioned_simple!(__nvJitLinkComplete_13_1, do_complete);
versioned_simple!(__nvJitLinkComplete_13_2, do_complete);
versioned_simple!(__nvJitLinkComplete_13_3, do_complete);
versioned_simple!(__nvJitLinkComplete_13_4, do_complete);
versioned_simple!(__nvJitLinkComplete_13_5, do_complete);

// --- GetLinkedCubinSize ---

versioned_get_size!(nvJitLinkGetLinkedCubinSize, do_get_linked_cubin_size);
versioned_get_size!(__nvJitLinkGetLinkedCubinSize_13_0, do_get_linked_cubin_size);
versioned_get_size!(__nvJitLinkGetLinkedCubinSize_13_1, do_get_linked_cubin_size);
versioned_get_size!(__nvJitLinkGetLinkedCubinSize_13_2, do_get_linked_cubin_size);
versioned_get_size!(__nvJitLinkGetLinkedCubinSize_13_3, do_get_linked_cubin_size);
versioned_get_size!(__nvJitLinkGetLinkedCubinSize_13_4, do_get_linked_cubin_size);
versioned_get_size!(__nvJitLinkGetLinkedCubinSize_13_5, do_get_linked_cubin_size);

// --- GetLinkedCubin ---

versioned_get_buf!(nvJitLinkGetLinkedCubin, do_get_linked_cubin);
versioned_get_buf!(__nvJitLinkGetLinkedCubin_13_0, do_get_linked_cubin);
versioned_get_buf!(__nvJitLinkGetLinkedCubin_13_1, do_get_linked_cubin);
versioned_get_buf!(__nvJitLinkGetLinkedCubin_13_2, do_get_linked_cubin);
versioned_get_buf!(__nvJitLinkGetLinkedCubin_13_3, do_get_linked_cubin);
versioned_get_buf!(__nvJitLinkGetLinkedCubin_13_4, do_get_linked_cubin);
versioned_get_buf!(__nvJitLinkGetLinkedCubin_13_5, do_get_linked_cubin);

// --- GetLinkedPtxSize ---

versioned_get_size!(nvJitLinkGetLinkedPtxSize, do_get_linked_ptx_size);
versioned_get_size!(__nvJitLinkGetLinkedPtxSize_13_0, do_get_linked_ptx_size);
versioned_get_size!(__nvJitLinkGetLinkedPtxSize_13_1, do_get_linked_ptx_size);
versioned_get_size!(__nvJitLinkGetLinkedPtxSize_13_2, do_get_linked_ptx_size);
versioned_get_size!(__nvJitLinkGetLinkedPtxSize_13_3, do_get_linked_ptx_size);
versioned_get_size!(__nvJitLinkGetLinkedPtxSize_13_4, do_get_linked_ptx_size);
versioned_get_size!(__nvJitLinkGetLinkedPtxSize_13_5, do_get_linked_ptx_size);

// --- GetLinkedPtx ---

versioned_get_char_buf!(nvJitLinkGetLinkedPtx, do_get_linked_ptx);
versioned_get_char_buf!(__nvJitLinkGetLinkedPtx_13_0, do_get_linked_ptx);
versioned_get_char_buf!(__nvJitLinkGetLinkedPtx_13_1, do_get_linked_ptx);
versioned_get_char_buf!(__nvJitLinkGetLinkedPtx_13_2, do_get_linked_ptx);
versioned_get_char_buf!(__nvJitLinkGetLinkedPtx_13_3, do_get_linked_ptx);
versioned_get_char_buf!(__nvJitLinkGetLinkedPtx_13_4, do_get_linked_ptx);
versioned_get_char_buf!(__nvJitLinkGetLinkedPtx_13_5, do_get_linked_ptx);

// --- GetLinkedLTOIRSize ---

versioned_get_size!(nvJitLinkGetLinkedLTOIRSize, do_get_linked_ltoir_size);
versioned_get_size!(__nvJitLinkGetLinkedLTOIRSize_13_0, do_get_linked_ltoir_size);
versioned_get_size!(__nvJitLinkGetLinkedLTOIRSize_13_1, do_get_linked_ltoir_size);
versioned_get_size!(__nvJitLinkGetLinkedLTOIRSize_13_2, do_get_linked_ltoir_size);
versioned_get_size!(__nvJitLinkGetLinkedLTOIRSize_13_3, do_get_linked_ltoir_size);
versioned_get_size!(__nvJitLinkGetLinkedLTOIRSize_13_4, do_get_linked_ltoir_size);
versioned_get_size!(__nvJitLinkGetLinkedLTOIRSize_13_5, do_get_linked_ltoir_size);

// --- GetLinkedLTOIR ---

versioned_get_buf!(nvJitLinkGetLinkedLTOIR, do_get_linked_ltoir);
versioned_get_buf!(__nvJitLinkGetLinkedLTOIR_13_0, do_get_linked_ltoir);
versioned_get_buf!(__nvJitLinkGetLinkedLTOIR_13_1, do_get_linked_ltoir);
versioned_get_buf!(__nvJitLinkGetLinkedLTOIR_13_2, do_get_linked_ltoir);
versioned_get_buf!(__nvJitLinkGetLinkedLTOIR_13_3, do_get_linked_ltoir);
versioned_get_buf!(__nvJitLinkGetLinkedLTOIR_13_4, do_get_linked_ltoir);
versioned_get_buf!(__nvJitLinkGetLinkedLTOIR_13_5, do_get_linked_ltoir);

// --- GetErrorLogSize ---

versioned_get_size!(nvJitLinkGetErrorLogSize, do_get_error_log_size);
versioned_get_size!(__nvJitLinkGetErrorLogSize_13_0, do_get_error_log_size);
versioned_get_size!(__nvJitLinkGetErrorLogSize_13_1, do_get_error_log_size);
versioned_get_size!(__nvJitLinkGetErrorLogSize_13_2, do_get_error_log_size);
versioned_get_size!(__nvJitLinkGetErrorLogSize_13_3, do_get_error_log_size);
versioned_get_size!(__nvJitLinkGetErrorLogSize_13_4, do_get_error_log_size);
versioned_get_size!(__nvJitLinkGetErrorLogSize_13_5, do_get_error_log_size);

// --- GetErrorLog ---

versioned_get_char_buf!(nvJitLinkGetErrorLog, do_get_error_log);
versioned_get_char_buf!(__nvJitLinkGetErrorLog_13_0, do_get_error_log);
versioned_get_char_buf!(__nvJitLinkGetErrorLog_13_1, do_get_error_log);
versioned_get_char_buf!(__nvJitLinkGetErrorLog_13_2, do_get_error_log);
versioned_get_char_buf!(__nvJitLinkGetErrorLog_13_3, do_get_error_log);
versioned_get_char_buf!(__nvJitLinkGetErrorLog_13_4, do_get_error_log);
versioned_get_char_buf!(__nvJitLinkGetErrorLog_13_5, do_get_error_log);

// --- GetInfoLogSize ---

versioned_get_size!(nvJitLinkGetInfoLogSize, do_get_info_log_size);
versioned_get_size!(__nvJitLinkGetInfoLogSize_13_0, do_get_info_log_size);
versioned_get_size!(__nvJitLinkGetInfoLogSize_13_1, do_get_info_log_size);
versioned_get_size!(__nvJitLinkGetInfoLogSize_13_2, do_get_info_log_size);
versioned_get_size!(__nvJitLinkGetInfoLogSize_13_3, do_get_info_log_size);
versioned_get_size!(__nvJitLinkGetInfoLogSize_13_4, do_get_info_log_size);
versioned_get_size!(__nvJitLinkGetInfoLogSize_13_5, do_get_info_log_size);

// --- GetInfoLog ---

versioned_get_char_buf!(nvJitLinkGetInfoLog, do_get_info_log);
versioned_get_char_buf!(__nvJitLinkGetInfoLog_13_0, do_get_info_log);
versioned_get_char_buf!(__nvJitLinkGetInfoLog_13_1, do_get_info_log);
versioned_get_char_buf!(__nvJitLinkGetInfoLog_13_2, do_get_info_log);
versioned_get_char_buf!(__nvJitLinkGetInfoLog_13_3, do_get_info_log);
versioned_get_char_buf!(__nvJitLinkGetInfoLog_13_4, do_get_info_log);
versioned_get_char_buf!(__nvJitLinkGetInfoLog_13_5, do_get_info_log);

// --- Version ---

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkVersion(major: *mut c_uint, minor: *mut c_uint) -> c_int {
    do_version(major, minor)
}

// --- __nvJitLinkAPI ---
// The real library exports this (some internal table). Forward to 12.9.2.

#[unsafe(no_mangle)]
pub extern "C" fn __nvJitLinkAPI() -> *const c_void {
    let lib = match real() {
        Ok(l) => l,
        Err(_) => return ptr::null(),
    };
    // Call the real library's __nvJitLinkAPI through dlsym.
    let name = CString::new("__nvJitLinkAPI").unwrap();
    let sym = unsafe { libc::dlsym(lib._handle, name.as_ptr()) };
    if sym.is_null() {
        return ptr::null();
    }
    let f: unsafe extern "C" fn() -> *const c_void =
        unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> *const c_void>(sym) };
    unsafe { f() }
}

// --- Unit tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_rewriting_changes_arch() {
        let opts = [
            CString::new("-arch=sm_75").unwrap(),
            CString::new("-maxrregcount=32").unwrap(),
            CString::new("-lto").unwrap(),
        ];
        let ptrs: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();
        let (rewritten, orig) = rewrite_options(ptrs.len() as c_uint, ptrs.as_ptr()).unwrap();
        assert_eq!(orig, Some("sm_75".to_string()));
        assert_eq!(rewritten.len(), 3);
        assert_eq!(rewritten[0].to_str().unwrap(), "-arch=sm_70");
        assert_eq!(rewritten[1].to_str().unwrap(), "-maxrregcount=32");
        assert_eq!(rewritten[2].to_str().unwrap(), "-lto");
    }

    #[test]
    fn option_rewriting_drops_new_options() {
        let opts = [
            CString::new("-arch=sm_90").unwrap(),
            CString::new("-Ofc=max").unwrap(),
            CString::new("-nvvm-latest").unwrap(),
            CString::new("-device-stack-protector").unwrap(),
            CString::new("-no-cache").unwrap(),
            CString::new("-verbose").unwrap(),
        ];
        let ptrs: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();
        let (rewritten, _) = rewrite_options(ptrs.len() as c_uint, ptrs.as_ptr()).unwrap();
        // Only -arch=sm_70 and -verbose survive.
        assert_eq!(rewritten.len(), 2);
        assert_eq!(rewritten[0].to_str().unwrap(), "-arch=sm_70");
        assert_eq!(rewritten[1].to_str().unwrap(), "-verbose");
    }

    #[test]
    fn option_rewriting_preserves_sm_70() {
        let opts = [CString::new("-arch=sm_70").unwrap()];
        let ptrs: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();
        let (rewritten, orig) = rewrite_options(ptrs.len() as c_uint, ptrs.as_ptr()).unwrap();
        assert_eq!(orig, None);
        assert_eq!(rewritten[0].to_str().unwrap(), "-arch=sm_70");
    }

    #[test]
    fn option_rewriting_takes_the_number_of_an_arch_variant() {
        for (given, wanted) in [
            ("-arch=sm_90a", "-arch=sm_70"),
            ("-arch=sm_100f", "-arch=sm_70"),
            ("-arch=sm_70", "-arch=sm_70"),
        ] {
            let option = CString::new(given).unwrap();
            let ptrs = [option.as_ptr()];
            let (rewritten, _) = rewrite_options(ptrs.len() as c_uint, ptrs.as_ptr()).unwrap();
            assert_eq!(rewritten[0].to_str().unwrap(), wanted, "{given}");
        }
    }

    #[test]
    fn option_rewriting_compute_arch() {
        let opts = [CString::new("-arch=compute_75").unwrap()];
        let ptrs: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();
        let (rewritten, orig) = rewrite_options(ptrs.len() as c_uint, ptrs.as_ptr()).unwrap();
        assert_eq!(orig, Some("compute_75".to_string()));
        assert_eq!(rewritten[0].to_str().unwrap(), "-arch=compute_70");
    }

    #[test]
    fn ptx_adaptation_uses_88_limit() {
        let policy = jitlink_policy();
        assert_eq!(policy.ptx_max, (8, 8));
        assert_eq!(policy.device_sm, 70);
        // PTX version 9.5 should be lowered to 8.8, not 9.0.
        let adapted = policy.adapt_ptx(b".version 9.5\n.target sm_75\n.address_size 64\n");
        assert_eq!(
            adapted.image.unwrap(),
            b".version 8.8\n.target sm_70\n.address_size 64\n"
        );
    }

    /// A fat binary with one uncompressed PTX entry for `sm`, as fatbinary writes it.
    fn ptx_fatbin(text: &[u8], sm: u32) -> Vec<u8> {
        let mut payload = text.to_vec();
        payload.push(0);
        payload.resize(payload.len().next_multiple_of(8), 0);
        let mut header = [0u8; 80];
        header[0..2].copy_from_slice(&vf_fatbin::KIND_PTX.to_le_bytes());
        header[2..4].copy_from_slice(&0x0101u16.to_le_bytes());
        header[4..8].copy_from_slice(&80u32.to_le_bytes());
        header[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        header[0x18..0x1a].copy_from_slice(&8u16.to_le_bytes());
        header[0x1a..0x1c].copy_from_slice(&8u16.to_le_bytes());
        header[0x1c..0x20].copy_from_slice(&sm.to_le_bytes());
        header[0x28..0x30].copy_from_slice(&0x11u64.to_le_bytes());
        let mut fat = vf_fatbin::MAGIC.to_le_bytes().to_vec();
        fat.extend_from_slice(&1u16.to_le_bytes());
        fat.extend_from_slice(&16u16.to_le_bytes());
        fat.extend_from_slice(&((header.len() + payload.len()) as u64).to_le_bytes());
        fat.extend_from_slice(&header);
        fat.extend_from_slice(&payload);
        fat
    }

    const BAD: &str = ".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n";
    const BAD2: &str = ".visible .entry bad2()\n{\n barrier.cluster.wait;\n ret;\n}\n";
    const GOOD: &str = ".visible .entry good()\n{\n ret;\n}\n";

    #[test]
    fn ptx_whose_kernels_all_trap_is_refused_as_text_and_in_a_fat_binary() {
        // (Until 2026-11-23 the fat binary was linked, into machine code that only traps.)
        for sm in [75, 90] {
            let only_bad = format!(".version 8.8\n.target sm_{sm}\n.address_size 64\n{BAD}{BAD2}");
            assert_eq!(
                adapt_ptx_data(only_bad.as_bytes()).unwrap_err(),
                NVJITLINK_ERROR_PTX_COMPILE
            );
            let fat = ptx_fatbin(only_bad.as_bytes(), sm);
            assert_eq!(
                check_fatbin_data(&fat).unwrap_err(),
                NVJITLINK_ERROR_PTX_COMPILE,
                "sm_{sm}"
            );

            // One kernel that runs: both compile, the other kernel as a trap.
            let half = format!(".version 8.8\n.target sm_{sm}\n.address_size 64\n{BAD}{GOOD}");
            // The kernel that traps is given back, for the note of the cubin that the link makes.
            let names = |refused: &[vf_adapt::Refused]| refused.iter().map(|r| r.kernel.clone()).collect::<Vec<_>>();
            assert_eq!(names(&adapt_ptx_data(half.as_bytes()).unwrap().1), ["bad"]);
            let (adapted, refused) = check_fatbin_data(&ptx_fatbin(half.as_bytes(), sm)).unwrap();
            assert_eq!(
                fatbin_kernels(&adapted.unwrap(), vf_fatbin::KIND_PTX, 70),
                ["bad", "good"]
            );
            assert_eq!(names(&refused), ["bad"]);
        }
    }

    #[test]
    fn a_file_whose_name_is_not_utf8_is_read_by_its_name() {
        // (Until 2026-11-23 the name was made into text: the byte 0xff became U+FFFD, and a file of
        // that name was read in its place.)
        use std::os::unix::ffi::OsStrExt as _;
        let dir = std::env::temp_dir().join(format!("vf-jitlink-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut name = dir.as_os_str().as_bytes().to_vec();
        name.extend_from_slice(b"/k\xff.ptx");
        let path = std::path::PathBuf::from(std::ffi::OsStr::from_bytes(&name));
        std::fs::write(&path, b"the file asked for").unwrap();
        std::fs::write(dir.join("k\u{fffd}.ptx"), b"another file").unwrap();
        let read = read_file(&CString::new(name).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(read.unwrap(), b"the file asked for");
    }

    #[test]
    fn ptx_88_sm70_is_native() {
        let policy = jitlink_policy();
        let adapted = policy.adapt_ptx(b".version 8.8\n.target sm_70\n");
        assert_eq!(adapted.outcome, vf_adapt::Outcome::Native);
        assert!(adapted.image.is_none());
    }

    #[test]
    fn version_reports_13_5() {
        let mut major = 0u32;
        let mut minor = 0u32;
        let result = do_version(&mut major, &mut minor);
        assert_eq!(result, NVJITLINK_SUCCESS);
        assert_eq!(major, 13);
        assert_eq!(minor, 5);
    }

    #[test]
    fn null_options_are_invalid_input() {
        // (Until 2026-11-22 a null array was read through, and so was a null option.)
        assert_eq!(
            rewrite_options(1, ptr::null()).unwrap_err(),
            NVJITLINK_ERROR_INVALID_INPUT
        );
        assert_eq!(
            rewrite_options(0, ptr::null()).unwrap_err(),
            NVJITLINK_ERROR_INVALID_INPUT
        );
        let arch = CString::new("-arch=sm_75").unwrap();
        let ptrs = [arch.as_ptr(), ptr::null()];
        assert_eq!(
            rewrite_options(2, ptrs.as_ptr()).unwrap_err(),
            NVJITLINK_ERROR_INVALID_INPUT
        );
        assert_eq!(rewrite_options(0, ptrs.as_ptr()).unwrap().0.len(), 0);
    }

    #[test]
    fn option_r_returns_error() {
        let opts = [CString::new("-arch=sm_75").unwrap(), CString::new("-r").unwrap()];
        let ptrs: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();
        let result = rewrite_options(ptrs.len() as c_uint, ptrs.as_ptr());
        assert_eq!(result.unwrap_err(), NVJITLINK_ERROR_UNRECOGNIZED_OPTION);
    }

    #[test]
    fn option_r_does_not_match_prefix() {
        // "-rdc=true" must NOT be caught by the exact-match "-r" check.
        // (12.9.2 may not know -rdc, but the point is it should not be caught as "-r".)
        let opts = [CString::new("-arch=sm_70").unwrap(), CString::new("-rdc=true").unwrap()];
        let ptrs: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();
        let (rewritten, _) = rewrite_options(ptrs.len() as c_uint, ptrs.as_ptr()).unwrap();
        assert_eq!(rewritten.len(), 2);
        assert_eq!(rewritten[1].to_str().unwrap(), "-rdc=true");
    }

    #[test]
    fn option_rewriting_passes_known_options() {
        let opts = [
            CString::new("-arch=sm_75").unwrap(),
            CString::new("-maxrregcount=32").unwrap(),
            CString::new("-time").unwrap(),
            CString::new("-verbose").unwrap(),
            CString::new("-lto").unwrap(),
            CString::new("-ptx").unwrap(),
            CString::new("-O3").unwrap(),
            CString::new("-g").unwrap(),
            CString::new("-lineinfo").unwrap(),
            CString::new("-ftz=1").unwrap(),
            CString::new("-prec-div=0").unwrap(),
            CString::new("-prec-sqrt=0").unwrap(),
            CString::new("-fma=1").unwrap(),
            CString::new("-kernels-used=foo").unwrap(),
            CString::new("-variables-used=bar").unwrap(),
            CString::new("-optimize-unused-variables").unwrap(),
            CString::new("-Xptxas=-v").unwrap(),
            CString::new("-split-compile=4").unwrap(),
            CString::new("-split-compile-extended=4").unwrap(),
            CString::new("-jump-table-density=50").unwrap(),
        ];
        let ptrs: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();
        let (rewritten, _) = rewrite_options(ptrs.len() as c_uint, ptrs.as_ptr()).unwrap();
        // All should pass through (arch rewritten).
        assert_eq!(rewritten.len(), opts.len());
        assert_eq!(rewritten[0].to_str().unwrap(), "-arch=sm_70");
        for (i, opt) in opts.iter().enumerate().skip(1) {
            assert_eq!(rewritten[i].to_str().unwrap(), opt.to_str().unwrap());
        }
    }
}

// --- Symbol versioning ---
// Bind every exported symbol to the `libnvJitLink.so.13` version node so that a program
// linked against the real CUDA 13.5 libnvJitLink.so.13 resolves its versioned references.

macro_rules! symver {
    ($sym:ident) => {
        std::arch::global_asm!(concat!(
            ".symver ",
            stringify!($sym),
            ",",
            stringify!($sym),
            "@@libnvJitLink.so.13"
        ));
    };
}

symver!(nvJitLinkCreate);
symver!(__nvJitLinkCreate_13_0);
symver!(__nvJitLinkCreate_13_1);
symver!(__nvJitLinkCreate_13_2);
symver!(__nvJitLinkCreate_13_3);
symver!(__nvJitLinkCreate_13_4);
symver!(__nvJitLinkCreate_13_5);
symver!(nvJitLinkDestroy);
symver!(__nvJitLinkDestroy_13_0);
symver!(__nvJitLinkDestroy_13_1);
symver!(__nvJitLinkDestroy_13_2);
symver!(__nvJitLinkDestroy_13_3);
symver!(__nvJitLinkDestroy_13_4);
symver!(__nvJitLinkDestroy_13_5);
symver!(nvJitLinkAddData);
symver!(__nvJitLinkAddData_13_0);
symver!(__nvJitLinkAddData_13_1);
symver!(__nvJitLinkAddData_13_2);
symver!(__nvJitLinkAddData_13_3);
symver!(__nvJitLinkAddData_13_4);
symver!(__nvJitLinkAddData_13_5);
symver!(nvJitLinkAddFile);
symver!(__nvJitLinkAddFile_13_0);
symver!(__nvJitLinkAddFile_13_1);
symver!(__nvJitLinkAddFile_13_2);
symver!(__nvJitLinkAddFile_13_3);
symver!(__nvJitLinkAddFile_13_4);
symver!(__nvJitLinkAddFile_13_5);
symver!(nvJitLinkComplete);
symver!(__nvJitLinkComplete_13_0);
symver!(__nvJitLinkComplete_13_1);
symver!(__nvJitLinkComplete_13_2);
symver!(__nvJitLinkComplete_13_3);
symver!(__nvJitLinkComplete_13_4);
symver!(__nvJitLinkComplete_13_5);
symver!(nvJitLinkGetLinkedCubinSize);
symver!(__nvJitLinkGetLinkedCubinSize_13_0);
symver!(__nvJitLinkGetLinkedCubinSize_13_1);
symver!(__nvJitLinkGetLinkedCubinSize_13_2);
symver!(__nvJitLinkGetLinkedCubinSize_13_3);
symver!(__nvJitLinkGetLinkedCubinSize_13_4);
symver!(__nvJitLinkGetLinkedCubinSize_13_5);
symver!(nvJitLinkGetLinkedCubin);
symver!(__nvJitLinkGetLinkedCubin_13_0);
symver!(__nvJitLinkGetLinkedCubin_13_1);
symver!(__nvJitLinkGetLinkedCubin_13_2);
symver!(__nvJitLinkGetLinkedCubin_13_3);
symver!(__nvJitLinkGetLinkedCubin_13_4);
symver!(__nvJitLinkGetLinkedCubin_13_5);
symver!(nvJitLinkGetLinkedPtxSize);
symver!(__nvJitLinkGetLinkedPtxSize_13_0);
symver!(__nvJitLinkGetLinkedPtxSize_13_1);
symver!(__nvJitLinkGetLinkedPtxSize_13_2);
symver!(__nvJitLinkGetLinkedPtxSize_13_3);
symver!(__nvJitLinkGetLinkedPtxSize_13_4);
symver!(__nvJitLinkGetLinkedPtxSize_13_5);
symver!(nvJitLinkGetLinkedPtx);
symver!(__nvJitLinkGetLinkedPtx_13_0);
symver!(__nvJitLinkGetLinkedPtx_13_1);
symver!(__nvJitLinkGetLinkedPtx_13_2);
symver!(__nvJitLinkGetLinkedPtx_13_3);
symver!(__nvJitLinkGetLinkedPtx_13_4);
symver!(__nvJitLinkGetLinkedPtx_13_5);
symver!(nvJitLinkGetLinkedLTOIRSize);
symver!(__nvJitLinkGetLinkedLTOIRSize_13_0);
symver!(__nvJitLinkGetLinkedLTOIRSize_13_1);
symver!(__nvJitLinkGetLinkedLTOIRSize_13_2);
symver!(__nvJitLinkGetLinkedLTOIRSize_13_3);
symver!(__nvJitLinkGetLinkedLTOIRSize_13_4);
symver!(__nvJitLinkGetLinkedLTOIRSize_13_5);
symver!(nvJitLinkGetLinkedLTOIR);
symver!(__nvJitLinkGetLinkedLTOIR_13_0);
symver!(__nvJitLinkGetLinkedLTOIR_13_1);
symver!(__nvJitLinkGetLinkedLTOIR_13_2);
symver!(__nvJitLinkGetLinkedLTOIR_13_3);
symver!(__nvJitLinkGetLinkedLTOIR_13_4);
symver!(__nvJitLinkGetLinkedLTOIR_13_5);
symver!(nvJitLinkGetErrorLogSize);
symver!(__nvJitLinkGetErrorLogSize_13_0);
symver!(__nvJitLinkGetErrorLogSize_13_1);
symver!(__nvJitLinkGetErrorLogSize_13_2);
symver!(__nvJitLinkGetErrorLogSize_13_3);
symver!(__nvJitLinkGetErrorLogSize_13_4);
symver!(__nvJitLinkGetErrorLogSize_13_5);
symver!(nvJitLinkGetErrorLog);
symver!(__nvJitLinkGetErrorLog_13_0);
symver!(__nvJitLinkGetErrorLog_13_1);
symver!(__nvJitLinkGetErrorLog_13_2);
symver!(__nvJitLinkGetErrorLog_13_3);
symver!(__nvJitLinkGetErrorLog_13_4);
symver!(__nvJitLinkGetErrorLog_13_5);
symver!(nvJitLinkGetInfoLogSize);
symver!(__nvJitLinkGetInfoLogSize_13_0);
symver!(__nvJitLinkGetInfoLogSize_13_1);
symver!(__nvJitLinkGetInfoLogSize_13_2);
symver!(__nvJitLinkGetInfoLogSize_13_3);
symver!(__nvJitLinkGetInfoLogSize_13_4);
symver!(__nvJitLinkGetInfoLogSize_13_5);
symver!(nvJitLinkGetInfoLog);
symver!(__nvJitLinkGetInfoLog_13_0);
symver!(__nvJitLinkGetInfoLog_13_1);
symver!(__nvJitLinkGetInfoLog_13_2);
symver!(__nvJitLinkGetInfoLog_13_3);
symver!(__nvJitLinkGetInfoLog_13_4);
symver!(__nvJitLinkGetInfoLog_13_5);
symver!(nvJitLinkVersion);
symver!(__nvJitLinkAPI);
