//! Kernel-level profiler for the proxy, enabled by `VF_PROFILE=FILE`.
//!
//! When on, every kernel launch is followed by a synchronisation of its stream
//! (`cuStreamSynchronize`), so that the time of the kernel is known: without it
//! `cuLaunchKernel` returns before the GPU finishes. This makes all launches serial within a
//! stream and blocks the CPU until the GPU is done, which changes how the program runs: a profile
//! is not a benchmark, it is a measurement of the kernels taken one at a time.
//!
//! The time of a launch is measured from the return of the launch call to the return of
//! `cuStreamSynchronize`: what the host waits for after the driver accepted the launch. It
//! includes anything queued in the stream before the kernel, and it does not include work the
//! launch call itself does: a first launch that makes the driver compile the kernel (lazy loading)
//! spends that time inside the call, before the timer starts.
//!
//! For every kernel that was launched at least once, the output file has a line with these fields
//! separated by tabs: kernel name, code origin (native / ptx-rewritten / machine-code-translated
//! / cache-hit), number of launches, total time in milliseconds, mean time in milliseconds, the
//! grid sizes seen, the block sizes seen (the most frequent of the first 64 distinct sizes, which
//! are all that is kept), the emulated instruction summary (the same text the layer prints when it
//! announces a kernel).
//!
//! One file per process: the PID is appended to the path (`FILE.PID`). The file is written when
//! the process ends (the `.fini_array` function).
//!
//! The first synchronisation that returns an error (a kernel that faulted ends the context, and
//! every call after it returns the same error) is said at once, with the kernel it followed: the
//! launches that the profile synchronised before ended without one.

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt::Write as FmtWrite;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use vf_ptx::EmulatedCounts;

use crate::{REALS, idx, log};

#[allow(dead_code)]
/// Per-kernel profiling state.
struct KernelProfile {
    /// How the code reached the device.
    origin: CodeOrigin,
    /// Emulated instruction counts, if any.
    emulated: Option<EmulatedCounts>,
    /// Number of launches.
    launches: u64,
    /// Total time from the return of the launch call to the return of `cuStreamSynchronize`
    /// (a compilation the driver does inside a first launch call is not in it).
    total_ns: u128,
    /// Grid sizes seen: `(gx, gy, gz)` -> count.
    grids: HashMap<(u32, u32, u32), u64>,
    /// Block sizes seen: `(bx, by, bz)` -> count.
    blocks: HashMap<(u32, u32, u32), u64>,
    /// Function attributes queried from the driver (once per function handle).
    func_attrs: Option<FuncAttrs>,
    /// Dynamic shared memory sizes seen at launch: bytes -> count.
    dynamic_smem: HashMap<u32, u64>,
}

/// Function attributes queried from the driver via `cuFuncGetAttribute`.
#[derive(Debug, Clone, Copy)]
struct FuncAttrs {
    /// Number of registers per thread (`CU_FUNC_ATTRIBUTE_NUM_REGS`).
    num_regs: u32,
    /// Static shared memory in bytes (`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`).
    static_smem_bytes: u32,
    /// Maximum threads per block (`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`).
    #[allow(dead_code)]
    max_threads_per_block: u32,
}

/// V100 (sm_70) multiprocessor limits (from the CUDA programming guide and `cuDeviceGetAttribute`).
/// These are verified at profiler initialization by querying the device.
struct SmLimits {
    /// Total shared memory per multiprocessor in bytes.
    shared_mem_per_sm: u32,
    /// Total registers per multiprocessor.
    regs_per_sm: u32,
    /// Maximum blocks per multiprocessor.
    max_blocks_per_sm: u32,
    /// Maximum warps per multiprocessor.
    max_warps_per_sm: u32,
    /// Register allocation granularity (threads per allocation unit).
    warp_alloc_granularity: u32,
    /// Shared memory allocation granularity in bytes.
    smem_alloc_granularity: u32,
}

impl SmLimits {
    /// V100 defaults. Overridden by `query_device_limits` when the device is available.
    const fn v100_defaults() -> Self {
        SmLimits {
            shared_mem_per_sm: 96 * 1024, // 96 KB
            regs_per_sm: 65536,
            max_blocks_per_sm: 32,
            max_warps_per_sm: 64,
            warp_alloc_granularity: 8,   // registers allocated in units of 8
            smem_alloc_granularity: 256, // shared memory allocated in units of 256 bytes
        }
    }
}

/// Occupancy result for one kernel launch on one multiprocessor.
#[derive(Debug, Clone, Copy)]
struct Occupancy {
    /// Maximum blocks of this launch that fit on one multiprocessor.
    blocks_per_sm: u32,
    /// What limits the number of blocks (the tightest constraint).
    limiting_factor: LimitingFactor,
    /// The number of blocks each individual limit gives.
    by_registers: u32,
    by_shared_memory: u32,
    by_block_limit: u32,
    by_warp_limit: u32,
}

/// What limits a kernel's blocks per multiprocessor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LimitingFactor {
    Registers,
    SharedMemory,
    BlockLimit,
    WarpLimit,
    /// Two or more limits tie.
    Tied(u8), // bitmask: 1=regs, 2=smem, 4=blocks, 8=warps
}

impl std::fmt::Display for LimitingFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LimitingFactor::Registers => f.write_str("registers"),
            LimitingFactor::SharedMemory => f.write_str("shared_memory"),
            LimitingFactor::BlockLimit => f.write_str("block_limit"),
            LimitingFactor::WarpLimit => f.write_str("warp_limit"),
            LimitingFactor::Tied(mask) => {
                let mut parts = Vec::new();
                if mask & 1 != 0 {
                    parts.push("registers");
                }
                if mask & 2 != 0 {
                    parts.push("shared_memory");
                }
                if mask & 4 != 0 {
                    parts.push("block_limit");
                }
                if mask & 8 != 0 {
                    parts.push("warp_limit");
                }
                f.write_str(&parts.join("+"))
            }
        }
    }
}

/// How a kernel's code reached the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeOrigin {
    /// Already for this GPU (sm_70 code in the image).
    Native,
    /// PTX for a later GPU, rewritten for Volta by the layer.
    PtxRewritten,
    /// Machine code translated to sm_70 in the proxy.
    MachineCodeTranslated,
    /// Taken from the proxy's cache (may be PTX-rewritten or translated).
    CacheHit,
    /// Not known (the lookup did not go through a path the profiler tracks).
    Unknown,
}

impl std::fmt::Display for CodeOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeOrigin::Native => f.write_str("native"),
            CodeOrigin::PtxRewritten => f.write_str("ptx-rewritten"),
            CodeOrigin::MachineCodeTranslated => f.write_str("machine-code-translated"),
            CodeOrigin::CacheHit => f.write_str("cache-hit"),
            CodeOrigin::Unknown => f.write_str("unknown"),
        }
    }
}

/// Global profiling state: `None` when profiling is off.
static STATE: LazyLock<Mutex<Option<ProfileState>>> = LazyLock::new(|| Mutex::new(None));

/// Kernel function handle -> function attributes (queried once from the driver).
static FUNC_ATTRS: LazyLock<Mutex<HashMap<usize, FuncAttrs>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// The multiprocessor limits of the device, queried once at initialization.
static SM_LIMITS: LazyLock<SmLimits> = LazyLock::new(|| query_device_limits().unwrap_or(SmLimits::v100_defaults()));

struct ProfileState {
    path: String,
    kernels: HashMap<String, KernelProfile>,
}

/// `CU_FUNC_ATTRIBUTE` values.
const FUNC_ATTR_MAX_THREADS_PER_BLOCK: c_int = 0;
const FUNC_ATTR_SHARED_SIZE_BYTES: c_int = 1;
const FUNC_ATTR_NUM_REGS: c_int = 4;

/// `CU_DEVICE_ATTRIBUTE` values.
const DEV_ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: c_int = 81;
const DEV_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR: c_int = 82;
const DEV_ATTR_MAX_BLOCKS_PER_MULTIPROCESSOR: c_int = 106;

/// Query the device limits from the driver. Called once at profiler initialization.
fn query_device_limits() -> Option<SmLimits> {
    let get_attr: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int =
        unsafe { std::mem::transmute::<usize, _>(REALS[idx::cuDeviceGetAttribute].load(Relaxed)) };
    let query = |attr: c_int| -> Option<u32> {
        let mut value: c_int = 0;
        let result = unsafe { get_attr(&mut value, attr, 0) };
        if result == 0 && value > 0 {
            Some(value as u32)
        } else {
            None
        }
    };
    let shared_mem = query(DEV_ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?;
    let regs = query(DEV_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR)?;
    let max_blocks = query(DEV_ATTR_MAX_BLOCKS_PER_MULTIPROCESSOR)?;
    // Max warps: 64 on V100, not a device attribute; use the known value.
    // Max threads per multiprocessor / 32 would be it, but we know it is 64 (2048 threads).
    let max_warps = 64u32;
    log::info(format_args!(
        "profiling: device SM limits: shared_mem={shared_mem} regs={regs} max_blocks={max_blocks} max_warps={max_warps}"
    ));
    Some(SmLimits {
        shared_mem_per_sm: shared_mem,
        regs_per_sm: regs,
        max_blocks_per_sm: max_blocks,
        max_warps_per_sm: max_warps,
        warp_alloc_granularity: 8,   // sm_70: registers allocated in units of 8 per thread
        smem_alloc_granularity: 256, // sm_70: shared memory allocated in units of 256 bytes
    })
}

/// Query the function attributes from the driver for a function handle. Cached per handle.
/// Tries `cuFuncGetAttribute` first (for CUfunction handles), then `cuKernelGetAttribute`
/// (for CUkernel handles from `cuLibraryGetKernel`).
unsafe fn query_func_attrs(function: *mut c_void) -> Option<FuncAttrs> {
    // Check the cache first.
    {
        let cache = FUNC_ATTRS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(attrs) = cache.get(&(function as usize)) {
            return Some(*attrs);
        }
    }
    // Try cuFuncGetAttribute (CUfunction handles).
    let get_func_attr: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void) -> c_int =
        unsafe { std::mem::transmute::<usize, _>(REALS[idx::cuFuncGetAttribute].load(Relaxed)) };
    let query_func = |attr: c_int| -> Option<u32> {
        let mut value: c_int = 0;
        let result = unsafe { get_func_attr(&mut value, attr, function) };
        if result == 0 { Some(value as u32) } else { None }
    };
    let attrs = if let Some(num_regs) = query_func(FUNC_ATTR_NUM_REGS) {
        FuncAttrs {
            num_regs,
            static_smem_bytes: query_func(FUNC_ATTR_SHARED_SIZE_BYTES).unwrap_or(0),
            max_threads_per_block: query_func(FUNC_ATTR_MAX_THREADS_PER_BLOCK).unwrap_or(0),
        }
    } else {
        // Try cuKernelGetAttribute (CUkernel handles, from cuLibraryGetKernel).
        let get_kernel_attr: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void, c_int) -> c_int =
            unsafe { std::mem::transmute::<usize, _>(REALS[idx::cuKernelGetAttribute].load(Relaxed)) };
        let query_kernel = |attr: c_int| -> Option<u32> {
            let mut value: c_int = 0;
            // device 0: the profiler runs on a single GPU.
            let result = unsafe { get_kernel_attr(&mut value, attr, function, 0) };
            if result == 0 { Some(value as u32) } else { None }
        };
        FuncAttrs {
            num_regs: query_kernel(FUNC_ATTR_NUM_REGS)?,
            static_smem_bytes: query_kernel(FUNC_ATTR_SHARED_SIZE_BYTES).unwrap_or(0),
            max_threads_per_block: query_kernel(FUNC_ATTR_MAX_THREADS_PER_BLOCK).unwrap_or(0),
        }
    };
    FUNC_ATTRS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(function as usize, attrs);
    Some(attrs)
}

/// The `limited_by` column. When not even one block fits (`blocks_per_sm` is 0: the launch would
/// fail with out of resources), the limit is followed by `:cannot_launch`, so that a 0 is not
/// mistaken for an unknown value.
fn limited_by_text(occ: &Occupancy) -> String {
    if occ.blocks_per_sm == 0 {
        format!("{}:cannot_launch", occ.limiting_factor)
    } else {
        occ.limiting_factor.to_string()
    }
}

/// Compute occupancy for a kernel launch given function attributes and launch parameters.
fn compute_occupancy(attrs: &FuncAttrs, dynamic_smem: u32, block_threads: u32, limits: &SmLimits) -> Occupancy {
    let warps_per_block = block_threads.div_ceil(32);
    let total_smem_per_block = {
        let raw = attrs.static_smem_bytes + dynamic_smem;
        // Round up to allocation granularity.
        if raw == 0 {
            0
        } else {
            raw.div_ceil(limits.smem_alloc_granularity) * limits.smem_alloc_granularity
        }
    };

    // Register limit, by the formula measured on a V100 (`docs/architecture.md`, "Launch").
    // allocated = ceil(num_regs / 8) * 8
    // per_warp = allocated * 32
    // per_sub = floor(16384 / per_warp) where 16384 = regs_per_sm / 4
    // max_warps_by_regs = per_sub * 4
    // max_blocks_by_regs = floor(max_warps_by_regs / warps_per_block)
    let by_registers = if attrs.num_regs > 0 && warps_per_block > 0 {
        let allocated = attrs.num_regs.div_ceil(limits.warp_alloc_granularity) * limits.warp_alloc_granularity;
        let per_warp = allocated * 32;
        let regs_per_sub = limits.regs_per_sm / 4;
        let warps_per_sub = regs_per_sub / per_warp;
        let max_warps = warps_per_sub * 4;
        max_warps / warps_per_block
    } else {
        limits.max_blocks_per_sm
    };

    // Shared memory limit.
    #[allow(clippy::manual_checked_ops)]
    let by_shared_memory = if total_smem_per_block > 0 {
        limits.shared_mem_per_sm / total_smem_per_block
    } else {
        limits.max_blocks_per_sm
    };

    // Block limit.
    let by_block_limit = limits.max_blocks_per_sm;

    // Warp limit.
    #[allow(clippy::manual_checked_ops)]
    let by_warp_limit = if warps_per_block > 0 {
        limits.max_warps_per_sm / warps_per_block
    } else {
        limits.max_blocks_per_sm
    };

    let min_blocks = by_registers
        .min(by_shared_memory)
        .min(by_block_limit)
        .min(by_warp_limit);

    // Determine the limiting factor(s).
    let mut mask: u8 = 0;
    if by_registers == min_blocks {
        mask |= 1;
    }
    if by_shared_memory == min_blocks {
        mask |= 2;
    }
    if by_block_limit == min_blocks {
        mask |= 4;
    }
    if by_warp_limit == min_blocks {
        mask |= 8;
    }
    let limiting_factor = if mask.count_ones() == 1 {
        match mask {
            1 => LimitingFactor::Registers,
            2 => LimitingFactor::SharedMemory,
            4 => LimitingFactor::BlockLimit,
            8 => LimitingFactor::WarpLimit,
            _ => unreachable!(),
        }
    } else {
        LimitingFactor::Tied(mask)
    };

    Occupancy {
        blocks_per_sm: min_blocks,
        limiting_factor,
        by_registers,
        by_shared_memory,
        by_block_limit,
        by_warp_limit,
    }
}

/// Initialise profiling: called once from `init`.
pub fn open() {
    let path = match std::env::var("VF_PROFILE") {
        Ok(p) if !p.is_empty() && p != "0" => p,
        _ => return,
    };
    let full_path = format!("{}.{}", path, std::process::id());
    log::warn(format_args!(
        "profiling: enabled, writing to {full_path} at exit (every launch is synchronised: this is not a benchmark)"
    ));
    *STATE.lock().unwrap_or_else(|e| e.into_inner()) = Some(ProfileState {
        path: full_path,
        kernels: HashMap::new(),
    });
}

/// Whether profiling is enabled.
pub fn enabled() -> bool {
    STATE.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// Record the origin at the lookup for a kernel name that is mapped through a module/library share.
pub fn record_kernel_info_by_name(name: &str, origin: CodeOrigin, emulated: Option<EmulatedCounts>) {
    if !enabled() {
        return;
    }
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ref mut ps) = *state {
        ps.kernels.entry(name.to_string()).or_insert_with(|| KernelProfile {
            origin,
            emulated,
            launches: 0,
            total_ns: 0,
            grids: HashMap::new(),
            blocks: HashMap::new(),
            func_attrs: None,
            dynamic_smem: HashMap::new(),
        });
    }
}

/// Kernel function handle -> name, recorded at the lookup (where the proxy knows the name the
/// program asked for). `cuFuncGetName` does not work for lazily compiled sub-modules: the proxy's
/// own map is the reliable source.
static HANDLE_NAMES: LazyLock<Mutex<HashMap<usize, String>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Module or library handle -> the kernel handles looked up in it, so that the three per-handle
/// maps (`HANDLE_NAMES`, `FUNC_ATTRS`) can be cleaned when the module is unloaded.
static MODULE_KERNELS: LazyLock<Mutex<HashMap<usize, Vec<usize>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record the name the program asked for when it looked up the kernel `handle` in `module`.
pub fn record_handle_name(module: usize, handle: usize, name: &str) {
    HANDLE_NAMES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(handle, name.to_string());
    let mut modules = MODULE_KERNELS.lock().unwrap_or_else(|e| e.into_inner());
    let kernels = modules.entry(module).or_default();
    if !kernels.contains(&handle) {
        kernels.push(handle);
    }
}

/// The modules or libraries `modules` are unloaded: drop what is known about the kernel handles
/// looked up in them, which the driver may give again to kernels of a module loaded later. The
/// launches already counted stay (they are kept by kernel name).
pub fn forget(modules: &[usize]) {
    let mut handles = Vec::new();
    {
        let mut by_module = MODULE_KERNELS.lock().unwrap_or_else(|e| e.into_inner());
        for module in modules {
            handles.extend(by_module.remove(module).unwrap_or_default());
        }
    }
    let mut names = HANDLE_NAMES.lock().unwrap_or_else(|e| e.into_inner());
    let mut attrs = FUNC_ATTRS.lock().unwrap_or_else(|e| e.into_inner());
    for handle in handles {
        names.remove(&handle);
        attrs.remove(&handle);
    }
}

/// The name of a kernel by its function handle: first from the proxy's own map (recorded at the
/// lookup), then from the driver's `cuFuncGetName`.
unsafe fn kernel_name(function: *mut c_void) -> String {
    // Try the proxy's own map first (works for lazily compiled sub-modules).
    if let Some(name) = HANDLE_NAMES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(function as usize))
        .cloned()
    {
        return name;
    }
    // Fall back to the driver.
    let name_of: unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> c_int =
        unsafe { std::mem::transmute::<usize, _>(REALS[idx::cuFuncGetName].load(Relaxed)) };
    let mut name: *const c_char = std::ptr::null();
    if unsafe { name_of(&mut name, function) } == 0 && !name.is_null() {
        unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
    } else {
        "?".to_string()
    }
}

/// Whether the stream is currently capturing a graph (CUDA graph capture mode). Synchronizing
/// a stream that is being captured is not allowed and would fail the capture.
unsafe fn is_capturing(stream: *mut c_void) -> bool {
    // CUstreamCaptureStatus: 0 = none, 1 = active, 2 = invalidated.
    let is_capturing: unsafe extern "C" fn(*mut c_void, *mut c_int) -> c_int =
        unsafe { std::mem::transmute::<usize, _>(REALS[idx::cuStreamIsCapturing].load(Relaxed)) };
    let mut status: c_int = 0;
    let result = unsafe { is_capturing(stream, &mut status) };
    result == 0 && status != 0
}

/// Synchronise the stream so that the kernel's GPU time is known. Skips synchronization when
/// the stream is capturing a graph (CUDA graph capture mode does not allow it). Returns the
/// driver's answer, or `None` when the stream was not synchronised.
unsafe fn sync_stream(stream: *mut c_void) -> Option<c_int> {
    if unsafe { is_capturing(stream) } {
        return None;
    }
    let sync: unsafe extern "C" fn(*mut c_void) -> c_int =
        unsafe { std::mem::transmute::<usize, _>(REALS[idx::cuStreamSynchronize].load(Relaxed)) };
    Some(unsafe { sync(stream) })
}

/// Whether a failed synchronisation was already said: the first is the one that names the kernel
/// that failed (a fault ends the context, and every later call fails with the same error).
static FAILURE_SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Profile a launch of `cuLaunchKernel`. Called from the hook.
///
/// # Safety
/// All arguments are those of the driver call.
pub unsafe fn profile_launch(
    function: *mut c_void,
    grid: [u32; 3],
    block: [u32; 3],
    shared_memory_bytes: u32,
    stream: *mut c_void,
) {
    let t0 = Instant::now();
    // The launch has already happened (the real function was called). Now synchronise.
    let synced = unsafe { sync_stream(stream) };
    let elapsed_ns = if synced == Some(0) { t0.elapsed().as_nanos() } else { 0 };

    let name = unsafe { kernel_name(function) };
    if let Some(code) = synced
        && code != 0
        && !FAILURE_SAID.swap(true, Relaxed)
    {
        // The launches before this one that the profile synchronised ended without an error: what
        // failed is this kernel, unless it is work that was not synchronised (another stream, a
        // graph, a launch made during a capture).
        log::error(format_args!(
            "profiling: the synchronisation after a launch of {name} (grid {}x{}x{}, block {}x{}x{}, \
             {shared_memory_bytes} bytes of dynamic shared memory) returned error {code}, the first that one did",
            grid[0], grid[1], grid[2], block[0], block[1], block[2]
        ));
    }

    // Query function attributes from the driver (cached per function handle).
    let func_attrs = unsafe { query_func_attrs(function) };

    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ref mut ps) = *state else { return };

    let entry = ps.kernels.entry(name).or_insert_with(|| KernelProfile {
        origin: CodeOrigin::Unknown,
        emulated: None,
        launches: 0,
        total_ns: 0,
        grids: HashMap::new(),
        blocks: HashMap::new(),
        func_attrs: None,
        dynamic_smem: HashMap::new(),
    });

    // Record function attributes (first seen wins; they are per-function, not per-launch).
    if entry.func_attrs.is_none() {
        entry.func_attrs = func_attrs;
    }

    entry.launches += 1;
    entry.total_ns += elapsed_ns;
    count_shape(&mut entry.grids, (grid[0], grid[1], grid[2]));
    count_shape(&mut entry.blocks, (block[0], block[1], block[2]));
    count_shape(&mut entry.dynamic_smem, shared_memory_bytes);
}

/// The most distinct sizes kept in each map of a kernel's launch shapes (grids, blocks, dynamic
/// shared memory). A program that runs for long, a server, can launch a kernel with a grid of its
/// own for each input it is given: the profile must not grow with them.
const SHAPES_KEPT: usize = 64;

/// Count a launch of size `size` in `seen`, unless `seen` is full and `size` is not in it.
fn count_shape<K: std::hash::Hash + Eq>(seen: &mut HashMap<K, u64>, size: K) {
    if let Some(launches) = seen.get_mut(&size) {
        *launches += 1;
    } else if seen.len() < SHAPES_KEPT {
        seen.insert(size, 1);
    }
}

/// Profile a launch of `cuLaunchKernelEx`.
pub unsafe fn profile_launch_ex(
    function: *mut c_void,
    grid: [u32; 3],
    block: [u32; 3],
    shared_memory_bytes: u32,
    stream: *mut c_void,
) {
    unsafe { profile_launch(function, grid, block, shared_memory_bytes, stream) };
}

fn format_dynamic_smem(smem: &HashMap<u32, u64>) -> String {
    if smem.is_empty() {
        return String::new();
    }
    if smem.len() == 1 {
        let (&bytes, _) = smem.iter().next().unwrap();
        return bytes.to_string();
    }
    let mut entries: Vec<_> = smem.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1));
    entries
        .iter()
        .take(5)
        .map(|&(&bytes, &n)| format!("{bytes}({n})"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_sizes(sizes: &HashMap<(u32, u32, u32), u64>) -> String {
    if sizes.len() == 1 {
        let ((x, y, z), _) = sizes.iter().next().unwrap();
        if *y == 1 && *z == 1 {
            return format!("{x}");
        }
        return format!("{x}x{y}x{z}");
    }
    let mut entries: Vec<_> = sizes.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1));
    let parts: Vec<String> = entries
        .iter()
        .take(5)
        .map(|((x, y, z), n)| {
            if *y == 1 && *z == 1 {
                format!("{x}({n})")
            } else {
                format!("{x}x{y}x{z}({n})")
            }
        })
        .collect();
    if entries.len() >= SHAPES_KEPT {
        format!(
            "{} +{} more, and maybe others (only {SHAPES_KEPT} sizes are kept)",
            parts.join(" "),
            entries.len() - 5
        )
    } else if entries.len() > 5 {
        format!("{} +{} more", parts.join(" "), entries.len() - 5)
    } else {
        parts.join(" ")
    }
}

/// Write the profile to the file. Called from `fini`.
pub fn write() {
    let state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ref ps) = *state else { return };
    if ps.kernels.is_empty() {
        return;
    }

    let mut lines: Vec<(String, &KernelProfile)> = ps.kernels.iter().map(|(n, p)| (n.clone(), p)).collect();
    // Sort by total time descending.
    lines.sort_by_key(|(_name, p)| std::cmp::Reverse(p.total_ns));

    let limits = &*SM_LIMITS;

    let mut content = String::new();
    let _ = writeln!(
        content,
        "kernel\torigin\tlaunches\ttotal_ms\tmean_ms\tgrids\tblocks\tregs\tstatic_smem\t\
         dynamic_smem\tblocks_per_sm\tlimited_by\tby_regs\tby_smem\tby_blocks\tby_warps\temulated"
    );
    for (name, profile) in &lines {
        let total_ms = profile.total_ns as f64 / 1_000_000.0;
        let mean_ms = if profile.launches > 0 {
            total_ms / profile.launches as f64
        } else {
            0.0
        };
        let emulated_str = profile
            .emulated
            .as_ref()
            .filter(|c| c.any())
            .map_or_else(String::new, |c| c.summary());

        // Function attributes and occupancy.
        let (
            regs_str,
            static_smem_str,
            dynamic_smem_str,
            occ_str,
            limit_str,
            by_regs_str,
            by_smem_str,
            by_blocks_str,
            by_warps_str,
        ) = if let Some(attrs) = &profile.func_attrs {
            // Use the most common block size and dynamic shared memory for occupancy computation.
            let block_threads = profile
                .blocks
                .iter()
                .max_by_key(|&(_, n)| n)
                .map(|((x, y, z), _)| x * y * z)
                .unwrap_or(128);
            let dyn_smem = profile
                .dynamic_smem
                .iter()
                .max_by_key(|&(_, n)| n)
                .map(|(s, _)| *s)
                .unwrap_or(0);
            let occ = compute_occupancy(attrs, dyn_smem, block_threads, limits);
            (
                attrs.num_regs.to_string(),
                attrs.static_smem_bytes.to_string(),
                format_dynamic_smem(&profile.dynamic_smem),
                occ.blocks_per_sm.to_string(),
                limited_by_text(&occ),
                occ.by_registers.to_string(),
                occ.by_shared_memory.to_string(),
                occ.by_block_limit.to_string(),
                occ.by_warp_limit.to_string(),
            )
        } else {
            (
                String::new(),
                String::new(),
                format_dynamic_smem(&profile.dynamic_smem),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            )
        };

        let _ = writeln!(
            content,
            "{name}\t{origin}\t{launches}\t{total_ms:.3}\t{mean_ms:.3}\t{grids}\t{blocks}\t\
             {regs}\t{static_smem}\t{dynamic_smem}\t{occ}\t{limit}\t{by_regs}\t{by_smem}\t{by_blocks}\t{by_warps}\t{emulated}",
            origin = profile.origin,
            launches = profile.launches,
            grids = format_sizes(&profile.grids),
            blocks = format_sizes(&profile.blocks),
            regs = regs_str,
            static_smem = static_smem_str,
            dynamic_smem = dynamic_smem_str,
            occ = occ_str,
            limit = limit_str,
            by_regs = by_regs_str,
            by_smem = by_smem_str,
            by_blocks = by_blocks_str,
            by_warps = by_warps_str,
            emulated = emulated_str,
        );
    }

    // Written as `VF_SUMMARY`'s file is: never through a link at that predictable name, its owner's
    // only.
    if let Err(e) = crate::summary::replace(std::path::Path::new(&ps.path), content.as_bytes()) {
        log::error(format_args!("profiling: cannot write {}: {e}", ps.path));
    } else {
        log::warn(format_args!(
            "profiling: written {} kernels to {}",
            lines.len(),
            ps.path
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_single_1d_grid() {
        let mut sizes = HashMap::new();
        sizes.insert((128, 1, 1), 10);
        assert_eq!(format_sizes(&sizes), "128");
    }

    #[test]
    fn format_single_3d_grid() {
        let mut sizes = HashMap::new();
        sizes.insert((4, 8, 2), 5);
        assert_eq!(format_sizes(&sizes), "4x8x2");
    }

    #[test]
    fn format_multiple_grids() {
        let mut sizes = HashMap::new();
        sizes.insert((128, 1, 1), 10);
        sizes.insert((256, 1, 1), 5);
        let s = format_sizes(&sizes);
        assert!(s.contains("128(10)"));
        assert!(s.contains("256(5)"));
    }

    #[test]
    fn launch_shapes_are_bounded() {
        let mut grids = HashMap::new();
        for x in 0..1000 {
            count_shape(&mut grids, (x, 1, 1));
        }
        count_shape(&mut grids, (3, 1, 1));
        assert_eq!(grids.len(), SHAPES_KEPT);
        assert_eq!(grids[&(3, 1, 1)], 2);
        assert!(!grids.contains_key(&(999, 1, 1)));
        let text = format_sizes(&grids);
        assert!(text.starts_with("3(2) "), "{text}");
        assert!(
            text.ends_with(" +59 more, and maybe others (only 64 sizes are kept)"),
            "{text}"
        );
    }

    #[test]
    fn code_origin_display() {
        assert_eq!(format!("{}", CodeOrigin::Native), "native");
        assert_eq!(format!("{}", CodeOrigin::PtxRewritten), "ptx-rewritten");
        assert_eq!(
            format!("{}", CodeOrigin::MachineCodeTranslated),
            "machine-code-translated"
        );
        assert_eq!(format!("{}", CodeOrigin::CacheHit), "cache-hit");
        assert_eq!(format!("{}", CodeOrigin::Unknown), "unknown");
    }

    #[test]
    fn occupancy_v100_attention_native() {
        // PyTorch attention.f16 native: 168 regs, 128 threads, some shared memory.
        let limits = SmLimits::v100_defaults();
        let attrs = FuncAttrs {
            num_regs: 168,
            static_smem_bytes: 0,
            max_threads_per_block: 1024,
        };
        // No shared memory.
        let occ = compute_occupancy(&attrs, 0, 128, &limits);
        // 168 regs -> allocated 168 (already mult of 8), per_warp = 168*32 = 5376,
        // per_sub = 16384/5376 = 3, max_warps = 12, warps_per_block = 4,
        // by_regs = 12/4 = 3.
        assert_eq!(occ.by_registers, 3);
        assert_eq!(occ.by_warp_limit, 16); // 64/4 = 16
        assert_eq!(occ.by_block_limit, 32);
    }

    #[test]
    fn occupancy_v100_attention_translated() {
        // Translated: 219 regs, 128 threads.
        let limits = SmLimits::v100_defaults();
        let attrs = FuncAttrs {
            num_regs: 219,
            static_smem_bytes: 0,
            max_threads_per_block: 256,
        };
        let occ = compute_occupancy(&attrs, 0, 128, &limits);
        // 219 -> allocated 224 (ceil(219/8)*8), per_warp = 224*32 = 7168,
        // per_sub = 16384/7168 = 2, max_warps = 8, warps_per_block = 4,
        // by_regs = 8/4 = 2.
        assert_eq!(occ.by_registers, 2);
    }

    #[test]
    fn occupancy_limited_by_shared_memory() {
        // 32 regs, 128 threads, 49152 bytes shared memory per block.
        let limits = SmLimits::v100_defaults();
        let attrs = FuncAttrs {
            num_regs: 32,
            static_smem_bytes: 0,
            max_threads_per_block: 1024,
        };
        let occ = compute_occupancy(&attrs, 49152, 128, &limits);
        // 96 KB / ~49152 = 2 blocks by shared memory.
        assert_eq!(occ.by_shared_memory, 2);
        // 32 regs -> allocated 32, per_warp = 1024, per_sub = 16, max_warps = 64,
        // by_regs = 64/4 = 16.
        assert_eq!(occ.by_registers, 16);
        // The minimum is by shared memory.
        assert_eq!(occ.blocks_per_sm, 2);
        assert!(matches!(occ.limiting_factor, LimitingFactor::SharedMemory));
    }

    #[test]
    fn occupancy_smem_allocation_granularity() {
        let limits = SmLimits::v100_defaults();
        let attrs = FuncAttrs {
            num_regs: 32,
            static_smem_bytes: 100,
            max_threads_per_block: 1024,
        };
        // 100 bytes static + 50 dynamic = 150 bytes, rounded up to 256.
        // 98304 / 256 = 384 blocks by shared memory.
        let occ = compute_occupancy(&attrs, 50, 128, &limits);
        assert_eq!(occ.by_shared_memory, 384);
    }

    #[test]
    fn occupancy_cannot_launch() {
        // 1,024 threads a block with 72 registers: 32 warps of 72 * 32 registers need 73,728
        // registers, more than the 65,536 of a multiprocessor: no block fits.
        let limits = SmLimits::v100_defaults();
        let attrs = FuncAttrs {
            num_regs: 72,
            static_smem_bytes: 0,
            max_threads_per_block: 1024,
        };
        let occ = compute_occupancy(&attrs, 0, 1024, &limits);
        assert_eq!(occ.by_registers, 0);
        assert_eq!(occ.blocks_per_sm, 0);
        assert_eq!(occ.limiting_factor, LimitingFactor::Registers);
        assert_eq!(limited_by_text(&occ), "registers:cannot_launch");
        // The same kernel with 512 threads fits one block, and says plainly what limits it.
        let occ = compute_occupancy(&attrs, 0, 512, &limits);
        assert_eq!(occ.blocks_per_sm, 1);
        assert_eq!(limited_by_text(&occ), "registers");
    }

    #[test]
    fn forget_drops_the_handles_of_an_unloaded_module() {
        let attrs = FuncAttrs {
            num_regs: 168,
            static_smem_bytes: 0,
            max_threads_per_block: 128,
        };
        // Module 0x1000 has kernel handle 0x2000 ("old"), module 0x1100 has 0x2100.
        record_handle_name(0x1000, 0x2000, "old");
        record_handle_name(0x1000, 0x2000, "old"); // looked up twice: one handle
        record_handle_name(0x1100, 0x2100, "other");
        FUNC_ATTRS.lock().unwrap().insert(0x2000, attrs);
        FUNC_ATTRS.lock().unwrap().insert(0x2100, attrs);
        // A launch already counted under the old name.
        *STATE.lock().unwrap() = Some(ProfileState {
            path: String::new(),
            kernels: HashMap::from([(
                "old".to_string(),
                KernelProfile {
                    origin: CodeOrigin::PtxRewritten,
                    emulated: None,
                    launches: 3,
                    total_ns: 0,
                    grids: HashMap::new(),
                    blocks: HashMap::new(),
                    func_attrs: Some(attrs),
                    dynamic_smem: HashMap::new(),
                },
            )]),
        });

        forget(&[0x1000]);

        let name_of = |handle: usize| HANDLE_NAMES.lock().unwrap().get(&handle).cloned();
        let attrs_of = |handle: usize| FUNC_ATTRS.lock().unwrap().get(&handle).copied();
        assert_eq!(name_of(0x2000), None);
        assert!(attrs_of(0x2000).is_none());
        assert!(!MODULE_KERNELS.lock().unwrap().contains_key(&0x1000));
        // The other module is untouched.
        assert_eq!(name_of(0x2100).as_deref(), Some("other"));
        assert!(attrs_of(0x2100).is_some());
        // The launches counted under the old name are kept.
        let state = STATE.lock().unwrap();
        assert_eq!(state.as_ref().unwrap().kernels["old"].launches, 3);
        drop(state);
        *STATE.lock().unwrap() = None; // the process must not write a file at its end

        // The handle comes back for another module's kernel: nothing of the old one is left.
        record_handle_name(0x1200, 0x2000, "new");
        assert_eq!(name_of(0x2000).as_deref(), Some("new"));
        assert!(attrs_of(0x2000).is_none());
    }

    #[test]
    fn format_dynamic_smem_single() {
        let mut m = HashMap::new();
        m.insert(8192, 10);
        assert_eq!(format_dynamic_smem(&m), "8192");
    }

    #[test]
    fn format_dynamic_smem_multiple() {
        let mut m = HashMap::new();
        m.insert(8192, 10);
        m.insert(0, 5);
        let s = format_dynamic_smem(&m);
        assert!(s.contains("8192(10)"));
        assert!(s.contains("0(5)"));
    }

    #[test]
    fn limiting_factor_display() {
        assert_eq!(format!("{}", LimitingFactor::Registers), "registers");
        assert_eq!(format!("{}", LimitingFactor::SharedMemory), "shared_memory");
        assert_eq!(format!("{}", LimitingFactor::BlockLimit), "block_limit");
        assert_eq!(format!("{}", LimitingFactor::WarpLimit), "warp_limit");
        assert_eq!(format!("{}", LimitingFactor::Tied(3)), "registers+shared_memory");
    }
}
