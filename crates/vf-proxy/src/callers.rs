//! Which compute capability a caller is told.
//!
//! A program chooses its kernels, and often its launch parameters, from the compute capability of the
//! GPU. Code built for sm_75 and later must be told a capability it has kernels for, or it gives up
//! (or launches kernels with the parameters of another architecture); once told 7.5 it loads its
//! Turing code, which `images` adapts. Code that does have Volta kernels must be told the truth, so
//! that it uses them: the CUDA 12 libraries that stand in for their CUDA 13 versions are in that case,
//! in the same process as the program built for CUDA 13.
//!
//! So the answer depends on who asks. The caller is the library the capability function returns to:
//! for example, a CUDA runtime, an application or a binding such as libffi.
//!
//! - A file that embeds code the driver uses on this GPU is told the real capability, always.
//! - `libcudart.so.13` and later, and a file whose embedded code starts at an architecture of the
//!   same generation that PTX can be retargeted from (sm_75 on Volta), are told that architecture.
//! - Under the launcher, a readable file with no embedded GPU code sees 7.5 for the device.
//!   Its queries about an individual kernel still return the architecture of the loaded binary.
//! - Older CUDA runtimes retain the physical capability by default.
//!
//! `VF_CAPABILITY` overrides the device answer for a direct client or code targeting later GPUs.
//! The PTX taken from the program follows that setting (`vf_adapt::Policy::told_sm`), and so does
//! what the caller is told, except where the program is given machine code of an earlier
//! generation: the translation of its sm_75 code, which is what runs of a library such as
//! PyTorch's, which has no PTX. Told 8.0, such a program launches its kernels for sm_80, which
//! that code has not. A caller is told no later generation than the machine code that is given
//! for its own embedded code, or for that of the libraries loaded with it as a dependency (a CUDA
//! runtime answers for the programs that use it): 7.5 in this case (`vf_adapt::Policy::told_limit`),
//! with a line at the default log level that says so.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::sync::{LazyLock, Mutex};

use vf_adapt::Embedded;

use crate::{config, log};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Caller {
    /// Has code for this GPU.
    Native,
    /// Has code from architecture `sm` on, which can be adapted.
    From(u32),
    /// A launcher client whose file contains no GPU code.
    Direct,
    /// Neither is known.
    Other,
}

/// First CUDA release without Volta: its runtime serves programs built for sm_75 and later.
const FIRST_RUNTIME_WITHOUT_DEVICE: u32 = 13;
const FIRST_SM_OF_THAT_RUNTIME: u32 = vf_adapt::TRANSLATED_FROM;

static UNDER_LAUNCHER: LazyLock<bool> = LazyLock::new(|| std::env::var_os("VOLTA_FOREVER").is_some());

/// What a caller is told.
#[derive(Debug, Clone, Copy)]
struct Decision {
    caller: Caller,
    /// The architecture that the code given for the caller limits what it is told to, below what
    /// `VF_CAPABILITY` asks (`vf_adapt::Policy::told_limit`).
    limit: Option<u32>,
}

#[derive(Default)]
struct Cache {
    unloads: u64,
    by_base: HashMap<usize, Decision>,
    /// `vf_adapt::Policy::told_limit_in` of the files read, by path.
    limits: HashMap<CString, Option<u32>>,
}

static CALLERS: LazyLock<Mutex<Cache>> = LazyLock::new(|| Mutex::new(Cache::default()));

/// glibc's unload counter changes even if another DSO takes the exact same address and filename.
fn unload_count() -> u64 {
    unsafe extern "C" fn first(info: *mut libc::dl_phdr_info, _: usize, data: *mut c_void) -> i32 {
        // SAFETY: glibc on the supported x86-64 Linux systems supplies dlpi_subs; data points to n.
        unsafe { *data.cast::<u64>() = (*info).dlpi_subs };
        1
    }
    let mut n = 0u64;
    unsafe { libc::dl_iterate_phdr(Some(first), (&raw mut n).cast()) };
    n
}

/// An object of the process (the main program or a shared library), as the dynamic loader lists it.
struct Object {
    /// The file its code can be read from: `/proc/self/exe` for the main program; `None` for
    /// an object without a name (the kernel's vDSO has no file either, and reads as nothing).
    path: Option<CString>,
    /// Whether the address asked about is in one of its segments.
    holds: bool,
    /// Its `DT_SONAME`, and the `DT_NEEDED` names of the libraries it was loaded with.
    soname: Option<Vec<u8>>,
    needed: Vec<Vec<u8>>,
}

/// The objects of the process, the one that holds `address` marked.
fn objects(address: usize) -> Vec<Object> {
    unsafe extern "C" fn each(info: *mut libc::dl_phdr_info, _: usize, data: *mut c_void) -> i32 {
        // SAFETY: `data` is the pair below; `info` is the loader's description of a loaded object,
        // whose program headers and segments are mapped while it is listed.
        let (address, list) = unsafe { &mut *data.cast::<(usize, Vec<Object>)>() };
        let first = list.is_empty();
        list.push(unsafe { object(&*info, *address, first) });
        0
    }
    let mut data = (address, Vec::<Object>::new());
    unsafe { libc::dl_iterate_phdr(Some(each), (&raw mut data).cast()) };
    data.1
}

/// # Safety
/// `info` must be what `dl_iterate_phdr` gives to its callback, used during the callback.
unsafe fn object(info: &libc::dl_phdr_info, address: usize, first: bool) -> Object {
    const PT_LOAD: u32 = 1;
    const PT_DYNAMIC: u32 = 2;
    let (base, count) = (info.dlpi_addr as usize, info.dlpi_phnum as usize);
    let headers = if info.dlpi_phdr.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(info.dlpi_phdr, count) }
    };
    let segments: Vec<(usize, usize)> = headers
        .iter()
        .filter(|h| h.p_type == PT_LOAD)
        .map(|h| {
            let start = base.wrapping_add(h.p_vaddr as usize);
            (start, start.saturating_add(h.p_memsz as usize))
        })
        .collect();
    let mapped = |at: usize, length: usize| {
        segments
            .iter()
            .any(|&(start, end)| at >= start && at.checked_add(length).is_some_and(|stop| stop <= end))
    };
    let name = if info.dlpi_name.is_null() {
        &[][..]
    } else {
        unsafe { CStr::from_ptr(info.dlpi_name) }.to_bytes()
    };
    // The loader lists the main program first, without a name.
    let path = match name {
        [] if first => Some(c"/proc/self/exe".to_owned()),
        [] => None,
        name => CString::new(name).ok(),
    };
    let mut found = Object {
        path,
        holds: mapped(address, 1),
        soname: None,
        needed: Vec::new(),
    };
    // The dynamic section: tags and values, up to DT_NULL.
    let Some(dynamic) = headers.iter().find(|h| h.p_type == PT_DYNAMIC) else {
        return found;
    };
    let dynamic = base.wrapping_add(dynamic.p_vaddr as usize);
    let (mut strings, mut size, mut names, mut soname) = (0usize, 0usize, Vec::new(), None);
    let mut at = dynamic;
    while mapped(at, 16) {
        // SAFETY: 16 bytes inside a segment of the object.
        let (tag, value) = unsafe { (*(at as *const i64), *((at + 8) as *const u64) as usize) };
        match tag {
            0 => break,
            1 => names.push(value),
            5 => strings = value,
            10 => size = value,
            14 => soname = Some(value),
            _ => {}
        }
        at += 16;
    }
    // The loader relocates the address of the string table in place, except in a read-only
    // dynamic section (the vDSO's).
    if !mapped(strings, size) {
        strings = strings.wrapping_add(base);
    }
    if size == 0 || !mapped(strings, size) {
        return found;
    }
    // SAFETY: the string table is inside a segment of the object.
    let table = unsafe { std::slice::from_raw_parts(strings as *const u8, size) };
    let string = |offset: usize| {
        let rest = table.get(offset..)?;
        Some(rest[..rest.iter().position(|&b| b == 0)?].to_vec())
    };
    found.soname = soname.and_then(string);
    found.needed = names.into_iter().filter_map(string).collect();
    found
}

/// A file mapped read-only.
struct Mapped {
    start: *mut c_void,
    len: usize,
}

impl Mapped {
    fn open(path: &CStr) -> Option<Mapped> {
        // SAFETY: plain system calls on a path; the mapping is private and read-only.
        unsafe {
            let fd = libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
            if fd < 0 {
                return None;
            }
            let mut stat: libc::stat = std::mem::zeroed();
            let len = if libc::fstat(fd, &mut stat) == 0 {
                stat.st_size as usize
            } else {
                0
            };
            let start = if len > 0 {
                libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, fd, 0)
            } else {
                libc::MAP_FAILED
            };
            libc::close(fd);
            (start != libc::MAP_FAILED).then_some(Mapped { start, len })
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping covers `len` readable bytes until `drop`.
        unsafe { std::slice::from_raw_parts(self.start as *const u8, self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: `start` and `len` describe the mapping made in `open`.
        unsafe { libc::munmap(self.start, self.len) };
    }
}

/// The major version in `libcudart.so.<major>[.minor...]`.
fn runtime_version(file_name: &str) -> Option<u32> {
    file_name.strip_prefix("libcudart.so.")?.split('.').next()?.parse().ok()
}

/// The class of the caller whose file is `path`, and, when `limits`, the limit that its own
/// embedded code sets to what it is told (`vf_adapt::Policy::told_limit_in`).
fn classify(path: &CStr, limits: bool) -> (Caller, Option<u32>) {
    let policy = &config().policy;
    let name = path.to_string_lossy();
    let file_name = name.rsplit('/').next().unwrap_or(&name);
    if let Some(version) = runtime_version(file_name) {
        let caller = if version >= FIRST_RUNTIME_WITHOUT_DEVICE && policy.retargetable(FIRST_SM_OF_THAT_RUNTIME) {
            Caller::From(FIRST_SM_OF_THAT_RUNTIME)
        } else {
            Caller::Other
        };
        return (caller, None);
    }
    let Some(file) = Mapped::open(path) else {
        return (Caller::Other, None);
    };
    let caller = match policy.embedded(file.bytes()) {
        Embedded::Usable => return (Caller::Native, None),
        Embedded::Nothing if *UNDER_LAUNCHER => Caller::Direct,
        Embedded::Other { lowest_sm } if policy.retargetable(lowest_sm) => Caller::From(lowest_sm),
        _ => Caller::Other,
    };
    let limit = if limits && matches!(caller, Caller::From(_)) {
        policy.told_limit_in(file.bytes())
    } else {
        None
    };
    (caller, limit)
}

/// The limit that the code of the libraries loaded with the caller as a dependency (named in
/// their `DT_NEEDED`) sets to what it is told, and the library that sets it: a CUDA runtime
/// answers for the programs that use it, the graph library of cuDNN for its engines. Their files
/// are read once each. A library loaded after the caller first asks is not seen.
fn dependents_limit(objects: &[Object], limits: &mut HashMap<CString, Option<u32>>) -> Option<(u32, CString)> {
    let caller = objects.iter().find(|object| object.holds)?;
    let file_name = caller.path.as_ref().map(|path| {
        path.to_bytes()
            .rsplit(|&b| b == b'/')
            .next()
            .unwrap_or_default()
            .to_vec()
    });
    let names: Vec<&[u8]> = caller
        .soname
        .iter()
        .chain(file_name.iter())
        .map(Vec::as_slice)
        .collect();
    let mut lowest: Option<(u32, CString)> = None;
    for object in objects.iter().filter(|object| !object.holds) {
        let Some(path) = &object.path else {
            continue;
        };
        if !object.needed.iter().any(|needed| names.contains(&needed.as_slice())) {
            continue;
        }
        let limit = *limits
            .entry(path.clone())
            .or_insert_with(|| Mapped::open(path).and_then(|file| config().policy.told_limit_in(file.bytes())));
        if let Some(sm) = limit
            && lowest.as_ref().is_none_or(|(lowest, _)| sm < *lowest)
        {
            lowest = Some((sm, path.clone()));
        }
    }
    lowest
}

fn caller(return_address: usize) -> Decision {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: the call site is live for the duration of this driver call.
    if unsafe { libc::dladdr(return_address as *const c_void, &mut info) } == 0 || info.dli_fname.is_null() {
        return Decision {
            caller: Caller::Other,
            limit: None,
        };
    }
    let unloads = unload_count();
    let mut cache = CALLERS.lock().unwrap_or_else(|e| e.into_inner());
    if cache.unloads != unloads {
        cache.by_base.clear();
        cache.limits.clear();
        cache.unloads = unloads;
    }
    let base = info.dli_fbase as usize;
    if let Some(decision) = cache.by_base.get(&base) {
        return *decision;
    }
    let objects = objects(return_address);
    // For the main program the loader's name is `argv[0]`, which names no file when the program
    // was started by name through PATH, or once it has changed directory: its file is read
    // through `/proc/self/exe`.
    let main = objects
        .first()
        .filter(|object| object.holds)
        .and_then(|object| object.path.as_deref());
    let path = main.unwrap_or_else(|| unsafe { CStr::from_ptr(info.dli_fname) });
    // What limits the answer matters only when the user asks for a later generation than Turing's.
    let limits = config()
        .policy
        .told_sm
        .is_some_and(|sm| sm / 10 > FIRST_SM_OF_THAT_RUNTIME / 10);
    let (caller, own) = classify(path, limits);
    let mut limit = own.map(|sm| (sm, path.to_owned()));
    if limits && caller != Caller::Native {
        let dependents = dependents_limit(&objects, &mut cache.limits);
        if let Some((sm, file)) = dependents
            && limit.as_ref().is_none_or(|(lowest, _)| sm < *lowest)
        {
            limit = Some((sm, file));
        }
    }
    let decision = Decision {
        caller,
        limit: limit.as_ref().map(|(sm, _)| *sm),
    };
    let name = match main {
        Some(path) if path == c"/proc/self/exe" => std::fs::read_link("/proc/self/exe").map_or_else(
            |_| path.to_string_lossy().into_owned(),
            |file| file.display().to_string(),
        ),
        _ => path.to_string_lossy().into_owned(),
    };
    match (caller, config().capability, &limit) {
        (Caller::Native, _, _) => log::info(format_args!(
            "{name} is told the real compute capability: it has code for this GPU"
        )),
        (_, Some((major, minor)), Some((sm, file))) => {
            let whose = if file.as_c_str() == path {
                "its code".to_string()
            } else {
                format!("the code of {}", file.to_string_lossy())
            };
            log::warn(format_args!(
                "{name} is told compute capability {}.{}, not the {major}.{minor} that VF_CAPABILITY asks: {whose} \
                 runs here as the translation of its machine code for sm_{sm}, in which the kernels that a program \
                 told {major}.{minor} chooses are not the real ones",
                sm / 10,
                sm % 10
            ))
        }
        (_, Some((major, minor)), None) => log::info(format_args!(
            "{name} is told compute capability {major}.{minor}, as VF_CAPABILITY asks"
        )),
        (Caller::From(sm), None, _) => log::info(format_args!(
            "{name} is told compute capability {}.{}: its code starts there",
            sm / 10,
            sm % 10
        )),
        (Caller::Direct, None, _) => log::info(format_args!(
            "{name} is told compute capability 7.5: it has no embedded GPU code"
        )),
        (Caller::Other, None, _) => log::info(format_args!("{name} is told the real compute capability")),
    }
    cache.by_base.insert(base, decision);
    decision
}

/// The capability reported for a physical sm_70 device.
pub fn device_capability(return_address: usize) -> Option<(i32, i32)> {
    capability(caller(return_address))
}

/// A client without embedded GPU code can identify the native architecture of an external kernel.
pub fn binary_capability(return_address: usize) -> Option<(i32, i32)> {
    match caller(return_address) {
        Decision {
            caller: Caller::Direct, ..
        } => None,
        other => capability(other),
    }
}

fn capability(decision: Decision) -> Option<(i32, i32)> {
    let as_pair = |sm: u32| ((sm / 10) as i32, (sm % 10) as i32);
    let told = config().capability.map(|asked| decision.limit.map_or(asked, as_pair));
    let sm = match decision.caller {
        Caller::Native => return None,
        Caller::Direct => FIRST_SM_OF_THAT_RUNTIME,
        Caller::From(sm) => sm,
        Caller::Other => return told,
    };
    told.or(Some(as_pair(sm)))
}
