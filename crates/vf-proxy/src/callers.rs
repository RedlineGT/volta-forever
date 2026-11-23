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
//! what the caller is told, except where the program is given code of an earlier generation: the
//! translation of its sm_75 code, which is what runs of a library such as PyTorch's, which has no
//! PTX, or its PTX for sm_75, when it has none for the generation asked. Told 8.0, such a program
//! launches its kernels for sm_80, which that code has not. A caller is told no later generation
//! than the code that is given for its own embedded code, or for that of the libraries loaded
//! with it as a dependency (a CUDA
//! runtime answers for the programs that use it; cuDNN's graph library for the library of its
//! precompiled engines): 7.5 in this case (`vf_adapt::Policy::told_limit`), with a line at the
//! default log level that says so. `VF_CAPABILITY_ALL=1` tells such a caller what `VF_CAPABILITY`
//! asks all the same, for a program that does not launch the kernels it would choose for it
//! (SGLang with the settings of its guide selects its quantized experts from PyTorch's capability).
//! The code given to a caller follows what it is told: the images it loads are adapted for the
//! architecture it is told (`told_sm`), so that a caller told 7.5 is given the sm_75 code of an
//! image that also has PTX for sm_80.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
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

/// Whether the file at `path` can be opened for reading.
fn readable(path: &CStr) -> bool {
    // SAFETY: plain system calls on a path.
    unsafe {
        let fd = libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
        fd >= 0 && libc::close(fd) == 0
    }
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

fn classify(path: &CStr) -> Caller {
    let policy = &config().policy;
    let name = path.to_string_lossy();
    let file_name = name.rsplit('/').next().unwrap_or(&name);
    if let Some(version) = runtime_version(file_name) {
        return if version >= FIRST_RUNTIME_WITHOUT_DEVICE && policy.retargetable(FIRST_SM_OF_THAT_RUNTIME) {
            Caller::From(FIRST_SM_OF_THAT_RUNTIME)
        } else {
            Caller::Other
        };
    }
    match Mapped::open(path).map(|file| policy.embedded(file.bytes())) {
        Some(Embedded::Usable) => Caller::Native,
        Some(Embedded::Nothing) if *UNDER_LAUNCHER => Caller::Direct,
        Some(Embedded::Other { lowest_sm }) if policy.retargetable(lowest_sm) => Caller::From(lowest_sm),
        _ => Caller::Other,
    }
}

/// The limit that the code embedded in the file at `path` sets to what a caller that runs it is
/// told (`vf_adapt::Policy::told_limit_in`), with the file.
fn file_limit(path: &CStr, limits: &mut HashMap<CString, Option<u32>>) -> Option<(u32, CString)> {
    let limit = *limits.entry(path.to_owned()).or_insert_with(|| {
        let file = Mapped::open(path)?;
        unpanicked(path, || config().policy.told_limit_in(file.bytes()))
    });
    limit.map(|sm| (sm, path.to_owned()))
}

/// What `scan` answers of the file at `path`, or `None` (no limit: the answer without the scan)
/// when it panics. The answer is given inside `extern "C"` functions of the driver's, where a
/// panic would end the process; the files read are any the process loaded.
fn unpanicked(path: &CStr, scan: impl FnOnce() -> Option<u32>) -> Option<u32> {
    catch_unwind(AssertUnwindSafe(scan)).unwrap_or_else(|_| {
        log::error(format_args!(
            "internal error while reading the GPU code of {}; it sets no limit to the capability told",
            path.to_string_lossy()
        ));
        None
    })
}

/// The lower of two limits.
fn lower(limit: Option<(u32, CString)>, other: Option<(u32, CString)>) -> Option<(u32, CString)> {
    match (limit, other) {
        (Some(a), Some(b)) => Some(if b.0 < a.0 { b } else { a }),
        (a, b) => a.or(b),
    }
}

/// The files read for what limits the caller of class `caller` whose file is `path`, besides the
/// libraries loaded with it as a dependency: its own, when it embeds code; for cuDNN's graph
/// library, which chooses the engines and loads the library of its precompiled ones only when it
/// needs them (after it asked), that library, beside it.
fn own_files(path: &CStr, caller: Caller) -> Vec<CString> {
    let bytes = path.to_bytes();
    let (directory, name) = match bytes.iter().rposition(|&b| b == b'/') {
        Some(at) => bytes.split_at(at + 1),
        None => (&b""[..], bytes),
    };
    let mut files = Vec::new();
    if matches!(caller, Caller::From(_)) && runtime_version(&String::from_utf8_lossy(name)).is_none() {
        files.push(path.to_owned());
    }
    if let Some(version) = name.strip_prefix(b"libcudnn_graph.so")
        && let Ok(engines) = CString::new([directory, b"libcudnn_engines_precompiled.so", version].concat())
    {
        files.push(engines);
    }
    files
}

/// The limit that the code of the libraries loaded with the caller as a dependency (named in
/// their `DT_NEEDED`) sets to what it is told, and the library that sets it: a CUDA runtime
/// answers for the programs that use it. Their files are read once each. A library loaded after
/// the caller first asks is not seen.
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
        if object.needed.iter().any(|needed| names.contains(&needed.as_slice())) {
            lowest = lower(lowest, file_limit(path, limits));
        }
    }
    lowest
}

/// What limits the answer matters only when the user asks for a later generation than Turing's.
fn limits_apply() -> bool {
    config()
        .policy
        .told_sm
        .is_some_and(|sm| sm / 10 > FIRST_SM_OF_THAT_RUNTIME / 10)
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
    // through `/proc/self/exe`, when that can be opened (without `/proc`, in a sandbox or a
    // chroot, the loader's name is read: it names the file when it is a path).
    let main = objects
        .first()
        .filter(|object| object.holds)
        .and_then(|object| object.path.as_deref())
        .filter(|&path| path != c"/proc/self/exe" || readable(path));
    let path = main.unwrap_or_else(|| unsafe { CStr::from_ptr(info.dli_fname) });
    let caller = classify(path);
    let mut limit = None;
    if limits_apply() && caller != Caller::Native {
        for file in own_files(path, caller) {
            limit = lower(limit, file_limit(&file, &mut cache.limits));
        }
        limit = lower(limit, dependents_limit(&objects, &mut cache.limits));
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
            if config().capability_all {
                log::warn(format_args!(
                    "{name} is told compute capability {major}.{minor}, as VF_CAPABILITY and VF_CAPABILITY_ALL ask, \
                     although {whose} runs here from its code for sm_{sm}, in which the \
                     kernels that a program told {major}.{minor} chooses are not the real ones"
                ))
            } else {
                log::warn(format_args!(
                    "{name} is told compute capability {}.{}, not the {major}.{minor} that VF_CAPABILITY asks: {whose} \
                     runs here from its code for sm_{sm}, in which the kernels that a \
                     program told {major}.{minor} chooses are not the real ones (VF_CAPABILITY_ALL=1 tells it \
                     {major}.{minor} all the same)",
                    sm / 10,
                    sm % 10
                ))
            }
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

/// The architecture for which the images that the caller at `return_address` loads are adapted
/// (`vf_adapt::Policy::told_sm`), when it is not the one `VF_CAPABILITY` asks: the one the caller
/// is told, where the code it runs limits it (`vf_adapt::Policy::told_limit`). Told 7.5, a caller
/// is then given the sm_75 code, or the PTX for sm_75, of every image it loads that has some, not
/// the code for sm_80 that the image may also have. The caller that loads an image is the caller
/// that asks what the GPU is, a CUDA runtime for the programs that use it. `None` with
/// `VF_CAPABILITY_ALL=1`, which tells every caller what `VF_CAPABILITY` asks.
pub fn told_sm(return_address: usize) -> Option<u32> {
    if config().capability_all || !limits_apply() {
        return None;
    }
    caller(return_address).limit
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
    let limit = decision.limit.filter(|_| !config().capability_all);
    let told = config().capability.map(|asked| limit.map_or(asked, as_pair));
    let sm = match decision.caller {
        Caller::Native => return None,
        Caller::Direct => FIRST_SM_OF_THAT_RUNTIME,
        Caller::From(sm) => sm,
        Caller::Other => return told,
    };
    told.or(Some(as_pair(sm)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scan_that_panics_sets_no_limit() {
        assert_eq!(unpanicked(c"/lib/a.so", || Some(75)), Some(75));
        assert_eq!(unpanicked(c"/lib/a.so", || panic!("an odd input")), None);
    }
}
