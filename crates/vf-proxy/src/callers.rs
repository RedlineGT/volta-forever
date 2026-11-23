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
//! with it as a dependency (a CUDA runtime answers for the programs that use it), or, for a caller
//! without code of its own, that of the libraries beside it that depend on it and that it may load
//! later (cuDNN's graph library loads the library of its engines when it needs them): 7.5 in this
//! case (`vf_adapt::Policy::told_limit`), with a line at the default log level that says so. `VF_CAPABILITY_ALL=1` tells a CUDA runtime and the code that uses it what `VF_CAPABILITY`
//! asks all the same, for a program that does not launch the kernels it would choose for it
//! (SGLang with the settings of its guide selects its quantized experts from PyTorch's capability);
//! a library that asks for itself without that runtime is still told what its code allows
//! (`Decision::program`).
//! The code given to a caller follows what it is told: the images it loads are adapted for the
//! architecture it is told (`told_sm`), so that a caller told 7.5 is given the sm_75 code of an
//! image that also has PTX for sm_80.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
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
    /// Whether the caller is a CUDA runtime (`libcudart.so.13` and later) or code loaded with one as
    /// a dependency: the program that uses the runtime, the callers whose limit
    /// `VF_CAPABILITY_ALL=1` lifts. A program reads the capability also for decisions that are not
    /// about its own kernels (SGLang its quantized experts, Triton its code), and its kernels that
    /// are stubs for the capability told are refused (`vf_adapt::Policy::adapt_fatbin`). A library
    /// that asks for itself without that runtime (cuDNN, cuFFT, cuSOLVER, which have none or have it
    /// linked in) chooses only its own kernels by the answer: told 8.0, they chose kernels that their
    /// sm_75 code does not have, refused or stubs (cuDNN's convolutions gave zeros, cuFFT stopped the
    /// GPU, cuSOLVER's Cholesky factorization failed).
    program: bool,
}

#[derive(Default)]
struct Cache {
    unloads: u64,
    by_base: HashMap<usize, Decision>,
    /// `vf_adapt::Policy::told_limit_in` of the files read, by path and what is told.
    limits: HashMap<(CString, Option<u32>), Option<u32>>,
}

/// Held only to read or write the cache: not over a walk of the loader's list (`dl_iterate_phdr`
/// takes the loader's lock, and a thread that holds that can ask what the GPU is from a callback),
/// nor over the reading of files.
static CALLERS: LazyLock<Mutex<Cache>> = LazyLock::new(|| Mutex::new(Cache::default()));

fn callers() -> std::sync::MutexGuard<'static, Cache> {
    CALLERS.lock().unwrap_or_else(|e| e.into_inner())
}

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
    /// The file its code can be read from: `/proc/self/exe` for the main program (without `/proc`,
    /// the loader's name for it: see `objects`); `None` for an object without a name (the
    /// kernel's vDSO has no file either, and reads as nothing).
    path: Option<CString>,
    /// The address of its first loaded segment.
    first: Option<usize>,
    /// What the file it was loaded from must have to be that file (`loaded_file`): its table of
    /// program headers and the contents of its notes (the build ID among them), as they are in
    /// memory.
    headers: Vec<u8>,
    notes: Vec<(usize, Vec<u8>)>,
    /// The inode of the file mapped at its first segment, as `/proc/self/maps` says (`None`
    /// without `/proc`).
    inode: Option<u64>,
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
    let mut list = data.1;
    // The main program's file is read through `/proc/self/exe`, its name for the loader being
    // `argv[0]`, which names no file when it was started by name through PATH or once it changed
    // directory. Without `/proc` (a sandbox, a chroot), that name is taken, as the loader gives it
    // (`dladdr`): it names the file when it is a path. Once here, for every reader of the list:
    // the caller's own code, and the code of the libraries it answers for.
    if let Some(main) = list.first_mut()
        && main.path.as_deref() == Some(c"/proc/self/exe")
        && !readable(c"/proc/self/exe")
    {
        main.path = main.first.and_then(|address| {
            let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
            // SAFETY: an address of the main program, which stays loaded.
            let found = unsafe { libc::dladdr(address as *const c_void, &mut info) } != 0;
            (found && !info.dli_fname.is_null()).then(|| unsafe { CStr::from_ptr(info.dli_fname) }.to_owned())
        });
    }
    // The inode of what is mapped where each object starts (`loaded_file`).
    if let Ok(maps) = std::fs::read("/proc/self/maps") {
        let mappings: Vec<(usize, usize, u64)> = maps
            .split(|&b| b == b'\n')
            .filter_map(|line| {
                let line = std::str::from_utf8(line).ok()?;
                let mut fields = line.split_ascii_whitespace();
                let (start, end) = fields.next()?.split_once('-')?;
                let inode = fields.nth(3)?.parse().ok()?;
                Some((
                    usize::from_str_radix(start, 16).ok()?,
                    usize::from_str_radix(end, 16).ok()?,
                    inode,
                ))
            })
            .collect();
        for object in &mut list {
            object.inode = object.first.and_then(|at| {
                mappings
                    .iter()
                    .find(|&&(start, end, _)| (start..end).contains(&at))
                    .map(|&(_, _, inode)| inode)
            });
        }
    }
    list
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
    const PT_NOTE: u32 = 4;
    let notes = headers
        .iter()
        .filter(|h| h.p_type == PT_NOTE)
        .filter_map(|h| {
            let (start, size) = (base.wrapping_add(h.p_vaddr as usize), h.p_filesz as usize);
            // SAFETY: `size` bytes inside a segment of the object.
            mapped(start, size).then(|| {
                (
                    h.p_offset as usize,
                    unsafe { std::slice::from_raw_parts(start as *const u8, size) }.to_vec(),
                )
            })
        })
        .collect();
    // SAFETY: the program headers that the loader gives, `count` of them.
    let table = unsafe { std::slice::from_raw_parts(headers.as_ptr().cast::<u8>(), std::mem::size_of_val(headers)) };
    let mut found = Object {
        path,
        first: segments.first().map(|&(start, _)| start),
        headers: table.to_vec(),
        notes,
        inode: None,
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
    inode: u64,
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
            let inode = stat.st_ino;
            let start = if len > 0 {
                libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, fd, 0)
            } else {
                libc::MAP_FAILED
            };
            libc::close(fd);
            (start != libc::MAP_FAILED).then_some(Mapped { start, len, inode })
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

/// The file of `object`, when the file at its path is the one it was loaded from: the inode that
/// `/proc/self/maps` gives for the object (the device is not compared: an overlay file system
/// gives another than `stat` does), and the same table of program headers and the same notes (the
/// build ID, when it has one) as are in memory. A file replaced at that path since, or one that
/// cannot be read, is not taken for it (`None`): what code the object has is then not known.
/// (Without `/proc`, a file of the same build whose GPU code is in sections that are not loaded,
/// as `objcopy --add-section` puts it, cannot be told from the one loaded.)
fn loaded_file(object: &Object) -> Option<Mapped> {
    let file = Mapped::open(object.path.as_deref()?)?;
    if object.inode.is_some_and(|inode| inode != file.inode) {
        return None;
    }
    let bytes = file.bytes();
    let u16_at = |at: usize| Some(u16::from_le_bytes(bytes.get(at..at.checked_add(2)?)?.try_into().ok()?));
    let u64_at = |at: usize| Some(u64::from_le_bytes(bytes.get(at..at.checked_add(8)?)?.try_into().ok()?) as usize);
    let same = || -> Option<bool> {
        if object.headers.is_empty() || bytes.get(..6)? != b"\x7fELF\x02\x01" || u16_at(0x36)? != 56 {
            return Some(false);
        }
        let table = u64_at(0x20)?;
        let headers = bytes.get(table..table.checked_add(usize::from(u16_at(0x38)?).checked_mul(56)?)?)?;
        Some(
            headers == object.headers.as_slice()
                && object.notes.iter().all(|(offset, note)| {
                    offset.checked_add(note.len()).and_then(|end| bytes.get(*offset..end)) == Some(note.as_slice())
                }),
        )
    };
    same().unwrap_or(false).then_some(file)
}

/// The major version in `libcudart.so.<major>[.minor...]`.
fn runtime_version(file_name: &str) -> Option<u32> {
    file_name.strip_prefix("libcudart.so.")?.split('.').next()?.parse().ok()
}

/// The caller of file `path`, whose contents are `file` when it is the file that was loaded
/// (`loaded_file`).
fn classify(path: &CStr, file: Option<&[u8]>) -> Caller {
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
    match file.map(|file| policy.embedded(file)) {
        Some(Embedded::Usable) => Caller::Native,
        Some(Embedded::Nothing) if *UNDER_LAUNCHER => Caller::Direct,
        Some(Embedded::Other { lowest_sm }) if policy.retargetable(lowest_sm) => Caller::From(lowest_sm),
        _ => Caller::Other,
    }
}

/// The limit that the code embedded in the file at `path` sets to what a caller that runs it is
/// told (`vf_adapt::Policy::told_limit_in`, with `policy`, which tells what the caller is told so
/// far), with the file. Each file is read once for what is told while the loader's count of
/// unloads is `unloads`.
fn file_limit(object: &Object, policy: &vf_adapt::Policy, unloads: u64) -> Option<(u32, CString)> {
    let path = object.path.as_deref()?;
    // A file that is not the one loaded, or cannot be read: what code the object runs is not
    // known, and it is taken to run from the lowest that adapted code does (7.5).
    let Some(file) = loaded_file(object) else {
        return unknown_limit(path);
    };
    let key = (path.to_owned(), policy.told_sm);
    let known = {
        let cache = callers();
        (cache.unloads == unloads)
            .then(|| cache.limits.get(&key).copied())
            .flatten()
    };
    let limit = known.unwrap_or_else(|| {
        let limit = unpanicked(path, || policy.told_limit_in(file.bytes()));
        let mut cache = callers();
        if cache.unloads == unloads {
            cache.limits.insert(key, limit);
        }
        limit
    });
    limit.map(|sm| (sm, path.to_owned()))
}

/// The limit of an object loaded from `path` whose file is not known (`loaded_file`): what
/// adapted code sets at least (7.5).
fn unknown_limit(path: &CStr) -> Option<(u32, CString)> {
    let description = format!(
        "{} (the file at that path is not the one that was loaded, or cannot be read: its code is taken to run \
         from code for sm_{}),",
        path.to_string_lossy(),
        vf_adapt::TRANSLATED_FROM
    );
    Some((vf_adapt::TRANSLATED_FROM, CString::new(description).ok()?))
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

/// The file name of `path`.
fn file_name(path: &[u8]) -> &[u8] {
    path.rsplit(|&b| b == b'/').next().unwrap_or_default()
}

/// The limit that the code in the file of the caller of class `caller` at `path` sets, when it
/// embeds code.
fn own_limit(
    object: Option<&Object>,
    caller: Caller,
    policy: &vf_adapt::Policy,
    unloads: u64,
) -> Option<(u32, CString)> {
    let object = object?;
    let path = object.path.as_deref()?;
    let own = matches!(caller, Caller::From(_))
        && runtime_version(&String::from_utf8_lossy(file_name(path.to_bytes()))).is_none();
    own.then(|| file_limit(object, policy, unloads)).flatten()
}

/// The names in the dynamic section of a shared object.
struct Names {
    soname: Option<Vec<u8>>,
    needed: Vec<Vec<u8>>,
}

/// The `DT_SONAME` and the `DT_NEEDED` names of the shared object in `file` (64-bit ELF), read
/// from its dynamic section. `None` when it is no such object.
fn dynamic_names(file: &[u8]) -> Option<Names> {
    const PT_LOAD: u32 = 1;
    const PT_DYNAMIC: u32 = 2;
    // (Every offset is the file's: added to with checks, and read within the file.)
    let u16_at = |at: usize| Some(u16::from_le_bytes(file.get(at..at.checked_add(2)?)?.try_into().ok()?));
    let u32_at = |at: usize| Some(u32::from_le_bytes(file.get(at..at.checked_add(4)?)?.try_into().ok()?));
    let u64_at = |at: usize| Some(u64::from_le_bytes(file.get(at..at.checked_add(8)?)?.try_into().ok()?) as usize);
    let field = |at: usize, offset: usize| at.checked_add(offset);
    if file.get(..6)? != b"\x7fELF\x02\x01" {
        return None;
    }
    let (table, size, count) = (u64_at(0x20)?, u16_at(0x36)? as usize, u16_at(0x38)? as usize);
    // (type, offset, address, size in the file) of each program header.
    let headers: Vec<(u32, usize, usize, usize)> = (0..count)
        .map(|i| {
            let at = table.checked_add(i.checked_mul(size)?)?;
            Some((
                u32_at(at)?,
                u64_at(field(at, 8)?)?,
                u64_at(field(at, 16)?)?,
                u64_at(field(at, 32)?)?,
            ))
        })
        .collect::<Option<_>>()?;
    let &(_, dynamic, _, length) = headers.iter().find(|header| header.0 == PT_DYNAMIC)?;
    let (mut strings, mut strings_size, mut names, mut soname) = (None, 0, Vec::new(), None);
    for at in (dynamic..dynamic.checked_add(length)?).step_by(16) {
        match (u64_at(at)?, u64_at(field(at, 8)?)?) {
            (0, _) => break,
            (1, name) => names.push(name),
            (5, address) => strings = Some(address),
            (10, size) => strings_size = size,
            (14, name) => soname = Some(name),
            _ => {}
        }
    }
    // The string table, by its address in a loaded segment.
    let strings = strings?;
    let &(_, offset, address, _) = headers.iter().find(|&&(kind, _, address, size)| {
        kind == PT_LOAD && (address..address.saturating_add(size)).contains(&strings)
    })?;
    let start = offset.checked_add(strings - address)?;
    let table = file.get(start..start.checked_add(strings_size)?)?;
    let string = |at: usize| {
        let rest = table.get(at..)?;
        Some(rest[..rest.iter().position(|&b| b == 0)?].to_vec())
    };
    Some(Names {
        soname: soname.and_then(string),
        needed: names.into_iter().filter_map(string).collect(),
    })
}

/// The sections of the 64-bit ELF object in `file` in which compilers put GPU code:
/// `.nv_fatbin`, and `__nv_relfatbin` for code linked again when it is loaded.
fn fat_binary_sections(file: &[u8]) -> Vec<&[u8]> {
    // (Every offset is the file's: added to with checks, and read within the file.)
    let u16_at = |at: usize| Some(u16::from_le_bytes(file.get(at..at.checked_add(2)?)?.try_into().ok()?) as usize);
    let u32_at = |at: usize| Some(u32::from_le_bytes(file.get(at..at.checked_add(4)?)?.try_into().ok()?) as usize);
    let u64_at = |at: usize| Some(u64::from_le_bytes(file.get(at..at.checked_add(8)?)?.try_into().ok()?) as usize);
    let sections = || -> Option<Vec<&[u8]>> {
        if file.get(..6)? != b"\x7fELF\x02\x01" {
            return None;
        }
        let (table, size, count, names) = (u64_at(0x28)?, u16_at(0x3a)?, u16_at(0x3c)?, u16_at(0x3e)?);
        // (name, offset, size) of each section header.
        let header = |i: usize| {
            let at = table.checked_add(i.checked_mul(size)?)?;
            Some((u32_at(at)?, u64_at(at.checked_add(24)?)?, u64_at(at.checked_add(32)?)?))
        };
        let (_, start, length) = header(names)?;
        let names = file.get(start..start.checked_add(length)?)?;
        let mut found = Vec::new();
        for i in 0..count {
            let (name, start, length) = header(i)?;
            let name = names.get(name..).and_then(|rest| rest.split(|&b| b == 0).next());
            if matches!(name, Some(b".nv_fatbin" | b"__nv_relfatbin")) {
                found.extend(file.get(start..start.checked_add(length)?));
            }
        }
        Some(found)
    };
    sections().unwrap_or_default()
}

/// For a caller without GPU code of its own at `path` (`Direct` or `Other`, not a CUDA runtime):
/// the libraries beside it (in its directory) that depend on it (`DT_NEEDED`), directly or
/// through each other, and that are not loaded, are libraries that it may load later, as cuDNN's
/// graph library loads the library of its engines when it needs them, often after it asked.
/// Which file the loader will take for one of them is not known before it is loaded (the search
/// path decides, or an audit library): when one of them has code of another architecture than
/// this GPU's, which is adapted, the limit is the lowest that adapted code sets (7.5). Once they
/// are loaded, the files that the loader took are read (`dependents_limit`).
fn beside_limit(path: &CStr, caller: Caller, objects: &[Object]) -> Option<(u32, CString)> {
    // The files read are any beside the caller: a panic while reading them sets no limit, as one
    // while reading the code of a caller does (`unpanicked`), instead of ending the process.
    catch_unwind(AssertUnwindSafe(|| beside_limit_unguarded(path, caller, objects))).unwrap_or_else(|_| {
        log::error(format_args!(
            "internal error while reading the libraries beside {}; they set no limit to the capability told",
            path.to_string_lossy()
        ));
        None
    })
}

fn beside_limit_unguarded(path: &CStr, caller: Caller, objects: &[Object]) -> Option<(u32, CString)> {
    let bytes = path.to_bytes();
    let own_name = file_name(bytes);
    if !matches!(caller, Caller::Direct | Caller::Other)
        || runtime_version(&String::from_utf8_lossy(own_name)).is_some()
    {
        return None;
    }
    let directory = &bytes[..bytes.len() - own_name.len()];
    let names_of = |object: &Object| {
        let file = object.path.as_ref().map(|path| file_name(path.to_bytes()).to_vec());
        object.soname.iter().cloned().chain(file).collect::<Vec<_>>()
    };
    let loaded: Vec<Vec<u8>> = objects.iter().flat_map(names_of).collect();
    let mut names = objects
        .iter()
        .find(|object| object.holds)
        .map(names_of)
        .unwrap_or_else(|| vec![own_name.to_vec()]);
    // The shared objects of the directory that are not loaded.
    struct Beside {
        path: CString,
        /// Its file name and its `DT_SONAME`, by which others need it.
        names: Vec<Vec<u8>>,
        needed: Vec<Vec<u8>>,
    }
    let directory_path = std::path::Path::new(std::ffi::OsStr::from_bytes(if directory.is_empty() {
        b"."
    } else {
        directory
    }));
    // A link is followed (cuDNN's libraries are links to files, and a link can lead to another
    // directory), and a file is read once, under all its names in the directory.
    let mut beside: Vec<Beside> = Vec::new();
    let mut files: HashMap<(u64, u64), Option<usize>> = HashMap::new();
    let mut entries: Vec<Vec<u8>> = std::fs::read_dir(directory_path)
        .ok()?
        .flatten()
        .map(|entry| entry.file_name().into_vec())
        .collect();
    entries.sort();
    for name in entries {
        if name.as_slice() == own_name || !name.windows(3).any(|w| w == b".so") || loaded.contains(&name) {
            continue;
        }
        let Ok(path) = CString::new([directory, &name].concat()) else {
            continue;
        };
        let Ok(metadata) = std::fs::metadata(std::ffi::OsStr::from_bytes(path.to_bytes())) else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        use std::os::unix::fs::MetadataExt as _;
        let file = (metadata.dev(), metadata.ino());
        if let Some(read) = files.get(&file) {
            if let Some(at) = *read {
                beside[at].names.push(name);
            }
            continue;
        }
        let Some(Names { soname, needed }) = Mapped::open(&path).and_then(|file| dynamic_names(file.bytes())) else {
            files.insert(file, None);
            continue;
        };
        if soname.as_ref().is_some_and(|soname| loaded.contains(soname)) {
            files.insert(file, None);
            continue;
        }
        files.insert(file, Some(beside.len()));
        let names = soname.into_iter().chain([name]).collect();
        beside.push(Beside { path, names, needed });
    }
    beside.sort_by(|a, b| a.path.cmp(&b.path));
    // Those that depend on the caller, directly or through each other.
    let mut dependents = Vec::new();
    loop {
        let (taken, rest): (Vec<_>, Vec<_>) = beside
            .into_iter()
            .partition(|object| object.needed.iter().any(|needed| names.contains(needed)));
        if taken.is_empty() {
            break;
        }
        beside = rest;
        for object in taken {
            names.extend(object.names);
            dependents.push(object.path);
        }
    }
    let policy = &config().policy;
    let adapted = dependents.into_iter().find(|file_path| {
        Mapped::open(file_path).is_some_and(|file| {
            fat_binary_sections(file.bytes()).into_iter().any(|section| {
                let embedded = catch_unwind(AssertUnwindSafe(|| policy.embedded(section)));
                matches!(embedded, Ok(vf_adapt::Embedded::Other { .. }))
            })
        })
    })?;
    let description = format!(
        "{}, which it may load, not loaded yet (until it is, which file the loader takes for it is not known, \
         and its code is taken to run from code for sm_{}),",
        adapted.to_string_lossy(),
        vf_adapt::TRANSLATED_FROM
    );
    Some((vf_adapt::TRANSLATED_FROM, CString::new(description).ok()?))
}

/// The limit that the code of the libraries loaded with the caller as a dependency sets to what
/// it is told, and the library that sets it: those that name it in their `DT_NEEDED`, and those
/// that name one of these, and so on (a CUDA runtime answers for the programs that use it, also
/// through a library of their own that has no GPU code). Their files are read once each. A
/// library loaded after the caller first asks is not seen.
fn dependents_limit(objects: &[Object], policy: &vf_adapt::Policy, unloads: u64) -> Option<(u32, CString)> {
    let names_of = |object: &Object| {
        let file_name = object.path.as_ref().map(|path| {
            path.to_bytes()
                .rsplit(|&b| b == b'/')
                .next()
                .unwrap_or_default()
                .to_vec()
        });
        object.soname.iter().cloned().chain(file_name).collect::<Vec<_>>()
    };
    let caller = objects.iter().position(|object| object.holds)?;
    let mut names = names_of(&objects[caller]);
    let mut seen = vec![false; objects.len()];
    seen[caller] = true;
    let mut lowest: Option<(u32, CString)> = None;
    loop {
        let mut more = false;
        for (at, object) in objects.iter().enumerate() {
            if seen[at] || !object.needed.iter().any(|needed| names.contains(needed)) {
                continue;
            }
            (seen[at], more) = (true, true);
            if object.path.is_some() {
                lowest = lower(lowest, file_limit(object, policy, unloads));
            }
            names.extend(names_of(object));
        }
        if !more {
            return lowest;
        }
    }
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
            program: false,
        };
    }
    let unloads = unload_count();
    let base = info.dli_fbase as usize;
    {
        let mut cache = callers();
        if cache.unloads != unloads {
            cache.by_base.clear();
            cache.limits.clear();
            cache.unloads = unloads;
        }
        if let Some(decision) = cache.by_base.get(&base) {
            return *decision;
        }
    }
    // Outside the lock (see `CALLERS`).
    let objects = objects(return_address);
    // The main program's file, as `objects` resolved it.
    let main = objects
        .first()
        .filter(|object| object.holds)
        .and_then(|object| object.path.as_deref());
    let path = main.unwrap_or_else(|| unsafe { CStr::from_ptr(info.dli_fname) });
    // Its code is read from its file only when that is the file it was loaded from: a path can
    // name another file once the object is loaded (replaced, or removed).
    let own = objects.iter().find(|object| object.holds);
    let file = own.and_then(loaded_file);
    let caller = classify(path, file.as_ref().map(Mapped::bytes));
    let runtime = runtime_version(&String::from_utf8_lossy(file_name(path.to_bytes()))).is_some();
    let unknown = (file.is_none() && !runtime).then(|| unknown_limit(path)).flatten();
    drop(file);
    let mut limit = None;
    if limits_apply() && caller != Caller::Native {
        // Told the lowest limit that its code and the code it answers for set, the caller is given
        // the code for it, which can set a lower one: lowered until nothing lowers it more
        // (`vf_adapt::Policy::lowered`), keeping the file that sets it.
        let beside = lower(beside_limit(path, caller, &objects), unknown);
        let mut policy = config().policy;
        loop {
            let found = lower(
                lower(own_limit(own, caller, &policy, unloads), beside.clone()),
                dependents_limit(&objects, &policy, unloads),
            );
            match found {
                Some((sm, file)) if policy.told_sm.is_some_and(|told| sm < told) => {
                    policy.told_sm = Some(sm);
                    limit = Some((sm, file));
                }
                _ => break,
            }
        }
    }
    let program = runtime
        || own.is_some_and(|object| {
            object.needed.iter().any(|name| {
                runtime_version(&String::from_utf8_lossy(name)).is_some_and(|v| v >= FIRST_RUNTIME_WITHOUT_DEVICE)
            })
        });
    let decision = Decision {
        caller,
        limit: limit.as_ref().map(|(sm, _)| *sm),
        program,
    };
    {
        let mut cache = callers();
        if cache.unloads == unloads {
            // (Another thread may have decided meanwhile, and said it.)
            if let Some(decided) = cache.by_base.get(&base) {
                return *decided;
            }
            cache.by_base.insert(base, decision);
        }
    }
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
            if config().capability_all && program {
                log::warn(format_args!(
                    "{name} is told compute capability {major}.{minor}, as VF_CAPABILITY and VF_CAPABILITY_ALL ask, \
                     although {whose} runs here from its code for sm_{sm}, in which the \
                     kernels that a program told {major}.{minor} chooses are not the real ones"
                ))
            } else {
                log::warn(format_args!(
                    "{name} is told compute capability {}.{}, not the {major}.{minor} that VF_CAPABILITY asks: {whose} \
                     runs here from its code for sm_{sm}, in which the kernels that a \
                     program told {major}.{minor} chooses are not the real ones ({})",
                    sm / 10,
                    sm % 10,
                    if program {
                        format!("VF_CAPABILITY_ALL=1 tells it {major}.{minor} all the same")
                    } else {
                        "VF_CAPABILITY_ALL=1 lifts the limit of a CUDA runtime and of the code that uses it only"
                            .to_string()
                    }
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
    decision
}

/// The architecture for which the images that the caller at `return_address` loads are adapted
/// (`vf_adapt::Policy::told_sm`), when it is not the one `VF_CAPABILITY` asks: the one the caller
/// is told, where the code it runs limits it (`vf_adapt::Policy::told_limit`). Told 7.5, a caller
/// is then given the sm_75 code, or the PTX for sm_75, of every image it loads that has some, not
/// the code for sm_80 that the image may also have. The caller that loads an image is the caller
/// that asks what the GPU is, a CUDA runtime for the programs that use it. `None` for a CUDA
/// runtime and the code that uses it with `VF_CAPABILITY_ALL=1`, which tells them what
/// `VF_CAPABILITY` asks (`Decision::program`).
pub fn told_sm(return_address: usize) -> Option<u32> {
    if !limits_apply() {
        return None;
    }
    let decision = caller(return_address);
    decision
        .limit
        .filter(|_| !(config().capability_all && decision.program))
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
    let limit = decision
        .limit
        .filter(|_| !(config().capability_all && decision.program));
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
    fn the_libraries_an_object_needs_are_read_from_its_file() {
        let file = std::fs::read("/proc/self/exe").unwrap();
        let Names { soname, needed } = dynamic_names(&file).unwrap();
        assert_eq!(soname, None);
        assert!(needed.iter().any(|name| name.starts_with(b"libc.so")), "{needed:?}");
        assert!(dynamic_names(b"\x7fELF\x02\x01 a file cut short").is_none());
        assert!(dynamic_names(b"not an object").is_none());
        assert!(fat_binary_sections(&file).is_empty());
        assert!(fat_binary_sections(b"\x7fELF\x02\x01 a file cut short").is_empty());
        // Tables said to be at the end of what an offset can be: nothing read, no overflow (until
        // 2026-11-23 the field after the table's offset was computed unchecked, which a debug build
        // stops at, in the capability hook).
        let mut wild = vec![0u8; 0x40];
        wild[..6].copy_from_slice(b"\x7fELF\x02\x01");
        for (table, size, count) in [(0x20, 0x36, 0x38), (0x28, 0x3a, 0x3c)] {
            wild[table..table + 8].copy_from_slice(&(u64::MAX - 3).to_le_bytes());
            wild[size..size + 2].copy_from_slice(&64u16.to_le_bytes());
            wild[count..count + 2].copy_from_slice(&1u16.to_le_bytes());
        }
        wild[0x3e..0x40].copy_from_slice(&0u16.to_le_bytes());
        assert!(dynamic_names(&wild).is_none());
        assert!(fat_binary_sections(&wild).is_empty());
    }

    #[test]
    fn a_scan_that_panics_sets_no_limit() {
        assert_eq!(unpanicked(c"/lib/a.so", || Some(75)), Some(75));
        assert_eq!(unpanicked(c"/lib/a.so", || panic!("an odd input")), None);
    }
}
