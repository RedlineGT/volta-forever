//! LD_AUDIT shared library that substitutes libraries by file name.
//!
//! The dynamic loader calls `la_objsearch` for every library it tries to open, including
//! `dlopen` of an absolute path. This library answers with a replacement path from a map file.
//!
//! Setup: `LD_AUDIT=/path/to/libvf_audit.so` is all a user needs.
//! The substitution table is read from `VF_AUDIT_MAP` (env, first priority: overrides the
//! file) and from a file named `vf-audit.map` next to the library (or `VF_AUDIT_MAP_FILE`).
//! Format: one `name=replacement` per line (# comments, empty lines allowed; replacements that
//! are relative paths are resolved relative to the map file's directory; an empty replacement
//! makes the library not found). The first matching entry wins, and env entries come first, so
//! the environment can override the file.
//!
//! A library named in the map as a replacement that opens a file by its path is left alone: a
//! forwarder opens the library it forwards to that way, and must get it. A bare name is looked up
//! in the map whoever asks, so that a substituted library finds its own companions and not the
//! program's (cuDNN opens its parts by name).
//!
//! One setting is not a library: `@cudnn-version=9.24.0` makes cuDNN's version queries
//! (`cudnnGetProperty`, `cudnnGetVersion`) answer that version. A program refuses a cuDNN older
//! than the one it was built with (XLA does, and then fails), while the cuDNN that still has code
//! for the GPU is an older one with the same interface. Under the VF launcher, the NVML compute
//! capability query also reports the application-facing architecture of a Volta GPU. Dispatchers
//! such as vLLM use NVML before initializing CUDA.
//!
//! Logging: `VF_LOG`, `VF_LOG_LEVEL` (level 2 = each substitution).

use std::ffi::{CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

// --- Logging ---

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
                Err(e) => eprintln!("[vf-audit] cannot open log file {path}: {e}"),
            }
        }
    }

    fn write(level: u8, args: Arguments<'_>) {
        if LEVEL.load(Relaxed) < level {
            return;
        }
        let line = format!("[vf-audit] {args}\n");
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
}

// --- Map entries ---

struct MapEntry {
    /// The library file name to match (e.g. "libcublas.so.13").
    name: String,
    /// The replacement path (absolute). Empty means "not found".
    replacement: CString,
}

struct AuditState {
    entries: Vec<MapEntry>,
    /// Cookies of our own libraries (libraries whose path matches a replacement).
    own_cookies: std::sync::Mutex<Vec<usize>>,
    /// What cuDNN's version queries answer (major, minor, patch), if set.
    cudnn_version: Option<[libc::c_int; 3]>,
    /// The compute capability advertised to NVML clients running under the VF launcher.
    nvml_sm: Option<u32>,
}

/// `@cudnn-version=9.24.0` among the entries: taken out of them.
fn take_cudnn_version(entries: &mut Vec<MapEntry>) -> Option<[libc::c_int; 3]> {
    let at = entries.iter().position(|entry| entry.name == "@cudnn-version")?;
    let entry = entries.remove(at);
    // The value went through `resolve_path`: only its last component is what was written.
    let text = String::from_utf8_lossy(basename(entry.replacement.to_bytes())).into_owned();
    let parts: Vec<libc::c_int> = text.split('.').filter_map(|part| part.parse().ok()).collect();
    match parts[..] {
        [major, minor, patch] if text.split('.').count() == 3 => Some([major, minor, patch]),
        _ => {
            log::error(format_args!(
                "@cudnn-version={text}: expected major.minor.patch; ignored"
            ));
            None
        }
    }
}

fn find_map_file() -> Option<PathBuf> {
    // Check VF_AUDIT_MAP_FILE first.
    if let Ok(path) = std::env::var("VF_AUDIT_MAP_FILE") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }
    // Look next to the library itself.
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    for line in maps.lines() {
        if line.contains("libvf_audit") {
            // Extract the path from the maps line.
            if let Some(path_start) = line.find('/') {
                let lib_path = Path::new(&line[path_start..]);
                if let Some(dir) = lib_path.parent() {
                    let map_file = dir.join("vf-audit.map");
                    if map_file.exists() {
                        return Some(map_file);
                    }
                }
            }
        }
    }
    None
}

fn parse_map_env(base_dir: Option<&Path>) -> Vec<MapEntry> {
    let mut entries = Vec::new();

    // Parse VF_AUDIT_MAP (colon-separated name=path pairs, like the prototype).
    if let Ok(spec) = std::env::var("VF_AUDIT_MAP") {
        for item in spec.split(':') {
            if let Some((name, replacement)) = item.split_once('=') {
                let name = name.trim().to_string();
                let replacement = resolve_path(replacement.trim(), base_dir);
                entries.push(MapEntry {
                    name,
                    replacement: CString::new(replacement).unwrap_or_default(),
                });
            }
        }
    }

    entries
}

fn parse_map_file(path: &Path) -> Vec<MapEntry> {
    let mut entries = Vec::new();
    let base_dir = path.parent();
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            log::error(format_args!("cannot read map file {}: {e}", path.display()));
            return entries;
        }
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, replacement)) = line.split_once('=') {
            let name = name.trim().to_string();
            let replacement = resolve_path(replacement.trim(), base_dir);
            entries.push(MapEntry {
                name,
                replacement: CString::new(replacement).unwrap_or_default(),
            });
        }
    }
    entries
}

fn resolve_path(replacement: &str, base_dir: Option<&Path>) -> Vec<u8> {
    if replacement.is_empty() {
        return Vec::new();
    }
    let path = Path::new(replacement);
    if path.is_absolute() {
        return replacement.as_bytes().to_vec();
    }
    // Relative path: resolve against base_dir.
    if let Some(base) = base_dir {
        let resolved = base.join(path);
        return resolved.as_os_str().as_bytes().to_vec();
    }
    replacement.as_bytes().to_vec()
}

/// Check that a non-empty replacement file exists. Log a warning if not.
fn validate_entry(entry: &MapEntry) {
    let repl = entry.replacement.to_bytes();
    if repl.is_empty() {
        // "not found" entry: no file to check.
        return;
    }
    let path = Path::new(std::ffi::OsStr::from_bytes(repl));
    if !path.exists() {
        log::warn(format_args!(
            "{} -> {}: replacement file does not exist (lookups will fail at load time)",
            entry.name,
            entry.replacement.to_string_lossy()
        ));
    }
}

fn init_state() -> AuditState {
    log::open();

    let map_file = find_map_file();
    let map_dir = map_file.as_deref().and_then(|p| p.parent());

    // Env entries FIRST (first match wins, so env overrides the file).
    let mut entries = parse_map_env(map_dir);

    if let Some(ref path) = map_file {
        log::info(format_args!("map file: {}", path.display()));
        let file_entries = parse_map_file(path);
        entries.extend(file_entries);
    }

    let mut cudnn_version = None;
    while let Some(version) = take_cudnn_version(&mut entries) {
        // The first one stands (the environment comes first).
        cudnn_version = cudnn_version.or(Some(version));
    }
    if let Some([major, minor, patch]) = cudnn_version {
        log::info(format_args!("cuDNN version queries answer {major}.{minor}.{patch}"));
    }

    for e in &entries {
        validate_entry(e);
        let repl = e.replacement.to_string_lossy();
        if repl.is_empty() {
            log::info(format_args!("{} -> (not found)", e.name));
        } else {
            log::info(format_args!("{} -> {}", e.name, repl));
        }
    }

    AuditState {
        entries,
        own_cookies: std::sync::Mutex::new(Vec::new()),
        cudnn_version,
        // (A value that is not used is said by the proxy, once.)
        nvml_sm: std::env::var_os("VOLTA_FOREVER").map(|_| {
            vf_adapt::requested_capability()
                .ok()
                .flatten()
                .unwrap_or(vf_adapt::TRANSLATED_FROM)
        }),
    }
}

static STATE: OnceLock<AuditState> = OnceLock::new();

fn state() -> &'static AuditState {
    STATE.get_or_init(init_state)
}

/// Extract the file name (basename) from a path.
fn basename(path: &[u8]) -> &[u8] {
    match path.iter().rposition(|&b| b == b'/') {
        Some(pos) => &path[pos + 1..],
        None => path,
    }
}

/// Check if a cookie belongs to one of our own libraries.
fn is_own_library(cookie: usize) -> bool {
    let s = state();
    let cookies = s.own_cookies.lock().unwrap_or_else(|e| e.into_inner());
    cookies.contains(&cookie)
}

// --- rtld-audit interface ---

// The x86_64 callbacks used here have the same ABI for audit versions 1 and 2.
const LAV_CURRENT: libc::c_uint = 2;
/// `la_objopen` flags: bindings to, and from, this object are reported to `la_symbind64`.
const LA_FLG_BINDTO: libc::c_uint = 1;
const LA_FLG_BINDFROM: libc::c_uint = 2;
/// `la_symbind64` flags: no call of the symbol is reported afterwards.
const LA_SYMB_NOPLTENTER: libc::c_uint = 1;
const LA_SYMB_NOPLTEXIT: libc::c_uint = 2;

/// Called by the dynamic loader to determine the audit interface version.
#[unsafe(no_mangle)]
pub extern "C" fn la_version(version: libc::c_uint) -> libc::c_uint {
    // Initialize state (reads the map).
    let _ = state();
    version.min(LAV_CURRENT)
}

/// Called when a shared object is loaded. We use this to record cookies of our own libraries.
///
/// # Safety
/// Called by the dynamic loader with valid `map` (a `struct link_map*`) and `cookie` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn la_objopen(
    map: *mut libc::c_void, // struct link_map*
    lmid: libc::c_long,
    cookie: *mut libc::c_ulong,
) -> libc::c_uint {
    if map.is_null() || cookie.is_null() {
        return 0;
    }

    // Get the library's name from the link_map.
    // struct link_map { l_addr, l_name, ... }
    // l_name is at offset sizeof(ElfW(Addr)) on the struct.
    // On x86_64: offset 8.
    let l_name_ptr: *const *const libc::c_char =
        unsafe { (map as *const u8).add(std::mem::size_of::<usize>()) as *const _ };
    let l_name = unsafe { *l_name_ptr };

    if !l_name.is_null() {
        let name = unsafe { CStr::from_ptr(l_name) };
        let name_bytes = name.to_bytes();

        // Check if this library's path matches any of our replacement paths.
        let s = state();
        for entry in &s.entries {
            let repl_bytes = entry.replacement.to_bytes();
            if !repl_bytes.is_empty() && name_bytes == repl_bytes {
                let cookie_val = unsafe { *cookie } as usize;
                let mut cookies = s.own_cookies.lock().unwrap_or_else(|e| e.into_inner());
                if !cookies.contains(&cookie_val) {
                    cookies.push(cookie_val);
                    log::info(format_args!(
                        "registered own library: {} (cookie {:#x})",
                        name.to_string_lossy(),
                        cookie_val
                    ));
                }
                break;
            }
        }
    }

    let s = state();
    if (s.cudnn_version.is_none() && s.nvml_sm.is_none()) || l_name.is_null() {
        return 0;
    }
    let name = basename(unsafe { CStr::from_ptr(l_name) }.to_bytes());
    if lmid == libc::LM_ID_BASE
        && s.nvml_sm.is_some()
        && (name == b"libnvidia-ml.so" || name.starts_with(b"libnvidia-ml.so."))
    {
        // A second copy has its own handles. Let that copy serve its own clients.
        if NVML_COOKIE
            .compare_exchange(0, unsafe { *cookie } as usize, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return LA_FLG_BINDTO | LA_FLG_BINDFROM;
        }
    }
    if s.cudnn_version.is_some() && name.starts_with(b"libcudnn") {
        return LA_FLG_BINDTO | LA_FLG_BINDFROM;
    }
    LA_FLG_BINDFROM
}

fn s_cudnn_version() -> Option<[libc::c_int; 3]> {
    state().cudnn_version
}

static NVML_COOKIE: AtomicUsize = AtomicUsize::new(0);
static REAL_NVML_CAPABILITY: AtomicUsize = AtomicUsize::new(0);

/// Release the recorded NVML instance so a later dlopen can bind its own implementation.
///
/// # Safety
/// The loader supplies the cookie from `la_objopen`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn la_objclose(cookie: *mut libc::c_ulong) -> libc::c_uint {
    if !cookie.is_null() {
        let _ = NVML_COOKIE.compare_exchange(unsafe { *cookie } as usize, 0, Ordering::Relaxed, Ordering::Relaxed);
    }
    0
}

unsafe extern "C" fn nvml_device_get_cuda_compute_capability(
    device: *mut libc::c_void,
    major: *mut libc::c_int,
    minor: *mut libc::c_int,
) -> libc::c_int {
    type Query = unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_int, *mut libc::c_int) -> libc::c_int;
    // la_symbind64 receives NVML's resolved address before it hands out this wrapper.
    let query: Query = unsafe { std::mem::transmute(REAL_NVML_CAPABILITY.load(Ordering::Relaxed)) };
    let status = unsafe { query(device, major, minor) };
    if status == 0
        && !major.is_null()
        && !minor.is_null()
        && unsafe { (*major, *minor) } == (7, 0)
        && let Some(sm) = state().nvml_sm
    {
        unsafe {
            *major = (sm / 10) as libc::c_int;
            *minor = (sm % 10) as libc::c_int;
        }
    }
    status
}

/// The functions of the loaded cuDNN that the two below stand for.
static REAL_PROPERTY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `cudnnGetProperty`: the loaded cuDNN's answer, with the version parts replaced.
unsafe extern "C" fn cudnn_get_property(kind: libc::c_int, value: *mut libc::c_int) -> libc::c_int {
    let real = REAL_PROPERTY.load(std::sync::atomic::Ordering::Relaxed);
    // Bound before this function was handed out.
    let real: unsafe extern "C" fn(libc::c_int, *mut libc::c_int) -> libc::c_int = unsafe { std::mem::transmute(real) };
    let status = unsafe { real(kind, value) };
    if let (0, Some(version), false) = (status, s_cudnn_version(), value.is_null())
        && let Some(part) = usize::try_from(kind).ok().and_then(|kind| version.get(kind))
    {
        unsafe { *value = *part };
    }
    status
}

/// `cudnnGetVersion`.
unsafe extern "C" fn cudnn_get_version() -> usize {
    let [major, minor, patch] = s_cudnn_version().unwrap_or_default();
    (major * 10000 + minor * 100 + patch) as usize
}

/// Called when a symbol is bound between two objects that `la_objopen` flagged. Returns the address
/// the reference gets.
///
/// # Safety
/// Called by the dynamic loader with valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn la_symbind64(
    sym: *mut libc::Elf64_Sym,
    _index: libc::c_uint,
    _referrer: *mut libc::c_ulong,
    definer: *mut libc::c_ulong,
    flags: *mut libc::c_uint,
    name: *const libc::c_char,
) -> usize {
    if !flags.is_null() {
        unsafe { *flags |= LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT };
    }
    let value = unsafe { (*sym).st_value } as usize;
    if name.is_null() {
        return value;
    }
    match unsafe { CStr::from_ptr(name) }.to_bytes() {
        b"nvmlDeviceGetCudaComputeCapability" if !definer.is_null() => {
            let cookie = unsafe { *definer } as usize;
            if cookie == 0 || cookie != NVML_COOKIE.load(Ordering::Relaxed) {
                return value;
            }
            REAL_NVML_CAPABILITY.store(value, Ordering::Relaxed);
            nvml_device_get_cuda_compute_capability as *const () as usize
        }
        b"cudnnGetProperty" if s_cudnn_version().is_some() => {
            REAL_PROPERTY.store(value, std::sync::atomic::Ordering::Relaxed);
            log::info(format_args!("cudnnGetProperty answers the configured version"));
            cudnn_get_property as *const () as usize
        }
        b"cudnnGetVersion" if s_cudnn_version().is_some() => {
            log::info(format_args!("cudnnGetVersion answers the configured version"));
            cudnn_get_version as *const () as usize
        }
        _ => value,
    }
}

/// Called for every library lookup. Returns the path to use, or NULL to make it not found.
///
/// # Safety
/// Called by the dynamic loader with a valid `name` C string and `cookie` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn la_objsearch(
    name: *const libc::c_char,
    cookie: *mut libc::c_ulong,
    _flag: libc::c_uint,
) -> *mut libc::c_char {
    if name.is_null() {
        return std::ptr::null_mut();
    }

    let name_cstr = unsafe { CStr::from_ptr(name) };
    let name_bytes = name_cstr.to_bytes();
    let base = basename(name_bytes);

    // One of our own libraries that names a file by its path gets that file.
    if !cookie.is_null() && name_bytes.contains(&b'/') {
        let cookie_val = unsafe { *cookie } as usize;
        if is_own_library(cookie_val) {
            return name as *mut libc::c_char;
        }
    }

    let s = state();
    for entry in &s.entries {
        if base != entry.name.as_bytes() {
            continue;
        }
        // Don't redirect if the name already IS the replacement.
        let repl_bytes = entry.replacement.to_bytes();
        if name_bytes == repl_bytes {
            break;
        }
        if repl_bytes.is_empty() {
            log::info(format_args!("{} -> (not found)", name_cstr.to_string_lossy()));
            return std::ptr::null_mut();
        }
        log::info(format_args!(
            "{} -> {}",
            name_cstr.to_string_lossy(),
            entry.replacement.to_string_lossy()
        ));
        return entry.replacement.as_ptr() as *mut libc::c_char;
    }

    name as *mut libc::c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn negotiate_audit_versions() {
        assert_eq!(la_version(0), 0);
        assert_eq!(la_version(1), 1);
        assert_eq!(la_version(2), 2);
        assert_eq!(la_version(3), 2);
    }

    #[test]
    fn basename_extracts_filename() {
        assert_eq!(basename(b"/usr/lib/libfoo.so"), b"libfoo.so");
        assert_eq!(basename(b"libfoo.so"), b"libfoo.so");
        assert_eq!(basename(b"/a/b/c/d.so.1"), b"d.so.1");
        assert_eq!(basename(b""), b"");
    }

    #[test]
    fn resolve_absolute_path() {
        assert_eq!(
            resolve_path("/usr/lib/libfoo.so", Some(Path::new("/tmp"))),
            b"/usr/lib/libfoo.so"
        );
    }

    #[test]
    fn resolve_relative_path() {
        let result = resolve_path("../lib/libfoo.so", Some(Path::new("/opt/cuda/map")));
        let expected = OsStr::new("/opt/cuda/map/../lib/libfoo.so");
        assert_eq!(OsStr::from_bytes(&result), expected);
    }

    #[test]
    fn resolve_empty_means_not_found() {
        assert_eq!(resolve_path("", Some(Path::new("/tmp"))), b"");
    }

    #[test]
    fn the_cudnn_version_setting_is_not_a_library() {
        let entry = |name: &str, value: &str| MapEntry {
            name: name.into(),
            replacement: CString::new(resolve_path(value, Some(Path::new("/maps")))).unwrap(),
        };
        let mut entries = vec![entry("libfoo.so", "/x/libfoo.so"), entry("@cudnn-version", "9.24.0")];
        assert_eq!(take_cudnn_version(&mut entries), Some([9, 24, 0]));
        assert_eq!(entries.len(), 1);
        assert_eq!(take_cudnn_version(&mut entries), None);
        let mut wrong = vec![entry("@cudnn-version", "9.24")];
        assert_eq!(take_cudnn_version(&mut wrong), None);
        assert!(wrong.is_empty());
    }

    #[test]
    fn env_overrides_file_first_match_wins() {
        // Simulate what init_state does: env entries first, then file entries.
        // First match wins, so if both have the same name, the env entry is used.
        let env_entry = MapEntry {
            name: "libfoo.so".into(),
            replacement: CString::new("/env/libfoo.so").unwrap(),
        };
        let file_entry = MapEntry {
            name: "libfoo.so".into(),
            replacement: CString::new("/file/libfoo.so").unwrap(),
        };
        let entries = vec![env_entry, file_entry];

        // la_objsearch uses first-match, so the env entry should win.
        let base = basename(b"libfoo.so");
        let mut matched = None;
        for entry in &entries {
            if base == entry.name.as_bytes() {
                matched = Some(entry.replacement.to_string_lossy().into_owned());
                break;
            }
        }
        assert_eq!(matched.as_deref(), Some("/env/libfoo.so"));
    }
}
