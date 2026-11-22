//! Images as programs hand them to the driver, and their replacements.
//!
//! A program gives the driver a pointer to a fat-binary wrapper, a fat binary, a cubin or PTX text, most
//! often without a length. `adapt` looks at what the pointer leads to and, when the image carries
//! nothing the driver can use on this GPU but can be adapted, builds the image to load instead.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CStr, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use vf_adapt::{Adapted, Emulated, Outcome, Refused, RefusedFunction};
use vf_fatbin as fatbin;

use crate::{config, kernels, log};

/// The 24-byte structure through which the CUDA runtime passes a fat binary.
#[repr(C)]
#[derive(Clone, Copy)]
struct Wrapper {
    magic: u32,
    version: u32,
    data: *const u8,
    /// Version 1: file name; version 2: array of further containers.
    extra: *const c_void,
}

/// The driver's error for PTX its compiler refuses.
pub const CUDA_ERROR_INVALID_PTX: i32 = 218;
/// Attempts at loading an image without the functions the compiler refuses in it. Its messages name
/// all of them at once unless it stops on one it cannot parse.
pub const REFUSAL_ROUNDS: usize = 8;
/// Size of the buffer given to the driver for the compiler's messages: room for thousands of them.
pub const ERROR_LOG_SIZE: usize = 1 << 20;

struct PrelinkedObjects {
    shared: OnceLock<vf_ptx::deps::LinkedShared>,
    objects: Vec<Box<[u8]>>,
    _pointers: Box<[*const u8]>,
}

/// An adapted image, owned by this library for as long as the driver may read it, and what the
/// adaptation found about its kernels.
pub struct Replacement {
    data: Box<[u8]>,
    wrapper: Option<Box<Wrapper>>,
    // A v2 wrapper's fourth word points to this null-terminated array. Every pointed-to object
    // and the pointer array itself must survive CUDA's deferred linking.
    prelinked: Option<PrelinkedObjects>,
    refused: Vec<Refused>,
    emulated: Vec<Emulated>,
    /// How the code reached the device, for the profiler.
    pub origin: Option<kernels::Origin>,
    /// For a second translation at launch (`fit`): the cubin that this is the translation of, when
    /// a kernel of the translation takes more registers than its budget.
    pub original_image: Option<Vec<u8>>,
}

// SAFETY: the wrapper points at owned, immutable data. Its fourth word is either the program
// file name (v1), or an immutable array and objects owned by this replacement (v2).
unsafe impl Send for Replacement {}
// Shared references are published only after the metadata is registered. The image, wrapper and
// relocation inputs remain immutable, just as they must for concurrent lookups by the driver.
unsafe impl Sync for Replacement {}

impl Replacement {
    fn new(data: Vec<u8>, wrapper: Option<Wrapper>) -> Replacement {
        let data = data.into_boxed_slice();
        let wrapper = wrapper.map(|w| {
            Box::new(Wrapper {
                data: data.as_ptr(),
                ..w
            })
        });
        Replacement {
            data,
            wrapper,
            prelinked: None,
            refused: Vec::new(),
            emulated: Vec::new(),
            origin: None,
            original_image: None,
        }
    }

    /// The image itself, where `as_ptr` may lead to a wrapper: a fat binary or text.
    pub fn data_ptr(&self) -> *const c_void {
        self.data.as_ptr() as *const c_void
    }

    /// Kernels of this image that would trap, which nothing keeps from the program on the path the
    /// image takes (the linker).
    pub fn refused(&self) -> &[Refused] {
        &self.refused
    }

    /// Kernels that run emulated instructions.
    pub fn emulated(&self) -> &[Emulated] {
        &self.emulated
    }

    /// The image to load after the driver's compiler refused functions of this one, as its `log`
    /// says, and those functions. See `vf_adapt::Policy::without_refused`.
    pub fn without_refused(&self, log: &str) -> Option<(Replacement, Vec<RefusedFunction>)> {
        if self.has_prelinked_objects() {
            return None;
        }
        let retry = config().policy.without_refused(&self.data, log)?;
        let mut next = Replacement::new(retry.image, self.wrapper.as_deref().copied());
        next.refused = self.refused.iter().cloned().chain(retry.kernels).collect();
        next.emulated = self.emulated.clone();
        next.origin = self.origin;
        next.original_image = self.original_image.clone();
        Some((next, retry.functions))
    }

    /// The linked inputs are meaningful only together, via `as_ptr`.
    pub fn has_prelinked_objects(&self) -> bool {
        self.prelinked.is_some()
    }

    /// A unique device-usable PTX input. If the driver could choose a different input, its linker
    /// must decide which definitions survive.
    fn linked_object_ptx(&self, index: usize) -> Option<std::borrow::Cow<'_, [u8]>> {
        let fat = fatbin::parse(self.prelinked.as_ref()?.objects.get(index)?).ok()?;
        let mut usable = fat.entries.iter().filter(|e| config().policy.usable(e));
        let entry = usable.next()?;
        if entry.kind() != fatbin::KIND_PTX || usable.next().is_some() {
            return None;
        }
        entry.payload().ok()
    }

    /// Keep the defining input indices in link order for entries present in the linked cubin.
    pub fn linked_entries(&self) -> Option<HashMap<String, Vec<usize>>> {
        let primary = fatbin::parse(&self.data).ok()?;
        let [binary] = primary.entries.as_slice() else {
            return None;
        };
        if binary.kind() != fatbin::KIND_ELF {
            return None;
        }
        let cubin = binary.payload().ok()?;
        let mut entries: HashMap<_, _> = fatbin::cubin_entries(&cubin)?
            .into_iter()
            .map(|name| (name.to_owned(), Vec::new()))
            .collect();
        drop(cubin);
        let prelinked = self.prelinked.as_ref()?;
        let mut shared = vf_ptx::deps::LinkedShared::default();
        for index in 0..prelinked.objects.len() {
            let text = self.linked_object_ptx(index)?;
            for name in shared.add_object(&text)? {
                if let Some(objects) = entries.get_mut(&name) {
                    objects.push(index);
                }
            }
        }
        prelinked.shared.get_or_init(|| shared);
        Some(entries)
    }

    /// NVCC can emit the same weak template in several objects. Use one complete leaf only
    /// when all candidate definitions are independent and their executable PTX agrees exactly.
    pub fn linked_leaf(&self, name: &str, indices: &[usize]) -> Option<Vec<u8>> {
        let shared = self.prelinked.as_ref()?.shared.get()?;
        let (&first, rest) = indices.split_first()?;
        let text = self.linked_object_ptx(first)?;
        let module = vf_ptx::deps::ModuleInfo::analyse(&text);
        if rest.is_empty() {
            return module.extract_linked_leaf(name, shared);
        }
        let definition = module.weak_leaf_definition(name, shared)?;
        for &index in rest {
            let text = self.linked_object_ptx(index)?;
            if vf_ptx::deps::ModuleInfo::analyse(&text).weak_leaf_definition(name, shared)? != definition {
                return None;
            }
        }
        module.extract(name)
    }

    /// What to pass to the driver in place of the program's pointer.
    pub fn as_ptr(&self) -> *const c_void {
        match &self.wrapper {
            Some(wrapper) => &**wrapper as *const Wrapper as *const c_void,
            None => self.data.as_ptr() as *const c_void,
        }
    }

    /// Size of the image in bytes (of the fat binary or the text, not of a wrapper).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Create a replacement from an owned byte vector (for sub-modules loaded lazily).
    pub fn from_vec(data: Vec<u8>) -> Replacement {
        Replacement::new(data, None)
    }

    /// What was loaded in the end in place of this replacement: `image`, in which `refused` kernels
    /// more cannot be given to the program.
    pub fn loaded_as(self, image: Vec<u8>, refused: Vec<Refused>) -> Replacement {
        let mut loaded = Replacement::new(image, None);
        loaded.refused = self.refused.into_iter().chain(refused).collect();
        loaded.emulated = self.emulated;
        loaded.origin = self.origin;
        loaded.original_image = self.original_image;
        loaded
    }

    /// The raw bytes of the image (fat binary or PTX text).
    pub fn data_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Replacements in use, by the module or library handle the driver returned for them. The driver may
/// read an image again after the load returns (lazy loading), so they live until the unload.
static KEPT: LazyLock<Mutex<HashMap<usize, Arc<Replacement>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Keep the complete device-linked wrapper alive while compiling outside the lazy-library lock.
pub fn retained(handle: usize) -> Option<Arc<Replacement>> {
    KEPT.lock().unwrap_or_else(|e| e.into_inner()).get(&handle).cloned()
}

/// The driver loaded `replacement` as `handle`: keep it, and what is known of its kernels.
pub fn keep(handle: usize, mut replacement: Replacement) {
    // Context destruction can retire a CUmodule without an explicit unload.
    // The driver's next load starts a new lifetime at the same address.
    kernels::forget(handle);
    if config().profile
        && let Some(origin) = replacement.origin
    {
        kernels::set_origin(handle, origin);
    }
    kernels::describe(
        handle,
        std::mem::take(&mut replacement.refused),
        std::mem::take(&mut replacement.emulated),
    );
    // For a second translation at launch (`fit`): the original of a translation that has a kernel
    // over its budget. Otherwise nothing: and nothing may be left there under a handle that the
    // driver gives again.
    match replacement.original_image.take() {
        Some(original) if !original.is_empty() => crate::fit::register_module(handle, original),
        _ => crate::fit::forget(handle),
    }
    KEPT.lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(handle, Arc::new(replacement));
}

pub fn release(handle: usize) {
    KEPT.lock().unwrap_or_else(|e| e.into_inner()).remove(&handle);
    kernels::forget(handle);
}

/// What the adaptation found in the inputs of a link.
#[derive(Default, Clone)]
struct Found {
    refused: Vec<Refused>,
    emulated: Vec<Emulated>,
}

/// The driver's linker gives back one image of machine code for this GPU, which the program then
/// loads as it got it or as a copy: by its bytes alone, its kernels that trap cannot be told from
/// the others. So what was found in the inputs of a link (`cuLinkAddData`) is kept by link state,
/// and passed on to the image the link makes (`cuLinkComplete`), kept by the length and digest of
/// its bytes. (Only links with something to say are kept: the images grow with the distinct links
/// of a program that refuses kernels.)
#[derive(Default)]
struct Links {
    inputs: HashMap<usize, Found>,
    outputs: HashMap<(usize, u64), Found>,
}

static LINKS: LazyLock<Mutex<Links>> = LazyLock::new(|| Mutex::new(Links::default()));

fn links() -> std::sync::MutexGuard<'static, Links> {
    LINKS.lock().unwrap_or_else(|e| e.into_inner())
}

fn digest(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut hasher = twox_hash::XxHash64::with_seed(0x6c696e6b);
    hasher.write(bytes);
    hasher.finish()
}

/// The extent of the cubin at `bytes`, by its tables, within `len` when it is known.
///
/// # Safety
/// `bytes` must point to a cubin the driver could read, or to `len` readable bytes.
unsafe fn cubin_extent(bytes: *const u8, len: Option<usize>) -> Option<usize> {
    let header = unsafe { std::slice::from_raw_parts(bytes, 0x40) };
    fatbin::cubin_tables_len(header)
        .filter(|&tables| len.is_none_or(|len| tables <= len))
        .and_then(|tables| fatbin::cubin_len(unsafe { std::slice::from_raw_parts(bytes, tables) }))
        .filter(|&length| len.is_none_or(|len| length <= len))
}

/// `replacement` was added to the link `state`.
pub fn link_input(state: usize, replacement: &Replacement) {
    if replacement.refused.is_empty() && replacement.emulated.is_empty() {
        return;
    }
    let mut links = links();
    let found = links.inputs.entry(state).or_default();
    found.refused.extend(replacement.refused.iter().cloned());
    found.emulated.extend(replacement.emulated.iter().cloned());
}

/// The link `state` made the `size` bytes at `image`.
///
/// # Safety
/// `image` must be null or point to `size` readable bytes.
pub unsafe fn link_output(state: usize, image: *const c_void, size: usize) {
    let mut links = links();
    let Some(found) = links.inputs.get(&state).cloned() else {
        return;
    };
    if image.is_null() || size < 0x40 || !fatbin::is_elf(unsafe { std::slice::from_raw_parts(image.cast(), 4) }) {
        return;
    }
    // Keyed as `linked` finds it again when the image is loaded.
    let Some(length) = (unsafe { cubin_extent(image.cast(), Some(size)) }) else {
        return;
    };
    let bytes = unsafe { std::slice::from_raw_parts(image.cast::<u8>(), length) };
    links.outputs.insert((length, digest(bytes)), found);
}

/// The link `state` is destroyed: the driver may give its handle to another.
pub fn link_destroyed(state: usize) {
    links().inputs.remove(&state);
}

/// The cubin at `bytes`, when a link made it from inputs in which kernels trap or run emulated
/// instructions: the same bytes, to load with what is known of those kernels.
///
/// # Safety
/// As for `adapt`, with `bytes` at a cubin.
unsafe fn linked(who: &str, bytes: *const u8, len: Option<usize>) -> Option<Replacement> {
    let links = links();
    if links.outputs.is_empty() {
        return None;
    }
    let length = unsafe { cubin_extent(bytes, len) }?;
    let cubin = unsafe { std::slice::from_raw_parts(bytes, length) };
    let found = links.outputs.get(&(length, digest(cubin)))?.clone();
    drop(links);
    if let Some(first) = found.refused.first() {
        log::info(format_args!(
            "{who}: cubin made by the driver's linker, in which {} kernels cannot run on this GPU (the first is {}: \
             {}); they are refused if the program asks for them",
            found.refused.len(),
            first.kernel,
            first.reason
        ));
    }
    let mut replacement = Replacement::new(cubin.to_vec(), None);
    replacement.refused = found.refused;
    replacement.emulated = found.emulated;
    Some(replacement)
}

fn report(who: &str, what: &str, outcome: &Outcome) {
    match outcome {
        Outcome::Native => log::trace(format_args!("{who}: {what} with code for this GPU")),
        Outcome::Unknown => log::trace(format_args!("{who}: {what} not recognised, loaded as is")),
        Outcome::Retargeted {
            entries,
            from_sm,
            tensor_rewritten,
            tanh_expanded,
            trapped,
            refused,
            emulated: _,
        } => {
            let device = config().policy.device_sm;
            let mut changes = format!("{entries} {}", if *entries == 1 { "module" } else { "modules" });
            if *tensor_rewritten > 0 {
                changes += &format!(", {tensor_rewritten} tensor-core instructions rewritten");
            }
            if *tanh_expanded > 0 {
                changes += &format!(", {tanh_expanded} tanh expanded");
            }
            log::info(format_args!(
                "{who}: {what} without sm_{device} code: PTX for sm_{from_sm} retargeted ({changes})"
            ));
            if let Some(first) = trapped.first() {
                log::info(format_args!(
                    "{who}: {} functions use instructions without an sm_{device} sequence and were made to trap \
                     (the first is {}: {}); the {} kernels that need them are refused if the program asks for them",
                    trapped.len(),
                    first.function,
                    first.reason,
                    refused.len()
                ));
            }
        }
        Outcome::Translated {
            entries,
            from_sm,
            functions,
            refused,
            ..
        } => {
            let device = config().policy.device_sm;
            let modules = if *entries == 1 {
                String::new()
            } else {
                format!(" in {entries} modules")
            };
            log::info(format_args!(
                "{who}: {what} without sm_{device} code: machine code for sm_{from_sm} translated \
                 ({functions} functions{modules})"
            ));
            if let Some(first) = refused.first() {
                log::info(format_args!(
                    "{who}: {} of these functions {} not translated (the first is {}: {}); their kernels are \
                     refused if the program asks for them",
                    refused.len(),
                    if refused.len() == 1 { "is" } else { "are" },
                    first.function,
                    first.reason
                ));
            }
        }
        Outcome::NotTranslated { from_sm, reason } => log::warn(format_args!(
            "{who}: {what} with machine code for sm_{from_sm} that could not be read ({reason}); loaded as is"
        )),
        Outcome::NeedsTranslation { lowest_sm } => log::warn(format_args!(
            "{who}: {what} with machine code for sm_{lowest_sm} and later only, which nothing here \
             translates; loaded as is"
        )),
    }
}

thread_local! {
    /// Profiler only: the last `adapt` of this thread found code for this GPU in the image and
    /// left it as it is. Read by `loaded_as_is`, which labels the module the driver then loads.
    static IMAGE_WAS_NATIVE: Cell<bool> = const { Cell::new(false) };
}

/// Reset metadata after the program's own image was loaded. Also label native
/// images when profiling is enabled.
///
/// # Safety
/// `handle` must be null or point to the handle the driver stored.
pub unsafe fn loaded_as_is(result: c_int, handle: *mut *mut c_void) {
    if result == 0 && !handle.is_null() {
        // (Nothing of a module that is gone may be left under a handle that the driver gives again.)
        let fresh = unsafe { *handle } as usize;
        release(fresh);
        crate::fit::forget(fresh);
    }
    if !config().profile {
        return;
    }
    if IMAGE_WAS_NATIVE.replace(false) && result == 0 && !handle.is_null() {
        kernels::set_origin(unsafe { *handle } as usize, kernels::Origin::Native);
    }
}

/// `original`: the image as the program gave it, when it is machine code that may have been
/// translated (kept, for `fit`, when the translation has a kernel over its budget).
fn finish(
    who: &str,
    what: &str,
    adapted: Adapted,
    wrapper: Option<Wrapper>,
    terminate: bool,
    original: Option<&[u8]>,
) -> Option<Replacement> {
    report(who, what, &adapted.outcome);
    if config().profile && matches!(adapted.outcome, Outcome::Native) {
        IMAGE_WAS_NATIVE.set(true);
    }
    let over_budget = matches!(adapted.outcome, Outcome::Translated { over_budget: true, .. });
    let origin = match &adapted.outcome {
        Outcome::Retargeted { .. } => Some(kernels::Origin::PtxRewritten),
        Outcome::Translated { .. } => Some(kernels::Origin::MachineCodeTranslated),
        _ => None,
    };
    let mut image = adapted.image?;
    if terminate {
        image.push(0);
    }
    let mut replacement = Replacement::new(image, wrapper);
    replacement.origin = origin;
    if over_budget
        && crate::fit::enabled()
        && let Some(original) = original
    {
        // Kept only when a second translation within the budget would lower a kernel's count (the
        // translation says so, and the cache keeps it said), and then the one cubin translated,
        // as it is stored, not the fat binary around it.
        replacement.original_image = config().policy.kept_for_second(original);
    }
    if let Outcome::Retargeted { refused, emulated, .. } | Outcome::Translated { refused, emulated, .. } =
        adapted.outcome
    {
        replacement.refused = refused;
        replacement.emulated = emulated;
    }
    Some(replacement)
}

/// What `adapt` makes of `image`, a whole fat binary or cubin. The translation of machine code
/// takes a tenth of a second a megabyte: it is kept (`cache`), and taken from there when the same
/// image was translated by an earlier run of the same build with the same policy.
fn adapted(who: &str, image: &[u8], adapt: impl FnOnce() -> Adapted) -> Adapted {
    let key = crate::cache::enabled().then(|| crate::cache::key_for(image));
    let start = std::time::Instant::now();
    if let Some(kept) = key.as_ref().and_then(crate::cache::load_translation) {
        log::timing(format_args!(
            "timing: {who}: translation from the cache {:.1} ms",
            start.elapsed().as_secs_f64() * 1000.0
        ));
        return kept;
    }
    let adapted = adapt();
    if let Outcome::Translated { .. } = adapted.outcome {
        log::timing(format_args!(
            "timing: {who}: machine code translated in {:.1} ms",
            start.elapsed().as_secs_f64() * 1000.0
        ));
        if let Some(key) = &key {
            crate::cache::save_translation(key, &adapted);
        }
    }
    adapted
}

fn adapt_container(who: &str, bytes: &[u8], fat: &fatbin::FatBinary<'_>) -> Result<Adapted, fatbin::Error> {
    let mut failed = None;
    let result = adapted(who, bytes, || {
        config().policy.adapt_fatbin(fat).unwrap_or_else(|e| {
            failed = Some(e);
            Adapted {
                outcome: Outcome::Unknown,
                image: None,
            }
        })
    });
    failed.map_or(Ok(result), Err)
}

/// `at`: where the fat binary is, for the bytes of the whole of it.
fn adapt_fatbin(
    who: &str,
    at: *const u8,
    fat: Result<fatbin::FatBinary<'_>, fatbin::Error>,
    wrapper: Option<Wrapper>,
) -> Option<Replacement> {
    let result = fat.and_then(|fat| {
        // SAFETY: `fat` was read from `at`, and is `fat.len` bytes there.
        let original = unsafe { std::slice::from_raw_parts(at, fat.len) };
        Ok((original, adapt_container(who, original, &fat)?))
    });
    match result {
        Ok((original, adapted)) => finish(who, "fat binary", adapted, wrapper, false, Some(original)),
        Err(e) => {
            log::warn(format_args!("{who}: {e}; loaded as is"));
            None
        }
    }
}

/// Adapt a device-linked image. CUDA first tries the fully linked binary. If it has no code
/// for the device, the driver can link the relocatable fat binaries in the fourth word of the
/// wrapper. A compilation unit can refer to a function or variable defined by another unit.
///
/// # Safety
/// The wrapper and the null-terminated array in `extra` are supplied by the CUDA runtime.
/// Each non-null array member is the address of a complete fat binary, not another wrapper.
unsafe fn adapt_linked(who: &str, wrapper: Wrapper) -> Option<Replacement> {
    let policy = &config().policy;
    let primary = unsafe { fatbin::parse_at(wrapper.data) }.ok()?;
    let linked = unsafe { std::slice::from_raw_parts(wrapper.data, primary.len) };
    let adapted = policy.adapt_fatbin(&primary).ok()?;
    if adapted.image.is_some() || matches!(adapted.outcome, Outcome::Native) {
        // This is the complete, already linked program.
        return finish(who, "linked fat binary", adapted, None, false, Some(linked));
    }
    if wrapper.extra.is_null() {
        return None;
    }

    let mut objects = Vec::new();
    let mut refused = Vec::new();
    let mut emulated = Vec::new();
    let mut origin = None;
    let mut object_traps = Vec::new();
    let mut check_external_calls = false;
    let mut next = wrapper.extra.cast::<*const u8>();
    loop {
        let ptr = unsafe { next.read() };
        if ptr.is_null() {
            break;
        }
        next = unsafe { next.add(1) };
        let fat = unsafe { fatbin::parse_at(ptr) }.ok()?;
        let source = unsafe { std::slice::from_raw_parts(ptr, fat.len) };
        let lowering = adapt_container(who, source, &fat).ok()?;
        let traps = match &lowering.outcome {
            Outcome::Retargeted { trapped, refused, .. } => {
                // An .entry cannot be called as a .func; entry-only refusals already have their
                // complete names. A trapped helper can also be reached from another object.
                check_external_calls |= trapped.iter().any(|t| !refused.iter().any(|r| r.kernel == t.function));
                trapped.clone()
            }
            Outcome::Translated { refused, .. } if !refused.is_empty() => {
                // A refused machine-code function might be called from a different object.
                // Its relocations do not give the PTX graph below the edges needed for this.
                log::warn(format_args!(
                    "{who}: a relocatable cubin has refused functions; loaded as is"
                ));
                return None;
            }
            _ => Vec::new(),
        };
        object_traps.push(traps);
        match finish(who, "relocatable fat binary", lowering, None, false, Some(source)) {
            Some(replaced) => {
                origin = origin.or(replaced.origin);
                refused.extend(replaced.refused);
                emulated.extend(replaced.emulated);
                objects.push(replaced.data);
            }
            None => objects.push(source.into()),
        }
    }
    let origin = origin?;
    if check_external_calls {
        let mut graph = vf_ptx::deps::LinkedDependencies::default();
        for (object, traps) in objects.iter().zip(&object_traps) {
            let fat = fatbin::parse(object).ok()?;
            for entry in fat.entries.iter().filter(|entry| policy.usable(entry)) {
                if entry.kind() == fatbin::KIND_ELF {
                    // A machine-code caller could reach a trapped PTX helper without a visible edge.
                    log::warn(format_args!(
                        "{who}: relocatable machine code may call a trapped PTX function; loaded as is"
                    ));
                    return None;
                }
                if !graph.add(&entry.payload().ok()?, traps) {
                    log::warn(format_args!(
                        "{who}: incomplete dependency information for relocatable PTX; loaded as is"
                    ));
                    return None;
                }
            }
        }
        refused.extend(graph.refused());
        refused.sort_by(|a, b| a.kernel.cmp(&b.kernel));
        refused.dedup_by(|a, b| a.kernel == b.kernel);
    }
    let pointers = objects
        .iter()
        .map(|obj| obj.as_ptr())
        .chain(std::iter::once(std::ptr::null()))
        .collect::<Box<[_]>>();
    let wrapper = Wrapper {
        extra: pointers.as_ptr().cast(),
        ..wrapper
    };
    let mut result = Replacement::new(linked.to_vec(), Some(wrapper));
    result.prelinked = Some(PrelinkedObjects {
        shared: OnceLock::new(),
        objects,
        _pointers: pointers,
    });
    result.refused = refused;
    result.emulated = emulated;
    result.origin = Some(origin);
    Some(result)
}

/// Copy the original image bytes at `ptr` exactly as the adaptation would consume them: the fat
/// binary a wrapper points to, the fat binary at `ptr`, or the PTX text (until the terminating zero).
/// Returns `None` for a null pointer, for machine code, for a format the adaptation would not touch,
/// or when adaptation is off.
///
/// # Safety
/// `ptr` must satisfy the same contract as `adapt`.
pub unsafe fn image_bytes(ptr: *const c_void, len: Option<usize>) -> Option<Vec<u8>> {
    if !config().adapt || ptr.is_null() {
        return None;
    }
    catch_unwind(AssertUnwindSafe(|| unsafe { image_bytes_inner(ptr, len) }))
        .ok()
        .flatten()
}

/// # Safety
/// As for `image_bytes`.
unsafe fn image_bytes_inner(ptr: *const c_void, len: Option<usize>) -> Option<Vec<u8>> {
    let bytes = ptr as *const u8;
    let mut head = [0u8; 4];
    for (i, byte) in head.iter_mut().enumerate() {
        if len.is_some_and(|len| i >= len) {
            break;
        }
        *byte = unsafe { *bytes.add(i) };
        if *byte == 0 && len.is_none() {
            break;
        }
    }
    let magic = u32::from_le_bytes(head);
    let binary = magic == fatbin::WRAPPER_MAGIC || magic == fatbin::MAGIC || head == *b"\x7fELF";
    if !binary && (head[0].is_ascii_graphic() || head[0].is_ascii_whitespace()) {
        let text = match len {
            Some(len) => unsafe { std::slice::from_raw_parts(bytes, len) },
            None => unsafe { CStr::from_ptr(ptr.cast()) }.to_bytes(),
        };
        return Some(text.to_vec());
    }
    if magic == fatbin::WRAPPER_MAGIC {
        let wrapper = unsafe { ptr.cast::<Wrapper>().read_unaligned() };
        if wrapper.version != 1 || wrapper.data.is_null() {
            return None;
        }
        let container_len =
            fatbin::container_len(unsafe { std::slice::from_raw_parts(wrapper.data, fatbin::HEADER_SIZE) }).ok()?;
        return Some(unsafe { std::slice::from_raw_parts(wrapper.data, container_len) }.to_vec());
    }
    if magic == fatbin::MAGIC {
        let container_len =
            fatbin::container_len(unsafe { std::slice::from_raw_parts(bytes, fatbin::HEADER_SIZE) }).ok()?;
        return Some(unsafe { std::slice::from_raw_parts(bytes, container_len) }.to_vec());
    }
    // Machine code (an ELF file): how far it reaches is not known here without reading its section
    // table, and its header alone does not tell it from another. It is not cached.
    None
}

/// # Safety
/// As for `adapt`.
unsafe fn adapt_unguarded(who: &str, ptr: *const c_void, len: Option<usize>) -> Option<Replacement> {
    let bytes = ptr as *const u8;
    let policy = &config().policy;
    // The first four bytes tell the binary formats apart. Text may be shorter than that: stop at its
    // terminating zero. (A fat binary starts with 0x50, the letter P, so text is what remains.)
    let mut head = [0u8; 4];
    for (i, byte) in head.iter_mut().enumerate() {
        if len.is_some_and(|len| i >= len) {
            break;
        }
        *byte = unsafe { *bytes.add(i) };
        if *byte == 0 && len.is_none() {
            break;
        }
    }
    let magic = u32::from_le_bytes(head);
    let binary = magic == fatbin::WRAPPER_MAGIC || magic == fatbin::MAGIC || head == *b"\x7fELF";
    if !binary && (head[0].is_ascii_graphic() || head[0].is_ascii_whitespace()) {
        let text = match len {
            Some(len) => unsafe { std::slice::from_raw_parts(bytes, len) },
            None => unsafe { CStr::from_ptr(ptr.cast()) }.to_bytes(),
        };
        // Text given with a length is not zero-terminated; neither is its replacement.
        return finish(who, "PTX text", policy.adapt_ptx(text), None, len.is_none(), None);
    }
    if magic == fatbin::WRAPPER_MAGIC {
        let wrapper = unsafe { ptr.cast::<Wrapper>().read_unaligned() };
        if wrapper.version == 2 && !wrapper.data.is_null() {
            return unsafe { adapt_linked(who, wrapper) };
        }
        if wrapper.version != 1 || wrapper.data.is_null() {
            log::warn(format_args!(
                "{who}: fat binary wrapper version {} loaded as is",
                wrapper.version
            ));
            return None;
        }
        return adapt_fatbin(
            who,
            wrapper.data,
            unsafe { fatbin::parse_at(wrapper.data) },
            Some(wrapper),
        );
    }
    if magic == fatbin::MAGIC {
        return adapt_fatbin(who, bytes, unsafe { fatbin::parse_at(bytes) }, None);
    }
    if head == *b"\x7fELF" {
        if len.is_some_and(|len| len < 0x40) {
            return None;
        }
        let header = unsafe { std::slice::from_raw_parts(bytes, 0x40) };
        if !policy.translates_cubin(header) {
            if let Some(linked) = unsafe { linked(who, bytes, len) } {
                return Some(linked);
            }
            return finish(who, "cubin", policy.adapt_cubin(header), None, false, None);
        }
        // To be translated, all of it: how far it reaches is in its tables, which the driver would
        // read as well.
        let whole = fatbin::cubin_tables_len(header)
            .filter(|&tables| len.is_none_or(|len| tables <= len))
            .and_then(|tables| fatbin::cubin_len(unsafe { std::slice::from_raw_parts(bytes, tables) }))
            .filter(|&length| len.is_none_or(|len| length <= len));
        let Some(length) = whole else {
            log::warn(format_args!(
                "{who}: cubin with machine code to translate, whose length its tables do not tell; loaded as is"
            ));
            return None;
        };
        let cubin = unsafe { std::slice::from_raw_parts(bytes, length) };
        let replacement = adapted(who, cubin, || policy.adapt_cubin(cubin));
        let text = matches!(replacement.outcome, Outcome::Retargeted { .. });
        return finish(who, "cubin", replacement, None, text, Some(cubin));
    }
    log::trace(format_args!(
        "{who}: image with unknown magic {magic:#010x} loaded as is"
    ));
    None
}

/// The image to load in place of the one at `ptr`, if it has to be adapted. `len` is its length when
/// the caller states one (`cuLinkAddData`).
///
/// # Safety
/// `ptr` must be null or point to an image the driver itself could read: a complete fat-binary wrapper,
/// fat binary or cubin, or text ended by a zero byte (or `len` bytes long).
pub unsafe fn adapt(who: &str, ptr: *const c_void, len: Option<usize>) -> Option<Replacement> {
    if config().profile {
        IMAGE_WAS_NATIVE.set(false);
    }
    if !config().adapt || ptr.is_null() {
        return None;
    }
    // Whatever goes wrong here, the program must get what it would get without this library.
    catch_unwind(AssertUnwindSafe(|| unsafe { adapt_unguarded(who, ptr, len) })).unwrap_or_else(|_| {
        log::error(format_args!(
            "{who}: internal error while adapting an image; loaded as is"
        ));
        None
    })
}

/// The image to load in place of the contents of a file, if it has to be adapted.
pub fn adapt_file(who: &str, path: &CStr) -> Option<Replacement> {
    if config().profile {
        IMAGE_WAS_NATIVE.set(false);
    }
    if !config().adapt {
        return None;
    }
    let mut contents = std::fs::read(path.to_str().ok()?).ok()?;
    if contents.len() < 4 {
        return None;
    }
    if contents.starts_with(&fatbin::MAGIC.to_le_bytes()) {
        // A file cut short must not be read past its end.
        fatbin::parse(&contents).ok()?;
    } else if fatbin::is_elf(&contents) {
        // A file cut short must not be read past its end either.
        if contents.len() < 0x40 || fatbin::cubin_len(&contents).is_none_or(|length| length > contents.len()) {
            return None;
        }
    } else if contents.starts_with(&fatbin::WRAPPER_MAGIC.to_le_bytes()) {
        return None; // a wrapper holds pointers: it has no meaning in a file
    } else {
        contents.push(0); // text
    }
    // SAFETY: `contents` is a complete fat binary, a complete cubin, or zero-terminated bytes.
    unsafe { adapt(who, contents.as_ptr().cast(), None) }
}
