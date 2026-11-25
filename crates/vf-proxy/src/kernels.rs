//! What this library knows about the kernels of the images it adapted, by the handle of their module
//! or library.
//!
//! - Kernels that must not be given to the program: they are, or call, a function that had to be made
//!   to trap (an instruction without a sequence for this GPU, or something the driver's compiler
//!   refused). Launched, such a kernel ends the CUDA context of the program; refused at the lookup, it
//!   is an error the program can report. A program can have a handle to one without a lookup (a
//!   private interface of the driver, an enumeration made before a refusal was known): what the
//!   driver says of a handle leads back to these records (`verdict`), for `guard`.
//! - Kernels that run emulated instructions: announced on the console when the program takes them.
//! - Kernels given with reported traps (`vf_ptx::Reported`): they are, or may reach, a function in
//!   which instructions that cannot run on this GPU, and that a correct program cannot reach on it,
//!   were replaced by a trap that names them (`vf_ptx::lower::Options::reported_traps`). Announced
//!   once each when the program takes them, counted at exit, and named in the report of a fault
//!   (`traps_of`).
//! - Kernels without code: their code in the image is a trap and nothing else (`vf_adapt::Stubs`).
//!   They are given at a lookup, as NVIDIA's driver gives them (programs look them up without
//!   launching them), and a use that would run one is refused (`guard`): launched, it would end
//!   the context, as it does on any GPU.
//!
//! Each is said once, within `VF_EMULATION_WARNINGS`, the second kernel of a name with what names its
//! image when that is known (`Kernels::shown`). A kernel with emulated instructions is its name and
//! the code it runs (`vf_adapt::emulation::Said`, which asks for the digest of the image given to the
//! driver, `images::digest_of`, only when a kernel of a name said before comes from another load):
//! two specializations of one kernel are each said. A refused kernel is its name and why it is
//! refused: its code is made to trap, so that two refused specializations differ in nothing that the
//! layer keeps but in why, if they do; one refused for another reason is said.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{LazyLock, Mutex};

use vf_adapt::emulation::{Announcer, KEPT_KERNELS, Load};
use vf_adapt::{Emulated, EmulatedCounts, Refused, Reported};

use crate::{REALS, config, idx, images, log, profile};

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
    /// By kernel: what it is given with of reported traps.
    reported: HashMap<String, Reported>,
    /// For the module of a library: that library, with which it goes away.
    library: Option<usize>,
    /// The handle is a library's (`cuLibraryLoadData`), not a module's: the driver makes a module
    /// of it in each context where it is used, whose functions the program may hold.
    is_library: bool,
    /// How the code of this module reached the device.
    origin: Option<Origin>,
    /// What names the image in the lines of its kernels, when it is known (`Replacement::shown`).
    shown: Option<String>,
    /// Kernels without code (their code is a trap and nothing else), and the image they are in, by
    /// its size and digest, for the line that says why one is not launched.
    trap_only: HashSet<String>,
    image: Option<(usize, u64)>,
    /// The architecture of the program's PTX that the layer retargeted for this GPU, when it was a
    /// later one (`retargeted`): what the program's code was compiled for, where the driver, which
    /// compiled the layer's PTX, says this GPU's (`retargeted_from`).
    retargeted_from: Option<u32>,
    /// The image's machine code that a GPU of the capability told loads, which the layer did not
    /// give (`elsewhere`), and the names of the kernels that were looked for there and are not in it
    /// (`not_given`; at most `KEPT_KERNELS`).
    elsewhere: Option<Elsewhere>,
    absent: HashSet<String>,
}

/// The machine code of an image that a GPU of the capability told loads, where the layer gave the
/// program the translation of the image's code for an earlier generation
/// (`vf_adapt::Policy::told_machine_code`): a program told that capability asks for its kernels by
/// the names they have in that code, and the code given may have no kernel of such a name. Said
/// when such a lookup fails (`not_given`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Elsewhere {
    /// The architecture of the machine code that the layer translated and gave.
    pub given_sm: u32,
    /// The capability that the program was told, as an SM number: the image was adapted for it.
    pub told_sm: u32,
    /// The architecture of the image's machine code that a GPU of that capability loads.
    pub loads_sm: u32,
    /// Why the layer gave the code for `given_sm` (`vf_adapt::Policy::why_given`).
    pub why: String,
    /// What names the program's image: the file it is in, its size and its digest.
    pub image: String,
}

static KNOWN: LazyLock<Mutex<HashMap<usize, Kernels>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
/// The number of modules and libraries known here with a refused kernel or a kernel without code:
/// while there is none, no handle that the program holds can be one of those, and `guard` asks
/// nothing.
static REFUSING: AtomicUsize = AtomicUsize::new(0);
/// Changed whenever what is known here changes, and whenever a module, a library or a context
/// goes (after which the driver may give its handles to others): what `guard` concluded of a
/// handle holds for one generation.
static GENERATION: AtomicU64 = AtomicU64::new(0);
static ANNOUNCER: Announcer = Announcer::new();
/// Refused kernels the program already asked for: the refusal is explained once.
static EXPLAINED: LazyLock<Mutex<Explained>> = LazyLock::new(|| Mutex::new(Explained::default()));
/// Whether the remedy for NCCL's kernels was given (`nccl_remedy`).
static NCCL_REMEDY_GIVEN: AtomicBool = AtomicBool::new(false);

/// Whether `name` is the name of one of the kernels of NCCL's collectives (`ncclDevKernel_...`;
/// `ncclKernel_...` before NCCL 2.19), as the compiler gives it: plain, or mangled as a function at
/// global scope (`_Z`, the length of the name, the name, its parameters). Not its kernels for
/// symmetric memory (`ncclSymkDevKernel_...`), of which NCCL goes on without those it cannot take
/// (it ignores a failure to look one up), and chooses them only among those its GPUs can run: the
/// layer refuses those that copy with `cp.async.bulk`, which NCCL 2.30 chooses on sm_100 and later
/// only, with `NCCL_SYM_TMA_ENABLE=1`.
pub(crate) fn nccl_kernel(name: &str) -> bool {
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
    ["ncclDevKernel_", "ncclKernel_"]
        .iter()
        .any(|prefix| identifier.starts_with(prefix))
}

/// For the first refused kernel of NCCL's collectives: what the program needs instead. Their
/// multicast instructions are given as reported traps where no GPU of the process can take part in
/// multicast (`images::policy`); elsewhere, with `VF_REPORTED_TRAPS=0`, or for another reason, the
/// CUDA 13 build of NCCL cannot run its collectives through the layer. Its CUDA 12 build has code
/// for Volta, and the launcher gives it to the program when it is installed.
fn nccl_remedy(name: &str) -> Option<String> {
    if !nccl_kernel(name) || NCCL_REMEDY_GIVEN.swap(true, Relaxed) {
        return None;
    }
    Some(format!(
        "{name} is a kernel of NCCL's collectives, which cannot run without it: a program that uses several \
         GPUs then needs NCCL's CUDA 12 build, which has code for this GPU. `vf fetch --only libnccl` downloads \
         it, and a program started through `vf` is given it in place of its own (`vf doctor` says whether it \
         is found; LIMITS.md, \"Several GPUs and NCCL\")"
    ))
}

/// The refused kernels that were explained, by name, function that traps and reason.
#[derive(Default)]
struct Explained {
    said: HashSet<(String, String, String)>,
    names: HashSet<String>,
    /// How many refused kernels had a name explained before.
    again: usize,
    /// Refused kernels came beyond the `KEPT_KERNELS` that `said` keeps.
    beyond: bool,
}

/// What to say of a refused kernel that the program asks for.
#[derive(Debug, PartialEq, Eq)]
enum Explain {
    /// Nothing: it was explained, or too many of its kind were.
    Nothing,
    /// Its refusal: the first kernel of its name.
    First,
    /// Its refusal, as another kernel of a name explained before.
    Again,
    /// That further kernels of names explained before are not listed.
    Further,
    /// That further refused kernels are not said: more than `KEPT_KERNELS` were.
    Beyond,
}

impl Explained {
    /// Kernel `name` is refused, for its `function` that traps, for `reason`: what to say. The first
    /// kernel of each name is explained; another of the same name, refused otherwise, while no more
    /// than `limit` have been (all of them with `all`; none with a limit of zero).
    fn refused(&mut self, name: &str, function: &str, reason: &str, limit: usize, all: bool) -> Explain {
        let kernel = (name.to_string(), function.to_string(), reason.to_string());
        if self.said.contains(&kernel) {
            return Explain::Nothing;
        }
        if self.said.len() + self.names.len() >= KEPT_KERNELS {
            return match std::mem::replace(&mut self.beyond, true) {
                false => Explain::Beyond,
                true => Explain::Nothing,
            };
        }
        self.said.insert(kernel);
        if self.names.insert(name.to_string()) {
            return Explain::First;
        }
        self.again += 1;
        match limit {
            0 => Explain::Nothing,
            _ if self.again <= limit || all => Explain::Again,
            _ if self.again == limit + 1 => Explain::Further,
            _ => Explain::Nothing,
        }
    }
}

fn known() -> std::sync::MutexGuard<'static, HashMap<usize, Kernels>> {
    KNOWN.lock().unwrap_or_else(|e| e.into_inner())
}

/// What is known here changed (`known` is the map, under its lock): `REFUSING` is counted again, and
/// a new generation begins.
fn changed(known: &HashMap<usize, Kernels>) {
    REFUSING.store(
        known
            .values()
            .filter(|k| !k.refused.is_empty() || !k.trap_only.is_empty())
            .count(),
        SeqCst,
    );
    new_generation();
}

/// A new generation begins (`GENERATION`): a module, a library or a context went, or what is known
/// here changed. (After what it changed.)
pub fn new_generation() {
    GENERATION.fetch_add(1, SeqCst);
}

/// The number of driver calls under way that end a module, a library or a context (`Ending`).
static ENDING: AtomicUsize = AtomicUsize::new(0);

/// While it lives, a driver call that ends a module, a library or a context is under way: once the
/// driver has freed the handles of its functions, another thread may be given one of them for
/// another function before the call returns. A generation begins when it is made, before the
/// driver's call; what `guard` concludes meanwhile is not kept (`ending`); and a generation begins
/// again when it is dropped, after the call.
pub struct Ending(());

/// A driver call that ends a module, a library or a context begins (`Ending`).
pub fn ending() -> Ending {
    ENDING.fetch_add(1, SeqCst);
    new_generation();
    Ending(())
}

impl Drop for Ending {
    fn drop(&mut self) {
        // (The generation first: a conclusion reached while the call was under way, and kept by a
        // thread that saw no call under way any more, is of the generation that this ends.)
        new_generation();
        ENDING.fetch_sub(1, SeqCst);
    }
}

/// Whether a driver call that ends a module, a library or a context is under way (`Ending`).
pub fn ending_now() -> bool {
    ENDING.load(SeqCst) != 0
}

/// Whether a module or library known here has a refused kernel.
#[inline]
pub fn refusing() -> bool {
    REFUSING.load(SeqCst) != 0
}

/// The current generation (`GENERATION`).
#[inline]
pub fn generation() -> u64 {
    GENERATION.load(SeqCst)
}

/// Module aliases consult their library's current metadata, including compiler
/// refusals learned after a lazy submodule first exposed a function.
/// The load of the image whose kernels `handle` has: its library's for the module of a library.
fn load_of(known: &HashMap<usize, Kernels>, handle: usize) -> Load {
    let owner = known.get(&handle).and_then(|k| k.library).unwrap_or(handle);
    Load {
        handle: owner,
        number: images::load_of(owner),
    }
}

fn metadata(known: &HashMap<usize, Kernels>, handle: usize) -> Option<&Kernels> {
    let entry = known.get(&handle)?;
    match entry.library {
        Some(library) => known.get(&library),
        None => Some(entry),
    }
}

/// Record what the adaptation found about the kernels of the image loaded as `handle`, a library's
/// when `library`, which `shown` names when that is known (a handle described before keeps what
/// named it).
pub fn describe(handle: usize, shown: Option<String>, refused: Vec<Refused>, emulated: Vec<Emulated>, library: bool) {
    if refused.is_empty() && emulated.is_empty() {
        return;
    }
    let mut known = known();
    let kernels = known.entry(handle).or_default();
    kernels.is_library |= library;
    if kernels.shown.is_none() {
        kernels.shown = shown;
    }
    for kernel in refused {
        kernels
            .refused
            .entry(kernel.kernel)
            .or_insert((kernel.function, kernel.reason));
    }
    for kernel in emulated {
        kernels.emulated.insert(kernel.kernel, kernel.counts);
    }
    changed(&known);
}

/// Record the kernels of the image loaded as `handle` (a library's when `library`), which `image`
/// names, that are given with reported traps, and say so in one line at level 1 (`image_traps_line`;
/// not with `VF_EMULATION_WARNINGS=0`). Each kernel is said by itself from level 2, when the program
/// takes it (`given`). (Nothing that `guard` concludes changes: no new generation.)
pub fn with_traps(handle: usize, image: Option<&str>, reported: Vec<Reported>, library: bool) {
    if reported.is_empty() {
        return;
    }
    if config().emulation_warnings > 0 {
        log::warn(format_args!("{}", image_traps_line(image, &reported)));
    }
    let mut known = known();
    let kernels = known.entry(handle).or_default();
    kernels.is_library |= library;
    for kernel in reported {
        kernels.reported.insert(kernel.kernel.clone(), kernel);
    }
}

/// The line that says, once for an image, which of its kernels are given with reported traps: how
/// many, the most instructions that one may reach, the first form of each (three at the most), the
/// first three names. (Until 2026-11-25 each kernel had a line of its own at level 1: about 40 lines of
/// up to 547 characters for each process of a program with the CUDA 13 NCCL.)
fn image_traps_line(image: Option<&str>, reported: &[Reported]) -> String {
    let most = reported.iter().map(|kernel| kernel.instructions).max().unwrap_or(0);
    let mut forms: Vec<&str> = reported.iter().map(|kernel| kernel.form.as_str()).collect();
    forms.sort_unstable();
    forms.dedup();
    let more_forms = if forms.len() > 3 { ", ..." } else { "" };
    let names: Vec<&str> = reported.iter().take(3).map(|kernel| kernel.kernel.as_str()).collect();
    let more_names = match reported.len() {
        0..=3 => String::new(),
        n => format!(" and {} more", n - 3),
    };
    format!(
        "{}: {} {} given with instructions that cannot run on this GPU replaced by reported traps (at most {} in \
         what one kernel may reach; {}{more_forms}): {}{more_names}. Such an instruction needs a multicast \
         address, which no GPU of this process can have: reached all the same, it stops its kernel with an \
         assertion that names it (CUDA_ERROR_ASSERT, 710) and the context ends. VF_LOG_LEVEL=2 says each \
         kernel when the program takes it; VF_REPORTED_TRAPS=0 refuses them instead",
        image.unwrap_or("an image"),
        reported.len(),
        if reported.len() == 1 {
            "kernel is"
        } else {
            "kernels are"
        },
        most,
        forms[..forms.len().min(3)].join(", "),
        names.join(", ")
    )
}

/// What kernel `name` of `handle` is given with of reported traps, if it is (for the report of a
/// fault).
pub fn traps_of(handle: usize, name: &str) -> Option<Reported> {
    metadata(&known(), handle).and_then(|kernels| kernels.reported.get(name).cloned())
}

/// The kernels with reported traps that were said, and how many were given: the line at exit.
#[derive(Default)]
struct TrapsSaid {
    /// The names said (at most `KEPT_KERNELS`).
    said: HashSet<String>,
    /// Kernels given, and whether more than were kept came.
    given: usize,
    beyond: bool,
    /// The first form replaced in what each kernel reaches, for the line at exit (the
    /// instructions replaced are not added up: kernels share the functions they reach).
    forms: std::collections::BTreeSet<String>,
}

static TRAPS_SAID: LazyLock<Mutex<TrapsSaid>> = LazyLock::new(|| Mutex::new(TrapsSaid::default()));

/// Kernel `name` is given with `reported`: counted for the line at exit, the first time a kernel of
/// its name is given, and said then from level 2 (`said`; at level 1 its image's line says it,
/// `with_traps`).
fn traps_given(name: &str, reported: &Reported, said_here: bool) -> Option<String> {
    let mut said = TRAPS_SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.said.contains(name) {
        return None;
    }
    if said.said.len() >= KEPT_KERNELS {
        said.beyond = true;
        return None;
    }
    said.said.insert(name.to_string());
    said.given += 1;
    said.forms.insert(reported.form.clone());
    said_here.then(|| format!("kernel {name} is given with {}", reported.summary()))
}

/// For the report at exit: the kernels given with reported traps.
pub fn traps_summary_line() -> Option<String> {
    let said = TRAPS_SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.given == 0 {
        return None;
    }
    let forms: Vec<&str> = said.forms.iter().map(String::as_str).take(4).collect();
    Some(format!(
        "{} {} given to the program with reported traps{} (in place of {}{})",
        said.given,
        if said.given == 1 { "kernel was" } else { "kernels were" },
        if said.beyond {
            ", and more that were not counted"
        } else {
            ""
        },
        forms.join(", "),
        if said.forms.len() > forms.len() { ", ..." } else { "" }
    ))
}

/// Record the kernels without code of the image loaded as `handle` (a library's when `library`):
/// `names`, whose code is a trap and nothing else, in the image of `image` (its size and digest).
pub fn without_code(handle: usize, names: Vec<String>, image: (usize, u64), library: bool) {
    if names.is_empty() {
        return;
    }
    let mut known = known();
    let kernels = known.entry(handle).or_default();
    kernels.is_library |= library;
    kernels.image.get_or_insert(image);
    kernels.trap_only.extend(names);
    changed(&known);
}

/// The code of the image loaded as `handle` (a library's when `library`) was retargeted by the
/// layer from PTX for `sm`, a later architecture than this GPU's. (Nothing that `guard` concludes
/// changes: no new generation.)
pub fn retargeted(handle: usize, sm: u32, library: bool) {
    let mut known = known();
    let kernels = known.entry(handle).or_default();
    kernels.is_library |= library;
    kernels.retargeted_from = Some(sm);
}

/// The image loaded as `handle` (a library's when `library`) has machine code that a GPU of the
/// capability told loads, which the layer did not give (`Elsewhere`). (Nothing that `guard`
/// concludes changes: no new generation.)
pub fn elsewhere(handle: usize, elsewhere: Elsewhere, library: bool) {
    let mut known = known();
    let kernels = known.entry(handle).or_default();
    kernels.is_library |= library;
    kernels.elsewhere = Some(elsewhere);
    kernels.absent.clear();
}

/// Kernels named by the line of `not_given`: said once for each name.
static NOT_GIVEN_SAID: LazyLock<Mutex<Guarded>> = LazyLock::new(|| Mutex::new(Guarded::default()));

/// The line of `not_given`, for kernel `name` of an image of which `elsewhere` is known; `tried`:
/// why the kernel is not given from the code that has it either (`told`).
fn not_given_line(name: &str, elsewhere: &Elsewhere, tried: Option<&str>) -> String {
    let Elsewhere {
        given_sm,
        told_sm,
        loads_sm,
        why,
        image,
    } = elsewhere;
    let tried = match tried {
        Some(tried) => format!(". The kernel alone is not given from the code for sm_{loads_sm} either: {tried}"),
        None => String::new(),
    };
    format!(
        "kernel {name} is not in the code that the layer gives for its image ({image}), its machine code for \
         sm_{given_sm} translated: it is only in the image's machine code for sm_{loads_sm}, which a GPU of the \
         capability told ({}.{}) loads. The lookup fails as the driver answers for a name that it does not have \
         (error 500): {why} (LIMITS.md, \"The capability told with VF_CAPABILITY\"){tried}",
        told_sm / 10,
        told_sm % 10
    )
}

/// Kernel `name`, which a lookup did not find, is in the machine code of its image that a GPU of
/// the capability told loads (`told_code`), and is not given from there (`tried`: why, when it was
/// tried, `told`): a line says so, once for each kernel. The program was told a capability for
/// which its image has the kernel, and was given other code.
pub fn not_given(name: &str, elsewhere: &Elsewhere, tried: Option<&str>) {
    if let Some(line) = not_given_said(name, elsewhere, tried) {
        log::warn(format_args!("{line}"));
    }
}

/// `not_given`: the line to say, if there is one (once for each kernel; the line that further
/// ones are not said, beyond `KEPT_KERNELS`).
fn not_given_said(name: &str, elsewhere: &Elsewhere, tried: Option<&str>) -> Option<String> {
    match NOT_GIVEN_SAID.lock().unwrap_or_else(|e| e.into_inner()).note(name) {
        Note::Nothing => None,
        Note::Beyond => Some(format!(
            "further kernels that are only in code which the layer did not give are not said: more than \
             {KEPT_KERNELS} were (each lookup still fails, error 500)"
        )),
        Note::First => Some(not_given_line(name, elsewhere, tried)),
    }
}

/// What `told_code` found: the program's module or library that has the image (a library's module
/// leads to its library), whether it is a library, and what is known of the image's code.
pub struct Told {
    pub owner: usize,
    pub library: bool,
    pub elsewhere: Elsewhere,
}

/// The lookup of kernel `name` in the module or library `handle` failed: the driver has no kernel of
/// that name in the code the layer gave. `Some` when the image's machine code that a GPU of the
/// capability told loads has one (`Elsewhere`); `None` when nothing of the kind is known of the
/// image, or that code has no kernel of the name either: a GPU of that capability has none. (A
/// name not found there is remembered, and that code is not read again for it.)
pub fn told_code(handle: usize, name: &str) -> Option<Told> {
    let (owner, library, elsewhere) = {
        let known = known();
        let owner = known.get(&handle).and_then(|k| k.library).unwrap_or(handle);
        match known.get(&owner) {
            Some(Kernels {
                elsewhere: Some(elsewhere),
                absent,
                is_library,
                ..
            }) if !absent.contains(name) => (owner, *is_library, elsewhere.clone()),
            _ => return None,
        }
    };
    // (Outside the lock of what is known: the cubins of that code are read, decompressed.)
    let load = images::load_of(owner);
    let found = images::retained(owner).is_some_and(|image| {
        vf_fatbin::parse(image.data_bytes()).is_ok_and(|fat| vf_adapt::has_kernel(&fat, elsewhere.loads_sm, name))
    });
    if !found {
        let mut known = known();
        if images::load_of(owner) == load
            && let Some(kernels) = known.get_mut(&owner)
            && kernels.absent.len() < KEPT_KERNELS
        {
            kernels.absent.insert(name.to_string());
        }
        return None;
    }
    Some(Told {
        owner,
        library,
        elsewhere,
    })
}

/// Why kernel `name` of `handle` is refused, when it is; nothing is said.
pub fn refusal(handle: usize, name: &str) -> Option<String> {
    let known = known();
    let (function, reason) = metadata(&known, handle)?.refused.get(name)?;
    Some(if function == name {
        reason.clone()
    } else {
        format!("{reason}, in {function}, which it calls")
    })
}

/// The architecture of the program's PTX from which the layer made the code of `handle`, a function
/// or a kernel (`retargeted`), as the driver says what `handle` is (`owner_of`): a function of a
/// module, of the module of a library in a context on this GPU (`share`), or in any context for a
/// deferred library (`lazy::module_owner`), or a kernel of a library, deferred or not. `None` when
/// its code is not known here as retargeted.
///
/// # Safety
/// `handle` must be one that the driver accepts for a question about a function or a kernel, or
/// that it refuses.
pub unsafe fn retargeted_from(handle: usize) -> Option<u32> {
    let (owner, _) = unsafe { owner_of(handle) }?;
    let of = |owner: usize| metadata(&known(), owner).and_then(|kernels| kernels.retargeted_from);
    of(owner).or_else(|| crate::lazy::module_owner(owner).and_then(of))
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

/// `module` is the module of library `library`: the same kernels. (A runtime may say it at every
/// launch, through `cuKernelGetFunction`: when it is known already, nothing changes, and no new
/// generation begins.)
pub fn share(module: usize, library: usize) {
    let mut known = known();
    if module == library {
        return;
    }
    if known.contains_key(&library) {
        if known
            .get(&module)
            .is_some_and(|kernels| kernels.library == Some(library))
        {
            return;
        }
        known.insert(
            module,
            Kernels {
                library: Some(library),
                ..Default::default()
            },
        );
    } else if known.remove(&module).is_none() {
        // (A context can return the former module address for a different library, including a
        // native one with no recorded refusals: what was known of it goes. Here nothing was.)
        return;
    }
    changed(&known);
}

/// The module or library is unloaded.
pub fn forget(handle: usize) {
    drop(take(handle));
}

/// `module`, the module of a library in a context on another GPU than this one, has none of the
/// library's refusals (`share` is for a context on this GPU). A runtime may say it at every launch,
/// through `cuKernelGetFunction`: when nothing was known of it, nothing changes, and no new
/// generation begins.
pub fn not_shared(module: usize) {
    let mut known = known();
    if known.remove(&module).is_some() {
        changed(&known);
    }
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
    // (Also when nothing was known of it: the handles of its functions may be given to others.)
    changed(&known);
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
    changed(&known);
}

/// Whether enumerations need to filter this module or library.
pub fn has_refused(handle: usize) -> bool {
    metadata(&known(), handle).is_some_and(|k| !k.refused.is_empty())
}

/// How many kernels of `handle` must not be given to the program, the first of them by name and
/// why; `None` when there is none.
pub fn first_refused(handle: usize) -> Option<(usize, String, String)> {
    let known = known();
    let refused = &metadata(&known, handle)?.refused;
    let (kernel, (_, reason)) = refused.iter().min_by(|a, b| a.0.cmp(b.0))?;
    Some((refused.len(), kernel.clone(), reason.clone()))
}

/// Kernel `name` of `handle` must not be given to the program. The first time the program asks for
/// it, the reason is printed.
pub fn refused(handle: usize, name: &str) -> bool {
    let known = known();
    let Some(kernels) = metadata(&known, handle) else {
        return false;
    };
    let Some((function, reason)) = kernels.refused.get(name) else {
        return false;
    };
    let shown = kernels.shown.clone();
    let explain = EXPLAINED.lock().unwrap_or_else(|e| e.into_inner()).refused(
        name,
        function,
        reason,
        config().emulation_warnings,
        log::at_level(2),
    );
    let through = if function == name {
        String::new()
    } else {
        format!(" (in {function}, which it calls)")
    };
    match explain {
        Explain::Nothing => {}
        Explain::First => {
            log::warn(format_args!(
                "kernel {name} cannot run on this GPU and is not given to the program: {reason}{through}"
            ));
            if let Some(remedy) = nccl_remedy(name) {
                log::warn(format_args!("{remedy}"));
            }
        }
        Explain::Again => {
            let which = shown.map_or(String::new(), |shown| format!(", {shown}"));
            log::warn(format_args!(
                "kernel {name} cannot run on this GPU and is not given to the program: {reason}{through} (another \
                 kernel of this name{which})"
            ));
        }
        Explain::Beyond => log::warn(format_args!(
            "further refused kernels are not said: more than {KEPT_KERNELS} were (each is still refused)"
        )),
        Explain::Further => log::warn(format_args!(
            "kernels refused under a name already said are not said any more (VF_LOG_LEVEL=2 says them all, \
             VF_EMULATION_WARNINGS=N says N)"
        )),
    }
    true
}

/// A refused kernel of which the program holds a handle: its name, the function it needs that
/// traps, and why.
pub struct Refusal {
    name: String,
    function: String,
    reason: String,
}

/// Refused kernels of which the program used a handle: the refusal is explained once.
static GUARDED: LazyLock<Mutex<Guarded>> = LazyLock::new(|| Mutex::new(Guarded::default()));

/// The names of the refused kernels whose use was explained, at most `KEPT_KERNELS` of them.
#[derive(Default)]
struct Guarded {
    said: HashSet<String>,
    /// Refused kernels were used beyond those kept.
    beyond: bool,
}

/// What to say of the use of a refused kernel.
#[derive(Debug, PartialEq, Eq)]
enum Note {
    /// Nothing: it was said, or it comes beyond those kept and that was said.
    Nothing,
    /// Its refusal: the first use of a kernel of its name.
    First,
    /// That the uses of further refused kernels are not said: more than `KEPT_KERNELS` were.
    Beyond,
}

impl Guarded {
    /// A handle to the refused kernel `name` was used: what to say. (The use is refused whatever
    /// this says: only the line depends on it.)
    fn note(&mut self, name: &str) -> Note {
        if self.said.contains(name) {
            return Note::Nothing;
        }
        if self.said.len() >= KEPT_KERNELS {
            return match std::mem::replace(&mut self.beyond, true) {
                false => Note::Beyond,
                true => Note::Nothing,
            };
        }
        self.said.insert(name.to_string());
        Note::First
    }
}

impl Refusal {
    /// Say, the first time for this kernel, that `call` with a handle to it is refused.
    pub fn explain(&self, call: &str) {
        match GUARDED.lock().unwrap_or_else(|e| e.into_inner()).note(&self.name) {
            Note::Nothing => return,
            Note::Beyond => {
                log::warn(format_args!(
                    "further refused kernels whose handles the program uses are not said: more than {KEPT_KERNELS} \
                     were (each use is still refused, error {})",
                    crate::guard::REFUSED
                ));
                return;
            }
            Note::First => {}
        }
        let name = &self.name;
        let through = if self.function == self.name {
            String::new()
        } else {
            format!(" (in {}, which it calls)", self.function)
        };
        log::warn(format_args!(
            "kernel {name} cannot run on this GPU: {}{through}. The program holds a handle to it, which a lookup by name \
             refuses, and {call} with it is refused (error {})",
            self.reason,
            crate::guard::REFUSED
        ));
        if let Some(remedy) = nccl_remedy(name) {
            log::warn(format_args!("{remedy}"));
        }
    }
}

type GetHandle = unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> c_int;
type GetName = unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> c_int;
type FindKernel = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;

/// NVIDIA's function `index`, of the C type `T` (not counted: it is the layer that asks).
///
/// # Safety
/// `T` must be the C type of that function.
unsafe fn driver<T: Copy>(index: usize) -> T {
    let address = REALS[index].load(Relaxed);
    // SAFETY: REALS holds NVIDIA's function, or a stub that ignores its arguments.
    unsafe { std::mem::transmute_copy::<usize, T>(&address) }
}

/// The name that the driver's `getter` (`cuFuncGetName`, `cuKernelGetName`) gives for `handle`.
unsafe fn name_by(getter: usize, handle: usize) -> Option<String> {
    let get: GetName = unsafe { driver(getter) };
    let mut name: *const c_char = std::ptr::null();
    if unsafe { get(&mut name, handle as *mut c_void) } != 0 || name.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned())
}

/// What the driver says `handle` is: a kernel of a library (`cuKernelGetLibrary`; a kernel of a
/// deferred library is the library the program holds), or a function of a module
/// (`cuFuncGetModule`): that library or module, with the driver's function that names it
/// (`cuKernelGetName`, `cuFuncGetName`). `None` when it says neither. (The library first: NVIDIA's
/// driver answers 400 to `cuKernelGetLibrary` for a function, measured on a V100 with driver
/// 580.178.04; `cuFuncGetName` takes a function only.)
unsafe fn owner_of(handle: usize) -> Option<(usize, usize)> {
    let get_library: GetHandle = unsafe { driver(idx::cuKernelGetLibrary) };
    let mut library = std::ptr::null_mut();
    if unsafe { get_library(&mut library, handle as *mut c_void) } == 0 && !library.is_null() {
        let owner = crate::lazy::kernel_library(handle).unwrap_or(library as usize);
        return Some((owner, idx::cuKernelGetName));
    }
    let get_module: GetHandle = unsafe { driver(idx::cuFuncGetModule) };
    let mut module = std::ptr::null_mut();
    if unsafe { get_module(&mut module, handle as *mut c_void) } != 0 || module.is_null() {
        return None;
    }
    Some((module as usize, idx::cuFuncGetName))
}

/// What the driver says `handle` is (`owner_of`), with its name.
unsafe fn identify(handle: usize) -> Option<(usize, String)> {
    let (owner, getter) = unsafe { owner_of(handle) }?;
    unsafe { name_by(getter, handle) }.map(|name| (owner, name))
}

/// Whether `handle` is the kernel `name` of `library` (a library's, in the driver), or its function
/// in the current context. A deferred library has it only once it was loaded whole.
unsafe fn of_library(handle: usize, library: usize, name: &str) -> bool {
    let whole = if crate::lazy::is_lazy(library) {
        match crate::lazy::materialized(library) {
            Some(whole) => Some(whole),
            None => return false,
        }
    } else {
        None
    };
    let target = whole.as_ref().map_or(library, |whole| whole.handle);
    let Ok(name) = std::ffi::CString::new(name) else {
        return false;
    };
    let find: FindKernel = unsafe { driver(idx::cuLibraryGetKernel) };
    let mut kernel = std::ptr::null_mut();
    if unsafe { find(&mut kernel, target as *mut c_void, name.as_ptr()) } != 0 || kernel.is_null() {
        return false;
    }
    if kernel as usize == handle {
        return true;
    }
    let function_of: GetHandle = unsafe { driver(idx::cuKernelGetFunction) };
    let mut function = std::ptr::null_mut();
    let found = unsafe { function_of(&mut function, kernel) };
    found == 0 && function as usize == handle
}

/// A kernel without code of which the program holds a handle: its name, and the image it is in.
pub struct NoCode {
    name: String,
    image: Option<(usize, u64)>,
}

/// Kernels without code whose use was refused: the refusal is said once for each name.
static NO_CODE_SAID: LazyLock<Mutex<Guarded>> = LazyLock::new(|| Mutex::new(Guarded::default()));

impl NoCode {
    /// Say, the first time for this kernel, that `call`, which would run it, is refused.
    pub fn explain(&self, call: &str) {
        match NO_CODE_SAID.lock().unwrap_or_else(|e| e.into_inner()).note(&self.name) {
            Note::Nothing => return,
            Note::Beyond => {
                log::warn(format_args!(
                    "further kernels without code whose launch is refused are not said: more than {KEPT_KERNELS} \
                     were (each launch is still refused, error {})",
                    crate::guard::REFUSED
                ));
                return;
            }
            Note::First => {}
        }
        let image = match self.image {
            Some((size, digest)) => format!("its image ({size} bytes, XXH64 {digest:016x})"),
            None => "its image".to_string(),
        };
        log::warn(format_args!(
            "kernel {} has no code in {image}: its body is a trap and nothing else; {call} with it is refused \
             (error {})",
            self.name,
            crate::guard::REFUSED
        ));
    }
}

/// What `verdict` says of a handle.
pub enum Verdict {
    /// The handle is that of a refused kernel.
    Refused(Refusal),
    /// The handle is that of a kernel without code: it may be asked about, not run.
    NoCode(NoCode),
    /// It is not, wherever it is used.
    Allowed,
    /// It is not, where it is used now, which another use may not be: so this is not kept.
    /// - A refused kernel of a library, or its function, used where the GPU is not the one whose
    ///   code the layer adapts: the driver runs the library's own code there (a library keeps the
    ///   entries of the other GPUs), as the module of the library in such a context gives the
    ///   kernel by name. Used on this GPU, it is refused. (Not for a deferred library, whose code is
    ///   the PTX adapted for this GPU on every GPU.)
    /// - A function of a module unknown here, named as the refused kernel of a library, that is
    ///   not that kernel's function in the current context: it may be its function in another
    ///   context, which the driver is asked about while another context, or none, is current.
    AllowedHere,
}

/// What `handle` is: a refused kernel, if it is a function or a kernel whose module or library is
/// known here with that kernel refused, or a function of a module unknown here that is the module
/// the driver made in this context for a library with that kernel refused (which the program did
/// not ask for with `cuLibraryGetModule` or `cuKernelGetFunction`). The refusals of a library that
/// is not deferred hold where the handle is used on this GPU: on `device`, or in the current
/// context when `None`.
///
/// # Safety
/// `handle` must be one that the driver accepts for a launch, or that it refuses.
pub unsafe fn verdict(handle: usize, device: Option<c_int>) -> Verdict {
    let Some((owner, name)) = (unsafe { identify(handle) }) else {
        return Verdict::Allowed;
    };
    let refusal = |kernels: &Kernels| {
        kernels.refused.get(&name).map(|(function, reason)| Refusal {
            name: name.clone(),
            function: function.clone(),
            reason: reason.clone(),
        })
    };
    let no_code = |kernels: &Kernels| {
        kernels.trap_only.contains(&name).then(|| NoCode {
            name: name.clone(),
            image: kernels.image,
        })
    };
    // A refusal, else a kernel without code, of a module or library.
    let finding = |kernels: &Kernels| {
        refusal(kernels)
            .map(Verdict::Refused)
            .or_else(|| no_code(kernels).map(Verdict::NoCode))
    };
    let on_this_gpu = || match device {
        Some(device) => crate::hooks::device_needs_adapting(device),
        None => crate::hooks::context_needs_adapting(),
    };
    // A refusal of `library` (or a kernel of it without code): for the GPU where the handle is used,
    // where the driver runs the library's own code for another GPU. (The module that the driver gives
    // for a library in a context is known here only for a context on this GPU.)
    let of_a_library = |library: usize, found: Option<Verdict>| match found {
        Some(_) if !crate::lazy::is_lazy(library) && !on_this_gpu() => Verdict::AllowedHere,
        Some(found) => found,
        None => Verdict::Allowed,
    };
    let libraries: Vec<usize> = {
        let known = known();
        if let Some(kernels) = metadata(&known, owner) {
            let found = finding(kernels);
            let library_kernel = known.get(&owner).is_some_and(|entry| entry.is_library);
            drop(known);
            return match found {
                Some(found) if library_kernel => of_a_library(owner, Some(found)),
                Some(found) => found,
                None => Verdict::Allowed,
            };
        }
        known
            .iter()
            .filter(|(_, kernels)| {
                kernels.is_library && (kernels.refused.contains_key(&name) || kernels.trap_only.contains(&name))
            })
            .map(|(&library, _)| library)
            .collect()
    };
    if libraries.is_empty() {
        return Verdict::Allowed;
    }
    for &library in &libraries {
        if unsafe { of_library(handle, library, &name) } {
            let found = known().get(&library).and_then(finding);
            return of_a_library(library, found);
        }
    }
    // (Compared with the function of each library's kernel in the current context only.)
    Verdict::AllowedHere
}

/// Kernel `name` of `handle` is given to the program: announce it if it runs emulated instructions.
/// When the profiler is on, also record the code origin and emulated counts for the kernel.
pub fn given(handle: usize, name: &str) {
    let (counts, origin_val, load, shown, reported) = {
        let k = known();
        let kernels = metadata(&k, handle);
        let counts = kernels.and_then(|k| k.emulated.get(name)).cloned();
        let origin_val = kernels.and_then(|k| k.origin);
        let shown = kernels.and_then(|k| k.shown.clone());
        let reported = kernels.and_then(|k| k.reported.get(name)).cloned();
        (counts, origin_val, load_of(&k, handle), shown, reported)
    };
    if let Some(reported) = &reported
        && let Some(line) = traps_given(name, reported, log::at_level(2))
    {
        log::info(format_args!("{line}"));
    }
    if let Some(ref counts) = counts {
        let limit = config().emulation_warnings;
        let all = log::at_level(2);
        for line in ANNOUNCER.given(name, load, &images::digest_of, shown.as_deref(), counts, limit, all) {
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

/// For the report at exit: the totals of what was announced.
pub fn summary_line() -> Option<String> {
    ANNOUNCER.summary(config().emulation_warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_architecture_retargeted_from_holds_for_the_modules_of_a_library() {
        let (library, module, other) = (0x7e57_1000, 0x7e57_1010, 0x7e57_1020);
        retargeted(library, 75, true);
        assert_eq!(metadata(&known(), library).and_then(|k| k.retargeted_from), Some(75));
        assert!(known().get(&library).is_some_and(|k| k.is_library));
        // The module that the driver gives for the library in a context.
        share(module, library);
        assert_eq!(metadata(&known(), module).and_then(|k| k.retargeted_from), Some(75));
        // What is known of refusals merges with it.
        describe(
            library,
            None,
            vec![Refused {
                kernel: "bad".into(),
                function: "bad".into(),
                reason: "a test".into(),
            }],
            Vec::new(),
            true,
        );
        assert_eq!(metadata(&known(), module).and_then(|k| k.retargeted_from), Some(75));
        assert!(metadata(&known(), module).is_some_and(|k| k.refused.contains_key("bad")));
        // Gone with the library, and nothing for a module that was not retargeted.
        forget(library);
        assert!(metadata(&known(), module).is_none());
        assert!(metadata(&known(), other).is_none());
        forget(module);
    }

    #[test]
    fn a_kernel_that_only_the_code_not_given_has_is_said_once_at_a_lookup_that_fails() {
        // `tests/kernels/k_sections.cu`: `k_from_80` only in the code for sm_80, `k_before_80` only
        // in the code for sm_75. Told 8.0, the program is given the translation of the sm_75 code.
        const SECTIONS: &[u8] = include_bytes!("../../../tests/kernels/k_sections.135.sm75-sm80.fatbin");
        let (library, module, other) = (0x7e57_2000, 0x7e57_2010, 0x7e57_2020);
        let policy = vf_adapt::Policy {
            told_sm: Some(80),
            keep_sources: true,
            ..vf_adapt::Policy::default()
        };
        let fat = vf_fatbin::parse(SECTIONS).unwrap();
        let loads_sm = policy.told_machine_code(&fat, 75).unwrap();
        let given = policy.adapt_fatbin(&fat).unwrap().image.unwrap();
        let elsewhere = Elsewhere {
            given_sm: 75,
            told_sm: 80,
            loads_sm,
            why: policy.why_given(75, loads_sm),
            image: "lib.so: 9184 bytes, XXH64 0123456789abcdef".to_string(),
        };
        let mut replacement = images::Replacement::from_vec(given.clone());
        replacement.elsewhere = Some(elsewhere.clone());
        images::keep(library, replacement);
        // Through the module that the driver gives for the library in a context, and the library.
        share(module, library);
        let told = told_code(module, "k_from_80").expect("the sm_80 code has k_from_80");
        assert!(told.owner == library && told.elsewhere == elsewhere);
        let line = not_given_said("k_from_80", &told.elsewhere, None).expect("a line for k_from_80");
        assert!(
            line.ends_with("(LIMITS.md, \"The capability told with VF_CAPABILITY\")"),
            "{line}"
        );
        assert!(
            line.starts_with(
                "kernel k_from_80 is not in the code that the layer gives for its image (lib.so: 9184 bytes, XXH64 \
                 0123456789abcdef), its machine code for sm_75 translated: it is only in the image's machine code \
                 for sm_80, which a GPU of the capability told (8.0) loads. The lookup fails as the driver answers \
                 for a name that it does not have (error 500): the layer gives an image that has machine code for \
                 sm_75 and sm_80 its code for sm_75"
            ),
            "{line}"
        );
        // Once for each kernel.
        assert!(told_code(library, "k_from_80").is_some());
        assert_eq!(not_given_said("k_from_80", &elsewhere, None), None);
        // Why the kernel is not given from the code that has it, when that was tried.
        assert!(
            not_given_line(
                "k_from_80",
                &elsewhere,
                Some("its machine code for sm_80 is not translated (a reason)")
            )
            .ends_with(
                "(LIMITS.md, \"The capability told with VF_CAPABILITY\"). The kernel alone is not given from the code \
                 for sm_80 either: its machine code for sm_80 is not translated (a reason)"
            )
        );
        // A name that the code for sm_80 does not have either: nothing, and it is not read again.
        for name in ["k_nowhere", "k_before_80"] {
            assert!(told_code(library, name).is_none(), "{name}");
            assert!(
                metadata(&known(), library).is_some_and(|k| k.absent.contains(name)),
                "{name}"
            );
        }
        // An image of which nothing of the kind is known.
        let mut plain = images::Replacement::from_vec(given);
        plain.elsewhere = None;
        images::keep(other, plain);
        assert!(told_code(other, "k_from_80").is_none());
        // Gone with the library.
        drop(images::release(library));
        assert!(metadata(&known(), module).is_none());
        drop(images::release(other));
    }

    #[test]
    fn nccl_kernels_are_known_by_their_names() {
        // Names of NCCL 2.30 for CUDA 13, as the layer refused them on a V100.
        for name in [
            "_Z39ncclDevKernel_AllReduce_Sum_f32_RING_LL24ncclDevKernelArgsStorageILm4096EE",
            "_Z22ncclDevKernel_SendRecv24ncclDevKernelArgsStorageILm4096EE",
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
            // Its kernels for symmetric memory: NCCL goes on without those it cannot have.
            "_Z32ncclSymkDevKernel_AllGather_STMC21ncclSymkDevWorkArgs4K",
            "_Z33ncclSymkDevKernel_AllGather_TmaST21ncclSymkDevWorkArgs4K",
            "_Z",
            "",
        ] {
            assert!(!nccl_kernel(name), "{name}");
        }
    }

    #[test]
    fn a_refused_kernel_is_its_name_and_why() {
        let mut explained = Explained::default();
        let why = |n: u32| format!("reason {n}");
        assert_eq!(explained.refused("k", "k", &why(1), 2, false), Explain::First);
        assert_eq!(explained.refused("k", "k", &why(1), 2, false), Explain::Nothing);
        assert_eq!(explained.refused("other", "other", &why(1), 2, false), Explain::First);
        // Kernels of the name refused otherwise, up to the limit, then once that more are not said.
        assert_eq!(explained.refused("k", "k", &why(2), 2, false), Explain::Again);
        assert_eq!(explained.refused("other", "f", &why(1), 2, false), Explain::Again);
        assert_eq!(explained.refused("k", "k", &why(4), 2, false), Explain::Further);
        assert_eq!(explained.refused("k", "k", &why(5), 2, false), Explain::Nothing);
        assert_eq!(explained.refused("k", "k", &why(2), 2, false), Explain::Nothing);
        // All of them at level 2; the first of a name always.
        assert_eq!(explained.refused("k", "k", &why(6), 2, true), Explain::Again);
        assert_eq!(explained.refused("new", "new", &why(6), 2, false), Explain::First);
        // With a limit of zero, the first of each name only.
        let mut silent = Explained::default();
        assert_eq!(silent.refused("k", "k", &why(1), 0, true), Explain::First);
        assert_eq!(silent.refused("k", "k", &why(2), 0, true), Explain::Nothing);
    }

    #[test]
    fn what_is_kept_of_the_refused_uses_said_is_bounded() {
        let mut guarded = Guarded::default();
        let notes: Vec<Note> = (0..KEPT_KERNELS + 100)
            .map(|n| guarded.note(&format!("k{n}")))
            .collect();
        assert_eq!(guarded.said.len(), KEPT_KERNELS);
        assert_eq!(notes.iter().filter(|n| **n == Note::First).count(), KEPT_KERNELS);
        assert_eq!(notes.iter().filter(|n| **n == Note::Beyond).count(), 1);
        assert_eq!(notes[KEPT_KERNELS], Note::Beyond);
        // A kernel kept is still known as said; one beyond is not said, nor kept.
        assert_eq!(guarded.note("k0"), Note::Nothing);
        assert_eq!(guarded.note("another"), Note::Nothing);
        assert_eq!(guarded.said.len(), KEPT_KERNELS);
    }

    #[test]
    fn what_is_kept_of_the_refusals_said_is_bounded() {
        let mut explained = Explained::default();
        let mut said = Vec::new();
        for n in 0..KEPT_KERNELS as u32 * 2 {
            said.push(explained.refused(&format!("k{}", n % 3000), "f", &format!("reason {n}"), 20, true));
        }
        assert!(explained.said.len() + explained.names.len() <= KEPT_KERNELS);
        assert_eq!(said.iter().filter(|e| **e == Explain::Beyond).count(), 1);
        assert_eq!(
            said.iter().filter(|e| **e == Explain::First).count(),
            explained.names.len()
        );
        // One kept is still known.
        assert_eq!(explained.refused("k0", "f", "reason 0", 20, true), Explain::Nothing);
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

    #[test]
    fn a_kernel_given_with_reported_traps_is_said_once_counted_and_known_to_the_report_of_a_fault() {
        let (library, module) = (0x7e57_3000, 0x7e57_3010);
        let reported = |kernel: &str, instructions| Reported {
            kernel: kernel.into(),
            function: "_Z43ncclDevFunc_AllGather_COLLNET_DIRECT_SIMPLEv".into(),
            form: "multimem.ld_reduce.acquire.sys.global.min.u64".into(),
            instructions,
            functions: 3,
        };
        with_traps(
            library,
            Some("lib"),
            vec![reported("ring", 12), reported("tree", 1)],
            true,
        );
        share(module, library);
        // Through the module of the library too, for the report of a fault.
        assert_eq!(traps_of(module, "ring"), Some(reported("ring", 12)));
        assert_eq!(traps_of(module, "other"), None);
        // One line for the image at level 1: how many, the most reached, the forms, the names.
        let many: Vec<Reported> = ["a", "b", "c", "d", "e"].iter().map(|name| reported(name, 7)).collect();
        assert_eq!(
            image_traps_line(Some("cuLibraryLoadData of 9 bytes"), &many),
            "cuLibraryLoadData of 9 bytes: 5 kernels are given with instructions that cannot run on this GPU \
             replaced by reported traps (at most 7 in what one kernel may reach; \
             multimem.ld_reduce.acquire.sys.global.min.u64): a, b, c and 2 more. Such an instruction needs a \
             multicast address, which no GPU of this process can have: reached all the same, it stops its kernel \
             with an assertion that names it (CUDA_ERROR_ASSERT, 710) and the context ends. VF_LOG_LEVEL=2 says \
             each kernel when the program takes it; VF_REPORTED_TRAPS=0 refuses them instead"
        );
        assert!(image_traps_line(None, &many[..1]).starts_with("an image: 1 kernel is given with"));
        // Each kernel from level 2, once for each name; all counted.
        assert_eq!(
            traps_given("ring", &reported("ring", 12), true).unwrap(),
            "kernel ring is given with 12 instructions that cannot run on this GPU replaced by a reported trap: \
             multimem.ld_reduce.acquire.sys.global.min.u64 in _Z43ncclDevFunc_AllGather_COLLNET_DIRECT_SIMPLEv and 2 \
             other functions"
        );
        assert_eq!(traps_given("ring", &reported("ring", 12), true), None);
        assert_eq!(traps_given("tree", &reported("tree", 1), false), None);
        assert_eq!(traps_given("third", &reported("third", 1), false), None);
        let summary = traps_summary_line().unwrap();
        assert_eq!(
            summary,
            "3 kernels were given to the program with reported traps (in place of \
             multimem.ld_reduce.acquire.sys.global.min.u64)"
        );
        forget(library);
        assert_eq!(traps_of(module, "ring"), None);
        forget(module);
    }
}
