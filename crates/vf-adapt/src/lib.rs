//! Decide what the driver of a Volta GPU should load in place of an image that carries nothing for it.
//!
//! An image is a fat binary, a cubin or PTX text. The driver picks from a fat binary the entry made for
//! its GPU: machine code for the same SM major version with a minor version it reaches, or PTX whose
//! target it reaches. CUDA 13 builds stop at sm_75, so on sm_70 nothing is picked. What can be done:
//!
//! - PTX entries for a later target are rewritten as sm_70 PTX, which the driver's JIT compiles
//!   natively: the instruction stream nvcc emits for sm_75 is valid for sm_70 except for a few
//!   instructions, which `vf_ptx::lower_to_volta` replaces by sequences Volta has (`tanh`, the
//!   Turing tensor-core instructions). A function that uses a form without such a sequence is made
//!   to trap, and the kernels that need it are listed: they are not to be given to the program.
//! - Where there is no PTX, supported sm_75 and sm_80 instructions are translated to sm_70
//!   (`vf_sass`), function by function. The functions the translator refuses are left as stubs,
//!   and listed: their kernels are not to be given to the program either.
//! - When the driver's compiler still refuses functions of an adapted image, `without_refused` makes
//!   the image to load instead.

pub mod emulation;

use vf_fatbin::{self as fatbin, Entry, FatBinary, KIND_ELF, KIND_PTX, NewEntry};
pub use vf_ptx::{Emulated, EmulatedCounts, Refused, RefusedFunction, Trapped};

/// The lowest CUDA 13 machine-code target and the default capability reported on Volta.
pub const TRANSLATED_FROM: u32 = 75;

/// The GPU and the driver the image is adapted for.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// SM architecture of the GPU (70 for Volta).
    pub device_sm: u32,
    /// Newest PTX ISA version the driver's JIT accepts (9.0 for the 580 branch).
    pub ptx_max: (u32, u32),
    /// What the rewriting of instructions may take for granted about the programs.
    pub lowering: vf_ptx::lower::Options,
    /// The architecture the programs are told the GPU has, when the user chose it
    /// (`VF_CAPABILITY`): the code taken from a fat binary is then the code for that architecture.
    /// `None`: a program is told the lowest architecture it has code for, and that code is taken.
    pub told_sm: Option<u32>,
    /// When supported machine code is translated for the device.
    pub translate: Translate,
    /// Whether pairs of products on whole rows are translated as such (`VF_PAIRS=0`: they are not).
    pub pairs: bool,
    /// Where the translation of machine code takes its registers (`VF_BORROW`; the values are
    /// those of `vf_sass::translate::Options::borrow`). The default, 7: above the program's
    /// wherever that translates, and for a kernel refused for want of registers, among the
    /// program's own that hold nothing, then with nothing kept in a register at all, kernels with
    /// tensor-core products included.
    pub borrow: u32,
}

/// When machine code is translated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Translate {
    /// Never: an image with nothing but such code is left as it is.
    Never,
    /// When an image has nothing else for the device: no PTX to rewrite.
    WithoutPtx,
    /// Whenever an image has such code, also where it has PTX that could be rewritten: to set the two
    /// ways against each other.
    First,
    /// Translate sm_75 machine code even when sm_70 code is available: to isolate the effect of the
    /// translation from the effect of the compiler target. The sm_70 entries are removed.
    Force,
}

/// A compute capability written `<major>.<minor>`, as an SM architecture number (8.6 is sm_86).
pub fn capability_sm(text: &str) -> Option<u32> {
    let (major, minor) = text.trim().split_once('.')?;
    let (major, minor): (u32, u32) = (major.parse().ok()?, minor.parse().ok()?);
    if minor >= 10 {
        return None;
    }
    major.checked_mul(10)?.checked_add(minor)
}

/// With `VF_TRANSLATE_DUMP=DIR`, every cubin given to the translator is written to
/// `DIR/<pid>-<number>.sm_<target>.cubin` as it was received: the machine code a program really loads,
/// to be translated and checked apart (`reference/validate.py`). Libraries keep some of
/// theirs where `vf extract` does not find it. (The directory must exist; what cannot be
/// written is left out, and there is nowhere here to say so.)
fn dump_cubin(cubin: &[u8], sm: u32) {
    static NUMBER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if let Ok(dir) = std::env::var("VF_TRANSLATE_DUMP") {
        let number = NUMBER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = std::fs::write(format!("{dir}/{}-{number:04}.sm_{sm}.cubin", std::process::id()), cubin);
    }
}

impl Default for Policy {
    /// The layer takes programs to keep the rows that `ldmatrix` reads at multiples of 16, as the PTX
    /// manual requires of them (`vf_ptx::lower::Options::aligned_rows`): the programs measured do,
    /// and testing it costs every program 17 % to a factor of two on its emulated tensor-core kernels
    /// (`docs/emulations.md`, "Matrix loads and moves"). A program that does not stops with a misaligned-address fault;
    /// `with_environment` lets its user ask for the test.
    fn default() -> Self {
        Policy {
            device_sm: 70,
            ptx_max: (9, 0),
            lowering: vf_ptx::lower::Options {
                aligned_rows: true,
                ..Default::default()
            },
            told_sm: None,
            translate: Translate::WithoutPtx,
            pairs: true,
            borrow: vf_sass::translate::BORROW_DEFAULT,
        }
    }
}

/// What was found in an image and what was done about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The image already holds code the driver uses for this GPU: left as is.
    Native,
    /// PTX for architecture `from_sm` was rewritten for the device (`entries` fat-binary entries, or 1
    /// for PTX text). `tensor_rewritten` and `tanh_expanded` count the tensor-core and `tanh`
    /// instructions replaced by sequences the device has; the `trapped` functions use an instruction
    /// without such a sequence and now trap.
    Retargeted {
        entries: usize,
        from_sm: u32,
        tensor_rewritten: usize,
        tanh_expanded: usize,
        trapped: Vec<Trapped>,
        /// The kernels that are, or call, a trapped function: not to be given to the program.
        refused: Vec<Refused>,
        /// The kernels that run emulated instructions, with the counts by kind.
        emulated: Vec<Emulated>,
    },
    /// Machine code for architecture `from_sm` was translated to machine code for the device
    /// (`entries` fat-binary entries, or 1 for a cubin), `functions` functions in all.
    Translated {
        entries: usize,
        from_sm: u32,
        functions: usize,
        /// The functions the translator refused, which are stubs in the image: not to be given to
        /// the program.
        refused: Vec<Refused>,
        /// The functions that have more instructions than they had (`EmulatedCounts::machine_code`).
        emulated: Vec<Emulated>,
        /// A function of the translation takes more registers than its budget (what it declared,
        /// rounded up to eight, 32 at least): the driver may refuse large blocks of it, which a
        /// second translation within the budget would have (`Policy::kept_for_second`).
        over_budget: bool,
    },
    /// Machine code for `from_sm` that the translator could not read: left as is.
    NotTranslated { from_sm: u32, reason: String },
    /// Only machine code for newer architectures that nothing translates; `lowest_sm` is the closest.
    NeedsTranslation { lowest_sm: u32 },
    /// Nothing recognised: left as is.
    Unknown,
}

/// What the fat binaries inside a file (a shared library, an executable) hold for the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Embedded {
    /// No fat binary.
    Nothing,
    /// Code the driver uses on the device.
    Usable,
    /// Code for other architectures only; `lowest_sm` is the lowest of them.
    Other { lowest_sm: u32 },
}

/// The result of adapting an image: `image` is the replacement, when there is one.
#[derive(Debug)]
pub struct Adapted {
    pub outcome: Outcome,
    pub image: Option<Vec<u8>>,
}

/// What to load after the driver's compiler refused functions of an adapted image.
#[derive(Debug)]
pub struct Retry {
    /// The image with those functions made to trap.
    pub image: Vec<u8>,
    pub functions: Vec<RefusedFunction>,
    /// The kernels that are or call one of them: not to be given to the program.
    pub kernels: Vec<Refused>,
}

impl Adapted {
    fn unchanged(outcome: Outcome) -> Adapted {
        Adapted { outcome, image: None }
    }

    fn complete_ptx(&self) -> bool {
        matches!(&self.outcome, Outcome::Retargeted { trapped, refused, .. } if trapped.is_empty() && refused.is_empty())
    }
}

/// A cubin translated for the device.
struct Translation {
    cubin: Vec<u8>,
    functions: usize,
    refused: Vec<Refused>,
    emulated: Vec<Emulated>,
    over_budget: bool,
}

impl Translation {
    /// Replacing this image with PTX cannot discard a working machine-code entry point.
    fn all_entries_refused(&self, cubin: &[u8]) -> bool {
        !self.refused.is_empty()
            && fatbin::cubin_entries(cubin).is_some_and(|names| {
                !names.is_empty() && names.iter().all(|name| self.refused.iter().any(|r| r.kernel == *name))
            })
    }
}

/// Reconstruct the single PTX module kept as NUL-separated lines by ptxas. Match the
/// architecture and the complete set of entry points before handing it to the usual PTX path.
fn embedded_ptx(cubin: &[u8], sm: u32) -> Option<(Vec<u8>, vf_ptx::Header)> {
    if fatbin::cubin_sm(cubin)? != sm {
        return None;
    }
    let source = fatbin::cubin_debug_ptx(cubin)?;
    if source.last() != Some(&0) || !source.is_ascii() {
        return None;
    }
    let mut text = source.to_vec();
    for byte in &mut text {
        if *byte == 0 {
            *byte = b'\n';
        }
    }
    let mut directives = text.split(|&b| b == b'\n').filter_map(|line| {
        let keyword = line.trim_ascii_start().split(|b| b.is_ascii_whitespace()).next()?;
        matches!(keyword, b".version" | b".target").then_some(keyword)
    });
    let header = vf_ptx::header(&text)?;
    if directives.next() != Some(b".version")
        || directives.next() != Some(b".target")
        || directives.next().is_some()
        || header.target != sm
    {
        return None;
    }
    let mut names = fatbin::cubin_entries(cubin)?;
    let mut entries: Vec<_> = vf_ptx::functions(&text)
        .into_iter()
        .filter(|f| f.entry)
        .map(|f| &text[f.name])
        .collect();
    names.sort_unstable();
    entries.sort_unstable();
    if entries.is_empty() || !names.iter().map(|name| name.as_bytes()).eq(entries) {
        return None;
    }
    Some((text, header))
}

impl Policy {
    /// Offline CUDA 12.9 tools accept PTX ISA through 8.8 on Volta.
    pub fn cuda_12_9() -> Self {
        Self {
            ptx_max: (8, 8),
            ..Self::default()
        }
    }

    /// This policy with what the environment asks for. `VF_ALIGNED_ROWS=0`: do not take the rows
    /// that `ldmatrix` reads to be at multiples of 16; test their addresses when the code runs, for a
    /// program that does not follow the PTX manual there (slower: see `Policy::default`).
    /// `VF_BF16_TC_GROUPS=1`: use guarded Volta FP16 Tensor Cores for groups of BF16 products.
    /// `VF_BF16_TC_TILES=1`: apply the same guarded conversion to two-by-two BF16 tiles.
    /// `VF_CAPABILITY=<major>.<minor>`: the compute capability the programs are told (`told_sm`).
    /// `VF_TRANSLATE=0`: do not translate machine code; `VF_TRANSLATE=first`: translate it also
    /// where an image has PTX (`Translate`).
    pub fn with_environment(mut self) -> Policy {
        match std::env::var("VF_TRANSLATE").as_deref() {
            Ok("0") => self.translate = Translate::Never,
            Ok("first") => self.translate = Translate::First,
            Ok("force") => self.translate = Translate::Force,
            _ => {}
        }
        if let Ok(value) = std::env::var("VF_ALIGNED_ROWS") {
            self.lowering.aligned_rows = value != "0";
        }
        if let Ok(value) = std::env::var("VF_BF16_TC_GROUPS") {
            self.lowering.bf16_tc_groups = value == "1";
        }
        if let Ok(value) = std::env::var("VF_BF16_TC_TILES") {
            self.lowering.bf16_tc_tiles = value == "1";
        }
        if let Ok(value) = std::env::var("VF_BF16_ROW") {
            use vf_ptx::lower::Bf16Row;
            self.lowering.bf16_row = match value.as_str() {
                "ordered" => Bf16Row::Ordered,
                "tree" => Bf16Row::Tree,
                _ => Bf16Row::Off,
            };
        }
        if let Some(sm) = std::env::var("VF_CAPABILITY").ok().as_deref().and_then(capability_sm) {
            self.told_sm = Some(sm);
        }
        if let Ok(value) = std::env::var("VF_PAIRS") {
            self.pairs = value != "0";
        }
        if let Ok(value) = std::env::var("VF_BORROW") {
            self.borrow = vf_sass::translate::borrow_from(Some(&value));
        }
        self
    }

    /// Machine code for `sm` runs on the device: same major version, minor version not above.
    fn runs(&self, sm: u32) -> bool {
        sm / 10 == self.device_sm / 10 && sm <= self.device_sm
    }

    fn jit_accepts(&self, entry: &Entry<'_>) -> bool {
        let (major, minor) = entry.content_version();
        entry.sm() <= self.device_sm && (major as u32, minor as u32) <= self.ptx_max
    }

    /// PTX that can be retargeted to the device: PTX written for any later architecture. What decides
    /// is its instructions: `vf_ptx::lower_to_volta` rewrites those it has a sequence for and traps
    /// the functions that use others it knows of. It does not know everything later targets have, so a
    /// retargeted module can still be refused by the compiler, which is for the caller to handle.
    pub fn retargetable(&self, sm: u32) -> bool {
        sm > self.device_sm
    }

    /// The driver uses this fat-binary entry on the device.
    pub fn usable(&self, entry: &Entry<'_>) -> bool {
        match entry.kind() {
            KIND_ELF => self.runs(entry.sm()),
            KIND_PTX => self.jit_accepts(entry),
            _ => false,
        }
    }

    /// Look through `file` for fat binaries until one holds code for the device.
    pub fn embedded(&self, file: &[u8]) -> Embedded {
        let magic = fatbin::MAGIC.to_le_bytes();
        let mut lowest: Option<u32> = None;
        let mut offset = 0;
        while let Some(found) = file[offset..].windows(4).position(|w| w == magic) {
            let at = offset + found;
            offset = at + 4;
            let Ok(fat) = fatbin::parse(&file[at..]) else {
                continue;
            };
            if fat.entries.is_empty() {
                continue;
            }
            if fat.entries.iter().any(|entry| self.usable(entry)) {
                return Embedded::Usable;
            }
            let code = fat.entries.iter().filter(|e| matches!(e.kind(), KIND_ELF | KIND_PTX));
            lowest = code.map(|e| e.sm()).chain(lowest).min();
            offset = at + fat.len;
        }
        lowest.map_or(Embedded::Nothing, |lowest_sm| Embedded::Other { lowest_sm })
    }

    /// `text` for the device: Turing-only instructions lowered, then the header.
    fn lower_ptx(&self, text: &[u8]) -> (Vec<u8>, vf_ptx::Lowering) {
        let (mut lowered, lowering) = if !vf_ptx::needs_lowering(text) {
            (text.to_vec(), vf_ptx::Lowering::default())
        } else {
            vf_ptx::lower_to_volta_with(text, self.lowering)
        };
        if self.device_sm == 70 {
            vf_ptx::division::specialize(&mut lowered);
        }
        vf_ptx::retarget(&mut lowered, self.ptx_max, self.device_sm);
        (lowered, lowering)
    }

    /// Adapt PTX text (no terminating zero needed, none added).
    pub fn adapt_ptx(&self, text: &[u8]) -> Adapted {
        let Some(header) = vf_ptx::header(text) else {
            return Adapted::unchanged(Outcome::Unknown);
        };
        if header.target <= self.device_sm && header.version <= self.ptx_max {
            return Adapted::unchanged(Outcome::Native);
        }
        if header.target > self.device_sm && !self.retargetable(header.target) {
            return Adapted::unchanged(Outcome::NeedsTranslation {
                lowest_sm: header.target,
            });
        }
        let (image, lowering) = self.lower_ptx(text);
        let outcome = Outcome::Retargeted {
            entries: 1,
            from_sm: header.target,
            tensor_rewritten: lowering.tensor_rewritten,
            tanh_expanded: lowering.tanh_expanded,
            trapped: lowering.trapped,
            refused: lowering.refused,
            emulated: lowering.emulated,
        };
        Adapted {
            outcome,
            image: Some(image),
        }
    }

    /// Machine code for `sm` can be translated for the device.
    fn translatable(&self, sm: u32) -> bool {
        self.translate != Translate::Never && self.device_sm == 70 && matches!(sm, 75 | 80)
    }

    /// Translate one image of the selected architecture into machine code for the device.
    fn translate_cubin(&self, cubin: &[u8], sm: u32) -> Result<Translation, String> {
        if fatbin::cubin_sm(cubin) != Some(sm) {
            return Err(format!("the cubin header does not match sm_{sm}"));
        }
        dump_cubin(cubin, sm);
        let opts = vf_sass::translate::Options {
            pairs: self.pairs,
            aligned_rows: self.lowering.aligned_rows,
            borrow: self.borrow,
        };
        let translated = vf_sass::translate::translate_with_options(cubin, false, &opts)?;
        let (mut refused, mut emulated) = (Vec::new(), Vec::new());
        for function in &translated.report {
            match &function.refused {
                Some(reason) => refused.push(Refused {
                    kernel: function.name.clone(),
                    function: function.name.clone(),
                    reason: format!("its machine code for sm_{sm} is not translated ({reason})"),
                }),
                None if function.after > function.before => emulated.push(Emulated {
                    kernel: function.name.clone(),
                    counts: EmulatedCounts {
                        machine_code: function.after - function.before,
                        ..EmulatedCounts::default()
                    },
                }),
                None => {}
            }
        }
        Ok(Translation {
            over_budget: translation_exceeds_budget(cubin, &translated.bytes),
            cubin: translated.bytes,
            functions: translated.report.len(),
            refused,
            emulated,
        })
    }

    /// Adapt a cubin. Native images need only their header. Later executables can retain PTX or
    /// use an instruction set the machine-code translator understands.
    pub fn adapt_cubin(&self, cubin: &[u8]) -> Adapted {
        match fatbin::cubin_sm(cubin) {
            Some(sm) if self.runs(sm) => Adapted::unchanged(Outcome::Native),
            // Ampere images with retained PTX already run through the PTX lowerer.
            Some(80)
                if self.device_sm == 70
                    && self.translate == Translate::WithoutPtx
                    && let Some((text, _)) = embedded_ptx(cubin, 80) =>
            {
                self.adapt_ptx(&text)
            }
            Some(sm) if self.translatable(sm) => match self.translate_cubin(cubin, sm) {
                Ok(translation)
                    if sm == 75
                        && self.translate == Translate::WithoutPtx
                        && translation.all_entries_refused(cubin)
                        && let Some(recovered) = embedded_ptx(cubin, sm)
                            .map(|(text, _)| self.adapt_ptx(&text))
                            .filter(Adapted::complete_ptx) =>
                {
                    recovered
                }
                Ok(translation) => Adapted {
                    outcome: Outcome::Translated {
                        entries: 1,
                        from_sm: sm,
                        functions: translation.functions,
                        refused: translation.refused,
                        emulated: translation.emulated,
                        over_budget: translation.over_budget,
                    },
                    image: Some(translation.cubin),
                },
                Err(reason) => Adapted::unchanged(Outcome::NotTranslated { from_sm: sm, reason }),
            },
            Some(sm) if self.retargetable(sm) => embedded_ptx(cubin, sm)
                .map(|(text, _)| self.adapt_ptx(&text))
                .unwrap_or_else(|| Adapted::unchanged(Outcome::NeedsTranslation { lowest_sm: sm })),
            Some(sm) => Adapted::unchanged(Outcome::NeedsTranslation { lowest_sm: sm }),
            None => Adapted::unchanged(Outcome::Unknown),
        }
    }

    /// Whether `adapt_cubin` would need all of a cubin with this header, not the header alone.
    pub fn translates_cubin(&self, header: &[u8]) -> bool {
        fatbin::cubin_sm(header).is_some_and(|sm| !self.runs(sm) && (self.translatable(sm) || self.retargetable(sm)))
    }

    /// Translate the machine code for one target, keeping entries for other GPUs. With the
    /// default policy, use a later target's complete retained PTX when it is available.
    fn translate_fatbin(&self, fat: &FatBinary<'_>) -> Result<Option<Adapted>, fatbin::Error> {
        let Some(source) = self.machine_source(fat) else {
            return Ok(None);
        };
        if self.translate == Translate::WithoutPtx
            && let Some(sm) = self.source_of(fat, KIND_ELF).filter(|&sm| sm >= 80)
        {
            let recovered = self.retarget_fatbin(fat, KIND_ELF, sm)?;
            if matches!(recovered.outcome, Outcome::Retargeted { .. }) {
                return Ok(Some(recovered));
            }
        }
        let mut entries = Vec::with_capacity(fat.entries.len());
        let (mut translated, mut functions, mut refused, mut emulated) = (0, 0, Vec::new(), Vec::new());
        let mut over_budget = false;
        let mut all_refused = source == 75 && self.translate == Translate::WithoutPtx;
        for entry in &fat.entries {
            if entry.kind() != KIND_ELF || entry.sm() != source {
                // In Force mode, drop native sm_70 entries so the driver uses the translated ones.
                if self.translate == Translate::Force && entry.kind() == KIND_ELF && self.runs(entry.sm()) {
                    continue;
                }
                entries.push(NewEntry::copy_of(entry));
                continue;
            }
            let payload = entry.payload()?;
            match self.translate_cubin(&payload, source) {
                Ok(translation) => {
                    all_refused = all_refused && translation.all_entries_refused(&payload);
                    entries.push(NewEntry::replace(
                        entry,
                        translation.cubin,
                        self.device_sm,
                        entry.content_version(),
                    ));
                    translated += 1;
                    functions += translation.functions;
                    over_budget |= translation.over_budget;
                    refused.extend(translation.refused);
                    emulated.extend(translation.emulated);
                }
                Err(reason) => {
                    return Ok(Some(Adapted::unchanged(Outcome::NotTranslated {
                        from_sm: entry.sm(),
                        reason,
                    })));
                }
            }
        }
        if all_refused {
            let recovered = self.retarget_fatbin(fat, KIND_ELF, source)?;
            if recovered.complete_ptx() {
                return Ok(Some(recovered));
            }
        }
        Ok(Some(Adapted {
            outcome: Outcome::Translated {
                entries: translated,
                from_sm: source,
                functions,
                refused,
                emulated,
                over_budget,
            },
            image: Some(fatbin::build(&entries)),
        }))
    }

    /// The architecture to use for PTX entries or, when none exist, PTX retained inside cubins.
    ///
    /// A program takes its decisions (which kernels to launch, with which parameters) for the GPU it
    /// is told it runs on, and the driver of that GPU would give it the code of the highest target
    /// the GPU reaches. The code for a lower target can be other code or none at all: the attention
    /// kernels of llama.cpp for sm_75 are traps where those for sm_80 are the real thing, and told
    /// 8.0 it launches them. So when the user chose what the programs are told, that entry is taken.
    /// Otherwise a program is told the lowest architecture it has code for (`callers` in the proxy):
    /// PTX at or below the device target in a dialect too new for the driver, else the nearest
    /// target above. The same when nothing reaches what the user chose.
    fn source_of(&self, fat: &FatBinary<'_>, kind: u16) -> Option<u32> {
        self.preferred_sm(
            fat.entries
                .iter()
                .filter(|e| e.kind() == kind)
                .map(Entry::sm)
                .filter(|&sm| sm <= self.device_sm || self.retargetable(sm)),
        )
    }

    fn machine_source(&self, fat: &FatBinary<'_>) -> Option<u32> {
        let targets = fat
            .entries
            .iter()
            .filter(|e| e.kind() == KIND_ELF)
            .map(Entry::sm)
            .filter(|&sm| self.translatable(sm));
        // The default Turing translation is the established path for multi-target libraries,
        // including PyTorch. Ampere's supported instruction set is currently narrower.
        if self.translate == Translate::WithoutPtx {
            targets.min()
        } else {
            self.preferred_sm(targets)
        }
    }

    fn preferred_sm(&self, targets: impl Iterator<Item = u32> + Clone) -> Option<u32> {
        let told = self
            .told_sm
            .and_then(|told| targets.clone().filter(|&sm| sm <= told).max());
        let nearest = targets.min_by_key(|&sm| if sm <= self.device_sm { 0 } else { sm });
        told.or(nearest)
    }

    /// Adapt a fat binary: when nothing in it is for the device, rewrite the PTX entries of one
    /// architecture (`source_of`) as device PTX. Without PTX, select one machine-code target.
    /// An executable for a later target can supply the PTX that ptxas kept with its line information.
    /// The other entries are kept.
    pub fn adapt_fatbin(&self, fat: &FatBinary<'_>) -> Result<Adapted, fatbin::Error> {
        if self.translate == Translate::Force
            && let Some(adapted) = self.translate_fatbin(fat)?
        {
            return Ok(adapted);
        }
        if fat.entries.iter().any(|entry| self.usable(entry)) {
            return Ok(Adapted::unchanged(Outcome::Native));
        }
        if self.translate == Translate::First
            && let Some(adapted) = self.translate_fatbin(fat)?
        {
            return Ok(adapted);
        }
        let (kind, source) = if let Some(source) = self.source_of(fat, KIND_PTX) {
            (KIND_PTX, source)
        } else {
            if let Some(adapted) = self.translate_fatbin(fat)? {
                return Ok(adapted);
            }
            let Some(source) = self.source_of(fat, KIND_ELF) else {
                return Ok(Adapted::unchanged(Outcome::Unknown));
            };
            (KIND_ELF, source)
        };
        self.retarget_fatbin(fat, kind, source)
    }

    fn retarget_fatbin(&self, fat: &FatBinary<'_>, kind: u16, source: u32) -> Result<Adapted, fatbin::Error> {
        let mut entries = Vec::with_capacity(fat.entries.len());
        let (mut rewritten, mut tensor_rewritten, mut tanh_expanded, mut trapped) = (0, 0, 0, Vec::new());
        let (mut refused, mut emulated) = (Vec::new(), Vec::new());
        for entry in &fat.entries {
            if entry.kind() != kind || entry.sm() != source {
                entries.push(NewEntry::copy_of(entry));
                continue;
            }
            let payload = entry.payload()?;
            let (text, version) = if kind == KIND_ELF {
                // Every cubin of the selected architecture must carry its complete source.
                // Returning a partially rewritten image would silently hide some entry points.
                let Some((ptx, header)) = embedded_ptx(&payload, source).filter(|_| self.retargetable(source)) else {
                    let lowest_sm = fat
                        .entries
                        .iter()
                        .filter(|e| e.kind() == KIND_ELF)
                        .map(Entry::sm)
                        .min()
                        .unwrap_or(source);
                    return Ok(Adapted::unchanged(Outcome::NeedsTranslation { lowest_sm }));
                };
                (std::borrow::Cow::Owned(ptx), header.version)
            } else {
                let (major, minor) = entry.content_version();
                (payload, (u32::from(major), u32::from(minor)))
            };
            let (text, lowering) = self.lower_ptx(&text);
            tensor_rewritten += lowering.tensor_rewritten;
            tanh_expanded += lowering.tanh_expanded;
            trapped.extend(lowering.trapped);
            refused.extend(lowering.refused);
            emulated.extend(lowering.emulated);
            let (major, minor) = version.min(self.ptx_max);
            let version = (major as u16, minor as u16);
            let target = self.device_sm.min(source);
            entries.push(if kind == KIND_ELF {
                NewEntry::from_ptx(text, target, version, entry)
            } else {
                NewEntry::replace(entry, text, target, version)
            });
            rewritten += 1;
        }
        Ok(Adapted {
            outcome: Outcome::Retargeted {
                entries: rewritten,
                from_sm: source,
                tensor_rewritten,
                tanh_expanded,
                trapped,
                refused,
                emulated,
            },
            image: Some(fatbin::build(&entries)),
        })
    }

    /// The image to load in place of `image`, which this policy made and of which the driver's
    /// compiler refused functions: its `log` names the lines. `image` is PTX text, with or without a
    /// final zero, or a fat binary whose one PTX entry for the device is what was compiled. `None`
    /// when the log names no line of a function, or when what was compiled is not known.
    pub fn without_refused(&self, image: &[u8], log: &str) -> Option<Retry> {
        let text_of =
            |bytes: &[u8]| -> usize { bytes.len() - bytes.iter().rev().take_while(|&&byte| byte == 0).count() };
        if !image.starts_with(&fatbin::MAGIC.to_le_bytes()) {
            let length = text_of(image);
            vf_ptx::header(&image[..length])?;
            let removal = vf_ptx::without_refused(&image[..length], log)?;
            let mut text = removal.text;
            text.extend_from_slice(&image[length..]);
            return Some(Retry {
                image: text,
                functions: removal.functions,
                kernels: removal.kernels,
            });
        }
        let fat = fatbin::parse(image).ok()?;
        let mut compiled = fat
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.kind() == KIND_PTX && self.jit_accepts(entry));
        let (index, entry) = compiled.next()?;
        if compiled.next().is_some() {
            return None;
        }
        let payload = entry.payload().ok()?;
        let removal = vf_ptx::without_refused(&payload[..text_of(&payload)], log)?;
        let mut text = Some(removal.text);
        let entries: Vec<NewEntry> = fat
            .entries
            .iter()
            .enumerate()
            .map(|(at, other)| match text.take_if(|_| at == index) {
                Some(text) => NewEntry::replace(other, text, other.sm(), other.content_version()),
                None => NewEntry::copy_of(other),
            })
            .collect();
        Some(Retry {
            image: fatbin::build(&entries),
            functions: removal.functions,
            kernels: removal.kernels,
        })
    }
}

/// Per-function register counts extracted from a cubin's ELF sections.
pub struct FunctionRegisters {
    pub name: String,
    pub registers: u32,
}

/// Read the register count of each `.text.<func>` section from a cubin.  The count is in the
/// upper 8 bits of the section's `info` field.
pub fn cubin_function_registers(cubin: &[u8]) -> Vec<FunctionRegisters> {
    let Ok(elf) = vf_sass::elf::Elf::parse(cubin) else {
        return Vec::new();
    };
    elf.sections
        .iter()
        .filter_map(|s| {
            s.name.strip_prefix(".text.").map(|name| FunctionRegisters {
                name: name.to_string(),
                registers: s.info >> 24,
            })
        })
        .collect()
}

/// Whether the translated cubin has any function whose register count exceeds its budget.  The
/// budget is `min(max(ceil8(declared), 32), 255)`.  `original` is the cubin before translation;
/// `translated` is the cubin after.  Returns `true` when a second translation within the budget
/// could lower the count of at least one function.
pub fn translation_exceeds_budget(original: &[u8], translated: &[u8]) -> bool {
    let orig = cubin_function_registers(original);
    let trans = cubin_function_registers(translated);
    for t in &trans {
        // Find the function's declared count in the original.
        let declared = orig
            .iter()
            .find(|o| o.name == t.name)
            .map_or(t.registers, |o| o.registers);
        let budget = (declared.div_ceil(8) * 8).clamp(32, 255);
        if t.registers > budget {
            return true;
        }
    }
    false
}

impl Policy {
    /// For a second translation at launch: what to keep of `original` (a cubin or a fat binary
    /// whose translation by this policy was `Outcome::Translated` with `over_budget`): the
    /// cubin, or a fat binary of the one entry that this policy translates, as it is stored
    /// (compressed: PyTorch's are a tenth of their size, and the whole fat binary holds code
    /// for six architectures). Nothing is decompressed here. `kept_cubin` gives the cubin back.
    pub fn kept_for_second(&self, original: &[u8]) -> Option<Vec<u8>> {
        if fatbin::is_elf(original) {
            return Some(original.to_vec());
        }
        let fat = fatbin::parse(original).ok()?;
        let source = self.machine_source(&fat)?;
        let mut entries = fat.entries.iter().filter(|e| e.kind() == KIND_ELF && e.sm() == source);
        let only = entries.next()?;
        // One module handle cannot identify which of several cubins owns a kernel.
        entries
            .next()
            .is_none()
            .then(|| fatbin::build(&[NewEntry::copy_of(only)]))
    }
}

/// The cubin in what `Policy::kept_for_second` gave to keep.
pub fn kept_cubin(kept: &[u8]) -> Option<std::borrow::Cow<'_, [u8]>> {
    if fatbin::is_elf(kept) {
        return Some(std::borrow::Cow::Borrowed(kept));
    }
    let fat = fatbin::parse(kept).ok()?;
    let cubin = fat.entries.first()?.payload().ok()?;
    Some(std::borrow::Cow::Owned(cubin.into_owned()))
}

/// Whether a second translation of a module must NOT be used for this kernel, because the kernel
/// can reach data of the module that code may write (`.nv.global`, `.nv.global.init`). A second
/// translation is loaded as another module: it has a copy of that data, which nothing keeps in step
/// with the first module's. True when, for the kernel or for a function of the cubin that its
/// relocations lead to:
///
/// - a relocation of the function's own sections (its code, its own constant banks) names such
///   data, or anything that is neither a constant section nor a function of the cubin;
/// - the function reads a constant bank that the module's kernels share (`.nv.constant4` holds the
///   addresses of the module's data) in which a relocation names such data. Tables that kernels
///   only read are there too (the compiler puts the tables of its mathematical functions in
///   `.nv.global.init`): they cannot be told from data that a kernel writes, and exclude the
///   kernel all the same;
/// - an instruction of the function cannot be decoded, or the cubin cannot be read.
///
/// On Turing, false for a name that the cubin has no function of. Other architectures require
/// a separate analysis of their constant-bank operands and stay in the original module.
pub fn kernel_names_global_data(cubin: &[u8], kernel_name: &str) -> bool {
    // Constant-bank operands have only been audited for Turing. Keep other kernels in their
    // original module, where writable globals cannot diverge between two translations.
    if fatbin::cubin_sm(cubin) != Some(75) {
        return true;
    }
    let Ok(elf) = vf_sass::elf::Elf::parse(cubin) else {
        return true;
    };
    let Ok(symbols) = elf.symbols() else {
        return true;
    };
    let is_data = |section: usize| {
        elf.sections
            .get(section)
            .is_some_and(|s| s.name == ".nv.global" || s.name == ".nv.global.init")
    };
    // The banks shared by the module's kernels (`.nv.constant<N>`, no function's name after the
    // number) in which a relocation names the module's data.
    let mut tainted = Vec::new();
    for section in &elf.sections {
        let Some(rest) = relocation_target(section) else {
            continue;
        };
        let Some(bank) = rest.strip_prefix(".nv.constant").and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        if relocated_symbols(section).any(|symbol| symbols.get(symbol).is_none_or(|s| is_data(s.shndx as usize))) {
            tainted.push(bank);
        }
    }
    let mut seen = vec![kernel_name.to_string()];
    let mut work = vec![kernel_name.to_string()];
    while let Some(function) = work.pop() {
        let Some(text) = elf.section(&format!(".text.{function}")) else {
            continue;
        };
        // What the relocations of the function's own sections name.
        for section in &elf.sections {
            let Some(rest) = relocation_target(section) else {
                continue;
            };
            let own_bank = rest
                .strip_prefix(".nv.constant")
                .and_then(|n| n.split_once('.'))
                .is_some_and(|(number, of)| number.parse::<u32>().is_ok() && of == function);
            if rest.strip_prefix(".text.") != Some(function.as_str()) && !own_bank {
                continue;
            }
            for symbol in relocated_symbols(section) {
                let Some(target) = symbols.get(symbol).and_then(|s| elf.sections.get(s.shndx as usize)) else {
                    return true;
                };
                if let Some(other) = target.name.strip_prefix(".text.") {
                    if !seen.iter().any(|name| name == other) {
                        seen.push(other.to_string());
                        work.push(other.to_string());
                    }
                } else if !target.name.starts_with(".nv.constant") {
                    return true;
                }
            }
        }
        // The banks its instructions read.
        let code = &elf.sections[text].data;
        if !code.len().is_multiple_of(16) {
            return true;
        }
        let mut decoder = vf_sass::decode::Decoder::new(75);
        for word in code.as_chunks::<16>().0 {
            let word = u128::from_le_bytes(*word);
            let Some(d) = decoder.decode(word) else {
                return true;
            };
            for field in ["Sa_bank", "Sb_bank", "Sc_bank", "bank5"] {
                if d.has_op_name(field) && tainted.contains(&d.op_or_name(field, 0)) {
                    return true;
                }
            }
        }
    }
    false
}

/// For a section of relocations (`SHT_REL`, `SHT_RELA`): the name of the section it relocates.
fn relocation_target(section: &vf_sass::elf::Section) -> Option<&str> {
    match section.stype {
        9 => section.name.strip_prefix(".rel"),
        4 => section.name.strip_prefix(".rela"),
        _ => None,
    }
}

/// The symbols that the entries of a section of relocations name, by their index.
fn relocated_symbols(section: &vf_sass::elf::Section) -> impl Iterator<Item = usize> + '_ {
    let entry = if section.stype == 9 { 16 } else { 24 };
    section
        .data
        .chunks_exact(entry)
        .map(|e| (u64::from_le_bytes(e[8..16].try_into().unwrap_or([0; 8])) >> 32) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ELF75_PTX75: &[u8] = include_bytes!("../../vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin");
    const ELF70_ELF75_PTX70: &[u8] = include_bytes!("../../vf-fatbin/tests/data/elf70-elf75-ptx70-lz4.fatbin");

    #[test]
    fn fat_binary_with_volta_code_is_left_alone() {
        let adapted = Policy::default()
            .adapt_fatbin(&fatbin::parse(ELF70_ELF75_PTX70).unwrap())
            .unwrap();
        assert_eq!(adapted.outcome, Outcome::Native);
        assert!(adapted.image.is_none());
    }

    #[test]
    fn turing_ptx_entry_becomes_volta_ptx() {
        let adapted = Policy::default()
            .adapt_fatbin(&fatbin::parse(ELF75_PTX75).unwrap())
            .unwrap();
        assert_eq!(
            adapted.outcome,
            Outcome::Retargeted {
                entries: 1,
                from_sm: 75,
                tensor_rewritten: 0,
                tanh_expanded: 0,
                trapped: vec![],
                refused: vec![],
                emulated: vec![],
            }
        );
        let image = adapted.image.unwrap();
        let fat = fatbin::parse(&image).unwrap();
        assert_eq!(fat.entries.len(), 2);
        assert_eq!((fat.entries[0].kind(), fat.entries[0].sm()), (KIND_ELF, 75));
        let ptx = &fat.entries[1];
        assert_eq!(
            (ptx.kind(), ptx.sm(), ptx.content_version(), ptx.is_compressed()),
            (KIND_PTX, 70, (9, 0), false)
        );
        let text = ptx.payload().unwrap();
        assert_eq!(
            vf_ptx::header(&text),
            Some(vf_ptx::Header {
                version: (9, 0),
                target: 70
            })
        );
        // Adapting the result again finds it native.
        assert_eq!(Policy::default().adapt_fatbin(&fat).unwrap().outcome, Outcome::Native);
    }

    /// What the translator makes of the seven functions of the fixture's machine code: all are
    /// taken; one has an instruction more (the address of an atomic reduction, which is a
    /// parameter), one three (three accesses to shared memory at an address with a stride).
    fn translated_fixture(entries: usize) -> Outcome {
        let added = |kernel: &str, machine_code| Emulated {
            kernel: kernel.to_string(),
            counts: EmulatedCounts {
                machine_code,
                ..EmulatedCounts::default()
            },
        };
        Outcome::Translated {
            entries,
            from_sm: 75,
            functions: 7,
            refused: vec![],
            emulated: vec![added("warp_sum", 1), added("block_sum", 3)],
            over_budget: false,
        }
    }

    #[test]
    fn machine_code_alone_is_translated() {
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let only_elf = fatbin::build(&[NewEntry::copy_of(&fat.entries[0])]);
        let adapted = Policy::default()
            .adapt_fatbin(&fatbin::parse(&only_elf).unwrap())
            .unwrap();
        assert_eq!(adapted.outcome, translated_fixture(1));
        let image = adapted.image.unwrap();
        let fat = fatbin::parse(&image).unwrap();
        assert_eq!(fat.entries.len(), 1);
        let entry = &fat.entries[0];
        assert_eq!((entry.kind(), entry.sm(), entry.is_compressed()), (KIND_ELF, 70, false));
        assert_eq!(fatbin::cubin_sm(&entry.payload().unwrap()), Some(70));
        // What comes out is for the device: nothing more to do to it.
        assert_eq!(Policy::default().adapt_fatbin(&fat).unwrap().outcome, Outcome::Native);
    }

    #[test]
    fn machine_code_before_ptx_when_asked() {
        // By default the PTX of this fat binary is rewritten (`turing_ptx_entry_becomes_volta_ptx`).
        let policy = Policy {
            translate: Translate::First,
            ..Policy::default()
        };
        let adapted = policy.adapt_fatbin(&fatbin::parse(ELF75_PTX75).unwrap()).unwrap();
        assert_eq!(adapted.outcome, translated_fixture(1));
        let image = adapted.image.unwrap();
        let fat = fatbin::parse(&image).unwrap();
        let kinds: Vec<(u16, u32)> = fat.entries.iter().map(|e| (e.kind(), e.sm())).collect();
        assert_eq!(kinds, [(KIND_ELF, 70), (KIND_PTX, 75)]);
    }

    #[test]
    fn a_cubin_is_translated() {
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let cubin = fat.entries[0].payload().unwrap();
        let policy = Policy::default();
        assert!(policy.translates_cubin(&cubin[..0x40]));
        let adapted = policy.adapt_cubin(&cubin);
        assert_eq!(adapted.outcome, translated_fixture(1));
        let image = adapted.image.unwrap();
        assert_eq!(fatbin::cubin_sm(&image), Some(70));
        assert_eq!(policy.adapt_cubin(&image).outcome, Outcome::Native);
        assert!(!policy.translates_cubin(&image[..0x40]));
        // Its header alone is not enough: said, and nothing loaded in its place.
        let cut = policy.adapt_cubin(&cubin[..0x40]);
        assert!(matches!(cut.outcome, Outcome::NotTranslated { from_sm: 75, .. }));
        assert!(cut.image.is_none());
    }

    #[test]
    fn machine_code_is_left_alone_when_told_so() {
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let only_elf = fatbin::build(&[NewEntry::copy_of(&fat.entries[0])]);
        let policy = Policy {
            translate: Translate::Never,
            ..Policy::default()
        };
        let adapted = policy.adapt_fatbin(&fatbin::parse(&only_elf).unwrap()).unwrap();
        assert_eq!(adapted.outcome, Outcome::NeedsTranslation { lowest_sm: 75 });
        assert!(adapted.image.is_none());
        let cubin = fat.entries[0].payload().unwrap();
        assert_eq!(
            policy.adapt_cubin(&cubin).outcome,
            Outcome::NeedsTranslation { lowest_sm: 75 }
        );
    }

    #[test]
    fn ptx_for_a_later_generation_is_retargeted() {
        let policy = Policy::default();
        // Since the instruction-level check is the real guard, PTX from any later architecture
        // is accepted for retargeting. If it contained sm_90-only instructions, lower_to_volta
        // would replace or trap them.
        let adapted = policy.adapt_ptx(b".version 9.5\n.target sm_90\n");
        assert_eq!(
            adapted.outcome,
            Outcome::Retargeted {
                entries: 1,
                from_sm: 90,
                tensor_rewritten: 0,
                tanh_expanded: 0,
                trapped: vec![],
                refused: vec![],
                emulated: vec![],
            }
        );
        assert_eq!(adapted.image.unwrap(), b".version 9.0\n.target sm_70\n");
    }

    #[test]
    fn cluster_directives_without_later_instructions_refuse_only_their_entry() {
        for directives in [
            ".explicitcluster\n.reqnctapercluster 2, 1, 1\n",
            ".explicitcluster .maxntid 128 .maxclusterrank 4 ",
            ".reqnctapercluster 2, /* rows */ 1, 1 ",
        ] {
            let ptx = format!(
                ".version 8.8\n.target sm_90\n.address_size 64\n\
                 .visible .entry clustered() {directives}{{ ret; }}\n\
                 .visible .entry plain() /* .explicitcluster */ {{ .reg .u32 %vf_tmp; /* .explicitcluster */ ret; }}\n"
            );
            let adapted = Policy::default().adapt_ptx(ptx.as_bytes());
            let Outcome::Retargeted { refused, .. } = adapted.outcome else {
                panic!("expected adapted PTX");
            };
            assert_eq!(refused.len(), 1);
            assert_eq!(refused[0].kernel, "clustered");
            let output = String::from_utf8(adapted.image.unwrap()).unwrap();
            let (cluster, plain) = output.split_once(".entry plain()").unwrap();
            assert!(cluster.contains("trap;"));
            assert!(!cluster.contains(".explicitcluster"));
            assert!(!cluster.contains(".reqnctapercluster"));
            assert!(!cluster.contains(".maxclusterrank"));
            if directives.contains(".maxntid") {
                assert!(cluster.contains(".maxntid 128"));
            }
            assert!(plain.contains(".reg .u32 %vf_tmp; /* .explicitcluster */ ret;"));
        }
    }

    #[test]
    fn ptx_text() {
        let policy = Policy::default();
        let adapted = policy.adapt_ptx(b".version 9.5\n.target sm_75\n.address_size 64\n");
        assert_eq!(
            adapted.outcome,
            Outcome::Retargeted {
                entries: 1,
                from_sm: 75,
                tensor_rewritten: 0,
                tanh_expanded: 0,
                trapped: vec![],
                refused: vec![],
                emulated: vec![],
            }
        );
        assert_eq!(
            adapted.image.unwrap(),
            b".version 9.0\n.target sm_70\n.address_size 64\n"
        );
        assert_eq!(
            policy.adapt_ptx(b".version 8.8\n.target sm_70\n").outcome,
            Outcome::Native
        );
        // An sm_70 target in a dialect the JIT does not know yet: only the version changes.
        let adapted = policy.adapt_ptx(b".version 9.5\n.target sm_70\n");
        assert_eq!(adapted.image.unwrap(), b".version 9.0\n.target sm_70\n");
        assert_eq!(policy.adapt_ptx(b"hello").outcome, Outcome::Unknown);
    }

    #[test]
    fn ptx_with_turing_instructions_is_lowered() {
        let text = b".version 9.5\n.target sm_75\n.entry a()\n{\n\ttanh.approx.f32 %f1, %f2;\n}\n\
                     .entry b()\n{\n\tmovmatrix.sync.aligned.m8n8.trans.b16 %r1, %r2;\n}\n\
                     .entry c()\n{\n\tmma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%r1, %r2}, {%r3}, {%r4}, {%r1, %r2};\n}\n";
        let adapted = Policy::default().adapt_ptx(text);
        let Outcome::Retargeted {
            entries: 1,
            from_sm: 75,
            tensor_rewritten: 1,
            tanh_expanded: 1,
            trapped,
            refused,
            emulated,
        } = &adapted.outcome
        else {
            panic!("{:?}", adapted.outcome);
        };
        // a has a tanh and b a movmatrix, both emulated; c is trapped, and refused as a kernel.
        let names: Vec<&str> = emulated.iter().map(|kernel| kernel.kernel.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(trapped.len(), 1);
        assert_eq!(trapped[0].function, "c");
        assert_eq!(refused.len(), 1);
        assert_eq!((refused[0].kernel.as_str(), refused[0].function.as_str()), ("c", "c"));
        let image = adapted.image.unwrap();
        assert!(vf_ptx::turing_only_features(&image).is_empty());
        assert_eq!(
            vf_ptx::header(&image),
            Some(vf_ptx::Header {
                version: (9, 0),
                target: 70
            })
        );
    }

    /// Two kernels, of which the second holds something the compiler refuses at line 9.
    const TWO_KERNELS: &str = ".version 9.5\n.target sm_75\n.address_size 64\n.entry good()\n{\n\tret;\n}\n.entry bad()\n{\n\tdiscard.global.L2 [0], 128;\n\tret;\n}\n";
    const LINE_10: &str =
        "ptxas application ptx input, line 10; error   : Feature 'discard' requires .target sm_80 or higher\n";

    #[test]
    fn text_without_what_the_compiler_refused() {
        let policy = Policy::default();
        for end in [&b""[..], b"\0"] {
            let mut image = policy.adapt_ptx(TWO_KERNELS.as_bytes()).image.unwrap();
            image.extend_from_slice(end);
            let retry = policy.without_refused(&image, LINE_10).unwrap();
            let text = String::from_utf8(retry.image).unwrap();
            assert!(text.contains(".entry good()\n{\n\tret;\n}\n.entry bad()\n{\n\ttrap;\n\tret;\n}\n"));
            assert_eq!(text.ends_with('\0'), !end.is_empty());
            assert_eq!(retry.functions.len(), 1);
            assert_eq!(retry.kernels.len(), 1);
            assert_eq!(retry.kernels[0].kernel, "bad");
            assert_eq!(
                retry.kernels[0].reason,
                "Feature 'discard' requires .target sm_80 or higher"
            );
            // Nothing the log names: nothing to try.
            assert!(
                policy
                    .without_refused(&image, "ptxas fatal   : Ptx assembly aborted due to errors\n")
                    .is_none()
            );
        }
        assert!(policy.without_refused(b"\x7fELF", LINE_10).is_none());
    }

    #[test]
    fn the_code_taken_is_the_code_for_what_the_program_is_told() {
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let ptx = |sm: u32| {
            let text = format!(".version 9.5\n.target sm_{sm}\n.address_size 64\n");
            NewEntry::replace(&fat.entries[1], text.into_bytes(), sm, (9, 5))
        };
        let program = fatbin::build(&[NewEntry::copy_of(&fat.entries[0]), ptx(90), ptx(75), ptx(80)]);
        let program = fatbin::parse(&program).unwrap();
        let taken = |told_sm: Option<u32>| {
            let policy = Policy {
                told_sm,
                ..Policy::default()
            };
            let adapted = policy.adapt_fatbin(&program).unwrap();
            let Outcome::Retargeted {
                entries: 1, from_sm, ..
            } = adapted.outcome
            else {
                panic!("{:?}", adapted.outcome);
            };
            // One entry became PTX for the device; the others are as they were.
            let image = adapted.image.unwrap();
            let adapted = fatbin::parse(&image).unwrap();
            let mut targets: Vec<u32> = adapted
                .entries
                .iter()
                .filter(|e| e.kind() == KIND_PTX)
                .map(|e| e.sm())
                .collect();
            targets.sort();
            let mut expected: Vec<u32> = [75, 80, 90]
                .iter()
                .map(|&sm| if sm == from_sm { 70 } else { sm })
                .collect();
            expected.sort();
            assert_eq!(targets, expected);
            from_sm
        };
        // Told nothing chosen by the user: the lowest architecture the program has code for.
        assert_eq!(taken(None), 75);
        // Told a capability: what the driver of such a GPU takes, the highest target it reaches.
        assert_eq!(taken(Some(75)), 75);
        assert_eq!(taken(Some(80)), 80);
        assert_eq!(taken(Some(86)), 80);
        assert_eq!(taken(Some(90)), 90);
        assert_eq!(taken(Some(121)), 90);
        // Nothing reaches it: the nearest above.
        assert_eq!(taken(Some(72)), 75);
        assert_eq!(capability_sm("8.6"), Some(86));
        assert_eq!(capability_sm("12.1"), Some(121));
        assert_eq!(capability_sm("8"), None);
        assert_eq!(capability_sm("8.10"), None);
    }

    #[test]
    fn fat_binary_without_what_the_compiler_refused() {
        let policy = Policy::default();
        // The fat binary of the other tests, its PTX replaced by the two kernels, then adapted.
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let program = fatbin::build(&[
            NewEntry::copy_of(&fat.entries[0]),
            NewEntry::replace(&fat.entries[1], TWO_KERNELS.as_bytes().to_vec(), 75, (9, 5)),
        ]);
        let adapted = policy
            .adapt_fatbin(&fatbin::parse(&program).unwrap())
            .unwrap()
            .image
            .unwrap();
        let retry = policy.without_refused(&adapted, LINE_10).unwrap();
        assert_eq!(retry.kernels.len(), 1);
        let again = fatbin::parse(&retry.image).unwrap();
        assert_eq!(again.entries.len(), 2);
        assert_eq!(again.entries[0].stored, fat.entries[0].stored);
        let ptx = &again.entries[1];
        assert_eq!((ptx.kind(), ptx.sm(), ptx.content_version()), (KIND_PTX, 70, (9, 0)));
        let payload = ptx.payload().unwrap();
        let text = std::str::from_utf8(&payload).unwrap().trim_end_matches('\0');
        assert!(text.starts_with(".version 9.0\n.target sm_70\n"));
        assert!(text.ends_with(".entry bad()\n{\n\ttrap;\n\tret;\n}\n"));
        // It is still what the driver would compile.
        assert_eq!(policy.adapt_fatbin(&again).unwrap().outcome, Outcome::Native);
    }

    #[test]
    fn code_embedded_in_a_file() {
        let policy = Policy::default();
        assert_eq!(policy.embedded(b"\x7fELF no fat binary here"), Embedded::Nothing);
        let mut file = b"\x7fELF....".to_vec();
        file.extend_from_slice(ELF75_PTX75);
        file.extend_from_slice(&[0x50, 0xed, 0x55, 0xba, 9, 9]); // the magic number, not a fat binary
        assert_eq!(policy.embedded(&file), Embedded::Other { lowest_sm: 75 });
        file.extend_from_slice(ELF70_ELF75_PTX70);
        assert_eq!(policy.embedded(&file), Embedded::Usable);
    }

    const DEBUG_CUBIN: &[u8] = include_bytes!("../../../tests/kernels/ptx-cases/copy-octal.sm80.cubin");

    #[test]
    fn embedded_ptx_uses_the_same_lowering_in_a_cubin_and_fat_binary() {
        let policy = Policy::default();
        let adapted = policy.adapt_cubin(DEBUG_CUBIN);
        assert!(matches!(&adapted.outcome, Outcome::Retargeted { from_sm: 80, refused, .. } if refused.is_empty()));
        let image = adapted.image.unwrap();
        assert_eq!(vf_ptx::header(&image).unwrap().target, 70);
        let template = fatbin::parse(ELF75_PTX75).unwrap().entries[0];
        let bytes = fatbin::build(&[NewEntry::replace(&template, DEBUG_CUBIN.to_vec(), 80, (1, 7))]);
        let rewritten = policy.adapt_fatbin(&fatbin::parse(&bytes).unwrap()).unwrap();
        assert_eq!(rewritten.outcome, adapted.outcome);
        let bytes = rewritten.image.unwrap();
        let fat = fatbin::parse(&bytes).unwrap();
        assert_eq!(fat.entries.len(), 1);
        let entry = fat.entries[0];
        assert_eq!(
            (entry.kind(), entry.sm(), entry.content_version()),
            (KIND_PTX, 70, (7, 0))
        );
        assert_eq!(&*entry.payload().unwrap(), image.as_slice());
    }

    #[test]
    fn choose_one_machine_target_and_keep_the_same_cubin_for_launch_retries() {
        let turing = fatbin::parse(ELF75_PTX75).unwrap().entries[0];
        let ampere = || NewEntry::replace(&turing, DEBUG_CUBIN.to_vec(), 80, (1, 7));
        let policy = Policy::default();
        let told_ampere = Policy {
            told_sm: Some(80),
            ..policy
        };
        for order in [true, false] {
            let turing_entry = NewEntry::copy_of(&turing);
            let entries = if order {
                [turing_entry, ampere()]
            } else {
                [ampere(), turing_entry]
            };
            let image = fatbin::build(&entries);
            let fat = fatbin::parse(&image).unwrap();
            assert_eq!(policy.machine_source(&fat), Some(75));
            assert_eq!(policy.adapt_fatbin(&fat).unwrap().outcome, translated_fixture(1));
            assert_eq!(told_ampere.machine_source(&fat), Some(75));
            assert!(matches!(
                told_ampere.adapt_fatbin(&fat).unwrap().outcome,
                Outcome::Retargeted { from_sm: 80, .. }
            ));
            let first = Policy {
                translate: Translate::First,
                ..told_ampere
            };
            assert_eq!(first.machine_source(&fat), Some(80));
            assert_eq!(
                kept_cubin(&first.kept_for_second(&image).unwrap()).as_deref(),
                Some(DEBUG_CUBIN)
            );
            assert_eq!(
                kept_cubin(&policy.kept_for_second(&image).unwrap()).as_deref(),
                Some(&*turing.payload().unwrap())
            );
        }
        let mut stripped = vf_sass::elf::Elf::parse(DEBUG_CUBIN).unwrap();
        let section = stripped.section(".nv_debug_ptx_txt").unwrap();
        stripped.sections[section].data.fill(0);
        let mixed = fatbin::build(&[
            NewEntry::replace(&turing, stripped.write(), 80, (1, 7)),
            NewEntry::copy_of(&turing),
        ]);
        assert_eq!(
            told_ampere
                .adapt_fatbin(&fatbin::parse(&mixed).unwrap())
                .unwrap()
                .outcome,
            translated_fixture(1)
        );
        let duplicate = fatbin::build(&[ampere(), ampere()]);
        assert!(told_ampere.kept_for_second(&duplicate).is_none());
        assert!(kernel_names_global_data(DEBUG_CUBIN, "k"));
        let newer_device = Policy {
            device_sm: 90,
            ..policy
        };
        assert_eq!(
            newer_device.adapt_cubin(DEBUG_CUBIN).outcome,
            Outcome::NeedsTranslation { lowest_sm: 80 }
        );
        let native_ampere = Policy {
            device_sm: 80,
            ..policy
        };
        assert_eq!(native_ampere.adapt_cubin(DEBUG_CUBIN).outcome, Outcome::Native);
    }

    #[test]
    fn stripped_ampere_code_keeps_its_per_kernel_refusals_through_the_public_policy() {
        let cubin = include_bytes!("../../../tests/kernels/translate/k_ampere.sm80.cubin");
        let policy = Policy::default();
        let bare = policy.adapt_cubin(cubin);
        assert!(
            matches!(&bare.outcome, Outcome::Translated { from_sm: 80, functions: 2, refused, .. }
            if refused.len() == 1 && refused[0].kernel == "unsupported_atomic")
        );
        let native = fatbin::parse(ELF70_ELF75_PTX70).unwrap().entries[0];
        let ampere = || NewEntry::replace(&native, cubin.to_vec(), 80, (1, 7));
        let single = fatbin::build(&[ampere()]);
        let single = fatbin::parse(&single).unwrap();
        assert_eq!(policy.adapt_fatbin(&single).unwrap().outcome, bare.outcome);
        let never = Policy {
            translate: Translate::Never,
            ..policy
        };
        assert_eq!(
            never.adapt_fatbin(&single).unwrap().outcome,
            Outcome::NeedsTranslation { lowest_sm: 80 }
        );
        assert_eq!(
            never.adapt_cubin(cubin).outcome,
            Outcome::NeedsTranslation { lowest_sm: 80 }
        );
        let mixed = fatbin::build(&[NewEntry::copy_of(&native), ampere()]);
        let mixed = fatbin::parse(&mixed).unwrap();
        let first = Policy {
            translate: Translate::First,
            ..policy
        };
        assert_eq!(first.adapt_fatbin(&mixed).unwrap().outcome, Outcome::Native);
        let force = Policy {
            translate: Translate::Force,
            ..policy
        };
        let forced = force.adapt_fatbin(&mixed).unwrap();
        assert_eq!(forced.outcome, bare.outcome);
        let bytes = forced.image.unwrap();
        let fat = fatbin::parse(&bytes).unwrap();
        assert_eq!(fat.entries.len(), 1);
        assert_eq!(fat.entries[0].sm(), 70);
    }

    #[test]
    fn recover_turing_ptx_only_when_no_translated_entry_can_run() {
        // ptxas predicates the Turing HMMA, but the retained PTX branches around the MMA.
        let cubin = include_bytes!("../../../tests/kernels/ptx-cases/guarded-mma.sm75.cubin");
        let policy = Policy::default();
        let machine = Policy {
            translate: Translate::First,
            ..policy
        };
        assert!(
            matches!(machine.adapt_cubin(cubin).outcome, Outcome::Translated { ref refused, .. } if refused.len() == 1)
        );
        let recovered = policy.adapt_cubin(cubin);
        assert!(recovered.complete_ptx());
        let fixture = fatbin::parse(ELF75_PTX75).unwrap();
        let good = fixture.entries[0];
        let guarded = || NewEntry::replace(&good, cubin.to_vec(), 75, (1, 7));
        let one = fatbin::build(&[guarded()]);
        let one = fatbin::parse(&one).unwrap();
        assert_eq!(policy.adapt_fatbin(&one).unwrap().outcome, recovered.outcome);
        assert_eq!(
            machine.adapt_fatbin(&one).unwrap().outcome,
            machine.adapt_cubin(cubin).outcome
        );
        let mixed = fatbin::build(&[guarded(), NewEntry::copy_of(&good)]);
        assert!(
            matches!(policy.adapt_fatbin(&fatbin::parse(&mixed).unwrap()).unwrap().outcome,
            Outcome::Translated { ref refused, entries: 2, .. } if refused.len() == 1)
        );
        let mut missing = cubin.to_vec();
        let section = b".nv_debug_ptx_txt";
        let offset = missing.windows(section.len()).position(|s| s == section).unwrap();
        missing[offset] = b'_';
        assert_eq!(policy.adapt_cubin(&missing).outcome, machine.adapt_cubin(cubin).outcome);
    }

    #[test]
    fn only_compiler_helpers_inside_an_entry_may_be_weak_symbols() {
        let input = include_bytes!("../../../tests/kernels/ptx-cases/signed-division.sm80.cubin");
        let policy = Policy {
            translate: Translate::Never,
            ..Policy::default()
        };
        assert!(matches!(
            policy.adapt_cubin(input).outcome,
            Outcome::Retargeted { from_sm: 80, .. }
        ));
        for edit in 0..4 {
            let mut elf = vf_sass::elf::Elf::parse(input).unwrap();
            let mut symbols = elf.symbols().unwrap();
            let index = symbols.iter().position(|s| s.name.ends_with("_div_s64")).unwrap();
            let helper = &mut symbols[index];
            assert_eq!(helper.info, 0x22);
            match edit {
                0 => helper.name_offset += 1, // outside ptxas's internal namespace
                1 => helper.value = u64::MAX, // outside its entry's range
                2 => helper.shndx = 0,        // external definition
                _ => {
                    let symtab = elf.section(".symtab").unwrap();
                    elf.sections[symtab].info = index as u32; // public weak symbol
                }
            }
            elf.set_symbols(&symbols);
            assert!(policy.adapt_cubin(&elf.write()).image.is_none());
        }
    }

    #[test]
    fn partial_mismatched_and_relocatable_debug_sources_are_left_alone() {
        let policy = Policy {
            translate: Translate::Never,
            ..Policy::default()
        };
        let edit = |from: &[u8], to: &[u8]| {
            let mut elf = vf_sass::elf::Elf::parse(DEBUG_CUBIN).unwrap();
            let index = elf.section(".nv_debug_ptx_txt").unwrap();
            let text = &mut elf.sections[index].data;
            let at = text.windows(from.len()).position(|w| w == from).unwrap();
            text.splice(at..at + from.len(), to.iter().copied());
            elf.write()
        };
        let mut relocatable = DEBUG_CUBIN.to_vec();
        relocatable[16] = 1;
        let mut with_global = vf_sass::elf::Elf::parse(DEBUG_CUBIN).unwrap();
        let mut symbols = with_global.symbols().unwrap();
        symbols.iter_mut().find(|s| s.name == "source").unwrap().info |= 0x10;
        with_global.set_symbols(&symbols);
        for bad in [
            edit(b".version 7.0", b".version 7.0\0.version 7.0"),
            edit(b".target sm_80", b".target sm_80\0.target sm_80"),
            edit(b".target sm_80", b".target sm_86"),
            edit(b".entry k(", b".entry q("),
            edit(b"ret;", b"\xff"),
            relocatable,
            with_global.write(),
        ] {
            assert_eq!(
                policy.adapt_cubin(&bad).outcome,
                Outcome::NeedsTranslation { lowest_sm: 80 }
            );
            let template = fatbin::parse(ELF75_PTX75).unwrap().entries[0];
            let bytes = fatbin::build(&[
                NewEntry::replace(&template, DEBUG_CUBIN.to_vec(), 80, (1, 7)),
                NewEntry::replace(&template, bad, 80, (1, 7)),
            ]);
            assert!(
                policy
                    .adapt_fatbin(&fatbin::parse(&bytes).unwrap())
                    .unwrap()
                    .image
                    .is_none()
            );
        }
    }

    #[test]
    fn cubins() {
        let policy = Policy::default();
        let fat = fatbin::parse(ELF70_ELF75_PTX70).unwrap();
        assert_eq!(policy.adapt_cubin(fat.entries[0].stored).outcome, Outcome::Native);
        // Machine code for Turing: translated, its seven functions taken.
        let adapted = policy.adapt_cubin(fat.entries[1].stored);
        let Outcome::Translated {
            from_sm: 75,
            functions: 7,
            refused,
            ..
        } = adapted.outcome
        else {
            panic!("{:?}", adapted.outcome);
        };
        assert_eq!(refused, []);
        assert_eq!(fatbin::cubin_sm(&adapted.image.unwrap()), Some(70));
        // The same with an instruction of one function that is none of Turing's: that function is
        // refused with the reason, the six others are translated.
        let mut damaged = fat.entries[1].stored.to_vec();
        let elf = vf_sass::elf::Elf::parse(&damaged).unwrap();
        let at = elf.sections[elf.section(".text.warp_sum").unwrap()].offset as usize;
        damaged[at..at + 16].fill(0xff);
        let adapted = policy.adapt_cubin(&damaged);
        let Outcome::Translated {
            functions: 7, refused, ..
        } = adapted.outcome
        else {
            panic!("{:?}", adapted.outcome);
        };
        assert_eq!(
            refused,
            [Refused {
                kernel: "warp_sum".to_string(),
                function: "warp_sum".to_string(),
                reason: "its machine code for sm_75 is not translated (the instruction \
                         ffffffffffffffffffffffffffffffff at 0x0 is not one of the sm_75 description)"
                    .to_string(),
            }]
        );
        assert_eq!(fatbin::cubin_sm(&adapted.image.unwrap()), Some(70));
        // For an architecture nothing translates.
        let mut later = fat.entries[1].stored.to_vec();
        later[0x30] = 86;
        assert_eq!(fatbin::cubin_sm(&later), Some(86));
        assert_eq!(
            policy.adapt_cubin(&later).outcome,
            Outcome::NeedsTranslation { lowest_sm: 86 }
        );
        assert_eq!(policy.adapt_cubin(b"\x7fELF").outcome, Outcome::Unknown);
    }

    /// The kernels of `tests/kernels/k_fit_test.cu` (the test of the second translation at launch).
    const FIT: &[u8] = include_bytes!("../../../tests/kernels/k_fit_test.135.sm75.cubin");

    #[test]
    fn data_of_the_module_excludes_a_kernel() {
        for kernel in ["k_fit_regs", "k_fit_shared", "k_fit_simple"] {
            assert!(!kernel_names_global_data(FIT, kernel), "{kernel}");
        }
        // These two read the bank that holds the addresses of `fit_scale` and `fit_seen`.
        for kernel in ["k_fit_reads_global", "k_fit_writes_global"] {
            assert!(kernel_names_global_data(FIT, kernel), "{kernel}");
        }
        assert!(!kernel_names_global_data(FIT, "no_such_kernel"));
        assert!(kernel_names_global_data(b"\x7fELF", "k_fit_regs"));
    }

    #[test]
    fn the_usual_translation_passes_the_budget_and_the_second_keeps_to_it() {
        let registers = |image: &[u8], name: &str| {
            cubin_function_registers(image)
                .into_iter()
                .find(|f| f.name == name)
                .map(|f| f.registers)
        };
        // k_fit_regs declares 64 registers: 1,024 threads of it can be launched. Translated the usual
        // way it takes 66, and they cannot.
        let usual = Policy::default().adapt_cubin(FIT).image.unwrap();
        assert_eq!(registers(FIT, "k_fit_regs"), Some(64));
        assert_eq!(registers(&usual, "k_fit_regs"), Some(66));
        assert!(translation_exceeds_budget(FIT, &usual));
        let second = Policy {
            borrow: 5,
            ..Policy::default()
        }
        .adapt_cubin(FIT)
        .image
        .unwrap();
        assert_eq!(registers(&second, "k_fit_regs"), Some(64));
        assert_eq!(registers(&second, "k_fit_reads_global"), Some(64));
        assert!(!translation_exceeds_budget(FIT, &second));
        // The translation says so itself, and what is kept for a second one is the cubin.
        let over = |policy: Policy| match policy.adapt_cubin(FIT).outcome {
            Outcome::Translated { over_budget, .. } => over_budget,
            other => panic!("{other:?}"),
        };
        assert!(over(Policy::default()));
        assert!(!over(Policy {
            borrow: 5,
            ..Policy::default()
        }));
        assert_eq!(Policy::default().kept_for_second(FIT).as_deref(), Some(FIT));
        assert_eq!(kept_cubin(FIT).as_deref(), Some(FIT));
        // From a fat binary whose machine code is for sm_75 (it has PTX too: translated when the
        // policy says machine code first): the one entry, as it is stored, and its cubin back.
        let policy = Policy {
            translate: Translate::First,
            ..Policy::default()
        };
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let adapted = policy.adapt_fatbin(&fat).unwrap();
        // (Its kernels are small: nothing of it is over its budget.)
        assert!(
            matches!(adapted.outcome, Outcome::Translated { over_budget: false, .. }),
            "{:?}",
            adapted.outcome
        );
        let cubin = fat.entries.iter().find(|entry| entry.kind() == KIND_ELF).unwrap();
        let kept = policy.kept_for_second(ELF75_PTX75).unwrap();
        assert!(kept.len() < ELF75_PTX75.len());
        assert_eq!(kept_cubin(&kept).as_deref(), Some(&*cubin.payload().unwrap()));
    }
}
