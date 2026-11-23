//! What the machine code of a kernel can do to memory that outlives it.
//!
//! A library compiled for several architectures holds, for each kernel, the code that its source
//! gives for each architecture, and a source can give nothing for some of them: PyTorch's attention
//! kernels for sm_80 are, in its code for sm_75, a call to `printf` and nothing else, and the program
//! launches them when it is told 8.0. Such code is told apart from a kernel by what it writes: a
//! kernel that writes no memory that outlives it computes nothing for the program, whatever it was
//! asked. The answer is read from the instructions alone, with the instruction set of sm_75: the
//! forms of sm_80 that write memory are decoded by it (they have the encodings of sm_75), and a word
//! it does not decode makes the answer unknown unless a write was found.

use crate::decode::Decoder;
use crate::elf::Elf;
use crate::field;
use crate::translate::find_relative_decoded;

/// What the code of a kernel can do to memory that outlives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// It writes none: no store to global or generic memory, no atomic or reduction, no write to a
    /// surface, and it calls no code but its own and the driver's `vprintf` and `__assertfail`. It
    /// can print, trap or wait; it cannot compute anything for the program.
    None,
    /// It has an instruction that writes such memory.
    Writes,
    /// Neither is shown: a word is not decoded, or it calls code that is not looked at.
    Unknown,
}

/// The instructions that write memory that outlives the kernel (local and shared memory do not).
const WRITES: [&str; 8] = ["STG", "ST", "ATOM", "ATOMG", "RED", "SUST", "SURED", "SUATOM"];

/// The functions of the driver that a kernel may call and still write nothing of the program's.
const HARMLESS: [&str; 2] = ["vprintf", "__assertfail"];

const SHT_RELA: u32 = 4;
const SHT_REL: u32 = 9;
const STT_FUNC: u8 = 2;
/// The mark of an entry point (a kernel) in the `st_other` of its symbol.
const STO_ENTRY: u8 = 0x10;

/// The kernels of `cubin` (machine code for sm_75 or sm_80), each with what its code can do.
pub fn kernel_effects(cubin: &[u8]) -> Result<Vec<(String, Effect)>, String> {
    let elf = Elf::parse(cubin)?;
    let symbols = elf.symbols()?;
    // The places in code and data where an address is relocated, by section: (section, offset,
    // symbol). A function whose address is taken outside the debugging information can be called
    // through a register; when none but the harmless ones is, a call through a register reaches
    // one of these.
    let mut relocations: Vec<(usize, u64, usize)> = Vec::new();
    for section in &elf.sections {
        let size = match section.stype {
            SHT_REL => 16,
            SHT_RELA => 24,
            _ => continue,
        };
        let target = section.info as usize;
        for entry in section.data.chunks_exact(size) {
            let offset = u64::from_le_bytes(entry[..8].try_into().unwrap());
            let info = u64::from_le_bytes(entry[8..16].try_into().unwrap());
            relocations.push((target, offset, (info >> 32) as usize));
        }
    }
    let harmless = |index: usize| {
        symbols
            .get(index)
            .is_some_and(|s| s.shndx == 0 && HARMLESS.contains(&s.name.as_str()))
    };
    let is_function = |index: usize| symbols.get(index).is_some_and(|s| s.info & 0xf == STT_FUNC);
    let debugging = |section: usize| elf.sections.get(section).is_none_or(|s| s.name.starts_with(".debug"));
    let indirect_calls_harmless = relocations
        .iter()
        .all(|&(section, _, symbol)| debugging(section) || !is_function(symbol) || harmless(symbol));

    let mut decoder = Decoder::new(75);
    let mut kernels = Vec::new();
    for symbol in &symbols {
        if symbol.info & 0xf != STT_FUNC || symbol.other & STO_ENTRY == 0 {
            continue;
        }
        let index = symbol.shndx as usize;
        let Some(section) = elf.sections.get(index) else {
            continue;
        };
        let code = &section.data;
        let (instructions, tail) = code.as_chunks::<16>();
        let mut writes = false;
        let mut unknown = !tail.is_empty() || instructions.is_empty();
        for (i, bytes) in instructions.iter().enumerate() {
            let at = 16 * i as u64;
            let word = u128::from_le_bytes(*bytes);
            let Some(d) = decoder.decode(word) else {
                // A form of a later architecture: a write when its opcode is that of writes only.
                let classes = Decoder::opcode_mnemonics_sm75(word);
                if !classes.is_empty() && classes.iter().all(|m| WRITES.contains(m)) {
                    writes = true;
                    break;
                }
                unknown = true;
                continue;
            };
            if WRITES.contains(&d.mnemonic) {
                writes = true;
                break;
            }
            if d.mnemonic != "CALL" {
                continue;
            }
            // A call relocated to another symbol, or to a place outside the kernel's own code, is to
            // code that is not looked at. One through a register reaches a function whose address is
            // taken.
            let relocated = relocations.iter().any(|&(s, offset, _)| s == index && offset == at);
            let inside = find_relative_decoded(&d, 75).map(|(fid, scale)| {
                let target = at as i64 + 16 + field::signed(fid, d.word) * i64::from(scale);
                target >= 0 && target < code.len() as i64
            });
            match inside {
                Some(inside) => unknown |= relocated || !inside,
                None => unknown |= !indirect_calls_harmless,
            }
        }
        let effect = if writes {
            Effect::Writes
        } else if unknown {
            Effect::Unknown
        } else {
            Effect::None
        };
        kernels.push((symbol.name.clone(), effect));
    }
    Ok(kernels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tests/kernels/k_stub.cu`, for sm_75 and sm_80.
    const STUBS: &[u8] = include_bytes!("../../../tests/kernels/k_stub.135.sm75-sm80.fatbin");

    fn cubin(sm: u32) -> Vec<u8> {
        let fat = vf_fatbin::parse(STUBS).unwrap();
        let entry = fat
            .entries
            .iter()
            .find(|e| e.kind() == vf_fatbin::KIND_ELF && e.sm() == sm)
            .unwrap();
        entry.payload().unwrap().into_owned()
    }

    fn effects(sm: u32) -> Vec<(String, Effect)> {
        let mut found = kernel_effects(&cubin(sm)).unwrap();
        found.sort_by(|a, b| a.0.cmp(&b.0));
        found
    }

    #[test]
    fn a_kernel_that_writes_no_memory_is_told_apart_in_each_architecture() {
        let of = |name: &str, effect| (name.to_string(), effect);
        // sm_75: the stub prints, `k_check` asserts, `k_wait` waits; `k_real` stores.
        assert_eq!(
            effects(75),
            [
                of("k_check", Effect::None),
                of("k_real", Effect::Writes),
                of("k_stub", Effect::None),
                of("k_wait", Effect::None),
            ]
        );
        // sm_80, read with the instruction set of sm_75: the stub's real body stores. (Its form of
        // the assertion's call is not one of sm_75: `k_check` is not known to write nothing there.)
        assert_eq!(
            effects(80),
            [
                of("k_check", Effect::Unknown),
                of("k_real", Effect::Writes),
                of("k_stub", Effect::Writes),
                of("k_wait", Effect::None),
            ]
        );
    }

    #[test]
    fn a_call_to_code_that_is_not_looked_at_makes_the_answer_unknown() {
        let mut elf = Elf::parse(&cubin(75)).unwrap();
        let symbols = elf.symbols().unwrap();
        // The address of the kernel `k_real` taken in data: a call through a register can reach it.
        let target = symbols.iter().position(|s| s.name == "k_real").unwrap() as u64;
        let relocation = elf
            .sections
            .iter()
            .position(|s| s.stype == SHT_REL && !s.name.starts_with(".rel.debug"))
            .expect("a section of relocations");
        let mut entry = Vec::new();
        entry.extend_from_slice(&0u64.to_le_bytes());
        entry.extend_from_slice(&((target << 32) | 2).to_le_bytes());
        elf.sections[relocation].data.extend_from_slice(&entry);
        let changed = kernel_effects(&elf.write()).unwrap();
        let effect = |name: &str| changed.iter().find(|(n, _)| n == name).unwrap().1;
        // The stub and the check call `vprintf` and `__assertfail` through a register: now unknown.
        assert_eq!(effect("k_stub"), Effect::Unknown);
        assert_eq!(effect("k_check"), Effect::Unknown);
        // No call: as before.
        assert_eq!(effect("k_wait"), Effect::None);
        assert_eq!(effect("k_real"), Effect::Writes);
    }

    #[test]
    fn what_is_not_a_cubin_is_an_error() {
        assert!(kernel_effects(b"not a cubin").is_err());
    }
}
