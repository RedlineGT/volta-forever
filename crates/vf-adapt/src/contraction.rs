//! Whether the PTX that ptxas retained in a cubin, compiled again, rounds as its machine code does.
//!
//! PTX lets the compiler contract a multiplication and an addition written without a rounding
//! modifier (`mul.f32` then `add.f32`) into one fused multiply-add, which rounds once where the two
//! round twice. Whether ptxas did is an option of the compilation (`--fmad`, true by default), and a
//! debug build (`nvcc -G`) contracts nothing: neither is written in the PTX. The driver's compiler,
//! given the retained PTX again, contracts by default. Compilers that ask for no contraction write
//! the explicit `.rn` forms (nvcc and NVRTC with `-fmad=false`, Triton with fusion off), which no
//! compiler contracts; PTX with the plain forms assembled with `--fmad=false` (an option given to
//! ptxas alone, written by hand or in inline assembly) or for debugging does not say so.
//!
//! The machine code says it. With line information (which a cubin that retains its PTX has, in
//! `.nv_debug_line_sass`), each instruction names the line of the PTX it comes from, and a fused
//! multiply-add is named by the line of the addition. So for each addition or subtraction in the
//! plain form that can take the result of a multiplication in the plain form:
//!
//! - an addition of the machine code from its line, and no fused multiply-add: it was not
//!   contracted, and is given the `.rn` form, which keeps it so;
//! - a fused multiply-add from its line, and no addition: it was, and stays as it is (the PTX allows
//!   it, and the driver's compiler contracts it as ptxas did);
//! - anything else (no line information, both kinds, neither, another statement on its line): what
//!   ptxas made of it is not known, and the retained PTX is not used.

use std::collections::{HashMap, HashSet};

use vf_ptx::scan::{self, Kind};

/// Floating-point types whose multiplication and addition contract into an instruction that Volta
/// has (FFMA, DFMA, HFMA2).
const TYPES: &[&str] = &["f32", "f64", "f16", "f16x2"];
const ROUNDING: &[&str] = &["rn", "rz", "rm", "rp"];

/// What the machine code made of a line of the retained PTX.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Made {
    /// An addition (FADD, DADD, HADD2).
    pub add: bool,
    /// A fused multiply-add (FFMA, DFMA, HFMA2).
    pub fused: bool,
}

/// An addition or a subtraction in the plain form that can take the result of a multiplication in
/// the plain form: where its opcode ends its operation (`add`, `sub`), and its line (from 1).
struct Candidate {
    after_operation: usize,
    line: u32,
    /// No other statement starts on its line, and it ends there.
    alone: bool,
}

/// `text`, the PTX retained in `cubin`, with the additions that its machine code did not contract
/// given the `.rn` form; `None` when what was contracted is not known (see the module).
pub(crate) fn as_compiled(cubin: &[u8], mut text: Vec<u8>) -> Option<Vec<u8>> {
    let candidates = candidates(&text)?;
    if candidates.is_empty() {
        return Some(text);
    }
    let made = machine_lines(cubin)?;
    let mut separate = Vec::new();
    for candidate in &candidates {
        if !candidate.alone {
            return None;
        }
        match made.get(&candidate.line).copied().unwrap_or_default() {
            Made {
                add: true,
                fused: false,
            } => separate.push(candidate.after_operation),
            Made {
                add: false,
                fused: true,
            } => {}
            _ => return None,
        }
    }
    separate.sort_unstable();
    for &at in separate.iter().rev() {
        text.splice(at..at, b".rn".iter().copied());
    }
    Some(text)
}

/// The words of an opcode in the plain form of one of `TYPES` (`add.ftz.f32`: `add`).
fn plain(opcode: &str) -> Option<&str> {
    let words: Vec<&str> = opcode.split('.').collect();
    let (&operation, &kind) = (words.first()?, words.last()?);
    (words.len() > 1 && TYPES.contains(&kind) && !words.iter().any(|w| ROUNDING.contains(w))).then_some(operation)
}

/// The candidates of `text` (`Candidate`), function by function. A register is taken to hold a
/// product wherever a multiplication in the plain form writes it, or a move or a negation copies
/// one that does: more additions are looked at than can be contracted, none fewer. `None` when the
/// body of a function does not divide into statements.
fn candidates(text: &[u8]) -> Option<Vec<Candidate>> {
    let starts: Vec<usize> = std::iter::once(0)
        .chain(
            text.iter()
                .enumerate()
                .filter(|&(_, &b)| b == b'\n')
                .map(|(at, _)| at + 1),
        )
        .collect();
    let line_of = |at: usize| starts.partition_point(|&start| start <= at) as u32;
    let mut found = Vec::new();
    for function in vf_ptx::functions(text) {
        let items = scan::items(text, function.body)?;
        let statements: Vec<(usize, usize, &str, Vec<&str>)> = items
            .iter()
            .filter_map(|item| match &item.kind {
                Kind::Statement { opcode, operands, .. } => Some((
                    item.range.start,
                    opcode.start,
                    std::str::from_utf8(&text[opcode.clone()]).ok()?,
                    scan::split_operands(std::str::from_utf8(&text[operands.clone()]).ok()?),
                )),
                _ => None,
            })
            .collect();
        let mut products: HashSet<&str> = statements
            .iter()
            .filter(|(_, _, opcode, _)| plain(opcode) == Some("mul"))
            .filter_map(|(_, _, _, operands)| scan::register(operands.first()?))
            .collect();
        loop {
            let before = products.len();
            for (_, _, opcode, operands) in &statements {
                let operation = opcode.split('.').next().unwrap_or_default();
                if let ("mov" | "neg", [destination, source]) = (operation, operands.as_slice())
                    && scan::register(source).is_some_and(|source| products.contains(source))
                    && let Some(destination) = scan::register(destination)
                {
                    products.insert(destination);
                }
            }
            if products.len() == before {
                break;
            }
        }
        let mut lines: HashMap<u32, usize> = HashMap::new();
        for item in &items {
            if matches!(item.kind, Kind::Statement { .. }) {
                *lines.entry(line_of(item.range.start)).or_default() += 1;
            }
        }
        for item in &items {
            let Kind::Statement { opcode, operands, .. } = &item.kind else {
                continue;
            };
            let Some(operation @ ("add" | "sub")) = std::str::from_utf8(&text[opcode.clone()]).ok().and_then(plain)
            else {
                continue;
            };
            let operands = scan::split_operands(std::str::from_utf8(&text[operands.clone()]).ok()?);
            if !operands
                .iter()
                .skip(1)
                .filter_map(|operand| scan::register(operand))
                .any(|source| products.contains(source))
            {
                continue;
            }
            let line = line_of(item.range.start);
            found.push(Candidate {
                after_operation: opcode.start + operation.len(),
                line,
                alone: lines.get(&line) == Some(&1) && line_of(item.range.end - 1) == line,
            });
        }
    }
    Some(found)
}

/// What the machine code of `cubin` made of each line of its retained PTX, by the line table that
/// ptxas writes with it (`.nv_debug_line_sass`, a DWARF line program whose addresses are those of
/// the instructions in the section of their function, given by relocations). `None` when the cubin
/// has no such table, or it cannot be read whole.
pub(crate) fn machine_lines(cubin: &[u8]) -> Option<HashMap<u32, Made>> {
    let elf = vf_sass::elf::Elf::parse(cubin).ok()?;
    let table = elf.section(".nv_debug_line_sass")?;
    let symbols = elf.symbols().ok()?;
    // By the offset of an address in the table: the symbol it is relative to, and the addend of
    // a relocation that has one (`SHT_RELA`).
    let mut relocated: HashMap<usize, (usize, Option<u64>)> = HashMap::new();
    for section in elf.sections.iter().filter(|s| s.info as usize == table) {
        let (size, addend) = match section.stype {
            9 => (16, false),
            4 => (24, true),
            _ => continue,
        };
        for entry in section.data.chunks_exact(size) {
            let word = |at: usize| u64::from_le_bytes(entry[at..at + 8].try_into().unwrap_or([0; 8]));
            relocated.insert(
                usize::try_from(word(0)).ok()?,
                ((word(8) >> 32) as usize, addend.then(|| word(16))),
            );
        }
    }
    let data = &elf.sections[table].data;
    let mut made: HashMap<u32, Made> = HashMap::new();
    // The rows of a sequence: (address, line), in the section of its function.
    let mut note = |section: usize, rows: &[(u64, u32)], end: u64| -> Option<()> {
        let code = &elf.sections.get(section)?.data;
        for (at, &(address, line)) in rows.iter().enumerate() {
            let stop = rows.get(at + 1).map_or(end, |next| next.0);
            if line == 0 || stop < address {
                continue;
            }
            let mut offset = usize::try_from(address).ok()?;
            while (offset as u64) < stop {
                let word = code.get(offset..offset.checked_add(8)?)?;
                let low = u64::from_le_bytes(word.try_into().ok()?);
                let entry = made.entry(line).or_default();
                match low & 0x1ff {
                    0x021 | 0x029 | 0x030 => entry.add = true,
                    0x023 | 0x02b | 0x031 | 0x035 => entry.fused = true,
                    _ => {}
                }
                offset = offset.checked_add(16)?;
            }
        }
        Some(())
    };
    let byte = |at: usize| data.get(at).copied();
    let u16_at = |at: usize| Some(u16::from_le_bytes(data.get(at..at + 2)?.try_into().ok()?));
    let u32_at = |at: usize| Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?));
    let u64_at = |at: usize| Some(u64::from_le_bytes(data.get(at..at + 8)?.try_into().ok()?));
    let leb = |at: &mut usize, signed: bool| -> Option<i64> {
        let (mut value, mut shift) = (0u64, 0u32);
        loop {
            let b = byte(*at)?;
            *at += 1;
            if shift >= 64 {
                return None;
            }
            value |= u64::from(b & 0x7f) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                if signed && shift < 64 && b & 0x40 != 0 {
                    value |= u64::MAX << shift;
                }
                return Some(value as i64);
            }
        }
    };
    let mut unit = 0;
    while unit < data.len() {
        let length = u32_at(unit)? as usize;
        // (0xffffffff: the 64-bit format, which ptxas does not write.)
        let end = unit
            .checked_add(4)?
            .checked_add(length)
            .filter(|&end| length < 0xffff_fff0 && end <= data.len())?;
        let version = u16_at(unit + 4)?;
        if !(2..=4).contains(&version) {
            return None;
        }
        let program = (unit + 10).checked_add(u32_at(unit + 6)? as usize)?;
        let mut at = unit + 10;
        let minimum = u64::from(byte(at)?);
        at += if version >= 4 { 3 } else { 2 };
        let line_base = byte(at)? as i8 as i64;
        let line_range = u64::from(byte(at + 1)?);
        let opcode_base = byte(at + 2)?;
        let lengths = data.get(at + 3..(at + 3).checked_add(usize::from(opcode_base).checked_sub(1)?)?)?;
        if line_range == 0 || program > end {
            return None;
        }
        at = program;
        let (mut address, mut line, mut section) = (0u64, 1i64, None::<usize>);
        let mut rows: Vec<(u64, u32)> = Vec::new();
        let row = |address: u64, line: i64, rows: &mut Vec<(u64, u32)>| -> Option<()> {
            rows.push((address, u32::try_from(line).ok()?));
            Some(())
        };
        while at < end {
            let opcode = byte(at)?;
            at += 1;
            if opcode >= opcode_base {
                let adjusted = u64::from(opcode - opcode_base);
                address = address.checked_add(adjusted / line_range * minimum)?;
                line = line.checked_add(line_base + (adjusted % line_range) as i64)?;
                row(address, line, &mut rows)?;
                continue;
            }
            match opcode {
                0 => {
                    let size = usize::try_from(leb(&mut at, false)?).ok()?;
                    let next = at.checked_add(size).filter(|&next| next <= end)?;
                    match byte(at)? {
                        // End of a sequence: its rows are noted, in the section of its function.
                        1 => {
                            note(section?, &rows, address)?;
                            rows.clear();
                            (address, line, section) = (0, 1, None);
                        }
                        // The address of the next row: relative to a function's symbol.
                        2 if size == 9 => {
                            let (symbol, addend) = relocated.get(&(at + 1))?;
                            let symbol = symbols.get(*symbol)?;
                            if !rows.is_empty() {
                                return None;
                            }
                            section = Some(usize::from(symbol.shndx));
                            address = addend.unwrap_or(u64_at(at + 1)?).checked_add(symbol.value)?;
                        }
                        2 => return None,
                        _ => {}
                    }
                    at = next;
                }
                1 => row(address, line, &mut rows)?,
                2 => address = address.checked_add(u64::try_from(leb(&mut at, false)?).ok()?.checked_mul(minimum)?)?,
                3 => line = line.checked_add(leb(&mut at, true)?)?,
                8 => address = address.checked_add((255 - u64::from(opcode_base)) / line_range * minimum)?,
                9 => {
                    address = address.checked_add(u64::from(u16_at(at)?))?;
                    at += 2;
                }
                _ => {
                    // Another standard opcode: skip its operands, as many as the header says.
                    for _ in 0..*lengths.get(usize::from(opcode) - 1)? {
                        leb(&mut at, false)?;
                    }
                }
            }
        }
        if !rows.is_empty() {
            return None;
        }
        unit = end;
    }
    Some(made)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEPARATE: &[u8] = include_bytes!("../../../tests/kernels/ptx-cases/product-sum.fmad-false.sm80.cubin");
    const FUSED: &[u8] = include_bytes!("../../../tests/kernels/ptx-cases/product-sum.fmad-true.sm80.cubin");

    #[test]
    fn the_line_table_names_what_each_line_became() {
        // product-sum.ptx: the multiplication is on line 19, the addition on line 20.
        let separate = machine_lines(SEPARATE).unwrap();
        assert_eq!(
            separate[&20],
            Made {
                add: true,
                fused: false
            }
        );
        assert_eq!(separate[&19], Made::default());
        let fused = machine_lines(FUSED).unwrap();
        assert_eq!(
            fused[&20],
            Made {
                add: false,
                fused: true
            }
        );
        assert!(!fused.contains_key(&19) || fused[&19] == Made::default());
        assert!(machine_lines(b"\x7fELF").is_none());
    }

    #[test]
    fn additions_that_can_take_a_product_are_found() {
        let text = b".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry k() {\n\
            .reg .f32 %f<9>;\n\
            mul.f32 %f1, %f2, %f3;\n\
            mov.f32 %f4, %f1;\n\
            neg.f32 %f5, %f4;\n\
            add.f32 %f6, %f5, %f7;\n\
            add.rn.f32 %f6, %f1, %f7;\n\
            add.f32 %f8, %f7, %f7;\n\
            mul.rn.f32 %f2, %f7, %f7;\n\
            @%p1 sub.ftz.f32 %f8, %f7, %f1; add.f32 %f8, %f8, %f1;\n\
            add.u32 %r1, %r1, 1;\n\
            ret;\n}\n";
        let found = candidates(text).unwrap();
        let summary: Vec<(u32, bool, &[u8])> = found
            .iter()
            .map(|c| (c.line, c.alone, &text[c.after_operation - 3..c.after_operation]))
            .collect();
        assert_eq!(
            summary,
            [(9, true, &b"add"[..]), (13, false, b"sub"), (13, false, b"add")]
        );
    }
}
