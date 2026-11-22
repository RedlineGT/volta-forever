//! The sm_80 instruction forms that can enter the Turing-to-Volta translator.
//! The table fixes all bits except explicitly identified operand fields. The forms with
//! different encodings on Ampere and Turing are decoded before testing shared forms.
use crate::generated::ampere::{LOAD16, PACK, SHARED_FORMS, STORE16, STORE32};

use crate::assemble::{
    OpVal::{Int, Str},
    assemble,
};
use crate::decode::{Decoded, Decoder, LETTER_U};
use crate::field;
use crate::generated::fields::*;
use crate::generated::special::{READ_BARRIER, USCHED, WAIT, WRITE_BARRIER};

use super::util::find_relative;
use super::{ARRIVES, Item, Refused};

const CONTROLS: u128 = 0x1ffff << USCHED;
const BYTE: u128 = 0xff;
fn matches_form(bits: u128, (fixed, variable): (u128, u128)) -> bool {
    bits & !variable == fixed
}

pub(super) fn source(header: &[u8]) -> Result<u8, String> {
    let arch = if header[8] >= 8 { header[0x31] } else { header[0x30] };
    if arch == 75 {
        return Ok(75);
    }
    if arch != 80 {
        return Err(format!("unsupported machine-code target sm_{arch}"));
    }
    let abi = (header[7], header[8]);
    if !matches!(abi, (0x33, 7) | (0x41, 8)) || header[16..20] != [2, 0, 190, 0] {
        return Err("sm_80 translation requires an executable CUDA ELF with a known ABI".into());
    }
    Ok(80)
}

struct Step {
    killed: u64,
    initialized: u64,
    uses: Option<u8>,
    branch: Option<usize>,
    falls_through: bool,
}

/// A binary has no textual descriptor on sm_80's default LDG/STG form. Even when
/// the uniform register number is known, its contents must be the system descriptor
/// on every path leading here before an ordinary Volta global pointer can replace it.
pub(super) fn normalize(code: &[u8], decoder: &mut Decoder) -> Result<Vec<u8>, Refused> {
    let (instructions, tail) = code.as_chunks::<16>();
    if instructions.is_empty() || !tail.is_empty() {
        return Err(Refused("sm_80 code is empty or contains a partial instruction".into()));
    }
    let mut result = Vec::with_capacity(code.len());
    let mut joins = [None; 16];
    let mut syncs = Vec::new();
    let mut steps = Vec::with_capacity(instructions.len());
    for (i, &chunk) in instructions.iter().enumerate() {
        let w = u128::from_le_bytes(chunk);
        let at = i * 16;
        let control = w & CONTROLS;
        if (w >> WRITE_BARRIER) & 7 == 6 || (w >> READ_BARRIER) & 7 == 6 {
            return Err(Refused(format!("reserved sm_80 scoreboard at 0x{at:x}")));
        }
        let bits = w & !CONTROLS;
        let addr = (w >> 24) & BYTE;
        let rd = (w >> 16) & BYTE;
        let rb = (w >> 32) & BYTE;
        let (word, descriptor) = if matches_form(bits, LOAD16) {
            let load = assemble(
                "ldg__sImmOffset",
                &[
                    ("e", Str("E")),
                    ("sz", Str("U16")),
                    ("sem", Str("CONSTANT")),
                    ("sco", Str("SYS")),
                    ("cop", Str("EN")),
                    ("Rd", Int(rd as u32)),
                    ("Ra", Int(addr as u32)),
                    ("Pu", Int(7)),
                ],
            );
            (load, Some((w >> 32) as u8 & 63))
        } else if matches_form(bits, STORE16) || matches_form(bits, STORE32) {
            let size = if matches_form(bits, STORE16) { "U16" } else { "_32" };
            let store = assemble(
                "stg__sImmOffset",
                &[
                    ("e", Str("E")),
                    ("sz", Str(size)),
                    ("sem", Str("WEAK")),
                    ("sco", Str("SYS")),
                    ("cop", Str("EN")),
                    ("Rb", Int(rb as u32)),
                    ("Ra", Int(addr as u32)),
                ],
            );
            (store, Some((w >> 64) as u8 & 63))
        } else if matches_form(bits, PACK) {
            // Let ordinary register allocation see the input and the destination of the conversion.
            let convert = assemble(
                "f2f_f32_downconvert__RRR_RRR",
                &[
                    ("dstfmt.srcfmt", Str("F16.F32")),
                    ("Rd", Int(rd as u32)),
                    ("Rb", Int(rb as u32)),
                ],
            );
            (convert, None)
        } else if SHARED_FORMS.iter().any(|&form| matches_form(bits, form)) {
            (Some(bits), None)
        } else {
            return Err(Refused(format!("unsupported sm_80 instruction {w:032x} at 0x{at:x}")));
        };
        if descriptor.is_some_and(|q| q > 60 || !q.is_multiple_of(2) || addr > 252 || !addr.is_multiple_of(2)) {
            return Err(Refused(format!("unsupported sm_80 address register pair at 0x{at:x}")));
        }
        let word = word.ok_or_else(|| Refused(format!("cannot encode sm_80 instruction for Volta at 0x{at:x}")))?;
        let word = (word & !CONTROLS) | control;
        let d = decoder
            .decode(word)
            .ok_or_else(|| Refused(format!("no decoded instruction for the sm_80 form at 0x{at:x}")))?;
        let mut killed = 0;
        for &(space, reg) in &d.writes {
            if space == LETTER_U && reg < 63 {
                killed |= 1u64 << (reg & !1);
            }
        }
        let q = d.op_or(OP_URD, 63);
        let initialized = if d.name == "uldc_const__RCR"
            && q <= 60
            && q.is_multiple_of(2)
            && d.op_or(OP_SZ, 0) == 5
            && d.op_or(OP_UPG, 7) == 7
            && d.op_or(OP_UPG_AT_NOT, 0) == 0
            && d.op_or(OP_SA_BANK, 1) == 0
            && d.op_or(OP_SA_ADDR, 0) == 0x118
        {
            1u64 << q
        } else {
            0
        };
        let branch = if let Some((fid, scale)) = find_relative(&d, 75) {
            let target = at as i64 + 16 + field::signed(fid, d.word) * i64::from(scale);
            if target < 0 || target >= code.len() as i64 || target % 16 != 0 {
                return Err(Refused(format!("sm_80 branch at 0x{at:x} leaves its function")));
            }
            Some(target as usize / 16)
        } else {
            None
        };
        // BSYNC can continue at its barrier's reconvergence PC. The supported regions
        // place that PC directly after BSYNC, so ordinary fallthrough covers that edge.
        if matches!(d.mnemonic, "BSSY" | "BSYNC") {
            let barrier = d.op_or_name("barReg", 16) as usize;
            let join = joins
                .get_mut(barrier)
                .ok_or_else(|| Refused(format!("invalid sm_80 convergence barrier at 0x{at:x}")))?;
            if d.mnemonic == "BSYNC" {
                syncs.push((i, barrier));
            } else if join.is_some() && *join != branch {
                return Err(Refused(format!("multiple sm_80 convergence targets for B{barrier}")));
            } else {
                *join = branch;
            }
        }
        let conditional = d.op_or(OP_PG, 7) != 7
            || d.op_or(OP_PG_AT_NOT, 0) != 0
            || d.op_or(OP_PP, 7) != 7
            || d.op_or(OP_PP_AT_NOT, 0) != 0
            || d.op_or(OP_COND, 0) != 0;
        steps.push(Step {
            killed,
            initialized,
            uses: descriptor,
            branch,
            falls_through: conditional || !matches!(d.mnemonic, "BRA" | "EXIT"),
        });
        result.extend_from_slice(&word.to_le_bytes());
    }
    for (i, barrier) in syncs {
        if joins[barrier] != Some(i + 1) {
            return Err(Refused(format!(
                "sm_80 BSYNC B{barrier} at 0x{:x} does not fall through to its convergence target",
                i * 16
            )));
        }
    }
    let mut state = vec![None; steps.len()];
    state[0] = Some(0u64);
    let mut pending = vec![0];
    while let Some(i) = pending.pop() {
        let s = &steps[i];
        let out = (state[i].unwrap() & !s.killed) | s.initialized;
        let fall = (s.falls_through && i + 1 < steps.len()).then_some(i + 1);
        for next in [s.branch, fall].into_iter().flatten() {
            let joined = state[next].map_or(out, |old| old & out);
            if state[next] != Some(joined) {
                state[next] = Some(joined);
                pending.push(next);
            }
        }
    }
    for (i, s) in steps.iter().enumerate() {
        if let Some(q) = s.uses
            && state[i].unwrap_or(0) & (1u64 << q) == 0
        {
            return Err(Refused(format!(
                "sm_80 global descriptor UR{q} is not established at 0x{:x}",
                i * 16
            )));
        }
    }
    Ok(result)
}

pub(super) fn pack(i: usize, d: &Decoded) -> Result<Vec<Item>, Refused> {
    let rd = Int(d.op_or(OP_RD, 255));
    // F2F converts the low half. PRMT selects its two low bytes and two bytes
    // from RZ; an ordinary 32-bit consumer sees exactly PACK_AB RZ, Rb.
    let convert = (d.word & !(7 << WRITE_BARRIER)) | (u128::from(ARRIVES) << WRITE_BARRIER) | (63 << WAIT);
    let mut first = Item::new(convert, i, "sm80: convert low half", true);
    first.then = Some(ARRIVES);
    let permute = assemble(
        "prmt__RuIR_RIR",
        &[("Rd", rd.clone()), ("Ra", rd), ("Sb", Int(0x5410)), ("Rc", Int(255))],
    )
    .ok_or_else(|| Refused("cannot assemble PRMT for a packed sm_80 conversion".into()))?;
    Ok(vec![first, Item::new(permute, i, "sm80: zero high half", true)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::Elf;
    use crate::translate::translate;

    const IMAGE: &[u8] = include_bytes!("../../../../tests/kernels/translate/k_ampere.sm80.cubin");

    #[test]
    fn translates_executable_only_and_preserves_individual_refusals() {
        let result = translate(IMAGE, true).unwrap();
        let accepted = result.report.iter().find(|r| r.name == "memory_check").unwrap();
        let refused = result.report.iter().find(|r| r.name == "unsupported_atomic").unwrap();
        assert_eq!(accepted.refused, None);
        assert!(refused.refused.as_deref().unwrap().contains("unsupported sm_80"));
        assert!(result.map.unwrap().iter().any(|r| r.3 == "sm80: zero high half"));
        for (offset, value) in [(16, 1), (16, 3), (18, 62), (7, 0), (8, 6), (0x30, 86)] {
            let mut wrong = IMAGE.to_vec();
            wrong[offset] = value;
            assert!(translate(&wrong, false).is_err());
        }
    }

    #[test]
    fn reschedules_shared_ampere_instructions_and_preserves_the_packed_word() {
        // Both producers are also legal Volta instructions. Their original stalls are too short.
        let executable = translate(IMAGE, false).unwrap().bytes;
        let source = Elf::parse(IMAGE).unwrap();
        let target = Elf::parse(&executable).unwrap();
        let src = &source.sections[source.section(".text.memory_check").unwrap()].data;
        let dst = &target.sections[target.section(".text.memory_check").unwrap()].data;
        let mut decoder = Decoder::new(70);
        let ops: Vec<_> = dst
            .as_chunks::<16>()
            .0
            .iter()
            .map(|&w| decoder.decode(u128::from_le_bytes(w)).unwrap())
            .collect();
        for (at, producer, consumer, old) in [(0x60, "IMAD", "LDG", 6), (0x90, "HADD2", "FADD", 5)] {
            let word = u128::from_le_bytes(src[at..at + 16].try_into().unwrap());
            assert_eq!(super::super::stall_of(word), old);
            let pair = ops
                .windows(2)
                .find(|a| a[0].mnemonic == producer && a[1].mnemonic == consumer)
                .unwrap();
            assert_eq!(super::super::stall_of(pair[0].word), 8);
        }
        let pair = ops
            .windows(2)
            .find(|w| w[0].mnemonic == "F2F" && w[1].mnemonic == "PRMT")
            .unwrap();
        assert_eq!(pair[0].op_or(OP_RD, 255), pair[1].op_or(OP_RA, 255));
        assert_eq!(pair[1].op_or(OP_RD, 255), pair[1].op_or(OP_RA, 255));
        assert_eq!(pair[1].op_or(OP_RC, 0), 255);
        assert_eq!(pair[1].op_or(OP_SB, 0), 0x5410);
        let sb = (pair[0].word >> WRITE_BARRIER) & 7;
        assert!(sb < 6 && ((pair[1].word >> WAIT) & (1 << sb)) != 0);
    }

    #[test]
    fn descriptor_must_be_canonical_along_every_path() {
        let e = Elf::parse(IMAGE).unwrap();
        let body = &e.sections[e.section(".text.memory_check").unwrap()].data;
        // The fixture loads UR4:UR5 at 0x30, then accesses through that descriptor at 0x70.
        let mut code = body.clone();
        let put = |code: &mut [u8], at: usize, word: u128| code[at..at + 16].copy_from_slice(&word.to_le_bytes());
        let word = |at: usize| u128::from_le_bytes(body[at..at + 16].try_into().unwrap());
        let check = |code: &[u8], why: &str| {
            let refusal = normalize(code, &mut Decoder::new(75)).unwrap_err();
            assert!(refusal.0.contains(why), "{}", refusal.0);
        };
        assert!(normalize(body, &mut Decoder::new(75)).is_ok());
        // The immediate in FADD is an operand. A different immediate retains its full encoding.
        let fadd = (word(0xa0) & !(u128::from(u32::MAX) << 32)) | (u128::from((-123.5f32).to_bits()) << 32);
        put(&mut code, 0xa0, fadd);
        let normalized = normalize(&code, &mut Decoder::new(75)).unwrap();
        assert_eq!(&normalized[0xa0..0xb0], &fadd.to_le_bytes());
        code.copy_from_slice(body);
        put(&mut code, 0x30, word(0x30) ^ (8 << 38));
        check(&code, "descriptor UR4 is not established");
        // Conditional BRA at 0x20 skips the descriptor load on one edge.
        let branch = 0x00000000038000000000001000000947 | (word(0x20) & CONTROLS);
        code.copy_from_slice(body);
        put(&mut code, 0x20, branch);
        check(&code, "descriptor UR4 is not established");
        for r in [4u128, 5] {
            // UMOV URr, 1, clobbering one half of the established descriptor.
            code.copy_from_slice(body);
            put(&mut code, 0x40, 0x000fe20000000000_0000000100007882 | r << 16);
            check(&code, "descriptor UR4 is not established");
        }
        // Verify the BSSY/BSYNC edge used for descriptor dataflow.
        code.copy_from_slice(body);
        let bssy = 0x000fe60003800000_0000000000007945;
        let bsync = 0x000fea0003800000_0000000000007941;
        put(&mut code, 0x10, bssy | (0x10 << 32)); // BSSY B0, 0x30
        put(&mut code, 0x20, bsync);
        assert!(normalize(&code, &mut Decoder::new(75)).is_ok());
        put(&mut code, 0x10, bssy | (0x30 << 32)); // BSYNC would jump past the ULDC.
        check(&code, "does not fall through to its convergence target");

        code.copy_from_slice(body);
        put(&mut code, 0x70, word(0x70) ^ (1 << 101));
        check(&code, "unsupported sm_80 instruction");
    }
}
