//! The sm_80 instruction forms that can enter the Turing-to-Volta translator.
//! The table fixes all bits except explicitly identified operand fields. The forms with
//! different encodings on Ampere and Turing are decoded before testing shared forms.
use std::collections::HashMap;

use crate::generated::ampere::{
    COMMIT, COPY128, HALF_CONSTANT, LOAD16, LOAD128, PACK, SHARED_FORMS, STORE16, STORE32, STORE128,
};

use crate::assemble::{
    OpVal::{Int, Str},
    assemble,
};
use crate::decode::{Decoded, Decoder, Kind, LETTER_R, LETTER_U};
use crate::field;
use crate::generated::fields::*;
use crate::generated::special::{READ_BARRIER, USCHED, WAIT, WRITE_BARRIER};

use super::context::Ctx;
use super::util::find_relative;
use super::{ARRIVES, Item, Refused, SEQUENCE};

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

#[derive(Debug)]
pub(super) enum Operation {
    Pack,
    GlobalToShared { global_offset: i32, shared_offset: i32 },
}

#[derive(Debug)]
pub(super) struct Normalized {
    pub words: Vec<u8>,
    pub operations: HashMap<usize, Operation>,
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
pub(super) fn normalize(code: &[u8], decoder: &mut Decoder, helper: Option<usize>) -> Result<Normalized, Refused> {
    let (instructions, tail) = code.as_chunks::<16>();
    if instructions.is_empty() || !tail.is_empty() {
        return Err(Refused("sm_80 code is empty or contains a partial instruction".into()));
    }
    let mut result = Vec::with_capacity(code.len());
    let mut operations = HashMap::new();
    let mut joins = [None; 16];
    let mut syncs = Vec::new();
    let mut returns = None;
    let mut previous_move = None;
    let mut calls = Vec::new();
    let mut saw_return = false;
    let mut helper_writes = 0;
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
        let load128 = matches_form(bits, LOAD128);
        let store128 = matches_form(bits, STORE128);
        // Both encodings carry the signed byte displacement in the same 24 bits.
        let offset = (w >> 40) as u32 & 0xff_ffff;
        if load128 || store128 {
            let vector = if load128 { rd } else { rb };
            if !vector.is_multiple_of(4) || vector > 248 || !addr.is_multiple_of(2) || addr > 252 {
                return Err(Refused(format!(
                    "invalid register group for sm_80 vector access at 0x{at:x}"
                )));
            }
        }
        let (word, descriptor) = if matches_form(bits, LOAD16) || load128 {
            let load = assemble(
                "ldg__sImmOffset",
                &[
                    ("e", Str("E")),
                    ("sz", Str(if load128 { "_128" } else { "U16" })),
                    ("sem", Str(if load128 { "WEAK" } else { "CONSTANT" })),
                    ("sco", Str("SYS")),
                    ("cop", Str("EN")),
                    ("Rd", Int(rd as u32)),
                    ("Ra", Int(addr as u32)),
                    ("Ra_offset", Int(offset)),
                    ("Pu", Int(7)),
                ],
            );
            (load, Some((w >> 32) as u8 & 63))
        } else if matches_form(bits, STORE16) || matches_form(bits, STORE32) || store128 {
            let size = if store128 {
                "_128"
            } else if matches_form(bits, STORE16) {
                "U16"
            } else {
                "_32"
            };
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
                    ("Ra_offset", Int(offset)),
                ],
            );
            (store, Some((w >> 64) as u8 & 63))
        } else if matches_form(bits, HALF_CONSTANT) {
            // RN(-0 * +0 + c) preserves a finite half, including the sign of zero.
            let pair = (w >> 32) as u32;
            if [pair, pair >> 16].iter().any(|&half| half & 0x7c00 == 0x7c00) {
                return Err(Refused(format!("nonfinite sm_80 half constant at 0x{at:x}")));
            }
            (
                assemble(
                    "imad__RRsI_RRI",
                    &[
                        ("Rd", Int(rd as u32)),
                        ("Ra", Int(255)),
                        ("Rb", Int(255)),
                        ("Sc", Int(pair)),
                    ],
                ),
                None,
            )
        } else if matches_form(bits, COPY128) {
            if addr > 252 || !addr.is_multiple_of(2) {
                return Err(Refused(format!(
                    "invalid address for sm_80 asynchronous copy at 0x{at:x}"
                )));
            }
            let global_offset = ((w >> 32) as i32) << 20 >> 20;
            let shared_offset = ((w >> 44) as i32) << 12 >> 12;
            if rd == 255 && shared_offset < 0 {
                return Err(Refused(format!("negative absolute shared address at 0x{at:x}")));
            }
            operations.insert(
                i,
                Operation::GlobalToShared {
                    global_offset,
                    shared_offset,
                },
            );
            // Liveness must see the global address pair, shared address and guard.
            // Emission replaces this store with the two Volta memory operations.
            let store = assemble(
                "stg__sImmOffset",
                &[
                    ("e", Str("E")),
                    ("sz", Str("_32")),
                    ("sem", Str("WEAK")),
                    ("sco", Str("SYS")),
                    ("cop", Str("EN")),
                    ("Ra", Int(addr as u32)),
                    ("Rb", Int(rd as u32)),
                    ("Pg", Int((w >> 12) as u32 & 7)),
                    ("Pg_not", Int((w >> 15) as u32 & 1)),
                ],
            );
            (store, Some((w >> 64) as u8 & 63))
        } else if matches_form(bits, COMMIT) {
            // The group now consists of ordinary shared stores.
            (assemble("membar_", &[("sem", Str("SC")), ("sco", Str("CTA"))]), None)
        } else if matches_form(bits, PACK) {
            // A single PRMT exposes both input registers and the output to liveness.
            operations.insert(i, Operation::Pack);
            let surrogate = assemble(
                "prmt__RuIR_RIR",
                &[
                    ("Rd", Int(rd as u32)),
                    ("Ra", Int(rb as u32)),
                    ("Rc", Int(addr as u32)),
                    ("Sb", Int(0x5410)),
                ],
            );
            (surrogate, None)
        } else if SHARED_FORMS.iter().any(|&form| matches_form(bits, form)) {
            (Some(bits), None)
        } else {
            return Err(Refused(format!("unsupported sm_80 instruction {w:032x} at 0x{at:x}")));
        };
        if descriptor.is_some_and(|q| q > 60 || !q.is_multiple_of(2) || addr > 252 || !addr.is_multiple_of(2)) {
            return Err(Refused(format!("unsupported sm_80 address register pair at 0x{at:x}")));
        }
        let word = word.ok_or_else(|| Refused(format!("cannot encode sm_80 instruction for Volta at 0x{at:x}")))?;
        let word = if matches_form(bits, COMMIT) || matches_form(bits, COPY128) {
            word | 63 << WAIT
        } else {
            (word & !CONTROLS) | control
        };
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
        let mut branch = if let Some((fid, scale)) = find_relative(&d, 75) {
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
        let is_call = d.mnemonic == "CALL";
        let is_return = d.mnemonic == "RET";
        let mut falls_through = conditional || !matches!(d.mnemonic, "BRA" | "EXIT" | "RET");
        if is_call || is_return || helper.is_some() {
            let Some(start) = helper else {
                return Err(Refused("sm_80 CALL or RET without a local function symbol".into()));
            };
            let in_helper = i >= start;
            if is_call {
                let Some((from, reg, back)) = previous_move else {
                    return Err(Refused(format!("unproven sm_80 return address at 0x{at:x}")));
                };
                if back as usize != at + 16
                    || in_helper
                    || branch != Some(start)
                    || i + 1 >= start
                    || conditional
                    || returns.is_some_and(|r| r != reg)
                {
                    return Err(Refused(format!("unsupported sm_80 local call at 0x{at:x}")));
                }
                returns = Some(reg);
                calls.push((from, i));
            } else if is_return {
                if !in_helper || returns.is_none() || d.op(OP_RA) != returns || branch != Some(0) || conditional {
                    return Err(Refused(format!("unproven sm_80 return at 0x{at:x}")));
                }
                saw_return = true;
                branch = None;
                falls_through = false;
            } else if branch.is_some_and(|target| (target >= start) != in_helper) || (i + 1 == start && falls_through) {
                return Err(Refused(format!(
                    "sm_80 control flow crosses local function boundary at 0x{at:x}"
                )));
            }
            if in_helper && returns.is_some_and(|r| d.writes.contains(&(LETTER_R, r))) {
                return Err(Refused(format!(
                    "sm_80 local function overwrites its return register at 0x{at:x}"
                )));
            }
            if in_helper {
                helper_writes |= killed;
            }
            if d.name == "mov__RI" && !conditional {
                previous_move = d
                    .op(OP_RD)
                    .filter(|r| *r < 255)
                    .zip(d.op(OP_SB))
                    .map(|(reg, back)| (i, reg, back));
            } else if d.kind == Kind::BRU
                || previous_move.is_some_and(|(_, reg, _)| d.writes.contains(&(LETTER_R, reg)))
            {
                previous_move = None;
            }
        }
        steps.push(Step {
            killed,
            initialized,
            uses: descriptor,
            branch,
            falls_through,
        });
        result.extend_from_slice(&word.to_le_bytes());
    }
    if helper.is_some() {
        if !saw_return
            || calls.is_empty()
            || steps.iter().any(|s| {
                s.branch
                    .is_some_and(|t| calls.iter().any(|&(from, call)| from < t && t <= call))
            })
        {
            return Err(Refused(
                "sm_80 local function lacks a proven call and return path".into(),
            ));
        }
        // A descriptor is valid on return only when neither half can be overwritten by
        // any instruction of the callee. Forget those pairs on both outgoing CALL edges.
        for (_, call) in calls {
            steps[call].killed |= helper_writes;
        }
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
    Ok(Normalized {
        words: result,
        operations,
    })
}

pub(super) fn emit(ctx: &mut Ctx<'_>, i: usize, d: &Decoded, op: &Operation) -> Result<Vec<Item>, Refused> {
    let (global_offset, shared_offset) = match op {
        Operation::Pack => return pack(ctx, i, d),
        Operation::GlobalToShared {
            global_offset,
            shared_offset,
        } => (global_offset, shared_offset),
    };
    // Scratch starts at an even register. Six consecutive registers always contain
    // a four-aligned group suitable for a 128-bit transaction.
    let temp = ctx
        .scratch_at_reg
        .ok_or_else(|| Refused("no dedicated register group for an sm_80 asynchronous copy".into()))?
        .next_multiple_of(4);
    let pg = d.op_or(OP_PG, 7);
    let invert = d.op_or(OP_PG_AT_NOT, 0);
    let load = assemble(
        "ldg__sImmOffset",
        &[
            ("e", Str("E")),
            ("sz", Str("_128")),
            ("sem", Str("WEAK")),
            ("sco", Str("SYS")),
            ("cop", Str("EN")),
            ("Pu", Int(7)),
            ("Rd", Int(temp)),
            ("Ra", Int(d.op_or(OP_RA, 255))),
            ("Ra_offset", Int(*global_offset as u32)),
            ("Pg", Int(pg)),
            ("Pg_not", Int(invert)),
            ("dst_wr_sb", Int(ARRIVES)),
        ],
    )
    .ok_or_else(|| Refused("cannot assemble the global load for an sm_80 copy".into()))?;
    let store = assemble(
        if d.op_or(OP_RB, 255) == 255 {
            "sts__uImmOffset"
        } else {
            "sts__sImmOffset"
        },
        &[
            ("sz", Str("_128")),
            ("Ra", Int(d.op_or(OP_RB, 255))),
            ("Ra_offset", Int(*shared_offset as u32)),
            ("Rb", Int(temp)),
            ("Pg", Int(pg)),
            ("Pg_not", Int(invert)),
            ("src_rel_sb", Int(SEQUENCE)),
        ],
    )
    .ok_or_else(|| Refused("cannot assemble the shared store for an sm_80 copy".into()))?;
    let mut a = Item::new(load | 63 << WAIT, i, "sm80: load asynchronous copy", true);
    a.then = Some(ARRIVES);
    let mut b = Item::new(store | 63 << WAIT, i, "sm80: store asynchronous copy", true);
    b.then = Some(SEQUENCE);
    Ok(vec![a, b])
}

pub(super) fn pack(ctx: &mut super::context::Ctx<'_>, i: usize, d: &Decoded) -> Result<Vec<Item>, Refused> {
    let rd = d.op_or(OP_RD, 255);
    let low = d.op_or(OP_RA, 255);
    let high = d.op_or(OP_RC, 255);
    let mut items = Vec::new();
    let mut conversion = |dest, source, why| -> Result<(), Refused> {
        let word = assemble(
            "f2f_f32_downconvert__RRR_RRR",
            &[
                ("dstfmt.srcfmt", Str("F16.F32")),
                ("Rd", Int(dest)),
                ("Rb", Int(source)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble F2F for a packed sm_80 conversion".into()))?;
        let word = (word & !(7 << WRITE_BARRIER)) | (u128::from(ARRIVES) << WRITE_BARRIER) | (63 << WAIT);
        let mut item = Item::new(word, i, why, true);
        item.then = Some(ARRIVES);
        items.push(item);
        Ok(())
    };
    // Convert the high input before writing the destination, which can alias either input.
    let upper = if high == 255 {
        255
    } else {
        let temp = ctx.pool(1)?;
        conversion(temp, high, "sm80: convert high half")?;
        temp
    };
    conversion(rd, low, "sm80: convert low half")?;
    let permute = assemble(
        "prmt__RuIR_RIR",
        &[
            ("Rd", Int(rd)),
            ("Ra", Int(rd)),
            ("Sb", Int(0x5410)),
            ("Rc", Int(upper)),
        ],
    )
    .ok_or_else(|| Refused("cannot assemble PRMT for a packed sm_80 conversion".into()))?;
    let why = if high == 255 {
        "sm80: zero high half"
    } else {
        "sm80: pack both halves"
    };
    items.push(Item::new(permute, i, why, true));
    Ok(items)
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
    fn two_source_pack_preserves_both_inputs_under_destination_aliasing() {
        let high = 2;
        let low = 3;
        for rd in [high, low, 0] {
            let mut input = Elf::parse(IMAGE).unwrap();
            let section = input.section(".text.memory_check").unwrap();
            let word = &mut input.sections[section].data[0xc0..0xd0];
            let source = u128::from_le_bytes(word.try_into().unwrap());
            assert!(matches_form(source & !CONTROLS, PACK));
            let operands = (u128::from(rd) << 16) | (u128::from(high) << 24) | (u128::from(low) << 32);
            word.copy_from_slice(&((source & !(0xff_ffff << 16)) | operands).to_le_bytes());
            let result = translate(&input.write(), false).unwrap();
            assert!(
                result
                    .report
                    .iter()
                    .any(|r| r.name == "memory_check" && r.refused.is_none())
            );
            let target = Elf::parse(&result.bytes).unwrap();
            let body = &target.sections[target.section(".text.memory_check").unwrap()].data;
            let mut decoder = Decoder::new(70);
            let ops: Vec<_> = body
                .as_chunks::<16>()
                .0
                .iter()
                .map(|&w| decoder.decode(u128::from_le_bytes(w)).unwrap())
                .collect();
            let group = ops
                .windows(3)
                .find(|w| w[0].mnemonic == "F2F" && w[1].mnemonic == "F2F" && w[2].mnemonic == "PRMT")
                .unwrap();
            let temporary = group[0].op_or(OP_RD, 255);
            assert!(![rd, low, high, 255].contains(&temporary));
            assert_eq!(group[0].op_or(OP_RB, 255), high);
            assert_eq!((group[1].op_or(OP_RD, 255), group[1].op_or(OP_RB, 255)), (rd, low));
            assert_eq!(
                (
                    group[2].op_or(OP_RD, 255),
                    group[2].op_or(OP_RA, 255),
                    group[2].op_or(OP_RC, 255)
                ),
                (rd, rd, temporary)
            );
            assert_eq!(group[2].op_or(OP_SB, 0), 0x5410);
            for pair in group.windows(2) {
                let scoreboard = pair[0].word >> WRITE_BARRIER & 7;
                assert!(scoreboard < 6 && pair[1].word >> WAIT & (1 << scoreboard) != 0);
            }
        }
    }

    #[test]
    fn guarded_copies_preserve_signed_offsets_and_refuse_unsupported_memory_forms() {
        let e = Elf::parse(include_bytes!(
            "../../../../tests/kernels/translate/k_guarded_copy.sm80.cubin"
        ))
        .unwrap();
        let body = &e.sections[e.section(".text.guarded_copy").unwrap()].data;
        let result = normalize(body, &mut Decoder::new(75), None).unwrap();
        let mut copies: Vec<_> = result
            .operations
            .iter()
            .filter_map(|(&i, op)| match op {
                Operation::GlobalToShared {
                    global_offset,
                    shared_offset,
                } => Some((i, *global_offset, *shared_offset)),
                _ => None,
            })
            .collect();
        copies.sort_unstable();
        // ptxas adjusted the second base by 512. Its signed displacement is now -528.
        assert_eq!(copies.len(), 2);
        assert_eq!((copies[0].1, copies[0].2), (512, 16));
        assert_eq!((copies[1].1, copies[1].2), (-528, 16));
        let words = body.as_chunks::<16>().0;
        let at = copies[1].0;
        let copy = u128::from_le_bytes(words[at]);
        let store_at = words
            .iter()
            .position(|&bytes| matches_form(u128::from_le_bytes(bytes) & !CONTROLS, STORE128))
            .unwrap();
        let store = u128::from_le_bytes(words[store_at]);
        for (i, wrong, why) in [
            (at, copy | 1 << 24, "invalid address"),
            (at, copy | 1 << 63, "negative absolute shared address"),
            (at, copy ^ (1 << 64), "address register pair"),
            (store_at, store | 1 << 32, "invalid register group"),
        ] {
            let mut input = body.clone();
            input[16 * i..16 * i + 16].copy_from_slice(&wrong.to_le_bytes());
            let error = normalize(&input, &mut Decoder::new(75), None).unwrap_err();
            assert!(error.0.contains(why), "{}", error.0);
        }
    }

    const LOCAL: &[u8] = include_bytes!("../../../../tests/kernels/translate/k_ampere_helper.sm80.cubin");

    #[test]
    fn relocates_the_private_helper_and_preserves_an_untouched_descriptor_across_the_call() {
        let translation = translate(LOCAL, false).unwrap();
        assert_eq!(translation.report.len(), 1);
        assert_eq!(translation.report[0].refused, None);
        let image = Elf::parse(&translation.bytes).unwrap();
        let section = image.section(".text.checked_add").unwrap();
        let length = image.sections[section].data.len() as u64;
        let symbols = image.symbols().unwrap();
        let helper = symbols
            .iter()
            .find(|s| s.info == 2 && s.shndx as usize == section)
            .unwrap();
        assert!(helper.value > 0);
        assert_eq!(helper.value + helper.size, length);
    }

    #[test]
    fn rejects_unproven_calls_returns_and_descriptor_clobbers() {
        let image = Elf::parse(LOCAL).unwrap();
        let code = &image.sections[image.section(".text.checked_add").unwrap()].data;
        let instruction = |at| u128::from_le_bytes(code[at..at + 16].try_into().unwrap());
        let check = |at, word: u128, reason: &str| {
            let mut changed = code.clone();
            changed[at..at + 16].copy_from_slice(&word.to_le_bytes());
            let error = normalize(&changed, &mut Decoder::new(75), Some(8)).unwrap_err();
            assert!(error.0.contains(reason), "{}", error.0);
        };
        // The caller establishes UR6:UR7 before the CALL and uses the pair after return.
        // Changing the callee's scalar ULDC to write either half invalidates that use.
        for half in [6u128, 7] {
            check(
                0x80,
                (instruction(0x80) & !(255 << 16)) | (half << 16),
                "descriptor UR6",
            );
        }
        // Move the call past the declared helper entry. Change its recorded return
        // address, or skip its setup with a direct branch to the call.
        check(0x30, instruction(0x30) + (16 << 32), "unsupported sm_80 local call");
        check(0x10, instruction(0x10) + (16 << 32), "unsupported sm_80 local call");
        let jump_to_call = 0x0000000003800000_0000002000000947 | (instruction(0) & CONTROLS);
        check(0, jump_to_call, "lacks a proven call and return path");
        // An indirect return through another register or an altered relative base
        // does not prove the fast path reaches the original continuation.
        check(0xc0, instruction(0xc0) ^ (1 << 24), "unproven sm_80 return");
        check(0xc0, instruction(0xc0) + (16 << 32), "unproven sm_80 return");
        check(
            0xb0,
            (instruction(0xb0) & !(255 << 16)) | (2 << 16),
            "overwrites its return register",
        );
        assert!(normalize(code, &mut Decoder::new(75), None).is_err());
    }

    #[test]
    fn rejects_helpers_exposed_as_an_entry_or_with_ambiguous_extent() {
        for variant in 0..4 {
            let mut image = Elf::parse(LOCAL).unwrap();
            let mut symbols = image.symbols().unwrap();
            let helper = symbols.iter_mut().find(|s| s.info == 2 && s.value != 0).unwrap();
            match variant {
                0 => helper.info |= 0x10,
                1 => helper.other |= 0x10,
                2 => helper.size -= 16,
                _ => helper.value += 1,
            }
            image.set_symbols(&symbols);
            let result = translate(&image.write(), false).unwrap();
            assert!(result.report[0].refused.as_deref().unwrap().contains("secondary entry"));
        }
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
            let refusal = normalize(code, &mut Decoder::new(75), None).unwrap_err();
            assert!(refusal.0.contains(why), "{}", refusal.0);
        };
        assert!(normalize(body, &mut Decoder::new(75), None).is_ok());
        // The immediate in FADD is an operand. A different immediate retains its full encoding.
        let fadd = (word(0xa0) & !(u128::from(u32::MAX) << 32)) | (u128::from((-123.5f32).to_bits()) << 32);
        put(&mut code, 0xa0, fadd);
        let normalized = normalize(&code, &mut Decoder::new(75), None).unwrap();
        assert_eq!(&normalized.words[0xa0..0xb0], &fadd.to_le_bytes());
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
        assert!(normalize(&code, &mut Decoder::new(75), None).is_ok());
        put(&mut code, 0x10, bssy | (0x30 << 32)); // BSYNC would jump past the ULDC.
        check(&code, "does not fall through to its convergence target");

        code.copy_from_slice(body);
        put(&mut code, 0x70, word(0x70) ^ (1 << 101));
        check(&code, "unsupported sm_80 instruction");
    }
}
