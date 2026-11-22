//! The sm_80 instruction forms that can enter the Turing-to-Volta translator.
//! The table fixes all bits except explicitly identified operand fields. The forms with
//! different encodings on Ampere and Turing are decoded before testing shared forms.
use std::collections::HashMap;

use crate::generated::ampere::{
    ATOM_ADD32, BRA_MASK_ALL, COMMIT, COPY128, HALF_CONSTANT, LOAD16, LOAD32, LOAD32_CONST, LOAD64_CONST, LOAD128,
    PACK, SHARED_FORMS, SHARED_INC, STORE16, STORE32, STORE64, STORE128,
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
    IncrementShared { offset: u32 },
    GlobalToShared { global_offset: i32, shared_offset: i32 },
}

#[derive(Debug)]
pub(super) struct Normalized {
    pub words: Vec<u8>,
    pub operations: HashMap<usize, Operation>,
}

#[derive(Clone, Copy)]
enum Convergence {
    Set(u32, usize),
    Wait(u32),
}

struct Step {
    convergence: Option<Convergence>,
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
    let mut barrier_mask = 0u16;
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
        let const64 = matches_form(bits, LOAD64_CONST);
        let store64 = matches_form(bits, STORE64);
        let load32 = matches_form(bits, LOAD32);
        let const32 = matches_form(bits, LOAD32_CONST);
        // Both encodings carry the signed byte displacement in the same 24 bits.
        let offset = (w >> 40) as u32 & 0xff_ffff;
        if load128 || store128 || const64 || store64 {
            let vector = if load128 || const64 { rd } else { rb };
            let width = if load128 || store128 { 4 } else { 2 };
            // A vector store can read four zeros from the architectural zero register.
            let zero_store = store128 && vector == 255;
            let valid_group = vector.is_multiple_of(width) && vector <= 256 - 2 * width;
            if !(valid_group || zero_store) || !addr.is_multiple_of(2) || addr > 252 {
                return Err(Refused(format!(
                    "invalid register group for sm_80 vector access at 0x{at:x}"
                )));
            }
        }
        let (word, descriptor) = if matches_form(bits, LOAD16) || load128 || load32 || const32 || const64 {
            let load = assemble(
                "ldg__sImmOffset",
                &[
                    ("e", Str("E")),
                    (
                        "sz",
                        Str(if load128 {
                            "_128"
                        } else if const64 {
                            "_64"
                        } else if load32 || const32 {
                            "_32"
                        } else {
                            "U16"
                        }),
                    ),
                    ("sem", Str(if load128 || load32 { "WEAK" } else { "CONSTANT" })),
                    ("sco", Str("SYS")),
                    ("cop", Str("EN")),
                    ("Rd", Int(rd as u32)),
                    ("Ra", Int(addr as u32)),
                    ("Ra_offset", Int(offset)),
                    ("Pu", Int(7)),
                    ("Pg", Int((w >> 12) as u32 & 7)),
                    ("Pg_not", Int((w >> 15) as u32 & 1)),
                ],
            );
            (load, Some((w >> 32) as u8 & 63))
        } else if matches_form(bits, STORE16) || matches_form(bits, STORE32) || store64 || store128 {
            let size = if store128 {
                "_128"
            } else if store64 {
                "_64"
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
                    ("Pg", Int((w >> 12) as u32 & 7)),
                    ("Pg_not", Int((w >> 15) as u32 & 1)),
                    ("Rb", Int(rb as u32)),
                    ("Ra", Int(addr as u32)),
                    ("Ra_offset", Int(offset)),
                ],
            );
            (store, Some((w >> 64) as u8 & 63))
        } else if matches_form(bits, BRA_MASK_ALL) {
            let branch = assemble("bra_", &[("Pg", Int(7)), ("Pp", Int(7)), ("cond", Str("DIV"))]);
            (branch.map(|op| op | bits & BRA_MASK_ALL.1), None)
        } else if matches_form(bits, ATOM_ADD32) {
            let add = assemble(
                "atomg__RaNonRZ",
                &[
                    ("Rd", Int(rd as u32)),
                    ("Pu", Int(7)),
                    ("Ra", Int(addr as u32)),
                    ("Ra_offset", Int(offset)),
                    ("Rb", Int(rb as u32)),
                    ("e", Str("E")),
                    ("sz", Str("32")),
                    ("op", Str("ADD")),
                    ("cop", Str("EN")),
                    ("sem", Str("STRONG")),
                    ("sco", Str("GPU")),
                ],
            );
            (add, Some((w >> 64) as u8 & 63))
        } else if matches_form(bits, SHARED_INC) {
            if addr == 255 {
                return Err(Refused(format!("missing sm_80 shared increment index at 0x{at:x}")));
            }
            operations.insert(i, Operation::IncrementShared { offset });
            // This store describes the guard and index read by the atomic reduction.
            let surrogate = assemble(
                "sts__sImmOffset",
                &[
                    ("Ra", Int(addr as u32)),
                    ("Rb", Int(255)),
                    ("sz", Str("_32")),
                    ("Pg", Int((w >> 12) as u32 & 7)),
                    ("Pg_not", Int((w >> 15) as u32 & 1)),
                ],
            );
            (surrogate, None)
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
        let convergence = match d.mnemonic {
            "BSSY" | "BSYNC" => {
                let reg = d.op_or_name("barReg", 16);
                if reg >= 16 {
                    return Err(Refused(format!("invalid sm_80 convergence barrier at 0x{at:x}")));
                }
                if d.mnemonic == "BSSY" {
                    Some(Convergence::Set(
                        reg,
                        branch.ok_or_else(|| Refused(format!("missing sm_80 convergence target at 0x{at:x}")))?,
                    ))
                } else {
                    barrier_mask |= 1u16 << reg;
                    Some(Convergence::Wait(reg))
                }
            }
            _ => None,
        };
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
            convergence,
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
    for reg in 0..16 {
        if barrier_mask & (1u16 << reg) != 0 {
            check_convergence(&steps, reg, helper)?;
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

/// At every BSYNC, all paths must have established the following PC as that
/// barrier's convergence target. An earlier completed region may reuse the same register.
fn check_convergence(steps: &[Step], reg: u32, helper: Option<usize>) -> Result<(), Refused> {
    // PCs lie below steps.len(). None is unreachable; unknown is reachable with no definite target.
    let unknown = steps.len();
    let callee_changes = helper.is_some_and(|start| {
        steps[start..]
            .iter()
            .any(|step| matches!(step.convergence, Some(Convergence::Set(r,_) | Convergence::Wait(r)) if r == reg))
    });
    let mut live = vec![None; steps.len()];
    let mut pending = vec![0];
    live[0] = Some(unknown);
    while let Some(i) = pending.pop() {
        let step = &steps[i];
        let before = live[i].unwrap();
        let after = match step.convergence {
            Some(Convergence::Set(r, target)) if r == reg => target,
            Some(Convergence::Wait(r)) if r == reg => unknown,
            _ if callee_changes && step.branch == helper => unknown,
            _ => before,
        };
        let fall = (step.falls_through && i + 1 < steps.len()).then_some(i + 1);
        for next in [step.branch, fall].into_iter().flatten() {
            // On the implicit edge to reconvergence, the region has completed.
            let out = if matches!(step.convergence, Some(Convergence::Set(r,target)) if r == reg && target == next) {
                unknown
            } else {
                after
            };
            let joined = if live[next].is_some_and(|previous| previous != out) {
                unknown
            } else {
                out
            };
            if live[next] != Some(joined) {
                live[next] = Some(joined);
                pending.push(next);
            }
        }
    }
    for (i, step) in steps.iter().enumerate() {
        if matches!(step.convergence, Some(Convergence::Wait(r)) if r == reg) && live[i] != Some(i + 1) {
            return Err(Refused(format!(
                "sm_80 BSYNC B{reg} at 0x{:x} does not fall through to its convergence target",
                i * 16
            )));
        }
    }
    Ok(())
}

pub(super) fn emit(ctx: &mut Ctx<'_>, i: usize, d: &Decoded, op: &Operation) -> Result<Vec<Item>, Refused> {
    let (global_offset, shared_offset) = match op {
        Operation::Pack => return pack(ctx, i, d),
        Operation::IncrementShared { offset } => return increment(ctx, i, d, *offset),
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

fn increment(ctx: &Ctx<'_>, i: usize, d: &Decoded, offset: u32) -> Result<Vec<Item>, Refused> {
    let temp = ctx
        .scratch_at_reg
        .ok_or_else(|| Refused("no dedicated registers for an sm_80 shared atomic increment".into()))?;
    let addr = assemble(
        "lea_lo_imm__RuIR_RIR",
        &[
            ("Rd", Int(temp)),
            ("Ra", Int(d.op_or(OP_RA, 255))),
            ("Sb", Int(0)),
            ("scaleU5", Int(2)),
            ("Pu", Int(7)),
        ],
    )
    .ok_or_else(|| Refused("cannot encode scaled shared address".into()))?;
    let one = assemble(
        "imad__RRsI_RRI",
        &[
            ("Rd", Int(temp + 1)),
            ("Ra", Int(255)),
            ("Rb", Int(255)),
            ("Sc", Int(1)),
        ],
    )
    .ok_or_else(|| Refused("cannot encode increment operand".into()))?;
    let word = assemble(
        "atoms__RaNonRZ",
        &[
            ("Ra", Int(temp)),
            ("Ra_offset", Int(offset)),
            ("Rb", Int(temp + 1)),
            ("Rd", Int(255)),
            ("op", Str("ADD")),
            ("sz", Str("32")),
            ("Pg", Int(d.op_or(OP_PG, 7))),
            ("Pg_not", Int(d.op_or(OP_PG_AT_NOT, 0))),
            ("src_rel_sb", Int(SEQUENCE)),
        ],
    )
    .ok_or_else(|| Refused("cannot encode shared atomic increment".into()))?;
    let mut add = Item::new(word | 63 << WAIT, i, "sm80: shared atomic increment", true);
    add.then = Some(SEQUENCE);
    Ok(vec![
        Item::new(addr | 63 << WAIT, i, "sm80: index shared counter", true),
        Item::new(one | 63 << WAIT, i, "sm80: increment by one", true),
        add,
    ])
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

    #[test]
    fn preserves_vector_store_sources_predicates_and_displacements() {
        // Actual ptxas output for a 128-bit zero fill and a predicated concatenation.
        let descriptor = 0x000fc60000000a00_0000460000047ab9u128;
        let zero = 0x000fe2000c101d04_000000ff02007986u128;
        let load = 0x000ee8000c1e9b00_0000000404048981u128;
        let store = 0x010fe8000c101b04_0800000608008986u128;
        let half = 0x000fe8000c101504_1000060d0a007986u128;
        let code = |inst: u128| [descriptor.to_le_bytes(), inst.to_le_bytes()].concat();
        let mut decoder = Decoder::new(75);
        for (word, mnemonic, size, reg_field, reg, pg, inverted, displacement) in [
            (zero, "STG", 6, OP_RB, 255, 7, 0, 0),
            (load, "LDG", 5, OP_RD, 4, 0, 1, 0),
            (store, "STG", 5, OP_RB, 6, 0, 1, 0x80000),
            (half, "STG", 2, OP_RB, 13, 7, 0, 0x100006),
        ] {
            let result = normalize(&code(word), &mut decoder, None).unwrap();
            let target = decoder
                .decode(u128::from_le_bytes(result.words[16..].try_into().unwrap()))
                .unwrap();
            assert_eq!(target.mnemonic, mnemonic);
            assert_eq!(target.op(OP_SZ), Some(size));
            assert_eq!(target.op(reg_field), Some(reg));
            assert_eq!((target.op(OP_PG), target.op(OP_PG_AT_NOT)), (Some(pg), Some(inverted)));
            assert_eq!(target.op_or_name("Ra_offset", 0), displacement);
        }
        for invalid in [
            zero ^ (1 << 32),                    // R254 cannot name a quad.
            zero | (1 << 24),                    // R3 cannot hold a 64-bit address.
            load | (1 << 16),                    // Odd destination.
            (load & !(255 << 16)) | (255 << 16), // RZ is only accepted for the zero store.
            store | (1 << 32),                   // Odd source.
        ] {
            assert!(
                normalize(&code(invalid), &mut decoder, None)
                    .unwrap_err()
                    .0
                    .contains("invalid register group")
            );
        }
        assert!(
            normalize(&load.to_le_bytes(), &mut decoder, None)
                .unwrap_err()
                .0
                .contains("descriptor")
        );
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
    fn full_mask_branch_and_local_weak_helper_are_required_for_the_scan() {
        let bytes = include_bytes!("../../../../tests/kernels/translate/k_warp_scan.sm80.cubin");
        let good = translate(bytes, false).unwrap();
        assert!(good.report[0].refused.is_none());
        let mut elf = Elf::parse(bytes).unwrap();
        let sec = elf.section(".text.warp_scan").unwrap();
        let symbols = elf.symbols().unwrap();
        let index = symbols
            .iter()
            .position(|s| s.info == 0x22 && s.shndx as usize == sec && s.value != 0)
            .unwrap();
        assert!(vf_fatbin::cubin_has_assembler_helper(bytes, index));
        assert!(!vf_fatbin::cubin_has_assembler_helper(bytes, usize::MAX));
        // Moving the same symbol to the externally visible part of the table changes its ABI.
        let symtab = elf.section(".symtab").unwrap();
        elf.sections[symtab].info = index as u32;
        let external = elf.write();
        assert!(!vf_fatbin::cubin_has_assembler_helper(&external, index));
        assert!(translate(&external, false).unwrap().report[0].refused.is_some());

        let code = &elf.sections[sec].data;
        let words = code.as_chunks::<16>().0;
        let pc = words
            .iter()
            .position(|&w| matches_form(u128::from_le_bytes(w) & !CONTROLS, BRA_MASK_ALL))
            .unwrap();
        let original = u128::from_le_bytes(words[pc]);
        for bit in [24, 30] {
            // A register other than URZ or the uncomplemented URZ changes the participation mask.
            let mut modified = code.clone();
            modified[pc * 16..pc * 16 + 16].copy_from_slice(&(original ^ (1 << bit)).to_le_bytes());
            let helper = Some(symbols[index].value as usize / 16);
            assert!(normalize(&modified, &mut Decoder::new(75), helper).is_err());
        }
    }

    #[test]
    fn shared_and_global_atomics_require_the_recognized_operands_and_memory_scope() {
        let bytes = include_bytes!("../../../../tests/kernels/translate/k_histogram.sm80.cubin");
        let elf = Elf::parse(bytes).unwrap();
        let code = &elf.sections[elf.section(".text.histogram").unwrap()].data;
        let words = code.as_chunks::<16>().0;
        let cases = [(ATOM_ADD32, 77), (ATOM_ADD32, 64), (SHARED_INC, 16), (SHARED_INC, 78)];
        for (form, bit) in cases {
            let pc = words
                .iter()
                .position(|&w| matches_form(u128::from_le_bytes(w) & !CONTROLS, form))
                .unwrap();
            let altered = u128::from_le_bytes(words[pc]) ^ (1 << bit);
            let mut modified = code.clone();
            modified[pc * 16..pc * 16 + 16].copy_from_slice(&altered.to_le_bytes());
            assert!(normalize(&modified, &mut Decoder::new(75), None).is_err());
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
        // B0 can be recycled once the first region is complete.
        put(&mut code, 0x40, bssy | (0x10 << 32));
        put(&mut code, 0x50, bsync);
        assert!(normalize(&code, &mut Decoder::new(75), None).is_ok());
        // A path entering the second BSYNC directly has no established convergence target.
        put(&mut code, 0, (branch & !(0xffu128 << 32)) | (0x40 << 32));
        check(&code, "does not fall through to its convergence target");
        put(&mut code, 0, word(0));
        put(&mut code, 0x10, bssy | (0x30 << 32)); // BSYNC would jump past the ULDC.
        check(&code, "does not fall through to its convergence target");

        code.copy_from_slice(body);
        put(&mut code, 0x70, word(0x70) ^ (1 << 101));
        check(&code, "unsupported sm_80 instruction");
    }
}
