//! Borrowed registers: registers of the program that hold nothing where the translation
//! needs them. Liveness analysis of registers, homes for uniform registers, the pool at
//! each instruction, and victims for room-making at level 3.

use std::collections::{HashMap, HashSet};

use crate::decode::{LETTER_R, LETTER_U};

use super::context::entries;
use super::parse::FuncData;
use super::{Refused, RegSet, regset_andnot, regset_empty, regset_or, regset_set, regset_test, unannounced};

/// Registers of the program that hold nothing where the translation needs registers: a home
/// for each uniform register through its whole life, and an even register and the next at an
/// instruction whose sequence computes into two (the pool). `docs/translator.md`, "Registers".
///
/// A register is free at an instruction when it is dead before it, dead after it, and neither
/// read nor written by it (an instruction may write a register that nothing reads).
/// The lives follow the ways of `pairing::Flow`; a write
/// under a predicate ends no life; a call within the section reads and writes what the code
/// it calls does; a call elsewhere reads every register the function names; at a return every
/// one of them is alive. R0 and R1 are never taken (R1 is the stack).
pub(super) struct Borrowed {
    /// The highest register the program uses.
    pub(super) top: u32,
    /// The function's own highest register (before any budget override).
    pub(super) highest: u32,
    /// At each instruction: a bit set of registers that are alive, read or written there.
    pub(super) busy: Vec<RegSet>,
    /// At each instruction: a bit set of registers taken by homes of uniform registers.
    pub(super) taken: Vec<RegSet>,
    /// At each instruction: a bit set of registers read or written by the instruction itself
    /// (for `victims_at`: registers the instruction does not touch can be put in the frame).
    pub(super) touch: Vec<RegSet>,
    /// A bit set of registers that some instruction writes late without saying when
    /// (never borrowed as victims either).
    pub(super) never: RegSet,
    /// {uniform register: ordinary register it lives in through its whole life}.
    pub(super) homes: HashMap<u32, u32>,
    /// The ordinary registers taken for homes or for the pool.
    pub(super) used: HashSet<u32>,
    /// Level 3: everything at the instruction.
    pub(super) everything: bool,
    /// {uniform register: its word in the frame}.
    pub(super) slots: HashMap<u32, u32>,
    /// Held uniform predicates whose value must survive an interval with every register busy.
    pub(super) predicate_slots: HashMap<u32, u32>,
    /// The frame size in bytes (a multiple of 16).
    pub(super) frame_bytes: u32,
    /// Where the stack pointer is at each instruction: `depth[i]` is how far R1 is, before
    /// instruction i, under where the first instruction puts it. None when the depth is not
    /// known at that instruction. The program moves R1 by `IADD3 R1, R1, imm` only.
    pub(super) depth: Vec<Option<i32>>,
    /// Whether the depth of R1 is known everywhere the first instruction leads.
    pub(super) r1_fixed: bool,
    /// Whether room is made where no register holds nothing (VF_BORROW >= 5): registers
    /// the instruction does not touch are put in the frame and brought back.
    pub(super) room: bool,
    /// Whether the frame is addressed from R1 itself (the stack pointer), not from a base
    /// register loaded from c[0x0][0x28].
    pub(super) from_r1: bool,
    /// The most registers any single instruction put away at once (for the final frame size).
    pub(super) spill: u32,
}

impl Borrowed {
    pub(super) fn new(fd: &FuncData, all_in_memory: bool, top_override: Option<u32>) -> Result<Self, Refused> {
        let ent = entries(fd);
        let flow = crate::pairing::Flow::new(&fd.src, &fd.tables, &ent);
        if !flow.known {
            return Err(Refused("the ways through the function are not known".into()));
        }
        let n = fd.src.len();
        // Build the reads and writes bit sets for registers (RegSet) and uniform registers (u64)
        // at each instruction.
        let mut reads_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut reads_u: Vec<u64> = vec![0u64; n];
        let mut writes_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut writes_u: Vec<u64> = vec![0u64; n];
        let mut every_r: RegSet = regset_empty();
        let mut every_u: u64 = 0;
        for (i, d) in fd.src.iter().enumerate() {
            for &(letter, x, _) in &d.reads {
                if letter == LETTER_R && x != 255 {
                    regset_set(&mut reads_r[i], x);
                } else if letter == LETTER_U && x != 63 {
                    reads_u[i] |= 1u64 << x;
                }
            }
            for &(letter, x) in &d.writes {
                if letter == LETTER_R && x != 255 {
                    regset_set(&mut writes_r[i], x);
                } else if letter == LETTER_U && x != 63 {
                    writes_u[i] |= 1u64 << x;
                }
            }
            every_r = regset_or(&every_r, &regset_or(&reads_r[i], &writes_r[i]));
            every_u |= reads_u[i] | writes_u[i];
        }
        let mut kills_r = writes_r.clone();
        let mut kills_u = writes_u.clone();
        let mut leaves_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut leaves_u: Vec<u64> = vec![0u64; n];
        for i in 0..n {
            if flow.kind[i] == "call" {
                let called = flow.called.get(&i);
                match called {
                    Some(None) => {
                        // A function not known here: reads everything.
                        reads_r[i] = regset_or(&reads_r[i], &every_r);
                        reads_u[i] |= every_u;
                    }
                    Some(Some((alive, ws))) => {
                        for &x in alive {
                            if x < 255 {
                                regset_set(&mut reads_r[i], x);
                            }
                        }
                        for &x in ws {
                            if x < 255 {
                                regset_set(&mut writes_r[i], x);
                            }
                        }
                        // And the uniform registers that the code it calls reads or writes:
                        // `flow.called` has the registers only. A uniform register set before
                        // the call and read by that code was taken for dead at the call
                        // until 2026-11-19.
                        if let Some(entry) = flow.named[i] {
                            let mut body = std::collections::BTreeSet::new();
                            let mut work = vec![entry];
                            while let Some(k) = work.pop() {
                                if body.insert(k) {
                                    work.extend(flow.succ[k].iter().copied());
                                }
                            }
                            for &k in &body {
                                for &(letter, x, _) in &fd.src[k].reads {
                                    if letter == LETTER_U && x != 63 {
                                        reads_u[i] |= 1u64 << x;
                                    }
                                }
                                for &(letter, x) in &fd.src[k].writes {
                                    if letter == LETTER_U && x != 63 {
                                        writes_u[i] |= 1u64 << x;
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        // Not in called map means it's not a recognized call pattern.
                        reads_r[i] = regset_or(&reads_r[i], &every_r);
                        reads_u[i] |= every_u;
                    }
                }
                kills_r[i] = regset_empty();
                kills_u[i] = 0;
            } else if !crate::pairing::unguarded(&fd.src[i]) {
                kills_r[i] = regset_empty();
                kills_u[i] = 0;
            }
            if fd.src[i].mnemonic == "RET" {
                leaves_r[i] = every_r;
                leaves_u[i] = every_u;
            }
        }
        // Backward dataflow: live_in[i] = reads[i] | (live_out[i] & ~kills[i])
        // where live_out[i] = union of live_in[t] for t in succ[i], plus leaves[i].
        let mut live_in_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut live_in_u: Vec<u64> = vec![0u64; n];
        let mut changed = true;
        while changed {
            changed = false;
            for i in (0..n).rev() {
                let mut out_r = leaves_r[i];
                let mut out_u = leaves_u[i];
                for &t in &flow.succ[i] {
                    out_r = regset_or(&out_r, &live_in_r[t]);
                    out_u |= live_in_u[t];
                }
                let new_r = regset_or(&reads_r[i], &regset_andnot(&out_r, &kills_r[i]));
                let new_u = reads_u[i] | (out_u & !kills_u[i]);
                if new_r != live_in_r[i] || new_u != live_in_u[i] {
                    live_in_r[i] = new_r;
                    live_in_u[i] = new_u;
                    changed = true;
                }
            }
        }
        // busy[i] = live_in[i] | live_out[i] | reads[i] | writes[i], register portion only.
        // busy_u[i] = same for uniform registers (for computing spans).
        let mut busy: Vec<RegSet> = Vec::with_capacity(n);
        let mut busy_u: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let mut out_r = leaves_r[i];
            let mut out_u = leaves_u[i];
            for &t in &flow.succ[i] {
                out_r = regset_or(&out_r, &live_in_r[t]);
                out_u |= live_in_u[t];
            }
            busy.push(regset_or(
                &regset_or(&live_in_r[i], &out_r),
                &regset_or(&reads_r[i], &writes_r[i]),
            ));
            busy_u.push(live_in_u[i] | out_u | reads_u[i] | writes_u[i]);
        }

        // Registers that some instruction writes late without saying when are never borrowed:
        // their result arrives over what was written meanwhile (`docs/translator.md`, "Timing rules").
        let mut never = regset_empty();
        for d in &fd.src {
            if unannounced(d) {
                for &(letter, x) in &d.writes {
                    if letter == LETTER_R && x != 255 {
                        regset_set(&mut never, x);
                    }
                }
            }
        }
        // To make room (`victims_at`): what each instruction reads or writes itself.
        let touch: Vec<RegSet> = (0..n).map(|i| regset_or(&reads_r[i], &writes_r[i])).collect();
        for b in &mut busy {
            *b = regset_or(b, &never);
        }

        // `top`: the highest register that may be taken. The function's own highest; or, with
        // a budget, registers above it too, which hold nothing anywhere.
        let top = match top_override {
            Some(t) => t.max(fd.highest),
            None => fd.highest,
        };
        // Where the stack pointer is at each instruction: `depth[i]` is how far R1 is, before
        // instruction i, under where the first instruction puts it. The program moves it by
        // constants only (IADD3 R1, R1, imm); known at an instruction when every way to it
        // agrees; code that is called has the depth of its calls.
        let mut depth: Vec<Option<i32>> = vec![None; n];
        let mut r1_fixed = n > 1;
        if n > 1 {
            let mut work: Vec<(usize, i32)> = flow.succ[0].iter().map(|&t| (t, 0i32)).collect();
            while let Some((i, at)) = work.pop() {
                if !r1_fixed {
                    break;
                }
                if let Some(existing) = depth[i] {
                    if existing != at {
                        r1_fixed = false;
                    }
                    continue;
                }
                depth[i] = Some(at);
                let mut after = at;
                // Check if instruction i writes R1.
                if writes_r[i]
                    .iter()
                    .enumerate()
                    .any(|(wi, &w)| if wi == 0 { w & 2 != 0 } else { false })
                {
                    // Does it write R1 by IADD3 R1, R1, imm (unguarded, Rc=255, no negate)?
                    let d = &fd.src[i];
                    if d.name == "iadd3_imm__RsIR_RIR"
                        && d.op_or(crate::generated::fields::OP_RD, 255) == 1
                        && d.op_or(crate::generated::fields::OP_RA, 255) == 1
                        && d.op_or(crate::generated::fields::OP_RC, 255) == 255
                        && crate::pairing::unguarded(d)
                        && d.op_or(crate::generated::fields::OP_RA_AT_NEGATE, 0) == 0
                        && d.op_or(crate::generated::fields::OP_RC_AT_NEGATE, 0) == 0
                    {
                        let sb = d.op_or(crate::generated::fields::OP_SB, 0);
                        let imm = sb as i32;
                        after = at - imm;
                    } else {
                        r1_fixed = false;
                    }
                }
                for &t in &flow.succ[i] {
                    work.push((t, after));
                }
                // Code that is called has the depth of its calls: of the calls whose function
                // is known whole (`called`), as in the prototype. A function that calls another
                // is not entered, and a uniform register in memory there is refused (`slot`).
                if flow.kind[i] == "call"
                    && flow.called.get(&i).is_some_and(|effect| effect.is_some())
                    && let Some(Some(entry)) = flow.named.get(i).copied()
                {
                    work.push((entry, at));
                }
            }
        }
        // The uniform registers, those of the longest lives first: each in the lowest register
        // that is free wherever it is alive, read or written, and that no other one has there.
        let mut taken: Vec<RegSet> = vec![regset_empty(); n];
        let mut homes: HashMap<u32, u32> = HashMap::new();
        let mut used: HashSet<u32> = HashSet::new();

        // Build the list of uniform register units with their spans.
        let pairs_set: HashSet<u32> = fd.pairs.iter().copied().collect();
        let mut units: Vec<(u32, u32, Vec<usize>)> = Vec::new(); // (u, width, span)
        for &u in &fd.uniform {
            if u > 0 && pairs_set.contains(&(u - 1)) {
                continue; // upper half of a pair
            }
            let width: u32 = if pairs_set.contains(&u) { 2 } else { 1 };
            let mask = (1u64 << width) - 1;
            let span: Vec<usize> = (0..n).filter(|&i| (busy_u[i] >> u) & mask != 0).collect();
            units.push((u, width, span));
        }
        // Sort: longest lives first, ties by number.
        units.sort_by(|a, b| b.2.len().cmp(&a.2.len()).then(a.0.cmp(&b.0)));

        if !all_in_memory {
            for (u, width, span) in &units {
                let width = *width;
                // Union of busy | taken over the span.
                let mut union = regset_empty();
                for &i in span {
                    union = regset_or(&union, &regset_or(&busy[i], &taken[i]));
                }
                // Find the lowest free register (starting from 2, stepping by width).
                // Python: range(2, top + 2 - width, width).
                let mut r = 2u32;
                let upper = top + 2 - width;
                let found = loop {
                    if r >= upper {
                        break None;
                    }
                    let all_clear = (0..width).all(|k| !regset_test(&union, r + k));
                    if all_clear {
                        break Some(r);
                    }
                    r += width;
                };
                if let Some(r) = found {
                    for k in 0..width {
                        homes.insert(*u + k, r + k);
                        used.insert(r + k);
                    }
                    for &i in span {
                        for k in 0..width {
                            regset_set(&mut taken[i], r + k);
                        }
                    }
                }
            }
        }

        Ok(Borrowed {
            top,
            highest: fd.highest,
            busy,
            taken,
            touch,
            never,
            homes,
            used,
            everything: false,
            slots: HashMap::new(),
            predicate_slots: HashMap::new(),
            frame_bytes: 0,
            depth,
            r1_fixed,
            room: false,
            from_r1: false,
            spill: 0,
        })
    }

    /// The lowest register that holds nothing at instruction `i`, or the lowest run of `width`
    /// registers from an even one: no home of a uniform register alive there, and not among
    /// `given` (a bit set: what the same instruction was given already). None when there is none.
    pub(super) fn free_at(&self, i: usize, width: u32, given: &RegSet) -> Option<u32> {
        let held = regset_or(&regset_or(&self.busy[i], &self.taken[i]), given);
        // Python: range(2, top + 2 - width, 1 if width == 1 else 2).
        let upper = self.top + 2 - width;
        let step = if width == 1 { 1u32 } else { 2 };
        let mut r = 2u32;
        loop {
            if r >= upper {
                return None;
            }
            let all_clear = (0..width).all(|k| !regset_test(&held, r + k));
            if all_clear {
                return Some(r);
            }
            r += step;
        }
    }

    /// Where no register holds nothing: the lowest register, or run of `width` from an even
    /// one, that instruction `i` neither reads nor writes itself, that is no home there, was
    /// not given already, and receives no result that nothing announces: registers that hold
    /// something for the program, to be put in the frame for the time of the sequence and
    /// brought back after it (`Ctx::make_room`). None when there is none.
    pub(super) fn victims_at(&self, i: usize, width: u32, given: &RegSet) -> Option<u32> {
        // Python: held = touch[i] | taken[i] | given | never | 3
        let mut held = regset_or(&regset_or(&self.touch[i], &self.taken[i]), given);
        held = regset_or(&held, &self.never);
        // R0 and R1 never.
        regset_set(&mut held, 0);
        regset_set(&mut held, 1);
        let _mask = (1u128 << width) - 1;
        // Python: range(2, highest + 2 - width, 1 if width == 1 else 2)
        let upper = self.highest + 2 - width;
        let step = if width == 1 { 1u32 } else { 2 };
        let mut r = 2u32;
        loop {
            if r >= upper {
                return None;
            }
            let all_clear = (0..width).all(|k| !regset_test(&held, r + k));
            if all_clear {
                return Some(r);
            }
            r += step;
        }
    }

    /// A register that holds nothing at any instruction of `span` and is no home there:
    /// taken through all of it. For a uniform predicate held in a register between the
    /// instructions that name it, at level 3. None when there is none.
    pub(super) fn kept(&mut self, span: &[usize]) -> Option<u32> {
        let mut union = regset_empty();
        for &i in span {
            union = regset_or(&union, &regset_or(&self.busy[i], &self.taken[i]));
        }
        // Python: range(2, top + 1).
        for r in 2..=self.top {
            if !regset_test(&union, r) {
                for &i in span {
                    regset_set(&mut self.taken[i], r);
                }
                self.used.insert(r);
                return Some(r);
            }
        }
        None
    }

    /// A word of the frame for each uniform register that has no home (the two of a pair
    /// side by side, the lower one first). Returns the size of the frame in bytes.
    pub(super) fn assign_slots(&mut self, uniform: &[u32], pairs: &HashSet<u32>) -> u32 {
        let mut slot = 0u32;
        for &u in uniform {
            if self.homes.contains_key(&u) || (u > 0 && pairs.contains(&(u - 1))) {
                continue;
            }
            if pairs.contains(&u) {
                self.slots.insert(u, slot);
                self.slots.insert(u + 1, slot + 1);
                slot += 2;
            } else {
                self.slots.insert(u, slot);
                slot += 1;
            }
        }
        self.frame_bytes = 4 * slot;
        if !self.frame_bytes.is_multiple_of(16) {
            self.frame_bytes = (self.frame_bytes + 15) & !15;
        }
        self.frame_bytes
    }

    pub(super) fn slot_count(&self) -> u32 {
        (self.slots.len() + self.predicate_slots.len()) as u32
    }
}
