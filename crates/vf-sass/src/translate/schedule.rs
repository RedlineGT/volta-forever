//! Scheduling: set the stall of each item so that every instruction waits long enough
//! for the registers it reads and the scoreboards it depends on. The schedule handles
//! what the translation added (saves, restores, waits for borrowed registers) in the
//! same pass as the program's own instructions.

use std::collections::{HashMap, HashSet};

use crate::decode::{Decoded, Kind, LETTER_P};
use crate::generated::fields::*;
use crate::generated::special::*;

use super::{Item, Refused, marked, stall_of, with_stall};

// ---- Scheduling ----

pub(super) fn schedule(items: &mut Vec<Item>, targets: &[u64]) -> Result<(), Refused> {
    // Split into basic blocks
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for (k, item) in items.iter().enumerate() {
        if k > start && item.first && targets.contains(&(16 * item.origin as u64)) {
            blocks.push((start, k));
            start = k;
        }
        if item.d.as_ref().unwrap().kind == Kind::BRU {
            blocks.push((start, k + 1));
            start = k + 1;
        }
    }
    if start < items.len() {
        blocks.push((start, items.len()));
    }

    let mut inserted: Vec<(usize, u128)> = Vec::new();
    let mut t: i32 = 0;
    // When each register or predicate was last written, by which item, and whether through the
    // mask of an R2P of the program (`masked`).
    let mut last: HashMap<(u8, u32), (i32, usize, bool)> = HashMap::new();
    // The writers whose scoreboard an instruction since has waited for.
    let mut waited: HashSet<usize> = HashSet::new();
    for &(a, b) in &blocks {
        // What was written before this block, and when, is known where the code before it goes
        // on into it (past a label, a branch not taken, the mark of a region): an instruction
        // the translation changed or added here is as far from those writers as from the ones
        // of its own block. Where the code before does not go on (or goes through other code
        // first: a call), nothing is known, as for what jumps here.
        let continued = goes_on(items, a);
        if !continued {
            t = 0;
            last.clear();
            waited.clear();
        }
        // The time and kind of the last reader of each register.
        let mut read_at: HashMap<(u8, u32), (i32, Kind)> = HashMap::new();
        for k in a..b {
            // Collect what we need from items[k] before potentially modifying items
            let item_changed = items[k].changed;
            let item_origin = items[k].origin;
            let d = items[k].d.as_ref().unwrap().clone();
            // (With what it reads and writes that its operands do not say.)
            let reads: Vec<(u8, u32, u8)> = d.reads.iter().chain(&items[k].more_reads).copied().collect();
            let writes: Vec<(u8, u32)> = d.writes.iter().chain(&items[k].more_writes).copied().collect();
            // The reads that the translation made up, and those of a mask: nothing of the
            // program's instruction says them.
            let through_mask = all_predicates(&d, &reads);
            let made_up: Vec<(u8, u32, u8)> = items[k]
                .more_reads
                .iter()
                .copied()
                .chain(through_mask.iter().copied())
                .collect();
            let mut need: i32 = 0;
            for &(letter, register, role) in d.reads.iter().chain(&made_up) {
                if let Some(&(wt, wi, through)) = last.get(&(letter, register)) {
                    let wr_changed = items[wi].changed;
                    let wr_word = items[wi].word;
                    if !(wr_changed || item_changed) {
                        continue;
                    }
                    if through && !made_up.contains(&(letter, register, role)) {
                        // (Written through the mask of an R2P of the program, and read by what
                        // the program's instruction read: as far as the compiler had them.)
                        continue;
                    }
                    if ((wr_word >> WRITE_BARRIER) & 7) != 7 || items[wi].covered {
                        // Announced on a scoreboard: the reader waits for it. What the program's
                        // instruction waited for, its translation waits for; a read that the
                        // translation made up (a predicate saved, written down, or all of them
                        // for R2P) has nothing of the program that waits for it: it does here,
                        // unless an instruction since has waited for that scoreboard (`waited`).
                        if made_up.contains(&(letter, register, role))
                            && ((wr_word >> WRITE_BARRIER) & 7) != 7
                            && !waited.contains(&wi)
                        {
                            if !item_changed {
                                return Err(Refused(format!(
                                    "the {} at 0x{:x} reads a predicate of the translation that a scoreboard announces",
                                    d.mnemonic,
                                    16 * item_origin
                                )));
                            }
                            items[k].word |= 1u128 << (WAIT + ((wr_word >> WRITE_BARRIER) & 7) as u32);
                        }
                        // (A result of double precision is not read earlier than Volta's own
                        // code reads one, whatever is waited for: `distance::waited`.)
                        let wd = items[wi].d.as_ref().unwrap();
                        if let Some(floor) = crate::distance::waited(letter, role, wd, d.kind) {
                            need = need.max(wt + floor as i32 - t);
                        }
                        continue;
                    }
                    let wd = items[wi].d.as_ref().unwrap();
                    let dist = crate::distance::least(letter, role, wd, d.kind) as i32;
                    need = need.max(wt + dist - t);
                }
            }
            // A writer that is not plain arithmetic, after a reader of what it writes: the
            // compiler's tables ask for up to 9 cycles between them (2 in most cases); the
            // original had another kind of writer here, with its own distance. (Not a
            // product after a product: they are done in order, and the compiler has the
            // steps of one product two cycles apart, the third writing what the first read.)
            // R2P is kept to this too, for another reason than the one first given for it
            // (2026-11-18, the same day: a reader does NOT go by what an R2P issued after it
            // gives back, at any distance): the instruction before a restore reads
            // what stood in the predicate, so the restore is nine cycles into its block or
            // more, and no writer of a predicate that ran before the block, on a path the
            // schedule does not see, is close enough for the R2P to undo (`all_predicates`).
            // What R2P writes is in a mask: `more_writes`.
            if item_changed && (!matches!(d.kind, Kind::FXU | Kind::IMAD | Kind::FMAI) || d.mnemonic == "R2P") {
                for &(letter, register) in &writes {
                    if let Some(&(rt, reader_kind)) = read_at.get(&(letter, register))
                        && !(d.kind == Kind::HMMA && reader_kind == Kind::HMMA)
                    {
                        need = need.max(rt + 9 - t);
                    }
                }
            }
            if need > 0 {
                if k == 0 || (k == a && !continued) {
                    return Err(Refused(format!(
                        "scheduling: nothing before 0x{:x} to wait on",
                        16 * item_origin
                    )));
                }
                t += delay(items, k - 1, need as u32, &mut inserted) as i32;
            }
            // (As a reader for the nine cycles above: a P2R with the predicates of its mask; not
            // an R2P with those it only writes back.)
            for &(letter, register, _) in &reads {
                read_at.insert((letter, register), (t, d.kind));
            }
            if d.mnemonic == "P2R" {
                for &(letter, register, _) in &through_mask {
                    read_at.insert((letter, register), (t, d.kind));
                }
            }
            // (What this instruction waits for is there when it is issued: the writers that
            // announced on those scoreboards are waited for.)
            let boards = (items[k].word >> WAIT) & 63;
            if boards != 0 {
                for &(_, wi, _) in last.values() {
                    if (boards >> ((items[wi].word >> WRITE_BARRIER) & 7)) & 1 != 0 {
                        waited.insert(wi);
                    }
                }
            }
            for &(letter, register) in &writes {
                last.insert((letter, register), (t, k, false));
            }
            for (letter, register) in masked(&d, &writes) {
                last.insert((letter, register), (t, k, true));
            }
            t += stall_of(items[k].word) as i32;
        }
        // Tail: ensure enough distance for the last writer
        let mut tail_idx = if items[b - 1].d.as_ref().unwrap().kind != Kind::BRU || b - a == 1 {
            b - 1
        } else {
            b - 2
        };
        let mut need: i32 = 0;
        for (&(letter, _register), &(when, wi, _)) in &last {
            let wr_changed = items[wi].changed;
            let wr_word = items[wi].word;
            if wr_changed && ((wr_word >> WRITE_BARRIER) & 7) == 7 && !items[wi].covered && wi != b - 1 {
                let wd = items[wi].d.as_ref().unwrap();
                need = need.max(when + crate::distance::longest(letter, wd) as i32 - t);
            }
        }
        let wd = items[b - 1].d.as_ref().unwrap().clone();
        let writes: Vec<(u8, u32)> = wd.writes.iter().chain(&items[b - 1].more_writes).copied().collect();
        if items[b - 1].changed && !writes.is_empty() && ((items[b - 1].word >> WRITE_BARRIER) & 7) == 7 {
            tail_idx = b - 1;
            for &(l, _) in &writes {
                need = need.max(crate::distance::longest(l, &wd) as i32 - stall_of(items[b - 1].word) as i32);
            }
        }
        if need > 0 {
            t += delay(items, tail_idx, need as u32, &mut inserted) as i32;
        }
    }

    // Insert NOP fillers
    inserted.sort_by(|a, b| b.cmp(a));
    for (k, word) in inserted {
        let origin = items[k].origin;
        // `finish` decodes it.
        items.insert(k + 1, Item::new(word, origin, "wait", true));
    }
    // A wait for a scoreboard comes two cycles or more after the instruction that names it, for
    // its result or for having read: native code of both architectures never has less. The
    // translation adds waits and names.
    for k in 1..items.len() {
        let mask = (items[k].word >> WAIT) & 63;
        if mask == 0 {
            continue;
        }
        let mut distance = 0;
        for j in (0..k).rev() {
            distance += stall_of(items[j].word);
            if distance >= 2 {
                break;
            }
            let named = [
                (items[j].word >> WRITE_BARRIER) & 7,
                (items[j].word >> READ_BARRIER) & 7,
            ];
            if (items[j].changed || items[k].changed || items[j].says)
                && named.iter().any(|&s| s != 7 && (mask >> s) & 1 != 0)
            {
                let before = items[k - 1].word;
                items[k - 1].word = with_stall(before, stall_of(before) + 2 - distance, marked(&items[k - 1].d));
                break;
            }
        }
    }
    Ok(())
}

/// Whether the item before item `a`, the first of a block of the schedule, goes on into it with
/// nothing else run between: an instruction that is not a branch, a branch that may not be taken,
/// the mark of a region (BSSY, BMOV). Not a call (other code runs first), nor anything that leaves.
fn goes_on(items: &[Item], a: usize) -> bool {
    if a == 0 {
        return false;
    }
    let d = items[a - 1].d.as_ref().unwrap();
    if d.kind != Kind::BRU || matches!(d.mnemonic, "BSSY" | "BMOV") {
        return true;
    }
    let always = d.op_or(OP_PG, 7) == 7
        && d.op_or(OP_PG_AT_NOT, 0) == 0
        && d.op_or(OP_PP, 7) == 7
        && d.op_or(OP_PP_AT_NOT, 0) == 0
        && d.op_or(OP_COND, 0) == 0;
    d.mnemonic == "BRA" && !always
}

/// What a P2R or an R2P reads of the predicates, which its operands (`reads`) do not say (they
/// are a mask, a number).
///
/// A P2R reads those of its mask.
///
/// An R2P reads all seven and writes them all back, those of its mask from its register and the
/// others as it read them; so a predicate outside its mask that an instruction wrote less than 4
/// cycles before it (5 for another R2P) gets its former value back. Measured on a V100,
/// 2026-11-18 (ISETP, PLOP3 or the carries of IADD3 1 to 3 cycles before an R2P of
/// another predicate are undone; two R2P of one predicate each, 1 to 4 cycles apart, lose the
/// first). Native sm_70 code has no writer of any predicate closer than 4 cycles before an R2P
/// (8,101 of them in PyTorch's and cuBLAS's). The translation had restores one cycle apart (two
/// predicates saved around one instruction) and so lost the first. (Not the predicates of its
/// mask, which get the register's bits whatever they held; a mask that is not a number: any of
/// them may be left as it was.)
fn all_predicates(d: &Decoded, reads: &[(u8, u32, u8)]) -> Vec<(u8, u32, u8)> {
    let named = |p: u32| reads.iter().any(|&(l, r, _)| l == LETTER_P && r == p);
    let wanted: u32 = match d.mnemonic {
        "P2R" => d.op_or(OP_SB, 0x7f) & 0x7f,
        "R2P" => {
            !(if d.name == "r2p__RIR" {
                d.op_or(OP_SB, 0) & 0x7f
            } else {
                0
            }) & 0x7f
        }
        _ => 0,
    };
    (0..7u32)
        .filter(|&p| (wanted >> p) & 1 != 0 && !named(p))
        .map(|p| (LETTER_P, p, 0u8))
        .collect()
}

/// What an R2P writes of the predicates, which its operands (`writes`) do not say: those of its
/// mask (any of them when the mask is not a number). The program's own R2P is a writer like any
/// other for what the translation reads after it: a predicate saved (P2R) one to four cycles
/// after the program's R2P gave it a value was saved as it was before (cuDNN's kernels had the
/// case, found by validate.py on the held-out corpus the day R2P was measured).
fn masked(d: &Decoded, writes: &[(u8, u32)]) -> Vec<(u8, u32)> {
    if d.mnemonic != "R2P" {
        return Vec::new();
    }
    let mask = if d.name == "r2p__RIR" {
        d.op_or(OP_SB, 0x7f) & 0x7f
    } else {
        0x7f
    };
    (0..7u32)
        .filter(|&p| (mask >> p) & 1 != 0 && !writes.contains(&(LETTER_P, p)))
        .map(|p| (LETTER_P, p))
        .collect()
}

fn delay(items: &mut [Item], k: usize, cycles: u32, inserted: &mut Vec<(usize, u128)>) -> u32 {
    let room = 15 - stall_of(items[k].word);
    let take = room.min(cycles);
    items[k].word = with_stall(items[k].word, stall_of(items[k].word) + take, marked(&items[k].d));
    let mut left = cycles as i32 - take as i32;
    while left > 0 {
        let step = (left as u32).min(15);
        inserted.push((k, with_stall(NOP, step, false)));
        left -= step as i32;
    }
    cycles
}
