//! Stall-count distance lookup: the shortest distance native sm_70 code keeps between
//! a writer and a reader, by kind.
//!
//! Port of `machine.py`'s `Machine.least` and `Machine.longest`.

use crate::decode::{Decoded, Kind};
use crate::generated::distances::{DISTANCES, LONGEST_KINDS};

/// The minimum distance native sm_70 code keeps from `writer` writing a register (letter "R")
/// or predicate (letter "P") with `role` ("data"/"guard") to a reader of kind `reader_kind`.
pub fn least(letter: u8, role: u8, writer: &Decoded, reader_kind: Kind) -> u32 {
    let d = least_in_tables(letter, role, writer, reader_kind);
    if letter == 2 && writer.mnemonic == "R2P" {
        // Predicates given back from a register (the tables do not say which R2P writes: its mask
        // is a number). In native sm_70 code a plain instruction reads one as an operand 5 cycles
        // later or more; the others seen (guards of memory instructions) 12 or more: 12 for
        // everything else.
        return d.max(if reader_kind == Kind::FXU && role == 0 { 5 } else { 12 });
    }
    d
}

fn least_in_tables(letter: u8, role: u8, writer: &Decoded, reader_kind: Kind) -> u32 {
    let letter_str = if letter == 2 { "P" } else { "R" };
    let role_str = if role == 1 { "guard" } else { "data" };
    let reader_str = reader_kind.name();

    // Try specific mnemonic first (e.g., "=DADD")
    let mnemonic_key = format!("={}", writer.mnemonic);
    for &(l, r, w, rk, d) in DISTANCES.iter() {
        if l == letter_str && r == role_str && w == mnemonic_key && rk == reader_str {
            return d as u32;
        }
    }

    // Then try by kind
    let writer_kind_str = writer.kind.name();
    for &(l, r, w, rk, d) in DISTANCES.iter() {
        if l == letter_str && r == role_str && w == writer_kind_str && rk == reader_str {
            return d as u32;
        }
    }

    // Never seen together: the longest distance this kind of writer keeps from anything, or 15.
    let mut max_d = 0u32;
    let mut found = false;
    for &(l, _r, w, _rk, d) in DISTANCES.iter() {
        if l == letter_str && w == writer_kind_str {
            max_d = max_d.max(d as u32);
            found = true;
        }
    }
    if found { max_d } else { 15 }
}

/// The distance that native sm_70 code keeps from `writer` to a reader that waits for the
/// scoreboard on which it announces its result, where a wait is not all it keeps: `None` for every
/// writer but the four of double precision. A reader of what DADD, DMUL, DFMA or DSETP writes
/// never comes closer than this in Volta's own code, scoreboard or not (8 cycles to an instruction
/// of the same pipe, 9 to any other; a predicate of DSETP 15 as a guard or to a branch), where
/// code for sm_80 has 6 and 14 (docs/validation.md, section 8). Port of `Machine.waited`.
pub fn waited(letter: u8, role: u8, writer: &Decoded, reader_kind: Kind) -> Option<u32> {
    if !matches!(writer.mnemonic, "DADD" | "DMUL" | "DFMA" | "DSETP") {
        return None;
    }
    if letter == 2 && (role == 1 || reader_kind == Kind::BRU) {
        return Some(15);
    }
    Some(if reader_kind == Kind::FMALITE { 8 } else { 9 })
}

/// `waited` for any reader: what is kept from `writer` to the end of straight-line code, where
/// the reader is not seen (it may be the first instruction of a branch's target). Port of
/// `Machine.waited_longest`.
pub fn waited_longest(letter: u8, writer: &Decoded) -> Option<u32> {
    if !matches!(writer.mnemonic, "DADD" | "DMUL" | "DFMA" | "DSETP") {
        return None;
    }
    Some(if letter == 2 { 15 } else { 9 })
}

/// The distance that is enough from `writer` to any reader of what it writes.
pub fn longest(letter: u8, writer: &Decoded) -> u32 {
    let mut max_d = 0u32;
    let roles: &[u8] = if letter == 2 { &[0, 1] } else { &[0] };
    for &role in roles {
        for &kind_str in LONGEST_KINDS.iter() {
            let kind = match kind_str {
                "FXU" => Kind::FXU,
                "IMAD" => Kind::IMAD,
                "FMAI" => Kind::FMAI,
                "FMALITE" => Kind::FMALITE,
                "MIO_FAST" => Kind::MioFast,
                "MIO_SLOW" => Kind::MioSlow,
                "BRU" => Kind::BRU,
                "HMMA" => Kind::HMMA,
                _ => continue,
            };
            max_d = max_d.max(least(letter, role, writer, kind));
        }
    }
    max_d
}
