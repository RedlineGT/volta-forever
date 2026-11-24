//! One spelling for each register.
//!
//! The assembler takes a member of a family of registers written with zeros before its number
//! for that member: after `.reg .b32 %d<4>;`, `%d00` is `%d0`, and `%d01` is `%d1`. The
//! rewriting of this module follows registers by their names, as text: what it knows of `%d0`
//! (a chain of products that accumulates in it, a value kept for it, that it holds zero) it
//! would not know of `%d00`, and a statement that reads the accumulator of a chain under that
//! spelling would be given the registers of the program as they were before the chain. So the
//! body is given one spelling first: every such name is written as the family declares it.
//!
//! This is done only where a name can mean nothing but that register. A name of that shape may
//! be declared as something else, and which declaration the assembler then follows depends on
//! the blocks, on the order of the declarations and on where the name is written:
//!
//! - a register declared on its own (`.reg .b32 %d00;` in a block under `%d<4>`: `%d00` is then
//!   the new register there, and `%d0` still the member), or two families with names in common
//!   (`%d<4>` and `%d0<4>`);
//! - a variable (`.shared .b32 %d00;` in a block: `mov.u32 %d1, %d00;` there takes its address;
//!   and with `.shared .b32 %d0;` in the block, `%d00` is still the register of the family
//!   outside, which `%d0` no longer names), a parameter (the address of `ld.param` is looked
//!   for among the parameters), a label.
//!
//! A function that writes a member with zeros and has such a declaration, under either
//! spelling, in its body or outside it, is refused. Without such a spelling it is left as it
//! is, and the declarations of each block are followed as they were.

use std::ops::Range;

use super::Unsupported;
use crate::scan::{self, Kind};

/// What the statements of a function declare, in all its blocks together.
#[derive(Default)]
struct Declared<'a> {
    /// `%name<n>`: the prefix and the count.
    families: Vec<(&'a str, usize)>,
    /// Registers declared on their own.
    single: Vec<&'a str>,
    /// What else the body gives a name to: the variables and parameters of its other
    /// directives (every name in them, their types included), and its labels.
    other: Vec<&'a str>,
}

impl<'a> Declared<'a> {
    fn declare(&mut self, operands: &'a str) {
        let operands = operands.trim();
        let (_, names) = operands.split_once(char::is_whitespace).unwrap_or((operands, ""));
        for part in names.split(',') {
            let part = part.trim();
            match part.split_once('<') {
                Some((name, count)) => {
                    if let Ok(count) = count.trim_end_matches('>').trim().parse::<usize>() {
                        self.families.push((name.trim(), count));
                    }
                }
                None if !part.is_empty() => self.single.push(part),
                None => {}
            }
        }
    }

    /// Fast zero-allocation filter: true only if `name` is a zero-padded member (`%d00` for
    /// `%d<4>`) of at least one declared family.
    fn has_noncanonical_member(&self, name: &str) -> bool {
        let bytes = name.as_bytes();
        let mut digit_start = bytes.len();
        while digit_start > 0 && bytes[digit_start - 1].is_ascii_digit() {
            digit_start -= 1;
        }
        let digits = &bytes[digit_start..];
        if digits.len() < 2 || !digits[..digits.len() - 1].contains(&b'0') {
            return false;
        }
        self.families.iter().any(|(prefix, count)| {
            name.strip_prefix(prefix).is_some_and(|number| {
                number.len() > 1
                    && number.as_bytes()[0] == b'0'
                    && number.bytes().all(|byte| byte.is_ascii_digit())
                    && number.parse::<usize>().is_ok_and(|index| index < *count)
            })
        })
    }

    /// The members of families that `name` can be, each as its family declares it (`%d0` for
    /// `%d00` under `%d<4>`), without repetition.
    fn members(&self, name: &str) -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for (prefix, count) in &self.families {
            let Some(number) = name.strip_prefix(prefix) else {
                continue;
            };
            if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            // (A number too long to read is beyond any family.)
            let Ok(index) = number.parse::<usize>() else {
                continue;
            };
            let member = format!("{prefix}{index}");
            if index < *count && !found.contains(&member) {
                found.push(member);
            }
        }
        found
    }
}

/// The body `text` with every member of a family of registers written as the family declares it,
/// or None when it is already so. `items` are the items of `text`; `outside`, the names that
/// the module has outside the bodies of its functions (parameters, variables), which a body
/// may name.
pub(super) fn canonical(text: &str, items: &[scan::Item], outside: &[String]) -> Result<Option<String>, Unsupported> {
    let mut declared = Declared::default();
    for item in items {
        match &item.kind {
            Kind::Statement { opcode, operands, .. } if &text[opcode.clone()] == ".reg" => {
                declared.declare(&text[operands.clone()]);
            }
            Kind::Statement { opcode, operands, .. } if text[opcode.clone()].starts_with('.') => {
                declared.other.extend(scan::names(&text[operands.clone()]));
            }
            Kind::Label => declared.other.extend(scan::names(&text[item.range.clone()])),
            _ => {}
        }
    }
    if declared.families.is_empty() {
        return Ok(None);
    }
    // The names written otherwise than their family declares them: where, and the member.
    let mut places: Vec<(Range<usize>, String)> = Vec::new();
    for item in items {
        let Kind::Statement { opcode, operands, .. } = &item.kind else {
            continue;
        };
        // (Not in a directive: `.reg` declares, and the others name no register.)
        if text[opcode.clone()].starts_with('.') {
            continue;
        }
        // The guard, before the opcode, and the operands.
        for range in [item.range.start..opcode.start, operands.clone()] {
            let part = &text[range.clone()];
            for name in scan::names(part) {
                if !declared.has_noncanonical_member(name) {
                    continue;
                }
                let members = declared.members(name);
                let Some(member) = members.iter().find(|member| member.as_str() != name) else {
                    continue;
                };
                let refused = |why: &str| {
                    Unsupported(format!(
                        "{name}: a register of a family written with zeros before its number, {why}"
                    ))
                };
                if members.len() > 1 {
                    return Err(refused("which two families of registers can mean"));
                }
                if !name.starts_with('%') {
                    return Err(refused("in a name that may not be a register"));
                }
                let at = range.start + (name.as_ptr() as usize - part.as_ptr() as usize);
                places.push((at..at + name.len(), member.clone()));
            }
        }
    }
    let Some((first, _)) = places.first() else {
        return Ok(None);
    };
    let refused = |why: String| {
        Unsupported(format!(
            "{}: a register of a family written with zeros before its number, {why}",
            &text[first.clone()]
        ))
    };
    // Each name must mean one register in every block: no register declared on its own is named
    // like a member of a family, under any spelling.
    if let Some(single) = declared
        .single
        .iter()
        .find(|single| !declared.members(single).is_empty())
    {
        return Err(refused(format!(
            "in a function where {single} is declared on its own beside a family that has that name"
        )));
    }
    // Nor anything else: a variable, a parameter or a label of that name, here or outside.
    if let Some(other) = declared
        .other
        .iter()
        .copied()
        .chain(outside.iter().map(String::as_str))
        .find(|other| !declared.members(other).is_empty())
    {
        return Err(refused(format!(
            "in a function where {other} is named in another declaration too (a variable, a parameter or a label)"
        )));
    }
    let mut renamed = String::with_capacity(text.len());
    let mut copied = 0;
    for (range, member) in &places {
        renamed.push_str(&text[copied..range.start]);
        renamed.push_str(member);
        copied = range.end;
    }
    renamed.push_str(&text[copied..]);
    Ok(Some(renamed))
}
