//! Signatures used by indirect calls. A missing signature can match every device function.
use crate::scan;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct Signature {
    returns: Vec<Option<u8>>,
    inputs: Vec<Option<u8>>,
}

/// Recognise ordinary scalar argument declarations. PTX can also pass aligned byte arrays,
/// use pointer qualifiers and specify attributes; leave unfamiliar declarations unconstrained.
fn params(text: &str) -> Option<(Vec<Option<u8>>, &str)> {
    let Some(text) = text.trim_start().strip_prefix('(') else {
        return Some((Vec::new(), text.trim_start()));
    };
    let (list, rest) = text.split_once(')')?;
    let mut widths = Vec::new();
    for param in scan::split_operands(list) {
        let mut words = param.split_ascii_whitespace();
        if !matches!(words.next(), Some(".reg" | ".param")) {
            return None;
        }
        let width = match words.next()? {
            ".b32" | ".s32" | ".u32" | ".f32" => Some(32),
            ".b64" | ".s64" | ".u64" | ".f64" => Some(64),
            ".pred" | ".b8" | ".s8" | ".u8" | ".b16" | ".s16" | ".u16" | ".f16" | ".bf16" => None,
            _ => return None,
        };
        let name = words.next()?;
        if !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'$' | b'%'))
            || words.next().is_some()
        {
            return None;
        }
        widths.push(width);
    }
    Some((widths, rest.trim_start()))
}

impl Signature {
    /// Both a .func and a .callprototype use [return-list] name [parameter-list].
    pub(super) fn parse(text: &str, name: &str) -> Option<Self> {
        let (returns, text) = params(text)?;
        let (inputs, remaining) = params(text.strip_prefix(name)?)?;
        matches!(remaining, "" | ".noreturn").then_some(Self { returns, inputs })
    }

    pub(super) fn compatible(&self, other: &Self) -> bool {
        fn compatible(a: &[Option<u8>], b: &[Option<u8>]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.is_none() || y.is_none() || x == y)
        }
        compatible(&self.returns, &other.returns) && compatible(&self.inputs, &other.inputs)
    }
}

/// A .calltargets list already references all its possible functions. A prototype specifies
/// the ABI only. Do not treat an unused prototype declaration as an indirect call.
pub(super) fn used(source: &str, items: &[scan::Item]) -> HashSet<Option<Signature>> {
    let mut prototypes = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let scan::Kind::Statement { opcode, operands, .. } = &item.kind else {
            continue;
        };
        if &source[opcode.clone()] != ".callprototype" {
            continue;
        }
        let Some(label) = index
            .checked_sub(1)
            .and_then(|i| items.get(i))
            .filter(|i| i.kind == scan::Kind::Label)
        else {
            return HashSet::from([None]);
        };
        let name = source[label.range.clone()].trim_end_matches(':').trim();
        // Labels can be reused in different lexical scopes. Retain every possibility.
        prototypes
            .entry(name)
            .and_modify(|s| *s = None)
            .or_insert_with(|| Signature::parse(&source[operands.clone()], "_"));
    }
    let mut used = HashSet::new();
    if !prototypes.is_empty() {
        for item in items {
            let scan::Kind::Statement { opcode, operands, .. } = &item.kind else {
                continue;
            };
            if source[opcode.clone()].split('.').next() == Some("call")
                && let Some(proto) = scan::split_operands(&source[operands.clone()])
                    .last()
                    .and_then(|target| prototypes.get(target))
            {
                used.insert(proto.clone());
            }
        }
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incompatible_scalar_inputs_or_returns_exclude_a_target() {
        let parse = |s| Signature::parse(s, "_").unwrap();
        let u32 = parse("_ (.param .u32 word)");
        assert!(u32.compatible(&parse("_ (.reg .f32 val)")));
        assert!(!u32.compatible(&parse("_ (.reg .b64 ptr)")));
        assert!(!u32.compatible(&parse("_ ()")));
        assert!(!u32.compatible(&parse("(.param .b32 result) _ (.reg .b32 ptr)")));
        assert!(!parse("(.reg .b64 x) _").compatible(&parse("(.reg .b32 x) _")));
        assert!(!u32.compatible(&parse("_ (.reg .b32 a, .reg .b32 b)")));
        assert!(u32.compatible(&parse("_ (.reg .u16 val)")));
        assert!(Signature::parse("_ (.param .align 8 .b8 _[32])", "_").is_none());
        assert!(Signature::parse("_ (.reg .b32 a) .new_attribute 1", "_").is_none());
    }
}
