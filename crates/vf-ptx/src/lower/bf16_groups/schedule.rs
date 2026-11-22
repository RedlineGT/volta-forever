//! Move independent register calculations ahead of products sharing one left fragment.
//! Keep both the calculation order and the product order. A gap can only cross a
//! product if neither instruction observes a register value changed by the other.
use super::{Item, Kind, Product, product};
use crate::lower::types::{Declarations, integer};
use crate::scan;
use std::collections::HashSet;

fn products<const N: usize>(
    text: &str,
    items: &[Item],
    first: &Product<'_>,
    outer: &Declarations,
) -> Option<[usize; N]> {
    let mut indices = [0; N];
    let mut count = 1;
    let mut read = HashSet::new();
    let mut written = HashSet::new();
    let mut prepared = HashSet::new();
    let mut local = None;
    let mut feeds_product = false;
    let left = &first.a;
    read.extend(first.a.iter().chain(&first.b).chain(&first.c).copied());
    written.extend(first.d.iter().copied());

    // Bound lookahead even in a function containing thousands of independent instructions.
    for (index, item) in items.iter().enumerate().take(256).skip(1) {
        if let Some(p) = product(item, text) {
            if local.is_some() || &p.a != left {
                return None;
            }
            feeds_product |= p.b.iter().any(|r| prepared.contains(r));
            read.extend(p.a.into_iter().chain(p.b).chain(p.c));
            written.extend(p.d);
            indices[count] = index;
            count += 1;
            if count == indices.len() {
                return feeds_product.then_some(indices);
            }
            continue;
        }
        match &item.kind {
            Kind::Open if local.is_none() => local = Some(Declarations::default()),
            Kind::Close if local.is_some() => local = None,
            Kind::Statement {
                guarded: false,
                opcode,
                operands,
            } => {
                let operands = &text[operands.clone()];
                let arity = match &text[opcode.clone()] {
                    ".reg" => {
                        local.as_mut()?.declare(operands);
                        continue;
                    }
                    "mov.b32" | "mov.u32" | "mov.f32" => 2,
                    "and.b32" | "or.b32" | "shr.u32" => 3,
                    "fma.rn.bf16x2" => 4,
                    _ => return None,
                };
                let args = scan::split_operands(operands);
                let (dest, sources) = args.split_first()?;
                let declared = |r: &str| {
                    scan::register(r).is_some()
                        && local.as_ref().and_then(|d| d.width(r)).or_else(|| outer.width(r)) == Some(32)
                };
                if args.len() != arity || !declared(dest) || read.contains(dest) || written.contains(dest) {
                    return None;
                }
                for src in sources {
                    if integer(src).is_none() && (!declared(src) || written.contains(src)) {
                        return None;
                    }
                }
                if local.as_ref().is_none_or(|d| d.width(dest).is_none()) {
                    prepared.insert(*dest);
                }
            }
            _ => return None,
        }
    }
    None
}

pub(super) fn prepare(text: &str, items: &[Item]) -> Option<(String, HashSet<usize>)> {
    let mut declarations = Declarations::default();
    let mut depth = 0;
    let mut i = 0;
    let mut copied = 0;
    let mut starts = HashSet::new();
    let mut out = String::new();
    while i < items.len() {
        let item = &items[i];
        match &item.kind {
            Kind::Open => depth += 1,
            Kind::Close => depth -= 1,
            Kind::Statement {
                guarded: false,
                opcode,
                operands,
            } if depth == 0 && &text[opcode.clone()] == ".reg" => {
                declarations.declare(&text[operands.clone()]);
            }
            _ if depth == 0 => {
                if let Some(p) = product(item, text) {
                    // A pair already amortizes the guard when its left fragment is shared.
                    // Prefer four products if the intervening calculations can all be moved.
                    let rest = &items[i..];
                    let four = products::<4>(text, rest, &p, &declarations);
                    let two;
                    let group: &[usize] = match &four {
                        Some(indices) => indices,
                        None => {
                            two = products::<2>(text, rest, &p, &declarations);
                            let Some(indices) = &two else {
                                i += 1;
                                continue;
                            };
                            indices
                        }
                    };
                    out.push_str(&text[copied..item.range.start]);
                    for pair in group.windows(2) {
                        out.push_str(&text[items[i + pair[0]].range.end..items[i + pair[1]].range.start]);
                    }
                    starts.insert(out.len());
                    for &offset in group {
                        out.push_str(&text[items[i + offset].range.clone()]);
                        out.push('\n');
                    }
                    i += group.last().unwrap();
                    copied = items[i].range.end;
                }
            }
            _ => {}
        }
        i += 1;
    }
    if copied == 0 {
        return None;
    }
    out.push_str(&text[copied..]);
    Some((out, starts))
}

#[cfg(test)]
mod tests {
    use super::*;

    const OP: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%d0,%d1,%d2,%d3},{%a0,%a1,%a2,%a3},{%b0,%b1},{%c0,%c1,%c2,%c3};";
    fn with_gap(gap: &str) -> Option<String> {
        let later = OP.replace("%b0", "%b2");
        let text = format!("{{.reg .b32 %d<4>,%a<4>,%b<4>,%c<4>; {OP} {gap} {later} {later} {later}}}");
        let items = scan::items(text.as_bytes(), 0..text.len()).unwrap();
        prepare(&text, &items).map(|(s, _)| s)
    }

    #[test]
    fn moves_complete_scopes_without_reordering_products_or_gap_calculations() {
        let gap = "{.reg .b32 local; and.b32 local,%b1,15; or.b32 %b2,local,7;}";
        let moved = with_gap(gap).unwrap();
        assert!(moved.find(gap).unwrap() < moved.find(OP).unwrap());
        assert_eq!(moved.matches(gap).count(), 1);
        assert_eq!(moved.matches("mma.sync").count(), 4);
    }

    #[test]
    fn declines_dependencies_scope_changes_effects_and_volatile_sources() {
        let feed = "mov.b32 %b2,%b3;";
        for fence in [
            "mov.b32 %b3,%d1;",
            "mov.u32 %d0,1;",
            "mov.u32 %c3,1;",
            "mov.u32 %a3,1;",
            "mov.u32 %b1,1;",
            "mov.u32 %b3,%clock;",
            "mov.b32 %b3,%b+1;",
            "L:",
            "bra L;",
            "ld.global.u32 %b3,[%a0];",
            "st.global.u32 [%a0],%b3;",
            "bar.sync 0;",
            "@%p mov.u32 %b3,1;",
            "{.reg .b32 local; mov.b32 %b3,local;} mov.b32 %b3,local;",
            ".loc 1 2 0\n",
        ] {
            assert!(with_gap(&format!("{fence} {feed}")).is_none(), "{fence}");
        }
        assert!(with_gap("").is_none());
    }
}
