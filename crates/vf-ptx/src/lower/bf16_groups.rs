//! Convert consecutive BF16 MMAs to FP16 MMAs when every participating lane has exactly
//! representable inputs. Keep the original operations behind a warp-uniform branch for all
//! other inputs. A group shares one vote and converts each distinct multiplicand once.
mod region;
mod row;
mod schedule;

use super::{Bf16Row, Options, types::Declarations};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

use super::parse::{self, Tensor};
use crate::scan::{self, Item, Kind};

const MMA: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";

struct Product<'a> {
    d: Vec<&'a str>,
    a: Vec<&'a str>,
    b: Vec<&'a str>,
    c: Vec<&'a str>,
}

fn product<'a>(item: &Item, text: &'a str) -> Option<Product<'a>> {
    let Kind::Statement {
        guarded: false,
        opcode,
        operands,
    } = &item.kind
    else {
        return None;
    };
    if &text[opcode.clone()] != MMA {
        return None;
    }
    let Tensor::Sums { d, a, b, c, .. } = parse::parse(MMA, &text[operands.clone()], &HashMap::new()).ok()? else {
        return None;
    };
    Some(Product { d, a, b, c })
}

fn mov_destination<'a>(item: &Item, text: &'a str) -> Option<&'a str> {
    let Kind::Statement {
        guarded: false,
        opcode,
        operands,
    } = &item.kind
    else {
        return None;
    };
    if &text[opcode.clone()] != "mov.b32" {
        return None;
    }
    let (d, s) = text[operands.clone()].split_once(',')?;
    scan::register(s.trim())?;
    scan::register(d.trim())
}

/// Reject a packed word if either half is nonzero and its absolute BF16 encoding is
/// outside 0x3880..=0x477f (the interval that converts to a *normal* FP16). With the
/// two sign bits cleared, each packed add/sub below stays within its own 16-bit half.
/// Folding the sign bits into a single predicate saves a warp-wide vote for every MMA.
fn write_guard(out: &mut String, reg: &str, id: usize, sources: &BTreeMap<&str, usize>) {
    let count = sources.len();
    writeln!(
        out,
        ".reg .b32 {reg}abs,{reg}nz,{reg}ge,{reg}le,{reg}ok,{reg}any,{reg}x,{reg}y;\n\
         .reg .pred {reg}bad,{reg}warp;\n\
         .reg .b32 {reg}h<{count}>;\n\
         mov.b32 {reg}any,0;"
    )
    .unwrap();
    for &src in sources.keys() {
        writeln!(
            out,
            "and.b32 {reg}abs,{src},0x7fff7fff;\n\
             add.u32 {reg}nz,{reg}abs,0x7fff7fff;\n\
             add.u32 {reg}ge,{reg}abs,0x47804780;\n\
             sub.u32 {reg}le,0xc77fc77f,{reg}abs;\n\
             and.b32 {reg}ok,{reg}ge,{reg}le;\n\
             not.b32 {reg}ok,{reg}ok;\n\
             and.b32 {reg}ok,{reg}ok,{reg}nz;\n\
             or.b32 {reg}any,{reg}any,{reg}ok;"
        )
        .unwrap();
    }
    writeln!(
        out,
        "and.b32 {reg}any,{reg}any,0x80008000;\n\
         setp.ne.u32 {reg}bad,{reg}any,0;\n\
         vote.sync.any.pred {reg}warp,{reg}bad,0xffffffff;\n\
         @{reg}warp bra.uni $bfgroup_slow_{id};"
    )
    .unwrap();
    // The low four BF16 exponent bits and seven fraction bits shift left by three.
    // Bit 14 is already the high bit of the target FP16 exponent. Both signs stay in place.
    for (&src, n) in sources {
        writeln!(
            out,
            "shl.b32 {reg}x,{src},3;\n\
             and.b32 {reg}x,{reg}x,0x3ff83ff8;\n\
             and.b32 {reg}y,{src},0xc000c000;\n\
             or.b32 {reg}h{n},{reg}x,{reg}y;"
        )
        .unwrap();
    }
}

pub(super) fn rewrite(text: &str, items: &[Item], outside: &[String], options: Options) -> Option<(String, usize)> {
    if !text.contains(MMA) || text.contains("bfgroup_") || outside.iter().any(|s| s.contains("bfgroup_")) {
        return None;
    }
    if let Some((scheduled, starts)) = schedule::prepare(text, items) {
        let scheduled_items = scan::items(scheduled.as_bytes(), 0..scheduled.len())?;
        return group(&scheduled, &scheduled_items, &starts, options);
    }
    group(text, items, &HashSet::new(), options)
}

fn group(text: &str, items: &[Item], scheduled: &HashSet<usize>, options: Options) -> Option<(String, usize)> {
    let mut out = String::new();
    let mut copied = 0;
    let mut first = 0;
    let mut groups = 0;
    let mut added_products = 0;
    let mut declarations = Declarations::default();
    let small_b = if options.bf16_row != Bf16Row::Off {
        super::known_bits::nibbles::at_products(text, items, MMA)
    } else {
        HashSet::new()
    };
    let mut depth = 0;
    while first < items.len() {
        match &items[first].kind {
            Kind::Open => depth += 1,
            Kind::Close => depth -= 1,
            Kind::Statement {
                guarded: false,
                opcode,
                operands,
            } if depth == 0 && options.bf16_row != Bf16Row::Off && &text[opcode.clone()] == ".reg" => {
                declarations.declare(&text[operands.clone()]);
            }
            _ => {}
        }
        if depth != 0 || product(&items[first], text).is_none() {
            first += 1;
            continue;
        }
        let mut products = Vec::new();
        let mut sources = BTreeMap::new();
        let mut destinations = HashSet::new();
        let mut last = first;
        for (idx, item) in items.iter().enumerate().skip(first) {
            if let Some(p) = product(item, text) {
                // The guard reads every multiplicand before the group. Copied or produced
                // intermediates must therefore never redefine any of those registers.
                if p.a.iter().chain(&p.b).any(|s| destinations.contains(s))
                    || p.d
                        .iter()
                        .any(|s| sources.contains_key(s) || p.a.contains(s) || p.b.contains(s))
                {
                    break;
                }
                for &s in p.a.iter().chain(&p.b) {
                    let n = sources.len();
                    sources.entry(s).or_insert(n);
                }
                destinations.extend(p.d.iter().copied());
                products.push((idx, p));
                last = idx;
                if products.len() == 16 {
                    break;
                }
            } else if let Some(d) = mov_destination(item, text) {
                if sources.contains_key(d) {
                    break;
                }
                destinations.insert(d);
            } else if !matches!(item.kind, Kind::Line) || !crate::deps::debug_location(&text[item.range.clone()]) {
                break;
            }
        }
        if options.bf16_row != Bf16Row::Off
            && let Some((batch, small)) = row::batch(
                &products,
                &declarations,
                &small_b,
                scheduled.contains(&items[first].range.start),
            )
        {
            if options.bf16_row == Bf16Row::Tree
                && small
                && batch.len() == 4
                && let Some(region) = region::Region::find(text, items, first, &declarations, &small_b)
            {
                let begin = items[first].range.start;
                out.push_str(&text[copied..begin]);
                region.write(&mut out, text, items, groups);
                groups += 2 * region.rounds.len();
                first = region.end() + 1;
                copied = items[first - 1].range.end;
                continue;
            }
            let begin = items[first].range.start;
            let last_product = batch.last().unwrap().0;
            let end = items[last_product].range.end;
            out.push_str(&text[copied..begin]);
            row::write(
                &mut out,
                groups,
                batch,
                options.bf16_row,
                small,
                row::Accumulator::Guarded(&text[begin..end]),
            );
            copied = end;
            groups += 1;
            first = last_product + 1;
            continue;
        }
        if products.len() < 4 && !(products.len() == 2 && scheduled.contains(&items[first].range.start)) {
            first += 1;
            continue;
        }
        if !options.bf16_tc_groups {
            first = last + 1;
            continue;
        }
        // In the V100 measurements, groups sharing a complete multiplicand amortized
        // the guard even with four outputs. For groups that change both fragments, small
        // accumulator sets were faster on the scalar path.
        let common_a = products.iter().all(|(_, p)| p.a == products[0].1.a);
        let common_b = products.iter().all(|(_, p)| p.b == products[0].1.b);
        let output_registers: HashSet<_> = products.iter().flat_map(|(_, p)| &p.d).collect();
        if !common_a && !common_b && output_registers.len() < 32 {
            first = last + 1;
            continue;
        }
        added_products += products.len();
        let begin = items[first].range.start;
        let end = items[last].range.end;
        out.push_str(&text[copied..begin]);
        let id = groups;
        groups += 1;
        // Keep declarations and labels in the body's outer scope so that RDC can prove
        // equivalence of weak definitions with different .loc records. Group order is stable.
        let reg = format!("%bfgroup_{id}_");
        write_guard(&mut out, &reg, id, &sources);
        let mut prev = begin;
        for (idx, p) in products {
            out.push_str(&text[prev..items[idx].range.start]);
            let convert = |v: &[&str]| {
                v.iter()
                    .map(|s| format!("{reg}h{}", sources[s]))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            writeln!(
                out,
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{}}},{{{}}},{{{}}},{{{}}};",
                p.d.join(","),
                convert(&p.a),
                convert(&p.b),
                p.c.join(",")
            )
            .unwrap();
            prev = items[idx].range.end;
        }
        writeln!(
            out,
            "bra.uni $bfgroup_done_{id};\n$bfgroup_slow_{id}:\n{}\n$bfgroup_done_{id}:\n",
            &text[begin..end]
        )
        .unwrap();
        copied = end;
        first = last + 1;
    }
    if copied == 0 {
        return None;
    }
    out.push_str(&text[copied..]);
    Some((out, added_products))
}

#[cfg(test)]
mod tests {
    use super::*;

    const OP: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%d0,%d1,%d2,%d3},{%a0,%a1,%a2,%a3},{%b0,%b1},{%c0,%c1,%c2,%c3};";

    fn scoped(source: &str, outside: &[String]) -> Option<String> {
        let text = format!("{{\n{source}\n}}");
        let items = scan::items(text.as_bytes(), 0..text.len()).unwrap();
        rewrite(
            &text,
            &items,
            outside,
            Options {
                bf16_tc_groups: true,
                ..Default::default()
            },
        )
        .map(|(text, _)| text)
    }

    #[test]
    fn preserves_the_slow_operations_and_converts_each_unique_source_once() {
        let body = [OP, ".loc 1 3 0", OP, "mov.b32 %c0,%d0;", OP, OP].join("\n");
        let out = scoped(&body, &[]).unwrap();
        assert_eq!(out.matches(MMA).count(), 4);
        assert_eq!(out.matches("f32.f16.f16.f32").count(), 4);
        assert_eq!(out.matches("shl.b32 %bfgroup_0_x").count(), 6);
        assert_eq!(out.matches("vote.sync.any.pred").count(), 1);
        assert!(out.contains(&body));
        assert_eq!(out.matches("mov.b32 %c0,%d0;").count(), 2);
    }

    #[test]
    fn refuses_gaps_predicates_and_operand_aliasing() {
        for gap in [
            "L:",
            "bra L;",
            "mov.b32 %b0,%d0;",
            "st.shared.u32 [%address],%c0;",
            ".loc 1 2 0 mov.b32 %b0,%d0;",
        ] {
            let body = [OP, OP, gap, OP, OP].join("\n");
            assert!(scoped(&body, &[]).is_none(), "{gap}");
        }
        let op = OP.replace("%d0,", "%a0,");
        assert!(scoped(&[op.as_str(); 4].join("\n"), &[]).is_none());
        let guarded = format!("@%pred {OP}");
        assert!(scoped(&[guarded.as_str(); 4].join("\n"), &[]).is_none());
        assert!(scoped(&[OP; 3].join("\n"), &[]).is_none());
        assert!(scoped(&[OP; 4].join("\n"), &["bfgroup_name".to_string()]).is_none());
    }

    #[test]
    fn reports_source_products_and_resolves_aliased_registers_before_grouping() {
        let options = super::super::Options {
            bf16_tc_groups: true,
            ..Default::default()
        };
        let lower = |gap: &str, options| {
            let body = format!("{{ .reg .b32 %a<4>,%b<2>,%d<4>,%c<4>; {OP} {OP} {gap} {OP} {OP} }}");
            super::super::body(&body, 64, options).unwrap()
        };
        let (fast, counts) = lower("", options);
        assert!(fast.contains("$bfgroup_slow_0:"));
        assert_eq!(counts.tensor, 4);
        let (default, counts) = lower("", Default::default());
        assert!(!default.contains("$bfgroup_slow_"));
        assert_eq!(counts.tensor, 4);

        // After the second product %a00 writes the same register as %a0 in the
        // third. Taking the half conversion for all four from the initial value is invalid.
        let (aliased, counts) = lower("mov.b32 %a00,%c0;", options);
        assert!(!aliased.contains("$bfgroup_slow_"));
        assert_eq!(counts.tensor, 4);
    }

    #[test]
    fn preserves_outer_control_flow_and_limits_the_group_size() {
        let nested = format!("L: {{ {} }}", [OP; 4].join(" "));
        assert!(scoped(&nested, &[]).is_none());

        let out = scoped(&[OP; 20].join(" "), &[]).unwrap();
        assert_eq!(out.matches("vote.sync.any.pred").count(), 2);
        assert_eq!(out.matches("f32.f16.f16.f32").count(), 20);
        assert_eq!(out.matches(MMA).count(), 20);
    }

    #[test]
    fn groups_shared_fragments_or_wide_accumulator_tiles() {
        let group = |n, shared_a, shared_b, dests| {
            (0..n)
                .map(|i| {
                    let vector = |name: &str, count: usize, tile: usize| {
                        (0..count)
                            .map(|j| format!("%{name}{}", tile * count + j))
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    let d = vector("d", 4, i % dests);
                    let a = vector("a", 4, if shared_a { 0 } else { i % 2 });
                    let b = vector("b", 2, if shared_b { 0 } else { i / 2 });
                    format!("{MMA} {{{d}}},{{{a}}},{{{b}}},{{{d}}};")
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(scoped(&group(4, true, false, 4), &[]).is_some());
        assert!(scoped(&group(4, false, true, 4), &[]).is_some());
        assert!(scoped(&group(8, false, false, 8), &[]).is_some());
        assert!(scoped(&group(8, false, false, 4), &[]).is_none());
    }

    #[test]
    fn packed_bit_conversion_and_guard_cover_every_bf16_input() {
        fn eligible(h: u16) -> bool {
            h & 0x7fff == 0 || (113..=142).contains(&((h >> 7) & 255))
        }
        fn half(h: u16) -> u16 {
            if h & 0x7fff == 0 {
                return h;
            }
            (h & 0x8000) | ((((h >> 7) & 255) - 112) << 10) | ((h & 127) << 3)
        }
        for i in 0..=u16::MAX {
            for other in [0, 0x8000, 0x3880, 0x477f, i.rotate_left(7)] {
                for (lo, hi) in [(i, other), (other, i)] {
                    let x = u32::from(lo) | (u32::from(hi) << 16);
                    let a = x & 0x7fff7fff;
                    let admitted = ((a + 0x7fff7fff) & !((a + 0x47804780) & (0xc77fc77f - a)) & 0x80008000) == 0;
                    assert_eq!(admitted, eligible(lo) && eligible(hi));
                    if admitted {
                        assert_eq!(
                            ((x << 3) & 0x3ff83ff8) | (x & 0xc000c000),
                            u32::from(half(lo)) | (u32::from(half(hi)) << 16)
                        );
                    }
                }
            }
        }
    }
}
