//! Factor equal guards out of a sequence of diamonds that execute zero, one or two BF16
//! products. Each path keeps its original scalar preparation, comparisons and product order.
//! For a defined use of mma.sync.aligned, all warp lanes must execute the same product.
//! Consequently the two predicates already agree across every participating warp.
//! The preparation contains no memory access or collective that could observe reconvergence.
use std::collections::HashMap;
use std::fmt::Write;

use super::Counts;
use super::types::{Declarations, integer};
use crate::scan::{self, Item, Kind};

const MMA: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";

struct Code<'a> {
    text: &'a str,
    items: &'a [Item],
    registers: Declarations,
    label_uses: HashMap<&'a str, usize>,
}

#[derive(PartialEq)]
struct Condition<'a> {
    outer: &'a str,
    opcode: &'a str,
    value: &'a str,
    immediate: &'a str,
}

struct Diamond<'a> {
    start: usize,
    condition: Condition<'a>,
    predicate: &'a str,
    join: &'a str,
}

impl<'a> Code<'a> {
    fn source(&self, i: usize) -> &'a str {
        &self.text[self.items[i].range.clone()]
    }

    fn statement(&self, i: usize) -> Option<(&'a str, &'a str, &'a str)> {
        let item = self.items.get(i)?;
        let Kind::Statement { opcode, operands, .. } = &item.kind else {
            return None;
        };
        Some((
            self.text[item.range.start..opcode.start].trim(),
            &self.text[opcode.clone()],
            self.text[operands.clone()].trim(),
        ))
    }

    fn guarded_branch(&self, i: usize) -> Option<(&'a str, &'a str)> {
        let (guard, "bra", label) = self.statement(i)? else {
            return None;
        };
        let pred = scan::register(guard.strip_prefix('@')?)?;
        (self.registers.width(pred) == Some(1)).then_some((pred, scan::register(label)?))
    }

    // @outer bra join; setp.eq inner, value, constant; mma; @inner bra join; mma; join:
    fn diamond(&self, i: usize) -> Option<Diamond<'a>> {
        let (outer, join) = self.guarded_branch(i)?;
        let ("", opcode @ ("setp.eq.u32" | "setp.eq.s32"), operands) = self.statement(i + 1)? else {
            return None;
        };
        let [pred, value, immediate]: [&str; 3] = scan::split_operands(operands).try_into().ok()?;
        if pred == outer
            || self.registers.width(value) != Some(32)
            || integer(immediate).is_none()
            || self.guarded_branch(i + 3)? != (pred, join)
            || self.items.get(i + 5)?.kind != Kind::Label
            || self.source(i + 5).strip_suffix(':')?.trim_end() != join
        {
            return None;
        }
        for index in [i + 2, i + 4] {
            let ("", MMA, args) = self.statement(index)? else {
                return None;
            };
            let [d, a, b, c]: [&str; 4] = scan::split_operands(args).try_into().ok()?;
            for (operand, expected, destination) in [(d, 4, true), (a, 4, false), (b, 2, false), (c, 4, false)] {
                let regs = scan::vector(operand)?;
                if regs.len() != expected
                    || regs.iter().any(|reg| self.registers.width(reg) != Some(32))
                    || destination && regs.contains(&value)
                {
                    return None;
                }
            }
        }
        Some(Diamond {
            start: i,
            condition: Condition {
                outer,
                opcode,
                value,
                immediate,
            },
            predicate: pred,
            join,
        })
    }

    /// Return the next outer instruction after an uninterrupted sequence of scalar operations.
    /// A nested block may declare a single temporary. No control flow, volatile register,
    /// memory operand, collective, predicate update or partial scope can be cloned.
    fn after_scalars(&self, mut i: usize, condition: &Condition) -> Option<(usize, usize)> {
        let mut nested = false;
        let mut local = None;
        let mut packed_fmas = 0;
        let start = i;
        while let Some(item) = self.items.get(i) {
            match item.kind {
                Kind::Open if !nested => nested = true,
                Kind::Close if nested => {
                    nested = false;
                    local = None;
                }
                Kind::Statement { .. } => {
                    let (guard, op, args) = self.statement(i)?;
                    let arity = match (guard, op) {
                        ("", ".reg") if nested && local.is_none() => {
                            let (ty, name) = args.split_once(char::is_whitespace)?;
                            let name = scan::register(name.trim())?;
                            if ty != ".b32" || self.registers.width(name).is_some() || name.starts_with('%') {
                                return None;
                            }
                            local = Some(name);
                            i += 1;
                            continue;
                        }
                        ("", "mov.b32" | "mov.u32") => 2,
                        ("", "shr.u32" | "and.b32" | "or.b32") => 3,
                        ("", "fma.rn.bf16x2") => 4,
                        _ => break,
                    };
                    let args = scan::split_operands(args);
                    if args.len() != arity || args[0] == condition.value {
                        return None;
                    }
                    let register = |s: &str| {
                        scan::register(s).is_some() && (local == Some(s) || self.registers.width(s) == Some(32))
                    };
                    if !register(args[0]) || !args[1..].iter().all(|s| register(s) || integer(s).is_some()) {
                        return None;
                    }
                    packed_fmas += usize::from(op == "fma.rn.bf16x2");
                }
                _ => break,
            }
            i += 1;
            if i - start > 128 {
                return None;
            }
        }
        (!nested).then_some((i, packed_fmas))
    }

    fn render(&self, group: &[Diamond], id: usize) -> String {
        let first = &group[0];
        let join = group.last().unwrap().join;
        let mut out = String::new();
        writeln!(
            out,
            "@{} bra $vf_cases_zero_{id};\n{}\n@{} bra $vf_cases_one_{id};",
            first.condition.outer,
            self.source(first.start + 1),
            first.predicate
        )
        .unwrap();
        for n in [2, 1, 0] {
            if n < 2 {
                let label = if n == 1 { "one" } else { "zero" };
                writeln!(out, "$vf_cases_{label}_{id}:").unwrap();
            }
            for (j, d) in group.iter().enumerate() {
                if j > 0 {
                    let prev = group[j - 1].start;
                    out.push_str(&self.text[self.items[prev + 5].range.end..self.items[d.start].range.start]);
                }
                // The first setp already ran at dispatch in both nonempty cases.
                if n > 0 {
                    if j != 0 {
                        writeln!(out, "{}", self.source(d.start + 1)).unwrap();
                    }
                    writeln!(out, "{}", self.source(d.start + 2)).unwrap();
                }
                if n == 2 {
                    writeln!(out, "{}", self.source(d.start + 4)).unwrap();
                }
            }
            if n != 0 {
                writeln!(out, "bra {join};").unwrap();
            }
        }
        out
    }
}

pub(super) fn rewrite<'a>(text: &'a str, items: &'a [Item]) -> Option<(String, Counts)> {
    if !text.contains("setp.eq.") || text.matches(MMA).take(16).count() < 16 {
        return None;
    }
    // Most large MMA kernels have no such branch diamonds. Avoid indexing all label
    // references or resolving register declarations for them.
    let candidates = items.windows(6).filter(|window| {
        window[5].kind == Kind::Label
            && matches!(&window[2].kind, Kind::Statement { opcode, guarded: false, .. } if &text[opcode.clone()] == MMA)
            && matches!(&window[4].kind, Kind::Statement { opcode, guarded: false, .. } if &text[opcode.clone()] == MMA)
            && matches!(&window[0].kind, Kind::Statement { guarded: true, .. })
            && matches!(&window[3].kind, Kind::Statement { guarded: true, .. })
    });
    if candidates.take(8).count() < 8 {
        return None;
    }
    let mut label_uses: HashMap<_, _> = items
        .iter()
        .filter(|x| x.kind == Kind::Label)
        .map(|x| (text[x.range.clone()].trim_end_matches(':').trim_end(), 0))
        .collect();
    for name in scan::names(text) {
        if let Some(count) = label_uses.get_mut(name) {
            *count += 1;
        }
    }
    let mut code = Code {
        text,
        items,
        registers: Declarations::default(),
        label_uses,
    };
    let mut out = String::new();
    let mut first = 0;
    let mut copied = 0;
    let mut groups = 0;
    let mut depth = 0;
    let mut growth = 0;
    let mut extra = Counts::default();
    while first < items.len() {
        match items[first].kind {
            Kind::Open => depth += 1,
            Kind::Close => depth -= 1,
            _ => {}
        }
        if depth == 0 {
            if let Some(("", ".reg", args)) = code.statement(first) {
                code.registers.declare(args);
            }
            if let Some(head) = code.diamond(first) {
                let mut chain = vec![head];
                let mut fmas = 0;
                while chain.len() < 16 {
                    let last = chain.last().unwrap();
                    // Removing an interior join is safe only when its two branches and its
                    // definition are its only uses anywhere in the function (including tables).
                    if code.label_uses[last.join] != 3 {
                        break;
                    }
                    let Some((next, packed)) = code.after_scalars(last.start + 6, &last.condition) else {
                        break;
                    };
                    let Some(d) = code.diamond(next) else { break };
                    if d.condition != last.condition {
                        break;
                    }
                    fmas += packed;
                    chain.push(d);
                }
                let last = chain.last().unwrap().start;
                let begin = items[first].range.start;
                let end = items[last + 5].range.start; // preserve the final join, including external edges
                if chain.len() >= 8 && end - begin <= 64 * 1024 {
                    let rendered = code.render(&chain, groups);
                    let added = rendered.len().saturating_sub(end - begin);
                    if growth + added <= text.len().min(128 * 1024) {
                        out.push_str(&text[copied..begin]);
                        out.push_str(&rendered);
                        growth += added;
                        extra.tensor += chain.len(); // the first product appears in the one and two paths
                        extra.other += 2 * fmas; // scalar preparation is present in all three paths
                        copied = end;
                        groups += 1;
                    }
                }
                first = last + 6;
                continue;
            }
        }
        first += 1;
    }
    if groups == 0 {
        return None;
    }
    out.push_str(&text[copied..]);
    Some((out, extra))
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGISTERS: &str = ".reg .pred %outer,%q<8>; .reg .b32 %n,%a,%b,%tmp,%d<4>;";
    const PRODUCT: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%d0,%d1,%d2,%d3},{%a,%a,%a,%a},{%b,%b},{%d0,%d1,%d2,%d3};";
    const PREP: &str = "{.reg .b32 c; mov.b32 c,%a; fma.rn.bf16x2 %tmp,c,%b,%tmp;}";

    fn region() -> String {
        (0..8)
            .map(|i| format!("@%outer bra L{i}; setp.eq.s32 %q{i},%n,1; {PRODUCT} @%q{i} bra L{i}; {PRODUCT} L{i}:"))
            .collect::<Vec<_>>()
            .join(PREP)
    }

    fn rewrite_body(body: &str) -> Option<(String, Counts)> {
        let t = format!("{{ {REGISTERS} {body} }}");
        let items = scan::items(t.as_bytes(), 0..t.len()).unwrap();
        rewrite(&t, &items)
    }

    #[test]
    fn preserves_external_final_join_and_source_counts() {
        let original = format!("@%outer bra L7; {} mov.b32 %tmp,%a;", region());
        let (coded, extra) = rewrite_body(&original).unwrap();
        assert_eq!(
            extra,
            Counts {
                tensor: 8,
                other: 14,
                tanh: 0
            }
        );
        assert_eq!(coded.matches("L7:").count(), 1);
        assert!(coded.contains("@%outer bra L7;"));
        assert_eq!(coded.matches("bra L7;").count(), 3);
        assert_eq!(coded.matches(PRODUCT).count(), 24);
        let (two, one) = coded.split_once("$vf_cases_one_0:").unwrap();
        let (one, zero) = one.split_once("$vf_cases_zero_0:").unwrap();
        assert_eq!([two, one, zero].map(|s| s.matches(PRODUCT).count()), [16, 8, 0]);
        assert_eq!([two, one, zero].map(|s| s.matches(PREP).count()), [7, 7, 7]);
        assert!(two.contains("setp.eq.s32 %q0,%n,1;") && one.contains("setp.eq.s32 %q7,%n,1;"));
        assert!(!zero.contains("setp.eq"));
        let source = format!("{{ {REGISTERS} {original} }}");
        let (_, counts) = super::super::body(&source, 64, Default::default()).unwrap();
        assert_eq!((counts.tensor, counts.other), (16, 7));
    }

    #[test]
    fn refuses_modified_guards_mma_destinations_and_internal_edges() {
        let body = region();
        for bad in [
            body.replacen("@%q0 bra", "@%q1 bra", 1),
            body.replacen("%q3,%n,1", "%q3,%a,1", 1),
            body.replacen("%q3,%n,1", "%q3,%n,2", 1),
            body.replacen("setp.eq.s32 %q4", "setp.eq.s32 %outer", 1),
            body.replacen("{%d0,%d1", "{%n,%d1", 1),
            body.replacen("{%a,%a", "{%clock,%a", 1),
            body.replacen("@%outer bra L4", "@!%outer bra L4", 1),
            body.replacen("@%q3 bra L3", "@%q3 bra L4", 1),
            format!("bra L4; {body}"),
            format!(".branchtargets L0,L1; {body}"),
        ] {
            assert!(rewrite_body(&bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn refuses_effectful_preparation_partial_scopes_and_volatile_registers() {
        let body = region();
        for gap in [
            "mov.b32 %tmp,%clock;",
            "mov.b32 %tmp,%envreg0;",
            "mov.b32 %tmp,%tid.x;",
            "mov.b32 %n,%tmp;",
            "st.shared.b32 [%a],%tmp;",
            "ld.shared.b32 %tmp,[%a];",
            "bar.sync 0;",
            "call f;",
            "C:",
            "@%outer mov.u32 %tmp,3;",
            "{.reg .b32 %n; mov.b32 %n,%tmp;}",
            ".pragma \"nounroll\";",
            ".reg .b32 temp;",
        ] {
            assert!(rewrite_body(&body.replacen(PREP, gap, 1)).is_none(), "{gap}");
        }
        assert!(rewrite_body(&format!("{{ {body} }}")).is_none());
        assert!(
            rewrite_body(
                &body
                    .replacen("@%q4 bra L4;", "@%q4 bra L4; {", 1)
                    .replace("L4:", "} L4:")
            )
            .is_none()
        );
    }

    #[test]
    fn canonicalizes_condition_before_testing_writes() {
        let region = region()
            .replace("%n,1", "%r0,1")
            .replacen(PREP, "mov.b32 %r00,%tmp;", 1);
        let source = format!("{{ {REGISTERS} .reg .b32 %r<2>; {region} }}");
        let (text, counts) = super::super::body(&source, 64, Default::default()).unwrap();
        assert!(!text.contains("$vf_cases_"));
        assert_eq!(counts.tensor, 16);
    }
}
