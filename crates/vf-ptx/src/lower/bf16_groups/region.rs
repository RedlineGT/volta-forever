//! Share the row and range checks across up to four straight-line updates to the
//! same accumulators. Both arms keep register preparation and MMA order intact.
use super::{Bf16Row, Declarations, Item, Kind, MMA, Product, product, row};
use crate::{lower::types::integer, scan};
use std::collections::{BTreeSet, HashSet};
use std::fmt::Write;

type Round<'a> = Vec<(usize, Product<'a>)>;

pub(super) struct Region<'a> {
    pub(super) rounds: Vec<Round<'a>>,
    private_sums: bool,
}

// A padded suffix must not let a second spelling of a register evade the write set.
fn canonical(name: &str) -> bool {
    let suffix = &name[name.trim_end_matches(|c: char| c.is_ascii_digit()).len()..];
    suffix.len() < 2 || !suffix.starts_with('0')
}

fn round<'a>(
    text: &'a str,
    items: &[Item],
    start: usize,
    declarations: &Declarations,
    small: &HashSet<usize>,
) -> Option<Round<'a>> {
    let products: Round<'_> = (start..start + 4)
        .map(|i| Some((i, product(items.get(i)?, text)?)))
        .collect::<Option<_>>()?;
    let (batch, bounded) = row::batch(&products, declarations, small, false)?;
    // The ordinary four-product path does not accept a destination that aliases a
    // multiplicand. Preserve that exclusion for every round, including later ones.
    let sums: HashSet<_> = products.iter().flat_map(|(_, p)| p.d.iter()).collect();
    if products.iter().any(|(_, p)| p.b.iter().any(|reg| sums.contains(reg))) {
        return None;
    }
    (bounded
        && batch.len() == 4
        && products
            .iter()
            .all(|(_, p)| p.c == p.d && p.a.iter().chain(&p.b).chain(&p.c).all(|r| canonical(r))))
    .then_some(products)
}

/// The operations between rounds are copied to each arm. Read no special registers,
/// move no memory accesses and admit only complete scopes that declare fresh scalar registers.
fn bridge<'a>(
    text: &'a str,
    items: &[Item],
    start: usize,
    outer: &Declarations,
    written: &mut HashSet<&'a str>,
    read: &mut HashSet<&'a str>,
) -> Option<usize> {
    let mut local = None;
    for (index, item) in items.iter().enumerate().skip(start).take(256) {
        match &item.kind {
            Kind::Open if local.is_none() => local = Some(Declarations::default()),
            Kind::Close if local.is_some() => local = None,
            Kind::Statement {
                guarded: false,
                opcode,
                operands,
            } => {
                let op = &text[opcode.clone()];
                let operands = &text[operands.clone()];
                if op == MMA {
                    return local.is_none().then_some(index);
                }
                if op == ".reg" {
                    let local = local.as_mut()?;
                    let (ty, names) = operands.trim().split_once(char::is_whitespace)?;
                    if ty != ".b32" {
                        return None;
                    }
                    if !names.split(',').all(|name| {
                        let name = name.trim();
                        scan::register(name).is_some() && canonical(name) && outer.width(name).is_none()
                    }) {
                        return None;
                    }
                    local.declare(operands);
                    continue;
                }
                let arity = match op {
                    "mov.b32" | "mov.u32" | "mov.f32" => 2,
                    "and.b32" | "or.b32" | "shr.u32" | "shl.b32" => 3,
                    "fma.rn.bf16x2" => 4,
                    _ => return None,
                };
                let words = scan::split_operands(operands);
                let (dest, src) = words.split_first()?;
                let declared = |s: &str| {
                    scan::register(s).is_some()
                        && canonical(s)
                        && local.as_ref().and_then(|l| l.width(s)).or_else(|| outer.width(s)) == Some(32)
                };
                if words.len() != arity || !declared(dest) || src.iter().any(|v| integer(v).is_none() && !declared(v)) {
                    return None;
                }
                written.insert(*dest);
                read.extend(src.iter().copied());
            }
            _ => return None,
        }
    }
    None
}

impl<'a> Region<'a> {
    pub(super) fn find(
        text: &'a str,
        items: &[Item],
        first: usize,
        decl: &Declarations,
        small: &HashSet<usize>,
    ) -> Option<Self> {
        let first = round(text, items, first, decl, small)?;
        let sums: HashSet<_> = first.iter().flat_map(|(_, p)| p.c.iter().copied()).collect();
        let mut activations: HashSet<_> = first[0].1.a.iter().copied().collect();
        if !activations.is_disjoint(&sums) {
            return None;
        }
        let mut region = Self {
            rounds: vec![first],
            private_sums: true,
        };
        let mut written = HashSet::new();
        while region.rounds.len() < 4 {
            let mut read = HashSet::new();
            let Some(idx) = bridge(text, items, region.end() + 1, decl, &mut written, &mut read) else {
                break;
            };
            let Some(next) = round(text, items, idx, decl, small) else {
                break;
            };
            if !next.iter().zip(&region.rounds[0]).all(|((_, p), (_, q))| p.c == q.c) {
                break;
            }
            activations.extend(next[0].1.a.iter().copied());
            if !activations.is_disjoint(&written) || !activations.is_disjoint(&sums) || !sums.is_disjoint(&written) {
                break;
            }
            region.private_sums &= sums.is_disjoint(&read);
            region.rounds.push(next);
        }
        (region.rounds.len() >= 2).then_some(region)
    }

    pub(super) fn end(&self) -> usize {
        self.rounds.last().unwrap().last().unwrap().0
    }

    pub(super) fn write(&self, out: &mut String, text: &str, items: &[Item], id: usize) {
        let r = format!("%bfgroup_region_{id}_");
        writeln!(
            out,
            ".reg .b32 {r}lane,{r}x,{r}y,{r}z,{r}flags,{r}bounds; .reg .pred {r}row,{r}bad,{r}warp;"
        )
        .unwrap();
        writeln!(
            out,
            "mov.u32 {r}lane,%laneid; setp.lt.u32 {r}row,{r}lane,4; mov.u32 {r}flags,0;"
        )
        .unwrap();
        let inputs = |a: usize, b: usize| {
            self.rounds
                .iter()
                .flat_map(|batch| {
                    let words = &batch[0].1.a;
                    [words[a], words[b]]
                })
                .collect::<BTreeSet<_>>()
        };
        let active = inputs(0, 2);
        for w in &active {
            writeln!(out, "or.b32 {r}flags,{r}flags,{w};").unwrap();
        }
        writeln!(out, "selp.b32 {r}flags,0,{r}flags,{r}row;").unwrap();
        for w in inputs(1, 3) {
            writeln!(out, "or.b32 {r}flags,{r}flags,{w};").unwrap();
        }

        // On active rows each round adds at most 16 * 15 * 2^32. Starting below
        // 2^118 leaves ample headroom under the individual guard's 2^120 limit
        // throughout all four rounds, including rounding of the FP32 additions.
        writeln!(out, "mov.u32 {r}bounds,0;").unwrap();
        for (_, p) in &self.rounds[0] {
            for c in &p.c[..2] {
                writeln!(out, "and.b32 {r}x,{c},0x7fffffff; max.u32 {r}bounds,{r}bounds,{r}x;").unwrap();
            }
        }
        writeln!(out, "setp.ge.and.u32 {r}bad,{r}bounds,0x7a800000,{r}row; selp.b32 {r}x,1,0,{r}bad; or.b32 {r}flags,{r}flags,{r}x;").unwrap();
        writeln!(out, "mov.u32 {r}bounds,0;").unwrap();
        for w in active {
            row::multiplicand_bounds(out, &r, w, Bf16Row::Tree);
        }
        writeln!(out,"and.b32 {r}bounds,{r}bounds,0x80008000; or.b32 {r}flags,{r}flags,{r}bounds; setp.ne.u32 {r}bad,{r}flags,0; vote.sync.any.pred {r}warp,{r}bad,0xffffffff; @{r}warp bra.uni $bfgroup_region_slow_{id};").unwrap();
        if self.private_sums {
            row::distribute_accumulators(out, &r, &self.rounds[0]);
        }
        for cold in [false, true] {
            let mut copied = items[self.rounds[0][0].0].range.start;
            for (n, batch) in self.rounds.iter().enumerate() {
                let start = items[batch[0].0].range.start;
                let end = items[batch.last().unwrap().0].range.end;
                out.push_str(&text[copied..start]);
                let round_id = id + n + if cold { self.rounds.len() } else { 0 };
                row::write(
                    out,
                    round_id,
                    batch,
                    Bf16Row::Tree,
                    true,
                    if cold {
                        row::Accumulator::Guarded(&text[start..end])
                    } else if self.private_sums {
                        row::Accumulator::Distributed {
                            prefix: &r,
                            last: n + 1 == self.rounds.len(),
                        }
                    } else {
                        row::Accumulator::Fragment
                    },
                );
                copied = end;
            }
            if !cold {
                writeln!(out, "bra.uni $bfgroup_region_done_{id};\n$bfgroup_region_slow_{id}:").unwrap();
            }
        }
        writeln!(out, "$bfgroup_region_done_{id}:").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{Options, body};

    fn source(rounds: usize, gap: &str) -> String {
        let mut text = String::from(
            "{.reg .pred %p; .reg .b32 %a<16>,%b<8>,%c<16>,%n,%q,%k,tmp; mov.b32 %k,0x43004300; mov.b32 tmp,0xbf80bf80;",
        );
        for i in 0..8 {
            write!(
                text,
                "and.b32 %n,%a0,0x000f000f; or.b32 %q,%n,0x43004300; fma.rn.bf16x2 %b{i},%k,tmp,%q;"
            )
            .unwrap();
        }
        for i in 0..rounds {
            if i > 0 {
                text.push_str(gap);
            }
            let a = (4 * i) % 16;
            for k in 0..4 {
                let c = (4 * k..4 * k + 4)
                    .map(|j| format!("%c{j}"))
                    .collect::<Vec<_>>()
                    .join(",");
                write!(
                    text,
                    "{MMA} {{{c}}},{{%a{a},%a{},%a{},%a{}}},{{%b{},%b{}}},{{{c}}};",
                    a + 1,
                    a + 2,
                    a + 3,
                    2 * k,
                    2 * k + 1
                )
                .unwrap();
            }
        }
        text.push('}');
        text
    }

    fn lower(text: &str, mode: Bf16Row) -> String {
        body(
            text,
            64,
            Options {
                bf16_row: mode,
                ..Default::default()
            },
        )
        .unwrap()
        .0
    }

    #[test]
    fn bounds_repeated_in_place_products_and_keeps_individual_fallbacks() {
        let label = "$bfgroup_region_slow_0:";
        for n in 2..=4 {
            let text = source(n, "{.reg .b32 local; mov.b32 local,%b0;}");
            let output = lower(&text, Bf16Row::Tree);
            assert!(output.contains(label));
            assert_eq!(output.matches("vote.sync.any.pred").count(), n + 1);
            assert_eq!(output.matches("0x7a800000").count(), 1);
            let (hot, cold) = output.split_once(label).unwrap();
            assert_eq!(hot.matches("vote.sync.any.pred").count(), 1);
            assert_eq!(cold.matches("vote.sync.any.pred").count(), n);
        }
        assert!(!lower(&source(1, ""), Bf16Row::Tree).contains(label));
        assert_eq!(
            lower(&source(8, ""), Bf16Row::Tree)
                .matches("bra.uni $bfgroup_region_done_")
                .count(),
            2
        );
        for mode in [Bf16Row::Ordered, Bf16Row::Off] {
            assert!(!lower(&source(2, ""), mode).contains(label));
        }
    }

    #[test]
    fn keeps_live_intermediate_fragments_visible_to_bridge_instructions() {
        for n in 2..=4 {
            let output = lower(&source(n, "mov.b32 %q,%b0;"), Bf16Row::Tree);
            let (hot, cold) = output.split_once("$bfgroup_region_slow_0:").unwrap();
            assert_eq!(hot.matches("add.rn.f32 %bfgroup_region_0_c0,").count(), n);
            assert_eq!(hot.matches("selp.b32 %c0,").count(), 1);
            assert!(!cold.contains("add.rn.f32 %bfgroup_region_0_c0,"));

            for gap in ["mov.b32 %q,%c0;", "{.reg .b32 local; mov.b32 local,%c15;}"] {
                let visible = lower(&source(n, gap), Bf16Row::Tree);
                let (hot, _) = visible.split_once("$bfgroup_region_slow_0:").unwrap();
                assert!(!hot.contains("add.rn.f32 %bfgroup_region_0_c0,"));
                assert_eq!(hot.matches("selp.b32 %c0,").count(), n);
            }
        }
    }

    #[test]
    fn rejects_unproved_bounds_changes_to_inputs_outputs_and_control_flow() {
        let allowed = source(2, "");
        let applies = |text: &str| lower(text, Bf16Row::Tree).contains("$bfgroup_region_slow_");
        assert!(applies(&allowed));
        for gap in [
            "mov.b32 %a4,%a3;",
            "mov.b32 %a0,%a4;",
            "mov.b32 %c2,%a5;",
            "@%p mov.b32 %n,0;",
            "join:",
            "bar.sync 0;",
            "ld.global.u32 %n,[%k];",
            "mov.u32 %n,%clock;",
            "{.reg .b32 %a4; mov.b32 %a4,%a3;}",
            "mov.b32 %a04,%n;",
        ] {
            assert!(!applies(&source(2, gap)), "{gap}");
        }
        for (old, new) in [
            ("0x000f000f", "0x001f001f"),
            ("%a4,%a5", "%c0,%a5"),
            ("{%c4,%c5,%c6,%c7},{%a4", "{%c5,%c4,%c6,%c7},{%a4"),
        ] {
            assert!(!applies(&allowed.replace(old, new)), "{old} => {new}");
        }
    }
}
