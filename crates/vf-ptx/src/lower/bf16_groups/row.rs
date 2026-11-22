//! Specialize independent MMA instructions for an A tile with only row zero active.
//! Four products can share the row guard when both B words of every product are proved small.
use super::{Bf16Row, Declarations, Product};
use std::collections::{BTreeSet, HashSet};
use std::fmt::Write;

type IndexedProduct<'a> = (usize, Product<'a>);

pub(super) fn batch<'a, 't>(
    products: &'a [IndexedProduct<'t>],
    declared: &Declarations,
    small_b: &HashSet<usize>,
    scheduled: bool,
) -> Option<(&'a [IndexedProduct<'t>], bool)> {
    for n in [4, 2] {
        let Some(batch) = products.get(..n) else { continue };
        let small = batch.iter().all(|(index, _)| small_b.contains(index));
        if !small && !(n == 2 && products.len() == 2 && scheduled) {
            continue;
        }
        let (first, left) = &batch[0];
        let mut written = BTreeSet::new();
        if batch.iter().enumerate().all(|(i, (index, p))| {
            let inputs = p.a.iter().chain(&p.b).chain(&p.c);
            *index == first + i
                && p.a == left.a
                && inputs
                    .chain(&p.d)
                    .all(|s| declared.width(s) == Some(32) && !written.contains(s))
                && p.d.iter().all(|d| written.insert(*d))
        }) {
            return Some((batch, small));
        }
    }
    None
}

pub(super) fn multiplicand_bounds(out: &mut String, r: &str, w: &str, mode: Bf16Row) {
    writeln!(out, "and.b32 {r}x,{w},0x7fff7fff;").unwrap();
    match mode {
        Bf16Row::Tree => {
            // Each 16-bit addition/subtraction is carry/borrow independent.
            writeln!(
                out,
                "add.u32 {r}y,{r}x,0x7fff7fff; add.u32 {r}z,{r}x,0x50805080;\n\
                     sub.u32 {r}x,0xcf7fcf7f,{r}x; and.b32 {r}x,{r}x,{r}z;\n\
                     not.b32 {r}x,{r}x; and.b32 {r}x,{r}x,{r}y;"
            )
            .unwrap();
        }
        Bf16Row::Ordered => writeln!(out, "add.u32 {r}x,{r}x,0x00800080;").unwrap(),
        Bf16Row::Off => unreachable!(),
    }
    writeln!(out, "or.b32 {r}bounds,{r}bounds,{r}x;").unwrap();
}

fn write_guard(out: &mut String, r: &str, id: usize, batch: &[IndexedProduct<'_>], mode: Bf16Row, small: bool) {
    let a = &batch[0].1.a;
    writeln!(
        out,
        "or.b32 {r}x,{},{}; selp.b32 {r}flags,0,{r}x,{r}row;\n\
         or.b32 {r}x,{},{}; or.b32 {r}flags,{r}flags,{r}x;",
        a[0], a[2], a[1], a[3]
    )
    .unwrap();
    // Dense B uses a dynamic finite check and requires inactive C to start at +0.
    // With statically nonnegative B, inactive C can instead receive its exact update.
    writeln!(out, "mov.b32 {r}bounds,0;").unwrap();
    let accumulators: BTreeSet<_> = batch.iter().map(|(_, p)| &p.c).collect();
    for c in accumulators {
        if !small {
            writeln!(
                out,
                "or.b32 {r}x,{},{}; selp.b32 {r}x,0,{r}x,{r}row;\n\
             or.b32 {r}y,{},{}; or.b32 {r}x,{r}x,{r}y; or.b32 {r}flags,{r}flags,{r}x;",
                c[0], c[1], c[2], c[3]
            )
            .unwrap();
        }
        if mode == Bf16Row::Tree {
            for v in c[..2].iter().collect::<BTreeSet<_>>() {
                writeln!(out, "and.b32 {r}x,{v},0x7fffffff; max.u32 {r}bounds,{r}bounds,{r}x;").unwrap();
            }
        }
    }
    let mut words: BTreeSet<&str> = BTreeSet::new();
    if !small {
        words.extend(batch.iter().flat_map(|(_, p)| p.b.iter().copied()));
    }
    if mode == Bf16Row::Tree {
        if small {
            // Only row-zero accumulators participate in the reassociated dot product.
            writeln!(out, "selp.b32 {r}bounds,{r}bounds,0,{r}row;").unwrap();
        }
        // |C| < 2^120, and each nonzero multiplicand is in [2^-32,2^32).
        // The 16 finite products and partial sums cannot overflow. Small terms can
        // still disappear by rounding; tree mode deliberately opts into that order.
        writeln!(
            out,
            "setp.ge.u32 {r}bad,{r}bounds,0x7b800000; selp.b32 {r}x,1,0,{r}bad;\n\
             or.b32 {r}flags,{r}flags,{r}x;"
        )
        .unwrap();
        words.extend([a[0], a[2]]);
    }
    writeln!(out, "mov.b32 {r}bounds,0;").unwrap();
    for w in words {
        multiplicand_bounds(out, r, w, mode);
    }
    writeln!(
        out,
        "and.b32 {r}bounds,{r}bounds,0x80008000; or.b32 {r}flags,{r}flags,{r}bounds;\n\
         setp.ne.u32 {r}bad,{r}flags,0; vote.sync.any.pred {r}warp,{r}bad,0xffffffff;\n\
         @{r}warp bra.uni $bfgroup_row_slow_{id};"
    )
    .unwrap();
}

pub(super) enum Accumulator<'a> {
    Guarded(&'a str),
    Fragment,
    Distributed { prefix: &'a str, last: bool },
}

/// Replicate the input C of each output column into its four neighboring lanes.
/// Each lane can then keep the column's sum in place throughout the region.
pub(super) fn distribute_accumulators(out: &mut String, r: &str, batch: &[IndexedProduct<'_>]) {
    writeln!(
        out,
        ".reg .b32 {r}c<4>,{r}col,{r}src,{r}even,{r}odd; .reg .pred {r}oddcol;"
    )
    .unwrap();
    writeln!(
        out,
        "shr.u32 {r}col,{r}lane,2; shr.u32 {r}src,{r}col,1; and.b32 {r}x,{r}col,1; setp.ne.u32 {r}oddcol,{r}x,0;"
    )
    .unwrap();
    for (k, (_, p)) in batch.iter().enumerate() {
        writeln!(out, "shfl.sync.idx.b32 {r}even,{}, {r}src,31,0xffffffff; shfl.sync.idx.b32 {r}odd,{}, {r}src,31,0xffffffff; selp.b32 {r}c{k},{r}odd,{r}even,{r}oddcol;", p.c[0],p.c[1]).unwrap();
    }
}

pub(super) fn write(
    out: &mut String,
    id: usize,
    batch: &[IndexedProduct<'_>],
    mode: Bf16Row,
    small: bool,
    accumulation: Accumulator<'_>,
) {
    let r = format!("%bfgroup_{id}_r");
    let a = &batch[0].1.a;
    writeln!(
        out,
        ".reg .b32 {r}lane,{r}t,{r}g,{r}col,{r}even,{r}odd,{r}lo,{r}hi,{r}word,{r}k,{r}acc,{r}next;\n\
         .reg .b32 {r}x,{r}y,{r}z,{r}flags,{r}bounds,{r}index,{r}evenlane,{r}oddlane,{r}a<16>;\n\
         .reg .pred {r}row,{r}bad,{r}warp,{r}oddcol;\n\
         mov.u32 {r}lane,%laneid; and.b32 {r}t,{r}lane,3; and.b32 {r}g,{r}lane,28;\n\
         shr.u32 {r}col,{r}lane,2; setp.eq.u32 {r}row,{r}g,0;"
    )
    .unwrap();
    if matches!(accumulation, Accumulator::Guarded(_)) {
        write_guard(out, &r, id, batch, mode, small);
    }
    writeln!(out, "shl.b32 {r}evenlane,{r}t,3; or.b32 {r}oddlane,{r}evenlane,4;").unwrap();
    // Before either product writes D, broadcast the original A fragments once.
    if mode == Bf16Row::Ordered {
        writeln!(
            out,
            "shr.u32 {r}index,{r}col,1; and.b32 {r}x,{r}col,1; setp.ne.u32 {r}oddcol,{r}x,0;"
        )
        .unwrap();
    }
    for block in 0..2 {
        let n = if mode == Bf16Row::Ordered { 4 } else { 1 };
        for t in 0..n {
            let src = if mode == Bf16Row::Ordered {
                t.to_string()
            } else {
                format!("{r}t")
            };
            let i = 2 * (block * n + t);
            writeln!(out, "shfl.sync.idx.b32 {r}word,{},{src},31,0xffffffff; shl.b32 {r}a{i},{r}word,16; and.b32 {r}a{},{r}word,0xffff0000;", a[2*block],i+1).unwrap();
        }
    }
    if small {
        writeln!(out, ".reg .b32 {r}zero<4>;").unwrap();
    }
    for (index, (_, p)) in batch.iter().enumerate() {
        if mode == Bf16Row::Ordered {
            writeln!(
                out,
                "shfl.sync.idx.b32 {r}even,{}, {r}index,31,0xffffffff;\n\
                 shfl.sync.idx.b32 {r}odd,{}, {r}index,31,0xffffffff;\n\
                 selp.b32 {r}acc,{r}odd,{r}even,{r}oddcol;",
                p.c[0], p.c[1]
            )
            .unwrap();
        } else {
            writeln!(out, "mov.b32 {r}acc,0;").unwrap();
        }
        for (block, b) in p.b.iter().enumerate() {
            let n = if mode == Bf16Row::Ordered { 4 } else { 1 };
            for t in 0..n {
                let word = if mode == Bf16Row::Ordered {
                    writeln!(
                        out,
                        "or.b32 {r}k,{r}g,{t}; shfl.sync.idx.b32 {r}word,{b},{r}k,31,0xffffffff;"
                    )
                    .unwrap();
                    format!("{r}word")
                } else {
                    b.to_string()
                };
                let i = 2 * (block * n + t);
                writeln!(
                    out,
                    "shl.b32 {r}lo,{word},16; and.b32 {r}hi,{word},0xffff0000;\n\
                     fma.rn.f32 {r}acc,{r}a{i},{r}lo,{r}acc; fma.rn.f32 {r}acc,{r}a{},{r}hi,{r}acc;",
                    i + 1
                )
                .unwrap();
            }
        }
        if mode == Bf16Row::Tree {
            for hop in [1, 2] {
                writeln!(
                    out,
                    "shfl.sync.bfly.b32 {r}next,{r}acc,{hop},31,0xffffffff; add.rn.f32 {r}acc,{r}acc,{r}next;"
                )
                .unwrap();
            }
        }
        // Once +0 is added to an inactive value, subsequent additions of +0
        // leave its bits unchanged, including signed zeros and quieted NaNs.
        let sum = if let Accumulator::Distributed { prefix, last } = accumulation {
            let carry = format!("{prefix}c{index}");
            writeln!(out, "add.rn.f32 {carry},{r}acc,{carry};").unwrap();
            if !last {
                continue;
            }
            carry
        } else {
            format!("{r}acc")
        };
        writeln!(out, "shfl.sync.idx.b32 {r}even,{sum},{r}evenlane,31,0xffffffff;\nshfl.sync.idx.b32 {r}odd,{sum},{r}oddlane,31,0xffffffff;").unwrap();
        if mode == Bf16Row::Tree && !matches!(accumulation, Accumulator::Distributed { .. }) {
            writeln!(
                out,
                "add.rn.f32 {r}even,{r}even,{}; add.rn.f32 {r}odd,{r}odd,{};",
                p.c[0], p.c[1]
            )
            .unwrap();
        }
        if small {
            // B is nonnegative and finite. On inactive rows every product is +0.
            // The first addition quiets a signaling NaN and turns -0 into +0.
            // Save every original C before writing D: even a permuted D/C alias is legal.
            for (i, c) in p.c.iter().enumerate() {
                writeln!(out, "add.rn.f32 {r}zero{i},{c},0f00000000;").unwrap();
            }
            writeln!(out,
                "selp.b32 {},{r}even,{r}zero0,{r}row; selp.b32 {},{r}odd,{r}zero1,{r}row; mov.b32 {},{r}zero2; mov.b32 {},{r}zero3;",
                p.d[0], p.d[1], p.d[2], p.d[3]).unwrap();
        } else {
            writeln!(
                out,
                "selp.b32 {},{r}even,0,{r}row; selp.b32 {},{r}odd,0,{r}row; mov.b32 {},0; mov.b32 {},0;",
                p.d[0], p.d[1], p.d[2], p.d[3]
            )
            .unwrap();
        }
    }
    if let Accumulator::Guarded(original) = accumulation {
        writeln!(
            out,
            "bra.uni $bfgroup_row_done_{id};\n$bfgroup_row_slow_{id}:\n{original}\n$bfgroup_row_done_{id}:"
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{Options, body};

    const MMA: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";
    fn test_pair(gap: &str, following_c: &str, mode: Bf16Row) -> (String, usize) {
        let source = format!(
            "{{.reg .b32 %a<4>,%b<4>,%c<8>,%d<8>;\n{MMA} {{%d0,%d1,%d2,%d3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%c0,%c1,%c2,%c3}};\n{gap}\n{MMA} {{%d4,%d5,%d6,%d7}},{{%a0,%a1,%a2,%a3}},{{%b2,%b3}},{{{following_c},%c5,%c6,%c7}};}}"
        );
        let (out, counts) = body(
            &source,
            64,
            Options {
                bf16_row: mode,
                ..Default::default()
            },
        )
        .unwrap();
        (out, counts.tensor)
    }

    #[test]
    fn a_single_vote_covers_two_independent_products() {
        let fast = "mov.b32 %b2,%b3;";
        for mode in [Bf16Row::Ordered, Bf16Row::Tree] {
            let (out, n) = test_pair(fast, "%c4", mode);
            assert_eq!(n, 2);
            assert_eq!(out.matches("vote.sync.any.pred").count(), 1);
            assert!(out.contains("$bfgroup_row_slow_0:"));
            assert_eq!(out.contains("shfl.sync.bfly.b32"), mode == Bf16Row::Tree);
        }
        for (gap, c, mode) in [
            (fast, "%c4", Bf16Row::Off),
            (fast, "%d00", Bf16Row::Tree),
            ("mov.b32 %b2,%d0;", "%c4", Bf16Row::Ordered),
            ("elsewhere: mov.b32 %b2,%b3;", "%c4", Bf16Row::Tree),
        ] {
            assert!(!test_pair(gap, c, mode).0.contains("$bfgroup_row_"));
        }
    }

    #[test]
    fn tree_uses_the_two_exact_f32_lanes_until_the_packed_word_changes() {
        let tree = |guard: &str, gap: &str| {
            let source = format!(
                "{{.reg .pred %p; .reg .b32 %a<4>,%b<4>,%n,%q,%k,tmp,%c<8>;
                 mov.b32 %k,0x43004300; mov.b32 tmp,0xbf80bf80;
                 and.b32 %n,%a0,0x000f000f; or.b32 %q,%n,0x43004300;
                 {guard}fma.rn.bf16x2 %b0,%k,tmp,%q;
                 {gap}
                 {MMA} {{%c0,%c1,%c2,%c3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%c0,%c1,%c2,%c3}};
                 mov.b32 %b2,%b3;
                 {MMA} {{%c4,%c5,%c6,%c7}},{{%a0,%a1,%a2,%a3}},{{%b2,%b3}},{{%c4,%c5,%c6,%c7}};}}"
            );
            body(
                &source,
                64,
                Options {
                    bf16_row: Bf16Row::Tree,
                    ..Default::default()
                },
            )
            .unwrap()
            .0
        };
        let from_low = "mov.b32 %bfgroup_0_rlo, %vf_r";
        let from_high = "mov.b32 %bfgroup_0_rhi, %vf_r";
        let unpack = "shl.b32 %bfgroup_0_rlo,%b0,16;";
        let exact = tree("", "");
        assert!(exact.contains(from_low) && exact.contains(from_high));
        assert!(!exact.contains(unpack));
        for (guard, gap) in [
            ("@%p ", ""),
            ("", "@%p mov.b32 %b00, %a0;"),
            ("", "{.reg .b32 %b<2>; mov.b32 %b0,0;}"),
            ("", "join:"),
        ] {
            let lowered = tree(guard, gap);
            assert!(lowered.contains(unpack), "{guard}{gap}");
            assert!(!lowered.contains(from_low) && !lowered.contains(from_high));
        }
    }

    #[test]
    fn bounded_quartets_use_original_c_and_require_all_eight_words() {
        let mut body_text = "{.reg .pred %p; .reg .b32 %a<4>,%b<8>,%n,%q,%k,tmp,%c<16>;".to_string();
        body_text.push_str("mov.b32 %k,0x43004300; mov.b32 tmp,0xbf80bf80;");
        for n in 0..8 {
            write!(
                body_text,
                "and.b32 %n,%a0,0x000f000f; or.b32 %q,%n,0x43004300; fma.rn.bf16x2 %b{n},%k,tmp,%q;"
            )
            .unwrap();
        }
        let proof_end = body_text.len();
        for k in 0..4 {
            let [i, j, m, n] = [4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3];
            write!(
                body_text,
                "{MMA} {{%c{j},%c{m},%c{n},%c{i}}},{{%a0,%a1,%a2,%a3}},{{%b{},%b{}}},{{%c{i},%c{j},%c{m},%c{n}}};",
                2 * k,
                2 * k + 1
            )
            .unwrap();
        }
        body_text.push('}');
        for mode in [Bf16Row::Ordered, Bf16Row::Tree] {
            let opt = Options {
                bf16_row: mode,
                ..Default::default()
            };
            let (fast, _) = body(&body_text, 64, opt).unwrap();
            assert_eq!(fast.matches("vote.sync.any.pred").count(), 1);
            for k in 0..4 {
                let before = format!("add.rn.f32 %bfgroup_0_rzero3,%c{},0f00000000;", 4 * k + 3);
                let write = format!("selp.b32 %c{},%bfgroup_0_reven,%bfgroup_0_rzero0,", 4 * k + 1);
                assert!(fast.find(&before).unwrap() < fast.find(&write).unwrap());
            }
            assert_eq!(fast.contains("setp.ge.u32 %bfgroup_0_rbad"), mode == Bf16Row::Tree);
            assert!(fast.contains("setp.ne.u32 %bfgroup_0_rbad,%bfgroup_0_rflags,0;"));
            let quartet = |text: &str| {
                body(text, 64, opt)
                    .unwrap()
                    .0
                    .matches("add.rn.f32 %bfgroup_0_rzero3,")
                    .count()
                    == 4
            };
            // A family member with leading zeroes is one register even across a scope.
            assert!(quartet(&body_text.replace("%b1,", "%b01,")));
            for gap in [
                "@%p mov.u32 %b01,%a3;",
                "join:",
                "{.reg .b32 %b<2>; mov.b32 %b01,%a3;}",
                "{.reg .b32 %b1; mov.b32 %b1,%a3;}",
                "bra join; mov.u32 %b1,0; join:",
            ] {
                let mut partial = body_text.clone();
                partial.insert_str(proof_end, gap);
                assert!(!quartet(&partial), "{gap}");
            }
            let mut ambiguous = body_text.replace("%b1,", "%b01,");
            ambiguous.insert_str(ambiguous.find(MMA).unwrap(), "{.reg .b32 %b1;}");
            assert!(body(&ambiguous, 64, opt).is_err());
        }
    }

    #[test]
    fn exponent_guards_match_independent_checks_on_both_halfwords() {
        for value in 0..=u16::MAX {
            for other in [0, 0x8000, 0x2f80, 0x4f7f, 0x7f80, value.rotate_right(3)] {
                for (lo, hi) in [(value, other), (other, value)] {
                    let mag = (u32::from(lo) | (u32::from(hi) << 16)) & 0x7fff7fff;
                    let bad_finite = (mag + 0x00800080) & 0x80008000;
                    let bad_tree = (mag + 0x7fff7fff) & !((mag + 0x50805080) & (0xcf7fcf7f - mag)) & 0x80008000;
                    let in_range = |h: u16| {
                        let mag = h & 0x7fff;
                        mag == 0 || (0x2f80..=0x4f7f).contains(&mag)
                    };
                    assert_eq!(bad_finite == 0, lo & 0x7fff < 0x7f80 && hi & 0x7fff < 0x7f80);
                    assert_eq!(bad_tree == 0, in_range(lo) && in_range(hi));
                }
            }
        }
    }
}
