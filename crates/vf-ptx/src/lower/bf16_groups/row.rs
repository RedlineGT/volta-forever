//! Specialize two independent MMA instructions for an A tile with only row zero active.
//! The scheduler has already moved independent preparations for B before the pair.
use super::{Bf16Row, Declarations, Product};
use std::collections::BTreeSet;
use std::fmt::Write;

pub(super) fn eligible(first: &Product<'_>, second: &Product<'_>, declared: &Declarations) -> bool {
    // The guard reads both original C fragments at the start; the fast path writes
    // destinations in program order. Never read a register clobbered by the first MMA.
    first.a == second.a
        && [first, second].iter().all(|p| {
            p.d.iter()
                .chain(&p.a)
                .chain(&p.b)
                .chain(&p.c)
                .all(|s| declared.width(s) == Some(32))
                && p.d.iter().collect::<BTreeSet<_>>().len() == 4
        })
        && second
            .a
            .iter()
            .chain(&second.b)
            .chain(&second.c)
            .chain(&second.d)
            .all(|s| !first.d.contains(s))
}

pub(super) fn write(out: &mut String, id: usize, pair: [&Product<'_>; 2], mode: Bf16Row, original: &str) {
    let r = format!("%bfgroup_{id}_r");
    let a = &pair[0].a;
    writeln!(
        out,
        ".reg .b32 {r}lane,{r}t,{r}g,{r}col,{r}even,{r}odd,{r}lo,{r}hi,{r}word,{r}k,{r}acc,{r}next;\n\
         .reg .b32 {r}x,{r}y,{r}z,{r}flags,{r}bounds,{r}index,{r}evenlane,{r}oddlane,{r}a<16>;\n\
         .reg .pred {r}row,{r}bad,{r}warp,{r}oddcol;\n\
         mov.u32 {r}lane,%laneid; and.b32 {r}t,{r}lane,3; and.b32 {r}g,{r}lane,28;\n\
         shr.u32 {r}col,{r}lane,2; setp.eq.u32 {r}row,{r}g,0;\n\
         or.b32 {r}x,{},{}; selp.b32 {r}flags,0,{r}x,{r}row;\n\
         or.b32 {r}x,{},{}; or.b32 {r}flags,{r}flags,{r}x;",
        a[0], a[2], a[1], a[3]
    )
    .unwrap();
    // Every inactive A and C element must be positive zero. With finite B,
    // a chain of fma.rn starting at +0 on those rows always returns +0.
    writeln!(out, "mov.b32 {r}bounds,0;").unwrap();
    let accumulators: BTreeSet<_> = pair.iter().map(|p| &p.c).collect();
    for c in accumulators {
        writeln!(
            out,
            "or.b32 {r}x,{},{}; selp.b32 {r}x,0,{r}x,{r}row;\n\
             or.b32 {r}y,{},{}; or.b32 {r}x,{r}x,{r}y; or.b32 {r}flags,{r}flags,{r}x;",
            c[0], c[1], c[2], c[3]
        )
        .unwrap();
        if mode == Bf16Row::Tree {
            for v in c[..2].iter().collect::<BTreeSet<_>>() {
                writeln!(out, "and.b32 {r}x,{v},0x7fffffff; max.u32 {r}bounds,{r}bounds,{r}x;").unwrap();
            }
        }
    }
    let mut words: BTreeSet<_> = pair.iter().flat_map(|p| p.b.iter()).collect();
    if mode == Bf16Row::Tree {
        // |C| < 2^120, and each nonzero multiplicand is in [2^-32,2^32).
        // The 16 finite products and partial sums cannot overflow. Small terms can
        // still disappear by rounding; tree mode deliberately opts into that order.
        writeln!(
            out,
            "setp.ge.u32 {r}bad,{r}bounds,0x7b800000; selp.b32 {r}x,1,0,{r}bad;\n\
             or.b32 {r}flags,{r}flags,{r}x;"
        )
        .unwrap();
        words.extend([&a[0], &a[2]]);
    }
    writeln!(out, "mov.b32 {r}bounds,0;").unwrap();
    for w in words {
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
    writeln!(
        out,
        "and.b32 {r}bounds,{r}bounds,0x80008000; or.b32 {r}flags,{r}flags,{r}bounds;\n\
         setp.ne.u32 {r}bad,{r}flags,0; vote.sync.any.pred {r}warp,{r}bad,0xffffffff;\n\
         @{r}warp bra.uni $bfgroup_row_slow_{id};\n\
         shl.b32 {r}evenlane,{r}t,3; or.b32 {r}oddlane,{r}evenlane,4;"
    )
    .unwrap();
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
    for p in pair {
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
        writeln!(
            out,
            "shfl.sync.idx.b32 {r}even,{r}acc,{r}evenlane,31,0xffffffff;\n\
             shfl.sync.idx.b32 {r}odd,{r}acc,{r}oddlane,31,0xffffffff;"
        )
        .unwrap();
        if mode == Bf16Row::Tree {
            writeln!(
                out,
                "add.rn.f32 {r}even,{r}even,{}; add.rn.f32 {r}odd,{r}odd,{};",
                p.c[0], p.c[1]
            )
            .unwrap();
        }
        writeln!(
            out,
            "selp.b32 {},{r}even,0,{r}row; selp.b32 {},{r}odd,0,{r}row; mov.b32 {},0; mov.b32 {},0;",
            p.d[0], p.d[1], p.d[2], p.d[3]
        )
        .unwrap();
    }
    writeln!(
        out,
        "bra.uni $bfgroup_row_done_{id};\n$bfgroup_row_slow_{id}:\n{original}\n$bfgroup_row_done_{id}:"
    )
    .unwrap();
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
