//! A chain of four BF16 matrix products, specialized when B has only column zero
//! active. Four lanes own two rows of A. Ordered mode sums K in the scalar
//! instruction order. Tree mode reduces four bounded partial sums per row.
use super::super::Bf16Column;
use super::{Bf16Row, Declarations, Product, row};
use crate::scan::{Item, Kind};
use std::collections::BTreeSet;
use std::fmt::Write;

type IndexedProduct<'a> = (usize, Product<'a>);

pub(super) fn accepts(products: &[IndexedProduct<'_>], items: &[Item], text: &str, declared: &Declarations) -> bool {
    let [first, rest @ ..] = products else { return false };
    let c = &first.1.c;
    products.len() == 4
        && c.iter().collect::<BTreeSet<_>>().len() == 4
        && products.iter().all(|(_, p)| {
            p.d == *c
                && p.c == *c
                && p.a
                    .iter()
                    .chain(&p.b)
                    .chain(c)
                    .all(|word| declared.width(word) == Some(32))
        })
        && rest.iter().zip(products).all(|(right, left)| {
            items[left.0 + 1..right.0]
                .iter()
                .all(|item| matches!(item.kind, Kind::Line) && crate::deps::debug_location(&text[item.range.clone()]))
        })
}
pub(super) fn write(out: &mut String, id: usize, products: &[IndexedProduct<'_>], fallback: &str, mode: Bf16Column) {
    let r = format!("%bfgroup_col{id}_");
    let c = &products[0].1.c;
    writeln!(out, ".reg .b32 {r}lane,{r}t,{r}flags,{r}bounds,{r}x,{r}y,{r}z,{r}word,{r}lo,{r}hi,{r}alo,{r}ahi,{r}acc0,{r}acc1,{r}tmp;\n\
             .reg .pred {r}dest,{r}first,{r}bad,{r}warp;\n\
             mov.u32 {r}lane,%laneid;\n\
             and.b32 {r}t,{r}lane,3;\n\
             setp.lt.u32 {r}first,{r}lane,4;\n\
             setp.eq.u32 {r}dest,{r}t,0;\n\
             mov.b32 {r}flags,0;").unwrap();

    let right: BTreeSet<_> = products.iter().flat_map(|(_, p)| &p.b).collect();
    for word in right {
        writeln!(out, "or.b32 {r}flags,{r}flags,{word};").unwrap();
    }
    writeln!(
        out,
        "and.b32 {r}flags,{r}flags,0x7fff7fff; selp.b32 {r}flags,0,{r}flags,{r}first;"
    )
    .unwrap();

    // Require +0 in unused C elements. Finite products by either sign of zero
    // leave those exact accumulators +0. Preserve exceptional inputs by fallback.
    writeln!(
        out,
        "or.b32 {r}x,{},{};\n\
             selp.b32 {r}x,0,{r}x,{r}dest;\n\
             or.b32 {r}flags,{r}flags,{r}x;\n\
             or.b32 {r}x,{},{};\n\
             or.b32 {r}flags,{r}flags,{r}x;",
        c[0], c[2], c[1], c[3]
    )
    .unwrap();
    if mode == Bf16Column::Tree {
        for active in [c[0], c[2]] {
            writeln!(
                out,
                "and.b32 {r}x,{active},0x7fffffff;\n\
             setp.ge.u32 {r}bad,{r}x,0x7b800000;\n\
             selp.b32 {r}y,1,0,{r}bad;\n\
             or.b32 {r}flags,{r}flags,{r}y;\n\
             setp.eq.u32 {r}bad,{active},0x80000000;\n\
             selp.b32 {r}y,1,0,{r}bad;\n\
             or.b32 {r}flags,{r}flags,{r}y;"
            )
            .unwrap();
        }
    }
    writeln!(out, "mov.b32 {r}bounds,0;").unwrap();
    let bound = match mode {
        Bf16Column::Ordered => Bf16Row::Ordered,
        Bf16Column::Tree => Bf16Row::Tree,
        Bf16Column::Off => unreachable!(),
    };
    let words: BTreeSet<_> = products.iter().flat_map(|(_, p)| p.a.iter().chain(&p.b)).collect();
    for &word in words {
        row::multiplicand_bounds(out, &r, word, bound);
    }
    writeln!(
        out,
        "and.b32 {r}bounds,{r}bounds,0x80008000;\n\
             or.b32 {r}flags,{r}flags,{r}bounds;\n\
             setp.ne.u32 {r}bad,{r}flags,0;\n\
             vote.sync.any.pred {r}warp,{r}bad,0xffffffff;\n\
             @{r}warp bra.uni $bfgroup_col_slow_{id};\n\
             mov.b32 {r}acc0,0;\n\
             mov.b32 {r}acc1,0;"
    )
    .unwrap();
    // Use the same K order and the same RN FMA as the scalar implementation.
    // All four lanes in a row group compute its two active sums; only lane t=0
    // owns column 0 in the output fragment.
    if mode == Bf16Column::Ordered {
        writeln!(
            out,
            "and.b32 {r}z,{r}lane,0xfffffffc;\n\
             shfl.sync.idx.b32 {r}acc0,{}, {r}z,31,0xffffffff;\n\
             shfl.sync.idx.b32 {r}acc1,{}, {r}z,31,0xffffffff;",
            c[0], c[2]
        )
        .unwrap();
        for (_, p) in products {
            for half in 0..2 {
                for t in 0..4 {
                    writeln!(
                        out,
                        "or.b32 {r}x,{r}z,{t};\n\
             shfl.sync.idx.b32 {r}word,{}, {t},31,0xffffffff;\n\
             shl.b32 {r}lo,{r}word,16;\n\
             and.b32 {r}hi,{r}word,0xffff0000;",
                        p.b[half]
                    )
                    .unwrap();
                    for row in 0..2 {
                        let a = p.a[half * 2 + row];
                        writeln!(
                            out,
                            "shfl.sync.idx.b32 {r}y,{a},{r}x,31,0xffffffff;\n\
             shl.b32 {r}alo,{r}y,16;\n\
             and.b32 {r}ahi,{r}y,0xffff0000;\n\
             fma.rn.f32 {r}acc{row},{r}alo,{r}lo,{r}acc{row};\n\
             fma.rn.f32 {r}acc{row},{r}ahi,{r}hi,{r}acc{row};"
                        )
                        .unwrap();
                    }
                }
            }
        }
        for row in 0..2 {
            writeln!(out, "@{r}dest mov.b32 {},{r}acc{row};", c[row * 2]).unwrap();
        }
    } else {
        for (_, p) in products {
            for half in 0..2 {
                writeln!(out, "shfl.sync.idx.b32 {r}word,{}, {r}t,31,0xffffffff; shl.b32 {r}lo,{r}word,16; and.b32 {r}hi,{r}word,0xffff0000;",p.b[half]).unwrap();
                for row in 0..2 {
                    let a = p.a[half * 2 + row];
                    writeln!(
                        out,
                        "shl.b32 {r}alo,{a},16;\n\
             and.b32 {r}ahi,{a},0xffff0000;\n\
             fma.rn.f32 {r}acc{row},{r}alo,{r}lo,{r}acc{row};\n\
             fma.rn.f32 {r}acc{row},{r}ahi,{r}hi,{r}acc{row};"
                    )
                    .unwrap();
                }
            }
        }
        for row in 0..2 {
            for hop in [1, 2] {
                writeln!(out, "shfl.sync.bfly.b32 {r}tmp,{r}acc{row},{hop},31,0xffffffff; add.rn.f32 {r}acc{row},{r}acc{row},{r}tmp;").unwrap();
            }
            writeln!(out, "@{r}dest add.rn.f32 {0},{0},{r}acc{row};", c[row * 2]).unwrap();
        }
    }
    writeln!(
        out,
        "bra.uni $bfgroup_col_done_{id};\n$bfgroup_col_slow_{id}:\n{fallback}\n$bfgroup_col_done_{id}:"
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::super::{Options, rewrite};
    use crate::lower::Bf16Column;
    use crate::scan;

    const OP: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%d0,%d1,%d2,%d3},{%a0,%a1,%a2,%a3},{%b0,%b1},{%d0,%d1,%d2,%d3};";
    fn lowered(ops: &str, mode: Bf16Column) -> Option<String> {
        let body = format!("{{\n.reg .b32 %d<4>, %a<4>, %b<2>;\n{ops}\n}}");
        let items = scan::items(body.as_bytes(), 0..body.len()).unwrap();
        rewrite(
            &body,
            &items,
            &[],
            Options {
                bf16_column: mode,
                ..Default::default()
            },
        )
        .map(|v| v.0)
    }

    #[test]
    fn bounded_and_ordered_use_distinct_arithmetic_and_keep_the_complete_fallback() {
        let ops = [OP; 4].join("\n");
        assert!(lowered(&ops, Bf16Column::Off).is_none());
        for (mode, fmas, reductions) in [(Bf16Column::Ordered, 128, 0), (Bf16Column::Tree, 32, 4)] {
            let text = lowered(&ops, mode).unwrap();
            assert_eq!(text.matches("fma.rn.f32").count(), fmas);
            assert_eq!(text.matches("shfl.sync.bfly").count(), reductions);
            assert_eq!(text.matches("vote.sync.any.pred").count(), 1);
            assert!(text.split_once("$bfgroup_col_slow_0:\n").unwrap().1.starts_with(&ops));
        }
    }

    #[test]
    fn only_rewrites_four_updates_to_the_same_independent_declared_destinations() {
        let ops = [OP; 4].join("\n");
        for invalid in [
            [OP; 3].join("\n"),
            [OP; 5].join("\n"),
            ops.replace("%d1", "%d0"),
            ops.replacen("{%d0,%d1,%d2,%d3};", "{%d1,%d0,%d2,%d3};", 1),
            ops.replace("%b0", "%d0"),
            ops.replace("%a0", "%clock"),
            ops.replacen(";\n", ";\nmov.b32 %a0,%b0;\n", 1),
            ops.replacen(";\n", ";\n@%p bra L;\n", 1),
        ] {
            assert!(lowered(&invalid, Bf16Column::Tree).is_none(), "{invalid}");
        }
    }
}
