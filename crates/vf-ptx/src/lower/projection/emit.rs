use super::Transfer;
use std::fmt::Write;
pub(super) fn write(t: &Transfer<'_>) -> String {
    let b = &t.inputs;
    let p = &t.params;
    let mut out = String::from("\n{\n");
    out.push_str(r#".reg .f32 %q4affine_lut;
.reg .b32 %q4affine_lane,%q4affine_a,%q4affine_q,%q4affine_tmp,%q4affine_lane4,%q4affine_bit,%q4affine_i,%q4affine_g,%q4affine_y,%q4affine_z,%q4affine_flags,%q4affine_word,%q4affine_x,%q4affine_lo,%q4affine_hi,%q4affine_meta,%q4affine_s<2>,%q4affine_b<2>;
.reg .f32 %q4affine_val,%q4affine_sum,%q4affine_gate,%q4affine_up,%q4affine_dot;
.reg .b64 %q4affine_X,%q4affine_W,%q4affine_meta_addr,%q4affine_off,%q4affine_Xg;
.reg .pred %q4affine_p,%q4affine_bad,%q4affine_odd;
mov.u32 %q4affine_lane,%laneid;
activemask.b32 %q4affine_a; setp.ne.u32 %q4affine_p,%q4affine_a,0xffffffff; @%q4affine_p bra $q4affine_COLD;
mov.pred %q4affine_bad,0;
"#);
    for (i, k) in [(1, 2048), (2, 9), (4, 32), (5, 16), (10, 512), (11, 0)] {
        writeln!(
            out,
            "ld.param.b32 %q4affine_a,[{}]; setp.ne.u32 %q4affine_p,%q4affine_a,{k}; or.pred %q4affine_bad,%q4affine_bad,%q4affine_p;",
            p[i]
        )
        .unwrap();
    }
    for (r, k) in [
        ("%ntid.x", 128),
        ("%ntid.y", 1),
        ("%ntid.z", 1),
        ("%nctaid.x", 36),
        (t.names[b.rows], 1),
        (t.names[b.counts], 9),
    ] {
        writeln!(
            out,
            "mov.b32 %q4affine_a,{r}; setp.ne.u32 %q4affine_p,%q4affine_a,{k}; or.pred %q4affine_bad,%q4affine_bad,%q4affine_p;"
        )
        .unwrap();
    }
    for (r, k) in [(b.expert, 65535), (b.first, 1048575), (b.member, 1048575)] {
        writeln!(
            out,
            "setp.gt.u32 %q4affine_p,{},{}; or.pred %q4affine_bad,%q4affine_bad,%q4affine_p;",
            t.names[r], k
        )
        .unwrap();
    }
    out.push_str("vote.sync.any.pred %q4affine_p,%q4affine_bad,0xffffffff; @%q4affine_p bra.uni $q4affine_COLD;\n");
    writeln!(
        out,
        "shfl.sync.idx.b32 %q4affine_i,{},0,31,0xffffffff; ld.param.u64 %q4affine_X,[{}]; cvta.to.global.u64 %q4affine_X,%q4affine_X;",
        t.names[b.row], p[0]
    )
    .unwrap();
    writeln!(out,"mul.wide.s32 %q4affine_off,%q4affine_i,4096; add.u64 %q4affine_X,%q4affine_X,%q4affine_off; mul.wide.u32 %q4affine_off,%q4affine_lane,16; sub.u64 %q4affine_W,{},%q4affine_off;",t.names[b.weights]).unwrap();
    out.push_str("or.b64 %q4affine_off,%q4affine_X,%q4affine_W; and.b64 %q4affine_off,%q4affine_off,15; setp.ne.u64 %q4affine_bad,%q4affine_off,0; vote.sync.any.pred %q4affine_p,%q4affine_bad,0xffffffff; @%q4affine_p bra.uni $q4affine_COLD;\n");
    // Four lanes own four columns. All four lanes load a different 16-element
    // quarter of K, and a lane computes four column dot products for that quarter.
    out.push_str(r#".reg .b32 %q4affine_coop_word<8>,%q4affine_coop_ax<16>,%q4affine_coop_w<4>,%q4affine_coop_tmp<4>,%q4affine_coop_col,%q4affine_coop_ix,%q4affine_coop_gate,%q4affine_coop_up;
.reg .f32 %q4affine_coop_p<4>,%q4affine_coop_s,%q4affine_coop_t,%q4affine_coop_0,%q4affine_coop_1;
.reg .b64 %q4affine_coop_x,%q4affine_coop_wg; .reg .pred %q4affine_coop_lo,%q4affine_coop_hi;
and.b32 %q4affine_lane4,%q4affine_lane,3; and.b32 %q4affine_tmp,%q4affine_lane,1; setp.ne.u32 %q4affine_coop_lo,%q4affine_tmp,0; and.b32 %q4affine_tmp,%q4affine_lane,2; setp.ne.u32 %q4affine_coop_hi,%q4affine_tmp,0;
shr.u32 %q4affine_coop_col,%q4affine_lane,2; shl.b32 %q4affine_tmp,%q4affine_lane4,3; or.b32 %q4affine_coop_col,%q4affine_coop_col,%q4affine_tmp;
and.b32 %q4affine_bit,%q4affine_coop_col,1; setp.ne.u32 %q4affine_odd,%q4affine_bit,0;
mul.wide.u32 %q4affine_off,%q4affine_lane,4; add.u64 %q4affine_Xg,%q4affine_X,%q4affine_off;
mul.wide.u32 %q4affine_off,%q4affine_lane4,32; add.u64 %q4affine_coop_x,%q4affine_X,%q4affine_off;
mul.wide.u32 %q4affine_off,%q4affine_lane,16; add.u64 %q4affine_coop_wg,%q4affine_W,%q4affine_off;
and.b32 %q4affine_i,%q4affine_coop_col,6; shl.b32 %q4affine_i,%q4affine_i,2; shr.u32 %q4affine_tmp,%q4affine_coop_col,3; or.b32 %q4affine_i,%q4affine_i,%q4affine_tmp;
mul.wide.u32 %q4affine_off,%q4affine_i,4; add.u64 %q4affine_meta_addr,%q4affine_W,%q4affine_off;
mov.b32 %q4affine_gate,0; mov.b32 %q4affine_up,0; mov.b32 %q4affine_flags,0; mov.u32 %q4affine_g,0;
and.b32 %q4affine_tmp,%q4affine_lane,15; cvt.rn.f32.u32 %q4affine_lut,%q4affine_tmp;
$q4affine_GROUP:
ld.global.nc.u32 %q4affine_word,[%q4affine_Xg];
and.b32 %q4affine_x,%q4affine_word,0x7fff7fff; add.u32 %q4affine_y,%q4affine_x,0x7fff7fff;
add.u32 %q4affine_z,%q4affine_x,0x50805080; sub.u32 %q4affine_x,0xcf7fcf7f,%q4affine_x;
and.b32 %q4affine_x,%q4affine_x,%q4affine_z; not.b32 %q4affine_x,%q4affine_x; and.b32 %q4affine_y,%q4affine_y,%q4affine_x; and.b32 %q4affine_y,%q4affine_y,0x80008000; or.b32 %q4affine_flags,%q4affine_flags,%q4affine_y;

mov.b32 %q4affine_sum,0;
"#);
    for half in 0..2 {
        load_four_words(&mut out, "%q4affine_coop_x", half * 16, half * 4);
    }
    for word in 0..8 {
        let lo = 2 * word;
        let hi = lo + 1;
        writeln!(out,"shl.b32 %q4affine_coop_ax{lo},%q4affine_coop_word{word},16; and.b32 %q4affine_coop_ax{hi},%q4affine_coop_word{word},0xffff0000;").unwrap();
        writeln!(out,"add.rn.f32 %q4affine_sum,%q4affine_sum,%q4affine_coop_ax{lo}; add.rn.f32 %q4affine_sum,%q4affine_sum,%q4affine_coop_ax{hi};").unwrap();
    }
    // Match the original input sum: fold each quarter from left to right,
    // then combine adjacent quarters, and finally the two half-warp pairs.
    for distance in [1, 2] {
        writeln!(out,"shfl.sync.bfly.b32 %q4affine_a,%q4affine_sum,{distance},31,0xffffffff; add.rn.f32 %q4affine_sum,%q4affine_sum,%q4affine_a;").unwrap();
    }
    for (table, acc) in ["%q4affine_gate", "%q4affine_up"].into_iter().enumerate() {
        write_table(&mut out, table, acc);
    }
    // Restore the accumulator-to-lane layout consumed by the original epilogue.
    out.push_str(r#"add.u64 %q4affine_coop_wg,%q4affine_coop_wg,2304; add.u64 %q4affine_meta_addr,%q4affine_meta_addr,2304; add.u64 %q4affine_Xg,%q4affine_Xg,128; add.u64 %q4affine_coop_x,%q4affine_coop_x,128; add.u32 %q4affine_g,%q4affine_g,1; setp.lt.u32 %q4affine_p,%q4affine_g,32; @%q4affine_p bra $q4affine_GROUP;
and.b32 %q4affine_coop_ix,%q4affine_lane,7; shl.b32 %q4affine_coop_ix,%q4affine_coop_ix,2; shr.u32 %q4affine_tmp,%q4affine_lane,3; or.b32 %q4affine_coop_ix,%q4affine_coop_ix,%q4affine_tmp;
shfl.sync.idx.b32 %q4affine_coop_gate,%q4affine_gate,%q4affine_coop_ix,31,0xffffffff; mov.b32 %q4affine_gate,%q4affine_coop_gate;
shfl.sync.idx.b32 %q4affine_coop_up,%q4affine_up,%q4affine_coop_ix,31,0xffffffff; mov.b32 %q4affine_up,%q4affine_coop_up;
"#);
    out.push_str("setp.ne.b32 %q4affine_p,%q4affine_flags,0; vote.sync.any.pred %q4affine_bad,%q4affine_p,0xffffffff; @%q4affine_bad bra.uni $q4affine_COLD;\nshl.b32 %q4affine_lane4,%q4affine_lane4,1;\n");
    for (r, (table, col)) in &t.outputs {
        let acc = ["%q4affine_gate", "%q4affine_up"][*table as usize];
        writeln!(
            out,
            "add.u32 %q4affine_i,%q4affine_lane4,{col}; shfl.sync.idx.b32 {},{acc},%q4affine_i,31,0xffffffff;",
            t.names[*r]
        )
        .unwrap();
    }
    writeln!(out, "bra.uni {};\n}}\n$q4affine_COLD:", t.label).unwrap();
    out
}

fn load_four_words(out: &mut String, base: &str, offset: usize, first: usize) {
    writeln!(out,"ld.global.nc.v4.u32 {{%q4affine_coop_word{first},%q4affine_coop_word{},%q4affine_coop_word{},%q4affine_coop_word{}}},[{base}+{offset}];",first+1,first+2,first+3).unwrap();
}

fn write_table(out: &mut String, table: usize, acc: &str) {
    let base = table * 1152;
    // Each lane reads two aligned 16-byte ranges. Together the warp reads
    // precisely the 1,024 packed-weight bytes from the verified source table.
    load_four_words(out, "%q4affine_coop_wg", base, 0);
    load_four_words(out, "%q4affine_coop_wg", base + 512, 4);
    out.push_str("mov.b32 %q4affine_dot,0;\n");
    for half in 0..2 {
        // A register at column offset c holds the weights of column c XOR
        // (lane % 4). This is the transpose needed by the reduction below.
        for c in 0..4 {
            writeln!(
                out,
                "selp.b32 %q4affine_coop_tmp{c},%q4affine_coop_word{},%q4affine_coop_word{},%q4affine_coop_lo;",
                (c ^ 1) * 2 + half,
                c * 2 + half
            )
            .unwrap();
        }
        for c in 0..4 {
            writeln!(
                out,
                "selp.b32 %q4affine_coop_w{c},%q4affine_coop_tmp{},%q4affine_coop_tmp{c},%q4affine_coop_hi;",
                c ^ 2
            )
            .unwrap();
        }
        for pair in 0..2 {
            for c in 0..4 {
                writeln!(out, "mov.b32 %q4affine_coop_p{c},0;").unwrap();
                for h in 0..4 {
                    let shift = 4 * (2 * pair + h / 2) + 16 * (h % 2);
                    let k = half * 8 + 4 * pair + h;
                    // Both halves of the full warp contain the lookup 0..15.
                    // SHFL.IDX discards the bits above the five-bit lane index.
                    writeln!(out,"shr.u32 %q4affine_q,%q4affine_coop_w{c},{shift}; shfl.sync.idx.b32 %q4affine_val,%q4affine_lut,%q4affine_q,31,0xffffffff; fma.rn.f32 %q4affine_coop_p{c},%q4affine_coop_ax{k},%q4affine_val,%q4affine_coop_p{c};").unwrap();
                }
            }
            // In lane q, p[j] contributes to column q XOR j. Each add has
            // the same two operands as the original pairwise tree (up to an
            // exchange of the finite FP32 summands). No K segment is reassociated.
            out.push_str(r#"shfl.sync.bfly.b32 %q4affine_coop_s,%q4affine_coop_p1,1,31,0xffffffff; add.rn.f32 %q4affine_coop_0,%q4affine_coop_p0,%q4affine_coop_s;
shfl.sync.bfly.b32 %q4affine_coop_s,%q4affine_coop_p3,1,31,0xffffffff; add.rn.f32 %q4affine_coop_1,%q4affine_coop_p2,%q4affine_coop_s;
shfl.sync.bfly.b32 %q4affine_coop_s,%q4affine_coop_1,2,31,0xffffffff; add.rn.f32 %q4affine_coop_t,%q4affine_coop_0,%q4affine_coop_s;
add.rn.f32 %q4affine_dot,%q4affine_coop_t,%q4affine_dot;
"#);
        }
    }
    for (j, dst) in [(0, "s"), (4, "b")] {
        writeln!(out,"ld.global.nc.u32 %q4affine_meta,[%q4affine_meta_addr+{}]; shl.b32 %q4affine_lo,%q4affine_meta,16; and.b32 %q4affine_hi,%q4affine_meta,0xffff0000; selp.b32 %q4affine_{dst}{table},%q4affine_hi,%q4affine_lo,%q4affine_odd;",base+1024+4*j).unwrap();
    }
    writeln!(out,"fma.rn.f32 {acc},%q4affine_dot,%q4affine_s{table},{acc}; fma.rn.f32 {acc},%q4affine_sum,%q4affine_b{table},{acc};").unwrap();
}
