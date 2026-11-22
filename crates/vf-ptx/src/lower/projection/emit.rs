use super::Transfer;
use std::fmt::Write;
pub(super) fn write(t: &Transfer<'_>) -> String {
    let b = &t.inputs;
    let p = &t.params;
    let mut out = String::from("\n{\n");
    out.push_str(r#".reg .f32 %q4affine_lut;
.reg .b32 %q4affine_lane,%q4affine_a,%q4affine_q,%q4affine_tmp,%q4affine_lane4,%q4affine_bit,%q4affine_i,%q4affine_g,%q4affine_y,%q4affine_z,%q4affine_flags,%q4affine_word,%q4affine_x,%q4affine_lo,%q4affine_hi,%q4affine_meta,%q4affine_v<4>,%q4affine_A<64>,%q4affine_s<2>,%q4affine_b<2>;
.reg .f32 %q4affine_val,%q4affine_sum,%q4affine_gate,%q4affine_up,%q4affine_dot,%q4affine_part<4>,%q4affine_g0,%q4affine_g1,%q4affine_co;
.reg .b64 %q4affine_X,%q4affine_W,%q4affine_meta_addr,%q4affine_off,%q4affine_Xg,%q4affine_Wg;
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
    out.push_str(r#"and.b32 %q4affine_lane4,%q4affine_lane,3; and.b32 %q4affine_bit,%q4affine_lane,1; setp.ne.u32 %q4affine_odd,%q4affine_bit,0;
mul.wide.u32 %q4affine_off,%q4affine_lane,4; add.u64 %q4affine_Xg,%q4affine_X,%q4affine_off;
and.b32 %q4affine_i,%q4affine_lane,7; shl.b32 %q4affine_tmp,%q4affine_i,4;
and.b32 %q4affine_i,%q4affine_lane,16; shl.b32 %q4affine_i,%q4affine_i,3; or.b32 %q4affine_tmp,%q4affine_tmp,%q4affine_i;
and.b32 %q4affine_i,%q4affine_lane,8; shr.u32 %q4affine_i,%q4affine_i,2; or.b32 %q4affine_tmp,%q4affine_tmp,%q4affine_i;
mul.wide.u32 %q4affine_off,%q4affine_tmp,4; add.u64 %q4affine_Wg,%q4affine_W,%q4affine_off;
and.b32 %q4affine_i,%q4affine_lane,6; shl.b32 %q4affine_i,%q4affine_i,2; shr.u32 %q4affine_tmp,%q4affine_lane,3; or.b32 %q4affine_i,%q4affine_i,%q4affine_tmp;
mul.wide.u32 %q4affine_off,%q4affine_i,4; add.u64 %q4affine_meta_addr,%q4affine_W,%q4affine_off;
mov.b32 %q4affine_gate,0; mov.b32 %q4affine_up,0; mov.b32 %q4affine_flags,0; mov.u32 %q4affine_g,0;
and.b32 %q4affine_tmp,%q4affine_lane,15; cvt.rn.f32.u32 %q4affine_lut,%q4affine_tmp;
$q4affine_GROUP:
ld.global.nc.u32 %q4affine_word,[%q4affine_Xg];
and.b32 %q4affine_x,%q4affine_word,0x7fff7fff; add.u32 %q4affine_y,%q4affine_x,0x7fff7fff;
add.u32 %q4affine_z,%q4affine_x,0x50805080; sub.u32 %q4affine_x,0xcf7fcf7f,%q4affine_x;
and.b32 %q4affine_x,%q4affine_x,%q4affine_z; not.b32 %q4affine_x,%q4affine_x; and.b32 %q4affine_y,%q4affine_y,%q4affine_x; and.b32 %q4affine_y,%q4affine_y,0x80008000; or.b32 %q4affine_flags,%q4affine_flags,%q4affine_y;
mov.b32 %q4affine_sum,0; shl.b32 %q4affine_i,%q4affine_lane4,3;
"#);
    // The upstream reduction folds each 16-element quarter from left to right,
    // then exchanges adjacent pairs and half-warp quarters at XOR distances 1,2.
    for j in 0..8 {
        writeln!(out,"add.u32 %q4affine_tmp,%q4affine_i,{j}; shfl.sync.idx.b32 %q4affine_a,%q4affine_word,%q4affine_tmp,31,0xffffffff; shl.b32 %q4affine_lo,%q4affine_a,16; and.b32 %q4affine_hi,%q4affine_a,0xffff0000; add.rn.f32 %q4affine_sum,%q4affine_sum,%q4affine_lo; add.rn.f32 %q4affine_sum,%q4affine_sum,%q4affine_hi;").unwrap();
    }
    for n in [1, 2] {
        writeln!(
            out,
            "shfl.sync.bfly.b32 %q4affine_a,%q4affine_sum,{n},31,0xffffffff; add.rn.f32 %q4affine_sum,%q4affine_sum,%q4affine_a;"
        )
        .unwrap();
    }
    for j in 0..32 {
        writeln!(out,"shfl.sync.idx.b32 %q4affine_a,%q4affine_word,{j},31,0xffffffff; shl.b32 %q4affine_A{},%q4affine_a,16; and.b32 %q4affine_A{},%q4affine_a,0xffff0000;",2*j,2*j+1).unwrap();
    }
    for (table, acc) in ["%q4affine_gate", "%q4affine_up"].into_iter().enumerate() {
        let base = table * 1152;
        out.push_str("mov.b32 %q4affine_dot,0;\n");
        for half in 0..2 {
            for q in 0..4 {
                writeln!(
                    out,
                    "ld.global.nc.u32 %q4affine_v{q},[%q4affine_Wg+{}];",
                    base + 16 * q + 4 * half
                )
                .unwrap();
            }
            for pair in 0..2 {
                for q in 0..4 {
                    writeln!(out, "mov.b32 %q4affine_part{q},0;").unwrap();
                    for h in 0..4 {
                        let shift = 4 * (2 * pair + h / 2) + 16 * (h % 2);
                        let k = 16 * q + 8 * half + 4 * pair + h;
                        // SHFL.IDX uses the low five bits of the source lane. Each
                        // half warp holds float(0)..float(15), so the fifth bit
                        // may come from the adjacent packed code. The full-warp
                        // guard above ensures every selected lane participates.
                        writeln!(out,"shr.u32 %q4affine_q,%q4affine_v{q},{shift}; shfl.sync.idx.b32 %q4affine_val,%q4affine_lut,%q4affine_q,31,0xffffffff; fma.rn.f32 %q4affine_part{q},%q4affine_A{k},%q4affine_val,%q4affine_part{q};").unwrap();
                    }
                }
                out.push_str("add.rn.f32 %q4affine_g0,%q4affine_part0,%q4affine_part1; add.rn.f32 %q4affine_g1,%q4affine_part2,%q4affine_part3; add.rn.f32 %q4affine_co,%q4affine_g0,%q4affine_g1; add.rn.f32 %q4affine_dot,%q4affine_co,%q4affine_dot;\n");
            }
        }
        for (j, dst) in [(0, "s"), (4, "b")] {
            writeln!(out,"ld.global.nc.u32 %q4affine_meta,[%q4affine_meta_addr+{}]; shl.b32 %q4affine_lo,%q4affine_meta,16; and.b32 %q4affine_hi,%q4affine_meta,0xffff0000; selp.b32 %q4affine_{dst}{table},%q4affine_hi,%q4affine_lo,%q4affine_odd;",base+1024+4*j).unwrap();
        }
        writeln!(
            out,
            "fma.rn.f32 {acc},%q4affine_dot,%q4affine_s{table},{acc}; fma.rn.f32 {acc},%q4affine_sum,%q4affine_b{table},{acc};"
        )
        .unwrap();
    }
    out.push_str("add.u64 %q4affine_Wg,%q4affine_Wg,2304; add.u64 %q4affine_meta_addr,%q4affine_meta_addr,2304; add.u64 %q4affine_Xg,%q4affine_Xg,128; add.u32 %q4affine_g,%q4affine_g,1; setp.lt.u32 %q4affine_p,%q4affine_g,32; @%q4affine_p bra $q4affine_GROUP;\nsetp.ne.b32 %q4affine_p,%q4affine_flags,0; vote.sync.any.pred %q4affine_bad,%q4affine_p,0xffffffff; @%q4affine_bad bra.uni $q4affine_COLD;\nshl.b32 %q4affine_lane4,%q4affine_lane4,1;\n");
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
