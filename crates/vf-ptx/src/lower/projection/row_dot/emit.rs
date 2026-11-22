use super::Program;
use std::fmt::Write;
pub(super) fn write(p: &Program<'_>, depths: &[i64]) -> String {
    let mut s = String::from(
        "{\n.reg .b32 %rowdot_r<36>; .reg .b64 %rowdot_d<22>; .reg .pred %rowdot_p<8>; .reg .f32 %rowdot_f<5>; .reg .b32 %rowdot_x<4>,%rowdot_w<4>;\n",
    );
    // X, W, output, row count, reduction depth, column count, optional row metadata.
    for (arg, kind, reg) in [
        (0, "u64", "d0"),
        (2, "u64", "d1"),
        (3, "u64", "d2"),
        (4, "u32", "r0"),
        (5, "u32", "r1"),
        (6, "u32", "r2"),
        (7, "u64", "d3"),
    ] {
        writeln!(s, "ld.param.{kind} %rowdot_{reg}, [{}];", p.params[arg]).unwrap();
    }
    s.push_str(
        r#"mov.u32 %rowdot_r3,%tid.x; mov.u32 %rowdot_r4,%ctaid.y;
mov.u32 %rowdot_r5,%ctaid.x; mov.u32 %rowdot_r6,%ctaid.z; or.b32 %rowdot_r5,%rowdot_r5,%rowdot_r6;
setp.ne.u32 %rowdot_p0,%rowdot_r5,0; @%rowdot_p0 bra $rowdot_fallback;
mov.u32 %rowdot_r5,%ntid.x; setp.ne.u32 %rowdot_p0,%rowdot_r5,256; @%rowdot_p0 bra $rowdot_fallback;
mov.u32 %rowdot_r5,%ntid.y; mov.u32 %rowdot_r6,%ntid.z; or.b32 %rowdot_r5,%rowdot_r5,%rowdot_r6;
setp.ne.u32 %rowdot_p0,%rowdot_r5,1; @%rowdot_p0 bra $rowdot_fallback;
setp.ne.u32 %rowdot_p0,%rowdot_r0,1; @%rowdot_p0 bra $rowdot_fallback;
setp.ne.u32 %rowdot_p1,0,0;
"#,
    );
    for k in depths {
        writeln!(
            s,
            "setp.eq.u32 %rowdot_p0,%rowdot_r1,{k}; or.pred %rowdot_p1,%rowdot_p1,%rowdot_p0;"
        )
        .unwrap()
    }
    s.push_str(r#"@!%rowdot_p1 bra $rowdot_fallback;
setp.gt.u32 %rowdot_p0,%rowdot_r2,65535; @%rowdot_p0 bra $rowdot_fallback;
setp.gt.u32 %rowdot_p0,%rowdot_r4,4095; @%rowdot_p0 bra $rowdot_fallback;
shl.b32 %rowdot_r5,%rowdot_r4,4; sub.s32 %rowdot_r6,%rowdot_r2,%rowdot_r5;
setp.eq.s32 %rowdot_p0,%rowdot_r6,1; setp.ge.s32 %rowdot_p1,%rowdot_r6,16; or.pred %rowdot_p0,%rowdot_p0,%rowdot_p1;
@!%rowdot_p0 bra $rowdot_fallback;
cvta.to.global.u64 %rowdot_d0,%rowdot_d0; cvta.to.global.u64 %rowdot_d1,%rowdot_d1; cvta.to.global.u64 %rowdot_d2,%rowdot_d2;
or.b64 %rowdot_d4,%rowdot_d0,%rowdot_d1; and.b64 %rowdot_d4,%rowdot_d4,15; setp.ne.u64 %rowdot_p0,%rowdot_d4,0; @%rowdot_p0 bra $rowdot_fallback;
and.b64 %rowdot_d4,%rowdot_d2,3; setp.ne.u64 %rowdot_p0,%rowdot_d4,0; @%rowdot_p0 bra $rowdot_fallback;
mul.wide.u32 %rowdot_d4,%rowdot_r1,2; add.u64 %rowdot_d5,%rowdot_d0,%rowdot_d4;
mul.wide.u32 %rowdot_d6,%rowdot_r2,%rowdot_r1; shl.b64 %rowdot_d6,%rowdot_d6,1; add.u64 %rowdot_d6,%rowdot_d6,%rowdot_d1;
mul.wide.u32 %rowdot_d7,%rowdot_r2,4; add.u64 %rowdot_d7,%rowdot_d7,%rowdot_d2;
"#);
    // Guard the new vector reads and reject intersection with the output,
    // including the metadata read used when a row-count pointer is nonnull.
    for (base, end) in [(0, 5), (1, 6), (2, 7)] {
        writeln!(s,"setp.eq.u64 %rowdot_p0,%rowdot_d{base},0; @%rowdot_p0 bra $rowdot_fallback; setp.lt.u64 %rowdot_p0,%rowdot_d{end},%rowdot_d{base}; @%rowdot_p0 bra $rowdot_fallback;").unwrap();
    }
    for (base, end) in [(0, 5), (1, 6), (3, 8)] {
        if base == 3 {
            s.push_str("setp.eq.u64 %rowdot_p0,%rowdot_d3,0; @%rowdot_p0 bra $rowdot_ready; cvta.to.global.u64 %rowdot_d3,%rowdot_d3; add.u64 %rowdot_d8,%rowdot_d3,12; setp.lt.u64 %rowdot_p0,%rowdot_d8,%rowdot_d3; @%rowdot_p0 bra $rowdot_fallback;\n")
        }
        writeln!(s,"setp.lt.u64 %rowdot_p0,%rowdot_d{base},%rowdot_d7; setp.lt.u64 %rowdot_p1,%rowdot_d2,%rowdot_d{end}; and.pred %rowdot_p0,%rowdot_p0,%rowdot_p1; @%rowdot_p0 bra $rowdot_fallback;").unwrap();
    }
    s.push_str(r#"ld.global.nc.u32 %rowdot_r6,[%rowdot_d3+8]; setp.ne.u32 %rowdot_p0,%rowdot_r6,1; @%rowdot_p0 bra $rowdot_fallback;
$rowdot_ready:
shr.u32 %rowdot_r7,%rowdot_r3,5; and.b32 %rowdot_r8,%rowdot_r3,31;
mad.lo.u32 %rowdot_r9,%rowdot_r7,2,%rowdot_r5;
setp.ge.u32 %rowdot_p0,%rowdot_r9,%rowdot_r2; @%rowdot_p0 bra $rowdot_exit;
add.u32 %rowdot_r10,%rowdot_r9,1; setp.lt.u32 %rowdot_p2,%rowdot_r10,%rowdot_r2;
mad.lo.u32 %rowdot_r11,%rowdot_r9,%rowdot_r1,0; mul.wide.u32 %rowdot_d8,%rowdot_r11,2;
add.u64 %rowdot_d8,%rowdot_d8,%rowdot_d1; add.u64 %rowdot_d9,%rowdot_d8,%rowdot_d4;
shl.b32 %rowdot_r12,%rowdot_r7,9; shl.b32 %rowdot_r13,%rowdot_r8,4;
"#);
    // Within each warp, consecutive lanes load 16 adjacent bytes of a
    // 256-element tile. X and the even weight row occupy the first shared
    // allocation. The odd row occupies the second. Two lanes run the original
    // dependent FP32 FMA chain for the warp's two output columns.
    for (sym, reg) in p.shared.iter().zip([14, 15]) {
        writeln!(s,"mov.u32 %rowdot_r{reg},{}; add.u32 %rowdot_r{reg},%rowdot_r{reg},15; and.b32 %rowdot_r{reg},%rowdot_r{reg},-16; add.u32 %rowdot_r{reg},%rowdot_r{reg},%rowdot_r12;",sym.0).unwrap();
    }
    s.push_str(r#"add.u32 %rowdot_r16,%rowdot_r14,4096;
add.u32 %rowdot_r17,%rowdot_r14,%rowdot_r13; add.u32 %rowdot_r18,%rowdot_r16,%rowdot_r13; add.u32 %rowdot_r19,%rowdot_r15,%rowdot_r13;
cvt.u64.u32 %rowdot_d10,%rowdot_r13; add.u64 %rowdot_d0,%rowdot_d0,%rowdot_d10; add.u64 %rowdot_d8,%rowdot_d8,%rowdot_d10; add.u64 %rowdot_d9,%rowdot_d9,%rowdot_d10;
add.u32 %rowdot_r20,%rowdot_r9,%rowdot_r8; setp.lt.u32 %rowdot_p3,%rowdot_r8,2; setp.lt.u32 %rowdot_p4,%rowdot_r20,%rowdot_r2; and.pred %rowdot_p3,%rowdot_p3,%rowdot_p4;
setp.eq.u32 %rowdot_p4,%rowdot_r8,0; selp.u32 %rowdot_r21,%rowdot_r16,%rowdot_r15,%rowdot_p4; mov.f32 %rowdot_f0,0f00000000;
mov.u32 %rowdot_r22,0;
$rowdot_tile:
ld.global.nc.v4.u32 {%rowdot_x0,%rowdot_x1,%rowdot_x2,%rowdot_x3},[%rowdot_d0]; st.shared.v4.u32 [%rowdot_r17],{%rowdot_x0,%rowdot_x1,%rowdot_x2,%rowdot_x3};
ld.global.nc.v4.u32 {%rowdot_w0,%rowdot_w1,%rowdot_w2,%rowdot_w3},[%rowdot_d8]; st.shared.v4.u32 [%rowdot_r18],{%rowdot_w0,%rowdot_w1,%rowdot_w2,%rowdot_w3};
@%rowdot_p2 ld.global.nc.v4.u32 {%rowdot_w0,%rowdot_w1,%rowdot_w2,%rowdot_w3},[%rowdot_d9]; @%rowdot_p2 st.shared.v4.u32 [%rowdot_r19],{%rowdot_w0,%rowdot_w1,%rowdot_w2,%rowdot_w3};
bar.warp.sync 0xffffffff;
@!%rowdot_p3 bra $rowdot_after;
mov.u32 %rowdot_r23,%rowdot_r14; mov.u32 %rowdot_r24,%rowdot_r21; mov.u32 %rowdot_r25,0;
$rowdot_fold:
ld.shared.v4.u32 {%rowdot_x0,%rowdot_x1,%rowdot_x2,%rowdot_x3},[%rowdot_r23]; ld.shared.v4.u32 {%rowdot_w0,%rowdot_w1,%rowdot_w2,%rowdot_w3},[%rowdot_r24];
"#);
    for i in 0..4 {
        writeln!(s,"shl.b32 %rowdot_r26,%rowdot_x{i},16; shl.b32 %rowdot_r27,%rowdot_w{i},16; mov.b32 %rowdot_f1,%rowdot_r26; mov.b32 %rowdot_f2,%rowdot_r27; fma.rn.f32 %rowdot_f0,%rowdot_f1,%rowdot_f2,%rowdot_f0;").unwrap();
        writeln!(s,"and.b32 %rowdot_r26,%rowdot_x{i},0xffff0000; and.b32 %rowdot_r27,%rowdot_w{i},0xffff0000; mov.b32 %rowdot_f1,%rowdot_r26; mov.b32 %rowdot_f2,%rowdot_r27; fma.rn.f32 %rowdot_f0,%rowdot_f1,%rowdot_f2,%rowdot_f0;").unwrap();
    }
    s.push_str(r#"add.u32 %rowdot_r23,%rowdot_r23,16; add.u32 %rowdot_r24,%rowdot_r24,16; add.u32 %rowdot_r25,%rowdot_r25,8;
setp.lt.u32 %rowdot_p0,%rowdot_r25,256; @%rowdot_p0 bra $rowdot_fold;
$rowdot_after:
bar.warp.sync 0xffffffff;
add.u64 %rowdot_d0,%rowdot_d0,512; add.u64 %rowdot_d8,%rowdot_d8,512; add.u64 %rowdot_d9,%rowdot_d9,512; add.u32 %rowdot_r22,%rowdot_r22,256;
setp.lt.u32 %rowdot_p0,%rowdot_r22,%rowdot_r1; @%rowdot_p0 bra $rowdot_tile;
mul.wide.u32 %rowdot_d10,%rowdot_r20,4; add.u64 %rowdot_d10,%rowdot_d10,%rowdot_d2; @%rowdot_p3 st.global.f32 [%rowdot_d10],%rowdot_f0;
$rowdot_exit:
ret;
}
$rowdot_fallback:
"#);
    s
}
