use super::*;
const SIG: &str = "(.param .u64 x, .param .u32 stride, .param .u64 w, .param .u64 y, .param .u32 m, .param .u32 k, .param .u32 n, .param .u64 meta) .maxntid 256,1,1";
const BODY: &str = r#"{
.reg .pred %p<8>; .reg .b32 %r<16>; .reg .b64 %d<12>; .reg .b16 %h<2>; .reg .f32 %f<4>;
.shared .align 2 .b8 xbuf[8256]; .shared .align 2 .b8 wbuf[8256];
ld.param.u64 %d0,[x]; ld.param.u64 %d1,[w]; ld.param.u64 %d2,[y]; ld.param.u64 %d3,[meta];
ld.param.u32 %r0,[m]; ld.param.u32 %r1,[k]; ld.param.u32 %r2,[n];
cvta.to.global.u64 %d0,%d0; cvta.to.global.u64 %d1,%d1; cvta.to.global.u64 %d2,%d2;
setp.eq.s64 %p0,%d3,0; @%p0 bra valid;
cvta.to.global.u64 %d3,%d3; ld.global.nc.u32 %r3,[%d3+8]; min.s32 %r0,%r0,%r3;
valid:
setp.eq.s32 %p0,%r0,0; @%p0 bra done;
mov.u32 %r4,%tid.x; mov.u32 %r5,%ctaid.y; shl.b32 %r5,%r5,4; add.s32 %r6,%r5,%r4;
setp.ge.s32 %p0,%r4,16; @%p0 bra staged;
setp.ge.s32 %p0,%r6,%r2; @%p0 bra staged;
mul.wide.s32 %d4,%r6,%r1; shl.b64 %d4,%d4,1; add.s64 %d4,%d1,%d4;
ld.global.nc.u16 %h0,[%d0]; ld.global.nc.u16 %h1,[%d4];
mov.u32 %r7,xbuf; mov.u32 %r8,wbuf; mad.lo.s32 %r7,%r4,2,%r7; mad.lo.s32 %r8,%r4,2,%r8;
st.shared.u16 [%r7],%h0; st.shared.u16 [%r8],%h1;
staged:
bar.sync 0; mov.f32 %f0,0f00000000;
@%p0 bra first;
ld.shared.u16 %h0,[%r7]; ld.shared.u16 %h1,[%r8];
mov.b32 %f1,{0,%h0}; mov.b32 %f2,{0,%h1}; fma.rn.f32 %f0,%f1,%f2,%f0;
first:
mov.u32 %r9,1;
loop:
@%p0 bra tick;
mul.wide.s32 %d5,%r9,2; add.s64 %d6,%d0,%d5; add.s64 %d7,%d4,%d5;
ld.global.nc.u16 %h0,[%d6]; ld.global.nc.u16 %h1,[%d7];
mov.b32 %f1,{0,%h0}; mov.b32 %f2,{0,%h1}; fma.rn.f32 %f0,%f1,%f2,%f0;
tick:
add.s32 %r9,%r9,1; and.b32 %r10,%r9,255; setp.ne.s32 %p1,%r10,0; @%p1 bra loop;
bar.sync 0; setp.lt.s32 %p1,%r9,%r1; @%p1 bra loop;
@%p0 bra done;
mul.wide.s32 %d5,%r6,4; add.s64 %d6,%d2,%d5; st.global.f32 [%d6],%f0;
done: ret;
}"#;
#[test]
fn recognizes_the_ordered_memory_trace_without_kernel_or_register_names() {
    let body = BODY
        .replace("%d", "%address")
        .replace("loop:", "back:")
        .replace("bra loop;", "bra back;")
        .replace("xbuf", "storage_a")
        .replace("wbuf", "storage_b");
    let result = rewrite(SIG, &body).unwrap();
    assert!(result.contains("$rowdot_tile:"));
    assert!(result.contains("storage_a"));
    assert!(result.contains(&body[body.find("ld.param.").unwrap()..]));
}
#[test]
fn refuses_a_different_result_or_effect() {
    for (original, change) in [
        ("[%d3+8]", "[%d3+4]"),
        ("ld.global.nc.u32 %r3,[%d3+8];", "mov.u32 %r3,1;"),
        ("add.s64 %d7,%d4,%d5;", "add.s64 %d7,%d0,%d5;"),
        ("mad.lo.s32 %r8,%r4,2,%r8;", "add.s32 %r8,%r8,0;"),
        ("mov.f32 %f0,0f00000000;", "mov.f32 %f0,0f80000000;"),
        ("%f1,%f2,%f0;", "%f2,%f1,%f0;"),
        ("st.global.f32 [%d6],%f0;", "st.global.f32 [%d6+4],%f0;"),
        ("shl.b32 %r5,%r5,4;", "shl.b32 %r5,%r5,3;"),
        ("ld.shared.u16 %h0,[%r7];", "ld.shared.u16 %h0,[%r8];"),
        ("staged:\nbar.sync 0;", "staged:\n"),
        ("setp.ge.s32 %p0,%r6,%r2;", "setp.gt.s32 %p0,%r6,%r2;"),
        ("xbuf[8256]", "xbuf[1024]"),
        ("xbuf[8256]", "xbuf[010240]"),
        (".align 2", ".align 02"),
        ("%r<16>", "%r<020>"),
        ("%r4,16;", "%r4,016;"),
        ("%d3+8", "%d3+010"),
        ("%r9,255", "%r09,255"),
        ("cvta.to.global.u64 %d0,%d0;", "mov.f32 %f3,0f00000000;"),
    ] {
        assert!(BODY.contains(original));
        assert!(rewrite(SIG, &BODY.replace(original, change)).is_none(), "{change}");
    }
}
