// k_mma_scaled_extra.cu: untested block-scaled MMA forms.
// Build: nvcc -arch=sm_120a -O3 -ptx -o k_mma_scaled_extra.ptx k_mma_scaled_extra.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)

// thread_id_a = 1 (scale_A from threads 2,3 of each quad instead of 0,1)
KERNEL(s_mxf4_tid_a1) {
    unsigned a0=in[GI*12+0], a1=in[GI*12+1], a2=in[GI*12+2], a3=in[GI*12+3];
    unsigned b0=in[GI*12+4], b1=in[GI*12+5];
    float c0,c1,c2,c3;
    __builtin_memcpy(&c0,&in[GI*12+6],4); __builtin_memcpy(&c1,&in[GI*12+7],4);
    __builtin_memcpy(&c2,&in[GI*12+8],4); __builtin_memcpy(&c3,&in[GI*12+9],4);
    unsigned sa=in[GI*12+10], sb=in[GI*12+11];
    float d0=c0,d1=c1,d2=c2,d3=c3;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "%10,{0,1},%11,{0,0};"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
    __builtin_memcpy(&out[GI*4+0],&d0,4); __builtin_memcpy(&out[GI*4+1],&d1,4);
    __builtin_memcpy(&out[GI*4+2],&d2,4); __builtin_memcpy(&out[GI*4+3],&d3,4);
}

// thread_id_b = 1
KERNEL(s_mxf4_tid_b1) {
    unsigned a0=in[GI*12+0], a1=in[GI*12+1], a2=in[GI*12+2], a3=in[GI*12+3];
    unsigned b0=in[GI*12+4], b1=in[GI*12+5];
    float c0,c1,c2,c3;
    __builtin_memcpy(&c0,&in[GI*12+6],4); __builtin_memcpy(&c1,&in[GI*12+7],4);
    __builtin_memcpy(&c2,&in[GI*12+8],4); __builtin_memcpy(&c3,&in[GI*12+9],4);
    unsigned sa=in[GI*12+10], sb=in[GI*12+11];
    float d0=c0,d1=c1,d2=c2,d3=c3;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "%10,{0,0},%11,{0,1};"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
    __builtin_memcpy(&out[GI*4+0],&d0,4); __builtin_memcpy(&out[GI*4+1],&d1,4);
    __builtin_memcpy(&out[GI*4+2],&d2,4); __builtin_memcpy(&out[GI*4+3],&d3,4);
}

// byte_id_a = 2 (the other valid value for 2X: scales from bytes 2,3 instead of 0,1)
KERNEL(s_mxf4_bid_a2) {
    unsigned a0=in[GI*12+0], a1=in[GI*12+1], a2=in[GI*12+2], a3=in[GI*12+3];
    unsigned b0=in[GI*12+4], b1=in[GI*12+5];
    float c0,c1,c2,c3;
    __builtin_memcpy(&c0,&in[GI*12+6],4); __builtin_memcpy(&c1,&in[GI*12+7],4);
    __builtin_memcpy(&c2,&in[GI*12+8],4); __builtin_memcpy(&c3,&in[GI*12+9],4);
    unsigned sa=in[GI*12+10], sb=in[GI*12+11];
    float d0=c0,d1=c1,d2=c2,d3=c3;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "%10,{2,0},%11,{2,0};"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
    __builtin_memcpy(&out[GI*4+0],&d0,4); __builtin_memcpy(&out[GI*4+1],&d1,4);
    __builtin_memcpy(&out[GI*4+2],&d2,4); __builtin_memcpy(&out[GI*4+3],&d3,4);
}

// scale_vec::2X with UE4M3 is REFUSED by ptxas 13.5:
// "Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3'"
// No kernel for it.

// E5M2 with singleton scale register vectors, also emitted by CUDA libraries.
KERNEL(s_mxf8_e5m2_1x) {
    unsigned a0=in[GI*12+0], a1=in[GI*12+1], a2=in[GI*12+2], a3=in[GI*12+3];
    unsigned b0=in[GI*12+4], b1=in[GI*12+5];
    float c0,c1,c2,c3;
    __builtin_memcpy(&c0,&in[GI*12+6],4); __builtin_memcpy(&c1,&in[GI*12+7],4);
    __builtin_memcpy(&c2,&in[GI*12+8],4); __builtin_memcpy(&c3,&in[GI*12+9],4);
    unsigned sa=in[GI*12+10], sb=in[GI*12+11];
    float d0=c0,d1=c1,d2=c2,d3=c3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e5m2.e5m2.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{0,0},{%11},{0,0};"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
    __builtin_memcpy(&out[GI*4+0],&d0,4); __builtin_memcpy(&out[GI*4+1],&d1,4);
    __builtin_memcpy(&out[GI*4+2],&d2,4); __builtin_memcpy(&out[GI*4+3],&d3,4);
}
