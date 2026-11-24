// Turing's FP16 tensor-core products whose bits show the order of their sums, for the cases of
// tests/gpu/run.sh that compare what the layer gives with and without pairs (docs/pairs.md) and in
// safe mode (VF_SAFE=1). Run by tests/kernels/ptxrun.c: kernel(out, turns), every thread of the
// block stores the same four words.
//
// Shared memory holds 512 rows of eight halves made from a hash of their place: normal values of
// many magnitudes and both signs, and about one in sixteen a subnormal. Each warp computes, at each
// of `turns` turns, the eight products of two A by four B (m16n8k8, f32 accumulators in place), A
// and B loaded by ldmatrix from rows that change with the lane and the turn: B plain in `plain`, B
// transposed in `trans`. The four words: a hash of the 16 results of A0's products, one of A1's, a
// sum of hashes of all of them, all over the warp, and the number of results that are not finite
// (0). In machine code the translator makes pairs of these products where it has the registers for
// them; it sums each result on the halves of k there, and on the even k, then the odd k, for a
// product by itself. (The PTX lowering of the same source makes no pairs of them: the readers after
// the loop are not at a place where the 32 lanes are known to run together.)
//
// Both kernels declare 62 registers for sm_75 (__launch_bounds__(1024)). Translated, they take 84
// with pairs and 79 without: blocks of 1,024 threads are refused (701, also by the second
// translation at launch, which takes no kernel with products to level 3). The same PTX with
// `.reqntid 1024` in place of `.maxntid 1024` (k_pair_bits.required.135.sm75.cubin) is translated
// within 64 registers, at level 3, where no pair is made (docs/translator.md, "A kernel that
// requires its block size").
//
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o tests/kernels/k_pair_bits.135.sm75.cubin tests/kernels/k_pair_bits.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -ptx   -o tests/kernels/k_pair_bits.135.sm75.ptx   tests/kernels/k_pair_bits.cu
//   sed 's/^\.maxntid 1024, 1, 1$/.reqntid 1024, 1, 1/' tests/kernels/k_pair_bits.135.sm75.ptx > k_pair_bits.required.ptx
//   toolkits/cuda-13.5.0/bin/ptxas -arch=sm_75 -O3 -o tests/kernels/k_pair_bits.required.135.sm75.cubin k_pair_bits.required.ptx

#define ROWS 512
#define MMA_F32 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "

#define LOAD_A4(a, addr)                                                           \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"         \
        : "=r"((a)[0]),"=r"((a)[1]),"=r"((a)[2]),"=r"((a)[3]) : "l"(addr))
#define LOAD_B4(b, addr)                                                           \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"         \
        : "=r"((b)[0]),"=r"((b)[1]),"=r"((b)[2]),"=r"((b)[3]) : "l"(addr))
#define LOAD_B4T(b, addr)                                                          \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3}, [%4];"   \
        : "=r"((b)[0]),"=r"((b)[1]),"=r"((b)[2]),"=r"((b)[3]) : "l"(addr))

// The eight products of two A by four B, each in place on its own accumulator.
#define GROUP_ALL(d, a, b) asm volatile(                                           \
    MMA_F32 "{%0,%1,%2,%3}, {%32,%33}, {%36}, {%0,%1,%2,%3};\n\t"                 \
    MMA_F32 "{%4,%5,%6,%7}, {%32,%33}, {%37}, {%4,%5,%6,%7};\n\t"                 \
    MMA_F32 "{%8,%9,%10,%11}, {%32,%33}, {%38}, {%8,%9,%10,%11};\n\t"             \
    MMA_F32 "{%12,%13,%14,%15}, {%32,%33}, {%39}, {%12,%13,%14,%15};\n\t"         \
    MMA_F32 "{%16,%17,%18,%19}, {%34,%35}, {%36}, {%16,%17,%18,%19};\n\t"         \
    MMA_F32 "{%20,%21,%22,%23}, {%34,%35}, {%37}, {%20,%21,%22,%23};\n\t"         \
    MMA_F32 "{%24,%25,%26,%27}, {%34,%35}, {%38}, {%24,%25,%26,%27};\n\t"         \
    MMA_F32 "{%28,%29,%30,%31}, {%34,%35}, {%39}, {%28,%29,%30,%31};"             \
    : "+r"((d)[0]),  "+r"((d)[1]),  "+r"((d)[2]),  "+r"((d)[3]),                  \
      "+r"((d)[4]),  "+r"((d)[5]),  "+r"((d)[6]),  "+r"((d)[7]),                  \
      "+r"((d)[8]),  "+r"((d)[9]),  "+r"((d)[10]), "+r"((d)[11]),                 \
      "+r"((d)[12]), "+r"((d)[13]), "+r"((d)[14]), "+r"((d)[15]),                 \
      "+r"((d)[16]), "+r"((d)[17]), "+r"((d)[18]), "+r"((d)[19]),                 \
      "+r"((d)[20]), "+r"((d)[21]), "+r"((d)[22]), "+r"((d)[23]),                 \
      "+r"((d)[24]), "+r"((d)[25]), "+r"((d)[26]), "+r"((d)[27]),                 \
      "+r"((d)[28]), "+r"((d)[29]), "+r"((d)[30]), "+r"((d)[31])                  \
    : "r"((a)[0]), "r"((a)[1]), "r"((a)[2]), "r"((a)[3]),                         \
      "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]))

static __device__ __forceinline__ unsigned mix(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// The half at place i: a sign, an exponent from 5 to 25 (2^-10 to 2^10), or 0 (a subnormal) for
// about one in sixteen, and ten bits of mantissa.
static __device__ __forceinline__ unsigned short half_at(unsigned i) {
    unsigned h = mix(i * 0x9e3779b9u + 0x632be5abu);
    unsigned exponent = (h >> 28) == 0 ? 0 : 5 + (h >> 16) % 21;
    return (unsigned short)((h & 0x8000) | (exponent << 10) | ((h >> 1) & 0x3ff));
}

static __device__ __forceinline__ void fill(unsigned short* tile) {
    for (int i = threadIdx.x; i < ROWS * 8; i += blockDim.x) tile[i] = half_at(i);
    __syncthreads();
}

// The four words of the warp, stored by every thread.
static __device__ __forceinline__ void words(const unsigned* d, unsigned* out) {
    const unsigned lane = threadIdx.x & 31;
    unsigned h0 = 0, h1 = 0, sum = 0, bad = 0;
    for (int i = 0; i < 32; i++) {
        unsigned w = d[i], at = lane * 32 + i;
        unsigned h = mix(w ^ mix(at + 0x51ed270bu));
        if (i < 16) h0 ^= h; else h1 ^= h;
        sum += mix(w + at * 0x2545f491u);
        bad += (w & 0x7f800000u) == 0x7f800000u;
    }
    for (int o = 16; o; o >>= 1) {
        h0 ^= __shfl_xor_sync(0xffffffffu, h0, o);
        h1 ^= __shfl_xor_sync(0xffffffffu, h1, o);
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
        bad += __shfl_xor_sync(0xffffffffu, bad, o);
    }
    unsigned* y = out + threadIdx.x * 4;
    y[0] = h0;
    y[1] = h1;
    y[2] = sum;
    y[3] = bad;
}

// The row that a lane gives to a matrix load at a turn: rows of A and B apart, all 16-byte aligned.
#define ROW_A(lane, s) ((const char*)tile + (((s) * 13 + (lane) * 5) & (ROWS / 2 - 1)) * 16)
#define ROW_B(lane, s) ((const char*)tile + (ROWS / 2 + (((s) * 29 + (lane) * 3) & (ROWS / 2 - 1))) * 16)

extern "C" __global__ void __launch_bounds__(1024) plain(unsigned* out, int turns) {
    __shared__ __align__(16) unsigned short tile[ROWS * 8];
    fill(tile);
    const unsigned lane = threadIdx.x & 31;
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int s = 0; s < turns; s++) {
        LOAD_A4(a, ROW_A(lane, s));
        LOAD_B4(b, ROW_B(lane, s));
        GROUP_ALL(d, a, b);
    }
    words(d, out);
}

extern "C" __global__ void __launch_bounds__(1024) trans(unsigned* out, int turns) {
    __shared__ __align__(16) unsigned short tile[ROWS * 8];
    fill(tile);
    const unsigned lane = threadIdx.x & 31;
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int s = 0; s < turns; s++) {
        LOAD_A4(a, ROW_A(lane, s));
        LOAD_B4T(b, ROW_B(lane, s));
        GROUP_ALL(d, a, b);
    }
    words(d, out);
}
