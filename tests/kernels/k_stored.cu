// Matrix loads into registers that stores before them still read, for ldsmrun.c. Each round of a
// kernel stores the four registers of the matrix load of the round before (four 32-bit stores to
// global memory), then loads four matrices from shared memory into the same four registers: the
// stores and the load are in one block of inline PTX, so that the compiler keeps this order and
// these registers. A store reads its register some time after it is issued, and names no
// scoreboard for that: the compiler counts on the load, which goes after the stores, to write
// its registers after they have read them.
//
// t_stored_plain: ldmatrix.x4; t_stored_trans: ldmatrix.x4.trans.
// out: [rounds][threads][4] words; shared memory: four matrices of 8 rows of 16 bytes for each warp,
// whose word i holds word_of(block, warp, i); in round k thread t gives the address of row
// (t & 7) + k (mod 8) of matrix t >> 3.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_stored.135.sm75.cubin k_stored.cu
#define WARPS 4

__device__ __forceinline__ unsigned word_of(unsigned block, unsigned warp, unsigned i) {
    unsigned x = (block * WARPS + warp) * 128u + i;
    x ^= x >> 7;
    x *= 0x9E3779B1u;
    return x ^ (x >> 15);
}

#define LOAD(form)                                                                                   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4" form ".shared.b16 {%0, %1, %2, %3}, [%4];"          \
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)                                                \
                 : "r"(row(0)))
#define STORE_THEN_LOAD(form, k)                                                                     \
    asm volatile("st.global.u32 [%5], %0;\n\t"                                                      \
                 "st.global.u32 [%5+4], %1;\n\t"                                                    \
                 "st.global.u32 [%5+8], %2;\n\t"                                                    \
                 "st.global.u32 [%5+12], %3;\n\t"                                                   \
                 "ldmatrix.sync.aligned.m8n8.x4" form ".shared.b16 {%0, %1, %2, %3}, [%4];"          \
                 : "+r"(a), "+r"(b), "+r"(c), "+r"(d)                                                \
                 : "r"(row(k)), "l"(out + ((size_t)((k) - 1) * threads + t) * 4)                     \
                 : "memory")

#define KERNEL(name, form)                                                                           \
    extern "C" __global__ void __launch_bounds__(WARPS * 32) name(unsigned* out, int rounds) {      \
        __shared__ __align__(16) unsigned tile[WARPS][128];                                         \
        unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;                                   \
        size_t threads = (size_t)gridDim.x * blockDim.x, t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; \
        for (unsigned i = lane; i < 128; i += 32) tile[warp][i] = word_of(blockIdx.x, warp, i);      \
        __syncwarp();                                                                                \
        unsigned base = (unsigned)__cvta_generic_to_shared(&tile[warp][0]);                         \
        auto row = [&](int k) { return base + ((lane >> 3) * 8 + ((lane + k) & 7)) * 16; };          \
        unsigned a, b, c, d;                                                                         \
        LOAD(form);                                                                                  \
        _Pragma("unroll 1") for (int k = 1; k < rounds; k++) STORE_THEN_LOAD(form, k);               \
        unsigned* y = out + ((size_t)(rounds - 1) * threads + t) * 4;                                \
        y[0] = a; y[1] = b; y[2] = c; y[3] = d;                                                      \
    }

KERNEL(t_stored_plain, "")
KERNEL(t_stored_trans, ".trans")
