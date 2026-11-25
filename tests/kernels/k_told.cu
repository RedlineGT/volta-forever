// Kernels that only the sm_80 machine code of a fat binary has (as tests/kernels/k_sections.cu),
// for what the layer does when a program told 8.0 asks for one: it gives the kernel from the sm_80
// code, translated, in a module of its own, when that is sound (docs/architecture.md, "Fat
// binaries and what is chosen"). One fat binary with machine code for sm_75 and sm_80, no PTX.
//
// k_both:        in the code of both; it stores the architecture it was compiled for.
// k_bf16_sum:    sm_80 only. The sum of two arrays of pairs of bf16 (HFMA2.BF16_V2 by one, what
//                the compiler makes of a bf16 sum on sm_80) and the pairs of bf16 nearest to
//                pairs of f32 (F2FP.BF16.PACK_AB): the two instructions of sm_80 for bf16 that
//                vLLM 0.34's vectorized bf16 norm uses, which Volta has not.
// k_divides:     sm_80 only. A division of f32, whose slow path is a function inside the kernel that
//                returns through a copy of its return register, entered after a BSSY that comes
//                before the load of the descriptor of global memory: as vLLM's norm has them.
// k_reads_data:  sm_80 only. It reads `told_scale`, a variable of the image that the program sets
//                through the module it holds: a module of the layer's own would have another copy.
//                It is not given.
// k_reads_table: sm_80 only. It reads `told_table`, a table of the image that no kernel writes:
//                through the same bank of addresses, so not given either.
// k_adds_const:  sm_80 only. It adds `told_bias[1]`, a `__constant__` variable, which is read from
//                a constant bank of the module with no relocation, and which the program sets
//                through the module it holds: not given.
// k_const_address: sm_80 only. It stores the address of `told_bias`, which it takes from the word of
//                bank 0 where the driver puts the address of the module's `__constant__` bank
//                (c[0x0][0x50]): bank 0 is not all the kernel's own. Not given.
// k_mixed_ptr:   sm_80 only. It reads through a pointer that is `told_bias` or its argument: the
//                same word of bank 0, and a load that the layer does not read. Not given.
// k_not_translated: sm_80 only. A sum of pairs of f16 (HADD2), which the layer does not translate
//                from sm_80 machine code: not given.
// k_before_80:   sm_75 only (the other overload).
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -O3 -fatbin -gencode arch=compute_75,code=sm_75 \
//       -gencode arch=compute_80,code=sm_80 -o k_told.135.sm75-sm80.fatbin k_told.cu
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ int told_scale = 1;
__device__ const int told_table[4] = {3, 5, 7, 11};
__constant__ float told_bias[4] = {0.5f, 1.5f, 2.5f, 3.5f};

extern "C" __global__ void k_both(int *out) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0) out[0] = __CUDA_ARCH__ / 10;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
extern "C" __global__ void k_bf16_sum(const __nv_bfloat162 *a, const __nv_bfloat162 *c, __nv_bfloat162 *sum,
                                      const float2 *f, __nv_bfloat162 *nearest, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sum[i] = __hadd2(a[i], c[i]);
    nearest[i] = __float22bfloat162_rn(f[i]);
}

extern "C" __global__ void k_divides(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

extern "C" __global__ void k_reads_data(int *out) {
    if (threadIdx.x == 0) out[1] = 80 * told_scale;
}

extern "C" __global__ void k_reads_table(int *out, int which) {
    if (threadIdx.x == 0) out[1] = told_table[which & 3];
}

extern "C" __global__ void k_adds_const(const float *a, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + told_bias[1];
}

extern "C" __global__ void k_const_address(const float **out) {
    if (threadIdx.x == 0) out[0] = told_bias;
}

extern "C" __global__ void k_mixed_ptr(const float *a, float *out, int n, int sel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float *p = sel ? told_bias : a;
    if (i < n) out[i] = p[i & 3];
}

extern "C" __global__ void k_not_translated(const __half2 *a, __half2 *out) {
    if (threadIdx.x == 0) out[0] = __hadd2(a[0], a[1]);
}
#else
extern "C" __global__ void k_before_80(int *out) {
    if (threadIdx.x == 0) out[1] = 75;
}
#endif
