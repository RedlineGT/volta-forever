// Kernels whose code differs by architecture as library kernels do, for the refusal of stubs
// (vf_sass::effects, vf_adapt::Policy::adapt_fatbin). In one fat binary with machine code for sm_75
// and sm_80 and no PTX, as PyTorch's library has it:
//
// k_stub:  for sm_80 and later it computes; for earlier GPUs it is a stub that prints and writes
//          nothing, as PyTorch's attention kernels for sm_80 are in its sm_75 code. A program told
//          8.0 launches it.
// k_real:  the same computation in every image.
// k_wait:  writes nothing in any image (it only waits), as a kernel that keeps a stream busy.
// k_check: writes nothing in any image (it only asserts), as a kernel that checks its input.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -O3 -fatbin -gencode arch=compute_75,code=sm_75 \
//       -gencode arch=compute_80,code=sm_80 -o k_stub.135.sm75-sm80.fatbin k_stub.cu
#include <cassert>
#include <cstdio>

extern "C" __global__ void k_stub(float *out, const float *in, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
#else
    printf("FATAL: kernel `k_stub` is for sm80 and later, but was built for sm%d\n", __CUDA_ARCH__ / 10);
#endif
}

extern "C" __global__ void k_real(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

extern "C" __global__ void k_wait(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

extern "C" __global__ void k_check(const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) assert(in[i] == in[i]);
}
