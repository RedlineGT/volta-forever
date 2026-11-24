// Kernels that the machine code of one fat binary has for some of its architectures only, as a
// library compiled for several has them: in one fat binary with machine code for sm_75 and sm_80
// and no PTX. The layer gives such an image from its sm_75 code (`docs/architecture.md`, "Fat
// binaries and what is chosen"); a program told 8.0 asks for the kernels of the sm_80 code.
//
// k_both:      in the code of every architecture; it stores the architecture it was compiled for
//              (__CUDA_ARCH__ / 10), so that a run says which code ran.
// k_from_80:   in the code for sm_80 and later only, as a library's kernel whose variant for a
//              later architecture exists only there: the host code of vLLM 0.34 asks for its
//              vectorized bf16 norm, `fused_add_rms_norm_kernel<c10::BFloat16, 8, ...>`, by the
//              name that its sm_80 code has (a template whose `enable_if` takes another overload,
//              of another name, in the code for sm_75). A GPU of 8.0 loads that code.
// k_before_80: in the code for earlier architectures only (the other overload).
// No kernel is named k_nowhere: a program that asks for it is told that there is none.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -O3 -fatbin -gencode arch=compute_75,code=sm_75 \
//       -gencode arch=compute_80,code=sm_80 -o k_sections.135.sm75-sm80.fatbin k_sections.cu

extern "C" __global__ void k_both(int *out) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0) out[0] = __CUDA_ARCH__ / 10;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
extern "C" __global__ void k_from_80(int *out) {
    if (threadIdx.x == 0) out[1] = 80;
}
#else
extern "C" __global__ void k_before_80(int *out) {
    if (threadIdx.x == 0) out[1] = 75;
}
#endif
