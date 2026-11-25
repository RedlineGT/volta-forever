// A kernel that only the code for sm_80 has and that calls a function of another unit: built
// with -rdc=true, its cubins are relocatable and a linker makes the image. For tests/proxy/told.py:
// the kernel is not given from the code for sm_80, and nothing is loaded for it.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -O3 -rdc=true -fatbin -gencode arch=compute_75,code=sm_75 \
//       -gencode arch=compute_80,code=sm_80 -o k_told_rdc.135.sm75-sm80.fatbin k_told_rdc.cu
extern __device__ float told_outside(float x);

extern "C" __global__ void k_rdc_both(float *out) {
    if (threadIdx.x == 0) out[0] = told_outside(1.0f);
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void k_rdc_calls_out(const float *a, float *out) {
    if (threadIdx.x == 0) out[0] = told_outside(a[0]);
}
#endif
