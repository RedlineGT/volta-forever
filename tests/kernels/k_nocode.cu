// Kernels without code for some GPUs, as a fat binary holds them: a kernel built for some
// architectures only is still emitted for the others, with a body that traps there (the `#else` of
// an architecture test). The layer answers the lookup of such a kernel and refuses its launch
// (vf_ptx::stubs, vf_sass::effects::trap_at_entry, vf-proxy guard). tests/gpu/run.sh `nocode-*`,
// tests/proxy/nocode_launch.py.
//
// no_code:         a trap and nothing else (for every architecture these images are built for).
// good:            writes v + its thread index.
// trap_if:         traps only when v is not zero, then writes 7: code, not a kernel without code.
// store_then_trap: writes, then traps: code (never launched by the tests).
// empty:           does nothing (a legitimate empty kernel: launched, it is not refused).
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -O3 -arch=compute_75 -ptx -o k_nocode.135.sm75.ptx k_nocode.cu
//   toolkits/cuda-13.5.0/bin/nvcc -O3 -cubin -arch=sm_75 -o k_nocode.135.sm75.cubin k_nocode.cu
//   toolkits/cuda-12.9.2/bin/nvcc -O3 -cubin -arch=sm_70 -o k_nocode.129.sm70.cubin k_nocode.cu
extern "C" __global__ void no_code(int *out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 600
    out[threadIdx.x] = 1;
#else
    __trap();
#endif
}

extern "C" __global__ void good(int *out, int v) { out[threadIdx.x] = v + (int)threadIdx.x; }

extern "C" __global__ void trap_if(int *out, int v) {
    if (v) __trap();
    out[threadIdx.x] = 7;
}

extern "C" __global__ void store_then_trap(int *out) {
    out[threadIdx.x] = 1;
    __trap();
}

extern "C" __global__ void empty() {}
