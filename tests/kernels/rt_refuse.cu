// rt_refuse: a program of the CUDA runtime whose fat binary has a kernel that the layer cannot run
// on a V100 (`bad`: a cluster barrier, which Volta does not have) beside two that it can. Built as
// PTX for sm_90 by tests/gpu/run.sh with the CUDA 12.9 and 13.5 compilers, and run under both
// values of CUDA_MODULE_LOADING: says what each launch and lookup answered.
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void good(int *v) { *v = 42; }
extern "C" __global__ void bad(int *v) {
    asm volatile("barrier.cluster.arrive;\n\tbarrier.cluster.wait;" ::: "memory");
    *v = 1;
}
extern "C" __global__ void also_good(int *v) { *v += 1; }

int main() {
    int *d = nullptr, h = 0;
    cudaError_t e = cudaMalloc(&d, sizeof h);
    printf("cudaMalloc -> %s\n", cudaGetErrorName(e));
    if (e != cudaSuccess) return 1;
    good<<<1, 1>>>(d);
    also_good<<<1, 1>>>(d);
    e = cudaGetLastError();
    cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("good kernels -> %s, value %d\n", cudaGetErrorName(e), h);
    cudaFuncAttributes attributes{};
    printf("cudaFuncGetAttributes(bad) -> %s\n", cudaGetErrorName(cudaFuncGetAttributes(&attributes, bad)));
    bad<<<1, 1>>>(d);
    printf("bad launch -> %s\n", cudaGetErrorName(cudaGetLastError()));
    good<<<1, 1>>>(d);
    e = cudaGetLastError();
    cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("good after bad -> %s, value %d, sync %s\n", cudaGetErrorName(e), h,
           cudaGetErrorName(cudaDeviceSynchronize()));
    return 0;
}
