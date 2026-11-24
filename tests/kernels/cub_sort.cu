// cub_sort: device-wide algorithms of CUB (a radix sort of pairs, a sum, an inclusive scan), which
// choose among the architectures that the program was compiled for by the PTX version that
// cudaFuncGetAttributes gives for a kernel of the program, and answer cudaErrorInvalidDeviceFunction
// when it is none of them (CUB of CUDA 13). Built by tests/gpu/run.sh as PTX for sm_75 only (as a CUDA
// 13 build for 75-virtual, such as Ollama's cuda_v13 runner), as machine code for sm_75 only, both
// with CUDA 13.5, and for sm_70 with CUDA 12.9. Says what the PTX and binary versions of a kernel of
// the program are, and what each algorithm answered; the results are compared with the host's.
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdio>
#include <vector>

__global__ void twice(float *x) { x[threadIdx.x] *= 2.0f; }

static bool failed = false;

static void say(const char *what, cudaError_t e, bool right) {
    printf("%s: %s, %s\n", what, cudaGetErrorString(e), e == cudaSuccess && right ? "right" : "WRONG");
    failed |= e != cudaSuccess || !right;
}

int main() {
    cudaFuncAttributes a{};
    cudaError_t e = cudaFuncGetAttributes(&a, twice);
    printf("cudaFuncGetAttributes: %s, ptxVersion %d, binaryVersion %d\n", cudaGetErrorString(e), a.ptxVersion,
           a.binaryVersion);
    const int n = 4096;
    std::vector<float> keys(n);
    std::vector<int> values(n);
    for (int i = 0; i < n; i++) {
        keys[i] = (float)((i * 7919) % n) - n / 2;
        values[i] = i;
    }
    float *k, *ko;
    int *v, *vo, *sum, *scan;
    cudaMalloc(&k, n * 4);
    cudaMalloc(&ko, n * 4);
    cudaMalloc(&v, n * 4);
    cudaMalloc(&vo, n * 4);
    cudaMalloc(&sum, 4);
    cudaMalloc(&scan, n * 4);
    cudaMemcpy(k, keys.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(v, values.data(), n * 4, cudaMemcpyHostToDevice);

    // The radix sort of pairs, as ggml's ARGSORT of more than 1,024 columns.
    size_t bytes = 0;
    e = cub::DeviceRadixSort::SortPairs(nullptr, bytes, k, ko, v, vo, n);
    say("SortPairs, the size of its temporary storage", e, true);
    if (e != cudaSuccess) return 1;
    void *t;
    cudaMalloc(&t, bytes);
    e = cub::DeviceRadixSort::SortPairs(t, bytes, k, ko, v, vo, n);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    std::vector<float> sorted(n);
    std::vector<int> order(n);
    cudaMemcpy(sorted.data(), ko, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(order.data(), vo, n * 4, cudaMemcpyDeviceToHost);
    std::vector<int> want(n);
    for (int i = 0; i < n; i++) want[i] = i;
    std::stable_sort(want.begin(), want.end(), [&](int x, int y) { return keys[x] < keys[y]; });
    bool right = true;
    for (int i = 0; i < n; i++) right &= order[i] == want[i] && sorted[i] == keys[want[i]];
    say("SortPairs", e, right);

    // A sum and an inclusive scan, as ggml's SUM and CUMSUM.
    cudaFree(t);
    bytes = 0;
    e = cub::DeviceReduce::Sum(nullptr, bytes, v, sum, n);
    if (e == cudaSuccess) e = cudaMalloc(&t, bytes);
    if (e == cudaSuccess) e = cub::DeviceReduce::Sum(t, bytes, v, sum, n);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    int total = -1;
    cudaMemcpy(&total, sum, 4, cudaMemcpyDeviceToHost);
    say("Sum", e, total == n * (n - 1) / 2);
    cudaFree(t);
    bytes = 0;
    e = cub::DeviceScan::InclusiveSum(nullptr, bytes, v, scan, n);
    if (e == cudaSuccess) e = cudaMalloc(&t, bytes);
    if (e == cudaSuccess) e = cub::DeviceScan::InclusiveSum(t, bytes, v, scan, n);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    std::vector<int> scanned(n);
    cudaMemcpy(scanned.data(), scan, n * 4, cudaMemcpyDeviceToHost);
    right = true;
    for (int i = 0, s = 0; i < n; i++) right &= scanned[i] == (s += i);
    say("InclusiveSum", e, right);
    printf("%s\n", failed ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return failed ? 1 : 0;
}
