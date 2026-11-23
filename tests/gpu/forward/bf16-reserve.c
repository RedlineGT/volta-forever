/* The program's share of graph memory (review section 5, G4). The driver reserves address space for a
 * graph's allocation nodes in granules of 32 MiB, at most about twice the device's memory for all the
 * graphs alive in a process, the program's and the runtime's together; the runtime holds at most
 * total memory / 32 MiB of them (its cap), the rest is the program's.
 *
 * First the ceiling is measured: graphs, each holding one 20 MiB allocation of the program's own, are
 * captured until the driver refuses one. Then the program captures 1,000 graphs, each with its own
 * 20 MiB allocation alive across two tensor-core calls on two forked streams (the runtime's two
 * allocations add two granules to such a graph): with the cap, the runtime serves until it holds its
 * share and forwards the rest, and every capture of the program succeeds (1,000 + 1,014 granules
 * under the ceiling); some graphs are replayed, served and forwarded, and give the right result.
 * Without the cap (control no-cap) the runtime would take three granules a graph and the program's
 * capture fails. Prints "ok bf16-reserve". */
#include "bf16-common.h"
#include <time.h>

static const int M = 1536, N = 256, K = 1536;
static cublasHandle_t h1, h2;
static cudaStream_t s1, s2;
static void *dA, *dB, *dC1, *dC2;
static float *ref;

static cudaError_t capture_graph(int with_calls, cudaGraph_t *g)
{
    const float one = 1.0f, zero = 0.0f;
    cudaEvent_t fork, join;
    cudaEventCreateWithFlags(&fork, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&join, cudaEventDisableTiming);
    void *own = NULL;
    cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
    cudaError_t r = cudaMallocAsync(&own, 20u << 20, s1);
    if (r == cudaSuccess) cudaMemsetAsync(own, 0, 4096, s1);
    if (with_calls) {
        cudaEventRecord(fork, s1);
        cudaStreamWaitEvent(s2, fork, 0);
        cublasGemmEx(h1, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, dC1,
                     CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(h2, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, dC2,
                     CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(join, s2);
        cudaStreamWaitEvent(s1, join, 0);
    }
    if (own) cudaFreeAsync(own, s1);
    const cudaError_t end = cudaStreamEndCapture(s1, g);
    cudaEventDestroy(fork);
    cudaEventDestroy(join);
    cudaGetLastError();
    return r != cudaSuccess ? r : end;
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    if (cublasCreate(&h1) || cublasCreate(&h2)) return 1;
    open_runtime();
    cublasSetStream(h1, s1);
    cublasSetStream(h2, s2);
    uint16_t *hA = bf16_random((size_t)M * K, 0.05), *hB = bf16_random((size_t)N * K, 1.0);
    dA = to_device(hA, (size_t)M * K * 2);
    dB = to_device(hB, (size_t)N * K * 2);
    CU(cudaMalloc(&dC1, (size_t)M * N * 4));
    CU(cudaMalloc(&dC2, (size_t)M * N * 4));
    const float one = 1.0f, zero = 0.0f;
    cublasGemmEx(h1, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, dC1,
                 CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaDeviceSynchronize());
    ref = from_device(dC1, (size_t)M * N * 4);

    /* the ceiling, with the program's graphs alone */
    enum { MAXG = 4096 };
    static cudaGraph_t gs[MAXG];
    int ceiling = 0;
    while (ceiling < MAXG && capture_graph(0, &gs[ceiling]) == cudaSuccess) ++ceiling;
    for (int i = 0; i < ceiling; ++i) cudaGraphDestroy(gs[i]);
    long long cap = 0, high = 0;
    if (rt_granules) rt_granules(0, &cap, &high);
    printf("note  the driver held %d graphs of one 20 MiB allocation each; the runtime's cap is %lld granules\n", ceiling,
           cap);
    CHECK(ceiling > 1000 + cap, "the ceiling (%d graphs) leaves no room for the test", ceiling);

    /* the program's 1,000 graphs, each with the runtime's two allocations */
    const Counts c0 = counts(API_GEMMEX);
    int made = 0;
    cudaError_t last = cudaSuccess;
    for (; made < 1000; ++made)
        if ((last = capture_graph(1, &gs[made])) != cudaSuccess) break;
    const Counts c1 = counts(API_GEMMEX);
    CHECK(made == 1000, "the program's capture %d of 1,000 failed (%s) with the runtime on", made + 1, cudaGetErrorName(last));
    const unsigned long long served = delta(&c0, &c1, SERVED_TC), at_cap = delta(&c0, &c1, FWD(R_CAP)),
                             alloc = delta(&c0, &c1, FWD(R_ALLOC));
    printf("note  2,000 calls in 1,000 graphs: %llu served, %llu forwarded at the cap, %llu for a failed allocation\n",
           served, at_cap, alloc);
    CHECK(!rt_counters || (at_cap > 0 && served <= (unsigned long long)cap), "the runtime did not stop at its cap");
    /* replays: the first graph (served) and the last (forwarded) */
    int wrong = 0;
    for (int i = 0; i < made; i += made - 1 > 0 ? made - 1 : 1) {
        cudaGraphExec_t e;
        cudaMemset(dC1, 0, (size_t)M * N * 4);
        cudaMemset(dC2, 0, (size_t)M * N * 4);
        if (cudaGraphInstantiate(&e, gs[i], 0) != cudaSuccess) {
            ++wrong;
            continue;
        }
        cudaGraphLaunch(e, s1);
        cudaStreamSynchronize(s1);
        float *o1 = from_device(dC1, (size_t)M * N * 4), *o2 = from_device(dC2, (size_t)M * N * 4);
        wrong += memcmp(o1, ref, (size_t)M * N * 4) != 0 || memcmp(o2, ref, (size_t)M * N * 4) != 0;
        free(o1);
        free(o2);
        cudaGraphExecDestroy(e);
        if (made == 1) break;
    }
    CHECK(wrong == 0, "%d of the graphs replayed gave a wrong result", wrong);
    for (int i = 0; i < made; ++i) cudaGraphDestroy(gs[i]);
    long long left = rt_granules ? rt_granules(0, NULL, NULL) : 0;
    for (int i = 0; i < 300 && left; ++i) { /* the destructors run on a thread of the driver */
        cudaDeviceSynchronize();
        struct timespec ts = {0, 10 * 1000 * 1000};
        nanosleep(&ts, NULL);
        left = rt_granules(0, NULL, NULL);
    }
    CHECK(left == 0, "%lld granules still counted after every graph was destroyed", left);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-reserve\n");
    return 0;
}
