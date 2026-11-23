/* The program's graph memory (review section 5, G4). The driver reserves address space for a graph's
 * allocation nodes in granules of 32 MiB, at most about twice the device's memory for all the graphs
 * alive in a process. A tensor-core call captured in a graph goes to 12.9 (it would add an allocation
 * node), so the variants take none of it: the ceiling is measured with graphs, each holding one
 * 20 MiB allocation of the program's own; then, with the variants, the program captures one graph
 * fewer than the ceiling, each with its own allocation alive across two tensor-core calls on two
 * forked streams: every capture succeeds, every captured call is forwarded, and the replays give 12.9's
 * result. With the control capture-tc (the captured calls served, with their allocation nodes) the
 * program's captures fail. Prints "ok bf16-reserve". */
#include "bf16-common.h"

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
    printf("note  the driver held %d graphs of one 20 MiB allocation each\n", ceiling);
    CHECK(ceiling > 100 && ceiling < MAXG, "the ceiling was not found (%d graphs)", ceiling);

    /* one graph fewer than the ceiling, each with two tensor-core calls */
    const int want = ceiling - 1;
    const Counts c0 = counts(API_GEMMEX);
    int made = 0;
    cudaError_t last = cudaSuccess;
    for (; made < want; ++made)
        if ((last = capture_graph(1, &gs[made])) != cudaSuccess) break;
    const Counts c1 = counts(API_GEMMEX);
    CHECK(made == want, "the program's capture %d of %d failed (%s) with the variants", made + 1, want,
          cudaGetErrorName(last));
    printf("note  %d graphs captured; %llu captured calls forwarded, %llu served by the tensor-core path\n", made,
           delta(&c0, &c1, FWD(R_CAPTURED)), delta(&c0, &c1, SERVED_TC));
    CHECK(!rt_counters || delta(&c0, &c1, SERVED_TC) == 0, "captured tensor-core calls were served");
    /* replays of the first and the last graph: 12.9's result */
    cublasHandle_t h12;
    c129.Create(&h12);
    c129.SetStream(h12, s1);
    void *C12;
    CU(cudaMalloc(&C12, (size_t)M * N * 4));
    c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, C12,
                CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(s1));
    float *r12 = from_device(C12, (size_t)M * N * 4);
    int wrong = 0;
    for (int i = 0; i < made; i += made > 1 ? made - 1 : 1) {
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
        wrong += memcmp(o1, r12, (size_t)M * N * 4) != 0 || memcmp(o2, r12, (size_t)M * N * 4) != 0;
        free(o1);
        free(o2);
        cudaGraphExecDestroy(e);
        if (made == 1) break;
    }
    CHECK(wrong == 0, "%d of the graphs replayed gave a result other than 12.9's", wrong);
    for (int i = 0; i < made; ++i) cudaGraphDestroy(gs[i]);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-reserve\n");
    return 0;
}
