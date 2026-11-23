/* Lifetimes (review section 5, G6): a graph replayed after the handle that captured it is destroyed:
 * a captured tensor-core call (computed by 12.9), and a captured skinny call, a kernel node of the
 * runtime's module, replayed after every handle of the variants is destroyed (the module lives as long
 * as its context); cudaDeviceReset with live handles, then a new handle and a call served in the new
 * context, with the bits it had before. Two V100s in one process and a V100 beside another GPU need
 * devices this test is not given (it says so). Prints "ok bf16-life". The control unload-module of the
 * test runtime (the module unloaded with the last handle) must make it fail: on driver 580 the replay
 * of the skinny call's graph then dies in cuGraphLaunch (SIGSEGV), after the line that announces it. */
#include "bf16-common.h"

static const int M = 1536, N = 256, K = 1536;
static uint16_t *hA, *hB;

static float *run_once(cublasHandle_t h, cudaStream_t s, int *served)
{
    void *dA = to_device(hA, (size_t)M * K * 2), *dB = to_device(hB, (size_t)N * K * 2), *C;
    CU(cudaMalloc(&C, (size_t)M * N * 4));
    const float one = 1.0f, zero = 0.0f;
    const Counts c0 = counts(API_GEMMEX);
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, C, CUDA_R_32F,
                 M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(s));
    const Counts c1 = counts(API_GEMMEX);
    *served = !rt_counters || delta(&c0, &c1, SERVED_TC) == 1;
    float *out = from_device(C, (size_t)M * N * 4);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(C);
    return out;
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    hA = bf16_random((size_t)M * K, 0.05);
    hB = bf16_random((size_t)N * K, 1.0);
    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cublasHandle_t h, keep; /* keep: a second handle, destroyed with the last one below */
    if (cublasCreate(&h) || cublasCreate(&keep)) return 1;
    open_runtime();
    cublasSetStream(h, s);
    int served;
    float *before = run_once(h, s, &served);
    CHECK(served, "the first call is not served");

    /* capture, destroy the handle, replay */
    void *dA = to_device(hA, (size_t)M * K * 2), *dB = to_device(hB, (size_t)N * K * 2), *C;
    CU(cudaMalloc(&C, (size_t)M * N * 4));
    const float one = 1.0f, zero = 0.0f;
    cudaGraph_t g;
    cudaGraphExec_t e;
    CU(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, C, CUDA_R_32F,
                 M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamEndCapture(s, &g));
    CU(cudaGraphInstantiate(&e, g, 0));
    cublasDestroy(h);
    CU(cudaGraphLaunch(e, s));
    const cudaError_t sync = cudaStreamSynchronize(s);
    CHECK(sync == cudaSuccess, "replay after the handle's destruction: %s", cudaGetErrorName(sync));
    float *replayed = from_device(C, (size_t)M * N * 4);
    /* (a captured tensor-core call is computed by 12.9: compare with 12.9's own result) */
    cublasHandle_t h12;
    c129.Create(&h12);
    c129.SetStream(h12, s);
    void *C12;
    CU(cudaMalloc(&C12, (size_t)M * N * 4));
    c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, C12,
                CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(s));
    float *r12 = from_device(C12, (size_t)M * N * 4);
    CHECK(!memcmp(replayed, r12, (size_t)M * N * 4), "replay after the handle's destruction: not 12.9's result");
    cudaFree(C12);
    cudaGraphExecDestroy(e);
    cudaGraphDestroy(g);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(C);

    /* a captured skinny call (the runtime's kernel), replayed after every handle is destroyed */
    {
        const int NS = 8;
        cublasHandle_t hs;
        if (cublasCreate(&hs)) return 1;
        cublasSetStream(hs, s);
        void *sA = to_device(hA, (size_t)M * K * 2), *sB = to_device(hB, (size_t)NS * K * 2), *sC, *sC12;
        CU(cudaMalloc(&sC, (size_t)M * NS * 4));
        CU(cudaMalloc(&sC12, (size_t)M * NS * 4));
        cudaGraph_t gs;
        cudaGraphExec_t es;
        const Counts c0 = counts(API_GEMMEX);
        CU(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        cublasGemmEx(hs, CUBLAS_OP_T, CUBLAS_OP_N, M, NS, K, &one, sA, CUDA_R_16BF, K, sB, CUDA_R_16BF, K, &zero, sC,
                     CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        CU(cudaStreamEndCapture(s, &gs));
        const Counts c1 = counts(API_GEMMEX);
        CHECK(!rt_counters || delta(&c0, &c1, SERVED_SKINNY) == 1, "the captured skinny-sized call is not served");
        CU(cudaGraphInstantiate(&es, gs, 0));
        cublasDestroy(hs);
        cublasDestroy(keep); /* no handle of the variants left */
        printf("replaying the captured skinny call after every handle of the variants is destroyed\n");
        fflush(stdout);
        CU(cudaGraphLaunch(es, s));
        const cudaError_t ss = cudaStreamSynchronize(s);
        CHECK(ss == cudaSuccess, "skinny replay after every handle's destruction: %s", cudaGetErrorName(ss));
        c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, NS, K, &one, sA, CUDA_R_16BF, K, sB, CUDA_R_16BF, K, &zero, sC12,
                    CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        CU(cudaStreamSynchronize(s));
        float *rs = from_device(sC, (size_t)M * NS * 4), *rs12 = from_device(sC12, (size_t)M * NS * 4);
        CHECK(!memcmp(rs, rs12, (size_t)M * NS * 4), "skinny replay after every handle's destruction: not 12.9's bits");
        free(rs);
        free(rs12);
        cudaGraphExecDestroy(es);
        cudaGraphDestroy(gs);
        cudaFree(sA);
        cudaFree(sB);
        cudaFree(sC);
        cudaFree(sC12);
    }

    /* cudaDeviceReset with a live handle, then a new handle in the new context */
    cublasHandle_t live;
    cublasCreate(&live);
    CU(cudaDeviceReset());
    CU(cudaFree(0));
    cudaStream_t s2;
    CU(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    cublasHandle_t h2;
    if (cublasCreate(&h2)) {
        printf("FAIL  cublasCreate after cudaDeviceReset\n");
        return 1;
    }
    cublasSetStream(h2, s2);
    float *after = run_once(h2, s2, &served);
    CHECK(served, "after cudaDeviceReset: a new handle's call is not served");
    CHECK(!memcmp(after, before, (size_t)M * N * 4), "after cudaDeviceReset: a different result");
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("note  %d device(s) visible: two V100s in one process and a V100 beside another GPU are not tested here\n",
           count);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-life\n");
    return 0;
}
