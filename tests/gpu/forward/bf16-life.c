/* Lifetimes (review section 5, G6): a graph replayed after the handle that captured it is destroyed
 * (the module lives as long as its context); cudaDeviceReset with live handles, then a new handle and
 * a call served in the new context, with the bits it had before. Two V100s in one process and a V100
 * beside another GPU need devices this test is not given (it says so). Prints "ok bf16-life". */
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
    cublasHandle_t h;
    if (cublasCreate(&h)) return 1;
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
