/* A stand-in for libcublas.so.12 for the host tests of the bf16 runtime: handles that keep their
 * settings, and products that only record their arguments (a forwarded call reaches 12.9 here).
 * stub_chaos plays another thread: right after the handle's next pointer-mode query (the runtime's
 * last), the handle's stream and pointer mode change. */
#include "stub.h"
#include <stdlib.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))

typedef struct {
    cudaStream_t stream;
    cublasPointerMode_t pmode;
    cublasMath_t math;
    size_t workspace;
    int chaos;
    cudaStream_t chaos_stream;
} Handle;

static void maybe_chaos(Handle *h)
{
    if (h->chaos) {
        h->chaos = 0;
        h->stream = h->chaos_stream;
        h->pmode = CUBLAS_POINTER_MODE_DEVICE;
        stub_event("chaos handle=%p stream=%p pmode=device", (void *)h, (void *)h->stream);
    }
}
EXPORT void stub_chaos(cublasHandle_t h, cudaStream_t s)
{
    ((Handle *)h)->chaos = 1;
    ((Handle *)h)->chaos_stream = s;
}

/* The version reported: 12.9.2, or STUB_CUBLAS_VERSION (major*10000 + minor*100 + patch). */
static int version(void)
{
    const char *v = getenv("STUB_CUBLAS_VERSION");
    return v && *v ? atoi(v) : 120902;
}
EXPORT cublasStatus_t cublasGetProperty(libraryPropertyType t, int *value)
{
    const int v = version();
    *value = t == MAJOR_VERSION ? v / 10000 : t == MINOR_VERSION ? v / 100 % 100 : v % 100;
    return CUBLAS_STATUS_SUCCESS;
}

EXPORT cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
{
    Handle *h = calloc(1, sizeof *h);
    h->workspace = 4u << 20;
    *handle = (cublasHandle_t)h;
    stub_event("cublasCreate handle=%p", (void *)h);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
{
    stub_event("cublasDestroy handle=%p", (void *)handle);
    free(handle);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *s)
{
    *s = ((Handle *)handle)->stream;
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t s)
{
    ((Handle *)handle)->stream = s;
    ((Handle *)handle)->workspace = 4u << 20; /* as cuBLAS 12: the default workspace again */
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *w, size_t size)
{
    ((Handle *)handle)->workspace = size;
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *m)
{
    *m = ((Handle *)handle)->pmode;
    maybe_chaos((Handle *)handle);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t m)
{
    ((Handle *)handle)->pmode = m;
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *m)
{
    *m = ((Handle *)handle)->math;
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t m)
{
    ((Handle *)handle)->math = m;
    return CUBLAS_STATUS_SUCCESS;
}

#define GEMM_EVENT(name, fmt_sizes, ...)                                                                      \
    stub_event(name " handle=%p ta=%d tb=%d " fmt_sizes " alpha=%p A=%p at=%d B=%p bt=%d beta=%p C=%p ct=%d " \
                    "compute=%d algo=%d stream=%p pmode=%d",                                                  \
               __VA_ARGS__)

EXPORT cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                   const void *alpha, const void *A, cudaDataType at, int lda, const void *B,
                                   cudaDataType bt, int ldb, const void *beta, void *C, cudaDataType ct, int ldc,
                                   cublasComputeType_t compute, cublasGemmAlgo_t algo)
{
    GEMM_EVENT("gemmex", "m=%d n=%d k=%d lda=%d ldb=%d ldc=%d", (void *)h, ta, tb, m, n, k, lda, ldb, ldc, alpha, A,
               at, B, bt, beta, C, ct, compute, algo, (void *)((Handle *)h)->stream, ((Handle *)h)->pmode);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasGemmEx_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int64_t m,
                                      int64_t n, int64_t k, const void *alpha, const void *A, cudaDataType at,
                                      int64_t lda, const void *B, cudaDataType bt, int64_t ldb, const void *beta,
                                      void *C, cudaDataType ct, int64_t ldc, cublasComputeType_t compute,
                                      cublasGemmAlgo_t algo)
{
    GEMM_EVENT("gemmex64", "m=%lld n=%lld k=%lld lda=%lld ldb=%lld ldc=%lld", (void *)h, ta, tb, (long long)m,
               (long long)n, (long long)k, (long long)lda, (long long)ldb, (long long)ldc, alpha, A, at, B, bt, beta, C,
               ct, compute, algo, (void *)((Handle *)h)->stream, ((Handle *)h)->pmode);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,
                                                 int n, int k, const void *alpha, const void *A, cudaDataType at,
                                                 int lda, long long sa, const void *B, cudaDataType bt, int ldb,
                                                 long long sb, const void *beta, void *C, cudaDataType ct, int ldc,
                                                 long long sc, int batch, cublasComputeType_t compute,
                                                 cublasGemmAlgo_t algo)
{
    GEMM_EVENT("gemmsb", "m=%d n=%d k=%d lda=%d ldb=%d ldc=%d sa=%lld sb=%lld sc=%lld batch=%d", (void *)h, ta, tb, m,
               n, k, lda, ldb, ldc, sa, sb, sc, batch, alpha, A, at, B, bt, beta, C, ct, compute, algo,
               (void *)((Handle *)h)->stream, ((Handle *)h)->pmode);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasGemmStridedBatchedEx_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                                    int64_t m, int64_t n, int64_t k, const void *alpha, const void *A,
                                                    cudaDataType at, int64_t lda, long long sa, const void *B,
                                                    cudaDataType bt, int64_t ldb, long long sb, const void *beta,
                                                    void *C, cudaDataType ct, int64_t ldc, long long sc, int64_t batch,
                                                    cublasComputeType_t compute, cublasGemmAlgo_t algo)
{
    GEMM_EVENT("gemmsb64", "m=%lld n=%lld k=%lld batch=%lld", (void *)h, ta, tb, (long long)m, (long long)n,
               (long long)k, (long long)batch, alpha, A, at, B, bt, beta, C, ct, compute, algo,
               (void *)((Handle *)h)->stream, ((Handle *)h)->pmode);
    return CUBLAS_STATUS_SUCCESS;
}
