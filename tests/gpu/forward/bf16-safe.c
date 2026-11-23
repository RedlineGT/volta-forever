/* VF_SAFE=1, the layer's conservative setting, with VF_CUBLAS_BF16_TC=1 (tests/gpu/forward-bf16.sh sets
 * both): the tensor-core path serves nothing, so every call it would have served has cuBLAS 12.9's bits
 * (legacy TN, NN and NT, and cuBLASLt TN with a bias); the skinny path, which gives 12.9's bits, still
 * serves its calls. 12.9's own results come from the same arguments given to the CUDA 12.9 libraries
 * directly. The test runtime's control safe-ignored (VF_SAFE not read) must make it fail. Prints
 * "ok bf16-safe". */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cublasLtHandle_t ltv, lt12;
static cudaStream_t st;

static void legacy(const char *what, int ta, int tb, int m, int n, int k, int want_skinny)
{
    uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)n * k, 1.0);
    void *A = to_device(hA, (size_t)m * k * 2), *B = to_device(hB, (size_t)n * k * 2), *C1, *C2;
    const size_t cb = (size_t)m * n * 4;
    CU(cudaMalloc(&C1, cb));
    CU(cudaMalloc(&C2, cb));
    const float one = 1.0f, zero = 0.0f;
    const int lda = ta ? k : m, ldb = tb ? n : k;
    const Counts k0 = counts(API_GEMMEX);
    const cublasStatus_t s1 = cublasGemmEx(hv, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one,
                                           A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb, &zero, C1, CUDA_R_32F, m,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(st));
    const Counts k1 = counts(API_GEMMEX);
    const cublasStatus_t s2 = c129.GemmEx(h12, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one,
                                          A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb, &zero, C2, CUDA_R_32F, m,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(st));
    float *r1 = from_device(C1, cb), *r2 = from_device(C2, cb);
    const size_t d = ndiff(r1, r2, cb, 4);
    const unsigned long long tc = delta(&k0, &k1, SERVED_TC), sk = delta(&k0, &k1, SERVED_SKINNY);
    printf("%-34s status %d / 12.9 %d; tensor-core %llu, skinny %llu; %zu elements differ from 12.9's\n", what, (int)s1,
           (int)s2, tc, sk, d);
    CHECK(s1 == 0 && s2 == 0 && d == 0 && (!rt_counters || (tc == 0 && sk == (unsigned long long)want_skinny)),
          "%s: not 12.9's bits, or served by the tensor-core path, or the skinny path did not serve as it should", what);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C1);
    cudaFree(C2);
    free(hA);
    free(hB);
    free(r1);
    free(r2);
}

static void lt_bias(int m, int n, int k)
{
    uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)n * k, 1.0), *hb = bf16_random(m, 1.0);
    void *A = to_device(hA, (size_t)m * k * 2), *B = to_device(hB, (size_t)n * k * 2), *bias = to_device(hb, (size_t)m * 2);
    void *D1, *D2, *ws;
    const size_t db = (size_t)m * n * 2;
    CU(cudaMalloc(&D1, db));
    CU(cudaMalloc(&D2, db));
    CU(cudaMalloc(&ws, 1u << 20));
    cublasLtMatmulDesc_t d;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const int32_t oa = CUBLAS_OP_T;
    const uint32_t e = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof e);
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, n, k);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, m, n, m);
    const float one = 1.0f, zero = 0.0f;
    const Counts k0 = counts(API_LT);
    const cublasStatus_t s1 = cublasLtMatmul(ltv, d, &one, A, la, B, lb, &zero, D1, lc, D1, lc, NULL, ws, 1u << 20, st);
    CU(cudaStreamSynchronize(st));
    const Counts k1 = counts(API_LT);
    const cublasStatus_t s2 = c129.LtMatmul(lt12, d, &one, A, la, B, lb, &zero, D2, lc, D2, lc, NULL, ws, 1u << 20, st);
    CU(cudaStreamSynchronize(st));
    uint16_t *r1 = from_device(D1, db), *r2 = from_device(D2, db);
    const size_t diff = ndiff(r1, r2, db, 2);
    const unsigned long long tc = delta(&k0, &k1, SERVED_TC);
    printf("cuBLASLt TN %d x %d x %d with a bias  status %d / 12.9 %d; tensor-core %llu; %zu elements differ from 12.9's\n",
           m, n, k, (int)s1, (int)s2, tc, diff);
    CHECK(s1 == 0 && s2 == 0 && diff == 0 && (!rt_counters || tc == 0),
          "cuBLASLt TN %d x %d x %d with a bias: not 12.9's bits, or served by the tensor-core path", m, n, k);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(d);
    cudaFree(A);
    cudaFree(B);
    cudaFree(bias);
    cudaFree(D1);
    cudaFree(D2);
    cudaFree(ws);
    free(hA);
    free(hB);
    free(hb);
    free(r1);
    free(r2);
}

int main(void)
{
    const char *safe = getenv("VF_SAFE"), *tc = getenv("VF_CUBLAS_BF16_TC");
    printf("VF_SAFE=%s VF_CUBLAS_BF16_TC=%s\n", safe ? safe : "(unset)", tc ? tc : "(unset)");
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasCreate(&hv) || c129.Create(&h12) || cublasLtCreate(&ltv) || c129.LtCreate(&lt12)) return 1;
    open_runtime();
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    legacy("TN 1536 x 256 x 1536", 1, 0, 1536, 256, 1536, 0);
    legacy("TN 4096 x 64 x 4096", 1, 0, 4096, 64, 4096, 0);
    legacy("NN 2048 x 128 x 2048", 0, 0, 2048, 128, 2048, 0);
    legacy("NT 1536 x 96 x 1536", 0, 1, 1536, 96, 1536, 0);
    legacy("TN 1536 x 8 x 1536 (skinny)", 1, 0, 1536, 8, 1536, 1);
    lt_bias(1536, 512, 1536);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-safe\n");
    return 0;
}
