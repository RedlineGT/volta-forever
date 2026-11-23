/* Discriminating arithmetic of the bf16 paths against cuBLAS 12.9 (review section 5, G2): inputs on
 * which a fused multiply-add chain, a multiply-and-add chain, flushing subnormals, and the orders of
 * the epilogue give different bits. Every case uses the NT layout (op(A) = A, op(B) = B^T), where
 * 12.9 runs its sequential kernel at every size of the kernel sweep, and compares every bit with 12.9.
 * Prints "ok bf16-arith". Controls (tests/gpu/forward-bf16.sh): kernels with separate multiply and
 * add, and with beta * C rounded before it is added, must fail. */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cublasLtHandle_t lt12;

static uint16_t bfv(float f) { return f2bf(f); }

/* C = alpha A B^T + beta C through the variant and through 12.9 (A m x k, B n x k, both column-major,
 * leading dimensions m and n); the number of output elements whose bits differ. */
static size_t gemm_pair(const char *what, int m, int n, int k, const uint16_t *hA, const uint16_t *hB, float alpha,
                        float beta, const void *hC, cudaDataType ct, int devmode, int want_path)
{
    const size_t es = ct == CUDA_R_32F ? 4 : 2, cb = (size_t)m * n * es;
    void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)n * k * 2);
    void *Cv = to_device(hC, cb), *C12 = to_device(hC, cb);
    float *dab;
    const float ab[2] = {alpha, beta};
    CU(cudaMalloc((void **)&dab, 8));
    CU(cudaMemcpy(dab, ab, 8, cudaMemcpyHostToDevice));
    const void *pa = devmode ? (const void *)dab : (const void *)&ab[0];
    const void *pb = devmode ? (const void *)(dab + 1) : (const void *)&ab[1];
    const cublasPointerMode_t pm = devmode ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST;
    cublasSetPointerMode(hv, pm);
    c129.SetPointerMode(h12, pm);
    const Counts c0 = counts(API_GEMMEX);
    const cublasStatus_t s1 = cublasGemmEx(hv, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, pa, dA, CUDA_R_16BF, m, dB,
                                           CUDA_R_16BF, n, pb, Cv, ct, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    const cublasStatus_t s2 = c129.GemmEx(h12, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, pa, dA, CUDA_R_16BF, m, dB,
                                          CUDA_R_16BF, n, pb, C12, ct, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaDeviceSynchronize());
    const Counts c1 = counts(API_GEMMEX);
    cublasSetPointerMode(hv, CUBLAS_POINTER_MODE_HOST);
    c129.SetPointerMode(h12, CUBLAS_POINTER_MODE_HOST);
    void *ov = from_device(Cv, cb), *o12 = from_device(C12, cb);
    const size_t d = ndiff(ov, o12, cb, es);
    CHECK(s1 == 0 && s2 == 0, "%s: status %d / %d", what, s1, s2);
    CHECK(!rt_counters || delta(&c0, &c1, want_path) == 1, "%s: not served by the expected path", what);
    if (d) {
        for (size_t i = 0; i < (size_t)m * n; ++i)
            if (memcmp((char *)ov + i * es, (char *)o12 + i * es, es)) {
                if (es == 4)
                    printf("      %s: element %zu: %a (variant) %a (12.9)\n", what, i, ((float *)ov)[i], ((float *)o12)[i]);
                else
                    printf("      %s: element %zu: 0x%04x (variant) 0x%04x (12.9)\n", what, i, ((uint16_t *)ov)[i],
                           ((uint16_t *)o12)[i]);
                break;
            }
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(Cv);
    cudaFree(C12);
    cudaFree(dab);
    free(ov);
    free(o12);
    return d;
}

static void *c_random(int m, int n, cudaDataType ct)
{
    const size_t mn = (size_t)m * n;
    if (ct == CUDA_R_32F) {
        float *c = malloc(mn * 4);
        for (size_t i = 0; i < mn; ++i) c[i] = (float)gauss();
        return c;
    }
    return bf16_random(mn, 1.0);
}

/* Products below 2^-126, inexact as fp32 subnormals: a fused multiply-add chain rounds once per
 * step, a multiply-and-add chain rounds each product to the subnormal grid first, a flushing kernel
 * makes them zero. Rows mix such products with normal ones. */
static void subnormal_products(void)
{
    const int m = 256, n = 8, k = 64;
    uint16_t *A = malloc((size_t)m * k * 2), *B = malloc((size_t)n * k * 2);
    for (int i = 0; i < m; ++i)
        for (int l = 0; l < k; ++l) {
            const int tiny = (i % 4) != 3; /* one row in four mixes in normal products */
            const float mant = (float)(1.0 + (rnd() % 128) / 128.0) * ((rnd() & 1) ? -1.0f : 1.0f);
            A[i + (size_t)l * m] = bfv(ldexpf(mant, tiny ? -70 - (int)(rnd() % 6) : -10));
        }
    for (int j = 0; j < n; ++j)
        for (int l = 0; l < k; ++l) {
            const float mant = (float)(1.0 + (rnd() % 128) / 128.0) * ((rnd() & 1) ? -1.0f : 1.0f);
            B[j + (size_t)l * n] = bfv(ldexpf(mant, -66 - (int)(rnd() % 6)));
        }
    float *C = calloc((size_t)m * n, 4);
    const size_t d = gemm_pair("subnormal products", m, n, k, A, B, 1.0f, 0.0f, C, CUDA_R_32F, 0, SERVED_SKINNY);
    CHECK(d == 0, "subnormal products: %zu elements differ from 12.9", d);
    free(A);
    free(B);
    free(C);
}

/* A product above FLT_MAX that the accumulator brings back: fma(2^64, 2^64, -2^104) = FLT_MAX, while
 * the separate product is inf. */
static void overflow_brought_back(void)
{
    const int m = 128, n = 2, k = 16;
    uint16_t *A = calloc((size_t)m * k, 2), *B = calloc((size_t)n * k, 2);
    for (int i = 0; i < m; ++i) {
        A[i + 0 * (size_t)m] = bfv(-ldexpf(1, 52));
        A[i + 1 * (size_t)m] = bfv(ldexpf(1, 64));
    }
    for (int j = 0; j < n; ++j) {
        B[j + 0 * (size_t)n] = bfv(ldexpf(1, 52));
        B[j + 1 * (size_t)n] = bfv(ldexpf(1, 64));
    }
    float *C = calloc((size_t)m * n, 4);
    const size_t d = gemm_pair("overflow brought back", m, n, k, A, B, 1.0f, 0.0f, C, CUDA_R_32F, 0, SERVED_SKINNY);
    CHECK(d == 0, "overflow brought back by the accumulator: %zu elements differ from 12.9", d);
    free(A);
    free(B);
    free(C);
}

/* The epilogue: alpha * acc rounded, then beta * C added by one fused multiply-add, against the
 * other orders: with alpha = 0.7 and beta = 0.3 or 1/3 neither product is exact. */
static void epilogue_order(void)
{
    const int m = 384, n = 12, k = 96;
    const float betas[2] = {0.3f, 1.0f / 3.0f};
    for (int ct = 0; ct < 2; ++ct)
        for (int b = 0; b < 2; ++b) {
            const cudaDataType t = ct ? CUDA_R_16BF : CUDA_R_32F;
            uint16_t *A = bf16_random((size_t)m * k, 0.05), *B = bf16_random((size_t)n * k, 1.0);
            void *C = c_random(m, n, t);
            char what[64];
            snprintf(what, sizeof what, "epilogue alpha 0.7 beta %s %s", b ? "1/3" : "0.3", ct ? "bf16" : "fp32");
            const size_t d = gemm_pair(what, m, n, k, A, B, 0.7f, betas[b], C, t, 0, SERVED_SKINNY);
            CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
            free(A);
            free(B);
            free(C);
        }
    /* The fallback of the tensor-core path has the same epilogue: rows that fail the check (a value
     * below the exponent window) are computed there, with finite results. */
    const int M = 512, N = 64, K = 512;
    uint16_t *A = bf16_random((size_t)M * K, 0.05), *B = bf16_random((size_t)N * K, 1.0);
    for (int i = 0; i < M; i += 7) A[i + (size_t)(i % K) * M] = bfv(1e-30f);
    float *C = c_random(M, N, CUDA_R_32F);
    float *Cv = NULL;
    (void)Cv;
    const size_t es = 4, cb = (size_t)M * N * es;
    void *dA = to_device(A, (size_t)M * K * 2), *dB = to_device(B, (size_t)N * K * 2);
    void *C1 = to_device(C, cb), *C2 = to_device(C, cb);
    const float alpha = 0.7f, beta = 1.0f / 3.0f;
    cublasGemmEx(hv, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, dA, CUDA_R_16BF, M, dB, CUDA_R_16BF, N, &beta, C1,
                 CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    c129.GemmEx(h12, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, dA, CUDA_R_16BF, M, dB, CUDA_R_16BF, N, &beta, C2,
                CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaDeviceSynchronize());
    float *o1 = from_device(C1, cb), *o2 = from_device(C2, cb);
    size_t bad = 0, rows = 0;
    for (int i = 0; i < M; i += 7, ++rows)
        for (int j = 0; j < N; ++j) bad += memcmp(&o1[i + (size_t)j * M], &o2[i + (size_t)j * M], 4) != 0;
    CHECK(bad == 0, "fallback rows with alpha 0.7 beta 1/3: %zu of %zu elements differ from 12.9", bad, rows * N);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(C1);
    cudaFree(C2);
    free(o1);
    free(o2);
    free(A);
    free(B);
    free(C);
}

/* cuBLASLt with a bias, alpha != 1 and beta != 0 together: the order alpha * acc, + beta * C, + bias. */
static void lt_bias(void)
{
    const int m = 512, n = 8, k = 384;
    for (int ct = 0; ct < 2; ++ct) {
        const cudaDataType t = ct ? CUDA_R_16BF : CUDA_R_32F;
        const size_t es = ct ? 2 : 4, cb = (size_t)m * n * es;
        uint16_t *A = bf16_random((size_t)m * k, 0.05), *B = bf16_random((size_t)n * k, 1.0);
        void *C = c_random(m, n, t), *bias = c_random(m, 1, t);
        void *dA = to_device(A, (size_t)m * k * 2), *dB = to_device(B, (size_t)n * k * 2), *dbias = to_device(bias, m * es);
        void *C1 = to_device(C, cb), *C2 = to_device(C, cb);
        cublasLtHandle_t ltv;
        cublasLtCreate(&ltv);
        cublasLtMatmulDesc_t desc;
        cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        const int32_t tb = CUBLAS_OP_T;
        const uint32_t epi = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi);
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof dbias);
        cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, m, k, m);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, n, k, n);
        cublasLtMatrixLayoutCreate(&lc, t, m, n, m);
        const float alpha = 0.7f, beta = 0.3f;
        const Counts c0 = counts(API_LT);
        const cublasStatus_t s1 = cublasLtMatmul(ltv, desc, &alpha, dA, la, dB, lb, &beta, C1, lc, C1, lc, NULL, NULL, 0, 0);
        const cublasStatus_t s2 = c129.LtMatmul(lt12, desc, &alpha, dA, la, dB, lb, &beta, C2, lc, C2, lc, NULL, NULL, 0, 0);
        CU(cudaDeviceSynchronize());
        const Counts c1 = counts(API_LT);
        void *o1 = from_device(C1, cb), *o2 = from_device(C2, cb);
        CHECK(s1 == 0 && s2 == 0, "cuBLASLt bias %s: status %d / %d", ct ? "bf16" : "fp32", s1, s2);
        CHECK(!rt_counters || delta(&c0, &c1, SERVED_SKINNY) == 1, "cuBLASLt bias %s: not served by the skinny path",
              ct ? "bf16" : "fp32");
        const size_t d = ndiff(o1, o2, cb, es);
        CHECK(d == 0, "cuBLASLt bias, alpha 0.7, beta 0.3, %s: %zu elements differ from 12.9", ct ? "bf16" : "fp32", d);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulDescDestroy(desc);
        cublasLtDestroy(ltv);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dbias);
        cudaFree(C1);
        cudaFree(C2);
        free(o1);
        free(o2);
        free(A);
        free(B);
        free(C);
        free(bias);
    }
}

/* Device pointer mode with alpha = 0 (read when the kernel runs): rows whose sum is negative give
 * -0, rows with NaN give NaN. All -0 product rows with alpha < 0 and beta = 0. NaN of both signs and
 * with payloads in A, in both output types. */
static void special_values(void)
{
    const int m = 256, n = 4, k = 64;
    for (int ct = 0; ct < 2; ++ct) {
        const cudaDataType t = ct ? CUDA_R_16BF : CUDA_R_32F;
        uint16_t *A = bf16_random((size_t)m * k, 0.05), *B = bf16_random((size_t)n * k, 1.0);
        for (int i = 0; i < m; i += 5) A[i + (size_t)(i % k) * m] = (i / 5) % 3 == 0 ? 0x7fc1 : (i / 5) % 3 == 1 ? 0xffa5 : 0xff80;
        void *C = c_random(m, n, t);
        char what[64];
        snprintf(what, sizeof what, "device alpha 0, NaN in A, %s", ct ? "bf16" : "fp32");
        size_t d = gemm_pair(what, m, n, k, A, B, 0.0f, 0.0f, C, t, 1, SERVED_SKINNY);
        CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
        snprintf(what, sizeof what, "NaN payloads in A, alpha 1, beta 0.5, %s", ct ? "bf16" : "fp32");
        d = gemm_pair(what, m, n, k, A, B, 1.0f, 0.5f, C, t, 0, SERVED_SKINNY);
        CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
        snprintf(what, sizeof what, "device alpha 0, beta -0, NaN in A, %s", ct ? "bf16" : "fp32");
        d = gemm_pair(what, m, n, k, A, B, 0.0f, -0.0f, C, t, 1, SERVED_SKINNY);
        CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
        for (int i = 0; i < m; i += 3)
            for (int l = 0; l < k; ++l) A[i + (size_t)l * m] = 0x8000; /* -0 */
        snprintf(what, sizeof what, "-0 product rows, alpha -1, beta 0, %s", ct ? "bf16" : "fp32");
        d = gemm_pair(what, m, n, k, A, B, -1.0f, 0.0f, C, t, 0, SERVED_SKINNY);
        CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
        snprintf(what, sizeof what, "-0 product rows, alpha -1, beta -0, %s", ct ? "bf16" : "fp32");
        d = gemm_pair(what, m, n, k, A, B, -1.0f, -0.0f, C, t, 0, SERVED_SKINNY);
        CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
        snprintf(what, sizeof what, "-0 product rows, alpha 1, beta -0, %s", ct ? "bf16" : "fp32");
        d = gemm_pair(what, m, n, k, A, B, 1.0f, -0.0f, C, t, 0, SERVED_SKINNY);
        CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
        free(A);
        free(B);
        free(C);
    }
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    if (cublasCreate(&hv) || c129.Create(&h12) || c129.LtCreate(&lt12)) {
        printf("FAIL  cannot create the handles\n");
        return 1;
    }
    open_runtime();
    subnormal_products();
    overflow_brought_back();
    epilogue_order();
    lt_bias();
    special_values();
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-arith\n");
    return 0;
}
