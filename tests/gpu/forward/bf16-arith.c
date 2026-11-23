/* Discriminating arithmetic of the bf16 paths against cuBLAS 12.9 (review section 5, G2): inputs on
 * which a fused multiply-add chain, a multiply-and-add chain, flushing subnormals, and the orders of
 * the epilogue give different bits. The cases use the NT layout (op(A) = A, op(B) = B^T), where 12.9
 * runs its sequential kernel at every size of the kernel sweep, and the cuBLASLt signed zeros with a
 * bias also TN, and compare every bit with 12.9. Prints "ok bf16-arith". Controls
 * (tests/gpu/forward-bf16.sh): kernels with separate multiply and add, with beta * C rounded before it
 * is added, without the +0 term, with it only when beta is zero, and with it also with a bias, must
 * fail. */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cublasLtHandle_t lt12;

static uint16_t bfv(float f) { return f2bf(f); }

/* C = alpha A B^T + beta C through the variant and through 12.9 (A m x k, B n x k, both column-major,
 * leading dimensions m and n); the number of output elements whose bits differ, over every row, or
 * with row_step > 0 over the rows i % row_step == 0 only. */
static size_t gemm_pair_rows(const char *what, int m, int n, int k, const uint16_t *hA, const uint16_t *hB, float alpha,
                             float beta, const void *hC, cudaDataType ct, int devmode, int want_path, int row_step)
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
    size_t d = 0;
    for (size_t i = 0; i < (size_t)m * n; ++i)
        d += (row_step <= 0 || (long long)(i % (size_t)m) % row_step == 0) &&
             memcmp((char *)ov + i * es, (char *)o12 + i * es, es) != 0;
    CHECK(s1 == 0 && s2 == 0, "%s: status %d / %d", what, s1, s2);
    CHECK(!rt_counters || delta(&c0, &c1, want_path) == 1, "%s: not served by the expected path", what);
    if (d) {
        for (size_t i = 0; i < (size_t)m * n; ++i)
            if ((row_step <= 0 || (long long)(i % (size_t)m) % row_step == 0) &&
                memcmp((char *)ov + i * es, (char *)o12 + i * es, es)) {
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

static size_t gemm_pair(const char *what, int m, int n, int k, const uint16_t *hA, const uint16_t *hB, float alpha,
                        float beta, const void *hC, cudaDataType ct, int devmode, int want_path)
{
    return gemm_pair_rows(what, m, n, k, hA, hB, alpha, beta, hC, ct, devmode, want_path, 0);
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
    const int m = 128, n = 2, k = 64;
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

/* Signed zeros: with alpha < 0 and an exactly zero dot product, alpha * acc = -0; when beta * C = -0
 * is added next (beta 1 or 0.5 with C = -0, beta -1 with C = +0), cuBLAS 12.9 gives +0, as when +0 is
 * added to alpha * acc before beta * C. A's rows i % 3 == 0 are zeros (+0 and -0 by turns); host and
 * device pointer modes, both output types. The skinny path: every element compared with 12.9. The
 * tensor-core path (its epilogue and its fallback finish the same way): the zero rows compared, since
 * elsewhere its bits are not 12.9's. The kernel control ZERO_BETA0 (+0 only when beta is zero, as
 * before) must fail these cases. */
static void signed_zeros(void)
{
    static const struct {
        float alpha, beta, c;
        int dev;
        const char *what;
    } cases[] = {
        {-1.0f, 1.0f, -0.0f, 0, "alpha -1, beta 1, C = -0"},
        {-1.0f, 0.5f, -0.0f, 0, "alpha -1, beta 0.5, C = -0"},
        {-1.0f, 0.5f, -0.0f, 1, "alpha -1, beta 0.5, C = -0, device pointers"},
        {-1.0f, -1.0f, 0.0f, 0, "alpha -1, beta -1, C = +0"},
        {-1.0f, -1.0f, 0.0f, 1, "alpha -1, beta -1, C = +0, device pointers"},
        {1.0f, 1.0f, -0.0f, 0, "alpha 1, beta 1, C = -0"},
    };
    for (int tc = 0; tc < 2; ++tc) {
        const int m = tc ? 512 : 256, n = tc ? 64 : 4, k = tc ? 512 : 64;
        for (int ct = 0; ct < 2; ++ct) {
            const cudaDataType t = ct ? CUDA_R_16BF : CUDA_R_32F;
            uint16_t *A = bf16_random((size_t)m * k, 0.05), *B = bf16_random((size_t)n * k, 1.0);
            for (int i = 0; i < m; i += 3)
                for (int l = 0; l < k; ++l) A[i + (size_t)l * m] = (i / 3) % 2 ? 0x8000 : 0x0000;
            void *C = malloc((size_t)m * n * 4);
            for (size_t c = 0; c < sizeof cases / sizeof *cases; ++c) {
                for (size_t e = 0; e < (size_t)m * n; ++e) {
                    if (ct)
                        ((uint16_t *)C)[e] = bfv(cases[c].c);
                    else
                        ((float *)C)[e] = cases[c].c;
                }
                char what[128];
                snprintf(what, sizeof what, "signed zeros, %s path, %s, %s", tc ? "tensor-core" : "skinny", cases[c].what,
                         ct ? "bf16" : "fp32");
                const size_t d = gemm_pair_rows(what, m, n, k, A, B, cases[c].alpha, cases[c].beta, C, t, cases[c].dev,
                                                tc ? SERVED_TC : SERVED_SKINNY, tc ? 3 : 0);
                CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
            }
            free(A);
            free(B);
            free(C);
        }
    }
}

/* cuBLASLt D = alpha op(A) op(B) + beta C (+ bias) through the variant and through 12.9 (D = C in place,
 * column-major, op(A) m x k and op(B) k x n as ta and tb say, a 32 MiB workspace); the number of output
 * elements whose bits differ, over every row or, with row_step > 0, over the rows i % row_step == 0, and
 * among those the -0 elements of 12.9's output and of the variant's (added to *neg12 and *negv). */
static size_t lt_pair(const char *what, int ta, int tb, int m, int n, int k, const uint16_t *hA, const uint16_t *hB,
                      float alpha, float beta, const void *hC, const void *hbias, cudaDataType ct, int devmode,
                      int want_path, int row_step, void *ws, size_t *neg12, size_t *negv)
{
    static cublasLtHandle_t ltv;
    if (!ltv) cublasLtCreate(&ltv);
    const size_t es = ct == CUDA_R_32F ? 4 : 2, cb = (size_t)m * n * es;
    void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)n * k * 2);
    void *Cv = to_device(hC, cb), *C12 = to_device(hC, cb), *dbias = hbias ? to_device(hbias, (size_t)m * es) : NULL;
    float *dab;
    const float ab[2] = {alpha, beta};
    CU(cudaMalloc((void **)&dab, 8));
    CU(cudaMemcpy(dab, ab, 8, cudaMemcpyHostToDevice));
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const int32_t oa = ta ? CUBLAS_OP_T : CUBLAS_OP_N, ob = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int32_t pm = devmode ? CUBLASLT_POINTER_MODE_DEVICE : CUBLASLT_POINTER_MODE_HOST;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof ob);
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof pm);
    if (hbias) {
        const uint32_t epi = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi);
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof dbias);
    }
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, ta ? k : m, ta ? m : k, ta ? k : m);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, tb ? n : k, tb ? k : n, tb ? n : k);
    cublasLtMatrixLayoutCreate(&lc, ct, m, n, m);
    const void *pa = devmode ? (const void *)dab : (const void *)&ab[0];
    const void *pb = devmode ? (const void *)(dab + 1) : (const void *)&ab[1];
    const Counts c0 = counts(API_LT);
    const cublasStatus_t s1 = cublasLtMatmul(ltv, desc, pa, dA, la, dB, lb, pb, Cv, lc, Cv, lc, NULL, ws, 32u << 20, 0);
    CU(cudaDeviceSynchronize());
    const Counts c1 = counts(API_LT);
    const cublasStatus_t s2 = c129.LtMatmul(lt12, desc, pa, dA, la, dB, lb, pb, C12, lc, C12, lc, NULL, ws, 32u << 20, 0);
    CU(cudaDeviceSynchronize());
    void *ov = from_device(Cv, cb), *o12 = from_device(C12, cb);
    size_t d = 0, first = (size_t)-1;
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        if (row_step > 0 && (long long)(i % (size_t)m) % row_step != 0) continue;
        const uint32_t bv = es == 4 ? ((uint32_t *)ov)[i] : ((uint16_t *)ov)[i];
        const uint32_t b12 = es == 4 ? ((uint32_t *)o12)[i] : ((uint16_t *)o12)[i];
        const uint32_t negz = es == 4 ? 0x80000000u : 0x8000u;
        *neg12 += b12 == negz;
        *negv += bv == negz;
        if (bv != b12 && !d++) first = i;
    }
    CHECK(s1 == 0 && s2 == 0, "%s: status %d / %d", what, s1, s2);
    CHECK(!rt_counters || delta(&c0, &c1, want_path) == 1, "%s: not served by the expected path", what);
    if (d) {
        if (es == 4)
            printf("      %s: element %zu: %a (variant) %a (12.9)\n", what, first, ((float *)ov)[first], ((float *)o12)[first]);
        else
            printf("      %s: element %zu: 0x%04x (variant) 0x%04x (12.9)\n", what, first, ((uint16_t *)ov)[first],
                   ((uint16_t *)o12)[first]);
    }
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(desc);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(Cv);
    cudaFree(C12);
    if (dbias) cudaFree(dbias);
    cudaFree(dab);
    free(ov);
    free(o12);
    return d;
}

/* Signed zeros of cublasLtMatmul with and without a bias. cuBLAS 12.9 always adds a bias term to
 * alpha * acc: the bias element, or +0 without a bias. So without a bias (or with a bias of +0) an
 * exactly zero result is +0, and with a bias of -0 it is -0 when alpha * acc and beta * C (beta not
 * zero) are -0 too. Rows i % 3 == 0 of op(A) are zeros (all +0, all -0, or both by turns); alpha -1
 * and 1 (host), -0 and +0 (device); beta 0, -0, 1, -1 and 0.5; C all +0 or all -0; no bias, a bias of
 * +0, a bias of -0 (of the output's type); both output types. The skinny path in the NT and TN layouts,
 * every element compared with 12.9; the tensor-core path (TN), the zero rows compared, since elsewhere
 * its bits are not 12.9's. 12.9 must give -0 somewhere in these cases. The kernel control ZERO_ALWAYS
 * (+0 added with a bias too) must fail them. */
static void lt_signed_zeros(void)
{
    static const struct {
        int ta, tb, m, n, k, path;
        const char *what;
    } shapes[] = {
        {0, 1, 512, 8, 384, SERVED_SKINNY, "skinny path, NT 512 x 8 x 384"},
        {1, 0, 1536, 8, 1536, SERVED_SKINNY, "skinny path, TN 1536 x 8 x 1536"},
        {1, 0, 1536, 64, 2048, SERVED_TC, "tensor-core path, TN 1536 x 64 x 2048"},
    };
    static const struct {
        float alpha;
        int dev;
    } alphas[] = {{-1.0f, 0}, {1.0f, 0}, {-0.0f, 1}, {0.0f, 1}};
    static const float betas[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f};
    void *ws;
    CU(cudaMalloc(&ws, 32u << 20));
    size_t neg12_all = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; ++s) {
        const int ta = shapes[s].ta, m = shapes[s].m, n = shapes[s].n, k = shapes[s].k;
        const int tc = shapes[s].path == SERVED_TC;
        uint16_t *A = bf16_random((size_t)m * k, 0.05), *B = bf16_random((size_t)n * k, 1.0);
        for (int i = 0; i < m; i += 3)
            for (int l = 0; l < k; ++l) {
                const int kind = (i / 3) % 3;
                A[ta ? (size_t)l + (size_t)i * k : (size_t)i + (size_t)l * m] = kind == 0 ? 0x0000 : kind == 1 ? 0x8000
                                                                                : (l % 2 ? 0x8000 : 0x0000);
            }
        for (int ct = 0; ct < 2; ++ct) {
            const cudaDataType t = ct ? CUDA_R_16BF : CUDA_R_32F;
            const size_t es = ct ? 2 : 4;
            void *C = malloc((size_t)m * n * es), *bias = malloc((size_t)m * es);
            for (int b = 0; b < 3; ++b) { /* no bias, +0, -0 */
                for (int i = 0; i < m; ++i) {
                    if (ct)
                        ((uint16_t *)bias)[i] = b == 2 ? 0x8000 : 0x0000;
                    else
                        ((float *)bias)[i] = b == 2 ? -0.0f : 0.0f;
                }
                size_t calls = 0, bad = 0, neg12 = 0, negv = 0;
                for (size_t a = 0; a < sizeof alphas / sizeof *alphas; ++a)
                    for (size_t be = 0; be < sizeof betas / sizeof *betas; ++be)
                        for (int cs = 0; cs < 2; ++cs) {
                            for (size_t e = 0; e < (size_t)m * n; ++e) {
                                if (ct)
                                    ((uint16_t *)C)[e] = cs ? 0x8000 : 0x0000;
                                else
                                    ((float *)C)[e] = cs ? -0.0f : 0.0f;
                            }
                            char what[192];
                            snprintf(what, sizeof what, "cuBLASLt signed zeros, %s, %s, alpha %s%s, beta %g, C = %s0, %s",
                                     shapes[s].what, b == 0 ? "no bias" : b == 1 ? "bias +0" : "bias -0",
                                     signbit(alphas[a].alpha) ? "-" : "", alphas[a].dev ? "0 (device)" : "1",
                                     (double)betas[be], cs ? "-" : "+", ct ? "bf16" : "fp32");
                            const size_t d = lt_pair(what, ta, shapes[s].tb, m, n, k, A, B, alphas[a].alpha, betas[be], C,
                                                     b ? bias : NULL, t, alphas[a].dev, shapes[s].path, tc ? 3 : 0, ws,
                                                     &neg12, &negv);
                            CHECK(d == 0, "%s: %zu elements differ from 12.9", what, d);
                            ++calls;
                            bad += d != 0;
                        }
                printf("cuBLASLt signed zeros, %s, %s, %s: %zu calls, %zu with elements differing from 12.9; -0 in %zu "
                       "elements from 12.9, %zu from the variant\n", shapes[s].what, b == 0 ? "no bias" : b == 1 ? "bias +0"
                       : "bias -0", ct ? "bf16" : "fp32", calls, bad, neg12, negv);
                neg12_all += neg12;
            }
            free(C);
            free(bias);
        }
        free(A);
        free(B);
    }
    CHECK(neg12_all > 0, "cuBLASLt signed zeros: 12.9 gave -0 nowhere: the cases do not test the bias of -0");
    cudaFree(ws);
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
    signed_zeros();
    lt_signed_zeros();
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-arith\n");
    return 0;
}
