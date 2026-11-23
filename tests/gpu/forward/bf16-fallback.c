/* The tensor-core path's fallback in the layout of linear layers (review F4): TN with aligned operands,
 * so the rows and columns that fail the strict conversion check are computed by vf_bf16_fallback_tn.
 * Columns of op(B) fail through one element far below the column's largest (1e-30), or an infinity or
 * a NaN; rows of op(A) the same way. Failing-column counts on both sides of 8 and 32 (1 to 300), with
 * failing rows; fp32 output with alpha 0.7, beta 0.3 and C in place, and bf16 output with beta 0.
 * Every element of a failing row or column must have the bits of a host model of the stated arithmetic
 * (one fp32 fused multiply-add chain from +0 over l = 0 .. k-1, alpha * acc rounded, then fma(beta, C, .)
 * or + 0, bf16 rounded to nearest even; a NaN is 0x7fffffff in fp32, 0x7fff in bf16); every other
 * element within 32 units of 2^-24 sum |a b| of fp64. Two cases have subnormal products in failing
 * columns, where a fat binary of separate multiplies and adds (VF_BF16_FATBIN=control-MULADD.fatbin)
 * gives other bits and makes the test fail. Prints "ok bf16-fallback". */
#include "bf16-common.h"

static cublasHandle_t hv;
static cudaStream_t st;

static uint32_t fbits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

/* The host model of one element (TN: op(A)(i, l) = A[l + i lda], op(B)(l, j) = B[l + j ldb]). */
static float model(const uint16_t *A, int lda, const uint16_t *B, int ldb, int k, int i, int j, float alpha, float beta,
                   float c0)
{
    float acc = 0.0f;
    for (int l = 0; l < k; ++l) acc = fmaf(bf2f(A[l + (long long)i * lda]), bf2f(B[l + (long long)j * ldb]), acc);
    volatile float r = alpha * acc;
    float out = beta != 0.0f ? fmaf(beta, c0, r) : r + 0.0f;
    return out;
}

static void one_case(int m, int n, int k, int ncols, int nrows, int bad_kind, int bf16_out, int tiny)
{
    const int lda = k, ldb = k;
    uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)n * k, 1.0);
    /* tiny: rows of op(A) of both signs near 2^-70..2^-75 and columns of op(B) near 2^-66..2^-71 (one
     * element of 2^-110 makes each column fail the check), as in bf16-arith.c: their products, below
     * 2^-126, are fp32 subnormals that a separate multiply rounds to the subnormal grid, and with mixed
     * signs the sum stays small enough for that rounding to show: a chain of fused multiply-adds and
     * one of multiplies and adds then differ in those elements. */
    const int tiny_rows[] = {5, 600, 1200}, tiny_cols[] = {7, 100, 250};
    char *tinyrow = calloc((size_t)m, 1), *tinycol = calloc((size_t)n, 1);
    for (int t = 0; tiny && t < 3; ++t) {
        const int i = tiny_rows[t] % m, j = tiny_cols[t] % n;
        tinyrow[i] = tinycol[j] = 1;
        for (int l = 0; l < k; ++l) {
            const float sa = (rnd() & 1) ? -1.0f : 1.0f, sb = (rnd() & 1) ? -1.0f : 1.0f;
            hA[(size_t)l + (size_t)i * lda] = f2bf(sa * ldexpf(1.0f + (float)(rnd() % 128) / 128.0f, -70 - (int)(rnd() % 6)));
            hB[(size_t)l + (size_t)j * ldb] = f2bf(sb * ldexpf(1.0f + (float)(rnd() % 128) / 128.0f, -66 - (int)(rnd() % 6)));
        }
        hB[(size_t)(k / 2) + (size_t)j * ldb] = f2bf(ldexpf(1.0f, -110));
    }
    const float bad = bad_kind == 0 ? 1e-30f : bad_kind == 1 ? INFINITY : NAN;
    char *colbad = calloc((size_t)n, 1), *rowbad = calloc((size_t)m, 1);
    for (int c = 0; c < ncols; ++c) {
        const int j = (int)(((long long)c * 7919) % n);
        colbad[j] = 1;
        hB[(size_t)(c % k) + (size_t)j * ldb] = f2bf(bad);
    }
    for (int r = 0; r < nrows; ++r) {
        const int i = (int)(((long long)r * 104729) % m);
        rowbad[i] = 1;
        hA[(size_t)((r * 3) % k) + (size_t)i * lda] = f2bf(bad);
    }
    /* With subnormal products, alpha 1 and beta 0 in fp32 output: the sum itself, which a beta * C of
     * ordinary size, or a rounding to bf16, would hide. */
    const float alpha = tiny ? 1.0f : 0.7f, beta = (bf16_out || tiny) ? 0.0f : 0.3f;
    const size_t es = bf16_out ? 2 : 4, cb = (size_t)m * n * es;
    void *hC0 = malloc(cb);
    if (bf16_out) {
        for (size_t i = 0; i < (size_t)m * n; ++i) ((uint16_t *)hC0)[i] = f2bf(NAN); /* not read: beta = 0 */
    } else {
        for (size_t i = 0; i < (size_t)m * n; ++i) ((float *)hC0)[i] = (float)gauss();
    }
    void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)n * k * 2), *dC = to_device(hC0, cb);
    CU(cudaDeviceSynchronize());
    const Counts c0 = counts(API_GEMMEX);
    const cublasStatus_t s = cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_16BF, lda, dB,
                                          CUDA_R_16BF, ldb, &beta, dC, bf16_out ? CUDA_R_16BF : CUDA_R_32F, m,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(st));
    const Counts c1 = counts(API_GEMMEX);
    void *hC = from_device(dC, cb);
    size_t fb = 0, fb_bad = 0, tc_bad = 0;
    double worst_tc = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            const size_t at = (size_t)i + (size_t)j * m;
            const float c_in = bf16_out ? 0.0f : ((float *)hC0)[at];
            if (colbad[j] || rowbad[i] || tinycol[j]) {
                ++fb;
                const float want = model(hA, lda, hB, ldb, k, i, j, alpha, beta, c_in);
                int same;
                if (bf16_out) {
                    same = ((uint16_t *)hC)[at] == f2bf(want);
                } else {
                    const float got = ((float *)hC)[at];
                    same = isnan(want) ? fbits(got) == 0x7fffffffu : fbits(got) == fbits(want);
                }
                if (!same) ++fb_bad;
            } else if (j % 7 == 0 && i % 5 == 0) { /* a sample of the tensor-core elements */
                double sum = 0, mag = 0;
                for (int l = 0; l < k; ++l) {
                    const double p = (double)bf2f(hA[l + (size_t)i * lda]) * (double)bf2f(hB[l + (size_t)j * ldb]);
                    sum += p;
                    mag += fabs(p);
                }
                sum = alpha * sum + (double)beta * c_in;
                mag = fabs(alpha) * mag + fabs((double)beta * c_in);
                const double got = bf16_out ? bf2f(((uint16_t *)hC)[at]) : ((float *)hC)[at];
                const double units = fabs(got - sum) / (mag > 0 ? mag * 0x1p-24 : 1.0);
                /* bf16 output: the rounding to 8 significant bits adds up to 2^-8 of the value */
                const double bound = bf16_out ? 32.0 + (fabs(sum) + 32.0 * mag * 0x1p-24) * 0x1p-8 / (mag * 0x1p-24) : 32.0;
                if (units > worst_tc) worst_tc = units;
                if (!(units <= bound)) ++tc_bad;
            }
        }
    static const char *const kinds[] = {"1e-30", "inf", "NaN"};
    printf("case TN %d x %d x %d, %s out%s: %d failing columns, %d failing rows (%s): %zu fallback elements, %zu differ "
           "from the model; tensor-core sample worst %.1f units, %zu beyond the bound; served %llu\n",
           m, n, k, bf16_out ? "bf16" : "fp32", tiny ? ", subnormal products" : "", ncols, nrows, kinds[bad_kind], fb,
           fb_bad, worst_tc, tc_bad,
           delta(&c0, &c1, SERVED_TC));
    CHECK(s == CUBLAS_STATUS_SUCCESS, "status %d", (int)s);
    CHECK(!rt_counters || delta(&c0, &c1, SERVED_TC) == 1, "not served by the tensor-core path");
    CHECK(fb_bad == 0, "%zu of %zu fallback elements differ from the host model", fb_bad, fb);
    CHECK(tc_bad == 0, "%zu tensor-core elements beyond the bound", tc_bad);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC0);
    free(hC);
    free(colbad);
    free(rowbad);
    free(tinyrow);
    free(tinycol);
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasCreate(&hv)) return 1;
    open_runtime();
    cublasSetStream(hv, st);
    static const int cols[] = {1, 5, 8, 9, 20, 32, 33, 100, 300};
    for (size_t c = 0; c < sizeof cols / sizeof *cols; ++c) one_case(1536, 320, 1536, cols[c], 0, 0, 0, 0);
    one_case(1536, 320, 1536, 9, 7, 0, 0, 0);
    one_case(1536, 320, 1536, 33, 33, 1, 0, 0);
    one_case(1536, 320, 1536, 5, 3, 2, 0, 0);
    one_case(1536, 320, 1536, 20, 9, 0, 1, 0);
    one_case(4608, 64, 1536, 8, 16, 0, 1, 0);
    one_case(896, 512, 4864, 32, 5, 0, 0, 0);
    one_case(1536, 320, 1536, 9, 0, 0, 0, 1);
    one_case(4608, 64, 1536, 5, 0, 0, 0, 1);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-fallback\n");
    return 0;
}
