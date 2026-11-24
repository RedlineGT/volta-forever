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
 * gives other bits and makes the test fail.
 * Then failing rows and columns spread over the chunks of a call (the fallback runs once, after the last
 * chunk, over all of them): TN 1536 x n x 8192, whose workspace takes 2 chunks of rows at n = 64, 6 at
 * n = 512 and 3 x 7 at n = 2048 (k = 8192, the wide rule's bound: with k > 8192 the path leaves n > 256 to
 * 12.9), with 1 failing column, 2 % of them, every column, and failing rows too;
 * fp32 output with alpha 0.7, beta 0.3 and C in place, and bf16 output. Every fallback element must have
 * the host model's bits (a sample of 4096 of them when there are more) and cuBLAS 12.9's bits from the
 * same call (all of them).
 * Every call also adds to the exit report's counters (a test build's vf_bf16_test_fallback_counters): one call
 * with failed rows or columns, one with more than a quarter of its rows or of its columns failed when it has, and
 * its fallback elements, exactly. A fat binary whose fallback adds nothing (VF_BF16_FATBIN=control-NO_STATS.fatbin)
 * makes the test fail there. Prints "ok bf16-fallback". */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cudaStream_t st;

/* The exit report's counters of the fallback (test builds only; NULL with a release runtime): [0] calls with
 * failed rows or columns, [1] those with more than a quarter of their rows or of their columns failed, [2] the
 * fallback's elements, [3] the elements of the calls the tensor-core path served. */
static int (*rt_fb_counters)(unsigned long long *, int);
typedef struct {
    unsigned long long v[4];
} FbCounts;
static FbCounts fb_counts(void)
{
    FbCounts c;
    memset(&c, 0, sizeof c);
    if (rt_fb_counters && rt_fb_counters(c.v, 4) != 4) {
        printf("FAIL  the fallback's counters could not be read\n");
        exit(1);
    }
    return c;
}
/* The counters' change over one call with ca failed rows and cb failed columns (m x n) and fb fallback elements. */
static void check_fb_counts(const char *what, const FbCounts *a, const FbCounts *b, int m, int n, int ca, int cb, size_t fb)
{
    if (!rt_fb_counters) return;
    const unsigned long long calls = b->v[0] - a->v[0], quarter = b->v[1] - a->v[1], elems = b->v[2] - a->v[2],
                             all = b->v[3] - a->v[3];
    const unsigned long long want_q = 4LL * ca > m || 4LL * cb > n;
    CHECK(calls == (ca > 0 || cb > 0) && quarter == want_q && elems == fb && all == (unsigned long long)m * n,
          "%s: the exit report's counters added %llu call, %llu over a quarter, %llu fallback elements of %llu (not %d, "
          "%llu, %zu of %llu)", what, calls, quarter, elems, all, ca > 0 || cb > 0, want_q, fb, (unsigned long long)m * n);
}

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
    /* tiny: rows of op(A) of both signs near 2^-70..2^-75 (whose largest, below 2^-24, makes each row fail the
     * check) and columns of op(B) near 2^-66..2^-71 (one element of 2^-110 makes each column fail it), as in
     * bf16-arith.c: their products, below
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
    const FbCounts f0 = fb_counts();
    const cublasStatus_t s = cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_16BF, lda, dB,
                                          CUDA_R_16BF, ldb, &beta, dC, bf16_out ? CUDA_R_16BF : CUDA_R_32F, m,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(st));
    const Counts c1 = counts(API_GEMMEX);
    const FbCounts f1 = fb_counts();
    void *hC = from_device(dC, cb);
    size_t fb = 0, fb_bad = 0, tc_bad = 0;
    double worst_tc = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            const size_t at = (size_t)i + (size_t)j * m;
            const float c_in = bf16_out ? 0.0f : ((float *)hC0)[at];
            if (colbad[j] || rowbad[i] || tinycol[j] || tinyrow[i]) { /* tiny rows fail the check too (largest < 2^-24) */
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
    int fr = 0, fc = 0;
    for (int i = 0; i < m; ++i) fr += rowbad[i] || tinyrow[i];
    for (int j = 0; j < n; ++j) fc += colbad[j] || tinycol[j];
    check_fb_counts("case", &f0, &f1, m, n, fr, fc, fb);
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

/* Failing rows and columns over several chunks: TN 1536 x n x 8192. cols: the number of failing columns
 * (spread evenly), -1 for every column. */
static void chunk_case(int n, int cols, int nrows, int bf16_out)
{
    const int m = 1536, k = 8192, lda = k, ldb = k;
    uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)n * k, 1.0);
    const int ncols = cols < 0 ? n : cols;
    char *colbad = calloc((size_t)n, 1), *rowbad = calloc((size_t)m, 1);
    for (int c = 0; c < ncols; ++c) {
        const int j = (int)(((long long)c * n) / ncols); /* spread over the columns, so over their chunks */
        colbad[j] = 1;
        hB[(size_t)((c * 131) % k) + (size_t)j * ldb] = f2bf(1e-30f);
    }
    for (int r = 0; r < nrows; ++r) { /* one in each chunk of rows, at least */
        const int i = (int)(((long long)r * m) / nrows) + 3;
        rowbad[i] = 1;
        hA[(size_t)((r * 977) % k) + (size_t)i * lda] = f2bf(1e-30f);
    }
    const float alpha = bf16_out ? 1.0f : 0.7f, beta = bf16_out ? 0.0f : 0.3f;
    const size_t es = bf16_out ? 2 : 4, cb = (size_t)m * n * es;
    void *hC0 = malloc(cb);
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        if (bf16_out)
            ((uint16_t *)hC0)[i] = f2bf(NAN); /* not read: beta = 0 */
        else
            ((float *)hC0)[i] = (float)gauss();
    }
    void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)n * k * 2), *dC = to_device(hC0, cb),
         *dC12 = to_device(hC0, cb);
    const cudaDataType ct = bf16_out ? CUDA_R_16BF : CUDA_R_32F;
    CU(cudaDeviceSynchronize());
    const Counts c0 = counts(API_GEMMEX);
    const FbCounts f0 = fb_counts();
    const cublasStatus_t s = cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_16BF, lda, dB,
                                          CUDA_R_16BF, ldb, &beta, dC, ct, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(st));
    const Counts c1 = counts(API_GEMMEX);
    const FbCounts f1 = fb_counts();
    const cublasStatus_t s12 = c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_16BF, lda, dB,
                                           CUDA_R_16BF, ldb, &beta, dC12, ct, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(st));
    void *hC = from_device(dC, cb), *hC12 = from_device(dC12, cb);
    size_t fb = 0, vs129 = 0, sampled = 0, vsmodel = 0;
    for (int j = 0; j < n; ++j) /* the fallback elements, counted first, then sampled for the host model */
        for (int i = 0; i < m; ++i)
            fb += colbad[j] || rowbad[i];
    const size_t step = fb > 4096 ? fb / 4096 : 1;
    size_t seen = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            if (!(colbad[j] || rowbad[i])) continue;
            const size_t at = (size_t)i + (size_t)j * m;
            const int same12 = bf16_out ? ((uint16_t *)hC)[at] == ((uint16_t *)hC12)[at]
                                        : fbits(((float *)hC)[at]) == fbits(((float *)hC12)[at]);
            vs129 += !same12;
            if (seen++ % step == 0) {
                ++sampled;
                const float want = model(hA, lda, hB, ldb, k, i, j, alpha, beta, bf16_out ? 0.0f : ((float *)hC0)[at]);
                const int same = bf16_out ? ((uint16_t *)hC)[at] == f2bf(want) : fbits(((float *)hC)[at]) == fbits(want);
                vsmodel += !same;
            }
        }
    printf("chunks TN %d x %d x %d, %s out: %d failing columns, %d failing rows: %zu fallback elements, %zu differ from "
           "cuBLAS 12.9's; of %zu sampled, %zu differ from the model; served %llu\n",
           m, n, k, bf16_out ? "bf16" : "fp32", ncols, nrows, fb, vs129, sampled, vsmodel, delta(&c0, &c1, SERVED_TC));
    CHECK(s == CUBLAS_STATUS_SUCCESS && s12 == CUBLAS_STATUS_SUCCESS, "status %d, 12.9 %d", (int)s, (int)s12);
    CHECK(!rt_counters || delta(&c0, &c1, SERVED_TC) == 1, "not served by the tensor-core path");
    CHECK(vsmodel == 0, "%zu of %zu sampled fallback elements differ from the host model", vsmodel, sampled);
    CHECK(vs129 == 0, "%zu of %zu fallback elements differ from cuBLAS 12.9's", vs129, fb);
    int fr = 0, fc = 0;
    for (int i = 0; i < m; ++i) fr += rowbad[i];
    for (int j = 0; j < n; ++j) fc += colbad[j];
    check_fb_counts("chunks", &f0, &f1, m, n, fr, fc, fb);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC12);
    free(hA);
    free(hB);
    free(hC0);
    free(hC);
    free(hC12);
    free(colbad);
    free(rowbad);
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasCreate(&hv) || c129.Create(&h12)) return 1;
    open_runtime();
    {
        void *rt = dlopen("libvf_cublas_bf16.so", RTLD_NOW | RTLD_NOLOAD);
        if (rt) *(void **)&rt_fb_counters = dlsym(rt, "vf_bf16_test_fallback_counters");
    }
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    static const int cols[] = {1, 5, 8, 9, 20, 32, 33, 100, 300};
    for (size_t c = 0; c < sizeof cols / sizeof *cols; ++c) one_case(1536, 320, 2048, cols[c], 0, 0, 0, 0);
    one_case(1536, 320, 2048, 9, 7, 0, 0, 0);
    one_case(1536, 320, 2048, 33, 33, 1, 0, 0);
    one_case(1536, 320, 2048, 5, 3, 2, 0, 0);
    one_case(1536, 320, 2048, 20, 9, 0, 1, 0);
    one_case(4608, 64, 2048, 8, 16, 0, 1, 0);
    one_case(896, 512, 4864, 32, 5, 0, 0, 0);
    one_case(1536, 320, 2048, 9, 0, 0, 0, 1);
    one_case(4608, 64, 2048, 5, 0, 0, 0, 1);
    static const int ns[] = {64, 512, 2048};
    for (int t = 0; t < 3; ++t) {
        chunk_case(ns[t], 1, 0, 0);
        chunk_case(ns[t], ns[t] / 50 > 1 ? ns[t] / 50 : 2, 0, 0); /* 2 % */
        chunk_case(ns[t], -1, 0, 0);
    }
    chunk_case(2048, 41, 7, 0); /* failing rows in every chunk of rows, with columns in every chunk of columns */
    chunk_case(2048, 41, 7, 1);
    chunk_case(512, -1, 7, 1);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-fallback\n");
    return 0;
}
