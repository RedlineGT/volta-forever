/* Rule S1 for the tensor-core path (a default's condition): each element's class is cuBLAS 12.9's (NaN, +inf,
 * -inf, +0, -0, finite; the sign of a finite value is not a class), a NaN has 12.9's bits, and C is left
 * untouched when beta = 0: not read (a NaN in C does not reach D) and, for cublasLtMatmul with C != D, not
 * written. Calls the tensor-core path takes (TN 1536 x 128 x 1024, at its gate, both interfaces, fp32 and bf16 output),
 * with special values put in A, B and C: NaN of both signs with payloads, infinities (inf - inf and inf x 0
 * among them), rows of +0 and -0 with alpha and beta of both signs, sums that overflow, results below the
 * normal range, beta 0 and -0 over a NaN C (host and device scalars), beta 1 over infinite and NaN C,
 * device-mode alpha 0 with NaN and inf in A. 12.9's results come from the same arguments given to it
 * directly. Controls (tests/gpu/forward-bf16.sh): VF_BF16_CONTROL=beta0-reads-c and the kernel control
 * NAN_BITS must make it fail. Prints "ok bf16-s1". */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cublasLtHandle_t ltv, lt12;
static cudaStream_t st;
static const int M = 1536, N = 128, K = 1024;

static int cls(float v) { return isnan(v) ? 0 : isinf(v) ? (v > 0 ? 1 : 2) : v == 0 ? (signbit(v) ? 4 : 3) : 5; }
static const char *const cname[6] = {"NaN", "+inf", "-inf", "+0", "-0", "finite"};

typedef struct {
    float alpha, beta;
    int dev, ct; /* device scalars; output bf16 */
} Case;

static float out_at(const void *p, size_t i, int ct, uint32_t *bits)
{
    if (ct) {
        const uint16_t b = ((const uint16_t *)p)[i];
        *bits = b;
        return bf2f(b);
    }
    float f;
    memcpy(&f, (const char *)p + 4 * i, 4);
    memcpy(bits, &f, 4);
    return f;
}

/* Runs one case through the variant and through 12.9, legacy and cuBLASLt (C != D); counts class and NaN-bit
 * differences, C's changes, and calls the tensor-core path did not serve. */
static void run_case(const char *what, const uint16_t *hA, const uint16_t *hB, const void *hC, const Case *c)
{
    const size_t es = c->ct ? 2 : 4, cb = (size_t)M * N * es;
    const cudaDataType t = c->ct ? CUDA_R_16BF : CUDA_R_32F;
    void *A = to_device(hA, (size_t)M * K * 2), *B = to_device(hB, (size_t)N * K * 2);
    float *ab;
    CU(cudaMalloc((void **)&ab, 8));
    const float sc[2] = {c->alpha, c->beta};
    CU(cudaMemcpy(ab, sc, 8, cudaMemcpyHostToDevice));
    const void *pa = c->dev ? (const void *)ab : (const void *)&sc[0], *pb = c->dev ? (const void *)(ab + 1) : (const void *)&sc[1];
    for (int api = 0; api < 2; ++api) {
        void *Dv = to_device(hC, cb), *D12 = to_device(hC, cb), *Cv = to_device(hC, cb), *C12 = to_device(hC, cb);
        const Counts k0 = counts(api ? API_LT : API_GEMMEX);
        cublasStatus_t s1, s2;
        if (!api) {
            const cublasPointerMode_t pm = c->dev ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST;
            cublasSetPointerMode(hv, pm);
            c129.SetPointerMode(h12, pm);
            s1 = cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, pa, A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, pb, Dv, t, M,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            CU(cudaStreamSynchronize(st));
            s2 = c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, pa, A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, pb, D12, t, M,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            CU(cudaStreamSynchronize(st));
            cublasSetPointerMode(hv, CUBLAS_POINTER_MODE_HOST);
            c129.SetPointerMode(h12, CUBLAS_POINTER_MODE_HOST);
        } else {
            cublasLtMatmulDesc_t d;
            cublasLtMatrixLayout_t la, lb, lc;
            cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            const int32_t oa = CUBLAS_OP_T, pm = c->dev ? CUBLASLT_POINTER_MODE_DEVICE : CUBLASLT_POINTER_MODE_HOST;
            cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
            cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof pm);
            cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, M, K);
            cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, N, K);
            cublasLtMatrixLayoutCreate(&lc, t, M, N, M);
            s1 = cublasLtMatmul(ltv, d, pa, A, la, B, lb, pb, Cv, lc, Dv, lc, NULL, NULL, 0, st);
            CU(cudaStreamSynchronize(st));
            s2 = c129.LtMatmul(lt12, d, pa, A, la, B, lb, pb, C12, lc, D12, lc, NULL, NULL, 0, st);
            CU(cudaStreamSynchronize(st));
            cublasLtMatrixLayoutDestroy(la);
            cublasLtMatrixLayoutDestroy(lb);
            cublasLtMatrixLayoutDestroy(lc);
            cublasLtMatmulDescDestroy(d);
        }
        const Counts k1 = counts(api ? API_LT : API_GEMMEX);
        void *ov = from_device(Dv, cb), *o12 = from_device(D12, cb), *cv = from_device(Cv, cb);
        size_t ncls = 0, nnan = 0, first = (size_t)-1;
        int ca = 0, cb12 = 0;
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            uint32_t bv, b12;
            const float v = out_at(ov, i, c->ct, &bv), w = out_at(o12, i, c->ct, &b12);
            if (cls(v) != cls(w)) {
                if (!ncls) first = i, ca = cls(v), cb12 = cls(w);
                ++ncls;
            } else if (cls(v) == 0 && bv != b12) {
                ++nnan;
            }
        }
        const int c_changed = api && memcmp(cv, hC, cb) != 0;
        const int served = !rt_counters || delta(&k0, &k1, SERVED_TC) == 1;
        printf("%-44s %-8s %s: status %d / %d, class differences %zu", what, api ? "cuBLASLt" : "legacy", c->ct ? "bf16" : "fp32",
               (int)s1, (int)s2, ncls);
        if (ncls) printf(" (first: element %zu, %s against 12.9's %s)", first, cname[ca], cname[cb12]);
        printf(", NaN bits differing %zu%s%s\n", nnan, c_changed ? ", C WRITTEN" : "", served ? "" : ", NOT the tensor-core path's");
        CHECK(s1 == s2 && !ncls && !nnan && !c_changed && served, "%s, %s, %s: not 12.9's classes and NaN bits, C changed, or "
              "not served by the tensor-core path", what, api ? "cuBLASLt" : "legacy", c->ct ? "bf16" : "fp32");
        free(ov);
        free(o12);
        free(cv);
        cudaFree(Dv);
        cudaFree(D12);
        cudaFree(Cv);
        cudaFree(C12);
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(ab);
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasCreate(&hv) || c129.Create(&h12) || cublasLtCreate(&ltv) || c129.LtCreate(&lt12)) return 1;
    open_runtime();
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    uint16_t *A0 = bf16_random((size_t)M * K, 0.02), *B0 = bf16_random((size_t)N * K, 1.0);
    uint16_t *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    float *C32 = malloc((size_t)M * N * 4);
    uint16_t *C16 = malloc((size_t)M * N * 2);
#define AT(i, l) A[(size_t)(i) * K + (l)]
#define BT(l, j) B[(size_t)(j) * K + (l)]
#define SETC(val)                                                   \
    do {                                                            \
        for (size_t e_ = 0; e_ < (size_t)M * N; ++e_) C32[e_] = (val), C16[e_] = f2bf(C32[e_]); \
    } while (0)
    for (int ct = 0; ct < 2; ++ct) {
        Case c = {1.0f, 0.0f, 0, ct};
        const void *C = ct ? (const void *)C16 : (const void *)C32;
        /* 1. NaN and infinities in A and B, beta 0 */
        memcpy(A, A0, (size_t)M * K * 2);
        memcpy(B, B0, (size_t)N * K * 2);
        AT(5, 7) = 0x7fc0;   /* quiet NaN */
        AT(9, 100) = 0x7fa1; /* signalling NaN with a payload */
        AT(11, 3) = 0xffc1;  /* negative NaN with a payload */
        BT(50, 3) = 0x7fc0;  /* NaN in a column of B */
        AT(20, 10) = 0x7f80; /* +inf against B[10] = 0 (NaN), > 0 and < 0 */
        BT(10, 0) = 0x0000, BT(10, 1) = 0x3f80, BT(10, 2) = 0xbf80;
        AT(21, 10) = 0x7f80, AT(21, 20) = 0xff80; /* +inf and -inf in one row: NaN where both products count */
        BT(20, 1) = 0x3f80, BT(20, 2) = 0x3f80;
        SETC(0.0f);
        run_case("NaN and infinities in A and B, beta 0", A, B, C, &c);
        /* 2. sums that overflow: row 30 = 1.5 x 2^100 at l = 0, 1, 2 */
        memcpy(A, A0, (size_t)M * K * 2);
        memcpy(B, B0, (size_t)N * K * 2);
        for (int l = 0; l < K; ++l) AT(30, l) = 0;
        AT(30, 0) = f2bf(0x1.8p100f), AT(30, 1) = f2bf(0x1.8p100f), AT(30, 2) = f2bf(0x1.8p100f);
        for (int j = 0; j < 4; ++j) BT(0, j) = BT(1, j) = BT(2, j) = 0;
        BT(0, 0) = f2bf(0x1p27f), BT(1, 0) = f2bf(0x1p27f), BT(2, 0) = f2bf(-0x1p27f);
        BT(0, 1) = f2bf(0x1p27f), BT(1, 1) = f2bf(0x1p27f), BT(2, 1) = f2bf(0x1p27f);
        BT(0, 2) = f2bf(0x1.4p26f), BT(1, 2) = f2bf(0x1.4p26f), BT(2, 2) = f2bf(0x1p25f);
        run_case("sums that overflow", A, B, C, &c);
        /* 3. zeros: rows of +0 and -0, alpha and beta of both signs */
        memcpy(A, A0, (size_t)M * K * 2);
        memcpy(B, B0, (size_t)N * K * 2);
        for (int l = 0; l < K; ++l) AT(40, l) = 0x0000, AT(41, l) = 0x8000;
        static const struct {
            float alpha, beta, cv;
            int dev;
            const char *what;
        } z[] = {{1, 0, 0, 0, "zeros, alpha 1, beta 0"},           {-1, 0, 0, 0, "zeros, alpha -1, beta 0"},
                 {-1, 1, -0.0f, 0, "zeros, alpha -1, beta 1, C = -0"}, {1, 1, -0.0f, 0, "zeros, alpha 1, beta 1, C = -0"},
                 {-1, 0.5f, -0.0f, 1, "zeros, alpha -1, beta 0.5, C = -0, device"},
                 {-1, -1, 0, 0, "zeros, alpha -1, beta -1, C = +0"}};
        for (size_t i = 0; i < sizeof z / sizeof *z; ++i) {
            SETC(z[i].cv);
            Case cz = {z[i].alpha, z[i].beta, z[i].dev, ct};
            run_case(z[i].what, A, B, C, &cz);
        }
        /* 4. beta 0 and -0 over a NaN C (host and device): C is not read */
        static const struct {
            float beta;
            int dev;
            const char *what;
        } nc[] = {{0.0f, 0, "beta 0 over a NaN C"}, {-0.0f, 0, "beta -0 over a NaN C"}, {0.0f, 1, "beta 0 over a NaN C, device"}};
        for (size_t i = 0; i < sizeof nc / sizeof *nc; ++i) {
            SETC(NAN);
            Case cn = {1.0f, nc[i].beta, nc[i].dev, ct};
            run_case(nc[i].what, A0, B0, C, &cn);
        }
        /* 5. beta 1 over infinite and NaN C */
        for (size_t e = 0; e < (size_t)M * N; ++e) {
            C32[e] = e % 7 == 0 ? INFINITY : e % 7 == 1 ? -INFINITY : e % 7 == 2 ? NAN : (float)gauss();
            C16[e] = f2bf(C32[e]);
        }
        Case c1 = {1.0f, 1.0f, 0, ct};
        run_case("beta 1 over infinite and NaN C", A0, B0, C, &c1);
        /* 6. device-mode alpha 0 with NaN and inf in A (beta 1, C finite) */
        memcpy(A, A0, (size_t)M * K * 2);
        AT(7, 9) = 0x7fc0, AT(13, 2) = 0x7f80;
        for (size_t e = 0; e < (size_t)M * N; ++e) C32[e] = (float)gauss(), C16[e] = f2bf(C32[e]);
        Case c0 = {0.0f, 1.0f, 1, ct};
        run_case("device alpha 0, NaN and inf in A", A, B0, C, &c0);
        /* 7. results below the normal range: A scaled by 2^-120 */
        for (size_t e = 0; e < (size_t)M * K; ++e) A[e] = f2bf(bf2f(A0[e]) * 0x1p-120f);
        SETC(0.0f);
        run_case("results below the normal range", A, B0, C, &c);
    }
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-s1\n");
    return 0;
}
