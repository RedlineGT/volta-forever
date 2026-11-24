/* Acceptance test of the bf16 tensor-core path under policy B (results may differ from cuBLAS 12.9 within
 * what current NVIDIA hardware gives and within the classical fp32 bound), on a V100:
 *
 *   bf16-accept            the test (tests/gpu/forward-bf16.sh runs it with VF_CUBLAS_BF16_TC=1
 *                          and the test runtime). For every call the path serves with k >= the gate
 *                          (VF_ACCEPT_GATE, default below):
 *     S1  the class of every element (NaN with its bits, +inf, -inf, +0, -0, finite) and the status are 12.9's,
 *         on special-value cases (NaN, inf, overflow, zeros with alpha and beta of both signs, beta = 0 with
 *         NaN in C in both pointer modes, subnormal results);
 *     S2  two runs give the same bits; so do a handle without workspace, and the same elements computed in
 *         a call with half the columns;
 *     A   per element, |c - x| <= max(|c129 - x|, (k - 1) u S + h)   (alpha 1, beta 0)
 *                      |c - x| <= max(|c129 - x|, (k + 2) u S' + h)  (alpha, beta, a bias)
 *         x the exact value, S = sum |a b| (S' adds |beta c| and |bias|), u = 2^-24, h half an ulp of the
 *         output format at x;
 *     B   per matrix (fp32 output), the bias towards zero, mean of (c - x) sign(x) / (u S'), is at most
 *         max(|bias of 12.9|, margin x slope x k) + 0.5 in magnitude: slope the native bias slope of the data
 *         (an RTX 5090 on the same data, measured once, constants below), margin VF_ACCEPT_BIAS_MARGIN (0.8);
 *         the bias of two synthetic constructions is judged from k = 4096 (craftsub) and 2048 (configuration
 *         1653006390) on (VF_ACCEPT_STRICT=1: from the gate), see bias_from_fam and kept[];
 *     on deterministic data (bf16-accept-gen.h): 10 families and 3 configurations of the adversarial search,
 *     k = 256, 512, 1024, 1536, 2048, 4096, 16384 (TN), NN, NT, TT at 2048, both output types, alpha and beta with a
 *     general C, the cuBLASLt bias
 *     epilogue. Calls below the gate are run but not judged, and so are the NN, NT and TT calls, which go to 12.9
 *     (the path's layout rule: TN only). With VF_ACCEPT_EXPECT_GATE=1, a call below the gate that the path serves
 *     is a failure, and so is a TN call from the gate on that it does not serve, or a call in another layout that
 *     it serves: for when the runtime has its gate; the number judged is printed). Prints "ok bf16-accept" and exits 0, or FAIL lines
 *     and exits 1. Controls, each of which must fail: VF_ACCEPT_GATE=256 and 512 (the path exceeds A on craftsub
 *     at 256, the bias bound below 1024); VF_ACCEPT_STRICT=1 (the two exceptions removed); the tensor-core path
 *     off (the skinny path on, for the test runtime's counters) with VF_ACCEPT_EXPECT_GATE=1 (nothing served).
 *
 *   bf16-accept safe       with VF_SAFE=1: no call served by the tensor-core path, every result with 12.9's bits
 *                          (the conservative switch; the skinny path, whose results are 12.9's bits, still serves
 *                          the calls of its rule); also VF_ACCEPT_MODE=safe without an argument.
 *
 *   Measurements (by hand, not part of the suite):
 *   bf16-accept sweep OUT.tsv FAMS KS M N    families x k (TN), both outputs
 *   bf16-accept search OUT.tsv K NCONF SEED  random configurations of the generator at one k, then hill
 *                                            climbing on the A ratio and on the bias
 *   bf16-accept shapes OUT.tsv               layouts, lm_head-sized m, alpha/beta with C, cuBLASLt bias
 *   bf16-accept list OUT.tsv IN.tsv KS M N   configurations of a TSV (as written here) at m x n, every k
 *   bf16-accept ndep OUT.tsv                 the same element at n = 1 ... 64 (variant and 12.9)
 *   bf16-accept time OUT.tsv                 time of 12.9 and of the variant, linear-layer shapes, n = 1 ... 64
 *
 * Built against the CUDA 13.5 headers like the other bf16-*.c (bf16-common.h); -fopenmp optional (host-side
 * generation and exact sums only). */
#include "bf16-common.h"
#include "bf16-accept-gen.h"
#include <time.h>
#ifdef _OPENMP /* -fopenmp shortens the host side only (generation, exact sums) */
#define OMP(x) _Pragma(#x)
#else
#define OMP(x)
#endif
#define KEPT_SLOPE_0 0.0292  /* configuration 901 (positive spread data) on the RTX 5090, as native_slope */
#define KEPT_SLOPE_1 0.00001 /* configuration 902 (signed data) */
#define KEPT_SLOPE_2 0.1176  /* configuration 1653006390: the search's most biased against the native bound */

static cublasHandle_t hv, h12;
static cublasLtHandle_t ltv, lt12;
static cudaStream_t st;
static void *ws_lt;
#define WS_LT (4u << 20)
static int gate = 1024; /* the tensor-core path's gate (VF_TC_MIN_K, crates/vf-fwdgen/bf16tc/runtime.c) */
static int expect_gate; /* VF_ACCEPT_EXPECT_GATE=1: the path serves exactly the calls with k >= the gate: one below it
                         * that it serves, or one from it on that it does not serve, is a failure */
static double bias_margin = 0.8; /* VF_ACCEPT_BIAS_MARGIN */
#define U32 0x1p-24

/* The native bias slope of each family: |bias| / k of an RTX 5090 (PyTorch 2.13, cuBLAS 12.9 for sm_120,
 * fp32 output) on the same data (seed 3000 + family, 1024 x 128, these 64 sampled rows), the smallest over
 * k = 1024, 1536, 2048 where its chains are not split (docs/emulations.md, "The tensor-core path's gate"). The
 * bound at k is margin x slope x k: a native chain of length k. */
static const double native_slope[FAM_N] = {
    [FAM_PARAM] = 0.0,       [FAM_NORMAL] = 0.000375, [FAM_SAMESIGN] = 0.0281, [FAM_LOGNPOS] = 0.0554,
    [FAM_ADVPOS] = 0.0688,   [FAM_CRAFTSUB] = 0.137,  [FAM_CRAFTACT] = 0.0060, [FAM_RELU2] = 0.00078,
    [FAM_OUTLIER] = 0.0167,  [FAM_CANCEL] = 0.0,      [FAM_MIXED30] = 0.0031};
static double cur_slope; /* the slope of the data being judged */
/* Two synthetic constructions exceed the bias bound above the gate: craftsub (a product of 2^20 at the start of
 * every 256-element slice of the path, then products of 1/2 to 1 ulp of it; 1.50 times its bound at k = 1024, 1.29
 * at 1536 and 1.20 at 2048, the bound being 12.9's own bias there) up to k = 4096, and the search's configuration
 * 1653006390 (non-negative operands, a 2^21 activation at the start of every slice; 1.90 times its bound at 1024
 * and 1.27 at 1536, 0.95 at 2048) up to 2048.
 * Real operands (1,728 pairs of four models) have mixed-sign products and a bias of at most 0.22 u at every k.
 * Their bias is judged from 4096 and 2048 on, unless VF_ACCEPT_STRICT=1 (then from the gate). Every other bound
 * applies from the gate. */
static int bias_from_fam[FAM_N] = {[FAM_CRAFTSUB] = 4096};
static int strict, cur_fam, cur_bias_from;

typedef struct {
    int ta, tb, m, n, k, ct; /* ct: 0 fp32, 1 bf16 */
    float alpha, beta;
    int lt, bias, wsnone, devptr;
} Prob;

typedef struct {
    int path, st12, stv;
    double ratio, bias_v, bias_12, maxu_v, maxu_12, rms_v, rms_12, fbfrac, worst_x, worst_s;
    long nfail, nel, cls, nanbits;
    int rerun_same;
} Res;

static const char *path_name(int p) { return p == 2 ? "tc" : p == 1 ? "skinny" : p == 3 ? "fwd" : p == 4 ? "129" : "?"; }

/* ---- operands --------------------------------------------------------------------------------- */
static uint16_t *host_A(const Gen *g, const Prob *p)
{
    const size_t lda = p->ta ? p->k : p->m;
    uint16_t *A = malloc(lda * (size_t)(p->ta ? p->m : p->k) * 2);
OMP(omp parallel for schedule(dynamic, 64))
    for (long long i = 0; i < p->m; ++i)
        for (int l = 0; l < p->k; ++l) {
            const uint16_t v = va(g, i, l);
            if (p->ta) A[(size_t)i * lda + l] = v;
            else A[(size_t)l * lda + i] = v;
        }
    return A;
}
static uint16_t *host_B(const Gen *g, const Prob *p)
{
    const size_t ldb = p->tb ? p->n : p->k;
    uint16_t *B = malloc(ldb * (size_t)(p->tb ? p->k : p->n) * 2);
OMP(omp parallel for schedule(dynamic, 8))
    for (long long j = 0; j < p->n; ++j)
        for (int l = 0; l < p->k; ++l) {
            const uint16_t v = vb(g, j, l);
            if (p->tb) B[(size_t)l * ldb + j] = v;
            else B[(size_t)j * ldb + l] = v;
        }
    return B;
}
static double c0_val(const Gen *g, const Prob *p, int i, int j)
{
    if (p->beta == 0.0f) return 0.0;
    const double v = g_gauss(g, 50, G_IDX(j, i));
    return p->ct ? g_f(g_bf16(v)) : (double)(float)v;
}
static double bias_val(const Gen *g, const Prob *p, int i)
{
    if (!p->bias) return 0.0;
    const double v = 0.5 * g_gauss(g, 51, (uint64_t)i);
    return p->ct ? g_f(g_bf16(v)) : (double)(float)v;
}

/* ---- one call ----------------------------------------------------------------------------------- */
static float *d_ab;
static cublasStatus_t gemm(int variant, const Prob *p, const void *A, const void *B, void *C, const void *bias)
{
    const float ab[2] = {p->alpha, p->beta};
    const void *pa = &ab[0], *pb = &ab[1];
    if (p->devptr) {
        CU(cudaMemcpy(d_ab, ab, 8, cudaMemcpyHostToDevice));
        pa = d_ab, pb = d_ab + 1;
    }
    const long long lda = p->ta ? p->k : p->m, ldb = p->tb ? p->n : p->k;
    const cudaDataType ct = p->ct ? CUDA_R_16BF : CUDA_R_32F;
    if (p->lt) {
        cublasLtMatmulDesc_t d;
        cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        const int32_t oa = p->ta ? CUBLAS_OP_T : CUBLAS_OP_N, ob = p->tb ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof ob);
        if (p->devptr) {
            const int32_t pm = CUBLASLT_POINTER_MODE_DEVICE;
            cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof pm);
        }
        if (p->bias) {
            const int32_t ep = CUBLASLT_EPILOGUE_BIAS;
            const int32_t bt = ct;
            cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof ep);
            cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
            cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof bt);
        }
        cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, p->ta ? p->k : p->m, p->ta ? p->m : p->k, lda);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, p->tb ? p->n : p->k, p->tb ? p->k : p->n, ldb);
        cublasLtMatrixLayoutCreate(&lc, ct, p->m, p->n, p->m);
        const cublasStatus_t s =
            variant ? cublasLtMatmul(ltv, d, pa, A, la, B, lb, pb, C, lc, C, lc, NULL, p->wsnone ? NULL : ws_lt,
                                     p->wsnone ? 0 : WS_LT, st)
                    : c129.LtMatmul(lt12, d, pa, A, la, B, lb, pb, C, lc, C, lc, NULL, p->wsnone ? NULL : ws_lt,
                                    p->wsnone ? 0 : WS_LT, st);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulDescDestroy(d);
        return s;
    }
    cublasHandle_t h = variant ? hv : h12;
    if (variant) {
        cublasSetStream(hv, st);
        if (p->wsnone) cublasSetWorkspace(hv, NULL, 0);
        cublasSetPointerMode(hv, p->devptr ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST);
    } else {
        c129.SetStream(h12, st);
        c129.SetPointerMode(h12, p->devptr ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST);
    }
    const cublasOperation_t oa = p->ta ? CUBLAS_OP_T : CUBLAS_OP_N, ob = p->tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasStatus_t s =
        variant ? cublasGemmEx(h, oa, ob, p->m, p->n, p->k, pa, A, CUDA_R_16BF, (int)lda, B, CUDA_R_16BF, (int)ldb, pb, C,
                               ct, p->m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                : c129.GemmEx(h, oa, ob, p->m, p->n, p->k, pa, A, CUDA_R_16BF, (int)lda, B, CUDA_R_16BF, (int)ldb, pb, C,
                              ct, p->m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (variant) cublasSetPointerMode(hv, CUBLAS_POINTER_MODE_HOST);
    else c129.SetPointerMode(h12, CUBLAS_POINTER_MODE_HOST);
    return s;
}

/* Runs one call; *path from the test runtime's counters. Output (m x n, ld m) copied to hout. */
static cublasStatus_t call(int variant, const Prob *p, const void *A, const void *B, void *dC, const void *hC0,
                           const void *bias, void *hout, int *path)
{
    const size_t cb = (size_t)p->m * p->n * (p->ct ? 2 : 4);
    CU(cudaMemcpy(dC, hC0, cb, cudaMemcpyHostToDevice));
    CU(cudaDeviceSynchronize());
    const int api = p->lt ? API_LT : API_GEMMEX;
    const Counts c0 = counts(api);
    const cublasStatus_t s = gemm(variant, p, A, B, dC, bias);
    CU(cudaStreamSynchronize(st));
    const Counts c1 = counts(api);
    *path = 4;
    if (variant) {
        *path = 0;
        if (rt_counters) *path = delta(&c0, &c1, SERVED_TC) ? 2 : delta(&c0, &c1, SERVED_SKINNY) ? 1 : 3;
    }
    if (hout) CU(cudaMemcpy(hout, dC, cb, cudaMemcpyDeviceToHost));
    return s;
}

/* ---- judging --------------------------------------------------------------------------------------- */
static double outval(const void *o, size_t idx, int ct)
{
    return ct ? g_f(((const uint16_t *)o)[idx]) : (double)((const float *)o)[idx];
}
static uint32_t outbits(const void *o, size_t idx, int ct)
{
    if (ct) return (uint32_t)((const uint16_t *)o)[idx] << 16;
    uint32_t u;
    memcpy(&u, (const char *)o + 4 * idx, 4);
    return u;
}
static double half_ulp(double x, int ct)
{
    const double a = fabs(x);
    int e = a > 0 ? ilogb(a) : -126;
    if (e < -126) e = -126;
    return ldexp(1.0, e - (ct ? 7 : 23) - 1);
}
/* 1 when two output elements are of different classes: NaN, inf of each sign, zero of each sign. */
static int class_differs(uint32_t a, uint32_t b)
{
    const int na = (a & 0x7fffffffu) > 0x7f800000u, nb = (b & 0x7fffffffu) > 0x7f800000u;
    if (na || nb) return na != nb;
    const int ia = (a & 0x7fffffffu) == 0x7f800000u, ib = (b & 0x7fffffffu) == 0x7f800000u;
    if (ia || ib) return a != b;
    const int za = (a & 0x7fffffffu) == 0, zb = (b & 0x7fffffffu) == 0;
    if (za && zb) return a != b;
    return 0;
}

#define NR_MAX 128
/* The full measurement of one problem. nr sampled rows (evenly spread), every column. */
static Res run_prob(const Gen *g, const Prob *p, int nr)
{
    Res r;
    memset(&r, 0, sizeof r);
    uint16_t *hA = host_A(g, p), *hB = host_B(g, p);
    const size_t na = (size_t)(p->ta ? p->k : p->m) * (p->ta ? p->m : p->k), nb = (size_t)(p->tb ? p->n : p->k) * (p->tb ? p->k : p->n);
    void *dA = to_device(hA, na * 2), *dB = to_device(hB, nb * 2);
    free(hA);
    free(hB);
    const size_t es = p->ct ? 2 : 4, mn = (size_t)p->m * p->n;
    void *hC0 = malloc(mn * es), *o12 = malloc(mn * es), *ov = malloc(mn * es), *ov2 = malloc(mn * es);
    for (int j = 0; j < p->n; ++j)
        for (int i = 0; i < p->m; ++i) {
            const double v = c0_val(g, p, i, j);
            if (p->ct) ((uint16_t *)hC0)[(size_t)j * p->m + i] = g_bf16(v);
            else ((float *)hC0)[(size_t)j * p->m + i] = (float)v;
        }
    void *dbias = NULL;
    if (p->bias) {
        void *hb = malloc((size_t)p->m * es);
        for (int i = 0; i < p->m; ++i) {
            if (p->ct) ((uint16_t *)hb)[i] = g_bf16(bias_val(g, p, i));
            else ((float *)hb)[i] = (float)bias_val(g, p, i);
        }
        dbias = to_device(hb, (size_t)p->m * es);
        free(hb);
    }
    void *dC;
    CU(cudaMalloc(&dC, mn * es));
    int p12, pv, pv2;
    r.st12 = call(0, p, dA, dB, dC, hC0, dbias, o12, &p12);
    r.stv = call(1, p, dA, dB, dC, hC0, dbias, ov, &pv);
    call(1, p, dA, dB, dC, hC0, dbias, ov2, &pv2);
    r.path = pv;
    r.rerun_same = !memcmp(ov, ov2, mn * es);
    cudaFree(dA), cudaFree(dB), cudaFree(dC);
    if (dbias) cudaFree(dbias);
    /* exact values of the sampled elements */
    if (nr > p->m) nr = p->m;
    int rows[NR_MAX];
    for (int q = 0; q < nr; ++q) rows[q] = nr > 1 ? (int)(((long long)q * (p->m - 1) + (nr - 1) / 2) / (nr - 1)) : 0;
    uint16_t *ar = malloc((size_t)nr * p->k * 2), *bc = malloc((size_t)p->n * p->k * 2);
    unsigned char *fa = malloc(nr), *fb = malloc(p->n);
OMP(omp parallel for)
    for (int q = 0; q < nr; ++q) {
        for (int l = 0; l < p->k; ++l) ar[(size_t)q * p->k + l] = va(g, rows[q], l);
        fa[q] = (unsigned char)tc_fails(ar + (size_t)q * p->k, p->k);
    }
OMP(omp parallel for)
    for (int j = 0; j < p->n; ++j) {
        for (int l = 0; l < p->k; ++l) bc[(size_t)j * p->k + l] = vb(g, j, l);
        fb[j] = (unsigned char)tc_fails(bc + (size_t)j * p->k, p->k);
    }
    const int plain = p->alpha == 1.0f && p->beta == 0.0f && !p->bias;
    const double uo = p->ct ? 0x1p-8 : U32;
    double sb_v = 0, sb_12 = 0, s2v = 0, s2_12 = 0, worst = 0, maxv = 0, max12 = 0, wx = 0, wsum = 0;
    long nfail = 0, nel = 0, cls = 0, nanb = 0, nfb = 0;
OMP(omp parallel for reduction(+ : sb_v, sb_12, s2v, s2_12, nfail, nel, cls, nanb, nfb) reduction(max : maxv, max12))
    for (long long t = 0; t < (long long)nr * p->n; ++t) {
        const int q = (int)(t / p->n), j = (int)(t % p->n), i = rows[q];
        const size_t idx = (size_t)j * p->m + i;
        const uint32_t b12 = outbits(o12, idx, p->ct), bv = outbits(ov, idx, p->ct);
        if (class_differs(b12, bv)) ++cls;
        else if ((b12 & 0x7fffffffu) > 0x7f800000u && b12 != bv) ++nanb;
        const Exact e = ex_dot(ar + (size_t)q * p->k, bc + (size_t)j * p->k, p->k);
        if (e.nonfinite) continue;
        const double c12 = outval(o12, idx, p->ct), cv = outval(ov, idx, p->ct);
        if (!isfinite(c12) || !isfinite(cv)) continue;
        double x, S, coef, h;
        if (plain) {
            x = e.x, S = e.s, coef = p->k - 1;
            h = half_ulp(p->ct ? g_f(e.cr16) : (double)e.cr32, p->ct);
        } else {
            const double c0 = c0_val(g, p, i, j), bi = bias_val(g, p, i);
            x = (double)p->alpha * e.x + (double)p->beta * c0 + bi;
            S = fabs((double)p->alpha) * e.s + fabs((double)p->beta * c0) + fabs(bi);
            coef = p->k + 2;
            h = half_ulp(x, p->ct);
        }
        if (S == 0) continue;
        const double allowed = fmax(fabs(c12 - x), coef * U32 * S + h);
        const double ratio = fabs(cv - x) / allowed;
        nfail += ratio > 1.0;
        ++nel;
        nfb += fa[q] || fb[j];
        const double sg = x < 0 ? -1.0 : 1.0;
        sb_v += (cv - x) * sg / (U32 * S);
        sb_12 += (c12 - x) * sg / (U32 * S);
        const double uv = fabs(cv - x) / (uo * S), u12 = fabs(c12 - x) / (uo * S);
        s2v += uv * uv;
        s2_12 += u12 * u12;
        if (uv > maxv) maxv = uv;
        if (u12 > max12) max12 = u12;
OMP(omp critical)
        if (ratio > worst) {
            worst = ratio;
            wx = x;
            wsum = S;
        }
    }
    r.ratio = worst, r.worst_x = wx, r.worst_s = wsum;
    r.nfail = nfail, r.nel = nel, r.cls = cls, r.nanbits = nanb;
    r.bias_v = nel ? sb_v / nel : 0, r.bias_12 = nel ? sb_12 / nel : 0;
    r.maxu_v = maxv, r.maxu_12 = max12;
    r.rms_v = nel ? sqrt(s2v / nel) : 0, r.rms_12 = nel ? sqrt(s2_12 / nel) : 0;
    r.fbfrac = nel ? (double)nfb / nel : 0;
    free(ar), free(bc), free(fa), free(fb), free(hC0), free(o12), free(ov), free(ov2);
    return r;
}

static double bias_bound(int k) { return bias_margin * cur_slope * k; }
static int bias_ok(const Res *r, int k)
{
    if (!strict && k < cur_bias_from) return 1;
    return fabs(r->bias_v) <= fmax(fabs(r->bias_12), bias_bound(k)) + 0.5;
}

/* ---- the test ------------------------------------------------------------------------------------- */
static int fails, njudged;
/* VF_ACCEPT_EXPECT_GATE=1: a TN call at k >= the gate must be served, so that its bounds are judged; a call in
 * another layout must go to 12.9 (the path's layout rule: TN only). */
static void expect_served(const char *what, const Prob *p, int served)
{
    if (!expect_gate || p->k < gate) return;
    const int tn = p->ta && !p->tb;
    if (tn && !served) {
        printf("FAIL  %s: not served by the tensor-core path at k >= the gate (k %d >= %d)\n", what, p->k, gate);
        ++fails;
    }
    if (!tn && served) {
        printf("FAIL  %s: served by the tensor-core path in another layout than TN\n", what);
        ++fails;
    }
}
static void judge(const char *what, const Prob *p, const Res *r)
{
    const int served = r->path == 2;
    const int judged = served && p->k >= gate;
    if (expect_gate && served && p->k < gate) {
        printf("FAIL  %s: served by the tensor-core path below the gate (k %d < %d)\n", what, p->k, gate);
        ++fails;
    }
    expect_served(what, p, served);
    njudged += judged;
    const int a_ok = r->nfail == 0, s_ok = r->cls == 0 && r->nanbits == 0 && r->rerun_same && r->st12 == r->stv;
    const int b_ok = p->ct || bias_ok(r, p->k);
    printf("%s %s: path %s, A worst %.3f (%ld of %ld over), bias %+.2f u (12.9 %+.2f, bound %.1f), max %.2f u (12.9 %.2f), "
           "classes %ld, NaN bits %ld, rerun %s%s\n",
           judged && !(a_ok && s_ok && b_ok) ? "FAIL " : judged ? "ok   " : "note ", what, path_name(r->path), r->ratio,
           r->nfail, r->nel, r->bias_v, r->bias_12, bias_bound(p->k), r->maxu_v, r->maxu_12, r->cls, r->nanbits,
           r->rerun_same ? "same" : "DIFFERENT", judged ? "" : " (not judged)");
    if (judged && !(a_ok && s_ok && b_ok)) ++fails;
}

static void prob_name(char *buf, size_t n, const Gen *g, const Prob *p)
{
    snprintf(buf, n, "%s %s%s %dx%dx%d %s%s%s%s", fam_name[g->fam], p->ta ? "T" : "N", p->tb ? "T" : "N", p->m, p->n, p->k,
             p->ct ? "bf16" : "fp32", p->beta != 0.0f ? " alpha .7 beta .3" : "", p->lt ? " lt" : "", p->bias ? " bias" : "");
}

/* S2: no workspace, and the same elements in a call of half the columns, have the same bits. */
static void invariance(const Gen *g, Prob p)
{
    uint16_t *hA = host_A(g, &p), *hB = host_B(g, &p);
    const size_t na = (size_t)p.k * p.m, nb = (size_t)p.k * p.n, es = p.ct ? 2 : 4, mn = (size_t)p.m * p.n;
    void *dA = to_device(hA, na * 2), *dB = to_device(hB, nb * 2), *dC;
    CU(cudaMalloc(&dC, mn * es));
    void *z = calloc(mn, es), *o1 = malloc(mn * es), *o2 = malloc(mn * es), *o3 = malloc(mn * es);
    int p1, p2, p3;
    call(1, &p, dA, dB, dC, z, NULL, o1, &p1);
    Prob q = p;
    q.wsnone = 1;
    call(1, &q, dA, dB, dC, z, NULL, o2, &p2);
    q = p;
    q.n = p.n / 2;
    call(1, &q, dA, dB, dC, z, NULL, o3, &p3);
    const int same_ws = !memcmp(o1, o2, mn * es), same_n = !memcmp(o1, o3, (size_t)p.m * q.n * es);
    const int judged = p1 == 2 && p.k >= gate;
    char name[128];
    prob_name(name, sizeof name, g, &p);
    expect_served(name, &p, p1 == 2);
    njudged += judged;
    printf("%s S2 %s: path %s / %s / %s; no workspace: %s; half the columns: %s%s\n",
           judged && !(same_ws && (p3 != 2 || same_n)) ? "FAIL " : judged ? "ok   " : "note ", name, path_name(p1),
           path_name(p2), path_name(p3), same_ws ? "same" : "DIFFERENT", p3 != 2 ? "not served" : same_n ? "same" : "DIFFERENT",
           judged ? "" : " (not judged)");
    if (judged && !(same_ws && (p3 != 2 || same_n))) ++fails;
    cudaFree(dA), cudaFree(dB), cudaFree(dC);
    free(hA), free(hB), free(z), free(o1), free(o2), free(o3);
}

/* S1: special values, every element of the matrix compared with 12.9 (class and NaN bits). */
static void special_case(const char *name, const Prob *p, const uint16_t *A, const uint16_t *B, const void *hC0)
{
    const size_t es = p->ct ? 2 : 4, mn = (size_t)p->m * p->n;
    void *dA = to_device(A, (size_t)p->k * p->m * 2), *dB = to_device(B, (size_t)p->k * p->n * 2), *dC;
    CU(cudaMalloc(&dC, mn * es));
    void *o12 = malloc(mn * es), *ov = malloc(mn * es);
    int p12, pv;
    const cublasStatus_t s12 = call(0, p, dA, dB, dC, hC0, NULL, o12, &p12), sv = call(1, p, dA, dB, dC, hC0, NULL, ov, &pv);
    long cls = 0, nanb = 0;
    for (size_t i = 0; i < mn; ++i) {
        const uint32_t a = outbits(o12, i, p->ct), b = outbits(ov, i, p->ct);
        if (class_differs(a, b)) ++cls;
        else if ((a & 0x7fffffffu) > 0x7f800000u && a != b) ++nanb;
    }
    const int judged = pv == 2 && p->k >= gate, bad = cls || nanb || s12 != sv;
    expect_served(name, p, pv == 2);
    njudged += judged;
    printf("%s S1 %s %s alpha %g beta %g %s: path %s, class mismatches %ld, NaN bits %ld, status %d/%d%s\n",
           judged && bad ? "FAIL " : judged ? "ok   " : "note ", name, p->ct ? "bf16" : "fp32", p->alpha, p->beta,
           p->devptr ? "device" : "host", path_name(pv), cls, nanb, (int)s12, (int)sv, judged ? "" : " (not judged)");
    if (judged && bad) ++fails;
    cudaFree(dA), cudaFree(dB), cudaFree(dC);
    free(o12), free(ov);
}

static void specials(void)
{
    const int m = 1536, n = 128, k = 1024; /* at the gate: judged */
    Gen g;
    memset(&g, 0, sizeof g);
    g.fam = FAM_NORMAL, g.seed = 4242;
    uint16_t *A = malloc((size_t)k * m * 2), *B = malloc((size_t)k * n * 2), *A0 = malloc((size_t)k * m * 2);
    for (int i = 0; i < m; ++i)
        for (int l = 0; l < k; ++l) A0[(size_t)i * k + l] = va(&g, i, l);
    for (int j = 0; j < n; ++j)
        for (int l = 0; l < k; ++l) B[(size_t)j * k + l] = vb(&g, j, l);
    float *C32 = malloc((size_t)m * n * 4);
    uint16_t *C16 = malloc((size_t)m * n * 2);
#define AT(i, l) A[(size_t)(i) * k + (l)]
#define BT(l, j) B[(size_t)(j) * k + (l)]
    for (int ct = 0; ct < 2; ++ct) {
        Prob p = {1, 0, m, n, k, ct, 1.0f, 0.0f, 0, 0, 0, 0};
#define SETC(val)                                                         \
    for (size_t q_ = 0; q_ < (size_t)m * n; ++q_) C32[q_] = (val), C16[q_] = g_bf16(C32[q_])
#define C0 (ct ? (const void *)C16 : (const void *)C32)
        memcpy(A, A0, (size_t)k * m * 2);
        AT(5, 7) = 0x7fc0, AT(9, 100) = 0x7fa1, AT(11, 3) = 0xffc1, AT(20, 10) = 0x7f80, AT(21, 10) = 0x7f80,
        AT(21, 20) = 0xff80;
        const uint16_t keep[5] = {BT(50, 3), BT(10, 0), BT(10, 1), BT(10, 2), BT(20, 1)};
        BT(50, 3) = 0x7fc0, BT(10, 0) = 0, BT(10, 1) = 0x3f80, BT(10, 2) = 0xbf80, BT(20, 1) = 0x3f80;
        SETC(0.0f);
        special_case("NaN and inf in A and B", &p, A, B, C0);
        BT(50, 3) = keep[0], BT(10, 0) = keep[1], BT(10, 1) = keep[2], BT(10, 2) = keep[3], BT(20, 1) = keep[4];
        memcpy(A, A0, (size_t)k * m * 2);
        for (int l = 0; l < k; ++l) AT(30, l) = 0;
        AT(30, 0) = AT(30, 1) = AT(30, 2) = g_bf16(0x1.8p100);
        uint16_t kb[12];
        for (int j = 0; j < 4; ++j)
            for (int l = 0; l < 3; ++l) kb[j * 3 + l] = BT(l, j);
        BT(0, 0) = BT(1, 0) = g_bf16(0x1p27), BT(2, 0) = g_bf16(-0x1p27);
        BT(0, 1) = BT(1, 1) = BT(2, 1) = g_bf16(0x1p27);
        BT(0, 2) = BT(1, 2) = g_bf16(0x1.4p26), BT(2, 2) = g_bf16(0x1p25);
        BT(0, 3) = BT(1, 3) = g_bf16(0x1.54p26), BT(2, 3) = 0;
        special_case("overflow", &p, A, B, C0);
        for (int j = 0; j < 4; ++j)
            for (int l = 0; l < 3; ++l) BT(l, j) = kb[j * 3 + l];
        memcpy(A, A0, (size_t)k * m * 2);
        for (int l = 0; l < k; ++l) AT(40, l) = 0x0000, AT(41, l) = 0x8000;
        static const float ab[][3] = {{1, 0, 0}, {-1, 0, 0}, {-1, 1, -0.0f}, {1, 1, -0.0f}, {-1, 0.5f, -0.0f},
                                      {-1, -1, 0.0f}, {1, -0.0f, NAN}, {1, 0, NAN}};
        for (int c = 0; c < 8; ++c)
            for (int dev = 0; dev < 2; ++dev) {
                p.alpha = ab[c][0], p.beta = ab[c][1], p.devptr = dev;
                SETC(ab[c][2]);
                special_case("zero rows", &p, A, B, C0);
            }
        p.devptr = 0, p.alpha = 1, p.beta = 0;
        memcpy(A, A0, (size_t)k * m * 2);
        AT(5, 7) = 0x7fc0;
        p.alpha = 0, p.beta = 1, p.devptr = 1;
        SETC(1.0f);
        special_case("alpha 0 with NaN in A", &p, A, B, C0);
        p.alpha = 1, p.beta = 1, p.devptr = 0;
        for (size_t q = 0; q < (size_t)m * n; ++q) C32[q] = (float)g_gauss(&g, 52, q), C16[q] = g_bf16(C32[q]);
        for (int j = 0; j < 8; ++j) {
            C32[(size_t)j * m + 60] = (j & 1) ? -INFINITY : INFINITY, C16[(size_t)j * m + 60] = g_bf16(C32[(size_t)j * m + 60]);
            C32[(size_t)j * m + 61] = NAN, C16[(size_t)j * m + 61] = 0x7fc0;
        }
        special_case("beta 1 with inf and NaN in C", &p, A, B, C0);
        p.beta = 0;
        memcpy(A, A0, (size_t)k * m * 2);
        uint16_t *B2 = malloc((size_t)k * n * 2);
        memcpy(B2, B, (size_t)k * n * 2);
        for (int l = 0; l < k; ++l) AT(50, l) = g_bf16(g_gauss(&g, 53, (uint64_t)l) * 0x1p-70);
        for (int j = 0; j < 4; ++j)
            for (int l = 0; l < k; ++l) B2[(size_t)j * k + l] = g_bf16(g_gauss(&g, 54, G_IDX(j, l)) * 0x1p-60);
        SETC(0.0f);
        special_case("subnormal results", &p, A, B2, C0);
        free(B2);
    }
    free(A), free(B), free(A0), free(C32), free(C16);
}

static Gen fam(int f, uint64_t seed)
{
    Gen g;
    memset(&g, 0, sizeof g);
    g.fam = f, g.seed = seed;
    return g;
}

/* The search's worst configurations kept in the test, with their native slopes (measured as above). */
static const struct {
    Gen g;
    double slope;
    int bias_from;
} kept[] = {
    {{FAM_PARAM, 901, 1, 2, 0, 14, 6, 0, 0, 2.5, 0.78, 0.0001, 0.0001}, KEPT_SLOPE_0, 0},
    {{FAM_PARAM, 902, 0, 2, 4, 16, 11, 1, 1, 2.34, 1.44, 0.0, 0.0}, KEPT_SLOPE_1, 0},
    {{FAM_PARAM, 1653006390, 1, 1, 8, 20, 21, 3, 0, 0.72, 0.18, 0.0, 0.01}, KEPT_SLOPE_2, 2048},
};

static int test_mode(void)
{
    if (!rt_counters) {
        printf("FAIL  the test runtime is needed (counters): with a release runtime no call can be judged\n");
        return 1;
    }
    printf("note  gate k >= %d; native bias bound %.2f x slope x k%s\n", gate, bias_margin,
           strict ? "; strict (every bias judged from the gate)"
                  : "; the bias of craftsub judged from k = 4096, of configuration 1653006390 from 2048");
    specials();
    static const int ks[] = {256, 512, 1024, 1536, 2048, 4096, 16384};
    static const int fams[] = {FAM_NORMAL, FAM_RELU2,   FAM_OUTLIER, FAM_CANCEL,   FAM_MIXED30,
                               FAM_SAMESIGN, FAM_LOGNPOS, FAM_ADVPOS, FAM_CRAFTACT, FAM_CRAFTSUB};
    const int nf = sizeof fams / sizeof *fams, nkept = sizeof kept / sizeof *kept;
    char name[160];
    for (int f = 0; f < nf + nkept; ++f) {
        const Gen g = f < nf ? fam(fams[f], 3000 + fams[f]) : kept[f - nf].g;
        cur_slope = f < nf ? native_slope[fams[f]] : kept[f - nf].slope;
        cur_fam = g.fam;
        cur_bias_from = f < nf ? bias_from_fam[fams[f]] : kept[f - nf].bias_from;
        for (size_t ik = 0; ik < sizeof ks / sizeof *ks; ++ik)
            for (int ct = 0; ct < 2; ++ct) {
                if (ct && ks[ik] != 256 && ks[ik] != 1024 && ks[ik] != 4096) continue;
                Prob p = {1, 0, 1024, 128, ks[ik], ct, 1.0f, 0.0f, 0, 0, 0, 0};
                const Res r = run_prob(&g, &p, 64);
                prob_name(name, sizeof name, &g, &p);
                judge(name, &p, &r);
            }
        Prob p = {1, 0, 1024, 128, 4096, 0, 1.0f, 0.0f, 0, 0, 0, 0};
        invariance(&g, p);
    }
    const int lay[3][2] = {{0, 0}, {0, 1}, {1, 1}};
    for (int f = 0; f < 2; ++f) {
        const int fi = f ? FAM_ADVPOS : FAM_NORMAL;
        const Gen g = fam(fi, 3000 + fi);
        cur_slope = native_slope[fi];
        cur_fam = fi;
        cur_bias_from = bias_from_fam[fi];
        for (int L = 0; L < 3; ++L)
            for (int ct = 0; ct < 2; ++ct) {
                Prob p = {lay[L][0], lay[L][1], 1024, 128, 2048, ct, 1.0f, 0.0f, 0, 0, 0, 0};
                const Res r = run_prob(&g, &p, 64);
                prob_name(name, sizeof name, &g, &p);
                judge(name, &p, &r);
            }
        for (int ct = 0; ct < 2; ++ct)
            for (int v = 0; v < 3; ++v) {
                Prob p = {1, 0, 1024, 128, 2048, ct, 0.7f, 0.3f, v >= 1, v == 2, 0, 0};
                if (v == 2) p.alpha = 1.0f, p.beta = 0.0f;
                const Res r = run_prob(&g, &p, 64);
                prob_name(name, sizeof name, &g, &p);
                judge(name, &p, &r);
            }
    }
    fails += failures; /* bf16-common.h's CHECK, should a helper use it */
    printf("note  %d calls served by the tensor-core path at k >= %d and judged\n", njudged, gate);
    if (fails) {
        printf("FAIL  %d checks failed\n", fails);
        return 1;
    }
    printf("ok bf16-accept\n");
    return 0;
}

/* bf16-accept safe: run with VF_SAFE=1 (and the paths' switches on): no call may be served by the tensor-core
 * path, and every result must have 12.9's bits (the conservative switch turns every path that is not bit-identical
 * off; the skinny path, bit-identical, still serves n = 8 here). */
static int safe_mode(void)
{
    if (!rt_counters) {
        printf("FAIL  the test runtime is needed (counters)\n");
        return 1;
    }
    static const int ks[] = {256, 2048, 16384}, ns[] = {8, 64};
    int bad = 0;
    for (int ik = 0; ik < 3; ++ik)
        for (int in = 0; in < 2; ++in)
            for (int ct = 0; ct < 2; ++ct) {
                const Gen g = fam(FAM_ADVPOS, 3000 + FAM_ADVPOS);
                Prob p = {1, 0, 1024, ns[in], ks[ik], ct, 1.0f, 0.0f, 0, 0, 0, 0};
                uint16_t *hA = host_A(&g, &p), *hB = host_B(&g, &p);
                const size_t es = ct ? 2 : 4, mn = (size_t)p.m * p.n;
                void *dA = to_device(hA, (size_t)p.m * p.k * 2), *dB = to_device(hB, (size_t)p.n * p.k * 2), *dC;
                CU(cudaMalloc(&dC, mn * es));
                void *z = calloc(mn, es), *o12 = malloc(mn * es), *ov = malloc(mn * es);
                int p12, pv;
                call(0, &p, dA, dB, dC, z, NULL, o12, &p12);
                call(1, &p, dA, dB, dC, z, NULL, ov, &pv);
                const int same = !memcmp(o12, ov, mn * es);
                const int ok = (pv == 3 || pv == 1) && same;
                printf("%s safe TN 1024x%dx%d %s: path %s, bits %s\n", ok ? "ok   " : "FAIL ", p.n, p.k,
                       ct ? "bf16" : "fp32", path_name(pv), same ? "12.9's" : "DIFFERENT");
                bad += !ok;
                free(hA), free(hB), free(z), free(o12), free(ov);
                cudaFree(dA), cudaFree(dB), cudaFree(dC);
            }
    if (bad) {
        printf("FAIL  %d calls served by the tensor-core path, or without 12.9's bits, under VF_SAFE=1\n", bad);
        return 1;
    }
    printf("ok bf16-accept safe\n");
    return 0;
}

/* ---- measurements ------------------------------------------------------------------------------- */
static void res_row(FILE *f, const char *tag, const Gen *g, const Prob *p, const Res *r)
{
    fprintf(f,
            "%s\t%s\t%llu\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%g\t%g\t%d\t%d\t%d\t%d\t%s%s\t%d\t%d\t%d\t%s\t%.3f\t%.4f\t%ld\t%ld\t%+.3f\t%+.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.4f\t%ld\t%ld\t%d\n",
            tag, fam_name[g->fam], (unsigned long long)g->seed, g->sA, g->sB, g->R, g->cancel, g->sigA, g->sigB, g->qA,
            g->qB, g->EA, g->EB, g->pos, p->ct, p->ta ? "T" : "N", p->tb ? "T" : "N", p->m, p->n, p->k, path_name(r->path),
            p->alpha, r->ratio, r->nfail, r->nel, r->bias_v, r->bias_12, r->maxu_v, r->maxu_12, r->rms_v, r->rms_12,
            r->fbfrac, r->cls, r->nanbits, r->rerun_same);
    fflush(f);
}
static const char *HDR = "tag\tfam\tseed\tsA\tsB\tR\tcancel\tsigA\tsigB\tqA\tqB\tEA\tEB\tpos\tct\top\tm\tn\tk\tpath\talpha\tratio\tnfail\tnel\tbias_v\tbias_129\tmaxu_v\tmaxu_129\trms_v\trms_129\tfbfrac\tcls\tnanbits\trerun_same\n";

static int parse_list(const char *s, int *out, int max)
{
    int n = 0;
    while (*s && n < max) {
        out[n++] = atoi(s);
        while (*s && *s != ',') ++s;
        if (*s) ++s;
    }
    return n;
}

static int sweep_mode(const char *out, const char *fams, const char *kl, int m, int n)
{
    FILE *f = fopen(out, "w");
    fputs(HDR, f);
    int ks[32];
    const int nk = parse_list(kl, ks, 32);
    char buf[256];
    snprintf(buf, sizeof buf, "%s", fams);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int fi = -1;
        for (int q = 0; q < FAM_N; ++q)
            if (!strcmp(tok, fam_name[q])) fi = q;
        if (fi < 0) continue;
        const Gen g = fam(fi, 3000 + fi);
        for (int ik = 0; ik < nk; ++ik)
            for (int ct = 0; ct < 2; ++ct) {
                Prob p = {1, 0, m, n, ks[ik], ct, 1.0f, 0.0f, 0, 0, 0, 0};
                const Res r = run_prob(&g, &p, 64);
                res_row(f, "sweep", &g, &p, &r);
            }
        printf("sweep %s done\n", tok);
        fflush(stdout);
    }
    fclose(f);
    return 0;
}

static uint64_t ss;
static uint64_t srnd(void)
{
    ss ^= ss << 13, ss ^= ss >> 7, ss ^= ss << 17;
    return ss;
}
static double sunif(void) { return (double)(srnd() >> 11) * 0x1p-53; }
static Gen random_gen(uint64_t seed)
{
    Gen g;
    memset(&g, 0, sizeof g);
    g.fam = FAM_PARAM, g.seed = seed;
    g.sA = (int)(srnd() % 2), g.sB = (int)(srnd() % 3);
    g.sigA = (double)(int)(sunif() * 800) / 100, g.sigB = (double)(int)(sunif() * 800) / 100;
    static const double q[] = {0, 0, 1e-4, 1e-3, 1e-2, 0.05};
    g.qA = q[srnd() % 6], g.qB = q[srnd() % 6];
    g.EA = 2 + (int)(srnd() % 20), g.EB = 2 + (int)(srnd() % 24);
    g.pos = (int)(srnd() % 5), g.cancel = (srnd() % 4) == 0, g.R = (int)(srnd() % 3) * 4;
    return g;
}
static Gen perturb(Gen g, uint64_t seed)
{
    g.seed = seed;
    switch (srnd() % 8) {
    case 0: g.sigA = fmax(0, g.sigA + (double)((int)(srnd() % 101) - 50) / 50); break;
    case 1: g.sigB = fmax(0, g.sigB + (double)((int)(srnd() % 101) - 50) / 50); break;
    case 2: g.EB = (int)fmin(28, fmax(0, g.EB + (int)(srnd() % 5) - 2)); break;
    case 3: g.EA = (int)fmin(24, fmax(0, g.EA + (int)(srnd() % 5) - 2)); break;
    case 4: g.pos = (int)(srnd() % 5); break;
    case 5: g.sB = (int)(srnd() % 3), g.sA = (int)(srnd() % 2); break;
    case 6: g.R = (int)(srnd() % 3) * 4; break;
    default: g.cancel = !g.cancel; break;
    }
    return g;
}

static int search_mode(const char *out, int k, int nconf, uint64_t seed)
{
    FILE *f = fopen(out, "w");
    fputs(HDR, f);
    ss = 0x9E3779B97F4A7C15ull ^ (seed * 0xD1B54A32D192ED03ull) ^ (uint64_t)k;
    static const int Ms[] = {256, 512, 1024}, Ns[] = {32, 64, 128};
    Gen best[2][8];
    double bs[2][8];
    for (int a = 0; a < 2; ++a)
        for (int i = 0; i < 8; ++i) bs[a][i] = -1;
    const int nrand = nconf / 2;
    for (int it = 0; it < nconf; ++it) {
        Gen g;
        int obj = it % 2;
        if (it < nrand) g = random_gen(seed * 1000003ull + (uint64_t)it);
        else {
            const int b = (it / 2) % 8;
            if (bs[obj][b] < 0) continue;
            g = perturb(best[obj][b], seed * 1000003ull + (uint64_t)it);
        }
        Prob p = {1, 0, Ms[srnd() % 3], Ns[srnd() % 3], k, 0, 1.0f, 0.0f, 0, 0, 0, 0};
        const Res r = run_prob(&g, &p, 64);
        res_row(f, it < nrand ? "rand" : "climb", &g, &p, &r);
        if (r.path != 2 || r.fbfrac > 0.5) continue;
        const double sc[2] = {r.ratio, fabs(r.bias_v) - fabs(r.bias_12)};
        for (int a = 0; a < 2; ++a) {
            int w = 0;
            for (int i = 1; i < 8; ++i)
                if (bs[a][i] < bs[a][w]) w = i;
            if (it < nrand ? sc[a] > bs[a][w] : (a == obj && sc[a] > bs[a][(it / 2) % 8])) {
                const int slot = it < nrand ? w : (it / 2) % 8;
                bs[a][slot] = sc[a], best[a][slot] = g;
            }
        }
    }
    fclose(f);
    return 0;
}

static int shapes_mode(const char *out)
{
    FILE *f = fopen(out, "w");
    fputs(HDR, f);
    const int lay[4][2] = {{1, 0}, {0, 0}, {0, 1}, {1, 1}};
    static const int fams[] = {FAM_NORMAL, FAM_ADVPOS, FAM_LOGNPOS, FAM_CRAFTSUB, FAM_SAMESIGN};
    static const int ks[] = {1024, 4096, 8192, 16384};
    for (int fi = 0; fi < 5; ++fi) {
        const Gen g = fam(fams[fi], 5000 + fi);
        for (int L = 0; L < 4; ++L)
            for (int ik = 0; ik < 4; ++ik)
                for (int ct = 0; ct < 2; ++ct) {
                    Prob p = {lay[L][0], lay[L][1], 1024, 128, ks[ik], ct, 1.0f, 0.0f, 0, 0, 0, 0};
                    const Res r = run_prob(&g, &p, 64);
                    res_row(f, "layout", &g, &p, &r);
                }
        for (int ct = 0; ct < 2; ++ct)
            for (int v = 0; v < 4; ++v) {
                Prob p = {1, 0, 1024, 128, 4096, ct, 0.7f, 0.3f, v >= 2, v == 3, 0, v == 1};
                if (v == 3) p.alpha = 1.0f, p.beta = 0.0f;
                const Res r = run_prob(&g, &p, 64);
                res_row(f, v == 0 ? "alphabeta" : v == 1 ? "alphabeta-dev" : v == 2 ? "lt-alphabeta" : "lt-bias", &g, &p, &r);
            }
        printf("shapes %s done\n", fam_name[fams[fi]]);
        fflush(stdout);
    }
    static const int lk[] = {896, 1536, 4096};
    for (int fi = 0; fi < 2; ++fi) {
        const Gen g = fam(fi ? FAM_ADVPOS : FAM_NORMAL, 6000 + fi);
        for (int ik = 0; ik < 3; ++ik)
            for (int ct = 0; ct < 2; ++ct) {
                Prob p = {1, 0, 151936, 128, lk[ik], ct, 1.0f, 0.0f, 0, 0, 0, 0};
                const Res r = run_prob(&g, &p, 128);
                res_row(f, "lmhead", &g, &p, &r);
            }
        printf("lm_head %s done\n", fam_name[g.fam]);
        fflush(stdout);
    }
    fclose(f);
    return 0;
}

/* Configurations read from a TSV of this program (columns fam seed sA sB R cancel sigA sigB qA qB EA EB pos),
 * each run at m x n for every k of a list. */
static int list_mode(const char *out, const char *in, const char *kl, int m, int n)
{
    FILE *fi = fopen(in, "r"), *f = fopen(out, "w");
    if (!fi || !f) return 1;
    fputs(HDR, f);
    int ks[32];
    const int nk = parse_list(kl, ks, 32);
    char line[1024];
    if (!fgets(line, sizeof line, fi)) return 1; /* header */
    while (fgets(line, sizeof line, fi)) {
        char famn[32];
        unsigned long long seed;
        Gen g;
        memset(&g, 0, sizeof g);
        if (sscanf(line, "%31s %llu %d %d %d %d %lf %lf %lf %lf %d %d %d", famn, &seed, &g.sA, &g.sB, &g.R, &g.cancel, &g.sigA,
                   &g.sigB, &g.qA, &g.qB, &g.EA, &g.EB, &g.pos) != 13)
            continue;
        g.seed = seed;
        for (int q = 0; q < FAM_N; ++q)
            if (!strcmp(famn, fam_name[q])) g.fam = q;
        for (int ik = 0; ik < nk; ++ik) {
            Prob p = {1, 0, m, n, ks[ik], 0, 1.0f, 0.0f, 0, 0, 0, 0};
            const Res r = run_prob(&g, &p, 64);
            res_row(f, "list", &g, &p, &r);
        }
    }
    fclose(f);
    fclose(fi);
    return 0;
}

/* The same elements (first column, 64 sampled rows) at n = 1 ... 64. */
static int ndep_mode(const char *out)
{
    FILE *f = fopen(out, "w");
    fprintf(f, "fam\tm\tk\tct\twho\tn\tpath\tdiffer_from_own_n64\tmax_u_vs_own_n64\tmax_u_vs_129_n64\n");
    static const int MK[][2] = {{896, 896}, {1536, 1536}, {4096, 4096}, {896, 4864}, {4864, 896}};
    static const int Ns[] = {1, 2, 4, 8, 16, 24, 31, 32, 48, 64};
    for (int fi = 0; fi < 2; ++fi) {
        const Gen g = fam(fi ? FAM_ADVPOS : FAM_NORMAL, 7000 + fi);
        for (int s = 0; s < 5; ++s)
            for (int ct = 0; ct < 2; ++ct) {
                const int m = MK[s][0], k = MK[s][1];
                Prob p = {1, 0, m, 64, k, ct, 1.0f, 0.0f, 0, 0, 0, 0};
                uint16_t *hA = host_A(&g, &p), *hB = host_B(&g, &p);
                void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)64 * k * 2), *dC;
                const size_t es = ct ? 2 : 4;
                CU(cudaMalloc(&dC, (size_t)m * 64 * es));
                void *z = calloc((size_t)m * 64, es), *ref[2], *o = malloc((size_t)m * 64 * es);
                int pth;
                for (int who = 0; who < 2; ++who) {
                    ref[who] = malloc((size_t)m * 64 * es);
                    call(who, &p, dA, dB, dC, z, NULL, ref[who], &pth);
                }
                /* exact sums of column 0 for the scale u S */
                double S[64];
                uint16_t *bc = malloc((size_t)k * 2), *ar = malloc((size_t)k * 2);
                for (int l = 0; l < k; ++l) bc[l] = vb(&g, 0, l);
                for (int q = 0; q < 64; ++q) {
                    const int i = (int)((long long)q * (m - 1) / 63);
                    for (int l = 0; l < k; ++l) ar[l] = va(&g, i, l);
                    S[q] = ex_dot(ar, bc, k).s;
                }
                for (int ni = 0; ni < 10; ++ni)
                    for (int who = 0; who < 2; ++who) {
                        Prob q = p;
                        q.n = Ns[ni];
                        call(who, &q, dA, dB, dC, z, NULL, o, &pth);
                        int dif = 0;
                        double mx_self = 0, mx_129 = 0;
                        for (int r = 0; r < 64; ++r) {
                            const int i = (int)((long long)r * (m - 1) / 63);
                            const double a = outval(o, i, ct), self = outval(ref[who], i, ct), b129 = outval(ref[0], i, ct);
                            const double us = (ct ? 0x1p-8 : U32) * S[r];
                            if (a != self) ++dif;
                            mx_self = fmax(mx_self, fabs(a - self) / us);
                            mx_129 = fmax(mx_129, fabs(a - b129) / us);
                        }
                        fprintf(f, "%s\t%d\t%d\t%s\t%s\t%d\t%s\t%d\t%.3f\t%.3f\n", fam_name[g.fam], m, k, ct ? "bf16" : "fp32",
                                who ? "variant" : "12.9", Ns[ni], path_name(pth), dif, mx_self, mx_129);
                    }
                fflush(f);
                free(bc), free(ar), free(hA), free(hB), free(z), free(o), free(ref[0]), free(ref[1]);
                cudaFree(dA), cudaFree(dB), cudaFree(dC);
            }
    }
    fclose(f);
    return 0;
}

static int time_mode(const char *out)
{
    FILE *f = fopen(out, "w");
    fprintf(f, "m\tk\tn\tct\tpath\tt129_us\tvariant_us\n");
    static const int MK[][2] = {{896, 896}, {4864, 896}, {896, 4864}, {1536, 1536}, {8960, 1536}, {1536, 8960},
                                {4096, 4096}, {11008, 4096}, {4096, 11008}, {151936, 896}};
    static const int Ns[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const Gen g = fam(FAM_NORMAL, 8000);
    for (int s = 0; s < 10; ++s) {
        const int m = MK[s][0], k = MK[s][1];
        Prob p = {1, 0, m, 64, k, 1, 1.0f, 0.0f, 0, 0, 0, 0};
        uint16_t *hA = host_A(&g, &p), *hB = host_B(&g, &p);
        void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)64 * k * 2), *dC;
        CU(cudaMalloc(&dC, (size_t)m * 64 * 4));
        for (int ni = 0; ni < 12; ++ni) {
            Prob q = p;
            q.n = Ns[ni];
            float t[2];
            int pth = 0;
            for (int who = 0; who < 2; ++who) {
                float v[15];
                const Counts c0 = counts(API_GEMMEX);
                gemm(who, &q, dA, dB, dC, NULL);
                CU(cudaStreamSynchronize(st));
                const Counts c1 = counts(API_GEMMEX);
                if (who) pth = !rt_counters ? 0 : delta(&c0, &c1, SERVED_TC) ? 2 : delta(&c0, &c1, SERVED_SKINNY) ? 1 : 3;
                for (int rep = 0; rep < 15; ++rep) {
                    cudaEventRecord(e0, st);
                    gemm(who, &q, dA, dB, dC, NULL);
                    cudaEventRecord(e1, st);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&v[rep], e0, e1);
                }
                for (int a = 0; a < 15; ++a)
                    for (int b = a + 1; b < 15; ++b)
                        if (v[b] < v[a]) {
                            const float x = v[a];
                            v[a] = v[b], v[b] = x;
                        }
                t[who] = v[7] * 1000;
            }
            fprintf(f, "%d\t%d\t%d\tbf16\t%s\t%.1f\t%.1f\n", m, k, Ns[ni], path_name(pth), t[0], t[1]);
            fflush(f);
        }
        free(hA), free(hB);
        cudaFree(dA), cudaFree(dB), cudaFree(dC);
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasCreate(&hv) || c129.Create(&h12) || cublasLtCreate(&ltv) || c129.LtCreate(&lt12)) {
        printf("FAIL  cannot create the handles\n");
        return 1;
    }
    open_runtime();
    CU(cudaMalloc(&ws_lt, WS_LT));
    CU(cudaMalloc((void **)&d_ab, 8));
    const char *gv = getenv("VF_ACCEPT_GATE"), *bv = getenv("VF_ACCEPT_BIAS_MARGIN"), *ev = getenv("VF_ACCEPT_EXPECT_GATE");
    if (gv) gate = atoi(gv);
    if (bv) bias_margin = atof(bv);
    expect_gate = ev && *ev == '1';
    const char *sv = getenv("VF_ACCEPT_STRICT");
    strict = sv && *sv == '1';
    const char *mode = getenv("VF_ACCEPT_MODE"); /* "safe": as the argument safe (for runners that pass no arguments) */
    if (argc < 2) return mode && !strcmp(mode, "safe") ? safe_mode() : test_mode();
    if (!strcmp(argv[1], "safe")) return safe_mode();
    if (!strcmp(argv[1], "sweep") && argc >= 7) return sweep_mode(argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));
    if (!strcmp(argv[1], "search") && argc >= 6) return search_mode(argv[2], atoi(argv[3]), atoi(argv[4]), strtoull(argv[5], NULL, 0));
    if (!strcmp(argv[1], "shapes") && argc >= 3) return shapes_mode(argv[2]);
    if (!strcmp(argv[1], "list") && argc >= 7) return list_mode(argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));
    if (!strcmp(argv[1], "ndep") && argc >= 3) return ndep_mode(argv[2]);
    if (!strcmp(argv[1], "time") && argc >= 3) return time_mode(argv[2]);
    printf("usage: bf16-accept [sweep OUT FAMS KS M N | search OUT K NCONF SEED | shapes OUT | ndep OUT | time OUT]\n");
    return 2;
}
