/* The skinny path's domain (review section 5, G1).
 *
 *   bf16-skinny kernels OUT.tsv   which kernels cuBLAS 12.9 launches for each configuration of a grid
 *                                 (shapes, the four operations, output types, alignments of the
 *                                 pointers and leading dimensions, handle workspaces, legacy and
 *                                 cuBLASLt): the call is captured into a graph that is never launched,
 *                                 and the kernel nodes' names are read. Nothing is computed.
 *   bf16-skinny check OUT.tsv     for each configuration of a smaller grid: the path the variant takes,
 *                                 whether its result has 12.9's bits, the error of both against fp64
 *                                 (on a sample of rows), and the time of both.
 *
 * tests/gpu/forward-bf16.sh runs both and analyses the files (tests/gpu/forward/bf16-skinny.py). */
#include "bf16-common.h"

#define NWS 10
static const size_t ws_size[NWS] = {0, 0, 32u << 20, 4u << 20, 8u << 20, 4u << 10, 64u << 10, 256u << 10, 1u << 20, 2u << 20};
static const char *const ws_name[NWS] = {"default", "none", "32MiB", "4MiB", "8MiB", "4KiB", "64KiB", "256KiB", "1MiB", "2MiB"};
static cublasHandle_t h12_ws[NWS]; /* 12.9 directly: its own default workspace (0), then the sizes above */
static cublasHandle_t hv;
static cublasLtHandle_t lt12;
static cudaStream_t cap_stream;
static char *big;          /* one buffer for every operand: captures read no data */
static size_t big_bytes;
static void *ws_big;

static const char *opname(int t) { return t ? "T" : "N"; }

/* Names of the kernels in a captured graph, joined by commas (prefixes: enough to tell them apart). */
static void kernel_names(cudaGraph_t g, char *out, size_t cap)
{
    out[0] = 0;
    size_t n = 0;
    cudaGraphGetNodes(g, NULL, &n);
    cudaGraphNode_t *nodes = calloc(n ? n : 1, sizeof *nodes);
    cudaGraphGetNodes(g, nodes, &n);
    for (size_t i = 0; i < n; ++i) {
        enum cudaGraphNodeType t;
        cudaGraphNodeGetType(nodes[i], &t);
        const char *name = NULL;
        char other[32];
        if (t == cudaGraphNodeTypeKernel) {
            CUDA_KERNEL_NODE_PARAMS p;
            memset(&p, 0, sizeof p);
            if (cuGraphKernelNodeGetParams((CUgraphNode)nodes[i], &p) == CUDA_SUCCESS && p.func)
                cuFuncGetName(&name, p.func);
            if (!name) name = "?";
        } else {
            snprintf(other, sizeof other, "node%d", (int)t);
            name = other;
        }
        char shortname[72];
        snprintf(shortname, sizeof shortname, "%.64s", name);
        if (strlen(out) + strlen(shortname) + 2 < cap) {
            if (out[0]) strcat(out, ",");
            strcat(out, shortname);
        }
    }
    free(nodes);
}

typedef struct {
    int ta, tb, m, n, k, ct; /* ct: 0 fp32, 1 bf16 */
    int aoff, boff, coff;    /* byte offsets of the operands from 256-byte aligned addresses */
    int ldpad;               /* added to every leading dimension */
    int ws;                  /* 0: the handle's default workspace, 1: none, 2: 32 MiB, 3: 4 MiB, 4: 8 MiB */
    int lt;                  /* through cublasLtMatmul */
} Cfg;

static int fits(const Cfg *c, long long *lda, long long *ldb, long long *ldc)
{
    *lda = (c->ta ? c->k : c->m) + c->ldpad;
    *ldb = (c->tb ? c->n : c->k) + c->ldpad;
    *ldc = c->m + c->ldpad;
    const long long a = *lda * (c->ta ? c->m : c->k) * 2, b = *ldb * (c->tb ? c->k : c->n) * 2,
                    cc = *ldc * c->n * (c->ct ? 2 : 4);
    return a + 4096 <= (long long)big_bytes / 2 && b + 4096 <= (long long)big_bytes / 4 &&
           cc + 4096 <= (long long)big_bytes / 4;
}

/* Captures 12.9's call for c; 0 when it was not captured (the configuration is skipped). */
static int capture_129(const Cfg *c, char *names, size_t cap)
{
    long long lda, ldb, ldc;
    if (!fits(c, &lda, &ldb, &ldc)) return 0;
    const void *A = big + c->aoff, *B = big + big_bytes / 2 + c->boff;
    void *C = big + big_bytes / 2 + big_bytes / 4 + c->coff;
    const float one = 1.0f, zero = 0.0f;
    cudaGraph_t g = NULL;
    if (cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeRelaxed) != cudaSuccess) return 0;
    cublasStatus_t st;
    if (!c->lt) {
        st = c129.GemmEx(h12_ws[c->ws], c->ta ? CUBLAS_OP_T : CUBLAS_OP_N, c->tb ? CUBLAS_OP_T : CUBLAS_OP_N, c->m, c->n,
                         c->k, &one, A, CUDA_R_16BF, (int)lda, B, CUDA_R_16BF, (int)ldb, &zero, C,
                         c->ct ? CUDA_R_16BF : CUDA_R_32F, (int)ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    } else {
        cublasLtMatmulDesc_t d;
        cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        const int32_t ta = c->ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb = c->tb ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
        cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, c->ta ? c->k : c->m, c->ta ? c->m : c->k, lda);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, c->tb ? c->n : c->k, c->tb ? c->k : c->n, ldb);
        cublasLtMatrixLayoutCreate(&lc, c->ct ? CUDA_R_16BF : CUDA_R_32F, c->m, c->n, ldc);
        const size_t wsz = c->ws == 0 ? (32u << 20) : ws_size[c->ws];
        st = c129.LtMatmul(lt12, d, &one, A, la, B, lb, &zero, C, lc, C, lc, NULL, wsz ? ws_big : NULL, wsz, cap_stream);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulDescDestroy(d);
    }
    if (cudaStreamEndCapture(cap_stream, &g) != cudaSuccess || !g) {
        cudaGetLastError();
        return 0;
    }
    if (st != CUBLAS_STATUS_SUCCESS) snprintf(names, cap, "status%d", (int)st);
    else kernel_names(g, names, cap);
    cudaGraphDestroy(g);
    return 1;
}

static void kernels_mode(const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) {
        printf("FAIL  cannot write %s\n", path);
        exit(1);
    }
    fprintf(f, "api\tta\ttb\tm\tn\tk\tout\taoff\tboff\tcoff\tldpad\tws\tkernels\n");
    const int ms[] = {1, 2, 3, 7, 8, 9, 16, 31, 32, 33, 63, 64, 97, 127, 128, 200, 251, 256, 320, 384, 416, 448, 509,
                      512, 640, 768, 777, 896, 1021, 1024, 1152, 1536, 1920, 2048, 3072, 4096, 4608, 7344, 9728,
                      16384, 50257, 151936};
    const int ns[] = {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 24, 31};
    const int ks[] = {16, 17, 64, 255, 256, 384, 512, 896, 1024, 1531, 1536, 1920, 2048, 3072, 4096, 4608, 4864,
                      8192, 16384, 65536, 1000000};
    const int nm = sizeof ms / sizeof ms[0], nn = sizeof ns / sizeof ns[0], nk = sizeof ks / sizeof ks[0];
    char names[1024];
    long long count = 0;
    const char *const *wsname = ws_name;
    /* 1. The grid: every shape, operation and output type, aligned and packed, with the handle's
     * default workspace and with none. */
    for (int w = 0; w < 2; ++w)
    for (int op = 0; op < 4; ++op)
        for (int ct = 0; ct < 2; ++ct)
            for (int a = 0; a < nm; ++a)
                for (int b = 0; b < nn; ++b)
                    for (int c = 0; c < nk; ++c) {
                        Cfg g = {op >> 1, op & 1, ms[a], ns[b], ks[c], ct, 0, 0, 0, 0, w, 0};
                        if ((long long)g.m * g.k > (1LL << 28)) continue;
                        if (capture_129(&g, names, sizeof names)) {
                            fprintf(f, "legacy\t%s\t%s\t%d\t%d\t%d\t%s\t0\t0\t0\t0\t%s\t%s\n", opname(g.ta),
                                    opname(g.tb), g.m, g.n, g.k, ct ? "bf16" : "fp32", wsname[w], names);
                            ++count;
                        }
                    }
    /* 2. Alignments, leading dimensions and workspaces, on a subset of shapes. */
    const int ms2[] = {64, 200, 448, 777, 1536, 4608, 16384}, ns2[] = {2, 5, 8, 16}, ks2[] = {64, 384, 1531, 1536, 4096};
    const int offs[][4] = {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 4, 0}, {8, 8, 8, 0}, {0, 0, 0, 1}, {0, 0, 0, 8}};
    for (int op = 0; op < 4; ++op)
        for (int ct = 0; ct < 2; ++ct)
            for (int a = 0; a < 7; ++a)
                for (int b = 0; b < 4; ++b)
                    for (int c = 0; c < 5; ++c) {
                        for (int v = 0; v < 6; ++v) {
                            Cfg g = {op >> 1, op & 1, ms2[a], ns2[b], ks2[c], ct, offs[v][0], offs[v][1], offs[v][2],
                                     offs[v][3], 0, 0};
                            if (capture_129(&g, names, sizeof names)) {
                                fprintf(f, "legacy\t%s\t%s\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%d\tdefault\t%s\n",
                                        opname(g.ta), opname(g.tb), g.m, g.n, g.k, ct ? "bf16" : "fp32", g.aoff,
                                        g.boff, g.coff, g.ldpad, names);
                                ++count;
                            }
                        }
                        for (int w = 1; w < NWS; ++w)
                            for (int lt = 0; lt < 2; ++lt) {
                                Cfg g = {op >> 1, op & 1, ms2[a], ns2[b], ks2[c], ct, 0, 0, 0, 0, w, lt};
                                if (capture_129(&g, names, sizeof names)) {
                                    fprintf(f, "%s\t%s\t%s\t%d\t%d\t%d\t%s\t0\t0\t0\t0\t%s\t%s\n", lt ? "lt" : "legacy",
                                            opname(g.ta), opname(g.tb), g.m, g.n, g.k, ct ? "bf16" : "fp32",
                                            wsname[w], names);
                                    ++count;
                                }
                            }
                    }
    /* 3. Validation: random shapes, not on the grid, inside and around the regions where 12.9 was
     * seen to run its sequential kernel; random alignments and workspaces for NT and TT. */
    rng_state = 0x5eed5eedull;
    for (int i = 0; i < 24000; ++i) {
        const int op = (int)(rnd() % 4), ct = (int)(rnd() % 2);
        Cfg g = {op >> 1, op & 1, 0, 0, 0, ct, 0, 0, 0, 0, 0, 0};
        if (!g.tb) {
            /* NN and TN: multiples of 8 near and inside the regions, with a workspace, aligned */
            g.m = 8 * (1 + (int)(rnd() % (g.ta ? 2560 : 1536)));
            g.k = 8 * (16 + (int)(rnd() % (g.ta ? 640 : 2400)));
            g.ws = (int)(rnd() % 3) == 0 ? 3 + (int)(rnd() % 2) : 0;
        } else {
            g.m = 1 + (int)(rnd() % 20000);
            g.k = 16 + (int)(rnd() % 16384);
            g.ws = (int)(rnd() % NWS);
            if (rnd() % 2) {
                g.aoff = 2 * (int)(rnd() % 8);
                g.boff = 2 * (int)(rnd() % 8);
                g.coff = (ct ? 2 : 4) * (int)(rnd() % 4);
                g.ldpad = (int)(rnd() % 9);
            }
        }
        g.n = 2 + (int)(rnd() % 30);
        if (g.n > 16 && (long long)g.m * g.k >= (1LL << 20)) g.n = 2 + (int)(rnd() % 15);
        if ((long long)g.m * g.k > (1LL << 28)) continue;
        g.lt = (int)(rnd() % 4) == 0;
        if (g.lt && g.ws == 0) g.ws = 2;
        if (capture_129(&g, names, sizeof names)) {
            fprintf(f, "%s\t%s\t%s\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%s\t%s\tvalidation\n", g.lt ? "lt" : "legacy",
                    opname(g.ta), opname(g.tb), g.m, g.n, g.k, ct ? "bf16" : "fp32", g.aoff, g.boff, g.coff, g.ldpad,
                    wsname[g.ws], names);
            ++count;
        }
    }
    fclose(f);
    printf("captured %lld configurations into %s\n", count, path);
}

/* ---- check: execute ---- */

static float time_call(int variant, const Cfg *c, const void *A, long long lda, const void *B, long long ldb, void *C,
                       long long ldc, int reps)
{
    const float one = 1.0f, zero = 0.0f;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0, cap_stream);
        if (variant)
            cublasGemmEx(hv, c->ta ? CUBLAS_OP_T : CUBLAS_OP_N, c->tb ? CUBLAS_OP_T : CUBLAS_OP_N, c->m, c->n, c->k, &one,
                         A, CUDA_R_16BF, (int)lda, B, CUDA_R_16BF, (int)ldb, &zero, C, c->ct ? CUDA_R_16BF : CUDA_R_32F,
                         (int)ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        else
            c129.GemmEx(h12_ws[c->ws], c->ta ? CUBLAS_OP_T : CUBLAS_OP_N, c->tb ? CUBLAS_OP_T : CUBLAS_OP_N, c->m, c->n, c->k,
                        &one, A, CUDA_R_16BF, (int)lda, B, CUDA_R_16BF, (int)ldb, &zero, C,
                        c->ct ? CUDA_R_16BF : CUDA_R_32F, (int)ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(e1, cap_stream);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return best * 1000.0f; /* microseconds */
}

/* Largest error of the rows sampled, in units of 2^-24 sum |a b| (fp64 reference). */
static double sample_error(const Cfg *c, const uint16_t *hA, long long lda, const uint16_t *hB, long long ldb,
                           const void *hC, long long ldc)
{
    double worst = 0;
    const int rows = c->m < 48 ? c->m : 48;
    for (int r = 0; r < rows; ++r) {
        const int i = (int)((long long)r * c->m / rows);
        for (int j = 0; j < c->n; ++j) {
            double sum = 0, mag = 0;
            for (int l = 0; l < c->k; ++l) {
                const double p = (double)opel(hA, c->ta, lda, i, l) * (double)opel(hB, !c->tb, ldb, j, l);
                sum += p;
                mag += fabs(p);
            }
            const double got = c->ct ? bf2f(((const uint16_t *)hC)[i + (long long)j * ldc])
                                     : ((const float *)hC)[i + (long long)j * ldc];
            const double e = fabs(got - sum) / (mag > 0 ? mag * 0x1p-24 : 1.0);
            if (e > worst) worst = e;
        }
    }
    return worst;
}

/* Uniform values in (-scale, scale), fast enough for operands of 10^8 elements. */
static uint16_t *bf16_uniform(size_t n, double scale)
{
    uint16_t *h = malloc(n * 2);
    for (size_t i = 0; i < n; ++i) h[i] = f2bf((float)(((int32_t)rnd() / 2147483648.0) * scale));
    return h;
}

static void check_one(FILE *f, const Cfg *c, const char *region)
{
    long long lda = (c->ta ? c->k : c->m) + c->ldpad, ldb = (c->tb ? c->n : c->k) + c->ldpad, ldc = c->m + c->ldpad;
    const size_t na = (size_t)lda * (c->ta ? c->m : c->k), nb = (size_t)ldb * (c->tb ? c->k : c->n);
    const size_t es = c->ct ? 2 : 4, cb = (size_t)ldc * c->n * es;
    if (na * 2 > big_bytes / 2 || nb * 2 > big_bytes / 4 || 2 * cb > big_bytes / 4) return;
    const int fast = !strcmp(region, "bounds") || na > (64u << 20);
    uint16_t *hA = fast ? bf16_uniform(na, 0.1) : bf16_random(na, 0.05);
    uint16_t *hB = fast ? bf16_uniform(nb, 2.0) : bf16_random(nb, 1.0);
    void *A = big + c->aoff, *B = big + big_bytes / 2 + c->boff, *C1 = big + big_bytes / 2 + big_bytes / 4 + c->coff,
         *C2 = (char *)C1 + cb + 256;
    /* the variant's handle with the workspace of the configuration (cublasSetStream gives the default back) */
    if (c->ws) cublasSetWorkspace(hv, ws_big, ws_size[c->ws]);
    CU(cudaMemcpy(A, hA, na * 2, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(B, hB, nb * 2, cudaMemcpyHostToDevice));
    const Counts k0 = counts(API_GEMMEX);
    const float tv = time_call(1, c, A, lda, B, ldb, C1, ldc, 5);
    const Counts k1 = counts(API_GEMMEX);
    const float t12 = time_call(0, c, A, lda, B, ldb, C2, ldc, 5);
    CU(cudaStreamSynchronize(cap_stream));
    if (c->ws) cublasSetStream(hv, cap_stream);
    void *o1 = from_device(C1, cb), *o2 = from_device(C2, cb);
    const char *path = delta(&k0, &k1, SERVED_SKINNY) ? "skinny" : delta(&k0, &k1, SERVED_TC) ? "tc" : "forward";
    char names[1024];
    Cfg g = *c;
    capture_129(&g, names, sizeof names);
    const char *kernel = strstr(names, "gemmSN") ? "gemmSN" : strstr(names, "magma_sgemmEx") ? "magma" : "other";
    const size_t diff = ndiff(o1, o2, cb, es);
    const int sample = (long long)c->m * c->n * c->k <= (1LL << 31);
    const double ev = sample ? sample_error(c, hA, lda, hB, ldb, o1, ldc) : -1;
    const double e12 = sample ? sample_error(c, hA, lda, hB, ldb, o2, ldc) : -1;
    fprintf(f, "%s\t%s%s\t%d\t%d\t%d\t%s\t%s\t%s\t%zu\t%.2f\t%.2f\t%.1f\t%.1f\t%d/%d/%d/%d/%s\n", region,
            opname(c->ta), opname(c->tb), c->m, c->n, c->k, c->ct ? "bf16" : "fp32", path, kernel, diff, ev, e12, tv, t12,
            c->aoff, c->boff, c->coff, c->ldpad, ws_name[c->ws]);
    fflush(f);
    free(hA);
    free(hB);
    free(o1);
    free(o2);
}

static void check_mode(const char *path, const char *which)
{
    FILE *f = fopen(path, "w");
    if (!f) exit(1);
    fprintf(f, "region\top\tm\tn\tk\tout\tpath\tkernel129\tdiffering\terr_variant\terr_129\tus_variant\tus_129\tvariant\n");
    rng_state = 0xc0ffee;
    /* 1. Bits, inside the rule's regions (both output types). */
    const int rule = !which || !strcmp(which, "rule"), bounds = !which || !strcmp(which, "bounds");
    /* 0. The rule's edges, alignments and workspaces: each just inside and just outside (forwarded:
     * 12.9's bits trivially), both output types, n = 8 and 16. */
    if (which && !strcmp(which, "edges")) {
        static const int e[][9] = {
            /* ta tb m k aoff boff coff ldpad ws */
            {1, 0, 440, 1536, 0, 0, 0, 0, 0}, {1, 0, 448, 1536, 0, 0, 0, 0, 0}, {1, 0, 456, 1536, 0, 0, 0, 0, 0},
            {1, 0, 16384, 1536, 0, 0, 0, 0, 0}, {1, 0, 16392, 1536, 0, 0, 0, 0, 0}, {1, 0, 1536, 248, 0, 0, 0, 0, 0},
            {1, 0, 1536, 256, 0, 0, 0, 0, 0}, {1, 0, 1536, 4096, 0, 0, 0, 0, 0}, {1, 0, 1536, 4104, 0, 0, 0, 0, 0},
            {0, 0, 120, 1536, 0, 0, 0, 0, 0}, {0, 0, 128, 1536, 0, 0, 0, 0, 0}, {0, 0, 9728, 1536, 0, 0, 0, 0, 0},
            {0, 0, 9736, 1536, 0, 0, 0, 0, 0}, {0, 0, 1536, 888, 0, 0, 0, 0, 0}, {0, 0, 1536, 896, 0, 0, 0, 0, 0},
            {0, 0, 1536, 16384, 0, 0, 0, 0, 0}, {0, 0, 1536, 16392, 0, 0, 0, 0, 0},
            {1, 0, 1536, 1536, 2, 0, 0, 0, 0}, {1, 0, 1536, 1536, 0, 2, 0, 0, 0}, {1, 0, 1536, 1536, 0, 0, 4, 0, 0},
            {1, 0, 1536, 1536, 0, 0, 0, 1, 0}, {1, 0, 1536, 1536, 0, 0, 0, 8, 0}, {1, 0, 1536, 1536, 0, 0, 0, 0, 1},
            {1, 0, 1536, 1536, 0, 0, 0, 0, 8}, {1, 0, 1536, 1536, 0, 0, 0, 0, 9}, {1, 0, 1536, 1536, 0, 0, 0, 0, 3},
            {1, 0, 1536, 1536, 0, 0, 0, 0, 4}, {1, 0, 1536, 1536, 0, 0, 0, 0, 2},
            {0, 0, 1536, 1536, 2, 0, 0, 0, 0}, {0, 0, 1536, 1536, 0, 0, 0, 0, 9}, {0, 0, 1536, 1536, 0, 0, 0, 0, 4},
            {0, 1, 1536, 1531, 2, 2, 4, 3, 1}, {1, 1, 1536, 1531, 2, 2, 4, 3, 1}, {0, 1, 777, 4096, 6, 0, 0, 5, 9},
            {1, 1, 4608, 256, 0, 6, 8, 1, 0}, {0, 1, 4608, 248, 0, 0, 0, 0, 0}, {0, 1, 4616, 1536, 0, 0, 0, 0, 0},
            {0, 1, 1536, 64, 0, 0, 0, 0, 0}, {0, 1, 1544, 64, 0, 0, 0, 0, 0}, {0, 1, 120, 64, 0, 0, 0, 0, 0},
        };
        for (size_t i = 0; i < sizeof e / sizeof e[0]; ++i)
            for (int ct = 0; ct < 2; ++ct)
                for (int n = 8; n <= 16; n += 8) {
                    Cfg g = {e[i][0], e[i][1], e[i][2], n, e[i][3], ct, e[i][4], e[i][5], e[i][6], e[i][7], e[i][8], 0};
                    check_one(f, &g, "edges");
                }
    }
    const int tn_m[] = {448, 512, 1024, 1536, 3072, 4608, 7344, 16384}, tn_k[] = {256, 512, 1536, 3072, 4096};
    const int nn_m[] = {128, 448, 1536, 4608, 9728}, nn_k[] = {896, 1536, 4096, 8192, 16384};
    const int t_m[] = {128, 256, 777, 1536, 4608, 16384, 50257}, t_k[] = {16, 64, 256, 1531, 4096, 16384};
    const int ns[] = {2, 3, 8, 13, 16, 24};
    for (int ct = 0; ct < 2 * rule; ++ct)
        for (int b = 0; b < 6; ++b) {
            for (int a = 0; a < 8; ++a)
                for (int c = 0; c < 5; ++c) {
                    Cfg g = {1, 0, tn_m[a], ns[b], tn_k[c], ct, 0, 0, 0, 0, 0, 0};
                    if (g.n > 16 && (long long)g.m * g.k >= (1LL << 20)) continue;
                    check_one(f, &g, "rule-TN");
                }
            for (int a = 0; a < 5; ++a)
                for (int c = 0; c < 5; ++c) {
                    Cfg g = {0, 0, nn_m[a], ns[b], nn_k[c], ct, 0, 0, 0, 0, 0, 0};
                    if (g.n > 16 && (long long)g.m * g.k >= (1LL << 20)) continue;
                    check_one(f, &g, "rule-NN");
                }
            for (int t = 0; t < 2; ++t)
                for (int a = 0; a < 7; ++a)
                    for (int c = 0; c < 6; ++c) {
                        Cfg g = {t, 1, t_m[a], ns[b], t_k[c], ct, 0, 0, 0, 0, 0, 0};
                        if (g.n > 16 && (long long)g.m * g.k >= (1LL << 20)) continue;
                        check_one(f, &g, t ? "rule-TT" : "rule-NT");
                    }
        }
    /* 2. Bounds: small m and long k (time against 12.9), NT (element loads) and TN (16-byte loads), fp32
     * out; run with VF_BF16_CONTROL=skinny-k,skinny-speed,seq-any so that the skinny path serves them. */
    const int bm[] = {2, 8, 32, 64, 128, 256, 512, 1024}, bk[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576};
    const int bn[] = {2, 8, 16};
    for (int op = 0; op < 2 * bounds; ++op)
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 3; ++b)
                for (int c = 0; c < 7; ++c) {
                    Cfg g = {op, op ? 0 : 1, bm[a], bn[b], bk[c], 0, 0, 0, 0, 0, 0, 0};
                    check_one(f, &g, "bounds");
                }
    fclose(f);
    printf("checked into %s\n", path);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("usage: bf16-skinny kernels|check OUT.tsv\n");
        return 2;
    }
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreateWithFlags(&cap_stream, cudaStreamNonBlocking));
    big_bytes = (size_t)2 << 30;
    CU(cudaMalloc((void **)&big, big_bytes));
    CU(cudaMalloc(&ws_big, 32u << 20));
    for (int i = 0; i < NWS; ++i) {
        if (c129.Create(&h12_ws[i])) {
            printf("FAIL  cublasCreate (12.9)\n");
            return 1;
        }
        c129.SetStream(h12_ws[i], cap_stream);
    }
    for (int i = 1; i < NWS; ++i) c129.SetWorkspace(h12_ws[i], ws_size[i] ? ws_big : NULL, ws_size[i]);
    c129.LtCreate(&lt12);
    if (cublasCreate(&hv)) return 1;
    open_runtime();
    if (!strcmp(argv[1], "kernels")) kernels_mode(argv[2]);
    else if (!strcmp(argv[1], "check")) {
        cublasSetStream(hv, cap_stream);
        check_mode(argv[2], argc > 3 ? argv[3] : NULL);
    } else {
        printf("FAIL  unknown mode %s\n", argv[1]);
        return 2;
    }
    if (failures) return 1;
    printf("ok bf16-skinny\n");
    return 0;
}
