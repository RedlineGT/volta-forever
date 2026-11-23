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
    else {
        printf("FAIL  unknown mode %s\n", argv[1]);
        return 2;
    }
    if (failures) return 1;
    printf("ok bf16-skinny\n");
    return 0;
}
