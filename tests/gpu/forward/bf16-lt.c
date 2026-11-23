/* cuBLASLt calls on the skinny path's shapes that must keep cuBLAS 12.9's behaviour (review F1, F2):
 * without an algorithm, and with the algorithm 12.9's heuristic gives (id 16), the variant serves them
 * with 12.9's bits; with an algorithm 12.9 computes otherwise (id 11) the call is forwarded, and the
 * result is 12.9's with that algorithm; so is id 16 with any configuration attribute other than
 * cublasLtMatmulAlgoInit leaves it (a split-K count of 0, tile 5, reduction scheme 1, custom option 1,
 * stages id 3, CTA swizzling 1: 12.9 refuses all but the last, delta review F1b), with 12.9's status;
 * a workspace size without a workspace gets 12.9's status; so does a workspace that is not 256-byte
 * aligned, and a legacy call with such a workspace has 12.9's bits; a bias that overlaps D is forwarded.
 * 12.9's own results come from the same descriptors given to libcublasLt.so.12 directly.
 * VF_BF16_CONTROL=algo (explicit algorithms served), ws-align and bias-alias make it fail. Prints
 * "ok bf16-lt". */
#include "bf16-common.h"

static cublasLtHandle_t ltv, lt12;
static cudaStream_t st;

typedef struct {
    int ta, m, n, k; /* TN (ta = 1) or NN (ta = 0) */
} Shape;

static int run(const Shape *s, const cublasLtMatmulAlgo_t *algo, int variant, void *dA, void *dB, void *dC, void *ws,
               size_t wss, cublasStatus_t *status)
{
    cublasLtMatmulDesc_t d;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const int32_t oa = s->ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, s->ta ? s->k : s->m, s->ta ? s->m : s->k, s->ta ? s->k : s->m);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, s->k, s->n, s->k);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, s->m, s->n, s->m);
    const float one = 1.0f, zero = 0.0f;
    *status = variant ? cublasLtMatmul(ltv, d, &one, dA, la, dB, lb, &zero, dC, lc, dC, lc, algo, ws, wss, st)
                      : c129.LtMatmul(lt12, d, &one, dA, la, dB, lb, &zero, dC, lc, dC, lc, algo, ws, wss, st);
    cudaStreamSynchronize(st);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(d);
    return 0;
}

/* A workspace that is not 256-byte aligned, and a bias that overlaps D: 12.9's to compute or refuse.
 * cuBLASLt: 12.9 refuses such a workspace (CUBLAS_STATUS_INVALID_VALUE); the call is forwarded and gets
 * that status. Legacy: with a 4 MiB workspace set 16 bytes past an aligned address, 12.9 runs gemmSN at TN
 * 448 x 8 x 256 instead of its sequential kernel; the call is forwarded and has 12.9's bits. A bias that
 * overlaps D is forwarded (12.9's own result there depends on the order of its threads: not compared). */
static void ws_and_bias(void *ws)
{
    const int m = 1536, n = 8, k = 1536;
    uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)n * k, 1.0), *hbias = bf16_random(m, 1.0);
    void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)n * k * 2), *c1, *c2;
    CU(cudaMalloc(&c1, (size_t)m * n * 4));
    CU(cudaMalloc(&c2, (size_t)m * n * 4));
    static const int offs[] = {16, 128};
    for (int o = 0; o < 2; ++o) {
        cublasStatus_t s1, s2;
        const Shape sh = {1, m, n, k};
        const Counts k0 = counts(API_LT);
        run(&sh, NULL, 1, dA, dB, c1, (char *)ws + offs[o], 32u << 20, &s1);
        const Counts k1 = counts(API_LT);
        run(&sh, NULL, 0, dA, dB, c2, (char *)ws + offs[o], 32u << 20, &s2);
        cudaGetLastError();
        printf("     cuBLASLt, a 32 MiB workspace %d bytes past an aligned address: status %d / 12.9 %d; %s\n", offs[o],
               (int)s1, (int)s2, delta(&k0, &k1, SERVED_SKINNY) ? "served" : "forwarded");
        CHECK(s1 == s2 && (!rt_counters || delta(&k0, &k1, FWD(R_ARGS)) == 1),
              "cuBLASLt, a workspace %d bytes past an aligned address: status %d where 12.9 gives %d, or not forwarded",
              offs[o], (int)s1, (int)s2);
    }
    /* legacy, through cublasSetWorkspace */
    cublasHandle_t hv, h12;
    if (cublasCreate(&hv) || c129.Create(&h12)) exit(1);
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    /* (shapes outside the tensor-core path's gate, which takes a call the skinny path does not) */
    static const int lshapes[][3] = {{448, 8, 256}, {1024, 8, 512}};
    for (int i = 0; i < 2; ++i) {
        const int lm = lshapes[i][0], ln = lshapes[i][1], lk = lshapes[i][2];
        const float one = 1.0f, zero = 0.0f;
        const cublasStatus_t w1 = cublasSetWorkspace(hv, (char *)ws + 16, 4u << 20),
                             w2 = c129.SetWorkspace(h12, (char *)ws + (16u << 20) + 16, 4u << 20);
        cudaMemset(c1, 0, (size_t)lm * ln * 4);
        cudaMemset(c2, 0xff, (size_t)lm * ln * 4);
        cudaDeviceSynchronize();
        const Counts k0 = counts(API_GEMMEX);
        const cublasStatus_t s1 = cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, lm, ln, lk, &one, dA, CUDA_R_16BF, lk, dB,
                                               CUDA_R_16BF, lk, &zero, c1, CUDA_R_32F, lm, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT);
        cudaStreamSynchronize(st);
        const Counts k1 = counts(API_GEMMEX);
        const cublasStatus_t s2 = c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, lm, ln, lk, &one, dA, CUDA_R_16BF, lk, dB,
                                              CUDA_R_16BF, lk, &zero, c2, CUDA_R_32F, lm, CUBLAS_COMPUTE_32F,
                                              CUBLAS_GEMM_DEFAULT);
        cudaStreamSynchronize(st);
        float *r1 = from_device(c1, (size_t)lm * ln * 4), *r2 = from_device(c2, (size_t)lm * ln * 4);
        const size_t diff = ndiff(r1, r2, (size_t)lm * ln * 4, 4);
        printf("     legacy TN %d x %d x %d, a 4 MiB workspace 16 bytes past an aligned address: set %d / 12.9 %d, status %d "
               "/ %d; %s; %zu elements differ from 12.9's\n", lm, ln, lk, (int)w1, (int)w2, (int)s1, (int)s2,
               delta(&k0, &k1, SERVED_SKINNY) ? "served by the skinny path" : "forwarded", diff);
        CHECK(s1 == s2 && w1 == w2 && diff == 0 && (!rt_counters || delta(&k0, &k1, FWD(R_SEQ)) == 1),
              "legacy TN %d x %d x %d with a 4 MiB workspace 16 bytes past an aligned address: not 12.9's result (%zu "
              "elements differ), or not forwarded", lm, ln, lk, diff);
        free(r1);
        free(r2);
    }
    cublasDestroy(hv);
    /* a bias that overlaps D (bf16 output: D's fourth column holds the bias) */
    {
        cublasLtMatmulDesc_t d;
        cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        const int32_t oa = CUBLAS_OP_T;
        const uint32_t e = CUBLASLT_EPILOGUE_BIAS;
        void *dbias = (char *)c1 + (size_t)3 * m * 2;
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof e);
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof dbias);
        cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, m, k);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, n, k);
        cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, m, n, m);
        CU(cudaMemcpy((char *)c1 + (size_t)3 * m * 2, hbias, (size_t)m * 2, cudaMemcpyHostToDevice));
        CU(cudaDeviceSynchronize());
        const float one = 1.0f, zero = 0.0f;
        const Counts k0 = counts(API_LT);
        const cublasStatus_t s1 = cublasLtMatmul(ltv, d, &one, dA, la, dB, lb, &zero, c1, lc, c1, lc, NULL, ws, 32u << 20, st);
        cudaStreamSynchronize(st);
        const Counts k1 = counts(API_LT);
        printf("     cuBLASLt, a bias that overlaps D: status %d; %s\n", (int)s1,
               delta(&k0, &k1, SERVED_SKINNY) ? "served by the skinny path" : "forwarded");
        CHECK(s1 == CUBLAS_STATUS_SUCCESS && (!rt_counters || delta(&k0, &k1, FWD(R_ALIAS)) == 1),
              "cuBLASLt, a bias that overlaps D: status %d, or not forwarded as aliasing", (int)s1);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulDescDestroy(d);
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(c1);
    cudaFree(c2);
    free(hA);
    free(hB);
    free(hbias);
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasLtCreate(&ltv) || c129.LtCreate(&lt12)) return 1;
    open_runtime();
    void *ws;
    const size_t WSS = 32u << 20;
    CU(cudaMalloc(&ws, WSS + 4096));
    static const Shape shapes[] = {{1, 1536, 8, 1536}, {1, 4608, 16, 4096}, {0, 1536, 16, 4096}};
    for (size_t i = 0; i < sizeof shapes / sizeof *shapes; ++i) {
        const Shape *s = &shapes[i];
        uint16_t *hA = bf16_random((size_t)s->m * s->k, 0.05), *hB = bf16_random((size_t)s->n * s->k, 1.0);
        void *dA = to_device(hA, (size_t)s->m * s->k * 2), *dB = to_device(hB, (size_t)s->n * s->k * 2), *c1, *c2;
        const size_t cb = (size_t)s->m * s->n * 4;
        CU(cudaMalloc(&c1, cb));
        CU(cudaMalloc(&c2, cb));
        cudaDeviceSynchronize();
        /* the algorithms: id 11 (accepted by cublasLtMatmulAlgoCheck), and the heuristic's first */
        cublasLtMatmulAlgo_t a11, ah;
        cublasLtMatmulAlgoInit(ltv, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F, 11,
                               &a11);
        {
            cublasLtMatmulDesc_t d;
            cublasLtMatrixLayout_t la, lb, lc;
            cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            const int32_t oa = s->ta ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
            cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, s->ta ? s->k : s->m, s->ta ? s->m : s->k, s->ta ? s->k : s->m);
            cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, s->k, s->n, s->k);
            cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, s->m, s->n, s->m);
            cublasLtMatmulHeuristicResult_t chk, hr[4];
            const cublasStatus_t ck = cublasLtMatmulAlgoCheck(ltv, d, la, lb, lc, lc, &a11, &chk);
            CHECK(ck == CUBLAS_STATUS_SUCCESS, "shape %zu: algorithm 11 not accepted by cublasLtMatmulAlgoCheck (%d)", i,
                  (int)ck);
            cublasLtMatmulPreference_t pref;
            cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WSS, sizeof WSS);
            int got = 0;
            cublasLtMatmulAlgoGetHeuristic(ltv, d, la, lb, lc, lc, pref, 4, hr, &got);
            CHECK(got > 0, "shape %zu: no heuristic result", i);
            ah = hr[0].algo;
            int32_t id = -1;
            size_t w = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&ah, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
            printf("shape %c%c %d x %d x %d: the heuristic's algorithm is id %d\n", s->ta ? 'T' : 'N', 'N', s->m, s->n,
                   s->k, id);
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(la);
            cublasLtMatrixLayoutDestroy(lb);
            cublasLtMatrixLayoutDestroy(lc);
            cublasLtMatmulDescDestroy(d);
        }
        /* id 16 as cublasLtMatmulAlgoInit leaves it, and with one configuration attribute changed */
        static const struct {
            int attr;
            uint32_t value;
            const char *what;
        } variants[] = {{-1, 0, "id 16 from AlgoInit"},
                        {CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 0, "id 16, split-K count 0"},
                        {CUBLASLT_ALGO_CONFIG_TILE_ID, 5, "id 16, tile 5"},
                        {CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 1, "id 16, reduction scheme 1"},
                        {CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 1, "id 16, custom option 1"},
                        {CUBLASLT_ALGO_CONFIG_STAGES_ID, 3, "id 16, stages id 3"},
                        {CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 1, "id 16, CTA swizzling 1"}};
        enum { NV = sizeof variants / sizeof variants[0] };
        cublasLtMatmulAlgo_t a16[NV];
        for (int v = 0; v < NV; ++v) {
            cublasLtMatmulAlgoInit(ltv, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F,
                                   16, &a16[v]);
            if (variants[v].attr >= 0) {
                const cublasStatus_t ss = cublasLtMatmulAlgoConfigSetAttribute(
                    &a16[v], (cublasLtMatmulAlgoConfigAttributes_t)variants[v].attr, &variants[v].value,
                    sizeof variants[v].value);
                CHECK(ss == CUBLAS_STATUS_SUCCESS, "shape %zu: %s: cublasLtMatmulAlgoConfigSetAttribute returned %d", i,
                      variants[v].what, (int)ss);
            }
        }
        const char *names[3 + NV] = {"no algorithm", "the heuristic's algorithm", "algorithm id 11"};
        const cublasLtMatmulAlgo_t *algos[3 + NV] = {NULL, &ah, &a11};
        int want_served[3 + NV] = {1, 1, 0};
        for (int v = 0; v < NV; ++v) {
            names[3 + v] = variants[v].what;
            algos[3 + v] = &a16[v];
            want_served[3 + v] = variants[v].attr < 0;
        }
        for (int a = 0; a < 3 + NV; ++a) {
            cublasStatus_t s1, s2;
            cudaMemset(c1, 0, cb);
            cudaMemset(c2, 0xff, cb);
            cudaDeviceSynchronize();
            const Counts k0 = counts(API_LT);
            run(s, algos[a], 1, dA, dB, c1, ws, WSS, &s1);
            const Counts k1 = counts(API_LT);
            run(s, algos[a], 0, dA, dB, c2, ws, WSS, &s2);
            float *r1 = from_device(c1, cb), *r2 = from_device(c2, cb);
            const size_t diff = ndiff(r1, r2, cb, 4);
            const int served = delta(&k0, &k1, SERVED_SKINNY) == 1, fwd = delta(&k0, &k1, FWD(R_ALGO)) == 1;
            printf("     %-28s status %d / 12.9 %d; %s; %zu elements differ from 12.9's\n", names[a], (int)s1, (int)s2,
                   served ? "served by the skinny path" : fwd ? "forwarded (explicit algorithm)" : "forwarded",
                   diff);
            CHECK(s1 == s2 && (s1 != CUBLAS_STATUS_SUCCESS || diff == 0),
                  "shape %zu, %s: not 12.9's result (%zu elements differ, status %d / %d)", i, names[a], diff, (int)s1,
                  (int)s2);
            CHECK(!rt_counters || (want_served[a] ? served : fwd), "shape %zu, %s: %s", i, names[a],
                  want_served[a] ? "not served by the skinny path" : "not forwarded for its algorithm");
            free(r1);
            free(r2);
        }
        /* a workspace size without a workspace: 12.9's status */
        cublasStatus_t s1, s2;
        const Counts k0 = counts(API_LT);
        run(s, NULL, 1, dA, dB, c1, NULL, WSS, &s1);
        const Counts k1 = counts(API_LT);
        run(s, NULL, 0, dA, dB, c2, NULL, WSS, &s2);
        cudaGetLastError();
        printf("     %-28s status %d / 12.9 %d; %s\n", "32 MiB without a workspace", (int)s1, (int)s2,
               delta(&k0, &k1, SERVED_SKINNY) ? "served" : "forwarded");
        CHECK(s1 == s2 && (!rt_counters || delta(&k0, &k1, FWD(R_ARGS)) == 1),
              "shape %zu: a size without a workspace: status %d where 12.9 gives %d, or not forwarded", i, (int)s1, (int)s2);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(c1);
        cudaFree(c2);
        free(hA);
        free(hB);
    }
    ws_and_bias(ws);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-lt\n");
    return 0;
}
