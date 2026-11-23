/* cuBLASLt calls on the skinny path's shapes that must keep cuBLAS 12.9's behaviour (review F1, F2):
 * without an algorithm, and with the algorithm 12.9's heuristic gives (id 16), the variant serves them
 * with 12.9's bits; with an algorithm 12.9 computes otherwise (id 11) the call is forwarded, and the
 * result is 12.9's with that algorithm; a workspace size without a workspace gets 12.9's status.
 * 12.9's own results come from the same descriptors given to libcublasLt.so.12 directly.
 * VF_BF16_CONTROL=algo (explicit algorithms served) makes it fail. Prints "ok bf16-lt". */
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
    CU(cudaMalloc(&ws, WSS));
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
        static const char *const names[] = {"no algorithm", "the heuristic's algorithm", "algorithm id 11"};
        const cublasLtMatmulAlgo_t *algos[] = {NULL, &ah, &a11};
        const int want_served[] = {1, 1, 0};
        for (int a = 0; a < 3; ++a) {
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
            CHECK(s1 == s2 && diff == 0, "shape %zu, %s: not 12.9's result (%zu elements differ, status %d / %d)", i,
                  names[a], diff, (int)s1, (int)s2);
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
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-lt\n");
    return 0;
}
