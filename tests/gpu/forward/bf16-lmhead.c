/* A language model's output layer, m = 151936 rows (Qwen2.5's vocabulary), on the GPU (the review's gap:
 * chunking at that size had only the host test): k = 896 (Qwen2.5-0.5B's hidden size) for the skinny path,
 * k = 2048 (Qwen2.5-3B's) for the tensor-core path, which takes k >= 1024 only. Run twice by
 * tests/gpu/forward-bf16.sh:
 *   VF_CUBLAS_BF16_TC=0 VF_CUBLAS_BF16_SKINNY=1   the skinny path alone forwards every such call (its rule:
 *                                                 TN up to m = 16384, NN 9728, NT and TT 4608), and each has
 *                                                 12.9's bits: TN at n = 2, 8, 16, NN, NT and TT at n = 8
 *   VF_CUBLAS_BF16_TC=1                           TN at n = 8, 64 and the tall rule's bound (128) goes to 12.9,
 *                                                 with its bits, for the tall rule (12.9 is as fast there); the
 *                                                 tensor-core path serves TN at one and 64 columns more outside a
 *                                                 capture, in chunks of at most 32 MiB of workspace: error
 *                                                 against fp64 within 32 units of 2^-24 sum |a b| on 512
 *                                                 sampled rows, the same bits twice; captured, the call is
 *                                                 forwarded and the replay has 12.9's bits
 * With a test build of the runtime, k is its gate when that is above 2048, so that the tensor-core path still takes
 * these calls. Prints "ok bf16-lmhead". */
#include "bf16-common.h"

static const int M = 151936;
static int K = 896; /* 2048 with the tensor-core path (or the runtime's gate, when above) */
static cublasHandle_t hv, h12;
static cudaStream_t st;

static cublasStatus_t gemm(cublasHandle_t h, int direct, int ta, int tb, int n, const void *A, const void *B, void *C)
{
    const float one = 1.0f, zero = 0.0f;
    const int lda = ta ? K : M, ldb = tb ? n : K;
    return (direct ? c129.GemmEx : cublasGemmEx)(h, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, M, n, K,
                                                 &one, A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb, &zero, C, CUDA_R_32F, M,
                                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

/* The largest error over 512 sampled rows of a TN product, in units of 2^-24 sum |a b|. */
static double sampled_err(const uint16_t *A, const uint16_t *B, int n, const float *C)
{
    double worst = 0;
    for (int s = 0; s < 512; ++s) {
        const long long i = ((long long)s * 104729) % M;
        for (int j = 0; j < n; ++j) {
            double x = 0, mag = 0;
            for (int l = 0; l < K; ++l) {
                const double p = (double)bf2f(A[l + i * K]) * (double)bf2f(B[l + (long long)j * K]);
                x += p;
                mag += fabs(p);
            }
            const double e = fabs((double)C[i + (long long)j * M] - x) / (mag > 0 ? mag * 0x1p-24 : 1.0);
            if (e > worst || isnan(e)) worst = isnan(e) ? INFINITY : e;
        }
    }
    return worst;
}

int main(void)
{
    const char *tcv = getenv("VF_CUBLAS_BF16_TC");
    const int tc = tcv && !strcmp(tcv, "1");
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasCreate(&hv) || c129.Create(&h12)) return 1;
    open_runtime();
    if (tc) {
        int (*min_k)(void) = NULL;
        void *rt = dlopen("libvf_cublas_bf16.so", RTLD_NOW | RTLD_NOLOAD);
        if (rt) *(void **)&min_k = dlsym(rt, "vf_bf16_test_tc_min_k");
        K = min_k && min_k() > 2048 ? min_k() : 2048;
    }
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    int tall_n = 128; /* the tall rule's n (a test build reports it) */
    {
        int (*bound)(int) = NULL;
        void *rt = dlopen("libvf_cublas_bf16.so", RTLD_NOW | RTLD_NOLOAD);
        if (rt) *(void **)&bound = dlsym(rt, "vf_bf16_test_rule_bound");
        if (bound) tall_n = bound(1);
    }
    const int NMAX = tall_n + 64;
    uint16_t *hA = bf16_random((size_t)M * K, 0.05), *hB = bf16_random((size_t)NMAX * K, 1.0);
    void *A = to_device(hA, (size_t)M * K * 2), *B = to_device(hB, (size_t)NMAX * K * 2), *C, *C2;
    CU(cudaMalloc(&C, (size_t)M * NMAX * 4));
    CU(cudaMalloc(&C2, (size_t)M * NMAX * 4));
    printf("m = %d, k = %d; the tensor-core path %s\n", M, K, tc ? "on" : "off");
    if (!tc) {
        static const int cases[][3] = {{1, 0, 2}, {1, 0, 8}, {1, 0, 16}, {0, 0, 8}, {0, 1, 8}, {1, 1, 8}};
        for (int c = 0; c < 6; ++c) {
            const int ta = cases[c][0], tb = cases[c][1], n = cases[c][2];
            const size_t cb = (size_t)M * n * 4;
            cudaMemset(C, 0, cb);
            cudaMemset(C2, 0xff, cb);
            CU(cudaDeviceSynchronize());
            const Counts k0 = counts(API_GEMMEX);
            const cublasStatus_t s1 = gemm(hv, 0, ta, tb, n, A, B, C);
            CU(cudaStreamSynchronize(st));
            const Counts k1 = counts(API_GEMMEX);
            const cublasStatus_t s2 = gemm(h12, 1, ta, tb, n, A, B, C2);
            CU(cudaStreamSynchronize(st));
            float *r1 = from_device(C, cb), *r2 = from_device(C2, cb);
            const size_t d = ndiff(r1, r2, cb, 4);
            const int served = delta(&k0, &k1, SERVED_SKINNY) || delta(&k0, &k1, SERVED_TC);
            printf("%c%c n = %2d: status %d / 12.9 %d, %s, %zu elements differ from 12.9's\n", ta ? 'T' : 'N', tb ? 'T' : 'N',
                   n, (int)s1, (int)s2, !rt_counters ? "(not counted)" : served ? "SERVED" : "forwarded", d);
            CHECK(s1 == 0 && s2 == 0 && d == 0 && (!rt_counters || !served), "%c%c n = %d: not forwarded with 12.9's bits",
                  ta ? 'T' : 'N', tb ? 'T' : 'N', n);
            free(r1);
            free(r2);
        }
    } else {
        /* tall and narrow: 12.9's, with its bits */
        const int nt[] = {8, 64, tall_n};
        for (int c = 0; c < 3; ++c) {
            const int n = nt[c];
            const size_t cb = (size_t)M * n * 4;
            const Counts k0 = counts(API_GEMMEX);
            const cublasStatus_t s1 = gemm(hv, 0, 1, 0, n, A, B, C);
            CU(cudaStreamSynchronize(st));
            const Counts k1 = counts(API_GEMMEX);
            const cublasStatus_t s2 = gemm(h12, 1, 1, 0, n, A, B, C2);
            CU(cudaStreamSynchronize(st));
            float *r1 = from_device(C, cb), *r2 = from_device(C2, cb);
            const size_t d = ndiff(r1, r2, cb, 4);
            printf("TN n = %2d: status %d / 12.9 %d, %s, %zu elements differ from 12.9's\n", n, (int)s1, (int)s2,
                   !rt_counters ? "(not counted)" : delta(&k0, &k1, FWD(R_TALL)) == 1 ? "forwarded (tall)" : "NOT forwarded "
                   "for the tall rule", d);
            CHECK(s1 == 0 && s2 == 0 && d == 0 && (!rt_counters || delta(&k0, &k1, FWD(R_TALL)) == 1),
                  "TN n = %d: not forwarded for the tall rule with 12.9's bits", n);
            free(r1);
            free(r2);
        }
        const int ns[] = {tall_n + 1, tall_n + 64};
        for (int c = 0; c < 2; ++c) {
            const int n = ns[c];
            const size_t cb = (size_t)M * n * 4;
            const Counts k0 = counts(API_GEMMEX);
            const cublasStatus_t s1 = gemm(hv, 0, 1, 0, n, A, B, C);
            const cublasStatus_t s1b = gemm(hv, 0, 1, 0, n, A, B, C2);
            CU(cudaStreamSynchronize(st));
            const Counts k1 = counts(API_GEMMEX);
            float *r1 = from_device(C, cb), *r1b = from_device(C2, cb);
            const double err = sampled_err(hA, hB, n, r1);
            printf("TN n = %2d, outside a capture: status %d, %s, error %.2f units on 512 sampled rows, the same bits twice: "
                   "%s\n", n, (int)s1, !rt_counters ? "(not counted)" : delta(&k0, &k1, SERVED_TC) == 2 ? "served by the "
                   "tensor-core path" : "NOT served by the tensor-core path", err, memcmp(r1, r1b, cb) ? "NO" : "yes");
            CHECK(s1 == 0 && s1b == 0 && err <= 32.0 && !memcmp(r1, r1b, cb) &&
                      (!rt_counters || delta(&k0, &k1, SERVED_TC) == 2),
                  "TN n = %d outside a capture: not served by the tensor-core path within 32 units, the same bits twice", n);
            free(r1);
            free(r1b);
        }
        /* captured: forwarded, and the replay has 12.9's bits */
        const int n = tall_n + 64;
        const size_t cb = (size_t)M * n * 4;
        cudaStream_t cs;
        CU(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        cublasSetStream(hv, cs);
        cudaGraph_t g = NULL;
        cudaGraphExec_t e;
        const Counts k0 = counts(API_GEMMEX);
        CU(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        gemm(hv, 0, 1, 0, n, A, B, C);
        CU(cudaStreamEndCapture(cs, &g));
        const Counts k1 = counts(API_GEMMEX);
        CU(cudaGraphInstantiate(&e, g, 0));
        CU(cudaGraphLaunch(e, cs));
        CU(cudaStreamSynchronize(cs));
        gemm(h12, 1, 1, 0, n, A, B, C2);
        CU(cudaStreamSynchronize(st));
        float *r1 = from_device(C, cb), *r2 = from_device(C2, cb);
        const size_t d = ndiff(r1, r2, cb, 4);
        printf("TN n = %d, captured: %s, the replay %s 12.9's bits (%zu elements differ)\n", n,
               !rt_counters ? "(not counted)" : delta(&k0, &k1, FWD(R_CAPTURED)) == 1 ? "forwarded" : "NOT forwarded",
               d ? "has not" : "has", d);
        CHECK(d == 0 && (!rt_counters || delta(&k0, &k1, FWD(R_CAPTURED)) == 1),
              "TN n = %d captured: not forwarded, or the replay has not 12.9's bits", n);
        free(r1);
        free(r2);
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    }
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-lmhead\n");
    return 0;
}
