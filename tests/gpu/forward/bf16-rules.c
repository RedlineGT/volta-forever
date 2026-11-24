/* The tensor-core path's layout rule, tall rule and wide rule on the GPU, with the products of a fine-tuning step as
 * cuBLAS gets them (a model of hidden size 896, MLP size 4864, vocabulary 151936, rank-16 adapters, 2048 tokens), an
 * output layer at a small batch and the down projection of a wide MLP:
 *   - the input gradients (NN, k = 4864) and the weight gradients (NT, k = the tokens) go to cuBLAS 12.9, with its bits,
 *     counted for the layout's reason;
 *   - the down projection and its adapter's product (TN, k = 4864) are served by the tensor-core path when k is in its
 *     gate (else forwarded for the gate's reason): error within 32 units of 2^-24 sum |a b| of fp64 on sampled
 *     elements, the same bits twice;
 *   - TN with m = 151936 up to the tall rule's n goes to 12.9, with its bits, for the tall rule; one column more, to
 *     the tensor-core path (k from the runtime's gate when that is above 2048);
 *   - TN with k past the wide rule's bound (the down projection of an MLP of 8960) goes to 12.9 with more columns
 *     than its n, for the wide rule, and to the tensor-core path at its n;
 *   - a long product of many rows (TN 7680 x 128 x 12800, of the class of the tall rule's former long-k branch) and an
 *     output layer at 128 columns (the tall rule's tier from 65 to 128 columns) go to the tensor-core path.
 * The bounds are the runtime's (a test build reports them). Run by tests/gpu/forward-bf16.sh with
 * VF_CUBLAS_BF16_TC=1 and a test build of the runtime, then with VF_BF16_CONTROL=tc-layouts, tc-tall and tc-wide
 * (each rule removed), where it must fail. Prints "ok bf16-rules". */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cudaStream_t st;
static int gate = 2048; /* the runtime's shortest k for the tensor-core path (test builds) */

/* Operands in [-scale, scale), from the shared generator (uniform: fast for the large ones). */
static uint16_t *fill(size_t n, float scale)
{
    uint16_t *h = malloc(n * 2);
    for (size_t i = 0; i < n; ++i) h[i] = f2bf(scale * ((float)(rnd() >> 8) * 0x1p-23f - 1.0f));
    return h;
}

static cublasStatus_t gemm(int variant, int ta, int tb, int m, int n, int k, const void *A, const void *B, void *C)
{
    const float one = 1.0f, zero = 0.0f;
    const int lda = ta ? k : m, ldb = tb ? n : k;
    return (variant ? cublasGemmEx : c129.GemmEx)(variant ? hv : h12, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                  tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_16BF, lda, B,
                                                  CUDA_R_16BF, ldb, &zero, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);
}

static const char *lay(int ta, int tb) { return ta ? (tb ? "TT" : "TN") : (tb ? "NT" : "NN"); }

/* One product through the variants and through 12.9: forwarded for the reason given (12.9's bits), or served by the
 * tensor-core path (want_reason < 0: within the bound on 256 sampled elements, the same bits twice). */
static void product(const char *what, int ta, int tb, int m, int n, int k, int want_reason)
{
    const size_t na = (size_t)m * k, nb = (size_t)n * k, cb = (size_t)m * n * 4;
    uint16_t *hA = fill(na, 0.05f), *hB = fill(nb, 1.0f);
    void *A = to_device(hA, na * 2), *B = to_device(hB, nb * 2), *Cv, *C2;
    CU(cudaMalloc(&Cv, cb));
    CU(cudaMalloc(&C2, cb));
    const Counts c0 = counts(API_GEMMEX);
    const cublasStatus_t s1 = gemm(1, ta, tb, m, n, k, A, B, Cv);
    CU(cudaStreamSynchronize(st));
    const Counts c1 = counts(API_GEMMEX);
    float *rv = from_device(Cv, cb);
    if (want_reason >= 0) {
        const cublasStatus_t s2 = gemm(0, ta, tb, m, n, k, A, B, C2);
        CU(cudaStreamSynchronize(st));
        float *r2 = from_device(C2, cb);
        const size_t d = ndiff(rv, r2, cb, 4);
        const int counted = !rt_counters || delta(&c0, &c1, FWD(want_reason)) == 1;
        printf("%s %d x %d x %d (%s): status %d / 12.9 %d, %s, %zu elements differ from 12.9's\n", lay(ta, tb), m, n, k,
               what, (int)s1, (int)s2, !rt_counters ? "(not counted)" : counted ? "forwarded" : "NOT forwarded", d);
        CHECK(s1 == 0 && s2 == 0 && counted, "%s %d x %d x %d (%s): not forwarded for the %s", lay(ta, tb), m, n, k, what,
              want_reason == R_LAYOUT ? "layout" : want_reason == R_TALL ? "tall rule" : want_reason == R_WIDE ? "wide rule" : "gate");
        CHECK(d == 0, "%s %d x %d x %d (%s): %zu elements differ from 12.9's", lay(ta, tb), m, n, k, what, d);
        free(r2);
    } else {
        const cublasStatus_t s2 = gemm(1, ta, tb, m, n, k, A, B, C2);
        CU(cudaStreamSynchronize(st));
        float *r2 = from_device(C2, cb);
        const int served = !rt_counters || delta(&c0, &c1, SERVED_TC) == 1;
        double worst = 0;
        for (int s = 0; s < 256; ++s) {
            const long long i = ((long long)s * 7919) % m, j = ((long long)s * 104729) % n;
            double x = 0, mag = 0;
            for (int l = 0; l < k; ++l) {
                const double p = (double)opel(hA, ta, ta ? k : m, i, l) * (double)opel(hB, !tb, tb ? n : k, j, l);
                x += p;
                mag += fabs(p);
            }
            const double e = fabs((double)rv[i + j * m] - x) / (mag > 0 ? mag * 0x1p-24 : 1.0);
            if (e > worst || isnan(e)) worst = isnan(e) ? INFINITY : e;
        }
        printf("%s %d x %d x %d (%s): status %d, %s, error %.2f units on 256 sampled elements, the same bits twice: %s\n",
               lay(ta, tb), m, n, k, what, (int)s1, !rt_counters ? "(not counted)" : served ? "served by the tensor-core "
               "path" : "NOT served by the tensor-core path", worst, memcmp(rv, r2, cb) ? "NO" : "yes");
        CHECK(s1 == 0 && s2 == 0 && served && worst <= 32.0 && !memcmp(rv, r2, cb),
              "%s %d x %d x %d (%s): not served by the tensor-core path within 32 units, the same bits twice", lay(ta, tb),
              m, n, k, what);
        free(r2);
    }
    free(rv);
    free(hA);
    free(hB);
    cudaFree(A);
    cudaFree(B);
    cudaFree(Cv);
    cudaFree(C2);
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasCreate(&hv) || c129.Create(&h12)) return 1;
    open_runtime();
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    int (*min_k)(void) = NULL;
    void *rt = dlopen("libvf_cublas_bf16.so", RTLD_NOW | RTLD_NOLOAD);
    if (rt) *(void **)&min_k = dlsym(rt, "vf_bf16_test_tc_min_k");
    if (min_k) gate = min_k();
    int (*bound)(int) = NULL;
    if (rt) *(void **)&bound = dlsym(rt, "vf_bf16_test_rule_bound");
    /* the tall rule's n and the wide rule's k and n (the documented values without a test build) */
    const int tall_n = bound ? bound(1) : 64, wide_k = bound ? bound(3) : 8192, wide_n = bound ? bound(4) : 256;
    printf("the tensor-core path's gate: k >= %d; tall rule: n <= %d forwarded; wide rule: k > %d with n > %d forwarded\n",
           gate, tall_n, wide_k, wide_n);
    const int T = 2048, kt = gate > 2048 ? gate : 2048; /* the tokens of the step; k of the tall rows */
    /* input gradients (NN) and weight gradients (NT): 12.9's */
    product("the gradient of an MLP's input through its gate and up projections", 0, 0, 896, T, 4864, R_LAYOUT);
    product("the gradient of a rank-16 adapter's activations", 0, 0, 16, T, 4864, R_LAYOUT);
    product("the weight gradient of a gate or up projection", 0, 1, 896, 4864, T, R_LAYOUT);
    product("the weight gradient of the down projection", 0, 1, 4864, 896, T, R_LAYOUT);
    product("the weight gradient of a q or o projection", 0, 1, 896, 896, T, R_LAYOUT);
    product("the weight gradient of a k or v projection", 0, 1, 896, 128, T, R_LAYOUT);
    product("the weight gradient of the output layer", 0, 1, 896, 151936, T, R_LAYOUT);
    product("a rank-16 adapter's B gradient, gate or up projection", 0, 1, 16, 4864, T, R_LAYOUT);
    product("a rank-16 adapter's B gradient, q or o projection", 0, 1, 16, 896, T, R_LAYOUT);
    product("a rank-16 adapter's A gradient, down projection", 0, 1, 4864, 16, T, R_LAYOUT);
    /* the forward products of the down projection (TN): the tensor-core path, from its gate on, unless wide */
    const int down = 4864 > wide_k && T > wide_n ? R_WIDE : 4864 >= gate ? -1 : R_SHORT_K;
    product("the down projection", 1, 0, 896, T, 4864, down);
    product("a rank-16 adapter's A product, down projection", 1, 0, 16, T, 4864, 4864 >= gate ? -1 : R_SHORT_K);
    /* an output layer (TN, m = 151936): 12.9's up to the tall rule's n, the tensor-core path one column more */
    product("an output layer, n = 2", 1, 0, 151936, 2, kt, R_TALL);
    product("an output layer, n at the tall rule's bound", 1, 0, 151936, tall_n, kt, R_TALL);
    product("an output layer, one column past it", 1, 0, 151936, tall_n + 1, kt, -1);
    /* the down projection of an MLP of 8960 (TN, k = 8960 past the wide rule's bound): 12.9's with more columns than
     * the wide rule's n, the tensor-core path at it */
    if (8960 > wide_k) {
        product("the down projection of an MLP of 8960, past the wide rule's n", 1, 0, 1536, wide_n + 64, 8960, R_WIDE);
        product("the down projection of an MLP of 8960, at the wide rule's n", 1, 0, 1536, wide_n, 8960, -1);
    }
    /* classes that the rules by shape class serve: a long product of many rows (one of the slowest points of its
     * sweep), an output layer at 128 columns */
    product("a long product of many rows", 1, 0, 7680, 128, 12800, -1);
    if (128 > tall_n) product("an output layer at 128 columns", 1, 0, 151936, 128, kt, -1);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-rules\n");
    return 0;
}
