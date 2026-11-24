/* Verify mode (VF_CUBLAS_BF16_VERIFY=1) with two threads. One verification is several launches on the call's stream
 * (C copied into the stream's scratch buffer, the skinny kernel, 12.9's call into the buffer, the comparison): calls
 * that two threads make on one stream must not interleave them on the buffer, or a comparison reads another call's
 * data and reports differences that are not there. Two threads, each with its own handle, make 300 skinny calls each
 * (TN 1536 x 8 x 1536, alpha 1, beta 0.3: C read; every call with its own C and its own output), all enqueued before
 * any is waited for, started together, on:
 *   - "one stream": a stream of the program's, the same for both threads;
 *   - "legacy stream": the legacy default stream (stream 0), the same for both threads;
 *   - "per-thread stream": the per-thread default stream (cudaStreamPerThread: another stream in each thread);
 *   - "a stream each": one stream per thread (their buffers were never shared: a check of the test itself).
 * In each case all 600 calls must be verified and none may differ (vf_bf16_test_verify, a test build's counters), and
 * each of the 600 outputs the program gets must be 12.9's bits (cuBLAS 12.9 called directly with the same arguments,
 * compared on the host). With the test build's control VF_BF16_CONTROL=verify-unlocked (no lock across a call's
 * launches, and one buffer for the per-thread default streams of all threads), the shared streams' cases report
 * differences that are not there, and the test fails. Prints "ok bf16-verify-threads". */
#include "bf16-common.h"
#include <pthread.h>

#define M 1536
#define N 8
#define K 1536
#define CALLS 300
#define THREADS 2

static int (*rt_verify)(unsigned long long *, int);
static void *dA;                 /* A: K x M (op(A) = A^T), shared */
static void *dB[THREADS];        /* B: K x N, one per thread */
static float *hC0[THREADS];      /* each call's C before the call: CALLS x M x N per thread */
static float *dC[THREADS];       /* the outputs the program gets */
static cublasHandle_t hv[THREADS];
static pthread_barrier_t start;

typedef struct {
    int t;
    cudaStream_t s;
    cublasStatus_t bad;
} Work;

static void *worker(void *arg)
{
    Work *w = arg;
    const float alpha = 1.0f, beta = 0.3f;
    /* the device's context current in this thread before its first call (without it, the first call finds no
     * current context and goes to 12.9) */
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    cublasSetStream(hv[w->t], w->s);
    pthread_barrier_wait(&start);
    for (int i = 0; i < CALLS; ++i) {
        const cublasStatus_t st = cublasGemmEx(hv[w->t], CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, CUDA_R_16BF, K,
                                               dB[w->t], CUDA_R_16BF, K, &beta, dC[w->t] + (size_t)i * M * N,
                                               CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS && !w->bad) w->bad = st;
    }
    if (cudaStreamSynchronize(w->s) != cudaSuccess && !w->bad) w->bad = CUBLAS_STATUS_EXECUTION_FAILED;
    return NULL;
}

static void vcounts(unsigned long long v[7])
{
    memset(v, 0, 7 * sizeof *v);
    CU(cudaDeviceSynchronize());
    if (rt_verify && rt_verify(v, 7) != 7) {
        printf("FAIL  verify mode's counters could not be read\n");
        exit(1);
    }
}

/* One case: both threads' calls on the streams given, then the counters and the outputs checked. */
static void run_case(const char *what, cudaStream_t s0, cudaStream_t s1, const float *ref)
{
    for (int t = 0; t < THREADS; ++t)
        CU(cudaMemcpy(dC[t], hC0[t], (size_t)CALLS * M * N * 4, cudaMemcpyHostToDevice));
    CU(cudaDeviceSynchronize());
    unsigned long long v0[7], v1[7];
    vcounts(v0);
    const Counts c0 = counts(API_GEMMEX);
    pthread_t th[THREADS];
    Work w[THREADS] = {{0, s0, CUBLAS_STATUS_SUCCESS}, {1, s1, CUBLAS_STATUS_SUCCESS}};
    pthread_barrier_init(&start, NULL, THREADS);
    for (int t = 0; t < THREADS; ++t) pthread_create(&th[t], NULL, worker, &w[t]);
    for (int t = 0; t < THREADS; ++t) pthread_join(th[t], NULL);
    pthread_barrier_destroy(&start);
    vcounts(v1);
    const Counts c1 = counts(API_GEMMEX);
    const unsigned long long served = delta(&c0, &c1, SERVED_SKINNY);
    size_t wrong = 0, calls_wrong = 0;
    for (int t = 0; t < THREADS; ++t) {
        float *out = from_device(dC[t], (size_t)CALLS * M * N * 4);
        for (int i = 0; i < CALLS; ++i) {
            const size_t nd = ndiff(out + (size_t)i * M * N, ref + ((size_t)t * CALLS + i) * M * N, (size_t)M * N * 4, 4);
            wrong += nd, calls_wrong += nd != 0;
        }
        free(out);
    }
    const unsigned long long ver = v1[0] - v0[0], dif = v1[1] - v0[1], el = v1[2] - v0[2], skip = v1[4] - v0[4];
    printf("case %s: %d calls (%d threads x %d, TN %d x %d x %d, beta 0.3): %llu served by the skinny path, %llu verified, "
           "%llu differed (%llu elements), %llu not verified; the outputs: %zu calls with elements other than 12.9's (%zu "
           "elements)\n", what, THREADS * CALLS, THREADS, CALLS, M, N, K, served, ver, dif, el, skip, calls_wrong, wrong);
    CHECK(w[0].bad == CUBLAS_STATUS_SUCCESS && w[1].bad == CUBLAS_STATUS_SUCCESS, "%s: status %d, %d", what, w[0].bad,
          w[1].bad);
    CHECK(wrong == 0, "%s: %zu outputs, %zu elements, differ from 12.9's", what, calls_wrong, wrong);
    if (rt_counters)
        CHECK(served == (unsigned long long)THREADS * CALLS, "%s: %llu calls served by the skinny path, not %d", what,
              served, THREADS * CALLS);
    if (rt_verify) {
        CHECK(ver == (unsigned long long)THREADS * CALLS, "%s: %llu calls verified, not %d", what, ver, THREADS * CALLS);
        CHECK(dif == 0 && el == 0, "%s: verify mode reports %llu calls differing, in %llu elements: the outputs are 12.9's",
              what, dif, el);
        CHECK(skip == 0, "%s: %llu calls not verified", what, skip);
    }
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    for (int t = 0; t < THREADS; ++t)
        if (cublasCreate(&hv[t])) return 1;
    cublasHandle_t h12;
    if (c129.Create(&h12)) return 1;
    open_runtime();
    void *rt = dlopen("libvf_cublas_bf16.so", RTLD_NOW | RTLD_NOLOAD);
    if (rt) *(void **)&rt_verify = dlsym(rt, "vf_bf16_test_verify");
    if (!rt_verify) printf("note  a release runtime: verify mode's counters are not checked\n");
    /* the data: A shared, B per thread, C per call; 12.9's results computed first, one call at a time */
    uint16_t *hA = bf16_random((size_t)K * M, 0.05);
    dA = to_device(hA, (size_t)K * M * 2);
    float *ref = malloc((size_t)THREADS * CALLS * M * N * 4);
    float *d12;
    CU(cudaMalloc((void **)&d12, (size_t)CALLS * M * N * 4));
    for (int t = 0; t < THREADS; ++t) {
        uint16_t *hB = bf16_random((size_t)K * N, 1.0);
        dB[t] = to_device(hB, (size_t)K * N * 2);
        free(hB);
        hC0[t] = malloc((size_t)CALLS * M * N * 4);
        for (size_t i = 0; i < (size_t)CALLS * M * N; ++i) hC0[t][i] = (float)gauss();
        CU(cudaMalloc((void **)&dC[t], (size_t)CALLS * M * N * 4));
        CU(cudaMemcpy(d12, hC0[t], (size_t)CALLS * M * N * 4, cudaMemcpyHostToDevice));
        const float alpha = 1.0f, beta = 0.3f;
        for (int i = 0; i < CALLS; ++i)
            if (c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, CUDA_R_16BF, K, dB[t], CUDA_R_16BF, K,
                            &beta, d12 + (size_t)i * M * N, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT))
                return 1;
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(ref + (size_t)t * CALLS * M * N, d12, (size_t)CALLS * M * N * 4, cudaMemcpyDeviceToHost));
    }
    cudaStream_t shared, own[THREADS];
    CU(cudaStreamCreateWithFlags(&shared, cudaStreamNonBlocking));
    for (int t = 0; t < THREADS; ++t) CU(cudaStreamCreateWithFlags(&own[t], cudaStreamNonBlocking));
    run_case("one stream", shared, shared, ref);
    run_case("legacy stream", 0, 0, ref);
    run_case("per-thread stream", cudaStreamPerThread, cudaStreamPerThread, ref);
    run_case("a stream each", own[0], own[1], ref);
    free(hA);
    free(ref);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-verify-threads\n");
    return 0;
}
