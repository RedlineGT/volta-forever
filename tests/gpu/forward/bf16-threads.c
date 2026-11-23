/* Capture modes and threads (review section 5, G5 and G7).
 *
 * G5: in global, thread-local and relaxed mode, a capture holds a tensor-core-sized call (forwarded to
 * 12.9: its workspace would be a node of the graph) and a skinny call (served: one kernel node); both
 * replay with 12.9's bits, and the graph holds no memory node. An eager tensor-core call by a thread
 * on a stream it is not capturing, during its own capture in global mode, is served and leaves the
 * capture valid. Two forked streams in one capture.
 * G7: two threads call through one shared handle while a third switches its stream (and, unless
 * VF_BF16_CONTROL is set, its pointer mode: the callers' alpha and beta are in managed memory, valid
 * in both modes); every call is served and every result is the one computed alone. Prints
 * "ok bf16-threads". */
#include "bf16-common.h"
#include <pthread.h>

static cublasHandle_t hv, h12;
static const int M = 1536, N = 256, K = 1536, NS = 8; /* NS: the skinny call's columns */
static void *dA, *dB;
static float *ref, *ref12, *sk12; /* tensor-core eager; 12.9's for that shape; 12.9's skinny-shape */

static cublasStatus_t gemm(cublasHandle_t h, int n, void *C, const float *alpha, const float *beta)
{
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, n, K, alpha, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, beta, C,
                        CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static int same(void *dC, const float *want, int n)
{
    float *c = from_device(dC, (size_t)M * n * 4);
    const int ok = !memcmp(c, want, (size_t)M * n * 4);
    free(c);
    return ok;
}

static size_t mem_nodes(cudaGraph_t g)
{
    size_t n = 0, m = 0;
    cudaGraphGetNodes(g, NULL, &n);
    cudaGraphNode_t *nodes = calloc(n ? n : 1, sizeof *nodes);
    cudaGraphGetNodes(g, nodes, &n);
    for (size_t i = 0; i < n; ++i) {
        enum cudaGraphNodeType t;
        cudaGraphNodeGetType(nodes[i], &t);
        m += t == cudaGraphNodeTypeMemAlloc || t == cudaGraphNodeTypeMemFree;
    }
    free(nodes);
    return m;
}

static void capture_mode(enum cudaStreamCaptureMode mode, const char *name)
{
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cublasSetStream(hv, s);
    void *C, *Cs;
    CU(cudaMalloc(&C, (size_t)M * N * 4));
    CU(cudaMalloc(&Cs, (size_t)M * NS * 4));
    const float one = 1.0f, zero = 0.0f;
    cudaGraph_t g = NULL;
    cudaGraphExec_t e = NULL;
    const Counts c0 = counts(API_GEMMEX);
    CU(cudaStreamBeginCapture(s, mode));
    gemm(hv, N, C, &one, &zero);
    gemm(hv, NS, Cs, &one, &zero);
    const cudaError_t end = cudaStreamEndCapture(s, &g);
    const Counts c1 = counts(API_GEMMEX);
    CHECK(end == cudaSuccess, "capture in %s mode: %s", name, cudaGetErrorName(end));
    CHECK(!rt_counters || (delta(&c0, &c1, FWD(R_CAPTURED)) == 1 && delta(&c0, &c1, SERVED_SKINNY) == 1),
          "capture in %s mode: the tensor-core call not forwarded, or the skinny call not served", name);
    if (end == cudaSuccess) {
        CHECK(mem_nodes(g) == 0, "capture in %s mode: %zu memory nodes in the graph", name, mem_nodes(g));
        CU(cudaGraphInstantiate(&e, g, 0));
        CU(cudaGraphLaunch(e, s));
        CU(cudaStreamSynchronize(s));
        CHECK(same(C, ref12, N) && same(Cs, sk12, NS), "capture in %s mode: a replay is not 12.9's result", name);
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    }
    cudaFree(C);
    cudaFree(Cs);
    cudaStreamDestroy(s);
}

/* The calling thread captures on s1 in global mode and makes an eager call on s2. */
static void eager_during_own_capture(void)
{
    cudaStream_t s1, s2;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    void *C1, *C2;
    CU(cudaMalloc(&C1, (size_t)M * N * 4));
    CU(cudaMalloc(&C2, (size_t)M * N * 4));
    const float one = 1.0f, zero = 0.0f;
    cublasHandle_t h2;
    cublasCreate(&h2);
    cublasSetStream(hv, s1);
    cublasSetStream(h2, s2);
    const Counts c0 = counts(API_GEMMEX);
    CU(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    gemm(hv, N, C1, &one, &zero);                              /* captured: forwarded */
    const cublasStatus_t eager = gemm(h2, N, C2, &one, &zero); /* eager, on a stream not capturing */
    cudaGraph_t g = NULL;
    const cudaError_t end = cudaStreamEndCapture(s1, &g);
    cudaGetLastError();
    const Counts c1 = counts(API_GEMMEX);
    CHECK(end == cudaSuccess, "an eager call during the thread's own capture in global mode invalidated it: %s",
          cudaGetErrorName(end));
    CHECK(eager == CUBLAS_STATUS_SUCCESS && (!rt_counters || delta(&c0, &c1, SERVED_TC) == 1),
          "the eager call during the capture: status %d, not served", eager);
    cudaStreamSynchronize(s2);
    CHECK(same(C2, ref, N), "the eager call's result is wrong");
    if (g) {
        cudaGraphExec_t e;
        CU(cudaGraphInstantiate(&e, g, 0));
        CU(cudaGraphLaunch(e, s1));
        CU(cudaStreamSynchronize(s1));
        CHECK(same(C1, ref12, N), "the captured call's replay is not 12.9's result");
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    }
    cublasDestroy(h2);
    cudaFree(C1);
    cudaFree(C2);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}

/* One capture, forked onto a second stream. */
static void two_streams_one_capture(void)
{
    cudaStream_t s1, s2;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaEvent_t fork, join;
    cudaEventCreateWithFlags(&fork, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&join, cudaEventDisableTiming);
    void *C1, *C2;
    CU(cudaMalloc(&C1, (size_t)M * NS * 4));
    CU(cudaMalloc(&C2, (size_t)M * N * 4));
    const float one = 1.0f, zero = 0.0f;
    cublasHandle_t h2;
    cublasCreate(&h2);
    cublasSetStream(hv, s1);
    cublasSetStream(h2, s2);
    CU(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    CU(cudaEventRecord(fork, s1));
    CU(cudaStreamWaitEvent(s2, fork, 0));
    gemm(hv, NS, C1, &one, &zero); /* skinny, served */
    gemm(h2, N, C2, &one, &zero);  /* tensor-core size, forwarded */
    CU(cudaEventRecord(join, s2));
    CU(cudaStreamWaitEvent(s1, join, 0));
    cudaGraph_t g = NULL;
    const cudaError_t end = cudaStreamEndCapture(s1, &g);
    CHECK(end == cudaSuccess, "two streams in one capture: %s", cudaGetErrorName(end));
    if (g) {
        cudaGraphExec_t e;
        CU(cudaGraphInstantiate(&e, g, 0));
        for (int r = 0; r < 3; ++r) CU(cudaGraphLaunch(e, s1));
        CU(cudaStreamSynchronize(s1));
        CHECK(same(C1, sk12, NS) && same(C2, ref12, N), "two streams in one capture: a result is wrong");
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    }
    cublasDestroy(h2);
    cudaFree(C1);
    cudaFree(C2);
}

/* ---- G7 ---- */
static cudaStream_t streams[2];
static volatile int stop;
static float *scal; /* alpha, beta in managed memory */
static void *outs[2][24];
static int toggle_pmode;

static void *toggler(void *arg)
{
    (void)arg;
    cudaSetDevice(0);
    for (int i = 0; !stop; ++i) {
        cublasSetStream(hv, streams[i & 1]);
        if (toggle_pmode) cublasSetPointerMode(hv, (i & 2) ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST);
    }
    return NULL;
}

static void *caller(void *arg)
{
    const int t = (int)(intptr_t)arg;
    cudaSetDevice(0); /* the thread's context, as a program's threads set it */
    cudaFree(0);
    for (int i = 0; i < 24; ++i) gemm(hv, N, outs[t][i], scal, scal + 1);
    return NULL;
}

static void shared_handle(void)
{
    const char *control = getenv("VF_BF16_CONTROL");
    toggle_pmode = !control || !*control; /* under a control, the stream only: a wrong result, never a fault */
    CU(cudaMallocManaged((void **)&scal, 8, cudaMemAttachGlobal));
    scal[0] = 1.0f;
    scal[1] = 0.0f;
    cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
    for (int t = 0; t < 2; ++t)
        for (int i = 0; i < 24; ++i) {
            CU(cudaMalloc(&outs[t][i], (size_t)M * N * 4));
            CU(cudaMemset(outs[t][i], 0xff, (size_t)M * N * 4));
        }
    CU(cudaDeviceSynchronize());
    const Counts c0 = counts(API_GEMMEX);
    pthread_t th[3];
    stop = 0;
    pthread_create(&th[2], NULL, toggler, NULL);
    pthread_create(&th[0], NULL, caller, (void *)0);
    pthread_create(&th[1], NULL, caller, (void *)1);
    pthread_join(th[0], NULL);
    pthread_join(th[1], NULL);
    stop = 1;
    pthread_join(th[2], NULL);
    const cudaError_t sync = cudaDeviceSynchronize();
    const Counts c1 = counts(API_GEMMEX);
    CHECK(sync == cudaSuccess, "shared handle: %s", cudaGetErrorName(sync));
    CHECK(!rt_counters || delta(&c0, &c1, SERVED_TC) == 48, "shared handle: %llu of 48 calls served",
          delta(&c0, &c1, SERVED_TC));
    int wrong = 0;
    for (int t = 0; t < 2; ++t)
        for (int i = 0; i < 24; ++i) wrong += !same(outs[t][i], ref, N);
    CHECK(wrong == 0, "shared handle switched by a third thread: %d of 48 results differ from the one computed alone",
          wrong);
    printf("note  shared handle: the third thread switched the stream%s\n", toggle_pmode ? " and the pointer mode" : "");
    cublasSetPointerMode(hv, CUBLAS_POINTER_MODE_HOST);
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    if (cublasCreate(&hv) || c129.Create(&h12)) return 1;
    open_runtime();
    uint16_t *hA = bf16_random((size_t)M * K, 0.05), *hB = bf16_random((size_t)N * K, 1.0);
    dA = to_device(hA, (size_t)M * K * 2);
    dB = to_device(hB, (size_t)N * K * 2);
    void *C, *Cs;
    CU(cudaMalloc(&C, (size_t)M * N * 4));
    CU(cudaMalloc(&Cs, (size_t)M * NS * 4));
    const float one = 1.0f, zero = 0.0f;
    const Counts c0 = counts(API_GEMMEX);
    gemm(hv, N, C, &one, &zero);
    CU(cudaDeviceSynchronize());
    const Counts c1 = counts(API_GEMMEX);
    CHECK(!rt_counters || delta(&c0, &c1, SERVED_TC) == 1, "the reference call is not served by the tensor-core path");
    ref = from_device(C, (size_t)M * N * 4);
    c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, C,
                CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, NS, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, Cs,
                CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaDeviceSynchronize());
    ref12 = from_device(C, (size_t)M * N * 4);
    sk12 = from_device(Cs, (size_t)M * NS * 4);
    const char *only = getenv("VF_BF16_THREADS_ONLY");
    if (!only || strcmp(only, "g7")) {
        capture_mode(cudaStreamCaptureModeGlobal, "global");
        capture_mode(cudaStreamCaptureModeThreadLocal, "thread-local");
        capture_mode(cudaStreamCaptureModeRelaxed, "relaxed");
        eager_during_own_capture();
        two_streams_one_capture();
    }
    if (!only || strcmp(only, "g5")) shared_handle();
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-threads\n");
    return 0;
}
