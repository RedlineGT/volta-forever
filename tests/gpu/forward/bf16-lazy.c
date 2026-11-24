/* Lazy setup: nothing of the runtime's is made when a handle is created; the first call a path would serve
 * in a context makes what that path needs (the module for both, and for the tensor-core path its memory
 * pool and cuBLASLt handle), in relaxed capture mode, since that call may come inside a capture. One case
 * per process (tests/gpu/forward-bf16.sh runs each), chosen by VF_BF16_LAZY:
 *   global        the first served call is a skinny call captured in global mode
 *   threadlocal   the same in thread-local mode
 *   other-thread  the first served call is an eager skinny call while another thread captures in global mode
 *   tc            the first served call is an eager tensor-core call on another stream while this thread
 *                 captures in global mode
 * Each: right after cublasCreate the runtime has made nothing (its test counters: no module, pool or
 * cuBLASLt handle, the context not set up); the capture ends valid; after the call the runtime has made
 * exactly what the path needs; a skinny result (the replay's, or the eager call's) has 12.9's bits; a
 * tensor-core result is served and equals a second eager call's. The test runtime's controls must make
 * it fail: eager-setup (everything made with the handle, as before) in every case; setup-not-relaxed (the
 * setup in the thread's own capture mode) in the case tc, where creating the pool during the thread's
 * global capture invalidates that capture (on driver 580 loading the module inside a global or
 * thread-local capture does not). Prints "ok bf16-lazy". */
#include "bf16-common.h"
#include <pthread.h>

static int (*rt_state)(int *, int);
static const int M = 1536, K = 2048;

static void state(int *s)
{
    memset(s, 0, 5 * sizeof *s);
    if (rt_state && rt_state(s, 5) != 5) {
        printf("FAIL  the runtime's state has not 5 values\n");
        exit(1);
    }
}

static void check_state(const char *when, int kernels, int tc, int modules, int pools, int lt)
{
    int s[5];
    state(s);
    printf("state %s: context kernels %d, tensor-core %d; made in the process: %d module(s), %d pool(s), %d cuBLASLt "
           "handle(s)\n", when, s[0], s[1], s[2], s[3], s[4]);
    CHECK(s[0] == kernels && s[1] == tc && s[2] == modules && s[3] == pools && s[4] == lt,
          "%s: the runtime's state is (%d, %d; %d, %d, %d), not (%d, %d; %d, %d, %d)", when, s[0], s[1], s[2], s[3], s[4],
          kernels, tc, modules, pools, lt);
}

static size_t kernel_nodes(cudaGraph_t g, const char *prefix, size_t *total)
{
    size_t n = 0, hit = 0;
    cudaGraphGetNodes(g, NULL, &n);
    cudaGraphNode_t *nodes = calloc(n ? n : 1, sizeof *nodes);
    cudaGraphGetNodes(g, nodes, &n);
    for (size_t i = 0; i < n; ++i) {
        enum cudaGraphNodeType t;
        cudaGraphNodeGetType(nodes[i], &t);
        if (t != cudaGraphNodeTypeKernel) continue;
        CUDA_KERNEL_NODE_PARAMS p;
        memset(&p, 0, sizeof p);
        const char *name = NULL;
        if (cuGraphKernelNodeGetParams((CUgraphNode)nodes[i], &p) == CUDA_SUCCESS && p.func) cuFuncGetName(&name, p.func);
        hit += name && !strncmp(name, prefix, strlen(prefix));
    }
    free(nodes);
    *total = n;
    return hit;
}

static cublasStatus_t gemm(cublasHandle_t h, int n, const void *A, const void *B, void *C, int direct)
{
    const float one = 1.0f, zero = 0.0f;
    return (direct ? c129.GemmEx : cublasGemmEx)(h, CUBLAS_OP_T, CUBLAS_OP_N, M, n, K, &one, A, CUDA_R_16BF, K, B,
                                                 CUDA_R_16BF, K, &zero, C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F,
                                                 CUBLAS_GEMM_DEFAULT);
}

static struct {
    cudaStream_t s;
    void *buf;
    volatile int capturing, release;
    cudaError_t end;
} other;

static void *other_capturer(void *arg)
{
    (void)arg;
    cudaGraph_t g = NULL;
    if (cudaStreamBeginCapture(other.s, cudaStreamCaptureModeGlobal) != cudaSuccess) {
        other.end = cudaErrorUnknown;
        __atomic_store_n(&other.capturing, 1, __ATOMIC_RELEASE);
        return NULL;
    }
    cudaMemsetAsync(other.buf, 0, 4096, other.s);
    __atomic_store_n(&other.capturing, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&other.release, __ATOMIC_ACQUIRE)) sched_yield();
    other.end = cudaStreamEndCapture(other.s, &g);
    if (g) cudaGraphDestroy(g);
    return NULL;
}

int main(void)
{
    const char *mode = getenv("VF_BF16_LAZY");
    if (!mode) mode = "global";
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    uint16_t *hA = bf16_random((size_t)M * K, 0.05), *hB = bf16_random((size_t)256 * K, 1.0);
    void *A = to_device(hA, (size_t)M * K * 2), *B = to_device(hB, (size_t)256 * K * 2), *C, *C2, *C12;
    CU(cudaMalloc(&C, (size_t)M * 256 * 4));
    CU(cudaMalloc(&C2, (size_t)M * 256 * 4));
    CU(cudaMalloc(&C12, (size_t)M * 256 * 4));
    cublasHandle_t h12;
    if (c129.Create(&h12)) return 1;
    cudaStream_t s, s2;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    c129.SetStream(h12, s);
    cublasHandle_t hv;
    if (cublasCreate(&hv)) {
        printf("FAIL  cublasCreate\n");
        return 1;
    }
    void *rt = dlopen("libvf_cublas_bf16.so", RTLD_NOW | RTLD_NOLOAD);
    if (rt) *(void **)&rt_state = dlsym(rt, "vf_bf16_test_state");
    open_runtime();
    if (!rt_state) {
        printf("FAIL  not the test build of the runtime\n");
        return 1;
    }
    cublasSetStream(hv, s);
    printf("case %s\n", mode);
    check_state("after cublasCreate", 0, 0, 0, 0, 0);
    const int NS = 8, NT = 256;
    const size_t cbs = (size_t)M * NS * 4, cbt = (size_t)M * NT * 4;
    if (!strcmp(mode, "global") || !strcmp(mode, "threadlocal")) {
        const int global = !strcmp(mode, "global");
        cudaGraph_t g = NULL;
        cudaGraphExec_t e;
        const Counts k0 = counts(API_GEMMEX);
        CU(cudaStreamBeginCapture(s, global ? cudaStreamCaptureModeGlobal : cudaStreamCaptureModeThreadLocal));
        const cublasStatus_t st = gemm(hv, NS, A, B, C, 0);
        const cudaError_t end = cudaStreamEndCapture(s, &g);
        const Counts k1 = counts(API_GEMMEX);
        CHECK(st == CUBLAS_STATUS_SUCCESS && end == cudaSuccess, "the first served call captured in %s mode: status %d, "
              "end of capture %s", mode, (int)st, cudaGetErrorName(end));
        CHECK(delta(&k0, &k1, SERVED_SKINNY) == 1, "the captured call was not served by the skinny path");
        check_state("after the first served call", 1, 0, 1, 0, 0);
        if (end == cudaSuccess) {
            size_t total = 0;
            const size_t sk = kernel_nodes(g, "vf_bf16_skinny", &total);
            printf("graph: %zu node(s), %zu kernel node(s) of the runtime's skinny kernels\n", total, sk);
            CHECK(total == 1 && sk == 1, "the graph holds %zu nodes, %zu of them skinny kernels (want 1 and 1)", total, sk);
            CU(cudaGraphInstantiate(&e, g, 0));
            CU(cudaGraphLaunch(e, s));
            gemm(h12, NS, A, B, C12, 1);
            CU(cudaStreamSynchronize(s));
            float *r = from_device(C, cbs), *r12 = from_device(C12, cbs);
            CHECK(!memcmp(r, r12, cbs), "the replay has not 12.9's bits");
            free(r);
            free(r12);
            cudaGraphExecDestroy(e);
            cudaGraphDestroy(g);
        }
    } else if (!strcmp(mode, "other-thread")) {
        other.s = s2;
        CU(cudaMalloc(&other.buf, 4096));
        pthread_t t;
        pthread_create(&t, NULL, other_capturer, NULL);
        while (!__atomic_load_n(&other.capturing, __ATOMIC_ACQUIRE)) sched_yield();
        const Counts k0 = counts(API_GEMMEX);
        const cublasStatus_t st = gemm(hv, NS, A, B, C, 0); /* eager, during the other thread's capture */
        const Counts k1 = counts(API_GEMMEX);
        __atomic_store_n(&other.release, 1, __ATOMIC_RELEASE);
        pthread_join(t, NULL);
        CHECK(st == CUBLAS_STATUS_SUCCESS && other.end == cudaSuccess, "the first served call during another thread's "
              "capture in global mode: status %d, end of that capture %s", (int)st, cudaGetErrorName(other.end));
        CHECK(delta(&k0, &k1, SERVED_SKINNY) == 1, "the eager call was not served by the skinny path");
        check_state("after the first served call", 1, 0, 1, 0, 0);
        gemm(h12, NS, A, B, C12, 1);
        CU(cudaDeviceSynchronize());
        float *r = from_device(C, cbs), *r12 = from_device(C12, cbs);
        CHECK(!memcmp(r, r12, cbs), "the eager call has not 12.9's bits");
        free(r);
        free(r12);
    } else if (!strcmp(mode, "tc")) {
        cublasSetStream(hv, s2);
        void *buf;
        CU(cudaMalloc(&buf, 4096));
        cudaGraph_t g = NULL;
        const Counts k0 = counts(API_GEMMEX);
        CU(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        cudaMemsetAsync(buf, 0, 4096, s);
        const cublasStatus_t st = gemm(hv, NT, A, B, C, 0); /* eager, on s2, during this thread's capture of s */
        const cudaError_t end = cudaStreamEndCapture(s, &g);
        const Counts k1 = counts(API_GEMMEX);
        CHECK(st == CUBLAS_STATUS_SUCCESS && end == cudaSuccess, "the first served call, eager, during this thread's "
              "capture in global mode: status %d, end of the capture %s", (int)st, cudaGetErrorName(end));
        CHECK(delta(&k0, &k1, SERVED_TC) == 1, "the eager call was not served by the tensor-core path");
        check_state("after the first served call", 1, 1, 1, 1, 1);
        if (g) cudaGraphDestroy(g);
        CU(cudaStreamSynchronize(s2));
        gemm(hv, NT, A, B, C2, 0);
        CU(cudaStreamSynchronize(s2));
        check_state("after a second call", 1, 1, 1, 1, 1);
        float *r = from_device(C, cbt), *r2 = from_device(C2, cbt);
        CHECK(!memcmp(r, r2, cbt), "the first call's result differs from a second eager call's");
        free(r);
        free(r2);
    } else {
        printf("FAIL  unknown case %s\n", mode);
        return 2;
    }
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-lazy\n");
    return 0;
}
