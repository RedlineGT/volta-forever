/* Graph captures (review 3.2, G3). The runtime gives every tensor-core call made during a capture to
 * cuBLAS 12.9 (no allocation, so no memory node in the program's graph), and the skinny path serves
 * captured calls with one kernel node of its own module (VF_BF16_GRAPHS_N=8): either way, with the
 * variant a program's graph must behave as with 12.9 alone. Each case runs twice in one process, once with the
 * call through the variant and once with the same call given to 12.9 directly, and prints what every
 * step returned. It fails when the variant's line differs from 12.9's (a step's status, a memory node,
 * or a replay's bits against 12.9's eager result), when the same call made eagerly is not served by
 * the path of its size, or when a captured call is not forwarded as captured (tensor-core size) or
 * not served by the skinny path (n = 8). VF_BF16_CONTROL=capture-tc (captured tensor-core calls served,
 * with allocation nodes) makes the first fail.
 * Prints "ok bf16-graphs". */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cudaStream_t st;
static const int M = 1536, K = 1536;
static int N = 512; /* VF_BF16_GRAPHS_N: 512 (tensor-core size, forwarded in captures) or 8 (skinny, served) */
static size_t CB;
static void *dA, *dB, *dC, *dRef;

static void gemm(int v3)
{
    const float one = 1.0f, zero = 0.0f;
    if (v3)
        cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, dC,
                     CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    else
        c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, dA, CUDA_R_16BF, K, dB, CUDA_R_16BF, K, &zero, dC,
                    CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static const char *E(cudaError_t e) { return e == cudaSuccess ? "ok" : cudaGetErrorName(e); }

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

/* Launches e on the stream (the output cleared first, on the same stream) and compares the output with
 * 12.9's eager result: "right", "WRONG", or "-" when the launch failed. */
static const char *launch_and_check(cudaGraphExec_t e, cudaError_t *rl)
{
    cudaMemsetAsync(dC, 0, CB, st);
    *rl = cudaGraphLaunch(e, st);
    const cudaError_t rs = cudaStreamSynchronize(st);
    cudaGetLastError();
    if (*rl != cudaSuccess || rs != cudaSuccess) return "-";
    float *a = from_device(dC, CB), *b = from_device(dRef, CB);
    const int ok = !memcmp(a, b, CB);
    free(a);
    free(b);
    return ok ? "right" : "WRONG";
}

static cudaGraph_t capture(int v3, cudaError_t *end)
{
    cudaGraph_t g = NULL;
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    gemm(v3);
    *end = cudaStreamEndCapture(st, &g);
    cudaGetLastError();
    return g;
}

/* What a case compares (out) and what it prints besides (extra). */
typedef struct {
    char out[512];
    char extra[64];
} Line;

static void second_instantiation(int v3, Line *w)
{
    cudaError_t end, rl = cudaErrorUnknown;
    cudaGraph_t g = capture(v3, &end);
    cudaGraphExec_t e1 = NULL, e2 = NULL;
    const cudaError_t r1 = end == cudaSuccess ? cudaGraphInstantiate(&e1, g, 0) : cudaErrorUnknown;
    const cudaError_t r2 = end == cudaSuccess ? cudaGraphInstantiate(&e2, g, 0) : cudaErrorUnknown;
    cudaGetLastError();
    const char *res = r2 == cudaSuccess ? launch_and_check(e2, &rl) : "-";
    snprintf(w->out, sizeof w->out, "capture %s, %zu memory nodes; 1st instantiation %s, 2nd %s; launch of the 2nd %s; result %s",
             E(end), g ? mem_nodes(g) : 0, E(r1), E(r2), r2 == cudaSuccess ? E(rl) : "-", res);
    if (e1) cudaGraphExecDestroy(e1);
    if (e2) cudaGraphExecDestroy(e2);
    if (g) cudaGraphDestroy(g);
}

static void child_graph(int v3, Line *w)
{
    cudaError_t end, ri = cudaErrorUnknown, rl = cudaErrorUnknown;
    cudaGraph_t g = capture(v3, &end), parent;
    cudaGraphCreate(&parent, 0);
    cudaGraphNode_t node;
    const cudaError_t r = end == cudaSuccess ? cudaGraphAddChildGraphNode(&node, parent, NULL, 0, g) : cudaErrorUnknown;
    cudaGetLastError();
    cudaGraphExec_t e = NULL;
    const char *res = "-";
    if (r == cudaSuccess) {
        ri = cudaGraphInstantiate(&e, parent, 0);
        cudaGetLastError();
        if (ri == cudaSuccess) res = launch_and_check(e, &rl);
    }
    snprintf(w->out, sizeof w->out, "add as a child graph %s; instantiate %s; launch %s; result %s", E(r),
             r == cudaSuccess ? E(ri) : "-", ri == cudaSuccess ? E(rl) : "-", res);
    if (e) cudaGraphExecDestroy(e);
    cudaGraphDestroy(parent);
    if (g) cudaGraphDestroy(g);
}

/* Instantiated for device launch, uploaded, and launched from the host (which a device graph allows). */
static void device_launch(int v3, Line *w)
{
    cudaError_t end, ru = cudaErrorUnknown, rl = cudaErrorUnknown;
    cudaGraph_t g = capture(v3, &end);
    cudaGraphExec_t e = NULL;
    const cudaError_t r =
        end == cudaSuccess ? cudaGraphInstantiateWithFlags(&e, g, cudaGraphInstantiateFlagDeviceLaunch) : cudaErrorUnknown;
    cudaGetLastError();
    const char *res = "-";
    if (r == cudaSuccess) {
        ru = cudaGraphUpload(e, st);
        cudaGetLastError();
        if (ru == cudaSuccess) res = launch_and_check(e, &rl);
    }
    snprintf(w->out, sizeof w->out, "instantiate for device launch %s; upload %s; launch from the host %s; result %s", E(r),
             r == cudaSuccess ? E(ru) : "-", ru == cudaSuccess ? E(rl) : "-", res);
    if (e) cudaGraphExecDestroy(e);
    if (g) cudaGraphDestroy(g);
}

/* A conditional node (CUDA 12.4): its body graph is filled by a capture into it. */
static void conditional_body(int v3, Line *w)
{
    cudaGraph_t parent;
    cudaGraphCreate(&parent, 0);
    cudaGraphConditionalHandle handle;
    cudaError_t r = cudaGraphConditionalHandleCreate(&handle, parent, 1, cudaGraphCondAssignDefault);
    struct cudaGraphNodeParams p;
    memset(&p, 0, sizeof p);
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = handle;
    p.conditional.type = cudaGraphCondTypeIf;
    p.conditional.size = 1;
    cudaGraphNode_t node;
    if (r == cudaSuccess) r = cudaGraphAddNode(&node, parent, NULL, NULL, 0, &p);
    if (r != cudaSuccess) {
        snprintf(w->out, sizeof w->out, "conditional node: %s", E(r));
        cudaGetLastError();
        cudaGraphDestroy(parent);
        return;
    }
    cudaGraph_t body = p.conditional.phGraph_out[0];
    const cudaError_t rb = cudaStreamBeginCaptureToGraph(st, body, NULL, NULL, 0, cudaStreamCaptureModeGlobal);
    enum cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    gemm(v3);
    cudaStreamIsCapturing(st, &cs);
    cudaGraph_t out = NULL;
    const cudaError_t re = cudaStreamEndCapture(st, &out);
    cudaGetLastError();
    cudaGraphExec_t e = NULL;
    cudaError_t ri = cudaErrorUnknown, rl = cudaErrorUnknown;
    const char *res = "-";
    if (re == cudaSuccess) {
        ri = cudaGraphInstantiate(&e, parent, 0);
        cudaGetLastError();
        if (ri == cudaSuccess) res = launch_and_check(e, &rl);
    }
    snprintf(w->out, sizeof w->out,
             "begin capture %s; capture status after the call %s; end capture %s; body memory nodes %zu; instantiate %s; "
             "launch %s; result %s",
             E(rb), cs == cudaStreamCaptureStatusActive ? "active" : cs == cudaStreamCaptureStatusInvalidated ? "INVALIDATED" : "none",
             E(re), mem_nodes(body), re == cudaSuccess ? E(ri) : "-", ri == cudaSuccess ? E(rl) : "-", res);
    if (e) cudaGraphExecDestroy(e);
    cudaGraphDestroy(parent);
}

static void capture_to_graph(int v3, Line *w)
{
    cudaGraph_t g;
    cudaGraphCreate(&g, 0);
    const cudaError_t rb = cudaStreamBeginCaptureToGraph(st, g, NULL, NULL, 0, cudaStreamCaptureModeGlobal);
    gemm(v3);
    cudaGraph_t out = NULL;
    const cudaError_t re = cudaStreamEndCapture(st, &out);
    cudaGetLastError();
    cudaGraphExec_t e = NULL;
    const cudaError_t ri = re == cudaSuccess ? cudaGraphInstantiate(&e, g, 0) : cudaErrorUnknown;
    cudaGetLastError();
    cudaError_t rl = cudaErrorUnknown;
    const char *res = ri == cudaSuccess ? launch_and_check(e, &rl) : "-";
    snprintf(w->out, sizeof w->out, "begin %s; end %s; memory nodes %zu; instantiate %s; launch %s; result %s", E(rb), E(re),
             mem_nodes(g), re == cudaSuccess ? E(ri) : "-", ri == cudaSuccess ? E(rl) : "-", res);
    if (e) cudaGraphExecDestroy(e);
    cudaGraphDestroy(g);
}

/* llama.cpp's pattern: capture again every step and update the executable graph from the new one. */
static void exec_update(int v3, Line *w)
{
    cudaError_t end1, end2, rl = cudaErrorUnknown;
    cudaGraph_t g1 = capture(v3, &end1), g2 = NULL;
    cudaGraphExec_t e = NULL;
    const cudaError_t ri = end1 == cudaSuccess ? cudaGraphInstantiate(&e, g1, 0) : cudaErrorUnknown;
    cudaGetLastError();
    g2 = capture(v3, &end2);
    cudaGraphExecUpdateResultInfo info;
    memset(&info, 0, sizeof info);
    const cudaError_t ru = ri == cudaSuccess && end2 == cudaSuccess ? cudaGraphExecUpdate(e, g2, &info) : cudaErrorUnknown;
    cudaGetLastError();
    const char *res = ru == cudaSuccess ? launch_and_check(e, &rl) : "-";
    snprintf(w->out, sizeof w->out, "instantiate %s; update from a recapture %s (result %d); launch %s; result %s", E(ri),
             E(ru), (int)info.result, ru == cudaSuccess ? E(rl) : "-", res);
    if (e) cudaGraphExecDestroy(e);
    if (g1) cudaGraphDestroy(g1);
    if (g2) cudaGraphDestroy(g2);
}

/* Memory of a graph's allocation nodes is mapped when the graph is first launched: with less than
 * 32 MiB free then, such a launch has nothing to map. */
static void physical_backing(int v3, Line *w)
{
    cudaError_t end;
    cudaGraph_t g = capture(v3, &end);
    cudaGraphExec_t e = NULL;
    cudaDeviceGraphMemTrim(0); /* nothing kept from the graphs before */
    size_t free_b = 0, total = 0;
    cudaMemGetInfo(&free_b, &total);
    void *filler = NULL;
    size_t want = free_b > (24u << 20) ? free_b - (16u << 20) : 0;
    while (want > 0 && cudaMalloc(&filler, want) != cudaSuccess) {
        cudaGetLastError();
        want -= 8u << 20;
    }
    size_t left = 0;
    cudaMemGetInfo(&left, &total);
    const cudaError_t ri = end == cudaSuccess ? cudaGraphInstantiate(&e, g, 0) : cudaErrorUnknown;
    cudaGetLastError();
    cudaError_t rl = cudaErrorUnknown;
    const char *res = ri == cudaSuccess ? launch_and_check(e, &rl) : "-";
    cudaFree(filler);
    snprintf(w->out, sizeof w->out, "instantiate %s; launch %s; result %s", E(ri), ri == cudaSuccess ? E(rl) : "-", res);
    snprintf(w->extra, sizeof w->extra, " (%zu MiB free)", left >> 20);
    if (e) cudaGraphExecDestroy(e);
    if (g) cudaGraphDestroy(g);
}

int main(void)
{
    const char *nv = getenv("VF_BF16_GRAPHS_N");
    if (nv && atoi(nv) > 0) N = atoi(nv);
    CB = (size_t)M * N * 4;
    const int skinny = N <= 16;
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    if (cublasCreate(&hv) || c129.Create(&h12)) return 1;
    open_runtime();
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    uint16_t *hA = bf16_random((size_t)M * K, 0.05), *hB = bf16_random((size_t)N * K, 1.0);
    dA = to_device(hA, (size_t)M * K * 2);
    dB = to_device(hB, (size_t)N * K * 2);
    CU(cudaMalloc(&dC, CB));
    CU(cudaMalloc(&dRef, CB));
    int broken = 0;
    /* 12.9's eager result: what a captured call must replay. */
    gemm(0);
    CU(cudaStreamSynchronize(st));
    CU(cudaMemcpy(dRef, dC, CB, cudaMemcpyDeviceToDevice));
    /* The same call made eagerly through the variant: served by the path of its size. */
    const Counts k0 = counts(API_GEMMEX);
    gemm(1);
    CU(cudaStreamSynchronize(st));
    const Counts k1 = counts(API_GEMMEX);
    if (rt_counters && delta(&k0, &k1, skinny ? SERVED_SKINNY : SERVED_TC) != 1) {
        printf("FAIL  the call made eagerly through the variant is not served by the %s path\n",
               skinny ? "skinny" : "tensor-core");
        broken = 1;
    }
    static const struct {
        const char *name;
        void (*run)(int, Line *);
    } cases[] = {
        {"second instantiation", second_instantiation},
        {"child graph", child_graph},
        {"device launch", device_launch},
        {"capture into a conditional body", conditional_body},
        {"cudaStreamBeginCaptureToGraph", capture_to_graph},
        {"cudaGraphExecUpdate after a recapture", exec_update},
        {"first launch with < 32 MiB free", physical_backing},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; ++i) {
        Line a, b;
        memset(&a, 0, sizeof a);
        memset(&b, 0, sizeof b);
        cases[i].run(1, &a);
        cases[i].run(0, &b);
        printf("case %-38s V3     %s%s\n", cases[i].name, a.out, a.extra);
        printf("case %-38s 12.9   %s%s\n", cases[i].name, b.out, b.extra);
        if (strcmp(a.out, b.out)) {
            printf("FAIL  %s: the variant's line differs from 12.9's\n", cases[i].name);
            broken = 1;
        }
    }
    const Counts k2 = counts(API_GEMMEX);
    const unsigned long long calls = delta(&k1, &k2, 0), fwd = delta(&k1, &k2, FWD(R_CAPTURED)),
                             tc = delta(&k1, &k2, SERVED_TC), sk = delta(&k1, &k2, SERVED_SKINNY);
    printf("n = %d: captured calls through the variant: %llu; forwarded as captured: %llu; served by the tensor-core "
           "path: %llu, by the skinny path: %llu\n",
           N, calls, fwd, tc, sk);
    if (rt_counters && !skinny && (calls == 0 || fwd != calls || tc != 0)) {
        printf("FAIL  a captured call through the variant was not forwarded as captured\n");
        broken = 1;
    }
    if (rt_counters && skinny && (calls == 0 || sk != calls)) {
        printf("FAIL  a captured skinny-sized call through the variant was not served by the skinny path\n");
        broken = 1;
    }
    if (broken) {
        printf("FAIL  with the variant, a program's graph does not behave as with cuBLAS 12.9 alone (above)\n");
        return 1;
    }
    printf("ok bf16-graphs\n");
    return 0;
}
