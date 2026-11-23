/* What a program sees of the graph-memory nodes that a captured tensor-core call adds (review 3.2,
 * G3): each case runs twice in one process, once with the call through the variant (served by the
 * tensor-core path: an allocation node and a free node in the graph) and once with the same call given
 * to cuBLAS 12.9 directly (no memory node), and prints what every step returned. A characterisation:
 * its lines go into LIMITS.md. It fails only when a case breaks the program's capture with the
 * variant (the capture invalidated, or a wrong result) where 12.9 alone does not. Prints "ok bf16-graphs". */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cudaStream_t st;
static const int M = 1536, N = 512, K = 1536;
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

/* The result of the last run equals the reference result of the same path (eager). */
static int result_ok(void)
{
    const size_t bytes = (size_t)M * N * 4;
    float *a = from_device(dC, bytes), *b = from_device(dRef, bytes);
    const int ok = !memcmp(a, b, bytes);
    free(a);
    free(b);
    return ok;
}

static void reference(int v3)
{
    cudaMemset(dC, 0, (size_t)M * N * 4);
    gemm(v3);
    cudaStreamSynchronize(st);
    cudaMemcpy(dRef, dC, (size_t)M * N * 4, cudaMemcpyDeviceToDevice);
    cudaMemset(dC, 0, (size_t)M * N * 4);
}

static cudaGraph_t capture(int v3, cudaError_t *end)
{
    cudaGraph_t g = NULL;
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    gemm(v3);
    *end = cudaStreamEndCapture(st, &g);
    return g;
}

static int broken; /* a case in which the variant broke what 12.9 alone did not */

static void report(const char *name, int v3, const char *what)
{
    printf("case %-34s %-6s %s\n", name, v3 ? "V3" : "12.9", what);
}

static void second_instantiation(int v3)
{
    reference(v3);
    cudaError_t end;
    cudaGraph_t g = capture(v3, &end);
    cudaGraphExec_t e1 = NULL, e2 = NULL;
    const cudaError_t r1 = cudaGraphInstantiate(&e1, g, 0), r2 = cudaGraphInstantiate(&e2, g, 0);
    cudaGetLastError();
    char buf[256];
    snprintf(buf, sizeof buf, "capture %s, %zu memory nodes; 1st instantiation %s, 2nd %s", E(end), mem_nodes(g), E(r1),
             E(r2));
    report("second instantiation", v3, buf);
    if (e1) cudaGraphExecDestroy(e1);
    if (e2) cudaGraphExecDestroy(e2);
    cudaGraphDestroy(g);
}

static void child_graph(int v3)
{
    cudaError_t end;
    cudaGraph_t g = capture(v3, &end), parent;
    cudaGraphCreate(&parent, 0);
    cudaGraphNode_t node;
    const cudaError_t r = cudaGraphAddChildGraphNode(&node, parent, NULL, 0, g);
    cudaGetLastError();
    cudaGraphExec_t e = NULL;
    cudaError_t ri = cudaErrorUnknown, rl = cudaErrorUnknown;
    int ok = 0;
    if (r == cudaSuccess) {
        ri = cudaGraphInstantiate(&e, parent, 0);
        if (ri == cudaSuccess) {
            reference(v3);
            rl = cudaGraphLaunch(e, st);
            cudaStreamSynchronize(st);
            ok = result_ok();
        }
        cudaGetLastError();
    }
    char buf[256];
    snprintf(buf, sizeof buf, "add as a child graph %s; instantiate %s; launch %s; result %s", E(r),
             r == cudaSuccess ? E(ri) : "-", ri == cudaSuccess ? E(rl) : "-", ri == cudaSuccess ? (ok ? "right" : "WRONG") : "-");
    report("child graph", v3, buf);
    if (ri == cudaSuccess && rl == cudaSuccess && !ok && v3) broken = 1;
    if (e) cudaGraphExecDestroy(e);
    cudaGraphDestroy(parent);
    cudaGraphDestroy(g);
}

static void device_launch(int v3)
{
    cudaError_t end;
    cudaGraph_t g = capture(v3, &end);
    cudaGraphExec_t e = NULL;
    const cudaError_t r = cudaGraphInstantiateWithFlags(&e, g, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGetLastError();
    char buf[256];
    snprintf(buf, sizeof buf, "instantiate for device launch %s", E(r));
    report("device launch", v3, buf);
    if (e) cudaGraphExecDestroy(e);
    cudaGraphDestroy(g);
}

/* A conditional node (CUDA 12.4): its body graph is filled by a capture into it. */
static void conditional_body(int v3)
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
    char buf[512];
    if (r != cudaSuccess) {
        snprintf(buf, sizeof buf, "conditional node: %s", E(r));
        report("capture into a conditional body", v3, buf);
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
    int ok = 0;
    if (re == cudaSuccess) {
        ri = cudaGraphInstantiate(&e, parent, 0);
        if (ri == cudaSuccess) {
            reference(v3);
            rl = cudaGraphLaunch(e, st);
            cudaStreamSynchronize(st);
            ok = result_ok();
        }
        cudaGetLastError();
    }
    snprintf(buf, sizeof buf,
             "begin capture %s; capture status after the call %s; end capture %s; body memory nodes %zu; instantiate %s; "
             "launch %s; result %s",
             E(rb), cs == cudaStreamCaptureStatusActive ? "active" : cs == cudaStreamCaptureStatusInvalidated ? "INVALIDATED" : "none",
             E(re), mem_nodes(body), re == cudaSuccess ? E(ri) : "-", ri == cudaSuccess ? E(rl) : "-",
             ri == cudaSuccess && rl == cudaSuccess ? (ok ? "right" : "WRONG") : "-");
    report("capture into a conditional body", v3, buf);
    /* A graph that 12.9 alone lets the program instantiate and the variant does not: broken. */
    if (v3 && (cs == cudaStreamCaptureStatusInvalidated || re != cudaSuccess || ri != cudaSuccess ||
               (rl == cudaSuccess && !ok)))
        broken = 1;
    if (e) cudaGraphExecDestroy(e);
    cudaGraphDestroy(parent);
}

static void capture_to_graph(int v3)
{
    cudaGraph_t g;
    cudaGraphCreate(&g, 0);
    const cudaError_t rb = cudaStreamBeginCaptureToGraph(st, g, NULL, NULL, 0, cudaStreamCaptureModeGlobal);
    gemm(v3);
    cudaGraph_t out = NULL;
    const cudaError_t re = cudaStreamEndCapture(st, &out);
    cudaGraphExec_t e = NULL;
    const cudaError_t ri = re == cudaSuccess ? cudaGraphInstantiate(&e, g, 0) : cudaErrorUnknown;
    cudaError_t rl = cudaErrorUnknown;
    int ok = 0;
    if (ri == cudaSuccess) {
        reference(v3);
        rl = cudaGraphLaunch(e, st);
        cudaStreamSynchronize(st);
        ok = result_ok();
    }
    cudaGetLastError();
    char buf[256];
    snprintf(buf, sizeof buf, "begin %s; end %s; memory nodes %zu; instantiate %s; launch %s; result %s", E(rb), E(re),
             mem_nodes(g), E(ri), ri == cudaSuccess ? E(rl) : "-", rl == cudaSuccess ? (ok ? "right" : "WRONG") : "-");
    report("cudaStreamBeginCaptureToGraph", v3, buf);
    if (v3 && (re != cudaSuccess || (rl == cudaSuccess && !ok))) broken = 1;
    if (e) cudaGraphExecDestroy(e);
    cudaGraphDestroy(g);
}

/* llama.cpp's pattern: capture again every step and update the executable graph from the new one. */
static void exec_update(int v3)
{
    cudaError_t end;
    cudaGraph_t g1 = capture(v3, &end), g2;
    cudaGraphExec_t e = NULL;
    cudaGraphInstantiate(&e, g1, 0);
    g2 = capture(v3, &end);
    cudaGraphExecUpdateResultInfo info;
    memset(&info, 0, sizeof info);
    const cudaError_t ru = cudaGraphExecUpdate(e, g2, &info);
    cudaGetLastError();
    cudaError_t rl = cudaErrorUnknown;
    int ok = 0;
    if (ru == cudaSuccess) {
        reference(v3);
        rl = cudaGraphLaunch(e, st);
        cudaStreamSynchronize(st);
        ok = result_ok();
    }
    char buf[256];
    snprintf(buf, sizeof buf, "update from a recapture %s (result %d); launch %s; result %s", E(ru), (int)info.result,
             ru == cudaSuccess ? E(rl) : "-", rl == cudaSuccess ? (ok ? "right" : "WRONG") : "-");
    report("cudaGraphExecUpdate after a recapture", v3, buf);
    if (v3 && ru == cudaSuccess && rl == cudaSuccess && !ok) broken = 1;
    cudaGraphExecDestroy(e);
    cudaGraphDestroy(g1);
    cudaGraphDestroy(g2);
}

/* Memory of the graph's allocation nodes is mapped when the graph is first launched: with less than
 * 32 MiB free then, the launch has nothing to map. */
static void physical_backing(int v3)
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
    const cudaError_t ri = cudaGraphInstantiate(&e, g, 0);
    const cudaError_t rl = ri == cudaSuccess ? cudaGraphLaunch(e, st) : ri;
    const cudaError_t rs = cudaStreamSynchronize(st);
    cudaGetLastError();
    cudaFree(filler);
    char buf[256];
    snprintf(buf, sizeof buf, "%zu MiB free at instantiation and first launch; instantiate %s; launch %s; synchronize %s",
             left >> 20, E(ri), E(rl), E(rs));
    report("first launch with < 32 MiB free", v3, buf);
    if (e) cudaGraphExecDestroy(e);
    cudaGraphDestroy(g);
}

int main(void)
{
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
    CU(cudaMalloc(&dC, (size_t)M * N * 4));
    CU(cudaMalloc(&dRef, (size_t)M * N * 4));
    for (int v3 = 1; v3 >= 0; --v3) {
        second_instantiation(v3);
        child_graph(v3);
        device_launch(v3);
        conditional_body(v3);
        capture_to_graph(v3);
        exec_update(v3);
        physical_backing(v3);
    }
    const Counts c = counts(API_GEMMEX);
    printf("served by the tensor-core path: %llu calls; forwarded as captured: %llu; for a failed allocation: %llu\n",
           c.v[SERVED_TC], c.v[FWD(R_CAPTURED)], c.v[FWD(R_ALLOC)]);
    if (broken) {
        printf("FAIL  a case broke the program's capture or result with the variant (above)\n");
        return 1;
    }
    printf("ok bf16-graphs\n");
    return 0;
}
