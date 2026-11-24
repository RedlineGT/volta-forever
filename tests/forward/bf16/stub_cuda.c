/* A stand-in for libcuda.so.1 for the host tests of the bf16 runtime (tests/forward/bf16.sh): the
 * driver functions the runtime binds, which record what they are asked (an event log shared with the
 * cuBLAS stand-ins), keep the state the tests set (current context, capturing streams, device), run
 * user-object destructors when a test destroys a graph, and fail on demand. No GPU. */
#include "stub.h"
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EXPORT __attribute__((visibility("default")))

/* ---- the event log ---- */
static char *events;
static size_t events_len, events_cap;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

EXPORT void stub_event(const char *fmt, ...)
{
    char line[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    pthread_mutex_lock(&mu);
    const size_t n = strlen(line);
    if (events_len + n + 2 > events_cap) {
        events_cap = (events_len + n + 2) * 2;
        events = realloc(events, events_cap);
    }
    memcpy(events + events_len, line, n);
    events[events_len + n] = '\n';
    events_len += n + 1;
    events[events_len] = 0;
    pthread_mutex_unlock(&mu);
}

/* The log so far, then cleared (the caller frees it). */
EXPORT char *stub_take_events(void)
{
    pthread_mutex_lock(&mu);
    char *e = events ? events : strdup("");
    events = NULL;
    events_len = events_cap = 0;
    pthread_mutex_unlock(&mu);
    return e;
}

/* ---- failures on demand: stub_fail("cuMemAllocFromPoolAsync", 1) makes the next call fail ---- */
static char fail_name[64];
static int fail_left;
EXPORT void stub_fail(const char *name, int count)
{
    snprintf(fail_name, sizeof fail_name, "%s", name);
    fail_left = count;
}
static int failing(const char *name)
{
    if (fail_left > 0 && !strcmp(name, fail_name)) {
        --fail_left;
        return 1;
    }
    return 0;
}
EXPORT int stub_failing(const char *name) { return failing(name); }

/* ---- contexts, devices ---- */
typedef struct {
    unsigned long long id;
    int device;
} Ctx;
static Ctx ctxs[8] = {{101, 0}, {102, 1}, {103, 0}, {104, 0}, {105, 0}, {106, 0}, {107, 0}, {108, 0}};
static __thread int current = 0; /* index + 1; 0: the first context */
static int cc_major[2] = {7, 7}, cc_minor[2] = {0, 0};
static int module_loads;

EXPORT void stub_set_context(int index) { current = index + 1; }
EXPORT void stub_set_device(int device, int major, int minor)
{
    cc_major[device] = major;
    cc_minor[device] = minor;
}
EXPORT int stub_module_loads(void) { return module_loads; }
static Ctx *cur(void) { return &ctxs[current ? current - 1 : 0]; }

EXPORT CUresult cuCtxGetCurrent(CUcontext *pctx)
{
    *pctx = (CUcontext)cur();
    return CUDA_SUCCESS;
}
EXPORT CUresult cuCtxGetId(CUcontext ctx, unsigned long long *id)
{
    *id = ((Ctx *)ctx)->id;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuCtxGetDevice(CUdevice *d)
{
    *d = cur()->device;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute a, CUdevice d)
{
    switch (a) {
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *pi = cc_major[d]; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *pi = cc_minor[d]; break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *pi = 80; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: *pi = 48 * 1024; break;
    default: return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

/* Streams: a test may give a stream to another context, or start a capture on it. */
typedef struct {
    CUstream s;
    int ctx; /* index + 1; 0: the current context */
    unsigned long long capture;
    CUgraph graph;
} StreamInfo;
static StreamInfo streams[64];
static StreamInfo *stream_info(CUstream s)
{
    for (int i = 0; i < 64; ++i)
        if (streams[i].s == s) return &streams[i];
    return NULL;
}
EXPORT void stub_set_stream(CUstream s, int ctx_index, unsigned long long capture, CUgraph graph)
{
    StreamInfo *x = stream_info(s);
    for (int i = 0; !x && i < 64; ++i)
        if (!streams[i].s) x = &streams[i];
    x->s = s;
    x->ctx = ctx_index + 1;
    x->capture = capture;
    x->graph = graph;
}
EXPORT CUresult cuStreamGetCtx(CUstream s, CUcontext *pctx)
{
    StreamInfo *x = stream_info(s);
    *pctx = (CUcontext)(x && x->ctx ? &ctxs[x->ctx - 1] : cur());
    return CUDA_SUCCESS;
}
EXPORT CUresult cuStreamGetCaptureInfo_v3(CUstream s, CUstreamCaptureStatus *st, cuuint64_t *id, CUgraph *graph,
                                          const CUgraphNode **deps, const CUgraphEdgeData **edges, size_t *ndeps)
{
    StreamInfo *x = stream_info(s);
    const int capturing = x && x->capture;
    *st = capturing ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
    if (id) *id = capturing ? x->capture : 0;
    if (graph) *graph = capturing ? x->graph : NULL;
    stub_event("captureinfo stream=%p", (void *)s);
    return CUDA_SUCCESS;
}

EXPORT CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode)
{
    static __thread CUstreamCaptureMode m = CU_STREAM_CAPTURE_MODE_GLOBAL;
    const CUstreamCaptureMode old = m;
    m = *mode;
    *mode = old;
    stub_event("capturemode %d", (int)m);
    return CUDA_SUCCESS;
}

/* ---- modules ---- */
typedef struct {
    char name[64];
} Func;
static Func funcs[512];
static int nfuncs;
EXPORT CUresult cuModuleLoadData(CUmodule *m, const void *image)
{
    if (failing("cuModuleLoadData")) return CUDA_ERROR_NOT_SUPPORTED;
    /* The image must be a fat binary (magic 0xBA55ED50). */
    if (*(const unsigned *)image != 0xBA55ED50u) return CUDA_ERROR_INVALID_VALUE;
    ++module_loads;
    stub_event("module load ctx=%llu", cur()->id);
    *m = (CUmodule)(uintptr_t)(0x1000 + module_loads);
    return CUDA_SUCCESS;
}
EXPORT CUresult cuModuleGetFunction(CUfunction *f, CUmodule m, const char *name)
{
    if (nfuncs >= 512) return CUDA_ERROR_OUT_OF_MEMORY;
    snprintf(funcs[nfuncs].name, sizeof funcs[nfuncs].name, "%s", name);
    *f = (CUfunction)&funcs[nfuncs++];
    return CUDA_SUCCESS;
}
EXPORT CUresult cuFuncLoad(CUfunction f) { return failing("cuFuncLoad") ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS; }
EXPORT CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute a, CUfunction f)
{
    switch (a) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = 256; break;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: *pi = 1024; break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS: *pi = 64; break;
    default: return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

/* A pause after each launch of one function (stub_pause_launch("vf_bf16_copy", 2000): 2 ms), set before the threads
 * that launch it start, so that another thread's launches can come in between where nothing holds them apart. */
static char pause_name[64];
static int pause_us;
EXPORT void stub_pause_launch(const char *name, int us)
{
    snprintf(pause_name, sizeof pause_name, "%s", name ? name : "");
    pause_us = us;
}

/* Launches: the function's name, the configuration, the stream, and the argument structure's first
 * STUB_ARGS bytes in hexadecimal (the kernels take one structure by value). */
EXPORT CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                               unsigned bz, unsigned smem, CUstream s, void **params, void **extra)
{
    const int fail = failing(((Func *)f)->name) || failing("cuLaunchKernel");
    char hex[2 * STUB_ARGS + 1];
    const unsigned char *a = params[0];
    for (int i = 0; i < STUB_ARGS; ++i) sprintf(hex + 2 * i, "%02x", a[i]);
    stub_event("launch %s grid=%u,%u,%u block=%u smem=%u stream=%p%s args=%s", ((Func *)f)->name, gx, gy, gz, bx, smem,
               (void *)s, fail ? " FAILED" : "", hex);
    if (pause_us > 0 && !strcmp(((Func *)f)->name, pause_name)) {
        const struct timespec t = {pause_us / 1000000, (long)(pause_us % 1000000) * 1000};
        nanosleep(&t, NULL);
    }
    return fail ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS;
}

/* ---- memory ---- */
static unsigned long long next_ptr = 0x700000000000ull;
EXPORT CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *props)
{
    if (failing("cuMemPoolCreate")) return CUDA_ERROR_OUT_OF_MEMORY;
    static int n;
    *pool = (CUmemoryPool)(uintptr_t)(0x2000 + ++n);
    stub_event("pool create device=%d", props->location.id);
    return CUDA_SUCCESS;
}
EXPORT CUresult cuMemPoolSetAttribute(CUmemoryPool p, CUmemPool_attribute a, void *v) { return CUDA_SUCCESS; }
EXPORT CUresult cuMemAllocFromPoolAsync(CUdeviceptr *d, size_t bytes, CUmemoryPool p, CUstream s)
{
    if (failing("cuMemAllocFromPoolAsync")) {
        stub_event("alloc bytes=%zu stream=%p FAILED", bytes, (void *)s);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&mu);
    *d = next_ptr;
    next_ptr += (bytes + 0xffff) & ~0xffffull;
    pthread_mutex_unlock(&mu);
    stub_event("alloc bytes=%zu stream=%p ptr=0x%llx", bytes, (void *)s, *d);
    return CUDA_SUCCESS;
}
EXPORT CUresult cuMemFreeAsync(CUdeviceptr d, CUstream s)
{
    stub_event("free ptr=0x%llx stream=%p", d, (void *)s);
    return CUDA_SUCCESS;
}
EXPORT CUresult cuMemsetD32Async(CUdeviceptr d, unsigned v, size_t n, CUstream s)
{
    stub_event("memset ptr=0x%llx words=%zu stream=%p", d, n, (void *)s);
    return failing("cuMemsetD32Async") ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS;
}

/* The fallback's counters: made on a stream that is then waited for, read back with their context pushed. The
 * copies to the host return what the test set (stub_set_dtoh). */
static unsigned long long dtoh_values[8];
static int dtoh_repeat; /* the values repeated over the whole copy, every dtoh_repeat words */
EXPORT void stub_set_dtoh(const unsigned long long *v, int n)
{
    memset(dtoh_values, 0, sizeof dtoh_values);
    memcpy(dtoh_values, v, (size_t)(n < 8 ? n : 8) * sizeof *v);
    dtoh_repeat = 0;
}
EXPORT void stub_set_dtoh_repeat(const unsigned long long *v, int n)
{
    stub_set_dtoh(v, n);
    dtoh_repeat = n < 8 ? n : 8;
}
EXPORT CUresult cuStreamSynchronize(CUstream s)
{
    const int fail = failing("cuStreamSynchronize");
    stub_event("synchronize stream=%p%s", (void *)s, fail ? " FAILED" : "");
    return fail ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS;
}
EXPORT CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes)
{
    const int fail = failing("cuMemcpyDtoH_v2");
    stub_event("copy to host ptr=0x%llx bytes=%zu context=%llu%s", src, bytes, cur()->id, fail ? " FAILED" : "");
    if (fail) return CUDA_ERROR_INVALID_CONTEXT;
    memset(dst, 0, bytes);
    if (dtoh_repeat)
        for (size_t w = 0; (w + 1) * 8 <= bytes; ++w) ((unsigned long long *)dst)[w] = dtoh_values[w % dtoh_repeat];
    else
        memcpy(dst, dtoh_values, bytes < sizeof dtoh_values ? bytes : sizeof dtoh_values);
    return CUDA_SUCCESS;
}
/* Verify mode: the runtime's own stream, and memory allocated outside the pool ("memalloc" events). */
EXPORT CUresult cuStreamCreate(CUstream *s, unsigned flags)
{
    static int n;
    const int fail = failing("cuStreamCreate");
    *s = fail ? NULL : (CUstream)(uintptr_t)(0x6000 + ++n);
    stub_event("stream create flags=%u%s", flags, fail ? " FAILED" : "");
    return fail ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
}
EXPORT CUresult cuMemAlloc_v2(CUdeviceptr *d, size_t bytes)
{
    if (failing("cuMemAlloc_v2")) {
        stub_event("memalloc bytes=%zu FAILED", bytes);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&mu);
    *d = next_ptr;
    next_ptr += (bytes + 0xffff) & ~0xffffull;
    pthread_mutex_unlock(&mu);
    stub_event("memalloc bytes=%zu ptr=0x%llx", bytes, *d);
    return CUDA_SUCCESS;
}
static __thread int pushed[8], npushed;
EXPORT CUresult cuCtxPushCurrent_v2(CUcontext c)
{
    if (failing("cuCtxPushCurrent_v2") || npushed == 8) {
        stub_event("push context=%llu FAILED", ((Ctx *)c)->id);
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    pushed[npushed++] = current;
    current = (int)((Ctx *)c - ctxs) + 1;
    stub_event("push context=%llu", ((Ctx *)c)->id);
    return CUDA_SUCCESS;
}
EXPORT CUresult cuCtxPopCurrent_v2(CUcontext *pc)
{
    if (pc) *pc = (CUcontext)cur();
    current = npushed ? pushed[--npushed] : 0;
    stub_event("pop context");
    return CUDA_SUCCESS;
}

EXPORT CUresult cuGetErrorName(CUresult e, const char **s)
{
    *s = "CUDA_ERROR_STUB";
    return CUDA_SUCCESS;
}
