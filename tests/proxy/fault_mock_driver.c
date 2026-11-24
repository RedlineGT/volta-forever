// A stand-in for NVIDIA's driver library, for fault.py: the layer loads it as the driver
// (VF_REAL_LIBCUDA). Safe to call from several threads; any number of contexts and modules;
// streams and events that belong to the context current when they were made; captures; a fault
// that becomes visible after a delay.
//
// Every module has the functions add_one (index 0), write_null (index 1) and ecc_fail (index 2):
// the handle of a function is (module << 4) | index. A fault, as NVIDIA's driver shows one
// (measured on a V100, tests/kernels/test_fault.c): the launch of write_null (or of a graph that
// holds it) returns 0, and so do the launches made right after it (they are queued); a call of its
// context that waits for the GPU or asks it (a synchronisation, a query, a copy) returns 700 once
// MOCK_DELAY (default 0) such calls of any context were made since the launch, and from then on
// every call that takes the context does (a launch, a module load and unload, a lookup,
// cuFuncGetName, cuStreamGetCtx); cuCtxGetCurrent, cuCtxPushCurrent, cuCtxPopCurrent and
// cuCtxDestroy return 0. ecc_fail does the same with 214 (an uncorrectable ECC error). A context
// made later has no fault.
//
// The legacy stream: a query or a synchronisation of the null stream while a blocking stream (one
// made with flags 0) of the same context is being captured returns 906 and invalidates that
// capture, as NVIDIA's driver does (the CUDA programming guide: "any use of the legacy stream
// while a blocking stream of the context is capturing is invalid"); a launch into an invalidated
// capture and its end return 901.
//
// The primary context of each device: made at its first retain, destroyed at its last release or
// at a reset. With MOCK_EXEC_FIXED set, every instantiated graph gets the same handle (as a driver
// may give the handle of one destroyed to the next one made).
//
// A lock of the driver's own, as NVIDIA's driver takes in its calls: cuStreamQuery,
// cuCtxPushCurrent_v2 and cuCtxPopCurrent_v2 take it, and with MOCK_SIGNAL set, cuCtxSynchronize
// raises SIGUSR1 while it holds it (a signal delivered during a call of the driver, whose handler
// may call exit: what the handlers of exit then ask of the driver waits for ever).
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
enum {
    INVALID_VALUE = 1,
    INVALID_CONTEXT = 201,
    INVALID_HANDLE = 400,
    NOT_FOUND = 500,
    ECC_UNCORRECTABLE = 214,
    ILLEGAL_ADDRESS = 700,
    CAPTURE_INVALIDATED = 901,
    CAPTURE_IMPLICIT = 906,
};

// Counted for the tests: launches that ran, synchronisations of streams, every wait.
_Atomic unsigned launched, synchronized, waits;

#define SLOTS 4096
static _Atomic uintptr_t next_context = 0x9000, next_module = 0x100, next_stream = 0x50000, next_event = 0x70000;
static _Atomic uintptr_t next_graph = 0x600000;
// By context handle modulo SLOTS: the fault met, and the fault under way with its delay.
static _Atomic int sticky[SLOTS], pending[SLOTS], delay[SLOTS];
// By stream (handle modulo SLOTS): its context, whether it is blocking, the graph it captures and
// whether that capture was invalidated. By event: its context.
static _Atomic uintptr_t stream_context[SLOTS], event_context[SLOTS], capturing[SLOTS];
static _Atomic int stream_blocking[SLOTS], invalidated[SLOTS];
// By graph: the error that its launch makes pending (0: none).
static _Atomic int graph_faults[SLOTS];

static _Thread_local uintptr_t stack[16];
static _Thread_local int depth;

static pthread_mutex_t driver_lock = PTHREAD_MUTEX_INITIALIZER;

static uintptr_t current(void) { return depth ? stack[depth - 1] : 0; }
static int slot(uintptr_t h) { return (int)((h >> 4) % SLOTS); }

static CUresult queued(uintptr_t context) { return context ? sticky[slot(context)] : INVALID_CONTEXT; }

// A call that waits for the GPU or asks it, of `context`.
static CUresult waited(uintptr_t context) {
    if (!context) return INVALID_CONTEXT;
    ++waits;
    int c = slot(context);
    if (pending[c]) {
        if (delay[c] > 0) {
            --delay[c];
            return 0;
        }
        sticky[c] = pending[c];
        pending[c] = 0;
    }
    return sticky[c];
}

static void make_pending(uintptr_t context, int error) {
    const char *d = getenv("MOCK_DELAY");
    delay[slot(context)] = d ? atoi(d) : 0;
    pending[slot(context)] = error;
}

CUresult cuInit(unsigned flags) { (void)flags; return 0; }
CUresult cuDriverGetVersion(int *version) { *version = 13000; return 0; }
CUresult cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
CUresult cuDeviceComputeCapability(int *major, int *minor, int device) {
    (void)device;
    *major = 7;
    *minor = 0;
    return 0;
}

static void fresh(uintptr_t c) { sticky[slot(c)] = pending[slot(c)] = delay[slot(c)] = 0; }

CUresult cuCtxCreate_v2(void **context, unsigned flags, int device) {
    (void)flags; (void)device;
    uintptr_t c = (next_context += 0x10);
    fresh(c);
    *context = (void *)c;
    if (!depth) depth = 1;
    stack[depth - 1] = c;
    return 0;
}
CUresult cuCtxDestroy_v2(void *context) {
    for (int d = 0; d < depth; ++d)
        if (stack[d] == (uintptr_t)context) stack[d] = 0;
    return 0;
}
CUresult cuCtxGetCurrent(void **context) { *context = (void *)current(); return 0; }
CUresult cuCtxSetCurrent(void *context) {
    if (!depth) depth = 1;
    stack[depth - 1] = (uintptr_t)context;
    return 0;
}
CUresult cuCtxPushCurrent_v2(void *context) {
    pthread_mutex_lock(&driver_lock);
    CUresult r = depth == 16 ? 1 : 0;
    if (!r) stack[depth++] = (uintptr_t)context;
    pthread_mutex_unlock(&driver_lock);
    return r;
}
CUresult cuCtxPopCurrent_v2(void **context) {
    pthread_mutex_lock(&driver_lock);
    CUresult r = depth ? 0 : INVALID_CONTEXT;
    if (!r) {
        if (context) *context = (void *)stack[depth - 1];
        --depth;
    }
    pthread_mutex_unlock(&driver_lock);
    return r;
}
CUresult cuCtxGetDevice(int *device) {
    if (!current()) return INVALID_CONTEXT;
    *device = 0;
    return 0;
}

// The primary context of devices 0 to 3.
#define PRIMARY(device) ((uintptr_t)0x8000 + 0x10 * (uintptr_t)(device))
static _Atomic int retains[4], active[4];
CUresult cuDevicePrimaryCtxRetain(void **context, int device) {
    if (device < 0 || device > 3) return INVALID_VALUE;
    if (retains[device]++ == 0 && !active[device]) fresh(PRIMARY(device));
    active[device] = 1;
    *context = (void *)PRIMARY(device);
    return 0;
}
CUresult cuDevicePrimaryCtxRelease_v2(int device) {
    if (device < 0 || device > 3 || !retains[device]) return INVALID_CONTEXT;
    if (!--retains[device]) active[device] = 0;
    return 0;
}
CUresult cuDevicePrimaryCtxRelease(int device) { return cuDevicePrimaryCtxRelease_v2(device); }
CUresult cuDevicePrimaryCtxReset_v2(int device) {
    if (device < 0 || device > 3) return INVALID_VALUE;
    active[device] = 0;
    return 0;
}
CUresult cuDevicePrimaryCtxReset(int device) { return cuDevicePrimaryCtxReset_v2(device); }
CUresult cuDevicePrimaryCtxGetState(int device, unsigned *flags, int *is_active) {
    if (device < 0 || device > 3) return INVALID_VALUE;
    *flags = 0;
    *is_active = active[device];
    return 0;
}

static const char *const names[] = {"add_one", "write_null", "ecc_fail"};
#define NAMES 3
CUresult cuModuleLoadData(void **module, const void *image) {
    (void)image;
    CUresult r = queued(current());
    if (r) return r;
    *module = (void *)(next_module++);
    return 0;
}
CUresult cuModuleLoadDataEx(void **module, const void *image, unsigned n, int *options, void **values) {
    (void)n; (void)options; (void)values;
    return cuModuleLoadData(module, image);
}
CUresult cuModuleUnload(void *module) { (void)module; return queued(current()); }
CUresult cuModuleGetFunction(void **function, void *module, const char *name) {
    CUresult r = queued(current());
    if (r) return r;
    for (int i = 0; i < NAMES; ++i)
        if (!strcmp(name, names[i])) {
            *function = (void *)(((uintptr_t)module << 4) | (uintptr_t)i);
            return 0;
        }
    return NOT_FOUND;
}
CUresult cuFuncGetName(const char **name, void *function) {
    CUresult r = queued(current());
    if (r) return r;
    uintptr_t i = (uintptr_t)function & 0xf;
    if (i >= NAMES) return INVALID_HANDLE;
    *name = names[i];
    return 0;
}

static int is_null_stream(void *stream) { return (uintptr_t)stream <= 2; }
static uintptr_t context_of(void *stream) {
    return is_null_stream(stream) ? current() : stream_context[slot((uintptr_t)stream)];
}
CUresult cuStreamCreate(void **stream, unsigned flags) {
    uintptr_t s = (next_stream += 0x10);
    stream_context[slot(s)] = current();
    stream_blocking[slot(s)] = (flags & 1) == 0;
    capturing[slot(s)] = 0;
    invalidated[slot(s)] = 0;
    *stream = (void *)s;
    return 0;
}
CUresult cuStreamDestroy_v2(void *stream) { (void)stream; return 0; }
CUresult cuStreamGetCtx(void *stream, void **context) {
    uintptr_t c = context_of(stream);
    if (sticky[slot(c)]) return sticky[slot(c)];
    *context = (void *)c;
    return 0;
}
CUresult cuStreamBeginCapture_v2(void *stream, int mode) {
    (void)mode;
    if (is_null_stream(stream)) return INVALID_VALUE;
    capturing[slot((uintptr_t)stream)] = next_graph++;
    graph_faults[slot(capturing[slot((uintptr_t)stream)])] = 0;
    invalidated[slot((uintptr_t)stream)] = 0;
    return 0;
}
CUresult cuStreamEndCapture(void *stream, void **graph) {
    if (is_null_stream(stream)) return INVALID_VALUE;
    int s = slot((uintptr_t)stream);
    uintptr_t g = capturing[s];
    if (!g) return INVALID_VALUE;
    capturing[s] = 0;
    if (invalidated[s]) {
        invalidated[s] = 0;
        return CAPTURE_INVALIDATED;
    }
    *graph = (void *)g;
    return 0;
}
CUresult cuStreamGetCaptureInfo_v2(void *stream, int *status, uint64_t *id, void **graph, const void **deps,
                                   size_t *count) {
    CUresult r = queued(context_of(stream));
    if (r) return r;
    int s = is_null_stream(stream) ? -1 : slot((uintptr_t)stream);
    uintptr_t g = s >= 0 ? capturing[s] : 0;
    *status = !g ? 0 : invalidated[s] ? 2 : 1;
    if (id) *id = 0;
    if (graph) *graph = (void *)g;
    if (deps) *deps = 0;
    if (count) *count = 0;
    return 0;
}

// A use of the legacy stream of `context`: invalid while a blocking stream of it is capturing,
// whose capture it invalidates.
static CUresult legacy_use(uintptr_t context) {
    int invalid = 0;
    for (uintptr_t h = 0x50010; h <= next_stream; h += 0x10) {
        int s = slot(h);
        if (capturing[s] && stream_blocking[s] && stream_context[s] == context) {
            invalidated[s] = 1;
            invalid = 1;
        }
    }
    return invalid ? CAPTURE_IMPLICIT : 0;
}

static _Atomic uintptr_t exec_graph;
CUresult cuGraphInstantiateWithFlags(void **exec, void *graph, unsigned long long flags) {
    (void)flags;
    if (getenv("MOCK_EXEC_FIXED")) {
        exec_graph = (uintptr_t)graph;
        *exec = (void *)0x777000;
    } else {
        *exec = (void *)((uintptr_t)graph + 0x10000);
    }
    return 0;
}
// (Launched with the graph it has at its launch.)
CUresult cuGraphLaunch(void *exec, void *stream) {
    uintptr_t c = context_of(stream);
    CUresult r = queued(c);
    if (r) return r;
    uintptr_t g = getenv("MOCK_EXEC_FIXED") ? exec_graph : (uintptr_t)exec - 0x10000;
    if (graph_faults[slot(g)]) make_pending(c, graph_faults[slot(g)]);
    return 0;
}
CUresult cuGraphDestroy(void *graph) { (void)graph; return 0; }
CUresult cuGraphExecDestroy(void *exec) { (void)exec; return 0; }

CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shared, void *stream, void **params, void **extra) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)shared; (void)params; (void)extra;
    uintptr_t c = context_of(stream);
    CUresult r = queued(c);
    if (r) return r;
    uintptr_t i = (uintptr_t)f & 0xf;
    if (i >= NAMES) return INVALID_HANDLE;
    int error = i == 1 ? ILLEGAL_ADDRESS : i == 2 ? ECC_UNCORRECTABLE : 0;
    if (!is_null_stream(stream) && capturing[slot((uintptr_t)stream)]) {
        if (invalidated[slot((uintptr_t)stream)]) return CAPTURE_INVALIDATED;
        if (error) graph_faults[slot(capturing[slot((uintptr_t)stream)])] = error;
        return 0;
    }
    ++launched;
    if (error) make_pending(c, error);
    return 0;
}
CUresult cuLaunchKernel_ptsz(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                             unsigned shared, void *stream, void **params, void **extra) {
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

CUresult cuCtxSynchronize(void) {
    if (getenv("MOCK_SIGNAL")) {
        pthread_mutex_lock(&driver_lock);
        raise(SIGUSR1);
        pthread_mutex_unlock(&driver_lock);
    }
    return waited(current());
}
CUresult cuStreamSynchronize(void *stream) {
    ++synchronized;
    if ((uintptr_t)stream == 0 || (uintptr_t)stream == 1) {
        CUresult r = legacy_use(current());
        if (r) return r;
    }
    return waited(context_of(stream));
}
CUresult cuStreamSynchronize_ptsz(void *stream) { return waited(context_of(stream)); }
CUresult cuStreamQuery(void *stream) {
    pthread_mutex_lock(&driver_lock);
    CUresult r = 0;
    if ((uintptr_t)stream == 0 || (uintptr_t)stream == 1) r = legacy_use(current());
    if (!r) r = waited(context_of(stream));
    pthread_mutex_unlock(&driver_lock);
    return r;
}

CUresult cuEventCreate(void **event, unsigned flags) {
    (void)flags;
    uintptr_t e = (next_event += 0x10);
    event_context[slot(e)] = current();
    *event = (void *)e;
    return 0;
}
CUresult cuEventRecord(void *event, void *stream) {
    event_context[slot((uintptr_t)event)] = context_of(stream);
    return queued(context_of(stream));
}
CUresult cuEventSynchronize(void *event) { return waited(event_context[slot((uintptr_t)event)]); }
CUresult cuEventQuery(void *event) { return waited(event_context[slot((uintptr_t)event)]); }

// Lookups: the functions of this library that the tests look up.
CUresult cuGetProcAddress(const char *symbol, void **pointer, int version, uint64_t flags) {
    (void)version; (void)flags;
    if (!strcmp(symbol, "cuStreamSynchronize")) *pointer = (void *)cuStreamSynchronize;
    else if (!strcmp(symbol, "cuCtxSynchronize")) *pointer = (void *)cuCtxSynchronize;
    else if (!strcmp(symbol, "cuLaunchKernel")) *pointer = (void *)cuLaunchKernel;
    else return NOT_FOUND;
    return 0;
}

CUresult cuMemAlloc_v2(uint64_t *pointer, size_t size) {
    (void)size;
    *pointer = 0x7f0000000000;
    return queued(current());
}
CUresult cuMemcpyDtoH_v2(void *host, uint64_t device, size_t size) {
    (void)device;
    memset(host, 0, size);
    return waited(current());
}
