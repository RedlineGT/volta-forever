// A stand-in for NVIDIA's driver library, for fault.py: the layer loads it as the driver
// (VF_REAL_LIBCUDA). It keeps modules with the names of the entries of the PTX they were given
// (".entry NAME("), a function for each name asked for, contexts, streams, and graphs captured on a
// stream.
//
// A fault, as NVIDIA's driver shows one (measured on a V100, tests/kernels/test_fault.c): the launch
// of a function named write_null, or of a graph that holds one, returns 0, and so do the launches
// made right after it (they are queued); the first call that waits for the GPU or asks it (a
// synchronisation, a query, a copy) returns 700, and from then on every call that takes the context
// (a launch, a module load, a lookup, cuFuncGetName) does; cuCtxGetCurrent, cuCtxPushCurrent,
// cuCtxPopCurrent and cuCtxDestroy return 0. A context made later has no fault.
#include <stdint.h>
#include <string.h>

typedef int CUresult;
enum { INVALID_HANDLE = 400, NOT_FOUND = 500, ILLEGAL_ADDRESS = 700 };

unsigned launched, synchronized;

#define MAX 64
#define NAMES 8
static struct {
    uintptr_t handle;
    int live;
    char names[NAMES][32];
    int count;
} modules[MAX];
static int module_count;
static uintptr_t module_counter = 0x1000;

static struct {
    uintptr_t handle;
    int module, name;
} functions[4 * MAX];
static int function_count;
static uintptr_t function_counter = 0x100000;

// By context (its handle modulo MAX): a fault met (`sticky`) or under way (`pending`).
static int sticky[MAX], pending[MAX];
static uintptr_t context_counter = 0x9000;
static _Thread_local uintptr_t stack[8];
static _Thread_local int depth;

static uintptr_t current(void) { return depth ? stack[depth - 1] : 0; }

// What a call that takes the current context returns, without waiting for the GPU: the fault, once a
// call that waits met it.
static CUresult queued_state(void) { return sticky[current() % MAX]; }

// What a call that waits for the GPU, or asks it, returns: the fault under way, if there is one.
static CUresult context_state(void) {
    uintptr_t c = current() % MAX;
    if (pending[c]) {
        sticky[c] = pending[c];
        pending[c] = 0;
        return sticky[c];
    }
    return sticky[c];
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
CUresult cuCtxCreate_v2(void **context, unsigned flags, int device) {
    (void)flags; (void)device;
    *context = (void *)context_counter++;
    sticky[(uintptr_t)*context % MAX] = pending[(uintptr_t)*context % MAX] = 0;
    if (!depth) depth = 1;
    stack[depth - 1] = (uintptr_t)*context;
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
    if (depth == 8) return 1;
    stack[depth++] = (uintptr_t)context;
    return 0;
}
CUresult cuCtxPopCurrent_v2(void **context) {
    if (!depth) return 201;
    if (context) *context = (void *)stack[depth - 1];
    --depth;
    return 0;
}
CUresult cuCtxGetDevice(int *device) {
    if (!current()) return 201;
    *device = 0;
    return 0;
}

CUresult cuModuleLoadData(void **module, const void *image) {
    CUresult r = queued_state();
    if (r) return r;
    int i = module_count++;
    memset(&modules[i], 0, sizeof modules[i]);
    modules[i].handle = module_counter++;
    modules[i].live = 1;
    for (const char *p = image; (p = strstr(p, ".entry ")) && modules[i].count < NAMES;) {
        p += 7;
        int n = 0;
        while (p[n] && p[n] != '(' && p[n] != ' ' && p[n] != '\n' && n < 31) ++n;
        memcpy(modules[i].names[modules[i].count], p, n);
        modules[i].names[modules[i].count++][n] = 0;
    }
    *module = (void *)modules[i].handle;
    return 0;
}
CUresult cuModuleLoadDataEx(void **module, const void *image, unsigned n, int *options, void **values) {
    (void)n; (void)options; (void)values;
    return cuModuleLoadData(module, image);
}
CUresult cuModuleUnload(void *module) {
    CUresult r = queued_state();
    if (r) return r;
    for (int i = 0; i < module_count; ++i)
        if (modules[i].live && modules[i].handle == (uintptr_t)module) {
            modules[i].live = 0;
            return 0;
        }
    return INVALID_HANDLE;
}
CUresult cuModuleGetFunction(void **function, void *module, const char *name) {
    CUresult r = queued_state();
    if (r) return r;
    for (int i = 0; i < module_count; ++i) {
        if (!modules[i].live || modules[i].handle != (uintptr_t)module) continue;
        for (int n = 0; n < modules[i].count; ++n) {
            if (strcmp(modules[i].names[n], name)) continue;
            for (int f = 0; f < function_count; ++f)
                if (functions[f].module == i && functions[f].name == n) {
                    *function = (void *)functions[f].handle;
                    return 0;
                }
            int f = function_count++;
            functions[f].handle = function_counter++ * 16;
            functions[f].module = i;
            functions[f].name = n;
            *function = (void *)functions[f].handle;
            return 0;
        }
        return NOT_FOUND;
    }
    return INVALID_HANDLE;
}
static const char *name_of(void *function) {
    for (int f = 0; f < function_count; ++f)
        if (functions[f].handle == (uintptr_t)function) return modules[functions[f].module].names[functions[f].name];
    return 0;
}
CUresult cuFuncGetName(const char **name, void *function) {
    CUresult r = queued_state();
    if (r) return r;
    *name = name_of(function);
    return *name ? 0 : INVALID_HANDLE;
}

// Streams, and the graphs captured on them.
#define STREAMS 16
static uintptr_t stream_handles[STREAMS], capturing[STREAMS];
static int stream_count;
static uintptr_t graph_counter = 0x600000;
static int graph_faults[MAX];

static int stream_index(void *stream) {
    for (int s = 0; s < stream_count; ++s)
        if (stream_handles[s] == (uintptr_t)stream) return s;
    return -1;
}
CUresult cuStreamCreate(void **stream, unsigned flags) {
    (void)flags;
    stream_handles[stream_count] = 0x5000 + 0x10 * stream_count;
    *stream = (void *)stream_handles[stream_count++];
    return 0;
}
CUresult cuStreamBeginCapture_v2(void *stream, int mode) {
    (void)mode;
    int s = stream_index(stream);
    if (s < 0) return INVALID_HANDLE;
    capturing[s] = graph_counter++;
    graph_faults[capturing[s] % MAX] = 0;
    return 0;
}
CUresult cuStreamEndCapture(void *stream, void **graph) {
    int s = stream_index(stream);
    if (s < 0 || !capturing[s]) return 1;
    *graph = (void *)capturing[s];
    capturing[s] = 0;
    return 0;
}
CUresult cuStreamGetCaptureInfo_v2(void *stream, int *status, uint64_t *id, void **graph, const void **deps,
                                   size_t *count) {
    CUresult r = context_state();
    if (r) return r;
    int s = stream_index(stream);
    *status = s >= 0 && capturing[s] ? 1 : 0;
    if (id) *id = 0;
    if (graph) *graph = *status ? (void *)capturing[s] : 0;
    if (deps) *deps = 0;
    if (count) *count = 0;
    return 0;
}
CUresult cuGraphInstantiateWithFlags(void **exec, void *graph, unsigned long long flags) {
    (void)flags;
    *exec = (void *)((uintptr_t)graph + 0x10000);
    return 0;
}
CUresult cuGraphLaunch(void *exec, void *stream) {
    (void)stream;
    CUresult r = queued_state();
    if (r) return r;
    if (graph_faults[((uintptr_t)exec - 0x10000) % MAX]) pending[current() % MAX] = ILLEGAL_ADDRESS;
    return 0;
}
CUresult cuGraphDestroy(void *graph) { (void)graph; return 0; }
CUresult cuGraphExecDestroy(void *exec) { (void)exec; return 0; }

CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shared, void *stream, void **params, void **extra) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)shared; (void)params; (void)extra;
    CUresult r = queued_state();
    if (r) return r;
    const char *name = name_of(f);
    if (!name) return INVALID_HANDLE;
    int s = stream_index(stream);
    if (s >= 0 && capturing[s]) {
        if (!strcmp(name, "write_null")) graph_faults[capturing[s] % MAX] = 1;
        return 0;
    }
    ++launched;
    if (!strcmp(name, "write_null")) pending[current() % MAX] = ILLEGAL_ADDRESS;
    return 0;
}

// The thread's own default stream: the same launches.
CUresult cuLaunchKernel_ptsz(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                             unsigned shared, void *stream, void **params, void **extra) {
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

static CUresult wait(void) { return context_state(); }
CUresult cuCtxSynchronize(void) { return wait(); }
CUresult cuStreamSynchronize(void *stream) {
    (void)stream;
    ++synchronized;
    return wait();
}
CUresult cuStreamSynchronize_ptsz(void *stream) { (void)stream; return wait(); }
CUresult cuStreamQuery(void *stream) { (void)stream; return wait(); }
CUresult cuStreamDestroy_v2(void *stream) {
    int s = stream_index(stream);
    if (s < 0) return INVALID_HANDLE;
    stream_handles[s] = 0;
    return 0;
}
CUresult cuMemAlloc_v2(uint64_t *pointer, size_t size) {
    (void)size;
    *pointer = 0x7f0000000000;
    return context_state();
}
CUresult cuMemcpyDtoH_v2(void *host, uint64_t device, size_t size) {
    (void)device;
    memset(host, 0, size);
    return context_state();
}
