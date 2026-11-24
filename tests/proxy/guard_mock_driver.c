// A stand-in for NVIDIA's driver library, for launch_guard.py: the layer loads it as the driver
// (VF_REAL_LIBCUDA), and the test also calls it directly, as a program reaches the driver without
// the layer (a private interface, which the layer does not see).
//
// It keeps what NVIDIA's driver keeps, by handle: modules and libraries with the names of the
// entries of the PTX they were given (".entry NAME("), a function for each name of a module asked
// for, a kernel for each name of a library, and the module that a library has in each context
// (cuLibraryGetModule, cuKernelGetFunction). A handle that is unloaded, or that goes with its context
// (cuCtxDestroy), is dead: a call with it fails with 400, as the driver answers for an invalid
// handle. The next module or library loaded, and the next function or kernel made, can be given a
// handle chosen by the test (`next_handle`, `next_function`), as the driver gives the handle of one
// that is gone.
//
// Counted for the test: the launches the driver makes (`launched`, by any launch function), the
// kernel nodes made or changed in graphs (`nodes`), the questions about a function that it answers
// (`asked`), and the calls by which the layer asks what a handle is (`identified`).
//
// cuGetExportTable gives a table whose entry 1 (after its size) is a private function that gives
// the function of the first entry of a module, as cuFFT takes its functions (`first_function`).
//
// Its compiler refuses a library whose PTX has the entries late() and other() and a line with 54321
// (in late): error 218, and that line in the error log, as ptxas writes it. Alone, late() is
// accepted: the layer learns that it is refused when a deferred library is loaded whole, after it
// gave the kernel compiled alone.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int CUresult;
enum { INVALID_VALUE = 1, INVALID_HANDLE = 400, NOT_FOUND = 500 };

unsigned launched, nodes, asked, identified, loads, unloads;
uintptr_t next_handle, next_function;

enum { MODULE = 1, LIBRARY, CONTEXT_MODULE };
#define MAX 256
#define NAMES 8
static struct {
    uintptr_t handle;
    int kind, live;
    uintptr_t context;
    int library; // of a CONTEXT_MODULE: the index of its library
    char names[NAMES][32];
    int count;
} objects[MAX];
static int object_count;

// Functions (of modules) and kernels (of libraries).
static struct {
    uintptr_t handle;
    int owner, name, live, kernel;
} handles[4 * MAX];
static int handle_count;
static uintptr_t counter = 0x1000, function_counter = 0x100000;

static _Thread_local uintptr_t current;
static uintptr_t context_counter = 0x9000;

static int find_object(void *handle) {
    for (int i = object_count - 1; i >= 0; --i)
        if (objects[i].live && objects[i].handle == (uintptr_t)handle) return i;
    return -1;
}

static int new_object(int kind, const char *text) {
    int i = object_count++;
    uintptr_t handle = next_handle ? next_handle : counter++;
    next_handle = 0;
    memset(&objects[i], 0, sizeof objects[i]);
    objects[i].handle = handle;
    objects[i].kind = kind;
    objects[i].live = 1;
    objects[i].context = current;
    for (const char *p = text; p && (p = strstr(p, ".entry ")) && objects[i].count < NAMES;) {
        p += 7;
        int n = 0;
        while (p[n] && p[n] != '(' && p[n] != ' ' && p[n] != '\n' && n < 31) ++n;
        memcpy(objects[i].names[objects[i].count], p, n);
        objects[i].names[objects[i].count++][n] = 0;
    }
    return i;
}

static int find_handle(void *handle) {
    for (int i = handle_count - 1; i >= 0; --i)
        if (handles[i].live && handles[i].handle == (uintptr_t)handle) return i;
    return -1;
}

// The function (kernel = 0) or kernel (1) of entry `name` of object `owner`.
static uintptr_t handle_of(int owner, int name, int kernel) {
    for (int i = 0; i < handle_count; ++i)
        if (handles[i].live && handles[i].owner == owner && handles[i].name == name && handles[i].kernel == kernel)
            return handles[i].handle;
    int i = handle_count++;
    handles[i].handle = next_function ? next_function : function_counter++ * 16 + (kernel ? 8 : 0);
    next_function = 0;
    handles[i].owner = owner;
    handles[i].name = name;
    handles[i].live = 1;
    handles[i].kernel = kernel;
    return handles[i].handle;
}

static int name_index(int owner, const char *name) {
    for (int n = 0; n < objects[owner].count; ++n)
        if (!strcmp(objects[owner].names[n], name)) return n;
    return -1;
}

static void kill(int object) {
    objects[object].live = 0;
    for (int i = 0; i < handle_count; ++i)
        if (handles[i].owner == object) handles[i].live = 0;
    for (int i = 0; i < object_count; ++i)
        if (objects[i].live && objects[i].kind == CONTEXT_MODULE && objects[i].library == object) kill(i);
}

// The module of library `library` in the current context.
static int context_module(int library) {
    for (int i = 0; i < object_count; ++i)
        if (objects[i].live && objects[i].kind == CONTEXT_MODULE && objects[i].library == library &&
            objects[i].context == current)
            return i;
    int i = new_object(CONTEXT_MODULE, 0);
    objects[i].library = library;
    objects[i].count = objects[library].count;
    memcpy(objects[i].names, objects[library].names, sizeof objects[i].names);
    return i;
}

CUresult cuInit(unsigned flags) { (void)flags; return 0; }
CUresult cuDriverGetVersion(int *version) { *version = 13000; return 0; }
CUresult cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
// Device 0 is the V100; device 1 a later GPU, whose contexts the layer gives the program's code as it
// is.
CUresult cuDeviceComputeCapability(int *major, int *minor, int device) {
    *major = device == 1 ? 9 : 7;
    *minor = 0;
    return 0;
}
static int devices[64];
CUresult cuCtxCreate_v2(void **context, unsigned flags, int device) {
    (void)flags;
    devices[context_counter % 64] = device;
    *context = (void *)context_counter++;
    current = (uintptr_t)*context;
    return 0;
}
CUresult cuCtxDestroy_v2(void *context) {
    for (int i = 0; i < object_count; ++i)
        if (objects[i].live && objects[i].kind != LIBRARY && objects[i].context == (uintptr_t)context) kill(i);
    if (current == (uintptr_t)context) current = 0;
    return 0;
}
CUresult cuCtxDestroy(void *context) { return cuCtxDestroy_v2(context); }
// (A green context: one of these, to the test.)
CUresult cuGreenCtxDestroy(void *context) { return cuCtxDestroy_v2(context); }
CUresult cuCtxGetCurrent(void **context) { *context = (void *)current; return 0; }
CUresult cuCtxSetCurrent(void *context) { current = (uintptr_t)context; return 0; }
CUresult cuCtxGetDevice(int *device) {
    if (!current) return 201;
    *device = devices[current % 64];
    return 0;
}
CUresult cuCtxGetDevice_v2(int *device, void *context) {
    if (!context) return cuCtxGetDevice(device);
    *device = devices[(uintptr_t)context % 64];
    return 0;
}

CUresult cuModuleLoadData(void **module, const void *image) {
    ++loads;
    *module = (void *)objects[new_object(MODULE, image)].handle;
    return 0;
}
CUresult cuModuleLoadDataEx(void **module, const void *image, unsigned n, int *options, void **values) {
    (void)n; (void)options; (void)values;
    return cuModuleLoadData(module, image);
}
CUresult cuLibraryLoadData(void **library, const void *image, int *jo, void **jv, unsigned jc, int *lo, void **lv,
                           unsigned lc) {
    (void)lo; (void)lv; (void)lc;
    const char *text = image, *hit = strstr(text, "54321");
    if (hit && strstr(text, "late(") && strstr(text, "other(")) {
        int line = 1;
        for (const char *p = text; p < hit; ++p) line += *p == '\n';
        char *buffer = 0;
        size_t size = 0;
        for (unsigned i = 0; i < jc; ++i) {
            if (jo[i] == 5) buffer = jv[i];
            if (jo[i] == 6) size = (size_t)(uintptr_t)jv[i];
        }
        if (buffer && size)
            snprintf(buffer, size, "ptxas application ptx input, line %d; error   : Feature 'x' requires .target sm_80\n",
                     line);
        return 218;
    }
    ++loads;
    *library = (void *)objects[new_object(LIBRARY, image)].handle;
    return 0;
}
CUresult cuModuleUnload(void *module) {
    int i = find_object(module);
    if (i < 0) return INVALID_HANDLE;
    ++unloads;
    kill(i);
    return 0;
}
CUresult cuLibraryUnload(void *library) { return cuModuleUnload(library); }

CUresult cuModuleGetFunction(void **function, void *module, const char *name) {
    int i = find_object(module), n = i < 0 ? -1 : name_index(i, name);
    if (i < 0 || objects[i].kind == LIBRARY) return INVALID_HANDLE;
    if (n < 0) {
        *function = 0;
        return NOT_FOUND;
    }
    *function = (void *)handle_of(i, n, 0);
    return 0;
}
CUresult cuLibraryGetKernel(void **kernel, void *library, const char *name) {
    int i = find_object(library), n = i < 0 ? -1 : name_index(i, name);
    if (i < 0 || objects[i].kind != LIBRARY) return INVALID_HANDLE;
    if (n < 0) {
        *kernel = 0;
        return NOT_FOUND;
    }
    *kernel = (void *)handle_of(i, n, 1);
    return 0;
}
CUresult cuLibraryGetModule(void **module, void *library) {
    int i = find_object(library);
    if (i < 0 || objects[i].kind != LIBRARY) return INVALID_HANDLE;
    *module = (void *)objects[context_module(i)].handle;
    return 0;
}
CUresult cuKernelGetFunction(void **function, void *kernel) {
    int k = find_handle(kernel);
    if (k < 0 || !handles[k].kernel) return INVALID_HANDLE;
    *function = (void *)handle_of(context_module(handles[k].owner), handles[k].name, 0);
    return 0;
}
CUresult cuFuncGetModule(void **module, void *function) {
    ++identified;
    int f = find_handle(function);
    if (f < 0 || handles[f].kernel) return INVALID_HANDLE;
    *module = (void *)objects[handles[f].owner].handle;
    return 0;
}
CUresult cuKernelGetLibrary(void **library, void *kernel) {
    ++identified;
    int k = find_handle(kernel);
    if (k < 0 || !handles[k].kernel) return INVALID_HANDLE;
    *library = (void *)objects[handles[k].owner].handle;
    return 0;
}
static CUresult name_of(const char **name, void *handle, int kernel) {
    int h = find_handle(handle);
    if (h < 0 || handles[h].kernel != kernel) return INVALID_HANDLE;
    *name = objects[handles[h].owner].names[handles[h].name];
    return 0;
}
CUresult cuFuncGetName(const char **name, void *function) { return name_of(name, function, 0); }
CUresult cuKernelGetName(const char **name, void *kernel) { return name_of(name, kernel, 1); }

static CUresult list(void **out, unsigned capacity, void *owner, int kernel) {
    int i = find_object(owner);
    if (i < 0) return INVALID_HANDLE;
    for (unsigned n = 0; n < capacity && n < (unsigned)objects[i].count; ++n) out[n] = (void *)handle_of(i, n, kernel);
    return 0;
}
static CUresult count_of(unsigned *count, void *owner) {
    int i = find_object(owner);
    if (i < 0) return INVALID_HANDLE;
    *count = objects[i].count;
    return 0;
}
CUresult cuModuleGetFunctionCount(unsigned *count, void *module) { return count_of(count, module); }
CUresult cuLibraryGetKernelCount(unsigned *count, void *library) { return count_of(count, library); }
CUresult cuModuleEnumerateFunctions(void **out, unsigned capacity, void *module) { return list(out, capacity, module, 0); }
CUresult cuLibraryEnumerateKernels(void **out, unsigned capacity, void *library) {
    return list(out, capacity, library, 1);
}

// ---- what is done with a function or a kernel: done when the handle is alive ----

static CUresult launch(void *function) {
    if (find_handle(function) < 0) return INVALID_HANDLE;
    ++launched;
    return 0;
}
CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shared, void *stream, void **params, void **extra) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)shared; (void)stream; (void)params; (void)extra;
    return launch(f);
}
CUresult cuLaunchKernel_ptsz(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                             unsigned shared, void *stream, void **params, void **extra) {
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}
CUresult cuLaunchKernelEx(const void *config, void *f, void **params, void **extra) {
    (void)config; (void)params; (void)extra;
    return launch(f);
}
CUresult cuLaunchKernelEx_ptsz(const void *config, void *f, void **params, void **extra) {
    return cuLaunchKernelEx(config, f, params, extra);
}
CUresult cuLaunch(void *f) { return launch(f); }
CUresult cuLaunchGrid(void *f, int w, int h) { (void)w; (void)h; return launch(f); }
CUresult cuLaunchGridAsync(void *f, int w, int h, void *stream) { (void)w; (void)h; (void)stream; return launch(f); }
CUresult cuLaunchCooperativeKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                                   unsigned bz, unsigned shared, void *stream, void **params) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)shared; (void)stream; (void)params;
    return launch(f);
}
CUresult cuLaunchCooperativeKernel_ptsz(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                                        unsigned bz, unsigned shared, void *stream, void **params) {
    return cuLaunchCooperativeKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params);
}
// CUDA_LAUNCH_PARAMS: the function first, 56 bytes.
CUresult cuLaunchCooperativeKernelMultiDevice(void *list, unsigned count, unsigned flags) {
    (void)flags;
    for (unsigned i = 0; i < count; ++i)
        if (find_handle(*(void **)((char *)list + 56 * i)) < 0) return INVALID_HANDLE;
    launched += count;
    return 0;
}

// Graphs: a node holds what it was given; the kernel node parameters start with the function, and
// those of _v2 have the kernel at byte 56 (used when the function is null); CUgraphNodeParams has
// its type first (0: a kernel) and the kernel parameters at byte 16.
static uintptr_t node_counter = 0x700000;
static CUresult node(void **out, const void *params, int v2) {
    void *function = *(void *const *)params;
    if (!function && v2) function = *(void *const *)((const char *)params + 56);
    if (find_handle(function) < 0) return INVALID_HANDLE;
    if (out) *out = (void *)node_counter++;
    ++nodes;
    return 0;
}
CUresult cuGraphCreate(void **graph, unsigned flags) { (void)flags; *graph = (void *)0x600000; return 0; }
CUresult cuGraphAddKernelNode(void **out, void *g, const void *deps, size_t n, const void *params) {
    (void)g; (void)deps; (void)n;
    return node(out, params, 0);
}
CUresult cuGraphAddKernelNode_v2(void **out, void *g, const void *deps, size_t n, const void *params) {
    (void)g; (void)deps; (void)n;
    return node(out, params, 1);
}
CUresult cuGraphKernelNodeSetParams(void *n, const void *params) { (void)n; return node(0, params, 0); }
CUresult cuGraphKernelNodeSetParams_v2(void *n, const void *params) { (void)n; return node(0, params, 1); }
CUresult cuGraphExecKernelNodeSetParams(void *e, void *n, const void *params) {
    (void)e; (void)n;
    return node(0, params, 0);
}
CUresult cuGraphExecKernelNodeSetParams_v2(void *e, void *n, const void *params) {
    (void)e; (void)n;
    return node(0, params, 1);
}
static CUresult any_node(void **out, const void *params) {
    if (*(const int *)params != 0) {
        if (out) *out = (void *)node_counter++;
        return 0;
    }
    return node(out, (const char *)params + 16, 1);
}
CUresult cuGraphAddNode(void **out, void *g, const void *deps, size_t n, const void *params) {
    (void)g; (void)deps; (void)n;
    return any_node(out, params);
}
CUresult cuGraphAddNode_v2(void **out, void *g, const void *deps, const void *edges, size_t n, const void *params) {
    (void)g; (void)deps; (void)edges; (void)n;
    return any_node(out, params);
}
CUresult cuGraphNodeSetParams(void *n, const void *params) { (void)n; return any_node(0, params); }
CUresult cuGraphExecNodeSetParams(void *e, void *n, const void *params) { (void)e; (void)n; return any_node(0, params); }

static CUresult ask(void *handle) {
    if (find_handle(handle) < 0) return INVALID_HANDLE;
    ++asked;
    return 0;
}
CUresult cuFuncGetAttribute(int *value, int attribute, void *f) { (void)attribute; *value = 1; return ask(f); }
CUresult cuFuncSetAttribute(void *f, int attribute, int value) { (void)attribute; (void)value; return ask(f); }
CUresult cuFuncSetCacheConfig(void *f, int config) { (void)config; return ask(f); }
CUresult cuFuncSetSharedMemConfig(void *f, int config) { (void)config; return ask(f); }
CUresult cuFuncGetParamInfo(void *f, size_t index, size_t *offset, size_t *size) {
    (void)index; *offset = 0; *size = 8;
    return ask(f);
}
CUresult cuFuncLoad(void *f) { return ask(f); }
CUresult cuKernelGetAttribute(int *value, int attribute, void *k, int device) {
    (void)attribute; (void)device; *value = 1;
    return ask(k);
}
CUresult cuKernelSetAttribute(int attribute, int value, void *k, int device) {
    (void)attribute; (void)value; (void)device;
    return ask(k);
}
CUresult cuKernelSetCacheConfig(void *k, int config, int device) { (void)config; (void)device; return ask(k); }
CUresult cuKernelGetParamInfo(void *k, size_t index, size_t *offset, size_t *size) {
    (void)index; *offset = 0; *size = 8;
    return ask(k);
}
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, void *f, int size, size_t shared) {
    (void)size; (void)shared; *blocks = 1;
    return ask(f);
}
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *blocks, void *f, int size, size_t shared,
                                                              unsigned flags) {
    (void)flags;
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(blocks, f, size, shared);
}
CUresult cuOccupancyMaxPotentialBlockSize(int *grid, int *block, void *f, void *of, size_t shared, int limit) {
    (void)of; (void)shared; (void)limit; *grid = *block = 1;
    return ask(f);
}
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *grid, int *block, void *f, void *of, size_t shared, int limit,
                                                   unsigned flags) {
    (void)flags;
    return cuOccupancyMaxPotentialBlockSize(grid, block, f, of, shared, limit);
}
CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *shared, void *f, int blocks, int size) {
    (void)blocks; (void)size; *shared = 0;
    return ask(f);
}
CUresult cuOccupancyMaxPotentialClusterSize(int *size, void *f, const void *config) {
    (void)config; *size = 1;
    return ask(f);
}
CUresult cuOccupancyMaxActiveClusters(int *clusters, void *f, const void *config) {
    (void)config; *clusters = 1;
    return ask(f);
}

// ---- a private interface: the function of the first entry of a module ----

static CUresult first_function(void *module, void **function) {
    int i = find_object(module);
    if (i < 0 || objects[i].count == 0) return INVALID_HANDLE;
    *function = (void *)handle_of(i, 0, 0);
    return 0;
}
static const void *table[] = {(void *)(2 * sizeof(void *)), (void *)first_function};
CUresult cuGetExportTable(const void **out, const void *id) {
    (void)id;
    *out = table;
    return 0;
}

CUresult cuGetProcAddress(const char *symbol, void **pointer, int version, uint64_t flags) {
    (void)version; (void)flags;
    if (!strcmp(symbol, "cuLaunchKernel")) *pointer = (void *)cuLaunchKernel;
    else if (!strcmp(symbol, "cuFuncGetAttribute")) *pointer = (void *)cuFuncGetAttribute;
    else return NOT_FOUND;
    return 0;
}
