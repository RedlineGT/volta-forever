// A stand-in for NVIDIA's driver library that threads can share, for guard_stress.c: the layer loads
// it as the driver (VF_REAL_LIBCUDA), and the test also calls it directly for its handles, as a
// program takes them through a private interface of the driver. It keeps what guard_mock_driver.c
// keeps (modules and libraries with the names of their entries, functions, kernels, the module of a
// library in each context, contexts on a V100 and on a later GPU), under one lock, and gives the
// handles of what is gone again, last freed first, to any thread, as an allocator may.
//
// Counted for the test: the launches and questions that reach it (`launched`, `asked`), those of a
// function or kernel whose name starts with "bad" (`launched_bad`, `asked_bad`: a refused kernel
// that reached the driver), and the calls by which the layer asks what a handle is (`identified`).
// A context destroy can wait after it freed the handles and before it returns (`destroy_sleep_us`),
// and cuLibraryGetKernel can be held for one library (`hold_library`, while `held`; `holding` says
// that a call waits there), so that a test can choose how threads interleave.
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int CUresult;
enum { INVALID_VALUE = 1, INVALID_CONTEXT = 201, INVALID_HANDLE = 400, NOT_FOUND = 500 };

static pthread_mutex_t big = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
#define LOCK pthread_mutex_lock(&big)
#define UNLOCK pthread_mutex_unlock(&big)

volatile unsigned long launched, launched_bad, nodes, asked, asked_bad, identified, loads, unloads, kernel_functions;
volatile int destroy_sleep_us;
volatile uintptr_t hold_library;   // cuLibraryGetKernel on this library waits while `held` is 1
volatile int held, holding;

enum { MODULE = 1, LIBRARY, CONTEXT_MODULE };
#define MAXOBJ 4096
#define MAXFN 16384
#define NAMES 4
static struct obj {
    uintptr_t handle;
    int kind, live;
    uintptr_t context;
    int library;
    char names[NAMES][32];
    int count;
} objects[MAXOBJ];
static int object_hw;
static struct fn {
    uintptr_t handle;
    int owner, name, live, kernel;
} fns[MAXFN];
static int fn_hw;

// LIFO free lists of handle values.
static uintptr_t free_obj[MAXOBJ], free_fn[MAXFN], free_ctx[1024];
static int n_free_obj, n_free_fn, n_free_ctx;
static uintptr_t obj_counter = 0x10000, fn_counter = 0x100000, ctx_counter = 0x9000;

static _Thread_local uintptr_t current;
static struct {
    uintptr_t ctx;
    int device, live;
} ctxs[1024];

static int find_object(void *handle) {
    for (int i = 0; i < object_hw; ++i)
        if (objects[i].live && objects[i].handle == (uintptr_t)handle) return i;
    return -1;
}

static int new_object(int kind, const char *text) {
    int i = 0;
    while (i < object_hw && objects[i].live) ++i;
    if (i == object_hw) {
        if (object_hw == MAXOBJ) abort();
        ++object_hw;
    }
    uintptr_t handle = n_free_obj ? free_obj[--n_free_obj] : (obj_counter += 0x40);
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

static int find_fn(void *handle) {
    for (int i = 0; i < fn_hw; ++i)
        if (fns[i].live && fns[i].handle == (uintptr_t)handle) return i;
    return -1;
}

static uintptr_t handle_of(int owner, int name, int kernel) {
    for (int i = 0; i < fn_hw; ++i)
        if (fns[i].live && fns[i].owner == owner && fns[i].name == name && fns[i].kernel == kernel)
            return fns[i].handle;
    int i = 0;
    while (i < fn_hw && fns[i].live) ++i;
    if (i == fn_hw) {
        if (fn_hw == MAXFN) abort();
        ++fn_hw;
    }
    fns[i].handle = n_free_fn ? free_fn[--n_free_fn] : (fn_counter += 0x40);
    fns[i].owner = owner;
    fns[i].name = name;
    fns[i].live = 1;
    fns[i].kernel = kernel;
    return fns[i].handle;
}

static int name_index(int owner, const char *name) {
    for (int n = 0; n < objects[owner].count; ++n)
        if (!strcmp(objects[owner].names[n], name)) return n;
    return -1;
}

static void kill(int object) {
    if (!objects[object].live) return;
    objects[object].live = 0;
    free_obj[n_free_obj++] = objects[object].handle;
    for (int i = 0; i < fn_hw; ++i)
        if (fns[i].live && fns[i].owner == object) {
            fns[i].live = 0;
            free_fn[n_free_fn++] = fns[i].handle;
        }
    for (int i = 0; i < object_hw; ++i)
        if (objects[i].live && objects[i].kind == CONTEXT_MODULE && objects[i].library == object) kill(i);
}

static int context_module(int library) {
    for (int i = 0; i < object_hw; ++i)
        if (objects[i].live && objects[i].kind == CONTEXT_MODULE && objects[i].library == library &&
            objects[i].context == current)
            return i;
    int i = new_object(CONTEXT_MODULE, 0);
    objects[i].library = library;
    objects[i].count = objects[library].count;
    memcpy(objects[i].names, objects[library].names, sizeof objects[i].names);
    return i;
}

static int device_of(uintptr_t ctx) {
    for (int i = 0; i < 1024; ++i)
        if (ctxs[i].live && ctxs[i].ctx == ctx) return ctxs[i].device;
    return -1;
}

CUresult cuInit(unsigned flags) { (void)flags; return 0; }
CUresult cuDriverGetVersion(int *version) { *version = 13000; return 0; }
CUresult cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
CUresult cuDeviceGetCount(int *count) { *count = 2; return 0; }
CUresult cuDeviceComputeCapability(int *major, int *minor, int device) {
    *major = device == 1 ? 9 : 7;
    *minor = 0;
    return 0;
}
CUresult cuCtxCreate_v2(void **context, unsigned flags, int device) {
    (void)flags;
    LOCK;
    uintptr_t c = n_free_ctx ? free_ctx[--n_free_ctx] : (ctx_counter += 0x10);
    for (int i = 0; i < 1024; ++i)
        if (!ctxs[i].live) {
            ctxs[i].ctx = c;
            ctxs[i].device = device;
            ctxs[i].live = 1;
            break;
        }
    UNLOCK;
    *context = (void *)c;
    current = c;
    return 0;
}
static CUresult destroy(void *context) {
    LOCK;
    for (int i = 0; i < object_hw; ++i)
        if (objects[i].live && objects[i].kind != LIBRARY && objects[i].context == (uintptr_t)context) kill(i);
    for (int i = 0; i < 1024; ++i)
        if (ctxs[i].live && ctxs[i].ctx == (uintptr_t)context) {
            ctxs[i].live = 0;
            free_ctx[n_free_ctx++] = ctxs[i].ctx;
        }
    UNLOCK;
    if (current == (uintptr_t)context) current = 0;
    // The handles are free: another thread may be given them before this returns.
    if (destroy_sleep_us) usleep(destroy_sleep_us);
    return 0;
}
CUresult cuCtxDestroy_v2(void *context) { return destroy(context); }
CUresult cuCtxDestroy(void *context) { return destroy(context); }
CUresult cuGreenCtxDestroy(void *context) { return destroy(context); }
// (A green context is its own context here: what is loaded through the one given for it is its.)
CUresult cuCtxFromGreenCtx(void **context, void *green) { *context = green; return 0; }
// (The last detach of a context made with cuCtxCreate destroys it.)
CUresult cuCtxDetach(void *context) { return destroy(context); }
CUresult cuCtxGetCurrent(void **context) { *context = (void *)current; return 0; }
CUresult cuCtxSetCurrent(void *context) { current = (uintptr_t)context; return 0; }
CUresult cuCtxGetDevice(int *device) {
    if (!current) return INVALID_CONTEXT;
    LOCK;
    *device = device_of(current);
    UNLOCK;
    return *device < 0 ? INVALID_CONTEXT : 0;
}

CUresult cuModuleLoadData(void **module, const void *image) {
    LOCK;
    ++loads;
    *module = (void *)objects[new_object(MODULE, image)].handle;
    UNLOCK;
    return 0;
}
CUresult cuModuleLoadDataEx(void **module, const void *image, unsigned n, int *options, void **values) {
    (void)n; (void)options; (void)values;
    return cuModuleLoadData(module, image);
}
CUresult cuLibraryLoadData(void **library, const void *image, int *jo, void **jv, unsigned jc, int *lo, void **lv,
                           unsigned lc) {
    (void)jo; (void)jv; (void)jc; (void)lo; (void)lv; (void)lc;
    LOCK;
    ++loads;
    *library = (void *)objects[new_object(LIBRARY, image)].handle;
    UNLOCK;
    return 0;
}
CUresult cuModuleUnload(void *module) {
    LOCK;
    int i = find_object(module);
    if (i < 0) {
        UNLOCK;
        return INVALID_HANDLE;
    }
    ++unloads;
    kill(i);
    UNLOCK;
    return 0;
}
CUresult cuLibraryUnload(void *library) { return cuModuleUnload(library); }

CUresult cuModuleGetFunction(void **function, void *module, const char *name) {
    LOCK;
    int i = find_object(module), n = i < 0 ? -1 : name_index(i, name);
    CUresult r = 0;
    if (i < 0 || objects[i].kind == LIBRARY) r = INVALID_HANDLE;
    else if (n < 0) {
        *function = 0;
        r = NOT_FOUND;
    } else
        *function = (void *)handle_of(i, n, 0);
    UNLOCK;
    return r;
}
CUresult cuLibraryGetKernel(void **kernel, void *library, const char *name) {
    if (hold_library && (uintptr_t)library == hold_library) {
        holding = 1;
        while (held) usleep(100);
        holding = 0;
    }
    LOCK;
    int i = find_object(library), n = i < 0 ? -1 : name_index(i, name);
    CUresult r = 0;
    if (i < 0 || objects[i].kind != LIBRARY) r = INVALID_HANDLE;
    else if (n < 0) {
        *kernel = 0;
        r = NOT_FOUND;
    } else
        *kernel = (void *)handle_of(i, n, 1);
    UNLOCK;
    return r;
}
CUresult cuLibraryGetModule(void **module, void *library) {
    LOCK;
    int i = find_object(library);
    CUresult r = 0;
    if (i < 0 || objects[i].kind != LIBRARY) r = INVALID_HANDLE;
    else
        *module = (void *)objects[context_module(i)].handle;
    UNLOCK;
    return r;
}
CUresult cuKernelGetFunction(void **function, void *kernel) {
    LOCK;
    ++kernel_functions;
    int k = find_fn(kernel);
    CUresult r = 0;
    if (k < 0 || !fns[k].kernel) r = INVALID_HANDLE;
    else if (!current) r = INVALID_CONTEXT;
    else
        *function = (void *)handle_of(context_module(fns[k].owner), fns[k].name, 0);
    UNLOCK;
    return r;
}
CUresult cuFuncGetModule(void **module, void *function) {
    LOCK;
    ++identified;
    int f = find_fn(function);
    CUresult r = 0;
    if (f < 0 || fns[f].kernel) r = INVALID_HANDLE;
    else
        *module = (void *)objects[fns[f].owner].handle;
    UNLOCK;
    return r;
}
CUresult cuKernelGetLibrary(void **library, void *kernel) {
    LOCK;
    ++identified;
    int k = find_fn(kernel);
    CUresult r = 0;
    if (k < 0 || !fns[k].kernel) r = INVALID_HANDLE;
    else
        *library = (void *)objects[fns[k].owner].handle;
    UNLOCK;
    return r;
}
static CUresult name_of(const char **name, void *handle, int kernel) {
    LOCK;
    int h = find_fn(handle);
    CUresult r = 0;
    if (h < 0 || fns[h].kernel != kernel) r = INVALID_HANDLE;
    else
        *name = objects[fns[h].owner].names[fns[h].name]; // (stays readable: the array is static)
    UNLOCK;
    return r;
}
CUresult cuFuncGetName(const char **name, void *function) { return name_of(name, function, 0); }
CUresult cuKernelGetName(const char **name, void *kernel) { return name_of(name, kernel, 1); }

static CUresult list(void **out, unsigned capacity, void *owner, int kernel) {
    LOCK;
    int i = find_object(owner);
    if (i < 0) {
        UNLOCK;
        return INVALID_HANDLE;
    }
    for (unsigned n = 0; n < capacity && n < (unsigned)objects[i].count; ++n) out[n] = (void *)handle_of(i, n, kernel);
    UNLOCK;
    return 0;
}
static CUresult count_of(unsigned *count, void *owner) {
    LOCK;
    int i = find_object(owner);
    if (i >= 0) *count = objects[i].count;
    UNLOCK;
    return i < 0 ? INVALID_HANDLE : 0;
}
CUresult cuModuleGetFunctionCount(unsigned *count, void *module) { return count_of(count, module); }
CUresult cuLibraryGetKernelCount(unsigned *count, void *library) { return count_of(count, library); }
CUresult cuModuleEnumerateFunctions(void **out, unsigned capacity, void *module) { return list(out, capacity, module, 0); }
CUresult cuLibraryEnumerateKernels(void **out, unsigned capacity, void *library) {
    return list(out, capacity, library, 1);
}

static int is_bad(int h) { return !strncmp(objects[fns[h].owner].names[fns[h].name], "bad", 3); }

static CUresult launch(void *function) {
    LOCK;
    int h = find_fn(function);
    if (h < 0) {
        UNLOCK;
        return INVALID_HANDLE;
    }
    ++launched;
    if (is_bad(h)) ++launched_bad;
    UNLOCK;
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
static CUresult ask(void *handle) {
    LOCK;
    int h = find_fn(handle);
    if (h < 0) {
        UNLOCK;
        return INVALID_HANDLE;
    }
    ++asked;
    if (is_bad(h)) ++asked_bad;
    UNLOCK;
    return 0;
}
CUresult cuFuncGetAttribute(int *value, int attribute, void *f) { (void)attribute; *value = 1; return ask(f); }
CUresult cuKernelGetAttribute(int *value, int attribute, void *k, int device) {
    (void)attribute; (void)device; *value = 1;
    return ask(k);
}
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, void *f, int size, size_t shared) {
    (void)size; (void)shared; *blocks = 1;
    return ask(f);
}
CUresult cuDevicePrimaryCtxGetState(int device, unsigned *flags, int *active) {
    (void)device; *flags = 0; *active = 0;
    return 0;
}
CUresult cuGetProcAddress(const char *symbol, void **pointer, int version, uint64_t flags) {
    (void)version; (void)flags;
    if (!strcmp(symbol, "cuLaunchKernel")) *pointer = (void *)cuLaunchKernel;
    else if (!strcmp(symbol, "cuFuncGetAttribute")) *pointer = (void *)cuFuncGetAttribute;
    else return NOT_FOUND;
    return 0;
}
