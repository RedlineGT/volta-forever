// A stand-in for NVIDIA's driver library, for the tests of what the layer keeps over the life of a
// process (lifecycle.sh). The layer loads it as the driver (VF_REAL_LIBCUDA). Every module,
// library and context it makes is given a handle of its own, which nothing else is given: what the
// layer keeps of a module is dropped only when the layer is told that the module is gone.
//
// Contexts: one of its own at each creation, current in the thread that made it; the primary
// context of each device, made at its first retain, destroyed at its last release or at a reset.
// cuModuleGetFunction gives 16 * module + 1, cuLibraryGetKernel 16 * library + 2;
// cuLibraryGetModule a module of its own at each call, cuKernelGetFunction a function of a module of
// its own (cuFuncGetModule: function / 16; cuKernelGetLibrary: kernel / 16). A launch of a function of the first module
// loaded after a context was made is refused with 701 (out of resources), as the driver refuses a
// block that takes more registers than there are; a function of a later module is launched.
#include <stdint.h>

static uintptr_t new_handle(void) {
    static uintptr_t next = 16;
    return __atomic_fetch_add(&next, 1, __ATOMIC_RELAXED);
}

int cuInit(unsigned flags) { (void)flags; return 0; }

static _Thread_local uintptr_t current;
static int retains[4], active[4];
static int loads_in_context;
static char first_in_context[4096];
#define PRIMARY(device) ((void *)(uintptr_t)(8 + (device)))

int cuCtxCreate_v2(void **context, unsigned flags, int device) {
    (void)flags; (void)device;
    *context = (void *)new_handle();
    current = (uintptr_t)*context;
    loads_in_context = 0;
    return 0;
}
int cuCtxDestroy_v2(void *context) {
    if (current == (uintptr_t)context) current = 0;
    return 0;
}
int cuCtxDestroy(void *context) { return cuCtxDestroy_v2(context); }
int cuCtxGetCurrent(void **context) { *context = (void *)current; return 0; }
int cuCtxSetCurrent(void *context) { current = (uintptr_t)context; return 0; }
int cuDevicePrimaryCtxRetain(void **context, int device) {
    retains[device]++;
    active[device] = 1;
    *context = PRIMARY(device);
    return 0;
}
int cuDevicePrimaryCtxRelease_v2(int device) {
    if (!retains[device]) return 201;
    if (!--retains[device]) active[device] = 0;
    return 0;
}
int cuDevicePrimaryCtxRelease(int device) { return cuDevicePrimaryCtxRelease_v2(device); }
int cuDevicePrimaryCtxReset_v2(int device) { active[device] = 0; return 0; }
int cuDevicePrimaryCtxReset(int device) { return cuDevicePrimaryCtxReset_v2(device); }
int cuDevicePrimaryCtxGetState(int device, unsigned *flags, int *is_active) {
    *flags = 0;
    *is_active = active[device];
    return 0;
}

int cuModuleLoadData(void **module, const void *image) {
    (void)image;
    *module = (void *)new_handle();
    first_in_context[(uintptr_t)*module % 4096] = loads_in_context++ == 0;
    return 0;
}
int cuModuleUnload(void *module) { (void)module; return 0; }
int cuModuleGetFunction(void **function, void *module, const char *name) {
    (void)name;
    *function = (void *)((uintptr_t)module * 16 + 1);
    return 0;
}

int cuLaunchKernel(void *function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                   unsigned shared, void *stream, void **parameters, void **extra) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)shared; (void)stream; (void)parameters;
    (void)extra;
    return first_in_context[(uintptr_t)function / 16 % 4096] ? 701 : 0;
}

int cuLibraryLoadData(void **library, const void *image, int *jit_options, void **jit_values, unsigned jit_count,
                      int *options, void **values, unsigned count) {
    (void)image; (void)jit_options; (void)jit_values; (void)jit_count; (void)options; (void)values; (void)count;
    *library = (void *)new_handle();
    return 0;
}
int cuLibraryUnload(void *library) { (void)library; return 0; }
int cuLibraryGetKernel(void **kernel, void *library, const char *name) {
    (void)name;
    *kernel = (void *)((uintptr_t)library * 16 + 2);
    return 0;
}
int cuLibraryGetModule(void **module, void *library) {
    (void)library;
    *module = (void *)new_handle();
    return 0;
}
int cuKernelGetFunction(void **function, void *kernel) {
    (void)kernel;
    *function = (void *)(new_handle() * 16 + 1);
    return 0;
}
int cuFuncGetModule(void **module, void *function) {
    *module = (void *)((uintptr_t)function / 16);
    return 0;
}
int cuKernelGetLibrary(void **library, void *kernel) {
    *library = (void *)((uintptr_t)kernel / 16);
    return 0;
}
