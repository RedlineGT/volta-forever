// A stand-in for NVIDIA's driver library, for the test of the second translation at launch
// without a GPU (fit_api.sh). The layer loads it as the driver (VF_REAL_LIBCUDA). It counts the
// images loaded, the launches and the unloads, and refuses with 701 (out of resources) every
// launch of a kernel of the first image loaded, as the driver refuses a block that takes more
// registers than there are; a kernel of the second image (the layer's second translation) is
// launched.
//
// Handles: each load gives the next number (1, 2, 3, ...), and the test sets the count back to 0
// before each case, so the image that the program loads is 1 and the one the layer loads is 2.
// cuModuleGetFunction gives 256 * module + 1, cuLibraryGetKernel and cuLibraryEnumerateKernels
// 256 * library + 2; a launch is refused when the kernel's module is odd. cuLibraryGetModule gives
// an odd handle made from the library's, which no load gives: functions taken from it are
// refused too.
//
// Attributes are kept by handle, the name queries on enumerated kernels are counted, and a load
// or a setting can be paused from another thread. The next load can be given a handle chosen by
// the test (fit_mock_next_module), as the driver gives the handle of a module that is gone to
// another. There is one context, current in every thread (fit_mock_context), which
// cuCtxDestroy destroys only in name.
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>

// What the test program reads.
int fit_mock_loads = 0;
int fit_mock_launches = 0;
int fit_mock_unloads = 0;
// Name queries on enumerated kernels: total and those on a sentinel (handle the driver did not
// write).
int fit_mock_name_queries = 0;
int fit_mock_bad_name_queries = 0;
// Attribute tracking: fit_mock_require_attr, when non-zero, makes do_launch refuse a second image
// whose slot in fit_mock_attrs does not hold 65536.
int fit_mock_require_attr = 0;
int fit_mock_set_calls = 0;
int fit_mock_attrs[1024];
// A load that can be paused: the second load (fit_mock_loads == 2) spins until released.
_Atomic int fit_mock_pause = 0;
_Atomic int fit_mock_started = 0;
_Atomic int fit_mock_release = 0;

// The handle that the next load gives, when it is not 0.
uintptr_t fit_mock_next_module = 0;
#define fit_mock_context ((void *)0x7000)

int cuInit(unsigned flags) { (void)flags; return 0; }
int cuCtxGetCurrent(void **c) { *c = fit_mock_context; return 0; }
int cuCtxDestroy_v2(void *c) { (void)c; return 0; }
int cuCtxDestroy(void *c) { (void)c; return 0; }
int cuModuleLoadData(void **m, const void *p) {
    (void)p;
    int n = ++fit_mock_loads;
    *m = (void*)(fit_mock_next_module ? fit_mock_next_module : (uintptr_t)n);
    fit_mock_next_module = 0;
    if (n == 2 && fit_mock_pause) {
        fit_mock_started = 1;
        while (!fit_mock_release) usleep(1000);
    }
    return 0;
}
int cuModuleLoadDataEx(void **m, const void *p, unsigned n, int *o, void **v) {
    (void)n; (void)o; (void)v;
    return cuModuleLoadData(m, p);
}
int cuLibraryLoadData(void **m, const void *p, int *a, void **b, unsigned c, int *d, void **e, unsigned f) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f;
    return cuModuleLoadData(m, p);
}
static void hold_second(uintptr_t handle, int call);
int cuModuleGetFunction(void **f, void *m, const char *n) {
    (void)n;
    *f = (void*)((uintptr_t)m * 256 + 1);
    hold_second((uintptr_t)m * 256, 3);
    return 0;
}
int cuLibraryGetKernel(void **f, void *m, const char *n) {
    (void)n;
    *f = (void*)((uintptr_t)m * 256 + 2);
    return 0;
}
int cuKernelGetFunction(void **f, void *k) {
    *f = (void*)((uintptr_t)k - 1);
    return 0;
}
int cuLibraryGetModule(void **m, void *l) {
    *m = (void*)(uintptr_t)(2 * (uintptr_t)l + 255);
    return 0;
}
int cuLibraryGetKernelCount(unsigned *n, void *l) {
    (void)l; *n = 1; return 0;
}
int cuLibraryEnumerateKernels(void **k, unsigned count, void *l) {
    if (count) *k = (void*)((uintptr_t)l * 256 + 2);
    return 0;
}
int cuModuleGetGlobal(uint64_t *p, size_t *s, void *m, const char *n) {
    (void)n; (void)m; *p = 4096; *s = 4; return 0;
}
int cuModuleGetGlobal_v2(uint64_t *p, size_t *s, void *m, const char *n) {
    (void)n; (void)m; *p = 4096; *s = 4; return 0;
}
int cuLibraryGetGlobal(uint64_t *p, size_t *s, void *l, const char *n) {
    (void)n; (void)l; *p = 4096; *s = 4; return 0;
}
int cuModuleGetTexRef(void **t, void *m, const char *n) {
    (void)n; (void)m; *t = (void*)0x1000; return 0;
}
int cuModuleGetSurfRef(void **t, void *m, const char *n) {
    (void)n; (void)m; *t = (void*)0x2000; return 0;
}

// A call on a second image (an even module) that can be held: a setting when fit_mock_hold_second
// is 1, a launch when it is 2, a lookup of a function when it is 3. The first such call takes
// effect, then does not return until
// fit_mock_second_release is set. An unload of that module meanwhile is counted (the driver may
// then give its handle to another function, which the call would reach).
_Atomic int fit_mock_hold_second = 0;
_Atomic int fit_mock_second_held = 0;
_Atomic int fit_mock_second_release = 0;
_Atomic uintptr_t fit_mock_busy = 0;
int fit_mock_unloads_during_call = 0;
static void hold_second(uintptr_t handle, int call) {
    int wanted = call;
    if (!((handle / 256) & 1) && atomic_compare_exchange_strong(&fit_mock_hold_second, &wanted, 0)) {
        fit_mock_busy = handle / 256;
        fit_mock_second_held = 1;
        while (!fit_mock_second_release) usleep(1000);
        fit_mock_busy = 0;
    }
}

// A launch: refused when the kernel's module (handle / 256) is odd. When fit_mock_require_attr is
// non-zero, a second image (even module) also needs fit_mock_attrs[module] == 65536.
static int do_launch(void *f) {
    fit_mock_launches++;
    uintptr_t module_of = (uintptr_t)f / 256;
    if (module_of & 1) return 701;
    if (fit_mock_require_attr && fit_mock_attrs[module_of] != 65536) return 701;
    hold_second((uintptr_t)f, 2);
    return 0;
}
int cuLaunchKernel(void *f, unsigned a, unsigned b, unsigned c, unsigned d, unsigned e,
                   unsigned g, unsigned h, void *s, void **p, void **x) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)g;(void)h;(void)s;(void)p;(void)x;
    return do_launch(f);
}
int cuLaunchKernel_ptsz(void *f, unsigned a, unsigned b, unsigned c, unsigned d, unsigned e,
                        unsigned g, unsigned h, void *s, void **p, void **x) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)g;(void)h;(void)s;(void)p;(void)x;
    return do_launch(f);
}
int cuLaunchKernelEx(void *config, void *f, void **p, void **e) {
    (void)config;(void)p;(void)e;
    return do_launch(f);
}
int cuLaunchKernelEx_ptsz(void *config, void *f, void **p, void **e) {
    (void)config;(void)p;(void)e;
    return do_launch(f);
}
int cuFuncGetName(const char **name, void *f) {
    fit_mock_name_queries++;
    if ((uintptr_t)f == 0xdeadbeef) { fit_mock_bad_name_queries++; return 400; }
    (void)f; *name = "k_fit_regs"; return 0;
}
int cuKernelGetName(const char **name, void *k) {
    fit_mock_name_queries++;
    if ((uintptr_t)k == 0xdeadbeef) { fit_mock_bad_name_queries++; return 400; }
    (void)k; *name = "k_fit_regs"; return 0;
}
int cuModuleUnload(void *m) {
    fit_mock_unloads++;
    if ((uintptr_t)m == fit_mock_busy) fit_mock_unloads_during_call++;
    return 0;
}
int cuLibraryUnload(void *m) {
    fit_mock_unloads++;
    if ((uintptr_t)m == fit_mock_busy) fit_mock_unloads_during_call++;
    return 0;
}
// A setting that can be held: when fit_mock_hold_set is a value, the first call that sets that
// value on a first image (an odd module) takes effect, then does not return until
// fit_mock_set_release is set. (Two calls at once: the driver takes one, then the other, and
// they need not return in that order.)
_Atomic int fit_mock_hold_set = 0;
_Atomic int fit_mock_set_held = 0;
_Atomic int fit_mock_set_release = 0;
static void hold_set(uintptr_t handle, int v) {
    int wanted = v;
    if (v != 0 && ((handle / 256) & 1) && atomic_compare_exchange_strong(&fit_mock_hold_set, &wanted, 0)) {
        fit_mock_set_held = 1;
        while (!fit_mock_set_release) usleep(1000);
    }
}
int cuKernelSetAttribute(int a, int v, void *k, int d) {
    (void)a; (void)d;
    fit_mock_set_calls++;
    fit_mock_attrs[(uintptr_t)k / 256] = v;
    hold_set((uintptr_t)k, v);
    hold_second((uintptr_t)k, 1);
    return 0;
}
int cuFuncSetAttribute(void *f, int a, int v) {
    (void)a;
    fit_mock_set_calls++;
    fit_mock_attrs[(uintptr_t)f / 256] = v;
    hold_set((uintptr_t)f, v);
    hold_second((uintptr_t)f, 1);
    return 0;
}
int cuCtxGetDevice(int *dev) {
    *dev = 0;
    return 0;
}
int cuGetProcAddress(const char *name, void **p, int version, unsigned long long flags) {
    (void)version;
    if (!strcmp(name, "cuLaunchKernel")) {
        *p = flags == 2 ? (void*)cuLaunchKernel_ptsz : (void*)cuLaunchKernel;
        return 0;
    }
    if (!strcmp(name, "cuLaunchKernelEx")) {
        *p = flags == 2 ? (void*)cuLaunchKernelEx_ptsz : (void*)cuLaunchKernelEx;
        return 0;
    }
    if (!strcmp(name, "cuModuleGetGlobal")) {
        *p = (void*)cuModuleGetGlobal_v2;
        return 0;
    }
    return 801;
}
