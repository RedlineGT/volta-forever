// The second translation at launch (docs/architecture.md, "Launch"), by every way a program can
// come to it, without a GPU.
//
// A stand-in for the driver (fit_mock_driver.c) refuses with 701 every launch of a kernel of the
// first image loaded, and launches a kernel of the second. The layer translates the cubin and
// loads it through the stand-in; at the refused launch it must translate again, load that, and
// launch from there, or leave the refusal to the program where it may not. Each case says what
// the program gets, how many images the driver was given and how many launches it saw:
//   - a variable or a surface reference of the module asked for by name, by either exported
//     name of cuModuleGetGlobal and by what cuGetProcAddress gives for it: no second
//     translation after that (it would have its own copy of the data);
//   - the launch entries for the per-thread default stream (cuLaunchKernel_ptsz,
//     cuLaunchKernelEx_ptsz), called and taken from cuGetProcAddress: launched from the second
//     translation as by the plain entries;
//   - a library's kernels taken by enumeration, and functions taken from the module that the
//     driver gives for a library: the same;
//   - a library and that module are one image: a variable asked for through either excludes
//     the kernels taken through the other, and the module's second translation is unloaded
//     with the library;
//   - an unload of the module while the second translation is given the kernel's settings at a
//     refused launch waits for the launch: the second translation is not unloaded under the
//     setting (its handle could be given to another function), and the launch is made; nor while
//     the launch from it is being made, nor while a setting that the program makes is given to it,
//     nor while its kernel is being looked up there at the first refused launch;
//   - a second translation made for a module that the program unloads meanwhile, and whose
//     handle the driver gives to the next module loaded, is not that module's: it is unloaded,
//     and that module's refused launch makes one of its own;
//   - a second translation made for a module whose context is destroyed meanwhile went with the
//     context: it is not unloaded (its handle may be another module's by then).
//
// usage: fit_api PROXY MOCK CUBIN
// PROXY is the layer's libcuda.so.1, MOCK the stand-in, CUBIN an sm_75 cubin with the kernel
// k_fit_regs (tests/kernels/k_fit_test.135.sm75.cubin).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <linux/futex.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef int CUresult;
typedef struct { unsigned grid[3], block[3]; unsigned smem; void *stream, *attrs; unsigned nattrs; } CUlaunchConfig;

static void *lib, *mock;
static int checks = 0, wrong = 0;

static void *sym(const char *n) {
    void *p = dlsym(lib, n);
    if (!p) { fprintf(stderr, "missing symbol in proxy: %s\n", n); exit(2); }
    return p;
}
static void reset_counters(void) {
    *(int*)dlsym(mock, "fit_mock_loads") = 0;
    *(int*)dlsym(mock, "fit_mock_launches") = 0;
    *(int*)dlsym(mock, "fit_mock_unloads") = 0;
    *(int*)dlsym(mock, "fit_mock_name_queries") = 0;
    *(int*)dlsym(mock, "fit_mock_bad_name_queries") = 0;
    *(int*)dlsym(mock, "fit_mock_set_calls") = 0;
    *(int*)dlsym(mock, "fit_mock_require_attr") = 0;
    memset(dlsym(mock, "fit_mock_attrs"), 0, 1024 * sizeof(int));
    *(_Atomic int*)dlsym(mock, "fit_mock_pause") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_started") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_release") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_hold_set") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_set_held") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_set_release") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_hold_second") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_second_held") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_second_release") = 0;
    *(int*)dlsym(mock, "fit_mock_unloads_during_call") = 0;
    *(uintptr_t*)dlsym(mock, "fit_mock_next_module") = 0;
}
static int mock_loads(void) { return *(int*)dlsym(mock, "fit_mock_loads"); }
static int mock_unloads(void) { return *(int*)dlsym(mock, "fit_mock_unloads"); }
static int mock_launches(void) { return *(int*)dlsym(mock, "fit_mock_launches"); }

#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuModuleLoadData, void**, const void*);
SYM(CUresult, cuModuleGetFunction, void**, void*, const char*);
SYM(CUresult, cuModuleGetGlobal, uint64_t*, size_t*, void*, const char*);
SYM(CUresult, cuModuleGetGlobal_v2, uint64_t*, size_t*, void*, const char*);
SYM(CUresult, cuModuleGetSurfRef, void**, void*, const char*);
SYM(CUresult, cuModuleUnload, void*);
SYM(CUresult, cuLibraryLoadData, void**, const void*, int*, void**, unsigned, int*, void**, unsigned);
SYM(CUresult, cuLibraryGetKernel, void**, void*, const char*);
SYM(CUresult, cuLibraryGetGlobal, uint64_t*, size_t*, void*, const char*);
SYM(CUresult, cuLibraryUnload, void*);
SYM(CUresult, cuLibraryEnumerateKernels, void**, unsigned, void*);
SYM(CUresult, cuLibraryGetModule, void**, void*);
SYM(CUresult, cuLaunchKernel, void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    unsigned, void*, void**, void**);
SYM(CUresult, cuLaunchKernel_ptsz, void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    unsigned, void*, void**, void**);
SYM(CUresult, cuLaunchKernelEx, CUlaunchConfig*, void*, void**, void**);
SYM(CUresult, cuLaunchKernelEx_ptsz, CUlaunchConfig*, void*, void**, void**);
SYM(CUresult, cuGetProcAddress, const char*, void**, int, unsigned long long);
SYM(CUresult, cuKernelSetAttribute, int, int, void*, int);
SYM(CUresult, cuFuncSetAttribute, void*, int, int);
SYM(CUresult, cuLibraryGetKernelCount, unsigned*, void*);
SYM(CUresult, cuCtxDestroy_v2, void*);

#define LOAD(name) *(void**)&name = sym(#name)

// A case: what the program got, and what the driver saw, against what they must be.
static void check(const char *name, CUresult result, int expect_result, int expect_loads, int expect_launches) {
    checks++;
    int loads = mock_loads(), launches = mock_launches();
    int ok = (result == expect_result) && (loads == expect_loads) && (launches == expect_launches);
    if (!ok) wrong++;
    printf("%s  %-50s result=%d (want %d)  loads=%d (want %d)  launches=%d (want %d)\n",
           ok ? "ok   " : "WRONG", name, result, expect_result, loads, expect_loads, launches, expect_launches);
}
static void check_unloads(const char *name, int expect) {
    checks++;
    int ok = mock_unloads() == expect;
    if (!ok) wrong++;
    printf("%s  %-50s unloads=%d (want %d)\n", ok ? "ok   " : "WRONG", name, mock_unloads(), expect);
}

struct launch_args { void *func; CUresult *result; };
// An unload made from a thread.
struct unload_args { void *module; _Atomic int tid, done; };
static void *unload_thread_fn(void *arg) {
    struct unload_args *p = arg;
    p->tid = (int)syscall(SYS_gettid);
    cuModuleUnload(p->module);
    p->done = 1;
    return NULL;
}
static void *launch_thread_fn(void *arg) {
    struct launch_args *p = arg;
    *p->result = cuLaunchKernel(p->func, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    return NULL;
}

// A setting made from a thread: on a function (kernel == 0) or on a library's kernel.
struct set_args { void *handle; int kernel; int value; _Atomic int tid, done; CUresult result; };
static void *set_thread_fn(void *arg) {
    struct set_args *p = arg;
    p->tid = (int)syscall(SYS_gettid);
    if (p->kernel) p->result = cuKernelSetAttribute(8, p->value, p->handle, 0);
    else p->result = cuFuncSetAttribute(p->handle, 8, p->value);
    p->done = 1;
    return NULL;
}

// A reaches its pause inside the mock before B starts. The worker has no other
// blocking calls between publishing its tid and returning from the setter. In
// the serialized proxy B waits on the setter mutex; without it B completes.
// A thread delayed before entering the setter satisfies neither observation.
static int waiting_on_setter(int tid) {
    if (!tid) return 0;
    char path[64];
    snprintf(path, sizeof path, "/proc/self/task/%d/syscall", tid);
    FILE *f = fopen(path, "r");
    if (!f) return 0;  // The worker may already have exited.
    long nr;
    unsigned long address, op;
    int parsed = fscanf(f, "%ld %lx %lx", &nr, &address, &op);
    fclose(f);
    if (parsed != 3 || nr != SYS_futex) return 0;
    unsigned long command = op & FUTEX_CMD_MASK;
    return command == FUTEX_WAIT || command == FUTEX_WAIT_BITSET;
}

// Two settings of one kernel at once (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 32768 and
// 65536), the kernel's second image being there already: the driver takes 32768 and keeps that
// call from returning, takes 65536 from the other thread if that call gets to it, then lets the
// first one return. Whatever the order, once both calls are over the first image and the second
// hold the same value, and a launch that needs the first image's value is made from the second.
static void two_settings_at_once(const char *name, void *handle, int kernel, void *launched,
                                 int first_module, int second_module) {
    int *attrs = dlsym(mock, "fit_mock_attrs");
    _Atomic int *held = dlsym(mock, "fit_mock_set_held");
    _Atomic int *release = dlsym(mock, "fit_mock_set_release");
    *held = 0;
    *release = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_hold_set") = 32768;
    struct set_args a = { .handle = handle, .kernel = kernel, .value = 32768 };
    struct set_args b = { .handle = handle, .kernel = kernel, .value = 65536 };
    pthread_t ta, tb;
    if (pthread_create(&ta, NULL, set_thread_fn, &a)) {
        fprintf(stderr, "%s: could not start first setter\n", name);
        exit(2);
    }
    for (int i = 0; i < 10000 && !*held; i++) usleep(1000);
    if (!*held) {
        fprintf(stderr, "%s: first setter did not reach its pause\n", name);
        exit(2);
    }
    if (pthread_create(&tb, NULL, set_thread_fn, &b)) {
        fprintf(stderr, "%s: could not start second setter\n", name);
        exit(2);
    }
    int observed = 0;
    for (int i = 0; i < 10000; i++) {
        if (b.done || waiting_on_setter(b.tid)) {
            observed = 1;
            break;
        }
        usleep(1000);
    }
    *release = 1;
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    if (!observed) {
        fprintf(stderr, "%s: could not observe the second setter while the first was held\n", name);
        exit(2);
    }
    char label[96];
    checks++;
    {
        int ok = !a.result && !b.result && attrs[first_module] == 65536 && attrs[second_module] == 65536;
        if (!ok) wrong++;
        snprintf(label, sizeof label, "%s_same_value", name);
        printf("%s  %-50s first image=%d second image=%d (want both 65536)\n",
               ok ? "ok   " : "WRONG", label, attrs[first_module], attrs[second_module]);
    }
    *(int*)dlsym(mock, "fit_mock_require_attr") = attrs[first_module] == 65536;
    int launches = mock_launches();
    CUresult r = cuLaunchKernel(launched, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    snprintf(label, sizeof label, "%s_launch", name);
    check(label, r, 0, 2, launches + 2);
    *(int*)dlsym(mock, "fit_mock_require_attr") = 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: fit_api PROXY_PATH MOCK_PATH CUBIN_PATH\n");
        return 2;
    }
    const char *proxy_path = argv[1], *mock_path = argv[2], *cubin_path = argv[3];

    mock = dlopen(mock_path, RTLD_NOW);
    if (!mock) { fprintf(stderr, "cannot open mock: %s\n", dlerror()); return 2; }
    lib = dlopen(proxy_path, RTLD_NOW);
    if (!lib) { fprintf(stderr, "cannot open proxy: %s\n", dlerror()); return 2; }

    LOAD(cuModuleLoadData); LOAD(cuModuleGetFunction);
    LOAD(cuModuleGetGlobal); LOAD(cuModuleGetGlobal_v2); LOAD(cuModuleGetSurfRef);
    LOAD(cuModuleUnload);
    LOAD(cuLibraryLoadData); LOAD(cuLibraryGetKernel); LOAD(cuLibraryGetGlobal); LOAD(cuLibraryUnload);
    LOAD(cuLibraryEnumerateKernels); LOAD(cuLibraryGetModule);
    LOAD(cuLaunchKernel); LOAD(cuLaunchKernel_ptsz);
    LOAD(cuLaunchKernelEx); LOAD(cuLaunchKernelEx_ptsz);
    LOAD(cuGetProcAddress);
    LOAD(cuKernelSetAttribute); LOAD(cuFuncSetAttribute);
    LOAD(cuLibraryGetKernelCount); LOAD(cuCtxDestroy_v2);

    FILE *f = fopen(cubin_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", cubin_path); return 2; }
    fseek(f, 0, SEEK_END); size_t size = ftell(f); fseek(f, 0, SEEK_SET);
    void *image = malloc(size);
    if (fread(image, 1, size, f) != size) { fprintf(stderr, "read error\n"); return 2; }
    fclose(f);

    CUresult r;
    void *module, *function, *library, *kernel;
    CUlaunchConfig config = {{1,1,1},{1024,1,1},0,NULL,NULL,0};

    // ---- a variable of the module asked for by name: no second translation after that ----

    // By the older exported name.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) { fprintf(stderr, "load failed: %d\n", r); return 1; }
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) { fprintf(stderr, "getfunc failed: %d\n", r); return 1; }
    { uint64_t dptr; size_t sz; cuModuleGetGlobal(&dptr, &sz, module, "fit_scale"); }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("global_v1_marks_module", r, 701, 1, 1);
    cuModuleUnload(module);

    // By the name that the headers of CUDA 13 give the call (cuModuleGetGlobal_v2).
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    { uint64_t dptr; size_t sz; cuModuleGetGlobal_v2(&dptr, &sz, module, "fit_scale"); }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("global_v2_marks_module", r, 701, 1, 1);
    cuModuleUnload(module);

    // By what cuGetProcAddress gives for "cuModuleGetGlobal".
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    {
        void *resolved;
        cuGetProcAddress("cuModuleGetGlobal", &resolved, 12000, 0);
        typedef CUresult (*GlobalFn)(uint64_t*, size_t*, void*, const char*);
        uint64_t dptr; size_t sz;
        ((GlobalFn)resolved)(&dptr, &sz, module, "fit_scale");
    }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("global_resolved_marks_module", r, 701, 1, 1);
    cuModuleUnload(module);

    // A surface reference.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    { void *sref; cuModuleGetSurfRef(&sref, module, "some_surface"); }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("surfref_marks_module", r, 701, 1, 1);
    cuModuleUnload(module);

    // Nothing asked for: the launch is made from a second image.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("no_global_allows_retry", r, 0, 2, 2);
    cuModuleUnload(module);

    // ---- the launch entries for the per-thread default stream ----

    // cuLaunchKernel_ptsz, called.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    r = cuLaunchKernel_ptsz(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("launch_ptsz_direct", r, 0, 2, 2);
    cuModuleUnload(module);

    // cuLaunchKernelEx_ptsz, called.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    r = cuLaunchKernelEx_ptsz(&config, function, NULL, NULL);
    check("launch_ex_ptsz_direct", r, 0, 2, 2);
    cuModuleUnload(module);

    // What cuGetProcAddress gives for "cuLaunchKernel" with the flag of the per-thread stream (2).
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    {
        void *resolved;
        cuGetProcAddress("cuLaunchKernel", &resolved, 12000, 2);
        typedef CUresult (*Fn)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, void*, void**, void**);
        r = ((Fn)resolved)(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    }
    check("launch_ptsz_resolved", r, 0, 2, 2);
    cuModuleUnload(module);

    // The same for "cuLaunchKernelEx".
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    {
        void *resolved;
        cuGetProcAddress("cuLaunchKernelEx", &resolved, 12000, 2);
        typedef CUresult (*Fn)(CUlaunchConfig*, void*, void**, void**);
        r = ((Fn)resolved)(&config, function, NULL, NULL);
    }
    check("launch_ex_ptsz_resolved", r, 0, 2, 2);
    cuModuleUnload(module);

    // ---- kernels of a library ----

    // Taken by name.
    reset_counters();
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryGetKernel(&kernel, library, "k_fit_regs"); if (r) return 1;
    r = cuLaunchKernel(kernel, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("library_get_kernel", r, 0, 2, 2);
    cuLibraryUnload(library);

    // Taken by enumeration.
    reset_counters();
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryEnumerateKernels(&kernel, 1, library); if (r) return 1;
    r = cuLaunchKernel(kernel, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("library_enumerate", r, 0, 2, 2);
    cuLibraryUnload(library);

    // A function of the module that the driver gives for the library. Its second translation is
    // a module of its own, which goes with the library: two unloads.
    reset_counters();
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    { void *mod; cuLibraryGetModule(&mod, library); cuModuleGetFunction(&function, mod, "k_fit_regs"); }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("library_to_module", r, 0, 2, 2);
    cuLibraryUnload(library);
    check_unloads("library_to_module_unloaded", 2);

    // A library and the module the driver gives for it are one image under two handles. A
    // variable asked for through the library: no second translation for a function of the module.
    reset_counters();
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    { void *mod; cuLibraryGetModule(&mod, library); cuModuleGetFunction(&function, mod, "k_fit_regs"); }
    { uint64_t dptr; size_t sz; cuLibraryGetGlobal(&dptr, &sz, library, "fit_scale"); }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("library_global_excludes_module", r, 701, 1, 1);
    cuLibraryUnload(library);

    // And one asked for through the module: none for a kernel of the library.
    reset_counters();
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryGetKernel(&kernel, library, "k_fit_regs"); if (r) return 1;
    { void *mod; uint64_t dptr; size_t sz; cuLibraryGetModule(&mod, library); cuModuleGetGlobal_v2(&dptr, &sz, mod, "fit_scale"); }
    r = cuLaunchKernel(kernel, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("module_global_excludes_library", r, 701, 1, 1);
    cuLibraryUnload(library);

    // The same when the kernel had already been launched from a second translation.
    reset_counters();
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryGetKernel(&kernel, library, "k_fit_regs"); if (r) return 1;
    r = cuLaunchKernel(kernel, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("library_kernel_before_module_global", r, 0, 2, 2);
    { void *mod; uint64_t dptr; size_t sz; cuLibraryGetModule(&mod, library); cuModuleGetGlobal_v2(&dptr, &sz, mod, "fit_scale"); }
    r = cuLaunchKernel(kernel, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("library_kernel_after_module_global", r, 701, 2, 3);
    cuLibraryUnload(library);

    // ---- enumeration with capacity larger than the number of kernels ----

    // A library with one kernel, enumerated with capacity 2: the second slot is left as the caller
    // set it (a sentinel), and must not be read by the proxy. The name query count says whether the
    // proxy asked the driver for the name of the sentinel.
    reset_counters();
    *(int*)dlsym(mock, "fit_mock_name_queries") = 0;
    *(int*)dlsym(mock, "fit_mock_bad_name_queries") = 0;
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    {
        void *kbuf[2]; kbuf[0] = NULL; kbuf[1] = (void*)(uintptr_t)0xdeadbeef;
        r = cuLibraryEnumerateKernels(kbuf, 2, library);
        checks++;
        int bad_queries = *(int*)dlsym(mock, "fit_mock_bad_name_queries");
        int ok = (r == 0) && (bad_queries == 0);
        if (!ok) wrong++;
        printf("%s  %-50s result=%d, bad name queries=%d (want 0)\n",
               ok ? "ok   " : "WRONG", "enum_capacity_larger_than_count", r, bad_queries);
        // The kernel from the first slot is launchable from a second translation.
        if (kbuf[0]) {
            r = cuLaunchKernel(kbuf[0], 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
            check("enum_valid_kernel_launches", r, 0, 2, 2);
        }
        // The sentinel was not touched.
        checks++;
        ok = (kbuf[1] == (void*)(uintptr_t)0xdeadbeef);
        if (!ok) wrong++;
        printf("%s  %-50s sentinel=%p (want 0xdeadbeef)\n",
               ok ? "ok   " : "WRONG", "enum_sentinel_untouched", kbuf[1]);
    }
    cuLibraryUnload(library);

    // ---- attributes of a library's kernel follow to the function of the library's module ----

    // A library's kernel gets an attribute; a function of the library's module with the same name
    // must receive that attribute on its second image. The mock requires that the attribute has
    // been set before it accepts the launch.
    reset_counters();
    *(int*)dlsym(mock, "fit_mock_set_calls") = 0;
    *(int*)dlsym(mock, "fit_mock_require_attr") = 1;
    memset(dlsym(mock, "fit_mock_attrs"), 0, 1024 * sizeof(int));
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryGetKernel(&kernel, library, "k_fit_regs"); if (r) return 1;
    // Set the attribute on the library's kernel (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8).
    cuKernelSetAttribute(8, 65536, kernel, 0);
    // Take the module that the driver gives for the library, and a function of it.
    { void *mod; cuLibraryGetModule(&mod, library); cuModuleGetFunction(&function, mod, "k_fit_regs"); }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("attr_kernel_to_module_function", r, 0, 2, 2);
    *(int*)dlsym(mock, "fit_mock_require_attr") = 0;
    cuLibraryUnload(library);

    // The function-level override wins: K at 65536, F set to 0 with cuFuncSetAttribute, the second
    // image has 0 and the launch is refused.
    reset_counters();
    *(int*)dlsym(mock, "fit_mock_set_calls") = 0;
    *(int*)dlsym(mock, "fit_mock_require_attr") = 1;
    memset(dlsym(mock, "fit_mock_attrs"), 0, 1024 * sizeof(int));
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryGetKernel(&kernel, library, "k_fit_regs"); if (r) return 1;
    cuKernelSetAttribute(8, 65536, kernel, 0);
    { void *mod; cuLibraryGetModule(&mod, library); cuModuleGetFunction(&function, mod, "k_fit_regs"); }
    // Override with cuFuncSetAttribute to 0 on the function: takes priority.
    cuFuncSetAttribute(function, 8, 0);
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("attr_function_overrides_kernel", r, 701, 2, 2);
    *(int*)dlsym(mock, "fit_mock_require_attr") = 0;
    cuLibraryUnload(library);

    // A setting of the library's kernel for another device than the current context's (the
    // stand-in's contexts are on device 0) is not the function's: its second image does not get
    // it, and the launch is refused there too.
    reset_counters();
    *(int*)dlsym(mock, "fit_mock_require_attr") = 1;
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryGetKernel(&kernel, library, "k_fit_regs"); if (r) return 1;
    cuKernelSetAttribute(8, 65536, kernel, 1);
    { void *mod; cuLibraryGetModule(&mod, library); cuModuleGetFunction(&function, mod, "k_fit_regs"); }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("attr_kernel_of_another_device", r, 701, 2, 2);
    *(int*)dlsym(mock, "fit_mock_require_attr") = 0;
    cuLibraryUnload(library);

    // A setting of the library's kernel made after the function's second image exists reaches
    // it at the next refused launch: refused without it, launched with it.
    reset_counters();
    *(int*)dlsym(mock, "fit_mock_require_attr") = 1;
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryGetKernel(&kernel, library, "k_fit_regs"); if (r) return 1;
    { void *mod; cuLibraryGetModule(&mod, library); cuModuleGetFunction(&function, mod, "k_fit_regs"); }
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("attr_kernel_not_set_yet", r, 701, 2, 2);
    cuKernelSetAttribute(8, 65536, kernel, 0);
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("attr_kernel_set_later", r, 0, 2, 4);
    *(int*)dlsym(mock, "fit_mock_require_attr") = 0;
    cuLibraryUnload(library);

    // ---- two settings of one kernel at once leave it and its second image with one value ----

    // On a function of a module.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("attr_two_at_once_function_first_launch", r, 0, 2, 2);
    two_settings_at_once("attr_two_at_once_function", function, 0, function, 1, 2);
    cuModuleUnload(module);

    // On a library's kernel.
    reset_counters();
    r = cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0); if (r) return 1;
    r = cuLibraryGetKernel(&kernel, library, "k_fit_regs"); if (r) return 1;
    r = cuLaunchKernel(kernel, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("attr_two_at_once_kernel_first_launch", r, 0, 2, 2);
    two_settings_at_once("attr_two_at_once_kernel", kernel, 1, kernel, 1, 2);
    cuLibraryUnload(library);

    // ---- data exclusion is atomic with the launch in progress ----

    // Thread A: first launch is refused (701), starts the second load (which the mock pauses).
    // Thread B: cuModuleGetGlobal_v2 returns and asked_for_data completes.
    // Thread B releases the paused load. Thread A's launch must be refused (701): the commit point
    // prevents it from launching a second image after asked_for_data returned. A later launch is
    // also refused.
    reset_counters();
    *(_Atomic int*)dlsym(mock, "fit_mock_pause") = 1;
    *(_Atomic int*)dlsym(mock, "fit_mock_started") = 0;
    *(_Atomic int*)dlsym(mock, "fit_mock_release") = 0;
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    // Thread A: launch in a thread (the second load will pause).
    CUresult thread_result = -1;
    struct launch_args targ = { function, &thread_result };
    pthread_t thread;
    pthread_create(&thread, NULL, launch_thread_fn, &targ);
    // Wait for the second load to reach the pause point.
    for (int i = 0; i < 10000 && !*(_Atomic int*)dlsym(mock, "fit_mock_started"); i++) usleep(1000);
    // Thread B: asked_for_data.
    { uint64_t dptr; size_t sz; cuModuleGetGlobal_v2(&dptr, &sz, module, "fit_scale"); }
    // Release the paused load.
    *(_Atomic int*)dlsym(mock, "fit_mock_release") = 1;
    pthread_join(thread, NULL);
    checks++;
    {
        int ok = (thread_result == 701);
        if (!ok) wrong++;
        printf("%s  %-50s overlapping launch=%d (want 701)\n",
               ok ? "ok   " : "WRONG", "data_during_fit_overlapping", thread_result);
    }
    // A later launch: also refused (the kernel is unavailable after asked_for_data).
    // Loads: 2 (the program's image and the proxy's second image that was built before the
    // exclusion took effect). Launches: 2 (the first failed launch and this one, which the proxy
    // blocks before reaching the driver).
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("data_during_fit_later", r, 701, 2, 2);
    cuModuleUnload(module);

    // ---- an unload while the second translation is given the kernel's settings ----

    // Thread A: the kernel has a setting; its first launch is refused, the second translation is
    // made, and the setting given to it is held in the stand-in. Thread B: the program unloads
    // the module. B's unload of the second translation waits for A's launch, which is made.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    cuFuncSetAttribute(function, 8, 65536);
    *(_Atomic int*)dlsym(mock, "fit_mock_hold_second") = 1;
    {
        CUresult launched = -1;
        struct launch_args a = { function, &launched };
        struct unload_args b = { .module = module };
        pthread_t ta, tb;
        pthread_create(&ta, NULL, launch_thread_fn, &a);
        _Atomic int *held = dlsym(mock, "fit_mock_second_held");
        for (int i = 0; i < 10000 && !*held; i++) usleep(1000);
        if (!*held) { fprintf(stderr, "unload_during_settings: the setting was not held\n"); return 2; }
        pthread_create(&tb, NULL, unload_thread_fn, &b);
        int observed = 0;
        for (int i = 0; i < 10000 && !observed; i++) {
            observed = b.done || waiting_on_setter(b.tid);
            if (!observed) usleep(1000);
        }
        *(_Atomic int*)dlsym(mock, "fit_mock_second_release") = 1;
        pthread_join(ta, NULL);
        pthread_join(tb, NULL);
        if (!observed) { fprintf(stderr, "unload_during_settings: could not observe the unload\n"); return 2; }
        int during = *(int*)dlsym(mock, "fit_mock_unloads_during_call");
        checks++;
        int ok = during == 0;
        if (!ok) wrong++;
        printf("%s  %-50s unloads during the setting=%d (want 0)\n", ok ? "ok   " : "WRONG",
               "unload_during_settings_waits", during);
        check("unload_during_settings_launch", launched, 0, 2, 2);
        check_unloads("unload_during_settings_unloaded", 2);
    }

    // The same while the launch from the second translation is being made.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    *(_Atomic int*)dlsym(mock, "fit_mock_hold_second") = 2;
    {
        CUresult launched = -1;
        struct launch_args a = { function, &launched };
        struct unload_args b = { .module = module };
        pthread_t ta, tb;
        pthread_create(&ta, NULL, launch_thread_fn, &a);
        _Atomic int *held = dlsym(mock, "fit_mock_second_held");
        for (int i = 0; i < 10000 && !*held; i++) usleep(1000);
        if (!*held) { fprintf(stderr, "unload_during_launch: the launch was not held\n"); return 2; }
        pthread_create(&tb, NULL, unload_thread_fn, &b);
        int observed = 0;
        for (int i = 0; i < 10000 && !observed; i++) {
            observed = b.done || waiting_on_setter(b.tid);
            if (!observed) usleep(1000);
        }
        *(_Atomic int*)dlsym(mock, "fit_mock_second_release") = 1;
        pthread_join(ta, NULL);
        pthread_join(tb, NULL);
        if (!observed) { fprintf(stderr, "unload_during_launch: could not observe the unload\n"); return 2; }
        int during = *(int*)dlsym(mock, "fit_mock_unloads_during_call");
        checks++;
        int ok = during == 0 && launched == 0;
        if (!ok) wrong++;
        printf("%s  %-50s unloads during the launch=%d (want 0), launch=%d (want 0)\n",
               ok ? "ok   " : "WRONG", "unload_during_launch_waits", during, launched);
    }

    // The same while a setting that the program makes on the kernel is given to its second
    // translation, which a launch made before.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    r = cuLaunchKernel(function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
    check("unload_during_own_setting_first_launch", r, 0, 2, 2);
    *(_Atomic int*)dlsym(mock, "fit_mock_hold_second") = 1;
    {
        struct set_args a = { .handle = function, .kernel = 0, .value = 32768 };
        struct unload_args b = { .module = module };
        pthread_t ta, tb;
        pthread_create(&ta, NULL, set_thread_fn, &a);
        _Atomic int *held = dlsym(mock, "fit_mock_second_held");
        for (int i = 0; i < 10000 && !*held; i++) usleep(1000);
        if (!*held) { fprintf(stderr, "unload_during_own_setting: the setting was not held\n"); return 2; }
        pthread_create(&tb, NULL, unload_thread_fn, &b);
        int observed = 0;
        for (int i = 0; i < 10000 && !observed; i++) {
            observed = b.done || waiting_on_setter(b.tid);
            if (!observed) usleep(1000);
        }
        *(_Atomic int*)dlsym(mock, "fit_mock_second_release") = 1;
        pthread_join(ta, NULL);
        pthread_join(tb, NULL);
        if (!observed) { fprintf(stderr, "unload_during_own_setting: could not observe the unload\n"); return 2; }
        int during = *(int*)dlsym(mock, "fit_mock_unloads_during_call");
        checks++;
        int ok = during == 0 && a.result == 0;
        if (!ok) wrong++;
        printf("%s  %-50s unloads during the setting=%d (want 0), setting=%d (want 0)\n",
               ok ? "ok   " : "WRONG", "unload_during_own_setting_waits", during, a.result);
    }

    // The same while the kernel is looked up in the second translation, just made at the first
    // refused launch.
    reset_counters();
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    *(_Atomic int*)dlsym(mock, "fit_mock_hold_second") = 3;
    {
        CUresult launched = -1;
        struct launch_args a = { function, &launched };
        struct unload_args b = { .module = module };
        pthread_t ta, tb;
        pthread_create(&ta, NULL, launch_thread_fn, &a);
        _Atomic int *held = dlsym(mock, "fit_mock_second_held");
        for (int i = 0; i < 10000 && !*held; i++) usleep(1000);
        if (!*held) { fprintf(stderr, "unload_during_lookup: the lookup was not held\n"); return 2; }
        pthread_create(&tb, NULL, unload_thread_fn, &b);
        int observed = 0;
        for (int i = 0; i < 10000 && !observed; i++) {
            observed = b.done || waiting_on_setter(b.tid);
            if (!observed) usleep(1000);
        }
        *(_Atomic int*)dlsym(mock, "fit_mock_second_release") = 1;
        pthread_join(ta, NULL);
        pthread_join(tb, NULL);
        if (!observed) { fprintf(stderr, "unload_during_lookup: could not observe the unload\n"); return 2; }
        int during = *(int*)dlsym(mock, "fit_mock_unloads_during_call");
        checks++;
        int ok = during == 0 && launched == 0;
        if (!ok) wrong++;
        printf("%s  %-50s unloads during the lookup=%d (want 0), launch=%d (want 0)\n",
               ok ? "ok   " : "WRONG", "unload_during_lookup_waits", during, launched);
        check_unloads("unload_during_lookup_unloaded", 2);
    }

    // ---- the module goes while its second translation is being made ----

    // Thread A: the first launch is refused, and the load of the second translation is paused.
    // The program unloads the module, and the driver gives its handle to the next module loaded,
    // whose function has the same handle as the first. A's translation is not that module's: it
    // is unloaded, and A gets the first answer. The new module's refused launch is made from a
    // second translation of its own (before: from A's, which was taken for it).
    reset_counters();
    *(_Atomic int*)dlsym(mock, "fit_mock_pause") = 1;
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    {
        CUresult launched = -1;
        struct launch_args a = { function, &launched };
        pthread_t ta;
        pthread_create(&ta, NULL, launch_thread_fn, &a);
        for (int i = 0; i < 10000 && !*(_Atomic int*)dlsym(mock, "fit_mock_started"); i++) usleep(1000);
        if (!*(_Atomic int*)dlsym(mock, "fit_mock_started")) { fprintf(stderr, "handle_again: the load was not paused\n"); return 2; }
        cuModuleUnload(module);
        void *again, *again_function;
        *(uintptr_t*)dlsym(mock, "fit_mock_next_module") = (uintptr_t)module;
        r = cuModuleLoadData(&again, image); if (r || again != module) { fprintf(stderr, "handle_again: load %d %p\n", r, again); return 2; }
        r = cuModuleGetFunction(&again_function, again, "k_fit_regs"); if (r) return 1;
        *(_Atomic int*)dlsym(mock, "fit_mock_release") = 1;
        pthread_join(ta, NULL);
        check("handle_again_first_launch", launched, 701, 3, 1);
        check_unloads("handle_again_translation_unloaded", 2);
        r = cuLaunchKernel(again_function, 1,1,1, 1024,1,1, 0, NULL, NULL, NULL);
        check("handle_again_own_translation", r, 0, 4, 3);
        cuModuleUnload(again);
    }

    // Thread A as above; the context of the module is destroyed meanwhile, and the module with it.
    // The translation, loaded in that context, went with it: it is not unloaded.
    reset_counters();
    *(_Atomic int*)dlsym(mock, "fit_mock_pause") = 1;
    r = cuModuleLoadData(&module, image); if (r) return 1;
    r = cuModuleGetFunction(&function, module, "k_fit_regs"); if (r) return 1;
    {
        CUresult launched = -1;
        struct launch_args a = { function, &launched };
        pthread_t ta;
        pthread_create(&ta, NULL, launch_thread_fn, &a);
        for (int i = 0; i < 10000 && !*(_Atomic int*)dlsym(mock, "fit_mock_started"); i++) usleep(1000);
        if (!*(_Atomic int*)dlsym(mock, "fit_mock_started")) { fprintf(stderr, "context_destroyed: the load was not paused\n"); return 2; }
        cuCtxDestroy_v2((void *)0x7000);
        *(_Atomic int*)dlsym(mock, "fit_mock_release") = 1;
        pthread_join(ta, NULL);
        check("context_destroyed_first_launch", launched, 701, 2, 1);
        check_unloads("context_destroyed_translation_left", 0);
    }

    printf("\n%d checks, %d wrong\n", checks, wrong);
    return wrong ? 1 : 0;
}
