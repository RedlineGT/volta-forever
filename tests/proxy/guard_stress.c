/*
 * guard_stress.c: the refusal of a refused kernel at its use (`guard` in crates/vf-proxy), under
 * threads, against guard_mt_mock_driver.c, a stand-in for the driver that gives the handles of what
 * is gone again, last freed first, to any thread. The handles are taken from the stand-in directly,
 * as a program takes them through a private interface of the driver.
 *
 *   guard_stress PROXY MOCK SCENARIO [THREADS [SECONDS]]
 *
 * WRONG, and status 1: a function or kernel named bad* reached the driver (a refused kernel was
 * launched or asked about), a launch of a good* one was refused (218), or what the scenario expects
 * did not happen. Libraries are loaded whole (VF_LAZY=0), so that the stand-in has their entries.
 *   modules   threads load modules (bad and good, good only, bad only) through the layer, take their
 *             functions from the driver, launch each, unload them; two more threads launch a function
 *             of a module that stays, all the time.
 *   contexts  one library (bad and good); threads make a context, take the library's functions in it
 *             from the driver (cuKernelGetFunction), launch them, and destroy the context through the
 *             layer (cuCtxDestroy_v2), while the driver gives the handles of the destroyed context's
 *             functions to the next thread. green: the same, destroyed with cuGreenCtxDestroy.
 *   samename  a module that the layer keeps nothing of, with a function named as the refused kernel
 *             of a library, and another library whose kernel of that name runs: all three run.
 *   nocontext the function of a library's refused kernel in context A, taken from the driver, asked
 *             about while no context is current, then while another is; then, in A, asked about and
 *             launched: refused there, whatever was concluded in the other contexts.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int CUresult;
#define REFUSED 218

static const char PTX[] = ".version 8.8\n.target sm_75\n.address_size 64\n"
                          ".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n"
                          ".visible .entry good()\n{\n ret;\n}\n";
static const char GOOD[] = ".version 8.8\n.target sm_75\n.address_size 64\n"
                           ".visible .entry good()\n{\n ret;\n}\n";
static const char BAD[] = ".version 8.8\n.target sm_75\n.address_size 64\n"
                          ".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n";
/* Native PTX (nothing adapted, nothing kept) with a function named as a refused kernel of a library. */
static const char SAME_NAME[] = ".version 7.0\n.target sm_70\n.address_size 64\n"
                                ".visible .entry badname()\n{\n ret;\n}\n";
static const char LIB_SAME_NAME[] = ".version 8.8\n.target sm_75\n.address_size 64\n"
                                    ".visible .entry badname()\n{\n barrier.cluster.arrive;\n ret;\n}\n";
static const char LIB_SAME_NAME_RUNS[] = ".version 7.0\n.target sm_70\n.address_size 64\n"
                                         ".visible .entry badname()\n{\n ret;\n}\n";

static void *api, *drv;
#define L(name) dlsym(api, name)
#define D(name) dlsym(drv, name)
typedef CUresult (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                             void **, void **);
static CUresult (*l_load)(void **, const void *), (*l_unload)(void *), (*l_ctx_create)(void **, unsigned, int),
    (*l_ctx_destroy)(void *), (*l_green_destroy)(void *), (*l_set_current)(void *),
    (*l_lib_load)(void **, const void *, void *, void *, unsigned, void *, void *, unsigned),
    (*l_kernel_function)(void **, void *), (*l_attribute)(int *, int, void *);
static CUresult (*d_get_function)(void **, void *, const char *), (*d_lib_kernel)(void **, void *, const char *),
    (*d_kernel_function)(void **, void *);
static launch_t l_launch;
static volatile unsigned long *launched_bad, *asked_bad, *launched, *identified;
static volatile int stop, failed;
static atomic_ulong false_refusals, good_launches, bad_refusals, other_errors;

static CUresult go(void *f) { return l_launch(f, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL); }

static void check(const char *what, int ok) {
    printf("  %s  %s\n", ok ? "ok    " : "WRONG ", what);
    failed |= !ok;
}

static void bind(const char *proxy, const char *mock) {
    api = dlopen(proxy, RTLD_NOW);
    drv = dlopen(mock, RTLD_NOW);
    if (!api || !drv) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        exit(2);
    }
    l_load = L("cuModuleLoadData");
    l_unload = L("cuModuleUnload");
    l_ctx_create = L("cuCtxCreate_v2");
    l_ctx_destroy = L("cuCtxDestroy_v2");
    l_green_destroy = L("cuGreenCtxDestroy");
    l_set_current = L("cuCtxSetCurrent");
    l_lib_load = L("cuLibraryLoadData");
    l_kernel_function = L("cuKernelGetFunction");
    l_attribute = L("cuFuncGetAttribute");
    l_launch = L("cuLaunchKernel");
    d_get_function = D("cuModuleGetFunction");
    d_lib_kernel = D("cuLibraryGetKernel");
    d_kernel_function = D("cuKernelGetFunction");
    launched_bad = D("launched_bad");
    asked_bad = D("asked_bad");
    launched = D("launched");
    identified = D("identified");
    CUresult (*init)(unsigned) = L("cuInit");
    if (init(0)) exit(2);
}

static void check_good(CUresult r) {
    if (r == REFUSED) atomic_fetch_add(&false_refusals, 1);
    else if (r) atomic_fetch_add(&other_errors, 1);
    else atomic_fetch_add(&good_launches, 1);
}
static void check_bad(CUresult r) {
    if (r == REFUSED) atomic_fetch_add(&bad_refusals, 1);
    else atomic_fetch_add(&other_errors, 1);
}

/* ---- modules ---- */
static void *module_worker(void *arg) {
    void *ctx;
    unsigned seed = (unsigned)(uintptr_t)arg;
    l_ctx_create(&ctx, 0, 0);
    while (!stop) {
        int kind = rand_r(&seed) % 3;
        const char *image = kind == 0 ? PTX : kind == 1 ? GOOD : BAD;
        void *m = NULL, *bad = NULL, *good = NULL;
        if (l_load(&m, image)) {
            atomic_fetch_add(&other_errors, 1);
            continue;
        }
        if (kind != 2) d_get_function(&good, m, "good");
        if (kind != 1) d_get_function(&bad, m, "bad");
        for (int i = 0; i < 4; ++i) {
            if (good) check_good(go(good));
            if (bad) check_bad(go(bad));
        }
        l_unload(m);
    }
    return NULL;
}
static void *steady_worker(void *arg) {
    void *m, *good;
    l_set_current(arg);
    l_load(&m, GOOD);
    d_get_function(&good, m, "good");
    while (!stop) check_good(go(good));
    l_unload(m);
    return NULL;
}

/* ---- contexts ---- */
static void *bad_kernel, *good_kernel;
static int green;
static void *context_worker(void *arg) {
    (void)arg;
    while (!stop) {
        void *ctx, *bad = NULL, *good = NULL;
        l_ctx_create(&ctx, 0, 0);
        d_kernel_function(&good, good_kernel);
        for (int i = 0; i < 4; ++i) check_good(go(good));
        d_kernel_function(&bad, bad_kernel);
        for (int i = 0; i < 4; ++i) check_bad(go(bad));
        (green ? l_green_destroy : l_ctx_destroy)(ctx);
    }
    return NULL;
}

static void run_threads(void *(*fn)(void *), int threads, int seconds, void *(*extra)(void *), void *extra_arg,
                        int extras) {
    pthread_t t[64], e[8];
    for (long i = 0; i < threads; ++i) pthread_create(&t[i], NULL, fn, (void *)(i + 1));
    for (int i = 0; i < extras; ++i) pthread_create(&e[i], NULL, extra, extra_arg);
    sleep(seconds);
    stop = 1;
    for (int i = 0; i < threads; ++i) pthread_join(t[i], NULL);
    for (int i = 0; i < extras; ++i) pthread_join(e[i], NULL);
}

/* The verdict of a scenario with threads. */
static void report(const char *what) {
    char line[400];
    snprintf(line, sizeof line,
             "%s: %lu launches of good kernels, %lu refused; %lu launches of bad ones refused, %lu reached the "
             "driver; %lu other errors; the driver was asked what a handle is %lu times",
             what, atomic_load(&good_launches), atomic_load(&false_refusals), atomic_load(&bad_refusals),
             *launched_bad, atomic_load(&other_errors), *identified);
    check(line, atomic_load(&false_refusals) == 0 && *launched_bad == 0 && atomic_load(&other_errors) == 0 &&
                    atomic_load(&good_launches) > 0 && atomic_load(&bad_refusals) > 0);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: guard_stress PROXY MOCK SCENARIO [THREADS [SECONDS]]\n");
        return 2;
    }
    bind(argv[1], argv[2]);
    const char *scenario = argv[3];
    int threads = argc > 4 ? atoi(argv[4]) : 8, seconds = argc > 5 ? atoi(argv[5]) : 1;
    if (threads > 64) threads = 64;
    if (!strcmp(scenario, "modules")) {
        void *ctx;
        l_ctx_create(&ctx, 0, 0);
        run_threads(module_worker, threads, seconds, steady_worker, ctx, 2);
        report("modules");
    } else if (!strcmp(scenario, "contexts") || !strcmp(scenario, "green")) {
        void *ctx, *library;
        green = !strcmp(scenario, "green");
        *(volatile int *)D("destroy_sleep_us") = 50;
        l_ctx_create(&ctx, 0, 0);
        if (l_lib_load(&library, PTX, NULL, NULL, 0, NULL, NULL, 0)) return 2;
        d_lib_kernel(&bad_kernel, library, "bad");
        d_lib_kernel(&good_kernel, library, "good");
        run_threads(context_worker, threads, seconds, NULL, NULL, 0);
        report(scenario);
    } else if (!strcmp(scenario, "samename")) {
        void *ctx, *lib, *lib2, *m, *f, *k2, *f2;
        int value;
        l_ctx_create(&ctx, 0, 0);
        if (l_lib_load(&lib, LIB_SAME_NAME, NULL, NULL, 0, NULL, NULL, 0) ||
            l_lib_load(&lib2, LIB_SAME_NAME_RUNS, NULL, NULL, 0, NULL, NULL, 0))
            return 2;
        l_load(&m, SAME_NAME);
        d_get_function(&f, m, "badname");
        d_lib_kernel(&k2, lib2, "badname");
        d_kernel_function(&f2, k2);
        unsigned long before = *launched;
        CUresult r1 = go(f), r2 = go(k2), r3 = go(f2), r4 = l_attribute(&value, 0, f);
        char line[300];
        snprintf(line, sizeof line,
                 "samename: a function of a module of nothing refused -> %d, the kernel of a library that runs -> %d, "
                 "its function -> %d, a question -> %d; %lu of 3 launches reached the driver",
                 r1, r2, r3, r4, *launched - before);
        check(line, !r1 && !r2 && !r3 && !r4 && *launched - before == 3);
    } else if (!strcmp(scenario, "nocontext")) {
        void *a, *b, *library, *bad;
        int value;
        l_ctx_create(&a, 0, 0);
        if (l_lib_load(&library, PTX, NULL, NULL, 0, NULL, NULL, 0)) return 2;
        d_lib_kernel(&bad_kernel, library, "bad");
        d_kernel_function(&bad, bad_kernel);
        for (int other = 0; other < 2; ++other) {
            if (other) l_ctx_create(&b, 0, 0);
            else l_set_current(NULL);
            CUresult elsewhere = l_attribute(&value, 0, bad);
            l_set_current(a);
            unsigned long launches = *launched_bad, questions = *asked_bad;
            CUresult asked = l_attribute(&value, 0, bad), launched_in_a = go(bad);
            char line[300];
            snprintf(line, sizeof line,
                     "nocontext: asked about while %s -> %d; then in its context: asked -> %d, launched -> %d; "
                     "%lu questions and %lu launches reached the driver",
                     other ? "another context is current" : "no context is current", elsewhere, asked,
                     launched_in_a, *asked_bad - questions, *launched_bad - launches);
            check(line, asked == REFUSED && launched_in_a == REFUSED && *launched_bad == launches &&
                            *asked_bad == questions);
        }
    } else {
        fprintf(stderr, "no scenario %s\n", scenario);
        return 2;
    }
    return failed;
}
