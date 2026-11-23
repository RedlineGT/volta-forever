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
 *   greenstale a module with a refused kernel `bad`, loaded in a green context that is destroyed; the
 *             driver gives the module's handle to the module that it makes for another library in a
 *             new context, whose kernel `bad` runs: it is launched and asked about.
 *   aba       a thread asks about the function `good` of a module with a refused kernel, and is held in
 *             the driver after its name was read; meanwhile the module goes, the driver gives the
 *             function's handle to the refused kernel of the module loaded next, and the layer's
 *             generation moves by exactly 65,536 (read through its symbol: GENERATION_OFFSET and
 *             INIT_OFFSET, of `cuda::kernels::GENERATION` and of `cuInit`, in hexadecimal); the
 *             thread goes on, and then the refused kernel is launched: refused.
 *   detach    a context is destroyed by its last cuCtxDetach: the function of a library's kernel that
 *             runs, found allowed there, and then the function of its refused kernel in a new context
 *             at the same handle, launched: refused. Then a context with a module of a refused kernel,
 *             attached to once (cuCtxAttach): detached once, the kernel is still refused; detached
 *             again (destroyed), the handle of the module that went is the module that the driver
 *             makes for another library, whose kernel `bad` runs: it is launched.
 *   storm     a function launched on the V100, 10,000 times in a row, then 10,000 times each after a
 *             launch on a later GPU of the function of a library's kernel there, asked for before each
 *             launch through the layer (cuKernelGetFunction), as a runtime may: the refusal asks the
 *             driver what a handle is a few times only, not at every launch.
 *   many      2,000 functions of 500 modules of nothing refused, beside a module with a refused kernel,
 *             launched in turn three times: once each found allowed, the refusal asks the driver about
 *             few of them again (at most 1 %), however their handles fall in its table.
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
    } else if (!strcmp(scenario, "aba")) {
        static const char NATIVE[] = ".version 7.0\n.target sm_70\n.address_size 64\n"
                                     ".visible .entry good()\n{\n ret;\n}\n";
        const char *g = getenv("GENERATION_OFFSET"), *i = getenv("INIT_OFFSET");
        if (!g || !i) {
            printf("  aba: skipped (the offsets of the layer's generation and of cuInit are not given)\n");
            return 0;
        }
        volatile uint64_t *generation =
            (volatile uint64_t *)((uintptr_t)L("cuInit") - strtoull(i, 0, 16) + strtoull(g, 0, 16));
        volatile uintptr_t *hold = D("hold_function");
        volatile int *held = D("held"), *holding = D("holding");
        void *ctx, *m, *f, *m2, *bad;
        pthread_t a;
        struct timespec ms = {0, 1000000};
        l_ctx_create(&ctx, 0, 0);
        l_load(&m, PTX);
        d_get_function(&f, m, "good");
        *hold = (uintptr_t)f;
        *held = 1;
        uint64_t read = *generation;
        pthread_create(&a, NULL, (void *(*)(void *))(void *)go, f);
        while (!*holding) nanosleep(&ms, NULL);
        l_unload(m);
        l_load(&m2, BAD);
        d_get_function(&bad, m2, "bad");
        int same = bad == f;
        // (A load and an unload through the layer each begin generations.)
        while (*generation - read < 65536 - 8) {
            void *x;
            l_load(&x, NATIVE);
            l_unload(x);
        }
        while (*generation - read < 65536) {
            void *x;
            l_load(&x, NATIVE);
        }
        uint64_t moved = *generation - read;
        *held = 0;
        pthread_join(a, NULL);
        *hold = 0;
        unsigned long before = *launched_bad;
        CUresult r = go(bad);
        char line[300];
        snprintf(line, sizeof line,
                 "aba: the handle given to a refused kernel: %s; the generation moved by %lu; its launch -> %d, %lu "
                 "reached the driver",
                 same ? "yes" : "no", (unsigned long)moved, r, *launched_bad - before);
        check(line, same && moved == 65536 && r == REFUSED && *launched_bad == before);
    } else if (!strcmp(scenario, "detach")) {
        static const char LIB_RUNS[] = ".version 7.0\n.target sm_70\n.address_size 64\n"
                                       ".visible .entry bad()\n{\n ret;\n}\n";
        CUresult (*l_detach)(void *) = L("cuCtxDetach"), (*l_attach)(void **, unsigned) = L("cuCtxAttach");
        CUresult (*d_func_module)(void **, void *) = D("cuFuncGetModule");
        void *library, *runs, *k_runs, *a, *b, *c, *d, *good_f, *bad_f, *m, *in_c, *again, *h, *module_of_h;
        char line[300];
        if (l_lib_load(&library, PTX, NULL, NULL, 0, NULL, NULL, 0) ||
            l_lib_load(&runs, LIB_RUNS, NULL, NULL, 0, NULL, NULL, 0))
            return 2;
        d_lib_kernel(&bad_kernel, library, "bad");
        d_lib_kernel(&good_kernel, library, "good");
        d_lib_kernel(&k_runs, runs, "bad");
        l_ctx_create(&a, 0, 0);
        d_kernel_function(&good_f, good_kernel);
        CUresult r1 = go(good_f);
        l_detach(a);
        l_ctx_create(&b, 0, 0);
        d_kernel_function(&bad_f, bad_kernel);
        unsigned long before = *launched_bad;
        CUresult r2 = go(bad_f);
        snprintf(line, sizeof line,
                 "detach: the kernel that runs -> %d; after the last cuCtxDetach, the refused kernel's function at "
                 "its handle: %s, launched -> %d, %lu reached the driver",
                 r1, bad_f == good_f ? "yes" : "no", r2, *launched_bad - before);
        check(line, r1 == 0 && bad_f == good_f && r2 == REFUSED && *launched_bad == before);
        l_ctx_create(&c, 0, 0);
        l_load(&m, PTX);
        d_get_function(&in_c, m, "bad");
        l_attach(&again, 0);
        l_detach(c);
        before = *launched_bad;
        CUresult r3 = go(in_c);
        unsigned long reached = *launched_bad - before;
        l_detach(c);
        l_ctx_create(&d, 0, 0);
        d_kernel_function(&h, k_runs);
        d_func_module(&module_of_h, h);
        unsigned long launches = *launched;
        CUresult r4 = go(h);
        snprintf(line, sizeof line,
                 "detach: attached once, detached once: the refused kernel -> %d (%lu reached the driver); detached "
                 "again: the handle of its module given to another library's: %s, whose kernel `bad` runs -> %d",
                 r3, reached, module_of_h == m ? "yes" : "no", r4);
        check(line,
              again == c && r3 == REFUSED && reached == 0 && module_of_h == m && r4 == 0 && *launched - launches == 1);
    } else if (!strcmp(scenario, "storm")) {
        void *v100, *later, *library, *m, *good, *f;
        char line[300];
        l_ctx_create(&v100, 0, 0);
        if (l_lib_load(&library, PTX, NULL, NULL, 0, NULL, NULL, 0)) return 2;
        d_lib_kernel(&good_kernel, library, "good");
        l_load(&m, GOOD);
        d_get_function(&good, m, "good");
        for (int i = 0; i < 10; ++i) go(good);
        unsigned long base = *identified;
        int wrong = 0;
        for (int i = 0; i < 10000; ++i) wrong |= go(good) != 0;
        unsigned long alone = *identified - base;
        l_ctx_create(&later, 0, 1);
        base = *identified;
        for (int i = 0; i < 10000; ++i) {
            l_set_current(later);
            wrong |= l_kernel_function(&f, good_kernel) != 0 || go(f) != 0;
            l_set_current(v100);
            wrong |= go(good) != 0;
        }
        // (The layer's cuKernelGetFunction itself asks the driver twice, for the library of the kernel
        // and the module of the function, which it follows.)
        unsigned long between = *identified - base - 2 * 10000;
        snprintf(line, sizeof line,
                 "storm: the refusal asked the driver what a handle is %lu times over 10,000 launches on the V100, "
                 "%lu times over 10,000 more each after a launch on a later GPU",
                 alone, between);
        check(line, !wrong && alone == 0 && between <= 10);
    } else if (!strcmp(scenario, "many")) {
        static const char FOUR[] = ".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry k0()\n{\n ret;\n}\n.visible .entry k1()\n{\n ret;\n}\n"
                                   ".visible .entry k2()\n{\n ret;\n}\n.visible .entry k3()\n{\n ret;\n}\n";
        enum { MODULES = 500, KERNELS = 4 * MODULES };
        static void *f[KERNELS];
        void *ctx, *refusing, *m;
        char line[300], name[4] = "k0";
        int wrong = 0;
        *(volatile int *)D("random_handles") = 1;
        l_ctx_create(&ctx, 0, 0);
        l_load(&refusing, PTX);
        for (int i = 0; i < MODULES; ++i) {
            wrong |= l_load(&m, FOUR) != 0;
            for (int j = 0; j < 4; ++j) {
                name[1] = (char)('0' + j);
                wrong |= d_get_function(&f[4 * i + j], m, name) != 0;
            }
        }
        unsigned long asked[3];
        for (int pass = 0; pass < 3; ++pass) {
            unsigned long base = *identified;
            for (int i = 0; i < KERNELS; ++i) wrong |= go(f[i]) != 0;
            // (Two questions for a function: is it a kernel of a library, and of which module is it.)
            asked[pass] = (*identified - base) / 2;
        }
        snprintf(line, sizeof line,
                 "many: of %d functions launched in turn, the refusal asked about %lu, then %lu and %lu again", KERNELS,
                 asked[0], asked[1], asked[2]);
        check(line, !wrong && asked[1] <= KERNELS / 100 && asked[2] <= KERNELS / 100);
    } else if (!strcmp(scenario, "greenstale")) {
        static const char LIB_RUNS[] = ".version 7.0\n.target sm_70\n.address_size 64\n"
                                       ".visible .entry bad()\n{\n ret;\n}\n";
        void *green_context, *b, *m, *library, *k, *h, *module_of_h;
        int value;
        CUresult (*d_func_module)(void **, void *) = D("cuFuncGetModule");
        if (l_lib_load(&library, LIB_RUNS, NULL, NULL, 0, NULL, NULL, 0)) return 2;
        d_lib_kernel(&k, library, "bad");
        l_ctx_create(&green_context, 0, 0);
        l_load(&m, PTX);
        l_green_destroy(green_context);
        l_ctx_create(&b, 0, 0);
        d_kernel_function(&h, k);
        d_func_module(&module_of_h, h);
        unsigned long before = *launched;
        CUresult r = go(h), q = l_attribute(&value, 0, h);
        char line[300];
        snprintf(line, sizeof line,
                 "greenstale: the library's module in the new context has the handle of the module that went: %s; "
                 "its kernel `bad`, which runs: launched -> %d (%lu reached the driver), asked about -> %d",
                 module_of_h == m ? "yes" : "no", r, *launched - before, q);
        check(line, module_of_h == m && r == 0 && q == 0 && *launched - before == 1);
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
