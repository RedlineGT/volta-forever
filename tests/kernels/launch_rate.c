/*
 * launch_rate.c: how fast kernels are launched through a libcuda.so.1, for the cost of what the
 * proxy does at a launch. An empty kernel of PTX for sm_70 (which the proxy gives the driver as it
 * is) is launched with cuLaunchKernel:
 *   - N times in a row, then the context is synchronised: launches per second;
 *   - in bursts of 64 launches, each after a synchronisation (the driver's queue never fills), 2,000
 *     times: the time of one cuLaunchKernel call on the CPU, median and mean of the bursts.
 * With `refusing`, a module with a kernel that the proxy refuses (an instruction of sm_90) is loaded
 * first, so that the proxy looks at every launch (`guard` in crates/vf-proxy).
 * With `many` (or `many-refusing`), N distinct empty kernels of one module (default 2,000) are
 * launched in turn, five times, each launch timed alone (a synchronisation every 32): the median and
 * the mean of each pass. The first pass is their first use; a module is loaded and unloaded before
 * the fourth (a new generation of the proxy's records). CUDA_MODULE_LOADING=EAGER keeps the
 * driver's own loading of each kernel out of the first pass.
 *
 * Build: gcc -O2 -Wall -o launch_rate launch_rate.c -ldl -lm
 * Run:   DRVRUN_LIBCUDA=<libcuda.so.1> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./launch_rate [plain|refusing|many|many-refusing [N]]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;

static const char empty_text[] =
    ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry empty()\n{\n    ret;\n}\n";
static const char refusing_text[] =
    ".version 9.0\n.target sm_75\n.address_size 64\n"
    ".visible .entry refused_by_layer()\n{\n    barrier.cluster.arrive;\n    ret;\n}\n"
    ".visible .entry other()\n{\n    ret;\n}\n";

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef CUresult (*launch_fn)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                              void **, void **);

/* `many`: n distinct kernels launched in turn, five times. */
static int in_turn(CUresult (*load)(void **, const void *), CUresult (*unload)(void *),
                   CUresult (*get)(void **, void *, const char *), CUresult (*synchronize)(void), launch_fn launch, long n,
                   const char *path, int refusing) {
    static const char small_text[] = ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry small()\n{\n    "
                                     "ret;\n}\n";
    char *text = malloc(64 + (size_t)n * 48), *p = text;
    void *module, *small, **f = malloc(sizeof(void *) * n);
    double *t = malloc(sizeof(double) * n);
    p += sprintf(p, ".version 7.0\n.target sm_70\n.address_size 64\n");
    for (long i = 0; i < n; ++i) p += sprintf(p, ".visible .entry k%ld()\n{\n    ret;\n}\n", i);
    if (load(&module, text)) return 2;
    for (long i = 0; i < n; ++i) {
        char name[24];
        snprintf(name, sizeof name, "k%ld", i);
        if (get(&f[i], module, name)) return 2;
    }
    for (int pass = 0; pass < 5; ++pass) {
        if (pass == 3 && (load(&small, small_text) || unload(small))) return 2;
        double sum = 0;
        for (long i = 0; i < n; ++i) {
            if (i % 32 == 0) synchronize();
            double start = now();
            if (launch(f[i], 1, 1, 1, 32, 1, 1, 0, NULL, NULL, NULL)) return 1;
            t[i] = (now() - start) * 1e9;
            sum += t[i];
        }
        synchronize();
        qsort(t, n, sizeof t[0], compare);
        printf("%s%s: %ld kernels in turn, pass %d%s: one launch call: median %.0f ns, mean %.0f ns\n", path,
               refusing ? " (a module with a refused kernel loaded)" : "", n, pass,
               pass == 0 ? " (first use)" : pass == 3 ? " (after a module was loaded and unloaded)" : "", t[n / 2], sum / n);
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "plain";
    if (argc > 3 || (strcmp(mode, "plain") && strcmp(mode, "refusing") && strcmp(mode, "many") &&
                     strcmp(mode, "many-refusing"))) {
        fprintf(stderr, "usage: launch_rate [plain|refusing|many|many-refusing [N]]\n");
        return 2;
    }
    int refusing = strstr(mode, "refusing") != NULL, many = !strncmp(mode, "many", 4);
    long n = argc > 2 ? atol(argv[2]) : many ? 2000 : 200000;
    const char *path = getenv("DRVRUN_LIBCUDA");
    void *cuda = dlopen(path ? path : "libcuda.so.1", RTLD_NOW);
    if (!cuda) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    CUresult (*init)(unsigned) = dlsym(cuda, "cuInit");
    CUresult (*retain)(void **, int) = dlsym(cuda, "cuDevicePrimaryCtxRetain");
    CUresult (*set_current)(void *) = dlsym(cuda, "cuCtxSetCurrent");
    CUresult (*load)(void **, const void *) = dlsym(cuda, "cuModuleLoadData");
    CUresult (*get)(void **, void *, const char *) = dlsym(cuda, "cuModuleGetFunction");
    CUresult (*synchronize)(void) = dlsym(cuda, "cuCtxSynchronize");
    CUresult (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                       void **) = dlsym(cuda, "cuLaunchKernel");
    void *context = NULL, *module = NULL, *other = NULL, *function = NULL;
    if (init(0) || retain(&context, 0) || set_current(context) || load(&module, empty_text) ||
        get(&function, module, "empty")) {
        fprintf(stderr, "no CUDA context or no kernel\n");
        return 2;
    }
    if (refusing && load(&other, refusing_text)) {
        fprintf(stderr, "the module with a refused kernel does not load\n");
        return 2;
    }
    if (many)
        return in_turn(load, dlsym(cuda, "cuModuleUnload"), get, synchronize, launch, n, path ? path : "libcuda.so.1",
                       refusing);
    /* Warm up. */
    for (int i = 0; i < 1000; ++i) launch(function, 1, 1, 1, 32, 1, 1, 0, NULL, NULL, NULL);
    synchronize();

    double start = now();
    for (long i = 0; i < n; ++i)
        if (launch(function, 1, 1, 1, 32, 1, 1, 0, NULL, NULL, NULL)) {
            fprintf(stderr, "launch %ld failed\n", i);
            return 1;
        }
    CUresult r = synchronize();
    double rate = n / (now() - start);

    enum { BURSTS = 2000, BURST = 64 };
    static double per_launch[BURSTS];
    double sum = 0;
    for (int b = 0; b < BURSTS; ++b) {
        synchronize();
        double t = now();
        for (int i = 0; i < BURST; ++i) launch(function, 1, 1, 1, 32, 1, 1, 0, NULL, NULL, NULL);
        per_launch[b] = (now() - t) / BURST * 1e9;
        sum += per_launch[b];
    }
    synchronize();
    qsort(per_launch, BURSTS, sizeof per_launch[0], compare);
    printf("%s%s: %ld launches in a row: %.0f per second (synchronise -> %d); one launch call: median %.1f ns, mean %.1f ns "
           "(%d bursts of %d)\n",
           path ? path : "libcuda.so.1", refusing ? " (a module with a refused kernel loaded)" : "", n, rate, r,
           per_launch[BURSTS / 2], sum / BURSTS, BURSTS, BURST);
    return r != 0;
}
