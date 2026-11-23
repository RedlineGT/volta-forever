/*
 * launch_rate.c: how fast kernels are launched through a libcuda.so.1, for the cost of what the
 * proxy does at a launch. An empty kernel of PTX for sm_70 (which the proxy gives the driver as it
 * is) is launched with cuLaunchKernel:
 *   - N times in a row, then the context is synchronised: launches per second;
 *   - in bursts of 64 launches, each after a synchronisation (the driver's queue never fills), 2,000
 *     times: the time of one cuLaunchKernel call on the CPU, median and mean of the bursts.
 * With `refusing`, a module with a kernel that the proxy refuses (an instruction of sm_90) is loaded
 * first, so that the proxy looks at every launch (`guard` in crates/vf-proxy).
 *
 * Build: gcc -O2 -Wall -o launch_rate launch_rate.c -ldl -lm
 * Run:   DRVRUN_LIBCUDA=<libcuda.so.1> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./launch_rate [plain|refusing [N]]
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

int main(int argc, char **argv) {
    if (argc > 3 || (argc > 1 && strcmp(argv[1], "plain") && strcmp(argv[1], "refusing"))) {
        fprintf(stderr, "usage: launch_rate [plain|refusing [N]]\n");
        return 2;
    }
    int refusing = argc > 1 && !strcmp(argv[1], "refusing");
    long n = argc > 2 ? atol(argv[2]) : 200000;
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
