/*
 * test_told.c: what a program told 8.0 gets of the kernels of tests/kernels/k_told.cu (machine code
 * for sm_75 and sm_80, no PTX) that only its code for sm_80 has: the layer gives such a kernel from
 * that code, translated, in a module of its own, when it reaches no data of the image and is
 * translated; tests/gpu/run.sh checks what is printed here against what the layer said.
 *
 *   test_told IMAGE module    load IMAGE with cuModuleLoadData, look up with cuModuleGetFunction
 *   test_told IMAGE file      load the file IMAGE with cuModuleLoad
 *   test_told IMAGE library   load IMAGE with cuLibraryLoadData, look up with cuLibraryGetKernel
 *                             (as the CUDA 13 runtime does), then in the library's module in this
 *                             context (cuLibraryGetModule, as the CUDA 12 runtime does)
 *
 * Prints `NAME: lookup R` for each lookup (500: the driver's answer for a name it does not have),
 * and for the kernels found what they computed, against the host:
 *   k_both, k_before_80   the architecture of the code that ran
 *   k_bf16_sum            sums of pairs of bf16 against the exact sum rounded once to bf16 (to
 *                         nearest, ties to even; the PTX manual's fma.rn.bf16x2 with a
 *                         multiplier of one), and pairs of f32 against the nearest bf16; a NaN
 *                         must give a NaN (the layer gives 0x7fff)
 *   k_divides             quotients of f32 against the host's (IEEE division), bit for bit
 *   k_reads_data          80 times the variable `told_scale`, which the program sets to 3
 *                         through the module it holds (a module interface only)
 *   k_reads_table         an element of the table `told_table`
 *   k_adds_const          0 plus `told_bias[1]`, which the program sets to 20 the same way (the
 *                         image has 1.5): a kernel in another module would add 1.5
 *   k_const_address       (looked up only) the address of `told_bias`, from a word of its own
 *                         bank 0
 * then unloads the image, loads it again and looks the bf16 kernel up and runs it once more.
 * With TOLD_ONLY=NAME in the environment, only that kernel is launched.
 *
 * Build: gcc -O2 -Wall -o test_told test_told.c -ldl -lm
 * Run:   DRVRUN_LIBCUDA=<proxy> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./test_told IMAGE module
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef CUresult (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                             void **, void **);

static void *cuda;
#define API(name) dlsym(cuda, name)
#define N 8192

static char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *data = calloc(1, n + 1);
    if (data && fread(data, 1, n, f) != (size_t)n) {
        free(data);
        data = NULL;
    }
    fclose(f);
    *size = n;
    return data;
}

static uint32_t state = 0x2545f491u;
static uint32_t rnd(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

/* The magnitude of the bf16 `pattern` (15 bits), with 0x7f80 read as 2^128. */
static double magnitude(uint32_t pattern) {
    uint32_t exponent = pattern >> 7 & 0xff, fraction = pattern & 0x7f;
    return exponent ? ldexp(fraction + 128.0, (int)exponent - 134) : ldexp(fraction, -133);
}

/* The bf16 nearest to x, ties to even, by its value among the values of the bf16; 0x7fff for a NaN. */
static uint32_t nearest_bf16(double x) {
    if (isnan(x)) return 0x7fff;
    uint32_t sign = signbit(x) ? 0x8000 : 0, below = 0, beyond = 0x7f81;
    double size = fabs(x);
    while (beyond - below > 1) {
        uint32_t middle = (below + beyond) / 2;
        if (magnitude(middle) <= size) below = middle; else beyond = middle;
    }
    if (below == 0x7f80) return sign | below;
    double to_below = size - magnitude(below), to_above = magnitude(below + 1) - size;
    int up = to_above < to_below || (to_above == to_below && (below & 1));
    return sign | (below + up);
}

static double value_bf16(uint32_t half) {
    uint32_t m = half & 0x7fff;
    double v = m > 0x7f80 ? NAN : m == 0x7f80 ? INFINITY : magnitude(m);
    return half & 0x8000 ? -v : v;
}

/* The bf16 nearest to the sum of two bf16, rounded once. */
static uint32_t nearest_bf16_sum(uint32_t a, uint32_t c) {
    double x = value_bf16(a), y = value_bf16(c);
    int distance = abs((int)(a >> 7 & 0xff) - (int)(c >> 7 & 0xff));
    /* So far apart that the smaller is less than a fourth of the last bit of the larger: the sum
       rounds to the larger. (Nearer, the sum is exact as a double.) */
    if (isfinite(x) && isfinite(y) && distance > 40) return fabs(x) > fabs(y) ? a : c;
    return nearest_bf16(x + y);
}

static int same_bf16(uint32_t got, uint32_t want) {
    if ((want & 0x7fff) > 0x7f80) return (got & 0x7fff) > 0x7f80;
    return got == want;
}

static const uint16_t halves[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x3f81,
                                  0x4000, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x7f81, 0x3e80,
                                  0x4b80, 0x3380, 0x3300, 0x3280, 0x0100, 0x8100, 0x7f00};
static const uint32_t singles[] = {0, 0x80000000u, 1, 0x7fff, 0x8000, 0x8001, 0x18000, 0x17fff, 0x3f808000,
                                   0x3f818000, 0x3f807fff, 0x3f808001, 0x7f7f8000, 0x7f7f7fff, 0x7f7fffff,
                                   0xff7f8000u, 0x7f800000, 0xff800000u, 0x7f800001, 0x7fc00000, 0xffffffffu,
                                   0x7fffffff, 0x007fffff, 0x00800000, 0x807f8000u};
#define COUNT(x) (sizeof(x) / sizeof(x)[0])

static CUdeviceptr device[8];
static CUresult (*to_host)(void *, CUdeviceptr, size_t);
static CUresult (*to_device)(CUdeviceptr, const void *, size_t);
static CUresult (*synchronize)(void);
static launch_t launch;

/* k_bf16_sum on N pairs of bf16 pairs and N pairs of f32. */
static void run_bf16(void *function, const char *when) {
    static uint32_t a[N], c[N], sum[N], nearest[N];
    static uint32_t f[2 * N];
    size_t special = COUNT(halves);
    for (int i = 0; i < N; ++i) {
        /* Every pair of the special halves first, each in both halves of a word; then at random,
           some with exponents near each other, where sums round. */
        uint32_t x, y, z, w;
        if ((size_t)i < special * special) {
            x = halves[i % special], y = halves[i / special], z = y, w = x;
        } else {
            x = rnd() >> 16, y = rnd() >> 16, z = rnd() >> 16, w = rnd() >> 16;
            if (i & 1) y = (x & 0x7f80) ^ (rnd() & 0x8780) ^ (y & 0x7f), w = (z & 0x7f80) ^ (rnd() & 0x8380) ^ (w & 0x7f);
        }
        a[i] = x | z << 16, c[i] = y | w << 16;
        f[2 * i] = (size_t)i < COUNT(singles) ? singles[i] : rnd();
        f[2 * i + 1] = (size_t)i < COUNT(singles) ? singles[COUNT(singles) - 1 - i] : rnd();
        if (i >= (int)COUNT(singles) && (i & 3) == 0) f[2 * i] = (f[2 * i] & 0xffff0000u) | 0x8000 | (rnd() & 1);
    }
    memset(sum, 0xee, sizeof sum);
    memset(nearest, 0xee, sizeof nearest);
    int n = N;
    void *args[] = {&device[0], &device[1], &device[2], &device[3], &device[4], &n};
    CUresult r = to_device(device[0], a, sizeof a);
    if (!r) r = to_device(device[1], c, sizeof c);
    if (!r) r = to_device(device[2], sum, sizeof sum);
    if (!r) r = to_device(device[3], f, sizeof f);
    if (!r) r = to_device(device[4], nearest, sizeof nearest);
    if (!r) r = launch(function, N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL);
    if (!r) r = synchronize();
    if (!r) r = to_host(sum, device[2], sizeof sum);
    if (!r) r = to_host(nearest, device[4], sizeof nearest);
    if (r) {
        printf("k_bf16_sum%s: launch %d\n", when, r);
        return;
    }
    int wrong_sums = 0, wrong_nearest = 0;
    for (int i = 0; i < N; ++i) {
        for (int half = 0; half < 2; ++half) {
            uint32_t x = a[i] >> 16 * half & 0xffff, y = c[i] >> 16 * half & 0xffff;
            uint32_t got = sum[i] >> 16 * half & 0xffff, want = nearest_bf16_sum(x, y);
            if (!same_bf16(got, want) && wrong_sums++ < 5)
                printf("  sum %04x + %04x: %04x, the host %04x\n", x, y, got, want);
            float single;
            memcpy(&single, &f[2 * i + half], 4);
            got = nearest[i] >> 16 * half & 0xffff, want = nearest_bf16(single);
            if (!same_bf16(got, want) && wrong_nearest++ < 5)
                printf("  nearest to %08x: %04x, the host %04x\n", f[2 * i + half], got, want);
        }
    }
    printf("k_bf16_sum%s: %d sums, %d differ from the sum rounded once to bf16\n", when, 2 * N, wrong_sums);
    printf("k_bf16_sum%s: %d conversions, %d differ from the nearest bf16\n", when, 2 * N, wrong_nearest);
}

/* k_divides on N pairs of f32, some of them where the division takes its slow path. */
static void run_divides(void *function) {
    static float a[N], b[N], out[N];
    static const uint32_t edge[] = {0x00000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0x3f800000, 0x80000001u,
                                    0x7f800000, 0x00000000, 0x80000000u, 0x7fc00000, 0x01000000, 0x7e800000};
    for (int i = 0; i < N; ++i) {
        uint32_t x = (size_t)i < COUNT(edge) * COUNT(edge) ? edge[i % COUNT(edge)] : rnd();
        uint32_t y = (size_t)i < COUNT(edge) * COUNT(edge) ? edge[i / COUNT(edge)] : rnd();
        if (i >= 512 && (i & 7) == 0) y = (y & 0x807fffffu) | 0x00800000; /* a small divisor: large quotients */
        if (i >= 512 && (i & 7) == 1) x = x & 0x807fffffu;               /* a subnormal dividend */
        memcpy(&a[i], &x, 4);
        memcpy(&b[i], &y, 4);
    }
    memset(out, 0xee, sizeof out);
    int n = N;
    void *args[] = {&device[0], &device[1], &device[2], &n};
    CUresult r = to_device(device[0], a, sizeof a);
    if (!r) r = to_device(device[1], b, sizeof b);
    if (!r) r = to_device(device[2], out, sizeof out);
    if (!r) r = launch(function, N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL);
    if (!r) r = synchronize();
    if (!r) r = to_host(out, device[2], sizeof out);
    if (r) {
        printf("k_divides: launch %d\n", r);
        return;
    }
    int wrong = 0;
    for (int i = 0; i < N; ++i) {
        volatile float want = a[i] / b[i];
        float expected = want;
        int same = isnan(expected) ? isnan(out[i]) : !memcmp(&out[i], &expected, 4);
        if (!same && wrong++ < 5) printf("  %a / %a: %a, the host %a\n", a[i], b[i], out[i], expected);
    }
    printf("k_divides: %d quotients, %d differ from the host's\n", N, wrong);
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[2], "module") && strcmp(argv[2], "file") && strcmp(argv[2], "library"))) {
        fprintf(stderr, "usage: test_told IMAGE module|file|library\n");
        return 2;
    }
    const char *interface = argv[2];
    int library_mode = !strcmp(interface, "library");
    const char *path = getenv("DRVRUN_LIBCUDA");
    cuda = dlopen(path ? path : "libcuda.so.1", RTLD_NOW);
    if (!cuda) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    size_t size = 0;
    char *image = read_file(argv[1], &size);
    if (!image) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    CUresult (*init)(unsigned) = API("cuInit");
    CUresult (*retain)(void **, int) = API("cuDevicePrimaryCtxRetain");
    CUresult (*set_current)(void *) = API("cuCtxSetCurrent");
    CUresult (*allocate)(CUdeviceptr *, size_t) = API("cuMemAlloc_v2");
    CUresult (*memset_d32)(CUdeviceptr, unsigned, size_t) = API("cuMemsetD32_v2");
    to_host = API("cuMemcpyDtoH_v2");
    to_device = API("cuMemcpyHtoD_v2");
    synchronize = API("cuCtxSynchronize");
    launch = API("cuLaunchKernel");
    void *context = NULL;
    CUdeviceptr out = 0;
    if (init(0) || retain(&context, 0) || set_current(context) || allocate(&out, 32 * sizeof(int))) {
        fprintf(stderr, "no CUDA context\n");
        return 2;
    }
    for (int i = 0; i < 8; ++i)
        if (allocate(&device[i], 8 * N)) return 2;
    printf("%s through the %s interface\n", argv[1], interface);

    CUresult (*library_load)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned) =
        API("cuLibraryLoadData");
    CUresult (*library_unload)(void *) = API("cuLibraryUnload");
    CUresult (*module_load)(void **, const void *) = API("cuModuleLoadData");
    CUresult (*file_load)(void **, const char *) = API("cuModuleLoad");
    CUresult (*module_unload)(void *) = API("cuModuleUnload");
    CUresult (*library_kernel)(void **, void *, const char *) = API("cuLibraryGetKernel");
    CUresult (*kernel_library)(void **, void *) = API("cuKernelGetLibrary");
    CUresult (*as_function)(void **, void *) = API("cuKernelGetFunction");
    CUresult (*module_function)(void **, void *, const char *) = API("cuModuleGetFunction");
    CUresult (*module_global)(CUdeviceptr *, size_t *, void *, const char *) = API("cuModuleGetGlobal_v2");
    CUresult (*library_module)(void **, void *) = API("cuLibraryGetModule");

    const char *names[] = {"k_both", "k_before_80", "k_bf16_sum", "k_divides", "k_reads_data", "k_not_translated",
                           "k_nowhere", "k_reads_table", "k_adds_const", "k_const_address"};
    const int count = sizeof names / sizeof names[0];
    int lookups = 0;
    for (int load = 0; load < 2; ++load) {
        void *loaded = NULL, *module = NULL;
        CUresult r;
        if (library_mode) r = library_load(&loaded, image, NULL, NULL, 0, NULL, NULL, 0);
        else if (!strcmp(interface, "file")) r = file_load(&loaded, argv[1]);
        else r = module_load(&loaded, image);
        printf("load: %d\n", r);
        if (r) return 1;
        if (load == 1) {
            /* Loaded again: the bf16 kernel once more. */
            void *function = NULL, *kernel = NULL;
            if (library_mode) {
                r = library_kernel(&kernel, loaded, "k_bf16_sum");
                if (!r) r = as_function(&function, kernel);
            } else {
                r = module_function(&function, loaded, "k_bf16_sum");
            }
            printf("k_bf16_sum, loaded again: lookup %d\n", r);
            if (!r && function) run_bf16(function, ", loaded again");
            r = library_mode ? library_unload(loaded) : module_unload(loaded);
            printf("unload: %d\n", r);
            break;
        }
        if (library_mode) {
            r = library_module(&module, loaded);
            printf("the library's module: %d\n", r);
        } else {
            /* The program sets the variable through the module it holds. */
            CUdeviceptr scale = 0;
            size_t bytes = 0;
            int three = 3;
            r = module_global(&scale, &bytes, loaded, "told_scale");
            if (!r) r = to_device(scale, &three, sizeof three);
            printf("told_scale set to 3: %d\n", r);
            float bias[4] = {10, 20, 30, 40};
            r = module_global(&scale, &bytes, loaded, "told_bias");
            if (!r) r = to_device(scale, bias, sizeof bias);
            printf("told_bias set to 10, 20, 30, 40: %d\n", r);
        }
        /* Each name twice through the main interface, once more through the library's module. */
        for (int round = 0; round < 3; ++round) {
            if (round == 2 && !module) break;
            for (int i = 0; i < count; ++i) {
                void *function = NULL, *kernel = NULL;
                if (round == 2) {
                    r = module_function(&function, module, names[i]);
                } else if (library_mode) {
                    r = library_kernel(&kernel, loaded, names[i]);
                    if (!r) {
                        void *of = NULL;
                        CUresult asked = kernel_library(&of, kernel);
                        if (round == 0)
                            printf("%s: its library is %s\n", names[i],
                                   asked ? "not said" : of == loaded ? "the one loaded" : "another");
                        r = as_function(&function, kernel);
                    }
                } else {
                    r = module_function(&function, loaded, names[i]);
                }
                ++lookups;
                printf("%s: lookup %d%s\n", names[i], r, round == 2 ? " (in the library's module)" : "");
                if (r || !function || round == 1) continue;
                /* TOLD_ONLY=NAME: only that kernel is launched (the others are still looked up). */
                if (getenv("TOLD_ONLY") && strcmp(getenv("TOLD_ONLY"), names[i])) continue;
                if (round == 2) {
                    /* A function of the library's module runs too. */
                    if (i == 2) run_bf16(function, " (in the library's module)");
                    continue;
                }
                if (i == 2) {
                    run_bf16(function, "");
                } else if (i == 3) {
                    run_divides(function);
                } else if (i == 8) {
                    /* Were it given: what it adds says whose copy of told_bias it read. */
                    float a[128] = {0}, sum[128];
                    int n = 128;
                    void *args[] = {&device[0], &device[1], &n};
                    r = to_device(device[0], a, sizeof a);
                    if (!r) r = launch(function, 1, 1, 1, 128, 1, 1, 0, NULL, args, NULL);
                    if (!r) r = synchronize();
                    if (!r) r = to_host(sum, device[1], sizeof sum);
                    if (r) printf("k_adds_const: launch %d\n", r);
                    else printf("k_adds_const: added %g\n", sum[0]);
                } else if (i < 2 || i == 4 || i == 7) {
                    int got[2] = {0, 0};
                    int which = 2;
                    void *args[] = {&out, &which};
                    memset_d32(out, 0, 32);
                    r = launch(function, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
                    if (!r) r = synchronize();
                    if (!r) r = to_host(got, out, sizeof got);
                    if (r) printf("%s: launch %d\n", names[i], r);
                    else if (i == 4 || i == 7) printf("%s: stored %d\n", names[i], got[1]);
                    else printf("%s: ran the code for sm_%d\n", names[i], i == 0 ? got[0] : got[1]);
                }
            }
        }
        r = library_mode ? library_unload(loaded) : module_unload(loaded);
        printf("unload: %d\n", r);
    }
    printf("%d lookups\n", lookups);
    /* The context still runs a kernel's worth of work. */
    int word = 0;
    CUresult r = memset_d32(out, 7, 1);
    if (!r) r = to_host(&word, out, sizeof word);
    printf("the context goes on: %d\n", r ? -r : word);
    return 0;
}
