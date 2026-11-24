/*
 * test_sections.c: what a program finds of the kernels of tests/kernels/k_sections.cu (machine code
 * for sm_75 and sm_80, no PTX), whose code for sm_80 has a kernel that its code for sm_75 has not.
 * It looks each kernel up, launches those it finds and says which code ran; tests/gpu/run.sh
 * checks that against what the layer gave and what it said.
 *
 *   test_sections IMAGE module    load IMAGE with cuModuleLoadData, look up with cuModuleGetFunction
 *   test_sections IMAGE file      load the file IMAGE with cuModuleLoad
 *   test_sections IMAGE library   load IMAGE with cuLibraryLoadData, look up with cuLibraryGetKernel
 *                                 (as the CUDA 13 runtime does), then in the library's module in
 *                                 this context (cuLibraryGetModule, as the CUDA 12 runtime does)
 *
 * Prints, for each lookup, `NAME: lookup R` (R the driver's result, 500 for a name it does not
 * have), and for a kernel found and launched `NAME: ran the code for sm_N`; then the number of
 * lookups made. The kernels only store one word each, with the first thread of one block.
 *
 * Build: gcc -O2 -Wall -o test_sections test_sections.c -ldl
 * Run:   DRVRUN_LIBCUDA=<proxy> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./test_sections IMAGE module
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef CUresult (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                             void **, void **);

static void *cuda;
#define API(name) dlsym(cuda, name)

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

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[2], "module") && strcmp(argv[2], "file") && strcmp(argv[2], "library"))) {
        fprintf(stderr, "usage: test_sections IMAGE module|file|library\n");
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
    CUresult (*to_host)(void *, CUdeviceptr, size_t) = API("cuMemcpyDtoH_v2");
    CUresult (*memset_d32)(CUdeviceptr, unsigned, size_t) = API("cuMemsetD32_v2");
    CUresult (*synchronize)(void) = API("cuCtxSynchronize");
    launch_t launch = API("cuLaunchKernel");
    void *context = NULL;
    CUdeviceptr out = 0;
    if (init(0) || retain(&context, 0) || set_current(context) || allocate(&out, 32 * sizeof(int))) {
        fprintf(stderr, "no CUDA context\n");
        return 2;
    }
    printf("%s through the %s interface\n", argv[1], interface);

    void *loaded = NULL, *module = NULL;
    CUresult r;
    if (library_mode) {
        CUresult (*load)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned) =
            API("cuLibraryLoadData");
        r = load(&loaded, image, NULL, NULL, 0, NULL, NULL, 0);
    } else if (!strcmp(interface, "file")) {
        CUresult (*load)(void **, const char *) = API("cuModuleLoad");
        r = load(&loaded, argv[1]);
    } else {
        CUresult (*load)(void **, const void *) = API("cuModuleLoadData");
        r = load(&loaded, image);
    }
    printf("load: %d\n", r);
    if (r) return 1;
    CUresult (*library_kernel)(void **, void *, const char *) = API("cuLibraryGetKernel");
    CUresult (*as_function)(void **, void *) = API("cuKernelGetFunction");
    CUresult (*module_function)(void **, void *, const char *) = API("cuModuleGetFunction");
    if (library_mode) {
        CUresult (*library_module)(void **, void *) = API("cuLibraryGetModule");
        r = library_module(&module, loaded);
        printf("the library's module: %d\n", r);
    }

    /* Each name twice through the main interface, once more through the library's module. */
    const char *names[] = {"k_both", "k_before_80", "k_from_80", "k_nowhere"};
    int lookups = 0;
    for (int round = 0; round < 3; ++round) {
        if (round == 2 && !module) break;
        for (int i = 0; i < 4; ++i) {
            void *function = NULL;
            if (round == 2) {
                r = module_function(&function, module, names[i]);
            } else if (library_mode) {
                void *kernel = NULL;
                r = library_kernel(&kernel, loaded, names[i]);
                if (!r) r = as_function(&function, kernel);
            } else {
                r = module_function(&function, loaded, names[i]);
            }
            ++lookups;
            printf("%s: lookup %d%s\n", names[i], r, round == 2 ? " (in the library's module)" : "");
            if (r || !function || round) continue;
            int got[2] = {0, 0};
            void *args[] = {&out};
            memset_d32(out, 0, 32);
            r = launch(function, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
            if (!r) r = synchronize();
            if (!r) r = to_host(got, out, sizeof got);
            if (r) {
                printf("%s: launch %d\n", names[i], r);
                continue;
            }
            /* k_both stores its architecture in the first word; the others 75 or 80 in the second. */
            printf("%s: ran the code for sm_%d\n", names[i], i == 0 ? got[0] : got[1]);
        }
    }
    printf("%d lookups\n", lookups);
    return 0;
}
