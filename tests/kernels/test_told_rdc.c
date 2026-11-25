/* An image that the driver's linker makes from relocatable code (two units built with -rdc=true,
 * each a fat binary of sm_75 and sm_80 cubins: k_told_rdc.cu, whose kernels call a function of
 * k_told_rdc2.cu), and a kernel that only the sm_80 code of its unit has:
 *
 *   test_told_rdc UNIT.fatbin OTHER.fatbin
 *
 * links the two (cuLinkAddData, cuLinkComplete), loads what the link made, looks k_rdc_both up
 * (and runs it when it is given: the function of the other unit gives 3 x + 1, 4 for 1; the layer
 * refuses it today, 218: a call that a relocation resolves is not translated) and looks
 * k_rdc_calls_out up: the module is of the image that the linker made, which has one code, and
 * the kernel is not in it (500). The layer loads nothing for it and says nothing of the code for
 * sm_80: tests/gpu/run.sh, told-8.0-rdc.
 *
 * Build: gcc -O2 -Wall -o test_told_rdc test_told_rdc.c -ldl
 * Run:   DRVRUN_LIBCUDA=<proxy> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./test_told_rdc A B
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

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
    if (argc != 3) {
        fprintf(stderr, "usage: test_told_rdc UNIT.fatbin OTHER.fatbin\n");
        return 2;
    }
    const char *path = getenv("DRVRUN_LIBCUDA");
    cuda = dlopen(path ? path : "libcuda.so.1", RTLD_NOW);
    if (!cuda) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    CUresult (*init)(unsigned) = API("cuInit");
    CUresult (*retain)(void **, int) = API("cuDevicePrimaryCtxRetain");
    CUresult (*set_current)(void *) = API("cuCtxSetCurrent");
    CUresult (*allocate)(CUdeviceptr *, size_t) = API("cuMemAlloc_v2");
    CUresult (*to_host)(void *, CUdeviceptr, size_t) = API("cuMemcpyDtoH_v2");
    CUresult (*synchronize)(void) = API("cuCtxSynchronize");
    CUresult (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                       void **) = API("cuLaunchKernel");
    CUresult (*create)(unsigned, int *, void **, void **) = API("cuLinkCreate_v2");
    CUresult (*add)(void *, int, void *, size_t, const char *, unsigned, int *, void **) = API("cuLinkAddData_v2");
    CUresult (*complete)(void *, void **, size_t *) = API("cuLinkComplete");
    CUresult (*destroy)(void *) = API("cuLinkDestroy");
    CUresult (*module_load)(void **, const void *) = API("cuModuleLoadData");
    CUresult (*module_function)(void **, void *, const char *) = API("cuModuleGetFunction");
    CUresult (*module_unload)(void *) = API("cuModuleUnload");
    void *context = NULL;
    CUdeviceptr out = 0;
    if (init(0) || retain(&context, 0) || set_current(context) || allocate(&out, 64)) {
        fprintf(stderr, "no CUDA context\n");
        return 2;
    }
    void *state = NULL, *image = NULL, *module = NULL, *function = NULL;
    size_t size = 0;
    CUresult r = create(0, NULL, NULL, &state);
    printf("link: create %d\n", r);
    if (r) return 1;
    for (int i = 1; i < 3; ++i) {
        size_t n = 0;
        char *unit = read_file(argv[i], &n);
        if (!unit) {
            fprintf(stderr, "cannot read %s\n", argv[i]);
            return 2;
        }
        r = add(state, 2 /* CU_JIT_INPUT_FATBINARY */, unit, n, i == 1 ? "unit" : "other", 0, NULL, NULL);
        printf("link: add %d: %d\n", i, r);
        if (r) return 1;
    }
    r = complete(state, &image, &size);
    printf("link: complete %d\n", r);
    if (r) return 1;
    r = module_load(&module, image);
    printf("load: %d\n", r);
    if (r) return 1;
    destroy(state);
    r = module_function(&function, module, "k_rdc_both");
    printf("k_rdc_both: lookup %d\n", r);
    if (!r) {
        float value = 0;
        void *params[] = {&out};
        r = launch(function, 1, 1, 1, 1, 1, 1, 0, NULL, params, NULL);
        if (!r) r = synchronize();
        if (!r) r = to_host(&value, out, sizeof value);
        printf("k_rdc_both: launch %d, stored %g\n", r, value);
    }
    function = (void *)1;
    r = module_function(&function, module, "k_rdc_calls_out");
    printf("k_rdc_calls_out: lookup %d, handle %s\n", r, function ? "left" : "null");
    printf("unload: %d\n", module_unload(module));
    printf("the context goes on: %d\n", synchronize());
    return 0;
}
