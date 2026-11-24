/*
 * test_nocode.c: a kernel without code (its body in the image is a trap and nothing else, as a
 * kernel built for other architectures only is emitted for the others) is given at the lookup and
 * answered about, as NVIDIA's driver does, but not launched: a launch returns 218 and a kernel node
 * of a graph is refused, while a capture in progress goes on; the context survives, and kernels
 * that are code run, among them one that traps only behind a test, and an empty one. The images are
 * tests/kernels/k_nocode.* (PTX for sm_75, retargeted by the layer; machine code for sm_75,
 * translated by it).
 *
 *   test_nocode IMAGE module    load IMAGE with cuModuleLoadData
 *   test_nocode IMAGE library   load IMAGE with cuLibraryLoadData (a deferred library unless VF_LAZY=0)
 *   test_nocode IMAGE native    the control, against NVIDIA's driver (DRVRUN_LIBCUDA): launch no_code
 *                               and say what the synchronisation returns. The GPU stops the context
 *                               (an Xid line): not part of tests/gpu/run.sh.
 *
 * Build: gcc -O2 -Wall -o test_nocode test_nocode.c -ldl
 * Run:   DRVRUN_LIBCUDA=<proxy> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./test_nocode IMAGE module
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

struct launch_config {
    unsigned grid[3], block[3], shared;
    void *stream, *attributes;
    unsigned count;
};
struct kernel_node {
    void *function;
    unsigned grid[3], block[3], shared;
    void **params, **extra;
    void *kernel, *context;
};

static void *cuda;
#define API(name) dlsym(cuda, name)
static int failed;

static void check(const char *what, int ok) {
    printf("  %s  %s\n", ok ? "ok    " : "WRONG ", what);
    failed |= !ok;
}

typedef CUresult (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                             void **, void **);

static char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *data = calloc(1, n + 1); /* PTX text is given with its terminating zero */
    if (data && fread(data, 1, n, f) != (size_t)n) {
        free(data);
        data = NULL;
    }
    fclose(f);
    *size = n;
    return data;
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[2], "module") && strcmp(argv[2], "library") && strcmp(argv[2], "native"))) {
        fprintf(stderr, "usage: test_nocode IMAGE module|library|native\n");
        return 2;
    }
    int library_mode = !strcmp(argv[2], "library"), native = !strcmp(argv[2], "native");
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
    printf("%s through the %s interface\n", argv[1], argv[2]);

    void *loaded = NULL;
    CUresult r;
    if (library_mode) {
        CUresult (*load)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned) =
            API("cuLibraryLoadData");
        r = load(&loaded, image, NULL, NULL, 0, NULL, NULL, 0);
    } else {
        CUresult (*load)(void **, const void *) = API("cuModuleLoadData");
        r = load(&loaded, image);
    }
    check("the image loads", r == 0);
    if (r) return 1;

    /* The functions, as a program takes them: from the module, or the library's kernel and its
       function in this context. */
    void *kernels[5] = {0}, *functions[5] = {0};
    const char *names[5] = {"no_code", "good", "trap_if", "empty", "store_then_trap"};
    char line[300];
    for (int i = 0; i < 5; ++i) {
        if (library_mode) {
            CUresult (*get)(void **, void *, const char *) = API("cuLibraryGetKernel");
            CUresult (*as_function)(void **, void *) = API("cuKernelGetFunction");
            r = get(&kernels[i], loaded, names[i]);
            if (!r) r = as_function(&functions[i], kernels[i]);
        } else {
            CUresult (*get)(void **, void *, const char *) = API("cuModuleGetFunction");
            r = get(&functions[i], loaded, names[i]);
        }
        snprintf(line, sizeof line, "%s is given at the lookup (got %d)", names[i], r);
        check(line, r == 0 && functions[i] != NULL);
    }
    void *no_code = functions[0], *good = functions[1], *trap_if = functions[2], *empty = functions[3];

    if (native) {
        r = launch(no_code, 1, 1, 1, 32, 1, 1, 0, NULL, (void *[]){&out}, NULL);
        CUresult s = synchronize();
        printf("NVIDIA's driver: cuLaunchKernel(no_code) returned %d, then cuCtxSynchronize %d\n", r, s);
        return 0;
    }

    /* Asked about: answered. */
    int value = 0;
    CUresult (*attribute)(int *, int, void *) = API("cuFuncGetAttribute");
    r = attribute(&value, 0 /* CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK */, no_code);
    snprintf(line, sizeof line, "cuFuncGetAttribute(no_code) is answered (got %d, %d threads)", r, value);
    check(line, r == 0 && value > 0);
    CUresult (*occupancy)(int *, void *, int, size_t) = API("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    r = occupancy(&value, no_code, 32, 0);
    snprintf(line, sizeof line, "cuOccupancyMaxActiveBlocksPerMultiprocessor(no_code) is answered (got %d)", r);
    check(line, r == 0 && value > 0);
    if (library_mode) {
        CUresult (*kernel_attribute)(int *, int, void *, int) = API("cuKernelGetAttribute");
        r = kernel_attribute(&value, 0, kernels[0], 0);
        snprintf(line, sizeof line, "cuKernelGetAttribute(no_code) is answered (got %d)", r);
        check(line, r == 0 && value > 0);
    }

    /* Launched: refused, and the driver not called. */
    void *args[] = {&out};
    r = launch(no_code, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
    snprintf(line, sizeof line, "cuLaunchKernel(no_code) is refused with 218 (got %d)", r);
    check(line, r == 218);
    struct launch_config config = {{1, 1, 1}, {32, 1, 1}, 0, NULL, NULL, 0};
    CUresult (*launch_ex)(const void *, void *, void **, void **) = API("cuLaunchKernelEx");
    r = launch_ex(&config, no_code, args, NULL);
    snprintf(line, sizeof line, "cuLaunchKernelEx(no_code) is refused with 218 (got %d)", r);
    check(line, r == 218);
    if (library_mode) {
        r = launch(kernels[0], 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
        snprintf(line, sizeof line, "cuLaunchKernel with the library's kernel no_code is refused with 218 (got %d)", r);
        check(line, r == 218);
    }
    r = synchronize();
    snprintf(line, sizeof line, "the context goes on (cuCtxSynchronize %d)", r);
    check(line, r == 0);

    /* Graphs: a node with it is not made; a capture in progress goes on, and records what runs. */
    CUresult (*graph_create)(void **, unsigned) = API("cuGraphCreate");
    CUresult (*add_node)(void **, void *, void **, size_t, const void *) = API("cuGraphAddKernelNode_v2");
    CUresult (*stream_create)(void **, unsigned) = API("cuStreamCreate");
    CUresult (*begin)(void *, int) = API("cuStreamBeginCapture_v2");
    CUresult (*end)(void *, void **) = API("cuStreamEndCapture");
    CUresult (*instantiate)(void **, void *, unsigned long long) = API("cuGraphInstantiateWithFlags");
    CUresult (*graph_launch)(void *, void *) = API("cuGraphLaunch");
    CUresult (*stream_sync)(void *) = API("cuStreamSynchronize");
    void *graph = NULL, *node = NULL, *stream = NULL, *captured = NULL, *exec = NULL;
    struct kernel_node bad = {no_code, {1, 1, 1}, {32, 1, 1}, 0, args, NULL, NULL, NULL};
    check("cuGraphCreate", graph_create(&graph, 0) == 0);
    r = add_node(&node, graph, NULL, 0, &bad);
    snprintf(line, sizeof line, "cuGraphAddKernelNode_v2 with no_code is refused with 218 (got %d)", r);
    check(line, r == 218);
    int v = 1000;
    void *good_args[] = {&out, &v};
    check("cuStreamCreate", stream_create(&stream, 0) == 0);
    check("cuStreamBeginCapture", begin(stream, 0) == 0);
    r = launch(no_code, 1, 1, 1, 32, 1, 1, 0, stream, args, NULL);
    snprintf(line, sizeof line, "cuLaunchKernel(no_code) in a capture is refused with 218 (got %d)", r);
    check(line, r == 218);
    r = launch(good, 1, 1, 1, 32, 1, 1, 0, stream, good_args, NULL);
    snprintf(line, sizeof line, "good is recorded in the same capture (got %d)", r);
    check(line, r == 0);
    r = end(stream, &captured);
    snprintf(line, sizeof line, "the capture ends with a graph (got %d)", r);
    check(line, r == 0 && captured != NULL);
    int got[32] = {0};
    if (!r && captured) {
        r = instantiate(&exec, captured, 0);
        if (!r) r = graph_launch(exec, stream);
        if (!r) r = stream_sync(stream);
        if (!r) r = to_host(got, out, sizeof got);
        snprintf(line, sizeof line, "the captured graph runs good: %d %d ... %d (got %d)", got[0], got[1], got[31], r);
        check(line, r == 0 && got[0] == 1000 && got[31] == 1031);
    }

    /* Code runs: good, trap_if behind its test, empty. */
    memset_d32(out, 0, 32);
    v = 100;
    r = launch(good, 1, 1, 1, 32, 1, 1, 0, NULL, good_args, NULL);
    if (!r) r = synchronize();
    if (!r) r = to_host(got, out, sizeof got);
    snprintf(line, sizeof line, "good(100) writes 100..131: %d ... %d (got %d)", got[0], got[31], r);
    check(line, r == 0 && got[0] == 100 && got[31] == 131);
    v = 0;
    r = launch(trap_if, 1, 1, 1, 32, 1, 1, 0, NULL, good_args, NULL);
    if (!r) r = synchronize();
    if (!r) r = to_host(got, out, sizeof got);
    snprintf(line, sizeof line, "trap_if(0) is launched and writes 7: %d (got %d)", got[5], r);
    check(line, r == 0 && got[5] == 7);
    r = launch(empty, 1, 1, 1, 32, 1, 1, 0, NULL, NULL, NULL);
    if (!r) r = synchronize();
    snprintf(line, sizeof line, "empty is launched (got %d)", r);
    check(line, r == 0);

    puts(failed ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return failed;
}
